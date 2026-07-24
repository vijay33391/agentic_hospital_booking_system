import logfire
import os
import asyncio
from contextlib import asynccontextmanager
from typing import Literal, Optional

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from nemoguardrails import RailsConfig, LLMRails

from src.agents.mcp_servers.config import resolve_relative_paths, mcp_config

# ---------------------------------------------------------------------------
# Env / logging
# ---------------------------------------------------------------------------

load_dotenv()
logfire.configure(token=os.getenv("write_token"))
logfire.info("check logfire is working or not")

# ---------------------------------------------------------------------------
# Schema reference (for humans, not executed)
# ---------------------------------------------------------------------------
"""create table public.adv_booking_system (
  date_slot timestamp without time zone null,
  specialization character varying null,
  doctor_name character varying null,
  is_available boolean null,
  patient_to_attend bigint null
) TABLESPACE pg_default;"""

# ---------------------------------------------------------------------------
# Validators / tools / state / nodes
# ---------------------------------------------------------------------------
from src.agents.utility.validators import (
    DateValidator,
    DateTimeModel,
    IdentifiactionNumberValidator,
    DOCTOR_NAMES,
)
from src.agents.utility.tools import (
    check_doctor_availability,
    check_doctor_availability_by_specialization,
    set_appointment,
    reschedule_appointment,
    cancel_appointment,
)
from src.agents.utility.agent_state import AgentState, Router
from src.agents.nodes.agents import (
    supervise_node,
    information_node,
    booking_node,
    init_mcp_and_agents,  # NEW: one-time async init for MCP tools + agents
)

# ---------------------------------------------------------------------------
# Primary LLM
# ---------------------------------------------------------------------------

# llama-3.1-8b-instant is unreliable for structured/tool-calling output.
# llama-3.3-70b-versatile / gpt-oss-120b are far more consistent for this workload.
'''llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)
'''
# ---------------------------------------------------------------------------
# Graph — compiled ONCE, with a checkpointer so `thread_id` config actually
# persists conversation state across calls. `booking_agent` is the single
# source of truth used everywhere in this file.

# NOTE: this graph can be safely built/compiled at import time because the
# node functions (supervise_node/information_node/booking_node) only need
# the MCP tools + cached agents *when they run*, not when the graph is
# assembled. Those are loaded once via `init_mcp_and_agents()` in the
# FastAPI lifespan below, before the server accepts any requests.
# ---------------------------------------------------------------------------

checkpointer = MemorySaver()  # swap for a Postgres/Redis checkpointer in production

graph = StateGraph(AgentState)
graph.add_node("supervisor", supervise_node)
graph.add_node("information_node", information_node)
graph.add_node("booking_node", booking_node)
graph.add_edge(START, "supervisor")
# NOTE: no other add_edge calls exist here — this only works because
# supervise_node/information_node/booking_node return `Command(goto=...)`
# objects to route dynamically.
booking_agent = graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# CLI entrypoint (for local testing without FastAPI)
# ---------------------------------------------------------------------------

'''async def main():
    from src.agents.nodes.agents import init_mcp_and_agents
    await init_mcp_and_agents()

    inputs = [HumanMessage(content="can you check if doctor jane smith available on 8 August 2024 ?")]
    state = {"messages": inputs, "user_id": "10232303"}
    config = {"configurable": {"thread_id": "cli-test-thread"}}

    result = await booking_agent.ainvoke(state, config=config)
    print(result)
    print('@' * 50)
    print(result['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main())'''


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

from fastapi import FastAPI, Response
from src.agents.guardrails.rails import initialize_rails, guard
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once, on the real event loop uvicorn is already driving. This is
    the ONLY correct place to load MCP tools and build the agents:
      - it's async, so `await` works (no asyncio.run() / bare top-level
        await, both of which break here)
      - it runs exactly once per process, before any request is served, so
        MCP tools are fetched once and both react agents are built once —
        not rebuilt on every /query call.
    """
    initialize_rails()
    await init_mcp_and_agents()
    logfire.info("startup complete: guardrails ready, MCP tools + agents initialized")
    yield


api = FastAPI(title="Enterprise Agentic RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"
    user_id: Optional[str] = "default_user"  # was referenced but never defined


@api.get("/")
def home():
    return {"message": "Enterprise LangGraph booking agent API is live."}


@api.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = booking_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


@api.post("/query")
async def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id
    user_id = request.user_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "query": q,
        "user_id": user_id,
        "next_action": "",
        "cur_reasoning": "",
    }

    # Configuration for Memory (Thread ID) — this is what makes booking_agent's
    # MemorySaver checkpointer actually persist/resume conversation state.
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = await guard(q)

        if rail_fired:
            logfire.info(f"guardrails fired | thread={thread_id}")
            return {
                "query": q,
                "answer": rail_response,
                "user_id": user_id,
                "next_action": "",
                "cur_reasoning": "skipped agentic workflow, guardrails fired",
                "status": "blocked",
            }

        # Gate 2: LangGraph pipeline
        final_output = await booking_agent.ainvoke(initial_state, config=config)
        logfire.info("guardrails did not fire; agentic workflow ran")
        return {
            "query": q,
            "answer": final_output["messages"][-1].content,
        }
    except Exception as e:
        logfire.error(f"Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
        }