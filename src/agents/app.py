
# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
import logfire
from typing import Literal, Optional
from fastapi import FastAPI, Response
from src.agents.guardrails.rails import initialize_rails, guard
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.agents.nodes.agents import (
    supervise_node,
    information_node,
    booking_node,
    init_mcp_and_agents,  # NEW: one-time async init for MCP tools + agents
)
from src.agents.graph import booking_agent

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
        
if __name__ == "__main__":
    # Specify the port number here
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
