'''how agent state is internally in agentic workflow.
class AgentState(TypedDict):
    """State shared across the graph."""
    user_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    next_action: str
    cur_reasoning: str
    query: str
    hop_count: int  # NEW: supervisor loop-cap counter, see agents.py
'''
# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
import asyncio
import uuid

import logfire
from typing import Optional
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.agents.guardrails.rails import initialize_rails, guard
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from src.agents.nodes.agents import (
    supervise_node,
    information_node,
    booking_node,
    init_mcp_and_agents,  # NEW: one-time async init for MCP tools + agents
)
from src.agents.graph import booking_agent

# Wall-clock budget for a single /query call. Guards against a hung MCP
# server, a stalled Groq call, or a supervisor loop slipping past the
# hop cap in agents.py — without this, a stuck request hangs forever
# with no signal to the caller.
QUERY_TIMEOUT_SECONDS = 45


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

# ---------------------------------------------------------------------------
# CORS — the browser-based UI (static/index.html) is served from this same
# origin via StaticFiles below, but CORS is left permissive here so the UI
# can also be opened from a different port/host during local development
# (e.g. a live-reload dev server) without silent fetch() failures.
# Tighten `allow_origins` to your real frontend origin(s) in production.
# ---------------------------------------------------------------------------
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=4000)
    # No shared static default here — see below. `None` means "the caller
    # didn't supply one", and we generate a unique id per request rather
    # than falling back to a fixed string that every anonymous caller
    # would collide on.
    thread_id: Optional[str] = None
    user_id: Optional[str] = None


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
        logfire.exception("Failed to generate graph image")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@api.post("/query")
async def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q.strip()
    if not q:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Field 'q' must not be blank.")

    # Bug fix: previously both defaulted to the literal string
    # "default_user". MemorySaver checkpoints conversation state by
    # thread_id, so every caller that omitted thread_id was silently
    # sharing ONE conversation/state with every other such caller —
    # including each other's booking/appointment context. Generate a
    # fresh unique id per request instead of a shared fallback.
    thread_id = request.thread_id or f"anon-{uuid.uuid4()}"
    user_id = request.user_id or f"anon-{uuid.uuid4()}"

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "query": q,
        "user_id": user_id,
        "next_action": "",
        "cur_reasoning": "",
        "hop_count": 0,
    }

    # Configuration for Memory (Thread ID) — this is what makes booking_agent's
    # MemorySaver checkpointer actually persist/resume conversation state.
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = await guard(q)

        if rail_fired:
            logfire.info("guardrails fired", thread_id=thread_id)
            return {
                "query": q,
                "answer": rail_response,
                "user_id": user_id,
                "next_action": "",
                "cur_reasoning": "skipped agentic workflow, guardrails fired",
                "status": "blocked",
            }

        # Gate 2: LangGraph pipeline, bounded by a wall-clock timeout so a
        # hung MCP call or LLM request can't hold the connection open
        # indefinitely.
        try:
            final_output = await asyncio.wait_for(
                booking_agent.ainvoke(initial_state, config=config),
                timeout=QUERY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logfire.error("Query timed out", thread_id=thread_id, timeout=QUERY_TIMEOUT_SECONDS)
            return {
                "query": q,
                "answer": "This is taking longer than expected — please try again.",
                "user_id": user_id,
                "status": "timeout",
            }

        answer = final_output["messages"][-1].content
        logfire.info("agentic workflow completed", thread_id=thread_id, answer=answer)

        return {
            "query": q,
            "answer": answer,
            "user_id": user_id,
            "status": "ok",
        }
    except Exception:
        # logfire.exception captures the full traceback, not just str(e) —
        # without this, failures like the earlier supervisor-loop / 429
        # issue are much harder to diagnose from logs alone.
        logfire.exception("Backend execution failed", thread_id=thread_id)
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
        }


# ---------------------------------------------------------------------------
# Serve the chat UI at /ui (static/index.html). Kept separate from "/" so
# the existing health-check JSON response above is untouched.
# ---------------------------------------------------------------------------
api.mount(
    "/ui",
    StaticFiles(directory="ui/static", html=True),
    name="ui",
)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:api",     
        host="0.0.0.0",
        port=8000,
        
    )