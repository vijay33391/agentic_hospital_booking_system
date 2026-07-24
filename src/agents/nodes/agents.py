import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("write_token"))

# ----------------******************--------------------------

import asyncio
import random
from typing import Literal

from langgraph.types import Command

from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from groq import RateLimitError  # groq-python raises this; langchain_groq re-raises it as-is

from src.agents.mcp_servers.config import resolve_relative_paths, mcp_config

# schema-------------------
"""create table public.adv_booking_system (
  date_slot timestamp without time zone null,
  specialization character varying null,
  doctor_name character varying null,
  is_available boolean null,
  patient_to_attend bigint null
) TABLESPACE pg_default;"""

# -----validators
from src.agents.utility.validators import (
    DateValidator,
    DateTimeModel,
    IdentifiactionNumberValidator,
    DOCTOR_NAMES,
)
# tools
from src.agents.utility.tools import (
    check_doctor_availability,
    check_doctor_availability_by_specialization,
    set_appointment,
    reschedule_appointment,
    cancel_appointment,
)
# state
from src.agents.utility.agent_state import AgentState, Router


# ---------------------------------------------------------------------------
# LLM — one shared client, reused by both agents.
# ---------------------------------------------------------------------------
from src.agents.gateways.config import llm_sub,llm_sup


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Hard ceiling on supervisor -> worker -> supervisor hops for a single user
# turn. This is a circuit breaker, not the primary fix — the primary fix is
# the deterministic FINISH check below. This just guarantees that even a
# pathological case can't loop indefinitely and burn through rate limits.
MAX_SUPERVISOR_HOPS = 4

# Retry policy for transient Groq errors (mainly 429 rate limits). Workers
# call the LLM at least once per hop, so a single flaky call shouldn't fail
# the whole turn.
_MAX_LLM_RETRIES = 3
_BASE_BACKOFF_SECONDS = 2.0


# ---------------------------------------------------------------------------
# MCP tools + agents: built EXACTLY ONCE, on the running event loop.
#
# Previously this module called `asyncio.run(mcp_client.get_tools())` (or an
# `await` statement) at *import time*. Both are broken:
#   - `asyncio.run()` at module scope fails the instant uvicorn is already
#     running an event loop ("cannot be called from a running event loop").
#   - a bare top-level `await` is a SyntaxError outside a notebook/REPL.
#
# It also rebuilt `create_agent(...)` from scratch inside information_node /
# booking_node on *every single request*, re-registering tools each time.
#
# Fix: expose an async `init_mcp_and_agents()` that FastAPI's `lifespan`
# calls once, on the real event loop, before the server starts accepting
# traffic. The nodes below just reuse the cached agent instances.
# ---------------------------------------------------------------------------

_supabase_tools: list = []
_information_agent = None
_booking_agent_runnable = None
_initialized = False
_init_lock = asyncio.Lock()


INFO_SYSTEM_PROMPT = """You are the Information Agent for a hospital appointment system.

SCOPE: doctor availability lookups only (by name or by specialization).

CRITICAL — AVAILABILITY IS LIVE, NOT CACHED:
Doctor availability changes constantly because bookings, cancellations,
and reschedules happen between messages — including earlier in THIS
conversation. Your conversation history may contain an availability
answer from a previous turn; that answer is a SNAPSHOT from the moment
it was fetched and may already be stale. Every time the user asks an
availability question — even if it looks identical to one asked earlier
in this same conversation, even for the same doctor and date — you MUST
call the tool again to get the current data. Never answer an
availability question directly from a previous tool result already in
your context. There is no such thing as "I already have this answer" for
availability queries.

RULES:
1. On EVERY new user message asking about doctor availability, call the
   appropriate tool (check_doctor_availability or
   check_doctor_availability_by_specialization) fresh — regardless of
   whether the same lookup appears earlier in the conversation history.
2. If a required field is missing (doctor name, specialization, or date),
   ask exactly ONE clarifying question and stop. Do not call a tool with
   guessed or placeholder values.
3. Within a single turn, once a tool call has returned a result for this
   message — including "no slots available" — that IS the answer for
   THIS message. Reply immediately in plain text. Do not call the exact
   same tool with the exact same arguments a second time within the same
   turn (this rule is scoped to avoiding redundant back-to-back calls
   before you've answered — it does NOT mean skipping the tool call on a
   later, new user message).
4. Never fabricate availability. Only report what the tool returned on
   THIS call.
5. Assume the current year is 2026 unless the user specifies otherwise.
6. Stay in scope: if asked to book, cancel, or reschedule, say that's
   handled by booking and do not attempt it yourself.
"""

BOOKING_SYSTEM_PROMPT = """You are the Booking Agent in a hospital appointment system.

SCOPE: setting, rescheduling, and cancelling appointments only.

RULES:
1. When the user wants to SET, CHANGE, or CANCEL an appointment, you MUST
   call exactly one of: set_appointment, cancel_appointment,
   reschedule_appointment. A tool call is mandatory for these intents —
   never simulate success in plain text.
2. If required information is missing (date, doctor name, or ID number),
   ask exactly ONE clear question, then stop and wait for the answer.
   Once you have everything, call the correct tool on your very next turn.
3. After a tool call returns, immediately relay its result to the user in
   plain text. Do not call the same tool again with the same arguments
   WITHIN THIS SAME TURN. This does not apply across turns: if the user
   later asks to book/reschedule/cancel again — even for the same
   doctor/date — treat it as a brand-new request and call the tool
   fresh, since the underlying schedule may have changed since any
   earlier attempt.
4. You may answer in plain text without calling a tool ONLY if the
   request is a general question unrelated to booking/cancelling/
   rescheduling (e.g. "what specializations do you have?").
5. Never fabricate a confirmation. Only report what a tool actually
   returned.
6. Assume the current year is 2026 unless the user specifies otherwise.
"""


async def init_mcp_and_agents() -> None:
    """
    Load MCP (Supabase) tools and build both react agents exactly once.

    MUST be awaited from FastAPI's `lifespan` startup (see graph.py), where
    a real event loop is already running. Safe to call more than once
    (e.g. multiple lifespan invocations in tests) — later calls are no-ops
    once initialization has succeeded, guarded by an asyncio.Lock so
    concurrent startup calls don't race and load tools twice.
    """
    global _supabase_tools, _information_agent, _booking_agent_runnable, _initialized

    if _initialized:
        return

    async with _init_lock:
        if _initialized:  # re-check inside the lock
            return

        try:
            config = mcp_config()
            client = MultiServerMCPClient(config["mcpServers"])
            _supabase_tools = await client.get_tools()
        except Exception:
            logfire.exception("Failed to load MCP tools during startup")
            raise  # fail startup loudly rather than serving with no tools

        _information_agent = create_agent(
            model=llm_sub,
            tools=[
                check_doctor_availability,
                check_doctor_availability_by_specialization,
                *_supabase_tools,
            ],
            system_prompt=INFO_SYSTEM_PROMPT,
        )

        _booking_agent_runnable = create_agent(
            model=llm_sub,
            tools=[
                set_appointment,
                cancel_appointment,
                reschedule_appointment,
                *_supabase_tools,
            ],
            system_prompt=BOOKING_SYSTEM_PROMPT,
        )

        _initialized = True
        logfire.info(
            "MCP tools + agents initialized once",
            tool_count=len(_supabase_tools),
        )


def _ensure_initialized() -> None:
    if not _initialized:
        raise RuntimeError(
            "MCP tools/agents were never initialized. Make sure FastAPI's "
            "lifespan calls `await init_mcp_and_agents()` on startup before "
            "any request reaches information_node/booking_node."
        )


async def _ainvoke_with_retry(agent, state, *, node_name: str):
    """
    Call agent.ainvoke(state) with bounded retries + exponential backoff on
    Groq 429 rate-limit errors. Without this, a single transient rate-limit
    blip surfaces as a hard failure to the user even though the very next
    call would likely succeed.
    """
    last_exc = None
    for attempt in range(1, _MAX_LLM_RETRIES + 1):
        try:
            return await agent.ainvoke(state)
        except RateLimitError as exc:
            last_exc = exc
            if attempt == _MAX_LLM_RETRIES:
                break
            backoff = _BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            backoff += random.uniform(0, 0.5)  # jitter, avoid thundering herd
            logfire.warning(
                "Groq rate limit hit, retrying",
                node=node_name,
                attempt=attempt,
                backoff_seconds=round(backoff, 2),
            )
            await asyncio.sleep(backoff)
    logfire.error(
        "Groq rate limit exceeded after retries",
        node=node_name,
        attempts=_MAX_LLM_RETRIES,
    )
    raise last_exc


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

worker_info = """WORKER: information_node
DESCRIPTION: Specialized agent to provide information related to doctor availability and hospital booking FAQs.

WORKER: booking_node
DESCRIPTION: Specialized agent to book, cancel, or reschedule appointments.

WORKER: FINISH
DESCRIPTION: If the user query has been answered, route to FINISH.
"""


system_prompt = (
    "You are a supervisor tasked with managing a conversation between the following workers.\n\n"
    "### SPECIALIZED ASSISTANTS:\n"
    f"{worker_info}\n\n"
    "Your primary role is to help users with doctor appointment booking, rescheduling, cancellation, "
    "doctor availability, and hospital booking FAQs.\n\n"
    "Delegate the task to the appropriate specialized worker exactly once per new user request. "
    "Given the following user request, respond with the worker to act next in the required JSON format.\n\n"

    "IMPORTANT CRITERIA FOR FINISH (apply strictly, in order):\n"
    "1. If the immediately preceding message is from a worker (information_node or booking_node), "
    "you MUST route to FINISH. A worker has already handled the request; do not send it back to "
    "the same or another worker to re-check or confirm the result.\n"

    "2. Only route to a worker if the immediately preceding message is from the user and has not yet "
    "been handled by any worker.\n"

    "3. Never route the same user request to the same worker more than once.\n"

    "4. If the user's request is unrelated to doctor availability, appointment booking, appointment "
    "rescheduling, appointment cancellation, or hospital booking FAQs, do NOT route the request to "
    "any worker. Instead, route directly to FINISH with the reasoning that the request is out of scope. "
    "The final response to the user must be exactly:\n"
    
    "\"I can help you with checking doctor availability, booking appointments, rescheduling appointments, "
    "cancelling appointments, and answering hospital booking FAQs. How can I assist you today?\""
)
    



def _last_message_is_unanswered_worker_reply(history) -> bool:
    """
    Returns True if the last message is a worker AIMessage that looks like a
    genuine clarifying question rather than a completed answer. Used only to
    decide whether the deterministic-FINISH short circuit still needs to let
    the graph continue (waiting on the user), as opposed to ending the turn.
    This is a narrow heuristic (trailing '?') — swap for a structured
    `needs_clarification: bool` flag returned by the workers themselves if
    you want something sturdier than punctuation-sniffing.
    """
    if not history:
        return False
    last = history[-1]
    return isinstance(last, AIMessage) and last.content.strip().endswith("?")


def supervise_node(agent_state: AgentState) -> Command[Literal["information_node", "booking_node", "__end__"]]:
    # 'query' may not exist yet on the very first call — use .get() instead
    # of agent_state['query'] to avoid a KeyError.
    current_query = agent_state.get("query", "")
    history = agent_state["messages"]
    hops = agent_state.get("hop_count", 0)

    # --- Deterministic short-circuit -------------------------------------
    # Previously, "should we FINISH" was decided by re-asking the LLM on
    # every hop, using only a natural-language instruction. That's a soft
    # constraint interpreted fresh each time, so a smaller/faster model
    # would occasionally misjudge it and route back into the same worker
    # repeatedly (see: same-turn information_node called 4x in a row,
    # eventually tripping the Groq 429). We now decide this in code first,
    # and only fall through to the LLM router when a worker hasn't yet
    # acted on the current request.
    if history and isinstance(history[-1], AIMessage) and history[-1].name in (
        "information_node",
        "booking_node",
    ):
        if _last_message_is_unanswered_worker_reply(history):
            # Worker asked a clarifying question — end this turn so the
            # user can reply; do NOT loop back into a worker.
            logfire.info("Deterministic FINISH — worker asked a clarifying question")
        else:
            logfire.info("Deterministic FINISH — worker already answered")
        return Command(
            goto="__end__",
            update={"next_action": "FINISH", "hop_count": 0},
        )

    # --- Hard circuit breaker --------------------------------------------
    if hops >= MAX_SUPERVISOR_HOPS:
        logfire.warning(
            "Max supervisor hops reached — forcing FINISH",
            hops=hops,
            user_query=current_query,
        )
        return Command(
            goto="__end__",
            update={
                "next_action": "FINISH",
                "hop_count": 0,
                "messages": [
                    AIMessage(
                        content=(
                            "I'm having trouble completing this request right now. "
                            "Could you try rephrasing it, or try again in a moment?"
                        ),
                        name="supervisor",
                    )
                ],
            },
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Query: {current_query}"},
    ] + history

    # Capture the very first user message as the canonical "query" once.
    new_query = ""
    if len(history) == 1:
        new_query = history[0].content

    try:
        response = llm_sup.with_structured_output(Router).invoke(messages)
    except RateLimitError:
        logfire.exception("Supervisor LLM rate-limited")
        return Command(
            goto="__end__",
            update={
                "next_action": "FINISH",
                "hop_count": 0,
                "messages": [
                    AIMessage(
                        content="We're a bit busy right now — please try again in a moment.",
                        name="supervisor",
                    )
                ],
            },
        )

    goto = response["next"]

    logfire.info("--------")
    logfire.info(
        "Supervisor routing decision",
        goto=goto,
        reasoning=response["reasoning"],
        current_node="supervisor",
        user_query=agent_state.get("query", ""),
        hops=hops,
    )

    if goto == "FINISH":
        return Command(
            goto="__end__",
            update={"next_action": "FINISH", "hop_count": 0},
        )

    if new_query:
        return Command(
            goto=goto,
            update={
                "next_action": goto,
                "query": new_query,
                "cur_reasoning": response["reasoning"],
                "hop_count": hops + 1,
                "messages": [
                    HumanMessage(content=f"user's identification number is {agent_state['user_id']}")
                ],
            },
        )

    return Command(
        goto=goto,
        update={
            "next_action": goto,
            "cur_reasoning": response["reasoning"],
            "hop_count": hops + 1,
        },
    )


# ---------------------------------------------------------------------------
# Information agent — reuses the cached agent built in init_mcp_and_agents()
# ---------------------------------------------------------------------------

async def information_node(state: AgentState) -> Command[Literal["supervisor"]]:
    _ensure_initialized()
    logfire.info("information_node invoked")

    try:
        # The Supabase MCP tools are async-only (no sync `func`, only a
        # coroutine), so this must be driven with ainvoke, not invoke().
        result = await _ainvoke_with_retry(_information_agent, state, node_name="information_node")
    except RateLimitError:
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="I'm unable to check availability right now due to high demand — please try again shortly.",
                        name="information_node",
                    )
                ]
            },
            goto="supervisor",
        )
    except Exception:
        logfire.exception("information_node failed unexpectedly")
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="Something went wrong while checking availability. Please try again.",
                        name="information_node",
                    )
                ]
            },
            goto="supervisor",
        )

    reply = result["messages"][-1].content
    logfire.info("Information node completed", node="information", reply=reply)

    return Command(
        # Only return the NEW message. 'messages' already uses the
        # add_messages reducer, which appends this to existing state.
        # Returning state["messages"] + [new] here would re-append the
        # entire history on every hop and duplicate it exponentially.
        update={"messages": [AIMessage(content=reply, name="information_node")]},
        goto="supervisor",
    )


# ---------------------------------------------------------------------------
# Booking agent — reuses the cached agent built in init_mcp_and_agents()
# ---------------------------------------------------------------------------

async def booking_node(state: AgentState) -> Command[Literal["supervisor"]]:
    _ensure_initialized()
    logfire.info("booking_node invoked")

    try:
        # Same reason as information_node: MCP tools are async-only.
        result = await _ainvoke_with_retry(_booking_agent_runnable, state, node_name="booking_node")
    except RateLimitError:
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="I'm unable to process this booking request right now due to high demand — please try again shortly.",
                        name="booking_node",
                    )
                ]
            },
            goto="supervisor",
        )
    except Exception:
        logfire.exception("booking_node failed unexpectedly")
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="Something went wrong while processing this booking. Please try again.",
                        name="booking_node",
                    )
                ]
            },
            goto="supervisor",
        )

    reply = result["messages"][-1].content
    logfire.info("Booking node completed", node="booking", reply=reply)

    return Command(
        update={"messages": [AIMessage(content=reply, name="booking_node")]},
        goto="supervisor",
    )