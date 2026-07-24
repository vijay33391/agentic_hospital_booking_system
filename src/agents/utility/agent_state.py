"""
Multi-agent hospital booking system — LLM client configuration.

All LLM calls go through Portkey (caching, retries, provider fallback).
Model selection is delegated entirely to the Portkey configs
(SUPERVISOR_CONFIG_ID / SUB_AGENTS_CONFIG_ID) via each config's
`targets[].override_params.model` — this file does not hardcode which
upstream model gets called; the dashboard config decides that.
"""

from typing import TypedDict, Any, Literal, Annotated
import os
import logging

from langgraph.graph import StateGraph, add_messages
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

# logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State shared across the graph."""
    user_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    next_action: str
    cur_reasoning: str
    query: str
    hop_count: int


class Router(TypedDict):
    next: Literal["information_node", "booking_node", "FINISH"]
    reasoning: str
    query: str


worker_info = """WORKER: information_node
 DESCRIPTION: specialized agent to provide information related to availability of doctors or any FAQs related to hospital.

WORKER: booking_node
 DESCRIPTION: specialized agent to only to book, cancel or reschedule appointment

WORKER: FINISH
DESCRIPTION: If User Query is answered and route to Finished"""

'''
# --------------------------------------------------------------------------
# Portkey configuration
# --------------------------------------------------------------------------
# PORTKEY_API_KEY must come from the environment — never hardcode it in
# source. Fail fast and loud if it's missing rather than silently breaking
# at the first LLM call.
try:
    PORTKEY_API_KEY = os.environ["PORTKEY_API_KEY"]
except KeyError as exc:
    raise RuntimeError(
        "PORTKEY_API_KEY environment variable is not set. "
        "Set it (e.g. via .env / your secrets manager) before starting the app."
    ) from exc

# Config IDs ("pc-...") select the routing/model/fallback/cache rules you've
# already defined in the Portkey dashboard. Overridable via env vars so you
# can point staging/prod at different configs without a code change.
SUPERVISOR_CONFIG_ID = os.getenv("PORTKEY_SUPERVISOR_CONFIG", "pc-adv-bo-a4f9e2")
SUB_AGENTS_CONFIG_ID = os.getenv("PORTKEY_SUBAGENTS_CONFIG", "pc-adv-bo-f4891c")

ENVIRONMENT = os.getenv("ENVIRONMENT", "production")


def _build_portkey_llm(
    *,
    config_id: str,
    feature: str,
    temperature: float = 0.7,
    model_hint: str = "portkey-routed",
) -> ChatOpenAI:
    """
    Build a Portkey-routed ChatOpenAI client.

    ChatOpenAI is used instead of ChatGroq because Portkey exposes an
    OpenAI-compatible proxy endpoint; ChatGroq is hardwired to Groq's API and
    has no base_url override. `model_hint` only satisfies ChatOpenAI's
    required `model` field — the actual upstream model is decided by each
    target's `override_params.model` in the Portkey config, so this value
    gets overridden at the gateway.
    """
    headers = createHeaders(
        api_key=PORTKEY_API_KEY,
        config=config_id,
        metadata={
            "feature": feature,
            "environment": ENVIRONMENT,
        },
    )
    return ChatOpenAI(
        api_key=PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=model_hint,
        temperature=temperature,
        default_headers=headers,
    )


def get_supervisor_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Supervisor agent LLM, routed via SUPERVISOR_CONFIG_ID (llama-3.3-70b-versatile + fallback)."""
    return _build_portkey_llm(
        config_id=SUPERVISOR_CONFIG_ID,
        feature="advance-booking-supervisor",
        temperature=temperature,
        model_hint="llama-3.3-70b-versatile",
    )


def get_subagent_llm(temperature: float = 0.7) -> ChatOpenAI:
    """information_node / booking_node LLM, routed via SUB_AGENTS_CONFIG_ID (gpt-oss-120b + fallback)."""
    return _build_portkey_llm(
        config_id=SUB_AGENTS_CONFIG_ID,
        feature="advance-booking-subagents",
        temperature=temperature,
        model_hint="openai/gpt-oss-120b",
    )


# Module-level singletons — same usage pattern as your original llm_sup / llm_sub,
# so the rest of your graph code (`llm_sup.invoke(...)`, `llm_sub.bind_tools(...)`, etc.)
# doesn't need to change.
llm_sup = get_supervisor_llm()
llm_sub = get_subagent_llm()


def extract_cache_status(response: Any) -> str:
    """
    Pull x-portkey-cache-status from a response's raw HTTP headers.

    LangChain's ChatOpenAI wraps the OpenAI SDK, which exposes the raw
    httpx response differently across versions, so this checks a few known
    attribute paths defensively, then falls back to `response_metadata`
    (where LangChain sometimes surfaces provider headers). Returns 'MISS'
    if nothing is found (e.g. streaming responses don't always carry headers
    the same way).
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            headers = getattr(raw, "headers", {})
            status = headers.get("x-portkey-cache-status", "")
            if status:
                return status.upper()

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        headers = metadata.get("headers", {})
        status = headers.get("x-portkey-cache-status", "")
        if status:
            return status.upper()

    logger.debug("Could not find x-portkey-cache-status header on response")
    return "MISS"'''