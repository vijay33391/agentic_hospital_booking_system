import logfire
from langchain_groq import ChatGroq
import os
from nemoguardrails import RailsConfig, LLMRails

from src.agents.guardrails.co_lang_rules import CO_LANG_RULE, YAML_CONTENT, RAIL_INDICATORS
from src.agents.gateways.config import guard_llm
_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses llama-3.1-8b-instant for fast intent classification at the gate —
    the heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
    """
    global _rails

    '''guard_llm = ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model="llama-3.1-8b-instant",
        temperature=0
    )
    '''
    
    config = RailsConfig.from_content(
        colang_content=CO_LANG_RULE,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (llama-3.1-8b-instant).")


async def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    NOTE: this MUST be `async def` and MUST call `_rails.generate_async(...)`,
    not `_rails.generate(...)`. `generate()` is a sync method that internally
    tries to manage its own event loop (via asyncio.run()/get_event_loop()).
    Calling it from inside an already-running loop (FastAPI/uvicorn) raises:
    "You are using the sync `generate` inside async code. You should use it
    with `await generate_async(...)`". Since graph.py already does
    `await guard(q)` inside `async def query(...)`, the fix is to make this
    function genuinely async end-to-end.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = await _rails.generate_async(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content.lower() for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None