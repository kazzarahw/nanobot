"""Latent-looped inference for the ReAct agent loop.

Provides an inner reasoning loop that runs *before* each action decision.
The model performs configurable latent passes — lightweight LLM calls with
no tools and a reduced token budget — to refine a compact state
representation (plan, key observations, uncertainties).  The refined state
is then injected into the next action-deciding LLM call so the model can
make better-informed decisions.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.config.schema import LatentLoopConfig
    from nanobot.providers.base import LLMProvider

# Maximum character length for the injected state summary.  Oldest
# observations are dropped first when this limit is exceeded.
_MAX_STATE_CHARS = 4000

# ── Prompts ──────────────────────────────────────────────────────────────

_INIT_PROMPT = """\
You are in a planning phase.  Analyze the conversation so far and produce \
an initial reasoning state.

Respond with ONLY a JSON object (no markdown fences) with exactly these keys:
{
  "plan": "<high-level plan / strategy for answering the user>",
  "key_observations": ["<fact 1>", "<fact 2>", ...],
  "uncertainties": ["<open question 1>", ...]
}
"""

_REFINE_PROMPT = """\
You are in a planning phase.  You will NOT execute any tools right now.

## Current Reasoning State
Plan: {plan}
Key observations: {observations}
Uncertainties: {uncertainties}

Refine your plan.  Update observations and uncertainties based on \
everything you know so far.  Be concise.

IMPORTANT GUARDRAILS:
- Stay focused ONLY on what the user asked. Do NOT add meetings, \
notifications, webhooks, or tasks not requested.
- Do NOT expand scope beyond the original request.
- If the plan already covers what was asked, keep it as-is.

Respond with ONLY a JSON object (no markdown fences):
{{"plan": "...", "key_observations": [...], "uncertainties": [...]}}
"""

_CONDENSE_PROMPT = """\
You just executed tools.  Condense the results into an updated reasoning \
state.

## Tool Results
{tool_results}

## Previous State
Plan: {plan}
Key observations: {observations}
Uncertainties: {uncertainties}

Update the state.  Keep only actionable information.  Resolve \
uncertainties that the tool results answer.  Be concise.

Respond with ONLY a JSON object (no markdown fences):
{{"plan": "...", "key_observations": [...], "uncertainties": [...]}}
"""


# ── Data ─────────────────────────────────────────────────────────────────


@dataclass
class LatentMetrics:
    """Tracks cost/performance of latent-loop passes."""

    total_passes: int = 0
    total_latent_tokens: int = 0
    converged_early: int = 0
    initialization_tokens: int = 0
    condensation_calls: int = 0
    parse_failures: int = 0


@dataclass
class LatentState:
    """Compact reasoning state carried across ReAct iterations."""

    plan: str = ""
    key_observations: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    iteration: int = 0
    refinement_count: int = 0
    last_refine_ok: bool = True

    # ── Serialization helpers ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "key_observations": self.key_observations,
            "uncertainties": self.uncertainties,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatentState:
        plan = data.get("plan", "")
        if not isinstance(plan, str):
            plan = str(plan)
        obs = data.get("key_observations", [])
        if isinstance(obs, str):
            obs = [obs]
        elif not isinstance(obs, list):
            obs = []
        else:
            obs = [str(o) for o in obs]
        unc = data.get("uncertainties", [])
        if isinstance(unc, str):
            unc = [unc]
        elif not isinstance(unc, list):
            unc = []
        else:
            unc = [str(u) for u in unc]
        return cls(plan=plan, key_observations=obs, uncertainties=unc)

    # ── Human-readable summary for injection ─────────────────────────

    def summary(self) -> str:
        obs = "\n".join(f"- {o}" for o in self.key_observations) or "(none)"
        unc = "\n".join(f"- {u}" for u in self.uncertainties) or "(none)"
        return (
            f"## Latent Reasoning State  (iteration {self.iteration}, "
            f"{self.refinement_count} refinements)\n\n"
            f"**Plan:** {self.plan}\n\n"
            f"**Key observations:**\n{obs}\n\n"
            f"**Uncertainties:**\n{unc}"
        )


# ── Reasoner ─────────────────────────────────────────────────────────────


# Per-tool truncation limits.  Tools not listed here use _DEFAULT_TOOL_TRUNCATION.
_TOOL_TRUNCATION: dict[str, int] = {
    "exec": 2000,
    "read_file": 2000,
    "edit_file": 1500,
    "web_fetch": 1200,
    "web_search": 800,
    "list_dir": 800,
    "write_file": 500,
}
_DEFAULT_TOOL_TRUNCATION = 1000
_TOTAL_TOOL_BUDGET = 6000

# Transient error patterns that warrant a retry.
_TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


def _repair_truncated_json(text: str) -> str | None:
    """Attempt to repair JSON truncated mid-string or mid-array.

    Handles common LLM failure modes:
    - Unterminated strings  (``"plan": "do stu``)
    - Unclosed arrays       (``"key_observations": ["a", "b``)
    - Missing closing brace
    Returns the repaired text, or *None* if repair was not possible.
    """
    if not text:
        return None
    # Strip trailing whitespace / partial tokens
    text = text.rstrip()

    # Close an unterminated string: odd number of unescaped quotes
    # means the last string was never closed.
    quote_count = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            i += 2
            continue
        if ch == '"':
            quote_count += 1
        i += 1

    if quote_count % 2 == 1:
        # Last quote is open — close it
        text += '"'

    # Close unclosed brackets / braces
    stack: list[str] = []
    in_string = False
    j = 0
    while j < len(text):
        ch = text[j]
        if ch == "\\" and in_string and j + 1 < len(text):
            j += 2
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch in ("{", "["):
                stack.append("}" if ch == "{" else "]")
            elif ch in ("}", "]") and stack:
                stack.pop()
        j += 1

    # Append missing closers in reverse order
    text += "".join(reversed(stack))

    # Validate the result
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return None


class LatentReasoner:
    """Manages the inner latent-reasoning loop."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        config: LatentLoopConfig,
    ) -> None:
        self.provider = provider
        self.model = model
        self.config = config
        self.metrics = LatentMetrics()
        self.disabled_due_to_instability = False
        self._parse_fail_iterations: list[int] = []

    # ── Public API ───────────────────────────────────────────────────

    async def initialize(self, messages: list[dict]) -> LatentState:
        """Create an initial latent state from the starting context."""
        condensed = self._condense_messages(messages, max_messages=6)
        prompt_messages: list[dict[str, Any]] = [
            {"role": "system", "content": _INIT_PROMPT},
            {"role": "user", "content": condensed},
        ]
        state, _ = await self._call_for_state(prompt_messages)
        logger.debug(f"Latent state initialized: plan={state.plan!r:.80}")
        return state

    async def refine(self, state: LatentState, messages: list[dict]) -> LatentState:
        """Run one latent reasoning pass — update plan/observations."""
        condensed = self._condense_messages(messages, max_messages=4)
        prompt = _REFINE_PROMPT.format(
            plan=state.plan,
            observations=json.dumps(state.key_observations),
            uncertainties=json.dumps(state.uncertainties),
        )
        prompt_messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": condensed},
        ]
        new_state, parse_ok = await self._call_for_state(prompt_messages, fallback=state)
        new_state.iteration = state.iteration
        new_state.refinement_count = state.refinement_count + 1
        new_state.last_refine_ok = parse_ok
        self.metrics.total_passes += 1
        logger.debug(
            f"Latent refinement #{new_state.refinement_count}: plan={new_state.plan!r:.80}"
        )
        return new_state

    async def incorporate_observations(
        self,
        state: LatentState,
        tool_results: list[tuple[str, str]],
    ) -> LatentState:
        """Condense tool results into the latent state.

        Args:
            state: Current latent state.
            tool_results: List of (tool_name, result_text) pairs.
        """
        results_text = self._truncate_tool_results(tool_results)
        prompt = _CONDENSE_PROMPT.format(
            tool_results=results_text,
            plan=state.plan,
            observations=json.dumps(state.key_observations),
            uncertainties=json.dumps(state.uncertainties),
        )
        prompt_messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Update the reasoning state with these tool results."},
        ]
        new_state, _ = await self._call_for_state(prompt_messages, fallback=state)
        new_state.iteration = state.iteration
        new_state.refinement_count = state.refinement_count
        self.metrics.condensation_calls += 1
        logger.debug("Latent state updated with tool observations")
        return new_state

    def inject(self, state: LatentState, messages: list[dict]) -> list[dict]:
        """Return a *copy* of messages with the latent state injected.

        The state is inserted as a second system message right after the
        first system prompt so the model can condition on it without
        polluting the conversation history.

        The original *state* and *messages* are never mutated.
        """
        # Work on a shallow copy of observations so we never mutate the caller's state
        observations = list(state.key_observations)
        summary = state.summary()

        # Truncate if too long (drop oldest observations first)
        while observations and len(summary) > _MAX_STATE_CHARS:
            observations.pop(0)
            # Rebuild summary with the trimmed list
            trimmed = LatentState(
                plan=state.plan,
                key_observations=observations,
                uncertainties=state.uncertainties,
                iteration=state.iteration,
                refinement_count=state.refinement_count,
            )
            summary = trimmed.summary()

        state_msg: dict[str, Any] = {"role": "system", "content": summary}

        # Insert after the first system message (index 1), or prepend.
        result = list(messages)
        insert_idx = 1 if result and result[0].get("role") == "system" else 0
        result.insert(insert_idx, state_msg)
        return result

    # ── Convergence & drift helpers ──────────────────────────────────

    @staticmethod
    def _plan_similarity(a: str, b: str) -> float:
        """Compute similarity ratio between two plan strings using SequenceMatcher."""
        if not a and not b:
            return 1.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _detect_plan_drift(plan: str, keywords: list[str]) -> bool:
        """Return True if the plan mentions any drift keywords."""
        plan_lower = plan.lower()
        return any(kw.lower() in plan_lower for kw in keywords)

    @staticmethod
    def _estimate_complexity(messages: list[dict]) -> int:
        """Estimate query complexity (0-3) based on message content.

        0 = trivial (greeting, short message) → skip latent passes
        1 = simple question
        2 = moderate task
        3 = complex multi-step task
        """
        # Extract the last real user message, ignoring internal loop prompts.
        user_text = LatentReasoner._latest_real_user_text(messages)

        if not user_text:
            return 0

        text_len = len(user_text)
        if text_len < 20:
            return 0

        # Check for action-oriented keywords
        action_keywords = [
            "write",
            "create",
            "build",
            "implement",
            "fix",
            "debug",
            "refactor",
            "deploy",
            "configure",
            "analyze",
            "search",
            "edit",
            "modify",
            "update",
            "install",
        ]
        text_lower = user_text.lower()
        action_count = sum(1 for kw in action_keywords if kw in text_lower)

        if text_len < 50 and action_count == 0:
            return 0
        if text_len < 100 and action_count <= 1:
            return 1
        if action_count >= 3 or text_len > 500:
            return 3
        return 2

    @staticmethod
    def _latest_real_user_text(messages: list[dict]) -> str:
        """Return latest user text that did not originate from internal prompts."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            if msg.get("_internal"):
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            return content if isinstance(content, str) else str(content)
        return ""

    # ── Internals ────────────────────────────────────────────────────

    async def _call_for_state(
        self,
        messages: list[dict],
        fallback: LatentState | None = None,
    ) -> tuple[LatentState, bool]:
        """Call the LLM and parse the response into a LatentState.

        Retries once on transient network errors.  On any parse failure
        the *fallback* state is returned unchanged (or an empty state if
        no fallback is provided).
        """
        last_exc: Exception | None = None
        for attempt in range(2):  # at most 1 retry
            try:
                latent_model = getattr(self.config, "latent_model", None) or self.model
                response = await self.provider.chat(
                    messages=messages,
                    tools=None,
                    model=latent_model,
                    temperature=self.config.latent_temperature,
                    max_tokens=self.config.latent_max_tokens,
                )
                # Track token usage
                usage = getattr(response, "usage", None) or {}
                tokens = usage.get("total_tokens") or usage.get("completion_tokens") or 0
                self.metrics.total_latent_tokens += tokens

                text = (response.content or "").strip()
                data = self._parse_json(text)
                return LatentState.from_dict(data), True
            except _TRANSIENT_EXCEPTIONS as e:
                last_exc = e
                if attempt == 0:
                    logger.debug(f"Latent call transient error ({e}), retrying in 2s")
                    await asyncio.sleep(2)
                    continue
                logger.warning(f"Latent call failed after retry ({e}), keeping previous state")
                return fallback if fallback is not None else LatentState(), False
            except Exception as e:
                self.metrics.parse_failures += 1
                logger.warning(f"Latent state parse failed ({e}), keeping previous state")
                return fallback if fallback is not None else LatentState(), False

        # Should not be reached, but guard against it.
        logger.warning(f"Latent call exhausted retries ({last_exc}), keeping previous state")
        return fallback if fallback is not None else LatentState(), False

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, handling markdown fences and truncation."""
        if not text:
            raise ValueError("empty response")

        # Strip optional markdown fences (```json ... ``` or ``` ... ```)
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        # First try a direct parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            raise ValueError(f"expected dict, got {type(data).__name__}")
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from surrounding prose
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        # Try to repair truncated JSON
        repaired = _repair_truncated_json(text)
        if repaired is not None:
            data = json.loads(repaired)
            if isinstance(data, dict):
                logger.debug("Repaired truncated JSON in latent response")
                return data

        raise ValueError(f"could not parse JSON from latent response: {text[:120]}")

    @staticmethod
    def _condense_messages(messages: list[dict], max_messages: int = 6) -> str:
        """Create a short textual summary of the most recent messages.

        Prioritises user and assistant messages over tool results (which
        are already captured in the latent state via incorporate_observations).
        """
        # Separate priority messages from tool results
        priority: list[dict] = []
        tools: list[dict] = []
        for msg in messages:
            if msg.get("_internal"):
                continue
            role = msg.get("role", "")
            if role in ("tool",):
                tools.append(msg)
            else:
                priority.append(msg)

        # Take recent priority messages first, fill remaining budget with tool results
        budget = max_messages
        selected = priority[-budget:]
        remaining = budget - len(selected)
        if remaining > 0 and tools:
            selected = tools[-remaining:] + selected

        lines: list[str] = []
        for msg in selected:
            role = msg.get("role", "?").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal content — extract text parts only
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            # Truncate — use shorter limit for tool results
            limit = 300 if msg.get("role") == "tool" else 500
            if len(content) > limit:
                content = content[:limit] + "..."
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    @staticmethod
    def _truncate_tool_results(tool_results: list[tuple[str, str]]) -> str:
        """Truncate tool results using per-tool limits within a total budget."""
        parts: list[str] = []
        total_chars = 0
        for name, result in tool_results:
            limit = _TOOL_TRUNCATION.get(name, _DEFAULT_TOOL_TRUNCATION)
            remaining_budget = max(0, _TOTAL_TOOL_BUDGET - total_chars)
            effective_limit = min(limit, remaining_budget)
            if effective_limit <= 0:
                parts.append(f"### {name}\n(truncated — budget exhausted)")
                continue
            truncated = result[:effective_limit]
            if len(result) > effective_limit:
                truncated += "..."
            parts.append(f"### {name}\n{truncated}")
            total_chars += len(truncated)
        return "\n\n".join(parts)


# ── Shared orchestration ────────────────────────────────────────────────


async def run_latent_passes(
    reasoner: LatentReasoner | None,
    state: LatentState | None,
    config: LatentLoopConfig | None,
    messages: list[dict],
    iteration: int,
) -> tuple[LatentState | None, list[dict]]:
    """Run inner latent passes and return ``(updated_state, action_messages)``.

    Shared by :class:`AgentLoop` and :class:`SubagentManager` so the
    orchestration logic lives in exactly one place.

    Supports early-exit when the plan converges (similarity-based) or
    drifts off-topic.  Also enforces session-wide token and refinement
    budgets, and skips passes for trivial queries.
    """
    if reasoner is None or state is None or config is None:
        return state, messages
    if reasoner.disabled_due_to_instability:
        return state, messages

    # Adaptive complexity: skip latent passes for trivial queries
    complexity = LatentReasoner._estimate_complexity(messages)
    if complexity == 0:
        logger.debug("Latent passes skipped (trivial query, complexity=0)")
        action_messages = messages
        if config.inject_state:
            action_messages = reasoner.inject(state, messages)
        return state, action_messages

    # Scale max passes by complexity (0 already handled above)
    effective_passes = min(config.max_latent_passes, max(1, complexity))

    if iteration > config.warmup_threshold:
        for _ in range(effective_passes):
            # Budget guards
            if reasoner.metrics.total_latent_tokens >= config.max_total_latent_tokens:
                logger.warning("Latent token budget exhausted, skipping refinement")
                break
            if reasoner.metrics.total_passes >= config.max_total_refinements:
                logger.warning("Latent refinement cap reached, skipping refinement")
                break

            prev_plan = state.plan
            state = await reasoner.refine(state, messages)
            if not state.last_refine_ok:
                reasoner._parse_fail_iterations.append(iteration)
                window = max(1, config.instability_window_iterations)
                min_iteration = max(1, iteration - window + 1)
                reasoner._parse_fail_iterations = [
                    i for i in reasoner._parse_fail_iterations if i >= min_iteration
                ]
                if (
                    config.disable_on_instability
                    and len(reasoner._parse_fail_iterations)
                    >= config.instability_parse_fail_threshold
                ):
                    reasoner.disabled_due_to_instability = True
                    logger.warning(
                        "Latent disabled due to instability: "
                        f"{len(reasoner._parse_fail_iterations)} parse failures within "
                        f"{window} iterations"
                    )
                    break
                # Parse failed; keep outer loop moving but do not treat as convergence.
                continue

            # Drift detection
            if LatentReasoner._detect_plan_drift(state.plan, config.plan_drift_keywords):
                logger.warning("Plan drift detected, reverting to previous plan")
                state = LatentState(
                    plan=prev_plan,
                    key_observations=state.key_observations,
                    uncertainties=state.uncertainties,
                    iteration=state.iteration,
                    refinement_count=state.refinement_count,
                )
                break

            # Similarity-based convergence
            similarity = LatentReasoner._plan_similarity(prev_plan, state.plan)
            if similarity >= config.convergence_similarity:
                reasoner.metrics.converged_early += 1
                logger.debug(f"Latent inner loop converged early (similarity={similarity:.2f})")
                break

    action_messages = messages
    if config.inject_state and not reasoner.disabled_due_to_instability:
        action_messages = reasoner.inject(state, messages)
    return state, action_messages


async def incorporate_tool_results(
    reasoner: LatentReasoner | None,
    state: LatentState | None,
    config: LatentLoopConfig | None,
    tool_results: list[tuple[str, str]],
    iteration: int,
) -> LatentState | None:
    """Incorporate tool results into the latent state.

    Shared helper matching :func:`run_latent_passes`.
    """
    if reasoner is None or state is None or config is None:
        return state
    if config.condense_tool_results:
        state = await reasoner.incorporate_observations(state, tool_results)
        state.iteration = iteration
    return state
