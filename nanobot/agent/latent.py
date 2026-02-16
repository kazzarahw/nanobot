"""Latent-looped inference for the ReAct agent loop.

Provides an inner reasoning loop that runs *before* each action decision.
The model performs configurable latent passes — lightweight LLM calls with
no tools and a reduced token budget — to refine a compact state
representation (plan, key observations, uncertainties).  The refined state
is then injected into the next action-deciding LLM call so the model can
make better-informed decisions.
"""

from __future__ import annotations

import json
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
class LatentState:
    """Compact reasoning state carried across ReAct iterations."""

    plan: str = ""
    key_observations: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    iteration: int = 0
    refinement_count: int = 0

    # ── Serialization helpers ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "key_observations": self.key_observations,
            "uncertainties": self.uncertainties,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatentState:
        return cls(
            plan=data.get("plan", ""),
            key_observations=data.get("key_observations", []),
            uncertainties=data.get("uncertainties", []),
        )

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

    # ── Public API ───────────────────────────────────────────────────

    async def initialize(self, messages: list[dict]) -> LatentState:
        """Create an initial latent state from the starting context."""
        condensed = self._condense_messages(messages, max_messages=6)
        prompt_messages: list[dict[str, Any]] = [
            {"role": "system", "content": _INIT_PROMPT},
            {"role": "user", "content": condensed},
        ]
        state = await self._call_for_state(prompt_messages)
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
        new_state = await self._call_for_state(prompt_messages, fallback=state)
        new_state.iteration = state.iteration
        new_state.refinement_count = state.refinement_count + 1
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
        results_text = "\n\n".join(
            f"### {name}\n{result[:1000]}" for name, result in tool_results
        )
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
        new_state = await self._call_for_state(prompt_messages, fallback=state)
        new_state.iteration = state.iteration
        new_state.refinement_count = state.refinement_count
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

    # ── Internals ────────────────────────────────────────────────────

    async def _call_for_state(
        self,
        messages: list[dict],
        fallback: LatentState | None = None,
    ) -> LatentState:
        """Call the LLM and parse the response into a LatentState.

        On any parse failure the *fallback* state is returned unchanged
        (or an empty state if no fallback is provided).
        """
        try:
            response = await self.provider.chat(
                messages=messages,
                tools=None,
                model=self.model,
                temperature=self.config.latent_temperature,
                max_tokens=self.config.latent_max_tokens,
            )
            text = (response.content or "").strip()
            # Strip optional markdown fences
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            return LatentState.from_dict(data)
        except Exception as e:
            logger.warning(f"Latent state parse failed ({e}), keeping previous state")
            return fallback if fallback is not None else LatentState()

    @staticmethod
    def _condense_messages(messages: list[dict], max_messages: int = 6) -> str:
        """Create a short textual summary of the most recent messages."""
        recent = messages[-max_messages:]
        lines: list[str] = []
        for msg in recent:
            role = msg.get("role", "?").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal content — extract text parts only
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            # Truncate individual messages
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)
