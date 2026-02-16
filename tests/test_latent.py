"""Tests for latent-looped inference (agent/latent.py)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.latent import LatentReasoner, LatentState
from nanobot.config.schema import LatentLoopConfig
from nanobot.providers.base import LLMResponse


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_response(state_dict: dict[str, Any]) -> LLMResponse:
    """Build a fake LLMResponse whose content is JSON-encoded *state_dict*."""
    return LLMResponse(content=json.dumps(state_dict))


def _make_provider(state_dict: dict[str, Any] | None = None) -> AsyncMock:
    """Return a mock LLMProvider whose .chat() returns a valid state response."""
    provider = AsyncMock()
    if state_dict is None:
        state_dict = {
            "plan": "test plan",
            "key_observations": ["obs1"],
            "uncertainties": ["unc1"],
        }
    provider.chat = AsyncMock(return_value=_make_response(state_dict))
    return provider


def _default_config(**overrides: Any) -> LatentLoopConfig:
    return LatentLoopConfig(**{"enabled": True, **overrides})


def _sample_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]


# ── LatentState unit tests ──────────────────────────────────────────────


class TestLatentState:
    def test_to_dict_roundtrip(self) -> None:
        state = LatentState(
            plan="do stuff",
            key_observations=["a", "b"],
            uncertainties=["x"],
            iteration=3,
            refinement_count=5,
        )
        d = state.to_dict()
        assert d == {
            "plan": "do stuff",
            "key_observations": ["a", "b"],
            "uncertainties": ["x"],
        }
        restored = LatentState.from_dict(d)
        assert restored.plan == "do stuff"
        assert restored.key_observations == ["a", "b"]
        assert restored.uncertainties == ["x"]
        # iteration / refinement_count are not in the dict (transient)
        assert restored.iteration == 0
        assert restored.refinement_count == 0

    def test_from_dict_missing_keys(self) -> None:
        state = LatentState.from_dict({})
        assert state.plan == ""
        assert state.key_observations == []
        assert state.uncertainties == []

    def test_summary_format(self) -> None:
        state = LatentState(
            plan="solve it",
            key_observations=["found x"],
            uncertainties=["why y?"],
            iteration=2,
            refinement_count=4,
        )
        s = state.summary()
        assert "iteration 2" in s
        assert "4 refinements" in s
        assert "solve it" in s
        assert "- found x" in s
        assert "- why y?" in s

    def test_summary_empty_lists(self) -> None:
        state = LatentState(plan="nothing yet")
        s = state.summary()
        assert "(none)" in s


# ── LatentReasoner tests ────────────────────────────────────────────────


class TestLatentReasoner:
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        provider = _make_provider(
            {"plan": "initial plan", "key_observations": ["o1"], "uncertainties": []}
        )
        reasoner = LatentReasoner(provider, "test-model", _default_config())

        state = await reasoner.initialize(_sample_messages())

        assert state.plan == "initial plan"
        assert state.key_observations == ["o1"]
        assert state.uncertainties == []
        provider.chat.assert_awaited_once()
        # Should call without tools
        call_kwargs = provider.chat.call_args
        assert call_kwargs.kwargs.get("tools") is None

    @pytest.mark.asyncio
    async def test_initialize_parse_failure_returns_empty(self) -> None:
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content="not json"))
        reasoner = LatentReasoner(provider, "test-model", _default_config())

        state = await reasoner.initialize(_sample_messages())

        assert state.plan == ""
        assert state.key_observations == []

    @pytest.mark.asyncio
    async def test_refine_increments_count(self) -> None:
        provider = _make_provider(
            {"plan": "refined plan", "key_observations": ["r1"], "uncertainties": ["u1"]}
        )
        reasoner = LatentReasoner(provider, "test-model", _default_config())
        initial = LatentState(plan="old", iteration=2, refinement_count=3)

        refined = await reasoner.refine(initial, _sample_messages())

        assert refined.plan == "refined plan"
        assert refined.iteration == 2  # preserved
        assert refined.refinement_count == 4  # incremented

    @pytest.mark.asyncio
    async def test_refine_parse_failure_keeps_fallback(self) -> None:
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content="bad json!"))
        reasoner = LatentReasoner(provider, "test-model", _default_config())
        original = LatentState(plan="original", key_observations=["keep"])

        result = await reasoner.refine(original, _sample_messages())

        assert result.plan == "original"
        assert result.key_observations == ["keep"]

    @pytest.mark.asyncio
    async def test_incorporate_observations(self) -> None:
        provider = _make_provider(
            {
                "plan": "updated plan",
                "key_observations": ["found file"],
                "uncertainties": [],
            }
        )
        reasoner = LatentReasoner(provider, "test-model", _default_config())
        state = LatentState(plan="old", iteration=5, refinement_count=2)

        updated = await reasoner.incorporate_observations(
            state, [("read_file", "contents of foo.py")]
        )

        assert updated.plan == "updated plan"
        assert updated.iteration == 5  # preserved
        assert updated.refinement_count == 2  # preserved (not a refinement)

    @pytest.mark.asyncio
    async def test_incorporate_truncates_long_results(self) -> None:
        """Tool results longer than per-tool limit are truncated in the prompt."""
        provider = _make_provider()
        reasoner = LatentReasoner(provider, "test-model", _default_config())
        state = LatentState(plan="p")

        long_result = "x" * 5000
        await reasoner.incorporate_observations(state, [("exec", long_result)])

        # exec has a 2000-char per-tool limit
        call_args = provider.chat.call_args
        prompt_messages = call_args.kwargs.get("messages") or call_args.args[0]
        system_content = prompt_messages[0]["content"]
        assert "x" * 2000 in system_content
        assert "x" * 2001 not in system_content

    def test_inject_inserts_after_system(self) -> None:
        reasoner = LatentReasoner(AsyncMock(), "m", _default_config())
        state = LatentState(plan="injected plan", iteration=1, refinement_count=1)
        messages = _sample_messages()

        result = reasoner.inject(state, messages)

        # Original messages unchanged
        assert len(messages) == 2
        # Injected list has one extra message
        assert len(result) == 3
        assert result[0]["role"] == "system"  # original system prompt
        assert result[1]["role"] == "system"  # injected state
        assert "injected plan" in result[1]["content"]
        assert result[2]["role"] == "user"

    def test_inject_no_system_message(self) -> None:
        reasoner = LatentReasoner(AsyncMock(), "m", _default_config())
        state = LatentState(plan="p")
        messages = [{"role": "user", "content": "hi"}]

        result = reasoner.inject(state, messages)

        assert len(result) == 2
        # Inserted at index 0 (before user message)
        assert result[0]["role"] == "system"

    def test_inject_truncates_observations(self) -> None:
        reasoner = LatentReasoner(AsyncMock(), "m", _default_config())
        # Create state with many observations to exceed _MAX_STATE_CHARS
        state = LatentState(
            plan="p",
            key_observations=["o" * 500 for _ in range(20)],  # 10000 chars
        )
        messages = _sample_messages()

        result = reasoner.inject(state, messages)

        injected_content = result[1]["content"]
        assert len(injected_content) <= 5000  # well under any reasonable limit

    def test_inject_does_not_mutate_state(self) -> None:
        """inject() must never modify the caller's LatentState object."""
        reasoner = LatentReasoner(AsyncMock(), "m", _default_config())
        original_obs = ["o" * 500 for _ in range(20)]
        state = LatentState(plan="p", key_observations=list(original_obs))
        messages = _sample_messages()

        reasoner.inject(state, messages)

        # The original state's observations must be untouched
        assert len(state.key_observations) == len(original_obs)
        assert state.key_observations == original_obs

    @pytest.mark.asyncio
    async def test_config_temperature_and_tokens(self) -> None:
        """Latent calls use the configured temperature and max_tokens."""
        provider = _make_provider()
        config = _default_config(latent_temperature=0.1, latent_max_tokens=512)
        reasoner = LatentReasoner(provider, "test-model", config)

        await reasoner.initialize(_sample_messages())

        call_kwargs = provider.chat.call_args.kwargs
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 512


# ── LatentReasoner._condense_messages tests ─────────────────────────────


class TestCondenseMessages:
    def test_basic_condensation(self) -> None:
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = LatentReasoner._condense_messages(messages, max_messages=2)
        assert "[USER] hello" in result
        assert "[ASSISTANT] hi there" in result
        assert "system prompt" not in result  # only last 2

    def test_truncates_long_messages(self) -> None:
        messages = [{"role": "user", "content": "a" * 1000}]
        result = LatentReasoner._condense_messages(messages)
        assert len(result) < 600  # 500 + "..." + prefix

    def test_handles_multimodal_content(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "text", "text": "describe this"},
                ],
            }
        ]
        result = LatentReasoner._condense_messages(messages)
        assert "describe this" in result


# ── Config defaults test ────────────────────────────────────────────────


class TestLatentLoopConfig:
    def test_defaults(self) -> None:
        config = LatentLoopConfig()
        assert config.enabled is False
        assert config.max_latent_passes == 3
        assert config.latent_max_tokens == 1024
        assert config.latent_temperature == 0.3
        assert config.condense_tool_results is True
        assert config.inject_state is True
        assert config.warmup_threshold == 0

    def test_agent_defaults_has_latent_loop(self) -> None:
        from nanobot.config.schema import AgentDefaults

        defaults = AgentDefaults()
        assert hasattr(defaults, "latent_loop")
        assert isinstance(defaults.latent_loop, LatentLoopConfig)
        assert defaults.latent_loop.enabled is False
