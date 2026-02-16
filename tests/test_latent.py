"""Tests for latent-looped inference (agent/latent.py)."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.latent import (
    LatentMetrics,
    LatentReasoner,
    LatentState,
    _repair_truncated_json,
    incorporate_tool_results,
    run_latent_passes,
)
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

    def test_latent_model_defaults_to_none(self) -> None:
        config = LatentLoopConfig()
        assert config.latent_model is None

    def test_latent_model_can_be_set(self) -> None:
        config = LatentLoopConfig(latent_model="anthropic/claude-haiku")
        assert config.latent_model == "anthropic/claude-haiku"


# ── from_dict type coercion tests ─────────────────────────────────────


class TestFromDictTypeCoercion:
    def test_plan_coerced_to_str(self) -> None:
        state = LatentState.from_dict({"plan": 42})
        assert state.plan == "42"

    def test_observations_string_becomes_list(self) -> None:
        state = LatentState.from_dict({"key_observations": "single observation"})
        assert state.key_observations == ["single observation"]

    def test_uncertainties_string_becomes_list(self) -> None:
        state = LatentState.from_dict({"uncertainties": "one question"})
        assert state.uncertainties == ["one question"]

    def test_observations_non_list_non_str_becomes_empty(self) -> None:
        state = LatentState.from_dict({"key_observations": 123})
        assert state.key_observations == []

    def test_nested_non_str_items_coerced(self) -> None:
        state = LatentState.from_dict({"key_observations": [1, True, "ok"]})
        assert state.key_observations == ["1", "True", "ok"]


# ── JSON repair tests ────────────────────────────────────────────────


class TestRepairTruncatedJson:
    def test_unterminated_string(self) -> None:
        text = '{"plan": "do stu'
        repaired = _repair_truncated_json(text)
        assert repaired is not None
        data = json.loads(repaired)
        assert data["plan"] == "do stu"

    def test_unclosed_array(self) -> None:
        text = '{"key_observations": ["a", "b"'
        repaired = _repair_truncated_json(text)
        assert repaired is not None
        data = json.loads(repaired)
        assert data["key_observations"] == ["a", "b"]

    def test_missing_closing_brace(self) -> None:
        text = '{"plan": "done", "key_observations": ["x"]'
        repaired = _repair_truncated_json(text)
        assert repaired is not None
        data = json.loads(repaired)
        assert data["plan"] == "done"

    def test_unterminated_string_in_array(self) -> None:
        text = '{"plan": "ok", "key_observations": ["a", "trun'
        repaired = _repair_truncated_json(text)
        assert repaired is not None
        data = json.loads(repaired)
        assert "a" in data["key_observations"]

    def test_valid_json_returned_unchanged(self) -> None:
        text = '{"plan": "ok"}'
        repaired = _repair_truncated_json(text)
        assert repaired is not None
        assert json.loads(repaired) == {"plan": "ok"}

    def test_empty_string_returns_none(self) -> None:
        assert _repair_truncated_json("") is None

    def test_total_garbage_returns_none(self) -> None:
        assert _repair_truncated_json("hello world no json here at all") is None


# ── _parse_json tests ─────────────────────────────────────────────────


class TestParseJson:
    def test_plain_json(self) -> None:
        data = LatentReasoner._parse_json('{"plan": "x"}')
        assert data == {"plan": "x"}

    def test_json_in_markdown_fences(self) -> None:
        text = '```json\n{"plan": "y"}\n```'
        data = LatentReasoner._parse_json(text)
        assert data["plan"] == "y"

    def test_json_fences_no_language_tag(self) -> None:
        text = '```\n{"plan": "z"}\n```'
        data = LatentReasoner._parse_json(text)
        assert data["plan"] == "z"

    def test_json_embedded_in_prose(self) -> None:
        text = 'Here is my plan:\n{"plan": "embedded"}\nDone.'
        data = LatentReasoner._parse_json(text)
        assert data["plan"] == "embedded"

    def test_truncated_json_repaired(self) -> None:
        text = '{"plan": "trunc'
        data = LatentReasoner._parse_json(text)
        assert data["plan"] == "trunc"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty response"):
            LatentReasoner._parse_json("")

    def test_non_dict_json_raises(self) -> None:
        with pytest.raises(ValueError):
            LatentReasoner._parse_json('["not", "a", "dict"]')


# ── Latent model override test ────────────────────────────────────────


class TestLatentModelOverride:
    @pytest.mark.asyncio
    async def test_uses_latent_model_when_set(self) -> None:
        provider = _make_provider()
        config = _default_config(latent_model="cheap-model")
        reasoner = LatentReasoner(provider, "expensive-model", config)

        await reasoner.initialize(_sample_messages())

        call_kwargs = provider.chat.call_args.kwargs
        assert call_kwargs["model"] == "cheap-model"

    @pytest.mark.asyncio
    async def test_falls_back_to_main_model(self) -> None:
        provider = _make_provider()
        config = _default_config(latent_model=None)
        reasoner = LatentReasoner(provider, "main-model", config)

        await reasoner.initialize(_sample_messages())

        call_kwargs = provider.chat.call_args.kwargs
        assert call_kwargs["model"] == "main-model"


# ── Metrics tests ─────────────────────────────────────────────────────


class TestLatentMetrics:
    def test_initial_metrics_are_zero(self) -> None:
        m = LatentMetrics()
        assert m.total_passes == 0
        assert m.total_latent_tokens == 0
        assert m.converged_early == 0
        assert m.condensation_calls == 0

    @pytest.mark.asyncio
    async def test_refine_increments_total_passes(self) -> None:
        provider = _make_provider({"plan": "p", "key_observations": [], "uncertainties": []})
        reasoner = LatentReasoner(provider, "m", _default_config())
        state = LatentState(plan="p")

        await reasoner.refine(state, _sample_messages())
        await reasoner.refine(state, _sample_messages())

        assert reasoner.metrics.total_passes == 2

    @pytest.mark.asyncio
    async def test_incorporate_increments_condensation_calls(self) -> None:
        provider = _make_provider()
        reasoner = LatentReasoner(provider, "m", _default_config())
        state = LatentState(plan="p")

        await reasoner.incorporate_observations(state, [("exec", "output")])

        assert reasoner.metrics.condensation_calls == 1


# ── Retry logic tests ────────────────────────────────────────────────


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self) -> None:
        provider = AsyncMock()
        provider.chat = AsyncMock(
            side_effect=[ConnectionError("network down"), _make_response({"plan": "recovered"})]
        )
        reasoner = LatentReasoner(provider, "m", _default_config())

        state = await reasoner.initialize(_sample_messages())

        assert state.plan == "recovered"
        assert provider.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_falls_back_after_two_transient_failures(self) -> None:
        provider = AsyncMock()
        provider.chat = AsyncMock(side_effect=ConnectionError("persistent failure"))
        reasoner = LatentReasoner(provider, "m", _default_config())
        fallback = LatentState(plan="fallback plan")

        state = await reasoner.refine(fallback, _sample_messages())

        assert state.plan == "fallback plan"
        assert provider.chat.await_count == 2


# ── Shared orchestration tests ────────────────────────────────────────


class TestRunLatentPasses:
    @pytest.mark.asyncio
    async def test_noop_when_disabled(self) -> None:
        messages = _sample_messages()
        state, action_msgs = await run_latent_passes(None, None, None, messages, 1)
        assert state is None
        assert action_msgs is messages  # identity, not copy

    @pytest.mark.asyncio
    async def test_skips_during_warmup(self) -> None:
        provider = _make_provider({"plan": "should not change"})
        config = _default_config(warmup_threshold=3, max_latent_passes=2)
        reasoner = LatentReasoner(provider, "m", config)
        initial_state = LatentState(plan="original")

        # iteration=2 is within warmup (threshold=3), so should not refine
        state, action_msgs = await run_latent_passes(
            reasoner, initial_state, config, _sample_messages(), iteration=2,
        )

        assert state.plan == "original"  # unchanged — no refinement
        # But inject_state should still work
        assert len(action_msgs) == 3  # original 2 + injected state

    @pytest.mark.asyncio
    async def test_runs_passes_after_warmup(self) -> None:
        provider = _make_provider({"plan": "new plan", "key_observations": [], "uncertainties": []})
        config = _default_config(warmup_threshold=0, max_latent_passes=2)
        reasoner = LatentReasoner(provider, "m", config)
        initial_state = LatentState(plan="old plan")

        # Use a complex enough message so complexity estimation doesn't skip passes
        complex_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a Python script to implement a file parser and search through all the data"},
        ]
        state, _ = await run_latent_passes(
            reasoner, initial_state, config, complex_messages, iteration=1,
        )

        assert state.plan == "new plan"

    @pytest.mark.asyncio
    async def test_early_exit_on_convergence(self) -> None:
        """When plan doesn't change, inner loop should break early."""
        provider = _make_provider({"plan": "stable", "key_observations": ["o"], "uncertainties": []})
        config = _default_config(max_latent_passes=5)
        reasoner = LatentReasoner(provider, "m", config)
        # Start with the same plan — first pass returns "stable", matches input
        initial_state = LatentState(plan="stable")

        complex_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a Python script to implement a file parser and search through all the data"},
        ]
        state, _ = await run_latent_passes(
            reasoner, initial_state, config, complex_messages, iteration=1,
        )

        # Only 1 pass should have run (converged immediately)
        assert reasoner.metrics.total_passes == 1
        assert reasoner.metrics.converged_early == 1


class TestIncorporateToolResults:
    @pytest.mark.asyncio
    async def test_noop_when_disabled(self) -> None:
        state = await incorporate_tool_results(None, None, None, [], 1)
        assert state is None

    @pytest.mark.asyncio
    async def test_updates_state_and_iteration(self) -> None:
        provider = _make_provider({"plan": "updated", "key_observations": ["new"], "uncertainties": []})
        config = _default_config()
        reasoner = LatentReasoner(provider, "m", config)
        initial = LatentState(plan="old", iteration=0)

        result = await incorporate_tool_results(
            reasoner, initial, config, [("exec", "output")], iteration=3,
        )

        assert result.plan == "updated"
        assert result.iteration == 3

    @pytest.mark.asyncio
    async def test_skipped_when_condense_disabled(self) -> None:
        config = _default_config(condense_tool_results=False)
        reasoner = LatentReasoner(AsyncMock(), "m", config)
        initial = LatentState(plan="unchanged")

        result = await incorporate_tool_results(
            reasoner, initial, config, [("exec", "output")], iteration=1,
        )

        assert result.plan == "unchanged"


# ── Per-tool truncation tests ─────────────────────────────────────────


class TestTruncateToolResults:
    def test_per_tool_limits_applied(self) -> None:
        results = [("web_search", "x" * 2000)]  # web_search limit is 800
        text = LatentReasoner._truncate_tool_results(results)
        assert "x" * 800 in text
        assert "x" * 801 not in text

    def test_total_budget_enforced(self) -> None:
        # 4 tools each with 2000-char results: only first ~3 should get content
        results = [("exec", "a" * 2000), ("read_file", "b" * 2000), ("exec", "c" * 2000), ("exec", "d" * 2000)]
        text = LatentReasoner._truncate_tool_results(results)
        assert "budget exhausted" in text

    def test_default_limit_for_unknown_tool(self) -> None:
        results = [("custom_tool", "y" * 3000)]
        text = LatentReasoner._truncate_tool_results(results)
        assert "y" * 1000 in text
        assert "y" * 1001 not in text


# ── Condensation priority tests ───────────────────────────────────────


class TestCondenseMessagesPriority:
    def test_tool_messages_deprioritized(self) -> None:
        messages = [
            {"role": "user", "content": "important question"},
            {"role": "tool", "content": "tool output A"},
            {"role": "tool", "content": "tool output B"},
            {"role": "assistant", "content": "important answer"},
        ]
        # Budget of 2: should pick user + assistant, not tool messages
        result = LatentReasoner._condense_messages(messages, max_messages=2)
        assert "important question" in result
        assert "important answer" in result

    def test_tool_results_shorter_truncation(self) -> None:
        messages = [{"role": "tool", "content": "x" * 1000}]
        result = LatentReasoner._condense_messages(messages, max_messages=5)
        # Tool results use 300-char limit
        assert "x" * 300 in result
        assert "x" * 301 not in result


# ── Plan similarity tests ────────────────────────────────────────────


class TestPlanSimilarity:
    def test_identical_plans(self) -> None:
        assert LatentReasoner._plan_similarity("write code", "write code") == 1.0

    def test_empty_plans(self) -> None:
        assert LatentReasoner._plan_similarity("", "") == 1.0

    def test_completely_different(self) -> None:
        sim = LatentReasoner._plan_similarity("write python code", "buy groceries tomorrow")
        assert sim < 0.5

    def test_similar_plans_above_threshold(self) -> None:
        a = "Write a Python script to parse JSON files"
        b = "Write a Python script to parse JSON data files"
        sim = LatentReasoner._plan_similarity(a, b)
        assert sim >= 0.85

    def test_moderate_similarity(self) -> None:
        a = "Implement user authentication with JWT"
        b = "Implement user auth with JSON Web Tokens and session management"
        sim = LatentReasoner._plan_similarity(a, b)
        assert 0.3 < sim < 0.9


# ── Plan drift detection tests ───────────────────────────────────────


class TestPlanDriftDetection:
    def test_detects_meeting_drift(self) -> None:
        plan = "Step 1: Write the code. Step 2: Schedule meeting with team to review."
        keywords = ["schedule meeting", "slack webhook", "team sync"]
        assert LatentReasoner._detect_plan_drift(plan, keywords) is True

    def test_detects_slack_drift(self) -> None:
        plan = "Configure Slack webhook for notifications"
        keywords = ["schedule meeting", "slack webhook"]
        assert LatentReasoner._detect_plan_drift(plan, keywords) is True

    def test_no_drift_on_topic(self) -> None:
        plan = "Read the file, parse the JSON, write the output"
        keywords = ["schedule meeting", "slack webhook", "team sync"]
        assert LatentReasoner._detect_plan_drift(plan, keywords) is False

    def test_case_insensitive(self) -> None:
        plan = "SCHEDULE MEETING with stakeholders"
        keywords = ["schedule meeting"]
        assert LatentReasoner._detect_plan_drift(plan, keywords) is True

    def test_empty_keywords(self) -> None:
        plan = "Do anything at all"
        assert LatentReasoner._detect_plan_drift(plan, []) is False


# ── Budget exhaustion tests ──────────────────────────────────────────


class TestBudgetExhaustion:
    @pytest.mark.asyncio
    async def test_token_budget_stops_refinement(self) -> None:
        """When token budget is exhausted, refinement should be skipped."""
        provider = _make_provider({"plan": "new plan", "key_observations": [], "uncertainties": []})
        config = _default_config(max_latent_passes=5, max_total_latent_tokens=100)
        reasoner = LatentReasoner(provider, "m", config)
        # Simulate already-exhausted budget
        reasoner.metrics.total_latent_tokens = 100
        state = LatentState(plan="original")

        result_state, _ = await run_latent_passes(
            reasoner, state, config, _sample_messages(), iteration=1,
        )

        # Plan should be unchanged since refinement was skipped
        assert result_state.plan == "original"
        assert reasoner.metrics.total_passes == 0

    @pytest.mark.asyncio
    async def test_refinement_cap_stops_refinement(self) -> None:
        """When refinement cap is reached, refinement should be skipped."""
        provider = _make_provider({"plan": "new plan", "key_observations": [], "uncertainties": []})
        config = _default_config(max_latent_passes=5, max_total_refinements=10)
        reasoner = LatentReasoner(provider, "m", config)
        # Simulate already-exhausted cap
        reasoner.metrics.total_passes = 10
        state = LatentState(plan="original")

        result_state, _ = await run_latent_passes(
            reasoner, state, config, _sample_messages(), iteration=1,
        )

        assert result_state.plan == "original"

    @pytest.mark.asyncio
    async def test_drift_reverts_plan(self) -> None:
        """When plan drift is detected, the plan should revert to previous."""
        provider = _make_provider({
            "plan": "Write code then schedule meeting with team",
            "key_observations": ["obs"],
            "uncertainties": [],
        })
        config = _default_config(
            max_latent_passes=3,
            plan_drift_keywords=["schedule meeting"],
        )
        reasoner = LatentReasoner(provider, "m", config)
        state = LatentState(plan="Write the code")

        result_state, _ = await run_latent_passes(
            reasoner, state, config, _sample_messages(), iteration=1,
        )

        # Should revert to previous plan
        assert result_state.plan == "Write the code"

    @pytest.mark.asyncio
    async def test_similarity_convergence(self) -> None:
        """Plans with high similarity should trigger early convergence."""
        provider = _make_provider({
            "plan": "Write a Python script to parse files",
            "key_observations": [],
            "uncertainties": [],
        })
        config = _default_config(
            max_latent_passes=5,
            convergence_similarity=0.85,
        )
        reasoner = LatentReasoner(provider, "m", config)
        state = LatentState(plan="Write a Python script to parse data files")

        await run_latent_passes(
            reasoner, state, config,
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "Write a parser for JSON files and data transformation"}],
            iteration=1,
        )

        # Should converge after 1 pass since plans are very similar
        assert reasoner.metrics.converged_early >= 1


# ── Adaptive complexity estimation tests ─────────────────────────────


class TestEstimateComplexity:
    def test_short_greeting_is_zero(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        assert LatentReasoner._estimate_complexity(messages) == 0

    def test_empty_messages_is_zero(self) -> None:
        assert LatentReasoner._estimate_complexity([]) == 0

    def test_simple_question(self) -> None:
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        c = LatentReasoner._estimate_complexity(messages)
        assert c <= 1

    def test_complex_multi_action(self) -> None:
        messages = [{"role": "user", "content": "Write a script to create a new database, implement the migration, and deploy the service with proper configuration and search indexing"}]
        c = LatentReasoner._estimate_complexity(messages)
        assert c >= 2

    def test_single_action_moderate(self) -> None:
        messages = [{"role": "user", "content": "Please fix the bug in the authentication module that causes login failures"}]
        c = LatentReasoner._estimate_complexity(messages)
        assert c >= 1

    @pytest.mark.asyncio
    async def test_trivial_query_skips_passes(self) -> None:
        """A trivial query (complexity=0) should skip latent passes entirely."""
        provider = _make_provider({"plan": "new", "key_observations": [], "uncertainties": []})
        config = _default_config(max_latent_passes=3)
        reasoner = LatentReasoner(provider, "m", config)
        state = LatentState(plan="original")

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        result_state, _ = await run_latent_passes(
            reasoner, state, config, messages, iteration=1,
        )

        # No refinement calls should have been made
        assert reasoner.metrics.total_passes == 0
        assert result_state.plan == "original"


# ── New config fields tests ──────────────────────────────────────────


class TestNewLatentLoopConfigFields:
    def test_new_defaults(self) -> None:
        config = LatentLoopConfig()
        assert config.max_total_latent_tokens == 50_000
        assert config.max_total_refinements == 30
        assert config.convergence_similarity == 0.85
        assert isinstance(config.plan_drift_keywords, list)
        assert len(config.plan_drift_keywords) > 0

    def test_drift_keywords_overridable(self) -> None:
        config = LatentLoopConfig(plan_drift_keywords=["custom keyword"])
        assert config.plan_drift_keywords == ["custom keyword"]

    def test_convergence_similarity_customizable(self) -> None:
        config = LatentLoopConfig(convergence_similarity=0.95)
        assert config.convergence_similarity == 0.95


# ── Registry enhanced error messages tests ───────────────────────────


class TestRegistryEnhancedErrors:
    @pytest.mark.asyncio
    async def test_error_includes_tool_list(self) -> None:
        from nanobot.agent.tools.registry import ToolRegistry
        from nanobot.agent.tools.base import Tool

        class FakeTool(Tool):
            @property
            def name(self) -> str:
                return "fake_tool"
            @property
            def description(self) -> str:
                return "A fake tool"
            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}
            async def execute(self, **kwargs) -> str:
                return "ok"

        registry = ToolRegistry()
        registry.register(FakeTool())
        result = await registry.execute("nonexistent_tool", {})
        assert "fake_tool" in result
        assert "not found" in result.lower() or "not found" in result

    @pytest.mark.asyncio
    async def test_escalating_error_on_repeat(self) -> None:
        from nanobot.agent.tools.registry import ToolRegistry
        from nanobot.agent.tools.base import Tool

        class FakeTool(Tool):
            @property
            def name(self) -> str:
                return "real_tool"
            @property
            def description(self) -> str:
                return "A real tool"
            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}
            async def execute(self, **kwargs) -> str:
                return "ok"

        registry = ToolRegistry()
        registry.register(FakeTool())

        # First two attempts: normal error
        await registry.execute("bad_tool", {})
        await registry.execute("bad_tool", {})
        # Third attempt: escalated
        result = await registry.execute("bad_tool", {})
        assert "STOP" in result
        assert "3 times" in result

    def test_get_error_stats(self) -> None:
        from nanobot.agent.tools.registry import ToolRegistry

        registry = ToolRegistry()
        # Trigger some errors synchronously by accessing internal state
        registry._invalid_tool_attempts["bad_tool"] = 5
        stats = registry.get_error_stats()
        assert stats == {"bad_tool": 5}
