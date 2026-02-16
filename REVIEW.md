# Review: Latent-Looped Iterations — Findings & Improvement Plan

## Summary

The latent-looped inference feature (PR #1, commits `bf0fb21`–`bd487af`) adds an
inner reasoning loop to the ReAct agent.  Before each action decision, the model
performs configurable lightweight LLM calls to refine a compact state (plan,
observations, uncertainties).  The design is clean, opt-in, and well-tested.

This document catalogs concrete issues found during review and proposes
prioritized improvements.

---

## 1. Code Duplication Between `loop.py` and `subagent.py`

**Severity: High (maintainability)**

The latent loop integration is copy-pasted between `AgentLoop._run_agent_loop()`
(`loop.py:184-248`) and `SubagentManager._run_subagent()` (`subagent.py:131-222`).
The two copies are nearly identical but will inevitably drift as one gets fixed
and the other doesn't.  The `inject()` mutation bug (`b7e1d33`) was fixed in
`latent.py` itself, but a future behavioral change to the orchestration (e.g.,
early-exit logic, adaptive pass count) would need to be applied in two places.

**Improvement:** Extract the latent loop orchestration into a shared helper —
either a method on `LatentReasoner` itself or a standalone async function — that
both `loop.py` and `subagent.py` call.  For example:

```python
# In latent.py
async def run_latent_passes(
    reasoner: LatentReasoner | None,
    state: LatentState | None,
    config: LatentLoopConfig | None,
    messages: list[dict],
    iteration: int,
) -> tuple[LatentState | None, list[dict]]:
    """Run inner latent passes and return (updated_state, action_messages)."""
    if reasoner and state and iteration > config.warmup_threshold:
        for _ in range(config.max_latent_passes):
            state = await reasoner.refine(state, messages)
    action_messages = messages
    if reasoner and state and config.inject_state:
        action_messages = reasoner.inject(state, messages)
    return state, action_messages
```

---

## 2. No Adaptive/Early-Exit Logic in the Inner Loop

**Severity: Medium (efficiency)**

The inner loop always runs exactly `max_latent_passes` times, even if the plan
has already stabilized (identical plan across consecutive passes).  For simple
queries this wastes tokens and adds latency.

**Improvement:** Add convergence detection.  After each `refine()` call, compare
the new plan to the previous one.  If the plan text is unchanged (or the
edit distance is below a threshold), break out of the inner loop early.

```python
for _ in range(config.max_latent_passes):
    new_state = await reasoner.refine(state, messages)
    if new_state.plan == state.plan:
        state = new_state
        break  # Plan has converged
    state = new_state
```

This should also be exposed as a config flag (`early_exit: bool = True`) so users
can disable it if they want deterministic pass counts.

---

## 3. Latent Reasoner Uses the Same (Expensive) Model

**Severity: Medium (cost)**

Both latent passes and action passes use `self.model` — the same model configured
for tool-calling.  Latent passes are short, no-tool reasoning calls that could use
a smaller/cheaper model (e.g., `claude-haiku` instead of `claude-opus`) without
quality loss on planning tasks.

**Improvement:** Add an optional `latent_model` field to `LatentLoopConfig`:

```python
class LatentLoopConfig(BaseModel):
    ...
    latent_model: str | None = None  # Falls back to the agent's main model
```

The `LatentReasoner.__init__` would accept this override and use it for all inner
calls.  This could cut latent-loop costs by 10-20x for providers with tiered
pricing.

---

## 4. `_condense_messages` Is Too Aggressive

**Severity: Medium (quality)**

`_condense_messages()` takes only the last N messages (default 4-6) and truncates
each to 500 characters.  For conversations with long tool outputs, the condensed
context may lose critical details that the latent loop needs to make good planning
decisions.

**Improvements:**
- Weight recent assistant reasoning and user messages higher than tool outputs
  (tool results are already in the latent state via `incorporate_observations`).
- Make `max_messages` and the per-message truncation limit configurable.
- Consider a token-aware truncation strategy rather than a character-based one.

---

## 5. No Observability / Metrics for Latent Passes

**Severity: Medium (operations)**

The latent loop logs at `debug` level but there's no structured observability.
Operators cannot easily answer: "How many latent tokens did this request use?",
"Did the plan converge?", "How many passes were actually needed?"

**Improvement:** Track and return latent-loop metrics alongside the response:

```python
@dataclass
class LatentMetrics:
    total_passes: int = 0
    total_latent_tokens: int = 0
    converged_early: bool = False
    initialization_tokens: int = 0
    condensation_calls: int = 0
```

These can be logged at `info` level at the end of each outer iteration and
optionally surfaced in the API response metadata.

---

## 6. Missing Error Recovery in Inner Loop

**Severity: Medium (resilience)**

If a latent LLM call fails (network error, rate limit, provider outage),
`_call_for_state` catches the exception and returns the fallback state.  This is
correct for parse failures, but for transient network errors the entire latent
budget for that iteration is silently lost.  There's no retry logic.

**Improvement:** Add a single retry with backoff for transient errors (timeout,
connection reset, 429/503 status codes) before falling back to the previous state.
The retry should be capped at 1 attempt to avoid compounding latency.

---

## 7. Tool Result Truncation Is Uniform

**Severity: Low (quality)**

In `incorporate_observations()`, every tool result is truncated to 1000
characters regardless of the tool type.  A `read_file` result losing 90% of its
content is much more damaging than truncating a `web_search` snippet.

**Improvement:** Allow per-tool truncation limits, or use a smarter strategy:
- Allocate a total budget (e.g., 4000 chars) across all tool results proportionally
- Prioritize the last tool result (most likely to be the "answer")
- Let `exec` results keep more (they tend to be short but information-dense)

---

## 8. `inject()` Always Uses a System Message

**Severity: Low (compatibility)**

Some LLM providers don't support multiple system messages or behave unexpectedly
when they appear mid-conversation.  The current approach inserts the latent state
as a second `system` message at index 1.

**Improvement:** Make the injection role configurable:

```python
class LatentLoopConfig(BaseModel):
    ...
    inject_role: str = "system"  # "system", "user", or "developer"
```

For providers that don't handle multi-system well, users can switch to `"user"`
injection (wrapped in XML tags like `<latent-state>...</latent-state>` to
distinguish it from actual user input).

---

## 9. `LatentState.from_dict` Doesn't Validate Types

**Severity: Low (robustness)**

`from_dict` trusts that `data["key_observations"]` is a list and
`data["plan"]` is a string.  If the LLM returns malformed JSON (e.g.,
`"key_observations": "just a string"`), the state will be created with
wrong types that could cause downstream errors.

**Improvement:** Add type coercion in `from_dict`:

```python
@classmethod
def from_dict(cls, data: dict) -> LatentState:
    plan = str(data.get("plan", ""))
    obs = data.get("key_observations", [])
    if isinstance(obs, str):
        obs = [obs]
    unc = data.get("uncertainties", [])
    if isinstance(unc, str):
        unc = [unc]
    return cls(plan=plan, key_observations=obs, uncertainties=unc)
```

---

## 10. Test Coverage Gaps

**Severity: Low (testing)**

The test suite (`test_latent.py`, 305 lines) is solid but has some gaps:

| Gap | Suggested Test |
|-----|---------------|
| Integration with `_run_agent_loop` | End-to-end test with mocked provider verifying latent calls happen in the right order |
| `warmup_threshold` behavior | Test that latent passes are skipped for iterations <= threshold |
| Subagent latent integration | Test that subagent loop also runs latent passes |
| Convergence (after early-exit is added) | Test that inner loop breaks when plan is stable |
| Error recovery (after retry is added) | Test that transient errors trigger retry |
| Markdown fence stripping edge cases | Test ```` ```json\n{...}\n``` ```` and ```` ``` ```` with no language tag |

---

## Priority Matrix

| # | Improvement | Effort | Impact | Priority |
|---|-------------|--------|--------|----------|
| 1 | Extract shared latent orchestration | Small | High | **P0** |
| 2 | Early-exit on plan convergence | Small | Medium | **P1** |
| 3 | Separate `latent_model` config | Small | Medium | **P1** |
| 5 | Latent metrics/observability | Medium | Medium | **P1** |
| 6 | Retry transient errors in inner loop | Small | Medium | **P1** |
| 4 | Smarter message condensation | Medium | Medium | **P2** |
| 7 | Per-tool truncation budgets | Medium | Low | **P2** |
| 9 | Type validation in `from_dict` | Small | Low | **P2** |
| 10 | Expand test coverage | Medium | Low | **P2** |
| 8 | Configurable injection role | Small | Low | **P3** |

---

## What's Done Well

- **Opt-in by default** — zero behavior change for existing users.
- **Non-mutating `inject()`** — the bug was caught and fixed with a test.
- **Fallback on parse failure** — graceful degradation instead of crashes.
- **Clean separation** — all latent logic lives in one 280-line module.
- **Comprehensive config** — every knob is exposed and documented.
- **Good test coverage** — 19 tests covering serialization, reasoning, injection,
  and config defaults.
