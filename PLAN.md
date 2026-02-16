# Latent-Looped Inference: Design Plan

## Overview

Latent-Looped Inference adds an **inner reasoning loop** inside each step of the
existing ReAct loop.  Before the agent commits to an action (tool call) or a
final answer, it performs configurable *latent passes* — short LLM calls that
refine a condensed state representation without executing tools.  This gives the
model more compute-per-step and keeps the context window manageable as
conversations grow.

```
                          ┌──────────────────────────┐
                          │    OUTER LOOP (ReAct)     │
                          │   max_iterations = 20     │
                          │                           │
  ┌──────────────────────►│  1. Build messages        │
  │                       │  2. ── INNER LOOP ──────  │
  │                       │  │  a. Condense state     │
  │                       │  │  b. Latent LLM call    │
  │                       │  │  c. Update state       │
  │                       │  │  d. Repeat N times     │
  │                       │  └────────────────────    │
  │                       │  3. Action LLM call       │
  │                       │     (with latent state)   │
  │                       │  4. Execute tool calls    │
  │                       │  5. Update latent state   │
  │                       │     with observations     │
  │                       └──────────┬───────────────┘
  │  has_tool_calls                  │ no tool calls
  └──────────────────────────────────┘      │
                                            ▼
                                      Final answer
```

## Current Architecture (what we're modifying)

### ReAct loop — `loop.py:149-205`

```python
async def _run_agent_loop(self, initial_messages):
    messages = initial_messages      # full history, grows each iteration
    iteration = 0
    while iteration < self.max_iterations:
        iteration += 1
        response = await self.provider.chat(messages, tools=..., ...)
        if response.has_tool_calls:
            # append assistant + tool results + reflection prompt
            ...
        else:
            final_content = response.content
            break
    return final_content, tools_used
```

**Problem this addresses:** Every LLM call receives the full, ever-growing
message list.  The model gets one shot per iteration to reason about tool
results, plan, and decide on an action.  For complex multi-step tasks the model
can lose coherence or make premature decisions because it lacks dedicated
reasoning time.

## Detailed Design

### 1. New module: `nanobot/agent/latent.py`

Houses all latent-loop logic in a single cohesive module.

```python
@dataclass
class LatentState:
    """Compact reasoning state carried across ReAct iterations."""
    plan: str                    # current high-level plan / strategy
    key_observations: list[str]  # extracted facts from tool results
    uncertainties: list[str]     # open questions / unknowns
    iteration: int               # outer-loop iteration counter
    refinement_count: int        # total latent passes performed

class LatentReasoner:
    """Manages the inner latent-reasoning loop."""

    def __init__(self, provider, model, config):
        ...

    async def initialize(self, messages) -> LatentState:
        """Create initial latent state from the starting context."""

    async def refine(self, state, messages) -> LatentState:
        """Run one latent reasoning pass — update plan/observations."""

    async def incorporate_observations(self, state, tool_results) -> LatentState:
        """Condense tool results into the latent state."""

    def inject(self, state, messages) -> list[dict]:
        """Prepend a latent-state summary message into the message list."""
```

#### Key design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| State format | Structured text (plan + observations + uncertainties) | LLMs work best with natural language; structured fields keep it parseable |
| Inner-loop LLM calls | Same provider, **no tools**, reduced `max_tokens` | Cheaper/faster calls focused purely on reasoning |
| State injection | Inserted as a `system` message after the main system prompt | Keeps it visible to the model without polluting conversation history |
| Condensation | LLM summarizes tool results into bullet points | More flexible than heuristic truncation |

### 2. Configuration: `nanobot/config/schema.py`

New Pydantic model added to `AgentDefaults`:

```python
class LatentLoopConfig(BaseModel):
    """Latent-looped inference configuration."""
    enabled: bool = False                    # opt-in, zero behavior change by default
    max_latent_passes: int = 3               # inner-loop iterations per ReAct step
    latent_max_tokens: int = 1024            # token budget for each latent call
    latent_temperature: float = 0.3          # lower temp for focused reasoning
    condense_tool_results: bool = True       # summarize tool output into state
    inject_state: bool = True                # include state in action LLM call
    warmup_threshold: int = 0               # skip latent loops for the first N outer iterations
```

Added as a field on `AgentDefaults`:

```python
class AgentDefaults(BaseModel):
    ...
    latent_loop: LatentLoopConfig = Field(default_factory=LatentLoopConfig)
```

### 3. Modified ReAct loop: `nanobot/agent/loop.py`

The `_run_agent_loop` method gains an optional inner loop:

```python
async def _run_agent_loop(self, initial_messages):
    messages = initial_messages
    iteration = 0
    final_content = None
    tools_used = []

    # Initialize latent reasoner if enabled
    latent = None
    latent_state = None
    if self.latent_config and self.latent_config.enabled:
        latent = LatentReasoner(self.provider, self.model, self.latent_config)
        latent_state = await latent.initialize(messages)

    while iteration < self.max_iterations:
        iteration += 1

        # ── INNER LOOP: latent reasoning passes ──
        if latent and latent_state and iteration > self.latent_config.warmup_threshold:
            for _ in range(self.latent_config.max_latent_passes):
                latent_state = await latent.refine(latent_state, messages)

        # Inject latent state into context for action decision
        action_messages = messages
        if latent and latent_state and self.latent_config.inject_state:
            action_messages = latent.inject(latent_state, messages)

        # ── ACTION: call LLM with tools ──
        response = await self.provider.chat(
            messages=action_messages,
            tools=self.tools.get_definitions(),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if response.has_tool_calls:
            # ... existing tool execution logic (unchanged) ...
            messages = self.context.add_assistant_message(...)
            for tool_call in response.tool_calls:
                result = await self.tools.execute(...)
                messages = self.context.add_tool_result(...)

            # ── UPDATE: incorporate tool results into latent state ──
            if latent and latent_state and self.latent_config.condense_tool_results:
                tool_results = [...]  # collect name+result pairs
                latent_state = await latent.incorporate_observations(
                    latent_state, tool_results
                )
                latent_state.iteration = iteration

            messages.append({"role": "user", "content": "Reflect on the results and decide next steps."})
        else:
            final_content = response.content
            break

    return final_content, tools_used
```

### 4. Latent prompts

The inner-loop LLM calls use focused prompts:

**Refinement prompt** (used in `refine()`):
```
You are in a planning phase. You will NOT execute any tools right now.

## Current State
Plan: {state.plan}
Key observations: {state.key_observations}
Uncertainties: {state.uncertainties}

## Recent context
{last few messages, condensed}

Refine your plan. Update observations and uncertainties.
Respond with JSON: {"plan": "...", "key_observations": [...], "uncertainties": [...]}
```

**Condensation prompt** (used in `incorporate_observations()`):
```
You just executed tools. Condense the results into updated state.

## Tool Results
{tool_name}: {result_summary}

## Previous State
{state}

Update the state. Keep only actionable information.
Respond with JSON: {"plan": "...", "key_observations": [...], "uncertainties": [...]}
```

### 5. Changes to `AgentLoop.__init__`

```python
def __init__(self, ..., latent_config=None):
    ...
    self.latent_config = latent_config
```

The `latent_config` parameter is threaded from `Config.agents.defaults.latent_loop`
through the existing initialization chain in `cli/commands.py` and channel
startup code.

### 6. Changes to `LLMResponse` / `providers/base.py`

No changes needed. The latent calls use the same `provider.chat()` interface
with `tools=None`. The existing `LLMResponse` dataclass already has all fields
we need (`content`, `usage`, `reasoning_content`).

### 7. Changes to `ContextBuilder` — none

The `inject()` method on `LatentReasoner` manipulates the message list directly
(inserting a system message). `ContextBuilder` doesn't need modification.

## Files Changed (Summary)

| File | Change Type | Description |
|------|-------------|-------------|
| `nanobot/agent/latent.py` | **NEW** | `LatentState`, `LatentReasoner` classes |
| `nanobot/agent/loop.py` | MODIFY | Add inner loop, initialize/update latent state |
| `nanobot/config/schema.py` | MODIFY | Add `LatentLoopConfig`, field on `AgentDefaults` |
| `nanobot/cli/commands.py` | MODIFY | Thread `latent_config` into `AgentLoop` constructor |
| `nanobot/agent/subagent.py` | MODIFY | Thread `latent_config` into subagent `AgentLoop` |
| `tests/test_latent.py` | **NEW** | Unit tests for latent reasoner |

## Step-by-Step Implementation Order

1. **`config/schema.py`** — Add `LatentLoopConfig` and field on `AgentDefaults`
2. **`agent/latent.py`** — Implement `LatentState` and `LatentReasoner`
3. **`agent/loop.py`** — Wire latent reasoner into `_run_agent_loop` and `__init__`
4. **`cli/commands.py`** — Pass `latent_config` from config to `AgentLoop`
5. **`agent/subagent.py`** — Pass `latent_config` to subagent loops
6. **`tests/test_latent.py`** — Tests for initialization, refinement, condensation, injection

## Risk Analysis

| Risk | Mitigation |
|------|------------|
| Extra LLM calls increase latency/cost | `enabled: false` by default; `warmup_threshold` skips early iterations; `max_latent_passes` caps inner loop; inner calls use small `latent_max_tokens` |
| Latent state drifts from reality | `incorporate_observations()` re-grounds state after every tool execution |
| JSON parsing failures in latent calls | Fallback: keep previous state unchanged on parse error; log warning |
| Context window overflow from injection | `inject()` enforces a character limit on the state summary; truncates oldest observations first |
| Breaking existing behavior | Feature is entirely opt-in (`enabled: false`); when disabled, zero code paths change |

## Token Budget Analysis

With default settings (`max_latent_passes=3`, `latent_max_tokens=1024`):

- **Per outer iteration**: up to 3 x ~1024 = ~3,072 extra completion tokens
- **20 outer iterations max**: up to ~61,440 extra tokens in worst case
- **Typical scenario** (5 outer iterations): ~15,360 extra tokens
- **Cost control**: `warmup_threshold` can skip latent loops on simple first steps

Compared to the existing loop which can use `20 x 8192 = 163,840` completion
tokens in the worst case, the latent overhead is modest (~19-37% increase in
typical scenarios).
