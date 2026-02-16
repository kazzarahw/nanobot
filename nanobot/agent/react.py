"""Shared ReAct loop implementation for agents and subagents."""

import json
from typing import Any

from loguru import logger

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider


def _build_reflect_prompt(has_errors: bool, error_summaries: list[str]) -> str:
    """Build a contextual reflect prompt based on tool execution results."""
    if has_errors:
        summary = "; ".join(error_summaries[:3])
        return (
            f"Tool calls failed: {summary}. "
            "Fix errors or try a different approach. "
            "Do NOT repeat the same failing call."
        )
    return (
        "Tool calls succeeded. "
        "Provide your final response if done, or proceed with the next action."
    )


async def run_react_loop(
    provider: LLMProvider,
    model: str,
    tools: ToolRegistry,
    initial_messages: list[dict],
    max_iterations: int,
    temperature: float,
    max_tokens: int,
    latent_config: Any | None,
    enable_circuit_breaker: bool = True,
    log_prefix: str = "",
    context_builder: Any | None = None,
) -> tuple[str | None, list[str]]:
    """
    Run the ReAct loop (reasoning + action).

    When latent-looped inference is enabled, each outer iteration may
    include inner *latent passes* — lightweight LLM calls (no tools,
    reduced token budget) that refine a compact reasoning state before
    the model commits to an action.

    Args:
        provider: LLM provider for making API calls.
        model: Model name to use.
        tools: Tool registry for available tools.
        initial_messages: Starting messages for the LLM conversation.
        max_iterations: Maximum number of iterations.
        temperature: Temperature for sampling.
        max_tokens: Maximum tokens per response.
        latent_config: Latent loop configuration (or None to disable).
        enable_circuit_breaker: Whether to use error circuit breaker.
        log_prefix: Prefix for log messages (e.g., "Subagent [abc123] ").
        context_builder: Optional context builder for message formatting.
                        If None, messages are appended directly.

    Returns:
        Tuple of (final_content, list_of_tools_used).
    """
    from nanobot.agent.latent import (
        LatentReasoner,
        incorporate_tool_results,
        run_latent_passes,
    )

    messages = initial_messages
    iteration = 0
    final_content = None
    tools_used: list[str] = []
    consecutive_error_count = 0
    recent_errors: list[str] = []

    # Initialize latent reasoner if enabled
    latent: LatentReasoner | None = None
    latent_state = None
    if latent_config and latent_config.enabled:
        latent = LatentReasoner(provider, model, latent_config)
        latent_state = await latent.initialize(messages)

    while iteration < max_iterations:
        iteration += 1

        # ── Inner loop: latent reasoning passes ──
        latent_state, action_messages = await run_latent_passes(
            latent, latent_state, latent_config, messages, iteration,
        )

        response = await provider.chat(
            messages=action_messages,
            tools=tools.get_definitions(),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if response.has_tool_calls:
            tool_call_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in response.tool_calls
            ]

            # Add assistant message (with or without context builder)
            if context_builder:
                messages = context_builder.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    }
                )

            tool_results_for_latent: list[tuple[str, str]] = []
            has_errors = False
            error_summaries: list[str] = []
            for tool_call in response.tool_calls:
                tools_used.append(tool_call.name)
                args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                logger.info(f"{log_prefix}Tool call: {tool_call.name}({args_str[:200]})")
                result = await tools.execute(tool_call.name, tool_call.arguments)

                # Add tool result (with or without context builder)
                if context_builder:
                    messages = context_builder.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        }
                    )

                tool_results_for_latent.append((tool_call.name, result))
                if result.startswith("Error:"):
                    has_errors = True
                    error_summaries.append(f"{tool_call.name}: {result[:120]}")

            # ── Update latent state with tool observations ──
            latent_state = await incorporate_tool_results(
                latent, latent_state, latent_config,
                tool_results_for_latent, iteration,
            )

            # ── Contextual reflect prompt ──
            reflect = _build_reflect_prompt(has_errors, error_summaries)
            messages.append({"role": "user", "content": reflect})

            # ── Error circuit breaker ──
            if enable_circuit_breaker and has_errors:
                consecutive_error_count += 1
                recent_errors.extend(error_summaries)
                if consecutive_error_count >= 3:
                    valid_tools = ", ".join(sorted(tools.tool_names))
                    intervention = (
                        f"STOP. You have had {consecutive_error_count} consecutive "
                        f"iterations with errors. Recent errors:\n"
                        + "\n".join(f"- {e}" for e in recent_errors[-6:])
                        + f"\n\nAvailable tools: {valid_tools}\n"
                        "You MUST try a completely different approach. "
                        "Do NOT repeat the same failing calls."
                    )
                    messages.append({"role": "user", "content": intervention})
                    logger.warning(f"{log_prefix}Circuit breaker triggered after {consecutive_error_count} error iterations")
                    consecutive_error_count = 0
                    recent_errors.clear()
            else:
                consecutive_error_count = 0
                recent_errors.clear()
        else:
            final_content = response.content
            break

    if latent:
        m = latent.metrics
        logger.info(
            f"{log_prefix}Latent loop stats: passes={m.total_passes}, "
            f"tokens={m.total_latent_tokens}, converged_early={m.converged_early}, "
            f"condensations={m.condensation_calls}"
        )

    return final_content, tools_used
