"""Token-budgeted context window preparation for action LLM calls."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class TrimResult:
    """Result of preparing an action message window."""

    messages: list[dict[str, Any]]
    before_tokens: int
    after_tokens: int
    dropped_internal: int
    dropped_other: int


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Approximate token usage for provider-agnostic budgeting."""
    total = 0
    for msg in messages:
        total += _estimate_message_tokens(msg)
    return total


def prepare_action_messages(
    messages: list[dict[str, Any]],
    *,
    max_input_tokens: int | None,
    reserve_tokens: int,
) -> TrimResult:
    """Build a token-budgeted sliding window for action decisions."""
    sanitized = [_sanitize_message(msg) for msg in messages]
    before_tokens = estimate_tokens(sanitized)

    if max_input_tokens is None:
        return TrimResult(
            messages=sanitized,
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            dropped_internal=0,
            dropped_other=0,
        )

    budget = max(256, max_input_tokens - max(0, reserve_tokens))
    if before_tokens <= budget:
        return TrimResult(
            messages=sanitized,
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            dropped_internal=0,
            dropped_other=0,
        )

    keep_indexes: set[int] = set()
    n = len(messages)

    # Always keep the first system prompt if present.
    if n and messages[0].get("role") == "system":
        keep_indexes.add(0)

    # Preserve latest latent state system message if present.
    latent_idx = _find_last_latent_state_index(messages)
    if latent_idx is not None:
        keep_indexes.add(latent_idx)

    # Preserve latest real user turn (ignoring internal loop prompts).
    latest_user_idx = _find_last_real_user_index(messages)
    if latest_user_idx is not None:
        keep_indexes.add(latest_user_idx)

    # Preserve latest tool-call chain (assistant tool call + subsequent tools).
    chain = _find_latest_tool_chain(messages)
    keep_indexes.update(chain)

    selected: list[int] = sorted(keep_indexes)
    selected_set = set(selected)

    # Fill from newest to oldest, skipping internal prompts first.
    for idx in range(n - 1, -1, -1):
        if idx in selected_set:
            continue
        msg = messages[idx]
        if _is_internal_prompt(msg):
            continue
        selected.append(idx)
        selected_set.add(idx)
        candidate = [_sanitize_message(messages[i]) for i in sorted(selected)]
        if estimate_tokens(candidate) > budget:
            selected.remove(idx)
            selected_set.remove(idx)

    # If still over budget, drop oldest optional messages.
    selected = sorted(selected)
    final_messages = [_sanitize_message(messages[i]) for i in selected]
    while estimate_tokens(final_messages) > budget:
        dropped = False
        for i, idx in enumerate(selected):
            if idx in keep_indexes:
                continue
            del selected[i]
            final_messages = [_sanitize_message(messages[j]) for j in selected]
            dropped = True
            break
        if not dropped:
            final_messages = _force_truncate_large_contents(final_messages, budget)
            break

    selected_set = set(selected)
    dropped_internal = 0
    dropped_other = 0
    for idx, msg in enumerate(messages):
        if idx in selected_set:
            continue
        if _is_internal_prompt(msg):
            dropped_internal += 1
        else:
            dropped_other += 1

    after_tokens = estimate_tokens(final_messages)
    return TrimResult(
        messages=final_messages,
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        dropped_internal=dropped_internal,
        dropped_other=dropped_other,
    )


def _is_internal_prompt(msg: dict[str, Any]) -> bool:
    return bool(msg.get("_internal"))


def _find_last_latent_state_index(messages: list[dict[str, Any]]) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and "## Latent Reasoning State" in content:
            return idx
    return None


def _find_last_real_user_index(messages: list[dict[str, Any]]) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "user" and not _is_internal_prompt(msg):
            return idx
    return None


def _find_latest_tool_chain(messages: list[dict[str, Any]]) -> list[int]:
    assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            assistant_idx = idx
            break
    if assistant_idx is None:
        return []

    chain = [assistant_idx]
    idx = assistant_idx + 1
    while idx < len(messages) and messages[idx].get("role") == "tool":
        chain.append(idx)
        idx += 1
    return chain


def _sanitize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Remove internal fields before sending provider requests."""
    role = msg.get("role")
    content = msg.get("content", "")
    if role in ("system", "user"):
        result = {"role": role, "content": content}
        if msg.get("name"):
            result["name"] = msg["name"]
        return result
    if role == "assistant":
        result: dict[str, Any] = {"role": role, "content": content}
        if msg.get("tool_calls"):
            result["tool_calls"] = msg["tool_calls"]
        if msg.get("reasoning_content"):
            result["reasoning_content"] = msg["reasoning_content"]
        return result
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": msg.get("tool_call_id", ""),
            "name": msg.get("name", ""),
            "content": content,
        }
    return {"role": role or "user", "content": content}


def _estimate_message_tokens(msg: dict[str, Any]) -> int:
    content = msg.get("content", "")
    if isinstance(content, list):
        text_content = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    text_content.append(text)
        content = " ".join(text_content)
    if not isinstance(content, str):
        content = str(content)

    # Rough overhead for role/formatting/tool metadata.
    overhead = 12
    if msg.get("tool_calls"):
        overhead += 24
    if msg.get("role") == "tool":
        overhead += 8
    return overhead + max(1, math.ceil(len(content) / 4))


def _force_truncate_large_contents(
    messages: list[dict[str, Any]], budget: int
) -> list[dict[str, Any]]:
    """As a final guard, truncate non-essential long text fields."""
    result = [dict(m) for m in messages]
    if estimate_tokens(result) <= budget:
        return result

    for msg in result:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 1200:
                msg["content"] = content[:1200] + "... (truncated for context budget)"
    if estimate_tokens(result) <= budget:
        return result

    for msg in result:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 600:
                msg["content"] = content[:600] + "... (truncated for context budget)"
    return result
