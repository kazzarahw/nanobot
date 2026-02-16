"""Conservative repair helpers for malformed tool calls."""

from __future__ import annotations

import json
import re
from typing import Any

_REPAIRABLE_TOOLS = {
    "list_dir": "path",
    "read_file": "path",
    "web_fetch": "url",
    "web_search": "query",
}


def repair_tool_call(
    tool_names: list[str],
    name: str,
    arguments: dict[str, Any],
) -> tuple[str, dict[str, Any], bool, str]:
    """
    Attempt conservative tool-call repair.

    Returns:
        (name, arguments, repaired, note)
    """
    if not isinstance(arguments, dict):
        return name, arguments, False, ""

    if name != "exec":
        return name, arguments, False, ""

    command = arguments.get("command")
    if not isinstance(command, str) and isinstance(arguments.get("raw"), str):
        command = arguments["raw"]
    if not isinstance(command, str):
        return name, arguments, False, ""

    repaired = _parse_exec_wrapped_tool(command, set(tool_names))
    if not repaired:
        return name, arguments, False, ""

    repaired_name, repaired_args = repaired
    note = (
        f"Tool-call normalization applied: converted exec wrapper into direct "
        f"'{repaired_name}' call."
    )
    return repaired_name, repaired_args, True, note


def _parse_exec_wrapped_tool(
    command: str, available_tools: set[str]
) -> tuple[str, dict[str, Any]] | None:
    cmd = command.strip()
    if not cmd:
        return None

    # Avoid repairing shell-ish or composite commands.
    if any(token in cmd for token in ("&&", "||", ";", "|", "\n")):
        return None

    fn_style = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*", cmd)
    if fn_style:
        tool = fn_style.group(1)
        inner = fn_style.group(2).strip()
        return _build_call_from_inner(tool, inner, available_tools)

    split_style = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)\s+(.+)", cmd)
    if split_style:
        tool = split_style.group(1)
        inner = split_style.group(2).strip()
        return _build_call_from_inner(tool, inner, available_tools)

    return None


def _build_call_from_inner(
    tool: str,
    inner: str,
    available_tools: set[str],
) -> tuple[str, dict[str, Any]] | None:
    if tool not in _REPAIRABLE_TOOLS or tool not in available_tools:
        return None

    param_name = _REPAIRABLE_TOOLS[tool]

    if inner.startswith("{") and inner.endswith("}"):
        parsed = _safe_json_loads(inner)
        if isinstance(parsed, dict):
            if param_name in parsed and isinstance(parsed[param_name], str):
                return tool, parsed
        return None

    # Quoted single positional argument.
    if (inner.startswith('"') and inner.endswith('"')) or (
        inner.startswith("'") and inner.endswith("'")
    ):
        unquoted = inner[1:-1].strip()
        if not unquoted:
            return None
        return _map_single_arg(tool, param_name, unquoted)

    return _map_single_arg(tool, param_name, inner)


def _map_single_arg(tool: str, param_name: str, value: str) -> tuple[str, dict[str, Any]] | None:
    value = value.strip()
    if not value:
        return None
    if tool == "web_fetch" and not value.startswith(("http://", "https://")):
        return None
    return tool, {param_name: value}


def _safe_json_loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
