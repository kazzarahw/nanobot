"""File system tools: read, write, edit."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.lsp.manager import LspManager


def _resolve_path(path: str, allowed_dir: Path | None = None) -> Path:
    """Resolve path and optionally enforce directory restriction.

    Also checks for likely misspelled home directories (e.g. /home/skaye
    instead of /home/skye) and returns a helpful error.
    """
    resolved = Path(path).expanduser().resolve()
    if allowed_dir and not str(resolved).startswith(str(allowed_dir.resolve())):
        raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")

    # Detect misspelled /home/<user> paths
    import re

    m = re.match(r"^/home/([^/]+)", str(resolved))
    if m:
        user_dir = Path(f"/home/{m.group(1)}")
        if not user_dir.exists():
            # Find actual home directories to suggest
            home_root = Path("/home")
            if home_root.exists():
                existing = [d.name for d in home_root.iterdir() if d.is_dir()]
                suggestion = f" Existing home directories: {', '.join(sorted(existing))}" if existing else ""
                raise FileNotFoundError(
                    f"Home directory '{user_dir}' does not exist.{suggestion} "
                    f"IMPORTANT: Use exact paths. Do NOT guess usernames."
                )

    return resolved


async def _get_diagnostics_safe(
    lsp_manager: LspManager, path: str, content: str, is_change: bool
) -> str | None:
    """Get formatted diagnostics string, never raises."""
    try:
        diags = await lsp_manager.notify_and_get_diagnostics(path, content, is_change)
        if not diags:
            return None
        return _format_diagnostics(diags)
    except Exception:
        return None


def _format_diagnostics(diags: list[dict]) -> str:
    """Format LSP diagnostics into a readable string for the LLM."""
    lines = []
    for d in diags:
        severity = {1: "Error", 2: "Warning", 3: "Info", 4: "Hint"}.get(d.get("severity", 0), "?")
        line_num = d.get("range", {}).get("start", {}).get("line", 0) + 1
        msg = d.get("message", "")
        source = d.get("source", "")
        prefix = f"[{source}] " if source else ""
        lines.append(f"  Line {line_num}: {severity}: {prefix}{msg}")
    return "\n".join(lines)


class ReadFileTool(Tool):
    """Tool to read file contents."""

    def __init__(self, allowed_dir: Path | None = None, lsp_manager: LspManager | None = None):
        self._allowed_dir = allowed_dir
        self._lsp_manager = lsp_manager

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file at the given path. "
            "Output may include an '--- LSP Diagnostics ---' section with "
            "errors/warnings — these are not part of the file content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "The file path to read"}},
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._allowed_dir)
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"

            content = file_path.read_text(encoding="utf-8")
            result = content

            if self._lsp_manager:
                diag_str = await _get_diagnostics_safe(
                    self._lsp_manager, str(file_path), content, is_change=False
                )
                if diag_str:
                    result += f"\n\n--- LSP Diagnostics ---\n{diag_str}"

            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WriteFileTool(Tool):
    """Tool to write content to a file."""

    def __init__(self, allowed_dir: Path | None = None, lsp_manager: LspManager | None = None):
        self._allowed_dir = allowed_dir
        self._lsp_manager = lsp_manager

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file at the given path. Creates parent directories if needed. "
            "Result may include '--- LSP Diagnostics ---' with errors/warnings to fix."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to write to"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._allowed_dir)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            result = f"Successfully wrote {len(content)} bytes to {path}"

            if self._lsp_manager:
                diag_str = await _get_diagnostics_safe(
                    self._lsp_manager, str(file_path), content, is_change=True
                )
                if diag_str:
                    result += f"\n\n--- LSP Diagnostics ---\n{diag_str}"

            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class EditFileTool(Tool):
    """Tool to edit a file by replacing text."""

    def __init__(self, allowed_dir: Path | None = None, lsp_manager: LspManager | None = None):
        self._allowed_dir = allowed_dir
        self._lsp_manager = lsp_manager

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing old_text with new_text. The old_text must exist exactly in the file. "
            "Result may include '--- LSP Diagnostics ---' with errors/warnings to fix."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to edit"},
                "old_text": {"type": "string", "description": "The exact text to find and replace"},
                "new_text": {"type": "string", "description": "The text to replace with"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(self, path: str, old_text: str, new_text: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._allowed_dir)
            if not file_path.exists():
                return f"Error: File not found: {path}"

            content = file_path.read_text(encoding="utf-8")

            if old_text not in content:
                return "Error: old_text not found in file. Make sure it matches exactly."

            # Count occurrences
            count = content.count(old_text)
            if count > 1:
                return f"Warning: old_text appears {count} times. Please provide more context to make it unique."

            new_content = content.replace(old_text, new_text, 1)
            file_path.write_text(new_content, encoding="utf-8")

            result = f"Successfully edited {path}"

            if self._lsp_manager:
                diag_str = await _get_diagnostics_safe(
                    self._lsp_manager, str(file_path), new_content, is_change=True
                )
                if diag_str:
                    result += f"\n\n--- LSP Diagnostics ---\n{diag_str}"

            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {str(e)}"


class ListDirTool(Tool):
    """Tool to list directory contents."""

    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List the contents of a directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "The directory path to list"}},
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            dir_path = _resolve_path(path, self._allowed_dir)
            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Not a directory: {path}"

            items = []
            for item in sorted(dir_path.iterdir()):
                prefix = "📁 " if item.is_dir() else "📄 "
                items.append(f"{prefix}{item.name}")

            if not items:
                return f"Directory {path} is empty"

            return "\n".join(items)
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"
