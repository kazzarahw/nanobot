"""Tool registry for dynamic tool management."""

from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._invalid_tool_attempts: dict[str, int] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """
        Execute a tool by name with given parameters.

        Args:
            name: Tool name.
            params: Tool parameters.

        Returns:
            Tool execution result as string.

        Raises:
            KeyError: If tool not found.
        """
        tool = self._tools.get(name)
        if not tool:
            self._invalid_tool_attempts[name] = self._invalid_tool_attempts.get(name, 0) + 1
            attempts = self._invalid_tool_attempts[name]
            valid_names = ", ".join(sorted(self._tools.keys()))
            if attempts >= 3:
                return (
                    f"Error: STOP trying '{name}' — it does not exist and never will. "
                    f"You have attempted it {attempts} times. "
                    f"Use ONLY these tools: {valid_names}"
                )
            return f"Error: Tool '{name}' not found. Available tools: {valid_names}"

        try:
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            return await tool.execute(**params)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)


def register_core_tools(
    registry: ToolRegistry,
    workspace: Path,
    restrict_to_workspace: bool = False,
    brave_api_key: str | None = None,
    exec_timeout: int = 120000,
    lsp_manager: Any = None,
) -> None:
    """
    Register the 7 core tools used by both agents and subagents.

    Args:
        registry: The tool registry to populate.
        workspace: The workspace path.
        restrict_to_workspace: Whether to restrict file ops to workspace.
        brave_api_key: Optional API key for web search.
        exec_timeout: Timeout for shell commands (ms).
        lsp_manager: Optional LSP manager for file tools.
    """
    from nanobot.agent.tools.filesystem import (
        EditFileTool,
        ListDirTool,
        ReadFileTool,
        WriteFileTool,
    )
    from nanobot.agent.tools.shell import ExecTool
    from nanobot.agent.tools.web import WebFetchTool, WebSearchTool

    allowed_dir = workspace if restrict_to_workspace else None
    registry.register(ReadFileTool(allowed_dir=allowed_dir, lsp_manager=lsp_manager))
    registry.register(WriteFileTool(allowed_dir=allowed_dir, lsp_manager=lsp_manager))
    registry.register(EditFileTool(allowed_dir=allowed_dir, lsp_manager=lsp_manager))
    registry.register(ListDirTool(allowed_dir=allowed_dir))
    registry.register(
        ExecTool(
            working_dir=str(workspace),
            timeout=exec_timeout,
            restrict_to_workspace=restrict_to_workspace,
        )
    )
    registry.register(WebSearchTool(api_key=brave_api_key))
    registry.register(WebFetchTool())
