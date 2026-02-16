"""Subagent manager for background task execution."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.react import run_react_loop
from nanobot.agent.tools.registry import ToolRegistry, register_core_tools
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider

if TYPE_CHECKING:
    from nanobot.config.schema import LatentLoopConfig


class SubagentManager:
    """
    Manages background subagent execution.

    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        latent_config: "LatentLoopConfig | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig

        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.latent_config = latent_config
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """
        Spawn a subagent to execute a task in the background.

        Args:
            task: The task description for the subagent.
            label: Optional human-readable label for the task.
            origin_channel: The channel to announce results to.
            origin_chat_id: The chat ID to announce results to.

        Returns:
            Status message indicating the subagent was started.
        """
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }

        # Create background task
        bg_task = asyncio.create_task(self._run_subagent(task_id, task, display_label, origin))
        self._running_tasks[task_id] = bg_task

        # Cleanup when done
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info(f"Subagent [{task_id}] starting task: {label}")

        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            register_core_tools(
                tools,
                workspace=self.workspace,
                restrict_to_workspace=self.restrict_to_workspace,
                brave_api_key=self.brave_api_key,
                exec_timeout=self.exec_config.timeout,
            )

            # Build messages with subagent-specific prompt
            system_prompt = self._build_subagent_prompt(task, tool_names=tools.tool_names)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # Run ReAct loop (limited iterations, no circuit breaker for subagents)
            final_result, _ = await run_react_loop(
                provider=self.provider,
                model=self.model,
                tools=tools,
                initial_messages=messages,
                max_iterations=15,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                latent_config=self.latent_config,
                enable_circuit_breaker=False,
                log_prefix=f"Subagent [{task_id}] ",
                context_builder=None,
            )

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            logger.info(f"Subagent [{task_id}] completed successfully")
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Subagent [{task_id}] failed: {e}")
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug(
            f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}"
        )

    def _build_subagent_prompt(self, task: str, tool_names: list[str] | None = None) -> str:
        """Build a focused system prompt for the subagent."""
        import time as _time
        from datetime import datetime
        from pathlib import Path

        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        home_dir = str(Path.home())

        prompt = f"""# Subagent

## Current Time
{now} ({tz})

You are a subagent spawned by the main agent to complete a specific task.

## Home Directory
IMPORTANT: The home directory is: {home_dir}
Use exact paths. Do NOT guess usernames or fabricate paths.

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings"""

        if tool_names:
            tool_list = ", ".join(sorted(tool_names))
            prompt += f"""

## Available Tools (use ONLY these exact names)
{tool_list}

Do NOT attempt tools like str_replace_editor, execute_python_code, bash — they do not exist.
If a tool call fails with "not found", check the list above and use the correct name."""

        prompt += f"""

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}
Skills are available at: {self.workspace}/skills/ (read SKILL.md files as needed)

When you have completed the task, provide a clear summary of your findings or actions."""

        return prompt

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
