from pathlib import Path

from nanobot.agent.context import ContextBuilder
from nanobot.agent.subagent import SubagentManager
from nanobot.providers.base import LLMResponse, LLMProvider


class DummyProvider(LLMProvider):
    async def chat(self, *args, **kwargs):  # type: ignore[override]
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "dummy-model"


def test_main_system_prompt_includes_exec_guardrail(tmp_path: Path):
    builder = ContextBuilder(tmp_path, tool_names=["exec", "list_dir", "read_file"])
    prompt = builder.build_system_prompt()

    assert "Never wrap built-in tools inside `exec`" in prompt
    assert 'exec({"command": "list_dir /tmp"})' in prompt
    assert 'list_dir({"path": "/tmp"})' in prompt


def test_subagent_prompt_includes_exec_guardrail(tmp_path: Path):
    manager = SubagentManager(
        provider=DummyProvider(),
        workspace=tmp_path,
        bus=object(),  # type: ignore[arg-type]
    )
    prompt = manager._build_subagent_prompt("task", tool_names=["exec", "list_dir"])

    assert "Never wrap built-in tools inside `exec`" in prompt
    assert 'exec({"command": "list_dir /tmp"})' in prompt
    assert 'list_dir({"path": "/tmp"})' in prompt
