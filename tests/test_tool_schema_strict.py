from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.filesystem import ListDirTool


class DummyTool(Tool):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "Dummy test tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_to_schema_adds_strict_and_blocks_extra_properties():
    schema = DummyTool().to_schema()
    fn = schema["function"]
    params = fn["parameters"]

    assert schema["type"] == "function"
    assert fn["strict"] is True
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    assert params["required"] == ["value"]


def test_existing_tool_schema_keeps_shape_and_adds_additional_properties_false():
    schema = ListDirTool().to_schema()
    params = schema["function"]["parameters"]

    assert schema["function"]["name"] == "list_dir"
    assert params["required"] == ["path"]
    assert params["additionalProperties"] is False
