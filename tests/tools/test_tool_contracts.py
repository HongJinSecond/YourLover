"""The registry exposes only real tools with working argument schemas."""

from agent.tools import tool_map
from agent.tools.BaseTool import BaseTool, _wrap_tool_instance


def test_registry_and_calculator():
    assert set(tool_map) == {"calculator", "file_manager", "rag"}
    assert "14" in str(tool_map["calculator"].invoke({"expression": "2 + 3 * 4"}))


def test_file_tool_preserves_optional_argument_and_round_trip(tmp_path):
    path = str(tmp_path / "nested" / "note.txt")
    schema = tool_map["file_manager"].args_schema.model_json_schema()
    assert set(schema["required"]) == {"action", "file_path"}
    assert schema["properties"]["content"]["default"] == ""
    tool_map["file_manager"].invoke({"action": "write", "file_path": path, "content": "hello"})
    assert tool_map["file_manager"].invoke({"action": "read", "file_path": path}) == "hello"


def test_tool_annotations_do_not_need_source_code_generation():
    class QuotedTool(BaseTool):
        name = "quoted_tool"
        description = 'A description containing """ and newline\ncharacters.'
        param_descriptions = {"query": 'A "quoted" query.'}

        def _tool_call(self, query: str, count: int = 2) -> str:
            return query * count

    wrapped = _wrap_tool_instance(QuotedTool())
    assert wrapped.invoke({"query": "x"}) == "xx"
    assert (
        wrapped.args_schema.model_json_schema()["properties"]["query"]["description"]
        == 'A "quoted" query.'
    )
