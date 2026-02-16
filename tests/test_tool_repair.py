from nanobot.agent.tools.repair import repair_tool_call


def test_repairs_exec_list_dir_space_syntax():
    name, args, repaired, note = repair_tool_call(
        ["exec", "list_dir"],
        "exec",
        {"command": "list_dir /home/skye/.nanobot"},
    )
    assert repaired is True
    assert name == "list_dir"
    assert args == {"path": "/home/skye/.nanobot"}
    assert "normalization" in note.lower()


def test_repairs_exec_function_style_json_args():
    name, args, repaired, _ = repair_tool_call(
        ["exec", "list_dir"],
        "exec",
        {"command": 'list_dir({"path":"/tmp"})'},
    )
    assert repaired is True
    assert name == "list_dir"
    assert args == {"path": "/tmp"}


def test_repairs_exec_raw_argument_fallback():
    name, args, repaired, _ = repair_tool_call(
        ["exec", "read_file"],
        "exec",
        {"raw": "read_file /tmp/data.txt"},
    )
    assert repaired is True
    assert name == "read_file"
    assert args == {"path": "/tmp/data.txt"}


def test_does_not_repair_disallowed_or_ambiguous_commands():
    name, args, repaired, _ = repair_tool_call(
        ["exec", "write_file"],
        "exec",
        {"command": "write_file /tmp/x hello"},
    )
    assert repaired is False
    assert name == "exec"
    assert args == {"command": "write_file /tmp/x hello"}

    name2, args2, repaired2, _ = repair_tool_call(
        ["exec", "list_dir"],
        "exec",
        {"command": "list_dir /tmp && pwd"},
    )
    assert repaired2 is False
    assert name2 == "exec"
    assert args2 == {"command": "list_dir /tmp && pwd"}
