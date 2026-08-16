from __future__ import annotations

import json

from codex_shim.server import ShimServer


def test_deepseek_parser_accepts_plural_wrappers_and_direct_tool_tag():
    calls = ShimServer._recover_deepseek_tool_calls(
        """
<_calls>
<tool_calls>
<exec_command name="exec_command">
{"cmd":"pwd"}
</exec_command>
</tool_calls>
</_calls>
"""
    )

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "exec_command"
    assert json.loads(calls[0]["function"]["arguments"]) == {"cmd": "pwd"}


def test_deepseek_parser_accepts_invoke_parameter_protocol():
    calls = ShimServer._recover_deepseek_tool_calls(
        """
<tool_calls>
<invoke name="exec_command">
<parameter name="cmd" string="true">cd /tmp &amp;&amp; pwd</parameter>
<parameter name="yield_time_ms" string="false">10000</parameter>
</invoke>
</tool_calls>
"""
    )

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "exec_command"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "cmd": "cd /tmp && pwd",
        "yield_time_ms": 10000,
    }


def test_deepseek_parser_recovers_tool_call_missing_opening_angle_bracket():
    calls = ShimServer._recover_deepseek_tool_calls(
        """
tool_call>
{"arguments":{"cmd":"sed -n '1,120p' server.py","workdir":"/tmp"},"name":"exec_command"}
</tool_call>
"""
    )

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "exec_command"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "cmd": "sed -n '1,120p' server.py",
        "workdir": "/tmp",
    }


def test_deepseek_parser_ignores_repeated_empty_wrappers():
    assert ShimServer._recover_deepseek_tool_calls(
        "<_calls>\n<tool_calls>\n<tool_calls>\n<tool_calls>"
    ) == []


def test_deepseek_parser_recovers_mismatched_opening_tag_with_parameters():
    calls = ShimServer._recover_deepseek_tool_calls(
        """
<oke name="exec_command">
<parameter name="cmd">sed -n '1,120p' README.md</parameter>
</invoke>
"""
    )

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "exec_command"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "cmd": "sed -n '1,120p' README.md"
    }


def test_deepseek_tool_arguments_drop_invalid_optional_array_values():
    schema = {
        "type": "object",
        "properties": {
            "cmd": {"type": "string"},
            "prefix_rule": {"type": "array", "items": {"type": "string"}},
            "login": {"type": "boolean"},
            "yield_time_ms": {"type": "integer"},
        },
        "required": ["cmd"],
        "additionalProperties": False,
    }

    cleaned = ShimServer._sanitize_deepseek_tool_arguments(
        {
            "cmd": "pwd",
            "prefix_rule": "",
            "login": "false",
            "yield_time_ms": "10000",
            "invented": "value",
        },
        schema,
    )

    assert cleaned == {
        "cmd": "pwd",
        "login": False,
        "yield_time_ms": 10000,
    }


def test_deepseek_tool_arguments_convert_integral_float_to_integer():
    schema = {
        "type": "object",
        "properties": {
            "cmd": {"type": "string"},
            "max_output_tokens": {"type": "integer"},
        },
        "required": ["cmd"],
    }

    cleaned = ShimServer._sanitize_deepseek_tool_arguments(
        {"cmd": "pwd", "max_output_tokens": 20000.0},
        schema,
    )

    assert cleaned == {"cmd": "pwd", "max_output_tokens": 20000}


def test_shim_detects_all_known_malformed_deepseek_xml_variants():
    for text in (
        "<_calls>",
        "<tool_calls>",
        '<invoke name="exec_command">',
        '<parameter name="cmd">',
        "tool_call>",
    ):
        assert ShimServer._is_malformed_tool_call(text)
