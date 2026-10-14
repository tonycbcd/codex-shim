from __future__ import annotations

import json

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer

from codex_shim import server as server_module
from codex_shim.server import (
    CLAUDE_EXECUTION_RULES,
    CLAUDE_LOOP_BREAK_TOOL_THRESHOLD,
    CLAUDE_REPEAT_TOOL_CALL_THRESHOLD,
    ChatGPTPassthroughTerminalGuard,
    CODEGRAPH_FIRST_RULES,
    DEEPSEEK_EXECUTION_RULES,
    KIMI_HARD_STOP_TOOL_THRESHOLD,
    PICKER_TOKEN_HEADER,
    ResponsesStreamState,
    ShimServer,
    _add_claude_execution_guidance,
    _await_with_sse_heartbeats,
    _check_and_strip_platform_prefix,
    _claude_current_turn_failed_patch_count,
    _claude_current_turn_tool_call_count,
    _claude_current_turn_repeated_tool_call,
    _configured_claude_gateway_model,
    _custom_tool_input,
    _current_managed_model,
    _deepseek_codegraph_project_hint,
    _deepseek_stream_failure,
    _estimate_kimi_tokens,
    _expand_deepseek_namespace_tools,
    _is_claude_tool_pairing_error,
    _repair_claude_tool_rounds,
    _is_deepseek_internal_result_text,
    _kimi_current_turn_tool_call_count,
    _normalize_apply_patch_input,
    _picker_html,
    _prime_loopback_platform,
    _rewrite_response_model,
    _sanitize_deepseek_codeproxy_event,
    _sanitize_deepseek_body,
    _sanitize_deepseek_completed_payload,
    _sanitize_chatgpt_passthrough_body,
    _set_active_model,
    _write_sse,
)
from codex_shim.settings import FALLBACK_CHATGPT_PASSTHROUGH_SLUGS
from codex_shim.translate import SHIM_ENCRYPTED_CONTENT_PREFIX


@pytest.fixture
def auth_present(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"tokens": {"access_token": "stub", "account_id": "acct"}}))
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", auth)
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", auth)
    return auth


@pytest.fixture
def auth_missing(monkeypatch, tmp_path):
    missing = tmp_path / "missing-auth.json"
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", missing)
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", missing)


def test_sanitize_chatgpt_passthrough_body_drops_shim_reasoning():
    body = {
        "model": "claude-local",
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {
                "id": "rs_shim",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "local thought"}],
                "encrypted_content": f"{SHIM_ENCRYPTED_CONTENT_PREFIX}deadbeef",
            },
            {
                "id": "rs_openai",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "openai thought"}],
                "encrypted_content": "openai-verifiable-content",
            },
        ],
    }

    sanitized = _sanitize_chatgpt_passthrough_body(body)

    assert sanitized is not body
    assert sanitized["input"] is not body["input"]
    assert [item["id"] for item in sanitized["input"] if item.get("type") == "reasoning"] == ["rs_openai"]
    assert sanitized["input"][1]["encrypted_content"] == "openai-verifiable-content"
    assert len(body["input"]) == 3


def test_sanitize_chatgpt_passthrough_body_repairs_item_id_prefixes():
    body = {
        "input": [
            {
                "id": "fc_bad_message",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            {
                "id": "msg_bad_tool",
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec_command",
                "arguments": "{}",
            },
            {
                "id": "msg_keep",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "continue"}],
            },
        ]
    }

    sanitized = _sanitize_chatgpt_passthrough_body(body)

    assert sanitized["input"][0]["id"] == "msg_bad_message"
    assert sanitized["input"][1]["id"] == "fc_bad_tool"
    assert sanitized["input"][2]["id"] == "msg_keep"


def test_configured_claude_gateway_model_prefers_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_GATEWAY_MODEL", "claude-opus-4.5")
    assert _configured_claude_gateway_model() == "claude-opus-4.5"


def test_configured_claude_gateway_model_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("CLAUDE_GATEWAY_MODEL", raising=False)
    assert _configured_claude_gateway_model() == "claude-sonnet-4.6"


def test_kimi_current_turn_tool_call_count_ignores_previous_turns():
    body = {
        "input": [
            {"type": "message", "role": "user", "content": "old turn"},
            {"type": "function_call", "call_id": "old", "name": "exec", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "old", "output": "done"},
            {"type": "message", "role": "user", "content": "new turn"},
            {"type": "function_call", "call_id": "new-1", "name": "exec", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "new-1", "output": "done"},
            {"type": "custom_tool_call", "call_id": "new-2", "name": "patch", "input": "x"},
        ]
    }

    assert _kimi_current_turn_tool_call_count(body) == 2


def test_estimate_kimi_tokens_never_returns_zero_for_content():
    assert _estimate_kimi_tokens("") > 0
    assert _estimate_kimi_tokens("hello") > 0
    assert _estimate_kimi_tokens({"content": "中文内容"}) > 0


def test_prime_routing_identity_rejects_non_loopback_peer():
    class Transport:
        @staticmethod
        def get_extra_info(name):
            assert name == "peername"
            return ("203.0.113.20", 12345)

    class Request:
        headers = {"Authorization": "Bearer local-codex-shim-chatgpt"}
        transport = Transport()

    with pytest.raises(web.HTTPForbidden) as exc:
        _prime_loopback_platform(Request())
    assert "loopback-only" in exc.value.text


def test_deepseek_stream_failure_detects_empty_upstream_error_before_sse_commit():
    line = json.dumps(
        {
            "error": {
                "message": (
                    "DeepSeek Web returned an empty response after browser recovery "
                    "and two automatic retries"
                )
            }
        }
    )

    assert _deepseek_stream_failure(line) == (
        "DeepSeek Web returned an empty response after browser recovery "
        "and two automatic retries"
    )
    assert _deepseek_stream_failure("[DONE]") == "DeepSeek Web returned no stream events"
    assert _deepseek_stream_failure(
        json.dumps({"choices": [{"delta": {"content": "正常回答"}}]})
    ) is None


def test_deepseek_stream_failure_detects_legacy_empty_retry_message():
    line = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "content": (
                            "DeepSeek Web did not produce a usable tool call or final "
                            "answer after two automatic retries. Please retry this turn."
                        )
                    }
                }
            ]
        }
    )

    assert _deepseek_stream_failure(line) == (
        "DeepSeek Web returned an empty response after automatic retries"
    )


def test_sanitize_chatgpt_passthrough_body_removes_nested_shim_encrypted_content():
    body = {
        "model": "claude-local",
        "input": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "done",
                        "encrypted_content": f"{SHIM_ENCRYPTED_CONTENT_PREFIX}deadbeef",
                    }
                ],
            }
        ],
    }

    sanitized = _sanitize_chatgpt_passthrough_body(body)

    assert "encrypted_content" not in sanitized["input"][0]["content"][0]
    assert "encrypted_content" in body["input"][0]["content"][0]


def test_sanitize_chatgpt_passthrough_body_keeps_old_reasoning_content_empty():
    body = {
        "model": "gpt-5.4",
        "input": [
            {
                "id": f"rs_{index}",
                "type": "reasoning",
                "content": [],
                "summary": [{"type": "summary_text", "text": f"thought {index}"}],
                "encrypted_content": f"openai-content-{index}",
            }
            for index in range(8)
        ],
    }

    sanitized = _sanitize_chatgpt_passthrough_body(body)

    assert sanitized["input"][0]["content"] == []
    assert sanitized["input"][0]["summary"] == []
    assert sanitized["input"][-1]["summary"] == [
        {"type": "summary_text", "text": "thought 7"}
    ]


def test_claude_execution_guidance_counts_only_latest_user_turn_tools():
    body = {
        "input": [
            {"type": "message", "role": "user", "content": "old task"},
            {"type": "function_call", "name": "exec_command", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "old", "output": "ok"},
            {"type": "message", "role": "user", "content": "new task"},
            {"type": "function_call", "name": "exec_command", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "new", "output": "ok"},
            {"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch"},
        ]
    }

    assert _claude_current_turn_tool_call_count(body) == 2


def test_claude_execution_guidance_adds_loop_breaker_after_threshold():
    body = {
        "input": [
            {"type": "message", "role": "user", "content": "fix it"},
            *[
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": f"sed -n '{index}p' file.php"}),
                }
                for index in range(CLAUDE_LOOP_BREAK_TOOL_THRESHOLD)
            ],
        ]
    }
    chat_body = {"messages": [{"role": "user", "content": "fix it"}]}

    count = _add_claude_execution_guidance(chat_body, body)

    assert count == CLAUDE_LOOP_BREAK_TOOL_THRESHOLD
    guidance = chat_body["messages"][0]
    assert guidance["role"] == "system"
    assert CLAUDE_EXECUTION_RULES in guidance["content"]
    assert "Claude tool-loop breaker" in guidance["content"]
    assert f"used {CLAUDE_LOOP_BREAK_TOOL_THRESHOLD} tool calls" in guidance["content"]


def test_claude_repeated_tool_call_hard_stops_exact_command_loop():
    repeated_call = {
        "type": "function_call",
        "name": "exec_command",
        "arguments": json.dumps({"cmd": "curl -s example.test | grep agd"}),
    }
    body = {
        "input": [
            {"type": "message", "role": "user", "content": "investigate"},
            *[dict(repeated_call) for _ in range(CLAUDE_REPEAT_TOOL_CALL_THRESHOLD)],
        ]
    }
    chat_body = {
        "messages": [{"role": "user", "content": "investigate"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "exec_command", "parameters": {}},
            }
        ],
    }

    assert _claude_current_turn_repeated_tool_call(body) == (
        "exec_command",
        CLAUDE_REPEAT_TOOL_CALL_THRESHOLD,
    )

    _add_claude_execution_guidance(chat_body, body)

    assert chat_body["tools"] == []
    assert "Claude repeated-tool hard stop" in chat_body["messages"][0]["content"]
    assert "Tools are disabled" in chat_body["messages"][0]["content"]


def test_claude_repeated_tool_call_ignores_old_turns_and_write_stdin_polling():
    body = {
        "input": [
            {"type": "message", "role": "user", "content": "old task"},
            *[
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                }
                for _ in range(5)
            ],
            {"type": "message", "role": "user", "content": "new task"},
            *[
                {
                    "type": "function_call",
                    "name": "write_stdin",
                    "arguments": '{"session_id":1,"chars":""}',
                }
                for _ in range(5)
            ],
        ]
    }

    assert _claude_current_turn_repeated_tool_call(body) == ("", 0)


def test_deepseek_skill_discovery_guidance_added_for_general_task():
    body = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "你去 /opt/codes/ms/search-service 结合codegraph 总结 waitlist 搜索逻辑",
            }
        ]
    }

    sanitized = _sanitize_deepseek_body(body)
    instructions = sanitized["instructions"]

    assert "DeepSeek skill-discovery protocol" in instructions
    assert "/config/.codex/skills" in instructions
    assert "/config/.codex/skills/.system" in instructions
    assert "~/.codex/skills" in instructions
    assert "read that skill's SKILL.md before any wider search" in instructions


def test_deepseek_skill_discovery_guidance_skips_plain_greeting():
    body = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "Hello",
            }
        ]
    }

    sanitized = _sanitize_deepseek_body(body)

    assert "DeepSeek skill-discovery protocol" not in sanitized["instructions"]


def test_detect_deepseek_internal_result_text():
    assert _is_deepseek_internal_result_text("<result>\n<check>patch applied</check>\n</result>\n</function_results>")
    assert _is_deepseek_internal_result_text('{"tool_calls":[{"name":"exec_command","arguments":{"cmd":"pwd"}}]}')
    assert not _is_deepseek_internal_result_text("已完成修改，并通过语法检查。")


def test_sanitize_deepseek_codeproxy_event_suppresses_internal_delta():
    payload = {
        "type": "response.output_text.delta",
        "delta": "<result>\n<check>patch applied</check>\n</result>\n</function_results>",
    }

    assert _sanitize_deepseek_codeproxy_event(payload, "[deepseek-pro]") is None


def test_sanitize_deepseek_completed_payload_replaces_internal_only_message():
    payload = {
        "type": "response.completed",
        "response": {
            "output": [
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": "{\"cmd\":\"pwd\"}",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "<result><check>patch applied</check></result></function_results>",
                        }
                    ],
                },
            ]
        },
    }

    sanitized = _sanitize_deepseek_completed_payload(payload, "[deepseek-pro]")
    messages = [item for item in sanitized["response"]["output"] if item.get("type") == "message"]

    assert messages
    assert messages[-1]["content"][0]["text"] == "[deepseek-pro] 已完成工具执行，请查看上方命令结果与代码变更。"


async def test_claude_fallback_uses_configured_kiro_gateway(monkeypatch):
    requested = {}

    class FailedResponse:
        status = 503

        async def text(self):
            return "test failure"

    class FakeSession:
        async def post(self, url, **kwargs):
            requested["url"] = url
            requested["headers"] = kwargs["headers"]
            return FailedResponse()

    shim = ShimServer()

    async def fake_get_session():
        return FakeSession()

    monkeypatch.setattr(shim, "_get_session", fake_get_session)
    monkeypatch.setenv(
        "CLAUDE_GATEWAY_URL",
        "http://127.0.0.1:18901/v1/chat/completions",
    )
    monkeypatch.setenv("CLAUDE_GATEWAY_API_KEY", "test-key")

    result = await shim._claude_gateway_fallback(
        None,
        {"input": [{"role": "user", "content": "hello"}]},
    )

    assert result is None
    assert requested["url"] == "http://127.0.0.1:18901/v1/chat/completions"
    assert requested["headers"]["Authorization"] == "Bearer test-key"


def test_deepseek_malformed_tool_call_detection_is_case_insensitive():
    shim = ShimServer()

    assert shim._is_malformed_tool_call('<INVOKE name="exec_command">')
    assert shim._is_malformed_tool_call('<parameter name="cmd">pwd</parameter>')
    assert not shim._is_malformed_tool_call("normal response")


def test_repair_claude_tool_rounds_moves_results_and_drops_dangling_items():
    chat_body = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "first", "arguments": "{}"},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "second", "arguments": "{}"},
                    },
                ],
            },
            {"role": "user", "content": "steering message"},
            {"role": "tool", "tool_call_id": "call_b", "content": "b result"},
            {"role": "tool", "tool_call_id": "call_a", "content": "a result"},
            {"role": "tool", "tool_call_id": "orphan", "content": "orphan"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_missing",
                        "type": "function",
                        "function": {"name": "missing", "arguments": "{}"},
                    }
                ],
            },
        ]
    }

    stats = _repair_claude_tool_rounds(chat_body)

    assert [message["role"] for message in chat_body["messages"]] == [
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    assert [
        message.get("tool_call_id")
        for message in chat_body["messages"][1:3]
    ] == ["call_a", "call_b"]
    assert stats == {
        "paired": 2,
        "dropped_calls": 1,
        "dropped_results": 1,
        "moved_results": 1,
    }


def test_claude_tool_pairing_error_detection():
    assert _is_claude_tool_pairing_error(
        "tool_use ids were found without tool_result blocks immediately after"
    )
    assert not _is_claude_tool_pairing_error("Token expired")


@pytest.mark.asyncio
async def test_claude_gateway_retries_tool_pairing_400_without_tool_history(
    monkeypatch,
):
    requests: list[dict] = []

    class FakeResponse:
        def __init__(self, status, text):
            self.status = status
            self._text = text

        async def text(self):
            return self._text

        def release(self):
            pass

    class FakeSession:
        async def post(self, url, **kwargs):
            requests.append(json.loads(json.dumps(kwargs["json"])))
            if len(requests) == 1:
                return FakeResponse(
                    400,
                    "tool_use ids were found without tool_result blocks "
                    "immediately after",
                )
            return FakeResponse(503, "test stop")

    shim = ShimServer()

    async def fake_get_session():
        return FakeSession()

    monkeypatch.setattr(shim, "_get_session", fake_get_session)
    result = await shim._claude_gateway_fallback(
        None,
        {
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
                {"type": "message", "role": "user", "content": "continue"},
                {
                    "type": "function_call_output",
                    "call_id": "call_a",
                    "output": "ok",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
        },
    )

    assert result is None
    assert len(requests) == 2
    first_roles = [message["role"] for message in requests[0]["messages"]]
    assert first_roles[-3:] == ["assistant", "tool", "user"]
    assert not any(
        message.get("role") == "tool" or message.get("tool_calls")
        for message in requests[1]["messages"]
    )


def test_platform_prefix_uses_newest_prefixed_user_message():
    body = {
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "[chatgpt] old"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "reply"}]},
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "[deepseek-pro] newest"}],
            },
        ]
    }

    stripped, platform = _check_and_strip_platform_prefix(body)

    assert platform == "deepseek-pro"
    assert stripped["input"][0]["content"][0]["text"] == "[chatgpt] old"
    assert stripped["input"][2]["content"][0]["text"] == "newest"


def test_latest_prefix_in_history_applies_to_unprefixed_followup():
    body = {
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "[chatgpt] old"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "reply"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "latest"}]},
        ]
    }

    stripped, platform = _check_and_strip_platform_prefix(body)

    assert platform == "chatgpt"
    assert stripped["input"][0]["content"][0]["text"] == "old"
    assert stripped["input"][2]["content"][0]["text"] == "latest"


@pytest.mark.parametrize("prefix", ["claude", "claud", "kiro"])
def test_claude_prefix_aliases_switch_back_to_claude(prefix):
    body = {
        "input": [
            {"role": "user", "content": "[chatgpt] old task"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": f"[{prefix}] use Claude now"},
            {"role": "user", "content": "synthetic follow-up"},
        ]
    }

    stripped, platform = _check_and_strip_platform_prefix(body)

    assert platform == "claude"
    assert stripped["input"][2]["content"] == "use Claude now"
    assert stripped["input"][0]["content"] == "[chatgpt] old task"


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("claude-4.5", "claude-4.5"),
        ("kimi-2.7", "kimi-k2.7"),
        ("kimi-3", "kimi-k3"),
        ("chatgpt-5.6", "chatgpt-5.6"),
        ("chatgpt-6", "chatgpt-6"),
        ("deepseek", "deepseek"),
        ("deepseek-pro", "deepseek-pro"),
    ],
)
def test_requested_wechat_prefixes_are_supported(prefix, expected):
    body = {"input": [{"role": "user", "content": f"[{prefix}] ping"}]}

    stripped, platform = _check_and_strip_platform_prefix(body)

    assert platform == expected
    assert stripped["input"][0]["content"] == "ping"


def test_batched_wechat_messages_use_the_last_explicit_prefix():
    body = {
        "input": [
            {
                "role": "user",
                "content": "[deepseek] 看上去还不错\n\n[chatgpt-5.6] Hi, what's your name ?",
            }
        ]
    }

    stripped, platform = _check_and_strip_platform_prefix(body)

    assert platform == "chatgpt-5.6"
    assert stripped["input"][0]["content"] == "看上去还不错\n\nHi, what's your name ?"


def test_new_prefix_updates_remembered_session_platform():
    shim = ShimServer()

    class Request:
        headers = {"session_id": "session-switch-model"}

    assert shim._resolve_session_platform(Request(), "chatgpt") == "chatgpt"
    assert shim._resolve_session_platform(Request(), "deepseek-pro") == "deepseek-pro"
    assert shim._resolve_session_platform(Request(), None) == "deepseek-pro"


def test_deepseek_context_keeps_current_task_and_compacts_optional_tools():
    body = {
        "input": [
            {"role": "user", "content": "stale task"},
            {"type": "reasoning", "encrypted_content": "old", "summary": []},
            {"type": "function_call_output", "call_id": "old", "output": "x" * 10_000},
            {"role": "user", "content": "previous task"},
            {
                "type": "function_call_output",
                "call_id": "previous",
                "output": "previous result",
            },
            {"role": "user", "content": "fix the Python code"},
            {"type": "function_call", "call_id": "new", "name": "exec_command", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "new", "output": "result"},
        ],
        "tools": [
            {"type": "function", "name": "exec_command", "description": "x" * 1_000},
            {"type": "namespace", "name": "mcp__pencil__", "description": "design"},
            {"type": "namespace", "name": "mcp__codegraph__", "description": "code"},
        ],
    }

    sanitized = _sanitize_deepseek_body(body)

    assert len(sanitized["input"]) == 6
    assert sanitized["input"][0]["content"] == "stale task"
    assert all(item.get("call_id") != "old" for item in sanitized["input"])
    assert any(item.get("call_id") == "previous" for item in sanitized["input"])
    assert "Do not end while an update_plan item is still in_progress" in sanitized["instructions"]
    assert all(item.get("type") != "reasoning" for item in sanitized["input"])
    assert [tool["name"] for tool in sanitized["tools"]] == [
        "exec_command",
        "mcp__codegraph__",
    ]
    assert sanitized["tools"][1]["tools"][0]["name"] == "codegraph_explore"
    assert len(sanitized["tools"][0]["description"]) <= 241


def test_deepseek_execution_rules_require_real_edits_and_verification():
    sanitized = _sanitize_deepseek_body(
        {
            "instructions": "Follow the repository instructions.",
            "input": [{"role": "user", "content": "fix the code"}],
            "tools": [],
        }
    )

    instructions = sanitized["instructions"]
    assert instructions.startswith("Follow the repository instructions.")
    assert DEEPSEEK_EXECUTION_RULES in instructions
    assert "Run dependent steps sequentially" in instructions
    assert "successful file write" in instructions
    assert "a real diff showing the intended change" in instructions
    assert "Never fabricate file contents, diffs, command output" in instructions
    assert "If no file was changed" in instructions
    assert "distinguish pre-existing changes from this task" in instructions
    assert "matching local skill" in instructions


def test_deepseek_office_scp_requests_get_skill_guidance():
    sanitized = _sanitize_deepseek_body(
        {
            "input": [{"role": "user", "content": "你将修改的代码 scp 到 office 对应文件中"}],
            "tools": [],
        }
    )

    instructions = sanitized["instructions"]
    assert "office-scp skill" in instructions
    assert "/config/.codex/skills/office-scp/SKILL.md" in instructions
    assert "/config/.codex/skills/office-scp/scripts/scp-to-office.sh" in instructions
    assert "~/.codex/secrets/office-ssh-password" in instructions
    assert "Do not search for alternate office credentials" in instructions


def test_deepseek_context_keeps_previous_task_for_continue_message():
    body = {
        "input": [
            {"role": "user", "content": "repair the routing bug"},
            {"type": "function_call", "call_id": "call_1", "name": "exec_command", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
            {"role": "user", "content": "继续修复"},
        ],
        "tools": [],
    }

    sanitized = _sanitize_deepseek_body(body)

    assert sanitized["input"][0]["content"] == "repair the routing bug"
    assert any(item.get("call_id") == "call_1" for item in sanitized["input"])
    assert sanitized["input"][-1]["content"] == "继续修复"


def test_deepseek_hydrates_codegraph_namespace_with_project_hint(tmp_path):
    project = tmp_path / "de4-web"
    (project / ".codegraph").mkdir(parents=True)
    body = {
        "input": [
            {
                "role": "user",
                "content": (
                    f"<environment_context><cwd>{tmp_path}</cwd></environment_context>\n"
                    "在de4-web 中结合 codegraph 修复问题"
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
            {"type": "namespace", "name": "mcp__codegraph__", "description": "code"},
        ],
    }

    assert _deepseek_codegraph_project_hint(body) == str(project)
    sanitized = _sanitize_deepseek_body(body)
    tools = {
        tool["name"]: tool
        for tool in sanitized["tools"]
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    }

    assert "mcp__codegraph__" in tools
    codegraph = tools["mcp__codegraph__"]["tools"][0]
    assert codegraph["parameters"]["required"] == ["query", "projectPath"]
    assert codegraph["parameters"]["properties"]["projectPath"]["default"] == str(project)
    assert str(project) in codegraph["description"]
    assert CODEGRAPH_FIRST_RULES in sanitized["instructions"]
    assert f"Indexed projectPath for this task: {project}" in sanitized["instructions"]


def test_codegraph_project_hint_walks_up_from_nested_file(tmp_path):
    project = tmp_path / "front"
    nested_file = project / "src/common/views/newmobile/editProfile.vue"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("<template />")
    (project / ".codegraph").mkdir()
    body = {
        "input": [
            {
                "role": "user",
                "content": f"检查代码 {nested_file} 中的问题",
            }
        ]
    }

    assert _deepseek_codegraph_project_hint(body) == str(project)


def test_kimi_expands_codegraph_namespace_with_project_hint(tmp_path):
    project = tmp_path / "front"
    (project / ".codegraph").mkdir(parents=True)
    tools = [
        {
            "type": "function",
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
        {"type": "namespace", "name": "mcp__codegraph__", "description": "code"},
    ]

    expanded = _expand_deepseek_namespace_tools(tools, str(project))
    codegraph = next(
        tool for tool in expanded if tool.get("name") == "codegraph_explore"
    )

    assert all(tool.get("type") != "namespace" for tool in expanded)
    assert codegraph["parameters"]["required"] == ["query", "projectPath"]
    assert codegraph["parameters"]["properties"]["projectPath"]["default"] == str(project)


def test_deepseek_rewrites_virtual_codegraph_call_to_exec_command():
    delta = {
        "tool_calls": [
            {
                "function": {
                    "name": "codegraph_explore",
                    "arguments": json.dumps(
                        {
                            "query": "deletePhotos verified status",
                            "projectPath": "/workspace/de4-web",
                            "maxFiles": 12,
                        }
                    ),
                }
            }
        ]
    }
    schemas = {
        "exec_command": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "workdir": {"type": "string"},
            },
            "required": ["cmd"],
            "additionalProperties": False,
        }
    }

    ShimServer._sanitize_deepseek_delta_tool_calls(
        delta,
        schemas,
        "/fallback/project",
    )

    function = delta["tool_calls"][0]["function"]
    assert function["name"] == "exec_command"
    assert json.loads(function["arguments"]) == {
        "cmd": "codegraph explore 'deletePhotos verified status'",
        "workdir": "/workspace/de4-web",
    }


def test_deepseek_context_keeps_previous_turn_tool_chain_only():
    body = {
        "input": [
            {"role": "user", "content": "old task"},
            {
                "type": "function_call",
                "call_id": "old_call",
                "name": "exec_command",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "old_call",
                "output": "old result",
            },
            {"role": "assistant", "content": "old task done"},
            {"role": "user", "content": "inspect the current bug"},
            {
                "type": "function_call",
                "call_id": "recent_call",
                "name": "exec_command",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "recent_call",
                "output": "recent result",
            },
            {"role": "assistant", "content": "the bug is in server.py"},
            {"role": "user", "content": "fix it"},
        ],
        "tools": [],
    }

    sanitized = _sanitize_deepseek_body(body)
    call_ids = {
        item.get("call_id")
        for item in sanitized["input"]
        if item.get("call_id")
    }

    assert "old_call" not in call_ids
    assert "recent_call" in call_ids
    assert sanitized["input"][-1]["content"] == "fix it"


def test_deepseek_context_walks_past_consecutive_continue_messages():
    body = {
        "input": [
            {"role": "user", "content": "repair the HRL scheduler in /opt/codes/ms/HRL"},
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
            {"role": "user", "content": "继续修复"},
            {"role": "assistant", "content": "still working"},
            {"role": "user", "content": "重试"},
        ],
        "tools": [],
    }

    sanitized = _sanitize_deepseek_body(body)

    assert sanitized["input"][0]["content"] == "repair the HRL scheduler in /opt/codes/ms/HRL"
    assert sanitized["input"][-1]["content"] == "重试"


def test_deepseek_context_keeps_recent_dialogue_for_followup_question():
    body = {
        "input": [
            {"role": "user", "content": "Here is the OpenSearch DSL: must_not Username.keyword"},
            {"type": "function_call", "call_id": "old", "name": "exec_command", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "old", "output": "large old output"},
            {
                "role": "assistant",
                "content": "[deepseek-pro] Use a lowercase normalizer on Username.keyword.",
            },
            {"role": "user", "content": "我测试了，还是不行"},
        ],
        "tools": [],
    }

    sanitized = _sanitize_deepseek_body(body)

    assert [item.get("role") for item in sanitized["input"]] == [
        "user",
        None,
        None,
        "assistant",
        "user",
    ]
    assert sanitized["input"][0]["content"].startswith("Here is the OpenSearch DSL")
    assert sanitized["input"][1]["call_id"] == "old"
    assert sanitized["input"][2]["call_id"] == "old"
    assert sanitized["input"][3]["content"] == (
        "Use a lowercase normalizer on Username.keyword."
    )
    assert sanitized["input"][4]["content"] == "我测试了，还是不行"


def test_deepseek_context_keeps_at_most_six_recent_user_turns():
    body = {
        "input": [
            item
            for index in range(8)
            for item in (
                {"role": "user", "content": f"user-{index}"},
                {"role": "assistant", "content": f"assistant-{index}"},
            )
        ]
        + [{"role": "user", "content": "latest followup"}],
        "tools": [],
    }

    sanitized = _sanitize_deepseek_body(body)
    user_messages = [
        item["content"]
        for item in sanitized["input"]
        if item.get("role") == "user"
    ]

    assert user_messages == [
        "user-3",
        "user-4",
        "user-5",
        "user-6",
        "user-7",
        "latest followup",
    ]


def test_custom_tool_input_unwraps_freeform_envelope():
    assert _custom_tool_input('{"input":"*** Begin Patch\\n*** End Patch"}') == (
        "*** Begin Patch\n*** End Patch"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            """*** Begin Patch
*** Update File: /tmp/example.php
*** Hunk: public count
-old line
+new line
*** End Patch""",
            """*** Begin Patch
*** Update File: /tmp/example.php
@@
-old line
+new line
*** End Patch""",
        ),
        (
            """*** Begin Patch
*** Update File: /tmp/example.php
*** Find:
old line
*** Replace:
new line
*** End Patch""",
            """*** Begin Patch
*** Update File: /tmp/example.php
@@
-old line
+new line
*** End Patch""",
        ),
        (
            """*** Begin Patch
*** Update File: /tmp/example.php
*** Update 1
old line
---
new line
*** End Patch""",
            """*** Begin Patch
*** Update File: /tmp/example.php
@@
-old line
+new line
*** End Patch""",
        ),
        (
            """*** Begin Patch
*** Update File: /tmp/example.php
@@ -1721,6 +1721,17 @@
 old line
+new line
*** End Patch""",
            """*** Begin Patch
*** Update File: /tmp/example.php
@@
 old line
+new line
*** End Patch""",
        ),
    ],
)
def test_normalize_apply_patch_input_recovers_claude_patch_dialects(raw, expected):
    assert _normalize_apply_patch_input(json.dumps({"patch": raw})) == expected


def test_claude_failed_patch_count_is_scoped_to_latest_user_task():
    body = {
        "input": [
            {"role": "user", "content": "old task"},
            {
                "type": "custom_tool_call_output",
                "output": "apply_patch verification failed",
            },
            {"role": "user", "content": "current task"},
            {
                "type": "custom_tool_call_output",
                "output": "apply_patch verification failed: invalid hunk",
            },
            {
                "type": "function_call_output",
                "output": "failed to find expected lines",
            },
        ]
    }

    assert _claude_current_turn_failed_patch_count(body) == 2


def test_claude_guidance_switches_away_from_repeated_invalid_patch():
    responses_body = {
        "input": [
            {"role": "user", "content": "fix it"},
            {
                "type": "custom_tool_call_output",
                "output": "apply_patch verification failed",
            },
            {
                "type": "custom_tool_call_output",
                "output": "apply_patch verification failed",
            },
        ]
    }
    chat_body = {
        "messages": [{"role": "user", "content": "fix it"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "apply_patch", "parameters": {}},
            },
            {
                "type": "function",
                "function": {"name": "exec_command", "parameters": {}},
            },
        ],
    }

    _add_claude_execution_guidance(chat_body, responses_body)

    guidance = chat_body["messages"][0]["content"]
    assert "Never emit `*** Hunk`" in guidance
    assert "stop using apply_patch for this turn" in guidance
    assert [tool["function"]["name"] for tool in chat_body["tools"]] == [
        "exec_command"
    ]


async def test_claude_placeholder_is_suppressed_and_patch_is_normalized(
    monkeypatch,
    tmp_path,
):
    raw_patch = """*** Begin Patch
*** Update 1: /tmp/example.php
*** Find
old line
*** Replace
new line
*** End Patch"""

    async def gateway(request):
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": "(empty placeholder)",
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_patch",
                                    "type": "function",
                                    "function": {
                                        "name": "apply_patch",
                                        "arguments": json.dumps({"patch": raw_patch}),
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {"delta": {}, "finish_reason": "tool_calls"}
                ]
            },
        ]
        for chunk in chunks:
            await response.write(
                f"data: {json.dumps(chunk)}\n\n".encode()
            )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", gateway)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()
    monkeypatch.setenv(
        "CLAUDE_GATEWAY_URL",
        str(upstream_client.make_url("/v1/chat/completions")),
    )

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim = ShimServer(settings)

    async def proxy(request):
        return await shim._claude_gateway_fallback(
            request,
            {
                "model": "claude-sonnet-4.6",
                "input": "fix it",
                "tools": [{"type": "apply_patch"}],
                "stream": True,
            },
        )

    app = web.Application()
    app.router.add_post("/test", proxy)
    app.on_cleanup.append(shim._cleanup)
    client = TestClient(TestServer(app))
    await client.start_server()

    response = await client.post("/test")
    assert response.status == 200
    events = _sse_events(await response.text())
    serialized = json.dumps(events, ensure_ascii=False)
    assert "(empty placeholder)" not in serialized
    done = [
        event
        for event in events
        if event.get("type") == "response.custom_tool_call_input.done"
    ][-1]
    assert done["input"] == """*** Begin Patch
*** Update File: /tmp/example.php
@@
-old line
+new line
*** End Patch"""

    await client.close()
    await upstream_client.close()


async def test_responses_first_platform_deepseek_pro_selects_pro(monkeypatch, tmp_path):
    captured = {}

    async def deepseek_passthrough(self, request, body, path):
        captured["body"] = body
        captured["path"] = path
        return web.json_response({"ok": True})

    monkeypatch.setattr(ShimServer, "_deepseek_passthrough", deepseek_passthrough)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "input": "hi",
            "first_platform": "deepseek-pro",
        },
    )

    assert resp.status == 200
    assert captured["body"]["model"] == "deepseek-v4-pro"
    assert captured["path"] == "/v1/responses"
    await shim_client.close()


async def test_deepseek_running_uses_codeproxy_responses_endpoint(
    monkeypatch,
    tmp_path,
):
    captured = {}

    async def responses(request):
        captured["body"] = await request.json()
        return web.json_response(
            {
                "id": "resp_deepseek",
                "object": "response",
                "status": "completed",
                "model": "deepseek",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/responses", responses)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    monkeypatch.setattr(
        server_module,
        "DEEPSEEK_API_BASE",
        str(upstream_client.make_url("")).rstrip("/"),
    )
    monkeypatch.setattr(server_module, "_get_deepseek_api_key", lambda: "secret")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim = ShimServer(settings)

    async def proxy(request):
        async with ClientSession() as session:
            return await shim._deepseek_passthrough_running(
                request,
                {
                    "model": "deepseek",
                    "stream": False,
                    "input": [{"role": "user", "content": "fix it"}],
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "mcp__codegraph__",
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "codegraph_explore",
                                    "parameters": {"type": "object"},
                                }
                            ],
                        }
                    ],
                },
                session,
            )

    app = web.Application()
    app.router.add_post("/test", proxy)
    client = TestClient(TestServer(app))
    await client.start_server()

    response = await client.post("/test")
    assert response.status == 200
    assert (await response.json())["id"] == "resp_deepseek"
    assert captured["body"]["model"] == server_module.DEEPSEEK_MODEL_STANDARD
    assert captured["body"]["stream"] is False
    assert captured["body"]["tools"][0]["type"] == "namespace"

    await client.close()
    await upstream_client.close()


async def test_explicit_deepseek_pro_never_falls_back_to_claude_when_unavailable(
    monkeypatch,
    tmp_path,
):
    async def claude(*args, **kwargs):
        pytest.fail("explicit DeepSeek Pro request must never route to Claude")

    monkeypatch.setattr(server_module, "_deepseek_available", lambda: False)
    monkeypatch.setattr(ShimServer, "_claude_gateway_fallback", claude)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "input": "[deepseek-pro] continue fixing",
            "stream": True,
        },
    )

    assert resp.status == 503
    assert "will not switch models" in await resp.text()
    await shim_client.close()


async def test_responses_defaults_to_claude_even_for_chatgpt_model(
    monkeypatch,
    tmp_path,
    auth_present,
):
    captured = {}

    async def claude(self, request, body, response_model_override=None, prepared_response=None):
        captured["claude"] = body
        captured["model"] = response_model_override
        return web.json_response({"platform": "claude"})

    async def chatgpt(*args, **kwargs):
        pytest.fail("default request must not route to ChatGPT")

    monkeypatch.setattr(ShimServer, "_claude_gateway_fallback", claude)
    monkeypatch.setattr(ShimServer, "_chatgpt_passthrough", chatgpt)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={"model": "gpt-5.4", "input": "hi", "stream": True},
    )

    assert resp.status == 200
    assert await resp.json() == {"platform": "claude"}
    assert captured["model"] == "claude-sonnet-4.6"
    await shim_client.close()


@pytest.mark.parametrize(
    ("request_body", "headers"),
    [
        (
            {"model": "gpt-5.4", "input": "hi", "first_platform": "chatgpt"},
            None,
        ),
        ({"model": "gpt-5.4", "input": "[chatgpt] hi"}, None),
        (
            {"model": "gpt-5.4", "input": "hi"},
            {"x-codex-shim-platform": "chatgpt"},
        ),
        (
            {"model": "gpt-5.4", "input": "hi"},
            {"Authorization": "Bearer local-codex-shim-chatgpt"},
        ),
    ],
)
async def test_responses_only_explicit_chatgpt_routes_to_chatgpt(
    monkeypatch,
    tmp_path,
    auth_present,
    request_body,
    headers,
):
    captured = {}

    async def chatgpt(self, request, body, response_model_override=None, upstream_model=None):
        captured["body"] = body
        return web.json_response({"platform": "chatgpt"})

    async def claude(*args, **kwargs):
        pytest.fail("explicit ChatGPT request must not route to Claude")

    monkeypatch.setattr(ShimServer, "_chatgpt_passthrough", chatgpt)
    monkeypatch.setattr(ShimServer, "_claude_gateway_fallback", claude)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json=request_body,
        headers=headers,
    )

    assert resp.status == 200
    assert await resp.json() == {"platform": "chatgpt"}
    assert "first_platform" not in captured["body"]
    await shim_client.close()


async def test_explicit_chatgpt_failure_does_not_fallback_to_claude(
    monkeypatch,
    tmp_path,
    auth_present,
):
    class FailedResponse:
        status = 400

        async def text(self):
            return "invalid history item id"

        def release(self):
            return None

    class FakeSession:
        async def post(self, *args, **kwargs):
            return FailedResponse()

    async def get_session(self):
        return FakeSession()

    async def claude(*args, **kwargs):
        pytest.fail("explicit ChatGPT request must not fall back to Claude")

    monkeypatch.setattr(ShimServer, "_get_session", get_session)
    monkeypatch.setattr(ShimServer, "_claude_gateway_fallback", claude)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    client = TestClient(TestServer(ShimServer(settings).app()))
    await client.start_server()
    try:
        resp = await client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.6-sol",
                "stream": True,
                "input": "[chatgpt-5.6] hello",
            },
        )
        text = await resp.text()
        assert "explicit_model_failed" in text
        assert "provider fallback is disabled" in text
    finally:
        await client.close()


async def test_chatgpt_platform_survives_context_compaction(
    monkeypatch,
    tmp_path,
    auth_present,
):
    routed = []

    async def chatgpt(self, request, body, response_model_override=None, upstream_model=None):
        routed.append(("responses", body))
        return web.json_response({"platform": "chatgpt"})

    async def compact(self, request, body, upstream_model=None):
        routed.append(("compact", body))
        return web.json_response(
            {
                "id": "resp_compact",
                "status": "completed",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Context automatically compacted.",
                            }
                        ],
                    }
                ],
            }
        )

    async def claude(*args, **kwargs):
        pytest.fail("remembered ChatGPT session must not route to Claude")

    monkeypatch.setattr(ShimServer, "_chatgpt_passthrough", chatgpt)
    monkeypatch.setattr(ShimServer, "_chatgpt_compact_passthrough", compact)
    monkeypatch.setattr(ShimServer, "_claude_gateway_fallback", claude)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    headers = {"session_id": "session-chatgpt-compact"}

    first = await shim_client.post(
        "/v1/responses",
        headers=headers,
        json={"model": "gpt-5.4", "input": "[chatgpt] fix the bug"},
    )
    compacted = await shim_client.post(
        "/v1/responses/compact",
        headers=headers,
        json={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": "fix the bug"}],
        },
    )
    continued = await shim_client.post(
        "/v1/responses",
        headers=headers,
        json={
            "model": "gpt-5.4",
            "input": [
                {
                    "role": "assistant",
                    "content": "Context automatically compacted.",
                },
                {"role": "user", "content": "continue"},
            ],
        },
    )

    assert first.status == 200
    assert compacted.status == 200
    assert continued.status == 200
    assert [endpoint for endpoint, _body in routed] == [
        "responses",
        "compact",
        "responses",
    ]
    assert routed[0][1]["input"] == "fix the bug"
    await shim_client.close()


def test_rewrite_response_model_only_rewrites_chatgpt_metadata():
    payload = {
        "model": "gpt-5.6-sol",
        "nested": [{"model": "gpt-5.6-sol"}, {"model": "other"}],
    }

    _rewrite_response_model(payload, "custom-model")

    assert payload == {
        "model": "custom-model",
        "nested": [{"model": "custom-model"}, {"model": "other"}],
    }


def test_image_generation_detection_is_conservative():
    shim = ShimServer()
    tools = [
        {"type": "function", "function": {"name": "shell"}},
        {"type": "image_generation", "name": "image_generation"},
    ]

    assert shim._needs_image_gen({"tools": tools, "input": [{"role": "user", "content": "write code for an icon component"}]}) is False
    assert shim._needs_image_gen({"tools": tools, "input": [{"role": "user", "content": "@image generate a neon fox"}]}) is True
    assert shim._needs_image_gen({"tools": tools, "tool_choice": {"type": "image_generation"}, "input": "hi"}) is True
    assert shim._needs_image_followup(
        {
            "input": [
                {"type": "image_generation_call", "id": "ig_1"},
                {"role": "user", "content": "make it brighter"},
            ]
        }
    ) is True


async def test_image_generation_routes_to_chatgpt_passthrough_and_rewrites_model(monkeypatch, tmp_path, auth_present):
    captured = {}
    response_data = {"id": "resp_img", "model": "gpt-5.6-sol", "output": [{"type": "image_generation_call", "model": "gpt-5.6-sol"}]}

    class FakeContent:
        async def iter_chunked(self, size):
            yield f"data: {json.dumps(response_data)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    class FakeUpstream:
        status = 200
        content_type = "text/event-stream"
        content = FakeContent()

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["body"] = json
        captured["headers"] = headers
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "openai",
                        "baseUrl": "http://example.invalid/v1",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={
            "model": "real-openai",
            "input": [{"role": "user", "content": "@image generate a neon fox"}],
            "tools": [{"type": "image_generation", "name": "image_generation"}],
            "first_platform": "ChatGPT",
        },
    )
    assert resp.status == 200
    # Response is SSE stream, read and parse it
    body = await resp.read()
    # Verify the request was forwarded correctly
    assert captured["body"]["model"] == "gpt-5.6-sol"
    assert captured["headers"]["Authorization"] == "Bearer stub"

    await shim_client.close()


async def test_chat_completions_routes_to_openai_chat(tmp_path):
    captured = {}

    async def chat(request):
        captured["headers"] = dict(request.headers)
        captured["body"] = await request.json()
        return web.json_response(
            {
                "id": "chatcmpl_fake",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "openai",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/chat/completions",
        json={"model": "real-openai", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 200
    payload = await resp.json()
    assert payload["choices"][0]["message"]["content"].startswith("hello")
    assert payload["usage"]["prompt_tokens"] == 2
    assert payload["usage"]["completion_tokens"] == 1
    assert captured["body"]["model"] == "real-openai"
    assert captured["headers"]["Authorization"] == "Bearer secret"

    await shim_client.close()
    await upstream_client.close()


async def test_missing_api_key_env_has_model_specific_error(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "glm-5.1",
                        "displayName": "OpenCode Go GLM-5.1",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": "https://opencode.ai/zen/go/v1",
                        "apiKeyEnv": "OPENCODE_GO_API_KEY",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/chat/completions",
        json={"model": "glm-5-1", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status == 401
    text = await resp.text()
    assert "OPENCODE_GO_API_KEY" in text
    assert "CURSOR_API_KEY" not in text

    await shim_client.close()


def _sse_events(text: str) -> list[dict]:
    events = []
    for block in text.split("\n\n"):
        if not block.startswith("data:"):
            continue
        data = block.removeprefix("data:").strip()
        if data and data != "[DONE]":
            events.append(json.loads(data))
    return events


def _named_sse_events(text: str) -> list[tuple[str | None, dict]]:
    events = []
    for block in text.split("\n\n"):
        event_name = None
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = line.removeprefix("data:").strip()
        if data and data != "[DONE]":
            events.append((event_name, json.loads(data)))
    return events


async def test_kimi_hard_stop_finishes_locally_without_upstream_request(tmp_path, monkeypatch):
    shim = ShimServer(_picker_settings_file(tmp_path))

    class UnexpectedClientSession:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Kimi upstream must not be called after the hard limit")

    monkeypatch.setattr(server_module, "ClientSession", UnexpectedClientSession)

    body = {
        "model": "kimi-k3",
        "stream": True,
        "input": [{"type": "message", "role": "user", "content": "investigate"}],
    }
    for index in range(KIMI_HARD_STOP_TOOL_THRESHOLD):
        call_id = f"call-{index}"
        body["input"].extend([
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "true"}),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": "ok",
            },
        ])

    async def handler(request):
        return await shim._kimi_api_passthrough(
            request,
            body,
            model="kimi-for-coding",
            source_tag="[kimi-k3]",
        )

    app = web.Application()
    app.router.add_post("/test", handler)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post("/test")
        assert resp.status == 200
        assert resp.headers["X-Codex-Shim-Usage-Estimated"] == "true"
        events = _sse_events(await resp.text())
        completed = [event for event in events if event.get("type") == "response.completed"]
        assert len(completed) == 1
        response = completed[0]["response"]
        assert response["status"] == "completed"
        assert response["usage"]["total_tokens"] > 0
        assert "已强制停止继续调用" in response["output"][0]["content"][0]["text"]
        assert str(KIMI_HARD_STOP_TOOL_THRESHOLD) in response["output"][0]["content"][0]["text"]
    finally:
        await client.close()


async def test_kimi_non_streaming_estimates_usage_when_upstream_omits_it(
    tmp_path,
    monkeypatch,
):
    async def kimi_chat(request):
        await request.json()
        return web.json_response({
            "id": "kimi-test",
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
        })

    upstream = web.Application()
    upstream.router.add_post("/chat/completions", kimi_chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    shim = ShimServer(_picker_settings_file(tmp_path))
    monkeypatch.setattr(
        server_module,
        "KIMI_API_BASE",
        str(upstream_client.make_url("/chat/completions")),
    )
    monkeypatch.setattr(server_module, "_get_kimi_api_key", lambda: "test-key")

    body = {
        "model": "kimi-for-coding",
        "stream": False,
        "input": [{"type": "message", "role": "user", "content": "check this"}],
    }

    async def handler(request):
        return await shim._kimi_api_passthrough(
            request,
            body,
            model="kimi-for-coding",
            source_tag="[kimi-k3]",
        )

    app = web.Application()
    app.router.add_post("/test", handler)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post("/test")
        assert resp.status == 200
        assert resp.headers["X-Codex-Shim-Usage-Estimated"] == "true"
        payload = await resp.json()
        assert payload["status"] == "completed"
        assert payload["usage"]["input_tokens"] > 0
        assert payload["usage"]["output_tokens"] > 0
        assert payload["usage"]["total_tokens"] == (
            payload["usage"]["input_tokens"] + payload["usage"]["output_tokens"]
        )
    finally:
        await client.close()
        await upstream_client.close()


async def test_streaming_openai_chat_response_completed_includes_usage(tmp_path):
    async def chat(request):
        await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
        await response.write(
            b'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6,"prompt_tokens_details":{"cached_tokens":3}}}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "openai",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post("/v1/responses", json={"model": "real-openai", "input": "hi", "stream": True})
    assert resp.status == 200
    events = _sse_events(await resp.text())
    completed = [event for event in events if event.get("type") == "response.completed"][-1]
    usage = completed["response"]["usage"]
    assert usage["input_tokens"] == 4
    assert usage["output_tokens"] == 2
    assert usage["total_tokens"] == 6
    assert usage["input_tokens_details"]["cached_tokens"] == 3

    await shim_client.close()
    await upstream_client.close()


async def test_streaming_anthropic_response_completed_includes_usage():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("claude-real")
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 5,
                    "cache_read_input_tokens": 4,
                    "output_tokens": 1,
                }
            },
        },
    )
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 3},
        },
    )
    await state.finish(downstream)

    events = _sse_events(b"".join(downstream.chunks).decode())
    completed = [event for event in events if event.get("type") == "response.completed"][-1]
    usage = completed["response"]["usage"]
    assert usage["input_tokens"] == 5
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == 8
    assert usage["input_tokens_details"]["cached_tokens"] == 4


async def test_responses_stream_state_dedupes_repeated_source_tags_in_text():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("claude-sonnet-4.6", source_tag="[claude-sonnet-4.6]")
    await state._text_delta(
        downstream,
        "[claude-opus-4.6] [claude-opus-4.6] 已同步。",
    )
    await state.finish(downstream)

    events = _sse_events(b"".join(downstream.chunks).decode())
    deltas = [event["delta"] for event in events if event.get("type") == "response.output_text.delta"]
    assert deltas[0] == "[claude-sonnet-4.6] 已同步。"
    assert deltas.count("[claude-sonnet-4.6] 已同步。") == 1


async def test_responses_stream_state_blocks_tool_calls_after_claude_hard_stop():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState(
        "claude-sonnet-4.6",
        block_tool_calls=True,
        blocked_tool_name="apply_patch",
    )
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_patch",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": "*** Begin Patch",
                                },
                            }
                        ]
                    }
                }
            ]
        },
    )
    await state.finish(downstream)

    events = _sse_events(b"".join(downstream.chunks).decode())
    event_types = {event.get("type") for event in events}
    assert "response.custom_tool_call_input.done" not in event_types
    assert "response.function_call_arguments.done" not in event_types
    assert "response.completed" in event_types
    text = "".join(
        event.get("delta", "")
        for event in events
        if event.get("type") == "response.output_text.delta"
    )
    assert "Stopped a repeated Claude tool loop" in text
    assert "apply_patch" in text


def test_sanitize_bedrock_messages_drops_trailing_assistant_after_merge():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "u1"}]},
        {"role": "assistant", "content": [{"type": "thinking", "text": "hidden"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "prefill"}]},
    ]

    sanitized = ShimServer._sanitize_bedrock_messages(messages)

    assert sanitized == [
        {"role": "user", "content": [{"type": "text", "text": "u1"}]},
    ]


def test_sanitize_bedrock_messages_drops_non_user_tail_after_cleanup():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "u1"}]},
        {"role": "assistant", "content": [{"type": "thinking", "text": "hidden"}]},
        {"role": "tool", "content": [{"type": "tool_result", "text": "ok"}]},
    ]

    sanitized = ShimServer._sanitize_bedrock_messages(messages)

    assert sanitized == [
        {"role": "user", "content": [{"type": "text", "text": "u1"}]},
    ]


def test_normalize_explicit_platform_model_claude_keeps_body_model():
    body = {"model": "us-anthropic-claude-opus-5", "input": [{"role": "user", "content": "hi"}]}

    normalized = ShimServer._normalize_explicit_platform_model(body, "claude")

    assert normalized["model"] == "us-anthropic-claude-opus-5"
    assert body["model"] == "us-anthropic-claude-opus-5"


async def test_explicit_claude_platform_bypasses_stale_bedrock_model(monkeypatch, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim = ShimServer(settings)
    captured: dict[str, object] = {}

    async def fake_gateway(request, body, response_model_override=None, prepared_response=None):
        captured["body"] = body
        captured["response_model_override"] = response_model_override
        return web.json_response({"ok": True, "via": "claude-gateway"})

    monkeypatch.setattr(shim, "_claude_gateway_fallback", fake_gateway)

    app = shim.app()
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post(
            "/v1/responses",
            json={
                "model": "us-anthropic-claude-opus-5",
                "stream": False,
                "input": [{"role": "user", "content": "[claude] hi"}],
            },
        )
        assert resp.status == 200
        payload = await resp.json()
        assert payload["via"] == "claude-gateway"
        assert captured["response_model_override"] == _configured_claude_gateway_model()
        assert isinstance(captured["body"], dict)
        assert captured["body"]["model"] == "us-anthropic-claude-opus-5"
    finally:
        await client.close()


async def test_claude_45_prefix_selects_sonnet_45(monkeypatch, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim = ShimServer(settings)
    captured: dict[str, object] = {}

    async def fake_gateway(request, body, **kwargs):
        captured.update(kwargs)
        return web.json_response({"ok": True})

    monkeypatch.setattr(shim, "_claude_gateway_fallback", fake_gateway)

    client = TestClient(TestServer(shim.app()))
    await client.start_server()
    try:
        resp = await client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.6-sol",
                "stream": False,
                "input": [{"role": "user", "content": "[claude-4.5] hi"}],
            },
        )
        assert resp.status == 200
        assert captured["upstream_model_override"] == "claude-sonnet-4.5"
    finally:
        await client.close()


async def test_explicit_claude_stream_emits_response_created_before_gateway_503(monkeypatch, tmp_path):
    class FailedResponse:
        status = 503

        async def text(self):
            return "gateway unavailable"

        def release(self):
            return None

    class FakeSession:
        async def post(self, url, **kwargs):
            return FailedResponse()

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim = ShimServer(settings)

    async def fake_get_session():
        return FakeSession()

    monkeypatch.setattr(shim, "_get_session", fake_get_session)
    monkeypatch.setenv("CLAUDE_GATEWAY_API_KEY", "test-key")

    shim_client = TestClient(TestServer(shim.app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "stream": True,
            "input": [{"role": "user", "content": "[claude] test retry handling"}],
        },
    )
    assert resp.status == 200
    events = _sse_events(await resp.text())
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "error"
    assert "Claude gateway returned HTTP 503" in events[-1]["error"]["message"]

    await shim_client.close()


async def test_bedrock_stream_failure_emits_sse_error_instead_of_aborting(tmp_path):
    from aiohttp import web as aiohttp_web

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "us.anthropic.claude-opus-4-6-v1",
                        "displayName": "Bedrock Claude",
                        "provider": "bedrock",
                        "baseURL": "bedrock://local",
                    }
                ]
            }
        )
    )
    shim = ShimServer(settings)

    async def fake_post_bedrock(request, route, body, as_responses, source_tag=None):
        response = aiohttp_web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await _write_sse(
            response,
            {
                "type": "response.created",
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "status": "in_progress",
                    "model": route.slug,
                    "output": [],
                },
            },
        )
        await _write_sse(
            response,
            {
                "type": "error",
                "error": {
                    "type": "server_error",
                    "message": "Bedrock invocation failed: assistant prefill rejected",
                },
            },
        )
        await response.write_eof()
        return response

    shim._post_bedrock = fake_post_bedrock  # type: ignore[method-assign]

    shim_client = TestClient(TestServer(shim.app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "stream": True,
            "input": [{"role": "user", "content": "[claude-opus-4.6] verify"}],
        },
    )
    assert resp.status == 200
    events = _sse_events(await resp.text())
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "error"
    assert "Bedrock invocation failed" in events[-1]["error"]["message"]

    await shim_client.close()


async def test_responses_compact_defaults_to_claude_gateway(monkeypatch, tmp_path):
    captured = {}

    async def chat(request):
        captured["body"] = await request.json()
        return web.json_response(
            {
                "id": "chatcmpl_compact",
                "choices": [{"message": {"role": "assistant", "content": "Task: keep implementing compact support."}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    monkeypatch.setenv(
        "CLAUDE_GATEWAY_URL",
        str(upstream_client.make_url("/v1/chat/completions")),
    )
    monkeypatch.setenv("CLAUDE_GATEWAY_API_KEY", "test-key")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses/compact",
        json={
            "model": "gpt-5.4",
            "input": [
                {"role": "user", "content": "implement compact"},
                {"type": "function_call_output", "call_id": "call_1", "output": "tests pass"},
            ],
            "service_tier": "priority",
            "stream": True,
        },
    )
    assert resp.status == 200
    payload = await resp.json()
    assert payload["status"] == "completed"
    assert payload["model"] == "claude-sonnet-4.6"
    assert payload["output"][0]["content"][0]["text"].startswith("Task: keep implementing compact support.")
    assert payload["usage"]["input_tokens"] == 9
    assert payload["usage"]["output_tokens"] == 2
    assert payload["usage"]["total_tokens"] == 11
    assert captured["body"]["model"] == "claude-sonnet-4.6"
    assert captured["body"]["stream"] is False
    assert "service_tier" not in captured["body"]
    assert "Compact the conversation" in captured["body"]["messages"][0]["content"]

    await shim_client.close()
    await upstream_client.close()


async def test_responses_compact_chatgpt_passthrough_uses_compact_endpoint(monkeypatch, tmp_path, auth_present):
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "application/json"

        async def json(self, content_type=None):
            return {"id": "resp_compact", "model": "gpt-5.6-sol", "output": [{"type": "message", "model": "gpt-5.6-sol"}]}

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["body"] = json
        captured["headers"] = headers
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post("/v1/responses/compact", json={"model": "openai-gpt-5-5-codex-max", "input": "hi", "stream": True, "first_platform": "ChatGPT"})
    assert resp.status == 200
    payload = await resp.json()
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["output"][0]["model"] == "gpt-5.6-sol"
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses/compact"
    assert captured["body"]["model"] == "gpt-5.6-sol"
    assert "stream" not in captured["body"]
    assert captured["headers"]["Accept"] == "application/json"

    await shim_client.close()


async def test_health_and_models_include_chatgpt_passthrough_when_auth_present(tmp_path, auth_present, monkeypatch):
    missing_cache = tmp_path / "missing-models-cache.json"
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_MODELS_CACHE", missing_cache)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    health = await shim_client.get("/health")
    assert health.status == 200
    body = await health.json()
    assert body["models"] == len(FALLBACK_CHATGPT_PASSTHROUGH_SLUGS)
    assert body["chatgpt_passthrough"] is True

    models = await shim_client.get("/v1/models")
    assert models.status == 200
    payload = await models.json()
    assert sorted(model["id"] for model in payload["data"]) == sorted(FALLBACK_CHATGPT_PASSTHROUGH_SLUGS)

    await shim_client.close()


async def test_health_and_models_hide_chatgpt_passthrough_when_auth_missing(tmp_path, auth_missing):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    health = await shim_client.get("/health")
    body = await health.json()
    assert body["models"] == 0
    assert body["chatgpt_passthrough"] is False

    models = await shim_client.get("/v1/models")
    payload = await models.json()
    assert payload["data"] == []

    await shim_client.close()


@pytest.fixture
def cursor_present(monkeypatch):
    def _on(**_kwargs):
        return True

    for target in (
        "codex_shim.cursor_passthrough.cursor_passthrough_available",
        "codex_shim.server.cursor_passthrough_available",
        "codex_shim.catalog.cursor_passthrough_available",
        "codex_shim.cli.cursor_passthrough_available",
    ):
        monkeypatch.setattr(target, _on)


@pytest.fixture
def cursor_missing(monkeypatch):
    monkeypatch.setattr("codex_shim.cursor_passthrough.cursor_passthrough_available", lambda **_: False)
    monkeypatch.setattr("codex_shim.server.cursor_passthrough_available", lambda **_: False)
    monkeypatch.setattr("codex_shim.catalog.cursor_passthrough_available", lambda **_: False)


async def test_health_and_models_include_cursor_passthrough_when_auth_present(tmp_path, cursor_present, auth_missing):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    health = await shim_client.get("/health")
    assert health.status == 200
    body = await health.json()
    assert body["models"] == 1
    assert body["cursor_passthrough"] is True

    models = await shim_client.get("/v1/models")
    payload = await models.json()
    assert [model["id"] for model in payload["data"]] == ["composer-2-5"]

    await shim_client.close()


async def test_health_and_models_hide_cursor_passthrough_when_auth_missing(tmp_path, cursor_missing, auth_missing):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    health = await shim_client.get("/health")
    body = await health.json()
    assert body["models"] == 0
    assert body["cursor_passthrough"] is False

    models = await shim_client.get("/v1/models")
    payload = await models.json()
    assert payload["data"] == []

    await shim_client.close()


async def test_chat_routes_to_openai_normalizes_developer_role(tmp_path):
    captured = {}

    async def chat(request):
        captured["body"] = await request.json()
        return web.json_response({"id": "chatcmpl_fake", "choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "deepseek-reasoner",
                        "displayName": "DeepSeek Reasoner",
                        "provider": "openai",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-reasoner", "messages": [{"role": "developer", "content": "rules"}, {"role": "user", "content": "hi"}]},
    )
    assert resp.status == 200
    assert [message["role"] for message in captured["body"]["messages"]] == ["system", "user"]

    await shim_client.close()
    await upstream_client.close()


async def test_chat_routes_to_anthropic(tmp_path):
    captured = {}

    async def messages(request):
        captured["headers"] = dict(request.headers)
        captured["body"] = await request.json()
        return web.json_response({"id": "msg_fake", "content": [{"type": "text", "text": "anthropic hello"}], "stop_reason": "end_turn"})

    upstream = web.Application()
    upstream.router.add_post("/v1/messages", messages)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "claude-real",
                        "displayName": "Claude Real",
                        "provider": "anthropic",
                        "baseUrl": str(upstream_client.make_url("")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post("/v1/chat/completions", json={"model": "claude-real", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status == 200
    payload = await resp.json()
    assert payload["choices"][0]["message"]["content"] == "anthropic hello"
    assert captured["body"]["model"] == "claude-real"
    assert captured["headers"]["x-api-key"] == "secret"
    assert "Authorization" not in captured["headers"]

    await shim_client.close()
    await upstream_client.close()


async def test_anthropic_messages_routes_to_openai_chat(tmp_path):
    captured = {}

    async def chat(request):
        captured["headers"] = dict(request.headers)
        captured["body"] = await request.json()
        return web.json_response(
            {
                "id": "chatcmpl_fake",
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "openai hello"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/messages",
        json={
            "model": "real-openai",
            "system": "System",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 42,
        },
    )
    assert resp.status == 200
    payload = await resp.json()
    assert payload["type"] == "message"
    assert payload["model"] == "real-openai"
    assert payload["content"][0]["type"] == "text"
    assert payload["content"][0]["text"].startswith("openai hello")
    assert payload["usage"]["input_tokens"] == 2
    assert payload["usage"]["output_tokens"] == 1
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "real-openai"
    assert captured["body"]["max_tokens"] == 42
    assert captured["body"]["messages"] == [{"role": "system", "content": "System"}, {"role": "user", "content": "hi"}]

    await shim_client.close()
    await upstream_client.close()


async def test_anthropic_messages_passes_through_anthropic_upstream(tmp_path):
    captured = {}

    async def messages(request):
        captured["headers"] = dict(request.headers)
        captured["body"] = await request.json()
        return web.json_response(
            {
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "model": "claude-upstream",
                "content": [{"type": "text", "text": "anthropic hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/messages", messages)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "claude-upstream",
                        "displayName": "Claude Upstream",
                        "provider": "anthropic",
                        "baseUrl": str(upstream_client.make_url("")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/messages",
        json={"model": "claude-upstream", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 42},
    )
    assert resp.status == 200
    payload = await resp.json()
    assert payload["model"] == "claude-upstream"
    assert payload["content"][0]["text"].startswith("anthropic hello")
    assert captured["body"]["model"] == "claude-upstream"
    assert captured["headers"]["x-api-key"] == "secret"
    assert "Authorization" not in captured["headers"]

    await shim_client.close()
    await upstream_client.close()


async def test_anthropic_messages_streams_openai_chat_as_anthropic_sse(tmp_path):
    captured = {}

    async def chat(request):
        captured["body"] = await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
        await response.write(
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/messages",
        json={"model": "real-openai", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 42, "stream": True},
    )
    assert resp.status == 200
    text = await resp.text()
    assert "[DONE]" not in text
    events = _named_sse_events(text)
    assert [event for event, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "hello"}
    assert events[4][1]["delta"]["stop_reason"] == "end_turn"
    assert events[4][1]["usage"]["input_tokens"] == 4
    assert events[4][1]["usage"]["output_tokens"] == 2
    assert captured["body"]["stream_options"] == {"include_usage": True}

    await shim_client.close()
    await upstream_client.close()



async def test_anthropic_messages_streams_tool_calls_as_anthropic_sse(tmp_path):
    async def chat(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":""}}]}}]}\n\n'
        )
        await response.write(
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\":\\"repo\\"}"}}]}}]}\n\n'
        )
        await response.write(
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/messages",
        json={"model": "real-openai", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 42, "stream": True},
    )
    assert resp.status == 200
    text = await resp.text()
    events = _named_sse_events(text)
    event_names = [event for event, _ in events]
    assert "message_start" in event_names
    assert "content_block_start" in event_names
    tool_start = next(payload for name, payload in events if name == "content_block_start" and payload.get("content_block", {}).get("type") == "tool_use")
    assert tool_start["content_block"]["id"] == "call_1"
    assert tool_start["content_block"]["name"] == "lookup"
    tool_deltas = [payload for name, payload in events if name == "content_block_delta" and payload.get("delta", {}).get("type") == "input_json_delta"]
    assert len(tool_deltas) >= 1
    message_delta = next(payload for name, payload in events if name == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert "message_stop" in event_names

    await shim_client.close()
    await upstream_client.close()


async def test_anthropic_messages_streams_reasoning_as_anthropic_sse(tmp_path):
    async def chat(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"delta":{"reasoning_content":"let me think"}}]}\n\n'
        )
        await response.write(
            b'data: {"choices":[{"delta":{"content":"the answer"}}]}\n\n'
        )
        await response.write(
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/messages",
        json={"model": "real-openai", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 42, "stream": True},
    )
    assert resp.status == 200
    events = _named_sse_events(await resp.text())
    thinking_starts = [p for n, p in events if n == "content_block_start" and p.get("content_block", {}).get("type") == "thinking"]
    assert len(thinking_starts) == 1
    thinking_deltas = [p for n, p in events if n == "content_block_delta" and p.get("delta", {}).get("type") == "thinking_delta"]
    assert len(thinking_deltas) == 1
    assert thinking_deltas[0]["delta"]["thinking"] == "let me think"
    text_deltas = [p for n, p in events if n == "content_block_delta" and p.get("delta", {}).get("type") == "text_delta"]
    assert len(text_deltas) == 1
    assert text_deltas[0]["delta"]["text"] == "the answer"

    await shim_client.close()
    await upstream_client.close()


async def test_anthropic_messages_returns_anthropic_error_for_upstream_failure(tmp_path):
    async def chat(request):
        return web.json_response(
            {"error": {"message": "invalid api key", "type": "invalid_request_error"}},
            status=401,
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "bad-key",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/messages",
        json={"model": "real-openai", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 42},
    )
    assert resp.status == 401
    payload = await resp.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "authentication_error"
    assert "invalid api key" in payload["error"]["message"]

    await shim_client.close()
    await upstream_client.close()


async def test_anthropic_messages_streams_anthropic_passthrough(tmp_path):
    async def messages(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-upstream","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n')
        await response.write(b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n')
        await response.write(b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}\n\n')
        await response.write(b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n')
        await response.write(b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}\n\n')
        await response.write(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/messages", messages)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "claude-upstream",
                        "displayName": "Claude Upstream",
                        "provider": "anthropic",
                        "baseUrl": str(upstream_client.make_url("")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/messages",
        json={"model": "claude-upstream", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 42, "stream": True},
    )
    assert resp.status == 200
    text = await resp.text()
    events = _named_sse_events(text)
    event_names = [event for event, _ in events]
    assert event_names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    text_delta = next(payload for name, payload in events if name == "content_block_delta")
    assert text_delta["delta"]["text"] == "hello"

    await shim_client.close()
    await upstream_client.close()

def _picker_settings_file(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "kimi-k26",
                        "displayName": "Kimi K2.6",
                        "provider": "openai",
                        "baseUrl": "http://example.invalid/v1",
                        "apiKey": "k",
                    },
                    {
                        "model": "deepseek-v4-pro",
                        "displayName": "DeepSeek V4 Pro",
                        "provider": "openai",
                        "baseUrl": "http://example.invalid/v1",
                        "apiKey": "k",
                    },
                ]
            }
        )
    )
    return settings


def _stub_codex_config(monkeypatch, tmp_path, *, model: str = "kimi-k26") -> "Path":
    config = tmp_path / "config.toml"
    config.write_text(
        f'model = "{model}"\n'
        'model_provider = "codex_shim"\n'
        '\n'
        '[model_providers.codex_shim]\n'
        'name = "Codex Shim"\n'
        'base_url = "http://127.0.0.1:8765/v1"\n'
        'wire_api = "responses"\n'
    )
    monkeypatch.setattr(server_module, "CODEX_CONFIG_PATH", config)
    return config


def _picker_headers(shim: ShimServer) -> dict[str, str]:
    return {PICKER_TOKEN_HEADER: shim.picker_token}


def test_picker_html_renders_self_contained_page():
    html = _picker_html("test-token")
    assert html.startswith("<!DOCTYPE html>")
    assert "/api/models" in html
    assert "/api/switch" in html
    assert PICKER_TOKEN_HEADER in html
    assert 'const PICKER_TOKEN = "test-token";' in html


def test_picker_html_json_escapes_token():
    token = 'tok"\'</script>'
    html = _picker_html(token)
    assert 'const PICKER_TOKEN = "tok\\"\'\\u003c/script>";' in html
    assert "<script>" not in html.split("const PICKER_TOKEN = ", 1)[1].split(";", 1)[0]


def test_current_managed_model_reads_top_level_model(monkeypatch, tmp_path):
    _stub_codex_config(monkeypatch, tmp_path, model="deepseek-v4-pro")
    assert _current_managed_model() == "deepseek-v4-pro"


def test_current_managed_model_returns_none_when_config_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "CODEX_CONFIG_PATH", tmp_path / "nope.toml")
    assert _current_managed_model() is None


def test_set_active_model_rewrites_model_and_provider_name(monkeypatch, tmp_path):
    config = _stub_codex_config(monkeypatch, tmp_path)
    _set_active_model("deepseek-v4-pro", "DeepSeek V4 Pro")
    text = config.read_text()
    assert 'model = "deepseek-v4-pro"' in text
    assert 'name = "DeepSeek V4 Pro"' in text


def test_set_active_model_no_op_when_config_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "CODEX_CONFIG_PATH", tmp_path / "nope.toml")
    # Should not raise.
    _set_active_model("anything", "Anything")


async def test_picker_page_served_at_picker(tmp_path, auth_missing):
    settings = _picker_settings_file(tmp_path)
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.get("/picker")
        assert resp.status == 200
        text = await resp.text()
        assert "/api/models" in text
    finally:
        await shim_client.close()


async def test_api_models_lists_configured_models_with_active_flag(
    monkeypatch, tmp_path, auth_missing
):
    settings = _picker_settings_file(tmp_path)
    _stub_codex_config(monkeypatch, tmp_path, model="deepseek-v4-pro")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.get("/api/models")
        assert resp.status == 200
        data = await resp.json()
        slugs = [m["slug"] for m in data]
        assert slugs == ["kimi-k26", "deepseek-v4-pro"]
        active = {m["slug"]: m["active"] for m in data}
        assert active == {"kimi-k26": False, "deepseek-v4-pro": True}
    finally:
        await shim_client.close()


async def test_api_models_includes_chatgpt_when_auth_present(
    monkeypatch, tmp_path, auth_present
):
    settings = _picker_settings_file(tmp_path)
    _stub_codex_config(monkeypatch, tmp_path, model="gpt-5.6-sol")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.get("/api/models")
        data = await resp.json()
        slugs = [m["slug"] for m in data]
        assert slugs[0] == "gpt-5.6-sol"
        assert data[0]["active"] is True
    finally:
        await shim_client.close()


async def test_switch_model_rewrites_config_without_restart(
    monkeypatch, tmp_path, auth_missing
):
    settings = _picker_settings_file(tmp_path)
    config = _stub_codex_config(monkeypatch, tmp_path, model="kimi-k2.6")
    restart_calls = []
    monkeypatch.setattr(server_module, "_restart_codex_app", lambda: restart_calls.append(True))

    shim = ShimServer(settings)
    shim_client = TestClient(TestServer(shim.app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/api/switch",
            json={"slug": "deepseek-v4-pro", "restart_codex": False},
            headers=_picker_headers(shim),
        )
        assert resp.status == 200
        payload = await resp.json()
        assert payload == {"ok": True, "model": "deepseek-v4-pro", "restarted": False}
        text = config.read_text()
        assert 'model = "deepseek-v4-pro"' in text
        assert 'name = "DeepSeek V4 Pro"' in text
        assert restart_calls == []
    finally:
        await shim_client.close()


async def test_switch_model_triggers_restart_when_requested(
    monkeypatch, tmp_path, auth_missing
):
    settings = _picker_settings_file(tmp_path)
    _stub_codex_config(monkeypatch, tmp_path, model="kimi-k2.6")
    restart_calls = []
    monkeypatch.setattr(server_module, "_restart_codex_app", lambda: restart_calls.append(True))

    shim = ShimServer(settings)
    shim_client = TestClient(TestServer(shim.app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/api/switch",
            json={"slug": "deepseek-v4-pro", "restart_codex": True},
            headers=_picker_headers(shim),
        )
        assert resp.status == 200
        payload = await resp.json()
        assert payload["restarted"] is True
        assert restart_calls == [True]
    finally:
        await shim_client.close()


async def test_switch_model_rejects_missing_picker_token(monkeypatch, tmp_path, auth_missing):
    settings = _picker_settings_file(tmp_path)
    config = _stub_codex_config(monkeypatch, tmp_path, model="kimi-k2.6")
    restart_calls = []
    monkeypatch.setattr(server_module, "_restart_codex_app", lambda: restart_calls.append(True))

    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/api/switch",
            json={"slug": "deepseek-v4-pro", "restart_codex": True},
        )
        assert resp.status == 403
        assert await resp.json() == {"error": "forbidden"}
        assert 'model = "kimi-k2.6"' in config.read_text()
        assert restart_calls == []
    finally:
        await shim_client.close()


async def test_switch_model_rejects_bad_picker_token(monkeypatch, tmp_path, auth_missing):
    settings = _picker_settings_file(tmp_path)
    config = _stub_codex_config(monkeypatch, tmp_path, model="kimi-k2.6")
    restart_calls = []
    monkeypatch.setattr(server_module, "_restart_codex_app", lambda: restart_calls.append(True))

    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/api/switch",
            json={"slug": "deepseek-v4-pro", "restart_codex": True},
            headers={PICKER_TOKEN_HEADER: "wrong"},
        )
        assert resp.status == 403
        assert await resp.json() == {"error": "forbidden"}
        assert 'model = "kimi-k2.6"' in config.read_text()
        assert restart_calls == []
    finally:
        await shim_client.close()


async def test_switch_model_rejects_unknown_slug(monkeypatch, tmp_path, auth_missing):
    settings = _picker_settings_file(tmp_path)
    _stub_codex_config(monkeypatch, tmp_path)
    shim = ShimServer(settings)
    shim_client = TestClient(TestServer(shim.app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post("/api/switch", json={"slug": "nope"}, headers=_picker_headers(shim))
        assert resp.status == 404
    finally:
        await shim_client.close()


async def test_switch_model_requires_slug(tmp_path, auth_missing):
    settings = _picker_settings_file(tmp_path)
    shim = ShimServer(settings)
    shim_client = TestClient(TestServer(shim.app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post("/api/switch", json={}, headers=_picker_headers(shim))
        assert resp.status == 400
    finally:
        await shim_client.close()


@pytest.mark.asyncio
async def test_await_with_sse_heartbeats_keeps_slow_stream_read_alive():
    class Response:
        def __init__(self):
            self.writes = []

        async def write(self, data):
            self.writes.append(data)

    async def slow_event():
        import asyncio

        await asyncio.sleep(0.04)
        return 'data: {"type":"response.created"}'

    response = Response()
    result = await _await_with_sse_heartbeats(
        slow_event(),
        response,
        timeout=0.2,
        interval=0.01,
    )

    assert result == 'data: {"type":"response.created"}'
    assert response.writes
    assert all(write == b": codex-shim keep-alive\n\n" for write in response.writes)


def test_chatgpt_terminal_guard_synthesizes_completed_on_early_eof():
    guard = ChatGPTPassthroughTerminalGuard("gpt-5.6-sol", "[chatgpt-5.6]")
    guard.observe(
        {
            "type": "response.created",
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "status": "in_progress",
                "model": "gpt-5.6-sol",
                "output": [],
            },
        }
    )
    guard.observe(
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "delta": "partial answer",
        }
    )

    completed = guard.synthetic_completed()

    assert completed["type"] == "response.completed"
    assert completed["response"]["status"] == "completed"
    text = completed["response"]["output"][0]["content"][0]["text"]
    assert text.startswith("[chatgpt-5.6]")
    assert "partial answer" in text
    assert "未自动重试" in text
    assert completed["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] == 0


def test_chatgpt_terminal_guard_preserves_completed_function_call():
    guard = ChatGPTPassthroughTerminalGuard("gpt-5.6-sol", "[chatgpt-5.6]")
    guard.observe(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "fc_test",
                "type": "function_call",
                "status": "completed",
                "name": "exec_command",
                "call_id": "call_test",
                "arguments": '{"cmd":"pwd"}',
            },
        }
    )

    completed = guard.synthetic_completed()
    output = completed["response"]["output"]

    assert any(item.get("type") == "function_call" for item in output)
    assert not any(
        item.get("type") == "message"
        and "提前结束" in item.get("content", [{}])[0].get("text", "")
        for item in output
    )
