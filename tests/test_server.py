from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from codex_shim import server as server_module
from codex_shim.server import (
    PICKER_TOKEN_HEADER,
    ResponsesStreamState,
    ShimServer,
    _check_and_strip_platform_prefix,
    _custom_tool_input,
    _current_managed_model,
    _picker_html,
    _rewrite_response_model,
    _sanitize_deepseek_body,
    _sanitize_chatgpt_passthrough_body,
    _set_active_model,
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


def test_old_chatgpt_prefix_does_not_override_unprefixed_latest_turn():
    body = {
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "[chatgpt] old"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "reply"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "latest"}]},
        ]
    }

    stripped, platform = _check_and_strip_platform_prefix(body)

    assert platform is None
    assert stripped == body


def test_deepseek_context_keeps_current_task_and_compacts_optional_tools():
    body = {
        "input": [
            {"role": "user", "content": "old task"},
            {"type": "reasoning", "encrypted_content": "old", "summary": []},
            {"type": "function_call_output", "call_id": "old", "output": "x" * 10_000},
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

    assert len(sanitized["input"]) == 4
    assert sanitized["input"][0]["content"] == "old task"
    assert all(item.get("call_id") != "old" for item in sanitized["input"])
    assert "Do not end while an update_plan item is still in_progress" in sanitized["instructions"]
    assert all(item.get("type") != "reasoning" for item in sanitized["input"])
    assert [tool["name"] for tool in sanitized["tools"]] == [
        "exec_command",
        "mcp__codegraph__",
    ]
    assert len(sanitized["tools"][0]["description"]) <= 241


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
    assert sanitized["input"][-1]["content"] == "继续修复"


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
        "assistant",
        "user",
    ]
    assert sanitized["input"][0]["content"].startswith("Here is the OpenSearch DSL")
    assert sanitized["input"][1]["content"] == (
        "Use a lowercase normalizer on Username.keyword."
    )
    assert sanitized["input"][2]["content"] == "我测试了，还是不行"


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
    "request_body",
    [
        {"model": "gpt-5.4", "input": "hi", "first_platform": "chatgpt"},
        {"model": "gpt-5.4", "input": "[chatgpt] hi"},
    ],
)
async def test_responses_only_explicit_chatgpt_routes_to_chatgpt(
    monkeypatch,
    tmp_path,
    auth_present,
    request_body,
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

    resp = await shim_client.post("/v1/responses", json=request_body)

    assert resp.status == 200
    assert await resp.json() == {"platform": "chatgpt"}
    assert "first_platform" not in captured["body"]
    await shim_client.close()


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
    assert payload["choices"][0]["message"]["content"] == "hello"
    assert payload["usage"] == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
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
    assert completed["response"]["usage"] == {
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
        "input_tokens_details": {"cached_tokens": 3},
    }

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
    assert completed["response"]["usage"] == {
        "input_tokens": 5,
        "output_tokens": 3,
        "total_tokens": 8,
        "input_tokens_details": {
            "cached_tokens": 4,
            "cache_read_input_tokens": 4,
        },
    }


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
    assert payload["output"][0]["content"][0]["text"] == "Task: keep implementing compact support."
    assert payload["usage"] == {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11}
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
    assert payload["model"] == "openai-gpt-5-5-codex-max"
    assert payload["output"][0]["model"] == "openai-gpt-5-5-codex-max"
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
    assert payload["content"] == [{"type": "text", "text": "openai hello"}]
    assert payload["usage"] == {"input_tokens": 2, "output_tokens": 1}
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
    assert payload["content"][0]["text"] == "anthropic hello"
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
    assert events[4][1]["usage"] == {"input_tokens": 4, "output_tokens": 2}
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
