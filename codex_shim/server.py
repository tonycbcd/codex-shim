from __future__ import annotations

import argparse
import asyncio
import html
import ipaddress
import json
import os
import re
import secrets
import shlex
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from aiohttp import ClientSession, ClientTimeout, web

from .cursor_passthrough import (
    CURSOR_MODEL_SLUG,
    build_cursor_prompt,
    cursor_passthrough_available,
    cursor_passthrough_display_names,
    cursor_upstream_model,
    is_cursor_passthrough_slug,
    iter_cursor_agent_events,
)
from . import router as router_module
from .hostguard import build_allowed_hosts, host_guard_middleware
from .settings import (
    CHATGPT_MODEL_SLUG,
    DEFAULT_CODEX_AUTH,
    DEFAULT_SETTINGS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    PROVIDER_NAME,
    ModelSettings,
    ShimModel,
    available_model_slugs,
    chatgpt_passthrough_available,
    chatgpt_passthrough_display_names,
    chatgpt_passthrough_slugs,
    byok_model_has_credentials,
    chatgpt_upstream_model,
    is_chatgpt_passthrough_slug,
    usable_byok_models,
)
from .translate import (
    SHIM_ENCRYPTED_CONTENT_PREFIX,
    anthropic_messages_to_chat,
    anthropic_to_chat_response,
    anthropic_to_response,
    chat_completion_to_anthropic_message,
    chat_completion_to_response,
    chat_to_anthropic,
    normalize_responses_usage,
    responses_to_anthropic,
    responses_to_chat,
    _chat_finish_to_anthropic_stop,
    _responses_usage_to_anthropic_usage,
)

DEBUG_DIR = Path(__file__).resolve().parents[1] / ".codex-shim"
CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
PICKER_TOKEN_HEADER = "X-Codex-Shim-Picker-Token"
# DeepSeek Web API configuration
DEEPSEEK_API_BASE = "http://127.0.0.1:8766"
DEEPSEEK_API_KEY_FILE = Path(__file__).resolve().parents[1] / ".deepseek-web-data" / ".api-key"
DEEPSEEK_RUN_SCRIPT = Path(__file__).resolve().parents[1] / ".deepseek-web-data" / "run.sh"
DEEPSEEK_LOG_FILE = Path(__file__).resolve().parents[1] / ".deepseek-web-data" / "server.log"
DEEPSEEK_CHROME_PROFILE = Path(__file__).resolve().parents[1] / ".deepseek-web-data" / "chrome-profile"
DEEPSEEK_CDP_FILE = Path(__file__).resolve().parents[1] / ".deepseek-web-data" / "chrome.cdp"
DEEPSEEK_SESSIONS_FILE = Path(__file__).resolve().parents[1] / ".deepseek-web-data" / "sessions.json"
DEEPSEEK_MAX_SESSIONS_BYTES = 32 * 1024 * 1024
DEEPSEEK_MODEL_STANDARD = "deepseek-v4-flash"
DEEPSEEK_MODEL_PRO = "deepseek-v4-pro"
DEFAULT_CLAUDE_GATEWAY_URL = "http://127.0.0.1:8901/v1/chat/completions"
# Keep the default on a model the current Kiro subscription can actually use.
# claude-opus-4.6 returns 400 "Invalid model ID or insufficient subscription
# level" on this account, which is why default shim->Kiro requests failed.
DEFAULT_CLAUDE_GATEWAY_MODEL = "claude-sonnet-4.6"

CLAUDE_EXECUTION_RULES = """Claude Codex execution contract:
- For a small, localized code change, perform one focused discovery pass, then edit immediately.
- Prefer apply_patch for source edits. Do not use line-number-based sed/Python deletion or backup-and-restore editing.
- apply_patch accepts only Codex patch grammar: `*** Begin Patch`, then `*** Update File: path` / `*** Add File: path` / `*** Delete File: path`, bare `@@` hunks with `-` and `+` lines, and `*** End Patch`.
- Never put unified-diff line ranges in a hunk header: use `@@`, not `@@ -1721,6 +1721,17 @@`.
- Never emit `*** Hunk`, `*** Find`, `*** Replace`, bare/numbered `*** Update` hunk headers, or a plain `---/+++` unified diff to apply_patch.
- Do not repeatedly read the same file ranges or restate the same plan. Reuse tool results already present in the conversation.
- After editing, inspect the diff once and run the narrowest relevant syntax check or test.
- If an attempted edit is wrong, inspect the current diff and correct it directly instead of restoring and restarting the investigation.
- Do not claim that requested logic is fully removed until references to the removed symbols/text have been searched and the diff has been verified.
- Preserve unrelated pre-existing changes and identify them separately in the final report."""

CLAUDE_LOOP_BREAK_TOOL_THRESHOLD = 8
CLAUDE_LOOP_BREAK_RULES = """Claude tool-loop breaker:
- This turn has already used {tool_count} tool calls.
- Stop broad exploration and do not reread ranges already inspected.
- If the requested patch is not yet applied, make the smallest safe edit now with apply_patch.
- If the patch is already applied, run at most one focused verification command and then give the final answer.
- Do not announce another plan, restore from a backup, or restart the task from the beginning."""

CLAUDE_PATCH_FAILURE_RULES = """Claude apply_patch recovery:
- Previous apply_patch calls in this same task failed validation.
- Do not repeat the same patch syntax. Use only the exact Codex patch grammar described above.
- If two apply_patch attempts have already failed, stop using apply_patch for this turn. Use one focused exec_command edit, then inspect git diff and run the relevant syntax check."""

DEEPSEEK_EXECUTION_RULES = """DeepSeek Codex execution contract:
- Treat requests to modify, fix, implement, create, or refactor code as execution tasks, not explanation-only tasks. Keep using tools until the requested change is applied or a concrete blocker is proven.
- Before editing, confirm the current workdir/repository, inspect git status, and verify the target file exists. Current tool output is the source of truth; never reuse a path, branch, file, or diff from an earlier workspace without checking it again.
- Run dependent steps sequentially: locate -> inspect -> edit -> diff -> format -> test/build. Only independent read-only checks may run in parallel.
- Prefer apply_patch or another explicit file-edit tool. Preserve pre-existing user changes and do not reset, checkout, revert, or broadly reformat unrelated files.
- A failed critical tool call invalidates dependent assumptions. Recover from the failure before continuing, and never describe a planned or failed command as if it succeeded.
- You may claim completion only after a successful file write, a real diff showing the intended change, git diff --check (or equivalent), and a relevant test/build/syntax verification. If a verification cannot run, state exactly what remains unverified.
- Never fabricate file contents, diffs, command output, exit codes, test results, or deployment status. If no file was changed, explicitly say that no code was modified.
- Do not end while an update_plan item is still in_progress or while the requested modification is unverified. A progress list, artifact list, source list, diagnosis, or proposed patch is not a completed implementation.
- If a tool call fails because of arguments, immediately retry with only valid required arguments. Keep every integer tool argument as an integer, never as a decimal such as 20000.0.
- Call configured tool names exactly. For CodeGraph use codegraph_explore with query and projectPath; never call an MCP namespace name directly.
- Stay in the request's current project/workdir. Never scan / or unrelated repositories to guess the task.
- Final reports for execution tasks must name changed files, summarize the actual diff, list verification commands and their real outcomes, and distinguish pre-existing changes from this task."""


def _claude_current_turn_tool_call_count(body: dict[str, Any]) -> int:
    inputs = body.get("input")
    if not isinstance(inputs, list):
        return 0

    latest_user_index = -1
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user" and item.get("type") in {None, "message"}:
            latest_user_index = index

    return sum(
        1
        for item in inputs[latest_user_index + 1 :]
        if isinstance(item, dict)
        and item.get("type") in {"function_call", "custom_tool_call", "computer_call"}
    )


def _claude_current_turn_failed_patch_count(body: dict[str, Any]) -> int:
    inputs = body.get("input")
    if not isinstance(inputs, list):
        return 0

    latest_user_index = -1
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user" and item.get("type") in {None, "message"}:
            latest_user_index = index

    failure_pattern = re.compile(
        r"apply_patch verification failed|invalid (?:patch|hunk)|"
        r"failed to find expected lines|patch does not apply|corrupt patch",
        re.IGNORECASE,
    )
    return sum(
        1
        for item in inputs[latest_user_index + 1 :]
        if isinstance(item, dict)
        and item.get("type") in {"function_call_output", "custom_tool_call_output"}
        and failure_pattern.search(json.dumps(item, ensure_ascii=False))
    )


def _add_claude_execution_guidance(
    chat_body: dict[str, Any],
    responses_body: dict[str, Any],
) -> int:
    tool_count = _claude_current_turn_tool_call_count(responses_body)
    failed_patch_count = _claude_current_turn_failed_patch_count(responses_body)
    guidance = CLAUDE_EXECUTION_RULES
    if failed_patch_count:
        guidance += "\n\n" + CLAUDE_PATCH_FAILURE_RULES
    if failed_patch_count >= 2:
        tools = chat_body.get("tools")
        if isinstance(tools, list):
            chat_body["tools"] = [
                tool
                for tool in tools
                if not (
                    isinstance(tool, dict)
                    and isinstance(tool.get("function"), dict)
                    and tool["function"].get("name") == "apply_patch"
                )
            ]
    if tool_count >= CLAUDE_LOOP_BREAK_TOOL_THRESHOLD:
        guidance += "\n\n" + CLAUDE_LOOP_BREAK_RULES.format(tool_count=tool_count)

    messages = chat_body.get("messages")
    if not isinstance(messages, list):
        messages = []
        chat_body["messages"] = messages
    messages.insert(0, {"role": "system", "content": guidance})
    return tool_count



class ShimServer:
    def __init__(self, settings_path: Path = DEFAULT_SETTINGS, host: str = DEFAULT_HOST):
       self.settings = ModelSettings(settings_path)
       self.host = host
       self.timeout = ClientTimeout(total=None, sock_connect=30, sock_read=None)
       self.picker_token = secrets.token_urlsafe(32)
       self._session: ClientSession | None = None
       self._deepseek_lock = asyncio.Lock()
       self._deepseek_users = 0
       self._deepseek_process: asyncio.subprocess.Process | None = None
       self._deepseek_log_handle: Any | None = None
       self._deepseek_idle_handle: asyncio.TimerHandle | None = None
       self._deepseek_idle_seconds: float = 30.0
       self._session_platforms: dict[str, str] = {}

    def _resolve_session_platform(
        self,
        request: web.Request,
        explicit_platform: str | None,
    ) -> str | None:
        """Keep an explicit platform choice stable for the Codex session.

        Codex preserves ``session_id`` across automatic context compaction,
        while the compacted input no longer contains the original
        ``[chatgpt]`` prefix. Remembering the explicit choice by session keeps
        the compact request and all following turns on the same platform.
        """
        session_id = str(request.headers.get("session_id") or "").strip()
        if explicit_platform:
            if session_id:
                if (
                    session_id not in self._session_platforms
                    and len(self._session_platforms) >= 1024
                ):
                    self._session_platforms.pop(next(iter(self._session_platforms)))
                self._session_platforms[session_id] = explicit_platform
            return explicit_platform
        if not session_id:
            return None
        platform = self._session_platforms.get(session_id)
        if platform:
            print(
                f"[shim] Reusing session platform: [{platform}] "
                f"session_id={session_id}",
                flush=True,
            )
        return platform

    async def _get_session(self) -> ClientSession:
        """Return a persistent ClientSession with connection pooling."""
        if self._session is None or self._session.closed:
            from aiohttp import TCPConnector
            connector = TCPConnector(
                limit=20,
                keepalive_timeout=120,
                enable_cleanup_closed=True,
                ttl_dns_cache=600,
            )
            self._session = ClientSession(
                timeout=self.timeout,
                connector=connector,
            )
        return self._session

    async def _reset_session(self) -> ClientSession:
        """Close stale session and create a fresh one."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        return await self._get_session()

    def app(self) -> web.Application:
        allowed_hosts = build_allowed_hosts(self.host)
        app = web.Application(
            client_max_size=64 * 1024 * 1024,
            middlewares=[host_guard_middleware(allowed_hosts)],
        )
        app.router.add_get("/health", self.health)
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat_completions)
        app.router.add_post("/v1/messages", self.anthropic_messages)
        app.router.add_post("/v1/responses", self.responses)
        app.router.add_post("/v1/responses/compact", self.responses_compact)
        app.router.add_get("/picker", self.picker_page)
        app.router.add_get("/api/models", self.api_models)
        app.router.add_post("/api/switch", self.switch_model)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _cleanup(self, _app: web.Application) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        await self._stop_deepseek_service()

    async def _deepseek_healthcheck(self) -> bool:
        timeout = ClientTimeout(total=1)
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.get(f"{DEEPSEEK_API_BASE}/health") as response:
                    return response.status < 500
        except Exception:
            return False

    async def _start_deepseek_service(self) -> None:
        process = self._deepseek_process
        if process is not None and process.returncode is None:
            return
        if not DEEPSEEK_RUN_SCRIPT.is_file():
            raise RuntimeError(f"DeepSeek launcher not found: {DEEPSEEK_RUN_SCRIPT}")

        self._reset_oversized_deepseek_sessions()

        # The managed browser outlives the on-demand Node service. A browser
        # left running for many hours can accept its CDP websocket connection
        # but never finish Playwright initialization. DeepSeek Web then emits
        # an HTTP-200 empty stream, so recycle orphaned Chrome before startup.
        await self._stop_deepseek_chrome()
        DEEPSEEK_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._deepseek_log_handle = DEEPSEEK_LOG_FILE.open("ab", buffering=0)
        self._deepseek_process = await asyncio.create_subprocess_exec(
            str(DEEPSEEK_RUN_SCRIPT),
            stdout=self._deepseek_log_handle,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            process = self._deepseek_process
            if process is None or process.returncode is not None:
                code = None if process is None else process.returncode
                await self._stop_deepseek_service()
                raise RuntimeError(f"DeepSeek Web API exited during startup (code={code})")
            if await self._deepseek_healthcheck():
                print(f"[shim] DeepSeek Web API started on demand pid={process.pid}", flush=True)
                return
            await asyncio.sleep(0.25)

        await self._stop_deepseek_service()
        raise RuntimeError("DeepSeek Web API did not become healthy within 20 seconds")

    @staticmethod
    def _reset_oversized_deepseek_sessions() -> None:
        """Reset the local index before JSON serialization can OOM Node.

        Codex resends prompt history, so losing this local lookup index only
        makes the adapter establish a fresh DeepSeek conversation.
        """
        try:
            size = DEEPSEEK_SESSIONS_FILE.stat().st_size
        except FileNotFoundError:
            return
        if size <= DEEPSEEK_MAX_SESSIONS_BYTES:
            return

        temporary = DEEPSEEK_SESSIONS_FILE.with_name(
            f"{DEEPSEEK_SESSIONS_FILE.name}.{os.getpid()}.reset"
        )
        try:
            temporary.write_text(
                '{"sessions":{},"convs":{}}\n',
                encoding="utf-8",
            )
            temporary.chmod(0o600)
            temporary.replace(DEEPSEEK_SESSIONS_FILE)
        except OSError as exc:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise RuntimeError(
                f"Unable to reset oversized DeepSeek session index: {exc}"
            ) from exc

        print(
            f"[shim] Reset oversized DeepSeek session index: "
            f"{size} bytes > {DEEPSEEK_MAX_SESSIONS_BYTES}",
            flush=True,
        )

    async def _stop_deepseek_service(self) -> None:
        process = self._deepseek_process
        self._deepseek_process = None
        if process is not None and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
            print(f"[shim] DeepSeek Web API stopped after request pid={process.pid}", flush=True)
        if self._deepseek_log_handle is not None:
            self._deepseek_log_handle.close()
            self._deepseek_log_handle = None
        if process is not None:
            await self._stop_deepseek_chrome()

    @staticmethod
    def _managed_deepseek_chrome_pid() -> int | None:
        lock_path = DEEPSEEK_CHROME_PROFILE / "SingletonLock"
        try:
            lock_target = os.readlink(lock_path)
            pid = int(lock_target.rsplit("-", 1)[-1])
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, OSError, ValueError):
            return None
        expected_profile = f"--user-data-dir={DEEPSEEK_CHROME_PROFILE}".encode()
        if b"chrome" not in cmdline.lower() or expected_profile not in cmdline:
            return None
        return pid

    async def _stop_deepseek_chrome(self) -> None:
        pid = self._managed_deepseek_chrome_pid()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            else:
                for _ in range(30):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    await asyncio.sleep(0.1)
                else:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            print(f"[shim] DeepSeek managed Chrome stopped pid={pid}", flush=True)

        try:
            DEEPSEEK_CDP_FILE.unlink()
        except FileNotFoundError:
            pass
        if pid is None or not Path(f"/proc/{pid}").exists():
            for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
                try:
                    (DEEPSEEK_CHROME_PROFILE / name).unlink()
                except FileNotFoundError:
                    pass

    async def _acquire_deepseek_service(self) -> None:
        async with self._deepseek_lock:
            # Cancel any pending idle-stop timer
            if self._deepseek_idle_handle is not None:
                self._deepseek_idle_handle.cancel()
                self._deepseek_idle_handle = None
            if self._deepseek_users == 0:
                await self._start_deepseek_service()
            self._deepseek_users += 1

    async def _release_deepseek_service(self) -> None:
        async with self._deepseek_lock:
            self._deepseek_users = max(0, self._deepseek_users - 1)
            if self._deepseek_users == 0:
                # Schedule stop after idle timeout instead of stopping immediately
                self._deepseek_idle_handle = asyncio.get_event_loop().call_later(
                    self._deepseek_idle_seconds, lambda: asyncio.ensure_future(self._idle_stop_deepseek())
                )

    async def _idle_stop_deepseek(self) -> None:
        async with self._deepseek_lock:
            if self._deepseek_users == 0:
                await self._stop_deepseek_service()

    async def picker_page(self, _request: web.Request) -> web.Response:
        return web.Response(text=_picker_html(self.picker_token), content_type="text/html")

    async def api_models(self, _request: web.Request) -> web.Response:
        current = _current_managed_model()
        data: list[dict[str, Any]] = []
        router_config = self._active_router()
        if router_config is not None:
            data.append(
                {
                    "slug": router_config.slug,
                    "display_name": router_config.display_name,
                    "provider": "auto",
                    "active": current == router_config.slug,
                }
            )
        if chatgpt_passthrough_available():
            for slug, display_name in chatgpt_passthrough_display_names().items():
                data.append(
                    {
                        "slug": slug,
                        "display_name": display_name,
                        "provider": "chatgpt",
                        "active": current == slug,
                    }
                )
        if cursor_passthrough_available():
            for slug, display_name in cursor_passthrough_display_names().items():
                data.append(
                    {
                        "slug": slug,
                        "display_name": display_name,
                        "provider": "cursor",
                        "active": current == slug,
                    }
                )
        for m in usable_byok_models(self.settings.load()):
            data.append(
                {
                    "slug": m.slug,
                    "display_name": m.display_name,
                    "provider": m.provider,
                    "active": current == m.slug,
                }
            )
        return web.json_response(data)

    def _valid_picker_token(self, request: web.Request) -> bool:
        token = request.headers.get(PICKER_TOKEN_HEADER, "")
        return secrets.compare_digest(token, self.picker_token)

    async def switch_model(self, request: web.Request) -> web.Response:
        if not self._valid_picker_token(request):
            return web.json_response({"error": "forbidden"}, status=403)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        slug = str(body.get("slug") or "").strip()
        if not slug:
            return web.json_response({"error": "slug is required"}, status=400)
        models = usable_byok_models(self.settings.load())
        valid = {m.slug for m in models}
        display_for: dict[str, str] = {m.slug: m.display_name for m in models}
        router_config = self._active_router()
        if router_config is not None:
            valid.add(router_config.slug)
            display_for[router_config.slug] = router_config.display_name
        if chatgpt_passthrough_available():
            valid.update(chatgpt_passthrough_slugs())
            display_for.update(chatgpt_passthrough_display_names())
        if cursor_passthrough_available():
            valid.update(cursor_passthrough_display_names())
            display_for.update(cursor_passthrough_display_names())
        if slug not in valid:
            return web.json_response({"error": f"unknown model: {slug}"}, status=404)
        _set_active_model(slug, display_for.get(slug, slug))
        restart = bool(body.get("restart_codex"))
        if restart:
            _restart_codex_app()
        return web.json_response({"ok": True, "model": slug, "restarted": restart})

    async def health(self, _request: web.Request) -> web.Response:
        models = usable_byok_models(self.settings.load())
        chatgpt_ok = chatgpt_passthrough_available()
        cursor_ok = cursor_passthrough_available()
        passthrough_count = len(chatgpt_passthrough_slugs()) if chatgpt_ok else 0
        if cursor_ok:
            passthrough_count += len(cursor_passthrough_display_names())
        count = len(models) + passthrough_count
        return web.json_response(
            {
                "ok": True,
                "models": count,
                "chatgpt_passthrough": chatgpt_ok,
                "cursor_passthrough": cursor_ok,
                "auto_router": self._active_router() is not None,
            }
        )

    async def models(self, _request: web.Request) -> web.Response:
        now = int(time.time())
        data: list[dict[str, Any]] = []
        router_config = self._active_router()
        if router_config is not None:
            data.append(router_module.router_models_entry(router_config, now))
        if chatgpt_passthrough_available():
            data.extend(
                {"id": slug, "object": "model", "created": now, "owned_by": "chatgpt"}
                for slug in sorted(chatgpt_passthrough_slugs())
            )
        if cursor_passthrough_available():
            data.extend(
                {
                    "id": slug,
                    "object": "model",
                    "created": now,
                    "owned_by": "cursor",
                }
                for slug in sorted(cursor_passthrough_display_names())
            )
        data.extend({"id": model.slug, "object": "model", "created": now, "owned_by": "codex-shim"} for model in usable_byok_models(self.settings.load()))
        return web.json_response({"object": "list", "data": data})

    async def chat_completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        body = await self._maybe_apply_auto_router(body)
        route = self._route(body)
        if route.is_openai_chat:
            forwarded = dict(body)
            forwarded["model"] = route.model
            if "messages" in forwarded:
                forwarded["messages"] = _normalize_roles(forwarded["messages"])
            return await self._post_openai_chat(request, route, forwarded, as_responses=False)
        if route.is_anthropic:
            forwarded = chat_to_anthropic(body, route.model, route.max_output_tokens)
            return await self._post_anthropic(request, route, forwarded, as_responses=False)
        raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")

    async def anthropic_messages(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        route = self._route(body)
        if route.is_openai_chat:
            forwarded = anthropic_messages_to_chat(body, route.model, route.max_output_tokens)
            return await self._post_openai_chat_as_anthropic(request, route, forwarded)
        if route.is_anthropic:
            forwarded = dict(body)
            forwarded["model"] = route.model
            return await self._post_anthropic_messages(request, route, forwarded)
        raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")

    async def responses(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        _log_incoming_request("/v1/responses", body)
        body = await self._maybe_apply_auto_router(body)
        # Prefer an explicit prefix in the newest user message. The patched
        # Codex client may instead send the same choice as first_platform.
        body, prefix_platform = _check_and_strip_platform_prefix(body)
        requested_platform = str(body.pop("first_platform", "") or "").strip().lower()
        header_platform = _prime_loopback_platform(request)
        # first_platform is an explicit per-request routing decision and must
        # override prefixes retained in older conversation history.
        explicit_platform = requested_platform or header_platform or prefix_platform or None
        platform = self._resolve_session_platform(request, explicit_platform)
        if platform:
            print(f"[shim] Platform prefix detected: [{platform}]", flush=True)
            if platform == "deepseek":
                if _deepseek_available():
                    body["model"] = DEEPSEEK_MODEL_STANDARD
                    return await self._deepseek_passthrough(request, body, "/v1/responses")
                else:
                    raise web.HTTPServiceUnavailable(
                        text="DeepSeek is not available; explicit [deepseek] requests will not switch models"
                    )
            elif platform == "deepseek-pro":
                if _deepseek_available():
                    body["model"] = DEEPSEEK_MODEL_PRO
                    return await self._deepseek_passthrough(request, body, "/v1/responses")
                else:
                    raise web.HTTPServiceUnavailable(
                        text=(
                            "DeepSeek Pro is not available; explicit [deepseek-pro] "
                            "requests will not switch models"
                        )
                    )
            elif platform == "chatgpt":
                if chatgpt_passthrough_available():
                    model = str(body.get("model") or CHATGPT_MODEL_SLUG)
                    return await self._chatgpt_passthrough(
                        request,
                        body,
                        upstream_model=chatgpt_upstream_model(model),
                    )
                else:
                    print("[shim] ⚠️  ChatGPT not available, falling back to Claude", flush=True)
            elif platform in {"claud", "claude", "kiro"}:
                pass
            else:
                print(
                    f"[shim] Unknown platform [{platform}], using default Claude gateway",
                    flush=True,
                )

        model = str(body.get("model") or "")
        if not is_chatgpt_passthrough_slug(model):
            if is_cursor_passthrough_slug(model):
                return await self._cursor_passthrough(
                    request,
                    body,
                    response_model_override=model,
                    upstream_model=cursor_upstream_model(model),
                )
            if self._needs_image_gen(body) or self._needs_image_followup(body):
                return await self._chatgpt_passthrough(request, body, response_model_override=model)
            route = self._route(body)
            if route.is_openai_chat:
                forwarded = responses_to_chat(body, route.model)
                return await self._post_openai_chat(request, route, forwarded, as_responses=True)
            if route.is_anthropic:
                forwarded = responses_to_anthropic(body, route.model, route.max_output_tokens)
                return await self._post_anthropic(request, route, forwarded, as_responses=True)
            raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")

        # Claude/Kiro is the default platform. A ChatGPT model slug selected
        # in the Codex UI must not silently opt into ChatGPT Web; only the
        # explicit [chatgpt] prefix or first_platform=chatgpt may do that.
        print("[shim] Routing request to default Claude gateway", flush=True)
        claude_result = await self._claude_gateway_fallback(
            request,
            body,
            response_model_override=DEFAULT_CLAUDE_GATEWAY_MODEL,
        )
        if claude_result is not None:
            return claude_result
        raise web.HTTPBadGateway(text="Claude gateway on port 8901 is unavailable")

    async def responses_compact(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        _log_incoming_request("/v1/responses/compact", body)
        body = await self._maybe_apply_auto_router(body)
        body, prefix_platform = _check_and_strip_platform_prefix(body)
        requested_platform = str(body.pop("first_platform", "") or "").strip().lower()
        header_platform = _prime_loopback_platform(request)
        explicit_platform = requested_platform or header_platform or prefix_platform or None
        platform = self._resolve_session_platform(request, explicit_platform)
        if platform == "chatgpt" and chatgpt_passthrough_available():
            model = str(body.get("model") or CHATGPT_MODEL_SLUG)
            upstream = chatgpt_upstream_model(model)
            return await self._chatgpt_compact_passthrough(request, body, upstream_model=upstream)
        if platform == "chatgpt":
            print("[shim] ChatGPT compact unavailable; using default Claude gateway", flush=True)
        model = str(body.get("model") or "")
        if not is_chatgpt_passthrough_slug(model):
            if is_cursor_passthrough_slug(model):
                compact_body = dict(body)
                compact_body["input"] = body.get("input") or []
                compact_body["instructions"] = (
                    f"{body.get('instructions') or ''}\n\nSummarize the conversation above into a compact "
                    "context window suitable for continuing the task."
                ).strip()
                return await self._cursor_passthrough(
                    request,
                    compact_body,
                    response_model_override=model,
                    upstream_model=cursor_upstream_model(model),
                    force_non_stream=True,
                )
            route = self._route(body)
            compact_body = _compact_request_body(body, route.model)
            if route.is_openai_chat:
                forwarded = responses_to_chat(compact_body, route.model)
                forwarded["stream"] = False
                response = await self._post_openai_chat(request, route, forwarded, as_responses=True)
                return await _as_compact_response(response, route.slug)
            if route.is_anthropic:
                forwarded = responses_to_anthropic(compact_body, route.model, route.max_output_tokens)
                forwarded["stream"] = False
                response = await self._post_anthropic(request, route, forwarded, as_responses=True)
                return await _as_compact_response(response, route.slug)
            raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")
        return await self._claude_gateway_compact(body)

    async def _claude_gateway_compact(self, body: dict[str, Any]) -> web.Response:
        """Compact context through the default Kiro/Claude gateway."""
        claude_url = os.environ.get(
            "CLAUDE_GATEWAY_URL",
            DEFAULT_CLAUDE_GATEWAY_URL,
        ).strip()
        claude_key = os.environ.get(
            "CLAUDE_GATEWAY_API_KEY",
            os.environ.get("CODEX_SHIM_FALLBACK_KEY", "my-super-secret-password-123"),
        )
        claude_model = os.environ.get(
            "CLAUDE_GATEWAY_MODEL",
            DEFAULT_CLAUDE_GATEWAY_MODEL,
        ).strip()
        compact_body = _compact_request_body(body, claude_model)
        chat_body = responses_to_chat(compact_body, claude_model)
        chat_body["stream"] = False
        headers = {
            "Authorization": f"Bearer {claude_key}",
            "Content-Type": "application/json",
        }
        session = await self._get_session()
        try:
            import asyncio

            upstream = await asyncio.wait_for(
                session.post(claude_url, json=chat_body, headers=headers),
                timeout=90,
            )
        except Exception as exc:
            raise web.HTTPBadGateway(text=f"Claude compact request failed: {exc}") from exc
        try:
            if upstream.status >= 400:
                error_text = await upstream.text()
                raise web.HTTPBadGateway(
                    text=f"Claude compact request returned {upstream.status}: {error_text[:200]}"
                )
            payload = await upstream.json(content_type=None)
        finally:
            upstream.release()
        response_payload = chat_completion_to_response(
            payload,
            claude_model,
            _build_tool_types(compact_body),
        )
        summary = _compact_summary_from_output(response_payload.get("output"))
        return web.json_response(
            _compact_response_payload(
                claude_model,
                summary,
                response_payload.get("usage"),
            )
        )

    def _needs_image_gen(self, body: dict[str, Any]) -> bool:
        tools = body.get("tools") or []
        image_tool_names: set[str] = set()
        non_image_tool_count = 0
        for tool in tools:
            if not isinstance(tool, dict):
                non_image_tool_count += 1
                continue
            tool_type = str(tool.get("type") or "")
            fn = tool.get("function") or tool.get("name") or {}
            name = fn.get("name") if isinstance(fn, dict) else fn
            normalized = f"{tool_type} {name or ''}".lower()
            is_image_tool = tool_type in {"image_generation", "image_gen"} or ("image" in normalized and "gen" in normalized)
            if is_image_tool:
                image_tool_names.add(str(name or tool_type))
            else:
                non_image_tool_count += 1
        if not image_tool_names:
            return False

        tool_choice = body.get("tool_choice")
        if isinstance(tool_choice, str):
            if any(name.lower() in tool_choice.lower() for name in image_tool_names):
                return True
        elif isinstance(tool_choice, dict):
            fn = tool_choice.get("function") or {}
            choice_name = str(tool_choice.get("name") or (fn.get("name") if isinstance(fn, dict) else "") or tool_choice.get("type") or "").lower()
            if any(name.lower() in choice_name for name in image_tool_names):
                return True

        if non_image_tool_count == 0:
            return True

        latest = self._latest_user_text(body).lower()
        if not latest:
            return False
        image_intent_markers = (
            "@image",
            "imagegen",
            "image gen",
            "image_gen",
            "generate image",
            "generate an image",
            "generate a picture",
            "generate a photo",
            "generate an illustration",
            "create image",
            "create an image",
            "create a picture",
            "create a photo",
            "draw image",
            "draw an image",
            "make image",
            "make an image",
            "render image",
        )
        if any(marker in latest for marker in image_intent_markers):
            return True
        code_words = {"code", "component", "react", "tsx", "jsx", "html", "css", "svg", "file"}
        latest_words = {"".join(ch for ch in word if ch.isalnum()) for word in latest.split()}
        if latest_words & code_words:
            return False
        creative_objects = ("icon", "logo", "wallpaper", "poster", "banner", "avatar")
        creative_verbs = ("generate", "create", "draw", "design", "make", "render")
        return any(verb in latest for verb in creative_verbs) and any(obj in latest for obj in creative_objects)

    def _needs_image_followup(self, body: dict[str, Any]) -> bool:
        if not self._has_image_generation_history(body):
            return False
        latest = self._latest_user_text(body).lower()
        if not latest:
            return False
        direct_image_refs = ("image", "picture", "photo", "icon", "logo", "illustration")
        followup_actions = (
            "inspect",
            "look at",
            "view",
            "describe",
            "what do you see",
            "analyze",
            "modify",
            "edit",
            "change",
            "improve",
            "enhance",
            "upscale",
            "variation",
            "use",
            "based on",
            "same",
        )
        if any(ref in latest for ref in direct_image_refs) and any(action in latest for action in followup_actions):
            return True
        pronoun_followups = (
            "inspect it",
            "look at it",
            "view it",
            "describe it",
            "analyze it",
            "modify it",
            "edit it",
            "change it",
            "improve it",
            "enhance it",
            "upscale it",
            "make it brighter",
            "make it darker",
            "make it more",
            "use it",
            "based on it",
        )
        return any(marker in latest for marker in pronoun_followups)

    def _has_image_generation_history(self, body: dict[str, Any]) -> bool:
        inputs = body.get("input") or []
        if not isinstance(inputs, list):
            return False
        return any(isinstance(item, dict) and item.get("type") == "image_generation_call" for item in inputs)

    def _latest_user_text(self, body: dict[str, Any]) -> str:
        inputs = body.get("input") or []
        if isinstance(inputs, str):
            return inputs
        if not isinstance(inputs, list):
            return ""
        for item in reversed(inputs):
            if isinstance(item, str):
                return item
            if not isinstance(item, dict):
                continue
            if item.get("role") == "user":
                text = self._content_to_debug_text(item.get("content"))
                if text:
                    return text
            elif item.get("type") in {"input_text", "text"}:
                text = self._content_to_debug_text(item)
                if text:
                    return text
        return ""

    def _content_to_debug_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            return str(content.get("text") or content.get("content") or "")
        return str(content)

    async def _chatgpt_passthrough(
        self,
        request: web.Request,
        body: dict[str, Any],
        response_model_override: str | None = None,
        upstream_model: str | None = None,
    ) -> web.StreamResponse:
        """Forward a Responses request to chatgpt.com using the user's Codex auth.

        Lets the picker expose OpenAI GPT models (ChatGPT subscription) as
        first-class models alongside configured BYOK entries.
        """
        auth_path = DEFAULT_CODEX_AUTH.expanduser()
        try:
            auth = json.loads(auth_path.read_text())
        except FileNotFoundError:
            raise web.HTTPUnauthorized(text="~/.codex/auth.json not found")
        tokens = auth.get("tokens") or {}
        access_token = tokens.get("access_token")
        account_id = tokens.get("account_id") or ""
        if not access_token:
            raise web.HTTPUnauthorized(text="auth.json has no access_token")
        forwarded = _sanitize_chatgpt_passthrough_body(body)
        requested_model = str(forwarded.get("model") or CHATGPT_MODEL_SLUG)
        original_client_model = (
            upstream_model
            or (
                chatgpt_upstream_model(requested_model)
                if is_chatgpt_passthrough_slug(requested_model)
                else CHATGPT_MODEL_SLUG
            )
        )
        forwarded["model"] = original_client_model
        # Force reasoning effort to medium for ChatGPT passthrough
        if isinstance(forwarded.get("reasoning"), dict):
            forwarded["reasoning"]["effort"] = "medium"
        else:
            forwarded["reasoning"] = {"effort": "medium"}
        # ChatGPT /codex/responses requires input to be a list
        if isinstance(forwarded.get("input"), str):
            forwarded["input"] = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": forwarded["input"]}]}]
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if forwarded.get("stream") else "application/json",
            "OpenAI-Beta": "responses=2026-02-06",
            "originator": "codex_cli_rs",
            "chatgpt-account-id": account_id,
            "session_id": request.headers.get("session_id", ""),
        }
        url = "https://chatgpt.com/backend-api/codex/responses"
        import asyncio as _asyncio
        from aiohttp import ClientConnectorError, ServerDisconnectedError, ClientOSError
        import os
        FALLBACK_TIMEOUT = int(os.environ.get("CODEX_SHIM_FALLBACK_TIMEOUT", "60"))
        session = await self._get_session()
        response = _sse_response()
        await response.prepare(request)
        await _safe_write(response, b": codex-shim connected\n\n")

        # Honor an explicitly selected ChatGPT model first. The configured
        # default remains a capacity fallback for compatible requests.
        # Fallback models will be added dynamically if "at capacity" error is detected
        models_to_try = [original_client_model]
        if CHATGPT_MODEL_SLUG != original_client_model:
            models_to_try.append(CHATGPT_MODEL_SLUG)
        capacity_fallbacks_added = False

        timed_out = False
        model_idx = 0
        try_model = models_to_try[0]  # Initialize for type checker
        while model_idx < len(models_to_try):
            try_model = models_to_try[model_idx]
            forwarded["model"] = try_model
            t0 = time.time()
            max_retries = 5
            model_failed = False
            upstream = None
            for attempt in range(max_retries):
                try:
                    upstream = await _await_with_sse_heartbeats(
                        session.post(url, json=forwarded, headers=headers),
                        response,
                        timeout=FALLBACK_TIMEOUT,
                    )
                except _asyncio.TimeoutError:
                    elapsed = time.time() - t0
                    print(f"\n{'='*60}", flush=True)
                    print(f"[shim] ⚠️  ChatGPT TIMEOUT ({try_model}) after {elapsed:.1f}s", flush=True)
                    print(f"{'='*60}\n", flush=True)
                    model_failed = True
                    break
                except (ClientConnectorError, ServerDisconnectedError, ClientOSError, ConnectionResetError) as e:
                    if attempt < max_retries - 1:
                        print(f"[shim] connection error, resetting session (attempt {attempt+1}): {e}", flush=True)
                        session = await self._reset_session()
                        await _asyncio.sleep(1)
                        t0 = time.time()
                        continue
                    model_failed = True
                    break
                t1 = time.time()
                print(f"[shim] POST /codex/responses status={upstream.status} elapsed={t1-t0:.2f}s attempt={attempt+1} model={try_model}", flush=True)
                if upstream.status in (429, 503, 529):
                    body_text = await upstream.text()
                    # usage_limit_reached is a hard limit for this model
                    if "usage_limit_reached" in body_text:
                        print(f"\n{'='*60}", flush=True)
                        print(f"[shim] ⚠️  ChatGPT USAGE LIMIT REACHED for {try_model}", flush=True)
                        print(f"{'='*60}\n", flush=True)
                        model_failed = True
                        break
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt + 1
                        print(f"[shim] retrying in {wait}s: {body_text[:200]}", flush=True)
                        await _asyncio.sleep(wait)
                        t0 = time.time()
                        continue
                    # Last attempt still rate-limited
                    print(f"\n{'='*60}", flush=True)
                    print(f"[shim] ⚠️  ChatGPT RATE-LIMITED after {max_retries} attempts for {try_model}", flush=True)
                    print(f"{'='*60}\n", flush=True)
                    model_failed = True
                    break
                if upstream.status >= 400:
                    err_text = await upstream.text()
                    print(f"\n{'='*60}", flush=True)
                    print(f"[shim] ⚠️  ChatGPT FAILED (status {upstream.status}) for {try_model}", flush=True)
                    print(f"[shim] Error: {err_text[:150]}", flush=True)
                    print(f"{'='*60}\n", flush=True)
                    # Check for "at capacity" error and add limited fallback models (up to gpt-5.3-codex)
                    if "at capacity" in err_text.lower() and not capacity_fallbacks_added:
                        capacity_fallbacks_added = True
                        for fb_model in ("gpt-5.5", "gpt-5.4", "gpt-5.3-codex"):
                            if fb_model not in models_to_try:
                                models_to_try.append(fb_model)
                        print(f"[shim] Model at capacity, added fallbacks: {models_to_try}", flush=True)
                    model_failed = True
                    break
                break

            if not model_failed:
                break  # Success, proceed with streaming
            # If this model failed but there's another to try, continue loop
            model_idx += 1
            if model_idx < len(models_to_try):
                print(f"[shim] Trying next model: {models_to_try[model_idx]}", flush=True)
                continue
            # All ChatGPT models exhausted, fallback to OpenAI API
            print(f"[shim] All ChatGPT models exhausted, trying Claude gateway", flush=True)
            timed_out = True

        if timed_out:
            claude_result = await self._claude_gateway_fallback(
                request,
                body,
                response_model_override,
                prepared_response=response,
            )
            if claude_result is not None:
                return claude_result
            print(f"[shim] Claude gateway also failed, switching to OpenAI API fallback", flush=True)
            return await self._openai_api_fallback(
                request,
                body,
                response_model_override,
                prepared_response=response,
            )



        if not forwarded.get("stream"):
            payload = await upstream.json(content_type=None)
            _rewrite_response_model(payload, response_model_override)
            return web.json_response(payload)

        # --- In-stream error detection: buffer early data before committing ---
        _STREAM_ERROR_PHRASES = (
            "high demand",
            "experiencing high demand",
            "temporarily unavailable",
            "server_error",
            "overloaded",
            "capacity",
        )

        def _stream_has_error(text: str) -> bool:
            low = text.lower()
            return any(phrase in low for phrase in _STREAM_ERROR_PHRASES)

        if response_model_override:
            # Buffer SSE lines until we see actual content or an error.
            # "high demand" errors arrive AFTER metadata events (response.created,
            # response.in_progress) but BEFORE any content delta. So we buffer
            # until we see a content delta, then commit. If we see an error first,
            # we fallback.
            buffered_lines: list[str] = []
            stream_error_detected = False
            lines_iter = _sse_lines(upstream)
            _sentinel = object()
            # Buffer up to 30 events (covers metadata + error/first-content)
            for _ in range(30):
                try:
                    line = await _asyncio.wait_for(anext(lines_iter, _sentinel), timeout=15)  # type: ignore[arg-type]
                except _asyncio.TimeoutError:
                    break
                if line is _sentinel:
                    break
                buffered_lines.append(line)
                if line == "[DONE]":
                    break
                if _stream_has_error(line):
                    stream_error_detected = True
                    break
                # If we see a content delta, it means real content is flowing — safe to commit
                if '"output_text.delta"' in line or '"response.output_item.added"' in line:
                    break

            if stream_error_detected:
                upstream.release()
                error_line = buffered_lines[-1] if buffered_lines else ""
                print(f"\n{'='*60}", flush=True)
                print(f"[shim] ⚠️  ChatGPT in-stream ERROR detected for {try_model}", flush=True)
                print(f"[shim] Trigger: {error_line[:200]}", flush=True)
                print(f"{'='*60}\n", flush=True)
                # Check for "at capacity" and add fallback models
                if "at capacity" in error_line.lower() and not capacity_fallbacks_added:
                    capacity_fallbacks_added = True
                    for fb_model in ("gpt-5.5", "gpt-5.4", "gpt-5.3-codex"):
                        if fb_model not in models_to_try:
                            models_to_try.append(fb_model)
                    print(f"[shim] Model at capacity, added fallbacks: {models_to_try}", flush=True)
                # Try next model if available
                model_idx += 1
                if model_idx < len(models_to_try):
                    print(f"[shim] Trying next model: {models_to_try[model_idx]}", flush=True)
                    forwarded["model"] = models_to_try[model_idx]
                    try_model = models_to_try[model_idx]
                    # Reset and retry with new model
                    session = await self._get_session()
                    upstream = await _await_with_sse_heartbeats(
                        session.post(url, json=forwarded, headers=headers),
                        response,
                        timeout=FALLBACK_TIMEOUT,
                    )
                    if upstream.status == 200:
                        # Re-run stream buffering for new model
                        buffered_lines = []
                        stream_error_detected = False
                        lines_iter = _sse_lines(upstream)
                        for _ in range(30):
                            try:
                                line = await _asyncio.wait_for(anext(lines_iter, _sentinel), timeout=15)  # type: ignore[arg-type]
                            except _asyncio.TimeoutError:
                                break
                            if line is _sentinel:
                                break
                            line = str(line)  # type cast for pyright
                            buffered_lines.append(line)
                            if line == "[DONE]":
                                break
                            if _stream_has_error(line):
                                stream_error_detected = True
                                break
                            if '"output_text.delta"' in line or '"response.output_item.added"' in line:
                                break
                        if stream_error_detected:
                            # Recursively handle - but for simplicity, fall through to Claude
                            upstream.release()
                            print(f"[shim] Fallback model {try_model} also had stream error, trying Claude gateway", flush=True)
                        else:
                            # Success with fallback model - continue to streaming below
                            pass
                    else:
                        print(f"[shim] Fallback model {try_model} returned status {upstream.status}", flush=True)
                        stream_error_detected = True

                if stream_error_detected:
                    # All models failed, try Claude gateway
                    claude_result = await self._claude_gateway_fallback(
                        request,
                        body,
                        response_model_override,
                        prepared_response=response,
                    )
                    if claude_result is not None:
                        return claude_result
                    print(f"[shim] Claude gateway also failed, switching to OpenAI API fallback", flush=True)
                    return await self._openai_api_fallback(
                        request,
                        body,
                        response_model_override,
                        prepared_response=response,
                    )

            # No error — prepare response and flush buffered lines
            _source_injected = False
            # Determine source tag: show model name if using a fallback
            _source_tag = "[ChatGPT]" if try_model == CHATGPT_MODEL_SLUG else f"[ChatGPT {try_model}]"
            try:
                for line in buffered_lines:
                    if line == "[DONE]":
                        await _safe_write(response, b"data: [DONE]\n\n")
                        break
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        await _safe_write(response, f"data: {line}\n\n".encode())
                        continue
                    # Inject source tag before first content delta
                    if not _source_injected and payload.get("type") == "response.output_text.delta":
                        _source_injected = True
                        source_evt = dict(payload)
                        source_evt["delta"] = f"{_source_tag} "
                        await _write_sse(response, source_evt)
                    _rewrite_response_model(payload, response_model_override)
                    await _write_sse(response, payload)
                else:
                    # Continue streaming remaining lines
                    async for line in lines_iter:
                        if line == "[DONE]":
                            await _safe_write(response, b"data: [DONE]\n\n")
                            break
                        if _stream_has_error(line):
                            # Late error — can't fallback now, just log
                            print(f"[shim] ⚠️  late in-stream error (already streaming): {line[:150]}", flush=True)
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            await _safe_write(response, f"data: {line}\n\n".encode())
                            continue
                        # Inject source tag before first content delta
                        if not _source_injected and payload.get("type") == "response.output_text.delta":
                            _source_injected = True
                            source_evt = dict(payload)
                            source_evt["delta"] = f"{_source_tag} "
                            await _write_sse(response, source_evt)
                        _rewrite_response_model(payload, response_model_override)
                        await _write_sse(response, payload)
            except ClientDisconnected:
                pass
            finally:
                upstream.release()
        else:
            # Raw chunk path — buffer first chunk to detect errors
            first_chunk = b""
            try:
                first_chunk = await _asyncio.wait_for(
                    upstream.content.read(4096), timeout=10
                )
            except (_asyncio.TimeoutError, Exception):
                pass

            first_chunk_text = first_chunk.decode("utf-8", errors="replace") if first_chunk else ""
            if first_chunk and _stream_has_error(first_chunk_text):
                upstream.release()
                print(f"\n{'='*60}", flush=True)
                print(f"[shim] ⚠️  ChatGPT in-stream ERROR detected (raw) for {try_model}", flush=True)
                print(f"[shim] Trigger: {first_chunk_text[:200]}", flush=True)
                print(f"{'='*60}\n", flush=True)

                # Check for "at capacity" and try fallback models
                if "at capacity" in first_chunk_text.lower() and not capacity_fallbacks_added:
                    capacity_fallbacks_added = True
                    for fb_model in ("gpt-5.5", "gpt-5.4", "gpt-5.3-codex"):
                        if fb_model not in models_to_try:
                            models_to_try.append(fb_model)
                    print(f"[shim] Model at capacity (raw), added fallbacks: {models_to_try}", flush=True)

                # Try next model if available
                model_idx += 1
                if model_idx < len(models_to_try):
                    print(f"[shim] Trying next model (raw): {models_to_try[model_idx]}", flush=True)
                    forwarded["model"] = models_to_try[model_idx]
                    try_model = models_to_try[model_idx]
                    session = await self._get_session()
                    try:
                        upstream = await _await_with_sse_heartbeats(
                            session.post(url, json=forwarded, headers=headers),
                            response,
                            timeout=FALLBACK_TIMEOUT,
                        )
                        if upstream.status == 200:
                            first_chunk = await _asyncio.wait_for(upstream.content.read(4096), timeout=10)
                            first_chunk_text = first_chunk.decode("utf-8", errors="replace") if first_chunk else ""
                            if not _stream_has_error(first_chunk_text):
                                # Success - continue to streaming below
                                pass
                            else:
                                upstream.release()
                                print(f"[shim] Fallback model {try_model} also had error (raw), trying Claude gateway", flush=True)
                                first_chunk = b""  # Signal to fall through to Claude
                        else:
                            print(f"[shim] Fallback model {try_model} returned status {upstream.status} (raw)", flush=True)
                            first_chunk = b""
                    except Exception as e:
                        print(f"[shim] Fallback model {try_model} failed (raw): {e}", flush=True)
                        first_chunk = b""

                if not first_chunk:
                    # All models failed, try Claude gateway
                    claude_result = await self._claude_gateway_fallback(
                        request,
                        body,
                        response_model_override,
                        prepared_response=response,
                    )
                    if claude_result is not None:
                        return claude_result
                    print(f"[shim] Claude gateway also failed (raw), switching to OpenAI API fallback", flush=True)
                    return await self._openai_api_fallback(
                        request,
                        body,
                        response_model_override,
                        prepared_response=response,
                    )

            try:
                if first_chunk:
                    await _safe_write(response, first_chunk)
                async for chunk in upstream.content.iter_chunked(4096):
                    await _safe_write(response, chunk)
            except ClientDisconnected:
                pass
            finally:
                upstream.release()

        try:
            await response.write_eof()
        except Exception:
            pass
        return response


    @staticmethod
    def _is_malformed_tool_call(text: str) -> bool:
        """Detect XML-like pseudo tool calls emitted as plain model text."""
        return bool(
            re.search(
                r"(?:<\s*|(?m:^\s*))(?:_?calls?|tool[_-]?calls?|invoke|parameter)\b[^>\n]*>",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _recover_deepseek_tool_calls(text: str) -> list[dict[str, Any]]:
        """Recover common DeepSeek XML tool-call variants missed upstream."""

        def attribute_name(attributes: str) -> str:
            match = re.search(r'\bname\s*=\s*["\']([^"\']+)["\']', attributes, re.IGNORECASE)
            return match.group(1).strip() if match else ""

        def arguments_from_parameters(body: str) -> dict[str, Any] | None:
            arguments: dict[str, Any] = {}
            for match in re.finditer(
                r"<\s*parameter\b(?P<attrs>[^>]*)>(?P<value>[\s\S]*?)<\s*/\s*parameter\s*>",
                body,
                re.IGNORECASE,
            ):
                name = attribute_name(match.group("attrs"))
                if not name:
                    continue
                raw = html.unescape(match.group("value")).strip()
                force_string = bool(
                    re.search(r'\bstring\s*=\s*["\']true["\']', match.group("attrs"), re.IGNORECASE)
                )
                if force_string:
                    arguments[name] = raw
                    continue
                try:
                    arguments[name] = json.loads(raw)
                except (TypeError, ValueError):
                    arguments[name] = raw
            return arguments or None

        calls: list[dict[str, Any]] = []
        # DeepSeek occasionally omits only the opening ``<`` and emits
        # ``tool_call> ... </tool_call>``. Repair it before parsing so the
        # complete JSON payload is still treated as a structured tool call.
        normalized_text = re.sub(
            r"(?im)^(?P<indent>\s*)(?P<tag>tool[_-]?call|_?call|invoke)\b(?P<attrs>[^<>\n]*)>",
            r"\g<indent><\g<tag>\g<attrs>>",
            text,
        )
        normalized_text = re.sub(
            r"<\s*exec_command\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)<\s*/\s*exec_calls\s*>",
            r"<exec_command\g<attrs>>\g<body></exec_command>",
            normalized_text,
            flags=re.IGNORECASE,
        )
        normalized_text = re.sub(
            r"<\s*invoke\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)<\s*/\s*exec_calls\s*>",
            r"<invoke\g<attrs>>\g<body></invoke>",
            normalized_text,
            flags=re.IGNORECASE,
        )
        normalized_text = re.sub(
            r"<\s*/?\s*(?:tool[_-]?calls|_?calls)\b[^>]*>",
            "",
            normalized_text,
            flags=re.IGNORECASE,
        )
        direct_block = re.compile(
            r"<\s*(?P<tag>[A-Za-z_][\w:.-]*)\b(?P<attrs>[^>]*)>"
            r"(?P<body>[\s\S]*?)<\s*/\s*(?P=tag)\s*>",
            re.IGNORECASE,
        )
        for match in direct_block.finditer(normalized_text):
            tag = match.group("tag")
            if re.fullmatch(r"(?:tool[_-]?calls|_?calls|parameter)", tag, re.IGNORECASE):
                continue
            attributes = match.group("attrs")
            body_text = match.group("body").strip()
            name = attribute_name(attributes)
            if not name and tag.lower() != "invoke":
                name = tag
            if not name:
                continue

            payload: Any = None
            try:
                payload, _ = json.JSONDecoder().raw_decode(body_text)
            except (TypeError, ValueError):
                pass

            arguments: Any = None
            if isinstance(payload, dict):
                function = payload.get("function")
                nested = function if isinstance(function, dict) else payload
                payload_name = nested.get("name") or payload.get("name")
                if isinstance(payload_name, str) and payload_name.strip():
                    name = payload_name.strip()
                arguments = nested.get("arguments")
                if arguments is None:
                    arguments = {
                        key: value
                        for key, value in nested.items()
                        if key not in {"name", "function"}
                    }
            if arguments is None:
                arguments = arguments_from_parameters(body_text)
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (TypeError, ValueError):
                    pass
            if arguments is None:
                continue

            calls.append(
                {
                    "index": len(calls),
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                    },
                }
            )

        # DeepSeek occasionally truncates or corrupts the opening tag while
        # keeping a usable tool name and complete <parameter> children, e.g.
        # ``<oke name="exec_command"> ... </invoke>``. Recover that form even
        # though the XML tags no longer match.
        if not calls:
            loose_open = re.search(
                r"<\s*(?!parameter\b)[^>]*\bname\s*=\s*[\"'](?P<name>[^\"']+)[\"'][^>]*>",
                normalized_text,
                re.IGNORECASE,
            )
            loose_arguments = arguments_from_parameters(normalized_text)
            if loose_open and loose_arguments:
                calls.append(
                    {
                        "index": 0,
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": loose_open.group("name").strip(),
                            "arguments": json.dumps(
                                loose_arguments,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                )
        return calls

    @staticmethod
    def _sanitize_deepseek_tool_arguments(
        value: Any,
        schema: dict[str, Any] | None,
    ) -> Any:
        """Coerce/remove invalid optional values DeepSeek invents for tools."""
        invalid = object()

        def clean(current: Any, current_schema: Any) -> Any:
            if not isinstance(current_schema, dict):
                return current
            expected = current_schema.get("type")
            expected_types = expected if isinstance(expected, list) else [expected]

            if "object" in expected_types or isinstance(current_schema.get("properties"), dict):
                if not isinstance(current, dict):
                    return invalid
                properties = current_schema.get("properties")
                properties = properties if isinstance(properties, dict) else {}
                required = set(current_schema.get("required") or [])
                additional = current_schema.get("additionalProperties", True)
                result: dict[str, Any] = {}
                for key, item in current.items():
                    child_schema = properties.get(key)
                    if child_schema is None:
                        if additional is not False:
                            result[key] = item
                        continue
                    cleaned = clean(item, child_schema)
                    if cleaned is invalid:
                        if key in required:
                            result[key] = item
                        continue
                    result[key] = cleaned
                return result

            if "array" in expected_types:
                if not isinstance(current, list):
                    return invalid
                item_schema = current_schema.get("items")
                result = []
                for item in current:
                    cleaned = clean(item, item_schema)
                    if cleaned is not invalid:
                        result.append(cleaned)
                return result

            if "boolean" in expected_types:
                if isinstance(current, bool):
                    return current
                if isinstance(current, str) and current.lower() in {"true", "false"}:
                    return current.lower() == "true"
                return invalid

            if "integer" in expected_types:
                if isinstance(current, int) and not isinstance(current, bool):
                    return current
                if isinstance(current, float) and current.is_integer():
                    return int(current)
                if isinstance(current, str) and re.fullmatch(r"-?\d+", current.strip()):
                    return int(current)
                return invalid

            if "number" in expected_types:
                if isinstance(current, (int, float)) and not isinstance(current, bool):
                    return current
                if isinstance(current, str):
                    try:
                        return float(current)
                    except ValueError:
                        return invalid
                return invalid

            if "string" in expected_types and not isinstance(current, str):
                return invalid
            if expected_types == ["null"] and current is not None:
                return invalid
            return current

        cleaned = clean(value, schema)
        return value if cleaned is invalid else cleaned

    @staticmethod
    def _deepseek_tool_schemas(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
        schemas: dict[str, dict[str, Any]] = {}
        for tool in body.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            definition = function if isinstance(function, dict) else tool
            name = definition.get("name")
            schema = definition.get("parameters") or definition.get("input_schema")
            if isinstance(name, str) and isinstance(schema, dict):
                schemas[name] = schema
        return schemas

    @classmethod
    def _sanitize_deepseek_delta_tool_calls(
        cls,
        delta: dict[str, Any],
        schemas: dict[str, dict[str, Any]],
        project_hint: str | None = None,
    ) -> None:
        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            arguments = function.get("arguments")
            if (
                isinstance(name, str)
                and name in {"codegraph_explore", "mcp__codegraph", "mcp__codegraph__"}
                and name not in schemas
                and isinstance(arguments, str)
            ):
                try:
                    codegraph_arguments = json.loads(arguments)
                except (TypeError, ValueError):
                    codegraph_arguments = None
                if isinstance(codegraph_arguments, dict):
                    query = codegraph_arguments.get("query") or codegraph_arguments.get("input")
                    if isinstance(query, str) and query.strip():
                        workdir = codegraph_arguments.get("projectPath") or project_hint
                        exec_arguments: dict[str, Any] = {
                            "cmd": f"codegraph explore {shlex.quote(query.strip())}",
                        }
                        if isinstance(workdir, str) and workdir.strip():
                            exec_arguments["workdir"] = workdir.strip()
                        function["name"] = "exec_command"
                        function["arguments"] = json.dumps(
                            exec_arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        name = "exec_command"
                        arguments = function["arguments"]
            schema = schemas.get(name) if isinstance(name, str) else None
            if not schema or not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except (TypeError, ValueError):
                continue
            cleaned = cls._sanitize_deepseek_tool_arguments(parsed, schema)
            function["arguments"] = json.dumps(
                cleaned,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    async def _deepseek_passthrough(
        self,
        request: web.Request,
        body: dict[str, Any],
        path: str = "/v1/responses",
    ) -> web.StreamResponse:
        """Run the heavy DeepSeek compatibility service only for this request."""
        api_key = _get_deepseek_api_key()
        if not api_key:
            raise web.HTTPServiceUnavailable(
                text=(
                    "DeepSeek API key not found; explicit DeepSeek requests "
                    "will not switch models"
                )
            )

        try:
            await self._acquire_deepseek_service()
        except Exception as exc:
            raise web.HTTPServiceUnavailable(
                text=(
                    f"DeepSeek startup failed: {exc}; explicit DeepSeek requests "
                    "will not switch models"
                )
            )

        session = ClientSession(timeout=self.timeout)
        try:
            return await self._deepseek_passthrough_running(
                request,
                body,
                session,
                path=path,
            )
        finally:
            await session.close()
            await self._release_deepseek_service()

    async def _deepseek_passthrough_running(
        self,
        request: web.Request,
        body: dict[str, Any],
        session: ClientSession,
        path: str = "/v1/responses",
    ) -> web.StreamResponse:
        """Forward a Responses request to DeepSeek Web API.

        deepseek-web-api emulates function calling by prompting DeepSeek with
        the available tool definitions, parsing its XML/JSON protocol, and
        returning standard OpenAI ``tool_calls`` deltas. Tools must therefore
        be preserved here rather than stripped.
        """
        api_key = _get_deepseek_api_key()
        if not api_key:
            raise web.HTTPServiceUnavailable(
                text=(
                    "DeepSeek API key not found; explicit DeepSeek requests "
                    "will not switch models"
                )
            )

        # DeepSeek protocol conversion is centralized in the vendored
        # @codeproxy/core adapter exposed by deepseek-web-api's /v1/responses
        # route. Keep the older in-process Chat->Responses implementation below
        # temporarily as unreachable rollback context while the migration is
        # exercised by the full regression suite.
        return await self._deepseek_codeproxy_responses(
            request,
            body,
            session,
            api_key=api_key,
        )

        use_pro = str(body.get("model") or "").lower() == DEEPSEEK_MODEL_PRO
        ds_model = DEEPSEEK_MODEL_PRO if use_pro else DEEPSEEK_MODEL_STANDARD
        source_tag = "[deepseek-pro]" if use_pro else "[deepseek]"

        tools = body.get("tools")
        if tools:
            print(
                f"[shim] DeepSeek: forwarding {len(tools)} tools through compatibility parser",
                flush=True,
            )

        # Convert to chat completions format
        forwarded_body = _sanitize_deepseek_body(body)
        project_hint = _deepseek_codegraph_project_hint(body)
        chat_body = responses_to_chat(forwarded_body, ds_model)
        chat_body["stream"] = body.get("stream", True)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        del path  # Reserved for parity with other passthrough handlers.
        url = f"{DEEPSEEK_API_BASE}/v1/chat/completions"
        t0 = time.time()

        import asyncio as _asyncio
        try:
            upstream = await _asyncio.wait_for(
                session.post(url, json=chat_body, headers=headers),
                timeout=120,
            )
        except _asyncio.TimeoutError:
            raise web.HTTPGatewayTimeout(
                text=(
                    "DeepSeek timed out after 120 seconds; explicit DeepSeek requests "
                    "will not switch models"
                )
            )
        except Exception as e:
            raise web.HTTPBadGateway(
                text=(
                    f"DeepSeek connection error: {e}; explicit DeepSeek requests "
                    "will not switch models"
                )
            )

        t1 = time.time()
        print(f"[shim] DeepSeek status={upstream.status} elapsed={t1-t0:.2f}s model={ds_model}", flush=True)

        if upstream.status >= 400:
            err_text = await upstream.text()
            raise web.HTTPBadGateway(
                text=(
                    f"DeepSeek upstream error ({upstream.status}): {err_text[:500]}; "
                    "explicit DeepSeek requests will not switch models"
                )
            )

        # Handle streaming response
        if chat_body.get("stream"):
            lines_iter = _sse_lines(upstream)
            try:
                first_line = await anext(lines_iter)
            except StopAsyncIteration:
                first_line = "[DONE]"

            early_failure = _deepseek_stream_failure(first_line)
            if early_failure:
                upstream.release()
                raise web.HTTPBadGateway(
                    text=(
                        f"DeepSeek ended before usable output: {early_failure}; "
                        "explicit DeepSeek requests will not switch models"
                    )
                )

            response = _sse_response()
            await response.prepare(request)

            model_name = "deepseek-pro" if use_pro else "deepseek"
            tool_types = _build_tool_types(body)
            tool_schemas = self._deepseek_tool_schemas(body)
            state = ResponsesStreamState(model_name, tool_types)

            chunk_count = 0
            source_injected = False
            recovered_tool_index = 10000
            try:
                await state.start(response)
                line = first_line
                while True:
                    if line == "[DONE]":
                        print(f"[shim] DeepSeek stream done, chunks={chunk_count} content_len={len(state.message_text)}", flush=True)
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        chunk = {}
                    chunk_count += 1
                    if isinstance(chunk.get("error"), dict):
                        error_message = str(
                            chunk["error"].get("message")
                            or "DeepSeek Web returned an empty error"
                        )
                        print(
                            f"[shim] DeepSeek upstream stream error: {error_message}",
                            flush=True,
                        )
                        await state.write_chat_delta(
                            response,
                            {
                                "choices": [
                                    {
                                        "delta": {
                                            "content": (
                                                "[deepseek] DeepSeek Web failed: "
                                                f"{error_message}. Please retry this turn."
                                            )
                                        },
                                        "finish_reason": "stop",
                                    }
                                ]
                            },
                        )
                        break

                    # Extract delta content
                    choices = chunk.get("choices") or []
                    delta = choices[0].get("delta") or {} if choices else {}
                    content = delta.get("content") or ""

                    # Never expose DeepSeek's malformed XML tool protocol to
                    # Codex. The compatibility service normally converts it
                    # into standard tool_calls; this is a final safety net for
                    # incomplete or previously unseen tag variants.
                    if content and self._is_malformed_tool_call(content):
                        existing_calls = delta.get("tool_calls")
                        recovered_calls = (
                            [] if isinstance(existing_calls, list) and existing_calls
                            else self._recover_deepseek_tool_calls(content)
                        )
                        print(
                            f"[shim] DeepSeek: suppressed malformed tool XML ({len(content)} chars), "
                            f"recovered_calls={len(recovered_calls)}",
                            flush=True,
                        )
                        delta["content"] = ""
                        if recovered_calls:
                            for recovered_call in recovered_calls:
                                recovered_call["index"] = recovered_tool_index
                                recovered_tool_index += 1
                            delta["tool_calls"] = recovered_calls
                        content = ""

                    if delta.get("tool_calls"):
                        self._sanitize_deepseek_delta_tool_calls(
                            delta,
                            tool_schemas,
                            project_hint,
                        )

                    # Inject source tag before first content.
                    if content and not source_injected:
                        source_injected = True
                        delta["content"] = f"{source_tag} {content}"

                    if choices:
                        await state.write_chat_delta(response, chunk)
                    try:
                        line = await anext(lines_iter)
                    except StopAsyncIteration:
                        break

                await state.finish(response)
            except Exception as e:
                print(f"[shim] DeepSeek stream error: {e}", flush=True)
            finally:
                upstream.release()

            return response

        # Handle non-streaming response
        payload = await upstream.json(content_type=None)
        choices = payload.get("choices") or []
        content = ""
        upstream_tool_calls: list[dict[str, Any]] = []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            if isinstance(message.get("tool_calls"), list):
                upstream_tool_calls = message["tool_calls"]
        if content and not upstream_tool_calls and self._is_malformed_tool_call(content):
            recovered_calls = self._recover_deepseek_tool_calls(content)
            if recovered_calls:
                print(
                    f"[shim] Recovered {len(recovered_calls)} malformed DeepSeek "
                    "tool call(s) from non-stream response",
                    flush=True,
                )
                upstream_tool_calls = recovered_calls
                content = ""

        # Build Responses API format response
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        output: list[dict[str, Any]] = []
        tool_schemas = self._deepseek_tool_schemas(body)
        for index, call in enumerate(upstream_tool_calls):
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not name or not isinstance(arguments, str):
                continue
            normalized = {"tool_calls": [call]}
            self._sanitize_deepseek_delta_tool_calls(
                normalized,
                tool_schemas,
                project_hint,
            )
            sanitized_call = normalized["tool_calls"][0]
            sanitized_function = sanitized_call.get("function") or function
            call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}")
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "status": "completed",
                    "call_id": call_id,
                    "name": sanitized_function["name"],
                    "arguments": sanitized_function["arguments"],
                }
            )
        if content or not output:
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"{source_tag} {content}",
                        }
                    ],
                }
            )
        result = {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": "deepseek-pro" if use_pro else "deepseek",
            "output": output,
            "usage": payload.get("usage", {}),
        }
        return web.json_response(result)

    async def _deepseek_codeproxy_responses(
        self,
        request: web.Request,
        body: dict[str, Any],
        session: ClientSession,
        *,
        api_key: str,
    ) -> web.StreamResponse:
        """Proxy Responses traffic through the vendored codeproxy adapter."""
        use_pro = str(body.get("model") or "").lower() == DEEPSEEK_MODEL_PRO
        ds_model = DEEPSEEK_MODEL_PRO if use_pro else DEEPSEEK_MODEL_STANDARD
        forwarded_body = _sanitize_deepseek_body(body)
        forwarded_body["model"] = ds_model
        forwarded_body["stream"] = body.get("stream", True)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{DEEPSEEK_API_BASE}/v1/responses"
        t0 = time.time()

        import asyncio as _asyncio

        try:
            upstream = await _asyncio.wait_for(
                session.post(url, json=forwarded_body, headers=headers),
                timeout=120,
            )
        except _asyncio.TimeoutError:
            raise web.HTTPGatewayTimeout(
                text=(
                    "DeepSeek timed out after 120 seconds; explicit DeepSeek "
                    "requests will not switch models"
                )
            )
        except Exception as exc:
            raise web.HTTPBadGateway(
                text=(
                    f"DeepSeek connection error: {exc}; explicit DeepSeek "
                    "requests will not switch models"
                )
            )

        print(
            f"[shim] DeepSeek codeproxy status={upstream.status} "
            f"elapsed={time.time() - t0:.2f}s model={ds_model}",
            flush=True,
        )
        if upstream.status >= 400:
            err_text = await upstream.text()
            raise web.HTTPBadGateway(
                text=f"DeepSeek Web failed ({upstream.status}): {err_text}"
            )

        if forwarded_body["stream"] is not True:
            try:
                payload = await upstream.json()
            except Exception:
                payload = json.loads(await upstream.text())
            return web.json_response(payload)

        downstream = web.StreamResponse(
            status=upstream.status,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await downstream.prepare(request)
        try:
            async for chunk in upstream.content.iter_any():
                await downstream.write(chunk)
        except (ConnectionResetError, _asyncio.CancelledError):
            upstream.close()
            raise
        finally:
            if not upstream.closed:
                upstream.close()
        await downstream.write_eof()
        return downstream

    async def _claude_gateway_fallback(
        self,
        request: web.Request,
        body: dict[str, Any],
        response_model_override: str | None = None,
        prepared_response: web.StreamResponse | None = None,
    ) -> web.StreamResponse | None:
        """Fallback to Claude via kiro-gateway. Returns None if it fails."""
        claude_url = os.environ.get(
            "CLAUDE_GATEWAY_URL",
            DEFAULT_CLAUDE_GATEWAY_URL,
        ).strip()
        claude_key = os.environ.get(
            "CLAUDE_GATEWAY_API_KEY",
            os.environ.get("CODEX_SHIM_FALLBACK_KEY", "my-super-secret-password-123"),
        )
        claude_model = os.environ.get(
            "CLAUDE_GATEWAY_MODEL",
            DEFAULT_CLAUDE_GATEWAY_MODEL,
        ).strip()

        chat_body = responses_to_chat(body, claude_model)
        claude_tool_count = _add_claude_execution_guidance(chat_body, body)
        if claude_tool_count >= CLAUDE_LOOP_BREAK_TOOL_THRESHOLD:
            print(
                f"[shim] Claude loop breaker injected after {claude_tool_count} tool calls",
                flush=True,
            )
        chat_body["stream"] = True
        if not chat_body.get("tools") and "parallel_tool_calls" in chat_body:
            del chat_body["parallel_tool_calls"]

        headers = {
            "Authorization": f"Bearer {claude_key}",
            "Content-Type": "application/json",
        }

        session = await self._get_session()
        response = prepared_response or _sse_response()
        if prepared_response is None and request is not None:
            await response.prepare(request)
            await _safe_write(response, b": codex-shim connected\n\n")
        t0 = time.time()
        try:
            if request is None:
                import asyncio

                upstream = await asyncio.wait_for(
                    session.post(claude_url, json=chat_body, headers=headers),
                    timeout=90,
                )
            else:
                upstream = await _await_with_sse_heartbeats(
                    session.post(claude_url, json=chat_body, headers=headers),
                    response,
                    timeout=90,
                )
        except Exception as e:
            print(f"[shim] Claude gateway failed: {e}", flush=True)
            return None

        t1 = time.time()
        print(
            f"[shim] CLAUDE gateway status={upstream.status} "
            f"elapsed={t1-t0:.2f}s model={claude_model}",
            flush=True,
        )

        if upstream.status >= 400:
            err_text = await upstream.text()
            print(f"[shim] Claude gateway error: {err_text[:300]}", flush=True)
            return None

        model_name = response_model_override or DEFAULT_CLAUDE_GATEWAY_MODEL
        tool_types = _build_tool_types(body)
        state = ResponsesStreamState(model_name, tool_types)

        chunk_count = 0
        source_injected = False
        pending_content = ""
        try:
            await state.start(response)
            async for line in _sse_lines(upstream):
                if line == "[DONE]":
                    print(f"[shim] CLAUDE stream done, chunks={chunk_count} content_len={len(state.message_text)}", flush=True)
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if chunk_count == 0:
                    print(f"[shim] CLAUDE first delta keys: {list(delta.keys())} finish_reason={choices[0].get('finish_reason')}", flush=True)
                chunk_count += 1
                content = delta.get("content")
                if isinstance(content, str) and not source_injected:
                    pending_content += content
                    delta = dict(delta)
                    delta.pop("content", None)
                    candidate = pending_content.strip().lower()
                    if candidate and not "(empty placeholder)".startswith(candidate):
                        source_injected = True
                        delta["content"] = "[Claude] " + pending_content
                        pending_content = ""
                    chunk = dict(chunk)
                    chunk["choices"] = [dict(choices[0], delta=delta)]
                await state.write_chat_delta(response, chunk)
            if pending_content:
                if pending_content.strip().lower() == "(empty placeholder)":
                    if not state.tool_calls:
                        await state.write_chat_delta(
                            response,
                            {
                                "choices": [
                                    {
                                        "delta": {
                                            "content": "[Claude] Claude 未生成有效回复，请重试本轮。"
                                        }
                                    }
                                ]
                            },
                        )
                else:
                    await state.write_chat_delta(
                        response,
                        {
                            "choices": [
                                {"delta": {"content": "[Claude] " + pending_content}}
                            ]
                        },
                    )
            await state.finish(response)
        except ClientDisconnected:
            pass
        finally:
            upstream.release()

        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    async def _openai_api_fallback(
        self,
        request: web.Request,
        body: dict[str, Any],
        response_model_override: str | None = None,
        prepared_response: web.StreamResponse | None = None,
    ) -> web.StreamResponse:
        """Fallback to OpenAI official API when ChatGPT passthrough fails."""
        import os
        OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
        # Use the model from the original Codex request (what user selected in Codex UI)
        OPENAI_MODEL = body.get("model") or os.environ.get("CODEX_SHIM_OPENAI_FALLBACK_MODEL", "gpt-5.6-sol")

        if not OPENAI_KEY:
            if prepared_response is not None:
                await _write_sse(
                    prepared_response,
                    {
                        "type": "error",
                        "error": {
                            "type": "server_error",
                            "message": "ChatGPT and Claude failed; no OPENAI_API_KEY is configured",
                        },
                    },
                )
                try:
                    await prepared_response.write_eof()
                except Exception:
                    pass
                return prepared_response
            raise web.HTTPServiceUnavailable(text="ChatGPT failed and no OPENAI_API_KEY set for fallback")

        # Convert responses-format to chat completions format
        chat_body = responses_to_chat(body, OPENAI_MODEL)
        chat_body["stream"] = True
        # Remove parallel_tool_calls when no tools are present (OpenAI rejects it)
        if not chat_body.get("tools") and "parallel_tool_calls" in chat_body:
            del chat_body["parallel_tool_calls"]

        headers = {
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": "application/json",
        }

        url = "https://api.openai.com/v1/chat/completions"
        session = await self._get_session()
        t0 = time.time()

        try:
            upstream = await session.post(url, json=chat_body, headers=headers)
        except Exception as e:
            print(f"[shim] OpenAI API fallback failed: {e}", flush=True)
            raise web.HTTPServiceUnavailable(text=f"Both ChatGPT and OpenAI API failed: {e}")

        t1 = time.time()
        print(f"[shim] FALLBACK OpenAI API status={upstream.status} elapsed={t1-t0:.2f}s model={OPENAI_MODEL}", flush=True)

        if upstream.status >= 400:
            err_text = await upstream.text()
            print(f"[shim] OpenAI API error: {err_text[:300]}", flush=True)
            raise web.HTTPServiceUnavailable(text=f"OpenAI API returned {upstream.status}: {err_text[:200]}")

        # Stream SSE back using ResponsesStreamState (handles text + tool_calls)
        response = prepared_response or _sse_response()
        if prepared_response is None:
            await response.prepare(request)

        model_name = response_model_override or "gpt-5.6-sol"
        tool_types = _build_tool_types(body)
        state = ResponsesStreamState(model_name, tool_types)

        chunk_count = 0
        source_injected = False
        try:
            await state.start(response)
            async for line in _sse_lines(upstream):
                if line == "[DONE]":
                    print(f"[shim] FALLBACK stream done, chunks={chunk_count} content_len={len(state.message_text)}", flush=True)
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[shim] FALLBACK bad json: {line[:100]}", flush=True)
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    if chunk_count == 0:
                        print(f"[shim] FALLBACK no choices in chunk: {line[:200]}", flush=True)
                    continue
                delta = choices[0].get("delta", {})
                if chunk_count == 0:
                    print(f"[shim] FALLBACK first delta keys: {list(delta.keys())} finish_reason={choices[0].get('finish_reason')}", flush=True)
                chunk_count += 1
                # Inject source indicator before first text content
                if not source_injected and delta.get("content"):
                    source_injected = True
                    await state.write_chat_delta(
                        response,
                        {"choices": [{"delta": {"content": "[OpenAI API] "}}]},
                    )
                await state.write_chat_delta(response, chunk)
            await state.finish(response)
        except ClientDisconnected:
            pass
        finally:
            upstream.release()

        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    async def _chatgpt_compact_passthrough(
        self,
        request: web.Request,
        body: dict[str, Any],
        upstream_model: str | None = None,
    ) -> web.StreamResponse:
        auth_path = DEFAULT_CODEX_AUTH.expanduser()
        try:
            auth = json.loads(auth_path.read_text())
        except FileNotFoundError:
            raise web.HTTPUnauthorized(text="~/.codex/auth.json not found")
        tokens = auth.get("tokens") or {}
        access_token = tokens.get("access_token")
        account_id = tokens.get("account_id") or ""
        if not access_token:
            raise web.HTTPUnauthorized(text="auth.json has no access_token")
        forwarded = _sanitize_chatgpt_passthrough_body(body)
        requested_model = str(forwarded.get("model") or CHATGPT_MODEL_SLUG)
        original_model = (
            upstream_model
            or (
                chatgpt_upstream_model(requested_model)
                if is_chatgpt_passthrough_slug(requested_model)
                else CHATGPT_MODEL_SLUG
            )
        )
        forwarded["model"] = original_model
        forwarded.pop("stream", None)
        forwarded.pop("store", None)
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "OpenAI-Beta": "responses=2026-02-06",
            "originator": "codex_cli_rs",
            "chatgpt-account-id": account_id,
            "session_id": request.headers.get("session_id", ""),
        }
        url = "https://chatgpt.com/backend-api/codex/responses/compact"
        import asyncio as _asyncio
        from aiohttp import ClientConnectorError, ServerDisconnectedError, ClientOSError
        session = await self._get_session()

        # Honor an explicitly selected ChatGPT model first. The configured
        # default remains a capacity fallback for compatible requests.
        # Fallback models will be added dynamically if "at capacity" error is detected
        models_to_try = [original_model]
        if CHATGPT_MODEL_SLUG != original_model:
            models_to_try.append(CHATGPT_MODEL_SLUG)
        capacity_fallbacks_added = False

        model_idx = 0
        while model_idx < len(models_to_try):
            try_model = models_to_try[model_idx]
            forwarded["model"] = try_model
            t0 = time.time()
            max_retries = 5
            model_failed = False
            upstream = None
            for attempt in range(max_retries):
                try:
                    upstream = await session.post(url, json=forwarded, headers=headers)
                except (ClientConnectorError, ServerDisconnectedError, ClientOSError, ConnectionResetError) as e:
                    if attempt < max_retries - 1:
                        print(f"[shim] compact connection error, resetting session (attempt {attempt+1}): {e}", flush=True)
                        session = await self._reset_session()
                        await _asyncio.sleep(1)
                        t0 = time.time()
                        continue
                    model_failed = True
                    break
                t1 = time.time()
                print(f"[shim] POST /codex/responses/compact status={upstream.status} elapsed={t1-t0:.2f}s attempt={attempt+1} model={try_model}", flush=True)
                if upstream.status in (429, 503, 529):
                    body_text = await upstream.text()
                    if "usage_limit_reached" in body_text:
                        print(f"\n{'='*60}", flush=True)
                        print(f"[shim] ⚠️  ChatGPT USAGE LIMIT REACHED for {try_model} (compact)", flush=True)
                        print(f"{'='*60}\n", flush=True)
                        model_failed = True
                        break
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt + 1
                        print(f"[shim] retrying in {wait}s: {body_text[:200]}", flush=True)
                        await _asyncio.sleep(wait)
                        t0 = time.time()
                        continue
                    model_failed = True
                    break
                if upstream.status >= 400:
                    err_text = await upstream.text()
                    # Check for "at capacity" error and add limited fallback models (up to gpt-5.3-codex)
                    if "at capacity" in err_text.lower() and not capacity_fallbacks_added:
                        capacity_fallbacks_added = True
                        for fb_model in ("gpt-5.5", "gpt-5.4", "gpt-5.3-codex"):
                            if fb_model not in models_to_try:
                                models_to_try.append(fb_model)
                        print(f"[shim] compact: Model at capacity, added fallbacks: {models_to_try}", flush=True)
                    model_failed = True
                    break
                break

            if not model_failed:
                break
            model_idx += 1
            if model_idx < len(models_to_try):
                print(f"[shim] compact: trying next model: {models_to_try[model_idx]}", flush=True)
                continue
            # All models exhausted — return last error
            if upstream:
                return await _error_response(upstream)
            raise web.HTTPBadGateway(text="All ChatGPT models failed (compact)")

        payload = await upstream.json(content_type=None)
        _rewrite_response_model(payload, original_model or None)
        return web.json_response(payload)

    async def _cursor_passthrough(
        self,
        request: web.Request,
        body: dict[str, Any],
        response_model_override: str | None = None,
        upstream_model: str | None = None,
        force_non_stream: bool = False,
    ) -> web.StreamResponse:
        """Route Composer through cursor-agent using Cursor subscription login."""
        if not cursor_passthrough_available():
            raise web.HTTPUnauthorized(
                text="Cursor subscription auth unavailable. Run `cursor-agent login`, then retry."
            )
        slug = response_model_override or CURSOR_MODEL_SLUG
        upstream = upstream_model or cursor_upstream_model(slug)
        prompt = build_cursor_prompt(body)
        stream = bool(body.get("stream")) and not force_non_stream

        if not stream:
            text = ""
            usage: dict[str, Any] | None = None
            async for event in iter_cursor_agent_events(prompt, upstream):
                if event["type"] == "completed":
                    text = str(event.get("text") or text)
                elif event["type"] == "usage":
                    usage = event.get("usage") if isinstance(event.get("usage"), dict) else None
                elif event["type"] == "error":
                    raise web.HTTPBadGateway(text=str(event.get("message") or "cursor-agent failed"))
            payload: dict[str, Any] = {
                "id": f"resp_{int(time.time() * 1000)}",
                "object": "response",
                "model": slug,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_0",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
            }
            normalized_usage = normalize_responses_usage(usage)
            if normalized_usage:
                payload["usage"] = normalized_usage
            return web.json_response(payload)

        response = _sse_response()
        await response.prepare(request)
        tool_types = _build_tool_types(body)
        state = ResponsesStreamState(slug, tool_types)
        try:
            await state.start(response)
            async for event in iter_cursor_agent_events(prompt, upstream):
                if event["type"] == "text_delta":
                    await state.write_chat_delta(
                        response,
                        {"choices": [{"delta": {"content": event["delta"]}}]},
                    )
                elif event["type"] == "usage":
                    normalized_usage = normalize_responses_usage(event.get("usage"))
                    if normalized_usage:
                        state.usage = normalized_usage
                elif event["type"] == "error":
                    message = str(event.get("message") or "cursor-agent failed")
                    await state.write_chat_delta(
                        response,
                        {"choices": [{"delta": {"content": message}}]},
                    )
                    break
            await state.finish(response)
        except ClientDisconnected:
            pass
        except Exception as exc:
            print(f"[err] cursor passthrough {slug}: {exc}", flush=True)
            raise web.HTTPBadGateway(text=str(exc)) from exc
        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    # ------------------------------------------------------------------
    # Auto Router
    # ------------------------------------------------------------------
    def _active_router(self):
        """Return the RouterConfig only when enabled and at least one candidate
        backend is usable, so discovery never advertises a dead Auto entry."""
        config = self.settings.load_router()
        if config and router_module.router_is_active(config, available_model_slugs(self.settings.load())):
            return config
        return None

    async def _maybe_apply_auto_router(self, body: dict[str, Any]) -> dict[str, Any]:
        """If the request targets the Auto Router slug, classify the task and
        rewrite ``model`` to the concrete backend that should handle it. Any
        failure leaves the body untouched so the request still routes normally."""
        config = self.settings.load_router()
        if not config or not config.effective_enabled:
            return body
        if str(body.get("model") or "") != config.slug:
            return body
        resolved = await self._resolve_auto_model(config, body)
        if resolved and resolved != config.slug:
            if router_module.router_log_enabled():
                print(f"[router] {config.slug} -> {resolved}", flush=True)
            new_body = dict(body)
            new_body["model"] = resolved
            return new_body
        return body

    async def _resolve_auto_model(self, config, body: dict[str, Any]) -> str | None:
        models = self.settings.load()
        candidates = router_module.filter_available(config, available_model_slugs(models))
        if not candidates:
            return None
        classify = None
        if config.classifier:
            classifier_model = self.settings.by_slug_or_model(config.classifier)
            if (
                classifier_model is not None
                and byok_model_has_credentials(classifier_model)
                and (classifier_model.is_openai_chat or classifier_model.is_anthropic)
            ):
                classify = self._make_classifier(classifier_model, config)
        log = (lambda message: print(message, flush=True)) if router_module.router_log_enabled() else None
        resolved, _info = await router_module.resolve_auto(config, candidates, body, classify, log=log)
        return resolved or router_module.fallback_slug(
            config, candidates, has_image_task=router_module.has_images(body)
        )

    def _make_classifier(self, model: ShimModel, config):
        timeout = ClientTimeout(total=config.timeout + 5, sock_connect=config.timeout, sock_read=config.timeout)

        async def classify(system_prompt: str, user_content: str) -> str:
            async with ClientSession(timeout=timeout) as session:
                if model.is_anthropic:
                    url = _join_url(model.base_url, "/messages")
                    payload = {
                        "model": model.model,
                        "max_tokens": config.max_tokens,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_content}],
                    }
                    upstream = await session.post(url, json=payload, headers=_anthropic_headers(model))
                    upstream.raise_for_status()
                    data = await upstream.json(content_type=None)
                    return _anthropic_text(data)
                url = _join_url(model.base_url, "/chat/completions")
                payload = {
                    "model": model.model,
                    "stream": False,
                    "temperature": 0,
                    "max_tokens": config.max_tokens,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                }
                upstream = await session.post(url, json=payload, headers=_openai_headers(model))
                upstream.raise_for_status()
                data = await upstream.json(content_type=None)
                message = (data.get("choices") or [{}])[0].get("message") or {}
                return str(message.get("content") or "")

        return classify

    def _route(self, body: dict[str, Any]) -> ShimModel:
        requested = str(body.get("model") or "")
        route = self.settings.by_slug_or_model(requested)
        if route is None:
            raise web.HTTPNotFound(text=f"Unknown model slug/model: {requested}")
        if not byok_model_has_credentials(route):
            raise web.HTTPUnauthorized(text=_missing_api_key_message(route))
        return route

    async def _post_openai_chat(
        self, request: web.Request, route: ShimModel, body: dict[str, Any], as_responses: bool
    ) -> web.StreamResponse:
        url = _join_url(route.base_url, "/chat/completions")
        headers = _openai_headers(route)
        _dump_debug_request(route.slug, url, body)
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=body, headers=headers)
            if upstream.status >= 400:
                return await _error_response(upstream, slug=route.slug)
            if body.get("stream"):
                return await self._stream_openai_chat(request, upstream, route, as_responses, body)
            payload = await upstream.json(content_type=None)
        if as_responses:
            tool_types = _build_tool_types(body)
            payload = chat_completion_to_response(payload, route.slug, tool_types)
            intercepted = _maybe_intercept_web_search(payload)
            return web.json_response(intercepted or payload)
        return web.json_response(payload)

    async def _post_openai_chat_as_anthropic(
        self, request: web.Request, route: ShimModel, body: dict[str, Any]
    ) -> web.StreamResponse:
        url = _join_url(route.base_url, "/chat/completions")
        headers = _openai_headers(route)
        _dump_debug_request(route.slug, url, body)
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=body, headers=headers)
            if upstream.status >= 400:
                return await _anthropic_error_response(upstream)
            if body.get("stream"):
                return await self._stream_openai_chat_as_anthropic(request, upstream, route)
            payload = await upstream.json(content_type=None)
        return web.json_response(chat_completion_to_anthropic_message(payload, route.slug))

    async def _post_anthropic(
        self, request: web.Request, route: ShimModel, body: dict[str, Any], as_responses: bool
    ) -> web.StreamResponse:
        url = _join_url(route.base_url, "/messages")
        headers = _anthropic_headers(route)
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=body, headers=headers)
            if upstream.status >= 400:
                return await _error_response(upstream)
            if body.get("stream"):
                return await self._stream_anthropic(request, upstream, route, as_responses, body)
            payload = await upstream.json(content_type=None)
        if as_responses:
            tool_types = _build_tool_types(body)
            payload = anthropic_to_response(payload, route.slug, tool_types)
            intercepted = _maybe_intercept_web_search(payload)
            return web.json_response(intercepted or payload)
        return web.json_response(anthropic_to_chat_response(payload, route.slug))

    async def _post_anthropic_messages(
        self, request: web.Request, route: ShimModel, body: dict[str, Any]
    ) -> web.StreamResponse:
        url = _join_url(route.base_url, "/messages")
        headers = _anthropic_headers(route)
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=body, headers=headers)
            if upstream.status >= 400:
                return await _error_response(upstream, slug=route.slug)
            if body.get("stream"):
                return await self._stream_raw_sse(request, upstream, route.slug)
            payload = await upstream.json(content_type=None)
        if isinstance(payload, dict):
            payload["model"] = route.slug
        return web.json_response(payload)

    async def _stream_openai_chat(
        self, request: web.Request, upstream, route: ShimModel, as_responses: bool, body: dict[str, Any] | None = None
    ) -> web.StreamResponse:
        response = _sse_response()
        await response.prepare(request)
        if as_responses:
            tool_types = _build_tool_types(body) if body else {}
            state = ResponsesStreamState(route.slug, tool_types)
        try:
            if as_responses:
                await state.start(response)
            async for line in _sse_lines(upstream):
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if as_responses:
                    await state.write_chat_delta(response, event)
                else:
                    await _write_sse(response, event)
            if as_responses:
                await state.finish(response)
            else:
                await _safe_write(response, b"data: [DONE]\n\n")
        except ClientDisconnected:
            pass
        finally:
            upstream.release()
        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    async def _stream_openai_chat_as_anthropic(
        self, request: web.Request, upstream, route: ShimModel
    ) -> web.StreamResponse:
        response = _sse_response()
        await response.prepare(request)
        state = AnthropicMessagesStreamState(route.slug)
        try:
            await state.start(response)
            async for line in _sse_lines(upstream):
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                await state.write_chat_delta(response, event)
            await state.finish(response)
        except ClientDisconnected:
            pass
        finally:
            upstream.release()
        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    async def _stream_anthropic(
        self, request: web.Request, upstream, route: ShimModel, as_responses: bool, body: dict[str, Any] | None = None
    ) -> web.StreamResponse:
        response = _sse_response()
        await response.prepare(request)
        if as_responses:
            tool_types = _build_tool_types(body) if body else {}
            state = ResponsesStreamState(route.slug, tool_types)
        try:
            if as_responses:
                await state.start(response)
            async for line in _sse_lines(upstream):
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if as_responses:
                    await state.write_anthropic_delta(response, event)
                else:
                    await _write_sse(response, _anthropic_stream_to_chat_chunk(event, route.slug))
            if as_responses:
                await state.finish(response)
            else:
                await _safe_write(response, b"data: [DONE]\n\n")
        except ClientDisconnected:
            pass
        finally:
            upstream.release()
        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    async def _stream_raw_sse(self, request: web.Request, upstream, model_slug: str | None = None) -> web.StreamResponse:
        response = _sse_response()
        await response.prepare(request)
        try:
            async for line in _sse_lines(upstream):
                if model_slug and line.startswith("{"):
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict) and event.get("type") == "message_start":
                            msg = event.get("message")
                            if isinstance(msg, dict):
                                msg["model"] = model_slug
                        await _write_anthropic_sse(response, event.get("type", "message"), event)
                        continue
                    except json.JSONDecodeError:
                        pass
                await _safe_write(response, f"data: {line}\n\n".encode())
        except ClientDisconnected:
            pass
        finally:
            upstream.release()
        try:
            await response.write_eof()
        except Exception:
            pass
        return response


_DROP_ITEM = object()


def _prime_loopback_platform(request: web.Request) -> str | None:
    header_platform = str(
        request.headers.get("x-codex-shim-platform", "") or ""
    ).strip().lower()
    bearer_identity = request.headers.get("Authorization") == (
        "Bearer local-codex-shim-chatgpt"
    )
    if not header_platform and not bearer_identity:
        return None

    peername = request.transport.get_extra_info("peername") if request.transport else None
    peer_host = peername[0] if isinstance(peername, tuple) and peername else ""
    try:
        is_loopback = ipaddress.ip_address(peer_host).is_loopback
    except ValueError:
        is_loopback = peer_host == "localhost"
    if not is_loopback:
        raise web.HTTPForbidden(
            text="Prime Codex-shim routing identity is loopback-only"
        )
    if bearer_identity:
        return "chatgpt"
    return header_platform


def _check_and_strip_platform_prefix(body: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Use the newest platform prefix found in the replayed user prompt history.

    Returns (modified_body, platform) where platform is one of:
    - "deepseek" for [deepseek]
    - "deepseek-pro" for [deepseek-pro]
    - "chatgpt" for [chatgpt]
    - "claude" for [claude], [claud], or [kiro]
    - None if no prefix found
    """
    import re
    prefix_pattern = re.compile(
        r'^\s*\[(deepseek-pro|deepseek|chatgpt|claude|claud|kiro)\]\s*',
        re.IGNORECASE,
    )

    def normalize_platform(value: str) -> str:
        platform = value.lower()
        return "claude" if platform in {"claud", "kiro"} else platform

    input_data = body.get("input")
    if not input_data:
        return body, None

    # Handle string input
    if isinstance(input_data, str):
        match = prefix_pattern.match(input_data)
        if match:
            platform = normalize_platform(match.group(1))
            new_body = dict(body)
            new_body["input"] = input_data[match.end():].lstrip()
            return new_body, platform
        return body, None

    # Codex resends the conversation on every turn. Search user messages in
    # reverse order so a newer explicit prefix switches the remembered session
    # platform, even when later synthetic/unprefixed user items are appended.
    if isinstance(input_data, list):
        for i in range(len(input_data) - 1, -1, -1):
            item = input_data[i]
            if not isinstance(item, dict):
                continue
            if item.get("role") != "user":
                continue
            content = item.get("content")
            # Handle string content
            if isinstance(content, str):
                match = prefix_pattern.match(content)
                if match:
                    platform = normalize_platform(match.group(1))
                    new_body = dict(body)
                    new_input = list(input_data)
                    new_item = dict(item)
                    new_item["content"] = content[match.end():].lstrip()
                    new_input[i] = new_item
                    new_body["input"] = new_input
                    return new_body, platform
            # Handle list content (e.g., [{"type": "input_text", "text": "..."}])
            if isinstance(content, list):
                for j in range(len(content) - 1, -1, -1):
                    part = content[j]
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text") or part.get("input_text")
                    if not text:
                        continue
                    match = prefix_pattern.match(text)
                    if match:
                        platform = normalize_platform(match.group(1))
                        new_body = dict(body)
                        new_input = list(input_data)
                        new_item = dict(item)
                        new_content = list(content)
                        new_part = dict(part)
                        text_key = "text" if "text" in part else "input_text"
                        new_part[text_key] = text[match.end():].lstrip()
                        new_content[j] = new_part
                        new_item["content"] = new_content
                        new_input[i] = new_item
                        new_body["input"] = new_input
                        return new_body, platform

    return body, None


def _deepseek_available() -> bool:
    """Check if DeepSeek Web API is available."""
    return DEEPSEEK_API_KEY_FILE.exists()


def _get_deepseek_api_key() -> str | None:
    """Read DeepSeek API key from file."""
    try:
        return DEEPSEEK_API_KEY_FILE.read_text().strip()
    except Exception:
        return None


def _deepseek_item_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text") or part.get("input_text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _clip_deepseek_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[older DeepSeek context truncated]...\n"
    remaining = max(0, limit - len(marker))
    head = remaining // 2
    return f"{text[:head]}{marker}{text[-(remaining - head):]}"


def _truncate_deepseek_message(item: dict[str, Any], limit: int) -> None:
    text = _deepseek_item_text(item)
    if not text or len(text) <= limit:
        return
    item["content"] = _clip_deepseek_text(text, limit)


def _strip_deepseek_source_tag(item: dict[str, Any]) -> None:
    """Keep replayed assistant text equal to the vendor's untagged session turn."""
    if item.get("role") != "assistant":
        return
    content = item.get("content")
    prefix = re.compile(r"^\s*\[(?:deepseek-pro|deepseek)\]\s*", re.IGNORECASE)
    if isinstance(content, str):
        item["content"] = prefix.sub("", content, count=1)
        return
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict):
            continue
        for key in ("text", "input_text"):
            if isinstance(part.get(key), str):
                part[key] = prefix.sub("", part[key], count=1)
                return


def _compact_deepseek_value(value: Any) -> Any:
    """Reduce verbose tool documentation without changing its schema."""
    if isinstance(value, list):
        return [_compact_deepseek_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    compacted: dict[str, Any] = {}
    for key, item in value.items():
        if key == "description" and isinstance(item, str) and len(item) > 240:
            compacted[key] = item[:240] + "…"
        else:
            compacted[key] = _compact_deepseek_value(item)
    return compacted


def _deepseek_codegraph_project_hint(body: dict[str, Any]) -> str | None:
    """Infer the intended indexed project from environment context and task text."""
    texts = [
        _deepseek_item_text(item)
        for item in body.get("input") or []
        if isinstance(item, dict)
    ]
    joined = "\n".join(texts)
    cwd_matches = re.findall(r"<cwd>\s*([^<]+?)\s*</cwd>", joined, re.IGNORECASE)
    cwd = cwd_matches[-1].strip() if cwd_matches else ""

    absolute_candidates = re.findall(r"(?<![\w.-])(/[^\s<>'\"`]+)", joined)
    for raw_candidate in reversed(absolute_candidates):
        candidate = raw_candidate.rstrip(".,:;)]}")
        path = Path(candidate)
        if path.name == ".codegraph":
            path = path.parent
        if (path / ".codegraph").is_dir():
            return str(path)
        if path.is_file() and (path.parent / ".codegraph").is_dir():
            return str(path.parent)

    project_names: list[str] = []
    for pattern in (
        r"(?:在|in)\s*([A-Za-z0-9_.-]+)\s*(?:中|里|project|repo|\b)",
        r"\b([A-Za-z0-9_.-]+)/(?:apps|src|service|common|tests?)\b",
    ):
        project_names.extend(re.findall(pattern, joined, re.IGNORECASE))

    if cwd:
        cwd_path = Path(cwd)
        if (cwd_path / ".codegraph").is_dir():
            return str(cwd_path)
        for project_name in reversed(project_names):
            candidate = cwd_path / project_name
            if (candidate / ".codegraph").is_dir():
                return str(candidate)
        indexed_children = [
            marker.parent
            for marker in cwd_path.glob("*/.codegraph")
            if marker.is_dir()
        ]
        if len(indexed_children) == 1:
            return str(indexed_children[0])
    return None


def _expand_deepseek_namespace_tools(
    tools: list[dict[str, Any]],
    project_hint: str | None,
) -> list[dict[str, Any]]:
    """Expose concrete callable tools instead of opaque MCP namespaces."""
    tool_names = {
        str(
            tool.get("name")
            or (
                tool.get("function", {}).get("name")
                if isinstance(tool.get("function"), dict)
                else ""
            )
        )
        for tool in tools
    }
    if "exec_command" not in tool_names:
        return tools

    expanded: list[dict[str, Any]] = []
    has_codegraph_function = "codegraph_explore" in tool_names
    for tool in tools:
        if (
            tool.get("type") == "namespace"
            and str(tool.get("name") or "") in {"mcp__codegraph__", "mcp__codegraph"}
        ):
            if not has_codegraph_function:
                project_note = (
                    f" Default projectPath for this task: {project_hint}."
                    if project_hint
                    else ""
                )
                expanded.append(
                    {
                        "type": "function",
                        "name": "codegraph_explore",
                        "description": (
                            "Explore an indexed codebase before grep/read. Supply a focused "
                            "query naming relevant symbols or the code flow."
                            f"{project_note}"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "projectPath": {
                                    "type": "string",
                                    "description": "Absolute indexed project directory.",
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Symbols or code-flow question to explore.",
                                },
                                "maxFiles": {
                                    "type": "integer",
                                    "description": "Maximum source files to return.",
                                },
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    }
                )
                has_codegraph_function = True
            continue
        expanded.append(tool)
    return expanded


def _hydrate_deepseek_namespace_tools(
    tools: list[dict[str, Any]],
    project_hint: str | None,
) -> list[dict[str, Any]]:
    """Give opaque CodeGraph namespaces concrete children for codeproxy."""
    hydrated: list[dict[str, Any]] = []
    for tool in tools:
        if (
            not isinstance(tool, dict)
            or tool.get("type") != "namespace"
            or str(tool.get("name") or "") not in {"mcp__codegraph__", "mcp__codegraph"}
        ):
            hydrated.append(tool)
            continue

        namespace = dict(tool)
        children = namespace.get("tools")
        if not isinstance(children, list) or not children:
            project_note = (
                f" Default projectPath for this task: {project_hint}."
                if project_hint
                else ""
            )
            namespace["tools"] = [
                {
                    "type": "function",
                    "name": "codegraph_explore",
                    "description": (
                        "Explore an indexed codebase before grep/read. Supply a focused "
                        "query naming relevant symbols or the code flow."
                        f"{project_note}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "projectPath": {
                                "type": "string",
                                "description": "Absolute indexed project directory.",
                                **(
                                    {"default": project_hint}
                                    if project_hint
                                    else {}
                                ),
                            },
                            "query": {
                                "type": "string",
                                "description": "Symbols or code-flow question to explore.",
                            },
                            "maxFiles": {
                                "type": "integer",
                                "description": "Maximum source files to return.",
                            },
                        },
                        "required": ["query", "projectPath"],
                        "additionalProperties": False,
                    },
                }
            ]
        hydrated.append(namespace)
    return hydrated


def _deepseek_tool_is_relevant(tool: dict[str, Any], latest_user_text: str) -> bool:
    """Drop only very large optional namespaces that are clearly unrelated."""
    if tool.get("type") != "namespace":
        return True
    name = str(tool.get("name") or "").lower()
    text = latest_user_text.lower()
    if "pencil" in name:
        return bool(re.search(r"\b(ui|ux|design|figma|canvas)\b|设计|界面|画布|原型|\.pen", text))
    if "sites" in name:
        return bool(
            re.search(
                r"\b(site|website|deploy|domain|analytics|d1|cloudflare)\b|"
                r"网站|部署|域名|流量|数据库",
                text,
            )
        )
    return True


def _is_deepseek_continuation(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if len(normalized) > 100:
        return False
    return bool(
        re.match(
            r"^(继续|接着|再试|重试|再检查|再看看|继续修复|还有问题|不对|"
            r"continue\b|retry\b|try again\b|keep going\b|fix it\b)",
            normalized,
        )
    )


def _sanitize_deepseek_body(body: dict[str, Any]) -> dict[str, Any]:
    """Keep the current task/tool chain small enough for DeepSeek Web."""
    sanitized = json.loads(json.dumps(body))
    existing_instructions = sanitized.get("instructions")
    sanitized["instructions"] = (
        f"{existing_instructions}\n\n{DEEPSEEK_EXECUTION_RULES}"
        if isinstance(existing_instructions, str) and existing_instructions.strip()
        else DEEPSEEK_EXECUTION_RULES
    )
    input_items = sanitized.get("input")
    if not isinstance(input_items, list):
        return sanitized

    user_indices: list[int] = []
    latest_user_text = ""
    for index, item in enumerate(input_items):
        if isinstance(item, dict) and item.get("role") == "user":
            user_indices.append(index)
    latest_user_index = user_indices[-1] if user_indices else None
    current_user_index = latest_user_index
    if latest_user_index is not None:
        latest_user_item = input_items[latest_user_index]
        latest_user_text = _deepseek_item_text(latest_user_item)
    if latest_user_index is not None and _is_deepseek_continuation(latest_user_text):
        continuation_text = latest_user_text
        # A long Codex session can contain several consecutive "继续/重试"
        # messages. Walk back to the nearest concrete user task instead of
        # anchoring recovery to another context-free continuation.
        for previous_user_index in reversed(user_indices[:-1]):
            previous_text = _deepseek_item_text(input_items[previous_user_index])
            if not previous_text.strip():
                continue
            latest_user_index = previous_user_index
            latest_user_text = f"{previous_text}\n{continuation_text}"
            if not _is_deepseek_continuation(previous_text):
                break

    # Keep several preceding user/assistant turns so follow-up questions retain
    # the DSL, code, or decision they refer to. Older tool transcripts are
    # intentionally dropped; only the active task's tool chain is replayed.
    context_start_index = latest_user_index
    tool_history_start_index = latest_user_index
    if latest_user_index is not None:
        anchor_position = max(
            index
            for index, user_index in enumerate(user_indices)
            if user_index <= latest_user_index
        )
        context_position = max(0, anchor_position - 5)
        context_start_index = user_indices[context_position]
        if current_user_index != latest_user_index:
            # A continuation such as "继续修复" depends on the full tool chain
            # since the concrete task it refers to.
            tool_history_start_index = latest_user_index
        else:
            # For a normal follow-up, keep the immediately preceding turn's
            # tool calls/results. They contain the files inspected, edits made,
            # and verification results that the new request refers to.
            current_position = len(user_indices) - 1
            previous_position = max(0, current_position - 1)
            tool_history_start_index = user_indices[previous_position]
        tool_history_offset = tool_history_start_index - context_start_index
        current_turn_offset = current_user_index - context_start_index
        input_items = input_items[context_start_index:]
    elif len(input_items) > 40:
        tool_history_offset = 0
        current_turn_offset = 0
        input_items = input_items[-40:]
    else:
        tool_history_offset = 0
        current_turn_offset = 0

    cleaned_items: list[Any] = []
    tool_output_indices = [
        index
        for index, item in enumerate(input_items)
        if isinstance(item, dict)
        and item.get("type") in {"function_call_output", "custom_tool_call_output"}
    ]
    recent_tool_outputs = set(tool_output_indices[-3:])
    for index, item in enumerate(input_items):
        if not isinstance(item, dict):
            cleaned_items.append(item)
            continue
        if index < tool_history_offset and item.get("type") in {
            "function_call",
            "function_call_output",
            "custom_tool_call",
            "custom_tool_call_output",
        }:
            continue
        if item.get("type") == "reasoning":
            continue
        cleaned = dict(item)
        cleaned.pop("encrypted_content", None)
        _strip_deepseek_source_tag(cleaned)
        if cleaned.get("role") == "user":
            _truncate_deepseek_message(
                cleaned,
                16_000 if index >= current_turn_offset else 8_000,
            )
        elif cleaned.get("role") == "assistant":
            _truncate_deepseek_message(cleaned, 6_000)
        if cleaned.get("type") in {"function_call_output", "custom_tool_call_output"}:
            max_length = 6000 if index in recent_tool_outputs else 2000
            _truncate_tool_output(cleaned, max_length)
        cleaned_items.append(cleaned)
    sanitized["input"] = cleaned_items

    tools = sanitized.get("tools")
    if isinstance(tools, list):
        relevant_tools = [
            tool
            for tool in tools
            if not isinstance(tool, dict)
            or _deepseek_tool_is_relevant(tool, latest_user_text)
        ]
        # Keep native namespace tools intact. The vendored @codeproxy/core
        # adapter now owns namespace flattening, upstream-safe name
        # sanitization, and restoration in Responses events.
        project_hint = _deepseek_codegraph_project_hint(body)
        hydrated_tools = _hydrate_deepseek_namespace_tools(
            relevant_tools,
            project_hint,
        )
        sanitized["tools"] = _compact_deepseek_value(hydrated_tools)

    before_size = len(json.dumps(body, ensure_ascii=False))
    after_size = len(json.dumps(sanitized, ensure_ascii=False))
    if after_size < before_size:
        print(
            f"[shim] DeepSeek context reduced {before_size} -> {after_size} chars; "
            f"input {len(body.get('input') or [])} -> {len(cleaned_items)}, "
            f"tools {len(body.get('tools') or [])} -> {len(sanitized.get('tools') or [])}",
            flush=True,
        )
    return sanitized


def _optimize_input_context(body: dict[str, Any]) -> dict[str, Any]:
    """Optimize input context to reduce token count without losing conversation structure.

    Strategies:
    1. Strip reasoning content from older messages (keep only recent 6 reasoning blocks)
    2. Truncate overly long tool outputs (keep first/last 1500 chars for old ones)
    """
    import os

    input_items = body.get("input")
    if not isinstance(input_items, list):
        return body

    item_count = len(input_items)

    # Apply env-based reasoning effort override if set (no auto-degradation)
    effort_env = os.environ.get("CODEX_SHIM_REASONING_EFFORT", "").strip()
    if effort_env and effort_env in ("low", "medium", "high"):
        reasoning = body.get("reasoning")
        if isinstance(reasoning, dict):
            reasoning["effort"] = effort_env
        else:
            body["reasoning"] = {"effort": effort_env}

    # Find reasoning items and strip old ones (keep last 6)
    reasoning_indices = []
    for i, item in enumerate(input_items):
        if isinstance(item, dict) and item.get("type") == "reasoning":
            reasoning_indices.append(i)

    if len(reasoning_indices) > 6:
        for idx in reasoning_indices[:-6]:
            item = input_items[idx]
            # Responses API reasoning items only accept an empty content array.
            # Adding an input_text placeholder makes the entire request invalid.
            item["content"] = []
            if "summary" in item:
                item["summary"] = []

    # Truncate long tool outputs (older than last 10 items)
    MAX_OUTPUT_LEN = 3000
    if item_count > 10:
        for item in input_items[:-10]:
            if isinstance(item, dict):
                _truncate_tool_output(item, MAX_OUTPUT_LEN)
                # Also handle nested content arrays
                content = item.get("content")
                if isinstance(content, list):
                    for sub in content:
                        if isinstance(sub, dict):
                            _truncate_tool_output(sub, MAX_OUTPUT_LEN)

    return body


def _truncate_tool_output(item: dict[str, Any], max_len: int) -> None:
    """Truncate tool output text if too long."""
    if item.get("type") in ("function_call_output", "custom_tool_call_output"):
        output = item.get("output") or item.get("text") or ""
        if isinstance(output, str) and len(output) > max_len:
            half = max_len // 2
            truncated = output[:half] + f"\n\n... [{len(output) - max_len} chars truncated] ...\n\n" + output[-half:]
            if "output" in item:
                item["output"] = truncated
            elif "text" in item:
                item["text"] = truncated


def _sanitize_chatgpt_passthrough_body(body: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_chatgpt_passthrough_value(body)
    if not isinstance(sanitized, dict):
        sanitized = {}
    sanitized["store"] = False
    sanitized["stream"] = True
    # Optimize context before sending
    sanitized = _optimize_input_context(sanitized)
    return sanitized


def _sanitize_chatgpt_passthrough_value(value: Any) -> Any:
    if isinstance(value, list):
        output = []
        for item in value:
            sanitized = _sanitize_chatgpt_passthrough_value(item)
            if sanitized is not _DROP_ITEM:
                output.append(sanitized)
        return output
    if isinstance(value, dict):
        if value.get("type") == "reasoning" and _has_shim_encrypted_content(value):
            return _DROP_ITEM
        output = {}
        for key, item in value.items():
            if key == "encrypted_content" and isinstance(item, str) and item.startswith(SHIM_ENCRYPTED_CONTENT_PREFIX):
                continue
            sanitized = _sanitize_chatgpt_passthrough_value(item)
            if sanitized is not _DROP_ITEM:
                output[key] = sanitized
        return output
    return value


def _has_shim_encrypted_content(value: dict[str, Any]) -> bool:
    encrypted_content = value.get("encrypted_content")
    return isinstance(encrypted_content, str) and encrypted_content.startswith(SHIM_ENCRYPTED_CONTENT_PREFIX)


def _rewrite_response_model(payload: Any, model: str | None) -> None:
    if not model:
        return
    if isinstance(payload, dict):
        if payload.get("model") == CHATGPT_MODEL_SLUG:
            payload["model"] = model
        for value in payload.values():
            _rewrite_response_model(value, model)
    elif isinstance(payload, list):
        for item in payload:
            _rewrite_response_model(item, model)


class AnthropicMessagesStreamState:
    """Translates OpenAI chat-completions chunks into Anthropic Messages SSE."""

    def __init__(self, model: str):
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.model = model
        self.next_index = 0
        self.text_index: int | None = None
        self.reasoning_index: int | None = None
        self.text_open = False
        self.reasoning_open = False
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, Any] | None = None
        self.stop_reason = "end_turn"

    async def start(self, response: web.StreamResponse) -> None:
        await _write_anthropic_sse(
            response,
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

    async def write_chat_delta(self, response: web.StreamResponse, chunk: dict[str, Any]) -> None:
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage = normalize_responses_usage(usage)
        choice = (chunk.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            self.stop_reason = _chat_finish_to_anthropic_stop(finish_reason)
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            await self._reasoning_delta(response, str(reasoning))
        content = delta.get("content")
        if content:
            if self.reasoning_open:
                await self._close_reasoning(response)
            await self._text_delta(response, str(content))
        for call in delta.get("tool_calls") or []:
            await self._tool_delta(response, call)

    async def finish(self, response: web.StreamResponse) -> None:
        if self.reasoning_open:
            await self._close_reasoning(response)
        if self.text_open:
            await self._close_text(response)
        for index in sorted(self.tool_calls):
            state = self.tool_calls[index]
            if not state.get("closed"):
                await self._close_tool(response, index, state)
        await _write_anthropic_sse(
            response,
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                "usage": _responses_usage_to_anthropic_usage(self.usage) or {"output_tokens": 0},
            },
        )
        await _write_anthropic_sse(response, "message_stop", {"type": "message_stop"})

    async def _text_delta(self, response: web.StreamResponse, text: str) -> None:
        if self.text_index is None:
            self.text_index = self.next_index
            self.next_index += 1
            self.text_open = True
            await _write_anthropic_sse(
                response,
                "content_block_start",
                {"type": "content_block_start", "index": self.text_index, "content_block": {"type": "text", "text": ""}},
            )
        await _write_anthropic_sse(
            response,
            "content_block_delta",
            {"type": "content_block_delta", "index": self.text_index, "delta": {"type": "text_delta", "text": text}},
        )

    async def _close_text(self, response: web.StreamResponse) -> None:
        if self.text_index is None:
            return
        await _write_anthropic_sse(response, "content_block_stop", {"type": "content_block_stop", "index": self.text_index})
        self.text_index = None
        self.text_open = False

    async def _reasoning_delta(self, response: web.StreamResponse, text: str) -> None:
        if self.reasoning_index is None:
            self.reasoning_index = self.next_index
            self.next_index += 1
            self.reasoning_open = True
            await _write_anthropic_sse(
                response,
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.reasoning_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
        await _write_anthropic_sse(
            response,
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self.reasoning_index,
                "delta": {"type": "thinking_delta", "thinking": text},
            },
        )

    async def _close_reasoning(self, response: web.StreamResponse) -> None:
        if self.reasoning_index is None:
            return
        await _write_anthropic_sse(
            response,
            "content_block_stop",
            {"type": "content_block_stop", "index": self.reasoning_index},
        )
        self.reasoning_index = None
        self.reasoning_open = False

    async def _tool_delta(self, response: web.StreamResponse, call: dict[str, Any]) -> None:
        index = int(call.get("index", 0))
        fn = call.get("function") or {}
        state = self.tool_calls.setdefault(
            index,
            {
                "id": "",
                "name": "",
                "arguments": "",
                "emitted": 0,
                "block_index": None,
                "open": False,
                "closed": False,
            },
        )
        if call.get("id"):
            state["id"] = call["id"]
        if fn.get("name"):
            state["name"] += fn["name"]
        if fn.get("arguments"):
            state["arguments"] += fn["arguments"]
        if not state["open"] and state["name"]:
            if self.reasoning_open:
                await self._close_reasoning(response)
            if self.text_open:
                await self._close_text(response)
            await self._open_tool(response, index, state)
        if state["open"] and len(state["arguments"]) > state["emitted"]:
            delta = state["arguments"][state["emitted"] :]
            state["emitted"] = len(state["arguments"])
            await _write_anthropic_sse(
                response,
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": state["block_index"],
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                },
            )

    async def _open_tool(self, response: web.StreamResponse, index: int, state: dict[str, Any]) -> None:
        state["block_index"] = self.next_index
        self.next_index += 1
        state["open"] = True
        if not state["id"]:
            state["id"] = f"call_{index}"
        await _write_anthropic_sse(
            response,
            "content_block_start",
            {
                "type": "content_block_start",
                "index": state["block_index"],
                "content_block": {
                    "type": "tool_use",
                    "id": state["id"],
                    "name": state["name"] or "tool",
                    "input": {},
                },
            },
        )

    async def _close_tool(self, response: web.StreamResponse, index: int, state: dict[str, Any]) -> None:
        if not state["open"]:
            await self._open_tool(response, index, state)
            if len(state["arguments"]) > state["emitted"]:
                delta = state["arguments"][state["emitted"] :]
                state["emitted"] = len(state["arguments"])
                await _write_anthropic_sse(
                    response,
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": state["block_index"],
                        "delta": {"type": "input_json_delta", "partial_json": delta},
                    },
                )
        await _write_anthropic_sse(
            response,
            "content_block_stop",
            {"type": "content_block_stop", "index": state["block_index"]},
        )
        state["open"] = False
        state["closed"] = True


class ResponsesStreamState:
    """Translates upstream chat-completions / anthropic stream events into the
    Codex Desktop Responses-API event sequence. Keeps the message item and
    each tool call as separate output items with stable indices, and emits
    proper .added / .delta / .done / .completed events plus a final
    `response.completed` with the full reconciled `output` array."""

    def __init__(self, model: str, tool_types: dict[str, str] | None = None):
        self.response_id = f"resp_{int(time.time() * 1000)}"
        self.message_item_id = f"msg_{int(time.time() * 1000)}"
        self.model = model
        self.message_index: int | None = None
        self.message_text = ""
        self.message_opened = False
        self.message_closed = False
        self.usage: dict[str, Any] | None = None
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.reasoning_blocks: dict[Any, dict[str, Any]] = {}
        self.next_output_index = 0
        # Map sanitized tool name -> original Responses tool type so we can
        # emit the correct output item type (e.g. custom_tool_call for freeform
        # apply_patch instead of generic function_call).
        self.tool_types = tool_types or {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self, response: web.StreamResponse) -> None:
        await _write_sse(response, {"type": "response.created", "response": self._response("in_progress")})

    async def finish(self, response: web.StreamResponse) -> None:
        for state in sorted(self.reasoning_blocks.values(), key=lambda s: s["output_index"]):
            if not state.get("closed"):
                await self._close_reasoning(response, state)
        if self.message_opened and not self.message_closed:
            await self._close_message(response)
        for state in sorted(self.tool_calls.values(), key=lambda s: s["output_index"]):
            if not state.get("closed"):
                await self._close_tool(response, state)
        await _write_sse(response, {"type": "response.completed", "response": self._response("completed", final=True)})
        await response.write(b"data: [DONE]\n\n")

    # ------------------------------------------------------------------
    # Chat-completions (OpenAI-style) deltas
    # ------------------------------------------------------------------
    async def write_chat_delta(self, response: web.StreamResponse, chunk: dict[str, Any]) -> None:
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage = normalize_responses_usage(usage)
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            await self._chat_reasoning_delta(response, reasoning)
        content = delta.get("content")
        if content:
            for state in list(self.reasoning_blocks.values()):
                if not state.get("closed"):
                    await self._close_reasoning(response, state)
            await self._text_delta(response, content)
        for call in delta.get("tool_calls") or []:
            await self._chat_tool_delta(response, call)

    async def _chat_reasoning_delta(self, response: web.StreamResponse, text: str) -> None:
        state = self.reasoning_blocks.get(("chat",))
        if state is None:
            state = await self._open_reasoning(response, key=("chat",))
        state["text"] += text
        await _write_sse(
            response,
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": state["id"],
                "output_index": state["output_index"],
                "summary_index": 0,
                "delta": text,
            },
        )

    async def _chat_tool_delta(self, response: web.StreamResponse, call: dict[str, Any]) -> None:
        index = int(call.get("index", 0))
        fn = call.get("function") or {}
        state = self.tool_calls.get(index)
        if state is None:
            call_id = call.get("id") or f"call_{index}"
            state = await self._open_tool(response, key=index, call_id=call_id, name=fn.get("name") or "")
        else:
            if fn.get("name"):
                state["name"] += fn["name"]
        arg_delta = fn.get("arguments") or ""
        if arg_delta:
            state["arguments"] += arg_delta
            if state.get("output_type") != "custom_tool_call":
                await _write_sse(
                    response,
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": state["id"],
                        "output_index": state["output_index"],
                        "delta": arg_delta,
                    },
                )

    # ------------------------------------------------------------------
    # Anthropic deltas
    # ------------------------------------------------------------------
    async def write_anthropic_delta(self, response: web.StreamResponse, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message") or {}
            usage = message.get("usage")
            if isinstance(usage, dict):
                self.usage = normalize_responses_usage(usage)
        if event_type == "content_block_start":
            block = event.get("content_block") or {}
            idx = int(event.get("index", 0))
            btype = block.get("type")
            if btype == "text":
                seed = block.get("text") or ""
                if seed:
                    await self._text_delta(response, seed)
            elif btype == "tool_use":
                await self._open_tool(
                    response,
                    key=("anthropic", idx),
                    call_id=block.get("id") or f"call_{idx}",
                    name=block.get("name") or "",
                )
            elif btype in {"thinking", "redacted_thinking"}:
                await self._open_reasoning(
                    response,
                    key=("anthropic_thinking", idx),
                    initial_text=block.get("thinking") or "",
                    initial_signature=block.get("signature") or "",
                    redacted=(btype == "redacted_thinking"),
                    redacted_data=block.get("data") or "",
                )
        elif event_type == "content_block_delta":
            idx = int(event.get("index", 0))
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                await self._text_delta(response, delta.get("text", ""))
            elif dtype == "input_json_delta":
                state = self.tool_calls.get(("anthropic", idx))
                if state is not None:
                    arg_delta = delta.get("partial_json") or ""
                    if arg_delta:
                        state["arguments"] += arg_delta
                        await _write_sse(
                            response,
                            {
                                "type": "response.function_call_arguments.delta",
                                "item_id": state["id"],
                                "output_index": state["output_index"],
                                "delta": arg_delta,
                            },
                        )
            elif dtype == "thinking_delta":
                state = self.reasoning_blocks.get(("anthropic_thinking", idx))
                if state is None:
                    state = await self._open_reasoning(response, key=("anthropic_thinking", idx))
                txt = delta.get("thinking") or ""
                if txt:
                    state["text"] += txt
                    await _write_sse(
                        response,
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": state["id"],
                            "output_index": state["output_index"],
                            "summary_index": 0,
                            "delta": txt,
                        },
                    )
            elif dtype == "signature_delta":
                state = self.reasoning_blocks.get(("anthropic_thinking", idx))
                if state is None:
                    state = await self._open_reasoning(response, key=("anthropic_thinking", idx))
                state["signature"] += delta.get("signature") or ""
        elif event_type == "message_delta":
            usage = event.get("usage")
            if isinstance(usage, dict):
                if self.usage is None or any(
                    key in usage for key in ("input_tokens", "prompt_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
                ):
                    normalized = normalize_responses_usage(usage)
                    if normalized is not None:
                        self.usage = normalized if self.usage is None else {**self.usage, **normalized}
                output_tokens = usage.get("output_tokens")
                if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
                    if self.usage is None:
                        self.usage = normalize_responses_usage(usage)
                    else:
                        self.usage["output_tokens"] = output_tokens
                        self.usage["total_tokens"] = int(self.usage.get("input_tokens") or 0) + output_tokens
        elif event_type == "content_block_stop":
            idx = int(event.get("index", 0))
            tool_state = self.tool_calls.get(("anthropic", idx))
            if tool_state is not None and not tool_state.get("closed"):
                await self._close_tool(response, tool_state)
            r_state = self.reasoning_blocks.get(("anthropic_thinking", idx))
            if r_state is not None and not r_state.get("closed"):
                await self._close_reasoning(response, r_state)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _open_message(self, response: web.StreamResponse) -> None:
        self.message_index = self.next_output_index
        self.next_output_index += 1
        self.message_opened = True
        await _write_sse(
            response,
            {
                "type": "response.output_item.added",
                "output_index": self.message_index,
                "item": {
                    "id": self.message_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
        )
        await _write_sse(
            response,
            {
                "type": "response.content_part.added",
                "item_id": self.message_item_id,
                "output_index": self.message_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )

    async def _close_message(self, response: web.StreamResponse) -> None:
        if not self.message_opened or self.message_closed:
            return
        self.message_closed = True
        await _write_sse(
            response,
            {
                "type": "response.output_text.done",
                "item_id": self.message_item_id,
                "output_index": self.message_index,
                "content_index": 0,
                "text": self.message_text,
            },
        )
        await _write_sse(
            response,
            {
                "type": "response.content_part.done",
                "item_id": self.message_item_id,
                "output_index": self.message_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": self.message_text, "annotations": []},
            },
        )
        await _write_sse(
            response,
            {
                "type": "response.output_item.done",
                "output_index": self.message_index,
                "item": self._message_item("completed"),
            },
        )

    async def _text_delta(self, response: web.StreamResponse, text: str) -> None:
        if not text:
            return
        if not self.message_opened:
            await self._open_message(response)
        self.message_text += text
        await _write_sse(
            response,
            {
                "type": "response.output_text.delta",
                "item_id": self.message_item_id,
                "output_index": self.message_index,
                "content_index": 0,
                "delta": text,
            },
        )

    async def _open_tool(self, response: web.StreamResponse, *, key: Any, call_id: str, name: str) -> dict[str, Any]:
        # Close the assistant message before opening tool items, matching the
        # OpenAI Responses-API ordering Codex expects.
        if self.message_opened and not self.message_closed:
            await self._close_message(response)
        output_index = self.next_output_index
        self.next_output_index += 1
        # Determine output item type based on original tool type.
        # Freeform tools (apply_patch with no schema) emit custom_tool_call
        # so Codex Desktop knows not to validate against a fixed enum.
        original_type = self.tool_types.get(name, "")
        output_type = "function_call"
        if original_type in {"apply_patch", "custom"} or name == "apply_patch":
            output_type = "custom_tool_call"
        elif original_type.startswith("web_search"):
            output_type = "web_search_call"
        state: dict[str, Any] = {
            "id": call_id,
            "call_id": call_id,
            "name": name,
            "arguments": "",
            "output_index": output_index,
            "closed": False,
            "output_type": output_type,
        }
        self.tool_calls[key] = state
        item = {
            "id": call_id,
            "type": output_type,
            "status": "in_progress",
            "call_id": call_id,
            "name": name,
        }
        item["input" if output_type == "custom_tool_call" else "arguments"] = ""
        await _write_sse(
            response,
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": item,
            },
        )
        return state

    async def _close_tool(self, response: web.StreamResponse, state: dict[str, Any]) -> None:
        state["closed"] = True
        if state.get("output_type") == "custom_tool_call":
            custom_input = _custom_tool_input(state["arguments"])
            if state.get("name") == "apply_patch":
                custom_input = _normalize_apply_patch_input(custom_input)
            state["input"] = custom_input
            await _write_sse(
                response,
                {
                    "type": "response.custom_tool_call_input.delta",
                    "item_id": state["id"],
                    "output_index": state["output_index"],
                    "delta": custom_input,
                },
            )
            await _write_sse(
                response,
                {
                    "type": "response.custom_tool_call_input.done",
                    "item_id": state["id"],
                    "output_index": state["output_index"],
                    "input": custom_input,
                },
            )
        else:
            await _write_sse(
                response,
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": state["id"],
                    "output_index": state["output_index"],
                    "arguments": state["arguments"],
                },
            )
        await _write_sse(
            response,
            {
                "type": "response.output_item.done",
                "output_index": state["output_index"],
                "item": self._tool_item(state, "completed"),
            },
        )

    async def _open_reasoning(
        self,
        response: web.StreamResponse,
        *,
        key: Any,
        initial_text: str = "",
        initial_signature: str = "",
        redacted: bool = False,
        redacted_data: str = "",
    ) -> dict[str, Any]:
        # Reasoning items are emitted before the assistant message/tool calls
        # so we open them eagerly. If a message/tool was already opened we
        # still slot them in at the next available output_index; Codex orders
        # by output_index when reconciling.
        output_index = self.next_output_index
        self.next_output_index += 1
        item_id = f"rs_{int(time.time() * 1000)}_{output_index}"
        state: dict[str, Any] = {
            "id": item_id,
            "output_index": output_index,
            "text": initial_text,
            "signature": initial_signature,
            "redacted": redacted,
            "redacted_data": redacted_data,
            "closed": False,
        }
        self.reasoning_blocks[key] = state
        await _write_sse(
            response,
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "id": item_id,
                    "type": "reasoning",
                    "status": "in_progress",
                    "summary": [],
                    "encrypted_content": None,
                },
            },
        )
        if initial_text:
            await _write_sse(
                response,
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "delta": initial_text,
                },
            )
        return state

    async def _close_reasoning(self, response: web.StreamResponse, state: dict[str, Any]) -> None:
        state["closed"] = True
        # Emit summary_text.done so renderers can finalize the reasoning bubble.
        await _write_sse(
            response,
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": state["id"],
                "output_index": state["output_index"],
                "summary_index": 0,
                "text": state["text"],
            },
        )
        await _write_sse(
            response,
            {
                "type": "response.output_item.done",
                "output_index": state["output_index"],
                "item": self._reasoning_item(state, "completed"),
            },
        )

    def _reasoning_item(self, state: dict[str, Any], status: str) -> dict[str, Any]:
        # Encode the original Anthropic thinking block in encrypted_content so
        # we can roundtrip it back on the next turn. Codex preserves this
        # field verbatim across turns.
        if state.get("redacted"):
            payload = {"type": "redacted_thinking", "data": state.get("redacted_data", "")}
        else:
            payload = {
                "type": "thinking",
                "thinking": state.get("text", ""),
                "signature": state.get("signature", ""),
            }
        encrypted = _encode_thinking_payload(payload)
        return {
            "id": state["id"],
            "type": "reasoning",
            "status": status,
            "summary": (
                [{"type": "summary_text", "text": state.get("text", "")}]
                if state.get("text") and not state.get("redacted")
                else []
            ),
            "encrypted_content": encrypted,
        }

    def _message_item(self, status: str) -> dict[str, Any]:
        return {
            "id": self.message_item_id,
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": self.message_text, "annotations": []}
            ] if self.message_text else [],
        }

    def _tool_item(self, state: dict[str, Any], status: str) -> dict[str, Any]:
        item = {
            "id": state["id"],
            "type": state.get("output_type", "function_call"),
            "status": status,
            "call_id": state["call_id"],
            "name": state["name"],
        }
        if item["type"] == "custom_tool_call":
            item["input"] = state.get("input") or _custom_tool_input(state["arguments"])
        else:
            item["arguments"] = state["arguments"]
        return item

    def _response(self, status: str, *, final: bool = False) -> dict[str, Any]:
        output: list[dict[str, Any]] = []
        if final:
            collected: list[tuple[int, dict[str, Any]]] = []
            for state in self.reasoning_blocks.values():
                collected.append((state["output_index"], self._reasoning_item(state, "completed")))
            if self.message_opened and self.message_text and self.message_index is not None:
                collected.append((self.message_index, self._message_item("completed")))
            for state in self.tool_calls.values():
                collected.append((state["output_index"], self._tool_item(state, "completed")))
            collected.sort(key=lambda pair: pair[0])
            output = [item for _, item in collected]
        payload = {
            "id": self.response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": status,
            "model": self.model,
            "output": output,
        }
        if self.usage is not None:
            payload["usage"] = self.usage
        elif final:
            payload["usage"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return payload


_THINKING_MAGIC = "anthropic-thinking-v1:"


def _encode_thinking_payload(payload: dict[str, Any]) -> str:
    import base64

    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return _THINKING_MAGIC + base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_thinking_payload(encoded: str) -> dict[str, Any] | None:
    import base64

    if not isinstance(encoded, str) or not encoded.startswith(_THINKING_MAGIC):
        return None
    blob = encoded[len(_THINKING_MAGIC) :]
    try:
        raw = base64.urlsafe_b64decode(blob.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _custom_tool_input(arguments: str) -> str:
    """Unwrap the chat-compatible JSON envelope for a freeform custom tool."""
    try:
        parsed = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        return arguments
    if isinstance(parsed, dict):
        for key in ("input", "patch"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value
    return arguments


def _normalize_apply_patch_input(arguments: str) -> str:
    """Normalize common Claude patch dialects into Codex apply_patch grammar."""
    patch = _custom_tool_input(arguments).strip()
    fenced = re.fullmatch(
        r"```(?:diff|patch)?\s*\n([\s\S]*?)\n```",
        patch,
        re.IGNORECASE,
    )
    if fenced:
        patch = fenced.group(1).strip()

    begin = patch.find("*** Begin Patch")
    end = patch.rfind("*** End Patch")
    if begin >= 0:
        patch = patch[begin:]
    if end >= 0:
        patch = patch[: end + len("*** End Patch")]

    lines = patch.splitlines()
    normalized: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if re.match(
            r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@(?:\s+.*)?$",
            line,
        ):
            normalized.append("@@")
            index += 1
            continue
        update_match = re.match(
            r"^\*\*\* Update(?:\s+\d+)?(?: File)?:\s*(.+)$",
            line,
        )
        if update_match:
            normalized.append(f"*** Update File: {update_match.group(1).strip()}")
            index += 1
            continue
        if re.match(
            r"^\*\*\* Update(?:\s+\d+|\s+Hunk(?::.*)?)?\s*$",
            line,
        ):
            index += 1
            old_lines: list[str] = []
            while index < len(lines) and lines[index] != "---":
                if lines[index].startswith("*** ") or lines[index].startswith("@@"):
                    break
                old_lines.append(lines[index])
                index += 1
            if index < len(lines) and lines[index] == "---":
                index += 1
                new_lines: list[str] = []
                while index < len(lines) and not (
                    lines[index].startswith("*** ")
                    or lines[index].startswith("@@")
                ):
                    new_lines.append(lines[index])
                    index += 1
                normalized.append("@@")
                normalized.extend(f"-{value}" for value in old_lines)
                normalized.extend(f"+{value}" for value in new_lines)
            else:
                normalized.append("@@")
                normalized.extend(old_lines)
            continue
        if re.match(r"^\*\*\* Hunk(?:\s+\d+)?(?::.*)?$", line):
            normalized.append("@@")
            index += 1
            continue
        if re.match(r"^\*\*\* Find:?\s*$", line):
            old_lines: list[str] = []
            index += 1
            while index < len(lines) and not re.match(
                r"^\*\*\* Replace:?\s*$",
                lines[index],
            ):
                old_lines.append(lines[index])
                index += 1
            if index >= len(lines):
                normalized.append(line)
                normalized.extend(old_lines)
                break
            index += 1
            new_lines: list[str] = []
            while index < len(lines) and not (
                lines[index].startswith("*** ")
                or lines[index].startswith("@@")
            ):
                new_lines.append(lines[index])
                index += 1
            normalized.append("@@")
            normalized.extend(f"-{value}" for value in old_lines)
            normalized.extend(f"+{value}" for value in new_lines)
            continue
        normalized.append(line)
        index += 1

    return "\n".join(normalized)


def _build_tool_types(body: dict[str, Any]) -> dict[str, str]:
    """Build a map sanitized tool name -> original tool type from the request tools array.

    Codex Desktop emits native tools like `{"type": "apply_patch"}` and MCP tools
    like `{"type": "mcp__node_repl", "function": {"name": "js"}}`. When we translate
    those into chat-completions `function` tools, the original type is lost. We
    preserve it here so the Responses streaming translator can emit the correct
    output item type (e.g. `custom_tool_call` for freeform apply_patch instead of
    generic `function_call`).
    """
    tool_types: dict[str, str] = {}
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "").strip().lower()
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            name = str(fn["name"]).strip()
        elif tool.get("name"):
            name = str(tool["name"]).strip()
        else:
            name = tool_type
        clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())[:64].strip("_")
        if clean:
            tool_types[clean] = tool_type
    return tool_types

async def _perform_web_search(query: str) -> str:
    """Execute a web search via DuckDuckGo and return text results.

    This is a server-side fallback for custom models whose provider does not
    have a native web-search capability.  Codex Desktop expects the shim to
    return results as a `function_call_output` (or `web_search_call`) item;
    when the model is BYOK, the Desktop app does not execute the search itself,
    so the shim must do it and feed the results back into the conversation.
    """
    import urllib.parse
    import urllib.request

    if not query or not query.strip():
        return "No search query provided."

    # DuckDuckGo lite HTML endpoint (no API key required)
    url = (
        "https://html.duckduckgo.com/html/"
        + "?q="
        + urllib.parse.quote_plus(query.strip())
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Web search failed: {exc}"

    # Extract title + snippet from result links
    results: list[str] = []
    # Each result is in a `.result` div with `.result__a` (title/link) and `.result__snippet`
    from html.parser import HTMLParser

    class _ResultParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.in_result = False
            self.in_a = False
            self.in_snippet = False
            self.current_title = ""
            self.current_snippet = ""
            self.results: list[dict[str, str]] = []
            self._tag_stack: list[str] = []
            self._class_stack: list[str] = []

        def _current_class(self) -> str:
            return self._class_stack[-1] if self._class_stack else ""

        def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
            attrs = dict(attrs_list)
            cls = (attrs.get("class") or "").lower()
            self._tag_stack.append(tag)
            self._class_stack.append(cls)
            if "result" in cls and tag == "div":
                self.in_result = True
                self.current_title = ""
                self.current_snippet = ""
            if self.in_result and tag == "a" and "result__a" in cls:
                self.in_a = True
            if self.in_result and ("result__snippet" in cls or "result__body" in cls):
                self.in_snippet = True

        def handle_endtag(self, tag: str) -> None:
            if self._tag_stack and self._tag_stack[-1] == tag:
                self._tag_stack.pop()
                self._class_stack.pop()
            if tag == "div" and self.in_result:
                if self.current_title or self.current_snippet:
                    self.results.append(
                        {
                            "title": self.current_title.strip(),
                            "snippet": self.current_snippet.strip(),
                        }
                    )
                self.in_result = False
            if tag == "a":
                self.in_a = False
            if tag in {"div", "span", "p"}:
                self.in_snippet = False

        def handle_data(self, data: str) -> None:
            if self.in_a:
                self.current_title += data
            if self.in_snippet:
                self.current_snippet += data

    parser = _ResultParser()
    parser.feed(html)
    for r in parser.results[:5]:
        title = r["title"].replace("\n", " ")
        snippet = r["snippet"].replace("\n", " ")
        if title and snippet:
            results.append(f"{title}\n{snippet}")
        elif title:
            results.append(title)
        elif snippet:
            results.append(snippet)

    if not results:
        return "No web search results found."
    return "\n\n".join(results)

def _maybe_intercept_web_search(payload: dict[str, Any]) -> dict[str, Any] | None:
    """If the response payload contains a web_search_call, execute it server-side
    and return a new payload with the results embedded as a function_call_output.

    Returns None if no web_search_call is present (pass through unchanged).
    """
    output = payload.get("output") or []
    if not isinstance(output, list):
        return None
    search_calls: list[tuple[int, dict[str, Any]]] = []
    for i, item in enumerate(output):
        if isinstance(item, dict) and item.get("type") == "web_search_call":
            search_calls.append((i, item))
    if not search_calls:
        return None

    # Build synthetic search results
    results: list[dict[str, Any]] = []
    for idx, call in search_calls:
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        query = args.get("query") or ""
        # Run the search synchronously (non-streaming path only)
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            result_text = loop.run_until_complete(_perform_web_search(query))
        except RuntimeError:
            result_text = "Web search unavailable in this context."
        results.append({
            "id": f"wso_{call.get('call_id', '0')}",
            "type": "function_call_output",
            "status": "completed",
            "call_id": call.get("call_id"),
            "output": result_text,
        })

    # Replace web_search_call items with their results
    new_output: list[dict[str, Any]] = []
    for i, item in enumerate(output):
        if isinstance(item, dict) and item.get("type") == "web_search_call":
            # Find matching result
            for r in results:
                if r.get("call_id") == item.get("call_id"):
                    new_output.append(r)
                    break
            else:
                new_output.append(item)
        else:
            new_output.append(item)

    new_payload = dict(payload)
    new_payload["output"] = new_output
    return new_payload


_VERSIONED_BASE_RE = re.compile(r"(?:^|/)v\d+$")


def _join_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if _VERSIONED_BASE_RE.search(base):
        # Already ends with /v<n> (e.g. /v1, /api/coding/v3) — append
        # the endpoint as-is rather than injecting another /v1/.
        return base + endpoint
    if endpoint == "/messages":
        return base + "/v1/messages"
    return urljoin(base + "/", "v1" + endpoint)


def _openai_headers(route: ShimModel) -> dict[str, str]:
    headers = {"Content-Type": "application/json", **route.extra_headers}
    if route.api_key:
        headers.setdefault("Authorization", f"Bearer {route.api_key}")
    return headers


def _anthropic_headers(route: ShimModel) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        **route.extra_headers,
    }
    if route.api_key:
        headers.setdefault("x-api-key", route.api_key)
    return headers


def _anthropic_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    parts = [
        str(block.get("text") or "")
        for block in (payload.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


def _sse_response() -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    return response


async def _safe_write(response: web.StreamResponse, data: bytes) -> None:
    try:
        await response.write(data)
    except (ConnectionResetError, ConnectionError):
        raise ClientDisconnected()
    except Exception as exc:
        if exc.__class__.__name__ in {
            "ClientConnectionResetError",
            "ClientConnectionError",
            "ClientPayloadError",
        }:
            raise ClientDisconnected() from exc
        raise


async def _await_with_sse_heartbeats(
    awaitable: Any,
    response: web.StreamResponse,
    *,
    timeout: float,
    interval: float = 1.0,
) -> Any:
    """Keep Codex connected while a slow upstream request returns headers."""
    import asyncio

    task = asyncio.ensure_future(awaitable)
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=min(interval, remaining),
                )
            except asyncio.TimeoutError:
                if task.done():
                    return task.result()
                # SSE comments are ignored by Codex's event parser but count
                # as response activity, preventing Reconnect 1..3 loops.
                await _safe_write(response, b": codex-shim keep-alive\n\n")
    except BaseException:
        if not task.done():
            task.cancel()
        raise


async def _write_sse(response: web.StreamResponse, payload: dict[str, Any]) -> None:
    try:
        await response.write(f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode())
    except (ConnectionResetError, ConnectionError) as exc:
        raise ClientDisconnected() from exc
    except Exception as exc:
        # aiohttp raises ClientConnectionResetError (an OSError subclass on
        # some versions, a ClientConnectionError on others). Trap both.
        if exc.__class__.__name__ in {
            "ClientConnectionResetError",
            "ClientConnectionError",
            "ClientPayloadError",
        }:
            raise ClientDisconnected() from exc
        raise


async def _write_anthropic_sse(response: web.StreamResponse, event: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":"))
    try:
        await response.write(f"event: {event}\ndata: {data}\n\n".encode())
    except (ConnectionResetError, ConnectionError) as exc:
        raise ClientDisconnected() from exc
    except Exception as exc:
        if exc.__class__.__name__ in {
            "ClientConnectionResetError",
            "ClientConnectionError",
            "ClientPayloadError",
        }:
            raise ClientDisconnected() from exc
        raise


class ClientDisconnected(Exception):
    """Raised when the downstream Codex client closes the SSE connection."""


def _log_incoming_request(endpoint: str, body: dict[str, Any]) -> None:
    try:
        tools = body.get("tools") or []
        names = []
        for t in tools[:80]:
            if isinstance(t, dict):
                name = t.get("name") or (t.get("function") or {}).get("name") or t.get("type")
                if name:
                    names.append(str(name))
        input_items = body.get("input") or []
        input_summary = []
        if isinstance(input_items, list):
            for item in input_items[-6:]:
                if isinstance(item, dict):
                    t = item.get("type") or item.get("role") or "?"
                    extra = ""
                    if t == "function_call":
                        extra = f"({item.get('name', '?')})"
                    elif t == "function_call_output":
                        extra = f"(call_id={str(item.get('call_id', ''))[:24]})"
                    input_summary.append(f"{t}{extra}")
        print(
            f"[req] {endpoint} model={body.get('model')!r} stream={body.get('stream')!r} "
            f"tools={len(tools)} ({names[:8]}) "
            f"input={len(input_items)} ({input_summary})",
            flush=True,
        )
    except Exception as exc:
        print(f"[req] failed to log: {exc}", flush=True)


async def _sse_lines(upstream) -> Any:
    buffer = b""
    async for chunk in upstream.content.iter_chunked(4096):
        buffer += chunk
        while b"\n" in buffer:
            raw, buffer = buffer.split(b"\n", 1)
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("data:"):
                yield line[5:].strip()
    tail = buffer.decode("utf-8", errors="replace").strip()
    if tail.startswith("data:"):
        yield tail[5:].strip()


def _deepseek_stream_failure(line: str) -> str | None:
    """Detect a DeepSeek failure before committing the downstream SSE response."""
    if line == "[DONE]":
        return "DeepSeek Web returned no stream events"
    try:
        chunk = json.loads(line)
    except json.JSONDecodeError:
        return None
    error = chunk.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "DeepSeek Web returned an empty error")
    choices = chunk.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = str(delta.get("content") or "")
    legacy_empty = (
        "DeepSeek Web did not produce a usable tool call or final answer "
        "after two automatic retries"
    )
    if legacy_empty in content:
        return "DeepSeek Web returned an empty response after automatic retries"
    return None


def _anthropic_stream_to_chat_chunk(event: dict[str, Any], model: str) -> dict[str, Any]:
    content = ""
    if event.get("type") == "content_block_delta":
        delta = event.get("delta") or {}
        if delta.get("type") == "text_delta":
            content = delta.get("text", "")
    return {"object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}


def _compact_request_body(body: dict[str, Any], upstream_model: str) -> dict[str, Any]:
    instructions = body.get("instructions") or _default_compact_instructions()
    return {
        "model": upstream_model,
        "instructions": instructions,
        "input": body.get("input") or [],
        "max_output_tokens": body.get("max_output_tokens") or body.get("max_tokens") or 4096,
        "stream": False,
    }


def _default_compact_instructions() -> str:
    return (
        "Compact the conversation into a concise state handoff for the next Codex turn. "
        "Preserve the active task, user requirements, important file paths, commands already run, "
        "tool results, decisions, blockers, and the latest state. Omit filler and repeated text."
    )


async def _as_compact_response(response: web.StreamResponse, model: str) -> web.Response:
    if not isinstance(response, web.Response) or response.status >= 400:
        return response
    try:
        payload = json.loads(response.text or "{}")
    except json.JSONDecodeError:
        return response
    output = payload.get("output") if isinstance(payload, dict) else None
    summary = _compact_summary_from_output(output)
    compacted = _compact_response_payload(model, summary, payload.get("usage") if isinstance(payload, dict) else None)
    return web.json_response(compacted)


def _compact_summary_from_output(output: Any) -> str:
    parts: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content = item.get("content") or []
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text"):
                            parts.append(str(part["text"]))
            elif item.get("type") == "output_text" and item.get("text"):
                parts.append(str(item["text"]))
    return "\n".join(part for part in parts if part).strip()


def _compact_response_payload(model: str, summary: str, usage: Any = None) -> dict[str, Any]:
    now = int(time.time())
    response_id = f"resp_compact_{now}"
    text = summary or "No prior conversation state was available to compact."
    payload = {
        "id": response_id,
        "object": "response",
        "created_at": now,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": f"msg_compact_{now}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


async def _error_response(upstream, *, slug: str | None = None) -> web.Response:
    text = await upstream.text()
    if slug:
        print(
            f"[err] upstream {slug} returned {upstream.status}: {text[:500]}",
            flush=True,
        )
    return web.Response(status=upstream.status, text=text, content_type=upstream.content_type or "text/plain")


async def _anthropic_error_response(upstream) -> web.Response:
    text = await upstream.text()
    message = text
    error_type = "api_error"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            message = str(err.get("message") or message)
            error_type = str(err.get("type") or error_type)
        elif payload.get("message"):
            message = str(payload["message"])
    status_type = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
    }.get(upstream.status)
    if status_type:
        error_type = status_type
    body = {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }
    request_id = upstream.headers.get("request-id") or upstream.headers.get("x-request-id")
    if request_id:
        body["request_id"] = request_id
    return web.json_response(body, status=upstream.status)


def _missing_api_key_message(route: ShimModel) -> str:
    env_name = route.raw.get("api_key_env") or route.raw.get("apiKeyEnv")
    if env_name:
        return f"Model {route.slug} has no API key. Set {env_name} or add api_key/apiKey for this model."
    return f"Model {route.slug} has no API key. Add api_key/apiKey or api_key_env/apiKeyEnv for this model."


def _normalize_roles(messages: list[dict]) -> list[dict]:
    result = []
    for message in messages:
        if isinstance(message, dict):
            message = dict(message)
            if message.get("role") == "developer":
                message["role"] = "system"
        result.append(message)
    return result


def _dump_debug_request(slug: str, url: str, body: dict[str, Any]) -> None:
    """Best-effort dump of the last forwarded request body for debugging.

    Writes ``.codex-shim/last_request.json`` next to the rest of the runtime
    state (catalog, pid, log). Failures are silently swallowed — this is a
    debug aid, not a code path the request should depend on.
    """
    try:
        dump_path = DEBUG_DIR / "last_request.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"slug": slug, "url": url, "body": body}
        full = json.dumps(payload, indent=2, default=str)
        if len(full) > 2_000_000:
            messages = body.get("messages") or []
            summary = {
                "slug": slug,
                "url": url,
                "_truncated": True,
                "_full_size": len(full),
                "message_count": len(messages),
                "tool_count": len(body.get("tools") or []),
                "last_3_messages": messages[-3:],
            }
            dump_path.write_text(json.dumps(summary, indent=2, default=str))
        else:
            dump_path.write_text(full)
    except OSError as exc:
        print(f"[debug] dump failed: {exc}", flush=True)


def _current_managed_model() -> str | None:
    """Return the first ``model = "..."`` value from ~/.codex/config.toml."""
    if not CODEX_CONFIG_PATH.exists():
        return None
    try:
        text = CODEX_CONFIG_PATH.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("model = "):
            return stripped.split("=", 1)[1].strip().strip('"')
    return None


_MODEL_LINE_RE = re.compile(r'(?m)^(\s*model\s*=\s*")[^"]*(")')
_PROVIDER_NAME_RE = re.compile(
    r'(\[model_providers\.' + re.escape(PROVIDER_NAME) + r'\][^\[]*?\n\s*name\s*=\s*")[^"]*(")',
    re.DOTALL,
)


def _set_active_model(slug: str, display_name: str | None = None) -> None:
    """Rewrite the active model + provider label in ~/.codex/config.toml."""
    if not CODEX_CONFIG_PATH.exists():
        return
    try:
        text = CODEX_CONFIG_PATH.read_text()
    except OSError:
        return
    text = _MODEL_LINE_RE.sub(rf'\g<1>{slug}\g<2>', text, count=1)
    if display_name:
        text = _PROVIDER_NAME_RE.sub(rf'\g<1>{display_name}\g<2>', text, count=1)
    try:
        CODEX_CONFIG_PATH.write_text(text)
    except OSError as exc:
        print(f"[switch] failed to write {CODEX_CONFIG_PATH}: {exc}", flush=True)
        return
    print(f"[switch] set active model to {slug} ({display_name})", flush=True)


def _restart_codex_app() -> None:
    """Quit and relaunch Codex Desktop in a background thread (non-blocking).

    Cross-platform: ``taskkill`` + ``Codex.exe`` on Windows, ``osascript`` +
    ``open -a Codex`` on macOS. Linux has no Codex Desktop build today, so
    the branch is a no-op there.
    """
    import os as _os
    import subprocess as _subprocess
    import threading as _threading
    import time as _time

    def _do_restart() -> None:
        try:
            if _os.name == "nt":
                _subprocess.run(
                    ["taskkill", "/IM", "Codex.exe", "/F"],
                    check=False,
                    stdout=_subprocess.DEVNULL,
                    stderr=_subprocess.DEVNULL,
                )
                _time.sleep(1.5)
                local_appdata = _os.environ.get("LOCALAPPDATA", "")
                codex_exe = Path(local_appdata) / "Programs" / "Codex" / "Codex.exe"
                if codex_exe.exists():
                    _subprocess.Popen([str(codex_exe)])
                else:
                    _subprocess.Popen(["Codex.exe"], shell=True)
            elif sys.platform == "darwin":
                quit_script = 'tell application "Codex" to if it is running then quit'
                _subprocess.run(
                    ["osascript", "-e", quit_script],
                    check=False,
                    stdout=_subprocess.DEVNULL,
                    stderr=_subprocess.DEVNULL,
                )
                _time.sleep(1.5)
                _subprocess.Popen(["open", "-a", "Codex"])
        except OSError:
            pass

    _threading.Thread(target=_do_restart, daemon=True).start()


def _picker_html(picker_token: str) -> str:
    token_json = json.dumps(picker_token).replace("<", "\\u003c")
    html = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Codex Shim - Model Picker</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh; padding: 20px;
  }
  .container { max-width: 500px; width: 100%; }
  h1 { font-size: 24px; margin-bottom: 8px; color: #f0f6fc; }
  .subtitle { color: #8b949e; margin-bottom: 24px; font-size: 14px; }
  .model-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 16px; margin-bottom: 12px; cursor: pointer;
    transition: all 0.15s ease; display: flex; align-items: center;
    justify-content: space-between;
  }
  .model-card:hover { border-color: #58a6ff; background: #1c2333; }
  .model-card.active { border-color: #3fb950; background: #1a2e1a; }
  .model-info { flex: 1; }
  .model-name { font-size: 16px; font-weight: 600; color: #f0f6fc; }
  .model-provider { font-size: 12px; color: #8b949e; margin-top: 4px; }
  .model-badge {
    font-size: 11px; padding: 2px 8px; border-radius: 12px;
    font-weight: 600; text-transform: uppercase;
  }
  .badge-active { background: #1a4d2e; color: #3fb950; }
  .badge-switch { background: #1c2333; color: #58a6ff; }
  .status { text-align: center; margin-top: 16px; font-size: 14px; min-height: 20px; }
  .status.ok { color: #3fb950; }
  .status.err { color: #f85149; }
  .status.loading { color: #d29922; }
  .restart-note { color: #8b949e; font-size: 12px; text-align: center; margin-top: 8px; }
  .opt { display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 12px; }
  .opt label { font-size: 13px; color: #8b949e; cursor: pointer; }
  .opt input { cursor: pointer; }
</style>
</head>
<body>
<div class="container">
  <h1>Model Picker</h1>
  <p class="subtitle">Choose the active model for Codex Desktop</p>
  <div id="models"><div class="status loading">Loading models...</div></div>
  <div class="opt">
    <input type="checkbox" id="autoRestart" checked>
    <label for="autoRestart">Auto-restart Codex after switching</label>
  </div>
  <div id="status" class="status"></div>
  <p class="restart-note">Codex needs to restart to use the new model</p>
</div>
<script>
const PICKER_TOKEN = @@TOKEN_JSON@@;
async function loadModels() {
  const res = await fetch('/api/models');
  const models = await res.json();
  const container = document.getElementById('models');
  container.innerHTML = '';
  models.forEach(m => {
    const card = document.createElement('div');
    card.className = 'model-card' + (m.active ? ' active' : '');
    const info = document.createElement('div');
    info.className = 'model-info';
    const name = document.createElement('div');
    name.className = 'model-name';
    name.textContent = m.display_name;
    const prov = document.createElement('div');
    prov.className = 'model-provider';
    prov.textContent = m.provider + ' \u00b7 ' + m.slug;
    info.appendChild(name);
    info.appendChild(prov);
    const badge = document.createElement('span');
    badge.className = 'model-badge ' + (m.active ? 'badge-active' : 'badge-switch');
    badge.textContent = m.active ? 'Active' : 'Switch';
    card.appendChild(info);
    card.appendChild(badge);
    if (!m.active) {
      card.onclick = () => switchModel(m.slug);
    }
    container.appendChild(card);
  });
}
async function switchModel(slug) {
  const status = document.getElementById('status');
  const restart = document.getElementById('autoRestart').checked;
  status.className = 'status loading';
  status.textContent = 'Switching to ' + slug + '...';
  try {
    const res = await fetch('/api/switch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', '@@PICKER_HEADER@@': PICKER_TOKEN},
      body: JSON.stringify({slug, restart_codex: restart})
    });
    const data = await res.json();
    if (data.ok) {
      status.className = 'status ok';
      status.textContent = 'Switched to ' + slug + (restart ? ' \u2014 Codex restarting...' : '');
      setTimeout(loadModels, 1000);
    } else {
      status.className = 'status err';
      status.textContent = data.error || 'Failed';
    }
  } catch(e) {
    status.className = 'status err';
    status.textContent = 'Error: ' + e.message;
  }
}
loadModels();
</script>
</body>
</html>'''
    return (
        html.replace("@@TOKEN_JSON@@", token_json, 1).replace("@@PICKER_HEADER@@", PICKER_TOKEN_HEADER, 1)
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    shim = ShimServer(args.settings, host=args.host)
    web.run_app(shim.app(), host=args.host, port=args.port, handle_signals=True)


if __name__ == "__main__":
    main()
