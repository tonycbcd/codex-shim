import asyncio
from pathlib import Path

import pytest

from codex_shim import server
from codex_shim.server import ShimServer


@pytest.mark.asyncio
async def test_deepseek_service_is_shared_and_stops_after_last_request(
    monkeypatch,
    tmp_path: Path,
):
    shim = ShimServer(tmp_path / "settings.json")
    events: list[str] = []

    async def start() -> None:
        events.append("start")

    async def stop() -> None:
        events.append("stop")

    monkeypatch.setattr(shim, "_start_deepseek_service", start)
    monkeypatch.setattr(shim, "_stop_deepseek_service", stop)

    await shim._acquire_deepseek_service()
    await shim._acquire_deepseek_service()
    assert events == ["start"]

    await shim._release_deepseek_service()
    assert events == ["start"]

    # Set idle timeout to 0 so the timer fires on the next event-loop tick
    shim._deepseek_idle_seconds = 0
    await shim._release_deepseek_service()
    # Stop is deferred via call_later; give the event loop a tick to execute it
    await asyncio.sleep(0.01)
    assert events == ["start", "stop"]
    assert shim._deepseek_users == 0


@pytest.mark.asyncio
async def test_deepseek_service_is_not_counted_when_startup_fails(
    monkeypatch,
    tmp_path: Path,
):
    shim = ShimServer(tmp_path / "settings.json")

    async def fail_start() -> None:
        raise RuntimeError("startup failed")

    monkeypatch.setattr(shim, "_start_deepseek_service", fail_start)

    with pytest.raises(RuntimeError, match="startup failed"):
        await shim._acquire_deepseek_service()

    assert shim._deepseek_users == 0


@pytest.mark.asyncio
async def test_deepseek_start_recycles_orphaned_managed_chrome(
    monkeypatch,
    tmp_path: Path,
):
    shim = ShimServer(tmp_path / "settings.json")
    events: list[str] = []
    health = iter((False, True))

    async def healthcheck() -> bool:
        return next(health)

    async def stop_chrome() -> None:
        events.append("stop_chrome")

    class Process:
        pid = 123
        returncode = None

    async def spawn(*args, **kwargs):
        events.append("spawn")
        return Process()

    run_script = tmp_path / "run.sh"
    run_script.write_text("#!/bin/sh\n")
    monkeypatch.setattr(shim, "_deepseek_healthcheck", healthcheck)
    monkeypatch.setattr(shim, "_stop_deepseek_chrome", stop_chrome)
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(server, "DEEPSEEK_RUN_SCRIPT", run_script)
    monkeypatch.setattr(server, "DEEPSEEK_LOG_FILE", tmp_path / "server.log")

    await shim._start_deepseek_service()

    assert events == ["stop_chrome", "spawn"]
    assert shim._deepseek_process is not None
    if shim._deepseek_log_handle is not None:
        shim._deepseek_log_handle.close()


def test_deepseek_resets_oversized_session_index(monkeypatch, tmp_path: Path):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text("x" * 101)
    sessions_file.chmod(0o644)
    monkeypatch.setattr(server, "DEEPSEEK_SESSIONS_FILE", sessions_file)
    monkeypatch.setattr(server, "DEEPSEEK_MAX_SESSIONS_BYTES", 100)

    ShimServer._reset_oversized_deepseek_sessions()

    assert sessions_file.read_text() == '{"sessions":{},"convs":{}}\n'
    assert sessions_file.stat().st_mode & 0o777 == 0o600


def test_deepseek_keeps_session_index_within_limit(monkeypatch, tmp_path: Path):
    sessions_file = tmp_path / "sessions.json"
    original = '{"sessions":{"keep":{}},"convs":{}}\n'
    sessions_file.write_text(original)
    monkeypatch.setattr(server, "DEEPSEEK_SESSIONS_FILE", sessions_file)
    monkeypatch.setattr(server, "DEEPSEEK_MAX_SESSIONS_BYTES", len(original))

    ShimServer._reset_oversized_deepseek_sessions()

    assert sessions_file.read_text() == original
