import asyncio
from pathlib import Path

import pytest

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
