import asyncio
import threading

import pytest

from core.src.aio import BackgroundLoop


@pytest.fixture
def loop():
    bg = BackgroundLoop("test-loop")
    yield bg
    bg.stop()


async def _thread_name():
    return threading.current_thread().name


async def _numbers(n, fail_at=None, log=None):
    try:
        for i in range(n):
            if i == fail_at:
                raise ValueError("boom")
            await asyncio.sleep(0)
            yield i
    finally:
        if log is not None:
            log.append("cleaned up")


def test_run_executes_on_the_background_thread(loop):
    assert loop.run(_thread_name()) == "test-loop"


def test_arun_from_another_event_loop(loop):
    assert asyncio.run(loop.arun(_thread_name())) == "test-loop"


def test_iterate_yields_items_and_propagates_errors(loop):
    assert list(loop.iterate(_numbers(3))) == [0, 1, 2]
    with pytest.raises(ValueError):
        list(loop.iterate(_numbers(3, fail_at=1)))


def test_aiterate_yields_items_and_propagates_errors(loop):
    async def collect(agen):
        return [x async for x in loop.aiterate(agen)]

    assert asyncio.run(collect(_numbers(3))) == [0, 1, 2]
    with pytest.raises(ValueError):
        asyncio.run(collect(_numbers(3, fail_at=2)))


def test_abandoned_generator_still_runs_its_cleanup(loop):
    log = []
    gen = loop.iterate(_numbers(5, log=log))
    assert next(gen) == 0
    gen.close()
    loop.run(asyncio.sleep(0.05))
    assert log == ["cleaned up"]


def test_run_from_its_own_loop_is_refused(loop):
    async def nested():
        return loop.run(_thread_name())

    with pytest.raises(RuntimeError, match="own loop"):
        loop.run(nested())


def test_restarts_after_stop(loop):
    loop.run(_thread_name())
    loop.stop()
    assert loop.run(_thread_name()) == "test-loop"
