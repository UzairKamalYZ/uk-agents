"""
core/src/aio.py — a private asyncio event loop on a daemon thread.

Async resources that must stay on one event loop (MCP client sessions, the
orchestrator's aiosqlite checkpointer) live on a BackgroundLoop.  Any caller
can use them:

    loop = BackgroundLoop("orchestrator")
    loop.run(coro)              # sync caller: block for the result
    await loop.arun(coro)       # async caller on another loop: await it
    for x in loop.iterate(agen):         ...   # sync caller, async generator
    async for x in loop.aiterate(agen):  ...   # async caller, async generator

Generators are pumped to completion on the background loop even if the caller
stops early, so their cleanup (finally blocks) always runs.
"""
import asyncio
import concurrent.futures
import queue
import threading
from collections.abc import AsyncIterator, Coroutine, Iterator
from typing import Any, TypeVar

T = TypeVar("T")

_DONE = object()


class BackgroundLoop:
    """One asyncio loop on a daemon thread, started on first use."""

    def __init__(self, name: str = "background-loop") -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                loop = asyncio.new_event_loop()
                self._thread = threading.Thread(target=loop.run_forever, name=self._name, daemon=True)
                self._thread.start()
                self._loop = loop
            return self._loop

    def submit(self, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        return asyncio.run_coroutine_threadsafe(coro, self._ensure())

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run on the background loop and block for the result (sync callers)."""
        if self._thread is threading.current_thread():
            coro.close()
            raise RuntimeError(f"{self._name}: run() called from its own loop — await the coroutine instead")
        return self.submit(coro).result()

    async def arun(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run on the background loop and await the result (async callers, any loop)."""
        return await asyncio.wrap_future(self.submit(coro))

    def iterate(self, agen: AsyncIterator[T]) -> Iterator[T]:
        """Consume an async generator from sync code, item by item as they are produced."""
        items: queue.Queue = queue.Queue()

        async def pump() -> None:
            try:
                async for item in agen:
                    items.put((item, None))
            except BaseException as exc:
                items.put((_DONE, exc))
            else:
                items.put((_DONE, None))

        self.submit(pump())
        while True:
            item, error = items.get()
            if item is _DONE:
                if error is not None:
                    raise error
                return
            yield item

    async def aiterate(self, agen: AsyncIterator[T]) -> AsyncIterator[T]:
        """Consume an async generator that must run on this loop from another loop."""
        caller = asyncio.get_running_loop()
        items: asyncio.Queue = asyncio.Queue()

        def put(entry: tuple) -> None:
            try:
                caller.call_soon_threadsafe(items.put_nowait, entry)
            except RuntimeError:  # caller's loop already closed
                pass

        async def pump() -> None:
            try:
                async for item in agen:
                    put((item, None))
            except BaseException as exc:
                put((_DONE, exc))
            else:
                put((_DONE, None))

        self.submit(pump())
        while True:
            item, error = await items.get()
            if item is _DONE:
                if error is not None:
                    raise error
                return
            yield item

    def stop(self) -> None:
        """Stop the loop; it starts again on the next use."""
        with self._lock:
            loop, self._loop = self._loop, None
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
