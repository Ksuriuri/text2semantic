"""Bounded async admission, one model owner, and short acoustic batch windows."""
from __future__ import annotations

import asyncio
import time


class MicrobatchQueue:
    def __init__(self, run_batch, *, max_batch_size=8, max_wait_ms=5,
                 capacity=64, frame_budget=6000):
        if min(max_batch_size, capacity, frame_budget) < 1 or max_wait_ms < 0:
            raise ValueError("Invalid batch queue limits")
        self.run_batch = run_batch
        self.max_batch_size = max_batch_size
        self.wait = max_wait_ms / 1000
        self.queue = asyncio.Queue(maxsize=capacity)
        self.frame_budget = frame_budget
        self.closed = False
        self.task = asyncio.create_task(self._run())
        self.last_batch_size = 0
        self.failed = None

    async def submit(self, payload):
        if self.closed or self.failed:
            raise RuntimeError("Acoustic worker unavailable")
        frames = int(payload["codes"].numel())
        if frames > self.frame_budget:
            raise ValueError("Request exceeds acoustic frame budget")
        future = asyncio.get_running_loop().create_future()
        self.queue.put_nowait((payload, future, time.monotonic(), frames))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # A GPU thread cannot be killed. Keep request files alive until it
            # has finished reading them, even after the client disconnects.
            try:
                await asyncio.shield(future)
            except Exception:
                pass
            raise

    def _execute(self, payloads):
        try:
            return self.run_batch(payloads)
        except Exception as error:
            # Split failed batches so a malformed reference or OOM does not
            # fail unrelated requests. Never retry a device-context failure.
            if "device-side assert" in str(error) or "illegal memory access" in str(error):
                self.failed = error
                return [error] * len(payloads)
            if len(payloads) == 1:
                return [error]
            middle = len(payloads) // 2
            return self._execute(payloads[:middle]) + self._execute(payloads[middle:])

    async def _run(self):
        pending = None
        try:
            while True:
                first = pending if pending is not None else await self.queue.get()
                pending = None
                if first is None:
                    return
                items, frames = [first], first[3]
                deadline = asyncio.get_running_loop().time() + self.wait
                while len(items) < self.max_batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        entry = await asyncio.wait_for(self.queue.get(), remaining)
                    except asyncio.TimeoutError:
                        break
                    if entry is None:
                        pending = None
                        self.closed = True
                        break
                    if frames + entry[3] > self.frame_budget:
                        pending = entry
                        break
                    items.append(entry)
                    frames += entry[3]
                live = [entry for entry in items if not entry[1].cancelled()]
                if live:
                    start = time.monotonic()
                    try:
                        results = await asyncio.to_thread(self._execute, [e[0] for e in live])
                        if len(results) != len(live):
                            raise RuntimeError("Acoustic result count mismatch")
                        self.last_batch_size = len(live)
                        for (_, future, arrived, _), result in zip(live, results):
                            if not future.done():
                                if isinstance(result, Exception):
                                    future.set_exception(result)
                                else:
                                    wave, info = result
                                    future.set_result((wave, dict(info, acoustic_queue_ms=(start-arrived)*1000)))
                    except Exception as error:
                        for _, future, _, _ in live:
                            if not future.done():
                                future.set_exception(error)
                if self.closed and pending is None and self.queue.empty():
                    return
        except BaseException as error:
            self.failed = error
            if pending is not None and not pending[1].done():
                pending[1].set_exception(RuntimeError("Acoustic worker stopped"))
            while not self.queue.empty():
                entry = self.queue.get_nowait()
                if entry is not None and not entry[1].done():
                    entry[1].set_exception(RuntimeError("Acoustic worker stopped"))
            raise

    async def close(self):
        if not self.closed:
            self.closed = True
            await self.queue.put(None)
        await self.task
