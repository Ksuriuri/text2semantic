"""Cancellation-safe offload for code that uses request-owned files/tensors."""
import asyncio


async def offload(function, *args, **kwargs):
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Python cannot stop a running worker thread. Finish ownership cleanup
        # before callers remove reference files or release their model lock.
        try:
            await asyncio.shield(task)
        except Exception:
            pass
        raise
