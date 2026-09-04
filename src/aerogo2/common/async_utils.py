"""Async compatibility helpers for the Python 3.8 hardware runtime."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Callable, Tuple, TypeVar

T = TypeVar("T")


async def run_blocking(
    function: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run a blocking SDK call without requiring asyncio.to_thread (Python 3.9+)."""

    loop = asyncio.get_running_loop()
    operation = partial(function, *args, **kwargs)
    return await loop.run_in_executor(None, operation)


async def await_nonabandonable(task: asyncio.Future[T]) -> Tuple[T, bool]:
    """Wait through repeated caller cancellation without cancelling ``task``.

    The boolean records whether cancellation was requested.  A caller can
    finish any remaining safety commit and then propagate or deliberately
    consume cancellation at its own established boundary.
    """

    cancellation_seen = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancellation_seen = True
    return task.result(), cancellation_seen


__all__ = ["await_nonabandonable", "run_blocking"]
