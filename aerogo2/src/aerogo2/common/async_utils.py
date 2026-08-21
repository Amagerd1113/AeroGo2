"""Async compatibility helpers for the Python 3.8 hardware runtime."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Callable, TypeVar

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


__all__ = ["run_blocking"]
