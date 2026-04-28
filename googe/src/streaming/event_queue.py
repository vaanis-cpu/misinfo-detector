"""Async event queue for claim streaming."""

import asyncio
from typing import Any, Optional, AsyncIterator
from collections import deque

from ..config import get_config


class EventQueue:
    """Async event queue for processing claims.

    This is a simplified in-memory queue. In production, this would be
    replaced with Kafka or another distributed queue.
    """

    def __init__(self, maxsize: int = 0):
        """Initialize event queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited).
        """
        cfg = get_config()
        self._maxsize = maxsize or cfg.streaming.queue_size
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._closed = False

    async def put(self, item: Any) -> None:
        """Add an item to the queue.

        Args:
            item: Item to add.

        Raises:
            asyncio.QueueFull: If queue is full.
        """
        if self._closed:
            raise RuntimeError("Queue is closed")
        await self._queue.put(item)

    async def get(self) -> Any:
        """Get an item from the queue.

        Returns:
            The next item from the queue.

        Raises:
            asyncio.queues.QueueEmpty: If queue is empty.
        """
        return await self._queue.get()

    async def get_nowait(self) -> Optional[Any]:
        """Get an item without waiting.

        Returns:
            Item if available, None otherwise.
        """
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def iterate(self) -> AsyncIterator[Any]:
        """Iterate over items in the queue asynchronously.

        Yields:
            Items from the queue until closed.
        """
        while not self._closed:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield item
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def qsize(self) -> int:
        """Return approximate queue size."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Return True if queue is empty."""
        return self._queue.empty()

    def full(self) -> bool:
        """Return True if queue is full."""
        return self._queue.full()

    async def close(self) -> None:
        """Close the queue."""
        self._closed = True

    async def join(self) -> None:
        """Wait until all items are processed."""
        await self._queue.join()

    def task_done(self) -> None:
        """Mark a task as complete."""
        self._queue.task_done()


class EventBus:
    """Simple pub/sub event bus for internal communication."""

    def __init__(self):
        self._subscribers: dict = {}
        self._lock = asyncio.Lock()

    def subscribe(self, topic: str, callback) -> None:
        """Subscribe to a topic.

        Args:
            topic: Topic name.
            callback: Async callback function.
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback) -> None:
        """Unsubscribe from a topic."""
        if topic in self._subscribers:
            self._subscribers[topic].remove(callback)

    async def publish(self, topic: str, data: Any) -> None:
        """Publish data to a topic.

        Args:
            topic: Topic name.
            data: Data to publish.
        """
        async with self._lock:
            if topic in self._subscribers:
                for callback in self._subscribers[topic]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
                    except Exception:
                        pass

    def clear(self, topic: str = None) -> None:
        """Clear subscribers for a topic or all topics."""
        if topic:
            self._subscribers.pop(topic, None)
        else:
            self._subscribers.clear()
