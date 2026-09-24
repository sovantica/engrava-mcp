"""Unit tests for :mod:`engrava_mcp.read_only`.

Two questions are kept deliberately separate:

* **Correctness** — does a read through :class:`~engrava_mcp.read_only.ReadOnlyStore`
  return what the same call against the underlying store returns?
  ``TestReadsMatchTheUnderlyingStore`` answers this against a tracking-*disabled*
  store, covering all eight tool-level entry points (nine store calls — ``memory_stats``
  makes two), so nothing about access counting can confound the comparison.
* **The guarantee the wrapper exists for** — does a read through it leave
  ``access_count`` untouched, even after an explicit flush, on a store where a plain
  read would buffer an update? ``TestAccessTrackingIsSuppressed`` answers this against a
  store built with ``access_tracking_enabled=True``, reading the counter back with raw
  SQL so no store method's own read can confound the measurement.

  Only two of the nine methods actually buffer an access — ``get_thought`` and
  ``search_hybrid`` — verified by reading
  :meth:`~engrava.SqliteEngravaCore._buffer_accesses`'s call sites in the installed
  library rather than assumed; the other seven (``list_thoughts``, ``search_fts``,
  ``metrics``, ``get_edges``, ``list_edges``, ``count_thoughts``, ``execute_mindql``)
  never call it. Each of those two therefore gets its own view/raw-store pair here, with
  the raw-store case as the positive control that proves the counter *would* have moved.
  A behavioural test for one of the other seven would never go red under any suppression
  mutation (there is nothing to suppress), so it would not be evidence of anything; those
  seven are guarded by :class:`~engrava_mcp.read_only.ReadOnlyMcpStore` instead — a
  method missing from :class:`~engrava_mcp.read_only.ReadOnlyStore` is a type error,
  and their correctness is covered in ``TestReadsMatchTheUnderlyingStore``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import aiosqlite
import pytest
from engrava import (
    CoreThoughtRecord,
    EdgeRecord,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
)

from engrava_mcp.read_only import ReadOnlyStore
from engrava_mcp.server import (
    get_edges_impl,
    get_thought_impl,
    list_edges_impl,
    list_memory_impl,
    memory_stats_impl,
    query_memory_impl,
    search_keywords_impl,
    search_memory_impl,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _thought(
    thought_id: str,
    *,
    essence: str,
    content: str,
    priority: Priority = Priority.P2,
) -> CoreThoughtRecord:
    """Build a core thought record for seeding.

    Args:
        thought_id: Stable identifier for the thought.
        essence: Compact canonical text.
        content: Full stored content.
        priority: Priority level.

    Returns:
        A constructed ``CoreThoughtRecord``, active by default.

    """
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.BELIEF,
        essence=essence,
        content=content,
        priority=priority,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
    )


@pytest.fixture
async def wrapped() -> AsyncIterator[tuple[SqliteEngravaCore, ReadOnlyStore]]:
    """Yield a tracking-disabled store seeded for correctness checks, and its view.

    Seeds two thoughts (distinct keywords, distinct priorities, so search and
    filtering have something to discriminate) and one edge between them.

    Yields:
        The raw backend and a :class:`ReadOnlyStore` wrapping it.

    """
    connection = await aiosqlite.connect(":memory:")
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA foreign_keys=ON")
    backend = SqliteEngravaCore(connection)
    await backend.ensure_schema()

    await backend.create_thought(
        _thought(
            "ro-alpha",
            essence="Coffee brewing notes",
            content="Pour-over coffee extracts best between 90 and 96 degrees.",
        )
    )
    await backend.create_thought(
        _thought(
            "ro-beta",
            essence="Tea steeping notes",
            content="Green tea steeps best below boiling to avoid bitterness.",
            priority=Priority.P1,
        )
    )
    await backend.create_edge(
        EdgeRecord(
            edge_id="ro-e1",
            from_thought_id="ro-alpha",
            to_thought_id="ro-beta",
            edge_type=EdgeType.ASSOCIATED,
            weight=1.0,
            created_cycle=0,
            source=KnowledgeSource.EXPERIENCE,
            metadata={"topic": "drinks"},
        )
    )

    try:
        yield backend, ReadOnlyStore(backend)
    finally:
        await connection.close()


class TestReadsMatchTheUnderlyingStore:
    """Every read through the view returns what the same call on the raw store returns.

    Covers all eight tool-level entry points — nine store calls, since ``memory_stats``
    makes two (``count_thoughts`` and ``metrics``) — against a tracking-disabled store,
    so nothing about access counting can confound the comparison.
    """

    async def test_get_thought(self, wrapped: tuple[SqliteEngravaCore, ReadOnlyStore]) -> None:
        _backend, view = wrapped
        result = await get_thought_impl(view, "ro-alpha")
        assert result["found"] is True
        assert result["thought"]["essence"] == "Coffee brewing notes"

    async def test_list_memory(self, wrapped: tuple[SqliteEngravaCore, ReadOnlyStore]) -> None:
        _backend, view = wrapped
        result = await list_memory_impl(view)
        assert result["count"] == 2
        assert {t["thought_id"] for t in result["thoughts"]} == {"ro-alpha", "ro-beta"}

    async def test_search_keywords(self, wrapped: tuple[SqliteEngravaCore, ReadOnlyStore]) -> None:
        _backend, view = wrapped
        result = await search_keywords_impl(view, "coffee")
        assert [r["thought_id"] for r in result["results"]] == ["ro-alpha"]

    async def test_search_memory(self, wrapped: tuple[SqliteEngravaCore, ReadOnlyStore]) -> None:
        _backend, view = wrapped
        result = await search_memory_impl(view, "tea")
        assert "ro-beta" in {r["thought_id"] for r in result["results"]}

    async def test_get_edges(self, wrapped: tuple[SqliteEngravaCore, ReadOnlyStore]) -> None:
        _backend, view = wrapped
        result = await get_edges_impl(view, "ro-alpha", direction="OUT")
        assert result["count"] == 1
        assert result["edges"][0]["edge_id"] == "ro-e1"

    async def test_list_edges(self, wrapped: tuple[SqliteEngravaCore, ReadOnlyStore]) -> None:
        _backend, view = wrapped
        result = await list_edges_impl(view)
        assert result["count"] == 1
        assert result["edges"][0]["edge_id"] == "ro-e1"

    async def test_memory_stats(self, wrapped: tuple[SqliteEngravaCore, ReadOnlyStore]) -> None:
        _backend, view = wrapped
        result = await memory_stats_impl(view)
        assert result["thought_count"] == 2
        assert result["metrics"]["thoughts"]["total"] == 2
        assert result["metrics"]["edges"]["total"] == 1

    async def test_query_memory(self, wrapped: tuple[SqliteEngravaCore, ReadOnlyStore]) -> None:
        _backend, view = wrapped
        result = await query_memory_impl(
            view,
            "FIND thoughts WHERE priority = 'P1'",
        )
        ids = {row["thought_id"] for row in result["rows"]}
        assert ids == {"ro-beta"}


@pytest.fixture
async def tracked() -> AsyncIterator[tuple[aiosqlite.Connection, SqliteEngravaCore]]:
    """Yield a store with access tracking on, seeded with one thought.

    The seeded content matches the query text ``TestAccessTrackingIsSuppressed`` uses
    for its ``search_hybrid`` cases, so the same fixture serves both the ``get_thought``
    and the ``search_hybrid`` pairs.

    Yields:
        The raw connection — used to read ``access_count`` back with plain SQL so no
        store method's own read can confound the measurement — and the store built
        over it.

    """
    connection = await aiosqlite.connect(":memory:")
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA foreign_keys=ON")
    backend = SqliteEngravaCore(connection, access_tracking_enabled=True)
    await backend.ensure_schema()
    await backend.create_thought(
        _thought(
            "tracked-1",
            essence="Coffee brewing notes",
            content="Pour-over coffee extracts best between 90 and 96 degrees.",
        )
    )
    try:
        yield connection, backend
    finally:
        await connection.close()


async def _access_count(connection: aiosqlite.Connection, thought_id: str) -> int:
    """Read ``access_count`` straight out of SQLite.

    Args:
        connection: The live connection to query.
        thought_id: Identifier of the thought to inspect.

    Returns:
        The persisted ``access_count`` value.

    """
    cursor = await connection.execute(
        "SELECT access_count FROM thought WHERE thought_id = ?",
        (thought_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


class TestAccessTrackingIsSuppressed:
    """``access_count`` stays untouched through the view; not through a plain read.

    One view/raw-store pair per method that actually buffers an access —
    ``get_thought`` and ``search_hybrid`` (via ``search_memory_impl``), see the module
    docstring for how that set was established. Each "through the view" test is the one
    the acceptance criteria call for: it asserts on the persisted counter, not on which
    tools are registered, so a fix that merely stops registering the write tools (the
    original defect) fails it even though every read still "succeeds" and returns
    correct data. Each "raw store" test is the positive control proving the counter
    *would* have moved for that exact call shape.
    """

    async def test_get_thought_through_the_view_leaves_the_counter_unchanged(
        self, tracked: tuple[aiosqlite.Connection, SqliteEngravaCore]
    ) -> None:
        connection, backend = tracked
        view = ReadOnlyStore(backend)
        for _ in range(3):
            await get_thought_impl(view, "tracked-1")
        # An explicit flush forces any buffered update to materialise; if suppression
        # had failed silently (a wrap that does not actually suppress), this is what
        # would expose it.
        await backend.flush_access_buffer()
        assert await _access_count(connection, "tracked-1") == 0

    async def test_get_thought_on_the_raw_store_raises_the_counter(
        self, tracked: tuple[aiosqlite.Connection, SqliteEngravaCore]
    ) -> None:
        connection, backend = tracked
        for _ in range(3):
            await get_thought_impl(backend, "tracked-1")
        await backend.flush_access_buffer()
        assert await _access_count(connection, "tracked-1") == 3

    async def test_search_hybrid_through_the_view_leaves_the_counter_unchanged(
        self, tracked: tuple[aiosqlite.Connection, SqliteEngravaCore]
    ) -> None:
        connection, backend = tracked
        view = ReadOnlyStore(backend)
        for _ in range(3):
            result = await search_memory_impl(view, "coffee")
            assert result["results"], "the search must actually hit the seeded thought"
        await backend.flush_access_buffer()
        assert await _access_count(connection, "tracked-1") == 0

    async def test_search_hybrid_on_the_raw_store_raises_the_counter(
        self, tracked: tuple[aiosqlite.Connection, SqliteEngravaCore]
    ) -> None:
        connection, backend = tracked
        for _ in range(3):
            result = await search_memory_impl(backend, "coffee")
            assert result["results"], "the search must actually hit the seeded thought"
        await backend.flush_access_buffer()
        assert await _access_count(connection, "tracked-1") == 3


class TestGetEdgesLimitThroughTheView:
    """``ReadOnlyStore.get_edges`` forwards ``limit`` and runs it under suppression.

    Both the inner ``get_edges`` and the inner ``suppress_access_tracking`` are spied
    on so the order of events is observable, not just the two facts independently: the
    forwarded ``limit`` alone would not show the inner ``get_edges`` call actually
    happened *inside* the suppression window rather than before or after it.
    """

    async def test_forwards_limit_and_calls_the_inner_store_while_suppressed(
        self,
        wrapped: tuple[SqliteEngravaCore, ReadOnlyStore],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend, view = wrapped
        events: list[str] = []
        calls: list[int | None] = []

        real_get_edges = backend.get_edges
        real_suppress = backend.suppress_access_tracking

        async def _spy_get_edges(
            thought_id: str,
            *,
            direction: str = "BOTH",
            limit: int | None = None,
        ) -> list[EdgeRecord]:
            events.append("get_edges")
            calls.append(limit)
            return await real_get_edges(thought_id, direction=direction, limit=limit)

        @asynccontextmanager
        async def _spy_suppress() -> AsyncIterator[None]:
            events.append("enter")
            async with real_suppress():
                yield
            events.append("exit")

        monkeypatch.setattr(backend, "get_edges", _spy_get_edges)
        monkeypatch.setattr(backend, "suppress_access_tracking", _spy_suppress)

        result = await view.get_edges("ro-alpha", direction="OUT", limit=1)

        assert calls == [1]
        assert events == ["enter", "get_edges", "exit"]
        assert [edge.edge_id for edge in result] == ["ro-e1"]
