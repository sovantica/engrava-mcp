"""Cross-cutting integration + regression suites for the engrava-0.6 surface.

These run against the real in-memory ``SqliteEngravaCore`` (the ``store``
fixture in :mod:`tests.conftest`), not fakes, so they exercise the whole MCP
edge surface end to end against real engrava 0.6 and lock the decisions that
were deliberately *deferred* for the 0.6 minor:

* **D3** — MCP writes still stamp the origin cycle (``created_cycle == 0``) and
  the server did NOT adopt ``max_cycle()`` write-stamping.
* **D4** — the concrete store's ``fts_match_failure_count`` diagnostic is NOT
  surfaced on the public ``memory_stats`` payload.

A third suite observes that ``search_memory``'s ``recency_now`` genuinely
reaches engrava's ranker: it runs a control search over a corpus whose only
differing scored field is the transaction timestamp, then the same search with
``recency_now``, and asserts the ranking moves.

Suite 2 also pins that engrava 0.6's FTS query-normalization keeps the MCP
keyword-search surface robust: a syntactically odd FTS5 query is sanitized and
falls back inside engrava and never propagates as an error to the client.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import aiosqlite
import pytest
from engrava import (
    CoreThoughtRecord,
    EdgeType,
    LifecycleStatus,
    Priority,
    SearchConfig,
    SqliteEngravaCore,
    ThoughtType,
)
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session as connect_client

from engrava_mcp.server import (
    SERVER_NAME,
    StoreProvider,
    _tool_errors,
    get_edges_impl,
    link_thoughts_impl,
    list_edges_impl,
    memory_stats_impl,
    register_tools,
    search_keywords_impl,
    search_memory_impl,
    store_thought_impl,
)
from tests.recency_corpus import (
    RECENCY_EXPECTED_ORDER,
    RECENCY_NOW,
    RECENCY_QUERY,
    RECENCY_THOUGHT_IDS,
    seed_recency_corpus,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mcp import ClientSession

#: The cycle MCP writes stamp: the origin, not the store's high-water mark.
#: A literal rather than the server's own constant, so the expectation is
#: independent of the value the production code happens to hold.
ORIGIN_CYCLE = 0

#: A cycle well above the origin, seeded to raise the store's high-water mark
#: so ``max_cycle()`` and the origin cycle are distinguishable numbers.
HIGH_WATER_CYCLE = 7

#: Syntactically odd but VERIFIED-clean FTS inputs: each is sanitized/fallback
#: -handled inside engrava 0.6 and returns cleanly (0 hits) rather than raising.
#: engrava logs an internal warning and falls back — that log is expected and is
#: deliberately NOT asserted on; only the clean return is.
RISKY_FTS_QUERIES = [
    '"',
    "AND",
    "foo AND",
    "a OR OR b",
    "NOT",
    "((",
    "foo*",
    '"unbalanced',
    ":",
    "a NEAR/",
    "*",
    "café",
]


@pytest.fixture
async def recency_store() -> AsyncIterator[SqliteEngravaCore]:
    """Yield a store under engrava's default search policy, seeded for recency.

    Deliberately not the shared ``store`` fixture.  That fixture builds its
    store with no ``SearchConfig``, which resolves the recency fusion weight to
    ``0.0`` — the very condition that makes a recency assertion undetectable —
    and it seeds unrelated thoughts that would rank alongside the corpus.
    Changing it would alter what the rest of the suite exercises for the benefit
    of one test, so the dedicated fixture is the narrower change.

    Yields:
        A ``SqliteEngravaCore`` holding the recency corpus.

    """
    connection = await aiosqlite.connect(":memory:")
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA foreign_keys=ON")
    backend = SqliteEngravaCore(connection, search_config=SearchConfig())
    await backend.ensure_schema()
    await seed_recency_corpus(backend)

    try:
        yield backend
    finally:
        await connection.close()


async def _raise_high_water(store: SqliteEngravaCore) -> None:
    """Seed one thought at :data:`HIGH_WATER_CYCLE` to lift ``max_cycle()``.

    Seeded here rather than in the shared ``store`` fixture: the fixture backs
    the rest of the suite, and moving its cycles would change what every other
    test exercises for the benefit of two.

    Args:
        store: The store whose high-water cycle should be raised.

    """
    await store.create_thought(
        CoreThoughtRecord(
            thought_id="thought-high-water",
            thought_type=ThoughtType.BELIEF,
            essence="high water mark",
            content="a thought written on a later cognitive cycle",
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=HIGH_WATER_CYCLE,
            updated_cycle=HIGH_WATER_CYCLE,
            source="test",
        )
    )


@asynccontextmanager
async def _client_for(store: SqliteEngravaCore) -> AsyncIterator[ClientSession]:
    """Open a connected client whose tools query the given store.

    Args:
        store: The seeded store the tools should query.

    Yields:
        A connected client session wired to ``store`` through the real tool
        boundary (so FastMCP's error wrapping and ``_tool_errors`` run).

    """
    server: FastMCP = FastMCP(SERVER_NAME)
    provider = StoreProvider()
    provider.set(store)
    register_tools(server, provider)
    async with connect_client(server) as client:
        yield client


class TestEdgeRoundTripIntegration:
    """End-to-end edge surface against the real store, with the D3 lock."""

    async def test_link_filter_traverse_round_trip(self, store: SqliteEngravaCore) -> None:
        first = await store_thought_impl(store, essence="link source", content="source body")
        second = await store_thought_impl(store, essence="link target", content="target body")
        from_id = first["thought"]["thought_id"]
        to_id = second["thought"]["thought_id"]

        linked = await link_thoughts_impl(
            store,
            from_id,
            to_id,
            EdgeType.ASSOCIATED,
            metadata={"session": "s1", "kind": "ref"},
        )
        edge_id = linked["edge"]["edge_id"]
        assert linked["edge"]["metadata"] == {"session": "s1", "kind": "ref"}

        # JSON-friendly metadata filter -> MetadataFilter, against a real store.
        match = await list_edges_impl(store, metadata_equals={"session": "s1"})
        assert [edge["edge_id"] for edge in match["edges"]] == [edge_id]
        miss = await list_edges_impl(store, metadata_equals={"session": "other"})
        assert miss["edges"] == []
        assert miss["count"] == 0

        # Direction-aware traversal; the returned edge carries the set metadata.
        # Source node: OUT returns the edge, IN returns none.
        out = await get_edges_impl(store, from_id, direction="OUT")
        assert [edge["edge_id"] for edge in out["edges"]] == [edge_id]
        assert out["edges"][0]["metadata"] == {"session": "s1", "kind": "ref"}
        source_in = await get_edges_impl(store, from_id, direction="IN")
        assert source_in["edges"] == []
        both = await get_edges_impl(store, from_id, direction="BOTH")
        assert [edge["edge_id"] for edge in both["edges"]] == [edge_id]

        # Target node: the inverse — IN returns the edge, OUT returns none —
        # so direction is genuinely honored both ways, not just on the source.
        target_in = await get_edges_impl(store, to_id, direction="IN")
        assert [edge["edge_id"] for edge in target_in["edges"]] == [edge_id]
        target_out = await get_edges_impl(store, to_id, direction="OUT")
        assert target_out["edges"] == []

    async def test_d3_thought_writes_stamp_origin_cycle_not_max_cycle(
        self, store: SqliteEngravaCore
    ) -> None:
        # D3 lock, thought path: an MCP write stamps the origin cycle and the
        # server did NOT adopt max_cycle() write-stamping.
        #
        # The high-water seed is what makes this discriminating. The shared
        # fixture seeds everything at cycle 0, so both behaviours would produce
        # the same 0 and the test could not fail. Raising the store's high-water
        # mark first separates them: origin-stamping keeps 0, max_cycle()
        # -stamping would write HIGH_WATER_CYCLE.
        await _raise_high_water(store)
        assert await store.max_cycle() == HIGH_WATER_CYCLE

        created = await store_thought_impl(store, essence="cycle source", content="source body")
        stored = await store.get_thought(created["thought"]["thought_id"])
        assert stored is not None
        assert stored.created_cycle == ORIGIN_CYCLE
        assert stored.updated_cycle == ORIGIN_CYCLE

    async def test_d3_edge_writes_stamp_origin_cycle_not_max_cycle(
        self, store: SqliteEngravaCore
    ) -> None:
        # D3 lock, edge path: the same, for the edge write. Separate from the
        # thought path so a mutation of either is isolated — pytest halts at
        # the first failing assertion, so one test covering both would leave
        # the second path unobserved whenever the first reddens.
        await _raise_high_water(store)
        assert await store.max_cycle() == HIGH_WATER_CYCLE

        first = await store_thought_impl(store, essence="cycle source", content="source body")
        second = await store_thought_impl(store, essence="cycle target", content="target body")
        from_id = first["thought"]["thought_id"]
        to_id = second["thought"]["thought_id"]
        await link_thoughts_impl(store, from_id, to_id, EdgeType.DEPENDS_ON)

        out = await get_edges_impl(store, from_id, direction="OUT")
        assert out["edges"][0]["created_cycle"] == ORIGIN_CYCLE


class TestRecencyPassthrough:
    """``recency_now`` reaches engrava's ranker and changes what comes back."""

    async def test_recency_now_reorders_over_a_real_store(
        self, recency_store: SqliteEngravaCore
    ) -> None:
        # A well-formed response is not evidence: the tool would return one just
        # the same if recency_now were dropped on the way through. What
        # discriminates is the corpus moving.
        #
        # The control runs first and is load-bearing — it pins that the corpus
        # carries no latent ordering that would produce the ranked order below
        # on its own.
        control = await search_memory_impl(recency_store, RECENCY_QUERY)
        assert "recency" not in control["backends_used"]
        control_order = [entry["thought_id"] for entry in control["results"]]
        assert sorted(control_order) == sorted(RECENCY_THOUGHT_IDS)
        assert len({entry["score"] for entry in control["results"]}) == 1
        assert control_order != RECENCY_EXPECTED_ORDER

        ranked = await search_memory_impl(recency_store, RECENCY_QUERY, recency_now=RECENCY_NOW)
        assert "recency" in ranked["backends_used"]
        assert [entry["thought_id"] for entry in ranked["results"]] == RECENCY_EXPECTED_ORDER


class TestFtsNormalizationRegression:
    """engrava-0.6 FTS normalization keeps keyword search robust (+ D4 lock)."""

    async def test_positive_control_finds_a_real_hit(self, store: SqliteEngravaCore) -> None:
        # Positive control: keyword search actually finds a stored term. Without
        # this, the risky-input tests below would also pass if search silently
        # returned nothing — this pins "functional", they pin "robust".
        stored = await store_thought_impl(
            store,
            essence="contains zqxwv-unique-term marker",
            content="body mentioning zqxwv-unique-term again",
        )
        thought_id = stored["thought"]["thought_id"]
        result = await search_keywords_impl(store, "zqxwv-unique-term")
        assert thought_id in [entry["thought_id"] for entry in result["results"]]

    @pytest.mark.parametrize("query", RISKY_FTS_QUERIES)
    async def test_risky_fts_query_returns_clean_dict(
        self, store: SqliteEngravaCore, query: str
    ) -> None:
        # Each odd query is sanitized/fallback-handled inside engrava and must
        # return a well-formed payload rather than raise.
        result = await search_keywords_impl(store, query)
        assert isinstance(result, dict)
        assert isinstance(result["results"], list)

    async def test_risky_fts_query_over_the_tool_boundary_does_not_error(
        self, store: SqliteEngravaCore
    ) -> None:
        # Through the real MCP boundary (which runs _tool_errors), a
        # syntactically-odd-but-handled query surfaces no ToolError.
        async with _client_for(store) as client:
            result = await client.call_tool("search_keywords", {"query": "foo AND"})
        assert result.isError is False
        assert result.structuredContent is not None
        assert isinstance(result.structuredContent["results"], list)

    async def test_risky_fts_query_through_tool_errors_guard(
        self, store: SqliteEngravaCore
    ) -> None:
        # The guard itself lets a handled odd query through unchanged.
        async with _tool_errors():
            result = await search_keywords_impl(store, '"unbalanced')
        assert isinstance(result["results"], list)

    async def test_d4_stats_omit_fts_match_failure_count(self, store: SqliteEngravaCore) -> None:
        # D4 lock: the public stats surface exposes exactly these metric groups
        # and never the concrete store's fts_match_failure_count diagnostic.
        stats = await memory_stats_impl(store)
        assert set(stats["metrics"]) == {"thoughts", "edges", "storage_total_bytes"}
        assert "fts_match_failure_count" not in json.dumps(stats)
