"""Cross-cutting integration + regression suites for the engrava-0.6 surface.

These run against the real in-memory ``SqliteEngravaCore`` (the ``store``
fixture in :mod:`tests.conftest`), not fakes, so they exercise the whole MCP
edge surface end to end against real engrava 0.6 and lock the decisions that
were deliberately *deferred* for the 0.6 minor:

* **D3** — MCP writes still stamp the origin cycle (``created_cycle == 0``) and
  the server did NOT adopt ``max_cycle()`` write-stamping.
* **D4** — the concrete store's ``fts_match_failure_count`` diagnostic is NOT
  surfaced on the public ``memory_stats`` payload.

Suite 2 also pins that engrava 0.6's FTS query-normalization keeps the MCP
keyword-search surface robust: a syntactically odd FTS5 query is sanitized and
falls back inside engrava and never propagates as an error to the client.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
from engrava import EdgeType
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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from engrava import SqliteEngravaCore
    from mcp import ClientSession

#: A transaction-time "now" for the recency passthrough (valid ISO-8601).
RECENCY_NOW = "2026-07-20T00:00:00Z"

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

    async def test_d3_writes_stamp_origin_cycle_not_max_cycle(
        self, store: SqliteEngravaCore
    ) -> None:
        # D3 lock: MCP edge writes stamp the origin cycle, and the server did
        # not adopt max_cycle() write-stamping — so both stay 0.
        first = await store_thought_impl(store, essence="cycle source", content="source body")
        second = await store_thought_impl(store, essence="cycle target", content="target body")
        from_id = first["thought"]["thought_id"]
        to_id = second["thought"]["thought_id"]
        await link_thoughts_impl(store, from_id, to_id, EdgeType.DEPENDS_ON)

        out = await get_edges_impl(store, from_id, direction="OUT")
        assert out["edges"][0]["created_cycle"] == 0
        assert await store.max_cycle() == 0

    async def test_recency_now_passthrough_over_real_store(self, store: SqliteEngravaCore) -> None:
        # Transaction-time recency executes end to end and returns well-formed.
        result = await search_memory_impl(store, "coffee", recency_now=RECENCY_NOW)
        assert isinstance(result["results"], list)
        assert isinstance(result["backends_used"], list)


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
