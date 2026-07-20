"""Unit tests for the MCP edge-read tool implementations.

The ``get_edges`` and ``list_edges`` tools are exercised directly against a
store seeded with thoughts and a small edge graph carrying metadata (see the
``edge_store`` fixture below).  The filter-translation error path is driven
through :func:`~engrava_mcp.server._tool_errors` so the client-facing message
is asserted for real — in particular that it never leaks the internal JSONPath
grammar.
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
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session as connect_client

from engrava_mcp.server import (
    DEFAULT_EDGE_LIST_LIMIT,
    SERVER_NAME,
    StoreProvider,
    _tool_errors,
    get_edges_impl,
    list_edges_impl,
    register_tools,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mcp import ClientSession


@asynccontextmanager
async def _client_for(store: SqliteEngravaCore) -> AsyncIterator[ClientSession]:
    """Open a connected client whose tools query the given store.

    Registers the tools against a provider pointed at ``store`` and connects
    the in-process client so the real tool boundary runs end to end.

    Args:
        store: The seeded store the tools should query.

    Yields:
        A connected client session wired to ``store``.

    """
    server: FastMCP = FastMCP(SERVER_NAME)
    provider = StoreProvider()
    provider.set(store)
    register_tools(server, provider)
    async with connect_client(server) as client:
        yield client


def _error_text(content: object) -> str:
    """Extract the text of a tool error result's first content block.

    Args:
        content: The ``content`` sequence of a ``CallToolResult``.

    Returns:
        The ``text`` attribute of the first content block.

    """
    assert isinstance(content, list)
    assert content, "an error result must carry a content block"
    text = content[0].text  # type: ignore[union-attr]
    assert isinstance(text, str)
    return text


def _assert_grammar_free_filter_error(text: str) -> None:
    """Assert a filter error message is clean and leaks no JSONPath grammar.

    Args:
        text: The client-facing error message to inspect.

    """
    assert "metadata filter is invalid" in text.lower()
    # None of engrava's JSONPath grammar may reach the client: not the "$."/"$["
    # path syntax, not the accepted regex, not the "JSONPath" term itself.
    assert "$." not in text
    assert "$[" not in text
    assert "paths must match" not in text
    assert "^\\$" not in text
    assert "JSONPath" not in text


def _thought(thought_id: str) -> CoreThoughtRecord:
    """Build a minimal active thought to anchor edges.

    Args:
        thought_id: Stable identifier for the thought.

    Returns:
        A constructed ``CoreThoughtRecord``.

    """
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.BELIEF,
        essence=f"Essence for {thought_id}",
        content=f"Content body for {thought_id}.",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
    )


@pytest.fixture
async def edge_store() -> AsyncIterator[SqliteEngravaCore]:
    """Yield a store seeded with three thoughts and a small edge graph.

    Edges (all created at cycle 0):

    * ``e1`` t1 -> t2, ASSOCIATED, EXPERIENCE, metadata ``{topic: drinks, batch: 1}``
    * ``e2`` t2 -> t3, DEPENDS_ON, SEEDED_LLM, metadata ``{topic: food, batch: 2}``
    * ``e3`` t1 -> t3, ASSOCIATED, EXPERIENCE, metadata ``{topic: drinks, batch: 3}``

    Yields:
        A ``SqliteEngravaCore`` seeded for the edge-read tools.

    """
    connection = await aiosqlite.connect(":memory:")
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA foreign_keys=ON")
    backend = SqliteEngravaCore(connection)
    await backend.ensure_schema()

    for thought_id in ("t1", "t2", "t3"):
        await backend.create_thought(_thought(thought_id))

    edges = (
        EdgeRecord(
            edge_id="e1",
            from_thought_id="t1",
            to_thought_id="t2",
            edge_type=EdgeType.ASSOCIATED,
            weight=1.0,
            created_cycle=0,
            source=KnowledgeSource.EXPERIENCE,
            metadata={"topic": "drinks", "batch": 1},
        ),
        EdgeRecord(
            edge_id="e2",
            from_thought_id="t2",
            to_thought_id="t3",
            edge_type=EdgeType.DEPENDS_ON,
            weight=1.0,
            created_cycle=0,
            source=KnowledgeSource.SEEDED_LLM,
            metadata={"topic": "food", "batch": 2},
        ),
        EdgeRecord(
            edge_id="e3",
            from_thought_id="t1",
            to_thought_id="t3",
            edge_type=EdgeType.ASSOCIATED,
            weight=1.0,
            created_cycle=0,
            source=KnowledgeSource.EXPERIENCE,
            metadata={"topic": "drinks", "batch": 3},
        ),
    )
    for edge in edges:
        await backend.create_edge(edge)

    try:
        yield backend
    finally:
        await connection.close()


class TestGetEdges:
    """Tests for the ``get_edges`` tool."""

    async def test_out_direction_returns_outgoing_edges(
        self, edge_store: SqliteEngravaCore
    ) -> None:
        result = await get_edges_impl(edge_store, "t1", direction="OUT")
        ids = {edge["edge_id"] for edge in result["edges"]}
        assert ids == {"e1", "e3"}
        assert result["count"] == 2

    async def test_in_direction_returns_incoming_edges(self, edge_store: SqliteEngravaCore) -> None:
        result = await get_edges_impl(edge_store, "t3", direction="IN")
        ids = {edge["edge_id"] for edge in result["edges"]}
        assert ids == {"e2", "e3"}
        assert result["count"] == 2

    async def test_both_direction_returns_either_end(self, edge_store: SqliteEngravaCore) -> None:
        # t2 is the target of e1 and the source of e2, so BOTH returns both.
        result = await get_edges_impl(edge_store, "t2", direction="BOTH")
        ids = {edge["edge_id"] for edge in result["edges"]}
        assert ids == {"e1", "e2"}

    async def test_default_direction_is_both(self, edge_store: SqliteEngravaCore) -> None:
        default = await get_edges_impl(edge_store, "t2")
        both = await get_edges_impl(edge_store, "t2", direction="BOTH")
        assert {e["edge_id"] for e in default["edges"]} == {e["edge_id"] for e in both["edges"]}

    async def test_unknown_thought_returns_empty(self, edge_store: SqliteEngravaCore) -> None:
        result = await get_edges_impl(edge_store, "does-not-exist")
        assert result == {"edges": [], "count": 0}

    async def test_returned_edge_includes_metadata(self, edge_store: SqliteEngravaCore) -> None:
        result = await get_edges_impl(edge_store, "t1", direction="OUT")
        by_id = {edge["edge_id"]: edge for edge in result["edges"]}
        assert by_id["e1"]["metadata"] == {"topic": "drinks", "batch": 1}
        # Full edge records are returned, not a trimmed projection.
        assert by_id["e1"]["from_thought_id"] == "t1"
        assert by_id["e1"]["to_thought_id"] == "t2"
        assert by_id["e1"]["edge_type"] == "ASSOCIATED"


class TestListEdges:
    """Tests for the ``list_edges`` browse tool."""

    async def test_no_filter_returns_all(self, edge_store: SqliteEngravaCore) -> None:
        result = await list_edges_impl(edge_store)
        assert {edge["edge_id"] for edge in result["edges"]} == {"e1", "e2", "e3"}
        assert result["count"] == 3

    async def test_edge_type_filter(self, edge_store: SqliteEngravaCore) -> None:
        result = await list_edges_impl(edge_store, edge_type=EdgeType.ASSOCIATED)
        assert {edge["edge_id"] for edge in result["edges"]} == {"e1", "e3"}

    async def test_source_filter(self, edge_store: SqliteEngravaCore) -> None:
        result = await list_edges_impl(edge_store, source=KnowledgeSource.SEEDED_LLM)
        assert {edge["edge_id"] for edge in result["edges"]} == {"e2"}

    async def test_metadata_equals_matches_exact_value(self, edge_store: SqliteEngravaCore) -> None:
        result = await list_edges_impl(edge_store, metadata_equals={"topic": "drinks"})
        assert {edge["edge_id"] for edge in result["edges"]} == {"e1", "e3"}

    async def test_metadata_in_matches_any_listed_value(
        self, edge_store: SqliteEngravaCore
    ) -> None:
        result = await list_edges_impl(edge_store, metadata_in={"batch": [1, 3]})
        assert {edge["edge_id"] for edge in result["edges"]} == {"e1", "e3"}

    async def test_metadata_filters_combine_as_and(self, edge_store: SqliteEngravaCore) -> None:
        # topic=drinks (e1, e3) AND batch in [1] narrows to e1 only.
        result = await list_edges_impl(
            edge_store,
            metadata_equals={"topic": "drinks"},
            metadata_in={"batch": [1]},
        )
        assert {edge["edge_id"] for edge in result["edges"]} == {"e1"}

    async def test_limit_is_honoured(self, edge_store: SqliteEngravaCore) -> None:
        result = await list_edges_impl(edge_store, limit=1)
        assert result["count"] == 1
        assert len(result["edges"]) == 1

    async def test_returned_edges_include_metadata(self, edge_store: SqliteEngravaCore) -> None:
        result = await list_edges_impl(edge_store, edge_type=EdgeType.DEPENDS_ON)
        edge = result["edges"][0]
        assert edge["metadata"] == {"topic": "food", "batch": 2}

    async def test_invalid_metadata_key_maps_to_clean_error(
        self, edge_store: SqliteEngravaCore
    ) -> None:
        # A key that is not a simple field name produces an invalid JSONPath
        # internally; the client must get a clean message that never echoes the
        # JSONPath grammar (the accepted regex or the "$."/"$[0]" examples).
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                await list_edges_impl(edge_store, metadata_equals={"bad key!": "x"})
        text = str(excinfo.value)
        assert "metadata filter is invalid" in text.lower()
        # No JSONPath grammar leaks: neither the regex nor the "$." dollar syntax.
        assert "$" not in text
        assert "paths must match" not in text
        assert "^\\$" not in text
        assert "JSONPath" not in text

    async def test_invalid_metadata_value_maps_to_clean_error(
        self, edge_store: SqliteEngravaCore
    ) -> None:
        # A non-scalar value for an equality predicate is rejected by engrava;
        # the client-facing message stays grammar-free and names no internals.
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                await list_edges_impl(edge_store, metadata_equals={"topic": {"nested": 1}})  # type: ignore[dict-item]
        text = str(excinfo.value)
        assert "metadata filter is invalid" in text.lower()
        assert "FieldOp" not in text
        assert "$" not in text

    @pytest.mark.parametrize("smuggling_key", ["outer.inner", "tags[0]"])
    async def test_nested_metadata_key_is_rejected_on_equals(
        self, edge_store: SqliteEngravaCore, smuggling_key: str
    ) -> None:
        # A dotted/bracketed key would smuggle a nested JSONPath ($.outer.inner,
        # $.tags[0]) past the thin surface. It must be rejected at the boundary,
        # and the client-facing message must never echo the JSONPath grammar.
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                await list_edges_impl(edge_store, metadata_equals={smuggling_key: "x"})
        _assert_grammar_free_filter_error(str(excinfo.value))

    @pytest.mark.parametrize("smuggling_key", ["outer.inner", "tags[0]"])
    async def test_nested_metadata_key_is_rejected_on_in(
        self, edge_store: SqliteEngravaCore, smuggling_key: str
    ) -> None:
        # The same smuggling guard applies to metadata_in keys.
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                await list_edges_impl(edge_store, metadata_in={smuggling_key: ["x"]})
        _assert_grammar_free_filter_error(str(excinfo.value))

    async def test_nested_metadata_key_rejected_over_the_wire(
        self, edge_store: SqliteEngravaCore
    ) -> None:
        # End to end: a nested key rejected through the real tool boundary
        # surfaces the same clean, grammar-free ToolError.
        async with _client_for(edge_store) as client:
            result = await client.call_tool(
                "list_edges",
                {"metadata_equals": {"outer.inner": "x"}},
            )
        assert result.isError is True
        _assert_grammar_free_filter_error(_error_text(result.content))


class TestEdgeToolsOverTheWire:
    """The edge-read tools work through the real MCP boundary."""

    async def test_get_edges_over_the_wire(self, edge_store: SqliteEngravaCore) -> None:
        async with _client_for(edge_store) as client:
            result = await client.call_tool("get_edges", {"thought_id": "t1", "direction": "OUT"})
        assert result.isError is False
        assert result.structuredContent is not None
        assert {edge["edge_id"] for edge in result.structuredContent["edges"]} == {"e1", "e3"}

    async def test_list_edges_over_the_wire(self, edge_store: SqliteEngravaCore) -> None:
        async with _client_for(edge_store) as client:
            result = await client.call_tool(
                "list_edges",
                {"metadata_equals": {"topic": "drinks"}},
            )
        assert result.isError is False
        assert result.structuredContent is not None
        assert {edge["edge_id"] for edge in result.structuredContent["edges"]} == {"e1", "e3"}


def test_default_edge_list_limit_is_documented() -> None:
    # The chosen MCP default is deliberately smaller than engrava's own 5000.
    assert DEFAULT_EDGE_LIST_LIMIT == 100
