"""Domain validation of every wire-supplied numeric bound.

The MCP protocol layer validates an argument's *type*, not its *domain*: ``-1``
is a valid ``int``, and SQLite reads ``LIMIT -1`` as "no limit", so an
unvalidated negative bound silently defeats the scan cap it was meant to
impose.  A huge value is unbounded in effect for the same reason, so both ends
are constrained.

These tests cover every bound the server accepts over the wire
(``search_memory.top_k``, ``search_keywords.top_k``, ``list_memory.limit`` /
``.offset``, ``list_edges.limit``, ``query_memory.limit``, and the
``summarize_recent_memory`` prompt's ``limit``) at the implementation level and
at the real MCP client boundary.

The tests that matter most are in
:class:`TestOutOfRangeBoundsAreRejectedAndPagingStaysExact`: each pairs a
deletion-sensitive rejection (remove the domain guard and the negative bound is
accepted, so the test fails) with an **exact** row count taken against a store
holding strictly more rows, so neither half can pass vacuously.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
from engrava import EdgeType
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session as connect_client

import engrava_mcp.server as server_module
from engrava_mcp.server import (
    MAX_PAGE_LIMIT,
    MAX_TOP_K,
    SERVER_NAME,
    OutOfRangeBoundError,
    StoreProvider,
    _tool_errors,
    link_thoughts_impl,
    list_edges_impl,
    list_memory_impl,
    query_memory_impl,
    recent_thoughts_impl,
    register_prompts,
    register_tools,
    search_keywords_impl,
    search_memory_impl,
    store_thought_impl,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from engrava import SqliteEngravaCore
    from mcp import ClientSession

#: A FIND query matching every thought the ``bulk_store`` fixture seeds.
FIND_ALL = "FIND thoughts WHERE lifecycle_status = 'CREATED'"

#: How many thoughts ``bulk_store`` seeds.  Larger than any page size these
#: tests request, so a bounded call is observably shorter than the full set.
SEEDED_COUNT = 30

#: Every thought ``bulk_store`` holds: the two the shared ``store`` fixture
#: seeds plus :data:`SEEDED_COUNT`.  Asserted exactly so a paging test cannot
#: pass against an unexpectedly empty or truncated store.
TOTAL_THOUGHTS = SEEDED_COUNT + 2

#: How many thoughts match the term the bulk rows share.  The two fixture
#: thoughts are about coffee and tea, so only the bulk rows match.
WIDGET_MATCHES = SEEDED_COUNT

#: How many rows :data:`FIND_ALL` returns: newly stored thoughts start in the
#: ``CREATED`` lifecycle state, while the two fixture thoughts are ``ACTIVE``.
CREATED_ROWS = SEEDED_COUNT

#: Values that must be rejected for a page ``limit`` / ``top_k``: zero and
#: negatives defeat the cap outright, and an excessive value is unbounded in
#: effect.
OUT_OF_RANGE_LIMITS = [-1, 0, MAX_PAGE_LIMIT + 1, 10**18]

#: The same, for the ranked-window bound.
OUT_OF_RANGE_TOP_K = [-1, 0, MAX_TOP_K + 1, 10**18]


@asynccontextmanager
async def _client_for(store: SqliteEngravaCore) -> AsyncIterator[ClientSession]:
    """Open a connected client whose tools and prompts query the given store.

    Args:
        store: The seeded store the surface should query.

    Yields:
        A connected client session wired to ``store`` through the real
        boundary, so pydantic's schema validation and ``_tool_errors`` both run.

    """
    server: FastMCP = FastMCP(SERVER_NAME)
    provider = StoreProvider()
    provider.set(store)
    register_tools(server, provider)
    register_prompts(server, provider)
    async with connect_client(server) as client:
        yield client


@pytest.fixture
async def bulk_store(store: SqliteEngravaCore) -> SqliteEngravaCore:
    """Seed the shared store with enough thoughts to observe a page cap.

    Args:
        store: The shared in-memory store fixture.

    Returns:
        The same store, carrying :data:`SEEDED_COUNT` extra thoughts that all
        share a common search term.

    """
    for index in range(SEEDED_COUNT):
        await store_thought_impl(
            store,
            essence=f"bulk thought {index} widget",
            content=f"bulk body {index} widget content",
        )
    return store


class TestOutOfRangeBoundsAreRejectedAndPagingStaysExact:
    """Out-of-range bounds are rejected, and in-range paging is exactly capped.

    Each test pairs two assertions:

    * the ``pytest.raises`` half is *deletion-sensitive* — remove the domain
      guard and the negative bound is accepted, so the test fails;
    * the paging half asserts an **exact** row count against a store holding
      strictly more rows, so it cannot pass vacuously (a broken search that
      returned nothing, or a cap that silently returned everything, both fail).

    The rejected calls raise, so they return no rows to count; the guard's
    effect is therefore asserted through the rejection, and the cap's reality
    through the exact in-range counts. The names say exactly that.
    """

    async def test_negative_list_limit_rejected_and_page_is_exact(
        self, bulk_store: SqliteEngravaCore
    ) -> None:
        # Without the guard SQLite reads LIMIT -1 as "no limit" and this
        # returns every row in the store.
        with pytest.raises(OutOfRangeBoundError):
            await list_memory_impl(bulk_store, limit=-1)

        # The store really does hold more rows than the page ...
        full = await list_memory_impl(bulk_store, limit=MAX_PAGE_LIMIT)
        assert full["count"] == TOTAL_THOUGHTS
        # ... and a paged call returns exactly the page, not the store.
        page = await list_memory_impl(bulk_store, limit=5)
        assert page["count"] == 5

    async def test_negative_top_k_rejected_and_window_is_exact(
        self, bulk_store: SqliteEngravaCore
    ) -> None:
        with pytest.raises(OutOfRangeBoundError):
            await search_keywords_impl(bulk_store, "widget", top_k=-1)

        # Positive control: the term really matches many thoughts ...
        unbounded = await search_keywords_impl(bulk_store, "widget", top_k=MAX_TOP_K)
        assert len(unbounded["results"]) == WIDGET_MATCHES
        # ... so an exact-3 window proves the cap bites, not that search is dead.
        bounded = await search_keywords_impl(bulk_store, "widget", top_k=3)
        assert len(bounded["results"]) == 3

    async def test_negative_query_limit_rejected_and_row_cap_is_exact(
        self, bulk_store: SqliteEngravaCore
    ) -> None:
        # query_memory's limit is interpolated into engrava's SQL string, so it
        # must be rejected before a query object is ever constructed.
        with pytest.raises(OutOfRangeBoundError):
            await query_memory_impl(bulk_store, FIND_ALL, limit=-1)

        unbounded = await query_memory_impl(bulk_store, FIND_ALL, limit=MAX_PAGE_LIMIT)
        assert len(unbounded["rows"]) == CREATED_ROWS
        bounded = await query_memory_impl(bulk_store, FIND_ALL, limit=4)
        assert len(bounded["rows"]) == 4

    async def test_negative_recent_limit_rejected_and_listing_is_exact(
        self, bulk_store: SqliteEngravaCore
    ) -> None:
        # This is the implementation the summarize_recent_memory prompt reaches
        # with its wire-supplied limit.
        with pytest.raises(OutOfRangeBoundError):
            await recent_thoughts_impl(bulk_store, limit=-1)

        unbounded = await recent_thoughts_impl(bulk_store, limit=MAX_PAGE_LIMIT)
        assert len(unbounded["thoughts"]) == TOTAL_THOUGHTS
        bounded = await recent_thoughts_impl(bulk_store, limit=2)
        assert len(bounded["thoughts"]) == 2

    async def test_over_max_limit_is_rejected(self, bulk_store: SqliteEngravaCore) -> None:
        # A lower bound alone would not close the hole: a huge value is equally
        # unbounded in effect, so the ceiling is enforced independently. (This
        # store is far smaller than the cap, so the assertion is the rejection
        # itself — the ceiling is a contract, not an observable row count here.)
        with pytest.raises(OutOfRangeBoundError):
            await list_memory_impl(bulk_store, limit=MAX_PAGE_LIMIT + 1)
        with pytest.raises(OutOfRangeBoundError):
            await list_memory_impl(bulk_store, limit=10**18)


class TestImplLevelBounds:
    """Every wire-supplied bound is domain-validated in its implementation."""

    @pytest.mark.parametrize("top_k", OUT_OF_RANGE_TOP_K)
    async def test_search_memory_top_k(self, store: SqliteEngravaCore, top_k: int) -> None:
        with pytest.raises(OutOfRangeBoundError):
            await search_memory_impl(store, "coffee", top_k=top_k)

    @pytest.mark.parametrize("top_k", OUT_OF_RANGE_TOP_K)
    async def test_search_keywords_top_k(self, store: SqliteEngravaCore, top_k: int) -> None:
        with pytest.raises(OutOfRangeBoundError):
            await search_keywords_impl(store, "coffee", top_k=top_k)

    @pytest.mark.parametrize("limit", OUT_OF_RANGE_LIMITS)
    async def test_list_memory_limit(self, store: SqliteEngravaCore, limit: int) -> None:
        with pytest.raises(OutOfRangeBoundError):
            await list_memory_impl(store, limit=limit)

    async def test_list_memory_negative_offset(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(OutOfRangeBoundError):
            await list_memory_impl(store, offset=-1)

    async def test_list_memory_zero_offset_is_valid(self, store: SqliteEngravaCore) -> None:
        # Zero is a legitimate page start and must not be rejected.
        result = await list_memory_impl(store, offset=0)
        assert result["offset"] == 0

    @pytest.mark.parametrize("limit", OUT_OF_RANGE_LIMITS)
    async def test_list_edges_limit(self, store: SqliteEngravaCore, limit: int) -> None:
        with pytest.raises(OutOfRangeBoundError):
            await list_edges_impl(store, limit=limit)

    @pytest.mark.parametrize("limit", OUT_OF_RANGE_LIMITS)
    async def test_query_memory_limit(self, store: SqliteEngravaCore, limit: int) -> None:
        with pytest.raises(OutOfRangeBoundError):
            await query_memory_impl(store, FIND_ALL, limit=limit)

    @pytest.mark.parametrize("limit", OUT_OF_RANGE_LIMITS)
    async def test_recent_thoughts_limit(self, store: SqliteEngravaCore, limit: int) -> None:
        with pytest.raises(OutOfRangeBoundError):
            await recent_thoughts_impl(store, limit=limit)

    async def test_max_boundary_values_are_accepted(self, store: SqliteEngravaCore) -> None:
        # The maxima themselves are inside the domain (inclusive bounds).
        assert (await list_memory_impl(store, limit=MAX_PAGE_LIMIT))["limit"] == MAX_PAGE_LIMIT
        assert isinstance(
            (await search_keywords_impl(store, "coffee", top_k=MAX_TOP_K))["results"], list
        )

    async def test_query_memory_without_limit_still_works(self, store: SqliteEngravaCore) -> None:
        # An omitted limit is not a bound to validate; the store's own default
        # applies and the call must not be rejected.
        result = await query_memory_impl(store, FIND_ALL)
        assert isinstance(result["rows"], list)


class TestBoundErrorsAreCleanAtTheBoundary:
    """An out-of-range bound surfaces a clean, actionable ToolError."""

    async def test_impl_error_maps_to_clean_tool_error(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                await list_memory_impl(store, limit=-1)
        text = str(excinfo.value)
        # Names the caller's own argument, its value, and the accepted range ...
        assert "limit" in text
        assert str(MAX_PAGE_LIMIT) in text
        # ... and leaks no internal symbol or stack frame.
        assert "OutOfRangeBoundError" not in text
        assert "Traceback" not in text
        assert "sqlite" not in text.lower()

    @pytest.mark.parametrize(
        ("tool", "arguments"),
        [
            ("list_memory", {"limit": -1}),
            ("list_memory", {"offset": -1}),
            ("search_keywords", {"query": "coffee", "top_k": -1}),
            ("search_memory", {"query_text": "coffee", "top_k": -1}),
            ("list_edges", {"limit": -1}),
            ("query_memory", {"query": FIND_ALL, "limit": -1}),
            ("list_memory", {"limit": 10**18}),
            ("search_keywords", {"query": "coffee", "top_k": MAX_TOP_K + 1}),
        ],
    )
    async def test_out_of_range_bound_is_rejected_over_the_wire(
        self,
        store: SqliteEngravaCore,
        tool: str,
        arguments: dict[str, object],
    ) -> None:
        # Rejected at the real MCP boundary — whether by the advertised schema
        # or by the implementation's own guard, the client gets an error rather
        # than an unbounded result.
        async with _client_for(store) as client:
            result = await client.call_tool(tool, arguments)
        assert result.isError is True

    async def test_valid_bounds_still_succeed_over_the_wire(self, store: SqliteEngravaCore) -> None:
        # The guard is presentation-only for in-domain values.
        async with _client_for(store) as client:
            result = await client.call_tool("list_memory", {"limit": 2, "offset": 0})
        assert result.isError is False
        assert result.structuredContent is not None
        assert result.structuredContent["limit"] == 2

    async def test_negative_prompt_limit_is_rejected_before_the_body_runs(
        self,
        store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The prompt's limit crosses the wire too, so it is bounded twice over:
        # by the advertised annotation, and by the domain guard inside the body
        # for the paths where the protocol layer does not apply. The claim here
        # is the first of those — and the error alone cannot carry it, because
        # the in-body guard also surfaces as an McpError mentioning "limit".
        #
        # What tells them apart is whether the body ran at all. The store call
        # the body makes is recorded, so with the bound dropped from the
        # annotation the body proceeds, the call is recorded, and this fails.
        reached_the_body: list[int] = []
        real_recent_thoughts = server_module.recent_thoughts_impl

        async def _recording_recent_thoughts(
            store: SqliteEngravaCore, *, limit: int
        ) -> dict[str, object]:
            reached_the_body.append(limit)
            return await real_recent_thoughts(store, limit=limit)

        monkeypatch.setattr(server_module, "recent_thoughts_impl", _recording_recent_thoughts)

        async with _client_for(store) as client:
            with pytest.raises(McpError) as excinfo:
                await client.get_prompt("summarize_recent_memory", {"limit": "-1"})
            # Control: an in-range limit does reach the body through the same
            # recorder, so the emptiness asserted below is the rejection and not
            # a recorder that never observes anything.
            await client.get_prompt("summarize_recent_memory", {"limit": "2"})

        assert reached_the_body == [2]
        # The rejection names the offending argument, not our internal symbol.
        text = str(excinfo.value)
        assert "limit" in text.lower()
        assert "OutOfRangeBoundError" not in text

    async def test_prompt_body_maps_a_bad_limit_to_a_curated_message(
        self, store: SqliteEngravaCore
    ) -> None:
        # The whole point of the second guard: exercise the prompt body on the
        # path where the protocol layer does NOT apply. Calling the registered
        # handler's unwrapped function bypasses pydantic's argument validation,
        # exactly as a caller reaching the body without that layer would.
        # Unguarded, this surfaces a raw OutOfRangeBoundError to the client.
        server: FastMCP = FastMCP(SERVER_NAME)
        provider = StoreProvider()
        provider.set(store)
        register_prompts(server, provider)

        registered = {prompt.name: prompt for prompt in server._prompt_manager.list_prompts()}
        body = registered["summarize_recent_memory"].fn.__wrapped__

        with pytest.raises(ToolError) as excinfo:
            await body(limit=-1)

        text = str(excinfo.value)
        assert "limit" in text
        assert str(MAX_PAGE_LIMIT) in text
        # The client never sees the internal exception name or a stack frame.
        assert "OutOfRangeBoundError" not in text
        assert "Traceback" not in text

    async def test_prompt_body_still_renders_for_a_valid_limit(
        self, store: SqliteEngravaCore
    ) -> None:
        # The guard is presentation-only on the happy path.
        server: FastMCP = FastMCP(SERVER_NAME)
        provider = StoreProvider()
        provider.set(store)
        register_prompts(server, provider)

        registered = {prompt.name: prompt for prompt in server._prompt_manager.list_prompts()}
        body = registered["summarize_recent_memory"].fn.__wrapped__

        rendered = await body(limit=2)
        assert isinstance(rendered, str)
        assert rendered


class TestBoundsAdvertisedInToolSchema:
    """The bounds are published in the advertised MCP tool schema."""

    async def test_schema_publishes_limit_and_top_k_bounds(self, store: SqliteEngravaCore) -> None:
        async with _client_for(store) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}

        list_memory_schema = tools["list_memory"].inputSchema["properties"]
        assert list_memory_schema["limit"]["minimum"] == 1
        assert list_memory_schema["limit"]["maximum"] == MAX_PAGE_LIMIT
        assert list_memory_schema["offset"]["minimum"] == 0

        keywords_schema = tools["search_keywords"].inputSchema["properties"]
        assert keywords_schema["top_k"]["minimum"] == 1
        assert keywords_schema["top_k"]["maximum"] == MAX_TOP_K

        # search_memory carries the same ranked-window bound as search_keywords.
        memory_schema = tools["search_memory"].inputSchema["properties"]
        assert memory_schema["top_k"]["minimum"] == 1
        assert memory_schema["top_k"]["maximum"] == MAX_TOP_K

        # Both ends of the edge-listing page size, not just the ceiling.
        edges_schema = tools["list_edges"].inputSchema["properties"]
        assert edges_schema["limit"]["minimum"] == 1
        assert edges_schema["limit"]["maximum"] == MAX_PAGE_LIMIT

    async def test_schema_publishes_bounds_on_the_nullable_query_limit(
        self, store: SqliteEngravaCore
    ) -> None:
        # query_memory's limit is optional, so its bounds may sit inside a
        # nullable union rather than on the property itself. Assert they are
        # advertised either way — and that omitting the limit stays legal.
        async with _client_for(store) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}

        limit_schema = tools["query_memory"].inputSchema["properties"]["limit"]
        branches = limit_schema.get("anyOf", [limit_schema])
        bounded = [branch for branch in branches if "maximum" in branch]
        assert bounded, f"no bounded branch in the advertised schema: {limit_schema!r}"
        assert bounded[0]["minimum"] == 1
        assert bounded[0]["maximum"] == MAX_PAGE_LIMIT

    async def test_prompt_argument_is_advertised(self, store: SqliteEngravaCore) -> None:
        # The prompt's limit crosses the wire, so a client can supply it and it
        # must appear in the advertised argument list.
        async with _client_for(store) as client:
            prompts = {prompt.name: prompt for prompt in (await client.list_prompts()).prompts}

        arguments = prompts["summarize_recent_memory"].arguments
        assert arguments is not None
        assert "limit" in {argument.name for argument in arguments}


class TestQueryObjectBoundary:
    """A value crossing the wire never becomes a query-object identifier."""

    async def test_limit_is_validated_before_a_query_object_is_built(
        self, store: SqliteEngravaCore
    ) -> None:
        # query_memory's limit is interpolated into engrava's SQL string rather
        # than bound, so validation must happen before the MindQLQuery is
        # constructed. A rejected bound must surface as the domain error, never
        # as a database error from a constructed query.
        with pytest.raises(OutOfRangeBoundError):
            await query_memory_impl(store, FIND_ALL, limit=-1)

    async def test_edge_metadata_write_then_bounded_edge_listing(
        self, store: SqliteEngravaCore
    ) -> None:
        # A bounded list_edges call still returns real data — the guard does not
        # break the happy path it protects.
        first = await store_thought_impl(store, essence="bound source", content="source body")
        second = await store_thought_impl(store, essence="bound target", content="target body")
        await link_thoughts_impl(
            store,
            first["thought"]["thought_id"],
            second["thought"]["thought_id"],
            EdgeType.ASSOCIATED,
        )
        listed = await list_edges_impl(store, limit=1)
        assert listed["count"] == 1
