"""Read-only view over an engrava store, for the server's read-only deployment mode.

**Guarantee, stated exactly:** every read through :class:`ReadOnlyStore` leaves
``access_count`` exactly as it found it, even with access tracking on — a plain read
would otherwise buffer a deferred update that is written out when the store closes.
Not "no writes at all": store *resolution* (:mod:`engrava_mcp.config`) can still create
or migrate schema before any tool call is possible, unaffected by and out of scope for
this module. The narrower claim is the one that needs to be true: an exposed read never
stages a persistent access-tracking mutation.

:class:`ReadOnlyMcpStore` names the nine store calls the read ``_impl`` functions in
:mod:`engrava_mcp.server` actually make — narrower than engrava's own
``EngravaReadProtocol`` — so a call site this module forgets to guard is a type error,
not a silent gap.

**One mechanism for all nine.** Every method applies
``async with self._inner.suppress_access_tracking()`` directly; there is no dependency
on engrava's own ``ReadOnlyEngrava`` view. An earlier version delegated six of the nine
to it and hand-covered the rest (two methods it does not expose, plus a third whose
signature there rejects a keyword this server always passes) — dropped because which six
it happens to reach is decided by another package's current signatures, and the
write-blocking it would otherwise contribute is unreachable here anyway (the MCP surface
exposes no writes, and the write tools hold their own unwrapped store reference — see
``StoreProvider`` in :mod:`engrava_mcp.server`). The cost: this module no longer
inherits improvements to that view, and carries suppression at nine sites instead of six
— discharged by the protocol (a missing method is a type error) plus a behavioural
``access_count`` test for each method that actually buffers. Verified against
:meth:`~engrava.SqliteEngravaCore._buffer_accesses`'s call sites rather than assumed:
only ``get_thought`` and ``search_hybrid`` do; the other seven never buffer anything and
are guarded by the type contract alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from engrava import (
        EdgeRecord,
        EdgeType,
        EngravaMetrics,
        HybridSearchResult,
        KnowledgeSource,
        MetadataFilter,
        MindQLExtension,
        MindQLQuery,
        MindQLResult,
        SqliteEngravaCore,
        ThoughtRecord,
    )


@runtime_checkable
class ReadOnlyMcpStore(Protocol):
    """The read surface this server's read tools and resources call through a store.

    Names exactly the nine store calls the read ``_impl`` functions in
    :mod:`engrava_mcp.server` make — not engrava's full
    :class:`~engrava.EngravaReadProtocol`, which also covers capabilities (``recall``,
    ``search_similar``, ``get_embedding``, ``get_actions``, ``max_cycle``) this server
    never calls. Both :class:`~engrava.SqliteEngravaCore` and :class:`ReadOnlyStore`
    satisfy this protocol structurally, so either can be handed to a read ``_impl``
    function typed against it.
    """

    async def get_thought(self, thought_id: str) -> ThoughtRecord | None:
        """Retrieve a thought by its identifier.

        Args:
            thought_id: Identifier of the thought to retrieve.

        Returns:
            The thought record, or ``None`` if not found.

        """
        ...

    async def list_thoughts(
        self,
        *,
        priority: str | None = None,
        lifecycle_status: str | None = None,
        thought_type: str | None = None,
        min_cycle: int | None = None,
        max_cycle: int | None = None,
        include_expired: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ThoughtRecord]:
        """List thoughts matching the given filters.

        Args:
            priority: Filter by priority level.
            lifecycle_status: Filter by lifecycle status.
            thought_type: Filter by thought type.
            min_cycle: Minimum ``updated_cycle`` (inclusive).
            max_cycle: Maximum ``updated_cycle`` (inclusive).
            include_expired: If ``True``, include expired thoughts.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            The matching thought records.

        """
        ...

    async def search_fts(
        self,
        query: str,
        top_k: int = 10,
        *,
        include_archived: bool = False,
    ) -> list[tuple[str, float]]:
        """Run a full-text BM25 keyword search.

        Args:
            query: Search query string (FTS5 syntax supported).
            top_k: Maximum number of results.
            include_archived: When ``True``, re-admit archived thoughts.

        Returns:
            ``(thought_id, bm25_score)`` pairs, most relevant first.

        """
        ...

    async def search_hybrid(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        include_reflections: bool = True,
        recency_now: str | None = None,
        include_archived: bool = False,
    ) -> HybridSearchResult:
        """Run a hybrid (lexical + vector + recency) ranked search.

        Args:
            query_text: Natural-language query text.
            top_k: Maximum number of ranked results.
            include_reflections: Whether consolidated reflection thoughts may
                appear in the results.
            recency_now: Optional ISO-8601 instant driving transaction-time
                recency.
            include_archived: When ``True``, re-admit archived thoughts.

        Returns:
            The ranked search result.

        """
        ...

    async def metrics(self) -> EngravaMetrics:
        """Return a point-in-time store-health metrics snapshot."""
        ...

    async def get_edges(
        self,
        thought_id: str,
        *,
        direction: str = "BOTH",
        limit: int | None = None,
    ) -> list[EdgeRecord]:
        """Retrieve the edges connected to a thought.

        Args:
            thought_id: Identifier of the thought whose edges to fetch.
            direction: ``"IN"``, ``"OUT"``, or ``"BOTH"``.
            limit: Optional cap on the number of edges returned, the
                highest-weight ones first.  ``None`` returns every edge.

        Returns:
            The matching edge records.

        """
        ...

    async def list_edges(
        self,
        *,
        edge_type: EdgeType | None = None,
        source: KnowledgeSource | None = None,
        filters: MetadataFilter | None = None,
        limit: int = 5000,
    ) -> list[EdgeRecord]:
        """List edges matching optional filters.

        Args:
            edge_type: If given, restrict to this edge type.
            source: If given, restrict to this knowledge source.
            filters: Optional typed metadata filter.
            limit: Maximum number of edges to return.

        Returns:
            The matching edge records.

        """
        ...

    async def count_thoughts(
        self,
        *,
        lifecycle_status: str | None = None,
        thought_type: str | None = None,
        priority: str | None = None,
        include_expired: bool = False,
    ) -> int:
        """Count thoughts matching the given filters.

        Args:
            lifecycle_status: Filter by lifecycle status.
            thought_type: Filter by thought type.
            priority: Filter by priority level.
            include_expired: If ``True``, include expired thoughts.

        Returns:
            The number of matching thoughts.

        """
        ...

    async def execute_mindql(
        self,
        query: MindQLQuery,
        *,
        extensions: dict[str, MindQLExtension] | None = None,
    ) -> MindQLResult:
        """Execute an already-parsed MindQL query.

        Args:
            query: A parsed MindQL query.
            extensions: Optional registered MindQL extension commands.

        Returns:
            The query result.

        """
        ...


class ReadOnlyStore:
    """Read-only view over a store that leaves ``access_count`` untouched.

    Every method below runs its call on the wrapped store inside
    ``self._inner.suppress_access_tracking()`` — one mechanism, applied uniformly to
    all nine methods :class:`ReadOnlyMcpStore` declares, rather than delegating some to
    engrava's own read-only view and hand-covering the rest (see the module docstring
    for why that split was dropped).

    Args:
        inner: The store to wrap.

    """

    def __init__(self, inner: SqliteEngravaCore) -> None:
        self._inner = inner

    async def get_thought(self, thought_id: str) -> ThoughtRecord | None:
        """Call directly on the wrapped store, suppressed; see :class:`ReadOnlyMcpStore`."""
        async with self._inner.suppress_access_tracking():
            return await self._inner.get_thought(thought_id)

    async def list_thoughts(
        self,
        *,
        priority: str | None = None,
        lifecycle_status: str | None = None,
        thought_type: str | None = None,
        min_cycle: int | None = None,
        max_cycle: int | None = None,
        include_expired: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ThoughtRecord]:
        """Call directly on the wrapped store, suppressed; see :class:`ReadOnlyMcpStore`."""
        async with self._inner.suppress_access_tracking():
            return await self._inner.list_thoughts(
                priority=priority,
                lifecycle_status=lifecycle_status,
                thought_type=thought_type,
                min_cycle=min_cycle,
                max_cycle=max_cycle,
                include_expired=include_expired,
                limit=limit,
                offset=offset,
            )

    async def search_fts(
        self,
        query: str,
        top_k: int = 10,
        *,
        include_archived: bool = False,
    ) -> list[tuple[str, float]]:
        """Call directly on the wrapped store, suppressed; see :class:`ReadOnlyMcpStore`."""
        async with self._inner.suppress_access_tracking():
            return await self._inner.search_fts(query, top_k, include_archived=include_archived)

    async def search_hybrid(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        include_reflections: bool = True,
        recency_now: str | None = None,
        include_archived: bool = False,
    ) -> HybridSearchResult:
        """Call directly on the wrapped store, suppressed; see :class:`ReadOnlyMcpStore`."""
        async with self._inner.suppress_access_tracking():
            return await self._inner.search_hybrid(
                query_text,
                top_k=top_k,
                include_reflections=include_reflections,
                recency_now=recency_now,
                include_archived=include_archived,
            )

    async def metrics(self) -> EngravaMetrics:
        """Call directly on the wrapped store, suppressed; see :class:`ReadOnlyMcpStore`."""
        async with self._inner.suppress_access_tracking():
            return await self._inner.metrics()

    async def get_edges(
        self,
        thought_id: str,
        *,
        direction: str = "BOTH",
        limit: int | None = None,
    ) -> list[EdgeRecord]:
        """Call directly on the wrapped store, suppressed; see :class:`ReadOnlyMcpStore`."""
        async with self._inner.suppress_access_tracking():
            return await self._inner.get_edges(thought_id, direction=direction, limit=limit)

    async def list_edges(
        self,
        *,
        edge_type: EdgeType | None = None,
        source: KnowledgeSource | None = None,
        filters: MetadataFilter | None = None,
        limit: int = 5000,
    ) -> list[EdgeRecord]:
        """Call directly on the wrapped store, suppressed; see :class:`ReadOnlyMcpStore`."""
        async with self._inner.suppress_access_tracking():
            return await self._inner.list_edges(
                edge_type=edge_type,
                source=source,
                filters=filters,
                limit=limit,
            )

    async def count_thoughts(
        self,
        *,
        lifecycle_status: str | None = None,
        thought_type: str | None = None,
        priority: str | None = None,
        include_expired: bool = False,
    ) -> int:
        """Call directly on the wrapped store, suppressed; see :class:`ReadOnlyMcpStore`."""
        async with self._inner.suppress_access_tracking():
            return await self._inner.count_thoughts(
                lifecycle_status=lifecycle_status,
                thought_type=thought_type,
                priority=priority,
                include_expired=include_expired,
            )

    async def execute_mindql(
        self,
        query: MindQLQuery,
        *,
        extensions: dict[str, MindQLExtension] | None = None,
    ) -> MindQLResult:
        """Call directly on the wrapped store, suppressed; see :class:`ReadOnlyMcpStore`."""
        async with self._inner.suppress_access_tracking():
            return await self._inner.execute_mindql(query, extensions=extensions)
