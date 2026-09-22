"""Tests for the MCP tool error contract.

When a tool hits a known failure condition, the client must receive a
clean, typed, actionable error — a message with a helpful hint and
``isError`` set — rather than a raw Python traceback or an internal class
name.  These tests drive the real tool boundary through the in-process MCP
client transport (so MCPServer's error wrapping runs for real) and assert on
the message the client actually sees.

The conditions covered are:

* the store is not yet available (a misconfigured deployment),
* ``query_memory`` receives a non-``FIND`` command (``SELECT`` / ``COUNT``),
* ``query_memory`` receives a query whose own verb is not classified as
  ``FIND`` and fails to parse — the message stays generic, since the parser's
  raw text for an unrecognised verb would name the full command set,
* ``query_memory`` receives a query whose own verb *is* classified as
  ``FIND`` but fails to parse (an unknown table, a bad condition) — this
  gets the parser's specific diagnosis, because that verb classification
  guarantees the failure is about the FIND's own content,
* ``query_memory`` receives a syntactically valid ``FIND`` that fails during
  execution (e.g. an unknown column) — the same specific-diagnosis treatment,
  for the same reason, at a different stage,
* ``update_thought`` names a thought that does not exist,
* ``link_thoughts`` names an endpoint that does not exist.

Two cross-cutting properties are asserted in addition to per-condition
hints: the ``FIND``-only guard on ``query_memory`` is preserved (a
``SELECT`` is still rejected and the message never invites raw SQL), and no
error message leaks a filesystem path, a stack frame, or an internal symbol
name.

The server and client are built inside each test (rather than via a
yielding fixture) so the in-process transport's task-bound cancel scopes
enter and exit within the same task — the pattern the end-to-end server
tests already use.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import inspect
import re
import sqlite3
import sys
import textwrap
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, NoReturn

import aiosqlite
import pytest
from engrava import (
    ConnectionQuarantinedError,
    DefaultEngravaHooks,
    DerivedRecord,
    DerivedRecordError,
    DeriveGates,
    EdgeType,
    EmbeddingQueryPrefixMismatchError,
    MindQLParseError,
    Priority,
    SqliteEngravaCore,
    StaleDataError,
    ThoughtType,
    VectorDimensionMismatchError,
    WriteContentionError,
    WriteLockTimeoutError,
)
from engrava.domain import exceptions as engrava_exceptions
from engrava.domain.exceptions import DuplicateEdgeError, EngravaError
from engrava.infrastructure.sqlite.engrava_core import _derived_thought_id
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import engrava_mcp.server as server_module
from engrava_mcp.server import (
    DUPLICATE_EDGE_MESSAGE,
    EDGE_ID_COLLISION_MESSAGE,
    SERVER_NAME,
    SQLITE_MAX_BOUND_INT,
    EmbeddingQueryNotSupportedError,
    StoreProvider,
    _DeriveProducerFailedError,
    _query_declares_find,
    _tool_errors,
    link_thoughts_impl,
    query_memory_impl,
    register_tools,
    store_thought_impl,
    update_thought_impl,
)
from tests.conftest import make_thought
from tests.inprocess_client import connect_client

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from engrava import DeriveContext, ThoughtRecord
    from mcp import Client

#: Substrings that would indicate a leaked traceback or internal symbol.
#: Error messages shown to a client must contain none of them.
_LEAK_MARKERS = (
    "Traceback",
    'File "',
    "StoreNotReadyError",
    "UnsupportedQueryError",
    "UnexecutableQueryError",
    "MalformedFindError",
    "MindQLParseError",
    "ThoughtNotFoundError",
    "ReferentialIntegrityError",
    "SqliteEngravaCore",
    "lifespan",
    # Database-constraint internals: a UNIQUE violation's raw message names the
    # edge table and its columns — none of these may reach the client.
    "IntegrityError",
    "UNIQUE constraint",
    "edge.from_thought_id",
    "edge.to_thought_id",
    "edge.edge_type",
    "edge.edge_id",
    # Domain-model-validation internals: Pydantic's raw message names the model
    # class and links its docs site.
    "ValidationError",
    "ThoughtRecord",
    "pydantic",
    "errors.pydantic.dev",
    # Lifecycle-transition internals: the raw InvalidTransitionError message
    # names the internal status type.
    "InvalidTransitionError",
    "Invalid LifecycleStatus transition",
    # StaleDataError's raw message names the internal entity-type symbol.
    "StaleDataError",
    "Stale data:",
    # The busy-store, long-write-timeout and unusable-store errors must not name
    # their own types.
    "WriteContentionError",
    "WriteLockTimeoutError",
    "ConnectionQuarantinedError",
    # A bare ValueError's raw phrasing (duplicate thought_id, oversized edge
    # metadata) must be replaced, not forwarded.
    "Thought already exists:",
    "metadata serialized size",
    # The embeddings-refusal and overflow guards must not name their own type.
    "EmbeddingQueryNotSupportedError",
    "OverflowError",
    "convert to SQLite INTEGER",
    # The search-time embedding-mismatch errors must not name their own type
    # or echo the store's raw dimension integers / prefix values.
    "VectorDimensionMismatchError",
    "query vector dimension mismatch",
    "store expects",
    "EmbeddingQueryPrefixMismatchError",
    "Embedding query prefix mismatch",
    "corpus was built to pair with",
    # DerivedRecordError's raw message names its own type's wrapping and the
    # library's internal per-reason phrasing; none of it may reach the client.
    "DerivedRecordError",
    "_DeriveProducerFailedError",
    "max_derived_per_source",
    "identity collides with",
    "[source=",
)

#: Phrases that would wrongly suggest raw SQL is runnable over the wire.
#: The ``FIND``-only rejection message must contain none of them.
_SQL_INVITATIONS = (
    "use select",
    "run select",
    "raw sql",
    "arbitrary sql",
    "try select",
    "select is",
    "select instead",
)


@asynccontextmanager
async def _client_for(store: SqliteEngravaCore) -> AsyncIterator[Client]:
    """Open a connected client whose tools query the given store.

    Builds a server, points a :class:`StoreProvider` at ``store``, registers
    the tools, and connects the in-process client so the real tool boundary
    (and MCPServer's error wrapping) runs end to end.

    Args:
        store: The seeded store the tools should query.

    Yields:
        A connected client session wired to ``store``.

    """
    server: MCPServer = MCPServer(SERVER_NAME)
    provider = StoreProvider()
    provider.set(store, read_store=store)
    register_tools(server, provider, read_only=False)
    async with connect_client(server) as client:
        yield client


@asynccontextmanager
async def _store_less_client() -> AsyncIterator[Client]:
    """Open a connected client whose provider never received a store.

    Registering the tools against an unpopulated :class:`StoreProvider`
    reproduces a deployment whose store has not been configured: the first
    tool call hits the store-not-ready condition at the real boundary.

    Yields:
        A connected client session backed by a store-less provider.

    """
    server: MCPServer = MCPServer(SERVER_NAME)
    register_tools(server, StoreProvider(), read_only=False)
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


def _assert_no_raw_duplicate_phrasing(text: str) -> None:
    """Assert the store's own duplicate-edge wording never reaches the client.

    The typed ``DuplicateEdgeError`` reads "edge relationship already exists:
    '<id>' -[TYPE]-> '<id>'". That phrasing belongs to the store and may change
    at any time, so forwarding it would put an upstream string on our wire
    contract.

    Args:
        text: The client-facing error message to inspect.

    """
    lowered = text.lower()
    assert "edge relationship" not in lowered, f"leaked the store's phrasing: {text!r}"
    assert "already exists:" not in lowered, f"leaked the store's phrasing: {text!r}"
    assert "-[" not in text, f"leaked the store's edge notation: {text!r}"


def _assert_no_leak(text: str) -> None:
    """Assert an error message leaks no path, stack frame, or symbol name.

    Args:
        text: The client-facing error message to inspect.

    """
    for marker in _LEAK_MARKERS:
        assert marker not in text, f"error message leaked {marker!r}: {text!r}"
    # No forward-slash path segment ...
    assert not re.search(r"/\w", text), f"error message leaked a '/' path: {text!r}"
    # ... and no backslash path segment.
    assert "\\" not in text, f"error message leaked a '\\' path: {text!r}"


class TestStoreNotReady:
    """The store-not-ready condition surfaces an actionable config hint."""

    async def test_reports_missing_store_with_env_var_hint(self) -> None:
        async with _store_less_client() as client:
            result = await client.call_tool("memory_stats", {})

        assert result.is_error is True
        text = _error_text(result.content)
        # Actionable: it names the two documented configuration env vars ...
        assert "ENGRAVA_DB_PATH" in text
        assert "ENGRAVA_MCP_CONFIG" in text
        # ... and leaks no path, stack frame, or internal symbol.
        _assert_no_leak(text)


class TestUnsupportedQuery:
    """A non-``FIND`` ``query_memory`` is rejected with the FIND contract."""

    async def test_select_is_rejected_and_message_states_find_only(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool(
                "query_memory",
                {"query": "SELECT thought_id FROM thought"},
            )

        assert result.is_error is True
        text = _error_text(result.content)
        # The guard still rejects the query and states the FIND-only contract.
        assert "FIND" in text
        assert "only FIND" in text
        _assert_no_leak(text)

    async def test_count_is_rejected_with_find_example(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool("query_memory", {"query": "COUNT thoughts"})

        assert result.is_error is True
        text = _error_text(result.content)
        assert "FIND" in text
        # A valid FIND example is offered to get the caller back on track.
        assert "FIND thoughts WHERE" in text
        _assert_no_leak(text)


class TestGuardPreservation:
    """The FIND-only guard must reject SELECT without ever inviting SQL."""

    async def test_select_rejection_does_not_invite_sql(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool(
                "query_memory",
                {"query": "SELECT * FROM thought WHERE 1=1"},
            )

        # The rejection itself is intact: a SELECT still fails.
        assert result.is_error is True
        text = _error_text(result.content)
        lowered = text.lower()

        # The message asserts the FIND-only contract ...
        assert "find" in lowered
        assert "only find" in lowered
        # ... and must NOT suggest that raw SQL / SELECT is runnable.
        for invite in _SQL_INVITATIONS:
            assert invite not in lowered, f"message invited SQL via {invite!r}: {text!r}"
        # The only mention of SELECT permitted is echoing the rejected verb.
        # Stripping that quoted echo, no bare "SELECT" remains — so the
        # message never presents SELECT as a usable command.
        assert "SELECT" not in text.replace("'SELECT'", "")

    async def test_extension_command_is_also_rejected(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # A made-up verb parses as an unknown command and is rejected too —
        # the surface stays restricted to FIND, not just "not SELECT".
        async with _client_for(store) as client:
            result = await client.call_tool("query_memory", {"query": "DROP thoughts"})

        assert result.is_error is True
        text = _error_text(result.content)
        lowered = text.lower()
        assert "only find" in lowered
        _assert_no_leak(text)
        # The message must NOT leak the parser's full command set. For an
        # unrecognised verb the raw parser error reads "Expected FIND, COUNT,
        # SELECT, or extension command" — naming COUNT / SELECT / extension
        # would advertise commands the MCP surface deliberately hides. The
        # input verb ("DROP") is not echoed, so none of these may appear.
        assert "COUNT" not in text
        assert "SELECT" not in text
        assert "extension" not in lowered

    async def test_lowercase_whitespace_padded_unknown_verb_is_also_rejected(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # The verb classifier that routes a MindQLParseError to the specific
        # or the generic message must not itself be fooled by case or
        # padding: a lowercase, whitespace-padded unknown verb is still "not
        # FIND", so it keeps the generic message and the command set stays
        # hidden. This is the guard the classifier exists to protect, not a
        # by-product of it.
        async with _client_for(store) as client:
            result = await client.call_tool("query_memory", {"query": "  drop   thoughts  "})

        assert result.is_error is True
        text = _error_text(result.content)
        lowered = text.lower()
        assert "only find" in lowered
        _assert_no_leak(text)
        assert "COUNT" not in text
        assert "SELECT" not in text
        assert "extension" not in lowered


class TestQueryDeclaresFind:
    """``_query_declares_find`` classifies the verb, and fails safe.

    This is the discriminator ``query_memory_impl`` uses to decide which
    message a ``MindQLParseError`` gets. It never interprets the query — it
    only reads the first token, after stripping an optional ``EXPLAIN``
    prefix the same way ``parse()`` does — and it returns ``False`` on
    anything ambiguous rather than risk a false positive that would route an
    unrecognised-verb failure to the specific-message branch.
    """

    @pytest.mark.parametrize(
        "query",
        [
            "FIND thoughts",
            "FIND",
            "FIND nosuchtable",
            "  find thoughts LIMIT 5",
            "FiNd nosuchtable",
            "EXPLAIN FIND thoughts",
            "EXPLAIN   FIND nosuchtable",
            "explain find thoughts",
        ],
    )
    def test_recognises_find_case_and_whitespace_insensitively(self, query: str) -> None:
        assert _query_declares_find(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "COUNT thoughts",
            "SELECT * FROM thought",
            "DROP thoughts",
            "  drop   thoughts  ",
            "FINDER thoughts",
            "",
            "   ",
            "EXPLAIN",
            "EXPLAIN   ",
            "explain",
        ],
    )
    def test_fails_safe_to_false_on_anything_not_confidently_find(self, query: str) -> None:
        # Every one of these either is not FIND (COUNT, SELECT, DROP,
        # "FINDER" is not "FIND") or is too ambiguous to call at all (empty,
        # a bare EXPLAIN with nothing after it) -- both fail the same way, to
        # the side that keeps the generic message.
        assert _query_declares_find(query) is False


class TestMalformedFind:
    """A query whose own verb is FIND reports the parser's specific diagnosis.

    ``query_memory_impl`` classifies the verb itself, before ``parse()`` ever
    runs (see ``_query_declares_find`` in ``server.py``): when the query's
    first token, after an optional ``EXPLAIN`` prefix, is ``FIND``
    case-insensitively, whatever ``parse()`` then rejects the query for
    cannot be the unrecognised-verb message — it is necessarily about that
    FIND's own content (a bad table, a bad condition, a missing table name)
    — so the raw diagnosis is surfaced verbatim instead of the generic
    FIND-only message. A query whose verb is *not* FIND (an unrecognised verb,
    or anything the classifier does not confidently recognise) keeps the
    pre-existing generic message; that side is covered by
    :class:`TestGuardPreservation`.
    """

    async def test_find_alone_reports_its_own_diagnosis(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool("query_memory", {"query": "FIND"})

        assert result.is_error is True
        text = _error_text(result.content)
        # The parser's own diagnosis reaches the client verbatim ...
        assert "FIND requires a table name" in text
        # ... rather than the generic FIND-only message.
        assert "could not be parsed" not in text
        _assert_no_leak(text)

    async def test_unknown_table_names_the_table(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool("query_memory", {"query": "FIND nosuchtable"})

        assert result.is_error is True
        text = _error_text(result.content)
        assert "nosuchtable" in text
        assert "could not be parsed" not in text
        _assert_no_leak(text)

    async def test_invalid_condition_names_the_condition(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool(
                "query_memory",
                {"query": "FIND thoughts WHERE priority ~~ 'P1'"},
            )

        assert result.is_error is True
        text = _error_text(result.content)
        # The rejected condition text reaches the client ...
        assert "priority ~~ 'P1'" in text
        # ... rather than the generic FIND-only message.
        assert "could not be parsed" not in text
        _assert_no_leak(text)

    async def test_explain_prefix_gets_the_same_specific_diagnosis(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # EXPLAIN is stripped before the verb is read, both by parse() and by
        # the classifier that decides which message to give — so a malformed
        # FIND behind EXPLAIN is diagnosed the same as one without it.
        async with _client_for(store) as client:
            result = await client.call_tool(
                "query_memory",
                {"query": "EXPLAIN FIND nosuchtable"},
            )

        assert result.is_error is True
        text = _error_text(result.content)
        assert "nosuchtable" in text
        assert "could not be parsed" not in text
        _assert_no_leak(text)

    async def test_mixed_case_find_still_gets_the_specific_diagnosis(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # The verb classifier folds case exactly like parse() does, so a
        # mixed-case verb is still recognised as FIND.
        async with _client_for(store) as client:
            result = await client.call_tool("query_memory", {"query": "FiNd nosuchtable"})

        assert result.is_error is True
        text = _error_text(result.content)
        assert "nosuchtable" in text
        assert "could not be parsed" not in text
        _assert_no_leak(text)

    async def test_lowercase_whitespace_padded_find_still_executes(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # A lowercase, whitespace-padded FIND that is otherwise valid is not
        # touched by the classifier at all: it parses and executes normally.
        # This guards against a classifier bug that would reject or mishandle
        # a perfectly good query just because of case or padding.
        async with _client_for(store) as client:
            result = await client.call_tool("query_memory", {"query": "  find thoughts LIMIT 5"})

        assert result.is_error is not True


class TestUnexecutableFind:
    """A FIND that parses fine but fails at execution reports the diagnosis.

    ``FIND thoughts WHERE nosuchfield = 'x'`` is a syntactically valid FIND —
    it parses — and only fails once the executor checks the column against
    the table's allowlist. Like the parse()-stage failures in
    :class:`TestMalformedFind`, this diagnosis is safe to surface verbatim:
    the query has committed to FIND (whether by parsing all the way through,
    here, or merely by its own leading verb, there) before failing, so the
    message cannot name another MindQL command the way the parser's
    unrecognised-verb message would.
    """

    async def test_unknown_column_names_the_column(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool(
                "query_memory",
                {"query": "FIND thoughts WHERE nosuchfield = 'x'"},
            )

        assert result.is_error is True
        text = _error_text(result.content)
        # The specific diagnosis reaches the client, naming the bad column ...
        assert "nosuchfield" in text
        # ... rather than being swallowed by the generic FIND-only message.
        assert "could not be parsed" not in text
        _assert_no_leak(text)


class TestUpdateMissingThought:
    """Updating an absent thought names the missing identifier."""

    async def test_missing_thought_reports_id_with_hint(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool(
                "update_thought",
                {"thought_id": "ghost-thought", "essence": "x"},
            )

        assert result.is_error is True
        text = _error_text(result.content)
        # The offending id is echoed so the caller knows which one is wrong.
        assert "ghost-thought" in text
        # An actionable next step is offered.
        assert "search_memory" in text or "list_memory" in text
        _assert_no_leak(text)


class TestLinkMissingEndpoint:
    """Linking to an absent endpoint names the missing identifier."""

    async def test_missing_endpoint_reports_id_with_hint(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": "thought-alpha",
                    "to_thought_id": "ghost-endpoint",
                    "edge_type": "ASSOCIATED",
                },
            )

        assert result.is_error is True
        text = _error_text(result.content)
        # The dangling endpoint id is echoed ...
        assert "ghost-endpoint" in text
        # ... and the message leaks no path, stack frame, or class name.
        _assert_no_leak(text)


class TestDuplicateEdge:
    """A duplicate ``link_thoughts`` edge is mapped to a schema-free message."""

    async def test_duplicate_link_reports_clean_message_no_schema(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # The two seeded thoughts can be linked once; a second identical link
        # violates the (source, target, type) UNIQUE constraint. The raw
        # sqlite3 message names the edge table and its columns — the client
        # must instead get a curated, schema-free message.
        async with _client_for(store) as client:
            first = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": "thought-alpha",
                    "to_thought_id": "thought-beta",
                    "edge_type": "ASSOCIATED",
                },
            )
            assert first.is_error is False  # the first link succeeds

            duplicate = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": "thought-alpha",
                    "to_thought_id": "thought-beta",
                    "edge_type": "ASSOCIATED",
                },
            )

        assert duplicate.is_error is True
        text = _error_text(duplicate.content)
        # Actionable: it explains the uniqueness rule in user terms ...
        assert "already" in text.lower()
        # ... it is OUR curated wording, not the store's own phrasing ...
        assert DUPLICATE_EDGE_MESSAGE in text
        _assert_no_raw_duplicate_phrasing(text)
        # ... and leaks no table/column names, raw constraint text, or symbol.
        _assert_no_leak(text)

    async def test_typed_duplicate_error_maps_to_the_curated_message(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # The store signals a duplicate edge with a typed DuplicateEdgeError
        # whose own message spells out the endpoints in the store's phrasing.
        # That wording is the store's to change at will, so it must never reach
        # the client: the guard maps it to our curated message instead. This is
        # the regression guard for that mapping going stale.
        await link_thoughts_impl(store, "thought-alpha", "thought-beta", EdgeType.ASSOCIATED)

        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                await link_thoughts_impl(
                    store, "thought-alpha", "thought-beta", EdgeType.ASSOCIATED
                )

        text = str(excinfo.value)
        assert text == DUPLICATE_EDGE_MESSAGE
        _assert_no_raw_duplicate_phrasing(text)
        _assert_no_leak(text)

    async def test_both_duplicate_paths_are_indistinguishable(self) -> None:
        # A duplicate may surface as the typed error or as a raw UNIQUE
        # violation depending on the path taken; a client must not be able to
        # tell which happened.
        typed = DuplicateEdgeError("a", "b", "ASSOCIATED")
        raw = sqlite3.IntegrityError("UNIQUE constraint failed: edge.from_thought_id")

        messages: list[str] = []
        for failure in (typed, raw):
            with pytest.raises(ToolError) as excinfo:
                async with _tool_errors():
                    raise failure
            messages.append(str(excinfo.value))

        assert messages[0] == messages[1] == DUPLICATE_EDGE_MESSAGE


class TestInvalidFieldValue:
    """An invalid field value is mapped without leaking Pydantic internals."""

    async def test_empty_essence_reports_clean_validation_message(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # ``essence`` has a minimum length; an empty string fails domain-model
        # validation. The raw Pydantic error names the model class and links
        # its docs site — the client must get a curated message instead.
        async with _client_for(store) as client:
            result = await client.call_tool(
                "store_thought",
                {"essence": "", "content": "some content"},
            )

        assert result.is_error is True
        text = _error_text(result.content)
        # Actionable: it points at the offending field and says it is invalid ...
        assert "invalid" in text.lower()
        assert "essence" in text.lower()
        # ... and leaks no Pydantic URL, model class name, or symbol.
        _assert_no_leak(text)

    async def test_out_of_range_confidence_reports_clean_message(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # ``confidence`` is constrained to [0.0, 1.0] by the domain model (not
        # by the tool's argument schema), so an out-of-range value reaches the
        # tool body and is rejected there — exercising the ValidationError
        # mapping. The curated message names the field, never the model.
        async with _client_for(store) as client:
            result = await client.call_tool(
                "store_thought",
                {"essence": "ok", "content": "c", "confidence": 5.0},
            )

        assert result.is_error is True
        text = _error_text(result.content)
        assert "invalid" in text.lower()
        assert "confidence" in text.lower()
        _assert_no_leak(text)


class TestIllegalTransition:
    """An illegal lifecycle change is mapped without the internal type name."""

    async def test_backwards_transition_reports_clean_message(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # The seeded thoughts are ACTIVE; ACTIVE -> CREATED is backwards and
        # illegal. MCPServer coerces the wire status to the LifecycleStatus enum,
        # so the store's transition guard fires. The raw message names the
        # internal status type — the client must get a curated message instead.
        async with _client_for(store) as client:
            result = await client.call_tool(
                "update_thought",
                {"thought_id": "thought-alpha", "lifecycle_status": "CREATED"},
            )

        assert result.is_error is True
        text = _error_text(result.content)
        # Actionable: it states the move in plain terms (the public state names
        # are fine; the internal type name is not) ...
        assert "ACTIVE" in text
        assert "CREATED" in text
        assert "not allowed" in text.lower() or "cannot" in text.lower()
        # ... and leaks neither the raw "Invalid LifecycleStatus transition"
        # phrasing nor the exception class name.
        _assert_no_leak(text)

    async def test_illegal_transition_does_not_change_state(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # The rejection is real: the thought stays ACTIVE after an illegal
        # update attempt, confirming the guard blocks the write (not just the
        # message wrapper).
        async with _client_for(store) as client:
            await client.call_tool(
                "update_thought",
                {"thought_id": "thought-alpha", "lifecycle_status": "CREATED"},
            )
            after = await client.call_tool("get_thought", {"thought_id": "thought-alpha"})

        assert after.is_error is False
        assert after.structured_content is not None
        assert after.structured_content["thought"]["lifecycle_status"] == "ACTIVE"


class TestSuccessPathUnchanged:
    """Mapping errors must not alter what a successful tool call returns."""

    async def test_valid_find_still_succeeds(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # The seeded store has two ACTIVE thoughts; a valid FIND returns rows
        # with no error, confirming the wrapper is presentation-only.
        async with _client_for(store) as client:
            result = await client.call_tool(
                "query_memory",
                {"query": "FIND thoughts WHERE lifecycle_status = 'ACTIVE'"},
            )

        assert result.is_error is False
        assert result.structured_content is not None
        assert "thought_id" in result.structured_content["columns"]
        assert len(result.structured_content["rows"]) == 2

    async def test_valid_keyword_search_still_succeeds(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # A well-formed call through a wrapped read tool returns its normal
        # payload unchanged — the error wrapper adds nothing on the happy path.
        async with _client_for(store) as client:
            result = await client.call_tool("search_keywords", {"query": "coffee"})

        assert result.is_error is False
        assert result.structured_content is not None
        assert "results" in result.structured_content


class TestUnrecognisedIntegrityError:
    """A non-UNIQUE integrity error is re-raised unchanged, never masked."""

    async def test_non_unique_integrity_error_propagates_unchanged(self) -> None:
        # The wrapper maps only the UNIQUE (duplicate-edge) case to a curated
        # message; any other constraint violation must propagate as the raw
        # sqlite3.IntegrityError so it is never silently described or masked.
        original = sqlite3.IntegrityError("FOREIGN KEY constraint failed")
        with pytest.raises(sqlite3.IntegrityError) as excinfo:
            async with _tool_errors():
                raise original
        assert excinfo.value is original

    async def test_unique_integrity_error_is_mapped_to_tool_error(self) -> None:
        # Sanity foil to the re-raise case: a UNIQUE violation is mapped to a
        # curated ToolError rather than re-raised.
        unique_violation = sqlite3.IntegrityError("UNIQUE constraint failed: edges.x")
        with pytest.raises(ToolError):
            async with _tool_errors():
                raise unique_violation


class TestDuplicateThoughtId:
    """A caller-supplied thought_id that collides with an existing thought.

    engrava reports this as a bare ``ValueError`` ("Thought already exists:
    <id>"), not a dedicated type -- the store's contract is the message text,
    not a catchable class -- so the client must not see it raw.
    """

    async def test_duplicate_id_reports_clean_message_with_alternatives(
        self, store: SqliteEngravaCore
    ) -> None:
        async with _client_for(store) as client:
            first = await client.call_tool(
                "store_thought",
                {"essence": "first", "content": "first body", "thought_id": "dup-thought"},
            )
            assert first.is_error is False

            duplicate = await client.call_tool(
                "store_thought",
                {"essence": "second", "content": "second body", "thought_id": "dup-thought"},
            )

        assert duplicate.is_error is True
        text = _error_text(duplicate.content)
        # Names the offending id ...
        assert "dup-thought" in text
        assert "already exists" in text.lower()
        # ... and offers both actionable alternatives.
        assert "thought_id" in text
        assert "deduplicate" in text
        _assert_no_leak(text)

    async def test_duplicate_id_does_not_overwrite_the_existing_thought(
        self, store: SqliteEngravaCore
    ) -> None:
        await store_thought_impl(
            store, essence="original", content="original body", thought_id="dup-thought-2"
        )

        with pytest.raises(ToolError):
            async with _tool_errors():
                await store_thought_impl(
                    store,
                    essence="replacement",
                    content="replacement body",
                    thought_id="dup-thought-2",
                )

        read_back = await store.get_thought("dup-thought-2")
        assert read_back is not None
        assert read_back.essence == "original"

    async def test_deduplicate_true_is_unaffected(self, store: SqliteEngravaCore) -> None:
        # deduplicate=True resolves by content hash, a different path that
        # never reaches the id-collision ValueError -- the fix above must not
        # have coupled the two.
        content = "shared body for dedup-true regression"
        first = await store_thought_impl(store, essence="a", content=content, deduplicate=True)
        second = await store_thought_impl(store, essence="b", content=content, deduplicate=True)
        assert first["thought"]["thought_id"] == second["thought"]["thought_id"]


class TestConcurrentThoughtUpdate:
    """A concurrent write that loses the optimistic-concurrency guard.

    ``update_thought`` reads the thought's ``revision`` and guards its
    write on that value unconditionally -- not opt-in, and not exposed as a
    wire argument. Simulated here by advancing the stored row underneath the
    read ``update_thought`` already took, the same window a second real
    writer on a second store instance would race into (see
    ``StaleDataError``'s own docstring: a second store on the same database
    file is outside what the in-process write lock can reach).
    """

    @staticmethod
    def _install_racing_read(monkeypatch: pytest.MonkeyPatch, store: SqliteEngravaCore) -> None:
        """Make the next ``_get_thought_row`` also advance the real row.

        The snapshot ``update_thought`` reads is returned unchanged, so its
        own ``expected_cycle`` is now stale relative to the row it is about
        to guard its write against -- reproducing what a second writer
        landing in that exact window would do.
        """
        real_get_thought_row = type(store)._get_thought_row

        async def _racing_get_thought_row(self: SqliteEngravaCore, thought_id: str) -> object:
            row = await real_get_thought_row(self, thought_id)
            await self._db.execute(
                "UPDATE thought SET revision = revision + 1 WHERE thought_id = ?",
                (thought_id,),
            )
            return row

        monkeypatch.setattr(type(store), "_get_thought_row", _racing_get_thought_row)

    async def test_reports_actionable_retry_message_over_the_wire(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_racing_read(monkeypatch, store)

        async with _client_for(store) as client:
            result = await client.call_tool(
                "update_thought",
                {"thought_id": "thought-alpha", "essence": "raced update"},
            )

        assert result.is_error is True
        text = _error_text(result.content)
        assert "thought-alpha" in text
        assert "changed" in text.lower() or "modified" in text.lower()
        # Actionable: names the retry path.
        assert "get_thought" in text
        _assert_no_leak(text)

    async def test_raced_update_writes_nothing(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_racing_read(monkeypatch, store)

        with pytest.raises(ToolError):
            async with _tool_errors():
                await update_thought_impl(store, "thought-alpha", essence="raced update")

        read_back = await store.get_thought("thought-alpha")
        assert read_back is not None
        assert read_back.essence != "raced update"

    async def test_direct_stale_data_error_maps_to_the_same_message_shape(self) -> None:
        # Regression guard for the message contract itself, independent of
        # how the race is reproduced.
        err = StaleDataError(
            entity_type="ThoughtRecord", entity_id="thought-alpha", expected_version=3
        )
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                raise err
        text = str(excinfo.value)
        assert "thought-alpha" in text
        assert "get_thought" in text
        _assert_no_leak(text)


class TestOversizedEdgeMetadata:
    """Edge metadata above engrava's serialized-size limit.

    Reported as a bare ``ValueError`` shared with the thought-metadata path
    inside engrava, but ``store_thought`` / ``update_thought`` do not expose
    ``metadata`` on this wire surface at all -- ``link_thoughts`` is the only
    reachable caller, so the curated message stays edge-specific rather than
    echoing the store's thought-oriented "store it in content instead" advice.
    """

    async def test_oversized_metadata_reports_clean_edge_specific_message(
        self, store: SqliteEngravaCore
    ) -> None:
        oversized = {"blob": "x" * 70_000}
        async with _client_for(store) as client:
            result = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": "thought-alpha",
                    "to_thought_id": "thought-beta",
                    "edge_type": "ASSOCIATED",
                    "metadata": oversized,
                },
            )

        assert result.is_error is True
        text = _error_text(result.content)
        assert "metadata" in text.lower()
        assert "too large" in text.lower()
        # The two byte counts the store computed are surfaced ...
        assert "65536" in text
        # ... but the store's own "content" field advice is thought-specific
        # and must not appear on this edge-only path.
        assert "`content`" not in text
        _assert_no_leak(text)

    async def test_oversized_metadata_creates_no_edge(self, store: SqliteEngravaCore) -> None:
        oversized = {"blob": "x" * 70_000}
        with pytest.raises(ToolError):
            async with _tool_errors():
                await link_thoughts_impl(
                    store,
                    "thought-alpha",
                    "thought-beta",
                    EdgeType.ASSOCIATED,
                    metadata=oversized,
                )

        edges = await store.get_edges("thought-alpha", direction="OUT")
        assert edges == []

    async def test_in_range_metadata_is_unaffected(self, store: SqliteEngravaCore) -> None:
        result = await link_thoughts_impl(
            store,
            "thought-alpha",
            "thought-beta",
            EdgeType.ASSOCIATED,
            metadata={"topic": "regression check"},
        )
        assert result["edge"]["metadata"] == {"topic": "regression check"}

    async def test_message_shape_change_falls_back_to_a_generic_message(self) -> None:
        # Defense in depth for the regex extracting the two byte counts: if
        # engrava's message ever carries the "metadata serialized size" prefix
        # without the exact "N bytes exceeds maximum M bytes" shape the regex
        # expects, the branch still raises a clean, if less specific, message
        # rather than crashing on a failed match or falling through unmapped.
        reshaped = ValueError("metadata serialized size is too large now")
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                raise reshaped
        text = str(excinfo.value)
        assert "too large" in text.lower()
        _assert_no_leak(text)


class TestUnrelatedValueErrorIsNeverSwallowed:
    """A ``ValueError`` matching neither curated prefix propagates unchanged.

    The regression this guards: the plain ``ValueError`` branch added for (a)
    and (c) must re-raise anything that is not one of those two specific
    shapes, so a future, unrelated ``ValueError`` is never silently
    misdescribed or swallowed.
    """

    async def test_unrelated_value_error_propagates_unchanged(self) -> None:
        original = ValueError("some other failure entirely")
        with pytest.raises(ValueError, match="some other failure entirely") as excinfo:
            async with _tool_errors():
                raise original
        assert excinfo.value is original


class TestEdgeIdCollision:
    """A caller-supplied ``edge_id`` colliding with an existing edge's PK.

    Distinct from the (from, to, type) UNIQUE violation :class:`TestDuplicateEdge`
    covers: SQLite raises the same ``sqlite3.IntegrityError`` with "UNIQUE" in
    the text either way (a PRIMARY KEY is a UNIQUE index internally), but the
    two are different constraints with different causes -- attributing this
    one to the other tells the caller to change the edge type, which does
    nothing for an id collision.
    """

    async def test_colliding_edge_id_reports_the_real_cause(self, store: SqliteEngravaCore) -> None:
        await link_thoughts_impl(
            store,
            "thought-alpha",
            "thought-beta",
            EdgeType.ASSOCIATED,
            edge_id="edge-fixed-id",
        )

        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                # A different (from, to, type) -- would succeed on its own --
                # but the same edge_id, so only the PRIMARY KEY collides.
                await link_thoughts_impl(
                    store,
                    "thought-beta",
                    "thought-alpha",
                    EdgeType.DEPENDS_ON,
                    edge_id="edge-fixed-id",
                )

        text = str(excinfo.value)
        assert text == EDGE_ID_COLLISION_MESSAGE
        # It must NOT be attributed to the (from, to, type) duplicate cause,
        # nor suggest changing the edge type as the repair.
        assert text != DUPLICATE_EDGE_MESSAGE
        assert "already links those two thoughts" not in text
        assert "changing the edge type will not resolve" in text.lower()
        _assert_no_leak(text)

    async def test_colliding_edge_id_over_the_wire(self, store: SqliteEngravaCore) -> None:
        async with _client_for(store) as client:
            first = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": "thought-alpha",
                    "to_thought_id": "thought-beta",
                    "edge_type": "ASSOCIATED",
                    "edge_id": "edge-wire-fixed",
                },
            )
            assert first.is_error is False

            collision = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": "thought-beta",
                    "to_thought_id": "thought-alpha",
                    "edge_type": "DEPENDS_ON",
                    "edge_id": "edge-wire-fixed",
                },
            )

        assert collision.is_error is True
        text = _error_text(collision.content)
        assert EDGE_ID_COLLISION_MESSAGE in text
        _assert_no_leak(text)

    async def test_the_from_to_type_duplicate_is_still_correctly_attributed(
        self, store: SqliteEngravaCore
    ) -> None:
        # Regression guard: distinguishing the PK collision must not break
        # the pre-existing, correct (from, to, type) mapping it sits beside.
        await link_thoughts_impl(store, "thought-alpha", "thought-beta", EdgeType.ASSOCIATED)
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                await link_thoughts_impl(
                    store, "thought-alpha", "thought-beta", EdgeType.ASSOCIATED
                )
        assert str(excinfo.value) == DUPLICATE_EDGE_MESSAGE


class TestOverflowingNumericBound:
    """A wire-supplied integer that would overflow SQLite's own bind range.

    The primary fix is the type: ``offset`` / ``min_cycle`` / ``max_cycle``
    are now bound to ``[..., SQLITE_MAX_BOUND_INT]`` (see ``test_wire_bounds.py``
    for the wire-level rejection), so the values here should no longer reach
    a bind call at all. This class covers the ``_tool_errors`` translation
    directly, as defense in depth for any such parameter the type fix has not
    caught, or a future one that forgets it.
    """

    async def test_direct_overflow_error_maps_to_a_clean_message(self) -> None:
        raw = OverflowError("Python int too large to convert to SQLite INTEGER")
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                raise raw
        text = str(excinfo.value)
        assert str(SQLITE_MAX_BOUND_INT) in text
        _assert_no_leak(text)

    async def test_offset_at_the_old_overflow_value_is_now_rejected_at_the_impl_layer(
        self, store: SqliteEngravaCore
    ) -> None:
        # 2**63 is exactly the value the deep scan probed: it used to reach
        # sqlite3's bind call inside list_memory_impl and raise a raw
        # OverflowError. It is now rejected by _check_bound before the store
        # is ever touched, with the same clean message this class's first
        # test pins.
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                await server_module.list_memory_impl(store, offset=2**63)
        text = str(excinfo.value)
        assert "offset" in text
        assert str(SQLITE_MAX_BOUND_INT) in text
        _assert_no_leak(text)

    async def test_offset_at_the_old_overflow_value_is_rejected_over_the_wire(
        self, store: SqliteEngravaCore
    ) -> None:
        # Over the real MCP boundary this value is rejected by the advertised
        # schema before list_memory_impl ever runs -- a different layer from
        # the previous test, so only the rejection itself is asserted here.
        # NOTE: MCPServer's own argument-schema rejection message leaks
        # "pydantic" and a docs URL (confirmed pre-existing on every bound
        # argument already annotated before this WS, e.g. limit=10**18, not
        # something this change introduced) -- out of this WS's scope, which
        # is the _tool_errors translation table, not MCPServer's own protocol-
        # layer error formatting. Reported, not fixed here.
        async with _client_for(store) as client:
            result = await client.call_tool("list_memory", {"offset": 2**63})
        assert result.is_error is True


class TestEmbeddingQueryRefusal:
    """``FIND embeddings`` is refused outright rather than crashing on the blob.

    The crash this replaces happens *after* ``query_memory_impl`` returns --
    inside MCPServer's own response serialisation, once the returned dict's
    ``rows`` carry the embedding's raw ``bytes`` -- which is outside
    ``_tool_errors``'s ``try``/``except`` by the time it happens. Refusing the
    table at the query boundary, before any row is ever fetched, is the only
    place in this call this failure can be reached.
    """

    async def test_find_embeddings_is_refused_over_the_wire(self, store: SqliteEngravaCore) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool("query_memory", {"query": "FIND embeddings"})

        assert result.is_error is True
        text = _error_text(result.content)
        assert "embed" in text.lower()
        assert "thoughts" in text.lower()
        _assert_no_leak(text)

    async def test_find_embeddings_is_refused_even_with_a_stored_vector(
        self, store: SqliteEngravaCore
    ) -> None:
        # The exact scenario the deep scan probed: a real stored vector
        # present, so an unguarded query would fetch its raw bytes and crash
        # unmapped during serialisation instead of returning cleanly.
        await store.store_embedding("thought-alpha", [0.1, 0.2, 0.3])

        async with _client_for(store) as client:
            result = await client.call_tool("query_memory", {"query": "FIND embeddings"})

        assert result.is_error is True
        _assert_no_leak(_error_text(result.content))

    async def test_impl_raises_typed_error_for_the_embeddings_target(
        self, store: SqliteEngravaCore
    ) -> None:
        await store.store_embedding("thought-alpha", [0.1, 0.2, 0.3])
        with pytest.raises(EmbeddingQueryNotSupportedError):
            await query_memory_impl(store, "FIND embeddings")

    async def test_other_tables_are_unaffected(self, store: SqliteEngravaCore) -> None:
        result = await query_memory_impl(store, "FIND thoughts")
        assert isinstance(result["rows"], list)


class TestBusyStore:
    """A guarded write that could not start because another writer held the database.

    ``WriteContentionError`` is raised when a second process (or connection) holds
    the database's write lock past SQLite's own busy wait. Its docstring promises
    that nothing was written and that retrying the whole call is safe, so the
    message may say both. The server does not retry on its own.
    """

    @pytest.mark.parametrize(
        ("operation", "attempts"), [("create_thought", 3), ("update_thought", 1)]
    )
    async def test_direct_error_maps_to_a_retry_safe_message(
        self, operation: str, attempts: int
    ) -> None:
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                raise WriteContentionError(operation=operation, attempts=attempts)

        text = str(excinfo.value)
        lowered = text.lower()
        assert "busy" in lowered
        assert "another writer" in lowered
        assert "nothing was changed" in lowered
        assert "retry" in lowered
        # The library's own bookkeeping and SQLite's vocabulary stay out.
        assert operation not in text
        assert "attempt" not in lowered
        assert "lock" not in lowered
        assert "sqlite" not in lowered
        assert "BEGIN" not in text
        _assert_no_leak(text)

    async def test_update_thought_reports_the_busy_store_over_the_wire(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _contended_update(
            self: SqliteEngravaCore, thought_id: str, **changes: object
        ) -> NoReturn:
            raise WriteContentionError(operation="update_thought", attempts=1)

        monkeypatch.setattr(type(store), "update_thought", _contended_update)

        async with _client_for(store) as client:
            result = await client.call_tool(
                "update_thought",
                {"thought_id": "thought-alpha", "essence": "contended update"},
            )

        # State first: the refused update wrote nothing.
        read_back = await store.get_thought("thought-alpha")
        assert read_back is not None
        assert read_back.essence != "contended update"

        assert result.is_error is True
        text = _error_text(result.content)
        assert "busy" in text.lower()
        assert "nothing was changed" in text.lower()
        assert "retry" in text.lower()
        _assert_no_leak(text)


class TestWriteTimeout:
    """A write that timed out waiting for a long-running write ahead of it.

    ``WriteLockTimeoutError`` is raised when a task cannot get the store's
    in-process write lock within its bound. The library does not promise that
    the timed-out call left nothing behind, so -- unlike the busy-store message
    -- the wording must not claim that, and must tell the caller to check first.
    """

    async def test_direct_error_maps_to_a_check_before_repeating_message(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                raise WriteLockTimeoutError(timeout_seconds=600.0)

        text = str(excinfo.value)
        lowered = text.lower()
        assert "long-running write" in lowered
        assert "timed out" in lowered
        assert "retry" in lowered
        assert "read the affected thought back" in lowered
        # No promise the library does not make ...
        assert "nothing" not in lowered
        assert "safe" not in lowered
        # ... and none of its internals.
        assert "600" not in text
        assert "second" not in lowered
        assert "lock" not in lowered
        assert "suspend_auto_commit" not in text
        _assert_no_leak(text)

    async def test_a_write_waiting_past_the_bound_reports_it_over_the_wire(self) -> None:
        # A real reproduction, not a stand-in: a store whose write-lock bound is
        # tiny, and a second task that holds a suspend_auto_commit() window open.
        # The tool call runs in a different task from the holder, so it can
        # neither share the hold nor outwait it.
        connection = await aiosqlite.connect(":memory:")
        connection.row_factory = aiosqlite.Row
        backend = SqliteEngravaCore(connection, write_lock_acquire_timeout_seconds=0.05)
        await backend.ensure_schema()
        window_open = asyncio.Event()
        close_window = asyncio.Event()

        async def _hold_the_window() -> None:
            async with backend.suspend_auto_commit():
                window_open.set()
                await close_window.wait()

        holder = asyncio.create_task(_hold_the_window())
        try:
            await window_open.wait()
            async with _client_for(backend) as client:
                result = await client.call_tool(
                    "store_thought",
                    {
                        "essence": "waited too long",
                        "content": "This write queues behind the open window.",
                        "thought_id": "thought-timed-out",
                    },
                )
            close_window.set()
            await holder

            assert await backend.get_thought("thought-timed-out") is None
        finally:
            close_window.set()
            await holder
            await connection.close()

        assert result.is_error is True
        text = _error_text(result.content)
        assert "long-running write" in text.lower()
        assert "read the affected thought back" in text.lower()
        assert "nothing" not in text.lower()
        _assert_no_leak(text)


class TestUnusableStore:
    """The store's connection was quarantined after an unrecoverable rollback failure.

    ``ConnectionQuarantinedError`` is terminal for the server's store instance:
    every later operation fails fast with it, and only a restart recovers. The
    message says so, and does not echo the quarantine's ``reason``.
    """

    async def test_direct_error_maps_to_a_restart_message(self) -> None:
        reason = "open transaction left behind by a failed rollback"
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                raise ConnectionQuarantinedError(reason)

        text = str(excinfo.value)
        lowered = text.lower()
        assert "unusable" in lowered
        assert "restarted" in lowered
        assert "retrying will not help" in lowered
        assert "whoever operates the server" in lowered
        # The reason is the library's own diagnostic text.
        assert reason not in text
        assert "quarantine" not in lowered
        _assert_no_leak(text)

    async def test_a_read_tool_reports_the_unusable_store_over_the_wire(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reason = "open transaction left behind by a failed rollback"

        async def _quarantined_read(self: SqliteEngravaCore, thought_id: str) -> NoReturn:
            raise ConnectionQuarantinedError(reason)

        monkeypatch.setattr(type(store), "get_thought", _quarantined_read)

        async with _client_for(store) as client:
            result = await client.call_tool("get_thought", {"thought_id": "thought-alpha"})

        assert result.is_error is True
        text = _error_text(result.content)
        assert "unusable" in text.lower()
        assert "restarted" in text.lower()
        assert reason not in text
        _assert_no_leak(text)


class _FixedDimensionProvider:
    """A minimal embedding-provider double that produces a fixed-size vector.

    Satisfies only the mandatory ``EmbeddingProviderProtocol`` (``dimension``,
    ``model_name``, ``embed``, ``embed_batch``) -- no ``query_prefix`` /
    ``document_prefix`` / role methods -- so ``_role_prefixes`` reports it
    unprefixed and every embed call goes through the plain ``embed`` path.
    """

    def __init__(self, dimension: int, *, model_name: str = "test-provider") -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        del text  # unused: this double ignores query content
        return [0.1] * self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


class _RoleAwareProvider(_FixedDimensionProvider):
    """A ``RoleAwareEmbeddingProvider`` double with a configurable query prefix.

    Implements the full role-aware capability (both prefixes and every role
    method), so ``engrava``'s ``isinstance`` capability check picks it up and
    ``search_hybrid`` embeds queries through ``embed_query`` -- the path
    ``_ensure_query_prefix_pairs`` guards.
    """

    def __init__(
        self,
        dimension: int,
        *,
        query_prefix: str,
        document_prefix: str = "passage: ",
        model_name: str = "test-role-aware-provider",
    ) -> None:
        super().__init__(dimension, model_name=model_name)
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    @property
    def document_prefix(self) -> str:
        return self._document_prefix

    async def embed_query(self, text: str) -> list[float]:
        return await self.embed(text)

    async def embed_document(self, text: str) -> list[float]:
        return await self.embed(text)

    async def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_batch(texts)

    async def embed_document_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_batch(texts)


class TestVectorDimensionMismatch:
    """The configured provider's vectors no longer match what the store holds.

    ``VectorDimensionMismatchError`` is reachable through search_memory
    whenever an embedding provider is configured: the tool embeds the query
    text itself with whatever provider is currently configured, so a size
    mismatch surfaces the moment the engrava.yaml is repointed at a different
    embedding model after the store already has vectors of the old size. No
    tool argument names a vector, so nothing a caller supplies can trigger or
    avoid this -- only the server's own embedding configuration can.
    """

    async def test_direct_error_maps_to_a_configuration_message(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                raise VectorDimensionMismatchError(expected=384, actual=768)

        text = str(excinfo.value)
        lowered = text.lower()
        assert "search_memory" in text
        assert "different size" in lowered
        assert "no result was returned" in lowered
        assert "nothing was changed" in lowered
        assert "retrying will not help" in lowered
        assert "whoever operates this server" in lowered
        # The raw dimension integers are the store's and the provider's own
        # numbers, not anything a caller can act on.
        assert "384" not in text
        assert "768" not in text
        _assert_no_leak(text)

    async def test_search_memory_reports_the_mismatch_over_the_wire(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A real reproduction, not a stand-in: store_embedding locks the
        # store's corpus at a 3-dimensional vector with no provider involved,
        # then the store's provider is reconfigured to one that produces
        # 5-dimensional vectors -- exactly what an engrava.yaml repointed at a
        # different embedding model after the store already has vectors
        # looks like.
        await store.store_embedding("thought-alpha", [0.1, 0.2, 0.3])
        monkeypatch.setattr(store, "_embedding_provider", _FixedDimensionProvider(5))

        async with _client_for(store) as client:
            result = await client.call_tool("search_memory", {"query_text": "coffee"})

        # State first: the failed search changed nothing -- the locked
        # embedding is exactly the one stored above, untouched. (The query
        # text is non-empty and FTS5 is available, so a real lexical pass
        # over the corpus already ran internally before the vector arm
        # raised; that pass is read-only and never reaches this assertion.)
        embedding = await store.get_embedding("thought-alpha")
        assert embedding is not None
        assert embedding.dimension == 3

        assert result.is_error is True
        text = _error_text(result.content)
        assert "search_memory" in text
        assert "different size" in text.lower()
        assert "no result was returned" in text.lower()
        assert "nothing was changed" in text.lower()
        _assert_no_leak(text)


class TestEmbeddingQueryPrefixMismatch:
    """The active query prefix no longer pairs with the stored corpus.

    ``EmbeddingQueryPrefixMismatchError`` is reachable through search_memory
    only for an asymmetric embedding model: it fires when the provider's
    active query prefix no longer matches the one the corpus's vectors were
    embedded to pair with, e.g. the engrava.yaml was repointed at a different
    prefix configuration after the store already has vectors. Neither prefix
    is a tool argument, so nothing a caller supplies can trigger or avoid
    this -- only the server's own embedding configuration can.
    """

    async def test_direct_error_maps_to_a_configuration_message(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                raise EmbeddingQueryPrefixMismatchError(
                    stored_query_prefix="query: ",
                    configured_query_prefix="search_query: ",
                )

        text = str(excinfo.value)
        lowered = text.lower()
        assert "search_memory" in text
        assert "query prefix" in lowered
        assert "nothing was searched" in lowered
        assert "retrying will not help" in lowered
        assert "whoever operates this server" in lowered
        # The raw stored/configured prefix values are the library's own.
        assert "query: " not in text
        assert "search_query: " not in text
        _assert_no_leak(text)

    async def test_search_memory_reports_the_mismatch_over_the_wire(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A real reproduction, not a stand-in: configure a role-aware
        # provider, lock the store's corpus with it (which records its query
        # prefix in _metadata), then reconfigure the store's provider to one
        # with a different query prefix but the same dimension -- exactly
        # what an engrava.yaml repointed at a different prefix configuration
        # after the store already has vectors looks like. Keeping the
        # dimension equal isolates the prefix mismatch from a dimension one.
        monkeypatch.setattr(
            store, "_embedding_provider", _RoleAwareProvider(4, query_prefix="query: ")
        )
        await store.store_embedding(
            "thought-alpha", [0.1, 0.2, 0.3, 0.4], model_name="role-aware-model"
        )
        monkeypatch.setattr(
            store,
            "_embedding_provider",
            _RoleAwareProvider(4, query_prefix="search_query: "),
        )

        async with _client_for(store) as client:
            result = await client.call_tool("search_memory", {"query_text": "coffee"})

        assert result.is_error is True
        text = _error_text(result.content)
        assert "search_memory" in text
        assert "query prefix" in text.lower()
        assert "nothing was searched" in text.lower()
        _assert_no_leak(text)


# ---------------------------------------------------------------------------
# store_thought's derive-records failures.
# ---------------------------------------------------------------------------


class _DeriveHooks(DefaultEngravaHooks):
    """Operator-supplied hooks double that also produces derived records.

    A real implementation of :class:`~engrava.DerivedRecordProducerProtocol`
    (detected structurally by ``isinstance`` -- it is a
    ``@runtime_checkable`` ``Protocol`` -- so no mock is needed), configured
    per test to reproduce exactly one of the four failure shapes this WS
    covers.

    ``derive_records`` only produces (or raises) anything for the one
    source content each test configures as its trigger; every other create
    on the same store -- in particular a test's own setup call that
    pre-seeds an unrelated, foreign thought at a chosen id -- derives
    nothing, so that setup call is a plain, inert insert.
    """

    def __init__(
        self,
        *,
        trigger_content: str,
        records: Sequence[DerivedRecord] = (),
        raises: BaseException | None = None,
    ) -> None:
        self._trigger_content = trigger_content
        self._records = records
        self._raises = raises

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        """Return the configured records, raise, or no-op for a non-trigger source."""
        if thought.content != self._trigger_content:
            return ()
        if self._raises is not None:
            raise self._raises
        return self._records


@asynccontextmanager
async def _derive_store(
    hooks: _DeriveHooks, *, max_derived_per_source: int = 32
) -> AsyncIterator[SqliteEngravaCore]:
    """Build a fresh in-memory store with derived records enabled and raising.

    ``on_error`` is always ``"raise"`` here -- the only policy under which
    any of this WS's four cases reach a client at all; the default,
    ``"log"``, swallows every one of them internally (logged, source left
    durable, remaining children/derivation simply skipped) and never raises
    through this path.

    Args:
        hooks: The producer double to install.
        max_derived_per_source: The over-cap threshold to configure.

    Yields:
        A schema-initialised store, closed on exit.

    """
    connection = await aiosqlite.connect(":memory:")
    connection.row_factory = aiosqlite.Row
    backend = SqliteEngravaCore(
        connection,
        hooks=hooks,
        derive_gates=DeriveGates(
            enabled=True,
            on_error="raise",
            max_derived_per_source=max_derived_per_source,
        ),
    )
    await backend.ensure_schema()
    try:
        yield backend
    finally:
        await connection.close()


class TestDerivedRecordOverCap:
    """A derive-records producer returns more records than the configured cap.

    ``_collect_derived`` rejects the over-cap return -- as a typed
    ``DerivedRecordError`` -- before any child is written, but strictly
    after the source thought's own insert has already committed (derivation
    dispatches only once ``_finish_create_thought`` runs, itself only
    reached after ``_insert_new_thought_row``'s own ``_maybe_commit``).
    """

    async def test_reports_a_clean_message_over_the_wire(self) -> None:
        hooks = _DeriveHooks(
            trigger_content="over-cap source content",
            records=[
                DerivedRecord(
                    content="over-cap derived child one",
                    thought_type=ThoughtType.OBSERVATION,
                    priority=Priority.P3,
                ),
                DerivedRecord(
                    content="over-cap derived child two",
                    thought_type=ThoughtType.OBSERVATION,
                    priority=Priority.P3,
                ),
            ],
        )
        async with _derive_store(hooks, max_derived_per_source=1) as backend:
            async with _client_for(backend) as client:
                result = await client.call_tool(
                    "store_thought",
                    {
                        "essence": "over-cap source",
                        "content": "over-cap source content",
                        "thought_id": "over-cap-source",
                    },
                )

            # State first: the source thought is durable even though its
            # derived records were rejected.
            source = await backend.get_thought("over-cap-source")
            assert source is not None
            assert source.content == "over-cap source content"

            assert result.is_error is True
            text = _error_text(result.content)
            assert "stored" in text.lower()
            assert "derived record" in text.lower()
            _assert_no_leak(text)

    async def test_direct_error_maps_to_the_same_message_shape(self) -> None:
        # Regression guard for the message contract itself, independent of
        # how the over-cap condition is reproduced.
        err = DerivedRecordError(
            "over-cap-source",
            "producer returned more than max_derived_per_source=1 records",
        )
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                raise err
        text = str(excinfo.value)
        lowered = text.lower()
        assert "stored" in lowered
        assert "derived record" in lowered
        assert "over-cap-source" not in text
        _assert_no_leak(text)


class TestDerivedRecordSelfCollision:
    """A derived record's identity collides with its own source thought.

    ``_persist_derived_child`` raises before any database work for this
    child -- a pure pre-check on the deterministic, content-addressed id --
    but the source thought itself is already durably committed by the time
    derivation (and this check) ever runs.
    """

    async def test_reports_a_clean_message_over_the_wire(self) -> None:
        child_content = "self-collision derived content"
        # engrava's own private derived-id function, imported to construct a
        # real, deterministic id collision -- not reimplemented, so a future
        # change to the hashing scheme cannot silently produce a false pass.
        colliding_id = _derived_thought_id(child_content)
        hooks = _DeriveHooks(
            trigger_content="self-collision source content",
            records=[
                DerivedRecord(
                    content=child_content,
                    thought_type=ThoughtType.OBSERVATION,
                    priority=Priority.P3,
                )
            ],
        )
        async with _derive_store(hooks) as backend:
            async with _client_for(backend) as client:
                result = await client.call_tool(
                    "store_thought",
                    {
                        "essence": "self-collision source",
                        "content": "self-collision source content",
                        "thought_id": colliding_id,
                    },
                )

            source = await backend.get_thought(colliding_id)
            assert source is not None
            assert source.content == "self-collision source content"

            assert result.is_error is True
            text = _error_text(result.content)
            assert "stored" in text.lower()
            assert "derived record" in text.lower()
            _assert_no_leak(text)


class TestDerivedRecordForeignCollision:
    """A derived record's identity collides with an unrelated stored thought.

    The pre-existing thought's stored content differs from what the
    producer derived, so the conflict-as-reuse hit is treated as a
    collision (no provenance edge attached) rather than silently asserting
    a false "derived from" relationship against someone else's row.
    """

    async def test_reports_a_clean_message_and_leaves_the_foreign_thought_untouched(
        self,
    ) -> None:
        child_content = "foreign-collision derived content"
        foreign_id = _derived_thought_id(child_content)
        hooks = _DeriveHooks(
            trigger_content="foreign-collision source content",
            records=[
                DerivedRecord(
                    content=child_content,
                    thought_type=ThoughtType.OBSERVATION,
                    priority=Priority.P3,
                )
            ],
        )
        async with _derive_store(hooks) as backend:
            # The pre-existing, unrelated thought at the derived child's own
            # deterministic id. derive_records returns nothing for it (its
            # content is not the configured trigger), so this setup insert
            # does not itself dispatch a colliding derivation.
            await backend.create_thought(
                make_thought(
                    foreign_id,
                    essence="unrelated foreign thought",
                    content="unrelated pre-existing content",
                )
            )

            async with _client_for(backend) as client:
                result = await client.call_tool(
                    "store_thought",
                    {
                        "essence": "foreign-collision source",
                        "content": "foreign-collision source content",
                        "thought_id": "foreign-collision-source",
                    },
                )

            # State first: the source thought is durable, and the unrelated
            # foreign thought was reused, not overwritten with the derived
            # content or given a false provenance edge.
            source = await backend.get_thought("foreign-collision-source")
            assert source is not None
            assert source.content == "foreign-collision source content"
            foreign = await backend.get_thought(foreign_id)
            assert foreign is not None
            assert foreign.content == "unrelated pre-existing content"

            assert result.is_error is True
            text = _error_text(result.content)
            assert "stored" in text.lower()
            assert "derived record" in text.lower()
            _assert_no_leak(text)


class TestDeriveProducerException:
    """The derive-records producer's own ``derive_records`` call raises.

    Not a ``DerivedRecordError`` -- the producer's own exception type,
    which engrava's ``_collect_derived`` re-raises bare (unwrapped) under
    ``on_error="raise"``. Its type belongs to an operator-supplied
    extension and cannot be named in a ``_tool_errors`` except clause in
    advance, so :func:`engrava_mcp.server._derive_producer_guard` wraps it
    in :class:`engrava_mcp.server._DeriveProducerFailedError` first.
    """

    async def test_reports_a_clean_message_and_never_leaks_the_producers_own_text(
        self,
    ) -> None:
        class _ProducerBoomError(RuntimeError):
            """The producer's own exception type -- unknown to this server."""

        producer_diagnostic = "proprietary producer diagnostic, never for the client"
        hooks = _DeriveHooks(
            trigger_content="producer-exception source content",
            raises=_ProducerBoomError(producer_diagnostic),
        )
        async with _derive_store(hooks) as backend:
            async with _client_for(backend) as client:
                result = await client.call_tool(
                    "store_thought",
                    {
                        "essence": "producer-exception source",
                        "content": "producer-exception source content",
                        "thought_id": "producer-exception-source",
                    },
                )

            source = await backend.get_thought("producer-exception-source")
            assert source is not None
            assert source.content == "producer-exception source content"

            assert result.is_error is True
            text = _error_text(result.content)
            assert "stored" in text.lower()
            assert "derived record" in text.lower()
            assert producer_diagnostic not in text
            assert "_ProducerBoomError" not in text
            _assert_no_leak(text)

    async def test_direct_error_maps_to_the_same_message_shape(self) -> None:
        # Regression guard for the message contract itself, independent of
        # how the producer failure is reproduced.
        with pytest.raises(ToolError) as excinfo:
            async with _tool_errors():
                raise _DeriveProducerFailedError
        text = str(excinfo.value)
        lowered = text.lower()
        assert "stored" in lowered
        assert "derived record" in lowered
        _assert_no_leak(text)


#: Namespaces :func:`_resolve_exception_class` searches, in order, to turn an
#: except-clause identifier into the class object it names. ``sqlite3`` is
#: needed for ``sqlite3.IntegrityError`` (an attribute access, which
#: :func:`_except_clause_type_names` resolves to just its trailing
#: identifier).
_DERIVE_GUARD_RESOLUTION_NAMESPACES: tuple[object, ...] = (server_module, builtins, sqlite3)


def _resolve_exception_class(
    name: str, namespaces: tuple[object, ...]
) -> type[BaseException] | None:
    """Resolve an identifier to the class object it names, in ``namespaces``.

    Args:
        name: An identifier an ``except`` clause spelled (see
            :func:`_except_clause_type_names`).
        namespaces: Objects to look ``name`` up on, in order; the first hit
            that is actually a class wins.

    Returns:
        The resolved class, or ``None`` if no namespace has it as a class.

    """
    for namespace in namespaces:
        candidate = getattr(namespace, name, None)
        if isinstance(candidate, type):
            return candidate
    return None


def _uncovered_derive_guard_names(
    named: set[str],
    *,
    exempt: set[str],
    covered: tuple[type[BaseException], ...],
    namespaces: tuple[object, ...],
) -> tuple[list[str], list[str]]:
    """Names in ``named`` that ``covered`` does not account for.

    The pure check both the real completeness test and its own failability
    demonstration call, so the demonstration exercises the actual logic
    rather than a copy of it.

    Args:
        named: Identifiers named in ``_tool_errors``'s own except clauses.
        exempt: Names to skip regardless (the guard's own output type).
        covered: The exclusion tuple to check ``issubclass`` membership
            against.
        namespaces: Objects to resolve each name against, in order.

    Returns:
        A ``(unresolved, uncovered)`` pair, both sorted: names that could not
        be resolved to a class at all, and resolved names ``covered`` does
        not account for.

    """
    unresolved: list[str] = []
    uncovered: list[str] = []
    for name in sorted(named - exempt):
        cls = _resolve_exception_class(name, namespaces)
        if cls is None:
            unresolved.append(name)
        elif not issubclass(cls, covered):
            uncovered.append(name)
    return unresolved, uncovered


class TestDeriveProducerGuardExclusionCompleteness:
    """``_ALREADY_HANDLED_DERIVE_EXCEPTIONS`` must cover every ``_tool_errors`` branch.

    The exclusion tuple :func:`engrava_mcp.server._derive_producer_guard`
    checks against is hand-maintained, not derived from ``_tool_errors``
    itself -- so it can silently fall out of sync with it: a type that gets
    its own ``_tool_errors`` branch in the future but is never added here
    would, if ever raised from inside the guarded ``store.create_thought``
    call, be wrapped as ``_DeriveProducerFailedError`` and produce the wrong
    client-facing message instead of reaching its own handler. This test
    makes that drift fail loudly: every type named in a ``_tool_errors``
    except clause must be covered (``issubclass``) by
    ``_ALREADY_HANDLED_DERIVE_EXCEPTIONS`` -- the one exemption is
    ``_DeriveProducerFailedError`` itself, the guard's own *output* type,
    never something ``store.create_thought`` can raise as an input to it.
    """

    def test_every_tool_errors_branch_is_covered_by_the_derive_guard_exclusion(self) -> None:
        named = _except_clause_type_names(server_module._tool_errors)
        unresolved, uncovered = _uncovered_derive_guard_names(
            named,
            exempt={"_DeriveProducerFailedError"},
            covered=server_module._ALREADY_HANDLED_DERIVE_EXCEPTIONS,
            namespaces=_DERIVE_GUARD_RESOLUTION_NAMESPACES,
        )

        assert not unresolved, (
            f"could not resolve {unresolved} to a class via engrava_mcp.server, "
            "builtins, or sqlite3 -- extend _DERIVE_GUARD_RESOLUTION_NAMESPACES"
        )
        assert not uncovered, (
            f"_tool_errors names {uncovered} in its own except clauses, but "
            "_ALREADY_HANDLED_DERIVE_EXCEPTIONS does not cover it -- "
            "_derive_producer_guard could wrap a real instance as "
            "_DeriveProducerFailedError instead of letting it reach its own "
            "_tool_errors branch. Add it to _ALREADY_HANDLED_DERIVE_EXCEPTIONS."
        )

    def test_sweep_fails_on_a_deliberately_uncovered_type(self) -> None:
        # Proves the completeness check above actually catches a gap, on the
        # exact shape of gap this test class exists to prevent: a real
        # _tool_errors branch (MindQLParseError, resolvable via this test
        # module's own import of it) for a type a too-narrow exclusion tuple
        # misses.
        unresolved, uncovered = _uncovered_derive_guard_names(
            {"MindQLParseError"},
            exempt=set(),
            covered=(ValueError,),
            namespaces=(sys.modules[__name__],),
        )
        assert unresolved == []
        assert uncovered == ["MindQLParseError"]

    def test_sweep_passes_once_the_type_is_added_to_the_exclusion(self) -> None:
        # Foil to the previous test: adding the missing type to the covered
        # tuple clears the gap, confirming the check treats that as fixed.
        unresolved, uncovered = _uncovered_derive_guard_names(
            {"MindQLParseError"},
            exempt=set(),
            covered=(ValueError, MindQLParseError),
            namespaces=(sys.modules[__name__],),
        )
        assert unresolved == []
        assert uncovered == []

    def test_sweep_ignores_an_exempted_name_regardless_of_coverage(self) -> None:
        # _DeriveProducerFailedError itself must never be flagged even
        # though it is not (and must not be) in the exclusion tuple.
        unresolved, uncovered = _uncovered_derive_guard_names(
            {"MindQLParseError", "_DeriveProducerFailedError"},
            exempt={"_DeriveProducerFailedError"},
            covered=(ValueError, MindQLParseError),
            namespaces=(sys.modules[__name__],),
        )
        assert unresolved == []
        assert uncovered == []


# ---------------------------------------------------------------------------
# Deliverable 5: the exception-surface sweep.
# ---------------------------------------------------------------------------

#: Every public exception type ``engrava.domain.exceptions`` defines that
#: ``_tool_errors`` deliberately does not map, with the reason. An entry
#: disappears from here only when a ``_tool_errors`` branch maps it -- never
#: by silent deletion. Scoped to ``engrava.domain.exceptions`` specifically
#: (not the MindQL parser's own exceptions, a materially smaller and already
#: separately-tested surface) because that module is what "engrava's public
#: exception types" means in the WS this sweep exists to serve: it is where
#: the count of defined types was taken from, and where it must be re-taken.
_OUT_OF_SCOPE: dict[str, str] = {
    "EngravaError": ("abstract base; engrava raises a concrete subclass, never this type itself"),
    "ActionNotFoundError": "no MCP tool touches the Action domain",
    "ReadOnlyViolationError": (
        "read-only mode is enforced by not registering write tools and by "
        "engrava_mcp's own ReadOnlyStore, never by engrava's ReadOnlyEngrava -- "
        "this type cannot be raised through this server"
    ),
    "EmbeddingProviderContractError": (
        "raised when the store needs an embedding provider's dimension (for a "
        "search) and the provider has no public one; the only providers this "
        "server can be given, through an engrava.yaml, are the four built-in "
        "ones, and all four have it"
    ),
    "EmbeddingModelMismatchError": (
        "raised by the first embedding a store writes (store_thought or "
        "update_thought, when the engrava.yaml turns on embeddings.auto_embed "
        "with a provider), not when the store opens -- nothing on this server's "
        "startup path checks the model; the check compares the embedding metadata "
        "the database already stores (model name, vector length, document-prefix "
        "fingerprint) with what the configured provider actually produces at that "
        "first write, and raises when they differ, whatever made them differ"
    ),
    "EmbeddingGenerationError": (
        "raised by store_thought or update_thought only when the engrava.yaml "
        "this server is pointed at turns on embeddings.auto_embed and "
        "embeddings.require_embedding and the embedding provider then fails; "
        "with either off, or with no provider (the bare ENGRAVA_DB_PATH launch), "
        "this type is never raised"
    ),
    "JournalIntegrityError": (
        "raised only by from_config's on-open journal verification, before the "
        "lifespan starts serving any tool call"
    ),
    "ExtensionMigrationError": "requires an installed extension; this server registers none",
    "CoreMigrationError": (
        "raised when a schema-migration step leaves its target structure missing; "
        "migrations run when the store opens, before any tool call, and the one "
        "step re-run later (at the first embedding a store writes) creates a "
        "small bookkeeping table and checks it exists, so it can only raise this "
        "if that create silently did nothing"
    ),
    "SchemaVersionError": (
        "raised only when the store is opened, which the server does once at "
        "startup: a database this build refuses to open (an old layout that "
        "already holds data, an old layout whose empty tables have an outdated "
        "shape, or a file written by a newer engrava) stops the server from "
        "starting rather than failing a tool call"
    ),
    "SourceThoughtNotFoundError": (
        "requires the derived-records backfill entry point; no tool calls it"
    ),
    "CycleProviderError": "requires a configured cycle provider; this server configures none",
    "RecencyModeConflictError": (
        "requires an explicit current_cycle together with recency_now; "
        "current_cycle is never wire-exposed, so no call can construct the conflict"
    ),
    "DedupLockReentryError": (
        "requires a store subclass that overrides a method the deduplicating "
        "create runs and re-enters that create from it; the server builds the "
        "library's own store class directly, so no override can be installed, and "
        "the window itself runs none of the configured hooks (on_store runs after "
        "both of its locks are released)"
    ),
}


def _except_clause_type_names(func: object) -> set[str]:
    """Every exception-type identifier a function's own ``except`` clauses name.

    Reads the identifier text an ``except`` clause spells, not what it
    resolves to -- resolution against the real class objects happens
    separately, in :func:`_handled_public_exception_names`, so a same-named
    but unrelated symbol cannot masquerade as "handled".

    Args:
        func: The function to inspect (its live source is read via
            :func:`inspect.getsource`, so edits to it are picked up without
            this test file changing).

    Returns:
        The set of identifiers named in ``except X`` / ``except (X, Y)`` /
        ``except X as e`` clauses anywhere in the function body.

    """
    source = textwrap.dedent(inspect.getsource(func))  # type: ignore[arg-type]
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            candidates = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            for candidate in candidates:
                if isinstance(candidate, ast.Name):
                    names.add(candidate.id)
                elif isinstance(candidate, ast.Attribute):
                    names.add(candidate.attr)
    return names


def _engrava_public_exception_types() -> dict[str, type[EngravaError]]:
    """Every public exception type ``engrava.domain.exceptions`` defines.

    Returns:
        A mapping of class name to class object, for every class the module
        itself defines (``__module__`` matches -- an imported re-export from
        elsewhere would not count).

    """
    return {
        name: obj
        for name, obj in inspect.getmembers(engrava_exceptions, inspect.isclass)
        if obj.__module__ == engrava_exceptions.__name__
    }


def _handled_public_exception_names(public_types: dict[str, type[EngravaError]]) -> set[str]:
    """Names from ``public_types`` that ``_tool_errors`` actually maps.

    A name is "handled" only when it is both named in one of
    ``_tool_errors``'s own ``except`` clauses AND bound, in
    :mod:`engrava_mcp.server`'s own module namespace, to the exact class
    object ``public_types`` gives it -- not merely text-matched -- so an
    unrelated same-named symbol cannot pass as coverage.

    Args:
        public_types: The mapping from :func:`_engrava_public_exception_types`.

    Returns:
        The subset of ``public_types`` keys ``_tool_errors`` maps.

    """
    named = _except_clause_type_names(server_module._tool_errors)
    return {
        name
        for name in named
        if name in public_types and getattr(server_module, name, None) is public_types[name]
    }


def _unaccounted_exception_types(
    public_types: dict[str, object],
    handled: set[str],
    out_of_scope: dict[str, str],
) -> list[str]:
    """Public type names that are neither mapped nor excused.

    The pure check both the real sweep and its own failability demonstration
    call, so the demonstration exercises the actual logic rather than a copy
    of it.

    Args:
        public_types: Name -> class for every type under sweep.
        handled: Names a ``_tool_errors`` branch maps.
        out_of_scope: Names excused, with a reason.

    Returns:
        Sorted names present in ``public_types`` but absent from both
        ``handled`` and ``out_of_scope``.

    """
    return sorted(name for name in public_types if name not in handled and name not in out_of_scope)


class TestExceptionSurfaceSweep:
    """``_tool_errors`` must map or explicitly excuse every public engrava
    exception type.

    engrava's exception surface grows behind this server's back: a type added
    upstream that this table does not recognise crosses the wire as
    ``Error executing tool <name>: <internal message>`` unless and until
    someone happens to probe it by hand -- exactly how the six cases this WS
    closes were found. This test makes that discovery automatic instead of
    incidental: every class ``engrava.domain.exceptions`` defines must either
    be named in a ``_tool_errors`` except clause, or be listed in
    ``_OUT_OF_SCOPE`` with a reason. Silence is not a valid third state.
    """

    def test_every_public_exception_is_mapped_or_excused(self) -> None:
        public_types = _engrava_public_exception_types()
        handled = _handled_public_exception_names(public_types)
        unaccounted = _unaccounted_exception_types(public_types, handled, _OUT_OF_SCOPE)
        assert not unaccounted, (
            "engrava added exception type(s) this server neither maps nor "
            f"excuses: {unaccounted}. Add a _tool_errors branch mapping it, or "
            "add a reasoned entry to _OUT_OF_SCOPE."
        )

    def test_out_of_scope_entries_are_not_stale(self) -> None:
        # An _OUT_OF_SCOPE entry must name a type engrava still defines, and
        # one _tool_errors still does not map -- otherwise it is either a
        # leftover for a type that no longer exists, or dead weight hiding a
        # mapping that already exists.
        public_types = _engrava_public_exception_types()
        handled = _handled_public_exception_names(public_types)
        stale = sorted(
            name for name in _OUT_OF_SCOPE if name not in public_types or name in handled
        )
        assert not stale, f"stale _OUT_OF_SCOPE entries, no longer accurate: {stale}"

    def test_out_of_scope_entries_carry_a_real_reason(self) -> None:
        empty = [name for name, reason in _OUT_OF_SCOPE.items() if not reason.strip()]
        assert not empty, f"_OUT_OF_SCOPE entries with no reason: {empty}"

    def test_sweep_fails_on_a_deliberately_unmapped_type(self) -> None:
        # Proves the sweep guards something, on synthetic input that exercises
        # the same pure check the real sweep above calls: a type present in
        # neither `handled` nor `out_of_scope` is flagged.
        public_types = {"ThoughtNotFoundError": object(), "StaleDataError": object()}
        handled = {"ThoughtNotFoundError"}
        out_of_scope: dict[str, str] = {}

        unaccounted = _unaccounted_exception_types(public_types, handled, out_of_scope)

        assert unaccounted == ["StaleDataError"]

    def test_sweep_passes_when_the_gap_is_closed_either_way(self) -> None:
        # Foil to the previous test: mapping OR excusing the same gap clears
        # it, confirming the check treats both routes as equally valid.
        public_types = {"ThoughtNotFoundError": object(), "StaleDataError": object()}
        handled = {"ThoughtNotFoundError", "StaleDataError"}
        assert _unaccounted_exception_types(public_types, handled, {}) == []

        handled_narrow = {"ThoughtNotFoundError"}
        excused = {"StaleDataError": "covered by a reason, not a branch"}
        assert _unaccounted_exception_types(public_types, handled_narrow, excused) == []

    def test_every_currently_mapped_type_is_reachable_through_the_import(self) -> None:
        # The identity check in _handled_public_exception_names depends on
        # engrava_mcp.server importing each mapped type by its real name; this
        # pins that precondition so a future refactor (e.g. importing under an
        # alias) fails here with a clear reason rather than silently reopening
        # every type it renamed as "unmapped".
        public_types = _engrava_public_exception_types()
        handled = _handled_public_exception_names(public_types)
        assert "ThoughtNotFoundError" in handled
        assert "DuplicateEdgeError" in handled
        assert "StaleDataError" in handled
        assert "WriteContentionError" in handled
        assert "WriteLockTimeoutError" in handled
        assert "ConnectionQuarantinedError" in handled
