"""Tests for the MCP tool error contract.

When a tool hits a known failure condition, the client must receive a
clean, typed, actionable error — a message with a helpful hint and
``isError`` set — rather than a raw Python traceback or an internal class
name.  These tests drive the real tool boundary through the in-process MCP
client transport (so FastMCP's error wrapping runs for real) and assert on
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

import re
import sqlite3
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
from engrava import EdgeType
from engrava.domain.exceptions import DuplicateEdgeError
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session as connect_client

from engrava_mcp.server import (
    DUPLICATE_EDGE_MESSAGE,
    SERVER_NAME,
    StoreProvider,
    _query_declares_find,
    _tool_errors,
    link_thoughts_impl,
    register_tools,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore
    from mcp import ClientSession

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
async def _client_for(store: SqliteEngravaCore) -> AsyncIterator[ClientSession]:
    """Open a connected client whose tools query the given store.

    Builds a server, points a :class:`StoreProvider` at ``store``, registers
    the tools, and connects the in-process client so the real tool boundary
    (and FastMCP's error wrapping) runs end to end.

    Args:
        store: The seeded store the tools should query.

    Yields:
        A connected client session wired to ``store``.

    """
    server: FastMCP = FastMCP(SERVER_NAME)
    provider = StoreProvider()
    provider.set(store, read_store=store)
    register_tools(server, provider, read_only=False)
    async with connect_client(server) as client:
        yield client


@asynccontextmanager
async def _store_less_client() -> AsyncIterator[ClientSession]:
    """Open a connected client whose provider never received a store.

    Registering the tools against an unpopulated :class:`StoreProvider`
    reproduces a deployment whose store has not been configured: the first
    tool call hits the store-not-ready condition at the real boundary.

    Yields:
        A connected client session backed by a store-less provider.

    """
    server: FastMCP = FastMCP(SERVER_NAME)
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

        assert result.isError is True
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

        assert result.isError is True
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

        assert result.isError is True
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
        assert result.isError is True
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

        assert result.isError is True
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

        assert result.isError is True
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

        assert result.isError is True
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

        assert result.isError is True
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

        assert result.isError is True
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

        assert result.isError is True
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

        assert result.isError is True
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

        assert result.isError is not True


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

        assert result.isError is True
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

        assert result.isError is True
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

        assert result.isError is True
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
            assert first.isError is False  # the first link succeeds

            duplicate = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": "thought-alpha",
                    "to_thought_id": "thought-beta",
                    "edge_type": "ASSOCIATED",
                },
            )

        assert duplicate.isError is True
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

        assert result.isError is True
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

        assert result.isError is True
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
        # illegal. FastMCP coerces the wire status to the LifecycleStatus enum,
        # so the store's transition guard fires. The raw message names the
        # internal status type — the client must get a curated message instead.
        async with _client_for(store) as client:
            result = await client.call_tool(
                "update_thought",
                {"thought_id": "thought-alpha", "lifecycle_status": "CREATED"},
            )

        assert result.isError is True
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

        assert after.isError is False
        assert after.structuredContent is not None
        assert after.structuredContent["thought"]["lifecycle_status"] == "ACTIVE"


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

        assert result.isError is False
        assert result.structuredContent is not None
        assert "thought_id" in result.structuredContent["columns"]
        assert len(result.structuredContent["rows"]) == 2

    async def test_valid_keyword_search_still_succeeds(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # A well-formed call through a wrapped read tool returns its normal
        # payload unchanged — the error wrapper adds nothing on the happy path.
        async with _client_for(store) as client:
            result = await client.call_tool("search_keywords", {"query": "coffee"})

        assert result.isError is False
        assert result.structuredContent is not None
        assert "results" in result.structuredContent


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
