"""Characterisation tests for known, deliberately deferred defects.

Every test in this file asserts behaviour the project considers **wrong**.
That is the point: each one is a tripwire pinning a defect that was found,
verified, and consciously left unfixed (a decision on the correct remedy is
still open), not a spec these behaviours are meant to satisfy.

Both defects here are claims about the **public MCP surface** — what a real
client can and cannot get the server to do — so both tests drive the
registered tools through a real, connected client (see ``_client_for``
below), the same in-process boundary ``tests/test_edge_tools.py`` uses for
its own over-the-wire tests. Calling the module-level ``*_impl`` functions
directly is not equivalent here: the MCP argument-binding layer normalises
its input before ``*_impl`` ever runs (see the first test's comment for what
that erases), so a test built on the ``*_impl`` shortcut can assert a claim
the real wire behaviour does not support.

Rules for this file:

- A test name must make the "this is a known defect, not a contract" reading
  unmistakable on its own — hence the ``_known_defect`` suffix on every test.
- Every test carries a comment stating what the correct behaviour would be,
  and that a failure here likely means someone fixed the underlying defect —
  in which case the right response is to update or remove the test, never to
  revert the fix.
- Nothing here is fixed by this file. If this set ever empties out, that is
  the intended end state: it means every defect it once pinned got fixed
  properly, with its own real test, elsewhere in the suite.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from mcp.server.mcpserver import MCPServer

from engrava_mcp.server import SERVER_NAME, StoreProvider, register_tools
from tests.inprocess_client import connect_client

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from engrava import SqliteEngravaCore
    from mcp import Client


@asynccontextmanager
async def _client_for(store: SqliteEngravaCore) -> AsyncIterator[Client]:
    """Open a connected client whose tools query the given store.

    Registers the tools against a provider pointed at ``store`` and connects
    the in-process client so the real tool boundary runs end to end — the
    same pattern ``tests/test_edge_tools.py`` uses for its own
    over-the-wire tests.

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


class TestConfidenceClearingKnownDefect:
    """``update_thought`` cannot clear ``confidence`` by sending ``null``."""

    async def test_wire_cannot_distinguish_omitted_confidence_from_explicit_null_known_defect(
        self, store: SqliteEngravaCore
    ) -> None:
        # The wire schema accepts `confidence: null`, and engrava itself uses
        # `None` to mean "unknown" — the intuitive reading of the defect is
        # that our code receives that null and then ignores it. It is worse
        # than that: MCP's own argument-binding layer (MCPServer's
        # ArgModelBase.model_dump_one_level, which dumps every field rather
        # than only the ones the caller set) already collapses "confidence
        # omitted" and "confidence: null" to the identical Python value
        # (`None`) before any of this server's own code runs. Confirmed by
        # driving the real registered tool through a connected client and
        # instrumenting what `update_thought_impl` received: both shapes of
        # call produced the exact same keyword arguments. Calling
        # `update_thought_impl` directly with `confidence=None` (as an
        # earlier version of this test did) cannot tell those two cases
        # apart either, and ends up only reasserting that *omission* leaves
        # the field untouched — which is correct, desired behaviour, not the
        # defect. This test instead drives both shapes of call through the
        # real client and shows they are indistinguishable in their effect.
        #
        # Correct behaviour would be that a client sending an explicit
        # `confidence: null` gets the field cleared, while a client omitting
        # the argument entirely leaves it untouched. That is constructible
        # entirely in the tool-registration wrapper (a non-`None` sentinel
        # default on the wire-facing parameter distinguishes "not supplied"
        # from "supplied as null" once more, since pydantic does not
        # validate an unvalidated default) without changing
        # `update_thought_impl`'s own contract at all — verified by hand
        # while pinning this test, not shipped here. Which mechanism to
        # expose that as (the sentinel-default trick above, or a documented
        # `clear_confidence` flag) is the open decision this test does not
        # take a side on, which is exactly why the defect is pinned here
        # instead of fixed.
        #
        # If this test goes red, it almost certainly means clearing now
        # works — update or delete this test, do not revert whatever change
        # made it pass.
        async with _client_for(store) as client:
            seed = await client.call_tool(
                "update_thought", {"thought_id": "thought-alpha", "confidence": 0.5}
            )
            assert seed.is_error is False

            explicit_null = await client.call_tool(
                "update_thought", {"thought_id": "thought-alpha", "confidence": None}
            )
            assert explicit_null.is_error is False
            after_explicit_null = await client.call_tool(
                "get_thought", {"thought_id": "thought-alpha"}
            )

            omitted = await client.call_tool("update_thought", {"thought_id": "thought-alpha"})
            assert omitted.is_error is False
            after_omitted = await client.call_tool("get_thought", {"thought_id": "thought-alpha"})

        assert after_explicit_null.structured_content is not None
        assert after_omitted.structured_content is not None
        confidence_after_explicit_null = after_explicit_null.structured_content["thought"][
            "confidence"
        ]
        confidence_after_omitted = after_omitted.structured_content["thought"]["confidence"]

        # Pinning the defect: an explicit null and an omitted argument
        # produced the exact same outcome — confidence is still 0.5 either
        # way. A client that wants to record "confidence is now unknown" has
        # no way to express that over this wire.
        assert confidence_after_explicit_null == 0.5
        assert confidence_after_omitted == 0.5


class TestLinkThoughtsMetadataValidationAsymmetryKnownDefect:
    """``link_thoughts`` accepts metadata keys that ``list_edges`` refuses."""

    async def test_link_thoughts_accepts_a_key_list_edges_will_then_refuse_known_defect(
        self, store: SqliteEngravaCore
    ) -> None:
        # list_edges validates every metadata-filter key against a
        # simple-top-level-field-name pattern and rejects a dotted or
        # bracketed key, because such a key would otherwise build a nested
        # JSONPath ($.outer.inner) the thin MCP surface deliberately does not
        # expose. link_thoughts builds the same kind of edge metadata but
        # never validates its keys at all, so the "simple field names only"
        # contract holds on the read side and not on the write side: a real
        # client can store a key it can never filter on again. Both calls
        # below go through the registered tools over a real connected
        # client, so this is the public surface's own behaviour, not an
        # artefact of calling an internal function directly.
        #
        # Correct behaviour would be the same validation on both sides —
        # either link_thoughts rejects a dotted/bracketed key at write time
        # too, or list_edges is taught to reach nested metadata. Neither is
        # done here because tightening the write path would reject metadata
        # previously accepted, which the project treats as a compatibility
        # break belonging at a minor version, not a silent behaviour change.
        #
        # If this test goes red, it almost certainly means the write path
        # now validates keys (or the read path stopped rejecting them) —
        # update or delete this test, do not revert whatever change made it
        # pass.
        async with _client_for(store) as client:
            created = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": "thought-alpha",
                    "to_thought_id": "thought-beta",
                    "edge_type": "ASSOCIATED",
                    "metadata": {"outer.inner": 1},
                },
            )
            assert created.is_error is False
            assert created.structured_content is not None
            # Pinning the defect: the write accepted a key list_edges cannot
            # filter on.
            assert created.structured_content["edge"]["metadata"] == {"outer.inner": 1}

            filtered = await client.call_tool("list_edges", {"metadata_equals": {"outer.inner": 1}})

        # Pinning the defect: the same key, sent to the same server, on the
        # filter side of the same feature, is refused.
        assert filtered.is_error is True
        assert "metadata filter is invalid" in _error_text(filtered.content).lower()
