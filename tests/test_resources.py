"""End-to-end tests for the MCP read-only resources.

Exercises the resources through the in-memory MCP client transport (so
registration, URI-template binding, and JSON serialisation all run for
real), mirroring the tool tests in :mod:`tests.mcp.test_server`.

Resources are reads by definition, so they are advertised in both the
default and the read-only deployment; the read-only cases below assert
that independence directly.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import TYPE_CHECKING

import aiosqlite
from engrava import (
    CoreThoughtRecord,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
)

from engrava_mcp import build_server
from engrava_mcp.config import CONFIG_ENV_VAR, DB_PATH_ENV_VAR
from engrava_mcp.server import READ_ONLY_ENV_VAR
from tests.inprocess_client import connect_client

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

#: Static resource URIs the server must advertise via ``list_resources``.
STATIC_RESOURCE_URIS = frozenset({"engrava://stats", "engrava://recent"})
#: Templated resource URI advertised via ``list_resource_templates``.
THOUGHT_TEMPLATE_URI = "engrava://thought/{thought_id}"


async def _seed_two_thoughts(path: Path) -> None:
    """Create a database file with two thoughts updated in a known order.

    The second thought carries the larger ``updated_cycle`` so it is the
    most recent — ``list_thoughts`` orders by descending ``updated_cycle``,
    so ``engrava://recent`` must return it first.

    Args:
        path: Filesystem path for the new database.

    """
    connection = await aiosqlite.connect(str(path))
    connection.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(connection)
    await store.ensure_schema()
    await store.create_thought(
        CoreThoughtRecord(
            thought_id="older-thought",
            thought_type=ThoughtType.BELIEF,
            essence="Older note",
            content="The earlier of the two seeded thoughts.",
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=1,
            updated_cycle=1,
            source="test",
        )
    )
    await store.create_thought(
        CoreThoughtRecord(
            thought_id="newer-thought",
            thought_type=ThoughtType.BELIEF,
            essence="Newer note",
            content="The later of the two seeded thoughts.",
            priority=Priority.P1,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=2,
            updated_cycle=2,
            source="test",
        )
    )
    await connection.close()


#: Ids and essences used to pin percent-decoding at the resource
#: boundary. Essences differ per id, so a resource handler that resolves
#: the wrong thought is caught by the essence assertion, not just by the
#: id it already knew to ask for.
_RESERVED_CHARACTER_THOUGHTS: tuple[tuple[str, str], ...] = (
    ("a/b", "Slash-bearing id"),
    ("a%2Fb", "Literal percent-two-F id"),
    ("x y?z", "Space and query-mark id"),
    ("plain-id", "Plain control id"),
)


async def _seed_reserved_character_thoughts(path: Path) -> None:
    """Create a database seeded with :data:`_RESERVED_CHARACTER_THOUGHTS`.

    Covers a literal slash, a literal percent-encoded slash, a space plus
    a question mark, and a plain control id, each with its own essence.

    Args:
        path: Filesystem path for the new database.

    """
    connection = await aiosqlite.connect(str(path))
    connection.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(connection)
    await store.ensure_schema()
    for cycle, (thought_id, essence) in enumerate(_RESERVED_CHARACTER_THOUGHTS, start=1):
        await store.create_thought(
            CoreThoughtRecord(
                thought_id=thought_id,
                thought_type=ThoughtType.BELIEF,
                essence=essence,
                content="Seeded to pin resource-URI percent-decoding.",
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=cycle,
                updated_cycle=cycle,
                source="test",
            )
        )
    await connection.close()


def _decode_single(result: object) -> dict[str, object]:
    """Parse the single JSON text payload of a ``read_resource`` result.

    Args:
        result: The ``ReadResourceResult`` returned by ``read_resource``.

    Returns:
        The decoded JSON object carried by the result's sole content
        block.

    """
    contents = result.contents  # type: ignore[attr-defined]
    assert len(contents) == 1
    block = contents[0]
    assert block.mime_type == "application/json"
    decoded = json.loads(block.text)
    assert isinstance(decoded, dict)
    return decoded


class TestResourceListing:
    """List resources and templates through a connected client."""

    async def test_static_resources_are_listed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "list.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_resources()

        assert {str(resource.uri) for resource in listed.resources} == STATIC_RESOURCE_URIS

    async def test_thought_template_is_listed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "templates.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_resource_templates()

        templates = {template.uri_template for template in listed.resource_templates}
        assert THOUGHT_TEMPLATE_URI in templates


class TestResourceReads:
    """Read each resource through a connected client."""

    async def test_thought_resource_returns_seeded_thought(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "thought.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.read_resource("engrava://thought/newer-thought")

        payload = _decode_single(result)
        assert payload["found"] is True
        thought = payload["thought"]
        assert isinstance(thought, dict)
        assert thought["thought_id"] == "newer-thought"
        assert thought["essence"] == "Newer note"

    async def test_thought_resource_unknown_id_is_graceful(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "missing.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        # An unknown identifier must not raise over the transport; it
        # returns a not-found payload, mirroring the get_thought tool.
        async with connect_client(server) as client:
            result = await client.read_resource("engrava://thought/no-such-id")

        payload = _decode_single(result)
        assert payload == {"found": False, "thought": None}

    async def test_recent_resource_orders_newest_first(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "recent.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.read_resource("engrava://recent")

        payload = _decode_single(result)
        thoughts = payload["thoughts"]
        assert isinstance(thoughts, list)
        ids = [thought["thought_id"] for thought in thoughts]
        # list_thoughts orders by descending updated_cycle, so the newer
        # thought comes first.
        assert ids == ["newer-thought", "older-thought"]
        assert payload["limit"] == 10

    async def test_stats_resource_matches_memory_stats_tool(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "stats.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            resource_result = await client.read_resource("engrava://stats")
            tool_result = await client.call_tool("memory_stats", {})

        resource_payload = _decode_single(resource_result)
        assert tool_result.structured_content is not None
        # The resource and the memory_stats tool share memory_stats_impl,
        # so they must agree field-for-field (no duplicate stats logic).
        assert resource_payload == tool_result.structured_content
        assert resource_payload["thought_count"] == 2


class TestThoughtResourceUriDecoding:
    """Pin that a thought id round-trips through its resource URI.

    The resource boundary decodes the id exactly once. These tests read
    through the real resource path with ids that carry reserved URI
    characters, so a regression in encoding or decoding shows up as the
    wrong thought (or none) coming back, not just a raised error.
    """

    async def test_quoted_ids_round_trip_to_the_right_thought(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "reserved.db"
        await _seed_reserved_character_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            for thought_id, essence in _RESERVED_CHARACTER_THOUGHTS:
                # Built the way a real client would build it: percent-encode
                # the whole id, then let the server decode it once.
                quoted = urllib.parse.quote(thought_id, safe="")
                result = await client.read_resource(f"engrava://thought/{quoted}")
                payload = _decode_single(result)
                assert payload["found"] is True
                thought = payload["thought"]
                assert isinstance(thought, dict)
                assert thought["thought_id"] == thought_id
                assert thought["essence"] == essence

    async def test_percent_collision_ids_are_not_mixed_up(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "collision.db"
        await _seed_reserved_character_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        # Hand-written, so the exact bytes that trigger the collision are
        # pinned in the test source, not only produced by quote().
        async with connect_client(server) as client:
            slash_result = await client.read_resource("engrava://thought/a%2Fb")
            literal_percent_result = await client.read_resource("engrava://thought/a%252Fb")

        slash_payload = _decode_single(slash_result)
        assert slash_payload["found"] is True
        slash_thought = slash_payload["thought"]
        assert isinstance(slash_thought, dict)
        assert slash_thought["thought_id"] == "a/b"
        assert slash_thought["essence"] == "Slash-bearing id"

        literal_percent_payload = _decode_single(literal_percent_result)
        assert literal_percent_payload["found"] is True
        literal_percent_thought = literal_percent_payload["thought"]
        assert isinstance(literal_percent_thought, dict)
        assert literal_percent_thought["thought_id"] == "a%2Fb"
        assert literal_percent_thought["essence"] == "Literal percent-two-F id"


class TestResourcesInReadOnlyMode:
    """Resources are reads, so they survive the write-tool gate."""

    async def test_resources_listed_in_read_only_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "ro_list.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(READ_ONLY_ENV_VAR, "1")

        server = build_server()
        async with connect_client(server) as client:
            static = await client.list_resources()
            templates = await client.list_resource_templates()

        # Read-only mode hides the write tools but must not hide resources.
        assert {str(resource.uri) for resource in static.resources} == STATIC_RESOURCE_URIS
        assert THOUGHT_TEMPLATE_URI in {
            template.uri_template for template in templates.resource_templates
        }

    async def test_resources_readable_in_read_only_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "ro_read.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(READ_ONLY_ENV_VAR, "1")

        server = build_server()
        async with connect_client(server) as client:
            stats = await client.read_resource("engrava://stats")
            recent = await client.read_resource("engrava://recent")
            thought = await client.read_resource("engrava://thought/newer-thought")

        assert _decode_single(stats)["thought_count"] == 2
        assert len(_decode_single(recent)["thoughts"]) == 2
        assert _decode_single(thought)["found"] is True
