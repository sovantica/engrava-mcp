"""End-to-end and store-resolution tests for the MCP server.

Exercises the server through the in-memory MCP client transport (so the
lifespan, tool registration, and JSON serialisation all run for real) and
covers store resolution from environment variables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest
from engrava import (
    CoreThoughtRecord,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
)
from mcp.server.mcpserver.exceptions import ToolError

from engrava_mcp import build_server
from engrava_mcp.config import (
    CONFIG_ENV_VAR,
    DB_PATH_ENV_VAR,
    ResolvedStore,
    StoreResolutionError,
    resolve_store,
)
from engrava_mcp.server import READ_ONLY_ENV_VAR
from tests.inprocess_client import connect_client

if TYPE_CHECKING:
    from pathlib import Path

READ_TOOL_NAMES = frozenset(
    {
        "get_thought",
        "search_memory",
        "search_keywords",
        "list_memory",
        "query_memory",
        "memory_stats",
        "get_edges",
        "list_edges",
    }
)
WRITE_TOOL_NAMES = frozenset(
    {"store_thought", "update_thought", "link_thoughts", "delete_thought", "delete_edge"}
)
#: The subset of write tools that remove data and therefore carry
#: ``destructive_hint=True``.
DESTRUCTIVE_TOOL_NAMES = frozenset({"delete_thought", "delete_edge"})
EXPECTED_TOOL_NAMES = READ_TOOL_NAMES | WRITE_TOOL_NAMES


async def _seed_database(path: Path) -> None:
    """Create a database file with a single active thought.

    Args:
        path: Filesystem path for the new database.

    """
    connection = await aiosqlite.connect(str(path))
    connection.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(connection)
    await store.ensure_schema()
    await store.create_thought(
        CoreThoughtRecord(
            thought_id="seeded-1",
            thought_type=ThoughtType.BELIEF,
            essence="Persisted note",
            content="A note that survives a fresh connection.",
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=0,
            updated_cycle=0,
            source="test",
        )
    )
    await connection.close()


class TestServerEndToEnd:
    """Drive the server through a connected in-memory client."""

    async def test_lists_read_and_write_tools_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "tools.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_tools()

        read_only_by_name: dict[str, bool | None] = {}
        idempotent_by_name: dict[str, bool | None] = {}
        destructive_by_name: dict[str, bool | None] = {}
        for tool in listed.tools:
            # Every tool must carry an annotation block.
            assert tool.annotations is not None
            read_only_by_name[tool.name] = tool.annotations.read_only_hint
            idempotent_by_name[tool.name] = tool.annotations.idempotent_hint
            destructive_by_name[tool.name] = tool.annotations.destructive_hint

        assert set(read_only_by_name) == EXPECTED_TOOL_NAMES
        # The read tools are read-only and the write tools are not.
        assert all(read_only_by_name[name] for name in READ_TOOL_NAMES)
        assert all(read_only_by_name[name] is False for name in WRITE_TOOL_NAMES)

        # Idempotency hints must match the real store semantics a client
        # would rely on for safe retries:
        #   - update_thought refreshes updated_at (and appends a journal
        #     entry on a journal-enabled store) on every call -> NOT idempotent
        #   - store_thought creates a fresh node each call    -> NOT idempotent
        #   - link_thoughts rejects a duplicate (from,to,type) -> NOT idempotent
        #   - delete_* of an absent id is a no-op, same end state -> idempotent
        assert idempotent_by_name["update_thought"] is False
        assert idempotent_by_name["store_thought"] is False
        assert idempotent_by_name["link_thoughts"] is False
        assert idempotent_by_name["delete_thought"] is True
        assert idempotent_by_name["delete_edge"] is True

        # Only the delete tools remove data, so only they are destructive.
        assert all(destructive_by_name[name] is True for name in DESTRUCTIVE_TOOL_NAMES)
        assert all(
            destructive_by_name[name] is False for name in WRITE_TOOL_NAMES - DESTRUCTIVE_TOOL_NAMES
        )

    async def test_read_only_mode_hides_write_tools(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "ro.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(READ_ONLY_ENV_VAR, "1")

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_tools()

        assert {tool.name for tool in listed.tools} == READ_TOOL_NAMES

    async def test_write_tools_round_trip_over_transport(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "writes.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            created = await client.call_tool(
                "store_thought",
                {"essence": "Live note", "content": "Stored over the transport."},
            )
            assert created.is_error is False
            assert created.structured_content is not None
            first_id = created.structured_content["thought"]["thought_id"]

            second = await client.call_tool(
                "store_thought",
                {"essence": "Second note", "content": "Another stored note."},
            )
            assert second.structured_content is not None
            second_id = second.structured_content["thought"]["thought_id"]

            updated = await client.call_tool(
                "update_thought",
                {"thought_id": first_id, "essence": "Edited note"},
            )
            assert updated.is_error is False
            assert updated.structured_content is not None
            assert updated.structured_content["thought"]["essence"] == "Edited note"

            linked = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": first_id,
                    "to_thought_id": second_id,
                    "edge_type": "ASSOCIATED",
                },
            )
            assert linked.is_error is False
            assert linked.structured_content is not None
            assert linked.structured_content["edge"]["from_thought_id"] == first_id

            fetched = await client.call_tool("get_thought", {"thought_id": first_id})

        assert fetched.structured_content is not None
        assert fetched.structured_content["found"] is True
        assert fetched.structured_content["thought"]["essence"] == "Edited note"

    async def test_delete_tools_round_trip_over_transport(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "deletes.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            first = await client.call_tool(
                "store_thought",
                {"essence": "From note", "content": "Source thought."},
            )
            assert first.structured_content is not None
            first_id = first.structured_content["thought"]["thought_id"]

            second = await client.call_tool(
                "store_thought",
                {"essence": "To note", "content": "Target thought."},
            )
            assert second.structured_content is not None
            second_id = second.structured_content["thought"]["thought_id"]

            linked = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": first_id,
                    "to_thought_id": second_id,
                    "edge_type": "ASSOCIATED",
                },
            )
            assert linked.structured_content is not None
            edge_id = linked.structured_content["edge"]["edge_id"]

            deleted_edge = await client.call_tool("delete_edge", {"edge_id": edge_id})
            assert deleted_edge.is_error is False
            assert deleted_edge.structured_content is not None
            assert deleted_edge.structured_content["deleted"] is True

            deleted_thought = await client.call_tool("delete_thought", {"thought_id": first_id})
            assert deleted_thought.is_error is False
            assert deleted_thought.structured_content is not None
            assert deleted_thought.structured_content["deleted"] is True

            # Deleting the same thought again converges on the same end state
            # (already gone) and reports it without erroring.
            again = await client.call_tool("delete_thought", {"thought_id": first_id})
            assert again.is_error is False
            assert again.structured_content is not None
            assert again.structured_content["deleted"] is False

            fetched = await client.call_tool("get_thought", {"thought_id": first_id})

        assert fetched.structured_content is not None
        assert fetched.structured_content["found"] is False

    async def test_get_thought_round_trip(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "seeded.db"
        await _seed_database(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.call_tool("get_thought", {"thought_id": "seeded-1"})

        assert result.is_error is False
        assert result.structured_content is not None
        assert result.structured_content["found"] is True
        assert result.structured_content["thought"]["thought_id"] == "seeded-1"

    async def test_query_memory_rejects_select_over_transport(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "reject.db"
        await _seed_database(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.call_tool(
                "query_memory",
                {"query": "SELECT * FROM thought"},
            )

        assert result.is_error is True
        assert "FIND" in result.content[0].text  # type: ignore[union-attr]

    async def test_memory_stats_reports_seeded_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "stats.db"
        await _seed_database(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.call_tool("memory_stats", {})

        assert result.structured_content is not None
        assert result.structured_content["thought_count"] == 1

    async def test_memory_stats_reports_unmeasured_when_metrics_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "stats-disabled.db"
        await _seed_database(db_path)

        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(
            f"database:\n  path: {db_path}\nmetrics:\n  enabled: false\n",
            encoding="utf-8",
        )
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
        monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))

        server = build_server()
        async with connect_client(server) as client:
            result = await client.call_tool("memory_stats", {})

        assert result.structured_content is not None
        # The live, ungated count still reflects the seeded thought...
        assert result.structured_content["thought_count"] == 1
        # ...while the gated metrics snapshot is a zero-filled placeholder,
        # and the flag says so.
        assert result.structured_content["metrics"]["measured"] is False
        assert result.structured_content["metrics"]["thoughts"]["total"] == 0


async def _seed_varied_database(path: Path) -> None:
    """Create a database file spanning several types, statuses, priorities.

    The seed gives the filter tools something to discriminate: a mix of
    ``TASK``/``NOTE`` thoughts, ``ACTIVE``/``CREATED`` states, and ``P1``/
    ``P3`` priorities, all sharing the keyword "widget" so a single query
    ranks every row.

    Args:
        path: Filesystem path for the new database.

    """
    connection = await aiosqlite.connect(str(path))
    connection.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(connection)
    await store.ensure_schema()
    seeds = [
        ("active-task", ThoughtType.TASK, LifecycleStatus.ACTIVE, Priority.P1, 1),
        ("created-note", ThoughtType.NOTE, LifecycleStatus.CREATED, Priority.P3, 2),
        ("active-note", ThoughtType.NOTE, LifecycleStatus.ACTIVE, Priority.P3, 3),
    ]
    for thought_id, thought_type, status, priority, cycle in seeds:
        await store.create_thought(
            CoreThoughtRecord(
                thought_id=thought_id,
                thought_type=thought_type,
                essence=f"Widget note {thought_id}",
                content=f"A widget thought stored as {thought_id}.",
                priority=priority,
                lifecycle_status=status,
                created_cycle=cycle,
                updated_cycle=cycle,
                source="test",
            )
        )
    await connection.close()


class TestFilterAndListOverTransport:
    """Drive the new filter and browse surface through a connected client."""

    async def test_search_memory_filter_round_trip(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "search_filter.db"
        await _seed_varied_database(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            unfiltered = await client.call_tool("search_memory", {"query_text": "widget"})
            filtered = await client.call_tool(
                "search_memory",
                {"query_text": "widget", "thought_type": "NOTE"},
            )

        assert unfiltered.structured_content is not None
        # The unfiltered response carries no ``filtered`` block.
        assert "filtered" not in unfiltered.structured_content

        assert filtered.structured_content is not None
        kept = {entry["thought_id"] for entry in filtered.structured_content["results"]}
        assert kept == {"created-note", "active-note"}
        # Ranking honesty: the dropped TASK hit is accounted for truthfully.
        block = filtered.structured_content["filtered"]
        assert block["criteria"] == {"thought_type": "NOTE"}
        assert block["matched"] == 2
        assert block["dropped"] == 1

    async def test_list_memory_round_trip(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "list.db"
        await _seed_varied_database(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.call_tool(
                "list_memory",
                {"lifecycle_status": "ACTIVE", "limit": 10},
            )
            paged = await client.call_tool("list_memory", {"limit": 1, "offset": 1})

        assert listed.structured_content is not None
        ids = {thought["thought_id"] for thought in listed.structured_content["thoughts"]}
        assert ids == {"active-task", "active-note"}

        assert paged.structured_content is not None
        # Newest first (created-note at cycle 2 is the second row), one per page.
        assert paged.structured_content["count"] == 1
        assert [t["thought_id"] for t in paged.structured_content["thoughts"]] == ["created-note"]

    async def test_list_memory_available_in_read_only_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "ro_list_tool.db"
        await _seed_varied_database(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(READ_ONLY_ENV_VAR, "1")

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_tools()
            result = await client.call_tool("list_memory", {})

        # list_memory is a read tool, so it survives the write-tool gate.
        assert "list_memory" in {tool.name for tool in listed.tools}
        assert result.structured_content is not None
        assert result.structured_content["count"] == 3


class TestReadOnlyModeAccessTracking:
    """A read-only session over a tracking-enabled store makes no writes.

    An operator who turns Engrava's dreaming extension on gets
    ``access_tracking_enabled=True`` by default, and read-only mode must not let a
    read against that store stage a deferred access-count write. This drives the
    real ``ENGRAVA_MCP_CONFIG`` route end to end (build the server, call a read tool
    over the transport, let the lifespan tear down) and then reads ``access_count``
    back from a fresh connection — after the server's own connection has closed and
    flushed — so nothing about the assertion depends on any store method's own read.
    """

    async def test_reads_over_the_transport_leave_access_count_untouched(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "tracked.sqlite"
        await _seed_database(db_path)

        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(
            f"database:\n  path: {db_path}\nextensions:\n  dreaming:\n    enabled: true\n",
            encoding="utf-8",
        )
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
        monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))
        monkeypatch.setenv(READ_ONLY_ENV_VAR, "1")

        server = build_server()
        async with connect_client(server) as client:
            for _ in range(3):
                result = await client.call_tool("get_thought", {"thought_id": "seeded-1"})
                assert result.is_error is False
            for _ in range(3):
                # search_hybrid also buffers an access for a hit it returns — the
                # path a prior round's validation found completely unguarded — so
                # this transport-level check must exercise it too, not only
                # get_thought.
                searched = await client.call_tool("search_memory", {"query_text": "persisted"})
                assert searched.is_error is False
                assert searched.structured_content is not None
                assert searched.structured_content["results"], (
                    "the search must actually hit the seeded thought"
                )

        connection = await aiosqlite.connect(str(db_path))
        try:
            cursor = await connection.execute(
                "SELECT access_count FROM thought WHERE thought_id = ?",
                ("seeded-1",),
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 0
        finally:
            await connection.close()


class TestReadOnlyDecisionIsCapturedOnce:
    """Registration and store-wrapping cannot disagree, even if the environment changes.

    ``build_server()`` reads ``ENGRAVA_MCP_READ_ONLY`` once and reuses that value both to
    decide which tools register (synchronously, inside ``build_server()``) and to decide
    whether the store the lifespan hands out is wrapped (later, when the lifespan actually
    runs). Reading the flag independently at each site would let an environment change in
    between produce a deployment that advertises itself as read-only while serving reads
    against the raw, unwrapped store — or the reverse. This flips the flag after
    ``build_server()`` has already decided, and asserts both facets still agree with the
    value captured at build time.
    """

    async def test_env_change_after_build_does_not_unwrap_or_re_register(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "frozen.sqlite"
        await _seed_database(db_path)

        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(
            f"database:\n  path: {db_path}\nextensions:\n  dreaming:\n    enabled: true\n",
            encoding="utf-8",
        )
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
        monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))
        monkeypatch.setenv(READ_ONLY_ENV_VAR, "1")

        server = build_server()

        # The environment changes after build_server() has already decided. A
        # second, independent read of the flag at serve time would now see
        # "not read-only" and hand read tools the raw store.
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        async with connect_client(server) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert names == READ_TOOL_NAMES, "registration must stay frozen at build time"

            for _ in range(3):
                result = await client.call_tool("get_thought", {"thought_id": "seeded-1"})
                assert result.is_error is False

        connection = await aiosqlite.connect(str(db_path))
        try:
            cursor = await connection.execute(
                "SELECT access_count FROM thought WHERE thought_id = ?",
                ("seeded-1",),
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 0, "store-wrapping must stay frozen at build time too"
        finally:
            await connection.close()


class TestStoreResolution:
    """Tests for environment-driven store resolution."""

    async def test_db_path_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "resolve.db"
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        resolved = await resolve_store()
        assert isinstance(resolved, ResolvedStore)
        try:
            assert await resolved.store.count_thoughts() == 0
        finally:
            await resolved.aclose()
        assert db_path.exists()

    async def test_config_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "from_config.db"
        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(
            f"database:\n  path: {db_path.as_posix()}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)

        resolved = await resolve_store()
        try:
            assert await resolved.store.count_thoughts() == 0
        finally:
            await resolved.aclose()

    async def test_config_takes_priority_over_db_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        config_db = tmp_path / "config_priority.db"
        config_path = tmp_path / "priority.yaml"
        config_path.write_text(
            f"database:\n  path: {config_db.as_posix()}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "ignored.db"))

        resolved = await resolve_store()
        await resolved.aclose()
        # The config path's database is the one that gets created.
        assert config_db.exists()
        assert not (tmp_path / "ignored.db").exists()

    async def test_no_configuration_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
        with pytest.raises(StoreResolutionError):
            await resolve_store()


class TestSurfaceAfterShutdown:
    """A tool called after the lifespan has ended answers, it does not crash."""

    async def test_a_tool_called_after_shutdown_reports_the_store_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Shutdown forgets the store as well as closing the connection. If it
        # only closed the connection, the surface would keep handing out a store
        # whose connection is gone and a late call would surface the driver's
        # own text instead of the curated "not available yet" message. Asserting
        # the curated wording is what tells the two apart: both raise.
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "shutdown.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        lifespan = server.settings.lifespan
        assert lifespan is not None, "the built server must carry a lifespan"
        async with lifespan(server):
            # Serving: the same call succeeds while the lifespan is running, so
            # the failure below is attributable to shutdown and not to the call.
            assert await server.call_tool("memory_stats", {}) is not None

        with pytest.raises(ToolError) as excinfo:
            await server.call_tool("memory_stats", {})

        text = str(excinfo.value)
        assert "The engrava memory store is not available yet" in text
        assert DB_PATH_ENV_VAR in text
