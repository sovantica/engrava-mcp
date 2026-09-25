"""Tests that the server's self-description matches its actual behaviour.

None of the source under test here changes behaviour: these tests pin what
the server, the README, ``server.json``, and the runtime warning *say* about
that unchanged behaviour, so a future edit cannot reintroduce a claim the
code does not back up.

Four classes of false self-description are covered:

* The audit trail is not reachable on the bare ``ENGRAVA_DB_PATH`` route —
  not in the README, and not in the startup warning that repeats the same
  claim at every no-provider launch.
* ``server.json`` describes only what the MCP surface actually exposes —
  MindQL, search, and the thought graph — not audit verification or Action
  Records, which live in the library and the CLI only.
* Every description of cycle-ordered output says so honestly: highest
  cognitive cycle first, never "newest" or "most recently updated" — and,
  since every MCP-created thought or edge ties at cycle 0 and the
  underlying queries carry no tiebreaker, never a promise about what that
  tied order *is* either (not "insertion order", not "deterministic", not
  "stable", not "write order") or about it representing recency (not
  "latest").
* ``delete_thought`` states its cascade, and ``update_thought`` is not
  annotated idempotent.

**A note on what this sweep cannot do.** ``_FORBIDDEN_RECENCY_PATTERNS`` and
``_FORBIDDEN_TIE_ORDER_PATTERNS`` are phrase lists, and a phrase list finds
copies, not paraphrases. The same claim can be worded "newest first",
"insertion order" or "deterministic": a grep for "insertion order" will
never catch "deterministic", and no phrase list is sure to catch a further
rewording of the same claim (no ordering guarantee exists among cycle ties).
This file does not claim to enumerate every way that claim could
be phrased. What actually bounds the risk is that the honest wording is
now centralised in :data:`~engrava_mcp.server._CYCLE_ORDERING_NOTE` and
consumed by reference almost everywhere (:class:`TestOrderingDescriptionsAgree`
pins that the wire descriptions still contain it verbatim); the phrase
lists exist to catch a *drafted-fresh* restatement elsewhere in the source,
not to certify that no such restatement is possible. Reviewing new
ordering/recency prose by reading it, not just by running this file, is
still necessary.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

from engrava_mcp import build_server
from engrava_mcp.config import _NO_PROVIDER_WARNING, CONFIG_ENV_VAR, DB_PATH_ENV_VAR
from engrava_mcp.server import READ_ONLY_ENV_VAR
from tests.inprocess_client import connect_client

#: Repository root, located from this test file's path (never hardcoded).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Phrases that claim wall-clock recency ordering the server cannot deliver.
#: Every MCP-created thought stamps ``created_cycle = updated_cycle = 0``, so
#: nothing this server writes is ever genuinely "newest" or "most recently
#: updated" — only "highest cognitive cycle", which the source must say
#: instead. ``latest`` is listed because "confirm the latest activity" and
#: "stays focused on the latest activity" make the same claim in words the
#: narrower patterns miss. See the module docstring for why this list can
#: never be complete.
_FORBIDDEN_RECENCY_PATTERNS = (
    re.compile(r"newest[- ]first", re.IGNORECASE),
    re.compile(r"most[- ]recently[- ]updated", re.IGNORECASE),
    re.compile(r"most recently stored", re.IGNORECASE),
    re.compile(r"most recent thought", re.IGNORECASE),
    re.compile(r"\blatest\b", re.IGNORECASE),
)

#: Phrase claiming the MindQL grammar itself lacks ``OFFSET``. It does not —
#: only the ``query_memory`` tool exposes no separate ``offset`` argument.
_FALSE_GRAMMAR_OFFSET_CLAIM = re.compile(r"grammar has no ``?OFFSET``?", re.IGNORECASE)

#: Phrases that promise a specific order *among cycle ties* (thoughts or
#: edges sharing the same ``updated_cycle`` / ``created_cycle``). The
#: underlying queries are ``ORDER BY <cycle column> DESC`` with no
#: tiebreaker, so SQLite guarantees nothing about the relative order of
#: equal values — observing one order in a probe is one execution, not a
#: contract. ``insertion order`` names such an order outright.
#: ``deterministic`` states the same guarantee as a property of the browse
#: itself ("Deterministic, unranked browse...") rather than of a named
#: order — a paraphrase, not a repeat, which is exactly what a
#: phrase list cannot be trusted to catch in general (see the module
#: docstring).
_FORBIDDEN_TIE_ORDER_PATTERNS = (
    re.compile(r"insertion order", re.IGNORECASE),
    re.compile(r"stable order", re.IGNORECASE),
    re.compile(r"in the order (they|it) (were|was) written", re.IGNORECASE),
    re.compile(r"write order", re.IGNORECASE),
    re.compile(r"deterministic", re.IGNORECASE),
)


def _server_source() -> str:
    """Read ``server.py`` as text.

    Returns:
        The full source of ``src/engrava_mcp/server.py``.

    """
    return (REPO_ROOT / "src" / "engrava_mcp" / "server.py").read_text(encoding="utf-8")


def _readme_text() -> str:
    """Read ``README.md`` as text.

    Returns:
        The full contents of the repository README.

    """
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


class TestNoRemainingRecencyClaims:
    """Grep-checked (not eyeballed) sweep for stale recency-ordering claims."""

    def test_server_source_has_no_forbidden_recency_phrase(self) -> None:
        source = _server_source()
        offenders = [
            pattern.pattern for pattern in _FORBIDDEN_RECENCY_PATTERNS if pattern.search(source)
        ]
        assert not offenders, f"stale recency claim(s) still present: {offenders}"

    def test_server_source_does_not_claim_grammar_lacks_offset(self) -> None:
        source = _server_source()
        assert not _FALSE_GRAMMAR_OFFSET_CLAIM.search(source)
        # The honest, narrower claim must be present instead.
        assert "exposes no separate" in source
        assert "``offset`` argument" in source

    def test_readme_has_no_forbidden_recency_phrase(self) -> None:
        text = _readme_text()
        offenders = [
            pattern.pattern for pattern in _FORBIDDEN_RECENCY_PATTERNS if pattern.search(text)
        ]
        assert not offenders, f"stale recency claim(s) still present in README: {offenders}"

    def test_server_source_promises_no_order_among_cycle_ties(self) -> None:
        source = _server_source()
        offenders = [
            pattern.pattern for pattern in _FORBIDDEN_TIE_ORDER_PATTERNS if pattern.search(source)
        ]
        assert not offenders, f"unsupported tie-order promise(s) still present: {offenders}"


class TestOrderingDescriptionsAgree:
    """``list_memory`` and ``engrava://recent`` must describe ordering identically."""

    async def test_tool_and_resource_share_the_ordering_clause(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from engrava_mcp.server import _CYCLE_ORDERING_NOTE

        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "ordering.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()

        list_memory = next(tool for tool in tools.tools if tool.name == "list_memory")
        recent_resource = next(
            resource for resource in resources.resources if str(resource.uri) == "engrava://recent"
        )

        assert list_memory.description is not None
        assert recent_resource.description is not None
        # Both descriptions carry the exact same ordering clause, sourced
        # from the same module-level constant, so they cannot drift apart.
        assert _CYCLE_ORDERING_NOTE in list_memory.description
        assert _CYCLE_ORDERING_NOTE in recent_resource.description

        # Neither claims "newest" or "most recently updated" ordering, and
        # neither promises what the order among cycle ties actually is.
        for description in (list_memory.description, recent_resource.description):
            for pattern in _FORBIDDEN_RECENCY_PATTERNS + _FORBIDDEN_TIE_ORDER_PATTERNS:
                assert not pattern.search(description)


class TestDeleteThoughtNamesTheCascade:
    """``delete_thought``'s description states what it takes with it."""

    async def test_delete_thought_description_names_the_cascade(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "cascade.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            tools = await client.list_tools()

        delete_thought = next(tool for tool in tools.tools if tool.name == "delete_thought")
        assert delete_thought.description is not None
        for term in ("edges", "embeddings", "action records"):
            assert term in delete_thought.description

    async def test_update_thought_is_not_annotated_idempotent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "not_idempotent.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            tools = await client.list_tools()

        update_thought = next(tool for tool in tools.tools if tool.name == "update_thought")
        assert update_thought.annotations is not None
        assert update_thought.annotations.idempotent_hint is False


class TestReadmeBareRouteCapabilities:
    """The ``ENGRAVA_DB_PATH`` row lists only capabilities that route has."""

    def _db_path_row(self) -> str:
        for line in _readme_text().splitlines():
            if line.startswith("| `ENGRAVA_DB_PATH`"):
                return line
        pytest.fail("README is missing the ENGRAVA_DB_PATH configuration row")

    def test_bare_route_row_does_not_claim_the_audit_trail(self) -> None:
        row = self._db_path_row()
        assert "audit trail" not in row.lower() or "not available" in row.lower()
        # More precisely: the row must not claim the audit trail *works*.
        assert "audit trail still work" not in row
        assert "the audit trail is not available" in row

    def test_bare_route_row_still_lists_its_real_capabilities(self) -> None:
        row = self._db_path_row()
        assert "full-text search" in row
        assert "the graph" in row
        assert "MindQL" in row

    def test_config_route_row_names_the_journal(self) -> None:
        for line in _readme_text().splitlines():
            if line.startswith("| `ENGRAVA_MCP_CONFIG`"):
                assert "journal" in line.lower()
                return
        pytest.fail("README is missing the ENGRAVA_MCP_CONFIG configuration row")


class TestStartupWarningDoesNotClaimTheAuditTrail:
    """The no-provider startup warning must not repeat the README's false claim."""

    def test_warning_does_not_mention_the_audit_trail(self) -> None:
        assert "audit trail" not in _NO_PROVIDER_WARNING.lower()

    def test_warning_still_names_what_is_actually_unaffected(self) -> None:
        assert "full-text search" in _NO_PROVIDER_WARNING.lower()
        assert "graph" in _NO_PROVIDER_WARNING.lower()
        assert "mindql" in _NO_PROVIDER_WARNING.lower()


class TestServerJsonDescribesTheMcpSurfaceOnly:
    """``server.json``'s registry description matches the MCP surface, not the library."""

    def _server_json(self) -> dict[str, object]:
        return json.loads((REPO_ROOT / "server.json").read_text(encoding="utf-8"))

    def test_description_does_not_claim_audit_or_action_records(self) -> None:
        description = self._server_json()["description"]
        assert isinstance(description, str)
        lowered = description.lower()
        assert "audit" not in lowered
        assert "action record" not in lowered
        assert "tamper-evident" not in lowered

    def test_description_names_a_capability_the_surface_actually_has(self) -> None:
        description = self._server_json()["description"]
        assert isinstance(description, str)
        assert "MindQL" in description

    def test_version_fields_are_untouched_by_the_description_change(self) -> None:
        # The guard job (release.yml) compares the tag against these two
        # version fields; editing the description must never touch them.
        manifest = self._server_json()
        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            pyproject = tomllib.load(handle)
        project = pyproject["project"]
        assert isinstance(project, dict)
        assert manifest["version"] == project["version"]
        packages = manifest["packages"]
        assert isinstance(packages, list)
        assert packages[0]["version"] == project["version"]


class TestGetEdgesLimitDescribedHonestly:
    """The ``get_edges`` tool description and the README state ``limit``'s two halves."""

    def _get_edges_readme_sentence(self) -> str:
        """Return only the README's ``get_edges`` sentence, whitespace-collapsed.

        Extracts the text from the ``get_edges`` sentence up to — but not
        including — the following ``list_edges`` sentence, so a match cannot be
        satisfied by ``list_edges``'s own wording spilling in from the same
        line-wrapped paragraph. Whitespace runs (including the source's line
        wraps) are collapsed to single spaces so the assertion does not depend
        on exactly where the paragraph happens to wrap.

        Returns:
            The ``get_edges`` sentence alone, with normalised whitespace.

        """
        text = _readme_text()
        start = text.index("`get_edges` traverses")
        end = text.index("`list_edges` browses", start)
        return " ".join(text[start:end].split())

    async def test_tool_description_states_both_halves(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "get_edges_limit.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            tools = await client.list_tools()

        get_edges = next(tool for tool in tools.tools if tool.name == "get_edges")
        assert get_edges.description is not None
        # The exact, condition-linked sentence: a reversed with/without pairing,
        # or a rephrase dropping "at most" or "highest-weight", both fail this —
        # unlike disconnected fragment checks, which either would pass.
        assert (
            "With limit, it returns at most that many edges, the highest-weight "
            "ones first; without it, it returns every edge."
        ) in get_edges.description

    def test_readme_line_states_both_halves(self) -> None:
        sentence = self._get_edges_readme_sentence()
        assert (
            "with `limit`, at most that many edges, the highest-weight ones first, "
            "and without it, every edge."
        ) in sentence
