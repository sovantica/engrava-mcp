"""The recency signal participates only when ``recency_now`` is supplied.

``search_memory``'s advertised description tells an agent client that recency
takes part in the ranking only when ``recency_now`` is passed.  That sentence is
a promise made over the wire, and this module holds it to the observables a
client can actually check: the ``backends_used`` list the response carries and
the scores it ranks by.

The claim is asserted where it is read — through a connected MCP client
session, against a store built by :func:`resolve_store`, the resolution the
server itself runs — rather than against a store the tests assemble by hand.
That distinction is load-bearing: a store built without a ``SearchConfig``
resolves its recency fusion weight to ``0.0``, so on a hand-built store recency
is inert whatever the argument, and the contract these tests are about could
not be exhibited at all.

Both launch routes are covered because the description is unqualified about
which one a client reached the server through.  Neither route can supply a
cognitive-cycle clock — engrava takes a cycle provider as a live runtime
object and never from serialized config — so ``recency_now`` is the only
recency reference either route has, and the promise has to hold on both.

The assertions are stated as a *difference* between the two arms rather than as
an exact backend list: which other backends run is engrava's business (a vector
backend appears once an embedding provider is configured), while the one thing
this module is about is that ``recency`` never joins them without the argument.
A positive control on each arm keeps that difference from being read off two
dead searches.  ``backends_used`` is a diagnostic, so the scoring is asserted
alongside it: what a caller asking for recency is after is an effect on the
results, not a report that a backend ran.  With the argument this corpus comes
back on scores separated in recency order; without it every entry carries the
same score.  Both readings are properties of *this* corpus, whose entries
differ in nothing the ranker scores except transaction time — elsewhere other
signals would separate the scores whatever recency did, so what carries the
evidence is the change between the two arms rather than either arm alone.

Where a difference in scores says the same thing as a difference in order, the
scores are asserted and the order is not: entries the ranker scores identically
come back in whatever order its tie-break gives them, and that tie-break is
engrava's business.  Pinning it would buy no discrimination and would fail this
module on a change that leaves the recency contract intact.

The description says recency takes part *only when* the argument is passed, and
stops there, because supplying it is necessary rather than sufficient: an
``engrava.yaml`` giving the recency signal no weight ranks without it however
the tool is called.  The last test pins that boundary — the behaviour, that is.
Nothing here reads the description itself: an assertion on its text would
compare one copy of a sentence with another and pass however wrong both were,
so what keeps the wording true is that the behaviour it describes is pinned and
reviewed against, not a string check.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session as connect_client

from engrava_mcp.config import CONFIG_ENV_VAR, DB_PATH_ENV_VAR, resolve_store
from engrava_mcp.server import SERVER_NAME, StoreProvider, register_tools
from tests.recency_corpus import (
    RECENCY_EXPECTED_ORDER,
    RECENCY_NOW,
    RECENCY_QUERY,
    RECENCY_THOUGHT_IDS,
    seed_recency_corpus,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from mcp import ClientSession

#: The backend name engrava reports for the lexical arm.  Present on every
#: launch here, so it serves as the positive control that the search ran at all.
LEXICAL_BACKEND = "fts5"

#: The backend name engrava reports for the recency arm — the one this module
#: is about.
RECENCY_BACKEND = "recency"


def _point_at_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra_sections: str) -> None:
    """Point store resolution at an ``engrava.yaml`` under the given directory.

    Args:
        monkeypatch: Fixture used to set the environment.
        tmp_path: Directory to write the config and database into.
        extra_sections: Yaml text appended after the ``database`` section.

    """
    config_file = tmp_path / "engrava.yaml"
    config_file.write_text(
        f"database:\n  path: {tmp_path / 'recency.sqlite'}\n{extra_sections}",
        encoding="utf-8",
    )
    monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_file))


@pytest.fixture(params=["bare-database", "engrava-yaml"])
def launch_route(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> str:
    """Point store resolution at one of the server's two launch routes.

    Args:
        request: Supplies the route name being exercised.
        monkeypatch: Fixture used to set the environment.
        tmp_path: Temporary directory for the database and config files.

    Returns:
        The name of the route the environment now selects.

    """
    route = str(request.param)
    if route == "bare-database":
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "recency.sqlite"))
    else:
        _point_at_yaml(monkeypatch, tmp_path, "")
    return route


@asynccontextmanager
async def _client_on_the_resolved_store() -> AsyncIterator[ClientSession]:
    """Open a client session over a freshly resolved, seeded store.

    Yields:
        A connected client whose ``search_memory`` tool queries a store holding
        the recency corpus, reached through the real MCP boundary.

    """
    resolved = await resolve_store()
    try:
        await seed_recency_corpus(resolved.store)
        server: FastMCP = FastMCP(SERVER_NAME)
        provider = StoreProvider()
        provider.set(resolved.store)
        register_tools(server, provider)
        async with connect_client(server) as client:
            yield client
    finally:
        await resolved.aclose()


async def _search(client: ClientSession, **arguments: Any) -> dict[str, Any]:  # noqa: ANN401
    """Call ``search_memory`` over the wire and return its structured payload.

    Args:
        client: The connected client session.
        **arguments: Extra arguments to pass alongside the query.

    Returns:
        The tool's structured response content.

    """
    result = await client.call_tool("search_memory", {"query_text": RECENCY_QUERY, **arguments})
    assert result.isError is False
    assert result.structuredContent is not None
    return dict(result.structuredContent)


class TestRecencyIsReportedOnlyWhenTheTimestampIsSupplied:
    """``backends_used`` never names ``recency`` without ``recency_now``.

    With the argument it does name it on both launch routes as they resolve
    here — under engrava's default search weights.  A configuration that
    weights the recency signal at zero is the boundary the last class covers.
    """

    async def test_recency_backend_appears_only_with_recency_now(self, launch_route: str) -> None:
        # The description's necessary condition, plus the positive case it is
        # worth having — two calls differing in nothing but the argument, on a
        # store carrying engrava's default weights.
        async with _client_on_the_resolved_store() as client:
            omitted = await _search(client)
            supplied = await _search(client, recency_now=RECENCY_NOW)

        # Positive control on both arms: the corpus is there and the search
        # really ran, so neither list is empty for an unrelated reason.  The
        # lexical backend is named rather than left as "some backend" because
        # ``backends_used`` is a response field clients read: the names in it
        # are part of what this server hands over the wire, and the rest of
        # this suite reads that one too.
        assert [entry["thought_id"] for entry in omitted["results"]] != []
        assert sorted(entry["thought_id"] for entry in supplied["results"]) == sorted(
            RECENCY_THOUGHT_IDS
        )
        assert LEXICAL_BACKEND in omitted["backends_used"]
        assert LEXICAL_BACKEND in supplied["backends_used"]

        # And the only thing supplying the timestamp changes about the reported
        # backends is that the recency arm joins them — on this route, which the
        # fixture names, so a failure says which launch broke the promise.
        assert RECENCY_BACKEND not in omitted["backends_used"], launch_route
        assert set(supplied["backends_used"]) - set(omitted["backends_used"]) == {
            RECENCY_BACKEND
        }, launch_route
        # The other direction is asserted too, which is what makes the sentence
        # above exact rather than one-sided: a timestamp that switched some
        # other arm off while switching recency on would satisfy the difference
        # and still be a change in what the caller gets back.
        assert set(omitted["backends_used"]) - set(supplied["backends_used"]) == set()

    async def test_ranking_follows_recency_only_with_recency_now(self, launch_route: str) -> None:
        # The diagnostic list above says a backend ran; this says the ranking
        # moved on a corpus built so that only recency can move it.
        async with _client_on_the_resolved_store() as client:
            omitted = await _search(client)
            supplied = await _search(client, recency_now=RECENCY_NOW)

        # Without the timestamp nothing separates the corpus: its entries differ
        # in no field the ranker scores except transaction time, so every score
        # is the same one and the order is whatever the ranker does with a tie.
        # The tie is the assertion, not the order it happens to produce — which
        # tie-break engrava applies is its business, and pinning it would make
        # this test fail on a change that leaves recency working perfectly.
        assert len({entry["score"] for entry in omitted["results"]}) == 1, launch_route
        assert sorted(entry["thought_id"] for entry in omitted["results"]) == sorted(
            RECENCY_THOUGHT_IDS
        ), launch_route

        # With it the same corpus comes back newest first on separated scores,
        # so the order is the ranker's and not a tie-break's.
        assert [entry["thought_id"] for entry in supplied["results"]] == RECENCY_EXPECTED_ORDER, (
            launch_route
        )
        supplied_scores = [entry["score"] for entry in supplied["results"]]
        assert len(set(supplied_scores)) == len(RECENCY_EXPECTED_ORDER), launch_route
        assert supplied_scores == sorted(supplied_scores, reverse=True), launch_route


class TestSupplyingTheTimestampIsNecessaryNotSufficient:
    """A configuration that gives recency no weight ranks without it regardless."""

    async def test_zero_recency_weight_keeps_recency_out_despite_the_argument(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The boundary of the description's claim, and the reason it says
        # "only when" rather than promising the argument switches recency on:
        # the fusion weight belongs to the store's configuration, which the
        # server passes through and does not override.
        _point_at_yaml(monkeypatch, tmp_path, "search:\n  default_recency_weight: 0.0\n")
        async with _client_on_the_resolved_store() as client:
            supplied = await _search(client, recency_now=RECENCY_NOW)

        # The search ran and the corpus is there ...
        assert LEXICAL_BACKEND in supplied["backends_used"]
        assert sorted(entry["thought_id"] for entry in supplied["results"]) == sorted(
            RECENCY_THOUGHT_IDS
        )
        # ... and recency still took no part in it: it is unreported, and the
        # scores tie exactly as they do when the argument is left out, so the
        # signal separated nothing.  Stated on the scores rather than on the
        # order for the same reason as above — the tie-break is not ours to pin.
        assert RECENCY_BACKEND not in supplied["backends_used"]
        assert len({entry["score"] for entry in supplied["results"]}) == 1
