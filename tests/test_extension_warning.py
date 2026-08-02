"""The bare-database launch reports extensions it cannot wire.

An engrava extension that hooks the store is attached through the ``hooks``
section of an ``engrava.yaml``.  The bare-``ENGRAVA_DB_PATH`` launch carries no
configuration, so such an extension is installed, discoverable, and completely
inert — historically with no error and no log line, which is the failure mode
an operator is least likely to notice.  Store resolution therefore names the
extensions installed packages advertise on that launch, and points at the launch
where a store hook can be configured.

**These tests never install a real extension.**  Registering a real
``engrava.extensions`` entry point in the environment would make the warning
fire in every other bare-database test in the suite, and would make the
"nothing advertised, stay silent" test fail for a reason unrelated to the code.
The entry-point lookup is faked through :func:`pytest.MonkeyPatch.setattr`
instead, which unwinds deterministically at the end of each test.  Two tests at
the bottom of this module pin that it really does: one closes a monkeypatch
context and observes the restoration within a single test, so it holds however
the suite is ordered, and one observes it across a test boundary.

The "nothing is imported" property is asserted, not inferred from the launch
surviving: an implementation that loaded each entry point, swallowed the failure
and reported the name anyway would survive too, and that is exactly the
third-party import at startup this must not do.  Three observations cover it,
each blind to what the next one catches: a spying ``load()`` for the direct
call; a real importable probe module that must not appear in
:data:`sys.modules`; and a counter of that probe's *body executions*, kept on
:mod:`sys` so that importing the target and then dropping it from
``sys.modules`` — which defeats both of the first two — is still visible.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import sys
from typing import TYPE_CHECKING, NamedTuple, cast

import aiosqlite
import pytest

from engrava_mcp.config import (
    CONFIG_ENV_VAR,
    DB_PATH_ENV_VAR,
    EXTENSIONS_ENTRY_POINT_GROUP,
    resolve_store,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator
    from pathlib import Path

#: Name the fake extension advertises.  Deliberately not a real project's name.
FAKE_EXTENSION_NAME = "alpha-fake-extension"

#: A second one, so the message is exercised with more than a single name.
#: Neither name is a substring of the other, so an assertion that one is
#: present cannot be satisfied by the other appearing.
OTHER_FAKE_EXTENSION_NAME = "beta-fake-extension"

#: Import target for the fakes.  It does not exist, so ``EntryPoint.load()``
#: raises — see the module docstring.
UNIMPORTABLE_TARGET = "engrava_mcp_no_such_extension_module:MANIFEST"

#: A real, importable module nothing else in the suite imports, so its presence
#: in :data:`sys.modules` is attributable to detection alone.
PROBE_MODULE = "tests.extension_probe"

#: Entry-point target naming :data:`PROBE_MODULE`.
PROBE_TARGET = f"{PROBE_MODULE}:MANIFEST"

#: Attribute on :mod:`sys` that counts executions of the probe's module body.
#: It lives there rather than on the probe so that dropping the probe from
#: :data:`sys.modules` cannot erase the evidence.
PROBE_COUNTER = "_engrava_mcp_extension_probe_executions"


def _probe_executions() -> int:
    """Count how many times the probe module's body has run.

    Returns:
        The current execution count, zero if the probe has never run.

    """
    return int(getattr(sys, PROBE_COUNTER, 0))


#: Phrase that identifies the unwired-extensions warning among everything the
#: launch logs — the bare-database launch also logs the no-embedding-provider
#: warning, so "a warning was logged" is not evidence.
#:
#: Deliberately **not** a phrase any test then asserts is present: selecting on
#: a string and asserting the same string is true by construction and proves
#: nothing.  Every content assertion in this module is on some other part of
#: the message.  This phrase is still pinned, by selection rather than by
#: assertion: drop it from the message and every test here reports finding no
#: warning at all.
WARNING_SELECTOR = "intentionally does not configure store hooks"


def _fake_entry_points(*names: str) -> list[importlib.metadata.EntryPoint]:
    """Build fake extension entry points.

    Args:
        names: Names the fake extensions should advertise.

    Returns:
        Real ``EntryPoint`` objects in the extensions group whose targets are
        unimportable.

    """
    return [
        importlib.metadata.EntryPoint(
            name=name,
            value=UNIMPORTABLE_TARGET,
            group=EXTENSIONS_ENTRY_POINT_GROUP,
        )
        for name in names
    ]


def _install_fake_extensions(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make the extensions group report the named fakes, for one test only.

    Only the extensions group is answered with the fakes; every other group is
    answered empty rather than delegated to the real metadata, so no test can
    depend on what happens to be installed in the environment.

    Args:
        monkeypatch: Fixture whose teardown restores the real lookup.
        names: Names the fake extensions should advertise.

    """
    entries = _fake_entry_points(*names)

    def _entry_points(*, group: str | None = None) -> importlib.metadata.EntryPoints:
        if group == EXTENSIONS_ENTRY_POINT_GROUP:
            return importlib.metadata.EntryPoints(entries)
        return importlib.metadata.EntryPoints(())

    monkeypatch.setattr(importlib.metadata, "entry_points", _entry_points)


class _Accesses(NamedTuple):
    """What a watched entry point observed.

    Attributes:
        loads: Names for which ``load()`` was called.
        value_reads: Names whose ``value`` was read.

    """

    loads: list[str]
    value_reads: list[str]


def _install_access_spy(
    monkeypatch: pytest.MonkeyPatch, *names: str, value: str = UNIMPORTABLE_TARGET
) -> _Accesses:
    """Advertise extensions that record how they are touched.

    Watches the two attributes a caller could use to reach an extension —
    ``load()`` and ``value`` — because reading the target without following it
    is still a layering breach the module's docstring rules out, and neither
    the :data:`sys.modules` check nor the execution counter would see it.

    Args:
        monkeypatch: Fixture whose teardown restores the real lookup.
        names: Names the fake extensions should advertise.
        value: Entry-point target the fakes should report when asked.

    Returns:
        The two live lists, empty until something touches those attributes.

    """
    accesses = _Accesses(loads=[], value_reads=[])

    class _WatchedEntryPoint:
        def __init__(self, name: str) -> None:
            self.name = name

        @property
        def value(self) -> str:
            accesses.value_reads.append(self.name)
            return value

        def load(self) -> object:
            accesses.loads.append(self.name)
            return object()

    entries = [_WatchedEntryPoint(name) for name in names]

    def _entry_points(*, group: str | None = None) -> importlib.metadata.EntryPoints:
        if group == EXTENSIONS_ENTRY_POINT_GROUP:
            return cast("importlib.metadata.EntryPoints", entries)
        return importlib.metadata.EntryPoints(())

    monkeypatch.setattr(importlib.metadata, "entry_points", _entry_points)
    return accesses


#: Message carried by the failures the two breakers below inject.
DISCOVERY_FAILURE_DETAIL = "metadata for engrava.extensions is unreadable"

#: Phrase that identifies the discovery-failure line, distinct from the
#: unwired-extensions warning and from the unrelated no-provider warning.
DISCOVERY_FAILURE_SELECTOR = "cannot report which engrava extensions are advertised"

#: Phrase identifying the unrelated no-embedding-provider warning, which every
#: launch in this module emits.  Naming it lets the silence tests assert on the
#: *whole* set of warnings rather than only on the absence of one phrase: a
#: stray extension diagnostic worded differently would slip past the latter.
NO_PROVIDER_SELECTOR = "No embedding provider configured"


class _ExplodingEntryPoints(importlib.metadata.EntryPoints):
    """Entry points that raise when iterated rather than when looked up.

    A metadata finder is free to defer work: the lookup can return an object
    that only fails once something walks it.  Guarding the call alone would
    leave that case unprotected, so it gets its own breaker.
    """

    def __iter__(self) -> Iterator[importlib.metadata.EntryPoint]:
        """Raise instead of yielding.

        Raises:
            OSError: Always.

        """
        raise OSError(DISCOVERY_FAILURE_DETAIL)


def _break_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the entry-point lookup itself raise, for one test only.

    A broken install or an unusual distribution finder can make reading
    distribution metadata raise.  Discovery is a diagnostic, so that must not
    take down a launch that is otherwise working.

    Args:
        monkeypatch: Fixture whose teardown restores the real lookup.

    """

    def _entry_points(*, group: str | None = None) -> importlib.metadata.EntryPoints:
        _ = group
        raise OSError(DISCOVERY_FAILURE_DETAIL)

    monkeypatch.setattr(importlib.metadata, "entry_points", _entry_points)


def _install_non_string_name(monkeypatch: pytest.MonkeyPatch, name: object) -> None:
    """Advertise an entry point whose ``name`` is not text, for one test only.

    Nothing stops a metadata finder handing back a name that is not a string.
    Joining such a name into the warning would raise, from a code path that
    exists only to produce a diagnostic, after the connection is already open.

    Args:
        monkeypatch: Fixture whose teardown restores the real lookup.
        name: The non-text name to advertise.

    """

    class _OddEntryPoint:
        def __init__(self) -> None:
            self.name = name

    def _entry_points(*, group: str | None = None) -> importlib.metadata.EntryPoints:
        _ = group
        # Deliberately not an EntryPoints: a finder is free to return any
        # iterable, and the point here is that resolution survives it.
        return cast("importlib.metadata.EntryPoints", [_OddEntryPoint()])

    monkeypatch.setattr(importlib.metadata, "entry_points", _entry_points)


def _break_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the lookup succeed but raise while it is walked, for one test only.

    Args:
        monkeypatch: Fixture whose teardown restores the real lookup.

    """

    def _entry_points(*, group: str | None = None) -> importlib.metadata.EntryPoints:
        _ = group
        return _ExplodingEntryPoints(())

    monkeypatch.setattr(importlib.metadata, "entry_points", _entry_points)


def _server_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every warning the server logger emitted, in order.

    Args:
        caplog: Fixture holding the captured records.

    Returns:
        The messages logged through the ``engrava_mcp`` logger.

    """
    return [record.getMessage() for record in caplog.records if record.name == "engrava_mcp"]


def _unwired_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Select the unwired-extensions warnings out of everything logged.

    Args:
        caplog: Fixture holding the captured records.

    Returns:
        The messages of the records carrying :data:`WARNING_SELECTOR`.

    """
    return [
        record.getMessage() for record in caplog.records if WARNING_SELECTOR in record.getMessage()
    ]


@pytest.fixture
def bare_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point store resolution at a throwaway bare database.

    Args:
        monkeypatch: Fixture used to set the environment.
        tmp_path: Temporary directory for the database file.

    """
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "extensions.sqlite"))


@pytest.fixture
def config_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point store resolution at a minimal ``engrava.yaml``.

    Args:
        monkeypatch: Fixture used to set the environment.
        tmp_path: Temporary directory for the config and database files.

    """
    config_file = tmp_path / "engrava.yaml"
    config_file.write_text(
        f"database:\n  path: {tmp_path / 'from-yaml.sqlite'}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_file))


@pytest.fixture
def captured_warnings(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture the server logger's warnings for the duration of a test.

    Args:
        caplog: The underlying capture fixture.

    Yields:
        The same fixture, with the server logger raised to ``WARNING``.

    """
    with caplog.at_level(logging.WARNING, logger="engrava_mcp"):
        yield caplog


class TestUnwiredExtensionWarning:
    """What the bare-database launch says about extensions it cannot wire."""

    async def test_warns_and_names_the_advertised_extension(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bare_db: None,
        captured_warnings: pytest.LogCaptureFixture,
    ) -> None:
        _install_fake_extensions(monkeypatch, FAKE_EXTENSION_NAME)

        resolved = await resolve_store()
        try:
            messages = _unwired_warnings(captured_warnings)
            assert len(messages) == 1, f"expected exactly one warning, got {messages}"
            message = messages[0]
            # Content, not merely "something was logged": the operator needs the
            # name of what is inert, the launch that left it inert, and the
            # launch that would wire it.
            assert FAKE_EXTENSION_NAME in message
            assert DB_PATH_ENV_VAR in message
            assert CONFIG_ENV_VAR in message
            assert "hooks.class" in message
            # The message stops short of asserting what the extension does:
            # entry-point metadata says a distribution advertises one, not that
            # it hooks the store, so it says "advertise" and scopes the effect
            # to an extension's store hooks.
            assert "advertise" in message
            # Not bare "store hooks": that is inside WARNING_SELECTOR, so
            # asserting it would be satisfied by the selection itself.
            assert "an extension's store hooks" in message
            # And it tells an operator who wanted nothing from extensions that
            # there is nothing to do here.
            assert "Otherwise no action is needed" in message
        finally:
            await resolved.aclose()

    async def test_names_every_advertised_extension(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bare_db: None,
        captured_warnings: pytest.LogCaptureFixture,
    ) -> None:
        # Still one warning, not one per extension, and every name is in it.
        # Advertised in reverse order on purpose: metadata iteration order is
        # not something a distribution finder promises, so the message must not
        # inherit it. Naming them in a stable order is what makes two launches
        # of the same environment produce the same line.
        _install_fake_extensions(monkeypatch, OTHER_FAKE_EXTENSION_NAME, FAKE_EXTENSION_NAME)

        resolved = await resolve_store()
        try:
            messages = _unwired_warnings(captured_warnings)
            assert len(messages) == 1, f"expected exactly one warning, got {messages}"
            assert FAKE_EXTENSION_NAME in messages[0]
            assert OTHER_FAKE_EXTENSION_NAME in messages[0]
            assert messages[0].index(FAKE_EXTENSION_NAME) < messages[0].index(
                OTHER_FAKE_EXTENSION_NAME
            ), "names were reported in metadata order rather than a stable one"
        finally:
            await resolved.aclose()

    async def test_manifests_are_never_loaded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bare_db: None,
        captured_warnings: pytest.LogCaptureFixture,
    ) -> None:
        # Two independent observations, because either alone can be evaded.
        #
        # The load() spy catches the direct call. On its own it would miss an
        # implementation that read entry_point.value and imported the module
        # itself, which is the same third-party-code-at-startup outcome by a
        # different route.
        #
        # So the entry point points at a real, importable, otherwise-unused
        # probe module, watched two further ways. It must not enter
        # sys.modules, which covers the ordinary import routes; and the probe's
        # module body must not have run, counted on sys rather than on the probe
        # so that importing and then dropping it from sys.modules cannot hide
        # the execution. The counter is the conclusive one: it is indifferent to
        # how execution was reached. An unimportable target could not tell
        # "never imported" from "tried and failed", hence a real one.
        sys.modules.pop(PROBE_MODULE, None)
        executions_before = _probe_executions()
        accesses = _install_access_spy(monkeypatch, FAKE_EXTENSION_NAME, value=PROBE_TARGET)

        resolved = await resolve_store()
        try:
            assert accesses.loads == [], f"detection called load() on {accesses.loads}"
            # Reading the target without following it is still a breach of the
            # layering the module docstring states, and neither of the two
            # checks below would notice it.
            assert accesses.value_reads == [], (
                f"detection read the target of {accesses.value_reads}"
            )
            assert PROBE_MODULE not in sys.modules, "detection imported the extension target"
            assert _probe_executions() == executions_before, (
                "the extension target's module body ran"
            )
            # And the name still reached the warning, so "nothing imported" is
            # not because detection quietly did nothing.
            messages = _unwired_warnings(captured_warnings)
            assert len(messages) == 1, f"expected exactly one warning, got {messages}"
            assert FAKE_EXTENSION_NAME in messages[0]
        finally:
            await resolved.aclose()

    async def test_silent_when_nothing_is_advertised(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bare_db: None,
        captured_warnings: pytest.LogCaptureFixture,
    ) -> None:
        # The environment this suite runs in has no extensions installed, but
        # asserting that would make this test depend on the environment. The
        # empty group is faked explicitly instead.
        _install_fake_extensions(monkeypatch)

        resolved = await resolve_store()
        try:
            # Asserted over every warning the launch emitted, not just over the
            # absence of one phrase: an extension diagnostic worded differently
            # would be invisible to a selector-only check. The one expected
            # warning is unrelated to extensions, which also means this is not
            # passing because nothing was logged at all.
            emitted = _server_warnings(captured_warnings)
            assert len(emitted) == 1, f"expected exactly one warning, got {emitted}"
            assert NO_PROVIDER_SELECTOR in emitted[0]
        finally:
            await resolved.aclose()

    async def test_config_path_never_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config_path: None,
        captured_warnings: pytest.LogCaptureFixture,
    ) -> None:
        # Extensions installed, but this launch can wire them, so there is
        # nothing to report. Same fakes as the warning test above — the launch
        # is the only thing that differs.
        _install_fake_extensions(monkeypatch, FAKE_EXTENSION_NAME)

        resolved = await resolve_store()
        try:
            # Again over the whole set: this launch may legitimately warn that
            # no embedding provider is configured, and nothing else.
            emitted = _server_warnings(captured_warnings)
            assert len(emitted) == 1, f"expected exactly one warning, got {emitted}"
            assert NO_PROVIDER_SELECTOR in emitted[0]
        finally:
            await resolved.aclose()


class TestAFailedLaunchStaysQuiet:
    """A launch that fails before the store is built says nothing about them.

    Narrowly that, and not "a failed launch is silent": a launch can also fail
    *during* the report, in which case the report has already happened.  That
    case is the one below this class.

    The scan runs before the connection is opened, which is what keeps it out of
    the window where a connection exists that no caller can close.  The risk in
    that ordering is the mirror image: reporting something about a launch that
    then fails, where previously nothing was said at all.  Reporting happens
    after the store is safely built, so it does not.
    """

    async def test_a_failed_launch_reports_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        captured_warnings: pytest.LogCaptureFixture,
    ) -> None:
        # Extensions are advertised, so there would be something to say; the
        # database path is unopenable, so the launch never gets far enough to
        # say it.
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "no-such-dir" / "store.sqlite"))
        _install_fake_extensions(monkeypatch, FAKE_EXTENSION_NAME)

        with pytest.raises(Exception, match="unable to open database file"):
            await resolve_store()

        assert _server_warnings(captured_warnings) == []

    async def test_a_failed_launch_stays_quiet_about_a_broken_scan_too(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        captured_warnings: pytest.LogCaptureFixture,
    ) -> None:
        # Same for the discovery-failure line: the scan fails, the launch fails,
        # and neither is reported because there is no store to report about.
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "no-such-dir" / "store.sqlite"))
        _break_discovery(monkeypatch)

        with pytest.raises(Exception, match="unable to open database file"):
            await resolve_store()

        assert _server_warnings(captured_warnings) == []


class TestDiscoveryFailureIsNotFatal:
    """An ordinary failure in a diagnostic must not take down a working launch.

    Bounded to ``Exception`` throughout, like the code: cancellation and
    interpreter shutdown still propagate, and should.
    """

    async def test_launch_survives_unreadable_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bare_db: None,
        captured_warnings: pytest.LogCaptureFixture,
    ) -> None:
        # Discovery raises. Resolution must still hand back a usable store —
        # the store is the product, the extension report is a courtesy — and
        # the failure must not itself be silent.
        _break_discovery(monkeypatch)

        resolved = await resolve_store()
        try:
            assert await resolved.store.max_cycle() == 0, "the store is not usable"
            assert _unwired_warnings(captured_warnings) == []
            reported = [
                record.getMessage()
                for record in captured_warnings.records
                if DISCOVERY_FAILURE_SELECTOR in record.getMessage()
            ]
            assert len(reported) == 1, f"expected one discovery-failure line, got {reported}"
            # The cause is carried, not swallowed: an operator debugging this
            # needs the underlying error, not just that something went wrong.
            assert DISCOVERY_FAILURE_DETAIL in reported[0]
        finally:
            await resolved.aclose()

    async def test_launch_survives_metadata_that_fails_when_walked(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bare_db: None,
        captured_warnings: pytest.LogCaptureFixture,
    ) -> None:
        # The lookup succeeds and the failure comes later, while the result is
        # being walked. Guarding only the call would miss this, so it is driven
        # separately rather than assumed to be the same case.
        _break_iteration(monkeypatch)

        resolved = await resolve_store()
        try:
            assert _unwired_warnings(captured_warnings) == []
            reported = [
                record.getMessage()
                for record in captured_warnings.records
                if DISCOVERY_FAILURE_DETAIL in record.getMessage()
            ]
            assert len(reported) == 1, f"expected one discovery-failure line, got {reported}"
        finally:
            await resolved.aclose()

    async def test_launch_survives_a_name_that_is_not_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bare_db: None,
        captured_warnings: pytest.LogCaptureFixture,
    ) -> None:
        # Lookup, iteration and name access all succeed; the name is simply not
        # a string. Building the warning is where that used to bite, after the
        # connection was open. Resolution must survive it and still report.
        _install_non_string_name(monkeypatch, 17)

        resolved = await resolve_store()
        try:
            messages = _unwired_warnings(captured_warnings)
            assert len(messages) == 1, f"expected exactly one warning, got {messages}"
            assert "17" in messages[0]
        finally:
            await resolved.aclose()


class TestLoggingFailureIsNotFatal:
    """A logging stack this process does not own cannot fail the *new* warning.

    Scoped twice over, and both bounds matter.  It covers the two diagnostics
    this change adds, not the pre-existing no-provider warning, which is
    deliberately still unguarded and can still fail a launch — the handler below
    raises only on the extension warning for exactly that reason.  And it is
    bounded to ``Exception``: a handler raising ``KeyboardInterrupt`` is not
    something a diagnostic should swallow.
    """

    async def test_launch_survives_a_handler_that_raises(
        self, monkeypatch: pytest.MonkeyPatch, bare_db: None
    ) -> None:
        # Handlers and filters belong to the embedding application, and one that
        # raises would otherwise take down a launch that had already succeeded
        # over a line of advice. Scoped to the diagnostics this change adds: the
        # pre-existing no-provider warning keeps whatever behaviour it had.
        _install_fake_extensions(monkeypatch, FAKE_EXTENSION_NAME)

        emitted: list[str] = []

        class _HostileHandler(logging.Handler):
            """Raises only on the extension warning.

            The no-provider warning is not routed through the soft emitter, so a
            handler that raised on everything would abort the launch there and
            never reach the diagnostic under test.
            """

            def emit(self, record: logging.LogRecord) -> None:
                message = record.getMessage()
                emitted.append(message)
                if WARNING_SELECTOR in message:
                    msg = "this handler is broken"
                    raise RuntimeError(msg)

        server_logger = logging.getLogger("engrava_mcp")
        handler = _HostileHandler()
        server_logger.addHandler(handler)
        previous_level = server_logger.level
        server_logger.setLevel(logging.WARNING)
        try:
            resolved = await resolve_store()
            try:
                assert await resolved.store.max_cycle() == 0, "the store is not usable"
            finally:
                await resolved.aclose()
        finally:
            server_logger.removeHandler(handler)
            server_logger.setLevel(previous_level)

        # The extension warning was attempted, raised, and did not escape. The
        # unrelated warning is asserted too, so this cannot pass because the
        # launch stopped emitting altogether.
        assert len(emitted) == 2, f"expected both startup warnings attempted, got {emitted}"
        assert len([m for m in emitted if NO_PROVIDER_SELECTOR in m]) == 1
        assert len([m for m in emitted if WARNING_SELECTOR in m]) == 1


class TestTheExceptionBoundaryIsWhereItSays:
    """The guards contain ``Exception`` and deliberately not ``BaseException``.

    Both guards belong to this change — the scan's and the soft emitter's — and
    widening either to ``BaseException`` would swallow a cancellation the caller
    asked for while leaving every other test green.
    """

    async def test_discovery_does_not_swallow_cancellation(
        self, monkeypatch: pytest.MonkeyPatch, bare_db: None
    ) -> None:
        def _entry_points(*, group: str | None = None) -> importlib.metadata.EntryPoints:
            _ = group
            raise asyncio.CancelledError

        monkeypatch.setattr(importlib.metadata, "entry_points", _entry_points)

        with pytest.raises(asyncio.CancelledError):
            await resolve_store()

    async def test_the_soft_emitter_does_not_swallow_cancellation(
        self, monkeypatch: pytest.MonkeyPatch, bare_db: None
    ) -> None:
        # Through a handler rather than by patching the emitter, so it is the
        # emitter's own suppression boundary under test and not a stand-in.
        #
        # The cancellation unwinds through the report, which sits between a
        # successful build and the return — the one window where the connection
        # exists and no caller can close it. So this asserts both halves: the
        # cancellation reaches the caller, and the connection does not outlive
        # the launch that opened it.
        _install_fake_extensions(monkeypatch, FAKE_EXTENSION_NAME)

        opened: list[aiosqlite.Connection] = []
        real_connect = aiosqlite.connect

        def _tracking_connect(*args: object, **kwargs: object) -> aiosqlite.Connection:
            connection = real_connect(*args, **kwargs)  # type: ignore[arg-type]
            opened.append(connection)
            return connection

        monkeypatch.setattr(aiosqlite, "connect", _tracking_connect)

        class _CancellingHandler(logging.Handler):
            """Cancels only on the extension warning, for the reason above."""

            def emit(self, record: logging.LogRecord) -> None:
                if WARNING_SELECTOR in record.getMessage():
                    raise asyncio.CancelledError

        server_logger = logging.getLogger("engrava_mcp")
        handler = _CancellingHandler()
        server_logger.addHandler(handler)
        previous_level = server_logger.level
        server_logger.setLevel(logging.WARNING)
        try:
            with pytest.raises(asyncio.CancelledError):
                await resolve_store()
        finally:
            server_logger.removeHandler(handler)
            server_logger.setLevel(previous_level)

        assert len(opened) == 1, f"expected exactly one connection, saw {len(opened)}"
        with pytest.raises(ValueError, match="no active connection"):
            await opened[0].execute("SELECT 1")

    async def test_a_failing_close_does_not_replace_the_cancellation(
        self, monkeypatch: pytest.MonkeyPatch, bare_db: None
    ) -> None:
        # The report is cancelled and closing then fails too. What the caller
        # gets must still be the cancellation: "the close failed as well" is not
        # the more useful half of that story, and there is nothing further to do
        # about it either way.
        _install_fake_extensions(monkeypatch, FAKE_EXTENSION_NAME)

        closes: list[str] = []
        pending: list[Callable[[], Awaitable[None]]] = []
        real_connect = aiosqlite.connect

        def _connect_with_failing_close(*args: object, **kwargs: object) -> aiosqlite.Connection:
            connection = real_connect(*args, **kwargs)  # type: ignore[arg-type]
            pending.append(connection.close)

            async def _close() -> None:
                closes.append("attempted")
                msg = "the close itself failed"
                raise OSError(msg)

            connection.close = _close  # type: ignore[method-assign]
            return connection

        monkeypatch.setattr(aiosqlite, "connect", _connect_with_failing_close)

        class _CancellingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if WARNING_SELECTOR in record.getMessage():
                    raise asyncio.CancelledError

        server_logger = logging.getLogger("engrava_mcp")
        handler = _CancellingHandler()
        server_logger.addHandler(handler)
        previous_level = server_logger.level
        server_logger.setLevel(logging.WARNING)
        try:
            # CancelledError, not OSError.
            with pytest.raises(asyncio.CancelledError):
                await resolve_store()
            assert closes == ["attempted"], "cleanup never tried to close the connection"
        finally:
            server_logger.removeHandler(handler)
            server_logger.setLevel(previous_level)
            # The stand-in close never completes, so this test owns the real one.
            for close in pending:
                await close()


class TestFakeExtensionsDoNotLeak:
    """The fake entry points are confined to the test that installs them.

    A monkeypatched entry-point lookup that outlived its test would make the
    warning fire in unrelated bare-database tests.  Those would go *red*, not
    quietly green — the silence tests assert over the whole warning set — so the
    cost is not a false pass but a failure landing far from its cause, in a test
    that has nothing to do with extensions.  That is worth pinning here, where
    it says what it is.

    The first test observes the restoration inside a single test by closing a
    monkeypatch context, so it holds no matter how the suite is ordered. The
    pair after it observes the same thing across a test boundary, which is the
    situation that actually bites, at the cost of depending on file order.
    """

    async def test_the_fake_is_gone_once_its_context_closes(self, bare_db: None) -> None:
        with pytest.MonkeyPatch.context() as patched:
            _install_fake_extensions(patched, FAKE_EXTENSION_NAME)
            inside = [
                entry_point.name
                for entry_point in importlib.metadata.entry_points(
                    group=EXTENSIONS_ENTRY_POINT_GROUP
                )
            ]
        outside = [
            entry_point.name
            for entry_point in importlib.metadata.entry_points(group=EXTENSIONS_ENTRY_POINT_GROUP)
        ]
        assert FAKE_EXTENSION_NAME in inside
        assert FAKE_EXTENSION_NAME not in outside

    async def test_first_installs_a_fake(
        self, monkeypatch: pytest.MonkeyPatch, bare_db: None
    ) -> None:
        _install_fake_extensions(monkeypatch, FAKE_EXTENSION_NAME)
        resolved = await resolve_store()
        try:
            assert FAKE_EXTENSION_NAME in [
                entry_point.name
                for entry_point in importlib.metadata.entry_points(
                    group=EXTENSIONS_ENTRY_POINT_GROUP
                )
            ]
        finally:
            await resolved.aclose()

    async def test_then_the_real_lookup_is_back(
        self, bare_db: None, captured_warnings: pytest.LogCaptureFixture
    ) -> None:
        # No fake installed here: the lookup must be the real one again, and
        # the real environment has no extensions, so resolution stays silent.
        assert FAKE_EXTENSION_NAME not in [
            entry_point.name
            for entry_point in importlib.metadata.entry_points(group=EXTENSIONS_ENTRY_POINT_GROUP)
        ]

        resolved = await resolve_store()
        try:
            assert _unwired_warnings(captured_warnings) == []
        finally:
            await resolved.aclose()
