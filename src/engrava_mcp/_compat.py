"""Engrava compatibility helpers for the MCP server.

The server consumes engrava's PUBLIC API. Two compatibility concerns live
here so they are unit-testable in isolation from the import machinery and
the running server:

* :func:`incompatible_engrava_message` builds the actionable message the
  package raises when the engrava public API cannot be imported at all (a
  hard, fatal incompatibility caught in :mod:`engrava_mcp`'s ``__init__``).
* :func:`warn_if_engrava_out_of_range` emits a soft, non-fatal warning at
  startup when the *installed* engrava version falls outside the range this
  server is tested against. It never raises: a dev/editable engrava reports
  a pre-release number whose public API is in practice compatible, and the
  install-time dependency range is the real gate.
"""

from __future__ import annotations

import importlib.metadata
import warnings

#: Lowest engrava version (inclusive) this server is tested against.
#: Keep in sync with the ``engrava>=0.5,<0.6`` requirement in ``pyproject.toml``.
ENGRAVA_MIN_VERSION = (0, 5)

#: Lowest engrava version (exclusive) that is out of range — the next minor.
#: Keep in sync with the ``engrava>=0.5,<0.6`` requirement in ``pyproject.toml``.
ENGRAVA_MAX_VERSION_EXCLUSIVE = (0, 6)

#: Human-readable supported range, used in user-facing messages. Keep in sync
#: with the ``engrava>=0.5,<0.6`` requirement in ``pyproject.toml``.
ENGRAVA_SUPPORTED_RANGE = ">=0.5,<0.6"


class EngravaVersionWarning(UserWarning):
    """Warn that the installed engrava version is outside the tested range.

    Emitted by :func:`warn_if_engrava_out_of_range` at server startup. It is
    a warning, never an error: behaviour with an untested engrava version is
    unverified rather than known-broken.
    """


def incompatible_engrava_message(exc: BaseException) -> str:
    """Build the actionable message for an unimportable engrava public API.

    The server consumes engrava's PUBLIC API. A missing symbol when importing
    it means an incompatible ``engrava`` is installed. This builds a message
    that names the dependency, states the supported range, tells the operator
    how to recover, and preserves the original error for diagnosis.

    Args:
        exc: The original import-time exception to quote in the message.

    Returns:
        An actionable, single-string error message.

    """
    return (
        "engrava-mcp could not load the engrava public API it depends on. "
        "This usually means an incompatible 'engrava' is installed — engrava-mcp "
        f"requires engrava {ENGRAVA_SUPPORTED_RANGE}. Reinstall engrava-mcp "
        "(which pulls a compatible engrava) or align your engrava version. "
        f"Original import error: {exc}"
    )


def _parse_major_minor(version: str) -> tuple[int, int] | None:
    """Parse the leading ``major.minor`` pair from a version string.

    Args:
        version: A version string, e.g. ``"0.5.2"`` or ``"0.6.0rc1"``.

    Returns:
        The ``(major, minor)`` tuple, or ``None`` if it cannot be parsed.

    """
    parts = version.split(".")
    if len(parts) < 2:  # noqa: PLR2004 - a version needs at least major.minor
        return None
    try:
        major = int(parts[0])
    except ValueError:
        return None
    # Strip any pre-release/build suffix on the minor segment (e.g. "6rc1").
    minor_digits = ""
    for char in parts[1]:
        if not char.isdigit():
            break
        minor_digits += char
    if not minor_digits:
        return None
    return (major, int(minor_digits))


def warn_if_engrava_out_of_range() -> None:
    """Warn if the installed engrava version is outside the tested range.

    Reads the installed engrava version via :func:`importlib.metadata.version`
    and, if its ``major.minor`` is outside the supported range
    (:data:`ENGRAVA_SUPPORTED_RANGE`), emits an :class:`EngravaVersionWarning`.
    It never raises: an unparseable or missing version, or an out-of-range one,
    only warns. Call this once at startup, not at module import, to avoid noise
    on every ``import engrava_mcp``.
    """
    try:
        installed = importlib.metadata.version("engrava")
    except importlib.metadata.PackageNotFoundError:
        # No distribution metadata (unusual in practice). The install-time
        # dependency range is the real gate; stay silent rather than warn on a
        # missing-metadata edge that does not imply incompatibility.
        return

    parsed = _parse_major_minor(installed)
    if parsed is None:
        return

    if not (ENGRAVA_MIN_VERSION <= parsed < ENGRAVA_MAX_VERSION_EXCLUSIVE):
        warnings.warn(
            f"Installed engrava version {installed} is outside engrava-mcp's "
            f"tested range ({ENGRAVA_SUPPORTED_RANGE}); behaviour is unverified. "
            "Align your engrava version if you hit unexpected results.",
            EngravaVersionWarning,
            stacklevel=2,
        )
