"""MCPServer server exposing engrava's read API as agent tools.

This module builds a Model Context Protocol server that wraps the public
async read API of :class:`~engrava.SqliteEngravaCore`.  It is an *API
consumer*, not an engrava extension: it registers no hooks, manifests, or
MindQL extension commands.  Think of it as a sibling of the command-line
interface that speaks MCP over stdio.

Eight read-only tools are exposed:

``get_thought``
    Fetch a single thought by identifier.
``search_memory``
    Hybrid (lexical + vector + recency) ranked search.  Optional
    ``thought_type`` / ``lifecycle_status`` / ``priority`` filters narrow
    the ranked hits *after* ranking (the hybrid ranker cannot filter), so
    a filtered call may return fewer than ``top_k`` results and reports
    how many ranked hits it dropped.
``search_keywords``
    Pure full-text BM25 keyword search.
``list_memory``
    Unranked browse over stored thoughts with the full
    filter matrix (``thought_type``, ``lifecycle_status``, ``priority``,
    updated-cycle range) and ``limit`` / ``offset`` pagination.  Returns
    thoughts ordered by highest cognitive cycle first with no score — the
    clean home for "list memory by structured field", complementing the
    ranked ``search_memory``.  Thoughts written through this server all
    carry cycle 0, so their relative order is unspecified
    (:data:`_CYCLE_ORDERING_NOTE`).
``query_memory``
    Structured ``FIND`` queries in the MindQL query language.  Only the
    ``FIND`` command is accepted; raw-SQL passthrough and every other
    command are rejected.  The grammar itself has an ``OFFSET`` clause, but
    this tool exposes no separate ``offset`` argument, so it paginates by
    ``limit`` only (a caller wanting an offset writes it directly in the
    query text).  An optional ``limit`` always overrides any ``LIMIT`` the
    query text carries; a query with no ``LIMIT`` at all is capped rather
    than run unbounded.
``memory_stats``
    Aggregate counts and store-health metrics.
``get_edges``
    Fetch the edges connected to a thought (``IN`` / ``OUT`` / ``BOTH``),
    returning full edge records including their metadata.
``list_edges``
    Browse stored edges filtered by edge type, knowledge source, and
    edge metadata (``metadata_equals`` / ``metadata_in``), returning full
    edge records including their metadata.

Five write tools complete the surface:

``store_thought``
    Create a new thought node.
``update_thought``
    Mutate selected fields of an existing thought.
``link_thoughts``
    Create a typed edge between two existing thoughts.
``delete_thought``
    Remove a thought by identifier.
``delete_edge``
    Remove an edge by identifier.

The write tools are gated by the :data:`READ_ONLY_ENV_VAR` environment
variable.  When it is set to a truthy value the write tools are not
registered at all, so a read-only deployment never advertises them to
clients.  The read tools are always available, and in read-only mode they run
against a read-only view (see :mod:`engrava_mcp.read_only`) rather than the
raw store, so a read never stages a write of its own either — including a
deferred access-count update a store with access tracking on would otherwise
buffer and flush on close.

Three read-only *resources* round out the surface.  Where tools are
*invoked*, resources are addressable ``engrava://`` URIs that clients
surface as attachable context:

``engrava://thought/{thought_id}``
    A single thought as a JSON document.  Reading an unknown identifier
    yields a graceful not-found payload rather than an error.
``engrava://stats``
    Store-health counts and size, identical to the ``memory_stats`` tool
    (both share :func:`memory_stats_impl`).
``engrava://recent``
    The stored thoughts, ordered by highest cognitive cycle first
    (:data:`_CYCLE_ORDERING_NOTE`), as a JSON document.

Resources are reads by definition, so — unlike the write tools — they are
*not* gated by :data:`READ_ONLY_ENV_VAR`; they are advertised in both the
default and read-only deployments.

Three *prompts* complete the surface.  Prompts are parameterised templates
that a client surfaces as slash-commands or buttons; each one renders a
ready-to-send instruction that guides the assistant to gather context with
the read tools and resources above.  They are templates only — they open no
write path and call no store method:

``summarize_recent_memory``
    Summarise the thoughts with the highest cognitive cycle
    (:data:`_CYCLE_ORDERING_NOTE`).  Takes an optional ``limit`` (how many
    to consider).
``find_related``
    Find and synthesise thoughts related to a required ``topic``.
``reflect_on_topic``
    Reflect over what memory holds about a required ``topic``.

Prompts are read-oriented, so — like the resources — they are *not* gated by
:data:`READ_ONLY_ENV_VAR` and are advertised in both deployments.

The active store is supplied to tool and resource calls through a
:class:`StoreProvider` that the server's lifespan populates on startup and
clears on shutdown.  Each tool delegates to a module-level implementation
function that takes an explicit store argument, which keeps the query and
mutation logic unit-testable without a running server.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Any, Literal

import anyio
from engrava import (
    ConnectionQuarantinedError,
    DerivedRecordError,
    DuplicateEdgeError,
    EdgeRecord,
    EdgeType,
    EmbeddingGenerationError,
    EmbeddingModelMismatchError,
    EmbeddingQueryPrefixMismatchError,
    FieldOp,
    FieldPredicate,
    InvalidFilterError,
    InvalidFilterPathError,
    InvalidRecencyArgumentError,
    InvalidTransitionError,
    KnowledgeSource,
    LifecycleStatus,
    MetadataFilter,
    MindQLCommand,
    MindQLParseError,
    MindQLQuery,
    Priority,
    ReferentialIntegrityError,
    StaleDataError,
    ThoughtNotFoundError,
    ThoughtRecord,
    ThoughtType,
    VectorDimensionMismatchError,
    WriteContentionError,
    WriteLockTimeoutError,
    parse,
)
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS, ToolAnnotations
from pydantic import Field, ValidationError

from engrava_mcp._compat import warn_if_engrava_out_of_range
from engrava_mcp.config import ResolvedStore, resolve_store
from engrava_mcp.read_only import ReadOnlyStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from engrava import SqliteEngravaCore

    from engrava_mcp.read_only import ReadOnlyMcpStore

#: Direction of edge traversal for ``get_edges``.  ``OUT`` follows edges whose
#: source is the given thought, ``IN`` follows edges whose target is it, and
#: ``BOTH`` returns either.  Declaring it as a :data:`~typing.Literal` makes the
#: MCP tool schema enumerate the three accepted values.
EdgeDirection = Literal["IN", "OUT", "BOTH"]

#: A JSON scalar accepted as a metadata filter value on ``list_edges`` and as a
#: metadata value on ``link_thoughts``.  The metadata surface is deliberately
#: kept to plain JSON scalars over the wire: nested objects and the typed filter
#: machinery are never exposed to clients.
JsonScalar = str | int | float | bool | None

#: Module logger.  Shares its name with :data:`engrava_mcp.config.logger`
#: (``logging.getLogger`` returns the same object for the same name), so a
#: shutdown-time warning logged from either module reaches the same handlers
#: and the same ``tests/test_shutdown.py`` capture.
logger = logging.getLogger("engrava_mcp")

#: Server name advertised to MCP clients.
SERVER_NAME = "engrava"

#: Distribution whose installed version is advertised to MCP clients as this
#: server's own ``serverInfo.version`` — never the ``mcp`` SDK's.  Keep in
#: sync with ``project.name`` in ``pyproject.toml``.
_DISTRIBUTION_NAME = "engrava-mcp"

#: Advertised when the installed distribution carries no version metadata to
#: read (a normal PEP 660 editable install does; a vendored or otherwise
#: unusual checkout might not).  Chosen over crashing at startup, and over
#: silently falling back to the ``mcp`` SDK's own version — the defect
#: :func:`_server_version` exists to fix.
_UNKNOWN_VERSION = "unknown"


def _server_version() -> str:
    """Resolve the version to advertise to MCP clients as this server's own.

    Reads the installed :data:`_DISTRIBUTION_NAME` distribution's version via
    :func:`importlib.metadata.version` — the same number already set by hand
    in ``pyproject.toml`` and in both ``server.json`` fields at release time —
    rather than a fourth hand-maintained literal that could drift from those.

    Returns:
        The installed ``engrava-mcp`` version, or :data:`_UNKNOWN_VERSION` if
        no distribution metadata can be found.

    """
    try:
        return importlib.metadata.version(_DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        return _UNKNOWN_VERSION


#: Default number of results returned by search tools.
DEFAULT_TOP_K = 10

#: Default number of thoughts returned by the ``engrava://recent`` resource.
DEFAULT_RECENT_LIMIT = 10

#: The ordering clause shared by every description of cycle-ordered output
#: (the ``list_memory`` tool, the ``engrava://recent`` resource, and the
#: ``summarize_recent_memory`` prompt).  Cognitive cycles are a signal the
#: consuming application supplies (see engrava's
#: :meth:`~engrava.SqliteEngravaCore.list_thoughts`); an MCP client has none,
#: so every thought this server writes stamps ``created_cycle =
#: updated_cycle = 0`` and no update advances it.  Ordering by descending
#: ``updated_cycle`` is real, but the underlying query has no tiebreaker, so
#: SQLite guarantees nothing at all about how the tied rows come back
#: relative to each other — a fixed rule is exactly the kind of guarantee
#: this constant must not assert, no matter how it is phrased.  Say only
#: what is true: the order among ties is unspecified and carries no
#: recency meaning.  Stated once so the tool description and the resource
#: description cannot drift apart.
_CYCLE_ORDERING_NOTE = (
    "ordered by highest cognitive cycle first; thoughts written through "
    "this server all carry cycle 0, so their relative order is "
    "unspecified and does not represent recency"
)

#: Default page size for the ``list_memory`` browse tool.  Matches the
#: store's own ``list_thoughts`` default so an unpaged listing behaves the
#: same whether driven through MCP or the core API directly.
DEFAULT_LIST_LIMIT = 50

#: Default number of edges returned by the ``list_edges`` browse tool.
#: Deliberately smaller than the store's own ``list_edges`` default (5000):
#: an MCP response is read into an agent's context, so a focused page is a
#: better default over the wire than a bulk dump.  Callers that genuinely want
#: more can raise ``limit`` explicitly.
DEFAULT_EDGE_LIST_LIMIT = 100

#: Largest page size any wire-supplied ``limit`` may request (``list_memory``,
#: ``list_edges``, ``query_memory``, and the recent-thoughts listing).  Mirrors
#: engrava's own ``list_edges`` ceiling.  A bound crossing the wire is a *scan
#: cap*: without an upper bound a caller can request the whole store in one
#: call, and without a lower bound a negative value reaches SQLite, which reads
#: ``LIMIT -1`` as "no limit" and defeats the cap entirely.
MAX_PAGE_LIMIT = 5000

#: Largest ``top_k`` the ranked search tools accept.  Lower than
#: :data:`MAX_PAGE_LIMIT` because a ranked window is read into an agent's
#: context rather than paged through.
MAX_TOP_K = 1000

#: A page size supplied over the wire: at least one row, never more than the
#: scan cap.  Expressed as an annotated type so the bound lands in the
#: *advertised* MCP tool schema and pydantic enforces it at the protocol layer.
PageLimit = Annotated[int, Field(ge=1, le=MAX_PAGE_LIMIT)]

#: A ranked-window size supplied over the wire.
TopK = Annotated[int, Field(ge=1, le=MAX_TOP_K)]

#: Largest value SQLite's ``INTEGER`` storage class can bind: a signed 64-bit
#: integer.  Binding anything outside ``[SQLITE_MIN_BOUND_INT,
#: SQLITE_MAX_BOUND_INT]`` does not reach SQLite at all — the ``sqlite3``
#: driver itself raises a bare ``OverflowError`` while converting the Python
#: ``int``, with no argument name or context attached.  Every wire-supplied
#: integer that ends up bound into a SQLite parameter must therefore be
#: constrained to this range before it can reach a bind call.
SQLITE_MAX_BOUND_INT = 2**63 - 1

#: Smallest value SQLite's ``INTEGER`` storage class can bind — the other end
#: of :data:`SQLITE_MAX_BOUND_INT`.
SQLITE_MIN_BOUND_INT = -(2**63)

#: A page offset supplied over the wire.  Zero is a valid page start.  The
#: upper bound is not "no ceiling" (an earlier version of this comment claimed
#: exactly that): ``offset`` is bound into a SQLite query parameter, so it is
#: constrained to SQLite's own signed-64-bit integer ceiling, the same as any
#: other SQLite-bound integer crossing this wire.
PageOffset = Annotated[int, Field(ge=0, le=SQLITE_MAX_BOUND_INT)]

#: An ``updated_cycle`` filter bound (``min_cycle`` / ``max_cycle`` on
#: ``list_memory``) supplied over the wire.  Bound into a SQLite comparison,
#: so — like :data:`PageOffset` — it is constrained to SQLite's own bind
#: range.  Unlike ``offset`` the lower end is left at SQLite's own floor
#: rather than 0: a negative bound is harmless (``updated_cycle`` is always
#: non-negative, so it simply excludes nothing) and rejecting it would change
#: accepted-input behaviour beyond what avoiding the overflow requires.
CycleFilterBound = Annotated[int, Field(ge=SQLITE_MIN_BOUND_INT, le=SQLITE_MAX_BOUND_INT)]

#: A metadata-filter key accepted on ``list_edges``.  The thin surface accepts
#: only simple, top-level field names — a dotted or bracketed key (``a.b``,
#: ``tags[0]``) would build a nested engrava JSONPath (``$.a.b``, ``$.tags[0]``)
#: and reach nested metadata the MCP surface deliberately does not support, so
#: it is rejected at the boundary before any path is constructed.
METADATA_FIELD_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")

#: Default number of thoughts the ``summarize_recent_memory`` prompt asks
#: the assistant to consider when the caller omits ``limit``.  Kept small
#: so the summary stays focused rather than embedding the whole store.
DEFAULT_SUMMARY_LIMIT = 5

#: MIME type advertised for every ``engrava://`` resource.  Resource
#: handlers return a JSON document as text, so clients receive a stable,
#: machine-parseable content type.
RESOURCE_MIME_TYPE = "application/json"

#: Default edge weight when a caller does not supply one.
DEFAULT_EDGE_WEIGHT = 1.0

#: Cycle counter assigned to thoughts and edges created through the MCP
#: write surface.  This API consumer has no notion of a cognitive cycle
#: clock, so new records start at the origin cycle.
INITIAL_CYCLE = 0

#: A valid MindQL ``FIND`` query, embedded verbatim in the actionable hints
#: that ``query_memory`` returns when a caller sends a malformed or
#: unsupported query.  Showing one correct example is the fastest way to get
#: a client back onto the supported path; it deliberately demonstrates only
#: the ``FIND`` command, never raw SQL.
FIND_QUERY_EXAMPLE = "FIND thoughts WHERE lifecycle_status = 'ACTIVE' LIMIT 10"

#: Client-facing message for a duplicate ``link_thoughts`` edge.  The store may
#: report the duplicate either as a typed ``DuplicateEdgeError`` or as a raw
#: database ``UNIQUE`` violation depending on the code path taken; both are
#: mapped to this one wording so the two are indistinguishable to a client and
#: neither exposes the store's internal phrasing or schema names.
DUPLICATE_EDGE_MESSAGE = (
    "An edge of that type already links those two thoughts. Edges "
    "are unique per (source, target, type), so this link already "
    "exists — no change was made."
)

#: Client-facing message for a caller-supplied ``edge_id`` that collides with
#: an existing edge's primary key. Distinct from :data:`DUPLICATE_EDGE_MESSAGE`
#: (the (source, target, type) uniqueness constraint): the two are different
#: constraints on the same table, and the repair that fixes one — changing the
#: edge type — does nothing for the other, since the collision here is on the
#: id itself.
EDGE_ID_COLLISION_MESSAGE = (
    "An edge with that edge_id already exists — edge_id values must be "
    "unique across the whole store. Omit edge_id to have one generated "
    "automatically, or supply a different one. Changing the edge type will "
    "not resolve this: the collision is on the id itself, not on the "
    "(source, target, type) relationship."
)

#: Extracts the two byte counts from engrava's raw oversized-metadata message
#: (``"metadata serialized size 70009 bytes exceeds maximum 65536 bytes"``),
#: so the curated edge-metadata message in :func:`_tool_errors` can state the
#: actual numbers without forwarding the store's own phrasing (which also
#: suggests moving the payload into ``content``, advice that does not apply to
#: edges).
_METADATA_SIZE_PATTERN = re.compile(
    r"metadata serialized size (\d+) bytes exceeds maximum (\d+) bytes"
)

#: Environment variable that, when truthy, suppresses registration of the
#: write tools so the server exposes a read-only surface.
READ_ONLY_ENV_VAR = "ENGRAVA_MCP_READ_ONLY"

#: Values that enable read-only mode (compared case-insensitively after
#: stripping surrounding whitespace).  Any other value — including unset
#: or empty — leaves the full read and write surface enabled.
READ_ONLY_TRUTHY_VALUES = frozenset({"1", "true", "yes"})

_READ_ONLY = ToolAnnotations(read_only_hint=True)

#: Annotation for a non-idempotent, non-destructive write.  Covers creating
#: a new thought node (repeating the call creates another node), creating a
#: typed edge (an edge is unique per source/target/type, so repeating an
#: identical link is rejected rather than converging), and updating a
#: thought (every call refreshes ``updated_at`` and, on a journal-enabled
#: store, appends a journal entry, so a retried identical call has
#: observable effects even though the visible fields converge on the same
#: values) — none of these is safe for a client to blindly retry.
_WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False)

#: Annotation for a destructive but idempotent write (deleting a thought or
#: edge).  It is marked idempotent because deleting an already-absent
#: identifier is a no-op that returns ``deleted=False`` and leaves the same
#: end state — the record is gone either way — so a client may safely retry a
#: delete that appeared to fail.
_WRITE_DESTRUCTIVE = ToolAnnotations(
    read_only_hint=False, destructive_hint=True, idempotent_hint=True
)


class StoreNotReadyError(RuntimeError):
    """Raised when a tool is invoked before a store has been provided.

    This indicates a lifecycle bug — tools should only run while the
    server lifespan is active.
    """


class UnsupportedQueryError(ValueError):
    """Raised when ``query_memory`` receives a non-``FIND`` command.

    The MCP read surface deliberately accepts only the MindQL ``FIND``
    command.  Raw-SQL passthrough (``SELECT``), aggregate ``COUNT``, and
    extension commands are rejected so the tool cannot be used to run
    arbitrary statements against the database.

    Args:
        command: The rejected command verb.

    """

    def __init__(self, command: str) -> None:
        self.command = command
        super().__init__(
            f"query_memory accepts only FIND queries; received {command!r}. "
            f"Use the FIND command, for example: {FIND_QUERY_EXAMPLE}"
        )


class EmbeddingQueryNotSupportedError(ValueError):
    """Raised when a ``FIND`` targets the ``embeddings`` table.

    The embedding table's ``vector_blob`` column holds the raw stored
    embedding as bytes.  ``SELECT *`` (what the executor runs for every
    ``FIND``) returns it verbatim, and a tool result crosses the wire as
    JSON — bytes have no JSON representation, so a successful, in-range query
    against this table cannot be returned at all; without this guard it fails
    only after ``query_memory_impl`` has already returned, deep inside
    MCPServer's own response serialisation, as a raw, unmapped ``TypeError``
    that :func:`_tool_errors` never sees (its ``try``/``except`` has already
    exited by the time serialisation runs). Refusing the table outright, at
    the query boundary, converts that unreachable-message crash into an
    ordinary, actionable refusal — narrower than the alternative of quietly
    dropping or encoding ``vector_blob``, and the tool description says so.

    Args:
        table: The rejected table's canonical name (``"embedding"``).

    """

    def __init__(self, table: str) -> None:
        self.table = table
        super().__init__(
            f"query_memory does not support FIND {table}s: a stored "
            "embedding's vector data cannot be represented in a JSON tool "
            "result. Query thoughts, edges, or actions instead."
        )


class UnexecutableQueryError(ValueError):
    """Raised when a query that parsed as ``FIND`` fails during execution.

    By the time ``store.execute_mindql`` runs, the query has already passed
    the ``FIND``-only guard — so whatever it rejects the query for (an
    unknown column, a disallowed comparison) is necessarily about the
    query's own content, never about the command set. That makes the raw
    diagnosis safe to surface verbatim, unlike a failure raised by ``parse()``
    itself: the discriminator is *where* the failure was raised, not what it
    says, because the MindQL executor's exceptions carry no structure to
    pattern-match on beyond that.

    Args:
        message: The executor's own diagnosis, forwarded unchanged.

    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class MalformedFindError(ValueError):
    """Raised when a query classified as ``FIND`` fails to parse.

    ``query_memory_impl`` determines the command verb itself, before ever
    calling ``parse()`` (see :func:`_query_declares_find`): when the first
    token, after stripping an optional ``EXPLAIN`` prefix, is ``FIND``
    case-insensitively, any ``MindQLParseError`` that ``parse()`` then raises
    is necessarily about that FIND's own content (an unknown table, a bad
    condition) — the unrecognised-verb message that names the full command
    set only fires for a verb ``parse()`` itself does not recognise as
    ``FIND``, ``COUNT``, ``SELECT``, or a registered extension, which by
    construction cannot be this case. The discriminator is the *input*,
    classified before parsing, not the exception's *text* — this module
    controls the former and never the latter.

    Args:
        message: The parser's own diagnosis, forwarded unchanged.

    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class OutOfRangeBoundError(ValueError):
    """Raised when a wire-supplied numeric bound falls outside its domain.

    The MCP protocol layer validates an argument's *type*, not its *domain*:
    ``-1`` is a valid ``int``, and SQLite reads ``LIMIT -1`` as "no limit", so
    an unvalidated negative bound silently defeats the scan cap it was meant to
    impose.  An excessive upper value is equally unbounded in effect.  Each
    implementation therefore re-checks its own bounds rather than trusting the
    protocol layer's coercion, which also covers direct callers.

    Args:
        name: The offending argument's name, as the caller supplied it.
        value: The rejected value.
        minimum: Smallest accepted value.
        maximum: Largest accepted value, or ``None`` when unbounded above.

    """

    def __init__(self, name: str, value: int, minimum: int, maximum: int | None) -> None:
        self.name = name
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        allowed = f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        super().__init__(f"{name} must be {allowed}; received {value}.")


def _check_bound(name: str, value: int, *, minimum: int, maximum: int | None = None) -> None:
    """Validate a wire-supplied numeric bound against its domain.

    Args:
        name: The argument's name, used verbatim in the error message.
        value: The supplied value.
        minimum: Smallest accepted value.
        maximum: Largest accepted value, or ``None`` when unbounded above.

    Raises:
        OutOfRangeBoundError: If ``value`` is outside ``[minimum, maximum]``.

    """
    if value < minimum or (maximum is not None and value > maximum):
        raise OutOfRangeBoundError(name, value, minimum, maximum)


class _PromptBoundError(MCPError, ToolError):
    """A wire-supplied prompt argument bound violation, reported cleanly either way.

    Tools reach an out-of-range bound through :func:`_check_bound` and
    :func:`_tool_errors`, which turns it into a curated :class:`ToolError` — the
    channel ``MCPServer.call_tool`` reports to the client without mangling.
    Prompts have no equivalent: ``Prompt.render`` (the ``mcp`` 2.x prompt-rendering
    step) re-raises only :class:`~mcp.shared.exceptions.MCPError` unchanged, and
    wraps *any* other exception it sees — including a plain
    :class:`OutOfRangeBoundError`, and including :class:`ToolError` itself — into
    a generic ``ValueError`` that reaches the client only as an opaque "Internal
    server error". A prompt handler therefore has to raise :class:`MCPError`
    directly to keep its message.

    Subclassing both lets a single exception serve two call sites without
    duplicating the bound-violation message: raised through the real MCP
    connection, it is the :class:`MCPError` ``Prompt.render`` passes through
    untouched; raised (or caught) by a test driving a prompt's unwrapped
    function body directly — the path a caller reaching it without the
    protocol layer would take — it is still the :class:`ToolError` that path
    expects, exactly as :func:`_tool_errors` would raise for the same
    violation inside a tool.

    Args:
        cause: The bound violation this wraps; its message is used verbatim.

    """

    def __init__(self, cause: OutOfRangeBoundError) -> None:
        MCPError.__init__(self, code=INVALID_PARAMS, message=str(cause))


def _check_prompt_bound(name: str, value: int, *, minimum: int, maximum: int | None = None) -> None:
    """Validate a wire-supplied prompt argument bound, raising a clean protocol error.

    The prompt-side counterpart to :func:`_check_bound`: same domain check, but
    raising :class:`_PromptBoundError` instead of a bare :class:`OutOfRangeBoundError`
    so the message survives crossing a prompt's ``Prompt.render`` boundary (see
    :class:`_PromptBoundError` for why tools and prompts need different exception
    types here).

    Args:
        name: The argument's name, used verbatim in the error message.
        value: The supplied value.
        minimum: Smallest accepted value.
        maximum: Largest accepted value, or ``None`` when unbounded above.

    Raises:
        _PromptBoundError: If ``value`` is outside ``[minimum, maximum]``.

    """
    try:
        _check_bound(name, value, minimum=minimum, maximum=maximum)
    except OutOfRangeBoundError as exc:
        raise _PromptBoundError(exc) from exc


#: ``sqlite3.IntegrityError.sqlite_errorcode`` value for a PRIMARY KEY
#: violation (``SQLITE_CONSTRAINT_PRIMARYKEY``).  Used in :func:`_tool_errors`
#: to distinguish an ``edge_id`` collision from the edge table's other UNIQUE
#: constraint without depending on message text; see that branch for why the
#: text is also checked as a fallback.
_SQLITE_CONSTRAINT_PRIMARYKEY = 1555

#: SQLite result codes that report database contention: ``SQLITE_BUSY`` (5)
#: and its extended forms ``SQLITE_BUSY_RECOVERY`` (261),
#: ``SQLITE_BUSY_SNAPSHOT`` (517) and ``SQLITE_BUSY_TIMEOUT`` (773). The same
#: set engrava's own (private) classifier uses; defined here rather than
#: imported, since this server does not depend on engrava's private symbols.
#: ``SQLITE_LOCKED`` (6) is a different family of locking conflict and is not
#: included.
_BUSY_ERRORCODES: frozenset[int] = frozenset(
    {
        getattr(sqlite3, "SQLITE_BUSY", 5),
        getattr(sqlite3, "SQLITE_BUSY_RECOVERY", 261),
        getattr(sqlite3, "SQLITE_BUSY_SNAPSHOT", 517),
        getattr(sqlite3, "SQLITE_BUSY_TIMEOUT", 773),
    }
)


def _is_busy_error(exc: sqlite3.OperationalError) -> bool:
    """Return whether *exc* means SQLite reported database contention.

    Classifies structurally, via :attr:`sqlite3.Error.sqlite_errorcode`,
    never by matching ``str(exc)`` -- driver and locale text for "database is
    locked" is not a stable contract. The other side of a busy error can be
    a writer, or -- in rollback-journal mode, which ``engrava.yaml`` can
    still select -- a reader blocking this connection's own commit; nothing
    here or in any caller may claim which, or that a lock is held at the
    moment the message is composed.

    Args:
        exc: The raised SQLite operational error.

    Returns:
        ``True`` when ``exc.sqlite_errorcode`` is one of :data:`_BUSY_ERRORCODES`.
        ``False`` for ``SQLITE_LOCKED`` (6), any other code, and for an
        exception with no (or a non-``int``) ``sqlite_errorcode`` -- e.g. one
        built by hand rather than raised by the driver.

    """
    errorcode = getattr(exc, "sqlite_errorcode", None)
    if not isinstance(errorcode, int):
        return False
    return errorcode in _BUSY_ERRORCODES


#: Client-facing message for a busy ``sqlite3.OperationalError`` from
#: ``store.create_edge`` (``link_thoughts``).  ``create_edge`` is one write
#: unit begun ``BEGIN IMMEDIATE``, with its journal append inside it, then one
#: commit through the store's own recovery path, and nothing written after it
#: -- so a busy error here, whether it comes from acquiring the lock or from
#: the commit itself, always means nothing was written, and retrying the
#: whole call is safe.  Unlike ``store_thought`` (see
#: :func:`_residual_write_guard`), nothing in this call can still write after
#: that commit, so the message never varies.
_LINK_THOUGHTS_BUSY_MESSAGE = (
    "SQLite reported database contention. Nothing was written, and retrying is safe."
)

#: Client-facing message for a busy ``sqlite3.OperationalError`` from
#: ``store.delete_thought`` (``delete_thought``) or ``store.delete_edge``
#: (``delete_edge``).  Each delete, when it opens its own transaction,
#: takes the write lock (``BEGIN IMMEDIATE``) before its database reads and
#: writes. A failure in its write unit (the delete, ``delete_thought``'s
#: vector purge, the journal append) or in its commit is rolled back, or
#: the connection quarantined, and nothing is written after the commit.
#: And a plain ``ROLLBACK`` cannot report ``SQLITE_BUSY``. So a busy error
#: from either call means nothing was deleted, and retrying is safe.
_DELETE_BUSY_MESSAGE = (
    "SQLite reported database contention. Nothing was deleted, and retrying is safe."
)


@asynccontextmanager
async def _busy_guard(message: str) -> AsyncIterator[None]:
    """Map a busy ``sqlite3.OperationalError`` from one store call to a curated ``ToolError``.

    Shared by :func:`link_thoughts_impl` (around ``store.create_edge``),
    :func:`delete_thought_impl` (around ``store.delete_thought``) and
    :func:`delete_edge_impl` (around ``store.delete_edge``) -- one guard,
    parameterised by *message*, rather than three copies. The ``with`` wraps
    that single store call only, never the surrounding tool body, so a
    failure in this module's own argument handling is never mistaken for a
    store failure.

    Each of the three guarded calls, when it opens its own transaction,
    takes the write lock (``BEGIN IMMEDIATE``) before its database reads
    and writes. A failure in its write unit (the write and its journal
    append -- and, for ``delete_thought``, the vector purge) or in its
    commit is rolled back, or the connection quarantined if that rollback
    itself fails, and nothing is written after the commit. And a plain
    ``ROLLBACK`` cannot report ``SQLITE_BUSY``. So a busy error from any
    of the three calls means that call's own write did not happen, and
    retrying is safe.

    Args:
        message: The client-facing text to raise when the guarded call fails
            with a busy error -- :data:`_LINK_THOUGHTS_BUSY_MESSAGE` for
            ``create_edge``, :data:`_DELETE_BUSY_MESSAGE` for the two delete
            calls.

    Yields:
        ``None``; the caller runs the guarded store call inside the ``with``.

    Raises:
        ToolError: When the guarded call raises a busy
            ``sqlite3.OperationalError`` (:func:`_is_busy_error`), carrying
            *message*, chained ``from`` the original exception. Any other
            exception raised inside the ``with`` -- including a non-busy
            ``sqlite3.OperationalError`` -- propagates unchanged.

    """
    try:
        yield
    except sqlite3.OperationalError as exc:
        if not _is_busy_error(exc):
            raise
        raise ToolError(message) from exc


class _ResidualWriteError(Exception):
    """Wraps a write-path exception :func:`_tool_errors`'s own chain does not curate.

    :func:`_residual_write_guard` wraps both ``store.create_thought`` (in
    :func:`store_thought_impl`) and ``store.update_thought`` (in
    :func:`update_thought_impl`) -- one guard, phase-agnostic: nothing here
    distinguishes which of a call's several post-commit steps raised, or
    whether the exception fired before the write's own commit or after it.
    "Residual" is defined by what :func:`_tool_errors`'s own chain does with
    the exception, not by a hand-maintained list of types: the chain re-raised
    it unchanged (its type is unnamed, or it is a named type no branch
    curates -- a bare ``ValueError`` matching neither recognised prefix, an
    unrecognised ``sqlite3.IntegrityError``, an unbranched
    :class:`~engrava.EngravaError` subclass), or classifying it raised
    something new (an extension's ``__str__`` that itself raises, say), or the
    guarded call raised a :class:`ToolError` itself, which the chain has no
    branch for and so passes through unchanged. See
    :func:`_residual_write_guard` for how that is decided. A busy
    ``sqlite3.OperationalError`` from ``store.create_thought`` is a known
    category, not a residual one: :func:`_residual_write_guard` recognises
    and answers it before this classification ever runs, so it never becomes
    a :class:`_ResidualWriteError`.

    Because a residual exception may have fired either before or after the
    write's own commit, and this guard cannot tell which, the message this
    maps to (see :func:`_tool_errors`) never states whether the write took
    effect, and never names a step -- neither is something this server can
    know from here.

    The original exception is kept only in exception chaining and is never
    copied into the client-facing message this maps to: neither its type
    name nor its text reaches the client. An unnamed exception may belong to
    an operator-supplied extension and carry detail that extension composed,
    not a diagnostic string engrava wrote.

    Args:
        tool: Which tool's write raised -- ``"store_thought"`` or
            ``"update_thought"`` -- so :func:`_tool_errors` states the right
            fact (stored, or applied) as unconfirmed.
        thought_id: The identifier the write attempted: the caller-supplied
            or server-generated id for ``store_thought``, or the caller's own
            id for ``update_thought``. Not an internal value, so it is safe
            to hand back as the id to check.
        deduplicate: The ``store_thought`` call's own ``deduplicate`` value.
            Unused for ``update_thought``, which has no such argument: the
            asymmetry in what reading ``thought_id`` back can prove applies
            only to ``store_thought``.

    """

    def __init__(
        self,
        *,
        tool: Literal["store_thought", "update_thought"],
        thought_id: str,
        deduplicate: bool = False,
    ) -> None:
        self.tool = tool
        self.thought_id = thought_id
        self.deduplicate = deduplicate
        super().__init__(f"{tool} raised an exception type engrava-mcp does not classify")


async def _store_thought_busy_message(
    store: SqliteEngravaCore, thought_id: str, *, deduplicate: bool
) -> str:
    """Build the client-facing message for a busy ``create_thought``.

    Unlike ``create_edge`` (:data:`_LINK_THOUGHTS_BUSY_MESSAGE`),
    ``create_thought``'s commit is followed by several steps that can each
    still raise (auto-embed, hygiene cleanup, the configured hooks class,
    derived-record dispatch), so a busy error here establishes nothing about
    whether the thought was stored: it can come from before that commit or
    from any of those later steps. This reads the attempted id back once,
    with :meth:`~engrava.SqliteEngravaCore.get_thought`, and reports only
    what that one read found -- never what this call did, and never a
    durability or retry-safety claim the read cannot support: a present
    thought could equally be a pre-existing one this call never touched, and
    a read-then-report is a snapshot, not proof of what a retry would do.

    Args:
        store: The store to read the attempted id back from.
        thought_id: The id ``store.create_thought`` attempted -- the
            caller-supplied or server-generated id, not an internal value.
        deduplicate: The ``store_thought`` call's own ``deduplicate``
            value, used only when the read-back itself raises: it selects
            between :func:`_ResidualWriteError`'s two ``store_thought``
            pieces of advice, since a duplicate-content match changes what
            reading ``thought_id`` back could have proven.

    Returns:
        The message text, unprefixed by MCPServer's own
        ``"Error executing tool <name>: "`` wrapper.

    """
    try:
        found = await store.get_thought(thought_id)
    except Exception:  # noqa: BLE001 -- the read-back itself can fail for any reason
        if deduplicate:
            return (
                f"store_thought for {thought_id!r} (deduplicate=True): SQLite "
                "reported database contention; whether the thought was stored "
                "could not be confirmed. With deduplicate=True the call may have "
                "matched an existing thought with identical content instead of "
                f"storing a new one, so reading {thought_id!r} back with "
                "get_thought cannot settle what happened: finding it shows a "
                "thought with that id exists, not that this call stored it."
            )
        return (
            f"store_thought for {thought_id!r}: SQLite reported database "
            "contention; whether the thought was stored could not be "
            "confirmed. Read it back with get_thought before retrying -- a "
            "blind retry can store a second copy."
        )
    if found is None:
        return (
            f"store_thought for {thought_id!r}: SQLite reported database "
            "contention; whether the thought was stored could not be "
            "confirmed. A read just now found no thought with this id."
        )
    return (
        f"store_thought for {thought_id!r}: SQLite reported database "
        "contention; whether the thought was stored could not be confirmed. "
        "A read just now found a thought with this id."
    )


@asynccontextmanager
async def _residual_write_guard(
    *,
    store: SqliteEngravaCore,
    tool: Literal["store_thought", "update_thought"],
    thought_id: str,
    deduplicate: bool = False,
) -> AsyncIterator[None]:
    """Classify a write failure once, and wrap it only if nothing curates it.

    Wraps only the ``store.create_thought`` / ``store.update_thought`` call
    itself -- not :func:`_tool_errors`'s own ``try`` block, which also covers
    this module's own argument handling and would turn a genuine bug in this
    server's own code into a misleading write-failure message.

    A busy ``sqlite3.OperationalError`` (:func:`_is_busy_error`) from
    ``store.create_thought`` is handled first, **before** anything reaches
    :func:`_tool_errors`'s own ordered chain: it is a known category, not an
    unrecognised one, so there is nothing to classify -- see
    :func:`_store_thought_busy_message` for why the answer still cannot say
    whether the thought was stored. A busy error from ``store.update_thought``
    is not special-cased here: engrava raises its own typed
    ``WriteContentionError`` for that path's contention (kept by
    :func:`_tool_errors`'s existing branch), and a raw busy
    ``sqlite3.OperationalError`` there falls through to the same
    classify-once handling as any other residual exception, unchanged from
    before.

    Anything else is classified with no list of its own. It runs
    :func:`_tool_errors`'s own ordered chain exactly once, by re-raising the
    caught exception (``exc``) inside a nested ``async with
    _tool_errors():``. Every current mapping branch there raises
    ``ToolError(msg) from exc``, so a caught :class:`ToolError` whose
    ``__cause__ is exc`` is a curated result -- re-raised here unchanged, and
    passed through untouched by the real, outer :func:`_tool_errors` this
    function's own caller runs inside, since no branch there matches a
    :class:`ToolError` either (it derives from ``MCPServerError`` ->
    ``Exception``, and the chain has no broad catch). Anything else is
    residual: the same exception re-raised unchanged because no branch names
    its type, a new exception raised while a branch was composing its
    message (a throwing ``__str__``, say), or a :class:`ToolError` the
    guarded call raised itself -- indistinguishable from a curated one by
    type alone, but its ``__cause__`` cannot be ``exc``, since no branch
    catches ``ToolError``.

    Args:
        store: The store the guarded call writes through. Used only to read
            a ``store_thought`` busy error's attempted id back -- unused for
            ``update_thought`` and for every non-busy exception.
        tool: Forwarded to :class:`_ResidualWriteError`.
        thought_id: Forwarded to :class:`_ResidualWriteError`, and -- for
            ``store_thought`` -- the id a busy error's own read-back checks.
        deduplicate: Forwarded to :class:`_ResidualWriteError`, and -- for
            ``store_thought`` -- forwarded again to
            :func:`_store_thought_busy_message`.

    Yields:
        ``None``; the caller runs the guarded store call inside the ``with``.

    Raises:
        ToolError: When ``store.create_thought`` raises a busy
            ``sqlite3.OperationalError`` (``tool="store_thought"`` only),
            carrying :func:`_store_thought_busy_message`'s text.
        _ResidualWriteError: When the guarded call raises anything else
            :func:`_tool_errors`'s own chain does not curate.

    """
    try:
        yield
    except Exception as exc:
        # raises, including a type this module has never seen; that is what
        # _ResidualWriteError exists for.
        if (
            tool == "store_thought"
            and isinstance(exc, sqlite3.OperationalError)
            and _is_busy_error(exc)
        ):
            message = await _store_thought_busy_message(store, thought_id, deduplicate=deduplicate)
            raise ToolError(message) from exc
        try:
            async with _tool_errors():
                raise  # re-raises exc, which is still the exception being handled here
        except ToolError as out:
            if out.__cause__ is exc:
                raise
            raise _ResidualWriteError(
                tool=tool, thought_id=thought_id, deduplicate=deduplicate
            ) from exc
        except Exception:  # noqa: BLE001 -- same reason as the outer clause above
            raise _ResidualWriteError(
                tool=tool, thought_id=thought_id, deduplicate=deduplicate
            ) from exc


# C901 (mccabe complexity), PLR0912 (branch count), and PLR0915 (statement
# count) are all waived here, and for the same reason: this function is a flat
# translation table, not branching logic. Its "branches" are one ``except``
# clause per recognised typed failure, each mapping that failure to a curated,
# client-facing message — typically a message literal and a ``raise``, so the
# statement count grows in step with the branch count. Both therefore grow by
# one (branches) or a handful (statements) every time a new failure mode is
# given a curated message — which is the function working as intended, not
# accruing complexity. A test in ``test_errors.py`` walks engrava's own public
# exception types against this table and fails when one is neither mapped here
# nor explicitly excused, so all three counts are expected to keep growing as
# that surface does; widen this suppression's rationale again rather than
# deleting it when they do.
#
# Two branches carry a *second* level of branching, for the same underlying
# reason: the failure they translate is not a distinct engrava type, so there
# is nothing else to ``except`` on.
#   * ``sqlite3.IntegrityError`` covers two different constraints on the same
#     table (the caller-visible duplicate-edge UNIQUE and the ``edge_id``
#     PRIMARY KEY) that share one exception class; telling them apart is the
#     branch's job, not a sign it should be two branches.
#   * The plain ``ValueError`` branch covers two of engrava's own untyped
#     failures (a duplicate ``thought_id``, oversized edge metadata) that the
#     store raises as bare ``ValueError`` rather than a dedicated type;
#     dispatching on the message prefix is the only discriminator available,
#     and anything that matches neither prefix is re-raised unchanged so a
#     future, unrelated ``ValueError`` is never silently swallowed here.
#
# Splitting the table into helpers or a dict-based lookup was considered and
# rejected: it would scatter the error contract that the tests pin, and it does
# not actually fit the shape of the code. Several branches read attributes off
# the specific exception they caught (``thought_id``, ``referenced_id``,
# ``current_state`` / ``target_state``) rather than formatting a fixed string,
# and the ``sqlite3.IntegrityError`` branch deliberately re-raises anything that
# is not a recognised constraint violation so it is never silently masked. A
# lookup table would express neither, fragmenting one mechanism into two.
@asynccontextmanager
async def _tool_errors() -> AsyncIterator[None]:  # noqa: C901, PLR0912, PLR0915
    """Translate known typed failures into clean, actionable MCP errors.

    Wraps the body of a tool handler so that the typed exceptions raised by
    the store, the MindQL parser, and this module's own consumer-policy
    guard surface to the client as a :class:`ToolError` carrying a curated,
    agent-facing message instead of an internal exception.  MCPServer reports
    a :class:`ToolError` to the client with ``isError`` set and the message
    as text, so the client receives an actionable hint rather than a raw
    traceback or an internal class name.

    This is *presentation only*: it adds no new capability and relaxes no
    guard.  Each branch re-raises an existing failure with a better message;
    the ``UnsupportedQueryError`` branch in particular preserves the
    ``FIND``-only contract verbatim and never suggests that raw SQL is
    runnable over the wire.  Conditions this module does not recognise are
    left to propagate unchanged.

    The messages name only the documented configuration *environment
    variables* (never a filesystem path), carry no stack frames, and expose
    no internal symbol names, so a misuse reply leaks nothing about the
    deployment. In particular, a database constraint violation (a duplicate
    ``link_thoughts`` edge or a colliding caller-supplied ``edge_id``), a
    domain-model validation error, and an illegal lifecycle transition are
    mapped to curated messages here so the raw SQLite table/column names,
    Pydantic's internal model details, and the internal status-type name never
    reach the client; an unrecognised integrity error is re-raised unchanged
    rather than described. A concurrent write that lost the optimistic-
    concurrency guard (``StaleDataError``), a duplicate ``thought_id``, and
    oversized edge metadata are likewise mapped rather than left as an
    internal store message, as are a busy store (another writer, or a
    long-running write ahead of this one) and a store that has become
    unusable and needs a restart. A numeric argument that overflows SQLite's own
    bound-integer range gets a generic but honest message, since the raw
    ``OverflowError`` carries no argument name to attribute it to. A write-path
    embedding failure under ``embeddings.auto_embed`` (a model, vector-size or
    document-prefix mismatch against what this store's embeddings were built
    with, or a provider failure under ``embeddings.require_embedding``) is
    likewise mapped, in both cases stating that the thought text was
    nonetheless stored and only its embedding is missing. A ``search_memory``
    whose configured embedding provider no longer matches what the store
    declares or holds -- a changed vector size, or a changed query prefix on
    an asymmetric model -- is likewise mapped rather than left as an internal
    store message; both are read-path, configuration-only conditions, so the
    messages report no result and no change rather than suggesting a retry.
    A ``store_thought`` whose configured derived-records producer rejects its
    own output (``DerivedRecordError``, over-cap or an identity collision) is
    likewise mapped, stating that the thought itself was nonetheless stored
    and only its derived record is missing. A ``store_thought`` or
    ``update_thought`` write that ends with an exception this function's own
    chain does not otherwise curate -- reachable only wrapped as
    :class:`_ResidualWriteError`; see :func:`_residual_write_guard` -- gets a
    message that states neither which step raised nor whether the write took
    effect, because a residual exception may have fired either before or
    after the write's own commit, and this function cannot tell which. A
    busy ``sqlite3.OperationalError`` from ``store.create_thought``,
    ``store.create_edge``, ``store.delete_thought`` or ``store.delete_edge``
    never reaches this chain at all: each is answered by a narrow guard
    around that one call (:func:`_residual_write_guard` for
    ``create_thought``; :func:`_busy_guard`, shared, for the other three)
    before classification would begin, because ``create_thought``'s commit
    is followed by steps that can still write, while the other three's
    commits are not, and a chain-level branch here could only give all of
    them the same answer.

    Yields:
        ``None``; the caller runs the guarded tool body inside the ``with``.

    Raises:
        ToolError: With an actionable message when a recognised typed
            failure occurs while the body runs.

    """
    try:
        yield
    except StoreNotReadyError as exc:
        msg = (
            "The engrava memory store is not available yet. Start the server "
            "with a store configured: set ENGRAVA_DB_PATH to a database file, "
            "or point ENGRAVA_MCP_CONFIG at an engrava.yaml that names one."
        )
        raise ToolError(msg) from exc
    except UnsupportedQueryError as exc:
        # The exception text already states the FIND-only contract and shows
        # a valid FIND example; echoing it keeps the guard's wording intact
        # and never invites raw SQL.
        raise ToolError(str(exc)) from exc
    except EmbeddingQueryNotSupportedError as exc:
        # A FIND targeting the embeddings table. The exception text already
        # states why (raw vector bytes have no JSON representation) and what
        # to query instead; echoing it keeps that wording in one place.
        raise ToolError(str(exc)) from exc
    except EmbeddingModelMismatchError as exc:
        # store_thought / update_thought with embeddings.auto_embed on: fires
        # from the first embedding this store instance ever writes (never on
        # startup -- nothing on this server's own startup path checks the
        # model). The thought row itself is already committed by the time
        # this can fire (create_thought / update_thought both commit before
        # auto-embed ever runs -- see _on_auto_embed_failure's own
        # docstring), so only the embedding write is what failed. The check
        # compares the embedding metadata this store already has on record
        # (model name, vector size, and document-prefix fingerprint) against
        # what the configured provider produces now, and raises on any
        # difference -- the raw values are the store's own internal
        # bookkeeping, not something the caller supplied, and the check does
        # not distinguish what produced the difference, so neither is echoed
        # or guessed at here.
        msg = (
            "The thought was stored, but its embedding could not be written: "
            "this store's existing embeddings were built with a different "
            "model, vector size, or document prefix than the embedding "
            "provider now configured for this server produces. Semantic "
            "search (search_memory) will not find this thought, and every "
            "future write's auto-embed will fail the same way until the "
            "mismatch is fixed. Retrying will not help -- the embedding "
            "configuration needs to be corrected (or restored), or a "
            "matching database used, by whoever operates this server."
        )
        raise ToolError(msg) from exc
    except EmbeddingGenerationError as exc:
        # store_thought / update_thought, and only when embeddings.auto_embed
        # AND embeddings.require_embedding are both on. This type's own
        # message embeds str() of the provider's original exception verbatim
        # (see its __init__), which may carry provider-internal detail, so it
        # is never echoed here. The thought row is already committed by the
        # time this can fire (see this type's own docstring: create_thought /
        # update_thought commit before auto-embed ever runs), so
        # require_embedding's fail-fast does not undo the write -- only the
        # embedding is missing.
        msg = (
            "The thought was stored, but its embedding could not be "
            "generated (this server requires embeddings -- "
            "embeddings.require_embedding is enabled). Semantic search "
            "(search_memory) will not find this thought until it is "
            "re-embedded -- update its essence or content again once the "
            "provider issue is resolved, to trigger a fresh embed attempt, "
            "or report the failure to whoever operates this server."
        )
        raise ToolError(msg) from exc
    except VectorDimensionMismatchError as exc:
        # search_hybrid gathers its lexical (FTS5) arm before its vector arm,
        # so whenever FTS5 is available and query_text is non-empty (the
        # common case) a real keyword pass over the corpus has already run
        # by the time this fires from the vector arm -- its results are just
        # discarded, never fused or returned, because the call raises before
        # fusion. The message must not claim nothing was searched; only that
        # no result reached the caller and the store itself was not written
        # to. The raw dimension integers are the store's and the provider's
        # own numbers, not anything a caller can act on, so they are not
        # echoed; retrying cannot help, since the mismatch depends only on
        # server configuration, never on tool arguments.
        msg = (
            "search_memory could not run: the configured embedding provider "
            "produces vectors of a different size than the ones already "
            "stored in this memory. No result was returned, and nothing was "
            "changed. Retrying will not help -- the embedding configuration "
            "needs to be corrected (or restored) by whoever operates this "
            "server."
        )
        raise ToolError(msg) from exc
    except EmbeddingQueryPrefixMismatchError as exc:
        # search_memory embeds the query text with the configured provider's
        # active query prefix before searching; this fires only for an
        # asymmetric embedding model whose active query prefix no longer
        # pairs with the one this store's vectors were embedded to pair
        # with -- typically because the engrava.yaml was pointed at a
        # different prefix configuration after the store already had
        # vectors. This is a read path: nothing was written. Neither prefix
        # is a tool argument, so retrying the same search cannot help; only
        # correcting the server's embedding configuration can.
        msg = (
            "search_memory could not run: the configured embedding "
            "provider's active query prefix no longer matches the one this "
            "memory's stored vectors were embedded to pair with. Nothing "
            "was searched or changed. Retrying will not help -- the "
            "embedding configuration needs to be corrected (or restored) by "
            "whoever operates this server."
        )
        raise ToolError(msg) from exc
    except OutOfRangeBoundError as exc:
        # A numeric bound outside its accepted domain. The message names only
        # the caller's own argument, its value, and the accepted range — no
        # internal symbols — so echoing it is safe and directly actionable.
        raise ToolError(str(exc)) from exc
    except OverflowError as exc:
        # A wire-supplied integer that reached a SQLite bind call outside
        # SQLite's own signed-64-bit range. This should be unreachable in
        # practice now that every SQLite-bound integer parameter is
        # constrained at the wire (see PageOffset / CycleFilterBound) — this
        # branch is defense in depth for any such parameter this table has not
        # caught up with yet. The raw OverflowError carries no argument name
        # or value, only "Python int too large to convert to SQLite INTEGER",
        # so the message here is necessarily generic rather than naming a
        # specific cause it cannot know.
        msg = (
            "A numeric argument is too large for the store to accept. The "
            f"accepted range is {SQLITE_MIN_BOUND_INT} to "
            f"{SQLITE_MAX_BOUND_INT}. Use a smaller value."
        )
        raise ToolError(msg) from exc
    except MindQLParseError as exc:
        # Do NOT echo the parser's raw message: for an unrecognised verb the
        # parser names the full MindQL command set ("Expected FIND, COUNT,
        # SELECT, or extension command"), which would leak commands the MCP
        # surface deliberately does not expose. query_memory accepts only
        # FIND, so the client-facing message states that and shows a valid
        # FIND example — never the parser's command list.
        msg = (
            "query_memory accepts only FIND queries and the query could not "
            f"be parsed as one. Use the FIND command, for example: {FIND_QUERY_EXAMPLE}"
        )
        raise ToolError(msg) from exc
    except MalformedFindError as exc:
        # See MalformedFindError's docstring: query_memory_impl classifies
        # the command verb itself, before calling parse(), so a
        # MindQLParseError wrapped in this type is guaranteed to be about a
        # FIND's own content and never the unrecognised-verb message.
        raise ToolError(str(exc)) from exc
    except UnexecutableQueryError as exc:
        # Raised by query_memory_impl only after the query already parsed as
        # FIND, so by construction it cannot name another MindQL command.
        # Its message names only the query's own content (a column, a table,
        # a value) and is safe to echo verbatim — unlike the MindQLParseError
        # branch above, which stays generic because it also covers the
        # unrecognised-verb case.
        raise ToolError(str(exc)) from exc
    except ThoughtNotFoundError as exc:
        msg = (
            f"No thought exists with id {exc.thought_id!r}. Check the "
            "identifier, or use search_memory or list_memory to find it."
        )
        raise ToolError(msg) from exc
    except StaleDataError as exc:
        # update_thought guards its write with the thought's revision at
        # the moment it was read; a zero-row match means another writer
        # changed (or deleted) the row in between, and nothing of this update
        # was applied. entity_type ("ThoughtRecord") is an internal class
        # name and is deliberately not echoed — on this surface StaleDataError
        # is reachable only through update_thought, so "thought" is unambiguous
        # without it.
        msg = (
            f"Could not update thought {exc.entity_id!r}: it was changed (or "
            "deleted) by another write since it was last read, so nothing was "
            "applied. Fetch it again with get_thought and retry the update "
            "with the current values."
        )
        raise ToolError(msg) from exc
    except InvalidTransitionError as exc:
        # An illegal lifecycle change on update_thought (the wire status is
        # coerced to the enum, so the state-machine guard fires). The raw
        # message names the internal status type; surface the move in plain
        # user terms instead. The state values are the public lifecycle names
        # (CREATED/ACTIVE/DONE/ARCHIVED), not internal symbols.
        msg = (
            f"Cannot change lifecycle status from {exc.current_state} to "
            f"{exc.target_state}: that transition is not allowed. The lifecycle "
            "advances CREATED -> ACTIVE -> DONE -> ARCHIVED and cannot move "
            "backwards or skip ahead."
        )
        raise ToolError(msg) from exc
    except WriteContentionError as exc:
        # Another connection or process holds the database's write lock past
        # SQLite's own busy wait. The library guarantees nothing was written and
        # that retrying the whole call is safe, so the message says both. It does
        # not echo the operation name or attempt count, and it does not retry
        # here: the busy wait has already been spent, and a further server-side
        # retry could push a blocked call past the client's own timeout.
        msg = (
            "The memory store is busy: another writer is holding the same "
            "database, so this write could not start in time. Nothing was "
            "changed, and retrying the call is safe. Pause briefly, then retry."
        )
        raise ToolError(msg) from exc
    except WriteLockTimeoutError as exc:
        # A write ahead of this one held the store past the bound. Unlike
        # WriteContentionError, the library does not promise the timed-out call
        # left nothing behind, so the message must not say "nothing was changed"
        # and instead points the caller at reading the thought back. The bound
        # itself is internal and is not echoed.
        msg = (
            "The memory store was held by a long-running write, and this "
            "request timed out waiting for it. Wait a while, then retry. If "
            "you are not sure whether the request was applied, read the "
            "affected thought back first (get_thought or search_memory) "
            "before repeating it."
        )
        raise ToolError(msg) from exc
    except ConnectionQuarantinedError as exc:
        # Terminal for this server's store: every later operation fails fast
        # with the same error, so retrying can never help. The library's reason
        # text is diagnostic for the operator and is not echoed to the client.
        msg = (
            "This server's memory store is in an unusable state, and every "
            "request that needs it will fail until the server is restarted. "
            "Retrying will not help. Report this to whoever operates the server."
        )
        raise ToolError(msg) from exc
    except DerivedRecordError as exc:
        # store_thought only, reachable only when the engrava.yaml this
        # server is pointed at turns on derived records with a hooks class
        # and sets the failure policy to raise (the default -- log --
        # never raises through this path at all). One message covers all
        # three reasons this type carries (an over-cap producer return, a
        # derived record's identity colliding with its own source thought,
        # or colliding with an unrelated pre-existing thought) rather than
        # three different ones: none is actionable differently by the
        # caller, and what all three share -- the thought itself is
        # durably stored (derivation dispatches only after the source
        # thought's own commit), only its derived record is not -- is what
        # the client needs to know. The library's own message differs per
        # reason and names only its internal mechanism (e.g.
        # "[source=...] derived record identity collides with..."), so it
        # is not echoed.
        msg = (
            "The thought itself was stored successfully, but a derived "
            "record could not be produced for it: the derived-records "
            "producer this server is configured with either returned more "
            "records than the configured limit, or produced one whose "
            "identity collided with an existing thought. This is a "
            "property of the server's derived-records configuration, not "
            "of the fields you supplied -- report it to whoever operates "
            "the server."
        )
        raise ToolError(msg) from exc
    except _ResidualWriteError as exc:
        # store_thought / update_thought: the write ended with an exception
        # _residual_write_guard's classification (running it once through
        # this function's own chain) did not curate -- an unnamed type, a
        # named type no branch above recognises, a failure raised while
        # classifying it, or a ToolError the guarded call raised itself. It
        # may have fired before the write's own commit (an error on the
        # thought row's own INSERT) or after it (every other step either
        # call can reach) -- this function cannot tell which, so the message
        # never states whether the write took effect, and never names which
        # step raised. exc.thought_id is the id the write attempted -- the
        # caller's own, or the one this server generated for store_thought --
        # not an internal value, so it is safe to hand back as the id to
        # check. exc.deduplicate only changes what checking that id back can
        # prove, and only for store_thought.
        if exc.tool == "update_thought":
            msg = (
                f"update_thought for {exc.thought_id!r} ended with an error "
                "this server does not recognise, so whether the update was "
                "applied could not be confirmed. Read the thought back with "
                "get_thought before retrying. The error may originate in "
                "engrava itself, or in an extension this server's "
                "engrava.yaml configures -- this server cannot tell which "
                "from here."
            )
        elif exc.deduplicate:
            msg = (
                f"store_thought for {exc.thought_id!r} (deduplicate=True) "
                "ended with an error this server does not recognise, so "
                "whether the thought was stored could not be confirmed. "
                "With deduplicate=True the call may have matched an "
                f"existing thought with identical content instead of "
                f"storing a new one, so reading {exc.thought_id!r} back "
                "with get_thought cannot settle what happened: finding it "
                "shows a thought with that id exists, not that this call "
                "stored it. The error may originate in engrava itself, or "
                "in an extension this server's engrava.yaml configures -- "
                "this server cannot tell which from here."
            )
        else:
            msg = (
                f"store_thought for {exc.thought_id!r} ended with an error "
                "this server does not recognise, so whether the thought was "
                "stored could not be confirmed. Read it back with "
                "get_thought before retrying -- a blind retry can store a "
                "second copy. The error may originate in engrava itself, or "
                "in an extension this server's engrava.yaml configures -- "
                "this server cannot tell which from here."
            )
        raise ToolError(msg) from exc
    except ReferentialIntegrityError as exc:
        msg = (
            f"Cannot link thoughts: no thought exists with id "
            f"{exc.referenced_id!r}. Create that thought first, or correct "
            "the identifier."
        )
        raise ToolError(msg) from exc
    except (InvalidFilterError, InvalidFilterPathError) as exc:
        # A malformed ``metadata_equals`` / ``metadata_in`` filter on list_edges.
        # The raw messages spell out the internal JSONPath grammar (the accepted
        # regex and ``$.key`` examples) or name a rejected engrava internal
        # value shape; neither may reach the client. State the filter contract
        # in plain terms — simple field names as keys, JSON scalars as values —
        # without echoing the grammar.
        msg = (
            "The metadata filter is invalid. Use simple field names as keys "
            "(for example 'session_id' or 'topic') and JSON scalars — a string, "
            "number, or boolean — as values. Nested paths and structured values "
            "are not accepted here."
        )
        raise ToolError(msg) from exc
    except InvalidRecencyArgumentError as exc:
        # An unparseable ``recency_now`` on search_memory. The raw message echoes
        # the rejected value with engrava's own phrasing; surface a clean, format
        # -only hint instead.
        msg = "recency_now must be an ISO-8601 timestamp, for example '2026-07-20T14:30:00Z'."
        raise ToolError(msg) from exc
    except ValidationError as exc:
        # A field value rejected by the domain model (e.g. an essence below the
        # minimum length, or a value outside an enum). Pydantic's own message
        # names the internal model class and links errors.pydantic.dev, so it
        # must NOT be echoed; surface the offending field names only.
        fields = ", ".join(
            ".".join(str(part) for part in err.get("loc", ())) for err in exc.errors()
        )
        detail = f" (check: {fields})" if fields else ""
        msg = (
            f"One or more fields are invalid{detail}. Correct the value(s) and "
            "retry — see the tool's argument descriptions for the accepted "
            "types and ranges."
        )
        raise ToolError(msg) from exc
    except ValueError as exc:
        # Two of engrava's own failure modes reach this surface as a bare
        # ValueError rather than a dedicated type, so a message prefix is the
        # only discriminator available. This branch MUST stay below every
        # ValueError *subclass* branch above (UnsupportedQueryError,
        # EmbeddingQueryNotSupportedError, OutOfRangeBoundError,
        # MalformedFindError, UnexecutableQueryError, ValidationError) —
        # except clauses are tried in order, so if this one moved above them
        # it would catch their instances too and silently re-raise them
        # unmapped below, masking every one of those curated messages.
        text = str(exc)
        if text.startswith("Thought already exists: "):
            # store_thought with an explicit, colliding thought_id.
            duplicate_id = text.removeprefix("Thought already exists: ")
            msg = (
                f"A thought with id {duplicate_id!r} already exists. Use a "
                "different thought_id, omit it to have one generated "
                "automatically, or pass deduplicate=True to reuse a matching "
                "thought instead of failing."
            )
            raise ToolError(msg) from exc
        if text.startswith("metadata serialized size "):
            # link_thoughts metadata over engrava's serialized-size limit. The
            # store's own message also suggests moving the payload into
            # `content` — that advice is for thought metadata; this path is
            # edge-only, so the curated message stays edge-specific instead.
            match = _METADATA_SIZE_PATTERN.search(text)
            if match:
                size, maximum = match.group(1), match.group(2)
                msg = (
                    f"Edge metadata is too large: {size} bytes were supplied, "
                    f"but the maximum is {maximum} bytes. Store less data in "
                    "metadata, or keep the bulk of it in your own system and "
                    "store only a reference (e.g. an id) here."
                )
            else:
                msg = (
                    "Edge metadata is too large for the store's size limit. "
                    "Store less data in metadata, or keep the bulk of it in "
                    "your own system and store only a reference (e.g. an id) "
                    "here."
                )
            raise ToolError(msg) from exc
        raise
    except DuplicateEdgeError as exc:
        # The store's typed duplicate-edge signal from link_thoughts. Its own
        # message spells out the endpoints in engrava's phrasing, which is the
        # store's to change at will — so it is mapped to our curated wording
        # rather than forwarded, keeping this path indistinguishable from the
        # raw-constraint one below.
        raise ToolError(DUPLICATE_EDGE_MESSAGE) from exc
    except sqlite3.IntegrityError as exc:
        # A constraint violation from the database. The raw message names the
        # internal table and columns, which must not reach the client. Two
        # reachable cases share this exception class and must not share a
        # message: the (from, to, type) UNIQUE constraint from link_thoughts
        # (DUPLICATE_EDGE_MESSAGE), and a caller-supplied edge_id colliding
        # with the edge table's PRIMARY KEY (EDGE_ID_COLLISION_MESSAGE) — a
        # PRIMARY KEY is implemented as a UNIQUE index internally, so both
        # raise the same exception class with "UNIQUE" in the text, and
        # attributing the second to the first would name a cause the message
        # cannot know and suggest a repair (change the edge type) that does
        # not apply. Prefer sqlite_errorcode, the extended result code the
        # sqlite3 driver attaches when it raises this itself (Python 3.11+):
        # it distinguishes SQLITE_CONSTRAINT_PRIMARYKEY (1555) from
        # SQLITE_CONSTRAINT_UNIQUE (2067) without depending on message text at
        # all. That attribute is unset on an exception built by hand rather
        # than raised by the driver (e.g. a test double), so the column name
        # in the text is the fallback — "edge.edge_id" names only the single-
        # column PRIMARY KEY, never the three-column UNIQUE constraint.
        text = str(exc)
        if "UNIQUE" not in text:
            raise
        errorcode = getattr(exc, "sqlite_errorcode", None)
        if errorcode == _SQLITE_CONSTRAINT_PRIMARYKEY or "edge.edge_id" in text:
            raise ToolError(EDGE_ID_COLLISION_MESSAGE) from exc
        raise ToolError(DUPLICATE_EDGE_MESSAGE) from exc


class StoreProvider:
    """Holds the active store for the lifetime of a running server.

    The server lifespan calls :meth:`set` on startup and :meth:`clear`
    on shutdown.  Write tools call :meth:`require` to obtain the full,
    mutable store; read tools and resources call :meth:`require_read`
    instead, which returns the read-only view the lifespan installed when
    the server is running in read-only mode (see
    :data:`~engrava_mcp.read_only.ReadOnlyMcpStore`) — so a read can never
    stage a write regardless of which surface it came through.
    """

    def __init__(self) -> None:
        self._store: SqliteEngravaCore | None = None
        self._read_store: ReadOnlyMcpStore | None = None

    def set(self, store: SqliteEngravaCore, *, read_store: ReadOnlyMcpStore) -> None:
        """Record the active store.

        Args:
            store: The full, mutable store that write tools should use.
            read_store: The store read tools and resources should use —
                either ``store`` itself or a read-only view over it,
                decided by the caller.

        """
        self._store = store
        self._read_store = read_store

    def clear(self) -> None:
        """Forget the active store after shutdown."""
        self._store = None
        self._read_store = None

    def require(self) -> SqliteEngravaCore:
        """Return the active store, for a write tool.

        Returns:
            The store recorded by the lifespan.

        Raises:
            StoreNotReadyError: If no store is currently active.

        """
        if self._store is None:
            msg = "No active engrava store; the server lifespan is not running."
            raise StoreNotReadyError(msg)
        return self._store

    def require_read(self) -> ReadOnlyMcpStore:
        """Return the active read surface, for a read tool or resource.

        Returns:
            The read surface recorded by the lifespan — a read-only view in
            read-only mode, the full store otherwise.

        Raises:
            StoreNotReadyError: If no store is currently active.

        """
        if self._read_store is None:
            msg = "No active engrava store; the server lifespan is not running."
            raise StoreNotReadyError(msg)
        return self._read_store


async def get_thought_impl(store: ReadOnlyMcpStore, thought_id: str) -> dict[str, Any]:
    """Fetch a single thought by identifier.

    Args:
        store: The store to query.
        thought_id: Identifier of the thought to retrieve.

    Returns:
        A dict with a ``found`` flag and a ``thought`` entry.  ``thought``
        is the JSON-serialisable thought when it exists, otherwise
        ``None``.

    """
    thought = await store.get_thought(thought_id)
    if thought is None:
        return {"found": False, "thought": None}
    return {"found": True, "thought": thought.model_dump(mode="json")}


def _filter_criteria(
    *,
    thought_type: ThoughtType | None,
    lifecycle_status: LifecycleStatus | None,
    priority: Priority | None,
) -> dict[str, str]:
    """Collect the active thought filters as a JSON-friendly mapping.

    Only the filters the caller actually supplied appear in the result;
    each enum is reduced to its string value so the mapping serialises
    cleanly into a tool response.

    Args:
        thought_type: Thought-type filter, or ``None`` if not filtering.
        lifecycle_status: Lifecycle-status filter, or ``None``.
        priority: Priority filter, or ``None``.

    Returns:
        A dict mapping each supplied filter's field name to its string
        value.  Empty when no filter was supplied.

    """
    criteria: dict[str, str] = {}
    if thought_type is not None:
        criteria["thought_type"] = thought_type.value
    if lifecycle_status is not None:
        criteria["lifecycle_status"] = lifecycle_status.value
    if priority is not None:
        criteria["priority"] = priority.value
    return criteria


def _thought_matches(
    thought: ThoughtRecord,
    *,
    thought_type: ThoughtType | None,
    lifecycle_status: LifecycleStatus | None,
    priority: Priority | None,
) -> bool:
    """Report whether a thought satisfies every supplied filter.

    A ``None`` filter is not applied, so a thought matches when it equals
    each filter that *was* supplied (logical AND).  With no filters
    supplied this trivially returns ``True``.

    Args:
        thought: The thought record to test.
        thought_type: Required thought type, or ``None`` to ignore.
        lifecycle_status: Required lifecycle state, or ``None`` to ignore.
        priority: Required priority level, or ``None`` to ignore.

    Returns:
        ``True`` when the thought matches every supplied filter.

    """
    if thought_type is not None and thought.thought_type is not thought_type:
        return False
    if lifecycle_status is not None and thought.lifecycle_status is not lifecycle_status:
        return False
    return not (priority is not None and thought.priority is not priority)


async def search_memory_impl(
    store: ReadOnlyMcpStore,
    query_text: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    include_reflections: bool = True,
    thought_type: ThoughtType | None = None,
    lifecycle_status: LifecycleStatus | None = None,
    priority: Priority | None = None,
    recency_now: str | None = None,
) -> dict[str, Any]:
    """Run a hybrid ranked search over stored memory.

    The hybrid ranker itself does not filter by type, status, or
    priority, so any of those filters are applied *after* ranking: the
    ranked hits are fetched and the ones that do not match every supplied
    filter are dropped.  Ranking order is preserved and scores are never
    altered or fabricated — a filtered response simply carries fewer
    entries than ``top_k`` and reports how many were dropped (see the
    ``filtered`` block below) so the caller is never misled into reading
    an empty or short list as "nothing was found".

    By default the ranker never ranks archived thoughts at all, so a
    post-rank ``lifecycle_status=ARCHIVED`` filter could never keep
    anything — it would always report a confident, misleading empty
    result.  To keep that value reachable, archived thoughts are
    admitted into the ranked window only when ``lifecycle_status`` is
    ``ARCHIVED``; every other value, including no filter at all, ranks
    with the default archived-excluded behaviour.

    Args:
        store: The store to query.
        query_text: Natural-language query text.
        top_k: Maximum number of ranked results to consider.  Filters are
            applied to this ranked window, so the returned list may be
            shorter when filters drop hits.
        include_reflections: Whether consolidated reflection thoughts may
            appear in the results.
        thought_type: When set, keep only hits of this type.
        lifecycle_status: When set, keep only hits in this lifecycle state.
            Setting this to ``ARCHIVED`` also admits archived thoughts
            into the ranked window (see above); every other value ranks
            with archived thoughts excluded, as if no filter were set.
        priority: When set, keep only hits at this priority level.
        recency_now: Optional ISO-8601 timestamp used as "now" for the
            recency signal, letting a stateless consumer score recency by
            transaction time instead of a cognitive-cycle clock.  This
            server runs no such clock, so this argument is the only
            recency reference it can supply: when it is omitted the
            recency signal takes no part in ranking and ``recency`` does
            not appear in ``backends_used``.  Supplying it is necessary
            rather than sufficient — a store whose configuration gives
            the recency signal no weight still ranks without it.

    Returns:
        A dict with a ``results`` list of ``{"thought_id", "score"}``
        entries (ranking order preserved) and a ``backends_used`` list
        naming the search backends that were available for the query.
        When at least one filter is supplied, a ``filtered`` block is
        added carrying the active ``criteria`` and the ``scanned`` /
        ``matched`` / ``dropped`` counts over the ranked window, so a
        short or empty list is never mistaken for "no hits ranked".

    Raises:
        OutOfRangeBoundError: If ``top_k`` is outside its accepted range.
        InvalidRecencyArgumentError: If ``recency_now`` is not a valid
            ISO-8601 timestamp.

    """
    _check_bound("top_k", top_k, minimum=1, maximum=MAX_TOP_K)
    result = await store.search_hybrid(
        query_text,
        top_k=top_k,
        include_reflections=include_reflections,
        recency_now=recency_now,
        include_archived=lifecycle_status is LifecycleStatus.ARCHIVED,
    )
    backends_used = sorted(result.backends_used)

    criteria = _filter_criteria(
        thought_type=thought_type,
        lifecycle_status=lifecycle_status,
        priority=priority,
    )
    if not criteria:
        # Unfiltered path: byte-for-byte the original response shape.
        return {
            "results": [
                {"thought_id": thought_id, "score": score} for thought_id, score in result.results
            ],
            "backends_used": backends_used,
        }

    kept: list[dict[str, Any]] = []
    for thought_id, score in result.results:
        thought = await store.get_thought(thought_id)
        if thought is not None and _thought_matches(
            thought,
            thought_type=thought_type,
            lifecycle_status=lifecycle_status,
            priority=priority,
        ):
            kept.append({"thought_id": thought_id, "score": score})

    scanned = len(result.results)
    return {
        "results": kept,
        "backends_used": backends_used,
        "filtered": {
            "criteria": criteria,
            "scanned": scanned,
            "matched": len(kept),
            "dropped": scanned - len(kept),
        },
    }


async def search_keywords_impl(
    store: ReadOnlyMcpStore,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Run a full-text BM25 keyword search over stored memory.

    Args:
        store: The store to query.
        query: Full-text query string (supports ``AND``, ``OR``, ``NOT``
            and prefix ``*`` operators).
        top_k: Maximum number of ranked results to return.

    Returns:
        A dict with a ``results`` list of ``{"thought_id", "score"}``
        entries ordered by descending relevance.

    Raises:
        OutOfRangeBoundError: If ``top_k`` is outside its accepted range.

    """
    _check_bound("top_k", top_k, minimum=1, maximum=MAX_TOP_K)
    matches = await store.search_fts(query, top_k=top_k)
    return {
        "results": [{"thought_id": thought_id, "score": score} for thought_id, score in matches],
    }


#: Matches an optional leading ``EXPLAIN`` keyword the same way ``parse()``
#: does (case-insensitively, followed by whitespace or end-of-string), so
#: :func:`_query_declares_find` classifies the verb using exactly the same
#: rule ``parse()`` itself will apply, rather than a rule that could drift
#: from it.
_EXPLAIN_PREFIX_RE = re.compile(r"EXPLAIN(\s+|$)", re.IGNORECASE)


def _query_declares_find(query: str) -> bool:
    """Classify whether ``query``'s own command verb is ``FIND``.

    Used only to decide which of two messages a ``MindQLParseError`` gets —
    never to interpret the query, which remains ``parse()``'s job entirely.
    Mirrors ``parse()``'s own ``EXPLAIN``-stripping and whitespace-splitting
    tokenization far enough to read the first token, so this can never
    disagree with what ``parse()`` itself would call the verb.

    Fails toward ``False`` on anything not confidently recognised — an empty
    query, a bare ``EXPLAIN`` with nothing after it, leading/trailing
    whitespace that leaves no token at all. A false positive here would
    disclose ``parse()``'s command set for a query that is not actually a
    FIND; a false negative only costs a caller the specific diagnosis and
    falls back to the pre-existing generic message, which is the safe
    direction to fail in.

    Args:
        query: The raw MindQL query string, exactly as the caller wrote it.

    Returns:
        ``True`` when the first command token, after stripping an optional
        ``EXPLAIN`` prefix, is ``FIND`` case-insensitively; ``False`` for
        every other input, including ones this function does not recognise.

    """
    stripped = query.strip()
    if not stripped:
        return False

    explain_match = _EXPLAIN_PREFIX_RE.match(stripped)
    if explain_match:
        stripped = stripped[explain_match.end() :].strip()
        if not stripped:
            return False

    verb = stripped.split(maxsplit=1)[0]
    return verb.upper() == "FIND"


async def query_memory_impl(
    store: ReadOnlyMcpStore,
    query: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run a MindQL ``FIND`` query over stored memory.

    Only the ``FIND`` command is accepted.  The grammar is
    ``FIND <table> WHERE <field> <op> '<value>' [LIMIT n]``.

    Every call is capped at :data:`MAX_PAGE_LIMIT` rows regardless of what
    ``query`` asks for: a query with no ``LIMIT`` clause of its own gets the
    cap injected, and one whose own ``LIMIT`` already exceeds the cap is
    refused rather than silently truncated to it. **Precedence:** the
    ``limit`` argument, when supplied, always replaces any ``LIMIT`` already
    present in ``query`` outright — the two are never combined or compared,
    so a caller that wants the query's own ``LIMIT`` honoured must omit the
    argument.

    Args:
        store: The store to query.
        query: A MindQL ``FIND`` query string.
        limit: Optional row cap.  When provided, it overrides any ``LIMIT``
            clause present in ``query`` and is validated against the same
            ``[1, MAX_PAGE_LIMIT]`` range. When omitted, the query's own
            ``LIMIT`` is used if present and in range, or ``MAX_PAGE_LIMIT``
            is injected if the query carries none.

    Returns:
        A dict with the result ``columns`` and matching ``rows``.

    Raises:
        UnsupportedQueryError: If the query is not a ``FIND`` command.
        EmbeddingQueryNotSupportedError: If the query targets the
            ``embeddings`` table — refused outright, since its stored vector
            cannot be returned in a JSON tool result (see that error's
            docstring).
        OutOfRangeBoundError: If ``limit``, or an in-query ``LIMIT`` used in
            its place, is outside its accepted range.
        MindQLParseError: If the query is malformed and its own verb is not
            classified as ``FIND`` (see :func:`_query_declares_find`).
        MalformedFindError: If the query's own verb classifies as ``FIND``
            but it fails to parse (e.g. an unknown table or condition).
        UnexecutableQueryError: If the query parses as ``FIND`` but fails
            during execution (e.g. an unknown column).

    """
    # The verb is classified from the input itself, before parse() ever
    # runs, so that a MindQLParseError it then raises can be routed by where
    # the query committed to FIND rather than by the message's text (see
    # _query_declares_find and MalformedFindError).
    declares_find = _query_declares_find(query)
    try:
        parsed = parse(query)
    except MindQLParseError as exc:
        if declares_find:
            raise MalformedFindError(str(exc)) from exc
        raise
    if parsed.command is not MindQLCommand.FIND:
        raise UnsupportedQueryError(parsed.command.value)
    if parsed.table == "embedding":
        raise EmbeddingQueryNotSupportedError(parsed.table)

    # A value crossing the wire never becomes a query-object identifier: the
    # bound is validated *before* it is built into a MindQLQuery, because the
    # executor interpolates the limit into the SQL string rather than binding
    # it. Never rely on the protocol layer's coercion to have done this.
    #
    # Precedence (see the docstring): the `limit` argument always wins over a
    # LIMIT already present in the query text, replacing it outright. Only
    # when the argument is absent does the query's own LIMIT get a say — and
    # even then it is capped exactly like the argument would be, and a query
    # with no LIMIT at all gets MAX_PAGE_LIMIT injected rather than reaching
    # the store unbounded.
    if limit is not None:
        _check_bound("limit", limit, minimum=1, maximum=MAX_PAGE_LIMIT)
        effective = _with_limit(parsed, limit)
    elif parsed.limit is None:
        effective = _with_limit(parsed, MAX_PAGE_LIMIT)
    else:
        _check_bound("the query's LIMIT clause", parsed.limit, minimum=1, maximum=MAX_PAGE_LIMIT)
        effective = parsed

    # Execute via the public store-level entry point. The store owns the
    # connection; this consumer must not reach into it. The FIND-only guard
    # above is intentionally kept here (a consumer exposure policy), and no
    # ``extensions`` map is passed — both keep the over-the-wire surface
    # restricted to FIND.
    #
    # A MindQLParseError raised here is discriminated from one raised by
    # parse() above by WHERE it was raised, never by what it says: the query
    # has already parsed as FIND at this point, so whatever the executor
    # rejects it for cannot name another MindQL command. Wrap it in a
    # distinct type so `_tool_errors` can map the two differently.
    try:
        result = await store.execute_mindql(effective)
    except MindQLParseError as exc:
        raise UnexecutableQueryError(str(exc)) from exc
    return {"columns": result.columns, "rows": result.rows}


async def memory_stats_impl(store: ReadOnlyMcpStore) -> dict[str, Any]:
    """Return aggregate counts and store-health metrics.

    Args:
        store: The store to inspect.

    Returns:
        A dict with the live ``thought_count`` plus a ``metrics`` block
        carrying thought/edge counts, a storage-byte total, and a
        ``measured`` flag. When ``measured`` is ``False`` the store's
        metrics collection is disabled and the counts are zero-filled
        placeholders, not a real measurement.

    """
    thought_count = await store.count_thoughts()
    metrics = await store.metrics()
    return {
        "thought_count": thought_count,
        "metrics": {
            "thoughts": {
                "total": metrics.thoughts.total,
                "by_type": metrics.thoughts.by_type,
                "by_status": metrics.thoughts.by_status,
            },
            "edges": {
                "total": metrics.edges.total,
                "by_type": metrics.edges.by_type,
            },
            "storage_total_bytes": metrics.storage.total_bytes,
            "measured": metrics.measured,
        },
    }


async def recent_thoughts_impl(
    store: ReadOnlyMcpStore,
    *,
    limit: int = DEFAULT_RECENT_LIMIT,
) -> dict[str, Any]:
    """Return the thoughts with the highest cognitive cycle.

    Wraps the public :meth:`~engrava.SqliteEngravaCore.list_thoughts`,
    which orders by descending ``updated_cycle``.  Cognitive cycles are a
    signal the consuming application supplies; every thought this server
    writes stamps ``created_cycle = updated_cycle = 0`` and no update
    advances it, so for thoughts written through this server the first
    entry is not "the one touched most recently" — the query carries no
    tiebreaker, so SQLite guarantees nothing about which tied row comes
    first, and this call's result order among them is unspecified.

    Args:
        store: The store to query.
        limit: Maximum number of thoughts to return, highest cognitive
            cycle first (:data:`_CYCLE_ORDERING_NOTE`).

    Returns:
        A dict with a ``thoughts`` list of JSON-serialisable thoughts
        (highest cognitive cycle first) and the ``limit`` that was applied.

    Raises:
        OutOfRangeBoundError: If ``limit`` is outside its accepted range.

    """
    _check_bound("limit", limit, minimum=1, maximum=MAX_PAGE_LIMIT)
    thoughts = await store.list_thoughts(limit=limit)
    return {
        "thoughts": [thought.model_dump(mode="json") for thought in thoughts],
        "limit": limit,
    }


async def list_memory_impl(
    store: ReadOnlyMcpStore,
    *,
    thought_type: ThoughtType | None = None,
    lifecycle_status: LifecycleStatus | None = None,
    priority: Priority | None = None,
    min_cycle: int | None = None,
    max_cycle: int | None = None,
    include_expired: bool = False,
    limit: int = DEFAULT_LIST_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """List thoughts with filters and pagination.

    A direct pass-through to the public
    :meth:`~engrava.SqliteEngravaCore.list_thoughts`, which orders by
    descending ``updated_cycle`` (highest cognitive cycle first;
    :data:`_CYCLE_ORDERING_NOTE`) and applies every filter server-side.
    Unlike :func:`search_memory_impl` this is a plain
    browse: there is no relevance ranking and therefore no score.  It is
    the right tool when a caller wants an exhaustive, paginated slice of
    memory narrowed by structured fields rather than the best matches for
    a query.

    Args:
        store: The store to query.
        thought_type: When set, keep only thoughts of this type.
        lifecycle_status: When set, keep only thoughts in this state.
        priority: When set, keep only thoughts at this priority level.
        min_cycle: Inclusive lower bound on ``updated_cycle``.
        max_cycle: Inclusive upper bound on ``updated_cycle``.
        include_expired: When ``True``, expired thoughts are included.
            Defaults to ``False`` so expired thoughts stay hidden.
        limit: Maximum number of thoughts to return (page size).
        offset: Number of leading thoughts to skip (page start).

    Returns:
        A dict with a ``thoughts`` list of JSON-serialisable thoughts
        (highest cognitive cycle first), the ``count`` of thoughts on this
        page, and the ``limit`` / ``offset`` that were applied so the
        caller can drive pagination.

    Raises:
        OutOfRangeBoundError: If ``limit``, ``offset``, ``min_cycle``, or
            ``max_cycle`` is outside its accepted range.

    """
    _check_bound("limit", limit, minimum=1, maximum=MAX_PAGE_LIMIT)
    _check_bound("offset", offset, minimum=0, maximum=SQLITE_MAX_BOUND_INT)
    if min_cycle is not None:
        _check_bound(
            "min_cycle", min_cycle, minimum=SQLITE_MIN_BOUND_INT, maximum=SQLITE_MAX_BOUND_INT
        )
    if max_cycle is not None:
        _check_bound(
            "max_cycle", max_cycle, minimum=SQLITE_MIN_BOUND_INT, maximum=SQLITE_MAX_BOUND_INT
        )
    thoughts = await store.list_thoughts(
        thought_type=thought_type.value if thought_type is not None else None,
        lifecycle_status=lifecycle_status.value if lifecycle_status is not None else None,
        priority=priority.value if priority is not None else None,
        min_cycle=min_cycle,
        max_cycle=max_cycle,
        include_expired=include_expired,
        limit=limit,
        offset=offset,
    )
    serialised = [thought.model_dump(mode="json") for thought in thoughts]
    return {
        "thoughts": serialised,
        "count": len(serialised),
        "limit": limit,
        "offset": offset,
    }


async def get_edges_impl(
    store: ReadOnlyMcpStore,
    thought_id: str,
    *,
    direction: EdgeDirection = "BOTH",
) -> dict[str, Any]:
    """Return the edges connected to a thought.

    A direct pass-through to the public
    :meth:`~engrava.SqliteEngravaCore.get_edges`.  This is the read
    counterpart to :func:`link_thoughts_impl` / :func:`delete_edge_impl`:
    the write surface can create and remove edges but, without this, a
    client could never read them back.

    Args:
        store: The store to query.
        thought_id: Identifier of the thought whose edges to fetch.
        direction: Which edges to return — ``OUT`` for edges leaving the
            thought, ``IN`` for edges arriving at it, or ``BOTH`` for
            either.  An unknown ``thought_id`` simply has no edges.

    Returns:
        A dict with an ``edges`` list of JSON-serialisable edge records
        (each including its ``metadata``) and their ``count``.

    """
    edges = await store.get_edges(thought_id, direction=direction)
    serialised = [edge.model_dump(mode="json") for edge in edges]
    return {"edges": serialised, "count": len(serialised)}


def _metadata_filter(
    metadata_equals: dict[str, JsonScalar] | None,
    metadata_in: dict[str, list[JsonScalar]] | None,
) -> MetadataFilter | None:
    """Translate JSON-friendly metadata filters into engrava's typed filter.

    Each ``metadata_equals`` entry becomes an equality predicate and each
    ``metadata_in`` entry a membership predicate; the predicates are
    AND-conjoined into a single :class:`~engrava.MetadataFilter`.  The
    typed predicate machinery and its JSONPath grammar are constructed
    entirely here and never exposed over the wire — a caller supplies only
    plain field names and JSON scalars.

    Every key is first validated against
    :data:`METADATA_FIELD_NAME_PATTERN`: only simple, top-level field names
    are accepted.  A dotted or bracketed key would otherwise build a nested
    engrava JSONPath and reach nested metadata the thin surface does not
    expose, so such a key is rejected here — before any path is constructed —
    with the same :class:`~engrava.domain.exceptions.InvalidFilterPathError`
    that a malformed value uses, keeping the grammar entirely off the wire.

    Args:
        metadata_equals: Field-name to required-value mapping; each pair
            must match exactly.
        metadata_in: Field-name to allowed-values mapping; each field must
            equal one of the listed values.

    Returns:
        A ``MetadataFilter`` combining every supplied predicate, or
        ``None`` when neither argument carries any entry (match-all).

    Raises:
        InvalidFilterPathError: If a key is not a simple field name.
        InvalidFilterError: If a value is not an accepted scalar.

    """
    predicates: list[FieldPredicate] = []
    for key, value in (metadata_equals or {}).items():
        predicates.append(FieldPredicate(_metadata_path(key), FieldOp.EQ, value))
    for key, values in (metadata_in or {}).items():
        predicates.append(FieldPredicate(_metadata_path(key), FieldOp.IN, tuple(values)))
    if not predicates:
        return None
    return MetadataFilter(predicates)


def _metadata_path(key: str) -> str:
    """Build the JSONPath for a validated simple metadata field name.

    Args:
        key: A metadata field name supplied over the wire.

    Returns:
        The ``$.<key>`` JSONPath for a key that is a simple field name.

    Raises:
        InvalidFilterPathError: If ``key`` is not a simple field name
            (matching :data:`METADATA_FIELD_NAME_PATTERN`); a dotted or
            bracketed key that would reach nested metadata is rejected here.

    """
    if not METADATA_FIELD_NAME_PATTERN.fullmatch(key):
        raise InvalidFilterPathError(key)
    return f"$.{key}"


async def list_edges_impl(
    store: ReadOnlyMcpStore,
    *,
    edge_type: EdgeType | None = None,
    source: KnowledgeSource | None = None,
    metadata_equals: dict[str, JsonScalar] | None = None,
    metadata_in: dict[str, list[JsonScalar]] | None = None,
    limit: int = DEFAULT_EDGE_LIST_LIMIT,
) -> dict[str, Any]:
    """List edges with optional filters.

    A pass-through to the public
    :meth:`~engrava.SqliteEngravaCore.list_edges` that translates the
    JSON-friendly ``metadata_equals`` / ``metadata_in`` arguments into
    engrava's typed :class:`~engrava.MetadataFilter` internally (see
    :func:`_metadata_filter`), so the typed predicate machinery and its
    JSONPath grammar never appear on the wire.  ``edge_type`` and
    ``source`` are applied server-side by engrava.  The underlying query
    orders by descending ``created_cycle`` with no tiebreaker — the same
    property :data:`_CYCLE_ORDERING_NOTE` documents for thoughts applies
    here too: every edge created through :func:`link_thoughts_impl` stamps
    ``created_cycle = 0``, so the relative order among this server's own
    edges is unspecified.

    Args:
        store: The store to query.
        edge_type: When set, keep only edges of this relationship type.
        source: When set, keep only edges from this knowledge source.
        metadata_equals: Field-name to required-value mapping applied to
            each edge's metadata (exact match on every pair).
        metadata_in: Field-name to allowed-values mapping applied to each
            edge's metadata (the field must equal one of the values).
        limit: Maximum number of edges to return.

    Returns:
        A dict with an ``edges`` list of JSON-serialisable edge records
        (each including its ``metadata``, in unspecified relative order
        among ties) and their ``count``.

    Raises:
        OutOfRangeBoundError: If ``limit`` is outside its accepted range.
        InvalidFilterPathError: If a metadata key is not a simple field name.
        InvalidFilterError: If a metadata value is not an accepted scalar.

    """
    _check_bound("limit", limit, minimum=1, maximum=MAX_PAGE_LIMIT)
    filters = _metadata_filter(metadata_equals, metadata_in)
    edges = await store.list_edges(
        edge_type=edge_type,
        source=source,
        filters=filters,
        limit=limit,
    )
    serialised = [edge.model_dump(mode="json") for edge in edges]
    return {"edges": serialised, "count": len(serialised)}


async def store_thought_impl(
    store: SqliteEngravaCore,
    essence: str,
    content: str,
    *,
    thought_type: ThoughtType = ThoughtType.NOTE,
    priority: Priority = Priority.P3,
    source: str = "agent",
    confidence: float | None = None,
    thought_id: str | None = None,
    deduplicate: bool = False,
) -> dict[str, Any]:
    """Create a new thought node in the store.

    A :class:`~engrava.ThoughtRecord` is constructed from the supplied
    fields and persisted.  The remaining record fields take their model
    defaults.  New thoughts start in the ``CREATED`` lifecycle state at
    the origin cycle.

    Args:
        store: The store to write to.
        essence: Compact canonical text used in prompts (1-200 chars).
        content: Full stored content (non-empty).
        thought_type: Classification of the thought content.
        priority: Urgency level (``P1`` highest).
        source: Origin label for the thought (e.g. ``"agent"``, ``"human"``).
        confidence: Optional reliability estimate in ``[0.0, 1.0]``.
        thought_id: Optional caller-supplied identifier.  When omitted a
            fresh UUID4 is generated.
        deduplicate: When ``True``, an existing thought whose content hash
            matches has its confirmation count incremented and is returned
            instead of inserting a duplicate.

    Returns:
        A dict with a ``thought`` entry carrying the persisted thought's
        ``thought_id``, ``essence``, ``thought_type``, ``priority`` and
        ``lifecycle_status``.  When deduplication collapses onto an
        existing record, its identifier is returned.

    Raises:
        ValidationError: If a supplied field fails domain validation while
            :class:`~engrava.ThoughtRecord` is constructed -- before the
            guarded store call, so this propagates unchanged, never reshaped
            by :func:`_residual_write_guard`.
        ToolError: If ``store.create_thought`` raises an exception
            :func:`_tool_errors`'s own chain curates -- a caller-supplied
            ``thought_id`` colliding with an existing thought, an
            embedding-model mismatch, a required embedding's provider
            failing, or a rejected derived record, among others -- reached
            through :func:`_residual_write_guard`'s classification, carrying
            that branch's own curated message. Also raised directly by
            :func:`_residual_write_guard`, without reaching that chain, when
            ``store.create_thought`` raises a busy ``sqlite3.OperationalError``
            -- see :func:`_store_thought_busy_message`.
        _ResidualWriteError: If ``store.create_thought`` fails and
            classifying that failure through the :func:`_tool_errors` chain
            does not yield a curated message -- the chain re-raised it, or
            raised something else while classifying it. Whether the thought
            was stored could not be confirmed (see
            :func:`_residual_write_guard`).

    """
    record = ThoughtRecord(
        thought_id=thought_id if thought_id is not None else str(uuid.uuid4()),
        thought_type=thought_type,
        essence=essence,
        content=content,
        priority=priority,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=INITIAL_CYCLE,
        updated_cycle=INITIAL_CYCLE,
        source=source,
        confidence=confidence,
    )
    async with _residual_write_guard(
        store=store, tool="store_thought", thought_id=record.thought_id, deduplicate=deduplicate
    ):
        created = await store.create_thought(record, deduplicate=deduplicate)
    return {
        "thought": {
            "thought_id": created.thought_id,
            "essence": created.essence,
            "thought_type": created.thought_type.value,
            "priority": created.priority.value,
            "lifecycle_status": created.lifecycle_status.value,
        }
    }


async def update_thought_impl(
    store: SqliteEngravaCore,
    thought_id: str,
    *,
    essence: str | None = None,
    content: str | None = None,
    priority: Priority | None = None,
    lifecycle_status: LifecycleStatus | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Update selected fields of an existing thought.

    Only the fields the caller supplies are changed; every omitted
    argument leaves its stored value untouched.  Field changes are
    applied with the store's optimistic-concurrency guard.

    Args:
        store: The store to write to.
        thought_id: Identifier of the thought to update.
        essence: New compact canonical text, if changing.
        content: New full content, if changing.
        priority: New urgency level, if changing.
        lifecycle_status: New lifecycle state, if changing. Supplied as the
            string name of a :class:`~engrava.LifecycleStatus` member; the
            store validates that the transition is allowed.
        confidence: New reliability estimate in ``[0.0, 1.0]``, if changing.

    Returns:
        A dict with a ``thought`` entry carrying the updated thought's
        ``thought_id``, ``essence``, ``priority`` and ``lifecycle_status``.

    Raises:
        ToolError: If ``store.update_thought`` raises an exception
            :func:`_tool_errors`'s own chain curates -- a missing thought, a
            concurrent-write conflict, an illegal lifecycle transition, an
            embedding-model mismatch, or a required embedding's provider
            failing, among others -- reached through
            :func:`_residual_write_guard`'s classification, carrying that
            branch's own curated message.
        _ResidualWriteError: If ``store.update_thought`` fails and
            classifying that failure through the :func:`_tool_errors` chain
            does not yield a curated message -- the chain re-raised it, or
            raised something else while classifying it. Whether the update
            was applied could not be confirmed (see
            :func:`_residual_write_guard`).

    """
    changes: dict[str, object] = {}
    if essence is not None:
        changes["essence"] = essence
    if content is not None:
        changes["content"] = content
    if priority is not None:
        changes["priority"] = priority
    if lifecycle_status is not None:
        changes["lifecycle_status"] = lifecycle_status
    if confidence is not None:
        changes["confidence"] = confidence

    async with _residual_write_guard(store=store, tool="update_thought", thought_id=thought_id):
        updated = await store.update_thought(thought_id, **changes)
    return {
        "thought": {
            "thought_id": updated.thought_id,
            "essence": updated.essence,
            "priority": updated.priority.value,
            "lifecycle_status": updated.lifecycle_status.value,
        }
    }


async def link_thoughts_impl(
    store: SqliteEngravaCore,
    from_thought_id: str,
    to_thought_id: str,
    edge_type: EdgeType,
    *,
    weight: float = DEFAULT_EDGE_WEIGHT,
    edge_id: str | None = None,
    metadata: dict[str, JsonScalar] | None = None,
) -> dict[str, Any]:
    """Create a typed edge between two existing thoughts.

    An :class:`~engrava.EdgeRecord` is constructed from the supplied
    endpoints and persisted.  Both endpoints must already exist.

    Args:
        store: The store to write to.
        from_thought_id: Identifier of the source thought.
        to_thought_id: Identifier of the target thought.
        edge_type: Classification of the relationship.
        weight: Relation strength in ``[0.0, 1.0]``.
        edge_id: Optional caller-supplied identifier.  When omitted a
            fresh UUID4 is generated.
        metadata: Optional JSON object of extra fields to store on the edge,
            keyed by simple field names with JSON-scalar values.  This is the
            same metadata that ``list_edges`` filters on; when omitted the
            edge is stored with empty metadata.

    Returns:
        A dict with an ``edge`` entry carrying the persisted edge's
        ``edge_id``, ``from_thought_id``, ``to_thought_id``, ``edge_type``,
        ``weight`` and ``metadata``.

    Raises:
        ReferentialIntegrityError: If either endpoint does not exist.
        DuplicateEdgeError: If an edge with the same source, target and type
            already exists.  Edges are unique per ``(from, to, type)``, so
            this write is not idempotent — repeating an identical link is
            rejected rather than ignored.
        IntegrityError: If a caller-supplied ``edge_id`` collides with an
            existing edge's primary key.
        ValueError: If ``metadata`` serializes above the store's size limit.
        ToolError: If ``store.create_edge`` raises a busy
            ``sqlite3.OperationalError`` (:func:`_is_busy_error`), carrying
            :data:`_LINK_THOUGHTS_BUSY_MESSAGE`. Raised by :func:`_busy_guard`,
            wrapped around this one call only -- never around this function's
            own argument handling.

    """
    record = EdgeRecord(
        edge_id=edge_id if edge_id is not None else str(uuid.uuid4()),
        from_thought_id=from_thought_id,
        to_thought_id=to_thought_id,
        edge_type=edge_type,
        weight=weight,
        created_cycle=INITIAL_CYCLE,
        metadata=dict(metadata) if metadata else {},
    )
    # create_edge is one write unit begun BEGIN IMMEDIATE, with its journal
    # append inside it, then one commit through the store's own recovery
    # path, and nothing written after it -- so a busy error here, whether
    # from acquiring the lock or from the commit itself (a reader blocking
    # the upgrade, in rollback-journal mode), always means nothing was
    # written. Anything else propagates unchanged to the caller's own
    # _tool_errors, as before this guard existed. See _busy_guard, shared
    # with the two delete tools.
    async with _busy_guard(_LINK_THOUGHTS_BUSY_MESSAGE):
        created = await store.create_edge(record)
    return {
        "edge": {
            "edge_id": created.edge_id,
            "from_thought_id": created.from_thought_id,
            "to_thought_id": created.to_thought_id,
            "edge_type": created.edge_type.value,
            "weight": created.weight,
            "metadata": created.metadata,
        }
    }


async def delete_thought_impl(store: SqliteEngravaCore, thought_id: str) -> dict[str, Any]:
    """Delete a thought by identifier.

    This cascades: engrava deletes the thought's edges, embeddings, and
    action records in the same operation.  Deleting an identifier that is
    not present is a no-op rather than an error: the call simply reports
    that nothing was removed.

    Args:
        store: The store to write to.
        thought_id: Identifier of the thought to delete.

    Returns:
        A dict with a ``deleted`` flag: ``True`` when a thought was
        removed, ``False`` when no thought had the given identifier.

    Raises:
        ToolError: If ``store.delete_thought`` raises a busy
            ``sqlite3.OperationalError`` (:func:`_is_busy_error`), carrying
            :data:`_DELETE_BUSY_MESSAGE`. Raised by :func:`_busy_guard`,
            wrapped around this one call only -- never around this
            function's own argument handling.

    """
    # delete_thought, when it opens its own transaction, takes the write lock
    # (BEGIN IMMEDIATE) before its reads and writes. A failure in its write
    # unit (the delete, the vector purge, the journal append) or in its commit
    # is rolled back, or the connection quarantined, and nothing is written
    # after the commit. And a plain ROLLBACK cannot report SQLITE_BUSY. So a
    # busy error here means nothing was deleted; anything else propagates
    # unchanged to the caller's own _tool_errors.
    async with _busy_guard(_DELETE_BUSY_MESSAGE):
        deleted = await store.delete_thought(thought_id)
    return {"deleted": deleted}


async def delete_edge_impl(store: SqliteEngravaCore, edge_id: str) -> dict[str, Any]:
    """Delete an edge by identifier.

    Deleting an identifier that is not present is a no-op rather than an
    error: the call simply reports that nothing was removed.

    Args:
        store: The store to write to.
        edge_id: Identifier of the edge to delete.

    Returns:
        A dict with a ``deleted`` flag: ``True`` when an edge was removed,
        ``False`` when no edge had the given identifier.

    Raises:
        ToolError: If ``store.delete_edge`` raises a busy
            ``sqlite3.OperationalError`` (:func:`_is_busy_error`), carrying
            :data:`_DELETE_BUSY_MESSAGE`. Raised by :func:`_busy_guard`,
            wrapped around this one call only -- never around this
            function's own argument handling.

    """
    # delete_edge, when it opens its own transaction, takes the write lock
    # (BEGIN IMMEDIATE) before its reads and writes. A failure in its write
    # unit (the delete, the journal append) or in its commit is rolled back,
    # or the connection quarantined, and nothing is written after the commit.
    # And a plain ROLLBACK cannot report SQLITE_BUSY. So a busy error here
    # means nothing was deleted; anything else propagates unchanged to the
    # caller's own _tool_errors.
    async with _busy_guard(_DELETE_BUSY_MESSAGE):
        deleted = await store.delete_edge(edge_id)
    return {"deleted": deleted}


def _read_only_enabled() -> bool:
    """Report whether the server should expose a read-only surface.

    Reads :data:`READ_ONLY_ENV_VAR` and compares it against
    :data:`READ_ONLY_TRUTHY_VALUES` after stripping surrounding whitespace
    and lower-casing.  An unset or empty value is treated as not
    read-only.

    Returns:
        ``True`` when the environment requests a read-only surface,
        otherwise ``False``.

    """
    raw = os.environ.get(READ_ONLY_ENV_VAR, "")
    return raw.strip().lower() in READ_ONLY_TRUTHY_VALUES


def _with_limit(parsed: MindQLQuery, limit: int) -> MindQLQuery:
    """Return a copy of a parsed query with its ``limit`` replaced.

    Args:
        parsed: The parsed ``MindQLQuery``.
        limit: The row cap to apply.

    Returns:
        A new ``MindQLQuery`` identical to ``parsed`` but with ``limit``
        set to the supplied value.

    """
    return replace(parsed, limit=limit)


def _summarize_recent_prompt(limit: int, recent: dict[str, Any]) -> str:
    """Build the ``summarize_recent_memory`` prompt text.

    The text embeds the highest-cognitive-cycle thoughts already gathered
    from the store so the assistant can summarise them directly, while
    still naming the read tools and resources it can use to widen the
    picture.  Embedding is read-only: ``recent`` is the output of
    :func:`recent_thoughts_impl`.

    Args:
        limit: Number of thoughts the summary should cover, highest
            cognitive cycle first.
        recent: The payload returned by :func:`recent_thoughts_impl`,
            carrying a ``thoughts`` list ordered the same way
            (:data:`_CYCLE_ORDERING_NOTE`).

    Returns:
        A ready-to-send instruction asking for a concise summary of the
        highest-cognitive-cycle stored memory.

    """
    thoughts = recent.get("thoughts", [])
    if thoughts:
        snapshot = json.dumps(thoughts, indent=2)
        data_section = (
            f"Here are the {len(thoughts)} thoughts ({_CYCLE_ORDERING_NOTE}), "
            f"as JSON:\n\n{snapshot}\n\n"
        )
    else:
        data_section = "The store currently holds no thoughts to summarise.\n\n"
    return (
        f"Summarise the {limit} highest-cognitive-cycle memories in this "
        "engrava store.\n\n"
        f"{data_section}"
        "If you need more detail, "
        "read the `engrava://recent` resource or call the `memory_stats` "
        "tool; use `get_thought` to expand any single thought by its "
        "identifier. Produce a concise summary that highlights the main "
        "themes, any recurring topics, and anything that looks important "
        "or unresolved. Keep it brief — a short paragraph or a few bullet "
        "points."
    )


def _find_related_prompt(topic: str) -> str:
    """Build the ``find_related`` prompt text.

    Args:
        topic: The subject to find related thoughts about.

    Returns:
        A ready-to-send instruction asking the assistant to gather and
        synthesise thoughts related to ``topic`` using ``search_memory``.

    """
    return (
        f"Find and synthesise what this engrava memory store holds about "
        f"{topic!r}.\n\n"
        f"Use the `search_memory` tool with a query for {topic!r} (it ranks "
        "results by lexical, vector, and recency signals); you can also try "
        "`search_keywords` for an exact-term pass. Expand the most relevant "
        "hits with `get_thought` to read their full content. Then synthesise "
        "the findings into a short, organised summary of what is known about "
        f"{topic!r}, grouping related points and noting any gaps or "
        "contradictions."
    )


def _reflect_on_topic_prompt(topic: str) -> str:
    """Build the ``reflect_on_topic`` prompt text.

    Args:
        topic: The subject to reflect on.

    Returns:
        A ready-to-send instruction that scaffolds a structured reflection
        over what the store holds about ``topic``.

    """
    return (
        f"Reflect on what this engrava memory store holds about {topic!r}.\n\n"
        f"First gather the relevant memories: call `search_memory` for "
        f"{topic!r} and read the strongest hits in full with `get_thought`. "
        "Then reflect rather than merely listing: structure your response "
        "around (1) what is well established about the topic, (2) open "
        "questions or gaps in what is stored, and (3) any tensions or "
        "contradictions between thoughts. Close with one or two concrete "
        "follow-ups worth recording. Ground every observation in the "
        "retrieved thoughts."
    )


#: Exit code passed to ``os._exit`` when the connection close hit its bound
#: and abandoned a worker thread that never answered (see
#: :func:`_forced_exit_is_warranted`).  ``1`` (generic failure), absent a more
#: specific convention: nothing in this server's MCP client compatibility
#: matrix attaches meaning to a particular non-zero code, so there is no
#: sharper choice to make here.
_FORCED_EXIT_CODE = 1

#: Logged immediately before :func:`build_server`'s ``lifespan`` forces the
#: process to exit.
_FORCED_EXIT_WARNING = (
    "The database connection close abandoned a worker thread that never "
    "answered within its bound. That thread is not a daemon, so this "
    "process cannot exit on its own while it is still running. Forcing exit "
    "with code %s so a process supervisor, or the client that launched this "
    "server, notices immediately -- rather than being left with an orphaned "
    "process still holding a stale handle to the database file."
)


def _forced_exit_is_warranted(*, connection_close_hit_its_bound: bool) -> bool:
    """Decide, and log, whether the server process must force its own exit.

    Called from ``lifespan``'s ``finally`` block (see :func:`build_server`)
    with what :meth:`~engrava_mcp.config.ResolvedStore.aclose` returned. Kept
    separate from the ``os._exit`` call it gates -- the actual exit is a
    single line at ``lifespan``'s own outermost point, untested by
    construction -- so this decision, including the warning it logs, is
    exercised by a test directly, without the process actually exiting.
    Nothing else in this module calls ``os._exit``, so a test that calls this
    function, or that calls :meth:`~engrava_mcp.config.ResolvedStore.aclose`
    directly the way every existing shutdown test in
    ``tests/test_shutdown.py`` does, can never trigger it.

    Args:
        connection_close_hit_its_bound: What
            :meth:`~engrava_mcp.config.ResolvedStore.aclose` returned: whether
            the connection close abandoned a worker that never answered
            within its bound, rather than closing cleanly. ``False`` on a
            clean close and on the
            :class:`~engrava.ConnectionQuarantinedError` soft-warning path --
            that close does not leave a wedged worker behind, so it must
            never force an exit.

    Returns:
        ``connection_close_hit_its_bound``, unchanged -- the caller's cue to
        call ``os._exit``.

    """
    if connection_close_hit_its_bound:
        logger.warning(_FORCED_EXIT_WARNING, _FORCED_EXIT_CODE)
    return connection_close_hit_its_bound


def build_server() -> MCPServer:
    """Build the engrava MCP server with its tools registered.

    The returned server resolves its store from the environment when its
    lifespan starts and attempts to close the connection when the lifespan
    ends -- unless that attempt hits its bound on a genuinely wedged worker
    thread, in which case the process forces its own exit
    (:func:`_forced_exit_is_warranted`) instead of returning normally; see
    :meth:`~engrava_mcp.config.ResolvedStore.aclose` for what that bound is
    on either launch route. The read tools, the resources, and the prompts
    are always registered; the write tools are registered unless
    :func:`_read_only_enabled` reports a read-only deployment.

    Returns:
        A configured :class:`MCPServer` server ready to ``run()``.

    """
    provider = StoreProvider()
    # Read once and reuse: this decides both which store a read gets (below) and
    # which tools register (in register_tools). Calling _read_only_enabled() twice
    # and independently would let the two disagree if the environment changed
    # between build_server() and serving — read-only registration paired with an
    # unwrapped store, or the reverse.
    read_only = _read_only_enabled()

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
        resolved: ResolvedStore = await resolve_store()
        # In read-only mode, reads go through a view that never stages a write —
        # not even the deferred access-count update a plain read would buffer on
        # a store with access tracking on — so read_only_hint=True is true on every
        # configuration, not only on the one where tracking happens to be off.
        read_store: ReadOnlyMcpStore = (
            ReadOnlyStore(resolved.store) if read_only else resolved.store
        )
        provider.set(resolved.store, read_store=read_store)
        try:
            yield
        finally:
            provider.clear()
            # Shield the connection teardown so it runs to completion even
            # when the surrounding server task is being cancelled (as it is
            # on stdio EOF).  Without the shield the database worker thread
            # can outlive the event loop and raise on a late callback.
            with anyio.CancelScope(shield=True):
                connection_close_hit_its_bound = await resolved.aclose()
            forced_exit = _forced_exit_is_warranted(
                connection_close_hit_its_bound=connection_close_hit_its_bound
            )
            if forced_exit:
                # A genuine call terminates the process outright, so this line
                # is untestable by construction; everything that decides to
                # reach it -- the fact and the warning -- is proven through
                # _forced_exit_is_warranted instead (see its docstring and
                # build_server's for why a wedged worker thread leaves no other
                # way to end the process).
                #
                # tests/test_shutdown.py's TestForcedExitOnAWedgedShutdown does
                # drive the real lifespan to this exact line, with os._exit
                # monkeypatched so no real exit happens -- but coverage.py does
                # not attribute the line even so (it follows a genuine anyio
                # timeout-cancellation being caught, a combination coverage.py
                # is known to under-report). The pragma reflects that
                # measurement gap, not an actual absence of exercise.
                os._exit(_FORCED_EXIT_CODE)  # pragma: no cover

    server: MCPServer = MCPServer(
        SERVER_NAME,
        version=_server_version(),
        instructions=(
            "Access to an engrava agent-memory store: fetch thoughts, run "
            "hybrid and keyword search, list thoughts with structured filters "
            "and pagination, run structured MindQL FIND queries, and read "
            "store statistics. Hybrid search (search_memory) can also be "
            "narrowed by thought type, lifecycle status, or priority, but it "
            "filters after ranking; for an exhaustive unranked listing by "
            "those fields use list_memory. Read the edges of the memory graph "
            "with get_edges (the edges connected to a thought) and list_edges "
            "(browse edges filtered by type, source, or metadata). Unless the "
            "server is started in "
            "read-only mode, you can also store new thoughts, update existing "
            "thoughts, link thoughts with typed edges, and delete thoughts or "
            "edges. Read-only resources are also available as attachable "
            "context: a single thought (engrava://thought/{thought_id}), store "
            "statistics (engrava://stats), and the highest-cognitive-cycle "
            "thoughts (engrava://recent). Guided prompts scaffold common retrieval "
            "workflows: summarize_recent_memory, find_related, and "
            "reflect_on_topic."
        ),
        lifespan=lifespan,
    )
    register_resources(server, provider)
    register_prompts(server, provider)
    register_tools(server, provider, read_only=read_only)
    return server


def register_resources(server: MCPServer, provider: StoreProvider) -> None:
    """Register the read-only MCP resources on a server.

    Three resources are registered.  They are reads by definition, so —
    unlike the write tools — they are *not* gated by the read-only
    environment flag and are advertised in every deployment:

    ``engrava://thought/{thought_id}``
        A single thought as a JSON document.  An unknown identifier
        yields a graceful not-found payload rather than an error.
    ``engrava://stats``
        Store-health counts and size.  Shares :func:`memory_stats_impl`
        with the ``memory_stats`` tool, so the two agree by construction.
    ``engrava://recent``
        The stored thoughts, ordered by highest cognitive cycle first
        (:data:`_CYCLE_ORDERING_NOTE`), as a JSON document.

    Each handler returns a JSON string with the ``application/json`` MIME
    type, so clients receive a stable, machine-parseable payload.

    Args:
        server: The server to register resources on.
        provider: Supplies the active store to each resource at read time.

    """

    @server.resource(
        "engrava://thought/{thought_id}",
        name="thought",
        title="Thought",
        description="A single thought by its identifier, as a JSON document.",
        mime_type=RESOURCE_MIME_TYPE,
    )
    async def thought_resource(thought_id: str) -> str:
        payload = await get_thought_impl(provider.require_read(), thought_id)
        return json.dumps(payload)

    @server.resource(
        "engrava://stats",
        name="stats",
        title="Store statistics",
        description=(
            "Aggregate thought and edge counts and total storage size. Counts "
            "are zero-filled placeholders (see 'measured') if metrics "
            "collection is disabled."
        ),
        mime_type=RESOURCE_MIME_TYPE,
    )
    async def stats_resource() -> str:
        payload = await memory_stats_impl(provider.require_read())
        return json.dumps(payload)

    @server.resource(
        "engrava://recent",
        name="recent",
        title="Recent thoughts",
        description=(
            f"Returns the stored thoughts as a JSON document. The thoughts "
            f"are {_CYCLE_ORDERING_NOTE}."
        ),
        mime_type=RESOURCE_MIME_TYPE,
    )
    async def recent_resource() -> str:
        payload = await recent_thoughts_impl(provider.require_read())
        return json.dumps(payload)


def register_prompts(server: MCPServer, provider: StoreProvider) -> None:
    """Register the guided retrieval prompts on a server.

    Three prompts are registered.  They are parameterised templates that a
    client surfaces as slash-commands or buttons; each renders a
    ready-to-send instruction guiding the assistant to gather context with
    the read tools and resources before answering.  Prompts are
    read-oriented, so — like the resources and unlike the write tools —
    they are *not* gated by the read-only environment flag and are
    advertised in every deployment:

    ``summarize_recent_memory``
        Summarise the highest-cognitive-cycle thoughts
        (:data:`_CYCLE_ORDERING_NOTE`).  Takes an optional ``limit``; this
        is the one prompt that reads the store, embedding those thoughts
        (read-only) so the assistant can summarise them inline.
    ``find_related``
        Find and synthesise thoughts related to a required ``topic``.
    ``reflect_on_topic``
        Reflect over what memory holds about a required ``topic``.

    Args:
        server: The server to register prompts on.
        provider: Supplies the active store to ``summarize_recent_memory``
            at render time; the topic prompts are pure templates and do not
            use it.

    """

    @server.prompt(
        name="summarize_recent_memory",
        title="Summarise recent memory",
        description=(
            f"Summarise the thoughts {_CYCLE_ORDERING_NOTE}. Optionally set how many to consider."
        ),
    )
    async def summarize_recent_memory(limit: int = DEFAULT_SUMMARY_LIMIT) -> str:
        # ``limit`` is wire-supplied, so an out-of-range value must surface as a
        # curated message rather than a raw typed error — guarded like the
        # tools, but not *through* the tools' own ``PageLimit`` annotation: a
        # pydantic ``Field`` constraint on the parameter would let the SDK
        # reject the value before this body runs, but on mcp 2.x the argument
        # ``ValidationError`` that rejection raises is caught by the prompt
        # SDK's own rendering step and reported to the client only as an
        # opaque "Internal server error" (see ``_PromptBoundError``). Checking
        # the bound explicitly here, first, keeps both properties: the store is
        # still never reached for a rejected value, and the client still gets
        # an actionable message.
        _check_prompt_bound("limit", limit, minimum=1, maximum=MAX_PAGE_LIMIT)
        async with _tool_errors():
            recent = await recent_thoughts_impl(provider.require_read(), limit=limit)
            return _summarize_recent_prompt(limit, recent)

    @server.prompt(
        name="find_related",
        title="Find related thoughts",
        description="Find and synthesise stored thoughts related to a topic.",
    )
    def find_related(topic: str) -> str:
        return _find_related_prompt(topic)

    @server.prompt(
        name="reflect_on_topic",
        title="Reflect on a topic",
        description="Reflect on what stored memory holds about a topic.",
    )
    def reflect_on_topic(topic: str) -> str:
        return _reflect_on_topic_prompt(topic)


# C901: the mccabe count is inflated by the nested ``@server.tool`` handler
# definitions (one trivial delegating wrapper per tool), not by branching logic
# — this function has a single branch, the read-only guard. Splitting the flat
# registration list would hurt readability, so the complexity cap is waived here
# deliberately.
def register_tools(server: MCPServer, provider: StoreProvider, *, read_only: bool) -> None:  # noqa: C901
    """Register the MCP tools on a server.

    The eight read tools (``get_thought``, ``search_memory``,
    ``search_keywords``, ``list_memory``, ``query_memory``,
    ``memory_stats``, ``get_edges``, ``list_edges``) are always
    registered.  The five write tools are registered only when ``read_only``
    is ``False``; in read-only mode they are never advertised to clients.

    Args:
        server: The server to register tools on.
        provider: Supplies the active store to each tool at call time.
        read_only: Whether this deployment is read-only. Callers should pass
            the same value used to decide the store ``provider`` was given
            (see :func:`build_server`), rather than re-reading
            :func:`_read_only_enabled` independently — reading it twice
            lets registration and store-wrapping disagree if the
            environment changes in between.

    """

    @server.tool(
        name="get_thought",
        description="Fetch a single thought by its identifier.",
        annotations=_READ_ONLY,
    )
    async def get_thought(thought_id: str) -> dict[str, Any]:
        async with _tool_errors():
            return await get_thought_impl(provider.require_read(), thought_id)

    @server.tool(
        name="search_memory",
        description=(
            "Hybrid ranked search (lexical + vector + recency) over stored "
            "memory. Returns ranked thought identifiers with scores and the "
            "search backends that were available. Optionally narrow the "
            "ranked hits by thought type, lifecycle status, or priority; "
            "these filters are applied after ranking, so a filtered call may "
            "return fewer than top_k results and reports how many ranked hits "
            "were dropped. Archived thoughts are excluded from ranking by "
            "default; set lifecycle_status=ARCHIVED to search them instead. "
            "For an exhaustive, unranked, paginated listing by "
            "those same fields, use list_memory instead. Recency takes part in "
            "the ranking only when you pass recency_now: an ISO-8601 timestamp "
            "giving the moment to measure age against (transaction time)."
        ),
        annotations=_READ_ONLY,
    )
    async def search_memory(
        query_text: str,
        top_k: TopK = DEFAULT_TOP_K,
        *,
        include_reflections: bool = True,
        thought_type: ThoughtType | None = None,
        lifecycle_status: LifecycleStatus | None = None,
        priority: Priority | None = None,
        recency_now: str | None = None,
    ) -> dict[str, Any]:
        async with _tool_errors():
            return await search_memory_impl(
                provider.require_read(),
                query_text,
                top_k=top_k,
                include_reflections=include_reflections,
                thought_type=thought_type,
                lifecycle_status=lifecycle_status,
                priority=priority,
                recency_now=recency_now,
            )

    @server.tool(
        name="list_memory",
        description=(
            "List stored thoughts with optional filters and "
            "pagination. Unlike search_memory this does no relevance ranking "
            f"and returns no scores: it is a plain browse over memory, "
            f"{_CYCLE_ORDERING_NOTE}. Filter by thought type, lifecycle status, "
            "priority, and an updated-cycle range; page through results with "
            "limit and offset. Use this to enumerate memory by structured "
            "fields; use search_memory when you want the best matches for a "
            "query."
        ),
        annotations=_READ_ONLY,
    )
    async def list_memory(
        thought_type: ThoughtType | None = None,
        lifecycle_status: LifecycleStatus | None = None,
        priority: Priority | None = None,
        *,
        min_cycle: CycleFilterBound | None = None,
        max_cycle: CycleFilterBound | None = None,
        include_expired: bool = False,
        limit: PageLimit = DEFAULT_LIST_LIMIT,
        offset: PageOffset = 0,
    ) -> dict[str, Any]:
        async with _tool_errors():
            return await list_memory_impl(
                provider.require_read(),
                thought_type=thought_type,
                lifecycle_status=lifecycle_status,
                priority=priority,
                min_cycle=min_cycle,
                max_cycle=max_cycle,
                include_expired=include_expired,
                limit=limit,
                offset=offset,
            )

    @server.tool(
        name="search_keywords",
        description=(
            "Full-text BM25 keyword search over stored memory. Returns ranked "
            "thought identifiers with scores. Archived thoughts are never "
            "returned; this tool has no filter to widen that."
        ),
        annotations=_READ_ONLY,
    )
    async def search_keywords(query: str, top_k: TopK = DEFAULT_TOP_K) -> dict[str, Any]:
        async with _tool_errors():
            return await search_keywords_impl(provider.require_read(), query, top_k=top_k)

    @server.tool(
        name="query_memory",
        description=(
            "Run a structured MindQL FIND query over stored memory, e.g. "
            "\"FIND thoughts WHERE lifecycle_status = 'ACTIVE' LIMIT 10\". "
            "Only the FIND command is supported, and only against thoughts, "
            "edges, or actions — FIND embeddings is refused, since a stored "
            "embedding is raw vector bytes with no JSON representation. "
            "Results are capped at "
            f"{MAX_PAGE_LIMIT} rows: a query with no LIMIT clause gets this "
            "cap injected, and a LIMIT above it is refused rather than "
            "truncated. The limit argument, if given, always replaces any "
            "LIMIT already in the query text — the two are never combined."
        ),
        annotations=_READ_ONLY,
    )
    async def query_memory(query: str, limit: PageLimit | None = None) -> dict[str, Any]:
        async with _tool_errors():
            return await query_memory_impl(provider.require_read(), query, limit=limit)

    @server.tool(
        name="memory_stats",
        description=(
            "Return aggregate statistics about the memory store: thought and "
            "edge counts and total storage size. If the store's metrics "
            "collection is disabled, the counts in this response are "
            "zero-filled placeholders rather than a real measurement — check "
            "the 'measured' flag to tell the two cases apart."
        ),
        annotations=_READ_ONLY,
    )
    async def memory_stats() -> dict[str, Any]:
        async with _tool_errors():
            return await memory_stats_impl(provider.require_read())

    @server.tool(
        name="get_edges",
        description=(
            "Fetch the edges connected to a thought by its identifier. Choose "
            "the direction: OUT for edges leaving the thought, IN for edges "
            "arriving at it, or BOTH (the default) for either. Returns full edge "
            "records including their metadata, and a count. This is the read "
            "companion to link_thoughts and delete_edge."
        ),
        annotations=_READ_ONLY,
    )
    async def get_edges(
        thought_id: str,
        direction: EdgeDirection = "BOTH",
    ) -> dict[str, Any]:
        async with _tool_errors():
            return await get_edges_impl(provider.require_read(), thought_id, direction=direction)

    @server.tool(
        name="list_edges",
        description=(
            "List stored edges with optional filters. Filter by edge type, by "
            "knowledge source, and by edge metadata: metadata_equals takes a "
            "mapping of field name to a required value (exact match on every "
            "pair), and metadata_in takes a mapping of field name to a list of "
            "allowed values (the field must equal one of them). Metadata keys "
            "are simple field names and values are JSON scalars. Returns full "
            "edge records including their metadata, and a count."
        ),
        annotations=_READ_ONLY,
    )
    async def list_edges(
        edge_type: EdgeType | None = None,
        source: KnowledgeSource | None = None,
        metadata_equals: dict[str, JsonScalar] | None = None,
        metadata_in: dict[str, list[JsonScalar]] | None = None,
        limit: PageLimit = DEFAULT_EDGE_LIST_LIMIT,
    ) -> dict[str, Any]:
        async with _tool_errors():
            return await list_edges_impl(
                provider.require_read(),
                edge_type=edge_type,
                source=source,
                metadata_equals=metadata_equals,
                metadata_in=metadata_in,
                limit=limit,
            )

    if read_only:
        return

    @server.tool(
        name="store_thought",
        description=(
            "Create a new thought node. Provide its essence (short canonical "
            "text) and full content; optionally set the thought type, "
            "priority, source, and confidence. Returns the created thought's "
            "identifier and key fields."
        ),
        annotations=_WRITE,
    )
    async def store_thought(
        essence: str,
        content: str,
        thought_type: ThoughtType = ThoughtType.NOTE,
        priority: Priority = Priority.P3,
        source: str = "agent",
        *,
        confidence: float | None = None,
        thought_id: str | None = None,
        deduplicate: bool = False,
    ) -> dict[str, Any]:
        async with _tool_errors():
            return await store_thought_impl(
                provider.require(),
                essence,
                content,
                thought_type=thought_type,
                priority=priority,
                source=source,
                confidence=confidence,
                thought_id=thought_id,
                deduplicate=deduplicate,
            )

    @server.tool(
        name="update_thought",
        description=(
            "Update fields of an existing thought by identifier. Only the "
            "fields you supply change; omit the rest. Can change essence, "
            "content, priority, lifecycle status, and confidence."
        ),
        annotations=_WRITE,
    )
    async def update_thought(
        thought_id: str,
        essence: str | None = None,
        content: str | None = None,
        priority: Priority | None = None,
        lifecycle_status: LifecycleStatus | None = None,
        *,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        async with _tool_errors():
            return await update_thought_impl(
                provider.require(),
                thought_id,
                essence=essence,
                content=content,
                priority=priority,
                lifecycle_status=lifecycle_status,
                confidence=confidence,
            )

    @server.tool(
        name="link_thoughts",
        description=(
            "Create a typed edge between two existing thoughts, identified by "
            "their identifiers. Choose the edge type and optionally a weight "
            "in [0.0, 1.0]. Optionally attach a metadata object (simple field "
            "names to JSON scalars) that list_edges can later filter on. Both "
            "endpoints must already exist. An edge is unique per (source, "
            "target, type): linking the same pair with the same type twice is "
            "rejected rather than ignored."
        ),
        annotations=_WRITE,
    )
    async def link_thoughts(
        from_thought_id: str,
        to_thought_id: str,
        edge_type: EdgeType,
        weight: float = DEFAULT_EDGE_WEIGHT,
        *,
        edge_id: str | None = None,
        metadata: dict[str, JsonScalar] | None = None,
    ) -> dict[str, Any]:
        async with _tool_errors():
            return await link_thoughts_impl(
                provider.require(),
                from_thought_id,
                to_thought_id,
                edge_type,
                weight=weight,
                edge_id=edge_id,
                metadata=metadata,
            )

    @server.tool(
        name="delete_thought",
        description=(
            "Delete a thought by its identifier. Use this to remove a memory "
            "that is wrong or no longer wanted. This cascades: the thought's "
            "edges, embeddings, and action records are deleted with it. "
            "Returns whether a thought was removed; deleting an identifier "
            "that does not exist is not an error and simply reports that "
            "nothing was removed."
        ),
        annotations=_WRITE_DESTRUCTIVE,
    )
    async def delete_thought(thought_id: str) -> dict[str, Any]:
        async with _tool_errors():
            return await delete_thought_impl(provider.require(), thought_id)

    @server.tool(
        name="delete_edge",
        description=(
            "Delete an edge between two thoughts by its identifier. Use this to "
            "remove a relationship that is wrong or no longer wanted. Returns "
            "whether an edge was removed; deleting an identifier that does not "
            "exist is not an error and simply reports that nothing was removed."
        ),
        annotations=_WRITE_DESTRUCTIVE,
    )
    async def delete_edge(edge_id: str) -> dict[str, Any]:
        async with _tool_errors():
            return await delete_edge_impl(provider.require(), edge_id)


def main() -> None:
    """Run the engrava MCP server over stdio.

    Builds the server and serves it on the stdio transport (the MCPServer
    default).  This is the console-script, the ``python -m engrava_mcp``,
    and the ``python -m engrava_mcp.server`` entry point.  A soft warning is
    emitted first if the installed engrava version is outside the tested range.
    """
    warn_if_engrava_out_of_range()
    build_server().run()


if __name__ == "__main__":  # pragma: no cover - module-run guard; covered via `python -m`
    main()
