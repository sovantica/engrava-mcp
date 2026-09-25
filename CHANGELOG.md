# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows a one-way version mirror of [Engrava](https://github.com/sovantica/engrava)
(`engrava-mcp X.Y.z` targets `engrava X.Y`).

## [0.7.0]

### Added

- Add an optional `limit` to `get_edges`, returning at most that many edges, the highest-weight ones first.
- Add a `measured` flag to `memory_stats` and `engrava://stats`, which tells zero-filled placeholder metrics (metrics collection off) from a real measurement.
- Log to stderr which configuration route the server is opening the store from, and when the store is ready.

### Changed

- **Breaking:** require the MCP SDK 2.x line (`mcp>=2.2,<3`); the 1.x line is no longer supported.
- Target Engrava 0.7 — now requires `engrava >=0.7,<0.8`.
- Cap `query_memory` at 5000 rows, and refuse a query whose own `LIMIT` is outside 1–5000 when no `limit` argument is passed.
- Make `update_thought` fail with a conflict error, applying nothing, when another write changes or deletes the thought during the update; it is no longer annotated idempotent.
- Declare `anyio`, `aiosqlite` and `pydantic` as direct dependencies instead of relying on other packages to install them.
- Recommend `uv tool install engrava-mcp` for daily use, and an embedding provider outside the server process for MCP deployments.

### Fixed

- Find archived thoughts when searching with `lifecycle_status=ARCHIVED`, which previously returned nothing.
- Keep reads in read-only mode from writing anything of their own, including deferred access-count updates.
- Report this server's own version in the `initialize` handshake instead of the MCP SDK's.
- Return clear tool errors for more library failures; when SQLite reports database contention, `link_thoughts`, `delete_thought` and `delete_edge` say that nothing changed and that a retry is safe.
- Exit at shutdown instead of hanging when closing the database is stuck.
- Stop describing results as newest-first (`list_memory`, `engrava://recent`, `summarize_recent_memory` and others): thoughts and edges written through this server all carry cycle 0, so the order among them is unspecified.
- State in `delete_thought`'s description that the thought's edges, embeddings and action records are deleted with it.
- Correct the README: the audit trail is available only through `ENGRAVA_MCP_CONFIG`; the `ENGRAVA_DB_PATH` route builds the store with no journal.
- Stop advertising audit verification and action records in the MCP Registry description; this server exposes neither.
- Fix the README's example `engrava.yaml`: `database.path` and `provider: openai-compatible` replace keys that were never loaded.
- Document in the README that a thought this server stores gets an embedding only when `embeddings.auto_embed` is on (off by default), and that without it `update_thought` keeps a thought's existing embedding.
- Document in the README that read-only mode still opens the database read-write, creates it, and upgrades its schema at startup.
- Fix the README's example config to use an absolute database path, and say where `OPENAI_API_KEY` must be set.
- Declare `ENGRAVA_DB_PATH` (required), `ENGRAVA_MCP_CONFIG` and `ENGRAVA_MCP_READ_ONLY` in the registry manifest, so a host that configures this server from the MCP Registry asks for a database path instead of starting a server that exits for lack of one.
- Bug fixes and stability improvements.
- `query_memory` refuses an `OFFSET` beyond SQLite's integer range with a clear message instead of an unexplained error.

## [0.6.0]

### Added

- `get_edges` and `list_edges` tools — read and browse the memory graph's edges: traverse a thought's edges by direction, or filter edges by type, source, or metadata.
- Optional `metadata` on `link_thoughts` — attach JSON fields to an edge that `list_edges` can filter on.
- Optional `recency_now` on `search_memory` — score recency against a caller-supplied timestamp (transaction time).
- Report the Engrava extensions advertised by installed packages at startup on the `ENGRAVA_DB_PATH` launch, which attaches no extension hooks and so leaves a store-hook extension inactive.

### Fixed

- Honour `recency_now` on the `ENGRAVA_DB_PATH` quick-start — recency is now scored against the supplied timestamp there, where the argument was previously accepted and silently ignored.

### Changed

- Target Engrava 0.6 — now requires `engrava >=0.6,<0.7`.
- Point the `Documentation` project URL at the MCP server guide instead of the repository.
- Validate the numeric bounds accepted by the search, list, and query tools — a negative, zero, or excessively large `limit`, `top_k`, or `offset` is now rejected instead of silently returning an unbounded result. Callers that previously passed a value outside the accepted range now get an error.

## [0.5.1]

### Added

- `mcp-name: ai.sovantica/engrava` marker in the README so the MCP Registry can validate PyPI-package ownership and list the server. No functional changes.

## [0.5.0]

First standalone release of the Engrava MCP server.

### Added

- Standalone, runnable MCP server for Engrava — `uvx engrava-mcp` (or `pip install engrava-mcp`).
- 11 tools, 3 resources, and 3 prompts exposed over Engrava's public API via stdio.
- Read-only mode via `ENGRAVA_MCP_READ_ONLY` — write tools are not registered when enabled.
- Store resolution from environment: `ENGRAVA_MCP_CONFIG` (full `engrava.yaml`) or `ENGRAVA_DB_PATH` (bare SQLite quick-start).
- Optional embedding-provider extras (`local`, `hf`, `openai`, `ollama`) mirroring Engrava's own extras.
- Requires `engrava >=0.5,<0.6`, pulled in transitively so `import engrava` is available in the same environment.

### Changed

- Extracted from the former `engrava[mcp]` extra into this standalone package. Install `engrava-mcp` (or `uvx engrava-mcp`) instead of `pip install "engrava[mcp]"`, and update any pinned `engrava[mcp]` requirements to depend on `engrava-mcp`.

[0.7.0]: https://github.com/sovantica/engrava-mcp/releases/tag/v0.7.0
[0.6.0]: https://github.com/sovantica/engrava-mcp/releases/tag/v0.6.0
[0.5.1]: https://github.com/sovantica/engrava-mcp/releases/tag/v0.5.1
[0.5.0]: https://github.com/sovantica/engrava-mcp/releases/tag/v0.5.0
