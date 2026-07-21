# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows a one-way version mirror of [Engrava](https://github.com/sovantica/engrava)
(`engrava-mcp X.Y.z` targets `engrava X.Y`).

## [0.6.0]

### Added

- `get_edges` and `list_edges` tools — read and browse the memory graph's edges: traverse a thought's edges by direction, or filter edges by type, source, or metadata.
- Optional `metadata` on `link_thoughts` — attach JSON fields to an edge that `list_edges` can filter on.
- Optional `recency_now` on `search_memory` — score recency against a caller-supplied timestamp (transaction time).

### Changed

- Target Engrava 0.6 — now requires `engrava >=0.6,<0.7`.

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

[0.5.1]: https://github.com/sovantica/engrava-mcp/releases/tag/v0.5.1
[0.5.0]: https://github.com/sovantica/engrava-mcp/releases/tag/v0.5.0
