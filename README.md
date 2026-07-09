<!-- mcp-name: ai.sovantica/engrava -->

# Engrava MCP

[![CI](https://github.com/sovantica/engrava-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/sovantica/engrava-mcp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/engrava-mcp.svg)](https://pypi.org/project/engrava-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/engrava-mcp.svg)](https://pypi.org/project/engrava-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The [Model Context Protocol](https://modelcontextprotocol.io) server for
[Engrava](https://github.com/sovantica/engrava)** — expose an agent memory
database to any MCP client (Claude Desktop, Claude Code, Cursor, Windsurf,
VS Code, …) over stdio.

`engrava-mcp` is a standalone, runnable package that consumes Engrava's public
API. It is the one way to run Engrava as a memory server; the `engrava` library
itself ships no MCP code.

```bash
uvx engrava-mcp        # run the server (no install step)
# or
pip install engrava-mcp
engrava-mcp            # spawned by your MCP client over stdio
```

Installing `engrava-mcp` pulls in `engrava` transitively, so you also get the
`import engrava` library in the same environment.

## Compatibility

`engrava-mcp` follows Engrava's version: **`engrava-mcp X.Y.z` targets `engrava X.Y`**
and requires `engrava >=X.Y,<X.(Y+1)`. This is a one-way version mirror for legibility —
**not** a lockstep: Engrava releases on its own cadence, and `engrava-mcp` patch releases
are independent.

| engrava-mcp | Works with engrava |
|---|---|
| `0.5.x` | `>=0.5,<0.6` |

The dependency range is the source of truth. Normal installs resolve a compatible
`engrava` automatically; if you pin `engrava` yourself, keep it within that range. If no
matching `engrava-mcp` exists yet for a newer `engrava` (e.g. a fresh `engrava 0.6`), that
pairing is **not yet verified/supported** — not broken; stay on a supported pair until a
matching `engrava-mcp` ships.

## Which package do I want?

| Goal | Install |
|---|---|
| Build on the Engrava Python API (memory DB in your own code) | `pip install engrava` |
| Run Engrava as a memory server for an MCP client | `uvx engrava-mcp` (or `pip install engrava-mcp`) |

There is no third option.

## Migrating from `engrava[mcp]`

The server used to ship inside Engrava as the `engrava[mcp]` extra and an
in-`engrava` `engrava-mcp` command. As of Engrava 0.5.0 it lives here instead.

| Before | After |
|---|---|
| `pip install "engrava[mcp]"` | `pip install engrava-mcp` (or `uvx engrava-mcp`) |
| `engrava-mcp` (installed by engrava) | `engrava-mcp` (installed by this package) |
| client `mcp.json`: `"command": "engrava-mcp"` | client `mcp.json`: `"command": "uvx", "args": ["engrava-mcp"]` |

- **Watch out:** `pip install "engrava[mcp]"` against Engrava 0.5 **does not
  fail** — pip ignores the now-unknown extra and quietly installs bare
  `engrava`, so it can look like the server installed when it did not. Install
  `engrava-mcp` instead.
- Update any pinned requirement strings (`engrava[mcp]>=...`) to depend on
  `engrava-mcp`, not just reinstall.
- **Your store configuration is unchanged** — the same `engrava.yaml` / env vars
  work exactly as before (see [Configuration](#configuration)).

## Configuration

The server resolves its store from environment variables, in priority order:

| Variable | Meaning |
|---|---|
| `ENGRAVA_MCP_CONFIG` | Path to an `engrava.yaml`. Built with the full configuration — embedding provider, vector backend, journal, TTL. **Recommended.** |
| `ENGRAVA_DB_PATH` | Path to a bare SQLite database file. Zero-config quick-start; no embedding provider is configured, so semantic (vector) search is inert — full-text search, the graph, MindQL, and the audit trail still work. |
| `ENGRAVA_MCP_READ_ONLY` | When set to `1` / `true` / `yes`, the write tools are not registered, so the server exposes a read-only surface. |

**Recommended:** give the MCP server the same `engrava.yaml` your application
uses. The `yaml` is the only place to declare an embedding provider (and its
model / key), which the server needs to embed a *new query* at search time for
semantic search. With only `ENGRAVA_DB_PATH` set, the server logs a startup
warning that semantic search is inert and points you at `ENGRAVA_MCP_CONFIG`.

### Example `engrava.yaml`

```yaml
db_path: ./memory.db
embeddings:
  provider: openai            # or: ollama, sentence-transformer, huggingface
  model: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}
```

## Client setup

Point your MCP client at the server over stdio. For example, a typical
`mcp.json` entry:

```json
{
  "mcpServers": {
    "engrava": {
      "command": "uvx",
      "args": ["engrava-mcp"],
      "env": {
        "ENGRAVA_MCP_CONFIG": "/absolute/path/to/engrava.yaml"
      }
    }
  }
}
```

Use `ENGRAVA_DB_PATH` instead of `ENGRAVA_MCP_CONFIG` for the zero-config
quick-start, and add `"ENGRAVA_MCP_READ_ONLY": "1"` for an app-writes /
agent-reads deployment.

### Running without uvx

```bash
engrava-mcp                  # console script
python -m engrava_mcp        # module run
python -m engrava_mcp.server # module run (server module directly)
```

## Optional providers

The default install supports the vector backend and HTTP-based embedding
providers (OpenAI / Ollama) once configured in the `yaml`. Heavier providers are
opt-in extras that mirror Engrava's own extras:

```bash
uvx --from "engrava-mcp[local]"  engrava-mcp   # sentence-transformers (local model)
uvx --from "engrava-mcp[hf]"     engrava-mcp   # HuggingFace Inference API
uvx --from "engrava-mcp[openai]" engrava-mcp   # OpenAI-compatible embeddings deps
uvx --from "engrava-mcp[ollama]" engrava-mcp   # Ollama embeddings deps
```

## The surface

- **Tools (11):** `get_thought`, `search_memory`, `search_keywords`,
  `list_memory`, `query_memory`, `memory_stats` (read); `store_thought`,
  `update_thought`, `link_thoughts`, `delete_thought`, `delete_edge` (write,
  gated by `ENGRAVA_MCP_READ_ONLY`).
- **Resources (3):** `engrava://thought/{thought_id}`, `engrava://stats`,
  `engrava://recent`.
- **Prompts (3):** `summarize_recent_memory`, `find_related`, `reflect_on_topic`.

`query_memory` accepts only MindQL `FIND` queries; raw SQL and every other
command are rejected.

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/
ruff format --check src/ tests/
mypy --strict src/
pytest --cov --cov-fail-under=90
```

## License

MIT
