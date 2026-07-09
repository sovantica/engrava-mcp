# Contributing to engrava-mcp

Thank you for your interest in contributing to **engrava-mcp**! This document
explains how to set up a development environment, submit changes, and what we
look for in contributions.

## Scope

engrava-mcp is the **[Model Context Protocol](https://modelcontextprotocol.io)
server for [Engrava](https://github.com/sovantica/engrava)** — a thin, standalone
package that exposes Engrava's public API to any MCP client over stdio.
Contributions should stay within this scope:

**In scope:**
- MCP tool / resource / prompt improvements over Engrava's public API
- Configuration and store-resolution ergonomics (env vars, `engrava.yaml`)
- Read-only gating and other deployment modes
- stdio transport robustness and error handling
- Documentation, examples, and client setup guides
- Bug fixes and test coverage

**Out of scope:**
- Memory-database internals — those belong in
  [`engrava`](https://github.com/sovantica/engrava), which this server consumes
  as a library. Open API or storage changes there.
- Application-layer logic (planners, reasoners, cognitive architectures).
- Non-stdio transports unless first discussed in an issue.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/sovantica/engrava-mcp.git
cd engrava-mcp

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\Activate.ps1    # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Quality Standards

This project maintains strict code quality. All contributions must pass:

### Linting (ruff)

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

All ruff rules are enabled (`select = ["ALL"]`). Fix any violations before
submitting.

### Type Checking (mypy)

```bash
mypy --strict src/
```

Zero errors required. Use proper type annotations — no `Any`, no `type: ignore`
without justification.

### Tests (pytest)

```bash
pytest --cov --cov-fail-under=90
```

- Coverage must stay at or above **90%**.
- Use `async def test_*` directly — `asyncio_mode = "auto"` is configured.
- Prefer real implementations over mocks.
- New features require corresponding tests.

## Pull Request Guidelines

1. **One feature per PR** — keep changes focused and reviewable.
2. **Run all checks locally** before submitting:
   ```bash
   ruff check src/ tests/
   ruff format --check src/ tests/
   mypy --strict src/
   pytest --cov --cov-fail-under=90
   ```
3. **Write clear commit messages** — Conventional Commits 1.0.0, imperative mood
   ("Add read-only gating", not "Added read-only gating").
4. **Update documentation** if your change affects the tool surface, configuration,
   or client setup.
5. **Add docstrings** — Google-style docstrings on all public symbols.

## Branching

This repo follows a three-tier `feature/* → release/vX.Y.Z → main` flow.

| Branch | Role |
|---|---|
| `main` | Stable mirror that users land on when cloning. Forward-merged from the active `release/vX.Y.Z` branch when a version is ready. |
| `release/v<X.Y.Z>` | The active integration and stabilisation branch for the next version. Feature branches land here (squash), then it is forward-merged into `main`. |
| `<type>/<kebab-description>` | Feature / fix / chore branches. Plain English; ≤ 50 chars; lowercase kebab-case (e.g. `feature/read-only-resources`, `fix/empty-config-warning`). |

Open PRs from your feature branch targeting the active `release/vX.Y.Z` branch. CI must be green before review. The CHANGELOG is hand-maintained: add your user-facing change to the current release's section as part of your PR.

## Commit messages — Conventional Commits 1.0.0

Format: `<type>(<scope>): <description>`

Types: `feat`, `fix`, `perf`, `docs`, `style`, `refactor`, `test`, `build`,
`ci`, `chore`, `revert`. Scopes: `server`, `tools`, `resources`, `prompts`,
`config`, `docs`, `deps`, `release`, `ci`, `build`.

Breaking changes: add `!` after type/scope AND a `BREAKING CHANGE:` footer.

Keep branch names, commit messages, and referenced symbols public-safe — describe
the change in plain English and avoid leaking any private workflow detail.

## Code Style

- **Async-first** — no sync wrappers around async operations.
- **Typed exceptions** — never bare `except`; never swallow errors silently.
- **No magic strings** — categorical values as enums/constants.
- Consume Engrava only through its **public API**; do not reach into its internals.

## Reporting Issues

- Use [GitHub Issues](https://github.com/sovantica/engrava-mcp/issues).
- Include Python version, OS, MCP client, and a minimal reproduction.
- For security vulnerabilities, email directly instead of filing a public issue.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.
