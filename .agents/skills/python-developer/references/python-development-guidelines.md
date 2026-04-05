---
applyTo: '**/*.py'
description: 'Python implementation standards: style, correctness, type safety, testing, and maintainability guidance for AI code generation and refactoring.'
---

# Python Development Guidelines

## Mission

Implement or refactor Python code that is safe, typed, testable, readable, and
maintainable while using modern Python features intentionally.

## Language Version

* Target Python 3.9+ unless the project explicitly pins a newer minimum.
* Guard newer syntax behind version checks or comments when the deployment
  target is ambiguous (e.g. `match`/`case` needs 3.10+).

## Style Rules

* Follow [PEP 8](https://peps.python.org/pep-0008/) for naming and layout.
* Use 4-space indentation, no tabs.
* Limit lines to 120 characters (not 79) for readability on modern displays.
* Use trailing commas in multi-line collections and function signatures.
* Prefer double quotes for strings unless the string contains double quotes.
* Use f-strings over `str.format()` or `%`-formatting.
* Sort imports with `isort` conventions: stdlib, third-party, local — each
  group separated by a blank line.

## Naming Conventions

* `snake_case` for functions, methods, variables, and module names.
* `PascalCase` for classes and type aliases.
* `UPPER_SNAKE_CASE` for module-level constants.
* `_single_leading_underscore` for internal/private names.
* Use descriptive, intent-revealing names; avoid single-letter variables
  outside short loops or comprehensions.

## Type Annotations

* Add type annotations to **all** function and method signatures — annotate
  **every** parameter and the return type. Bare, untyped signatures
  (e.g. `def foo(x, y):`) are not acceptable.
* Use `from __future__ import annotations` at the top of every module to
  enable postponed evaluation and allow forward references.
* Prefer built-in generics (`list[str]`, `dict[str, int]`, `tuple[int, ...]`)
  over `typing.List`, `typing.Dict`, etc. (available from 3.9+).
* Use `Optional[X]` or `X | None` for nullable types.
* Use `TypeAlias`, `TypeVar`, and `Protocol` where they improve clarity.
* Annotate class attributes and instance variables, not just methods.
* Avoid `Any` unless interfacing with untyped third-party code; add a comment
  explaining why.

## Architecture Guidelines

* Keep modules focused on a single responsibility.
* Separate I/O (file system, network, database, Docker) from business logic.
* Prefer dependency injection over hard-coded imports for external services.
* Avoid circular imports between modules.
* Prefer pure functions where possible to reduce side effects and improve
  testability.
* Use dataclasses or Pydantic models for structured data instead of raw
  dicts.
* Prefer `pathlib.Path` over `os.path` for file system operations.

## Function and Method Design

* Keep functions focused and concise; prefer small, single-purpose functions.
* Avoid deep nesting; use early returns and guard clauses.
* Limit positional parameters to ~5; use keyword-only arguments (after `*`)
  for optional configuration.
* Avoid mutable default arguments (`def f(items=[])`); use `None` and
  assign inside the body.
* Return early on error conditions rather than wrapping the entire body in
  an `if`.

## Docstrings

All public functions, classes, and modules must have docstrings.

Use reStructuredText/Sphinx-style docstring fields for functions and methods:

```python
def fetch_logs(container_name: str, tail: int = 200) -> str:
    """Fetch the most recent logs from a Docker container.

  :param container_name: Name or ID of the container.
  :param tail: Number of trailing log lines to retrieve.
  :return: Log output with timestamps.
  :raises ContainerNotFoundError: If the container does not exist.
    """
```

* The first line is a concise imperative summary (not a sentence fragment).
* Separate the summary from the body with a blank line.
* **Every** docstring for a function or method **must** include:
  * A `:param arg_name: description` entry for **every** parameter.
  * A `:return: description` entry (omit only if the function returns `None`).
  * A `:raises ExceptionType: description` entry for every exception the
    function intentionally raises.
  * A `:yields: description` entry for generator functions.
* For classes, document `__init__` parameters in the class docstring or
  `__init__` docstring — not both.

## Error Handling

* Use specific exception types; avoid bare `except:` or `except Exception`.
* Create domain-relevant exception classes when built-in types are too
  generic.
* Validate inputs at module boundaries (API endpoints, CLI entry points,
  public functions).
* Always handle errors explicitly; avoid swallowing exceptions with empty
  `except` blocks.
* Log errors with structured context (operation name, relevant IDs, paths).
* Use `raise ... from e` to preserve exception chains.

## Logging

* Use the `logging` module, not `print()`, for operational output.
* Obtain loggers with `logging.getLogger(__name__)`.
* Use lazy formatting in log calls: `logger.info("Loaded %d items", count)`
  — not f-strings.
* Choose appropriate levels: `DEBUG` for internal detail, `INFO` for
  operational events, `WARNING` for recoverable problems, `ERROR` for
  failures.
* Include context in log messages (container name, file path, operation).

## Async / Concurrency

* Prefer `async`/`await` for I/O-bound concurrency (HTTP, Docker, file).
* Use `asyncio.create_subprocess_exec` over `subprocess.run` in async code.
* Bridge blocking calls to async with `asyncio.to_thread()` or
  `loop.run_in_executor()`.
* Avoid mixing `asyncio` and threading unless necessary; when combining
  them, use `asyncio.run_coroutine_threadsafe()` explicitly.
* Use `asyncio.TaskGroup` (3.11+) or `asyncio.gather()` for parallel async
  work.
* Always set timeouts on external calls to prevent indefinite hangs.

## Dependency Management

* Pin direct dependencies in `requirements.txt` with minimum version
  constraints (e.g. `fastapi>=0.104.0`).
* Use virtual environments (`venv` or `virtualenv`) — never install into
  the system Python.
* Keep `requirements.txt` sorted alphabetically.
* For applications (not libraries), consider pinning exact versions with a
  lock file for reproducible builds.

## FastAPI Specifics

* Use Pydantic models for all request and response bodies.
* Keep route functions thin — delegate to service classes for business logic.
* Use dependency injection (`Depends()`) for shared resources (DB sessions,
  service instances).
* Return proper HTTP status codes (201 for creation, 204 for no content,
  etc.).
* Add `response_model` to all routes for automatic documentation and
  validation.
* Use `APIRouter` with a prefix for logical grouping of endpoints.

## CLI / Script Specifics

* Use `argparse` or `click` for command-line interfaces.
* Provide `--help` text for all arguments.
* Return meaningful exit codes (0 for success, non-zero for failure).
* Write to `stdout` for normal output, `stderr` for errors and diagnostics.
* Support `--verbose` / `--quiet` flags where appropriate.

## Definition of Done

* Code runs without errors in the target runtime.
* Type annotations are present on all public APIs.
* `mypy` or `pyright` reports no new errors.
* `pylint` reports no new warnings or errors (see Pylint section below).
* Unit tests pass.
* Public functions and classes have docstrings.
* No generated, build output, or `.pyc` files are committed.

## Pylint

Run `pylint` on all changed Python files before considering work complete.

```bash
pylint <files>
```

### Rules to enforce

* **W0621 (redefined-outer-name)** — Do not shadow outer-scope names in
  function parameters or local variables. Wrap top-level logic in a
  `main()` function so that module-level names remain in a narrow scope.
* **C0411 (wrong-import-order)** — Follow the standard import order:
  `__future__`, stdlib, third-party, local. Use `isort` conventions.
* **C0103 (invalid-name)** — Use `UPPER_SNAKE_CASE` only for true
  module-level constants. Local variables and function-scoped values
  must use `snake_case`, even when they are semantically constant within
  that scope.
* **C0415 (import-outside-toplevel)** — Keep imports at the top of the
  module. Local imports are acceptable only when they break circular
  dependencies or defer expensive side effects; add a comment explaining
  why.
* **C0301 (line-too-long)** — Keep lines within 120 characters (the
  project limit). pylint defaults to 100; configure or override as
  needed.
* **R0914 / R0912 / R0915 (too-many-locals / branches / statements)** —
  Treat these as signals to refactor: extract helper functions,
  introduce early returns, or split into smaller methods.
* **R0801 (duplicate-code)** — Extract shared logic into a common utility
  or helper function rather than duplicating it across modules.

### Acceptable suppressions

Suppress individual pylint diagnostics inline only when the violation is
intentional and unavoidable. Always include a justification:

```python
host = os.environ.get(...)  # pylint: disable=redefined-outer-name  # shadow is intentional
```

Do **not** blanket-disable categories (e.g. `--disable=C,R`) in CI or
commit hooks.
