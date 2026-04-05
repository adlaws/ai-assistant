````skill
---
name: python-reviewer
description: >
  Use this skill when performing a code review of Python source files in this
  repository.  It provides a structured, checklist-driven review workflow
  covering correctness, security, maintainability, performance, and testing,
  with additional Python-specific review points for type safety, Pythonic
  idiom usage, error handling, resource management, and modern language
  features.
---

# Python Code Reviewer

## When to Use

- Reviewing pull requests or diffs containing Python code.
- Auditing an existing Python codebase for defects, security gaps, or
  maintainability issues.
- Pre-merge quality gates where a structured checklist is needed.
- Post-refactor validation to ensure no regressions were introduced.

## Prerequisites

- Read and understand the project-specific coding guidelines in
  `.agents/skills/python-developer/references/python-development-guidelines.md`
  and
  `.agents/skills/python-developer/references/python-testing-guidelines.md`.
- Ensure the code runs and all tests pass **before** starting the review:
  ```bash
  python -m pytest --tb=short
  ```
- Confirm the target Python version for the project (e.g. 3.9+, 3.10+,
  3.12+) so that version-gated features are reviewed correctly.

## Review Workflow

1. **Scope** — Identify the files or diff under review.
2. **Lint & Test** — Confirm the code passes `pylint`, `mypy`/`pyright`, and
   the test suite.
3. **Checklist Pass** — Walk through each section below, annotating findings.
4. **Classify** — Assign a severity to each finding (Critical / Major / Minor
   / Informational).
5. **Report** — Present findings grouped by severity, referencing the file
   and line range where the issue occurs, a description of the problem, its
   likely consequences, and concrete advice on how to fix it.

---

## Part A — Language-Agnostic Review Checklist

These principles apply broadly to any imperative/OOP language.

### A1. Correctness

| # | Check | Details |
|---|-------|---------|
| A1.1 | **Logic errors** | Off-by-one, inverted conditions, short-circuit evaluation misuse, unreachable code, incorrect operator precedence. |
| A1.2 | **Boundary / edge cases** | Empty inputs, zero-length collections, `None` values, maximum-value integers, empty strings, missing dictionary keys. |
| A1.3 | **Error handling** | Every fallible operation must have an error path; errors must not be silently swallowed. Bare `except:` or `except Exception` without re-raise is almost always wrong. |
| A1.4 | **Return values** | Functions must return consistent types. Ensure `None` is not returned implicitly when a meaningful value is expected. Check that callers use the returned value. |
| A1.5 | **Preconditions / postconditions** | Verify callers satisfy documented preconditions; ensure functions deliver promised postconditions. |
| A1.6 | **Arithmetic** | Integer overflow (rare in Python, but relevant in `ctypes`/`struct` code), division by zero, floating-point precision issues, `math.inf` / `math.nan` handling. |

### A2. Security & Input Validation

| # | Check | Details |
|---|-------|---------|
| A2.1 | **Input validation** | All external input (files, network, CLI args, environment variables, user-supplied data) must be validated before use. |
| A2.2 | **Injection risks** | User-supplied data must never flow directly into `eval()`, `exec()`, `subprocess` shell commands (`shell=True`), SQL strings, or `format`/f-string template injection vectors. |
| A2.3 | **Deserialisation** | Never use `pickle.loads()` or `yaml.load()` (without `SafeLoader`) on untrusted data. Prefer `json` or schema-validated formats. |
| A2.4 | **Sensitive data** | Passwords, keys, tokens must not be logged, hard-coded, or stored in plain text. Use environment variables or secrets management. |
| A2.5 | **Path traversal** | File-path inputs must be canonicalised (`pathlib.Path.resolve()`) and restricted to an expected directory. |
| A2.6 | **Temporary files** | Use `tempfile` module with context managers; avoid predictable temporary file names. |

### A3. Maintainability & Readability

| # | Check | Details |
|---|-------|---------|
| A3.1 | **Naming** | Variables, functions, classes, and constants follow PEP 8 naming conventions: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants. Names are descriptive and intent-revealing. |
| A3.2 | **Function length** | Prefer short, single-responsibility functions. Flag functions exceeding ~40 lines for possible extraction. |
| A3.3 | **Cognitive complexity** | Deeply nested control flow (>3 levels) warrants refactoring. Prefer early returns and guard clauses. |
| A3.4 | **Code duplication** | Identical or near-identical blocks should be extracted into shared utilities or helper functions. |
| A3.5 | **Docstrings** | All public functions, classes, and modules must have docstrings. Every docstring for a function or method **must** include a `:param arg_name: description` entry for **every** parameter, a `:return: description` entry (unless the function returns `None`), and a `:raises ExceptionType: description` entry for every exception the function intentionally raises. Use reST/Sphinx-style fields. Docstrings explain *what* and *why*, not *how*. |
| A3.6 | **Magic values** | Literal numbers/strings should be replaced with named constants (`UPPER_SNAKE_CASE`). |
| A3.7 | **Dead code** | Unreachable branches, unused variables/imports, commented-out code should be removed. |
| A3.8 | **Consistent style** | Indentation, spacing, quote style, and import ordering follow the project conventions and PEP 8. |

### A4. Testing

| # | Check | Details |
|---|-------|---------|
| A4.1 | **Test coverage** | New/changed behaviour must have corresponding unit tests. |
| A4.2 | **Edge-case tests** | Tests cover empty, boundary, `None`, and error cases — not just the happy path. |
| A4.3 | **Test independence** | Tests must not depend on execution order or shared mutable state. |
| A4.4 | **Assertion quality** | Tests use specific assertions (`assert x == y`, `pytest.raises`, `pytest.approx`) rather than vague `assert result`. Test names describe the expected behaviour. |
| A4.5 | **Arrange–Act–Assert** | Tests follow the AAA pattern with clear separation of setup, action, and verification. |
| A4.6 | **Mocking discipline** | Mock at boundaries (I/O, network, filesystem), not internal logic. Use `spec=True` or `autospec=True` to catch interface mismatches. |

### A5. Performance (General)

| # | Check | Details |
|---|-------|---------|
| A5.1 | **Algorithmic complexity** | Verify that algorithms scale appropriately for expected data sizes. Flag O(n²) or worse when a better alternative exists. |
| A5.2 | **Unnecessary allocation** | Avoid building large intermediate lists when a generator or iterator would suffice. |
| A5.3 | **Unnecessary copies** | Avoid deep-copying objects or converting between types needlessly. |
| A5.4 | **Premature optimisation** | Flag micro-optimisations that harm readability without measured justification. |

---

## Part B — Python-Specific Review Checklist

These checks layer on top of Part A for Python code.

### B1. Pythonic Idioms & Style

| # | Check | Details |
|---|-------|---------|
| B1.1 | **Use built-in iteration** | Prefer `for item in iterable` over index-based `for i in range(len(x))`. Use `enumerate()` when both index and value are needed. Use `zip()` to iterate over parallel sequences. (PEP 20: "Beautiful is better than ugly.") |
| B1.2 | **Truthiness** | Use implicit truthiness (`if items:` not `if len(items) > 0:`). Use `is None` / `is not None` for `None` checks — never `== None`. |
| B1.3 | **Comprehensions & generators** | Prefer list/dict/set comprehensions over `map()`/`filter()` with lambda for simple transformations. Avoid overly complex comprehensions (multiple `for` clauses or deeply nested logic) — extract to a function instead. |
| B1.4 | **Unpacking** | Use tuple/sequence unpacking (`a, b = pair`) and starred unpacking (`first, *rest = items`) for clarity. |
| B1.5 | **String formatting** | Prefer f-strings for string interpolation. Use `%`-style formatting in `logging` calls only. Do not concatenate strings with `+` in loops (use `"".join()`). |
| B1.6 | **Default iterators** | Use `for key in dict:` not `for key in dict.keys():`. Use `for line in file:` not `for line in file.readlines()`. |
| B1.7 | **Ternary expressions** | Use `x if cond else y` for simple cases. Avoid nested ternaries. |
| B1.8 | **Context managers (`with`)** | Use `with` for all resource management (files, sockets, locks, database connections). Prefer `contextlib.contextmanager` for custom context managers. |
| B1.9 | **EAFP vs LBYL** | Prefer "Easier to Ask Forgiveness than Permission" (try/except) over "Look Before You Leap" (if/then check) for operations likely to succeed. Use LBYL when failure is common or the check is trivial. |
| B1.10 | **Avoid power features** | Avoid metaclasses, `__getattr__` hacks, bytecode manipulation, `exec()`/`eval()`, and other "magic" unless genuinely necessary. (PEP 20: "Simple is better than complex.") |

### B2. Type Annotations & Safety

| # | Check | Details |
|---|-------|---------|
| B2.1 | **Annotate all method signatures** | **Every** function and method signature must use typed arguments — annotate every parameter and the return type. This applies to all public APIs, and is strongly recommended for private/internal functions as well. Bare, untyped signatures (e.g. `def foo(x, y):`) are not acceptable. |
| B2.2 | **Use modern generics** | Prefer `list[str]`, `dict[str, int]`, `tuple[int, ...]` over `typing.List`, `typing.Dict`, etc. (Python 3.9+). Use `X \| None` over `Optional[X]` (Python 3.10+) when the version allows. |
| B2.3 | **`from __future__ import annotations`** | Use at the top of modules to enable postponed evaluation of annotations and allow forward references. |
| B2.4 | **Avoid `Any`** | Avoid `typing.Any` unless interfacing with untyped third-party code. Add a comment explaining why when used. |
| B2.5 | **`None` handling** | Explicitly annotate nullable types (`X \| None`). Never rely on implicit `Optional`. Validate `None` before use. |
| B2.6 | **Class and instance attributes** | Annotate class-level and instance-level attributes, not just method signatures. |
| B2.7 | **Type checker clean** | Code should pass `mypy --strict` or `pyright` without new errors. |

### B3. Error Handling & Exceptions

| # | Check | Details |
|---|-------|---------|
| B3.1 | **Specific exceptions** | Catch specific exception types. Never use bare `except:` or `except Exception:` without re-raising or a strong justification. |
| B3.2 | **Exception chaining** | Use `raise NewError(...) from original` to preserve tracebacks and context. |
| B3.3 | **Custom exceptions** | Define domain-specific exception classes inheriting from a relevant built-in exception. Name them with an `Error` suffix (e.g. `ConfigurationError`). |
| B3.4 | **Minimal `try` blocks** | Keep the body of `try` blocks as small as possible. Only wrap the specific operation that may fail. |
| B3.5 | **No silent swallowing** | Empty `except` blocks are forbidden. At minimum, log the error. Prefer to re-raise or handle concretely. |
| B3.6 | **`assert` misuse** | Do not use `assert` for input validation or control flow in production code. Assertions can be stripped with `python -O`. Use `if`/`raise ValueError` instead. |
| B3.7 | **`finally` / cleanup** | Use `finally` or context managers for cleanup that must always run. |

### B4. Resource Management

| # | Check | Details |
|---|-------|---------|
| B4.1 | **Files and sockets** | Always close files and sockets explicitly via `with` statements. Never rely on garbage collection / `__del__` for cleanup. |
| B4.2 | **`pathlib` over `os.path`** | Prefer `pathlib.Path` for filesystem operations over `os.path` string manipulation. |
| B4.3 | **Database connections** | Use connection pools and context managers. Close connections in `finally` or via `with`. |
| B4.4 | **Temporary resources** | Use `tempfile.NamedTemporaryFile` or `tempfile.TemporaryDirectory` with context managers. |

### B5. Mutable Default Arguments & Common Pitfalls

| # | Check | Details |
|---|-------|---------|
| B5.1 | **Mutable defaults** | Never use mutable objects (`[]`, `{}`, `set()`) as default argument values. Use `None` and assign inside the function body. |
| B5.2 | **Late binding closures** | Watch for closures capturing loop variables by reference. Use default argument binding (`lambda x=x: ...`) or `functools.partial` when needed. |
| B5.3 | **Global / mutable state** | Avoid mutable module-level variables. Prefer dependency injection or configuration objects. |
| B5.4 | **Circular imports** | Restructure code to avoid circular imports. Factor shared types into a separate module. |
| B5.5 | **Name shadowing** | Do not shadow built-in names (`list`, `dict`, `type`, `id`, `input`, `open`, etc.) or outer-scope variables with local names. |

### B6. Imports & Module Structure

| # | Check | Details |
|---|-------|---------|
| B6.1 | **Import ordering** | Follow `isort` conventions: `__future__`, stdlib, third-party, local — separated by blank lines. |
| B6.2 | **Absolute imports** | Prefer absolute imports over relative imports. |
| B6.3 | **Top-level imports** | Keep imports at the top of the module. Local imports are acceptable only to break circular dependencies or defer expensive side effects — add a comment explaining why. |
| B6.4 | **Import what you use** | Remove unused imports. Do not import symbols that are never referenced. |
| B6.5 | **No wildcard imports** | Never use `from module import *`. Be explicit about what you import. |
| B6.6 | **`__all__` for public APIs** | Modules intended as public APIs should define `__all__` to control exported names. |
| B6.7 | **`if __name__ == "__main__"`** | Executable scripts must guard top-level code behind `if __name__ == "__main__":` so the module can be safely imported. |

### B7. Logging & Observability

| # | Check | Details |
|---|-------|---------|
| B7.1 | **`logging` over `print`** | Use the `logging` module for operational output. Reserve `print()` for CLI user-facing output or debugging only. |
| B7.2 | **Logger naming** | Obtain loggers with `logging.getLogger(__name__)`. |
| B7.3 | **Lazy formatting** | Use `%`-style formatting in log calls (`logger.info("Loaded %d items", count)`) — not f-strings — to avoid formatting when the log level is disabled. |
| B7.4 | **Appropriate levels** | `DEBUG` for internals, `INFO` for operational events, `WARNING` for recoverable problems, `ERROR` for failures, `CRITICAL` for system-wide failures. |
| B7.5 | **Structured context** | Include relevant identifiers, paths, and operation names in log messages. |

### B8. Async / Concurrency (if applicable)

| # | Check | Details |
|---|-------|---------|
| B8.1 | **`async`/`await` for I/O** | Prefer `async`/`await` for I/O-bound concurrency. Do not mix blocking calls into the event loop without `asyncio.to_thread()` or `run_in_executor()`. |
| B8.2 | **Thread safety** | Shared mutable state between threads must be protected by locks (`threading.Lock`, `threading.RLock`). Use `queue.Queue` for inter-thread communication. |
| B8.3 | **Timeouts** | Always set timeouts on external calls (network, subprocess) to prevent indefinite hangs. |
| B8.4 | **Task management** | Use `asyncio.TaskGroup` (3.11+) or `asyncio.gather()` for parallel async work. Handle task cancellation gracefully. |
| B8.5 | **`asyncio.run()` lifecycle** | Ensure `asyncio.run()` is called once at the top level. Do not nest event loops. |

### B9. Data Modelling & API Design

| # | Check | Details |
|---|-------|---------|
| B9.1 | **Dataclasses / Pydantic** | Use `dataclasses` or Pydantic models for structured data instead of raw `dict`s or tuples. |
| B9.2 | **Immutability** | Prefer `frozen=True` dataclasses or `NamedTuple` for value objects that should not change after creation. |
| B9.3 | **Parameter passing** | Limit positional parameters to ~5. Use keyword-only arguments (after `*`) for optional configuration. |
| B9.4 | **Consistent return types** | Functions should return a consistent type. Avoid returning `None` on one path and a value on another unless the return type is explicitly `X \| None`. |
| B9.5 | **`__repr__` / `__str__`** | Classes should define `__repr__` for debugging. `__str__` for user-facing output when applicable. |
| B9.6 | **Enum over magic strings** | Use `enum.Enum` or `enum.StrEnum` (3.11+) instead of string constants for fixed sets of values. |

---

## Severity Classification

| Severity | Meaning | Action |
|----------|---------|--------|
| **Critical** | Security vulnerability, data loss, crash, exceptions in production, use of `eval`/`exec` on untrusted input, `pickle` on untrusted data. | Must fix before merge. |
| **Major** | Resource leak, logic error, missing error handling, bare `except`, mutable default argument, missing tests for new behaviour. | Should fix before merge. |
| **Minor** | Style violation, missing type annotation, suboptimal idiom, missing `const`/`Final`, naming inconsistency, missing docstring. | Fix when convenient. |
| **Informational** | Suggestion for improvement; no defect. Performance hint, Pythonic alternative, structural recommendation. | Optional. |

---

## Report Template

For each finding, include:
1. **File and line range** where the issue occurs.
2. **Check reference** (e.g. B3.1, A2.2) for traceability.
3. **Description** of the nature of the problem.
4. **Consequences** if not addressed.
5. **Recommendation** — a brief, constructive, practical, concrete summary of
   how to fix it.

```
## Code Review Summary

**Scope**: <files/areas reviewed>
**Lint**: pylint PASS / FAIL — mypy PASS / FAIL
**Tests**: <n> passed, <m> failed

### Critical

| File:Lines | Check | Description | Consequence | Recommendation |
|------------|-------|-------------|-------------|----------------|
| ...        | ...   | ...         | ...         | ...            |

### Major

| File:Lines | Check | Description | Consequence | Recommendation |
|------------|-------|-------------|-------------|----------------|
| ...        | ...   | ...         | ...         | ...            |

### Minor

| File:Lines | Check | Description | Consequence | Recommendation |
|------------|-------|-------------|-------------|----------------|
| ...        | ...   | ...         | ...         | ...            |

### Informational

| File:Lines | Check | Description | Consequence | Recommendation |
|------------|-------|-------------|-------------|----------------|
| ...        | ...   | ...         | ...         | ...            |
```

---

## References

- [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 20 — The Zen of Python](https://peps.python.org/pep-0020/)
- [PEP 257 — Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 — Type Hints](https://peps.python.org/pep-0484/)
- [PEP 526 — Syntax for Variable Annotations](https://peps.python.org/pep-0526/)
- [PEP 585 — Type Hinting Generics in Standard Collections](https://peps.python.org/pep-0585/)
- [PEP 604 — Allow writing union types as X | Y](https://peps.python.org/pep-0604/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [The Hitchhiker's Guide to Python — Code Style](https://docs.python-guide.org/writing/style/)
- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [Effective Python by Brett Slakin — Key Idioms](https://effectivepython.com/)
- Project guidelines: `.agents/skills/python-developer/references/python-development-guidelines.md`
- Project testing guidelines: `.agents/skills/python-developer/references/python-testing-guidelines.md`

````
