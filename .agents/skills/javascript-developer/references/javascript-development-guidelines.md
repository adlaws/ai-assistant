---
applyTo: '**/*.js,**/*.mjs,**/*.cjs'
description: 'JavaScript implementation standards: style, correctness, testing, and maintainability guidance for AI code generation and refactoring.'
---

# JavaScript Development Guidelines

## Mission

Implement or refactor JavaScript code that is safe, testable, readable and maintainable while
using modern JS features intentionally.

## Rules

* Target ES2020+ implementations unless compatibility constraints require otherwise.
* Prefer `const` over `let`; never use `var`.
* Use strict equality (`===` / `!==`) exclusively.
* Prefer arrow functions for callbacks and short expressions; use named function declarations for top-level functions.
* Prefer template literals over string concatenation.
* Use destructuring for objects and arrays where it improves clarity.
* Prefer `async`/`await` over raw Promises and callbacks.
* Use optional chaining (`?.`) and nullish coalescing (`??`) instead of manual null checks.
* Prefer `Map` and `Set` over plain objects when the use case calls for keyed collections or uniqueness.
* Use standard library methods (`Array.prototype.map`, `.filter`, `.reduce`, `.find`, etc.) over manual loops where intent is clearer.
* Avoid mutating function arguments; return new values instead.

## Naming Conventions

* Use `PascalCase` for classes and constructor functions.
* Use `camelCase` for variables, functions, and method names.
* Use `UPPER_SNAKE_CASE` for module-level constants.
* Use descriptive, intent-revealing names; avoid single-letter variables outside short loops.

## Architecture Guidelines

* Keep modules focused on a single responsibility.
* **Avoid monolithic source files.** No single file should accumulate multiple
  unrelated concerns.  When a file exceeds ~300 lines of non-template logic,
  split it into focused modules (composables, helpers, or services).
* When adding new functionality, create a new focused module *first* rather
  than appending to an existing large file.
* In Vue applications, extract cohesive groups of reactive state and methods
  into composable functions (`use*.js`) so the main app file remains a thin
  orchestrator.
* Move pure utility functions (formatting, escaping, text transforms) into
  shared helper files rather than embedding them in component logic.
* Prefer named exports over default exports for discoverability and refactoring safety.
* Avoid circular dependencies between modules.
* Prefer pure functions where possible to reduce side effects and improve testability.
* Separate I/O (file system, network, database) from business logic.
* Prefer dependency injection over hard-coded imports for external services.

## General Guidelines

* Keep functions focused and concise; prefer small, single-purpose functions.
* Avoid deep nesting; use early returns and guard clauses.
* Handle errors explicitly; avoid swallowing exceptions with empty `catch` blocks.
* Use meaningful error messages that include context about what failed and why.
* Prefer `structuredClone()` or spread syntax for cloning objects over `JSON.parse(JSON.stringify())`.
* Use `Number.isFinite()`, `Number.isNaN()`, and `Number.isInteger()` instead of the global equivalents.

## Function and Method Documentation

All exported functions and methods should have clear JSDoc comments describing their purpose,
parameters, return values, and any exceptions.

For example:

```javascript
/**
 * A short description of what the function does.
 *
 * @param {string} name - Description of the parameter.
 * @param {number} [timeout=5000] - Optional parameter with default.
 * @returns {Promise<Result>} Description of the return value.
 * @throws {TypeError} If name is not a string.
 */
```

## Error Handling

* Use domain-relevant error types or error codes rather than generic errors.
* Validate inputs at module boundaries.
* Always handle Promise rejections; avoid unhandled rejection warnings.
* Prefer explicit error returns or thrown errors over silent failure.
* Log errors with structured context (e.g., operation name, relevant IDs).

## Testing Expectations

* Add or update tests for each behaviour change.
* Cover the normal flow and at least one failure or edge case.
* Keep tests deterministic and independent from external systems.
* Use mocks or stubs for I/O boundaries (network, file system, timers).

## Definition of Done

* Code runs without errors in the target runtime.
* Unit tests pass.
* Exported functions have JSDoc comments.
* No generated or build output files are committed.

```
