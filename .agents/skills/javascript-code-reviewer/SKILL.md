````skill
---
name: javascript-code-reviewer
description: >
  Use this skill when performing a code review of JavaScript source files in
  this repository.  It provides a structured, checklist-driven review workflow
  that is mostly language-agnostic (correctness, security, maintainability,
  performance) with additional JavaScript-specific review points for DOM safety,
  XSS prevention, async correctness, modern ES2020+ idiom usage, and
  browser/runtime compatibility.
---

# JavaScript Code Reviewer

## When to Use

- Reviewing pull requests or diffs that include `.js`, `.mjs`, or `.cjs` files.
- Auditing an existing JavaScript codebase for defects, security gaps, or
  maintainability issues.
- Pre-merge quality gates where a structured checklist is needed.
- Post-refactor validation to ensure no regressions were introduced.

## Prerequisites

- Read and understand the project-specific coding guidelines in
  `.agents/skills/javascript-developer/references/javascript-development-guidelines.md`.
- Read the testing guidelines in
  `.agents/skills/javascript-developer/references/javascript-testing-guidelines..md`.
- Ensure the application builds and serves correctly before starting the review.

## Review Workflow

1. **Scope** – Identify the files or diff under review.
2. **Build & Serve** – Confirm the project builds and serves without errors.
3. **Checklist Pass** – Walk through each section below, annotating findings.
4. **Classify** – Assign a severity to each finding (Critical / Major / Minor /
   Informational).
5. **Report** – Present findings grouped by severity, linking to file and line.

---

## Part A — Language-Agnostic Review Checklist

These principles apply to any imperative/OOP language.

### A1. Correctness

| # | Check | Details |
|---|-------|---------|
| A1.1 | **Logic errors** | Off-by-one, inverted conditions, short-circuit evaluation misuse, unreachable code. |
| A1.2 | **Boundary / edge cases** | Empty inputs, zero-length arrays, `null`/`undefined`/`NaN` values, maximum-value numbers. |
| A1.3 | **Error handling** | Every fallible operation must have an error path; errors must not be silently swallowed. |
| A1.4 | **Return values** | Functions that return success/failure or computed values must have their return values used by callers. |
| A1.5 | **Preconditions / postconditions** | Verify callers satisfy documented preconditions; ensure functions deliver promised postconditions. |
| A1.6 | **Arithmetic** | Floating-point precision issues, `NaN` propagation, division by zero, integer overflow beyond `Number.MAX_SAFE_INTEGER`. |

### A2. Security & Input Validation

| # | Check | Details |
|---|-------|---------|
| A2.1 | **Input validation** | All external input (user input, URL parameters, WebSocket messages, file contents) must be validated before use. |
| A2.2 | **Bounds checking** | Array accesses with dynamic indices must be range-checked. |
| A2.3 | **Injection / XSS** | User-supplied data must never flow directly into `innerHTML`, `document.write()`, `eval()`, `new Function()`, template literals used as HTML, or `setTimeout`/`setInterval` with string arguments. Use `textContent` or proper sanitisation. |
| A2.4 | **Sensitive data** | Passwords, keys, tokens must not be logged, stored in `localStorage` in plain text, or embedded in client-side code. |
| A2.5 | **Prototype pollution** | Avoid `Object.assign` or spread on untrusted objects without validation. Do not use `__proto__`, `constructor.prototype` paths from user input. |
| A2.6 | **URL / path handling** | Use the `URL` constructor for URL parsing. Validate and sanitise URL origins before use. Avoid string concatenation for URL building. |

### A3. Maintainability & Readability

| # | Check | Details |
|---|-------|---------|
| A3.1 | **Naming** | Variables, functions, classes, and constants follow project naming conventions (`camelCase`, `PascalCase`, `UPPER_SNAKE_CASE`). Names are descriptive without being excessively long. |
| A3.2 | **Function length** | Prefer short, single-responsibility functions. Flag functions exceeding ~40 lines for possible extraction. |
| A3.3 | **Cognitive complexity** | Deeply nested control flow (>3 levels) warrants refactoring. Use early returns and guard clauses. |
| A3.4 | **Code duplication** | Identical or near-identical blocks should be extracted into shared utilities. |
| A3.5 | **Comments & docs** | Exported functions and methods should have JSDoc comments with `@param`, `@returns`, `@throws`. Inline comments explain *why*, not *what*. |
| A3.6 | **Magic values** | Literal numbers/strings should be replaced with named constants. |
| A3.7 | **Dead code** | Unreachable branches, unused variables, commented-out code should be removed. |
| A3.8 | **Consistent style** | Indentation, quoting, semicolons, brace style follow the project style consistently. |

### A4. Testing

| # | Check | Details |
|---|-------|---------|
| A4.1 | **Test coverage** | New/changed behaviour must have corresponding unit tests. |
| A4.2 | **Edge-case tests** | Tests cover empty, boundary, and error cases, not just the happy path. |
| A4.3 | **Test independence** | Tests must not depend on execution order or shared mutable state. |
| A4.4 | **Assertion quality** | Tests should use specific assertions (`toEqual`, `toStrictEqual`, `toContain`, `toThrow`) rather than generic truthy checks. |

### A5. Performance (General)

| # | Check | Details |
|---|-------|---------|
| A5.1 | **Algorithmic complexity** | Verify that algorithms scale appropriately for expected data sizes. Flag O(n²) or worse when a better alternative exists. |
| A5.2 | **Unnecessary allocation** | Avoid allocating in hot loops (creating objects/arrays/closures repeatedly). |
| A5.3 | **Unnecessary copies** | Avoid deep-cloning large data structures when a shallow copy or reference suffices. |
| A5.4 | **Premature optimisation** | Flag micro-optimisations that harm readability without measured justification. |

---

## Part B — JavaScript-Specific Review Checklist

These checks layer on top of Part A for JavaScript code.

### B1. Modern Language Usage (ES2020+)

| # | Check | Details |
|---|-------|---------|
| B1.1 | **`const` / `let` only** | Never use `var`. Prefer `const` over `let` when the binding is not reassigned. |
| B1.2 | **Strict equality** | Use `===` / `!==` exclusively. Never use `==` / `!=`. |
| B1.3 | **Template literals** | Prefer template literals over string concatenation for readability. |
| B1.4 | **Destructuring** | Use destructuring for objects and arrays where it improves clarity. |
| B1.5 | **Optional chaining / nullish coalescing** | Use `?.` and `??` instead of manual null/undefined checks and `||` for default values (which coerces falsy values). |
| B1.6 | **Arrow functions** | Use arrow functions for callbacks and short expressions. Use named function declarations for top-level functions. |
| B1.7 | **`for...of` / array methods** | Prefer `for...of` or `.map()`, `.filter()`, `.find()`, `.reduce()` over `for (let i = 0; ...)` when the index is not needed. |
| B1.8 | **Spread / rest** | Use spread syntax for shallow cloning and array merging instead of `Object.assign` or `Array.prototype.concat`. |
| B1.9 | **`Map` / `Set`** | Prefer `Map` and `Set` over plain objects when the use case is keyed collections or uniqueness, especially when keys are not strings. |

### B2. Async & Concurrency

| # | Check | Details |
|---|-------|---------|
| B2.1 | **`async`/`await`** | Prefer `async`/`await` over raw `.then()` chains and callbacks for readability. |
| B2.2 | **Unhandled rejections** | Every `Promise` must have rejection handling (`try`/`catch` around `await`, or `.catch()`). Avoid fire-and-forget promises without error handling. |
| B2.3 | **Race conditions** | Identify code that performs async reads followed by async writes to shared state; the state may change between read and write. |
| B2.4 | **`Promise.all` vs sequential** | Use `Promise.all` or `Promise.allSettled` for independent async operations instead of `await`-ing them sequentially. |
| B2.5 | **Timer cleanup** | `setTimeout` and `setInterval` return IDs that must be cleared (`clearTimeout` / `clearInterval`) when the component/context is disposed. |
| B2.6 | **String eval in timers** | Never pass strings to `setTimeout` / `setInterval`. Always pass function references. |

### B3. DOM & Browser Safety

| # | Check | Details |
|---|-------|---------|
| B3.1 | **XSS via `innerHTML`** | Never set `innerHTML`, `outerHTML`, or use `document.write()` with unsanitised data. Use `textContent`, `createElement`, or a sanitisation library. |
| B3.2 | **Event listener cleanup** | Event listeners added with `addEventListener` must be removed when the element or component is destroyed to prevent memory leaks. |
| B3.3 | **DOM query efficiency** | Cache DOM queries outside loops. Avoid querying the DOM inside `requestAnimationFrame` or tight loops. |
| B3.4 | **Layout thrashing** | Avoid interleaving DOM reads (e.g., `offsetHeight`) and writes that trigger forced reflows. Batch reads and writes separately. |
| B3.5 | **Defensive element access** | `document.querySelector` / `getElementById` may return `null`. Always check before accessing properties or methods. |
| B3.6 | **`data-*` attributes** | Use `dataset` API instead of `getAttribute`/`setAttribute` for custom data attributes. |

### B4. WebSocket & Network

| # | Check | Details |
|---|-------|---------|
| B4.1 | **Connection lifecycle** | WebSocket connections must handle `onopen`, `onerror`, and `onclose` events. Implement reconnection logic for transient failures. |
| B4.2 | **Message validation** | Always validate and parse incoming WebSocket/fetch messages with error handling (`try`/`catch` around `JSON.parse`). |
| B4.3 | **State synchronisation** | UI state derived from server messages must handle out-of-order delivery and stale data gracefully. |
| B4.4 | **Fetch error handling** | `fetch()` does not reject on HTTP error statuses (4xx/5xx). Always check `response.ok` or `response.status` before reading the body. |
| B4.5 | **Request cancellation** | Long-running or repeated fetch/XHR requests should use `AbortController` to cancel in-flight requests when they become irrelevant. |

### B5. Error Handling & Robustness

| # | Check | Details |
|---|-------|---------|
| B5.1 | **Catch specificity** | `catch` blocks should handle or rethrow errors meaningfully, not swallow them silently. Log with context (operation name, relevant IDs). |
| B5.2 | **Type coercion traps** | Watch for implicit type coercion (`+` with mixed types, comparisons with `null`/`undefined`/`NaN`, truthy/falsy checks on `0` or `""`). |
| B5.3 | **`typeof` / `instanceof` checks** | Validate types at module boundaries. Use `typeof`, `instanceof`, `Array.isArray()`, or `Number.isFinite()` instead of the global equivalents. |
| B5.4 | **`JSON.parse` safety** | Always wrap `JSON.parse` in `try`/`catch`. Validate the structure of the parsed result before using it. |
| B5.5 | **Global state** | Minimise mutable global state. Prefer module-scoped state with explicit accessors. Never pollute `window` with application variables. |
| B5.6 | **`this` binding** | Ensure `this` is correctly bound in methods used as callbacks. Prefer arrow functions or `.bind()` to avoid unexpected `this`. |

### B6. Module & Architecture

| # | Check | Details |
|---|-------|---------|
| B6.1 | **Single responsibility** | Each module/file should have a clear, focused purpose. |
| B6.2 | **Circular dependencies** | Detect and break circular import chains. |
| B6.3 | **Named exports** | Prefer named exports over default exports for discoverability and refactoring safety. |
| B6.4 | **Side effects on import** | Modules should not trigger side effects (DOM mutations, network calls) when imported. |
| B6.5 | **Argument mutation** | Functions should not mutate their arguments. Return new values instead. |
| B6.6 | **API surface** | Only export what is needed. Keep internal helper functions private to the module. |

---

## Severity Classification

| Severity | Meaning |
|----------|---------|
| **Critical** | XSS vulnerability, data exposure, crash, unhandled rejection causing data loss. Must fix before merge. |
| **Major** | Memory leak, logic error, missing error handling, unvalidated input, missing tests. Should fix before merge. |
| **Minor** | Style violation, suboptimal idiom, missing `const`, naming inconsistency, missing JSDoc. Fix when convenient. |
| **Informational** | Suggestion for improvement; no defect. Optional. |

---

## Report Template

```
## Code Review Summary

**Scope**: <files/areas reviewed>
**Build**: PASS / FAIL
**Tests**: <n> passed, <m> failed (or N/A)

### Critical

| File:Line | Check | Description |
|-----------|-------|-------------|
| ...       | ...   | ...         |

### Major

| File:Line | Check | Description |
|-----------|-------|-------------|
| ...       | ...   | ...         |

### Minor

| File:Line | Check | Description |
|-----------|-------|-------------|
| ...       | ...   | ...         |

### Informational

| File:Line | Check | Description |
|-----------|-------|-------------|
| ...       | ...   | ...         |
```

## References

- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [OWASP DOM-Based XSS Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/DOM_based_XSS_Prevention_Cheat_Sheet.html)
- [MDN JavaScript Guide](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide)
- [Google JavaScript Style Guide](https://google.github.io/styleguide/jsguide.html)
- [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- [Node.js Best Practices](https://github.com/goldbergyoni/nodebestpractices)
- Project guidelines: `.agents/skills/javascript-developer/references/javascript-development-guidelines.md`
- Project testing guidelines: `.agents/skills/javascript-developer/references/javascript-testing-guidelines..md`
````
