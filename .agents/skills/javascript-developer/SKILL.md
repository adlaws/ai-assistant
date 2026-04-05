---
name: javascript-developer
description: Use this skill when implementing or refactoring JavaScript application logic in this repository, especially for modern JavaScript features, API design, and unit-test-backed changes in jerrycan/src and jerrycan/test.
---

# JavaScript Developer Skill

Use this skill for source-level JavaScript changes.

For coding and testing expectations, read `references/javascript-development-guidelines.md`. and `references/javascript-testing-guidelines.md`.

## Key Principles

### Avoid Monolithic Source Files

Do **not** allow any single source file to grow into a monolith.  When a file
accumulates multiple unrelated concerns, break it apart into smaller, focused
modules — each with a single clear responsibility.

Signs a file needs splitting:

- More than ~300 lines of non-template logic.
- Contains state, methods, or helpers that serve different features.
- You have to scroll past large blocks of code to find the piece you need.
- Multiple developers frequently touch the same file for unrelated changes.

Preferred approach:

- **Composables** (`use*.js`) — encapsulate a cohesive piece of reactive state
  and the methods that operate on it.  Each composable should be independently
  understandable and testable.
- **Helper / utility modules** (`helpers.js`, `formatters.js`, etc.) — pure
  functions with no framework dependencies.
- **Orchestrator / shell** (`app.js`) — wires composables together, manages
  lifecycle hooks, and holds the template.  This file should contain minimal
  business logic; it delegates to composables.

When adding new functionality, create a new composable or helper file *first*
rather than appending to an existing large file.
