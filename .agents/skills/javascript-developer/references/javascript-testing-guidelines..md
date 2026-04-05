```instructions
---
applyTo: '**/*.test.js,**/*.spec.js,**/__tests__/**/*.js'
description: 'JavaScript testing standards: unit test structure, mocking strategy, and quality expectations for AI-generated tests.'
---

# JavaScript Testing Guidelines

## Mission

Write tests that are fast, deterministic, readable, and provide clear diagnostics when they fail.

## Test Framework Conventions

* Use the test framework already present in the project (e.g., Jest, Vitest, Mocha). Do not
  introduce a new framework without explicit approval.
* Organise tests in files named `<module>.test.js` or `<module>.spec.js` alongside the source,
  or in a `__tests__/` directory mirroring the source structure.

## Unit Test Rules

* Each test should verify one specific behaviour.
* Use descriptive test names that state the condition and expected outcome
  (e.g., `"returns null when input is empty"`).
* Prefer `describe` / `it` or `describe` / `test` blocks to group related cases.
* Keep test setup minimal; avoid sharing mutable state between tests.
* Use `beforeEach` for per-test setup and `afterEach` for teardown when needed.
* Never rely on test execution order.

## Assertions

* Use specific assertions (`toEqual`, `toStrictEqual`, `toContain`, `toThrow`) over generic
  truthy checks.
* When testing floating-point results, use `toBeCloseTo` with an appropriate precision.
* When testing async code, always `await` the result or return the promise; do not use
  callback-based `done()` unless the framework requires it.

## Mocking and Test Doubles

* Mock external boundaries (network, file system, timers, databases), not internal modules.
* Prefer dependency injection to make units testable without patching module internals.
* Reset or restore mocks in `afterEach` to avoid test pollution.
* Avoid over-mocking; if a test requires extensive mocking the design may need refactoring.

## Edge Cases and Error Paths

* Test at least one error or edge case per function (e.g., empty input, null, boundary values,
  invalid types).
* Verify that errors are thrown or rejected with the expected type and message.

## What Is Not a Unit Test

* Tests that depend on real network calls, databases, or file system state.
* Tests that use `setTimeout` / `setInterval` without fake timers.
* Tests that compare entire snapshots without diagnosing which behaviour changed.
* Tests with randomised inputs that lack a fixed seed and reproducibility.

Classify these as integration, E2E, or snapshot tests and maintain them in a separate suite.

## Quality Checklist

* [ ] Each test has a clear, descriptive name.
* [ ] Tests are deterministic and pass on every run.
* [ ] Failure messages identify the specific behaviour that regressed.
* [ ] Mocks are scoped to I/O boundaries and cleaned up after each test.
* [ ] No test depends on the outcome or side effects of another test.

```
