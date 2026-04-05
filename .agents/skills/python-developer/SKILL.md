```skill
---
name: python-developer
description: 'Use this skill when implementing or refactoring Python application logic in this repository, especially for modern Python features, API design, type annotations, async patterns, and test-backed changes. Covers FastAPI services, CLI tools, Docker orchestration scripts, and general Python best practices.'
---

# Python Developer Skill

Use this skill for source-level Python changes in this repository.

## When to Use This Skill

* User asks to implement or refactor Python code
* User asks to add type annotations or improve type safety
* User asks about Python packaging, virtual environments, or dependency management
* User asks to create or modify FastAPI services, CLI tools, or automation scripts
* User asks to write or improve Python tests
* User asks about async patterns, error handling, or logging in Python

## Scope and Context

* This repository contains Python 3.9+ code including FastAPI web services,
  shell-wrapping automation scripts, and data-processing utilities.
* Prefer modern Python idioms (3.9+) while respecting the runtime version
  available in the project's environment.
* Keep changes consistent with existing project structure and style.

## Core Guidance

For full coding, testing, and documentation expectations, read:

* `references/python-development-guidelines.md` — implementation standards
* `references/python-testing-guidelines.md` — testing conventions

## Pre-Flight Checklist

* Confirm the target Python version before using syntax features (e.g.
  `match`/`case` requires 3.10+, `type` aliases require 3.12+).
* Check that new dependencies are added to the correct `requirements.txt`
  or `pyproject.toml`.
* Run the existing test suite before and after changes.
* Verify type annotations pass `mypy` or `pyright` without new errors.
* Ensure docstrings exist on all public functions, classes, and modules.
* Ensure docstrings use reST/Sphinx fields where applicable:
  * `:param arg_name: description`
  * `:return: description`
  * `:raises SomeException: description`

## Response Style

* Provide practical, working Python examples.
* Include clear file placement and architectural rationale.
* Call out trade-offs and compatibility constraints where relevant.
* Keep solutions minimal before introducing advanced abstractions.

## Related Skills

* For Markdown documentation formatting, see the `markdown-formatter` skill.
* For JavaScript interop or frontend work, see the `javascript-developer`
  and `vuejs-developer` skills.
```
