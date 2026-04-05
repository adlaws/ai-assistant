---
applyTo: '**/test_*.py,**/*_test.py,**/tests/**/*.py'
description: 'Python testing conventions: structure, assertions, mocking, fixtures, and coverage guidance for AI-generated and AI-refactored tests.'
---

# Python Testing Guidelines

## Mission

Write deterministic, isolated, readable tests that verify behaviour — not
implementation details — and that fail with clear diagnostic messages.

## Framework and Tooling

* Use `pytest` as the primary test framework.
* Use `pytest-asyncio` for testing async code.
* Use `unittest.mock` (stdlib) for mocking; prefer `pytest-mock`'s
  `mocker` fixture for convenience.
* Use `pytest.fixture` for shared setup and teardown.
* Use `coverage` or `pytest-cov` to measure test coverage.

## Test File Layout

* Place tests in a `tests/` directory mirroring the source structure.
* Name test files `test_<module>.py`.
* Name test functions `test_<behaviour_under_test>`.
* Group related tests in classes prefixed with `Test` (e.g.
  `TestDockerService`).

Example:

```text
app/
    docker_service.py
    config_service.py
tests/
    test_docker_service.py
    test_config_service.py
    conftest.py
```

## Writing Tests

### Structure

Use the **Arrange–Act–Assert** pattern:

```python
def test_read_env_returns_key_values(tmp_path):
    # Arrange
    env_file = tmp_path / ".env"
    env_file.write_text('FOO="bar"\nBAZ=123\n')
    service = ConfigService(tmp_path)

    # Act
    result = service.read_env()

    # Assert
    assert result["FOO"] == "bar"
    assert result["BAZ"] == "123"
```

### Naming

* Test names should describe the expected behaviour, not the
  implementation:
    * Good: `test_compose_up_returns_success_on_zero_exit_code`
    * Poor: `test_compose_up` or `test_run_compose`

### Assertions

* Use plain `assert` statements — not `self.assertEqual()`.
* Assert one logical concept per test; multiple `assert` calls for the
  same concept are fine.
* Use `pytest.raises(ExceptionType)` for expected exceptions.
* Use `pytest.approx()` for floating-point comparisons.
* Use descriptive assertion messages when the failure reason is not
  obvious from the expression.

### Fixtures

* Define shared fixtures in `conftest.py`.
* Prefer function-scoped fixtures (the default) for test isolation.
* Use `tmp_path` for temporary file operations — never write to the
  real file system.
* Use `monkeypatch` for patching environment variables and attributes.
* Yield fixtures for setup/teardown patterns:

```python
@pytest.fixture
def mock_docker_client(mocker):
    client = mocker.MagicMock()
    client.containers.list.return_value = []
    yield client
```

### Mocking

* Mock at the boundary, not in the middle — mock I/O, not logic.
* Prefer `mocker.patch()` over `unittest.mock.patch()` for cleaner
  syntax.
* Use `spec=True` or `autospec=True` to catch interface mismatches.
* Avoid over-mocking; if a test needs more than 3 mocks, consider
  restructuring the code.
* Verify call arguments with `assert_called_once_with()` or
  `call_args` when the interaction matters.

### Async Tests

* Mark async test functions with `@pytest.mark.asyncio`.
* Use `AsyncMock` for mocking async functions.
* Test both success and error paths for async operations.

```python
@pytest.mark.asyncio
async def test_compose_up_handles_failure(mock_service):
    mock_service._run_compose = AsyncMock(
        return_value={"returncode": 1, "stdout": "", "stderr": "error"}
    )
    result = await mock_service.compose_up()
    assert result["returncode"] == 1
```

### Parametrize

Use `@pytest.mark.parametrize` for testing multiple inputs:

```python
@pytest.mark.parametrize("status,expected_class", [
    ("running", "docker_bench"),
    ("exited", "docker_bench"),
])
def test_classify_container(status, expected_class):
    ...
```

## What to Test

* Normal (happy-path) behaviour.
* At least one error or edge case per function.
* Boundary conditions (empty inputs, missing files, None values).
* Error handling paths (exceptions raised, error messages).
* Input validation at public API boundaries.

## What Not to Test

* Third-party library internals (trust the library, mock the boundary).
* Private/internal methods directly — test them through the public API.
* Exact log message text (test that logging occurs, not the wording).

## Test Quality Checklist

* Tests are deterministic — no randomness, no reliance on system time,
  no network calls.
* Tests are independent — can run in any order.
* Tests are fast — mock slow I/O.
* No test modifies shared state (files, environment, global variables)
  without cleanup.
* Each test has a clear name describing the expected behaviour.
* Failed tests produce diagnostic output sufficient to locate the bug
  without re-running with a debugger.

## Coverage Expectations

* Aim for high coverage on business logic and service layers.
* Do not chase 100% coverage on boilerplate, entry points, or
  configuration wiring.
* Treat uncovered error-handling paths as a test gap worth filling.
