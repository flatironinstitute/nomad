import pytest

# See https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration-tests",
        action="store_true",
        default=False,
        help="Run integration tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as a (slow to run) integration test"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration-tests"):
        return
    skip_integration = pytest.mark.skip(
        reason="requires --run-integration-tests option to run"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
