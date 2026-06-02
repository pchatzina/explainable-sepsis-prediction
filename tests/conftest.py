"""Shared pytest fixtures for database integration tests."""

import pytest
from src.utils.database import get_engine


@pytest.fixture(scope="module")
def db_engine():
    """Create a single DB engine for all tests in a module.

    Uses the project's centralized database configuration to ensure
    consistent connection settings across all test files.
    """
    engine = get_engine()
    yield engine
    engine.dispose()
