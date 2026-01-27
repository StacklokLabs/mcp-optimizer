"""Shared fixtures for response_optimizer tests."""

from pathlib import Path

import pytest


class MockSummarizer:
    """Mock summarizer for testing that simply truncates text."""

    async def summarize(self, text: str, target_tokens: int) -> str:
        """Truncate text to fit target token count."""
        # Rough estimate: 4 chars per token
        max_chars = target_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20] + " [...SUMMARIZED]"

    def is_available(self) -> bool:
        """Mock summarizer is always available."""
        return True


@pytest.fixture
def mock_summarizer() -> MockSummarizer:
    """Return a mock summarizer for testing."""
    return MockSummarizer()


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent.parent.parent / "summarize_data"


@pytest.fixture
def json_test_content(test_data_dir: Path) -> str:
    """Load JSON test content from file."""
    return (test_data_dir / "json_gh_output.json").read_text()


@pytest.fixture
def markdown_test_content(test_data_dir: Path) -> str:
    """Load Markdown test content from file."""
    return (test_data_dir / "markdown_gh_output.md").read_text()


@pytest.fixture
def text_test_content(test_data_dir: Path) -> str:
    """Load plain text test content from file."""
    return (test_data_dir / "txt_output.txt").read_text()


@pytest.fixture
def simple_token_counter():
    """Return a simple character-based token estimator for testing."""

    def count(text: str) -> int:
        return len(text) // 4

    return count
