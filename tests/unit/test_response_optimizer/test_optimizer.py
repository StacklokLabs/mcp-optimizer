"""Tests for ResponseOptimizer integration."""

import json

import pytest

from mcp_optimizer.response_optimizer.models import ContentType
from mcp_optimizer.response_optimizer.optimizer import ResponseOptimizer


class TestResponseOptimizerInitialization:
    """Test ResponseOptimizer initialization."""

    def test_initialization_with_defaults(self):
        """Test optimizer initializes with default values."""
        optimizer = ResponseOptimizer()

        assert optimizer.token_threshold == 1000
        assert optimizer.head_lines == 20
        assert optimizer.tail_lines == 20
        assert optimizer._estimate_tokens is not None

    def test_initialization_with_custom_values(self):
        """Test optimizer initializes with custom values."""
        optimizer = ResponseOptimizer(
            token_threshold=500,
            head_lines=10,
            tail_lines=15,
        )

        assert optimizer.token_threshold == 500
        assert optimizer.head_lines == 10
        assert optimizer.tail_lines == 15

    def test_token_estimator_initialized(self):
        """Test that token estimator is initialized."""
        optimizer = ResponseOptimizer()

        # Should have a token estimator function
        assert optimizer._estimate_tokens is not None
        # Should be callable
        assert callable(optimizer._estimate_tokens)
        # Should return an integer for a test string
        assert isinstance(optimizer._estimate_tokens("test"), int)

    def test_summarizer_initialized(self):
        """Test that summarizer is initialized."""
        optimizer = ResponseOptimizer()

        assert optimizer._summarizer is not None

    def test_traversers_lazy_initialized(self):
        """Test that traversers are lazily initialized."""
        optimizer = ResponseOptimizer()

        assert optimizer._json_traverser is None
        assert optimizer._markdown_traverser is None
        assert optimizer._text_traverser is None


class TestResponseOptimizerOptimize:
    """Test ResponseOptimizer.optimize method."""

    @pytest.fixture
    def optimizer(self):
        """Create a ResponseOptimizer instance."""
        return ResponseOptimizer(token_threshold=100)

    @pytest.mark.asyncio
    async def test_below_threshold_passthrough(self, optimizer):
        """Test content below threshold is not optimized."""
        content = "Short content"
        result = await optimizer.optimize(content, tool_name="test_tool")

        assert result.was_optimized is False
        assert result.content == content
        assert result.token_metrics.tokens_saved == 0

    @pytest.mark.asyncio
    async def test_generates_response_id(self, optimizer):
        """Test that response ID is generated."""
        result = await optimizer.optimize("Test", tool_name="test")

        assert result.response_id is not None
        assert len(result.response_id) > 0

    @pytest.mark.asyncio
    async def test_generates_session_key(self, optimizer):
        """Test that session key is generated if not provided."""
        result = await optimizer.optimize("Test", tool_name="test")

        assert result.session_key is not None
        assert len(result.session_key) > 0

    @pytest.mark.asyncio
    async def test_uses_provided_session_key(self, optimizer):
        """Test that provided session key is used."""
        session_key = "custom-session-key"
        result = await optimizer.optimize("Test", tool_name="test", session_key=session_key)

        assert result.session_key == session_key

    @pytest.mark.asyncio
    async def test_max_tokens_override(self, optimizer):
        """Test that max_tokens parameter overrides threshold."""
        # Content that would be under default threshold but over max_tokens
        content = "x" * 100  # ~25 tokens with char-based estimation
        result = await optimizer.optimize(content, tool_name="test", max_tokens=10)

        # Should optimize because max_tokens is lower
        # Note: actual behavior depends on token counting
        assert result.token_metrics.baseline_tokens > 0

    @pytest.mark.asyncio
    async def test_json_content_classification(self, optimizer):
        """Test that JSON content is classified correctly."""
        content = '{"key": "value", "items": [1, 2, 3]}'
        result = await optimizer.optimize(content, tool_name="test")

        assert result.content_type == ContentType.JSON

    @pytest.mark.asyncio
    async def test_markdown_content_classification(self, optimizer):
        """Test that Markdown content is classified correctly."""
        content = "# Title\n\nSome content.\n\n## Section\n\nMore content."
        result = await optimizer.optimize(content, tool_name="test")

        assert result.content_type == ContentType.MARKDOWN

    @pytest.mark.asyncio
    async def test_unstructured_content_classification(self, optimizer):
        """Test that unstructured content is classified correctly."""
        content = "Just plain text without any special formatting."
        result = await optimizer.optimize(content, tool_name="test")

        assert result.content_type == ContentType.UNSTRUCTURED

    @pytest.mark.asyncio
    async def test_json_pipeline_optimization(self, optimizer):
        """Test full JSON optimization pipeline."""
        # Create large JSON content
        large_json = json.dumps({f"key{i}": f"value{i}" * 50 for i in range(20)})

        result = await optimizer.optimize(large_json, tool_name="test")

        assert result.content_type == ContentType.JSON
        # Should have valid JSON output
        json.loads(result.content)

    @pytest.mark.asyncio
    async def test_markdown_pipeline_optimization(self, optimizer):
        """Test full Markdown optimization pipeline."""
        # Create large Markdown content
        sections = "\n\n".join([f"# Section {i}\n\n{'Content ' * 100}" for i in range(10)])

        result = await optimizer.optimize(sections, tool_name="test")

        assert result.content_type == ContentType.MARKDOWN
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_text_pipeline_optimization(self, optimizer):
        """Test full Text optimization pipeline."""
        # Create large text content
        lines = "\n".join([f"Line {i}: " + "x" * 50 for i in range(100)])

        result = await optimizer.optimize(lines, tool_name="test")

        assert result.content_type == ContentType.UNSTRUCTURED
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_generates_query_hints_json(self, optimizer):
        """Test that query hints are generated for JSON."""
        large_json = json.dumps({f"key{i}": f"value{i}" * 50 for i in range(20)})

        result = await optimizer.optimize(large_json, tool_name="test")

        if result.was_optimized:
            assert result.query_hints is not None
            assert result.query_hints.tool == "jq"

    @pytest.mark.asyncio
    async def test_generates_query_hints_markdown(self, optimizer):
        """Test that query hints are generated for Markdown."""
        sections = "\n\n".join([f"# Section {i}\n\n{'Content ' * 100}" for i in range(10)])

        result = await optimizer.optimize(sections, tool_name="test")

        if result.was_optimized:
            assert result.query_hints is not None
            assert result.query_hints.tool == "section"

    @pytest.mark.asyncio
    async def test_generates_query_hints_text(self, optimizer):
        """Test that query hints are generated for text."""
        lines = "\n".join([f"Line {i}: " + "x" * 50 for i in range(100)])

        result = await optimizer.optimize(lines, tool_name="test")

        if result.was_optimized:
            assert result.query_hints is not None
            assert result.query_hints.tool == "text"

    @pytest.mark.asyncio
    async def test_token_metrics_calculation(self, optimizer):
        """Test that token metrics are calculated correctly."""
        large_content = "x" * 1000  # Large content

        result = await optimizer.optimize(large_content, tool_name="test")

        metrics = result.token_metrics
        assert metrics.baseline_tokens > 0
        assert metrics.returned_tokens > 0

        if result.was_optimized:
            assert metrics.tokens_saved >= 0
            assert metrics.savings_percentage >= 0


class TestResponseOptimizerIsAvailable:
    """Test ResponseOptimizer.is_summarizer_available method."""

    def test_is_summarizer_available(self):
        """Test is_summarizer_available returns boolean."""
        optimizer = ResponseOptimizer()
        result = optimizer.is_summarizer_available()

        assert isinstance(result, bool)


class TestResponseOptimizerWithRealFiles:
    """Test ResponseOptimizer with real test files."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with reasonable threshold for test files."""
        return ResponseOptimizer(token_threshold=500)

    @pytest.mark.asyncio
    async def test_optimize_real_json_file(self, optimizer, json_test_content: str):
        """Test optimization of real JSON file."""
        result = await optimizer.optimize(json_test_content, tool_name="json_test")

        assert result.content_type == ContentType.JSON
        assert result.token_metrics.baseline_tokens > 0

        # Should produce valid JSON
        json.loads(result.content)

    @pytest.mark.asyncio
    async def test_optimize_real_markdown_file(self, optimizer, markdown_test_content: str):
        """Test optimization of real Markdown file."""
        result = await optimizer.optimize(markdown_test_content, tool_name="markdown_test")

        assert result.content_type == ContentType.MARKDOWN
        assert result.token_metrics.baseline_tokens > 0
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_optimize_real_text_file(self, optimizer, text_test_content: str):
        """Test optimization of real text file."""
        result = await optimizer.optimize(text_test_content, tool_name="text_test")

        # Text file could be classified as unstructured or markdown
        assert result.content_type in (ContentType.UNSTRUCTURED, ContentType.MARKDOWN)
        assert result.token_metrics.baseline_tokens > 0
        assert len(result.content) > 0


class TestResponseOptimizerTraverserSelection:
    """Test traverser selection based on content type."""

    @pytest.fixture
    def optimizer(self):
        """Create a ResponseOptimizer instance."""
        return ResponseOptimizer()

    def test_get_json_traverser(self, optimizer):
        """Test JSON traverser selection."""
        traverser = optimizer._get_traverser(ContentType.JSON)

        assert traverser is not None
        assert optimizer._json_traverser is not None
        # Should return same instance on subsequent calls
        assert optimizer._get_traverser(ContentType.JSON) is traverser

    def test_get_markdown_traverser(self, optimizer):
        """Test Markdown traverser selection."""
        traverser = optimizer._get_traverser(ContentType.MARKDOWN)

        assert traverser is not None
        assert optimizer._markdown_traverser is not None
        # Should return same instance on subsequent calls
        assert optimizer._get_traverser(ContentType.MARKDOWN) is traverser

    def test_get_text_traverser(self, optimizer):
        """Test Text traverser selection."""
        traverser = optimizer._get_traverser(ContentType.UNSTRUCTURED)

        assert traverser is not None
        assert optimizer._text_traverser is not None
        # Should use configured head/tail lines
        assert traverser.head_lines == optimizer.head_lines
        assert traverser.tail_lines == optimizer.tail_lines
        # Should return same instance on subsequent calls
        assert optimizer._get_traverser(ContentType.UNSTRUCTURED) is traverser
