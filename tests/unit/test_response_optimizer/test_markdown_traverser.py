"""Tests for Markdown traverser."""

import pytest

from mcp_optimizer.response_optimizer.traversers.markdown_traverser import (
    MarkdownTraverser,
    Section,
)


class TestMarkdownTraverser:
    """Test MarkdownTraverser class."""

    @pytest.fixture
    def traverser(self, simple_token_counter):
        """Create a MarkdownTraverser with simple token counter."""
        return MarkdownTraverser(simple_token_counter)

    @pytest.mark.asyncio
    async def test_traverse_content_within_budget(self, traverser, mock_summarizer):
        """Test that content within budget is returned as-is."""
        content = "# Title\n\nSome content."
        result = await traverser.traverse(content, max_tokens=1000, summarizer=mock_summarizer)

        assert result.content == content
        assert result.sections_summarized == 0

    @pytest.mark.asyncio
    async def test_traverse_simple_document(self, traverser, mock_summarizer):
        """Test traversal of simple Markdown document."""
        content = """# Title

This is the introduction.

## Section 1

Content for section 1.

## Section 2

Content for section 2.
"""
        result = await traverser.traverse(content, max_tokens=1000, summarizer=mock_summarizer)

        assert "Title" in result.content
        assert result.original_tokens > 0

    @pytest.mark.asyncio
    async def test_traverse_header_hierarchy(self, traverser, mock_summarizer):
        """Test that header hierarchy is preserved."""
        content = """# H1 Title

## H2 Section

### H3 Subsection

Content here.
"""
        result = await traverser.traverse(content, max_tokens=1000, summarizer=mock_summarizer)

        assert "H1 Title" in result.content
        assert "H2 Section" in result.content
        assert "H3 Subsection" in result.content

    @pytest.mark.asyncio
    async def test_traverse_builds_toc(self, traverser, mock_summarizer):
        """Test that table of contents is built."""
        content = """# First
## First A
## First B
# Second
## Second A
"""
        result = await traverser.traverse(content, max_tokens=100, summarizer=mock_summarizer)

        # TOC should include section titles
        assert "First" in result.content
        assert "Second" in result.content

    @pytest.mark.asyncio
    async def test_traverse_section_truncation(self, traverser, mock_summarizer):
        """Test that sections exceeding budget are summarized."""
        long_content = "x" * 1000
        content = f"""# Title

{long_content}

## Section 2

Short content.
"""
        result = await traverser.traverse(content, max_tokens=100, summarizer=mock_summarizer)

        # Should have summarization marker
        assert "[...SUMMARIZED]" in result.content or "[SUMMARIZED" in result.content

    @pytest.mark.asyncio
    async def test_traverse_with_summarizer(self, traverser):
        """Test traversal with real LLMLingua summarizer."""
        from mcp_optimizer.response_optimizer.summarizers.llmlingua import LLMLinguaSummarizer

        summarizer = LLMLinguaSummarizer()
        long_content = "This is important content that should be preserved. " * 50
        content = f"""# Title

{long_content}
"""
        result = await traverser.traverse(content, max_tokens=100, summarizer=summarizer)

        # Should have compressed the content
        assert result.result_tokens < result.original_tokens

    @pytest.mark.asyncio
    async def test_traverse_extreme_budget_constraint(self, traverser, mock_summarizer):
        """Test traversal with very tight budget."""
        content = """# Section 1

Content for section 1.

# Section 2

Content for section 2.

# Section 3

Content for section 3.
"""
        result = await traverser.traverse(content, max_tokens=20, summarizer=mock_summarizer)

        # Should have summarized structure
        assert "[SUMMARIZED:" in result.content or "Section" in result.content

    @pytest.mark.asyncio
    async def test_traverse_content_before_first_header(self, traverser, mock_summarizer):
        """Test content appearing before the first header."""
        content = """Some intro text here.

# First Header

Header content.
"""
        result = await traverser.traverse(content, max_tokens=1000, summarizer=mock_summarizer)

        assert "intro text" in result.content or "First Header" in result.content

    @pytest.mark.asyncio
    async def test_traverse_empty_sections(self, traverser, mock_summarizer):
        """Test handling of empty sections."""
        content = """# Empty Section

# Another Section

Some content here.
"""
        result = await traverser.traverse(content, max_tokens=1000, summarizer=mock_summarizer)

        assert "Empty Section" in result.content
        assert "Another Section" in result.content

    @pytest.mark.asyncio
    async def test_traverse_deep_nesting(self, traverser, mock_summarizer):
        """Test deeply nested header levels."""
        content = """# Level 1

## Level 2

### Level 3

#### Level 4

##### Level 5

###### Level 6

Deep content.
"""
        result = await traverser.traverse(content, max_tokens=1000, summarizer=mock_summarizer)

        assert "Level 1" in result.content

    @pytest.mark.asyncio
    async def test_traverse_real_markdown_file_low_budget(
        self, traverser, markdown_test_content: str, mock_summarizer
    ):
        """Test traversal of real Markdown file with very low budget."""
        result = await traverser.traverse(
            markdown_test_content, max_tokens=50, summarizer=mock_summarizer
        )

        # Should produce non-empty content
        assert len(result.content) > 0

        # Should have significantly reduced size
        assert result.result_tokens < result.original_tokens

    @pytest.mark.asyncio
    async def test_traverse_real_markdown_file_default_budget(
        self, traverser, markdown_test_content: str, mock_summarizer
    ):
        """Test traversal of real Markdown file with default budget (1000 tokens)."""
        result = await traverser.traverse(
            markdown_test_content, max_tokens=1000, summarizer=mock_summarizer
        )

        # Should produce non-empty content
        assert len(result.content) > 0
        assert result.original_tokens > 0

    @pytest.mark.asyncio
    async def test_traverse_real_markdown_file_large_budget(
        self, traverser, markdown_test_content: str, mock_summarizer
    ):
        """Test traversal of real Markdown file with very large budget - all content returned."""
        result = await traverser.traverse(
            markdown_test_content, max_tokens=100000, summarizer=mock_summarizer
        )

        # Should return original content unchanged
        assert result.content == markdown_test_content
        assert result.sections_summarized == 0
        assert result.result_tokens == result.original_tokens

    @pytest.mark.asyncio
    async def test_traverse_preserves_code_blocks(self, traverser, mock_summarizer):
        """Test that code blocks in content are handled."""
        content = """# Code Example

```python
def hello():
    print("Hello, World!")
```

## Another Section

More content.
"""
        result = await traverser.traverse(content, max_tokens=1000, summarizer=mock_summarizer)

        # Code should be present
        assert "hello" in result.content.lower() or "Code" in result.content


class TestSection:
    """Test Section dataclass."""

    def test_section_creation(self):
        """Test creating a Section."""
        section = Section(level=1, title="Test Title", content="Test content")
        assert section.level == 1
        assert section.title == "Test Title"
        assert section.content == "Test content"
        assert section.children == []

    def test_section_with_children(self):
        """Test Section with children."""
        child = Section(level=2, title="Child")
        parent = Section(level=1, title="Parent", children=[child])

        assert len(parent.children) == 1
        assert parent.children[0].title == "Child"

    def test_section_defaults(self):
        """Test Section default values."""
        section = Section(level=1, title="Test")
        assert section.content == ""
        assert section.children == []
