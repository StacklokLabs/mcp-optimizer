"""Markdown traverser with breadth-first structure-aware compression."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import mistune

from mcp_optimizer.response_optimizer.models import TraversalResult
from mcp_optimizer.response_optimizer.traversers.base import BaseTraverser, Summarizer


@dataclass
class Section:
    """Represents a section in a Markdown document."""

    level: int
    title: str
    content: str = ""
    children: list["Section"] = field(default_factory=list)


class MarkdownTraverser(BaseTraverser):
    """
    Markdown traverser using breadth-first expansion.

    Algorithm:
    1. Extract document structure (all headers)
    2. Include header hierarchy first (H1 -> H2 -> H3)
    3. Add content under each header level-by-level
    4. Summarize sections exceeding budget
    """

    def __init__(self, token_estimator: Callable[[str], int]):
        super().__init__(token_estimator)
        self._md = mistune.create_markdown(renderer=None)

    async def traverse(
        self,
        content: str,
        max_tokens: int,
        summarizer: Summarizer | None = None,
    ) -> TraversalResult:
        """Traverse Markdown content using breadth-first expansion."""
        original_tokens = self.estimate_tokens(content)

        # If already within budget, return as-is
        if original_tokens <= max_tokens:
            return TraversalResult(
                content=content,
                original_tokens=original_tokens,
                result_tokens=original_tokens,
                sections_summarized=0,
            )

        # Parse into section tree
        sections = self._parse_sections(content)

        # Build output breadth-first
        result, sections_summarized = await self._build_output(sections, max_tokens, summarizer)

        result_tokens = self.estimate_tokens(result)

        return TraversalResult(
            content=result,
            original_tokens=original_tokens,
            result_tokens=result_tokens,
            sections_summarized=sections_summarized,
        )

    def _parse_sections(self, content: str) -> list[Section]:
        """Parse Markdown content into a section tree."""
        lines = content.split("\n")
        sections: list[Section] = []
        current_section: Section | None = None
        content_buffer: list[str] = []

        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            match = header_pattern.match(line)

            if match:
                # Save content from previous section
                if current_section is not None:
                    current_section.content = "\n".join(content_buffer).strip()
                elif content_buffer:
                    # Content before first header
                    sections.append(
                        Section(level=0, title="", content="\n".join(content_buffer).strip())
                    )
                content_buffer = []

                # Create new section
                level = len(match.group(1))
                title = match.group(2).strip()
                new_section = Section(level=level, title=title)

                # Find parent section
                if not sections or level == 1:
                    sections.append(new_section)
                else:
                    self._add_to_tree(sections, new_section, level)

                current_section = new_section
            else:
                content_buffer.append(line)

        # Save final section content
        if current_section is not None:
            current_section.content = "\n".join(content_buffer).strip()
        elif content_buffer:
            sections.append(Section(level=0, title="", content="\n".join(content_buffer).strip()))

        return sections

    def _add_to_tree(self, sections: list[Section], new_section: Section, level: int) -> None:
        """Add a section to the appropriate place in the tree."""
        # Find the last section at a higher level (lower number)
        for section in reversed(sections):
            if section.level < level:
                section.children.append(new_section)
                return
            # Check children recursively
            parent = self._find_parent(section, level)
            if parent:
                parent.children.append(new_section)
                return

        # No suitable parent found, add at root level
        sections.append(new_section)

    def _find_parent(self, section: Section, target_level: int) -> Section | None:
        """Find a suitable parent section for the target level."""
        if section.level < target_level:
            # Check if any child is a better parent
            for child in reversed(section.children):
                if child.level < target_level:
                    better_parent = self._find_parent(child, target_level)
                    return better_parent if better_parent else child
            return section
        return None

    async def _build_output(
        self,
        sections: list[Section],
        max_tokens: int,
        summarizer: Summarizer | None,
    ) -> tuple[str, int]:
        """Build output using breadth-first expansion."""
        sections_summarized = 0
        output_parts: list[str] = []
        remaining_budget = max_tokens

        # Phase 1: Include all headers (TOC-style)
        toc = self._build_toc(sections)
        toc_tokens = self.estimate_tokens(toc)

        if toc_tokens >= max_tokens:
            # Even TOC doesn't fit, summarize structure
            summary = f"[SUMMARIZED: Document with {self._count_sections(sections)} sections]"
            return summary, 1

        output_parts.append(toc)
        remaining_budget -= toc_tokens

        # Phase 2: Add content level by level
        # Start with content under H1, then H2, etc.
        for level in range(1, 7):
            if remaining_budget <= 0:
                break

            sections_at_level = self._get_sections_at_level(sections, level)
            if not sections_at_level:
                continue

            for section in sections_at_level:
                if remaining_budget <= 0:
                    break

                if section.content:
                    content_tokens = self.estimate_tokens(section.content)

                    if content_tokens <= remaining_budget:
                        # Add full content
                        section_output = self._format_section_with_content(section)
                        output_parts.append(section_output)
                        remaining_budget -= self.estimate_tokens(section_output)
                    elif summarizer:
                        # Summarize content
                        summarized = await summarizer.summarize(
                            section.content, remaining_budget // 2
                        )
                        section_output = self._format_section_with_summary(section, summarized)
                        output_parts.append(section_output)
                        remaining_budget -= self.estimate_tokens(section_output)
                        sections_summarized += 1
                    else:
                        # Truncate
                        truncated = self._truncate_content(section.content, remaining_budget // 2)
                        section_output = self._format_section_with_summary(
                            section, truncated + " [TRUNCATED]"
                        )
                        output_parts.append(section_output)
                        remaining_budget -= self.estimate_tokens(section_output)
                        sections_summarized += 1

        return "\n\n".join(output_parts), sections_summarized

    def _build_toc(self, sections: list[Section], prefix: str = "") -> str:
        """Build a table of contents from sections."""
        lines = []
        for section in sections:
            if section.title:
                indent = "  " * (section.level - 1) if section.level > 0 else ""
                lines.append(f"{indent}- {section.title}")
            for child in section.children:
                child_toc = self._build_toc([child])
                if child_toc:
                    lines.append(child_toc)
        return "\n".join(lines)

    def _count_sections(self, sections: list[Section]) -> int:
        """Count total number of sections."""
        count = len(sections)
        for section in sections:
            count += self._count_sections(section.children)
        return count

    def _get_sections_at_level(self, sections: list[Section], level: int) -> list[Section]:
        """Get all sections at a specific level."""
        result = []
        for section in sections:
            if section.level == level:
                result.append(section)
            result.extend(self._get_sections_at_level(section.children, level))
        return result

    def _format_section_with_content(self, section: Section) -> str:
        """Format a section with its full content."""
        header = "#" * section.level + " " + section.title if section.title else ""
        if header:
            return f"{header}\n\n{section.content}"
        return section.content

    def _format_section_with_summary(self, section: Section, summary: str) -> str:
        """Format a section with summarized content."""
        header = "#" * section.level + " " + section.title if section.title else ""
        if header:
            return f"{header}\n\n[SUMMARIZED]\n{summary}"
        return f"[SUMMARIZED]\n{summary}"

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within budget."""
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content
        return content[: max_chars - 20]
