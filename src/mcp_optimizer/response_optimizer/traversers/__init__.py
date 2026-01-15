"""Traversers for structure-aware content compression."""

from mcp_optimizer.response_optimizer.traversers.base import BaseTraverser
from mcp_optimizer.response_optimizer.traversers.json_traverser import JsonTraverser
from mcp_optimizer.response_optimizer.traversers.markdown_traverser import MarkdownTraverser
from mcp_optimizer.response_optimizer.traversers.text_traverser import TextTraverser

__all__ = [
    "BaseTraverser",
    "JsonTraverser",
    "MarkdownTraverser",
    "TextTraverser",
]
