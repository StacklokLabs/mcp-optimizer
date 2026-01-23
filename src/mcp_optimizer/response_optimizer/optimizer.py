"""Main response optimizer that orchestrates the optimization pipeline."""

import uuid
from typing import Literal

import structlog

from mcp_optimizer.db.models import TokenMetrics
from mcp_optimizer.response_optimizer.classifier import classify_content
from mcp_optimizer.response_optimizer.hints import generate_query_hints
from mcp_optimizer.response_optimizer.models import ContentType, OptimizedResponse
from mcp_optimizer.response_optimizer.summarizers.base import BaseSummarizer
from mcp_optimizer.response_optimizer.summarizers.llmlingua import LLMLinguaSummarizer
from mcp_optimizer.response_optimizer.summarizers.truncation import TruncationSummarizer
from mcp_optimizer.response_optimizer.token_counter import TokenCounter, estimate_tokens
from mcp_optimizer.response_optimizer.traversers.base import BaseTraverser
from mcp_optimizer.response_optimizer.traversers.json_traverser import JsonTraverser
from mcp_optimizer.response_optimizer.traversers.markdown_traverser import MarkdownTraverser
from mcp_optimizer.response_optimizer.traversers.text_traverser import TextTraverser

logger = structlog.get_logger(__name__)

SummarizerMethod = Literal["llmlingua", "truncation"]


class ResponseOptimizer:
    """
    Main class for optimizing tool responses.

    Pipeline:
    1. Check token count against threshold
    2. Classify content type (JSON, Markdown, Unstructured)
    3. Apply appropriate traverser for structural compression
    4. Use summarizer for content that still exceeds budget
    5. Generate query hints for retrieving original content
    6. Return optimized response with metadata
    """

    def __init__(
        self,
        token_threshold: int = 1000,
        head_lines: int = 20,
        tail_lines: int = 20,
        token_counter: TokenCounter | None = None,
        summarizer_method: SummarizerMethod = "llmlingua",
    ):
        """
        Initialize the response optimizer.

        Args:
            token_threshold: Token count threshold for optimization
            head_lines: Lines to preserve from start for text content
            tail_lines: Lines to preserve from end for text content
            token_counter: Optional token counter for accurate counts
            summarizer_method: Method for summarization ("llmlingua" or "truncation").
                If "llmlingua" is selected but unavailable, falls back to "truncation".
        """
        self.token_threshold = token_threshold
        self.head_lines = head_lines
        self.tail_lines = tail_lines

        # Set up token estimation
        if token_counter:
            self._estimate_tokens = token_counter.count_tokens
        else:
            self._estimate_tokens = estimate_tokens

        # Initialize summarizer based on method with fallback
        self._summarizer = self._create_summarizer(summarizer_method)

        # Initialize traversers (lazy)
        self._json_traverser: JsonTraverser | None = None
        self._markdown_traverser: MarkdownTraverser | None = None
        self._text_traverser: TextTraverser | None = None

    def _create_summarizer(self, method: SummarizerMethod) -> BaseSummarizer:
        """Create the appropriate summarizer based on method with fallback.

        Args:
            method: The requested summarization method

        Returns:
            A summarizer instance (LLMLingua if available, otherwise Truncation)
        """
        if method == "truncation":
            logger.info("Using truncation summarizer as configured")
            return TruncationSummarizer()

        # Try to use LLMLingua
        llmlingua = LLMLinguaSummarizer()
        if llmlingua.is_available():
            logger.info("Using LLMLingua summarizer")
            return llmlingua

        # Fall back to truncation with warning
        logger.warning(
            "LLMLingua model not available, falling back to truncation summarizer. "
            "To use LLMLingua, ensure the ONNX model is installed at the configured path."
        )
        return TruncationSummarizer()

    def _get_traverser(self, content_type: ContentType) -> BaseTraverser:
        """Get the appropriate traverser for the content type."""
        if content_type == ContentType.JSON:
            if self._json_traverser is None:
                self._json_traverser = JsonTraverser(self._estimate_tokens)
            return self._json_traverser

        elif content_type == ContentType.MARKDOWN:
            if self._markdown_traverser is None:
                self._markdown_traverser = MarkdownTraverser(self._estimate_tokens)
            return self._markdown_traverser

        else:  # UNSTRUCTURED
            if self._text_traverser is None:
                self._text_traverser = TextTraverser(
                    self._estimate_tokens,
                    head_lines=self.head_lines,
                    tail_lines=self.tail_lines,
                )
            return self._text_traverser

    async def optimize(
        self,
        content: str,
        tool_name: str,
        session_key: str | None = None,
        max_tokens: int | None = None,
    ) -> OptimizedResponse:
        """
        Optimize a tool response for reduced token usage.

        Args:
            content: The tool response content to optimize
            tool_name: Name of the tool that generated the response
            session_key: Optional session key for grouping responses.
                        If not provided, a new UUID is generated.
            max_tokens: Optional override for token threshold

        Returns:
            OptimizedResponse with compressed content and metadata
        """
        # Generate IDs
        response_id = str(uuid.uuid4())
        if session_key is None:
            session_key = str(uuid.uuid4())

        # Calculate token count
        original_tokens = self._estimate_tokens(content)
        threshold = max_tokens or self.token_threshold

        # Check if optimization is needed
        if original_tokens <= threshold:
            logger.debug(
                "Content within threshold, no optimization needed",
                tool_name=tool_name,
                original_tokens=original_tokens,
                threshold=threshold,
            )
            # Classify content type for the response
            content_type = classify_content(content)
            # Return unoptimized response with actual content
            return OptimizedResponse(
                content=content,
                response_id=response_id,
                session_key=session_key,
                content_type=content_type,
                was_optimized=False,
                query_hints=None,
                token_metrics=TokenMetrics(
                    baseline_tokens=original_tokens,
                    returned_tokens=original_tokens,
                    tokens_saved=0,
                    savings_percentage=0.0,
                ),
            )

        # Classify content type
        content_type = classify_content(content)
        logger.info(
            "Optimizing tool response",
            tool_name=tool_name,
            content_type=content_type.value,
            original_tokens=original_tokens,
            threshold=threshold,
        )

        # Get appropriate traverser
        traverser = self._get_traverser(content_type)

        # Traverse and compress
        result = await traverser.traverse(
            content=content,
            max_tokens=threshold,
            summarizer=self._summarizer,
        )

        # Generate query hints
        query_hints = generate_query_hints(content, content_type, response_id)

        # Calculate final token count and metrics
        final_tokens = result.result_tokens
        tokens_saved = original_tokens - final_tokens
        compression_ratio = final_tokens / original_tokens if original_tokens > 0 else 1.0
        savings_percentage = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0.0

        logger.info(
            "Response optimization complete",
            tool_name=tool_name,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            compression_ratio=f"{compression_ratio:.1%}",
            sections_summarized=result.sections_summarized,
        )

        return OptimizedResponse(
            content=result.content,
            response_id=response_id,
            session_key=session_key,
            content_type=content_type,
            was_optimized=True,
            query_hints=query_hints,
            token_metrics=TokenMetrics(
                baseline_tokens=original_tokens,
                returned_tokens=final_tokens,
                tokens_saved=tokens_saved,
                savings_percentage=savings_percentage,
            ),
        )

    def is_summarizer_available(self) -> bool:
        """Check if the summarizer model is available."""
        return self._summarizer.is_available()
