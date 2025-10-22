# Research: Token Savings Metrics

**Feature**: Token Savings Metrics for find_tool
**Phase**: Phase 0 - Research & Design Decisions
**Date**: 2025-10-20

## Overview

This document captures research findings and design decisions for implementing token savings metrics in the find_tool endpoint.

## Research Areas

### 1. tiktoken Library and Encoding Selection

**Decision**: Use `tiktoken` with `cl100k_base` encoding

**Rationale**:
- `tiktoken` is the official OpenAI tokenizer library, providing accurate token counting for GPT models
- `cl100k_base` is the encoding used by GPT-4, GPT-3.5-turbo, and text-embedding-ada-002
- This encoding provides the most accurate token counts for the majority of LLM use cases
- Alternative encodings considered:
  - `p50k_base` (used by older GPT-3 models) - less relevant for current LLMs
  - `r50k_base` (used by Code models) - too specialized
  - `o200k_base` (used by GPT-4o) - newer but cl100k_base is more widely compatible

**Implementation**:
```python
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
token_count = len(encoding.encode(text))
```

**References**:
- tiktoken documentation: https://github.com/openai/tiktoken
- Token counting best practices from OpenAI

### 2. Tool Serialization for Token Counting

**Decision**: Use `details.model_dump_json()` output from McpTool for token counting

**Rationale**:
- The `details` field in the tool table stores the complete JSON serialized McpTool
- This is the same representation sent to LLMs when tools are listed
- Counting tokens on the JSON representation provides accurate savings metrics
- Alternative approaches considered:
  - Count only name + description - underestimates actual token usage
  - Count inputSchema separately - adds complexity without benefit
  - Use custom tool representation - diverges from actual LLM consumption

**Implementation**:
- Calculate tokens from `details.model_dump_json()` during ingestion
- Store the integer count in new `token_count` column
- At query time, sum token counts instead of recalculating

### 3. Response Format for Token Savings

**Decision**: Extend find_tool response with custom response wrapper containing token metrics

**Rationale**:
- MCP `ListToolsResult` has a fixed structure (tools: list[Tool])
- Cannot modify the MCP protocol types directly without breaking compatibility
- Best approach: Return custom response with both tools and token metrics
- This is a breaking change but necessary to meet spec requirements

**Implementation**:
```python
return {
    "tools": matching_tools,
    "token_metrics": {
        "baseline_tokens": total_tokens,
        "returned_tokens": filtered_tokens,
        "tokens_saved": total_tokens - filtered_tokens,
        "savings_percentage": ((total_tokens - filtered_tokens) / total_tokens) * 100 if total_tokens > 0 else 0
    }
}
```

**Alternative Considered**:
- Log-only approach (non-breaking, limited visibility): Log token savings to structured logs, but clients can't access metrics programmatically. Rejected because spec requires the metric in the response.

### 4. Performance Considerations

**Decision**: Calculate tokens once during ingestion, aggregate at query time

**Rationale**:
- Tool ingestion happens once per server start (database is ephemeral)
- Typical deployment: 50-200 tools total
- Token calculation with tiktoken for 200 tools takes ~10-50ms total
- Query-time aggregation (simple SUM) takes <1ms
- This is far better than calculating tokens for all tools on every query

**Performance Targets**:
- Ingestion overhead: <50ms for 200 tools (acceptable for startup)
- Query-time overhead: <10ms for aggregation
- Memory overhead: 4 bytes per tool (INTEGER column)

**Scaling Analysis**:
- At 1000 tools: ~50-250ms ingestion overhead (still acceptable)
- At 10,000 tools: May need optimization (batch token calculation)
- Current scale (50-200 tools): No optimization needed

### 5. Database Schema Changes

**Decision**: Add INTEGER column `token_count` to `tool` table, make it NOT NULL with default 0

**Rationale**:
- INTEGER is sufficient for token counts (max ~8000 for typical tools, supports up to 2^31-1)
- NOT NULL constraint ensures data integrity
- Default 0 allows graceful handling of calculation failures
- No index needed (only used for SUM aggregation, not filtering)

**Schema Change**:
Since database is ephemeral and recreated on each server start, update the CREATE TABLE statement in migrations file:
```sql
CREATE TABLE tool (
    id TEXT PRIMARY KEY,
    mcpserver_id TEXT NOT NULL,
    details TEXT NOT NULL,
    details_embedding BLOB NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,  -- NEW
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (mcpserver_id) REFERENCES mcpserver (id) ON DELETE CASCADE
)
```

## Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tokenization library | tiktoken with cl100k_base | Official OpenAI library, GPT-4 compatible |
| Calculation timing | During ingestion | Performance optimization, calculate once |
| Storage location | INTEGER column in tool table | Minimal storage, fast aggregation |
| Response format | Custom response with token_metrics | Required by spec, document as breaking change |
| Serialization format | details.model_dump_json() | Matches actual LLM consumption |

## Open Questions

None - all design decisions have been made based on requirements and technical constraints.

## Next Steps

Proceed to Phase 1:
1. Create data-model.md with updated Tool entity
2. Define API contracts for extended find_tool response
3. Create quickstart.md for development guide
4. Update agent context with tiktoken dependency
