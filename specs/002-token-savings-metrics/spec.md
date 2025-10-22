# Feature Specification: Token Savings Metrics for find_tool

**Feature Branch**: `002-token-savings-metrics`
**Created**: 2025-10-20
**Status**: Draft
**Input**: User description: "The mcp-optimizer server returns a subset of tools that the input tool_description instead of returning all tools of all running mcp servers. This reduces tokens sent to LLM. Modify find_tool call to return the token savings as the difference between tokens across all 'running' MCP server tools and the tokens of the subset of tools returned by find_tool tool."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - View Token Savings from Tool Filtering (Priority: P1)

When an AI system or developer calls the find_tool endpoint, they can see how many tokens were saved by returning only relevant tools instead of all available tools. This helps them understand the efficiency gains from using the mcp-optimizer server's intelligent tool filtering.

**Why this priority**: This is the core value proposition - demonstrating the token efficiency that mcp-optimizer provides. Without this metric, users cannot quantify the benefit of using mcp-optimizer over listing all tools.

**Independent Test**: Can be fully tested by calling find_tool with any query and verifying the response includes token savings information. Delivers immediate value by showing optimization impact.

**Acceptance Scenarios**:

1. **Given** mcp-optimizer has 3 running servers with 50 total tools, **When** a user calls find_tool with "search the web" that returns 5 matching tools, **Then** the response includes token savings showing the difference between tokens for 50 tools vs 5 tools
2. **Given** mcp-optimizer has no running servers, **When** a user calls find_tool, **Then** the response shows zero token savings
3. **Given** mcp-optimizer returns all available tools (no filtering occurs), **When** a user calls find_tool, **Then** the response shows zero token savings

---

### Edge Cases

- What happens when all tools match the query (no filtering occurs)?
  - Token savings should be zero or minimal, indicating no optimization benefit
- How does the system handle when no tools are available from running servers?
  - Should return zero token savings since baseline and filtered set are both empty
- What if tool serialization size varies significantly between tools?
  - Token calculation should accurately account for each tool's actual serialized size using tiktoken
- How are tokens counted for tools with complex schemas or long descriptions?
  - Use tiktoken with consistent encoding model for accurate LLM tokenization

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST calculate token count for each tool during ingestion using tiktoken on the tool details field
- **FR-002**: System MUST persist the token count in the tool database table
- **FR-003**: System MUST sum token counts for all tools from running MCP servers to establish baseline
- **FR-004**: System MUST sum token counts for the filtered subset of tools returned by find_tool
- **FR-005**: System MUST compute token savings as the difference between baseline tokens and filtered tokens
- **FR-006**: System MUST include token savings metric in the find_tool response payload
- **FR-007**: System MUST use tiktoken for tokenization to match LLM token counting
- **FR-008**: System MUST handle cases where no tools are available (return zero savings)
- **FR-009**: System MUST handle cases where all tools match the query (return zero or minimal savings)

### Key Entities *(include if feature involves data)*

- **Token Savings Metric**: Represents the efficiency gain from tool filtering
  - Baseline token count (all running server tools)
  - Filtered token count (returned tools)
  - Savings value (difference)
  - Percentage reduction (optional)

- **Tool Record**: Database representation of a tool with token count
  - Token count (calculated from details field during ingestion)
  - Tool details field (used as source for token calculation)
  - Associated with running MCP server status

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: find_tool responses include token savings information in 100% of successful calls
- **SC-002**: Token count calculation during ingestion adds minimal overhead to server startup
- **SC-003**: Token counting uses tiktoken and matches LLM tokenization
- **SC-004**: Users can identify token savings value from the find_tool response without additional processing
- **SC-005**: System demonstrates measurable token savings when returning fewer than 50% of available tools
- **SC-006**: Calculating token savings from stored counts adds less than 10ms to find_tool response time

## Assumptions

- Token counting will use tiktoken library with an appropriate encoding (e.g., cl100k_base for GPT-4/GPT-3.5-turbo)
- Token counts are calculated once during tool ingestion and stored in the database
- The database is ephemeral and recreated on each server start, so tools are re-ingested each time
- The tool details field contains the complete serialized tool information needed for token counting
- The token savings metric will be included in the existing ListToolsResult response structure or a wrapper/extension of it
- The metric is primarily for visibility and does not affect the actual tool filtering logic
- Clients consuming the find_tool API can handle additional fields in the response without breaking changes
