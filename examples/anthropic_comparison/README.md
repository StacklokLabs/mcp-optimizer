# Anthropic Comparison Example

This example compares MCP Optimizer's semantic tool search against Anthropic's native tool search approaches (BM25, regex, and hybrid).

## Setup

### Download Required Data Files

The example requires test data files that are excluded from version control. Download them from the GitHub release:

```bash
./download_data.sh
```

This downloads:
- `mcp_tools_cleaned.json` (786K) - Tool definitions
- `mcp_tools_cleaned_tests_claude-sonnet-4.json` (1.2M) - Test cases
- `results.json` (13M) - Pre-generated comparison results
- `results_accuracy_comparison.png` (154K) - Results visualization

**Note:** If you want to regenerate results yourself, you only need the first two files. The script will skip files that already exist locally.

## Prerequisites

- `ANTHROPIC_API_KEY` environment variable set
- `OPENROUTER_API_KEY` environment variable set

## Running the Comparison

### 1. Ingest Test Data

First, load the test tools into the MCP Optimizer database:

```bash
uv run python ingest_test_data.py
```

This creates `mcp_optimizer_test.db` with all test tools and their embeddings.

### 2. Run Comparison

Execute the comparison across all test cases:

```bash
uv run python tool_search_comparison.py
```

Results are saved to `./results.json`.

## Options

The comparison script supports several options:

```bash
uv run python tool_search_comparison.py --help
```

Key options:
- `--limit N` - Limit to N test cases (useful for testing)
- `--max-concurrency N` - Set concurrent test executions (default: 10)
- `--resume` - Resume from existing results, retrying only failed tests
- `--output-file PATH` - Custom output file path

## Output

The comparison generates:
- JSON report with detailed metrics
- Console summary with aggregate statistics
- Per-test-case accuracy, token usage, and retrieval metrics
