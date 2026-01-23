# Call Tool Optimizer Experiments

This directory contains experiments for testing the MCP Optimizer's `call_tool` functionality and response optimization.

## AppWorld Experiment

Run experiments against AppWorld tasks using MCP Optimizer tools (`find_tool`, `call_tool`, `search_in_tool_response`) with a Pydantic AI agent.

### Prerequisites

1. Install AppWorld data (runs in isolated environment via `uv run --no-project`):
   ```bash
   # AppWorld is run downdloading from source. PyPi version has issues running in 3.13
   task appworld-install
   ```

2. Set environment variable:
   ```bash
   export OPENROUTER_API_KEY=your_api_key
   ```

3. Start AppWorld servers in separate terminals:
   ```bash
   # Terminal 1: API server
   task appworld-serve-api

   # Terminal 2: MCP server
   task appworld-serve-mcp
   ```

### Running the Experiment

```bash
# Run new experiment (limited to 5 tasks)
task appworld-experiment -- --limit 5

# Run with custom settings
task appworld-experiment -- --model anthropic/claude-opus-4 --threshold 500 --verbose
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--experiment-name` | Name for experiment (auto-generated if not provided, auto-resumes matching config) | (auto) |
| `--dataset` | AppWorld dataset (train, dev, test_normal, test_challenge) | train |
| `--limit` | Limit number of tasks to run | all |
| `--model` | LLM model for the agent (OpenRouter format) | anthropic/claude-sonnet-4 |
| `--threshold` | Token threshold for response optimization | 1000 |
| `--head-lines` | Lines to preserve from start for unstructured text | 20 |
| `--tail-lines` | Lines to preserve from end for unstructured text | 20 |
| `--max-steps` | Maximum agent steps per task | 50 |
| `--appworld-mcp-url` | AppWorld MCP server URL | http://localhost:10000 |
| `--appworld-api-url` | AppWorld API server URL for remote_apis_url | http://localhost:9000 |
| `--state-file` | Path to state file | {experiment_name}_state.json |
| `--output` | Path to output results file | {experiment_name}_results.json |
| `--db-path` | Path to database file (shared across experiments) | experiments_shared.db |
| `--force` | Delete existing state file and start fresh (does not delete shared database) | False |
| `--baseline` | Run baseline agent using direct MCP (ignores optimizer-specific options) | False |
| `--verbose` | Enable verbose output (debug logging) | False |

### Output Files

- **State file** (`{name}_state.json`): Tracks progress for resume capability
- **Results file** (`{name}_results.json`): Aggregated experiment results
- **Database** (`experiments_shared.db`): MCP Optimizer database with ingested tools (shared across experiments)

### Experiment Flow

1. Check if AppWorld MCP server is running
2. Load or create experiment state
3. Ingest AppWorld tools to MCP Optimizer database (if not done)
4. For each AppWorld task:
   - Get task instruction from AppWorld
   - Run Pydantic AI agent with `find_tool`, `call_tool`, `search_in_tool_response`
   - Evaluate task completion using AppWorld's `world.evaluate()`
   - Save state after each task (enables resume)
5. Generate and save aggregated results

---

## Response Optimizer Experiment

Tests the response optimizer module that compresses tool responses while preserving task-relevant information.

### Overview

The response optimizer uses:
1. **Content Classification**: Detects JSON, Markdown, or unstructured text
2. **Structure-Aware Traversal**: Breadth-first traversal that preserves structure
3. **LLMLingua-2 Summarization**: Token-level compression using ONNX model
4. **Query Hints**: Instructions for retrieving original content

### Running the Experiment

```bash
# Run with default settings (threshold=1000 tokens)
uv run python examples/call_tool_optimizer/sample_responses.py

# Note: The original run_experiment.py has been replaced with the AppWorld experiment.
# The sample_responses.py file contains the test data for response optimization testing.
```

### Sample Responses

The experiment includes three sample responses in `sample_responses.py`:

1. **Large JSON API Response**: 50-item paginated API result with nested metadata
2. **Markdown Documentation**: Multi-section API documentation
3. **Unstructured Log Output**: Application log with startup, requests, and shutdown

### ONNX Model

The LLMLingua-2 summarizer requires an ONNX model. To download:

```bash
task download-models
```

If the model is not available, the optimizer falls back to simple truncation.
