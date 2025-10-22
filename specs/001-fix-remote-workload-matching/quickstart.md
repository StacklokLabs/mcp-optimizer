# Quickstart: Fix Remote Workload Matching

**Feature**: 001-fix-remote-workload-matching
**Date**: 2025-10-20

## Overview

This quickstart guide helps you test and validate the remote workload matching fix. It covers setting up test scenarios, running the ingestion service, and verifying correct behavior.

## Prerequisites

- ToolHive running (Docker or Kubernetes mode)
- MCP-Optimizer development environment set up
- Access to ToolHive API (default: localhost:8080)

## Test Scenario 1: Remote Workload with Custom Name

**Goal**: Verify remote workload with custom name is correctly matched by URL

### Setup

1. Deploy a remote MCP workload with a custom name:

```bash
# Using ToolHive CLI
thv run --remote \
  --name "my-custom-github-server" \
  --url "https://api.github.com/mcp" \
  --transport sse
```

2. Ensure registry has an entry for the same URL:

```bash
# Check registry
thv registry list | grep github
# Should show entry with URL: https://api.github.com/mcp
```

### Expected Behavior

**Before Fix** (Current):
- Workload ingested with name "my-custom-github-server"
- Matching fails (name doesn't match registry name "github")
- Workload immediately deleted in cleanup phase
- Logs show: "Custom workload no longer exists in ToolHive, deleting"

**After Fix** (Expected):
- Workload details fetched: GET /api/v1beta/workloads/my-custom-github-server
- URL extracted: https://api.github.com/mcp
- Matched to registry entry by URL
- Workload persists with status=RUNNING
- Logs show: "Updated MCP server from workload"

### Verification

```bash
# Run ingestion manually
uv run mtm

# Query database to check workload exists
sqlite3 ~/.config/mcp-optimizer/mcp_optimizer.db \
  "SELECT name, package, remote, status, from_registry FROM mcpserver WHERE name='my-custom-github-server';"

# Expected result:
# my-custom-github-server|https://api.github.com/mcp|1|running|1
```

**Validation Checks**:
- ✅ Workload exists in database
- ✅ `package` column contains URL (https://api.github.com/mcp)
- ✅ `remote` is True (1)
- ✅ `status` is RUNNING
- ✅ `from_registry` is True (1)

## Test Scenario 2: Custom Remote Workload (Not in Registry)

**Goal**: Verify remote workload with URL not in registry creates custom entry

### Setup

1. Deploy a remote MCP workload with a URL not in registry:

```bash
thv run --remote \
  --name "my-custom-api" \
  --url "https://custom.api.com/mcp" \
  --transport sse
```

### Expected Behavior

- Workload details fetched
- URL extracted: https://custom.api.com/mcp
- No registry match found
- New server created with from_registry=False
- Workload persists (not deleted)

### Verification

```bash
sqlite3 ~/.config/mcp-optimizer/mcp_optimizer.db \
  "SELECT name, package, remote, status, from_registry FROM mcpserver WHERE name='my-custom-api';"

# Expected result:
# my-custom-api|https://custom.api.com/mcp|1|running|0
```

**Validation Checks**:
- ✅ Workload exists in database
- ✅ `package` contains custom URL
- ✅ `from_registry` is False (0)
- ✅ Workload not deleted in cleanup

## Test Scenario 3: Container Workload (Regression Test)

**Goal**: Verify container workload matching remains unchanged

### Setup

1. Deploy a container MCP workload:

```bash
thv run mcp/time
```

### Expected Behavior

- Container workload matched by package name (mcp/time)
- No workload detail fetch (not a remote workload)
- Existing matching logic works correctly

### Verification

```bash
sqlite3 ~/.config/mcp-optimizer/mcp_optimizer.db \
  "SELECT name, package, remote, status FROM mcpserver WHERE package='mcp/time';"

# Expected result:
# time|mcp/time|0|running
```

**Validation Checks**:
- ✅ Container workload matched correctly
- ✅ `package` contains image name (not URL)
- ✅ `remote` is False (0)
- ✅ No regression in container handling

## Test Scenario 4: Workload Detail Fetch Fails

**Goal**: Verify graceful error handling when API call fails

### Setup

1. Simulate failure by temporarily stopping ToolHive or using invalid name

2. Check logs for proper error handling

### Expected Behavior

- Error logged: "Failed to fetch workload details"
- Workload skipped for current cycle
- Other workloads in batch still processed
- No ingestion failure

### Verification

```bash
# Check logs
tail -f ~/.config/mcp-optimizer/mcp_optimizer.log | grep "workload details"

# Expected log entries:
# WARNING: Failed to fetch workload details, workload_name=..., error=...
# INFO: Successfully processed workload, workload_name=<other workload>
```

**Validation Checks**:
- ✅ Error logged with structured context
- ✅ Failed workload skipped
- ✅ Other workloads processed
- ✅ Ingestion batch completes

## Test Scenario 5: Multiple Workloads Same URL

**Goal**: Verify multiple workloads with same URL are both kept

### Setup

1. Deploy two remote workloads with different names but same URL:

```bash
thv run --remote --name "github-prod" --url "https://api.github.com/mcp"
thv run --remote --name "github-staging" --url "https://api.github.com/mcp"
```

### Expected Behavior

- Both workloads fetched for details
- Both have same URL
- Both matched to same registry entry
- Both persisted in database

### Verification

```bash
sqlite3 ~/.config/mcp-optimizer/mcp_optimizer.db \
  "SELECT name, package FROM mcpserver WHERE package='https://api.github.com/mcp';"

# Expected result:
# github-prod|https://api.github.com/mcp
# github-staging|https://api.github.com/mcp
```

**Validation Checks**:
- ✅ Both workloads exist
- ✅ Both have same package (URL)
- ✅ Both preserved (not deleted)

## Running Tests

### Unit Tests

```bash
# Run specific test file
uv run pytest tests/test_ingestion.py -v

# Run tests with coverage
uv run pytest tests/test_ingestion.py --cov=mcp_optimizer.ingestion --cov-report=term-missing
```

### Integration Tests

```bash
# Run all tests
task test

# Run with ToolHive integration
TOOLHIVE_HOST=localhost TOOLHIVE_PORT=8080 uv run pytest tests/test_ingestion.py -v
```

## Troubleshooting

### Issue: Workload still being deleted

**Check**:
1. Verify workload details API returns URL
2. Check registry has matching URL entry
3. Verify package column updated to URL (not name)

**Debug**:
```bash
# Check workload details API directly
curl http://localhost:8080/api/v1beta/workloads/<workload-name>

# Check registry entry
thv registry get <server-name>

# Check database package value
sqlite3 ~/.config/mcp-optimizer/mcp_optimizer.db \
  "SELECT name, package, remote FROM mcpserver WHERE name='<workload-name>';"
```

### Issue: Performance degradation

**Check**:
1. Number of remote workloads
2. API response times
3. Concurrent processing

**Measure**:
```bash
# Time ingestion cycle
time uv run mtm

# Check logs for timing
grep "ingestion completed" ~/.config/mcp-optimizer/mcp_optimizer.log
```

### Issue: API call fails

**Check**:
1. ToolHive is running
2. Network connectivity
3. Workload name is correct

**Debug**:
```bash
# Test API manually
curl -v http://localhost:8080/api/v1beta/workloads/<workload-name>

# Check ToolHive logs
docker logs toolhive-ui
```

## Success Criteria Validation

After running all test scenarios, verify:

1. ✅ **SC-001**: Remote workloads with custom names remain ingested (0% false deletion rate)
   - Test Scenario 1 passes

2. ✅ **SC-002**: Remote workloads correctly matched to registry within one cycle (100% match rate)
   - Test Scenario 1 shows correct matching

3. ✅ **SC-003**: Custom remote workloads successfully created (100% success rate)
   - Test Scenario 2 passes

4. ✅ **SC-004**: Container workloads unchanged (0% regression)
   - Test Scenario 3 passes

5. ✅ **SC-005**: No performance degradation
   - Ingestion time similar to before fix
   - Measured via timing tests

## Cleanup

After testing, clean up test workloads:

```bash
thv rm my-custom-github-server
thv rm my-custom-api
thv rm github-prod
thv rm github-staging
```

## Next Steps

Once all test scenarios pass:
1. Run full test suite: `task test`
2. Run quality gates: `task format && task lint && task typecheck && task test`
3. Commit changes
4. Create pull request with test evidence
