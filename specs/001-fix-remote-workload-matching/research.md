# Research: Fix Remote Workload Matching

**Feature**: 001-fix-remote-workload-matching
**Date**: 2025-10-20

## Problem Analysis

### Current Implementation Issue

The ingestion service in `src/mcp_optimizer/ingestion.py` currently matches workloads to registry entries using a brittle name-based approach. For remote workloads, this causes failures when:
1. A remote workload is deployed with a custom name different from its registry name
2. The matching logic relies on `workload.package` which may not contain the correct identifier
3. Cleanup logic incorrectly identifies the workload as not matching any registry entry
4. The workload gets ingested, then immediately deleted in the cleanup phase

### Root Cause

From code analysis of `ingestion.py`:
- Line 498-504: Container workloads use `workload.package` as identifier, remote use `workload.name`
- Line 948-949: Active workload identifiers are built using package for containers, name for remotes
- Line 1236-1241: Workload identifiers collected during ingestion
- The `package` column is populated with `workload.name` for remote workloads (line 502)
- Registry matching happens by comparing these identifiers, not URLs
- URLs are stable identifiers for remote workloads, names are not

## Decision: URL-Based Matching for Remote Workloads

### Rationale

**Why URLs are better identifiers:**
1. URLs are the actual endpoint addresses - they're what registry entries define
2. URLs remain stable even when workload names change
3. URLs are unique per service endpoint
4. Registry entries already contain URL information

**Why this solves the problem:**
1. A remote workload with name "custom-github" but URL "https://api.github.com/mcp" will correctly match registry entry "github" with the same URL
2. Multiple workloads can point to the same URL (valid use case)
3. Custom remote workloads (not in registry) still work - they get from_registry=False

### Implementation Approach

**1. Fetch Workload Details for Remote Workloads**

Add method to ToolhiveClient (`src/mcp_optimizer/toolhive/toolhive_client.py`):
```python
async def get_workload_details(self, workload_name: str) -> Workload:
    """Fetch detailed workload information including URL.

    Endpoint: GET /api/v1beta/workloads/{name}
    Returns: Full Workload object with url field populated
    """
```

**2. Update Package Column Semantics**

The `package` column in McpServer table will store:
- Container workloads: Docker image name (unchanged)
- Remote workloads: URL from workload details (changed from name)

**3. Modify Ingestion Logic**

In `IngestionService._upsert_server()` (ingestion.py:470-564):
- For remote workloads: fetch details to get URL, use URL as package identifier
- For container workloads: keep existing logic (use package name)

In `IngestionService._cleanup_removed_servers()` (ingestion.py:925-1002):
- Workload identifiers for remotes will be (URL, True) instead of (name, True)
- Matching will work correctly since package column contains URLs for remotes

**4. Registry Server Matching**

In `IngestionService._upsert_registry_server()` (ingestion.py:723-859):
- Remote registry servers already use `server_metadata.name` as package
- Need to update to use URL from registry metadata for remotes
- Extract URL from `RemoteServerMetadata.url` field

**5. Error Handling**

When fetching workload details fails:
- Log error with structured logging (structlog)
- Skip that workload for current cycle
- Allow retry on next ingestion cycle
- Don't fail entire ingestion batch

### Alternatives Considered

**Alternative 1: Keep name-based matching, enforce naming convention**
- **Rejected because**: Forces operational constraints on users
- Users should be able to name workloads freely
- Breaks existing deployments with custom names

**Alternative 2: Add separate URL column to McpServer**
- **Rejected because**: `package` column already serves as workload identifier
- Adding another column increases schema complexity
- URL is the stable identifier we need - reuse existing column

**Alternative 3: Store both name and URL, match on either**
- **Rejected because**: Ambiguous matching logic
- What if name matches one entry but URL matches another?
- Increases complexity without clear benefit

## Testing Strategy

### Unit Tests Required

1. **Test ToolhiveClient.get_workload_details()**
   - Success case: returns Workload with URL
   - Error cases: 404, network error, timeout
   - Mock httpx responses

2. **Test Remote Workload Matching**
   - Remote workload with custom name matches registry by URL
   - Remote workload without registry match creates custom entry
   - Container workload matching unchanged (regression test)

3. **Test Error Handling**
   - Workload detail fetch fails: logged, skipped, doesn't break batch
   - Partial failures in batch: other workloads still processed

4. **Test Cleanup Logic**
   - Remote workload with matching URL not deleted
   - Remote workload without matching URL treated correctly (from_registry flag)
   - Container workload cleanup unchanged (regression test)

### Integration Tests Required

1. **End-to-End Ingestion Cycle**
   - Deploy remote workload with custom name
   - Run ingestion
   - Verify workload persists (not deleted)
   - Verify correct registry match

2. **Registry Update Scenario**
   - Registry URL changes
   - Workload still using old URL
   - Verify treated as custom deployment

### Test Data Setup

- Mock ToolHive API responses for `/api/v1beta/workloads/{name}`
- Mock registry with remote server entries containing URLs
- Test database with existing McpServer entries

## Performance Considerations

### Additional API Calls

**Impact**: One additional GET request per remote workload per ingestion cycle

**Analysis**:
- Current implementation: List all workloads (1 API call)
- New implementation: List + detail fetch per remote workload (1 + N calls where N = # remote workloads)
- Typical N: 0-10 remote workloads in most deployments
- API call overhead: ~10-50ms per call
- Total added latency: ~100-500ms for 10 remote workloads

**Mitigation**:
- Already using batch processing for workload ingestion
- HTTP connection pooling in httpx client
- Concurrent detail fetches (asyncio.gather already in use)
- Detail fetch only for remote workloads, not containers

**Acceptable**: User explicitly stated "no performance degradation" as constraint, and with typical remote workload counts (< 10), the added latency is negligible compared to total ingestion time.

## Database Migration

**Not required**: The `package` column already exists and accepts string values. We're changing the semantic meaning of what's stored (URL vs name for remotes), but the schema remains identical.

**Migration approach**: Gradual - on next ingestion cycle, remote workloads will naturally update their package value to URLs.

## Backward Compatibility

**Container workloads**: Zero impact, matching logic unchanged
**Existing remote workloads**: Will be re-ingested with URL-based matching on next cycle
**Registry entries**: Need to ensure remote registry entries use URL as package identifier

## Implementation Phases

1. **Add ToolhiveClient.get_workload_details()** with error handling
2. **Update IngestionService._upsert_server()** for URL-based remote matching
3. **Update registry ingestion** to use URLs for remote servers
4. **Update cleanup logic** to use URL-based identifiers
5. **Add comprehensive tests** (unit + integration)
6. **Run quality gates** (format, lint, typecheck, test)

## Dependencies and Assumptions

**Dependencies**:
- httpx: Already in use for ToolHive API calls
- ToolHive API: `/api/v1beta/workloads/{name}` endpoint exists and returns `url` field
- Pydantic Workload model: Already has `url` field

**Assumptions**:
- ToolHive API is reliable and returns workload details consistently
- Workload URLs are stable identifiers (don't change frequently)
- Registry entries for remote servers contain valid URLs
- K8s runtime mode has equivalent workload detail fetch capability

**User-Provided Information**:
- Workload details endpoint: `<toolhive_server>/api/v1beta/workloads/{name}`
- URL field location: `url` field in response
- Package column usage: Should store URL for remotes
- Testing requirement: Verify new functionality and prevent regressions
