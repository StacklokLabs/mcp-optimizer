# ToolHive API Contract: Get Workload Details

**Feature**: 001-fix-remote-workload-matching
**Date**: 2025-10-20

## Endpoint

```
GET /api/v1beta/workloads/{name}
```

**Purpose**: Fetch detailed information about a specific workload, including its URL

**Base URL**: `http://{toolhive_host}:{toolhive_port}`

## Request

### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| name | string | Yes | Workload name from list_workloads response |

### Headers

```
Accept: application/json
```

### Example Request

```http
GET /api/v1beta/workloads/github-server HTTP/1.1
Host: localhost:8080
Accept: application/json
```

## Response

### Success Response (200 OK)

**Content-Type**: `application/json`

**Body**: Workload object with full details

```json
{
  "name": "github-server",
  "url": "https://api.github.com/mcp",
  "package": "mcp-github",
  "remote": true,
  "status": "running",
  "tool_type": "remote",
  "proxy_mode": "sse",
  "port": 8081,
  "group": "production",
  "created_at": "2025-10-20T10:00:00Z",
  "transport_type": "sse"
}
```

### Key Response Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | Yes | Workload identifier |
| url | string | **Yes** | **URL where workload is accessible - critical for matching** |
| package | string | No | Package/image name |
| remote | boolean | Yes | Whether this is a remote workload |
| status | string | Yes | Current workload status |
| tool_type | string | Yes | Type of tool ("mcp" or "remote") |
| proxy_mode | string | No | Proxy mode for transport |
| port | integer | No | Port number |
| group | string | No | Group membership |

### Error Responses

**404 Not Found**
- Workload with given name does not exist

```json
{
  "error": "workload not found",
  "message": "No workload found with name: github-server"
}
```

**500 Internal Server Error**
- Server error occurred

```json
{
  "error": "internal server error",
  "message": "Failed to retrieve workload details"
}
```

## Client Implementation

### Method Signature

```python
async def get_workload_details(
    self,
    workload_name: str
) -> Workload:
    """Fetch detailed workload information.

    Args:
        workload_name: Name of the workload to fetch

    Returns:
        Workload object with url field populated

    Raises:
        httpx.HTTPStatusError: If request fails (404, 500, etc.)
        httpx.TimeoutException: If request times out
        httpx.RequestError: If network error occurs
    """
```

### Error Handling Strategy

1. **404 Not Found**: Log warning, skip workload, continue with batch
2. **Timeout**: Log error, skip workload, retry next cycle
3. **Network Error**: Log error, skip workload, retry next cycle
4. **500 Server Error**: Log error, skip workload, retry next cycle

**Key Principle**: Never fail entire ingestion batch due to single workload detail fetch failure

### Retry Strategy

- No immediate retries (avoid cascading delays)
- Natural retry on next ingestion cycle (typically 30-60 seconds)
- Structured logging for monitoring and debugging

## Usage in Ingestion Flow

### When to Call

```python
# In IngestionService.ingest_workloads()
for workload in all_workloads:
    if workload.remote and workload.tool_type == "remote":
        # Fetch details to get accurate URL
        try:
            detailed_workload = await toolhive_client.get_workload_details(workload.name)
            # Use detailed_workload.url for matching
        except Exception as e:
            logger.warning("Failed to fetch workload details", error=e)
            continue  # Skip this workload
```

### URL Field Importance

The `url` field from the detailed response is:
1. Used to match against registry entry URLs
2. Stored in McpServer.package column
3. Used as stable identifier in cleanup logic
4. Critical for correct workload-registry matching

## Testing Considerations

### Mock Responses for Tests

**Success Case**:
```python
mock_response = {
    "name": "test-remote",
    "url": "https://test.api.com/mcp",
    "remote": True,
    "status": "running",
    "tool_type": "remote"
}
```

**Error Cases to Test**:
1. 404 response
2. Timeout exception
3. Network connection error
4. Invalid JSON response
5. Missing url field in response

### Contract Validation

Ensure response matches Pydantic Workload model:
- All required fields present
- Field types match model definition
- URL field is non-empty for remote workloads

## Performance Characteristics

**Expected Response Time**: 10-50ms
**Acceptable Timeout**: 5-10 seconds (configurable via mcp_timeout)
**Concurrent Requests**: Supported via asyncio.gather
**Rate Limiting**: None expected (internal API)

## Backward Compatibility

**Existing Endpoints**: Not affected
- `GET /api/v1beta/workloads` (list) - still used for initial discovery
- Other workload operations - unchanged

**Version**: v1beta indicates this may evolve, but URL field is fundamental and unlikely to change
