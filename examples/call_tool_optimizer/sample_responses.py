"""Sample tool responses for testing the response optimizer."""

import json

# Large JSON response (API result with many items)
LARGE_JSON_RESPONSE = json.dumps(
    {
        "results": [
            {
                "id": i,
                "name": f"Item {i}",
                "description": f"This is a detailed description for item {i}. "
                f"It contains information about the product, its features, "
                f"specifications, and other relevant details that might be "
                f"useful for users looking to understand what this item offers.",
                "metadata": {
                    "created_at": "2025-01-15T12:00:00Z",
                    "updated_at": "2025-01-15T12:00:00Z",
                    "category": f"category_{i % 5}",
                    "tags": [f"tag_{j}" for j in range(i % 10)],
                    "attributes": {
                        "color": ["red", "blue", "green", "yellow"][i % 4],
                        "size": ["small", "medium", "large"][i % 3],
                        "weight": i * 0.5,
                        "price": 9.99 + i * 2.5,
                    },
                },
                "related_items": [i + 1, i + 2, i + 3] if i < 47 else [],
            }
            for i in range(50)
        ],
        "pagination": {"total": 500, "page": 1, "per_page": 50, "total_pages": 10},
        "metadata": {"query_time_ms": 142, "cache_hit": False, "server": "api-server-1"},
        "links": {
            "self": "/api/items?page=1",
            "next": "/api/items?page=2",
            "last": "/api/items?page=10",
        },
    },
    indent=2,
)

# Markdown documentation page
MARKDOWN_RESPONSE = """# API Documentation

## Overview

This API provides access to the item management system. It supports CRUD operations
for items, categories, and user management.

## Authentication

All API endpoints require authentication using a Bearer token.

### Getting a Token

To obtain an authentication token, make a POST request to `/auth/token`:

```bash
curl -X POST https://api.example.com/auth/token \\
  -H "Content-Type: application/json" \\
  -d '{"username": "user@example.com", "password": "your_password"}'
```

### Using the Token

Include the token in the Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.example.com/api/items
```

## Endpoints

### Items

#### List Items

**GET** `/api/items`

Returns a paginated list of items.

| Parameter | Type | Description |
|-----------|------|-------------|
| page | int | Page number (default: 1) |
| per_page | int | Items per page (default: 50, max: 100) |
| category | string | Filter by category |
| sort | string | Sort field (name, created_at, price) |
| order | string | Sort order (asc, desc) |

**Response:**

```json
{
  "results": [...],
  "pagination": {
    "total": 500,
    "page": 1,
    "per_page": 50
  }
}
```

#### Get Item

**GET** `/api/items/{id}`

Returns a single item by ID.

**Response:**

```json
{
  "id": 1,
  "name": "Item 1",
  "description": "...",
  "metadata": {...}
}
```

#### Create Item

**POST** `/api/items`

Creates a new item.

**Request Body:**

```json
{
  "name": "New Item",
  "description": "Description here",
  "category": "category_1",
  "price": 29.99
}
```

#### Update Item

**PUT** `/api/items/{id}`

Updates an existing item.

#### Delete Item

**DELETE** `/api/items/{id}`

Deletes an item.

### Categories

#### List Categories

**GET** `/api/categories`

Returns all available categories.

#### Create Category

**POST** `/api/categories`

Creates a new category.

## Error Handling

The API returns standard HTTP status codes:

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 500 | Internal Server Error |

Error responses include a JSON body:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request was invalid",
    "details": [...]
  }
}
```

## Rate Limiting

API requests are rate limited to:
- 100 requests per minute for standard users
- 1000 requests per minute for premium users

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Webhooks

Configure webhooks to receive notifications about item changes.

### Webhook Events

- `item.created` - New item created
- `item.updated` - Item updated
- `item.deleted` - Item deleted

### Webhook Payload

```json
{
  "event": "item.created",
  "timestamp": "2025-01-15T12:00:00Z",
  "data": {
    "id": 123,
    "name": "New Item"
  }
}
```

## SDK Support

Official SDKs are available for:
- Python
- JavaScript/TypeScript
- Go
- Ruby
- PHP

Install the Python SDK:

```bash
pip install example-api-client
```

## Changelog

### v2.0.0 (2025-01-01)
- Added webhook support
- Improved pagination
- Breaking: Changed authentication endpoint

### v1.5.0 (2024-12-01)
- Added category management
- Performance improvements

### v1.0.0 (2024-01-01)
- Initial release
"""

# Unstructured log output
UNSTRUCTURED_LOG_RESPONSE = """2025-01-15 12:00:00.123 INFO  [main] Starting application...
2025-01-15 12:00:00.456 INFO  [main] Loading configuration from /etc/app/config.yaml
2025-01-15 12:00:00.789 DEBUG [config] Found 15 configuration keys
2025-01-15 12:00:01.012 INFO  [db] Connecting to database at localhost:5432
2025-01-15 12:00:01.234 DEBUG [db] Connection pool size: 10
2025-01-15 12:00:01.567 INFO  [db] Database connection established
2025-01-15 12:00:02.890 INFO  [cache] Initializing Redis cache at localhost:6379
2025-01-15 12:00:03.123 INFO  [cache] Cache connection established
2025-01-15 12:00:03.456 INFO  [auth] Loading authentication providers...
2025-01-15 12:00:03.789 DEBUG [auth] Enabled providers: oauth2, jwt, basic
2025-01-15 12:00:04.012 INFO  [api] Registering API routes...
2025-01-15 12:00:04.234 DEBUG [api] Registered route: GET /api/items
2025-01-15 12:00:04.345 DEBUG [api] Registered route: POST /api/items
2025-01-15 12:00:04.456 DEBUG [api] Registered route: GET /api/items/{id}
2025-01-15 12:00:04.567 DEBUG [api] Registered route: PUT /api/items/{id}
2025-01-15 12:00:04.678 DEBUG [api] Registered route: DELETE /api/items/{id}
2025-01-15 12:00:04.789 DEBUG [api] Registered route: GET /api/categories
2025-01-15 12:00:04.890 DEBUG [api] Registered route: POST /api/categories
2025-01-15 12:00:05.012 INFO  [api] 7 routes registered
2025-01-15 12:00:05.234 INFO  [worker] Starting background workers...
2025-01-15 12:00:05.456 DEBUG [worker] Worker pool size: 4
2025-01-15 12:00:05.678 INFO  [worker] Email worker started
2025-01-15 12:00:05.890 INFO  [worker] Notification worker started
2025-01-15 12:00:06.012 INFO  [worker] Cleanup worker started
2025-01-15 12:00:06.234 INFO  [worker] Analytics worker started
2025-01-15 12:00:06.456 INFO  [metrics] Initializing Prometheus metrics...
2025-01-15 12:00:06.678 INFO  [metrics] Metrics endpoint available at /metrics
2025-01-15 12:00:06.890 INFO  [health] Health check endpoint available at /health
2025-01-15 12:00:07.012 INFO  [server] Starting HTTP server on 0.0.0.0:8080
2025-01-15 12:00:07.234 INFO  [server] Server started successfully
2025-01-15 12:00:07.456 INFO  [main] Application startup complete
2025-01-15 12:00:10.123 INFO  [api] Request: GET /api/items from 192.168.1.100
2025-01-15 12:00:10.345 DEBUG [db] Executing query: SELECT * FROM items LIMIT 50
2025-01-15 12:00:10.567 DEBUG [db] Query returned 50 rows in 45ms
2025-01-15 12:00:10.789 INFO  [api] Response: 200 OK (89ms)
2025-01-15 12:00:15.012 INFO  [api] Request: POST /api/items from 192.168.1.100
2025-01-15 12:00:15.234 DEBUG [api] Request body: {"name": "New Item", "price": 29.99}
2025-01-15 12:00:15.456 DEBUG [db] Executing INSERT INTO items...
2025-01-15 12:00:15.678 DEBUG [db] Insert completed, id=1234
2025-01-15 12:00:15.890 INFO  [cache] Invalidating cache for items list
2025-01-15 12:00:16.012 INFO  [api] Response: 201 Created (1000ms)
2025-01-15 12:00:20.123 WARN  [api] Slow request detected: GET /api/items/search?q=test (2500ms)
2025-01-15 12:00:25.234 INFO  [api] Request: GET /api/items/1234 from 192.168.1.101
2025-01-15 12:00:25.345 DEBUG [cache] Cache hit for item:1234
2025-01-15 12:00:25.456 INFO  [api] Response: 200 OK (15ms)
2025-01-15 12:00:30.567 ERROR [api] Request failed: GET /api/items/9999
2025-01-15 12:00:30.678 ERROR [api] Item not found: id=9999
2025-01-15 12:00:30.789 INFO  [api] Response: 404 Not Found (25ms)
2025-01-15 12:00:35.890 INFO  [worker] Email worker processed 5 messages
2025-01-15 12:00:40.012 INFO  [worker] Notification worker processed 12 notifications
2025-01-15 12:00:45.123 DEBUG [cleanup] Running scheduled cleanup task
2025-01-15 12:00:45.234 DEBUG [cleanup] Deleted 23 expired sessions
2025-01-15 12:00:45.345 DEBUG [cleanup] Cleanup completed in 112ms
2025-01-15 12:00:50.456 INFO  [metrics] Current connections: 45
2025-01-15 12:00:50.567 INFO  [metrics] Requests/min: 127
2025-01-15 12:00:50.678 INFO  [metrics] Average response time: 85ms
2025-01-15 12:00:55.789 WARN  [db] Connection pool running low (2/10 available)
2025-01-15 12:01:00.890 INFO  [db] Connection pool recovered (8/10 available)
2025-01-15 12:01:05.012 INFO  [api] Request: PUT /api/items/1234 from 192.168.1.100
2025-01-15 12:01:05.234 DEBUG [api] Request body: {"price": 34.99}
2025-01-15 12:01:05.456 DEBUG [db] Executing UPDATE items SET price=34.99 WHERE id=1234
2025-01-15 12:01:05.678 INFO  [cache] Invalidating cache for item:1234
2025-01-15 12:01:05.890 INFO  [api] Response: 200 OK (876ms)
2025-01-15 12:01:10.012 INFO  [webhook] Triggering webhook for item.updated
2025-01-15 12:01:10.234 DEBUG [webhook] Sending to https://example.com/webhook
2025-01-15 12:01:10.567 INFO  [webhook] Webhook delivered successfully (333ms)
2025-01-15 12:01:15.678 INFO  [api] Request: DELETE /api/items/1235 from 192.168.1.102
2025-01-15 12:01:15.890 DEBUG [db] Executing DELETE FROM items WHERE id=1235
2025-01-15 12:01:16.012 INFO  [cache] Invalidating cache for item:1235
2025-01-15 12:01:16.234 INFO  [api] Response: 204 No Content (556ms)
2025-01-15 12:05:00.000 INFO  [health] Health check: all systems operational
2025-01-15 12:10:00.000 INFO  [health] Health check: all systems operational
2025-01-15 12:15:00.000 INFO  [health] Health check: all systems operational
2025-01-15 12:20:00.000 INFO  [health] Health check: all systems operational
2025-01-15 12:25:00.000 INFO  [health] Health check: all systems operational
2025-01-15 12:30:00.000 INFO  [health] Health check: all systems operational
Application shutdown requested (SIGTERM)
2025-01-15 12:30:01.000 INFO  [server] Graceful shutdown initiated...
2025-01-15 12:30:01.100 INFO  [server] Stopping HTTP server...
2025-01-15 12:30:01.200 INFO  [worker] Stopping background workers...
2025-01-15 12:30:01.300 INFO  [db] Closing database connections...
2025-01-15 12:30:01.400 INFO  [cache] Closing cache connections...
2025-01-15 12:30:01.500 INFO  [main] Application shutdown complete
Exit code: 0
"""

# Dictionary of all samples for easy access
SAMPLE_RESPONSES = {
    "json": {
        "name": "Large JSON API Response",
        "content": LARGE_JSON_RESPONSE,
        "expected_type": "json",
    },
    "markdown": {
        "name": "Markdown Documentation",
        "content": MARKDOWN_RESPONSE,
        "expected_type": "markdown",
    },
    "unstructured": {
        "name": "Unstructured Log Output",
        "content": UNSTRUCTURED_LOG_RESPONSE,
        "expected_type": "unstructured",
    },
}
