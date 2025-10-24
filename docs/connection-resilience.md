# Connection Resilience

MCP Optimizer includes robust connection retry logic to handle scenarios where ToolHive (`thv serve`) becomes unavailable or restarts on a different port.

## Overview

When ToolHive restarts (e.g., after a crash or manual restart), it may bind to a different port within the configured range. MCP Optimizer automatically detects connection failures and attempts to rediscover and reconnect to ToolHive, minimizing service interruptions.

## Features

### 1. Automatic Port Rediscovery

When a connection failure is detected, MCP Optimizer:
- Rescans the configured port range (default: 50000-50100)
- Tries the initially configured port first (if specified)
- Updates its internal connection details when a new port is found
- Logs the old and new port for visibility

### 2. Exponential Backoff

To avoid overwhelming the system during extended outages:
- Starts with a 1-second delay (configurable)
- Doubles the delay after each failed attempt: 1s → 2s → 4s → 8s → 16s → 32s
- Caps at a maximum delay of 60 seconds (configurable)
- Resets to initial delay after successful port rediscovery

### 3. Configurable Retry Limits

- Default: 100 retry attempts (approximately 100 minutes)
- Configurable range: 1-500 attempts
- After exhausting retries, MCP Optimizer logs a critical error and exits gracefully

### 4. Comprehensive Coverage

Retry logic applies to all ToolHive API operations:
- Listing workloads
- Fetching workload details
- Getting registry information
- Installing new servers
- Periodic polling operations

## Configuration

### Environment Variables

```bash
# Maximum number of retry attempts (1-500)
export TOOLHIVE_MAX_RETRIES=100

# Initial backoff delay in seconds (0.1-10.0)
export TOOLHIVE_INITIAL_BACKOFF=1.0

# Maximum backoff delay in seconds (1.0-300.0)
export TOOLHIVE_MAX_BACKOFF=60.0
```

### CLI Options

```bash
mcp-optimizer \
  --toolhive-max-retries 15 \
  --toolhive-initial-backoff 2.0 \
  --toolhive-max-backoff 120.0
```

### Configuration File

These settings follow the standard MCP Optimizer configuration hierarchy:
1. CLI options (highest priority)
2. Environment variables
3. Default values (lowest priority)

## Usage Examples

### Example 1: Default Behavior

```bash
# Terminal 1: Start ToolHive
thv serve

# Terminal 2: Start MCP Optimizer with defaults
mcp-optimizer
```

**Behavior:**
- MCP Optimizer connects to ToolHive
- If ToolHive restarts, MCP Optimizer retries for ~3-4 minutes
- Exits if ToolHive doesn't come back online

### Example 2: Extended Retry Window

For production environments where you want more resilience:

```bash
# Allow up to 20 retries with longer delays
export TOOLHIVE_MAX_RETRIES=20
export TOOLHIVE_MAX_BACKOFF=120.0

mcp-optimizer
```

**Behavior:**
- Retries for approximately 10-15 minutes
- Suitable for environments with slower restart times

### Example 3: Quick Failure

For development environments where you want fast feedback:

```bash
# Fail quickly with fewer retries
mcp-optimizer --toolhive-max-retries 3 --toolhive-initial-backoff 0.5
```

**Behavior:**
- Retries only 3 times
- Exits within ~4 seconds if ToolHive is unavailable
- Useful for quick iteration in development

## Testing Scenarios

### Scenario 1: Port Change During Operation

This tests the automatic port rediscovery feature:

```bash
# Terminal 1: Start ToolHive
thv serve
# Note the port (e.g., 50001)

# Terminal 2: Start MCP Optimizer
mcp-optimizer

# Terminal 1: Kill ToolHive
pkill -f 'thv serve'

# Terminal 1: Restart ToolHive (will use different port)
thv serve
# Note the new port (e.g., 50002)
```

**Expected Result:**
- MCP Optimizer detects connection failure
- Logs: "ToolHive connection failed"
- Logs: "Attempting to rediscover ToolHive port"
- Logs: "Successfully rediscovered ToolHive on new port"
- Continues operation normally

### Scenario 2: Extended Outage

This tests the exponential backoff and ultimate failure handling:

```bash
# Terminal 1: Start ToolHive
thv serve

# Terminal 2: Start MCP Optimizer
mcp-optimizer

# Terminal 1: Kill ToolHive (don't restart)
pkill -f 'thv serve'
```

**Expected Result:**
- MCP Optimizer detects connection failure
- Retries with increasing delays
- Logs each attempt with backoff time
- After 100 attempts (~100 minutes), logs critical error
- Exits with error code 1

### Scenario 3: Startup Without ToolHive

This tests initial connection retry:

```bash
# Make sure ToolHive is not running
pkill -f 'thv serve'

# Start MCP Optimizer
mcp-optimizer

# In another terminal, start ToolHive within retry window
thv serve
```

**Expected Result:**
- MCP Optimizer attempts initial connection
- Retries with exponential backoff
- If ToolHive starts during retry window: connects and continues
- If ToolHive doesn't start: exits after max retries

## Logging

MCP Optimizer provides detailed logging at each stage:

### Connection Failure
```
WARNING  ToolHive connection failed attempt=1 max_retries=100 error="Connection refused" backoff_seconds=1.0
```

### Port Rediscovery
```
WARNING  Attempting to rediscover ToolHive port previous_port=50001
INFO     Successfully rediscovered ToolHive on new port old_port=50001 new_port=50002
```

### Successful Recovery
```
INFO     Port rediscovery successful, retrying immediately new_port=50002
```

### Ultimate Failure
```
ERROR    All retry attempts exhausted. ToolHive is unavailable. total_attempts=100
CRITICAL ToolHive connection failure - exiting max_retries=100 last_error="Connection refused"
```

## Operational Considerations

### When to Increase Retries

Consider increasing retry attempts when:
- ToolHive typically takes longer to restart in your environment
- You want to maximize availability during maintenance windows
- Running in production with high availability requirements

### When to Decrease Retries

Consider decreasing retry attempts when:
- You want fast failure detection in development
- Running health checks that need quick responses
- Implementing custom restart orchestration

### Monitoring

Monitor these log patterns to detect connection issues:
- `"ToolHive connection failed"` - Connection issues occurring
- `"Successfully rediscovered ToolHive on new port"` - Port changes detected
- `"ToolHive connection failure - exiting"` - Service becoming unavailable

## Integration with Other Features

### Runtime Modes

Connection resilience works in both runtime modes:
- **Docker mode**: Retries connection to ToolHive API
- **K8s mode**: Retries connection to Kubernetes API (no ToolHive connection needed)

### Polling Manager

The polling manager also implements connection resilience:
- Workload polling includes retry logic
- Registry polling includes retry logic
- Exits gracefully if connection cannot be restored during polling

## Troubleshooting

### Problem: MCP Optimizer exits too quickly

**Solution:** Increase retry attempts and/or backoff delays:
```bash
export TOOLHIVE_MAX_RETRIES=20
export TOOLHIVE_MAX_BACKOFF=120.0
```

### Problem: MCP Optimizer takes too long to fail

**Solution:** Decrease retry attempts:
```bash
mcp-optimizer --toolhive-max-retries 3
```

### Problem: Logs show repeated connection failures

**Cause:** ToolHive is not running or not accessible

**Solution:**
1. Check if ToolHive is running: `ps aux | grep "thv serve"`
2. Check ToolHive port: `thv ls` (if ToolHive is running)
3. Check port range configuration matches ToolHive's range
4. Check firewall/network connectivity

### Problem: Port rediscovery not finding new port

**Cause:** New port is outside configured scan range

**Solution:** Expand the port scan range:
```bash
export TOOLHIVE_START_PORT_SCAN=50000
export TOOLHIVE_END_PORT_SCAN=50200
```

## See Also

- [RETRY_IMPLEMENTATION.md](../RETRY_IMPLEMENTATION.md) - Full implementation details
- [Runtime Modes](runtime-modes.md) - Runtime mode configuration
- [Configuration](../src/mcp_optimizer/config.py) - All available configuration options

