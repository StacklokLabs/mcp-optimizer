#!/usr/bin/env bash

set -euo pipefail

OUTPUT_DIR="src/mcp_optimizer/toolhive/api_models"
TEMP_DIR="$(mktemp -d)"
THV_PID=""
MANAGE_THV="${MANAGE_THV:-true}"

# Cleanup function
cleanup() {
    local exit_code=$?
    if [ -n "$THV_PID" ] && [ "$MANAGE_THV" = "true" ]; then
        # Check if process exists before attempting to kill
        if kill -0 "$THV_PID" 2>/dev/null; then
            echo "Stopping thv serve (PID: $THV_PID)..."
            kill "$THV_PID" 2>/dev/null || true
            wait "$THV_PID" 2>/dev/null || true
        fi
    fi
    rm -rf "$TEMP_DIR"
    exit $exit_code
}

trap cleanup EXIT INT TERM

# Start thv serve if we're managing it
if [ "$MANAGE_THV" = "true" ]; then
    echo "Starting thv serve --openapi on port 8080..."
    thv serve --openapi --port 8080 &
    THV_PID=$!

    echo "Waiting for thv serve to be ready..."
    MAX_ATTEMPTS=30
    for i in $(seq 1 $MAX_ATTEMPTS); do
        if curl -s --max-time 5 http://127.0.0.1:8080/api/openapi.json > /dev/null 2>&1; then
            echo "thv serve is ready!"
            break
        fi
        if [ $i -eq $MAX_ATTEMPTS ]; then
            echo "ERROR: thv serve did not become ready"
            exit 1
        fi
        sleep 1
    done
fi

# Save current models to temp directory for comparison
if [ -d "$OUTPUT_DIR" ]; then
    echo "Backing up current models..."
    cp -r "$OUTPUT_DIR" "$TEMP_DIR/backup"
fi

# Remove old models (with safety check)
if [ -n "$OUTPUT_DIR" ] && [ "$OUTPUT_DIR" != "/" ]; then
    rm -rf "$OUTPUT_DIR"
else
    echo "ERROR: Invalid OUTPUT_DIR value"
    exit 1
fi

# Generate new models
echo "Generating models from OpenAPI specification..."
uv run datamodel-codegen \
    --url http://127.0.0.1:8080/api/openapi.json \
    --output "$OUTPUT_DIR" \
    --input-file-type openapi \
    --use-standard-collections \
    --use-subclass-enum \
    --snake-case-field \
    --collapse-root-models \
    --target-python-version 3.13 \
    --output-model-type pydantic_v2.BaseModel

# Check if there are meaningful changes (excluding timestamp)
if [ -d "$TEMP_DIR/backup" ]; then
    echo "Checking for meaningful changes..."

    # Create copies with timestamp lines removed for comparison
    mkdir -p "$TEMP_DIR/new" "$TEMP_DIR/old"

    # Process new files
    if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR"/*.py 2>/dev/null)" ]; then
        for file in "$OUTPUT_DIR"/*.py; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                grep --text -v "^#   timestamp:" "$file" > "$TEMP_DIR/new/$filename" || true
            fi
        done
    fi

    # Process old files
    for file in "$TEMP_DIR/backup"/*.py; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            grep --text -v "^#   timestamp:" "$file" > "$TEMP_DIR/old/$filename" || true
        fi
    done

    # Compare directories and collect changed files
    CHANGED_FILES=()
    for file in "$TEMP_DIR/new"/*.py; do
        filename=$(basename "$file")
        old_file="$TEMP_DIR/old/$filename"

        if [ ! -f "$old_file" ]; then
            CHANGED_FILES+=("$filename (new file)")
        elif ! diff "$file" "$old_file" > /dev/null 2>&1; then
            CHANGED_FILES+=("$filename")
        fi
    done

    # Check for deleted files
    for file in "$TEMP_DIR/old"/*.py; do
        filename=$(basename "$file")
        new_file="$TEMP_DIR/new/$filename"

        if [ ! -f "$new_file" ]; then
            CHANGED_FILES+=("$filename (deleted)")
        fi
    done

    if [ ${#CHANGED_FILES[@]} -eq 0 ]; then
        echo "No meaningful changes detected (only timestamp updated)"
        echo "Restoring original files..."
        rm -rf "$OUTPUT_DIR"
        mv "$TEMP_DIR/backup" "$OUTPUT_DIR"
        exit 0
    else
        echo "Meaningful changes detected in ${#CHANGED_FILES[@]} file(s):"
        printf '  - %s\n' "${CHANGED_FILES[@]}"
    fi
fi

echo "Model generation complete!"
