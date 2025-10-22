#!/usr/bin/env bash

OUTPUT_DIR="src/mcp_optimizer/toolhive/api_models"
rm -r $OUTPUT_DIR
uv run datamodel-codegen \
    --url http://127.0.0.1:8080/api/openapi.json \
    --output $OUTPUT_DIR \
    --input-file-type openapi \
    --use-standard-collections \
    --use-subclass-enum \
    --snake-case-field \
    --collapse-root-models \
    --target-python-version 3.13 \
    --output-model-type pydantic_v2.BaseModel
