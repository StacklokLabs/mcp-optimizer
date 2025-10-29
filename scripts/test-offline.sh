#!/bin/bash
# Test script for offline Docker container functionality
# This simulates a completely airgapped/offline environment

set -e

# Build image only if SKIP_BUILD is not set (useful for CI where image is already built)
if [ -z "${SKIP_BUILD}" ]; then
  echo "🔧 Building Docker image..."
  docker build -t mcp-optimizer:latest .
else
  echo "⏭️  Skipping Docker build (SKIP_BUILD is set)"
fi

echo ""
echo "🔌 Testing offline mode (no network access)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker run --rm --network none mcp-optimizer:latest /app/.venv/bin/python -c "
import os
print('Environment variables:')
fastembed_cache_path = os.environ.get('FASTEMBED_CACHE_PATH')
print(f'  FASTEMBED_CACHE_PATH: {fastembed_cache_path}')
print(f'  TIKTOKEN_CACHE_DIR: {os.environ.get(\"TIKTOKEN_CACHE_DIR\")}')
print()
print('Testing embeddings...')
from mcp_optimizer.embeddings import EmbeddingManager
manager = EmbeddingManager(model_name='BAAI/bge-small-en-v1.5', enable_cache=True, threads=2, fastembed_cache_path=fastembed_cache_path)
embedding = manager.generate_embedding(['test offline mode'])
print(f'  ✓ Fastembed works! Embedding shape: {embedding.shape}')
print()
print('Testing tiktoken...')
import tiktoken
enc = tiktoken.get_encoding('cl100k_base')
tokens = enc.encode('test tiktoken offline')
print(f'  ✓ Tiktoken works! Encoded {len(tokens)} tokens')
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All offline tests passed!"
