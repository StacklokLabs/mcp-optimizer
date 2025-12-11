#!/bin/bash

# Script to download example data files from GitHub Release
# These files are excluded from git due to their size

set -e

REPO="StacklokLabs/mcp-optimizer"
RELEASE_TAG="anthropic-comparison-blogpost-data-v1.0.0"
BASE_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}"

FILES=(
    "mcp_tools_cleaned.json"
    "mcp_tools_cleaned_tests_claude-sonnet-4.json"
    "results.json"
    "results_accuracy_comparison.png"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Downloading anthropic comparison example data files..."
echo "Target directory: ${SCRIPT_DIR}"
echo ""

for file in "${FILES[@]}"; do
    if [ -f "${SCRIPT_DIR}/${file}" ]; then
        echo "⏭️  Skipping ${file} (already exists)"
    else
        echo "⬇️  Downloading ${file}..."
        if curl -L -f -o "${SCRIPT_DIR}/${file}" "${BASE_URL}/${file}"; then
            echo "✅ Downloaded ${file}"
        else
            echo "❌ Failed to download ${file}"
            echo "   URL: ${BASE_URL}/${file}"
            exit 1
        fi
    fi
    echo ""
done

echo "✨ All data files downloaded successfully!"
echo ""
echo "File sizes:"
du -sh "${SCRIPT_DIR}"/*.json "${SCRIPT_DIR}"/*.png 2>/dev/null || true
