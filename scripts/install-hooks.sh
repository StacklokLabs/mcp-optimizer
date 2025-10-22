#!/bin/bash

# Script to install git hooks for the project

HOOKS_DIR=".git/hooks"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Installing git hooks..."

# Create hooks directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/$HOOKS_DIR"

# Install pre-commit hook
cat > "$PROJECT_ROOT/$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash

# Pre-commit hook to run ruff check with fixes and type checking before committing

echo "Running ruff check with fixes..."

# Run ruff check with fixes including unsafe fixes
uv run ruff check . --fix --unsafe-fixes

# Check if ruff made any changes
if [ $? -eq 0 ]; then
    echo "✓ Ruff check completed successfully"
    
    # Check if there are any changes that were made by ruff
    if ! git diff --quiet; then
        echo "⚠️  Ruff has made automatic fixes to your code."
        echo "Please review the changes and stage them before committing."
        echo ""
        echo "You can see the changes with: git diff"
        echo "Stage the changes with: git add -u"
        echo ""
        exit 1
    fi
else
    echo "✗ Ruff check failed"
    exit 1
fi

echo "Running type checking..."

# Run type checking with ty
uv run ty check .

if [ $? -eq 0 ]; then
    echo "✓ Type checking passed"
else
    echo "✗ Type checking failed"
    echo "Please fix the type errors before committing."
    exit 1
fi

exit 0
EOF

# Make the hook executable
chmod +x "$PROJECT_ROOT/$HOOKS_DIR/pre-commit"

echo "✓ Git hooks installed successfully!"
echo ""
echo "The pre-commit hook will now run:"
echo "  - 'ruff check . --fix --unsafe-fixes' to fix code style issues"
echo "  - 'uv run ty check .' to check types"
echo "before each commit."
