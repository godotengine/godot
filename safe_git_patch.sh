#!/bin/bash
set -euo pipefail

PATCH_FILE="$1"
TOKEN="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"

echo "🛠  Applying patch: $PATCH_FILE"

git apply --3way "$PATCH_FILE" || {
    echo "❌ Patch failed to apply."
    exit 1
}

echo "✅ Patch applied to working directory."

# Run pre-commit validations
if pre-commit run --all-files; then
    echo "✅ Pre-commit checks passed."
else
    echo "⚠️ Pre-commit had issues. Fix before committing!"
    exit 1
fi

git add -A
git commit -m "Applied patch automatically"
echo "✅ Commit created."

# Push using token-based authentication directly
REMOTE_URL="https://$TOKEN@github.com/FromAriel/godotheadless.git"
git push "$REMOTE_URL" HEAD:work

echo "🚀 Pushed successfully to 'work' branch."
