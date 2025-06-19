#!/bin/bash

# === CONFIGURATION ===

DEFAULT_BRANCH="work"
DEFAULT_REMOTE="origin"
DEFAULT_COMMIT_MESSAGE="Applied patch automatically"

# === SAFETY START ===

set -e  # Exit immediately on any failure
set -o pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 path/to/patchfile.diff"
  exit 1
fi

PATCH_FILE="$1"

if [ ! -f "$PATCH_FILE" ]; then
  echo "ERROR: Patch file '$PATCH_FILE' not found."
  exit 1
fi

echo "🛠 Applying patch: $PATCH_FILE"

# Ensure we're inside a git repo
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "ERROR: Not inside a git repository."
  exit 1
fi

# Apply the patch with 3-way fallback
git apply --3way "$PATCH_FILE"

echo "✅ Patch applied to working directory."

# Stage all changes
git add -A

# If nothing to commit, exit gracefully
if git diff --cached --quiet; then
  echo "⚠️ No changes to commit. Patch may have been already applied."
  exit 0
fi

# Commit the staged changes
git commit -m "$DEFAULT_COMMIT_MESSAGE"

echo "✅ Commit created."

# Push immediately
echo "🚀 Pushing to $DEFAULT_REMOTE/$DEFAULT_BRANCH ..."
git push "$DEFAULT_REMOTE" "$DEFAULT_BRANCH"

echo "🎉 All done. Your patch is now safely saved in GitHub."

exit 0
