#!/usr/bin/env bash
set -euo pipefail

cd /workspace/godotheadless

PATCH_FILE="gitpatch.txt"
PATCH_BRANCH="mass-move-godot"
MAIN_BRANCH="$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's@^origin/@@' || echo main)"

# 1. Check for patch file
if [ ! -f "$PATCH_FILE" ]; then
  echo "❌ Patch file not found: $PATCH_FILE"
  exit 1
fi

# 2. Ensure clean working state
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "❌ Uncommitted changes detected! Please commit or stash before running this script."
  exit 2
fi

# 3. Create a new branch for the patch
git fetch origin
git checkout "$MAIN_BRANCH"
git pull
git checkout -b "$PATCH_BRANCH"

# 4. Dry-run patch application to preview errors
echo "🕵️ Checking if patch can be applied cleanly (dry-run)..."
if ! git apply --check "$PATCH_FILE"; then
  echo "❌ Patch cannot be applied cleanly! Aborting. (Try to resolve conflicts, or ensure you are on the right base commit.)"
  exit 3
fi
echo "✅ Patch check passed. Applying patch for real..."

# 5. Apply patch with whitespace fixes
if git apply --whitespace=fix "$PATCH_FILE"; then
  echo "✅ Patch applied successfully!"
else
  echo "❌ Patch application failed! Check above for errors. Try manual patching or splitting the patch."
  exit 4
fi

# 6. Stage all changes and prompt for review
git add .

echo
echo "📝 Patch applied and staged! Please REVIEW your repo now."
echo "---------------------------------------------"
echo "Run:  git status   to see changes"
echo "Run:  git diff     to view unstaged diffs"
echo "Run:  git diff --cached   to view staged diffs"
echo "---------------------------------------------"
echo "Press ENTER to commit and push, or Ctrl+C to abort..."
read -r

# 7. Commit and push (waits for your confirmation)
git commit -m "Apply large patch: reorganize Godot, add gdverify tool"
git push -u origin "$PATCH_BRANCH"

echo
echo "✅ Patch committed and pushed to branch: $PATCH_BRANCH"
echo "Now create a Pull Request from '$PATCH_BRANCH' to '$MAIN_BRANCH' in your Git provider's web UI!"
