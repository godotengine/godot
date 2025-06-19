#!/usr/bin/env bash
set -euo pipefail

PATCH_FILE="/workspace/godotheadless/gitpatch.txt"
PATCH_BRANCH="massive-rename-commit"
MAIN_BRANCH="$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's@^origin/@@' || echo main)"

cd /workspace/godotheadless

# 1. Sanity checks
if [ ! -f "$PATCH_FILE" ]; then
  echo "❌ Patch file not found: $PATCH_FILE"
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "❌ Working tree not clean. Commit or stash changes first."
  exit 2
fi

# 2. Checkout base branch, update, and branch
git fetch origin
git checkout "$MAIN_BRANCH"
git pull
git checkout -b "$PATCH_BRANCH"

# 3. Try dry-run with three-way merge
echo "🕵️ Dry-run patch application..."
if ! git apply --3way --check "$PATCH_FILE"; then
  echo "❌ Patch will NOT apply cleanly. Please fix conflicts manually."
  exit 3
fi

# 4. Actually apply the patch (three-way merge helps auto-resolve renames)
echo "🔧 Applying patch (with --3way)..."
if git apply --3way "$PATCH_FILE"; then
  echo "✅ Patch applied."
else
  echo "❌ Patch apply failed. (See output above for why.)"
  exit 4
fi

# 5. Stage all changes
git add .

echo
echo "📝 Patch is staged. Please review your repo for correctness."
echo "Run:  git status      # to check status"
echo "Run:  git diff        # to see changes"
echo "Run:  git diff --cached   # to see what will be committed"
echo "---------------------------------------------"
echo "Press ENTER to commit and push, or Ctrl+C to abort and review further..."
read -r

# 6. Commit and push
git commit -m "Mass rename and gdverify integration (giant patch applied via script)"
git push -u origin "$PATCH_BRANCH"

echo
echo "✅ All done! Create a Pull Request from '$PATCH_BRANCH' to '$MAIN_BRANCH' on your repo host."
