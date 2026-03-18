#!/bin/bash
set -e

MERGED=0
CLOSED=0
MERGE_FAILED=0

# ═══════════════════════════════════════════════════════════════
# MERGE — Clean, correct PRs ready to ship
# ═══════════════════════════════════════════════════════════════

merge_pr() {
  local pr=$1
  echo "MERGING #$pr..."
  if gh pr merge "$pr" --merge --delete-branch 2>&1; then
    MERGED=$((MERGED + 1))
    echo "  ✓ Merged #$pr"
  else
    MERGE_FAILED=$((MERGE_FAILED + 1))
    echo "  ✗ Failed to merge #$pr"
  fi
}

merge_pr 1367  # LOD bias config mutation fix (real bug fix — merge first)
merge_pr 1366  # ChunkLayoutHint fallback determinism tests
merge_pr 1372  # Format import comparison values
merge_pr 1381  # Disable gs_native_arch in CI + build stamp
merge_pr 1382  # Compute infrastructure tests
merge_pr 1387  # Shared sort preflight validator
merge_pr 1389  # Guard blend-dependent dispatch
merge_pr 1390  # Explicit frame execution states
merge_pr 1399  # LOD selection producer/consumer dedup
merge_pr 1400  # Surface binning counters in HUD

# ═══════════════════════════════════════════════════════════════
# CLOSE — PRs that need work, are rejected, or need minor fixes
# ═══════════════════════════════════════════════════════════════

close_pr() {
  local pr=$1
  local reason="$2"
  echo "CLOSING #$pr..."
  if gh pr close "$pr" --comment "Closed per architecture audit review.

**Verdict:** $reason

See repository issues for full context. Reopen with fixes if applicable." 2>&1; then
    CLOSED=$((CLOSED + 1))
    echo "  ✓ Closed #$pr"
  else
    echo "  ✗ Failed to close #$pr"
  fi
}

# REJECT
close_pr 1374 "REJECT — References non-existent \`GaussianEditorServices\` class. Won't compile."

# NEEDS WORK
close_pr 1360 "NEEDS WORK — Dead snapshot variables (\`(void)\`-casted, never used). Inferior to existing pre-alpha-release fix."
close_pr 1361 "NEEDS WORK — Functionally correct but messy. Inferior to existing pre-alpha-release fix."
close_pr 1362 "NEEDS WORK — Uses wrong buffer for \`painterly_meta\` second field (flags vs brush_override_ids). Data corruption risk."
close_pr 1364 "NEEDS WORK — Merge conflicts. Synchronous GPU readback in async path defeats purpose of indirect dispatch. See #1365 for better approach."
close_pr 1371 "NEEDS WORK — Adds unused allowlists over existing checks. Logic bug where optional dirs skip source collection."
close_pr 1376 "NEEDS WORK — \`volatile\` misuse for thread safety, thread-unsafe \`source_path\` access, conflicts with #1375."
close_pr 1378 "NEEDS WORK — Silently overrides runtime premultiplied toggle. Dead code in blit shader."
close_pr 1383 "NEEDS WORK — Cache key collision risk (XOR before murmur), no invalidation strategy, logging while holding mutex."
close_pr 1386 "NEEDS WORK — Correct direction but massive whitespace churn mixed with logic changes makes it unreviewable. Split formatting from functional refactor."
close_pr 1392 "NEEDS WORK — Correct extraction targets but 8500+ line diff with massive reformatting. Split into one PR per extracted service."
close_pr 1394 "NEEDS WORK — Over-engineered stateless wrapper classes add \`std::function\` indirection with no behavioral benefit."
close_pr 1396 "NEEDS WORK — Good invariant guards buried in 8400-line whitespace reformat. Split formatting from functional changes."

# MERGE WITH MINOR FIXES — close with clear guidance on what to fix
close_pr 1363 "MERGE WITH MINOR FIXES — Clean optimization. Minor: verify \`#include <utility>\` necessity. Reopen after minor cleanup."
close_pr 1365 "MERGE WITH MINOR FIXES — Better approach than #1364. Fix: zero-count anomaly fires on legitimately empty frames. Document \`effective_element_count\` as advisory."
close_pr 1368 "MERGE WITH MINOR FIXES — Good architecture. Fix: \`get_device_api_name().find(\"metal\")\" heuristic is fragile. Mixed signedness in \`MAX<uint64_t>(1u, ...)\`."
close_pr 1369 "MERGE WITH MINOR FIXES — Correct fast-math scoping. Fix: verify \`Path.as_posix()\` on Windows edge cases."
close_pr 1370 "MERGE WITH MINOR FIXES — Good infrastructure. Fix: verify SCons \`from build_metadata_manifest import\` path resolution."
close_pr 1373 "MERGE WITH MINOR FIXES — Good feature. Fix: F9 toggle doesn't work when \`start_in_fast_mode_for_large_scenes\` is false."
close_pr 1375 "MERGE WITH MINOR FIXES — Sound debounce design. Fix: verify \`get_tree()\` return type (raw pointer, not Ref)."
close_pr 1377 "MERGE WITH MINOR FIXES — Good defensive measure. Fix: guard macros are no-ops without consumers defining expected versions."
close_pr 1379 "MERGE WITH MINOR FIXES — Good additions. Fix: line-ending noise pollutes diff. Assertion counter reuse is unclean. Loose tolerances."
close_pr 1380 "MERGE WITH MINOR FIXES — Solid defensive work. Fix: per-pixel validation is a perf regression — keep raster-side checks \`#ifdef DEBUG_ENABLED\` only."
close_pr 1384 "MERGE WITH MINOR FIXES — Solid fallback design. Fix: move settings registration to GaussianSplatManager, cache config at frame start."
close_pr 1385 "MERGE WITH MINOR FIXES — Clean code split. Fix: remove empty facade marker classes that serve no purpose."
close_pr 1388 "MERGE WITH MINOR FIXES — Well-designed estimator. Fix: cache config load to avoid redundant ProjectSettings lookups per frame."
close_pr 1391 "MERGE WITH MINOR FIXES — Good profile refactor. Fix: hoist static profile array to file scope."
close_pr 1393 "MERGE WITH MINOR FIXES — Good simplification. Fix: document that \`set_lod_level\`/\`start_streaming\` now include synchronous work in caller's frame."
close_pr 1395 "MERGE WITH MINOR FIXES — Sound VRAM-aware clamping. Fix: whitespace noise, non-static \`_clamp_u64_to_u32\` free function."
close_pr 1397 "MERGE WITH MINOR FIXES — Clean LOD resolution. Fix: \`LODThresholdResolutionState::reset()\` duplicates default member initializers."
close_pr 1398 "MERGE WITH MINOR FIXES — Correct lifecycle mutex. Fix: \`Thread::start\` failure path is effectively dead code. Document."

# The painterly PR (not reviewed by agents but still open)
close_pr 1402 "NOT REVIEWED — Large feature PR not included in audit batch. Requires separate review."

echo ""
echo "═══════════════════════════════════════════"
echo "Merged: $MERGED  Closed: $CLOSED  Merge Failed: $MERGE_FAILED"
echo "═══════════════════════════════════════════"
