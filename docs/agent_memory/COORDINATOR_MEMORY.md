# Coordinator Memory

Last updated: 2026-03-19
Coordinator branch: `fix/windows-editor-tests`
Baseline commit: `6dde6a82c3b`
Integration branch: `integration/all-issues`

## Wave Status

- Wave 1 (active): `agent-build-ci`, `agent-compute`, `agent-gpu-sorting`
- Wave 2 (paused): `agent-core-data`, `agent-tile-renderer`, `agent-shaders`
- Wave 3 (paused): `agent-lod`, `agent-editor`, `agent-streaming`
- Wave 2 spillover (queued due slot limit): `agent-tile-renderer`, `agent-shaders`
- Wave 1 spillover (resolved): `agent-compute` now active
- Wave 4 (queued): `agent-qa-correctness`, `agent-qa-performance`

### Active Agent Sessions

- `agent-build-ci`: `019d0571-b295-7a73-99b2-11c733c7e9a0`
- `agent-compute`: `019d0575-15cb-7a53-adab-6775ba67b403`
- `agent-gpu-sorting`: `019d0575-15d5-73b3-95c2-d56ed88a31e7`
- Previous interrupted session IDs:
- `agent-lod`: `019d055c-7bab-7cc0-9efe-92f3cd5cfa9f`
- `agent-editor`: `019d055c-7bb6-7791-b5f7-bd14d2a9525a`
- `agent-streaming`: `019d055c-7bd6-7a72-80dd-cfe62ad492a7`
- `agent-core-data`: `019d055c-a590-7bc3-ab9f-b9f42726565c`
- `agent-gpu-sorting`: `019d055c-a5b5-71c0-ae3c-4a9d2709c9cc`
- `agent-build-ci`: `019d055c-a5ea-7423-9394-27ba6b84fd3e`

## Dirty Main Worktree (ignored by plan)

- Modified tracked fixture: `tests/examples/godot/test_project/tests/fixtures/test_splats.gsplatworld`
- Untracked generated assets: `templates/gaussian_splat_template/assets/`
- Untracked generated fixture: `tests/fixtures/test_splats.ply`

Rule: do not use these files as merge source. All work happens in dedicated worktrees.

## Worktree Map

- `.worktrees/agent-core-data` -> `agent/core-data`
- `.worktrees/agent-gpu-sorting` -> `agent/gpu-sorting`
- `.worktrees/agent-tile-renderer` -> `agent/tile-renderer`
- `.worktrees/agent-streaming` -> `agent/streaming`
- `.worktrees/agent-lod` -> `agent/lod`
- `.worktrees/agent-compute` -> `agent/compute`
- `.worktrees/agent-shaders` -> `agent/shaders`
- `.worktrees/agent-editor` -> `agent/editor`
- `.worktrees/agent-build-ci` -> `agent/build-ci`
- `.worktrees/agent-qa-correctness` -> `agent/qa-correctness`
- `.worktrees/agent-qa-performance` -> `agent/qa-performance`
- `.worktrees/integration-all-issues` -> `integration/all-issues`

## Agent Operating Rules

- Agents only edit files in their owned domain packet.
- Cross-domain edits require coordinator reassignment in this file.
- Never revert unrelated user changes.
- Every branch must pass owned test suite before merge request.

## Merge Log

- Pending first Wave 1 deliveries.
- Merge policy: only merge into `integration/all-issues` after packet tests pass and issue rows are updated to `Ready for Merge`.

## Reassignment Log

- Pending.

## Risks and Blocks

- `ISSUE-007` / `ISSUE-008` ambiguity resolved on 2026-03-19 with local contract evidence in `docs/agent_memory/GAUSSIAN_ISSUE_BOARD.md` (Wave 1 notes).
- Some issue fixes exist on other branches/historical commits and may need cherry-pick instead of reimplementation.
- Partial uncommitted diffs currently observed:
- `agent-compute`: `modules/gaussian_splatting/renderer/tile_render_resources.h`
- `agent-core-data`: `modules/gaussian_splatting/core/gaussian_data.h`
