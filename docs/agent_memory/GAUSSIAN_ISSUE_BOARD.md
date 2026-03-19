# Gaussian Splatting Issue Board

Baseline commit: `6dde6a82c3b`
Tracking scope: `ISSUE-001` to `ISSUE-045` (all subsystem Top-5 items)

Status legend:
- `Planned`: Assigned but not started in current orchestration wave.
- `In Progress`: Agent is actively implementing in its worktree.
- `Ready for Merge`: Agent branch is green and ready for coordinator review.
- `Done`: Merged into integration branch with validation evidence.
- `Blocked`: Waiting on dependency or unresolved issue definition.

| Issue ID | Subsystem | Title | Severity | Owner Agent | Status | Dependency | Evidence |
|---|---|---|---|---|---|---|---|
| ISSUE-001 | Core Data Structures | Core threading safety with RWLock + animation cache mutex ordering | Critical | agent-core-data | Planned | None | a7dd73fcd06 |
| ISSUE-002 | Core Data Structures | GPU resource ownership verification before uniform set creation | Critical | agent-core-data | Planned | None | 902f781f1b7 |
| ISSUE-003 | Core Data Structures | Guard GaussianData animation cache mutable fields from data races | High | agent-core-data | Planned | ISSUE-001 | a7dd73fcd06 |
| ISSUE-004 | Core Data Structures | Pack thread startup partial-failure deadlock fix | Critical | agent-core-data | Planned | ISSUE-001 | 102d229bb88 |
| ISSUE-005 | Core Data Structures | Document lock hierarchy and enforce lock-level assertions | Medium | agent-core-data | Planned | ISSUE-001 | d2fb0d32c2e |
| ISSUE-006 | GPU Sorting | Split oversized streaming/sorting integration unit for maintainability | High | agent-gpu-sorting | Planned | ISSUE-001 | 299e8f51706 |
| ISSUE-007 | GPU Sorting | Define missing canonical issue scope (no historical metadata found) | High | agent-gpu-sorting | Blocked | None | unresolved |
| ISSUE-008 | GPU Sorting | Define missing canonical issue scope (no historical metadata found) | High | agent-gpu-sorting | Blocked | None | unresolved |
| ISSUE-009 | GPU Sorting | Undo/redo safety for brush painting data mutation flows | Medium | agent-gpu-sorting | Planned | ISSUE-001 | 2128a1d20a9 |
| ISSUE-010 | GPU Sorting | Generation-safe cleanup for stale RenderingDevice pointers | High | agent-gpu-sorting | Planned | ISSUE-002 | 902f781f1b7 |
| ISSUE-011 | Tile Renderer | Zero-element and sort key config validation before GPU dispatch | High | agent-tile-renderer | Planned | ISSUE-010 | 57d317dca12 |
| ISSUE-012 | Tile Renderer | Replace global mutex bottlenecks with RWLock read concurrency | High | agent-tile-renderer | In Progress | ISSUE-001 | a7dd73fcd06; partial diff in `agent-core-data` worktree |
| ISSUE-013 | Tile Renderer | Optional AVX2 path with safe fallback for portability | Medium | agent-tile-renderer | Planned | None | 3ac1ab161ad |
| ISSUE-014 | Tile Renderer | Wire memory leak detection into runtime test coverage | Medium | agent-tile-renderer | Planned | None | 72a42aded5a |
| ISSUE-015 | Tile Renderer | Compile-time rasterizer statistics gating via build define | Medium | agent-tile-renderer | Planned | ISSUE-014 | b05c2844928 |
| ISSUE-016 | Streaming System | Merge culling orchestration into quality orchestration flow | High | agent-streaming | Planned | ISSUE-020 | da3b1076889 |
| ISSUE-017 | Streaming System | Decompose GaussianData god-object into focused companion units | Medium | agent-streaming | Planned | ISSUE-001 | 2251facd89e |
| ISSUE-018 | Streaming System | Consolidate duplicated ProjectSettings helper implementations | Medium | agent-streaming | Planned | None | 88465f14577 |
| ISSUE-019 | Streaming System | Remove unnecessary friend declarations and tighten visibility | Low | agent-streaming | Planned | ISSUE-017 | c72bafd8c0d |
| ISSUE-020 | Streaming System | Split monolithic IRenderer into role-segregated interfaces | High | agent-streaming | Planned | None | dd644a56109 |
| ISSUE-021 | LOD System | Validate frame dependency contracts before render entry points | High | agent-lod | Planned | ISSUE-020 | dd644a56109 |
| ISSUE-022 | LOD System | Replace static mutex one-time init with call_once semantics | Medium | agent-lod | Planned | None | 57d317dca12 |
| ISSUE-023 | LOD System | Forward/backward scene persistence compatibility safeguards | High | agent-lod | Planned | None | 61b094c398e |
| ISSUE-024 | LOD System | Make animation blend progression frame-rate independent | Medium | agent-lod | Planned | ISSUE-018 | 88465f14577 |
| ISSUE-025 | LOD System | Add unloaded-asset diagnostics for getter misuse | Medium | agent-lod | Planned | ISSUE-023 | ba143184765 |
| ISSUE-026 | Compute Infrastructure | Guard viewport texture state transitions against double-fire | Medium | agent-compute | Planned | ISSUE-020 | 88465f14577 |
| ISSUE-027 | Compute Infrastructure | SH buffer reallocation slack + shrink hysteresis | High | agent-compute | In Progress | ISSUE-023 | 50f58c3ab92; partial diff in `agent-compute` worktree |
| ISSUE-028 | Compute Infrastructure | Split monolithic tile render stages implementation | Medium | agent-compute | Planned | ISSUE-020 | 3d47157d998 |
| ISSUE-029 | Compute Infrastructure | Extract renderer nested POD types to standalone headers | Medium | agent-compute | Planned | ISSUE-020 | 11d0e29fdb2 |
| ISSUE-030 | Compute Infrastructure | Extract tile_binning shader helpers into includes | Medium | agent-compute | Planned | ISSUE-028 | 6534278a25e |
| ISSUE-031 | Shaders | Consolidate pack telemetry into cache-aligned data layout | Medium | agent-shaders | Planned | ISSUE-030 | 3c1d70036c8 |
| ISSUE-032 | Shaders | Make async readback resilient to partial callback failures | High | agent-shaders | Planned | ISSUE-027 | 50f58c3ab92 |
| ISSUE-033 | Shaders | Discard stale overflow stats in async GPU telemetry path | High | agent-shaders | Planned | ISSUE-032 | 19ca3b82c49 |
| ISSUE-034 | Shaders | Enforce compute shader include dependencies in build graph | Medium | agent-shaders | Planned | None | f362be08037 |
| ISSUE-035 | Shaders | Clarify CPU recording timing metric naming and docs | Low | agent-shaders | Planned | ISSUE-033 | 19ca3b82c49 |
| ISSUE-036 | Editor Integration | Cross-vendor `exp()` validation override path (`GS_SAFE_EXP`) | Medium | agent-editor | Planned | None | 32439d77809 |
| ISSUE-037 | Editor Integration | Runtime tile rasterizer shared-memory sizing contract checks | High | agent-editor | Planned | ISSUE-030 | 32439d77809 |
| ISSUE-038 | Editor Integration | Document SH sign convention consistency (PLY and shader) | Medium | agent-editor | Planned | ISSUE-034 | f362be08037 |
| ISSUE-039 | Editor Integration | Disk-backed thumbnail cache with fingerprint keying | Medium | agent-editor | Planned | None | 3ac1ab161ad |
| ISSUE-040 | Editor Integration | Add `reset_to_defaults()` to color grading resource | Low | agent-editor | Planned | None | 08c3b4b1706 |
| ISSUE-041 | Build System | Document painterly shader performance toggle route | Low | agent-build-ci | Ready for Merge | ISSUE-034 | `tests/ci/collect_production_evidence.ps1:320,329,463`; `tests/ci/README.md:18` |
| ISSUE-042 | Build System | Consolidate sort algorithm selection through unified traits | Medium | agent-build-ci | Ready for Merge | ISSUE-010 | `modules/gaussian_splatting/renderer/gpu_sorter.h:55,454,470`; `modules/gaussian_splatting/renderer/gpu_sorter.cpp:953,1037` |
| ISSUE-043 | Build System | Document theoretical complexity units and scaling assumptions | Low | agent-build-ci | Ready for Merge | ISSUE-042 | `modules/gaussian_splatting/renderer/gpu_sorter.h:183-188`; `tests/ci/TEST_COVERAGE.md:78` |
| ISSUE-044 | Build System | Protect against circular clip chains and enforce depth cap | High | agent-build-ci | Ready for Merge | ISSUE-040 | `modules/gaussian_splatting/animation/animation_state_machine.cpp:512,517-519` |
| ISSUE-045 | Build System | Track layout version in incremental saver contracts | Medium | agent-build-ci | Ready for Merge | ISSUE-023 | `modules/gaussian_splatting/persistence/incremental_saver.h:45-50`; `modules/gaussian_splatting/persistence/incremental_saver.cpp:763,809` |

## Wave Plan

- Wave 1: `agent-build-ci`, `agent-compute`, `agent-gpu-sorting`
- Wave 2: `agent-tile-renderer`, `agent-shaders`, `agent-core-data`
- Wave 3: `agent-lod`, `agent-editor`, `agent-streaming`
- Wave 4: `agent-qa-correctness`, `agent-qa-performance`

## Gate Checklist

- Each issue has a reproducible failing case or explicit contract gap.
- Fix includes error handling and resource lifetime safety.
- Tests added/updated for the issue domain.
- Issue row updated with commit hash, branch, and test evidence.
