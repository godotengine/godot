# Gaussian Renderer Refactor Memory

Last updated: 2026-03-24 (Europe/Berlin)
Owner role: Rendering architecture lead / refactor orchestrator  
Branch context: `/mnt/c/projects/godotgs-clean-refactor` (`refactor/gs-renderer-architecture`, dirty worktree)

## Purpose
This document is the implementation memory for the staged Gaussian Splatting renderer refactor. It records what we validated in code, what we generated, why the sequence is structured this way, and what must remain stable while migrating.

This is not a rewrite plan. It is a migration plan that preserves shipping behavior and keeps `GaussianSplatRenderer` as the public facade.

## Reality Check Performed
1. Regenerated architecture artifacts from current branch code using:
   - `python3 scripts/generate_architecture_diagrams.py`
2. Confirmed generated outputs under `docs/architecture/generated`:
   - `README.md`
   - `subsystem-dependencies.md`
   - `renderer-coupling.md`
   - `renderer-direct-access.md`
   - `coupling-report.md`
   - `local-dependencies.csv`
   - `summary.json`
3. Latest generated metrics:
   - `source_files: 272`
   - `include_edges: 836`
   - `symbol_reference_edges: 1685`
4. Parallel investigation was delegated to agents:
   - Agent Carver: full state/config consumer classification and mutability risk map.
   - Agent Ptolemy: sorting seam + composition-root coupling analysis and sequence.

## Findings (Ordered By Risk)
1. **Mutable-from-const provider contract is a primary blocker and must be explicitly addressed early.**
   - `IFrameStateProvider` returns mutable references from `const` methods in `gaussian_splat_renderer.h`.
   - Key locations: `modules/gaussian_splatting/renderer/gaussian_splat_renderer.h:332-340`.
   - Fallback provider path returns mutable static fallback objects in `gaussian_splat_renderer.cpp`.
   - Key locations: `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp:467-527`.
   - Impact: impossible to enforce snapshot-only read paths while mutable escape hatches remain exposed through `const` interfaces.

2. **State/config coupling is broad and concentrated in renderer-centric hot paths, not only in orchestrator count.**
   - Generated direct-access hotspots confirm this: `render_diagnostics_orchestrator.cpp`, `render_sorting_orchestrator.cpp`, `render_resource_orchestrator.cpp`, `performance_monitors.cpp`.
   - Evidence: `docs/architecture/generated/coupling-report.md` and `summary.json`.
   - Impact: renaming or moving orchestrators without reducing mutable access paths will not materially improve architecture.

3. **Observability consumers are mixed; treating the whole bucket as read-only snapshot migration would fail.**
   - Clean snapshot candidate: `core/performance_monitors.cpp` (read-mostly pull model).
   - Mixed mutator consumers:
     - `interfaces/debug_overlay_system.cpp:183-287` mutates renderer debug config/state.
     - `interfaces/painterly_renderer.cpp:1549-1562` pulls broad mutable renderer state on production path.
   - Impact: split phase required (`1a` snapshots for query consumers, `1b` explicit mutator APIs for mixed consumers).

4. **Sorting seam still leaks renderer ownership below orchestrator layer.**
   - Renderer-dependent APIs still present in pipeline header:
     - `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.h:64-65`
     - `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.h:86`
     - `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.h:237`
   - Implementation still reaches through renderer state in:
     - `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.cpp:2007+`
     - `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.cpp:2757+`
   - Impact: sorting is not decoupled enough to be a stable service seam.

5. **Composition root has real callback/pointer mesh complexity and renderer anchoring.**
   - Constructor wiring is dense and renderer-pointer based:
     - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp:554-704`.
   - Orchestrator headers store renderer pointers directly:
     - `modules/gaussian_splatting/renderer/render_streaming_orchestrator.h:13,38`
     - `modules/gaussian_splatting/renderer/render_sorting_orchestrator.h:21,71`
   - Impact: composition-root cleanup is justified, but only if it removes dependency direction issues (not just constructor reshuffling).

6. **Tests currently rely on mutating internals directly.**
   - Examples:
     - `modules/gaussian_splatting/tests/test_renderer_pipeline.h:537-538`
     - `modules/gaussian_splatting/tests/test_renderer_pipeline.h:638`
     - `modules/gaussian_splatting/tests/test_renderer_pipeline.h:1856`
   - Impact: tightening mutable access without replacement test hooks will break compatibility surfaces.

## Renderer State/Config Consumer Map

### Category A: Read-only diagnostics/monitoring (snapshot-first)
- `modules/gaussian_splatting/core/performance_monitors.cpp` (high call volume, mostly query usage)
- Read-mostly node/director integration:
  - `modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp`
  - `modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp`
  - `modules/gaussian_splatting/core/gaussian_splat_scene_director.cpp`

### Category B: Read-write production path (mutator-first)
- `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`
- `modules/gaussian_splatting/renderer/render_sorting_orchestrator.cpp`
- `modules/gaussian_splatting/renderer/render_resource_orchestrator.cpp`
- `modules/gaussian_splatting/renderer/render_data_orchestrator.cpp`
- `modules/gaussian_splatting/renderer/render_streaming_orchestrator.cpp`
- `modules/gaussian_splatting/renderer/render_output_orchestrator.cpp`
- `modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp`
- `modules/gaussian_splatting/interfaces/painterly_renderer.cpp`
- `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.cpp`

### Category C: Test-only
- `modules/gaussian_splatting/tests/test_renderer_pipeline.h`
- `modules/gaussian_splatting/tests/test_gaussian_splat_node.cpp`

### Category D: Editor/tooling and mixed diagnostics
- `modules/gaussian_splatting/interfaces/debug_overlay_system.cpp`
- `modules/gaussian_splatting/renderer/render_debug_state_orchestrator.cpp`
- `modules/gaussian_splatting/interfaces/interactive_state_manager.cpp`
- `modules/gaussian_splatting/renderer/render_diagnostics_orchestrator.cpp` (mixed: query + mutation side effects)

### Snapshot-first candidates
- `performance_monitors.cpp`
- read-only stats export surfaces in nodes/director paths
- query-only subsets of diagnostics once const-escape is removed

### Mutator-first candidates
- debug overlay mutators (`set_renderer_overlay_opacity`, invalidation/rebuild paths)
- painterly population path
- sorting/streaming/data orchestrators and pipeline stages

## Phase 1a Execution Log (Completed)

### Scope guardrails
- Query/read paths only.
- No debug overlay redesign.
- No painterly mutation redesign.
- No sorting API cleanup.
- Rollback remains limited to:
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.h`
  - `modules/gaussian_splatting/renderer/render_diagnostics_orchestrator.cpp`
  - `modules/gaussian_splatting/core/performance_monitors.cpp`

### Slice 1 (Completed): streaming/LOD monitor snapshot seam
- Date: 2026-03-23
- Introduced:
  - `GaussianSplatRenderer::MonitorStreamingSnapshot`
  - `GaussianSplatRenderer::get_monitor_streaming_snapshot() const`
- Migrated monitor families:
  - VRAM budget monitors
  - streaming core monitors
  - LOD monitors
  - memory-stream monitors
  - chunk-capacity + advanced LOD analytics
  - SH-compression monitors
- Verification:
  - Windows build + module lane passed (external verification run).

### Slice 2 (Completed): query-only completion for timing/route/projection fallback reads
- Date: 2026-03-23
- Added read-only snapshot fields for monitor fallbacks:
  - route/sort route identifiers
  - stage timing fields used by GPU timing monitors
  - performance/frame counters used by projection monitors
- Replaced remaining direct timing/route/projection monitor reads with snapshot reads.
- Verification:
  - `python3 tests/ci/run_module_tests.py --guard-only` passed locally.

### Slice 3 (Completed): cache-contract hardening without redesign
- Date: 2026-03-23
- Hardening changes:
  - monitor snapshot cache state is now `thread_local` (eliminates shared cross-thread mutable cache state),
  - cache key expanded from `(renderer, frame_counter)` to a compact monitor-focused key (frame + route ids + stage timing signals + key perf counters),
  - `_get_visible_splat_count()` fallback now uses snapshot (removes remaining projection fallback seam leak).
- Verification:
  - `python3 tests/ci/run_module_tests.py --guard-only` passed locally.

### Residual risks carried to Phase 1a closeout
- Low: cache key is intentionally partial, so some non-timing fields can still be stale within the same frame if updated after the first monitor read.
  - This is accepted for minimal Slice 3 scope and must be documented as a cache contract.
- Low: route invalidation keying uses hashed strings (`route_uid` / `sort_route_uid`), so collision risk is non-zero but negligible in practice.

### Phase 1a closeout evidence
- Native Windows verification (post Slice 3 and const accessor fix):
  - Build: `scons platform=windows target=editor dev_build=yes tests=yes module_gaussian_splatting_enabled=yes -j%NUMBER_OF_PROCESSORS%` passed.
  - Guard lane: passed.
  - Module lane: passed (`GaussianSplatting` 144 tests / 4,066 assertions).
  - GPU-dependent lanes stayed advisory in headless/no-GPU context.
- Architecture pack regenerated after Phase 1a implementation:
  - `python3 scripts/generate_architecture_diagrams.py`
  - Updated summary: `source_files=272`, `include_edges=836`, `symbol_reference_edges=1685`.
  - Notable seam signal: `core/performance_monitors.cpp` dropped from renderer state-access hotspot 101 to 3.

## Phase 1b Prep Map (Planning Only, No Code Changes)

### Exact inventory: mutable-from-const provider contracts to remove
1. `IFrameStateProvider` currently exposes mutable state from `const` methods:
   - `SortingState &get_sorting_state() const`
   - `RenderConfig &get_render_config() const`
   - `JacobianDebugConfig &get_jacobian_debug() const`
   - `ResourceState &get_resource_state() const`
   - `FrameState &get_frame_state() const`
   - `PerformanceState &get_performance_state() const`
   - `SubsystemState &get_subsystem_state() const`
   - Location: `modules/gaussian_splatting/renderer/gaussian_splat_renderer.h:334-340`
2. `FrameStateProvider` returns mutable static fallback objects for the same methods when `renderer == nullptr`:
   - Locations: `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp:467-528`
3. `RenderFrameContext::FrameDeps` stores mutable pointers for frame/state buckets while the provider is passed around as `const IFrameStateProvider *`:
   - Location: `modules/gaussian_splatting/renderer/gaussian_splat_renderer.h:275-297`
4. Stage/context structs carry `const IFrameStateProvider *` while calling mutating state methods through it:
   - Locations: `gaussian_splat_renderer.h:272`, `377`, `385`, `436`, `448`

### Full callsite migration map (grouped)

#### Query-only consumers
- `modules/gaussian_splatting/core/performance_monitors.cpp`
  - Status: already migrated to snapshot seam in Phase 1a.
  - Remaining direct state reads are cache-key inputs only (`get_frame_state()`, `get_debug_state()`, `get_performance_state()`), not monitor payload read paths.
- `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`
  - Query-only provider callsites that should move to read-only view contracts:
    - sort/index resource lookup: line `195`
    - culler availability checks: line `444`
    - render device / pipeline feature / frame plan reads: lines `869`, `1092`, `1540`, `1788`, `1841`, `2034`, `2174`, `2227`
    - render config signature reads: line `1272`

#### Mutating production consumers
- `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`
  - State mutation through provider-returned mutable refs:
    - frame/sort count writes and skip resets: lines `725-743`, `1462-1490`
    - raster/composite metrics writes: lines `1957`, `1981`, `2005`, `2013`, `2020-2032`, `2104-2106`, `2221-2232`, `2248`, `2265`, `2315`, `2342`
    - resource/compositor cache mutation via provider-derived state: lines `1971-1972`, `2042-2046`, `2212-2217`
- `modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp`
  - Provider used as mutable execution context for cull/sort stage orchestration:
    - lines `155-186`
- `modules/gaussian_splatting/interfaces/painterly_renderer.cpp`
  - Direct `GaussianSplatRenderer *` facade consumer, not an `IFrameStateProvider` consumer.
  - Broad mutable facade reach-through on production path:
    - state/config bundle capture at `1549-1562`
    - mutation-heavy painterly/raster integration and debug/perf updates: `1843-1991`

#### Tooling/debug consumers
- `modules/gaussian_splatting/interfaces/debug_overlay_system.cpp`
  - Direct `GaussianSplatRenderer *` facade consumer, not an `IFrameStateProvider` consumer.
  - Explicit mutators:
    - `set_renderer_overlay_opacity`: `183-197`
    - `invalidate_renderer_overlay`: `199-218`
    - `invalidate_renderer_hud`: `220-235`
    - `rebuild_renderer_overlay_statistics_from_cache`: `237-275`
    - `rebuild_renderer_performance_hud_lines`: `277-430`
- `modules/gaussian_splatting/interfaces/debug_overlay_macros.h`
  - Macros generate renderer-state mutating setters:
    - `GS_DEBUG_OVERLAY_RENDERER_SETTER_OVERLAY_IMPL`: `63-76`
    - `GS_DEBUG_OVERLAY_RENDERER_SETTER_HUD_IMPL`: `82-95`

#### Test-only consumers
- `modules/gaussian_splatting/tests/test_renderer_pipeline.h`
  - Direct internal mutation:
    - streaming system tear-down via mutable state: `537-538`, `638`
    - compositor cache internals mutation: `1856-1859`
  - Read-only but internal-state-dependent assertions:
    - streaming state checks: `554`, `652`, `776`
    - subsystem output compositor access: `1701`, `1832`
- `modules/gaussian_splatting/tests/test_gaussian_splat_node.cpp`
  - read-only scene state assertions: `746-758`

### Proposed replacement seams by group

#### Query-only consumers
- Introduce `IFrameStateView` (const-only) for stage/query paths.
- Move query-only stage helpers and signatures to `const IFrameStateView &`.
- Keep monitor read paths on snapshot seam; no direct mutable state exposure.
- `JacobianDebugConfig` starts in `IFrameStateView` by default unless a concrete production-path mutator is identified during migration.

#### Mutating production consumers
- Introduce `IFrameMutationAccess` for production writes, separate from read-only view.
- `IFrameMutationAccess` owns only currently-required mutable buckets:
  - `FrameState`, `PerformanceState`, `ResourceState`, `SortingState`, and explicit subsystem mutator access where needed.
- Stage inputs that mutate become `IFrameMutationAccess *`; pure query stages remain on `IFrameStateView *`.
- Treat painterly as an explicit `1b.2` hotspot sub-slice because it currently pulls a very wide renderer facade surface in one path.

#### Tooling/debug consumers
- Split overlay into command/query seams:
  - `DebugOverlayQueryView` (read-only metrics and snapshots)
  - `DebugOverlayCommandSink` (invalidate HUD/overlay, set flags/opacity)
- Keep renderer facade stable by delegating current public methods to these seams.
- Note: this is a direct-facade migration, not a provider-interface migration.

#### Test-only consumers
- Replace direct mutable internals with narrow test hooks on renderer/output compositor seams:
  - `test_set_streaming_system_for_test(Ref<GaussianStreamingSystem>)`
  - `test_clear_streaming_system_for_test()`
  - `test_get_output_cache_snapshot() const`
  - `test_override_output_cache_for_test(const OutputCacheOverride &)`
  - Optional friend test harness only for unavoidable edge cases.

### Phase 1b execution order and rollback points
1. **1b.0 Provider split scaffold (no callsite migration yet)**
   - Add `IFrameStateView` + `IFrameMutationAccess`.
   - Keep adapters so existing callsites compile unchanged.
   - Rollback point RP-1: remove new interfaces/adapters only.
2. **1b.1 Query-only callsite migration**
   - Move query-only stage helpers to `IFrameStateView`.
   - Prove no behavior change with guard/module lanes.
   - Acceptance caveat: state-bucket mutation via provider is blocked for migrated query paths, but service-pointer mutation risk remains until service seams are narrowed.
   - Rollback point RP-2: revert query-only signature/callsite set.
3. **1b.2 Mutating production migration**
   - Move mutating stage/provider paths to `IFrameMutationAccess`.
   - Split execution into:
     - `1b.2a` provider-based production writers (`render_pipeline_stages`, `render_instancing_orchestrator`)
     - `1b.2b` painterly direct-facade migration
   - Remove mutable-from-const methods from `IFrameStateProvider` adapters once all mutating production callsites are migrated.
   - Rollback point RP-3: re-enable adapter bridge methods temporarily.
4. **1b.3 Tooling/debug mixed-consumer migration**
   - Introduce explicit debug command/query seams and migrate direct-facade `debug_overlay_system` callsites.
   - Keep behavior and HUD/overlay output stable.
   - Rollback point RP-4: keep new seams, restore old delegations.
5. **1b.4 Test-hook migration before lock-down**
   - Replace direct test internals mutation with test hooks.
   - Rollback point RP-5: temporarily keep compatibility shim methods under test-only naming.
6. **1b.5 Lockdown prep completion criteria**
   - No mutable-from-const provider methods remain.
   - Query-only consumers do not require mutable renderer state access.
   - Debug/painterly/test paths have replacement seams/hooks in place.

### Phase 1b.0 implementation status (scaffold-only, behavior-preserving)
- Date: 2026-03-23
- Scope applied:
  - Added `IFrameStateView` and `IFrameMutationAccess` in `modules/gaussian_splatting/renderer/gaussian_splat_renderer.h`.
  - Kept existing `IFrameStateProvider` callsite surface intact as a legacy compatibility layer.
  - Added adapter bridge methods on `IFrameStateProvider`:
    - view side: `get_*_view()` methods forward to legacy getters,
    - mutation side: `get_*_mut()` methods forward to legacy getters.
- Explicitly preserved for this slice:
  - No callsite signature migrations (`render_pipeline_stages`, orchestrators, tooling, painterly, tests unchanged).
  - No behavior changes in render path execution.
  - `JacobianDebugConfig` remains classified as view-first (`get_jacobian_debug_view()`), not added to mutation access.
- Caveat carried forward:
  - `IFrameStateView` is currently const-only for state buckets, but still exposes mutable service pointers (`OutputCompositor *`, `GPUCuller *`, `PainterlyRenderer *`, `GPUSortingPipeline *`, `RenderingDevice *`).
  - Therefore, Phase `1b.1` does not yet provide full compile-time mutation prevention through service dependencies; this must be tightened in subsequent narrowing slices.
- Rollback boundary:
  - Revert only the interface scaffold/bridge hunk in `gaussian_splat_renderer.h` (RP-1).
- Verification status:
  - `git diff --check` passed for scaffold edits.
  - Full native Windows build/test rerun passed on the branch in subsequent validation workflows:
    - Build: pass.
    - Guard lane: pass.
    - Module lane: pass.

### Phase 1b.1 implementation status (query-only callsite migration, slice 1)
- Date: 2026-03-23
- Scope applied:
  - Query-only helper signatures moved from `IFrameStateProvider` to `IFrameStateView` in `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`:
    - `_get_sort_indices_buffer(...)`
    - `_compute_cull_config_signature(...)`
  - Query-only config/debug reads moved to view getters:
    - `get_render_config_view()` for color-grading signature path
    - `get_render_config_view()` and `get_jacobian_debug_view()` inside tile fallback setup
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay migration.
  - No painterly migration.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only the query-only signature/callsite edits in `render_pipeline_stages.cpp` (RP-2).
- Verification status:
  - `git diff --check` passed for the slice.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 1b.2b --no-regen-architecture`.
  - Native Windows verification passed via `Gaussian Production Gates` on `refactor/gs-renderer-architecture`:
    - Build: pass (Windows self-hosted module-validation lane).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).
    - Runtime/benchmark gates: pass (runtime harness, world-streaming gate, large-scene benchmark, eviction-churn benchmark).
  - Native Windows verification passed:
    - Build: pass (incremental, `render_pipeline_stages.cpp` recompiled).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).

### Phase 1b.1 implementation status (query-only callsite migration, slice 3)
- Date: 2026-03-23
- Scope applied:
  - Additional query-only provider reads were migrated to `IFrameStateView` aliases in:
    - `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`
    - `modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp`
    - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
  - In `render_pipeline_stages.cpp`, query-fetch usage now routes through `state_view` in:
    - `SortStage::execute(...)`
    - `RasterCompositeStage::execute(...)`
    - `RasterStage::resolve_painterly_output(...)`
    - `RasterStage::render_baseline_stage(...)`
    - `CompositeStage::execute(...)`
    - `render_sorted_splats_with_context(...)`
  - `build_frame_plan(...)` callsites now source scene/streaming/sorting/resource/subsystem/pipeline-feature inputs via `frame_provider` view accessors instead of broad direct renderer getters.
  - Snapshot initialization in `render_sorted_splats(...)` now reads `FrameState` and `SortingState` via view getters (`get_frame_state_view()`, `get_sorting_state_view()`).
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay migration.
  - No painterly direct-facade redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only slice-3 query-only alias/read edits in:
    - `render_pipeline_stages.cpp`
    - `render_instancing_orchestrator.cpp`
    - `gaussian_splat_renderer.cpp`
- Verification status:
  - `git diff --check` passed for the slice.
  - Native Windows verification passed:
    - Build: pass (incremental, touched renderer files rebuilt).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).

### Phase 1b.1 implementation status (query-only callsite migration, slice 4)
- Date: 2026-03-23
- Scope applied:
  - Remaining query-only provider reads in `RasterStage::render_tile_fallback(...)` were routed through `IFrameStateView`:
    - `get_frame_plan()`
    - `get_pipeline_features()`
    - `FrameState::frame_counter` read used for wind-time calculation
  - Location: `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay migration.
  - No painterly direct-facade redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only slice-4 query-only read edits in `render_pipeline_stages.cpp` (RP-2).
- Verification status:
  - `git diff --check` passed for the slice.
  - Native Windows verification passed (build + guard-only + module lane):
    - Build: pass (incremental, 3 files recompiled).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).

### Phase 1b.1 implementation status (query-only callsite migration, slice 5)
- Date: 2026-03-23
- Scope applied:
  - Additional query-fetch service reads in `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp` now route through existing `IFrameStateView` aliases:
    - `CullStage::execute(...)`: `get_gpu_culler()`, `get_rendering_device()`
    - `SortStage::execute(...)`: `get_gpu_culler()`, `get_sorting_pipeline()`, `get_rendering_device()`
    - `RasterStage::render_tile_fallback(...)`: `get_rendering_device()`
    - `RasterStage::try_reuse_cached_render(...)`: `get_output_compositor()`
    - `RasterStage::render_baseline_stage(...)`: `get_output_compositor()`
    - `RasterStage::render_painterly_or_baseline_stage(...)`: `get_painterly_renderer()`
    - `render_sorted_splats_with_context(...)`: `get_output_compositor()`
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay migration.
  - No painterly direct-facade redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only slice-5 query-fetch alias/read edits in `render_pipeline_stages.cpp` (RP-2).
- Verification status:
  - Implementation status: present in branch (not superseded).
  - Review status: approved in-thread (service-pointer caveat acknowledged).
  - `git diff --check` passed for the slice.
  - Native Windows build + module lane status not yet re-asserted in this thread (treat as pending until explicitly confirmed).

### Phase 1b.1 implementation status (query-only callsite migration, slice 6)
- Date: 2026-03-23
- Scope applied:
  - Query-only provider fetch in frame-entry skip handling now routes through `IFrameStateView`:
    - `execute_frame_entry(...)`: `get_gpu_culler()`
  - Location: `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`
  - No behavior/path changes were introduced; this slice also normalized indentation in already-touched `render_pipeline_stages.cpp` blocks to keep the migration diff reviewable.
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay migration.
  - No painterly direct-facade redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only slice-6 query-only alias/read edit and local formatting cleanup in `render_pipeline_stages.cpp` (RP-2).
- Verification status:
  - Implementation status: present in branch (not superseded).
  - Review status: approved in-thread (service-pointer caveat acknowledged).
  - `git diff --check` passed for the slice.
  - Native Windows build + module lane status not yet re-asserted in this thread (treat as pending until explicitly confirmed).

### Phase 1b.1 implementation status (query-only callsite migration, slice 7)
- Date: 2026-03-23
- Scope applied:
  - Additional query-only reads in `modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp` now route through `IFrameStateView`:
    - Readiness precheck: streaming-state read (`get_streaming_state()`) now sourced from a local `FrameStateProvider` view alias.
    - Instanced render loop: frame-state reads for `before_frame_counter`, `render_time_ms`, and `visible_splat_count` now source from `frame_state_view.get_frame_state_view()`.
  - Local indentation cleanup was applied in the touched loop block for reviewability; no API or behavior-path changes were introduced.
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay migration.
  - No painterly direct-facade redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only slice-7 query-only read edits and local formatting cleanup in `render_instancing_orchestrator.cpp`.
- Verification status:
  - `git diff --check` passed for the slice.
  - Native Windows verification passed (build + guard-only + module lane), per reported external runner result on 2026-03-23.

### Phase 1b.1 implementation status (query-fetch migration under service-pointer caveat, slice 8)
- Date: 2026-03-23
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp` (`RasterStage::render_tile_fallback(...)`):
    - Streaming-state fallback read for `total_gaussians` now sources from `state_view.get_streaming_state()` instead of direct renderer getter.
    - Debug-overlay projection-log gate now reads from `state_view.get_subsystem_state_view().debug_overlay_system` instead of direct renderer getter.
  - The function still mutates service-owned state/render flow later; this slice is explicitly query-fetch migration under the known `IFrameStateView` service-pointer caveat, not immutable query isolation.
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay redesign.
  - No painterly direct-facade redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only slice-8 query-read substitutions in `render_pipeline_stages.cpp`.
- Verification status:
  - `git diff --check` passed for the slice.
  - Native Windows verification passed (build + guard-only + module lane):
    - Build: pass (incremental, `render_pipeline_stages.cpp` only).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).

### Phase 1b.1 implementation status (query-only callsite migration, slice 9)
- Date: 2026-03-23
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`:
    - Pipeline trace/frame-event frame-counter reads now source from `IFrameStateView` via local `FrameStateProvider` aliases (`_begin_pipeline_trace(...)`, `_record_pipeline_event(...)`).
    - `prepare_render_frame_context(...)` frame-id assignment now sources from `state_view.get_frame_state_view().frame_counter`.
    - `log_stage_result(...)` 60-frame log cadence check now sources from `IFrameStateView` frame-state read.
  - This slice is read-only query migration; no service-pointer fetch or mutator signature changes were introduced.
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay or painterly redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only slice-9 frame-counter query-read substitutions in `render_pipeline_stages.cpp`.
- Verification status:
  - `git diff --check` passed for the slice.
  - Native Windows verification covered by the subsequent slice-10 lane (same touched file state):
    - Build: pass (incremental, `render_pipeline_stages.cpp` only).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).

### Phase 1b.1 implementation status (query-only callsite migration, slice 10)
- Date: 2026-03-23
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp` (`execute_frame_entry(...)` pre-plan state selection):
    - Added a local pre-plan `FrameStateProvider` + `IFrameStateView` alias for read-side fallback sourcing before final provider wiring.
    - Fallback reads for `SceneState`, `StreamingState`, and `PipelineFeatureSet` now route through `preplan_view` instead of direct renderer getters.
  - Mutating fallback references (`SortingState`, `ResourceState`, `SubsystemState`) remain unchanged in this slice.
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay or painterly redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only slice-10 pre-plan fallback query-read substitutions in `render_pipeline_stages.cpp`.
- Verification status:
  - `git diff --check` passed for the slice.
  - Native Windows verification passed (build + guard-only + module lane):
    - Build: pass (incremental, `render_pipeline_stages.cpp` only).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).

### Phase 1b.1 implementation status (batched frame-context provider wiring + query-fetch migration under caveat, slice 11)
- Date: 2026-03-23
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`:
    - `prepare_render_frame_context(...)` dependency assembly now routes service fetches through `IFrameStateView` (`output_compositor`, `gpu_culler`, `painterly_renderer`, `sorting_pipeline`, `rendering_device`, `pipeline_features`) instead of direct renderer getters.
    - Mutable state-bucket dep pointers in frame context (`sorting_state`, `render_config`, `jacobian_debug`, `resource_state`, `frame_state`, `performance_state`, `subsystem_state`) now route through `FrameStateProvider` accessors instead of direct renderer getters.
    - `execute_frame_entry(...)` pre-plan fallback for mutable buckets (`sorting_state`, `resource_state`, `subsystem_state`) now routes through the local pre-plan provider.
  - This slice intentionally includes mutable-bucket wiring through the legacy provider contract; it is not immutable query isolation.
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay or painterly redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only slice-11 frame-context/pre-plan wiring substitutions in `render_pipeline_stages.cpp`.
- Verification status:
  - `git diff --check` passed for the slice.
  - Native Windows verification passed (build + guard-only + module lane):
    - Build: pass (incremental, `render_pipeline_stages.cpp` only).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).

### Phase 1b.1 implementation status (batched instancing provider wiring + query-fetch migration under caveat, slice 12)
- Date: 2026-03-23
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp` (`RenderInstancingOrchestrator::render_instanced(...)`):
    - Added a root `FrameStateProvider` and split aliases for `IFrameMutationAccess` and `IFrameStateView`.
    - Streaming readiness query now reads through `IFrameStateView` (`get_streaming_state()`), replacing direct renderer state-bucket access.
    - Readiness-failure write path now updates `FrameState`, `SortingState`, and `PerformanceMetrics` through provider mutation access instead of direct renderer getters.
    - Instanced frame-plan build now sources scene/streaming/sorting/resource/subsystem/pipeline-feature queries through `IFrameStateView`.
    - Loop/frame aggregation reads (`frame_counter`, `render_time_ms`, `visible_splat_count`) now route through view aliases; frame-counter rollback and final aggregate writes now route through root mutation access.
  - This slice intentionally uses the legacy provider contract for mutable buckets and keeps direct facade calls where no provider seam exists (instance-pipeline buffers/debug state).
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay or painterly redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only slice-12 provider/view rewiring in `render_instancing_orchestrator.cpp`.
- Verification status:
  - `git diff --check` passed for touched files.
  - Local guard-only lane passed (`python3 tests/ci/run_module_tests.py --guard-only`).
  - Native Windows verification passed (build + guard-only + module lane):
    - Build: pass (incremental, `render_instancing_orchestrator.cpp` only).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).

### Phase 1b.1 implementation status (batched data-orchestrator provider wiring + query-fetch migration under caveat, slice 13)
- Date: 2026-03-23
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_data_orchestrator.cpp`:
    - Added local `FrameStateProvider` aliases in constructor and data/update paths (`IFrameStateView` + `IFrameMutationAccess`) and rewired state-bucket reads/writes away from direct renderer getters for:
      - `SubsystemState` access used by memory-stream device manager wiring and static-chunk culler state.
      - `PerformanceState` metrics initialization/reset during data activation and failure fallback.
      - `SortingState`/`FrameState` resets in data clear and streaming initialization paths.
      - `ResourceState` buffer-manager readiness and clear path in data upload prep.
    - Preserved direct renderer calls for non-provider surfaces (`DeviceState`, performance settings, cull multiplier/slack).
  - This slice intentionally keeps legacy provider mutability semantics and does not claim immutable dependency isolation.
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay or painterly redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only slice-13 provider/view rewiring in `render_data_orchestrator.cpp`.
- Verification status:
  - `git diff --check` passed for touched files.
  - Local guard-only lane passed (`python3 tests/ci/run_module_tests.py --guard-only`).
  - Native Windows verification passed (build + guard-only + module lane):
    - Build: pass (incremental, `render_data_orchestrator.cpp` only).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).

### Phase 1b.1 implementation status (batched renderer-orchestrator provider/view rewiring under caveat, slice 14)
- Date: 2026-03-23
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_device_orchestrator.cpp`:
    - Frame-id diagnostics now route through `FrameStateProvider` view reads instead of direct `renderer->get_frame_state()` reads in device/submission/sync error paths.
    - Rasterizer output tracking now routes through `IFrameStateView::get_subsystem_state_view()` instead of direct subsystem getter access.
  - `modules/gaussian_splatting/renderer/render_streaming_orchestrator.cpp`:
    - Added local provider aliases in `ensure_instance_streaming_system(...)`, `sync_instance_pipeline_assets(...)`, `render_streaming_frame(...)`, and `tick_streaming_only(...)`.
    - Rewired `SceneState`, `SubsystemState`, `PerformanceState`, `ResourceState`, and `FrameState` accesses to provider view/mutation paths where available.
    - Kept direct facade reads for non-provider surfaces and known 1b.1 caveat surfaces (`StreamingState` mutation paths, `DeviceState`, performance settings, cull multiplier/slack, debug state).
  - This slice remains migration-under-caveat work; it does not claim immutable dependency isolation.
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay or painterly redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only slice-14 rewiring in `render_device_orchestrator.cpp` and `render_streaming_orchestrator.cpp`.
- Verification status:
  - `git diff --check` passed for touched files.
  - Local guard-only lane passed (`python3 tests/ci/run_module_tests.py --guard-only`).
  - Native Windows verification passed via `Gaussian Production Gates` on `refactor/gs-renderer-architecture`:
    - Build: pass (Windows self-hosted module-validation lane).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).

### Phase 1b.2a implementation status (stage-writer split with explicit mutation access, slice 15)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.h`:
    - Stage carriers were split so query-only stage inputs (`CullStageInput`, `SortStageInput`) now carry `const IFrameStateView *`.
    - Mutating raster-stage input now carries both `const IFrameStateView *` and `IFrameMutationAccess *`.
    - `RenderFrameContext` continues to carry the legacy `state_provider` bridge, but also exposes explicit `state_view` / `mutation_access` slots for in-flight stage execution.
  - `modules/gaussian_splatting/renderer/render_pipeline_stages.h`:
    - `reset_render_state_for_frame(...)` now accepts explicit `IFrameStateView` + `IFrameMutationAccess` instead of a const provider.
  - `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`:
    - Added `_resolve_state_view(...)` and `_resolve_mutation_access(...)` helpers so older provider-only callers can bridge into the new split contract without behavior changes.
    - `execute_frame_entry(...)` now resolves explicit view/mutation access before stage dispatch and uses mutation access for frame-count writes.
    - `RasterCompositeStage::execute(...)` now builds raster/composite inputs with explicit `state_view` / `mutation_access` instead of provider-only stage contracts.
    - Mutating stage helpers now source bucket writes from `IFrameMutationAccess`:
      - `reset_render_state_for_frame(...)`
      - `RasterStage::render_tile_fallback(...)`
      - `RasterStage::try_reuse_cached_render(...)`
      - `RasterStage::render_baseline_stage(...)`
      - `RasterStage::render_painterly_or_baseline_stage(...)`
      - `render_sorted_splats_with_context(...)`
    - Query-only cull/sort stage paths continue to read through `IFrameStateView`.
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay or painterly direct-facade redesign.
  - No sorting-seam API work.
  - No public `GaussianSplatRenderer` facade behavior change.
- Remaining caveat:
  - The legacy provider bridge still exists in `RenderFrameContext::state_provider`, and `_resolve_mutation_access(...)` still falls back through that bridge for older callers that have not populated explicit mutation access yet.
  - Service pointers exposed by `IFrameStateView` remain mutable and are still under the known 1b caveat.
- Rollback boundary:
  - Revert only slice-15 stage-writer split changes in `gaussian_splat_renderer.h`, `render_pipeline_stages.h`, and `render_pipeline_stages.cpp`.
- Verification status:
  - `git diff --check` passed for touched files.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 1b.2a --no-regen-architecture`.
  - Native Windows verification passed via `Gaussian Production Gates` on `refactor/gs-renderer-architecture`:
    - Build: pass (Windows self-hosted module-validation lane).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).
    - Runtime/benchmark gates: pass (runtime harness, world-streaming gate, large-scene benchmark, eviction-churn benchmark).

### Phase 1b.2b implementation status (painterly provider-backed bucket reads, slice 16)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/interfaces/painterly_renderer.cpp`:
    - Introduced local `GaussianSplatRenderer::FrameStateProvider` instances in painterly production helpers that already sit on provider-backed buckets.
    - Rewired `clear_painterly_gpu_resources(...)` and `update_painterly_gpu_resources(...)` to mutate `SubsystemState` through `IFrameMutationAccess` instead of direct renderer getters.
    - Rewired `render_painterly_frame(...)` to query `SubsystemState` through `IFrameStateView` for tile-raster output access.
    - Rewired `populate_painterly_gbuffer(...)` to source `scene_state`, `resource_state`, `streaming_state`, `sorting_state`, `subsystem_state`, `frame_state`, `performance_state`, and `jacobian_debug` through `IFrameStateView` / `IFrameMutationAccess` aliases instead of direct renderer getters.
    - Also moved direct render-device / render-config reads that already had provider seams:
      - `_resolve_tracked_device(...)`
      - `_update_painterly_texture_tracking(...)`
      - `_ensure_painterly_composite_resources(...)`
      - `update_painterly_gpu_resources(...)`
      - `populate_painterly_gbuffer(...)` opacity multiplier
    - Kept direct access for surfaces that still lack a provider seam in this slice, including `device_state`, `view_state`, `tile_renderer_state`, `debug_state`, `culling_config`, and `painterly_config`.
- Explicitly preserved for this slice:
  - No painterly public API redesign.
  - No service-pointer narrowing.
  - No debug overlay, sorting-seam, or composition-root work.
  - No attempt to force non-provider surfaces through the new aliases.
- Remaining caveat:
  - `device_state`, `view_state`, `tile_renderer_state`, `debug_state`, `culling_config`, and `painterly_config` remain renderer-direct by design until matching seams exist.
- Rollback boundary:
  - Revert only slice-16 provider-alias rewiring in `modules/gaussian_splatting/interfaces/painterly_renderer.cpp`.
- Verification status:
  - `git diff --check` passed for the slice.

### Phase 1b.1 implementation status (query-only callsite migration, slice 2)
- Date: 2026-03-23
- Scope applied:
  - Additional query-only usages were switched to `IFrameStateView` aliases in `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`:
    - `SortStage::execute(...)` query helper reads (`_get_sort_indices_buffer(...)`) now route through `state_view`.
    - `RasterCompositeStage::execute(...)` query reads (`get_output_compositor`, cull/color signatures, sort-index buffer helper) now route through `state_view`, while mutating downstream stage pointers still use the existing provider pointer.
    - `RasterStage::resolve_painterly_output(...)` reads painterly renderer through `state_view`.
    - `RasterStage::render_baseline_stage(...)` query-only reads (`_get_sort_indices_buffer`, `get_frame_plan`) now route through `state_view`.
    - `CompositeStage::execute(...)` query fetches (`get_output_compositor`, `get_rendering_device`) now route through `state_view`.
    - `render_sorted_splats_with_context(...)` query reads (`get_frame_plan`, `_get_sort_indices_buffer`) now route through `state_view`.
- Explicitly preserved for this slice:
  - No service-pointer narrowing.
  - No debug overlay migration.
  - No painterly direct-facade redesign.
  - No sorting-seam API work.
  - No mutating-stage provider signature changes.
- Rollback boundary:
  - Revert only query-only alias/read edits in `render_pipeline_stages.cpp` introduced by slice 2 (RP-2).
- Verification status:
  - `git diff --check` passed for the slice.
  - Native Windows verification passed:
    - Build: pass (incremental, `render_pipeline_stages.cpp` recompiled).
    - Guard lane: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).

### Explicit Phase Boundary: `gpu_sorting_pipeline.cpp`
- `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.cpp` is not a Phase `1b` provider-migration target.
- It remains explicitly deferred to Phase `2+3` sorting-seam cleanup because its current coupling is renderer-direct rather than `IFrameStateProvider`-direct.
- Phase `1b` may document its current read/write surface, but should not expand scope into sorting-pipeline API redesign.

## Minimal Composition-Root Abstraction Needed
Use a narrow service bundle for sorting-related wiring, without changing behavior:

- Keep existing interfaces:
  - `ISortResultSink`
  - `ISortBufferHostContext`
- Add one new execution interface (working name: `ISortExecutionContext`) exposing only currently-used execution dependencies:
  - ensure/get render device for sorting
  - access to required sort/frame/performance state
  - culler update hooks required by instance sorting path
  - sorter refresh hook
  - submission-device provider for async/sync readback paths
- Composition-root bundle (working name: `SortingWiringBundle`):
  - `ISortExecutionContext *exec`
  - `ISortBufferHostContext *buffer_host`
  - `ISortResultSink *result_sink`
  - `GPUSortingPipeline *pipeline`
  - `GPUCuller *gpu_culler`
  - existing cull/error callbacks already used by sorting orchestrator

This is the minimum coherent seam because it matches current dependencies used in `gpu_sorting_pipeline.cpp` and `render_sorting_orchestrator.cpp` without introducing speculative abstraction.

## Remaining GPUSortingPipeline APIs Coupled To GaussianSplatRenderer
Current renderer-dependent APIs/pathways to remove in staged migration:
- Header/API surface:
  - `ensure_sort_buffers(GaussianSplatRenderer *, uint32_t)`
  - `release_sort_buffers(GaussianSplatRenderer *)`
  - `sort_gaussians_gpu(GaussianSplatRenderer *, const Transform3D &)`
  - `_sort_instance_pipeline(GaussianSplatRenderer &, const Transform3D &)`
- Implementation coupling:
  - `set_sort_result_sink(&renderer)` + `set_sort_buffer_host_context(&renderer)`
  - direct reads/writes through renderer state in `_sort_instance_pipeline`

## Phased Refactor Plan

### Phase 0: Migration Ledger And Dependency Map (Done)
- Purpose:
  - Establish fresh architecture baseline from current code.
  - Freeze factual coupling model before any behavior-affecting refactor.
- Target files:
  - `docs/architecture/generated/*`
  - `docs/architecture/gaussian-renderer-refactor-memory.md`
- Exact dependencies:
  - `scripts/generate_architecture_diagrams.py`
- Compatibility risks:
  - Low (analysis/docs only).
- Verification strategy:
  - Regeneration succeeds.
  - Generated file set complete.
- What must not change:
  - Runtime behavior and public API.
- Rollback point:
  - Drop generated/doc changes only.

### Phase 1a: Read-Only Snapshots For Diagnostics/Monitoring
- Purpose:
  - Migrate query-only consumers away from broad mutable renderer state access.
- Target files:
  - `modules/gaussian_splatting/core/performance_monitors.cpp`
  - selected query paths in `render_diagnostics_orchestrator.cpp`
  - snapshot DTO/header definitions under `renderer/render_types/*` (new or existing location)
- Exact dependencies:
  - renderer snapshot assembly function(s) in facade
  - no mutator access from monitoring call sites
- Compatibility risks:
  - stale snapshot risk, missing fields in telemetry/HUD output
- Verification strategy:
  - compare monitor values pre/post on same deterministic frame
  - run tests that validate diagnostics fields in `test_renderer_pipeline.h`
- What must not change:
  - monitor names, semantics, and published values
  - telemetry cadence
- Rollback point:
  - keep snapshot types but switch consumers back to direct reads if values regress

### Phase 1b: Explicit Mutator APIs For Mixed Consumers + Provider Contract Hardening
- Purpose:
  - Split read/query interfaces from explicit state mutation commands.
  - Remove mutable-from-const provider shape.
- Target files:
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.h`
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
  - `modules/gaussian_splatting/interfaces/debug_overlay_system.cpp`
  - `modules/gaussian_splatting/interfaces/painterly_renderer.cpp`
  - `modules/gaussian_splatting/renderer/render_diagnostics_orchestrator.cpp`
- Exact dependencies:
  - new narrow mutator entrypoints on facade or dedicated command interfaces
  - separate read-only provider (`const` only) and mutator provider (`non-const` only), or equivalent split
- Compatibility risks:
  - accidental behavior shifts in overlay invalidation/HUD refresh ordering
  - painterly route regressions from changed state write sequencing
- Verification strategy:
  - existing debug overlay/painterly tests and runtime smoke scenes
  - assert no mutable fallback static path remains
- What must not change:
  - visible overlay/HUD behavior
  - painterly fallback semantics
  - facade external API contract
- Rollback point:
  - revert provider split and keep explicit mutator wrappers as adapters

### Phase 2 + Phase 3 (Coupled): Composition Root Cleanup + Sorting Legacy Renderer-Path Removal
- Purpose:
  - Remove renderer-centric sorting dependencies while simultaneously cleaning constructor/wiring ownership.
  - Avoid introducing temporary composition abstractions that do not reduce coupling.
- Target files:
  - `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.h`
  - `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.cpp`
  - `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline_interfaces.h`
  - `modules/gaussian_splatting/renderer/render_sorting_orchestrator.h/.cpp`
  - `modules/gaussian_splatting/renderer/render_streaming_orchestrator.h/.cpp`
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
- Exact dependencies:
  - `ISortResultSink`, `ISortBufferHostContext`, new `ISortExecutionContext`
  - composition root bundle for sorting wiring
  - no direct `GaussianSplatRenderer` type in sorting pipeline public API
- Compatibility risks:
  - sorting fallback route regressions
  - async readback timeline/wait semantics regressions
  - cross-device resource ownership regressions
- Verification strategy:
  - sort correctness checks under CPU fallback and GPU paths
  - instance pipeline scenes with async count readback
  - targeted tests in `test_renderer_pipeline.h` around sorting metrics and fallback flags
- What must not change:
  - sort output determinism tolerances
  - fallback policy behavior and diagnostics fields
  - renderer facade entrypoints for external callers
- Rollback points:
  - RP1: after adding interface adapters, before orchestrator callsite migration
  - RP2: after orchestrator migration, before deleting renderer overloads
  - RP3: before removing singleton fallback paths

### Phase 2 + Phase 3 implementation status (sorting seam batch 1, slice 19)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.h` / `.cpp` / `_interfaces.h`:
    - Removed public/legacy sorting entrypoints that took `GaussianSplatRenderer *` or `GaussianSplatRenderer &`.
    - Added `SortFrameContext` plus explicit `set_sort_frame_context(...)` / `clear_sort_frame_context()` so instance sorting reads only the state buckets and execution dependencies it actually needs.
    - `_sort_instance_pipeline(...)` now executes from `SortFrameContext` instead of reaching through the renderer facade.
  - `modules/gaussian_splatting/renderer/render_sorting_orchestrator.cpp`:
    - Added explicit host/sink binding helpers and `SortFrameContext` construction from the renderer-owned buckets.
    - Replaced removed `ensure_sort_buffers(renderer, ...)`, `release_sort_buffers(renderer)`, and `sort_gaussians_gpu(renderer, ...)` callsites with explicit host/context wiring plus the new context-less pipeline entrypoints.
    - Both the instance-only fast path and the common GPU-sort path now populate `SortFrameContext` before dispatch.
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`:
    - Sorting-pipeline shutdown now binds sink/host explicitly before releasing sort buffers, instead of using the deleted renderer-taking overload.
- Fixes required before accepting this batch:
  - Corrected `SortFrameContext` bucket types to the renderer-owned types used in the current branch.
  - Corrected interface include paths in `gpu_sorting_pipeline_interfaces.h` (`render_frame_context_manager.h`, `render_performance_types.h`, `render_state_types.h`) so Windows build graph resolution matched the new interface location.
  - Blocked instance-cache and instance-GPU fast paths while `sorting_state.sorter_needs_rebuild` is set, so forced algorithm/capacity changes cannot be bypassed by early returns.
- Explicitly preserved for this batch:
  - No composition-root callback cleanup yet.
  - No debug overlay, painterly, or test-hook redesign.
  - No public `GaussianSplatRenderer` facade entrypoint changes.
- Remaining caveat:
  - This closes the renderer-taking sorting API seam, but the broader composition-root cleanup is still pending for the next `2+3` batch.
  - Sorting still uses explicit host/sink/context wiring rather than the final bundled composition-root contract.
- Rollback boundary:
  - Revert only the sorting seam batch in:
    - `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.h`
    - `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.cpp`
    - `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline_interfaces.h`
    - `modules/gaussian_splatting/renderer/render_sorting_orchestrator.cpp`
    - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
- Verification status:
  - `git diff --check` passed for each landed fixup.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 2-3 --no-regen-architecture`.
  - Native Windows verification passed via `Gaussian Production Gates` run `23482731572` on commit `9bc9032b54`:
    - Build: pass.
    - Smoke tests: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).
    - Runtime harness: pass.
    - World-streaming gate: pass.
    - Large-scene benchmark gate: pass.
    - Eviction-churn benchmark gate: pass.

### Phase 2 + Phase 3 implementation status (composition-root batch 2, slice 20)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_streaming_orchestrator.h` / `.cpp`:
    - Added `RenderStreamingOrchestratorDependencies` with a narrow `RuntimePorts` bundle for the renderer callbacks this orchestrator still needs during streaming bootstrap and frame execution.
    - Replaced direct renderer method dispatch for device bootstrap, cull-projection construction/validation, instance-buffer clearing/upload, cull/sort pipeline dispatch, and cull radius/slack lookups with explicit runtime-port calls.
    - Stored and validated the runtime-port bundle in the orchestrator constructor so the seam is explicit and constructor-time failures are surfaced early.
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`:
    - Switched streaming orchestrator construction to the dependency bundle form instead of the old raw constructor argument mesh.
- Fixes required before accepting this batch:
  - Corrected `RuntimePorts` function-pointer signatures for `validate_cull_projection_contract(...)` and `run_cull_sort_pipeline_frame(...)`, including the qualified `GaussianSplatRenderer::RenderFallbackReason` type, after the first Windows build exposed the mismatch.
- Explicitly preserved for this batch:
  - No streaming logic redesign.
  - No service-pointer narrowing beyond the selected runtime-port calls above.
  - No sorting-seam, debug/tooling, painterly, or test-hook work.
  - No public `GaussianSplatRenderer` facade entrypoint changes.
- Remaining caveat:
  - `RenderStreamingOrchestrator` still retains many direct renderer/state interactions outside the new runtime-port bundle. This batch only extracts the constructor/runtime-callback seam; broader orchestrator dependency narrowing remains later-phase work.
- Rollback boundary:
  - Revert only:
    - `modules/gaussian_splatting/renderer/render_streaming_orchestrator.h`
    - `modules/gaussian_splatting/renderer/render_streaming_orchestrator.cpp`
    - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
- Verification status:
  - `git diff --check` passed.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 2-3 --no-regen-architecture`.
  - Native Windows verification passed via `Gaussian Production Gates` run `23484521277` on commit `458951d24c`:
    - Build: pass.
    - Smoke tests: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).
    - Runtime harness: pass.
    - World-streaming gate: pass.
    - Large-scene benchmark gate: pass.
    - Eviction-churn benchmark gate: pass.

### Phase 4: Orchestrator Dependency Narrowing
- Purpose:
  - Remove remaining broad renderer pointer dependencies from orchestrator internals.
- Target files:
  - `render_streaming_orchestrator.*`
  - `render_sorting_orchestrator.*`
  - potentially `render_data_orchestrator.*`, `render_output_orchestrator.*` for follow-up narrowing
- Exact dependencies:
  - small contracts/service bundles injected at construction
  - reduced callback surface with explicit ownership
- Compatibility risks:
  - initialization order regressions
  - null dependency handling regressions
- Verification strategy:
  - constructor-time dependency validation
  - render-loop smoke + stage metrics sanity checks
- What must not change:
  - orchestration order and frame-stage semantics
- Rollback point:
  - restore previous constructor signatures while preserving internal refactors that are behavior-neutral

### Phase 4 implementation status (output orchestrator batch 1, slice 21)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_output_orchestrator.h` / `.cpp`:
    - Replaced the old constructor callback/raw-pointer mesh with an explicit `Dependencies` bundle and nested `RuntimePorts` contract.
    - Routed output-path runtime callbacks through explicit ports for device bootstrap, texture-format lookup, viewport-format updates, and GPU-resource creation.
    - Switched practical frame/resource bucket access to local `FrameStateProvider` view/mutation aliases inside `render_for_view(...)`, `copy_final_texture_to_target(...)`, `commit_to_render_buffers(...)`, and `test_copy_final_output(...)`.
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`:
    - Updated output orchestrator construction to use the dependency bundle form.
- Explicitly preserved for this batch:
  - No sorting-seam work.
  - No debug/tooling or painterly redesign.
  - No public `GaussianSplatRenderer` facade entrypoint changes.
  - No attempt to force `view_state`, `device_state`, or render submission paths through speculative new interfaces.
- Remaining caveat:
  - `RenderOutputOrchestrator` still reaches through `renderer` directly for `view_state`, `device_state`, test-data presence checks, resource-owner lookup, and `render_gaussians(...)`. This batch narrows the orchestrator’s constructor/runtime seam and selected state-bucket access; it is not a full ownership inversion.
- Follow-up fix after initial acceptance:
  - Resolved the viewport-copy status getters against the live compositor from `FrameStateProvider`, so the copy path and its status readers observe the same compositor instance if renderer subsystem bindings change.
- Rollback boundary:
  - Revert only:
    - `modules/gaussian_splatting/renderer/render_output_orchestrator.h`
    - `modules/gaussian_splatting/renderer/render_output_orchestrator.cpp`
    - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
- Verification status:
  - `git diff --check` passed.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 4 --no-regen-architecture`.
  - Native Windows verification passed via `Gaussian Production Gates` run `23485345156` on commit `f92dbab666`:
    - Build: pass.
    - Smoke tests: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).
    - Runtime harness: pass.
    - World-streaming gate: pass.
    - Large-scene benchmark gate: pass.
    - Eviction-churn benchmark gate: pass.

### Phase 4 implementation status (state/control orchestrator batch 2, slice 22)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_quality_orchestrator.h` / `.cpp`:
    - Replaced the raw constructor parameter list with an explicit `Dependencies` bundle and narrow `RuntimePorts` contract for sorter refresh.
    - Routed mutable sorting/frame/performance bucket access in `set_max_splats(...)`, `set_quality_preset(...)`, and `cull_for_view(...)` through local `FrameStateProvider` view/mutation aliases instead of direct broad renderer state getters.
    - Kept `set_quality_preset(...)` lazy by only marking sorter rebuild needed; no eager sorter refresh is triggered from that setter.
  - `modules/gaussian_splatting/renderer/render_config_orchestrator.h` / `.cpp`:
    - Replaced the raw constructor parameter list with a `Dependencies` bundle for renderer, interactive state manager, and painterly renderer wiring.
  - `modules/gaussian_splatting/renderer/render_data_orchestrator.h` / `.cpp`:
    - Replaced the callback-heavy constructor signature with a `Dependencies` bundle.
    - Routed selected state/control reads and writes in `set_gaussian_data(...)`, `update_gpu_buffers_with_real_data(...)`, and static-chunk culler updates through local provider-backed view/mutation access.
  - `modules/gaussian_splatting/renderer/render_instancing_orchestrator.h` / `.cpp`:
    - Replaced the raw constructor parameter list with a `Dependencies` bundle.
    - Rebound readiness checks, per-instance frame-plan reads, and aggregate frame/sorting/performance writes through explicit provider-backed view/mutation access.
    - Resolved the output compositor live from provider-backed state at execution time instead of relying solely on the ctor-captured pointer.
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`:
    - Updated constructor wiring for quality, config, data, and instancing orchestrators to the new dependency-bundle forms.
  - `modules/gaussian_splatting/tests/test_renderer_pipeline.h`:
    - Updated the instancing orchestrator test helper to construct the orchestrator via the new `Dependencies` bundle after the constructor surface changed.
- Explicitly preserved for this batch:
  - No sorting-seam work.
  - No debug/tooling redesign.
  - No painterly redesign beyond constructor wiring already owned by the config batch surface.
  - No broad service-pointer narrowing or composition-root rewrite outside these orchestrator constructor/runtime seams.
  - No public `GaussianSplatRenderer` facade break.
- Remaining caveat:
  - This batch narrows selected constructor/runtime state-control dependencies, but it does not fully invert ownership. These orchestrators still rely on renderer-facing services and direct renderer helpers for some device, test-data, and submission paths.
- Follow-up fix after initial attempt:
  - Removed an unsafe `const_cast`-backed `FrameStateProvider` use from `GaussianSplatRenderer::get_async_upload_enabled() const` and reverted `set_quality_preset(...)` to lazy sorter rebuild semantics.
  - Initial Windows workflow run `23487001647` on commit `3e183a65dc` failed only at build time because `modules/gaussian_splatting/tests/test_renderer_pipeline.h` still constructed `RenderInstancingOrchestrator` with the old five-argument constructor form.
  - The batch was fixed by updating that stale test helper to the new `Dependencies` construction form in commit `1627e30a02`.
- Rollback boundary:
  - Revert only:
    - `modules/gaussian_splatting/renderer/render_quality_orchestrator.h`
    - `modules/gaussian_splatting/renderer/render_quality_orchestrator.cpp`
    - `modules/gaussian_splatting/renderer/render_config_orchestrator.h`
    - `modules/gaussian_splatting/renderer/render_config_orchestrator.cpp`
    - `modules/gaussian_splatting/renderer/render_data_orchestrator.h`
    - `modules/gaussian_splatting/renderer/render_data_orchestrator.cpp`
    - `modules/gaussian_splatting/renderer/render_instancing_orchestrator.h`
    - `modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp`
    - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
    - `modules/gaussian_splatting/tests/test_renderer_pipeline.h`
- Verification status:
  - `git diff --check` passed for the batch and its follow-up fix.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 4 --no-regen-architecture`.
  - Native Windows verification passed via `Gaussian Production Gates` run `23487132952` on commit `1627e30a02`:
    - Build: pass.
    - Smoke tests: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).
    - Runtime harness: pass.
    - World-streaming gate: pass.
    - Large-scene benchmark gate: pass.
    - Eviction-churn benchmark gate: pass.

### Phase 4 implementation status (resource orchestrator batch 3, slice 23)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_resource_orchestrator.h` / `.cpp`:
    - Replaced the raw constructor parameter list with an explicit `Dependencies` bundle for renderer, device state, and pipeline-feature storage.
    - Introduced local `FrameStateProvider` aliases in `initialize_shaders()`, `create_gpu_resources_safe()`, and `update_gpu_pass_metrics_from_tile_renderer()` so practical subsystem and performance-bucket reads/writes go through local view/mutation access instead of repeated direct renderer state fan-in.
    - Narrowed painterly/interactive shader setup and GPU timing metric publication to local provider-backed access where that reduced direct renderer-state reach-through.
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`:
    - Updated resource orchestrator construction to the dependency-bundle form.
- Explicitly preserved for this batch:
  - No sorting-seam work.
  - No diagnostics/debug-tooling redesign.
  - No painterly redesign.
  - No test-data upload redesign.
  - No tile-renderer/rasterizer ownership redesign.
  - No public `GaussianSplatRenderer` facade break.
- Remaining caveat:
  - `RenderResourceOrchestrator` still legitimately uses direct renderer helpers for device/bootstrap operations (`ensure_rendering_device`, submission/main device access), test-data upload/state, tile-renderer state, and resource-ownership bookkeeping. This batch narrows constructor/runtime dependency surfaces; it does not invert the actual resource-management ownership model.
- Rollback boundary:
  - Revert only:
    - `modules/gaussian_splatting/renderer/render_resource_orchestrator.h`
    - `modules/gaussian_splatting/renderer/render_resource_orchestrator.cpp`
    - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
- Verification status:
  - `git diff --check` passed for the batch.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 4 --no-regen-architecture`.
  - Native Windows verification passed via `Gaussian Production Gates` run `23487524729` on commit `b1a8430808`:
    - Build: pass.
    - Smoke tests: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).
    - Runtime harness: pass.
    - World-streaming gate: pass.
    - Large-scene benchmark gate: pass.
    - Eviction-churn benchmark gate: pass.

### Phase 4 implementation status (diagnostics/debug control-plane batch 4, slice 24)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_debug_state_orchestrator.h` / `.cpp`:
    - Replaced the raw constructor argument mesh with an explicit `Dependencies` bundle plus `RuntimePorts` for anomaly-dump operations that still need renderer-owned behavior (`dump_pipeline_trace_to_file`, `get_resource_owner`).
    - Preserved existing overlay query/command-sink usage while narrowing direct control-plane reach-through to local dependency access.
  - `modules/gaussian_splatting/renderer/render_diagnostics_orchestrator.h` / `.cpp`:
    - Replaced the raw constructor argument list with an explicit `Dependencies` bundle plus a narrow `RuntimePorts` callback for GPU pass metric refresh.
    - Kept frame-finalize ordering intact while routing the tile-renderer GPU metric refresh through the explicit runtime port instead of a direct renderer call.
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`:
    - Updated diagnostics/debug orchestrator construction to the dependency-bundle form and wired the explicit runtime ports.
- Explicitly preserved for this batch:
  - No sorting-seam redesign.
  - No painterly redesign.
  - No debug overlay redesign.
  - No public `GaussianSplatRenderer` facade break.
- Remaining caveat:
  - The diagnostics/debug orchestrators still legitimately reach renderer-owned live state and services where that remains the real ownership boundary; this batch narrows constructor/control-plane fan-in, it does not relocate those ownership responsibilities.
- Rollback boundary:
  - Revert only:
    - `modules/gaussian_splatting/renderer/render_debug_state_orchestrator.h`
    - `modules/gaussian_splatting/renderer/render_debug_state_orchestrator.cpp`
    - `modules/gaussian_splatting/renderer/render_diagnostics_orchestrator.h`
    - `modules/gaussian_splatting/renderer/render_diagnostics_orchestrator.cpp`
    - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
- Verification status:
  - `git diff --check` passed for the batch.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 4 --no-regen-architecture`.
  - Native Windows verification passed via `Gaussian Production Gates` run `23489071182` on commit `67f3018406`:
    - Build: pass.
    - Smoke tests: pass.
    - Module lane: pass (`GaussianSplatting` 144 tests / 4,066 assertions).
    - Runtime harness: pass.
    - World-streaming gate: pass.
    - Large-scene benchmark gate: pass.
    - Eviction-churn benchmark gate: pass.

### Phase 4 implementation status (diagnostics/debug state-path batch 5, slice 25)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_diagnostics_orchestrator.cpp`:
    - Moved frame/performance/sorting/resource/subsystem reads in production-metrics assembly, render-stats assembly, sort-history helpers, frame-transition bookkeeping, and runtime diagnostic snapshot building onto local `FrameStateProvider` view/mutation access.
    - Preserved frame-finalize ordering while keeping the explicit GPU-metric runtime port and debug overlay query/command seam unchanged.
  - `modules/gaussian_splatting/renderer/render_debug_state_orchestrator.cpp`:
    - Moved pipeline-trace frame stamping, cull-guardrail state reads, anomaly-dump frame/visible-count reads, and anomaly snapshot device/compositor resolution onto local `FrameStateProvider` view/mutation access.
    - Kept anomaly-dump side effects on the existing runtime ports.
- Explicitly preserved for this batch:
  - No sorting-seam redesign.
  - No painterly redesign.
  - No debug overlay redesign.
  - No constructor signature changes.
  - No public `GaussianSplatRenderer` facade break.
- Remaining caveat:
  - Direct reads of `view_state`, `painterly_config`, and `debug_config` remain where no matching read-only seam exists yet; this batch narrows state/config fan-in without inventing new interfaces just to remove those accesses.
- Rollback boundary:
  - Revert only:
    - `modules/gaussian_splatting/renderer/render_diagnostics_orchestrator.cpp`
    - `modules/gaussian_splatting/renderer/render_debug_state_orchestrator.cpp`
- Verification status:
  - `git diff --check` passed for the batch.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 4 --no-regen-architecture`.
  - Native Windows verification passed via `Gaussian Production Gates` run `23490508420` on commit `a11f35010c` (build, smoke tests, module lane, runtime validation, world-streaming gate, large-scene benchmark, eviction-churn benchmark).

### Phase 4 implementation status (sorting bootstrap/benchmark batch 6, slice 26)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/renderer/render_sorting_orchestrator.cpp`:
    - Moved frame/sorting/resource access in `refresh_gpu_sorter(...)`, `initialize_sorting()`, `run_sort_benchmark(...)`, and `benchmark_sorting_performance()` onto local `FrameStateProvider` view/mutation access where state buckets are the right seam.
    - Switched benchmark buffer allocation/device metadata to the local `RenderingDevice *` from `IFrameStateView`.
    - Consolidated benchmark queue-free cleanup through a tiny local helper that writes through `IFrameMutationAccess` to preserve the existing deletion-queue path.
- Explicitly preserved for this batch:
  - No sorting-seam redesign.
  - No `GPUSortingPipeline` signature changes.
  - No changes to `sort_gaussians_for_view(...)`, `force_sort_for_view(...)`, or sort-cache helpers.
  - No painterly/debug overlay/composition-root work.
  - No public `GaussianSplatRenderer` facade break.
- Remaining caveat:
  - Direct renderer access remains for non-frame buckets and owned service/control boundaries in these bootstrap paths, including device initialization, performance settings, test-data sizing, and host-context binding. This batch narrows state-bucket fan-in without pretending the whole sorting subsystem is decoupled.
- Rollback boundary:
  - Revert only:
    - `modules/gaussian_splatting/renderer/render_sorting_orchestrator.cpp`
- Verification status:
  - `git diff --check` passed for the batch.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 4 --no-regen-architecture`.
  - Native Windows verification passed via `Gaussian Production Gates` run `23491356476` on commit `f09286a0cb` (build, smoke tests, module lane, runtime validation, world-streaming gate, large-scene benchmark, eviction-churn benchmark).

### Phase 5: Lock Down Mutable Renderer State Access
- Purpose:
  - remove/deny broad mutable `get_*_state()` usage outside sanctioned mutator surfaces.
- Target files:
  - facade headers + main mutator consumers + any remaining offenders from ledger
- Exact dependencies:
  - read-only snapshots + explicit command surfaces already in place
- Compatibility risks:
  - test breakage where internals were mutated directly
- Verification strategy:
  - compile-time failures become migration checklist
  - targeted unit/integration test run
- What must not change:
  - external facade behavior
- Rollback point:
  - temporarily re-enable adapter methods with deprecation guards if migration gaps remain

### Phase 6: Optional Cleanup Of Thin Config-Style Wrappers
- Purpose:
  - remove temporary wrappers that no longer carry architectural value after coupling reduction.
- Target files:
  - only wrappers proven redundant by prior phases
- Exact dependencies:
  - completion of phases 1-5
- Compatibility risks:
  - low if constrained to dead/duplicative wrappers
- Verification strategy:
  - no public API diff
  - no behavior diff in diagnostics/tests
- What must not change:
  - no semantic changes disguised as cleanup
- Rollback point:
  - restore removed wrappers if any downstream consumer still relies on them

### Phase 1b.3 implementation status (debug query/command seam batch, slice 17)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/interfaces/debug_overlay_system.h` / `.cpp`:
    - Introduced explicit `DebugOverlayQueryView` and `DebugOverlayCommandSink` seam structs with builder helpers on `DebugOverlaySystem`.
    - Kept legacy `get_*` / `set_*` renderer helpers as compatibility delegates.
    - Query view exposes overlay options and HUD/tile-density snapshot access used by the debug-facing orchestrators.
    - Command sink forwards overlay/HUD invalidation and flag/opacity mutations through the existing compatibility helpers.
  - `modules/gaussian_splatting/interfaces/debug_overlay_macros.h`:
    - Debug orchestrator setter macro now uses `build_command_sink(renderer)` instead of direct legacy renderer helper calls.
  - `modules/gaussian_splatting/renderer/render_debug_state_orchestrator.cpp`:
    - Debug option application now reads from `build_query_view(renderer)` / `DebugOverlayOptions` rather than direct `debug_overlay_system` getters.
    - Direct debug-overlay setter paths now use `build_command_sink(renderer)` for overlay/HUD mutation.
  - `modules/gaussian_splatting/renderer/render_diagnostics_orchestrator.cpp`:
    - Render-stat aggregation now sources debug overlay options, HUD lines, and tile-density values from the new query view.
    - HUD invalidation / rebuild paths now use the new command sink overloads.
- Explicitly preserved for this slice:
  - No sorting-seam work.
  - No composition-root work.
  - No painterly redesign.
  - No broad service narrowing beyond the explicit debug overlay seam.
- Remaining caveat:
  - The legacy renderer helpers remain available as compatibility delegates for holdout callsites outside this batch.
- Rollback boundary:
  - Revert only the `debug_overlay_system` seam wrapper additions and the orchestrator callsite rewiring in the files above.
- Verification status:
  - `git diff --check` passed for the batch.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 1b.3 --no-regen-architecture`.
  - Native Windows verification passed via `Gaussian Production Gates` run `23479912351` on commit `349347af77` (build, smoke tests, module lane, runtime validation, world-streaming gate, large-scene benchmark, eviction-churn benchmark).

### Phase 1b.4 implementation status (test-hook migration batch, slice 18)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.h` / `.cpp`:
    - Added explicit test-only seams for streaming-system reset/readiness and output-compositor access (`test_release_current_streaming_system()`, `test_has_current_streaming_system()`, `test_get_output_compositor()`).
  - `modules/gaussian_splatting/interfaces/output_compositor.h` / `.cpp`:
    - Added a narrow test-only hook to reset viewport-copy bookkeeping without mutating `get_cache_state()` directly.
  - `modules/gaussian_splatting/tests/test_renderer_pipeline.h`:
    - Replaced direct streaming-system mutation with the explicit renderer test hooks.
    - Replaced direct `get_subsystem_state().output_compositor` access with the explicit test-named renderer seam.
    - Replaced direct output-cache mutation with the compositor test hook.
  - `modules/gaussian_splatting/tests/test_gaussian_splat_node.cpp`:
    - Replaced `get_scene_state().gaussian_data` assertions with the stable public `get_gaussian_data()` facade accessor.
- Explicitly preserved for this slice:
  - No production behavior changes.
  - No sorting-seam work.
  - No composition-root work.
  - No provider lock-down yet.
- Remaining caveat:
  - Tests still exercise `OutputCompositor` directly through an explicit test seam; this is intentional for `1b.4` and narrower than the previous subsystem-state reach-through.
- Rollback boundary:
  - Revert only the explicit test-seam additions in renderer/output compositor and the matching test callsite rewrites.
- Verification status:
  - `git diff --check` passed for the batch.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 1b.4 --no-regen-architecture`.
  - Native Windows verification passed via `Gaussian Production Gates` run `23480421484` on commit `f659aeff64` (build, smoke tests, module lane, runtime validation, world-streaming gate, large-scene benchmark, eviction-churn benchmark).

### Phase 1b.5 implementation status (provider lock-down completion batch, slice 19)
- Date: 2026-03-24
- Scope applied:
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.h` / `.cpp`:
    - Removed the legacy `IFrameStateProvider` compatibility surface.
    - Promoted `FrameStateProvider` to implement `IFrameStateView` + `IFrameMutationAccess` directly with explicit `*_view()` / `*_mut()` methods.
    - Removed `RenderFrameContext::state_provider`; stage-entry contexts now carry only explicit `state_view` and `mutation_access`.
    - Updated renderer entry paths that create frame contexts to populate the explicit view/mutation seams directly.
  - `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`:
    - Removed the `const_cast`-based legacy provider bridge from `_resolve_mutation_access(...)`.
    - Rebound frame-entry execution to explicit view/mutation seams only.
    - Final runtime fix: when `execute_frame_entry(...)` copies `RenderFrameContext`, it now always rebinds both seams against the copied `deps` object before raster/composite execution so `frame_plan` and state pointers cannot go stale.
  - `modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp`:
    - Updated instanced render entry to populate explicit `state_view` + `mutation_access` on per-instance frame contexts.
- Explicitly preserved for this slice:
  - No sorting-seam work.
  - No composition-root cleanup.
  - No debug/tooling redesign.
  - No painterly redesign.
  - No public `GaussianSplatRenderer` facade break.
- Remaining caveat:
  - `IFrameStateView` no longer hides mutable state buckets behind `const`, but it still exposes mutable service pointers (`OutputCompositor *`, `GPUCuller *`, `PainterlyRenderer *`, `GPUSortingPipeline *`, `RenderingDevice *`) until later seam narrowing.
- Rollback boundary:
  - Revert only the explicit provider-lockdown changes in `gaussian_splat_renderer.h`, `gaussian_splat_renderer.cpp`, `render_pipeline_stages.cpp`, and `render_instancing_orchestrator.cpp`.
- Verification status:
  - `git diff --check` passed for the batch.
  - Local phase checks passed via `python3 scripts/refactor_phase_runner.py local-checks --phase 1b.5 --no-regen-architecture`.
  - Initial Windows workflow run `23481131357` on commit `30fbd0efa3` failed only in the world-streaming runtime gate; the batch was fixed by rebinding frame-entry seams to the copied `RenderFrameContext::deps` in `render_pipeline_stages.cpp`.
  - Native Windows verification passed via `Gaussian Production Gates` run `23481647454` on commit `620996d67e` (build, smoke tests, module lane, runtime validation, world-streaming gate, large-scene benchmark, eviction-churn benchmark).

## Tests That Mutated Internals And The Current Replacement Hooks
Replaced in `1b.4`:
- Streaming-system teardown in `test_renderer_pipeline.h` now uses `GaussianSplatRenderer::test_release_current_streaming_system()` and `test_has_current_streaming_system()`.
- Output-compositor reach-through in `test_renderer_pipeline.h` now uses `GaussianSplatRenderer::test_get_output_compositor()`.
- Viewport-copy cache resets in `test_renderer_pipeline.h` now use `OutputCompositor::test_reset_last_viewport_copy_state()`.
- Scene-data assertions in `test_gaussian_splat_node.cpp` now use the stable public `GaussianSplatRenderer::get_gaussian_data()` facade.

Rules for remaining test seams:
- keep hooks test-only and explicitly named
- do not expose broad mutable production state
- preserve existing test intent and assertions
- prefer read-only snapshots over additional subsystem reach-through if more output-compositor coverage is needed later

## Non-Negotiables During Implementation
- `GaussianSplatRenderer` remains the stable public facade.
- No orchestrator collapse back into renderer.
- No full rewrite.
- No “interface explosion”; add contracts only where coupling is proven.
- Diagnostics/HUD/performance monitors/tests/editor integration remain first-class compatibility surfaces.
- If a phase touches too many consumers at once, stop and resequence instead of pushing through.

## Decision Log (Why This Sequence)
1. Start with fresh generated artifacts because stale architecture reports lead to wrong sequencing.
2. Split phase 1 into `1a` and `1b` because observability consumers are mixed (query + mutation).
3. Handle mutable-from-const provider contracts early because they undermine all later snapshot guarantees.
4. Couple composition-root cleanup with sorting seam removal so constructor cleanup tracks real dependency direction change.
5. Delay strict mutable access lock-down until replacement seams and test hooks exist, to avoid broad breakage.

## Implementation Entry Criteria
Before any code phase starts:
- update this memory file with phase-specific migration map and selected consumer set
- list exact callsites being moved in that phase
- identify read-only vs mutator consumers explicitly
- define rollback commit boundary before first edit

## Phase 4.1 Implementation Status (sorting bootstrap + diagnostics only)

### Scope applied
- `modules/gaussian_splatting/renderer/render_sorting_orchestrator.cpp`
  - Bootstrapping and benchmark/diagnostic entry points now bind the renderer-owned frame buckets through the existing `FrameStateProvider` seam where it already fits:
    - `refresh_gpu_sorter`
    - `initialize_sorting`
    - `run_sort_benchmark`
    - `benchmark_sorting_performance`
  - The batch keeps backoff timing, sorter rebuild flow, benchmark math, logging, and cleanup behavior intact.
  - A tiny local helper centralizes benchmark buffer queue-free so the resource cleanup path stays identical.
- `modules/gaussian_splatting/renderer/render_sorting_orchestrator.h`
  - No signature changes were required for this batch.
- `docs/architecture/gaussian-renderer-refactor-memory.md`
  - Added this batch note and rollback boundary.

### Explicitly preserved
- No changes to `sort_gaussians_for_view`.
- No changes to `force_sort_for_view`.
- No cache-helper changes.
- No `GPUSortingPipeline` signature changes.
- No sorting-seam redesign.
- No painterly, debug-overlay, or composition-root changes.

### Caveat
- This is dependency narrowing, not full sorting decoupling.
- The target functions still use direct renderer access for things the current provider does not expose, such as device initialization and non-frame buckets like performance settings and test data.
- That is deliberate; I did not widen the seam just to remove every renderer getter.

### Rollback boundary
- Revert only:
  - `modules/gaussian_splatting/renderer/render_sorting_orchestrator.cpp`
  - `docs/architecture/gaussian-renderer-refactor-memory.md`
- If this batch regresses, do not roll back unrelated renderer or pipeline work.

## Phase 4.2 Implementation Status (sorting execution dependency narrowing, slice 27)

### Scope applied
- `modules/gaussian_splatting/renderer/render_sorting_orchestrator.cpp`
  - Routed `sort_gaussians_for_view(...)` through a local `FrameStateProvider` plus explicit `IFrameStateView` / `IFrameMutationAccess` aliases.
  - Moved route-UID publication, visible-count publication, and sort-metric resets/timings onto local frame/perf/debug references, with tiny local helpers for the repeated metric/count updates.
  - Preserved strict-global-sort fallback behavior, cache reuse, cull-signature tracking, and the existing device/buffer ownership flow.
  - Kept `force_sort_for_view(...)` focused on projection/viewport preparation and the streaming fallback path; it now uses local view-state aliasing for the read side without widening the seam.
- `docs/architecture/gaussian-renderer-refactor-memory.md`
  - Added this batch note and verification status.

### Explicitly preserved
- No `GPUSortingPipeline` signature changes.
- No sorting-seam redesign.
- No painterly/debug overlay/composition-root work.
- No broad new interface just to remove every remaining renderer getter.

### Caveat
- `force_sort_for_view(...)` still reaches through `renderer` for the current view-state and streaming orchestrator because the existing frame provider seam does not expose a view-state contract. That is intentional for this slice.

### Rollback boundary
- Revert only:
  - `modules/gaussian_splatting/renderer/render_sorting_orchestrator.cpp`
  - `docs/architecture/gaussian-renderer-refactor-memory.md`

### Verification status
- `git diff --check` passed.
- `python3 scripts/refactor_phase_runner.py local-checks --phase 4 --no-regen-architecture` passed.
- Native Windows verification passed via `Gaussian Production Gates` run `23491967749` on commit `a931f3a1e6` (build, smoke tests, module lane, runtime validation, world-streaming gate, large-scene benchmark, eviction-churn benchmark).

## Phase 4.3 Implementation Status (sorting helper/cache support-path narrowing, slice 28)

### Scope applied
- `modules/gaussian_splatting/renderer/render_sorting_orchestrator.cpp`
  - Narrowed `_set_instance_sort_inputs(...)` to consume explicit instance-pipeline buffers plus the render device from the existing frame-state seam instead of pulling the device from `renderer`.
  - Narrowed `_build_sort_frame_context(...)` to build `SortFrameContext` from the explicit `IFrameStateView` / `IFrameMutationAccess` seams and a passed-in view-state reference, instead of reading all frame/perf/device fields from `renderer`.
  - Routed instance-sort cache hit/miss publication through a local `FrameStateProvider` inside the cache helper so cache metrics no longer write to `renderer->get_performance_state()` directly.
  - Kept route UID, visible-count, timing, strict-global-sort, cache-reuse, and device/buffer semantics unchanged.
- `docs/architecture/gaussian-renderer-refactor-memory.md`
  - Added this batch note and verification status.

### Explicitly preserved
- No `GPUSortingPipeline` signature changes.
- No composition-root work.
- No diagnostics/debug overlay work.
- No test-hook changes.
- No broad renderer facade redesign.

### Caveat
- The residual instance-pipeline buffer lookup in `sort_gaussians_for_view(...)` still comes from `renderer` because the current frame-state seam does not expose instance buffer ownership. That is intentionally outside the seam narrowed in this slice.

### Rollback boundary
- Revert only:
  - `modules/gaussian_splatting/renderer/render_sorting_orchestrator.cpp`
  - `docs/architecture/gaussian-renderer-refactor-memory.md`

### Verification status
- `git diff --check` passed.
- `python3 scripts/refactor_phase_runner.py local-checks --phase 4 --no-regen-architecture` passed.
- Native Windows verification passed via `Gaussian Production Gates` run `23492944688` on commit `1e1c5d28eb` (build, smoke tests, module lane, runtime validation, world-streaming gate, large-scene benchmark, eviction-churn benchmark).

## Phase 4.4 Implementation Status (sorting bootstrap/benchmark lifecycle narrowing, slice 29)

### Scope applied
- `modules/gaussian_splatting/renderer/render_sorting_orchestrator.h` / `.cpp`
  - Introduced an explicit `Dependencies` bundle for the sorting orchestrator bootstrap and benchmark lifecycle paths.
  - Routed sorter refresh, initialization, and benchmark device bootstrap through explicit `performance_settings`, `test_data_state`, `device_state`, and `ensure_rendering_device` dependencies instead of repeated renderer reach-through in those paths.
  - Preserved sorter rebuild, benchmark timing/output, and failure/fallback behavior.
- `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
  - Updated sorting orchestrator construction to supply the explicit lifecycle dependencies.

### Explicitly preserved
- No `GPUSortingPipeline` signature changes.
- No changes to the already-closed core sort execution path beyond constructor wiring.
- No route UID behavior changes.
- No diagnostics/debug overlay work.
- No composition-root redesign.
- No resource/output/painterly/quality/test changes.

### Caveat
- This batch narrows bootstrap/benchmark lifecycle fan-in, not the remaining execution-path reads that still legitimately depend on renderer-owned services and buffers.

### Rollback boundary
- Revert only:
  - `modules/gaussian_splatting/renderer/render_sorting_orchestrator.h`
  - `modules/gaussian_splatting/renderer/render_sorting_orchestrator.cpp`
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
  - `docs/architecture/gaussian-renderer-refactor-memory.md`

### Verification status
- `git diff --check` passed.
- `python3 scripts/refactor_phase_runner.py local-checks --phase 4 --no-regen-architecture` passed.
- Native Windows verification passed via `Gaussian Production Gates` run `23495754350` on commit `72d89b1884` (build, smoke tests, module lane, runtime validation, world-streaming gate, large-scene benchmark, eviction-churn benchmark).

## Phase 4.5 Implementation Status (output residual dependency narrowing, slice 30)

### Scope applied
- `modules/gaussian_splatting/renderer/render_output_orchestrator.h` / `.cpp`
  - Added explicit `view_state` and `test_data_state` dependencies so output-path camera/viewport bookkeeping and test-data emptiness checks no longer reach through broad renderer getters.
  - Added output-specific runtime ports for `get_resource_owner(...)` and `render_gaussians(...)` so `render_for_view(...)` narrows resource-owner lookup and dispatch through explicit behavior seams instead of direct renderer helper calls.
  - Switched `copy_final_texture_to_target(...)`, `render_for_view(...)`, and `test_copy_final_output(...)` to use the explicit state pointers and provider-backed rendering-device lookup for compositor initialization.
- `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
  - Updated output orchestrator construction to supply the narrowed output-state pointers and runtime ports.

### Explicitly preserved
- No sorting-seam work.
- No resource-orchestrator changes.
- No painterly redesign.
- No public `GaussianSplatRenderer` facade entrypoint changes.
- No Phase 5 mutable-access lockdown work.

### Caveat
- This batch narrows the residual output seam, but it intentionally leaves painterly pass-graph use, GPU culler interaction, and renderer-owned viewport-format behavior on their current contracts.

### Rollback boundary
- Revert only:
  - `modules/gaussian_splatting/renderer/render_output_orchestrator.h`
  - `modules/gaussian_splatting/renderer/render_output_orchestrator.cpp`
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
  - `docs/architecture/gaussian-renderer-refactor-memory.md`

### Verification status
- `git diff --check` passed.
- `python3 scripts/refactor_phase_runner.py local-checks --phase 4 --no-regen-architecture` passed.
- Native Windows verification passed via `Gaussian Production Gates` run `23496475589` on commit `69ade68383` (build, smoke tests, module lane, runtime validation, world-streaming gate, large-scene benchmark, eviction-churn benchmark).

## Phase 4.6 Implementation Status (resource dependency-bundle narrowing, slice 31)

### Scope applied
- `modules/gaussian_splatting/renderer/render_resource_orchestrator.h` / `.cpp`
  - Added an explicit dependency bundle for resource setup paths: `performance_settings`, `painterly_config`, `debug_config`, `test_data_state`, `tile_renderer_state`, and `subsystem_state`.
  - Added resource-specific runtime ports for renderer-owned behavior that still belongs on the facade: `ensure_rendering_device(...)`, `get_submission_device()`, `get_main_rendering_device()`, `refresh_gpu_sorter(...)`, `track_resource_owner(...)`, and `free_owned_resource(...)`.
  - Rewired `initialize_shaders()`, `create_gpu_resources_safe()`, and `load_graphics_shader()` to use those explicit dependencies and ports instead of repeated direct `renderer->get_*` reach-through.
  - Kept `update_gpu_pass_metrics_from_tile_renderer()` bounded to the current metric seam; it still uses local provider access where metric mutation remains the correct contract.
- `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
  - Updated resource orchestrator construction to supply the narrowed dependency bundle and runtime ports.

### Explicitly preserved
- No sorting-seam changes.
- No painterly redesign.
- No debug-overlay redesign.
- No Phase 5 mutable-access lock-down.
- No public `GaussianSplatRenderer` facade entrypoint changes.

### Caveat
- This batch narrows the large resource-setup fan-in, but it does not invert ownership for painterly or interactive subsystem behavior. Calls that inherently require the renderer facade still use the explicit runtime ports or existing subsystem APIs.
- `update_gpu_pass_metrics_from_tile_renderer()` still uses local provider access for metric mutation and rasterizer timing access; that seam is intentionally left for the later hardening phase.

### Rollback boundary
- Revert only:
  - `modules/gaussian_splatting/renderer/render_resource_orchestrator.h`
  - `modules/gaussian_splatting/renderer/render_resource_orchestrator.cpp`
  - `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp`
  - `docs/architecture/gaussian-renderer-refactor-memory.md`

### Verification status
- `git diff --check` passed.
- `python3 scripts/refactor_phase_runner.py local-checks --phase 4 --no-regen-architecture` passed.
- Native Windows verification passed via `Gaussian Production Gates` run `23497309698` on commit `74aa214511` (build, smoke tests, module lane, runtime validation, world-streaming gate, large-scene benchmark, eviction-churn benchmark).
