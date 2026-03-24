# Gaussian Renderer Refactor Memory

Last updated: 2026-03-23 (Europe/Berlin)  
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
  - Full native Windows build/test rerun pending external lane execution (Claude build runner).

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

## Tests That Currently Mutate Internals And Replacement Hooks
Current direct-mutation examples:
- `test_renderer_pipeline.h` mutates streaming internals (`get_streaming_state().current_streaming_system.unref()`).
- `test_renderer_pipeline.h` mutates compositor cache internals (`output_compositor->get_cache_state()`).

Proposed replacement hooks:
1. `test_set_streaming_system_for_test(Ref<GaussianStreamingSystem>)`
2. `test_clear_streaming_system_for_test()`
3. `test_get_output_cache_snapshot() const` (read-only)
4. `test_override_output_cache_for_test(const OutputCacheOverride &)`
5. Optional `friend` test harness type for strictly-scoped mutation where unavoidable

Rules for replacement hooks:
- keep hooks test-only (guarded naming and location)
- do not expose broad mutable production state
- preserve existing test intent and assertions

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
