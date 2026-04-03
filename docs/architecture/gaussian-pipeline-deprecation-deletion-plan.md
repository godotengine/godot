# Gaussian Pipeline Deprecation And Deletion Plan

Status: living post-refactor cleanup plan  
Scope: compatibility removal, route consolidation, and API narrowing after Stages 0-3  
Audience: rendering, tools, editor, and test engineers

## Purpose

The refactor achieved the main architectural goal:

`source asset or baked world -> shared renderer contract -> resident or streaming backend -> sort/render`

But it did not finish the aggressive cleanup pass.

This document tracks what still exists only for compatibility, migration, tests, or editor convenience, and defines what should be:

- kept as a supported feature
- narrowed and explicitly marked low-level
- deprecated with a replacement workflow
- deleted once preconditions are met

This is not a new architecture proposal. It is the closeout plan for code and route consolidation that the refactor intentionally deferred.

## What The Refactor Already Consolidated

The main structural split is already fixed:

- `GaussianSplatWorld3D` no longer owns the runtime render route. It submits through the director in [gaussian_splat_world_3d.cpp](../../modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp).
- The shared instance renderer can now run resident data without a `GaussianStreamingSystem`, with backend choice resolved through [gs_project_settings.h](../../modules/gaussian_splatting/core/gs_project_settings.h) and the resident atlas publisher.
- Route/debug/effective-settings surfaces now explain the active path and provenance through existing HUD/editor/diagnostics surfaces.
- The editor no longer dual-writes `splat_asset` and `ply_file_path` during normal import flows.

That means the remaining cleanup work is concentrated in compatibility seams, not the core render architecture.

## Cleanup Principles

1. Do not reopen the accepted architecture.
2. Do not delete real runtime features until replacement workflows exist.
3. Prefer narrowing a surface before deleting it outright.
4. Migrate tests before removing the hook they rely on.
5. Keep one canonical editor workflow:
   imported `GaussianSplatAsset` -> shared renderer registration.
6. Keep one canonical backend policy surface:
   resolved `route_policy`, not overlapping booleans.

## Inventory Summary

| Surface | Current role | Category | Recommended action | Earliest bucket |
| --- | --- | --- | --- | --- |
| `rendering/gaussian_splatting/streaming/enabled` | Removed legacy compatibility toggle | Removed in Bucket B | Route policy is now the sole supported backend-policy setting | Complete |
| `upsert_world_submission()` / `unregister_world_submission()` | Removed scaffolding-only storage helpers | Removed in follow-up cleanup | Runtime world submission now goes through `submit_world_submission()` / `release_world_submission()` only | Complete |
| Editor preview direct `renderer->set_gaussian_data()` | Editor-only bypass of canonical submission path | Editor compatibility route | Keep as the single accepted editor-only preview exception | Bucket A |
| `GaussianSplatNode3D::ply_file_path` | Raw file load, auto-load, drag/drop, validation | Runtime and editor compatibility route | Demote from editor-primary path first; later decide whether runtime raw-load survives | Bucket C |
| `GaussianSplatDynamicInstance3D::ply_file_path` | Raw file to `GaussianData` bypass | Runtime compatibility route | Deprecate earlier than normal node path; require `splat_asset` or explicit `GaussianData` later | Bucket B/C |
| `GaussianSplatRenderer::set_gaussian_data()` | Low-level renderer/test/tool hook | Public low-level API | Keep for now, narrow usage contract, remove only after replacements exist | Bucket C |
| `GaussianSplatContainer::apply_to_renderer()` | Container-level direct renderer bypass | Low-level convenience API | Demote in README/API docs now; keep only while the direct raw-data renderer route still exists | Bucket C |
| Duplicated source-path resolution helpers | Same logic in node/editor code | Internal duplication | Consolidate into one shared helper | Bucket A |
| Explicit-resident legacy resident fallback | Accepted runtime fallback when resident atlas publish is infeasible | Runtime compatibility behavior | Keep until resident atlas covers explicit-resident edge cases or the fallback is intentionally retired | Bucket C |

## Deletion Buckets

### Bucket A: Safe Cleanup And Narrowing

These changes are low-risk and should not change user-visible behavior.

- Clarify public-vs-scaffolding API boundaries in the director.
- Consolidate duplicated source-path helper logic.
- Keep the direct preview upload as the single accepted editor-only exception.
- Remove dead preview-submission scaffolding.

### Bucket B: Compatibility Surface Removal

These changes are structurally safe but require test and project migration.

- Remove legacy `streaming/enabled` compatibility once projects/tests stop depending on it. Completed in Bucket B.
- Remove world-submission scaffolding APIs once tests stop relying on them.
- Deprecate `GaussianSplatDynamicInstance3D::ply_file_path`.

### Bucket C: Product Workflow Deprecation

These are the risky removals because they still back real workflows.

- Demote or remove `GaussianSplatNode3D::ply_file_path` as an editor-primary route.
- Decide the long-term fate of `GaussianSplatRenderer::set_gaussian_data()`.
- Decide whether explicit-resident legacy fallback should remain supported.

## Detailed Surfaces

### 1. Legacy `streaming/enabled`

Canonical files:

- [gs_project_settings.h](../../modules/gaussian_splatting/core/gs_project_settings.h)
- [gaussian_splat_manager.cpp](../../modules/gaussian_splatting/core/gaussian_splat_manager.cpp)
- [effective_config_snapshot.h](../../modules/gaussian_splatting/core/effective_config_snapshot.h)

Current state:

- Renderer/orchestrator code consumes the resolved enum from `gs::settings::get_streaming_route_policy()`.
- `rendering/gaussian_splatting/streaming/enabled` has been removed from module settings registration and no longer participates in route resolution.

Bucket B result:

1. Tests now assert enum `route_policy` behavior directly.
2. The bool has been removed from project settings registration.
3. `get_streaming_route_policy()` and `get_streaming_route_policy_source()` no longer translate or label a legacy bool path.
4. Diagnostics no longer expose `legacy_streaming_enabled_forced_resident`.

Deletion criteria:

- No tests rely on `streaming/enabled`.
- No shipped project configs rely on the bool.
- Diagnostics no longer need to distinguish the legacy source.

### 2. `upsert_world_submission()` / `unregister_world_submission()`

Canonical files:

- [gaussian_splat_scene_director.h](../../modules/gaussian_splatting/core/gaussian_splat_scene_director.h)
- [gaussian_splat_scene_director.cpp](../../modules/gaussian_splatting/core/gaussian_splat_scene_director.cpp)

Current state:

- `submit_world_submission()` / `release_world_submission()` are now the only public world-submission entrypoints.
- The scaffolding-only `upsert_*` / `unregister_*` helpers have been removed.

Result:

1. Runtime world submission remains covered through `submit_*` / `release_*` tests.
2. The director no longer exposes a metadata-only world-submission path on the public tools/test surface.

Deletion criteria:

- Complete.

### 3. Editor Preview Direct Upload

Canonical files:

- [gaussian_editor_plugin.cpp](../../modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp)
- [gaussian_splat_scene_director.h](../../modules/gaussian_splatting/core/gaussian_splat_scene_director.h)
- [gaussian_splat_scene_director.cpp](../../modules/gaussian_splatting/core/gaussian_splat_scene_director.cpp)

Current state:

- Editor preview still does direct bare-renderer upload through `current_renderer->set_gaussian_data(...)`.
- The unused director preview-submission scaffolding has been removed.

Decision:

- Direct preview upload remains the single supported preview path.

Rationale:

- No runtime bug currently requires preview migration.
- Preview-submission scaffolding had no production caller and only added a second competing preview path.

Future reconsideration:

- If preview ever moves into the director, it should be introduced as a dedicated runtime preview path rather than by reviving the old scaffolding API unchanged.

### 4. `GaussianSplatDynamicInstance3D::ply_file_path`

Canonical files:

- [gaussian_splat_dynamic_instance_3d.h](../../modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.h)
- [gaussian_splat_dynamic_instance_3d.cpp](../../modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp)

Current state:

- This class can still load raw file data directly into `GaussianData`.
- It then converges on instance registration, but it bypasses imported assets entirely.
- Bucket B marks the path as deprecated in runtime/docs while keeping it functional.

Why it is a better early deprecation target than the normal node path:

- It is less editor-facing.
- It already supports `splat_asset` and explicit `gaussian_data`.
- It is a stronger bypass of the canonical asset model than the main node.

Blockers:

- Public API and class docs still expose `ply_file_path`.
- Tests still cover direct data / direct file behavior.

Removal plan:

1. Deprecate `ply_file_path` in docs and class comments.
2. Emit a deprecation warning when the property is set.
3. Keep `set_gaussian_data()` and `set_splat_asset()` as supported alternatives.
4. Remove `_load_from_file()` and the `ply_file_path` property once tests/docs migrate.

Deletion criteria:

- All supported workflows can use `splat_asset` or explicit `gaussian_data`.
- No product flow depends on dynamic-instance raw file loading.

### 5. `GaussianSplatNode3D::ply_file_path`

Canonical files:

- [gaussian_splat_node_3d.h](../../modules/gaussian_splatting/nodes/gaussian_splat_node_3d.h)
- [gaussian_splat_node_3d.cpp](../../modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp)
- [gaussian_splat_node_helpers.cpp](../../modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp)

Current state:

- `ply_file_path` is still a real public runtime and editor route.
- It still powers:
  - auto-load on enter tree
  - direct runtime loading
  - validation/warnings
- Raw drag-and-drop now prefers asset-backed assignment and only falls back to `ply_file_path` if the asset-backed load fails.

Why it cannot simply be deleted now:

- The replacement workflow is not fully designed.
- The accepted refactor plan explicitly deferred this until replacements exist.

Recommended deprecation sequence:

1. Remove editor-primary behaviors first:
   - keep drag-and-drop asset-first and avoid reintroducing `ply_file_path` as the primary editor assignment path
   - steer users toward imported assets or explicit import tools
2. Keep runtime/script loading temporarily.
3. Introduce a clearer external-file workflow if runtime direct loading remains important.
4. Only then decide whether `ply_file_path` becomes:
   - metadata only, or
   - a supported runtime-only escape hatch

Blockers:

- Tests in [test_gaussian_splat_node.h](../../modules/gaussian_splatting/tests/test_gaussian_splat_node.h) still treat it as supported behavior.
- Class docs and examples still show `ply_file_path`.

Deletion criteria:

- Drag/drop, runtime load, auto-load, and validation all have replacement workflows.
- Class docs no longer present `ply_file_path` as the normal workflow.
- Tests are migrated away from direct-file-as-primary-authoring-path assumptions.

### 6. `GaussianSplatRenderer::set_gaussian_data()`

Canonical files:

- [gaussian_splat_renderer.h](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.h)
- [render_data_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_data_orchestrator.cpp)
- [gaussian_splat_scene_director.cpp](../../modules/gaussian_splatting/core/gaussian_splat_scene_director.cpp)
- [render_streaming_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_streaming_orchestrator.cpp)
- [gpu_culler.cpp](../../modules/gaussian_splatting/interfaces/gpu_culler.cpp)

Current state:

- The public low-level setter still exists and is still used by tests, tools, helpers, and editor preview.
- The underlying non-instance `gaussian_data` route is also still load-bearing in production runtime flows.

Investigation result:

- Do not remove the direct non-instance `gaussian_data` route yet.
- The correct near-term action is to narrow callers and document the remaining blockers, not to delete the route.

Verified blockers:

- World-backed runtime still realizes submissions through primary scene data and static chunks in the director.
- Streaming bootstrap for world/static/no-instance scenes still depends on primary `scene_state.gaussian_data`.
- The accepted explicit-resident legacy fallback still depends on the legacy primary-data route when resident atlas publication is rejected.
- The legacy non-instance cull/sort path is therefore still live in `GPUCuller` and sorting, even though normal node and dynamic-instance flows already use the shared instance/director path.

Secondary callers that should not drive the keep/remove decision by themselves:

- Editor preview direct upload in [gaussian_editor_plugin.cpp](../../modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp)
- [GaussianSplatContainer::apply_to_renderer()](../../modules/gaussian_splatting/nodes/gaussian_splat_container.cpp)

Why it still exists:

- It is still required by real runtime behavior, not just tests.
- It remains useful for focused low-level validation while the runtime route still exists.

Recommended action:

- Do not delete it yet.
- Narrow its contract instead:
  - document `set_gaussian_data()` as a low-level/test/tool/editor-preview API, not the canonical high-level scene path
  - demote `GaussianSplatContainer::apply_to_renderer()` in README/API docs to legacy low-level convenience usage
  - migrate or explicitly bless editor preview
  - retire container-level convenience usage
  - keep high-level scene code off it
  - remove the underlying runtime route only after world submission realization, streaming bootstrap, and explicit-resident fallback no longer depend on primary `gaussian_data`

Longer-term options:

- keep it permanently as the renderer's low-level data hook, or
- replace it with a more explicitly named test/tool upload API later

Deletion criteria:

- World-backed runtime no longer realizes submissions through primary `gaussian_data`.
- Streaming bootstrap no longer depends on primary raw scene data or synthetic primary fallback instances.
- Explicit-resident fallback no longer needs the legacy non-instance route.
- Editor preview no longer depends on it directly.
- Low-level tests have replacement helpers or intentionally keep this as a supported low-level API.

### 7. Duplicated Source-Path Resolution

Canonical files:

- [gaussian_editor_plugin.cpp](../../modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp)
- [gaussian_splat_node_3d.cpp](../../modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp)

Current state:

- Source-path resolution logic exists in more than one place.
- The accepted plan already tracked this as a consolidation candidate.

Recommended action:

- Extract one shared helper for:
  - asset source metadata lookup
  - node source selection precedence

Deletion criteria:

- One helper owns source-path precedence.
- Editor and node warnings/origin labels share the same rules.

### 8. Explicit-Resident Legacy Fallback

Canonical files:

- [gaussian_splat_renderer.cpp](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp)
- [gaussian-resident-instanced-contract.md](gaussian-resident-instanced-contract.md)

Current state:

- If resident atlas publication is rejected for explicit-resident requests, the renderer may still fall back to the legacy resident path.
- This is accepted current behavior, not a bug.

Why it is not a deletion target yet:

- Removing it would change runtime behavior for explicit resident requests.
- The resident atlas publisher still intentionally rejects some cases, including per-chunk quantization.

Recommended action:

- Keep it for now.
- Keep the explicit-resident quantization fallback test as baseline coverage.
- Revisit only after resident atlas coverage is wider and the legacy resident path is truly redundant.

Deletion criteria:

- Explicit-resident use cases no longer need the fallback.
- The fallback has dedicated coverage proving it is safe to remove or intentionally preserve.

## Tests That Currently Block Deletion

The current test suite still locks in several compatibility surfaces:

- `ply_file_path` as a supported node workflow:
  [test_gaussian_splat_node.h](../../modules/gaussian_splatting/tests/test_gaussian_splat_node.h)
- diagnostic chaining and mixed-hint behavior:
  [test_integration.cpp](../../modules/gaussian_splatting/tests/test_integration.cpp),
  [test_scene_director_submission_scaffolding.h](../../modules/gaussian_splatting/tests/test_scene_director_submission_scaffolding.h)

This is useful: it tells us exactly what must migrate before deletion is honest rather than accidental.

## Recommended Execution Order

### Pass 1: Safe Consolidation

- consolidate source-path helper logic
- keep world scaffolding APIs behind tools/test-only compilation
- keep direct preview as the documented editor-only exception

### Pass 2: Remove Easy Compatibility Debt

- remove legacy `streaming/enabled` after settings/test migration
- update docs and diagnostics labels accordingly
- narrow world scaffolding to test-only compilation
- deprecate `GaussianSplatDynamicInstance3D::ply_file_path`

### Pass 3: Product Workflow Deprecation

- deprecate `GaussianSplatDynamicInstance3D::ply_file_path`
- demote `GaussianSplatNode3D::ply_file_path` from editor-primary path
- remove raw drag-and-drop to `ply_file_path`
- decide whether runtime raw loading remains supported or is replaced

### Pass 4: Optional Further Narrowing

- decide whether `GaussianSplatRenderer::set_gaussian_data()` stays as a supported low-level API
- decide whether explicit-resident legacy fallback remains a supported compatibility behavior

## Completion Criteria

The original consolidation goal should be considered complete only when:

- no legacy boolean and enum route knobs overlap
- no public scaffolding APIs remain in the director without a real runtime purpose
- editor preview has exactly one intentional submission path
- imported assets are the only normal editor authoring route
- raw-file workflows are either explicitly runtime-only or fully replaced
- the low-level renderer upload surface is either intentionally kept or intentionally removed

## Relationship To Existing Docs

This document is the concrete closeout companion to:

- [Unified Gaussian pipeline refactor plan](gaussian-pipeline-unification-plan.md)
- [Resident-instanced renderer contract](gaussian-resident-instanced-contract.md)

Those documents explain how the architecture was unified. This document explains how the remaining compatibility seams should be retired.
