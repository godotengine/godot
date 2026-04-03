# Unified Gaussian Splat Pipeline Refactor Plan

Status: living design document  
Scope: high-level architecture and migration sequencing  
Audience: rendering, tools, and pipeline engineers

## Purpose

This document is the working refactor plan for unifying the Gaussian Splat data and rendering pipeline.

It is not a rewrite proposal. It is a staged migration plan that:

- converges the current submission paths on one renderer-facing contract
- separates scene submission from backend residency policy
- makes routing and effective settings inspectable
- preserves shipping behavior where possible while removing architectural ambiguity

## Why We Are Doing This

The current module works, but it is difficult to reason about because several concerns are mixed together:

- asset ingress
- scene submission
- route selection
- residency policy
- fallback behavior
- settings precedence

That has three costs:

1. Debugging is too expensive.  
   We often spend hours figuring out which path is active before we can investigate the actual bug.

2. Optimization is fragile.  
   Resident vs streaming, node vs world, and preset vs project behavior are not separated cleanly enough to profile with confidence.

3. User control is unclear.  
   Users cannot reliably answer what path their asset is taking, why that path was chosen, or how to override it.

## Target Architecture

The target model is:

`source asset or baked world -> shared renderer scene contract -> resident or streaming backend -> sort/render`

This means:

- one user-facing mental model
- one renderer-facing contract
- one effective route/debug surface
- one inspectable effective-settings model, with precedence normalization completed only when the current distributed rules are safe to collapse
- resident and streaming treated as backend policies, not separate scene pipelines

Important clarification:

“One path” does not mean every source must call the same function. It means different adapters must converge on the same renderer contract and the same inspectable effective state.

## Guiding Decisions

1. Submission and residency are orthogonal.  
   The shared instance-oriented renderer model is the normal rendering model. Resident and streaming are backend data policies behind it.

2. World baking remains a capability, not a separate render concept.  
   `GaussianSplatWorld` and `GaussianSplatWorld3D` may remain useful packaging or world-data adapters, but they should not imply a distinct renderer contract.

3. Resolution logic should not be centralized prematurely.  
   Effective settings are currently resolved in multiple places. The first goal is to expose provenance, not to invent a second resolver.

4. Debuggability is a first-class outcome.  
   We should be able to answer “what path is active and why?” from the editor or HUD.

5. The renderer facade stays stable during migration.  
   Low-level methods that tests and tools depend on can be deprecated before removal, but should not disappear before replacement hooks exist.

## Verified Current-State Constraints

These current facts shape the plan:

- `GaussianSplatNode3D` and `GaussianSplatDynamicInstance3D` already submit through [GaussianSplatSceneDirector](../../modules/gaussian_splatting/core/gaussian_splat_scene_director.cpp).
- `GaussianSplatWorld3D` still pushes renderer data directly through [gaussian_splat_world_3d.cpp](../../modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp).
- `GaussianSplatWorld3D` also owns renderer-setting application, tier overrides, and renderer ownership arbitration.
- The data-upload path still instantiates a streaming system in [render_data_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_data_orchestrator.cpp).
- The instanced path still depends on streaming readiness in [render_instancing_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp) and [gaussian_splat_renderer.cpp](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp).
- The current instance-pipeline contract is atlas-shaped across cull, sort, raster, and tile stages.
- Route control is now owned solely by enum-based `route_policy` in [gs_project_settings.h](../../modules/gaussian_splatting/core/gs_project_settings.h).
- The editor still sets both `splat_asset` and `ply_file_path` in [gaussian_editor_plugin.cpp](../../modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp).
- `ply_file_path` is still a real public API and load path in [gaussian_splat_node_3d.cpp](../../modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp) and [gaussian_splat_node_helpers.cpp](../../modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp).
- Route IDs and renderer stats already exist, but they are not explained well to users.

## Refactor Strategy

The work is split into one enabling stage, two structural stages, one observability stage, and one deferred deprecation stage.

Not every stage is release-worthy on its own. Some stages are structural checkpoints.

## Stage 0: Visibility And Ingress Guardrails

Goal: improve observability and remove the worst editor ambiguity before deeper structural work lands.

This stage is intentionally low-risk and can run before or alongside Stage 1.

### Changes

- Stop the editor plugin from dual-writing both `splat_asset` and `ply_file_path` when it already has an imported asset.
- Add asset-origin labeling in the inspector.
- Add configuration warnings for inconsistent dual-path state.
- Expand existing HUD/debug output so route IDs are explained in human terms.
- Expose existing `route_uid`, `sort_route_uid`, and `cull_route_uid` more clearly instead of inventing a new debug surface first.

### What Stays Stable

- `ply_file_path` remains functional.
- Runtime load behavior does not change.
- Submission routes do not change yet.
- Settings resolution logic does not move yet.

### Exit Criteria

- Editor import no longer sets both `splat_asset` and `ply_file_path` by default.
- Inspector shows asset origin metadata.
- HUD/debug surfaces can explain the active route in plain language.

### Risk

Low to low-medium.

## Stage 1: Unified Submission Contract

Goal: all high-level scene submission surfaces converge on one renderer-facing contract brokered by the director.

This is the first real architectural stage.

### Core Decision

We do not overload `InstanceRecord` for world data.

Instead, the director gains explicit submission adapters:

- instance-backed submissions for node and dynamic-instance paths
- world-backed submissions for baked/static world payloads
- preview-backed submissions for editor preview

Exact type names can vary, but the model must stay explicit.

### Changes

- Introduce an explicit world-submission record that can carry:
  - `GaussianData`
  - static chunks
  - bounds
  - metadata
  - desired residency hint
  - desired renderer/config overrides
- Move `GaussianSplatWorld3D` off direct renderer mutation and onto director-managed world submission.
- Move world-renderer ownership arbitration into the director.
- Move direct world data upload and direct static-chunk upload into the director-owned submission flow.
- Keep `GaussianSplatNode3D` and `GaussianSplatDynamicInstance3D` on the existing director-based instance path.
- Add a dedicated preview adapter so editor preview can participate in the same submission model without requiring a fake `World3D`.

### Important Constraint

Stage 1 does not centralize effective-settings resolution.

The director becomes the broker for submission and authority ownership, but not yet the single resolver for quality, tier, and feature settings. Existing resolution logic remains where it is until Stage 3.

In practical terms, Stage 1 relocates renderer ownership arbitration and direct data/chunk submission. It does not yet move full settings resolution or tier-limit computation into the director. World-side code may still compute desired overrides and pass them through the world-submission adapter during this stage.

Stage 1 also preserves the current single active world-backed source per scenario unless and until a separate aggregation design is approved. World-source merging is not part of this stage.

### What Stays Stable

- Tile rasterization, sorting, streaming internals, and compositing.
- Low-level renderer entrypoints used by tests and tools.
- Node- and dynamic-instance-level registration semantics.

### Migration Rule

`renderer->set_gaussian_data()` stays public during this stage for tests, probes, and low-level tools. High-level scene code should stop calling it directly, but the method is not removed yet.

Known migration surface: there are currently 18 direct test calls to `set_gaussian_data()` in the module test suite. Stage 4 must explicitly retire or replace that coverage before the method can be removed.

### Current Implementation Notes

- `submit_world_submission()` and `release_world_submission()` are the runtime world-submission entrypoints. They apply and clear renderer-owned world state in addition to tracking the active world-backed source.
- `upsert_world_submission()` and `unregister_world_submission()` remain as scaffolding and introspection helpers. They update only the director-owned record and intentionally have no renderer side effects.
- The current one-active-world-per-scenario rule is enforced only by the runtime `submit_*` / `release_*` path. The scaffolding helpers remain compiled only for tools/test builds, not as a normal runtime surface.

### Exit Criteria

- `GaussianSplatWorld3D` no longer directly calls renderer data/chunk upload methods.
- Renderer ownership arbitration is no longer embedded in `GaussianSplatWorld3D`.
- World-side settings and tier overrides are forwarded as submission inputs, not re-resolved by a new director-owned resolver.
- Director receives world, instance, and preview submissions through explicit entrypoints.
- One breakpoint in the director is enough to trace all high-level submissions.

### Risk

High.

Reason: this stage relocates authority, not just call sites.

## Stage 2: Backend Policy Split

Goal: resident and streaming become backend policies behind the shared submission contract.

This is the most technically risky stage.

### Core Decision

Do not start by naming which buffers are “streaming-only.”

First define a resident-instanced renderer contract that is valid for the existing cull, sort, raster, and tile stages, or explicitly define the minimal alternative stage inputs needed for resident data.

This is a hard prerequisite for Stage 2 implementation, not a follow-up refinement. Stage 2 should not begin coding until that contract is written down and reviewed.

Stage 2A deliverable: [Resident-instanced renderer contract](gaussian-resident-instanced-contract.md)

### Changes

- Add a shared residency hint concept across all submission types, not just world submissions.
- Let the renderer/backend resolve final `ResidencyMode` from:
  - global route policy
  - submission hint
  - device/runtime limits
  - budget constraints
- Split upload, readiness, and routing together:
  - resident upload path
  - resident-instanced readiness path
  - resident-instanced render route
- Remove unconditional streaming-system creation for resident data.
- Collapse legacy `streaming/enabled` behavior into enum-based route policy.

### Important Constraint

The final residency verdict belongs to the renderer/backend layer, not to the director alone. The director may carry hints, but the renderer owns the final backend choice.

### What Stays Stable

- Streaming remains a supported backend.
- Existing streaming backend internals stay largely intact unless required by the new contract.
- The renderer facade remains stable from the outside where possible.

### Current Implementation Notes

- The resident atlas publisher intentionally rejects per-chunk quantization. When resident publication is rejected for that reason, the renderer records `resident_quantization_unsupported` and falls back to streaming publication when the requested route is streaming-capable; explicit resident requests fall back to the legacy resident path instead of inventing a second stage contract.
- `backend_selection_reason` may be chained with ` -> ` when a rejected resident attempt leads into streaming publication or fallback. The chained reason is intentional and is part of the accepted diagnostics surface.
- Submission-hint collapse is conservative in the accepted implementation: conflicting instance-submission hints on one shared renderer collapse to no effective hint (`mixed_instance_submissions`), while preview and active world hints still take precedence over instance hints. Cross-source mixed-hint normalization is intentionally deferred because it would change backend-policy semantics.

### Exit Criteria

- A resident scene can render through the instance/shared renderer path without a `GaussianStreamingSystem`.
- The resident-instanced contract is explicitly defined and satisfies the input requirements of the existing cull, sort, raster, and tile stages, or the approved alternative stage inputs are documented.
- Route control is owned by one setting model, not overlapping booleans.
- A new resident-instanced route is explicit in diagnostics.
- Streaming scenes still work under the unified contract.

### Risk

High.

Reason: this is a renderer-contract change, not just a memory-allocation change.

## Stage 3: Effective Configuration And Route Provenance

Goal: make effective values and route decisions inspectable without moving resolution logic too early.

### Core Decision

Do not put a new master resolver into the director first.

Instead, provenance is added to the existing resolution sites and then composed into an inspectable snapshot.

### Changes

- Add human-readable explanations for existing route IDs.
- Extend current HUD/editor surfaces to show:
  - active route
  - route reason
  - cull/sort route details
- Add provenance logging to current resolution sites:
  - node helper quality resolution
  - tier caps
  - pipeline feature resolution
  - project-setting overrides
- Add a read-only effective-config snapshot assembled from current resolvers.
- Show effective values and their sources in the inspector and diagnostics.

### What Stays Stable

- Existing resolution algorithms remain the source of truth.
- No renderer-route changes happen in this stage.
- No GPU resource changes happen in this stage.

### Exit Criteria

- Users can answer “why is this streaming?” or “why is this capped?” from the editor/HUD.
- Effective values show source attribution.
- Route explanation is human-readable instead of raw UID-only.

### Risk

Medium.

Reason: the work is mostly additive, but it touches multiple existing resolution sites and must not duplicate rules.

## Stage 4: Consolidation, Deprecations And Removals

Goal: remove compatibility surfaces and normalize the remaining policy model only after replacement workflows are stable.

This stage is intentionally deferred.

### Candidates For Deprecation

- `ply_file_path` as a primary editor workflow
- compatibility remnants for direct high-level world-to-renderer mutation after Stage 1 migration
- post-migration cleanup once the enum `route_policy` is the only supported settings control
- tests and samples that still encode the old split as the preferred path

### Candidates For Consolidation

- publish one explicit precedence model for effective settings if Stage 3 provenance proves the distributed rules can be collapsed safely
- remove transitional compatibility hooks that remain public only for tests, tools, or migration support
- remove `upsert_world_submission()` — this is now a tests-only scaffolding entrypoint and should be retired once all coverage uses the ownership-aware submission path
- collapse the duplicated source-path resolution logic between `GaussianSplatNode3D::_get_asset_source_path()` and the editor plugin's `_get_asset_source_path()` into a single shared function

### Validation Status

- The explicit resident-plus-quantization-rejection fallback path now has dedicated coverage in `modules/gaussian_splatting/tests/test_scene_director_submission_scaffolding.h`. Keep that test as baseline coverage while the legacy explicit-resident fallback remains supported.

### Important Constraint

`ply_file_path` is still a real runtime feature today. It is not safe to demote it to metadata until replacement workflows are defined for:

- drag and drop
- runtime direct loading
- auto-load on scene entry
- validation and warnings

### Exit Criteria

- High-level editor and scene workflows use the unified asset/submission model.
- Deprecated APIs have migration notes and replacement paths.
- Legacy compatibility code can be removed without breaking normal authoring workflows.

### Risk

Deferred until earlier stages are stable.

## Invariants During The Refactor

These rules apply throughout the migration:

- `GaussianSplatRenderer` remains the stable public facade.
- We do not rewrite tile rasterization, sort algorithms, or compositor behavior as part of this plan.
- We do not centralize all settings logic in the first structural phase.
- We do not add new overlapping route knobs while trying to remove old ones.
- We do not remove low-level test hooks before replacements exist.

## Validation Strategy

Each stage should update both code and observability.

### Required Validation Areas

- unit tests for new adapters and routing behavior
- renderer tests for route and readiness transitions
- runtime scenes for world, instance, and preview paths
- benchmark lanes for resident vs streaming comparison
- explicit migration tests for route-policy behavior

### Required Runtime Scenarios

- single imported asset
- multi-instance scene
- world/baked static scene
- editor preview without scene node ownership
- small resident-fit scene
- large streaming-required scene

## Open Decisions

These decisions should be made early and recorded here as they are resolved:

1. Can one scenario host multiple world submissions, or is there still one active world-backed source per scenario?
   Current plan: preserve one active world-backed source per scenario through Stage 1. Revisit aggregation only after the unified contract is stable.
2. What exact resident-instanced GPU contract will satisfy the existing raster/tile stages?
   This decision is a prerequisite for Stage 2 start, not just a design follow-up.
3. Which low-level renderer methods remain public permanently, and which are migration-only?
4. What replaces `ply_file_path` if it is eventually demoted from first-class editor workflow?

## Stage Checkpoint Guidance

Stages are checkpointable, but not all checkpoints are product-complete.

- Stage 0 is a useful shipping-quality cleanup.
- Stage 1 is a structural checkpoint.
- Stage 2 is the main architecture payoff stage.
- Stage 3 turns the new architecture into something users and developers can actually understand.
- Stage 4 is cleanup, not a prerequisite for the architecture to be considered successful.

## Maintenance Notes

Update this document when any of the following changes:

- stage boundaries
- success criteria
- migration constraints
- open decisions
- public compatibility promises

When a stage begins, add:

- branch or PR reference
- owner
- current status
- validation result

This document should stay accurate enough that a new engineer can join the refactor without re-deriving the architecture from source.
