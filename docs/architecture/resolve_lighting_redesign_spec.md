# Resolve-Mode Lighting Redesign Spec

**Status**: design, not implemented
**Target**: post-tier-2 master (`e53d60ff77` or later)
**Supersedes**: the deferred portions of PR #220's resolve-lighting overhaul that were intentionally dropped during the PR #243 salvage

## 1. Summary

The Gaussian splatting resolve pass produces per-pixel lit output from blended splat fragments. PR #220 reported three resolve-mode lighting artifacts (chrome noise, dark patches, view-dependent brightness shifts) and bundled the fixes with a larger redesign that never fully landed. PR #243 ported the safe subset (weighted receiver depth, α-coverage weighting, SH double-scale fix, cascade-validity guards); the remaining normal-handling, ambient, and SH-occlusion changes were explicitly deferred because they require behavior decisions and cross-scene verification.

This spec defines the deferred work as a single focused design task. **It is not a cherry-pick.** The original commits targeted a slightly older codebase; the intent survives, the specific code needs re-derivation against current master.

### Not in scope
- Compositor default-policy flip (`depth_test=false`, relaxed scene-depth) — separate concern, separate PR.
- Color grading per-node independence — addressed by the color-grading fix PR.
- Per-splat SH precision or DC decode — already resolved on master.

## 2. Problems to solve

| ID | Symptom | Where it manifests |
|---|---|---|
| 1 | **Chrome/sparkle noise** on brightly lit resolve output under camera motion | Orbit camera around a scene with a strong directional light; specular-like flicker on overlap regions |
| 2 | **Dark patches** on silhouettes and overlap regions of shadowed splats | Static shadowed scene with overlapping splats; dark halos at silhouettes |
| 3 | **View-dependent brightness shifts** on static lighting | Orbit camera with no light-direction change; splat brightness changes as the camera moves |

Common cause: all three trace back to the resolve lighting pipeline using *per-pixel blended/flipped normals* that change with view angle and overlap depth. The BRDF evaluation is sensitive to normal direction, and the `sh_occlusion` term couples shadow strength into the base color multiplier, which amplifies artifacts under heavy shadow.

## 3. Current state on master

What's already landed (from PR #243 and adjacent merges):

- `tile_raster_common.glsl:662-672` — raster accumulates α-weighted `weighted_depth`, output to `lighting_depth` for stable PSSM cascade selection
- `tile_resolve.glsl:175,267-273` — resolve reconstructs lighting/shadow positions from `lighting_depth`
- `tile_resolve.glsl:380-381` — direct-light contribution is α-coverage weighted, not α² weighted
- `gs_directional_shadow.glsl:54-125` — cascade-range guards with graceful fallback to next cascade for blend_splits

What's still broken:

- `gs_lighting_common.glsl:50-54` — `shadow_normal` flipped if `dot(normal, light_dir) < 0`
- `gs_lighting_common.glsl:60-63` — `h_normal` flipped the same way before BRDF evaluation
- `gs_lighting_common.glsl:57` — `sh_occlusion = max(sh_occlusion, 1.0 - shadow)` couples shadow state into the SH contribution
- `tile_resolve.glsl:309-313` — resolve normal fallback is `view_dir` when `normal_sample` is degenerate
- `tile_resolve.glsl:363-366` — base SH color multiplied by `sh_occlusion`
- The resolve pass already binds the scene reflection atlas (`tile_render_resolve.cpp:916-920`, `:1007-1011` fallback), so an ambient/radiance source IS available — the gap is in shader use, not in CPU plumbing

## 4. Design decisions needed

The implementer must choose one option per decision. Defaults are what I'd recommend, but each has tradeoffs worth a reviewer thinking through before code lands.

### 4.1 Normal handling for BRDF evaluation

| Option | Description | Pros | Cons |
|---|---|---|---|
| A (default) | **World-up normal** for both diffuse and specular BRDF. No flipping. Use the blended `normal_sample` only for shadow direction. | Kills chrome noise and view-dependent shifts; behavior is deterministic per world position | Flat shading look — loses per-splat normal detail |
| B | **Blended normal, no flipping**. Trust the blended `normal_sample`, accept that dot(normal, light) can be negative (resulting in 0 NdotL). | Preserves per-splat surface detail | Silhouettes go dark because blended normals at silhouettes face away from light |
| C | **Hybrid**: world-up for diffuse, blended for specular. | Keeps some per-splat shimmer without chrome noise | More moving parts; harder to reason about |

### 4.2 SH occlusion handling

| Option | Description | Pros | Cons |
|---|---|---|---|
| A (default) | **Remove the `sh_occlusion` coupling**. Base SH color is always applied at full strength; shadow only affects the direct-light term. | Fixes dark patches caused by SH * shadow multiplication | Shadowed splats look brighter (show more baked color) — may or may not match intended look |
| B | Keep `sh_occlusion` but scale it softer (e.g., `sh_occlusion * 0.5` instead of `max(..., 1.0 - shadow)`). | Preserves some darken-in-shadow behavior | Tunable magic number; scene-dependent |
| C | Let each node expose a per-node `sh_shadow_coupling` parameter. | Scene artist control | Adds a param, adds surface area |

### 4.3 Ambient / indirect contribution

| Option | Description | Pros | Cons |
|---|---|---|---|
| A (default) | **Constant ambient term** read from `scene_data_block.data.ambient_light_color_energy.rgb` (already plumbed via the existing scene_data uniform set), **gated on `bool(scene_data_block.data.flags & SCENE_DATA_FLAGS_USE_AMBIENT_LIGHT)`**. The RGB channels are *already* premultiplied by the ambient energy when the scene UBO is built (`render_scene_data_rd.cpp:172-182`), so use `.rgb` directly — do NOT multiply by `.w` again or you double-scale. The flag must be checked because RGB is populated unconditionally in the UBO, while `SCENE_DATA_FLAGS_USE_AMBIENT_LIGHT` is only set when the environment actually selects an ambient source (`render_scene_data_rd.cpp:184-187`); skipping the flag would inject ambient into scenes that intentionally disable it. This mirrors `forward_clustered/scene_forward_clustered.glsl:1639` exactly. | Simple, cheap, no new buffers; uses existing scene env consistently with mesh shading | Not physically motivated for outdoors HDR; requires artist tuning |
| B | **Reflection-atlas sample** via the already-bound texture at `tile_render_resolve.cpp:916-920`. The cubemap is in the descriptor set; this option only adds the shader-side sample (sample at world-up or blended normal). | Physically motivated; matches mesh rendering; no new CPU plumbing | Cubemap may not be populated for all scenes; the shader should fall back to ambient_light_color_energy.rgb when the atlas slot is empty |
| C | Expose both, pick via project setting. | Flexibility | More surface area, more CI |

### 4.4 Fallback normal when `normal_sample` is degenerate

| Option | Description | Pros | Cons |
|---|---|---|---|
| A (default) | **World-up (`vec3(0,1,0)`)** as degenerate fallback. | View-independent; eliminates the view-dir artifact | Incorrect for walls / sideways surfaces |
| B | Keep current `view_dir` fallback. | Matches current behavior | Produces view-dependent artifacts — Issue 3 |
| C | Blend with surface tangent somehow. | Smarter | Not obviously better; complexity |

## 5. Proposed approach (if you pick all "A" options)

Minimal, coherent v1:

1. **BRDF path**: replace normal flipping in `gs_lighting_common.glsl:50-63` with a world-up normal for **both diffuse and specular**. This is strict Option 4.1A — no per-splat blended normal feeds the BRDF at all, which is what fully kills view-dependent chrome noise. (If the implementer wants to keep blended normal for specular only, that's the hybrid Option 4.1C; document the choice in the PR description and accept the partial chrome noise it preserves.)
2. **Shadow sampling**: uses blended `normal_sample` for receiver-bias direction (current), but `shadow_normal` is NOT flipped. If `dot(normal, light) < 0`, the splat is treated as facing away from the light → shadow term is 1.0 (unshadowed) but `NdotL = 0` on the diffuse side. Simpler than the current dual-flip logic.
3. **SH occlusion**: delete the `sh_occlusion = max(sh_occlusion, 1.0 - shadow)` line at `gs_lighting_common.glsl:57`. Remove the multiplication at `tile_resolve.glsl:363-366`. Base SH color applied at full strength.
4. **Ambient**: in `tile_resolve.glsl`, read `scene_data_block.data.ambient_light_color_energy.rgb` directly from the already-bound scene_data uniform set and add it into the shading composition **only when `bool(scene_data_block.data.flags & SCENE_DATA_FLAGS_USE_AMBIENT_LIGHT)` is true**. **No new resolve-param uniform, no `TileRenderParamsGPU` field, no CPU plumbing** — this is Option 4.3A. The RGB channels are already energy-premultiplied at UBO build time (`render_scene_data_rd.cpp:172-182`); do not multiply by `.w`. The flag check is required because RGB is populated unconditionally but the flag is only set when the environment selects an ambient source (`render_scene_data_rd.cpp:184-187`); without the gate, scenes that intentionally disable ambient would have light injected. Matches `scene_forward_clustered.glsl:1639` exactly.
5. **Normal fallback**: at `tile_resolve.glsl:309-313`, change the fallback from `view_dir` to `vec3(0, 1, 0)`.

## 6. Shader / dispatch changes

Files modified (estimated):

- `modules/gaussian_splatting/shaders/includes/gs_lighting_common.glsl` — remove normal flips, remove `sh_occlusion` coupling. ~10 lines deleted, ~3 added.
- `modules/gaussian_splatting/shaders/tile_resolve.glsl` — normal fallback, remove SH × occlusion multiplication, inject constant ambient. ~15 lines.
- (Option 4.3A) **No CPU plumbing needed** — the shader reads `scene_data_block.data.ambient_light_color_energy` from the already-bound scene_data uniform set. No new field on `TileRenderParamsGPU`, no layout version bump, no `sizeof` assertion change.
- (Option 4.3B) `modules/gaussian_splatting/renderer/tile_render_resolve.cpp` — already binds reflection_atlas at `:916-920`; only need shader-side sample plumbing in `tile_resolve.glsl`. No new uniform binding.

No new shaders, no new dispatch passes, no new buffers. The plumbing rides the existing scene_data uniform set and the already-bound reflection atlas.

## 7. Integration with existing systems

- **Compatible with PR #243's weighted_depth work**. The normal changes are decoupled from the depth-reconstruction path.
- **Compatible with PR #241's cascade guards**. Shadow sampling still uses them; only the receiver_normal input changes.
- **No interaction with PR #237's hotspot pruning**. Resolve runs after binning; the cull-time changes don't touch resolve logic.
- **Resident instance contract** (PR #236 / 905eb1a0ab): ambient color is per-frame (read from scene_data each frame), not per-instance-contract — no cache key change needed.

## 8. Tests

### New unit-test cases (in `tests/test_renderer_pipeline.h` or a new `test_resolve_lighting.h`)

- **Normal-stability under camera orbit**: render a single lit splat at 12 camera angles around a fixed position, measure output color variance. Variance should be below a threshold (e.g. < 5%) for Option 4.1A.
- **SH color preserved in full shadow**: render a brightly-colored splat fully occluded by a shadow caster, assert output contains the base color at > 80% strength (current master: ~0% due to SH × occlusion).
- **No dark halos at silhouettes**: render a cluster of overlapping splats under one directional light, check pixels at silhouette edges for brightness continuity with interior.
- **Normal fallback doesn't flip with view**: render a splat whose `normal_sample` is degenerate (zero-length write from raster) at two opposing camera angles; output should be identical.

### Regression coverage
- Keep the existing resolve-mode tests passing. If any test locks current (buggy) behavior explicitly, update it.
- Re-run PR #243's tests to confirm weighted-depth path isn't regressed.

### Manual verification scenes
- A dream_memory-style scene with ~200k visible splats, directional light, overlapping tiles — eyeball under camera orbit.
- A single-splat test scene at known position under known light — smoke test for each decision option.

## 9. Metrics

No new GPU monitors needed — the change is in-shader with no dispatch boundary changes. The existing `gpu_time_resolve_ms` monitor will reflect any performance delta.

If the implementer wants visibility into the new ambient term, add one optional line to `Performance` custom monitors: `gaussian_splatting/resolve_ambient_intensity` (scalar, reflecting `dot(scene_data_block.data.ambient_light_color_energy.rgb, vec3(1.0))` at the moment of the last submission).

## 10. Risks and open questions

### Risks

- **Visual regression on existing scenes**: dropping per-splat normal detail (Option 4.1A) WILL look different from current behavior on some assets. Worth an eyeball pass before committing.
- **SH-in-shadow brightness**: removing the occlusion coupling (Option 4.2A) may make shadowed splats look "too bright" in some scenes that were relying on the accidental darkening.
- **Ambient color source**: use `scene_data_block.data.ambient_light_color_energy.rgb` directly, gated on `bool(scene_data_block.data.flags & SCENE_DATA_FLAGS_USE_AMBIENT_LIGHT)`. RGB is already energy-premultiplied at UBO build time (`render_scene_data_rd.cpp:172-182`); multiplying by `.w` again would double-scale. The flag is required because RGB is populated unconditionally in the UBO but `SCENE_DATA_FLAGS_USE_AMBIENT_LIGHT` is only set when an ambient source is selected (`render_scene_data_rd.cpp:184-187`); without it, scenes that intentionally disable ambient would have light injected. Do NOT use `.w == 0` as a "missing environment" sentinel — `.w == 0` is a legitimate authored configuration (e.g., zero background energy) and the no-environment path explicitly seeds rgb to (1,1,1) and `.w` to 1 (`render_scene_data_rd.cpp:153-156`), not zero. Match what mesh shading paths in `scene_forward_clustered.glsl:1639` do — flag-gated RGB read, no value-based fallback. If a real "no env" fallback is ever needed, detect via scene environment validity, not via the field's value.
- **Backwards compatibility**: scenes authored under current (buggy) behavior may need re-tuning. The PR should note this in its release-channels doc update.

### Open questions (human decisions before implementation)

1. **Pick one option from each of 4.1, 4.2, 4.3, 4.4** — the "A defaults" above are recommendations, not commitments. The spec stays valid regardless of choice; the implementation checklist below expands based on the choice.
2. **Feature flag?** Should the new behavior be gated behind a project setting (`rendering/gaussian_splatting/resolve/lighting_mode`) with a default flip, or hard-replace the old path? Recommendation: hard-replace. The "old behavior" is a bug, not a feature.
3. **Sizing of the constant ambient**: resolved in favor of Option 4.3A — value comes from `scene_data_block.data.ambient_light_color_energy.rgb` (already energy-premultiplied at UBO build time). No new project setting or per-node scalar in v1; revisit if artists need scene-local override (see §12).

## 11. Implementation checklist

Ordered, each task small enough for one commit:

- [ ] **1. Fallback normal**: change `tile_resolve.glsl:309-313` to use `vec3(0, 1, 0)` instead of `view_dir`. Add a test for the orbit-stability invariant. Commit.
- [ ] **2. Remove normal flipping**: delete the `shadow_normal` and `h_normal` flip blocks in `gs_lighting_common.glsl:50-63`. For Option 4.1A, replace with a world-up normal for **both diffuse and specular** BRDF evaluation (matching §5 step 1). For Option 4.1C (hybrid), keep blended normal for specular only and document that choice in the PR. Commit.
- [ ] **3. Remove SH occlusion coupling**: delete the `sh_occlusion = max(..., 1.0 - shadow)` line and the `sh_occlusion` multiplication at `tile_resolve.glsl:363-366`. Add the SH-color-in-shadow test. Commit.
- [ ] **4. Resolve shader ambient injection** (Option 4.3A — no CPU plumbing): in `tile_resolve.glsl`, gate on `bool(scene_data_block.data.flags & SCENE_DATA_FLAGS_USE_AMBIENT_LIGHT)` and only when set add `+ scene_data_block.data.ambient_light_color_energy.rgb * base_color * alpha` (or equivalent) in the composition step. The flag gate is required: RGB is populated unconditionally in the UBO (`render_scene_data_rd.cpp:172-182`) but `SCENE_DATA_FLAGS_USE_AMBIENT_LIGHT` is only set when the environment selects an ambient source (`render_scene_data_rd.cpp:184-187`); without it, scenes that intentionally disable ambient receive injected light. Use `.rgb` directly — it is already premultiplied by the ambient energy at UBO build time; multiplying by `.w` would double-scale and disagree with mesh shading. No fallback gate on `.w` (which is a legitimate authored value, not a missing-env sentinel). Mirrors `scene_forward_clustered.glsl:1639` exactly. Commit.
- [ ] **5. (deferred to Option 4.3B if pursued)** Add reflection-atlas sample in resolve shader using the already-bound texture from `tile_render_resolve.cpp:916-920`. No new CPU bindings. Skip if 4.3A is sufficient.
- [ ] **6. Test scenes**: add a new `test_resolve_lighting.h` doctest file under `modules/gaussian_splatting/tests/` with the four unit-test cases from §8. Wire it into `test_gaussian_splatting.h`. Add the filter to `tests/ci/run_module_tests.py` per the pattern used by `Gaussian Diagnostics`/`Gaussian Logger` (PR #244). Commit.
- [ ] **7. Docs**: update `docs/development/release-channels.md` or a new `docs/reference/resolve-lighting.md` to describe the new behavior. Commit.
- [ ] **8. PR** against master. Title suggestion: `fix: stabilize resolve-mode lighting (normals, SH occlusion, ambient)`.

## 12. Future work (out of scope)

- Per-node ambient color override (if artists want scene-local control).
- Full radiance cubemap sample for ambient (Option 4.3B) — if the constant ambient approach proves insufficient.
- Cluster-domain lighting precompute — could tie in with the separately-specified cluster-culling work.
