# Engine Patches — GodotGS Fork Delta

This document lists every modification made to the Godot Engine source tree
at repository root by the GodotGS project. These changes are required for the
Gaussian Splatting module to integrate with Godot's rendering pipeline.

The upstream engine is no longer nested under `godot-source/`. In this fork,
engine files live at repository root and `modules/gaussian_splatting/` is the
authoritative in-tree module path.

**Base version:** Godot 4.5.0-rc @ `2dd26a027a99` (see `GODOT_VERSION`)
**Patch file:** `engine_modifications.patch` (2,348 lines, 38 files, +1,267 / −194)

---

## Summary

| Category | Files | Purpose |
|---|---|---|
| Rendering pipeline plumbing | 17 | Register `INSTANCE_GAUSSIAN_SPLAT` as a first-class instance type |
| New engine files | 3 | Storage class + tests (don't exist upstream) |
| Alpha blending pipeline | 3 | Compositing splat output onto the framebuffer |
| Backend stubs (GLES3/Dummy) | 4 | Interface compliance for backends that don't support splats |
| Editor integration | 2 | Drag-and-drop `.ply` → Gaussian Splat node |
| GPU resource management | 2 | Buffer validity checking and device instance tracking |
| Build/tooling | 3 | GLSL sanitization, Vulkan include path, test registration |
| Engine core / minor | 4 | Validation layer setter, CLI flag, minor refactors |

---

## Tier 1 — Rendering Pipeline (Critical)

These register Gaussian Splats as a new rendering instance type, the same pattern
Godot uses for meshes, particles, and fog volumes. **Without these, the module
cannot function.**

### `servers/rendering_server.h` (+9 / −1)
Added `INSTANCE_GAUSSIAN_SPLAT` to the `InstanceType` enum between
`INSTANCE_FOG_VOLUME` and `INSTANCE_MAX`.

### `servers/rendering/renderer_scene_cull.h` (+70 / −~20)
- Added `InstanceGaussianSplatData` struct
- Added `HiddenVisibilityIndexingPolicy` enum
- Added `allow_hidden_visibility_indexing` flag on instances
- Added `PagedArray<RID> gaussian_splats` to scene cull data
- Added `allows_hidden_visibility_indexing()` / `should_skip_visibility_indexing()`

### `servers/rendering/renderer_scene_cull.cpp` (+224 / −~40)
Implementation of the above — instance creation, culling logic, visibility
indexing for gaussian splat instances.

### `servers/rendering/renderer_scene_render.h` (+2 / −1)
Added `const PagedArray<RID> &p_gaussian_splats` parameter to the virtual
`render_scene()` signature.

### `servers/rendering/renderer_rd/renderer_scene_render_rd.h` (+14 / −~5)
Added `render_gaussian_splats_forward()` and `commit_gaussian_splats()` method
declarations.

### `servers/rendering/renderer_rd/renderer_scene_render_rd.cpp` (+128 / −~10)
Implementation of gaussian splat forward rendering and commit logic within the
RD (RenderingDevice) renderer.

### `servers/rendering/renderer_rd/forward_clustered/render_forward_clustered.cpp` (+253 / −~5)
~200+ lines of Gaussian shadow rendering infrastructure: shadow atlas helpers,
directional/omni/spot shadow dispatch structs. This is the largest single change.

### `servers/rendering/renderer_compositor.h` (+11 / −~3)
Added `virtual RendererRD::GaussianSplatStorage *get_gaussian_storage() = 0` to
the base compositor interface.

### `servers/rendering/renderer_rd/renderer_compositor_rd.h` (+25 / −~5)
Added `gaussian_storage` member, accessor method.

### `servers/rendering/renderer_rd/renderer_compositor_rd.cpp` (+18 / −~5)
Creates `GaussianSplatStorage` in constructor, deletes in destructor, returns
from `get_gaussian_storage()`.

### `servers/rendering/renderer_rd/storage_rd/render_data_rd.h` (+20 / −~5)
Added `const PagedArray<RID> *gaussian_splats` pointer and
`LocalVector<Ref<GaussianSplatRenderer>> gaussian_splat_renderers` to `RenderDataRD`.

### `servers/rendering/renderer_rd/storage_rd/utilities.cpp` (+25 / −~5)
Added `INSTANCE_GAUSSIAN_SPLAT` case to `get_base_type()` and `free()` methods
in the RD utility storage.

### `servers/rendering/rendering_server_globals.h` (+11 / −~3)
Added `static RendererRD::GaussianSplatStorage *gaussian_storage` global pointer.

### `servers/rendering/rendering_server_globals.cpp` (+1)
Initialize the gaussian_storage global.

### `servers/rendering/rendering_server_default.cpp` (+9 / −~3)
Wires `gaussian_storage` into global initialization path.

### `servers/rendering/renderer_viewport.cpp` (+4)
Commented-out debug logging (no functional change, could be removed).

### `servers/rendering/renderer_rd/storage_rd/light_storage.h` (+2)
Added `get_omni_light_count()` and `get_spot_light_count()` public getters
(used by the gaussian shadow dispatch code).

---

## Tier 2 — New Engine Files (Critical)

These files are entirely new — they don't exist in upstream Godot.

### `servers/rendering/renderer_rd/storage_rd/gaussian_splat_storage.h` (+76, new)
RID-based storage class for Gaussian Splat renderer references and AABBs.
Follows the same pattern as `mesh_storage.h`, `particle_storage.h`, etc.

### `servers/rendering/renderer_rd/storage_rd/gaussian_splat_storage.cpp` (+94, new)
Implementation of the above.

### `tests/servers/rendering/test_renderer_scene_cull.h` (+131, new)
Unit tests for the visibility policy and scene cull additions.

---

## Tier 3 — Alpha Blending Pipeline (Required)

### `servers/rendering/renderer_rd/effects/copy_effects.h` (+23 / −~5)
Added `COPY_TO_FB_PIPELINE_ALPHA` and `COPY_TO_FB_PIPELINE_PREMULT_ALPHA`
pipeline variants. Added `p_enable_blend` / `p_use_premultiplied_alpha` params.

### `servers/rendering/renderer_rd/effects/copy_effects.cpp` (+87 / −~15)
Pipeline creation for alpha and premultiplied-alpha blend modes. Modified
`copy_to_fb_rect()` to support blending for gaussian splat compositing.

### `servers/rendering/renderer_rd/effects/SCsub` (+1)
Added `#thirdparty/vulkan/include` to the include path.

---

## Tier 4 — Backend Stubs (Required for Compilation)

These ensure non-RD backends (GLES3, Dummy) still compile after the interface
changes. The implementations are no-ops.

### `drivers/gles3/rasterizer_gles3.h` (+9 / −~3)
`get_gaussian_storage()` returns `nullptr`.

### `drivers/gles3/rasterizer_scene_gles3.h` (+2 / −1)
Updated `render_scene()` signature (parameter unused).

### `drivers/gles3/rasterizer_scene_gles3.cpp` (+2 / −1)
Updated `render_scene()` signature (parameter unused).

### `servers/rendering/dummy/rasterizer_dummy.h` (+7 / −~3)
`get_gaussian_storage()` returns `nullptr`.

### `servers/rendering/dummy/rasterizer_scene_dummy.h` (+2 / −1)
Updated `render_scene()` signature.

---

## Tier 5 — Editor Integration (Optional)

### `editor/docks/scene_tree_dock.cpp` (+76 / −~5)
Added `_perform_create_gaussian_splats()` — handles drag-and-drop of `.ply`
files into the scene tree, creating a `GaussianSplatNode3D`. Guarded by
`#ifdef MODULE_GAUSSIAN_SPLATTING_ENABLED`.

### `editor/docks/scene_tree_dock.h` (+3)
Declaration for the above.

---

## Tier 6 — GPU Resource Management (Helpful)

### `servers/rendering/rendering_device.h` (+21 / −~5)
- Added `buffer_is_valid()` for checking RID validity across all buffer pools
- Added `device_instance_counter` / `device_instance_id` / `get_device_instance_id()`
  for multi-device tracking
- Enhanced invalid-free error diagnostics with sampling

### `servers/rendering/rendering_device.cpp` (+45 / −~10)
Implementation of the above.

---

## Tier 7 — Build & Tooling

### `glsl_builders.py` (+21)
Added `_sanitize_glsl_lines()` to strip `#extension GL_KHR_vulkan_glsl` and
`GL_GOOGLE_include_directive` directives from GLSL before feeding them to
Godot's RenderingDevice shader compiler.

### `tests/test_main.cpp` (+3)
Added `#include "tests/servers/rendering/test_renderer_scene_cull.h"`.

---

## Tier 8 — Minor / Quality-of-Life

### `core/config/engine.h` (+11 / −~3) and `core/config/engine.cpp` (+4)
Added `set_validation_layers_enabled(bool)` method (setter complement to the
existing getter).

### `main/main.cpp` (+12 / −~5)
Added `--diagnose-gaussian-rendering` CLI option. Minor refactoring of
process exit handling (named bool for return value).

### `platform/windows/os_windows.cpp` (+5 / −~3)
Minor refactoring: `Main::iteration()` return stored in named `exit_requested`
variable (no functional change).

---

## Applying the Patch

These changes now live directly in the fork root. `engine_modifications.patch`
is retained as a maintenance aid for rebases and delta review.

To apply the documented engine-only delta to a fresh Godot 4.5 checkout:

```bash
# 1. Clone Godot at the base commit
git clone https://github.com/godotengine/godot.git godotgs
cd godotgs
git checkout 2dd26a027a99633231184616d4dd287bbdd1c0a3

# 2. Apply the patch from the checkout root
git apply /path/to/engine_modifications.patch

# 3. Build from repository root
scons platform=windows target=editor dev_build=yes -j10
```

## Rebasing to a New Godot Version

1. Create a branch from the new Godot version
2. Replay the GodotGS engine changes directly in the fork root
3. Resolve conflicts (most likely in `renderer_scene_cull` and `render_forward_clustered`)
4. Update `GODOT_VERSION` with the new base commit
5. Regenerate `engine_modifications.patch` so its paths are rooted at repository root and exclude the in-tree module directory
