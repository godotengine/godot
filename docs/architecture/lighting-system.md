# Lighting System Architecture

This document explains how lighting and shadow controls flow from settings and node state into GPU parameters and shader execution.

## Canonical Sources

- Lighting defaults and renderer integration: [../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp)
- Lighting/shadow parameter assembly in stage setup: [../../modules/gaussian_splatting/renderer/render_pipeline_stages.cpp](../../modules/gaussian_splatting/renderer/render_pipeline_stages.cpp)
- GPU param packing (`TileRenderParamsGPU`): [../../modules/gaussian_splatting/renderer/tile_render_stages.cpp](../../modules/gaussian_splatting/renderer/tile_render_stages.cpp)
- GPU param layout contract: [../../modules/gaussian_splatting/renderer/gaussian_gpu_layout.h](../../modules/gaussian_splatting/renderer/gaussian_gpu_layout.h)
- Lighting bridge and shadow sampling shaders: [../../modules/gaussian_splatting/shaders/includes/gs_lighting_bridge.glsl](../../modules/gaussian_splatting/shaders/includes/gs_lighting_bridge.glsl), [../../modules/gaussian_splatting/shaders/includes/gs_lighting_common.glsl](../../modules/gaussian_splatting/shaders/includes/gs_lighting_common.glsl), [../../modules/gaussian_splatting/shaders/includes/gs_directional_shadow.glsl](../../modules/gaussian_splatting/shaders/includes/gs_directional_shadow.glsl)
- Resolve pass shader: [../../modules/gaussian_splatting/shaders/tile_resolve.glsl](../../modules/gaussian_splatting/shaders/tile_resolve.glsl)

## Control Surfaces

### Project Settings (Global)

Primary global lighting controls are initialized under `rendering/gaussian_splatting/lighting/*`:

- `direct_light_scale`
- `indirect_sh_scale`
- `shadow_strength`
- `dc_logit`
- `shadow_receiver_bias_scale`
- `shadow_receiver_bias_min`
- `shadow_receiver_bias_max`

See defaults setup in `GaussianSplatRenderer::_initialize_lighting_project_settings_defaults`.

### Node Controls (Per-Node)

Node-level controls that directly affect lighting/shadow-related behavior:

- `rendering/cast_shadow`
- `rendering/color_grading`
- `rendering/wind_*` override controls for animation deformation inputs

Source: [../../modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp](../../modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp)

## Data Flow: CPU to GPU

1. `render_pipeline_stages.cpp` reads project settings and assembles `TileRenderer::RenderParams`.
2. `tile_render_stages.cpp` packs these values into `TileRenderParamsGPU`.
3. `gaussian_gpu_layout.h` defines the authoritative memory layout and semantic fields: `lighting_config`, `shadow_strength`, `shadow_bias_config`, `lighting_mode`, and `light_counts`.
4. Resolve/cull/raster shaders consume these fields.

## Direct Lighting Path

Resolve-stage direct lighting runs in [tile_resolve.glsl](../../modules/gaussian_splatting/shaders/tile_resolve.glsl), using scene data, light buffers, and shadow helpers.

`lighting_mode.x` controls where direct lighting is applied:

- `0`: resolve
- `1`: per-splat
- `2`: both

Field semantics are defined in [../../modules/gaussian_splatting/renderer/gaussian_gpu_layout.h](../../modules/gaussian_splatting/renderer/gaussian_gpu_layout.h).

## Clustered vs Unclustered Lights

Cluster usage is determined by cluster buffer/config availability and `light_counts.z` semantics in GPU params.

- Cluster decision helper in shader: `gs_use_clustered_lights()` in [../../modules/gaussian_splatting/shaders/includes/gs_lighting_common.glsl](../../modules/gaussian_splatting/shaders/includes/gs_lighting_common.glsl)
- Debug override (force unclustered): [../../modules/gaussian_splatting/renderer/gpu_debug_utils.h](../../modules/gaussian_splatting/renderer/gpu_debug_utils.h)

## Shadow System

### Runtime Shadow Sampling

Shadow factors are computed per light type in [../../modules/gaussian_splatting/shaders/includes/gs_directional_shadow.glsl](../../modules/gaussian_splatting/shaders/includes/gs_directional_shadow.glsl):

- directional: `gs_directional_shadow`
- omni: `gs_omni_shadow_factor`
- spot: `gs_spot_shadow_factor`

### Directional Shadow Atlas Path

Directional shadow maps are rendered and blitted through:

- `GaussianSplatRenderer::render_directional_shadow_map`
- shadow output compositor setup
- shadow blit pipeline resources

Source: [../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp)

## Current Constraint: Soft Shadows in Compute Path

`gs_lighting_bridge.glsl` explicitly disables soft-shadow sampling in the compute compatibility path (`sc_*_shadow_samples()` return `0`). This is a current design constraint, not a documentation omission.

## Related Docs

- [Architecture overview](overview.md)
- [Render pipeline architecture](render-pipeline.md)
- [Project settings reference](../reference/project-settings.md)
- [Generated shader inventory](../api/shader_reference.md)
