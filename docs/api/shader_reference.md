# Shader Reference

Last generated: 2026-03-20

Coverage summary: `49` documented functions, `146` undocumented functions, `68` documented uniform fields, `123` undocumented uniform fields.

Undocumented entries are omitted by default. Use `--include-undocumented` to list them.

## Shader

```
modules/gaussian_splatting/shaders/gs_shadow_blit.glsl
```

### Uniform Blocks

#### Block

```
Params (params)
```

<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>uv_scale_offset</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>xy = scale, zw = offset</td>
    </tr>
    <tr>
      <td><pre><code>invert_depth</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>1.0 to invert (reversed depth)</td>
    </tr>
  </tbody>
</table>

#### Block

```
Params (params)
```

<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>uv_scale_offset</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>xy = scale, zw = offset</td>
    </tr>
    <tr>
      <td><pre><code>invert_depth</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>1.0 to invert (reversed depth)</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/color_grading_binning.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>rgb_to_hsv(vec3 c)</code></pre></td>
      <td>Binning Stage Color Grading Applied after SH evaluation, before packing into ProjectedGaussian Cost: ~20 ALU operations per splat = ~0.02ms for 1M splats RGB to HSV (copy from tonemap.glsl)</td>
    </tr>
    <tr>
      <td><pre><code>hsv_to_rgb(vec3 c)</code></pre></td>
      <td>HSV to RGB (copy from tonemap.glsl)</td>
    </tr>
    <tr>
      <td><pre><code>apply_color_grading_binning(vec3 color)</code></pre></td>
      <td>Apply color grading to splat color (binning stage) Operates in linear color space, before R11G11B10 packing</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/gaussian_splat_common_inc.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>dither_noise(vec2 frag_coord)</code></pre></td>
      <td>Simple hash-based noise for dithering (spatially varying) Uses fragment position to generate pseudo-random value in [-0.5, 0.5]</td>
    </tr>
    <tr>
      <td><pre><code>dither_noise_rgb(vec2 frag_coord)</code></pre></td>
      <td>Generate RGB dither noise using different offsets for each channel This breaks up color banding from RGB9E5 quantization</td>
    </tr>
    <tr>
      <td><pre><code>compute_sh_basis_with_bands(vec3 dir, uint max_band, out float basis_values[16])</code></pre></td>
      <td>SH sign convention (ISSUE-038): Condon-Shortley phase included. This evaluation uses the real spherical harmonics basis with the Condon-Shortley (CS) phase factor applied.  Odd-m basis functions carry a leading minus sign (e.g. Y_1^{-1} = -C1*y, Y_1^1 = -C1*x). PLY coefficients from 3DGS training (Kerbl et al. 2023) are stored with CS phase already baked in, so they are consumed here without any sign adjustment.  The import side (ply_loader.cpp, assemble_sh_coefficients) documents the same convention. Compute SH basis functions up to the specified band level basis_values[0] = DC (l=0) basis_values[1-3] = 1st order (l=1) basis_values[4-8] = 2nd order (l=2) basis_values[9-15] = 3rd order (l=3)</td>
    </tr>
    <tr>
      <td><pre><code>compute_sh_basis(vec3 dir, out float basis_values[16])</code></pre></td>
      <td>Legacy compute_sh_basis for backwards compatibility (computes all bands)</td>
    </tr>
    <tr>
      <td><pre><code>evaluate_sh_color_with_bands(uint gaussian_index, vec3 dir, uint sh_band_level)</code></pre></td>
      <td>Evaluate SH color with configurable band level sh_band_level: 0=DC only, 1=1st order, 2=2nd order, 3=3rd order</td>
    </tr>
    <tr>
      <td><pre><code>evaluate_sh_color(uint gaussian_index, vec3 dir)</code></pre></td>
      <td>Legacy evaluate_sh_color for backwards compatibility (uses all available bands)</td>
    </tr>
    <tr>
      <td><pre><code>evaluate_sh_color_dithered(uint gaussian_index, vec3 dir, uint sh_band_level, vec2 frag_coord)</code></pre></td>
      <td>Evaluate SH color with dithering to mitigate RGB9E5 quantization banding frag_coord: fragment screen position (gl_FragCoord.xy) for spatially-varying dither sh_band_level: 0=DC only, 1=1st order, 2=2nd order, 3=3rd order</td>
    </tr>
  </tbody>
</table>

### Uniform Blocks

#### Block

```
SceneData (scene_data)
```

<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>camera_position</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>xyz: world camera position, w: time or unused</td>
    </tr>
    <tr>
      <td><pre><code>viewport</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x: width, y: height, z: near plane, w: far plane</td>
    </tr>
    <tr>
      <td><pre><code>misc</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x: sigma multiplier override, y: gaussian count, z,w: unused</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/gs_culling_utils.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>gs_should_distance_cull(uint p_stable_splat_key, float world_distance)</code></pre></td>
      <td>Returns true if splat should be culled based on distance. p_stable_splat_key must be stable across camera-motion-induced sort order changes.</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/gs_deformation.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>gs_apply_sphere_effector(vec3 world_position, vec4 effector_sphere, vec4 effector_config, float time_seconds, uint stable_seed)</code></pre></td>
      <td>Sphere effector: animates splats within a spherical region effector_sphere: xyz = center, w = radius effector_config: x = enabled, y = strength, z = falloff, w = animation frequency (Hz)</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/gs_directional_shadow.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>gs_directional_shadow(uint idx, vec3 vertex, vec3 geo_normal, float taa_frame_count, float receiver_bias)</code></pre></td>
      <td>Directional shadow sampling for Gaussian splats. Adapted from Godot's forward clustered directional shadow path (no soft shadows).</td>
    </tr>
    <tr>
      <td><pre><code>gs_omni_shadow_factor(uint idx, vec3 vertex, vec3 geo_normal, float taa_frame_count, out float attenuation)</code></pre></td>
      <td>Omni shadow factor + attenuation (hard shadow path, soft shadows disabled).</td>
    </tr>
    <tr>
      <td><pre><code>gs_spot_shadow_factor(uint idx, vec3 vertex, vec3 geo_normal, float taa_frame_count, out float attenuation)</code></pre></td>
      <td>Spot shadow factor + attenuation (hard shadow path, soft shadows disabled).</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/gs_eigen_binning.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>compute_opacity_aware_sigma(float opacity, float visibility_threshold, float max_sigma)</code></pre></td>
      <td>Opacity-Aware Bounding (FlashGS Optimization) Computes the effective sigma multiplier based on opacity and visibility threshold.  Reduces tile-Gaussian pairs by ~94% for low-opacity splats. Reference: FlashGS — Efficient Gaussian Splatting with Adaptive Bounds Compute the sigma multiplier for opacity-aware bounds. Returns the effective number of sigmas to use based on opacity. For high opacity (close to 1.0), returns close to max_sigma (conservative). For low opacity, returns a smaller value (aggressive culling).</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/gs_render_params.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>gs_get_sh_band_level()</code></pre></td>
      <td>Helper to get current SH band level from params</td>
    </tr>
    <tr>
      <td><pre><code>gs_is_opacity_aware_culling_enabled()</code></pre></td>
      <td>Helper to check if opacity-aware culling is enabled</td>
    </tr>
    <tr>
      <td><pre><code>gs_get_visibility_threshold()</code></pre></td>
      <td>Helper to get visibility threshold (tau)</td>
    </tr>
    <tr>
      <td><pre><code>gs_get_lod_blend_factor()</code></pre></td>
      <td>Helper to get LOD blend factor from params (LODGE technique)</td>
    </tr>
    <tr>
      <td><pre><code>gs_is_lod_blend_enabled()</code></pre></td>
      <td>Helper to check if LOD blending is enabled</td>
    </tr>
  </tbody>
</table>

### Uniform Blocks

#### Block

```
RenderParams (params)
```

<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>view_matrix</code></pre></td>
      <td><pre><code>mat4</code></pre></td>
      <td>World-to-view matrix (may include instance transform)</td>
    </tr>
    <tr>
      <td><pre><code>inv_view_matrix</code></pre></td>
      <td><pre><code>mat4</code></pre></td>
      <td>Inverse of view_matrix</td>
    </tr>
    <tr>
      <td><pre><code>projection_matrix</code></pre></td>
      <td><pre><code>mat4</code></pre></td>
      <td>View-to-clip projection matrix</td>
    </tr>
    <tr>
      <td><pre><code>inv_projection_matrix</code></pre></td>
      <td><pre><code>mat4</code></pre></td>
      <td>Clip-to-view inverse projection matrix (GS render projection)</td>
    </tr>
    <tr>
      <td><pre><code>viewport_size</code></pre></td>
      <td><pre><code>vec2</code></pre></td>
      <td>Viewport size in pixels</td>
    </tr>
    <tr>
      <td><pre><code>tile_count</code></pre></td>
      <td><pre><code>vec2</code></pre></td>
      <td>Tile grid dimensions (x,y)</td>
    </tr>
    <tr>
      <td><pre><code>total_gaussians</code></pre></td>
      <td><pre><code>uint</code></pre></td>
      <td>Total splats in the source buffer</td>
    </tr>
    <tr>
      <td><pre><code>visible_gaussians</code></pre></td>
      <td><pre><code>uint</code></pre></td>
      <td>Visible splats after culling</td>
    </tr>
    <tr>
      <td><pre><code>near_plane</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>Camera near plane distance</td>
    </tr>
    <tr>
      <td><pre><code>far_plane</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>Camera far plane distance</td>
    </tr>
    <tr>
      <td><pre><code>debug_flags</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x: tile_grid_or_bounds, y: collect_raster_stats, z: overflow_tiles, w: projection_issues</td>
    </tr>
    <tr>
      <td><pre><code>debug_overlay_opacity</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>Blend strength for debug overlays</td>
    </tr>
    <tr>
      <td><pre><code>opacity_multiplier</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>Global opacity scaling factor</td>
    </tr>
    <tr>
      <td><pre><code>_pad0</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>Explicit padding to align camera_position to 16 bytes (std140)</td>
    </tr>
    <tr>
      <td><pre><code>_pad1</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>std140 padding</td>
    </tr>
    <tr>
      <td><pre><code>camera_position</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>xyz used, w reserved</td>
    </tr>
    <tr>
      <td><pre><code>alpha_floor</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>Minimum alpha when force_solid_coverage is enabled</td>
    </tr>
    <tr>
      <td><pre><code>force_solid_coverage</code></pre></td>
      <td><pre><code>uint</code></pre></td>
      <td>Nonzero enforces alpha_floor for any covered pixel</td>
    </tr>
    <tr>
      <td><pre><code>overlap_record_count</code></pre></td>
      <td><pre><code>uint</code></pre></td>
      <td>Total overlap records in the sorted buffer</td>
    </tr>
    <tr>
      <td><pre><code>_pad_overlap</code></pre></td>
      <td><pre><code>uint</code></pre></td>
      <td>std140 padding</td>
    </tr>
    <tr>
      <td><pre><code>cull_far_tolerance</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>Extra tolerance past the far plane for culling</td>
    </tr>
    <tr>
      <td><pre><code>tiny_splat_screen_radius</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>Drop splats smaller than this pixel radius</td>
    </tr>
    <tr>
      <td><pre><code>max_conic_aspect</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>Max aspect ratio for conic clamping (lower = less stretching)</td>
    </tr>
    <tr>
      <td><pre><code>_pad_before_jacobian</code></pre></td>
      <td><pre><code>float</code></pre></td>
      <td>std140: align jacobian_diag_flags (vec4) to 16 bytes</td>
    </tr>
    <tr>
      <td><pre><code>jacobian_diag_flags</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>Jacobian diagnostic toggles: x=bypass_radius_depth_floor, y=bypass_j_col2_clamp, z=invert_j_col2_sign, w=reserved</td>
    </tr>
    <tr>
      <td><pre><code>debug_overlay_flags</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>Debug overlay flags: x=overlay_mode (0=off, 1=density_heatmap, 2=shadow_opacity), y=depth_visualization, z=projection_z_mismatch, w=white_albedo_lighting_isolation</td>
    </tr>
    <tr>
      <td><pre><code>sh_config</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>Spherical Harmonics configuration: x=sh_bands (0-3), y=amortization_divisor, z=amortization_phase, w=force_full_update sh_bands: 0=DC only, 1=1st order, 2=2nd order, 3=3rd order (full)</td>
    </tr>
    <tr>
      <td><pre><code>sh_decode_config</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>SH decode configuration: x=dc_logit (1.0=decode DC with sigmoid), yzw=reserved</td>
    </tr>
    <tr>
      <td><pre><code>opacity_culling_config</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>Opacity-aware culling (FlashGS optimization): x=enabled (1.0=true), y=visibility_threshold (tau), z=reserved, w=reserved When enabled, splat radii are calculated as: r = sqrt(2 * ln(alpha/tau) * lambda_max) This reduces tile-Gaussian pairs by ~94% for low-opacity splats</td>
    </tr>
    <tr>
      <td><pre><code>lod_blend_config</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>LOD blending configuration (LODGE technique): x=lod_blend_factor (0-1, global blend factor for the frame) y=lod_blend_enabled (0 or 1) z=lod_blend_distance (world units) w=reserved</td>
    </tr>
    <tr>
      <td><pre><code>distance_cull_config</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>Distance-based culling configuration: x=start_distance, y=max_cull_rate, z=enabled, w=overlap_keep_ratio overlap_keep_ratio is used by global-sort binning to thin overlap records deterministically when the overlap budget is saturated.</td>
    </tr>
    <tr>
      <td><pre><code>color_grading_primary</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>Color grading parameters (aligned to vec4): x=enabled (0/1), y=exposure, z=contrast, w=saturation</td>
    </tr>
    <tr>
      <td><pre><code>color_grading_secondary</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x=temperature, y=tint, z=hue_shift (degrees), w=reserved</td>
    </tr>
    <tr>
      <td><pre><code>lighting_config</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>Lighting configuration: x=direct_light_scale, y=indirect_sh_scale, z=enable_direct_lighting, w=normal_mode (0=depth gradients, 1=view dir, 2=depth gradients with fallback)</td>
    </tr>
    <tr>
      <td><pre><code>shadow_strength</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>Shadow configuration: x=shadow_strength (0..1), yzw=reserved</td>
    </tr>
    <tr>
      <td><pre><code>shadow_bias_config</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>Shadow receiver bias configuration: x=receiver_bias_scale (per-splat radius multiplier), y=receiver_bias_min, z=receiver_bias_max (0=disabled), w=reserved</td>
    </tr>
    <tr>
      <td><pre><code>lighting_mode</code></pre></td>
      <td><pre><code>uvec4</code></pre></td>
      <td>Lighting mode: x=direct_lighting_mode (0=resolve, 1=per-splat, 2=both), yzw=reserved</td>
    </tr>
    <tr>
      <td><pre><code>light_counts</code></pre></td>
      <td><pre><code>uvec4</code></pre></td>
      <td>Light counts: x=omni_light_count, y=spot_light_count, z=cluster_enabled, w=light_mask</td>
    </tr>
    <tr>
      <td><pre><code>cluster_config</code></pre></td>
      <td><pre><code>uvec4</code></pre></td>
      <td>Cluster configuration: x=cluster_shift, y=cluster_width, z=cluster_type_size, w=max_cluster_element_count_div_32</td>
    </tr>
    <tr>
      <td><pre><code>instance_rotation_inv_col0</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>Instance rotation inverse for SH view direction correction (mat3 stored as 3 vec4s for std140).</td>
    </tr>
    <tr>
      <td><pre><code>wind_dir_strength</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>Global procedural wind configuration: wind_dir_strength: xyz = direction (world), w = displacement strength (meters)</td>
    </tr>
    <tr>
      <td><pre><code>wind_time_config</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>wind_time_config: x = time_seconds, y = temporal_frequency, z = spatial_frequency, w = enabled (0/1)</td>
    </tr>
    <tr>
      <td><pre><code>effector_sphere</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>Single global sphere effector (foundation for capped multi-effector support): effector_sphere: xyz = center (world), w = radius effector_config: x = enabled (0/1), y = displacement strength (meters), z = falloff exponent, w = reserved</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/gs_sh_binning.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>compute_sh_basis(vec3 dir, uint max_band, out float basis[16])</code></pre></td>
      <td>SH sign convention (ISSUE-038): Condon-Shortley phase included. This evaluation uses the real spherical harmonics basis with the Condon-Shortley (CS) phase factor.  Matches ply_loader.cpp import convention and gaussian_splat_common_inc.glsl.  See those files for full documentation. Compute SH basis functions up to the specified band level basis[0] = DC (l=0) basis[1-3] = 1st order (l=1) basis[4-8] = 2nd order (l=2) basis[9-15] = 3rd order (l=3)</td>
    </tr>
    <tr>
      <td><pre><code>compute_sh_basis_1st_order(vec3 dir, out float basis[4])</code></pre></td>
      <td>Legacy 1st order basis for backwards compatibility</td>
    </tr>
    <tr>
      <td><pre><code>evaluate_sh_with_bands(Gaussian g, vec3 view_dir, uint sh_band_level)</code></pre></td>
      <td>Evaluate SH color with configurable band level sh_band_level: 0=DC only, 1=1st order, 2=2nd order, 3=3rd order</td>
    </tr>
    <tr>
      <td><pre><code>evaluate_sh_1st_order(Gaussian g, vec3 view_dir)</code></pre></td>
      <td>Legacy evaluate function that uses 1st order only (for backwards compatibility)</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/gs_sort_key.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>gs_float_to_sortable_uint(float value)</code></pre></td>
      <td>64-bit key layout: hi = sortable depth, lo = tie-break.</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/painterly_features.glsl
```

### Uniform Blocks

#### Block

```
PainterlyPalette (painterly_palette)
```

<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>params</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x: count, y: blend strength, z: noise strength, w: preserve luminance (>0.5)</td>
    </tr>
  </tbody>
</table>

#### Block

```
PainterlyBrush (painterly_brush)
```

<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>params0</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x: scale, y: softness, z: anisotropy, w: rotation jitter</td>
    </tr>
    <tr>
      <td><pre><code>params1</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x: texture noise strength, yzw: reserved</td>
    </tr>
  </tbody>
</table>

#### Block

```
PainterlyLighting (painterly_lighting)
```

<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>ambient_color</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>rgb: ambient tint, a: ambient intensity</td>
    </tr>
    <tr>
      <td><pre><code>shadow_color</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>rgb: artistic shadow tint, a: shadow strength</td>
    </tr>
    <tr>
      <td><pre><code>highlight_color</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>rgb: highlight tint, a: highlight strength</td>
    </tr>
    <tr>
      <td><pre><code>rim_color_power</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>rgb: rim tint, a: rim power</td>
    </tr>
    <tr>
      <td><pre><code>style_params0</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x: shading style, y: cel band count, z: painterly mix strength, w: brush influence</td>
    </tr>
    <tr>
      <td><pre><code>style_params1</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x: rim blend, y: color temperature (K), z: temperature strength, w: temporal stability</td>
    </tr>
    <tr>
      <td><pre><code>style_params2</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x: gooch cool mix, y: gooch warm mix, z: cel softness, w: unused</td>
    </tr>
    <tr>
      <td><pre><code>light_control</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x: light count, y: global intensity, z/w: reserved</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/platform_compat.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>gs_exp_fast(float x)</code></pre></td>
      <td>Fast exp approximation using Schraudolph's bit manipulation method exp(x) ≈ 2^(x / ln(2)) represented as IEEE 754 float bits</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/quantization_dequant.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>extract_quantized_position(uvec2 position_chunk)</code></pre></td>
      <td>Extract quantized position components from packed data</td>
    </tr>
    <tr>
      <td><pre><code>extract_chunk_id(uvec2 position_chunk)</code></pre></td>
      <td>Extract chunk ID from packed position data</td>
    </tr>
    <tr>
      <td><pre><code>extract_quantized_scale(uint scale_lo, uint scale_hi)</code></pre></td>
      <td>Extract quantized scale components from packed data (scalar version for struct layout)</td>
    </tr>
    <tr>
      <td><pre><code>extract_quantized_scale(uvec2 scale_area)</code></pre></td>
      <td>Legacy overload for uvec2 (backward compatibility)</td>
    </tr>
    <tr>
      <td><pre><code>extract_area(uint scale_hi)</code></pre></td>
      <td>Extract area from packed scale/area data (scalar version)</td>
    </tr>
    <tr>
      <td><pre><code>extract_area(uvec2 scale_area)</code></pre></td>
      <td>Legacy overload for uvec2</td>
    </tr>
    <tr>
      <td><pre><code>extract_rotation(uint rotation_lo, uint rotation_hi)</code></pre></td>
      <td>Extract rotation quaternion from packed data (scalar version for struct layout)</td>
    </tr>
    <tr>
      <td><pre><code>extract_rotation(uvec2 rotation_packed)</code></pre></td>
      <td>Legacy overload for uvec2 (backward compatibility)</td>
    </tr>
    <tr>
      <td><pre><code>dequantize_position(uvec3 quantized, ChunkQuantization chunk)</code></pre></td>
      <td>Dequantize position using asset-local chunk bounds.</td>
    </tr>
    <tr>
      <td><pre><code>dequantize_scale(uvec3 quantized, ChunkQuantization chunk)</code></pre></td>
      <td>Dequantize scale using asset-local chunk bounds.</td>
    </tr>
    <tr>
      <td><pre><code>extract_normal(uint normal_xy, uint normal_z_stroke)</code></pre></td>
      <td>Extract normal from packed data</td>
    </tr>
    <tr>
      <td><pre><code>extract_stroke_age(uint normal_z_stroke)</code></pre></td>
      <td>Extract stroke age from packed normal data</td>
    </tr>
    <tr>
      <td><pre><code>get_max_position_error(ChunkQuantization chunk)</code></pre></td>
      <td>Utility: Compute quantization error bounds Calculate maximum position quantization error for a chunk</td>
    </tr>
    <tr>
      <td><pre><code>get_max_scale_error(ChunkQuantization chunk)</code></pre></td>
      <td>Calculate maximum scale quantization error for a chunk</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/tile_projection_common.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>gs_pack_conic_y_and_index(float conic_y, uint global_idx)</code></pre></td>
      <td>Legacy function kept for API compatibility</td>
    </tr>
    <tr>
      <td><pre><code>gs_unpack_conic_y_and_index(uint packed, out float conic_y, out uint global_idx)</code></pre></td>
      <td>Legacy function kept for API compatibility</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/includes/tile_raster_common.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>gs_dither_noise(vec2 frag_coord)</code></pre></td>
      <td>Simple hash-based noise for dithering (spatially varying) Uses fragment position to generate pseudo-random value in [-0.5, 0.5]</td>
    </tr>
    <tr>
      <td><pre><code>gs_dither_noise_rgb(vec2 frag_coord)</code></pre></td>
      <td>Generate RGB dither noise using different offsets for each channel This breaks up color banding from quantization</td>
    </tr>
    <tr>
      <td><pre><code>gs_apply_color_dither(vec3 color, vec2 frag_coord)</code></pre></td>
      <td>Apply dithering to a color to reduce quantization banding Uses flat dithering (not scaled by color) for consistent banding reduction across all tones</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/tile_prefix_scan.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>main()</code></pre></td>
      <td>Pass 3: add workgroup offsets into base ranges.</td>
    </tr>
  </tbody>
</table>

### Uniform Blocks

#### Block

```
PrefixParams (params)
```

<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>global_sort_capacity</code></pre></td>
      <td><pre><code>uint</code></pre></td>
      <td>Used for overflow detection in Pass 3</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/shaders/viewport_blit.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>srgb_to_linear(vec3 color)</code></pre></td>
      <td>Fast sRGB approximations using polynomial/sqrt instead of pow() These avoid expensive pow() calls while maintaining good accuracy Max error ~0.4% which is imperceptible</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/compute/cluster_cull.glsl
```

### Functions

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>aabb_frustum_visible(vec3 aabb_min, vec3 aabb_max)</code></pre></td>
      <td>AABB-frustum intersection test (conservative)</td>
    </tr>
  </tbody>
</table>

### Uniform Blocks

#### Block

```
ClusterCullParams (params)
```

<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>fine_cull_workgroup_size</code></pre></td>
      <td><pre><code>uint</code></pre></td>
      <td>Typically 256</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/compute/depth_compute.glsl
```

### Uniform Blocks

#### Block

```
Params (params)
```

<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>camera_position_ortho</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>xyz = camera position, w = orthographic flag</td>
    </tr>
    <tr>
      <td><pre><code>cull_screen_distance</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x = pixel_scale_y, y = tiny_splat_radius_px, z = min_screen_threshold_px, w = max_distance_sq</td>
    </tr>
    <tr>
      <td><pre><code>cull_frustum_radius</code></pre></td>
      <td><pre><code>vec4</code></pre></td>
      <td>x = radius_multiplier, y = frustum_plane_slack, z = enable_frustum, w = reserved</td>
    </tr>
  </tbody>
</table>


## Shader

```
modules/gaussian_splatting/compute/instance_chunk_dispatch.glsl
```

### Uniform Blocks

#### Block

```
Params (params)
```

<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>max_visible_chunks</code></pre></td>
      <td><pre><code>uint</code></pre></td>
      <td>Uses InstanceDepthParamsGPU.visible_chunk_count slot.</td>
    </tr>
    <tr>
      <td><pre><code>dispatch_group_x</code></pre></td>
      <td><pre><code>uint</code></pre></td>
      <td>Uses InstanceDepthParamsGPU.pad0 slot.</td>
    </tr>
  </tbody>
</table>


Generated by:

```
scripts/generate_shader_docs.py
```
