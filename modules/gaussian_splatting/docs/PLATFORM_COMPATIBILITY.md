# Platform Compatibility and Performance Guide

This document provides platform compatibility information and performance tuning guidance for the Gaussian Splatting module.

## Platform Compatibility Matrix

| Platform | GPU Vendor | Status | Notes |
|----------|------------|--------|-------|
| Windows | NVIDIA | Tested | Primary development platform. Full feature support including subgroup operations. |
| Windows | AMD | Partial | Requires testing. RDNA2+ recommended for subgroup support. |
| Windows | Intel | Partial | Requires testing. Arc GPUs recommended for compute performance. |
| Linux | NVIDIA | Partial | Requires CI testing. Driver version 525+ recommended. |
| Linux | AMD (Mesa) | Partial | Requires testing. Mesa 23.0+ recommended for RADV. |
| macOS | MoltenVK | Experimental | Vulkan translation layer. Performance overhead expected. Metal-specific workgroup sizes applied. |
| Android | Any | Not Supported | Mobile GPU constraints too restrictive for real-time Gaussian splatting. |
| iOS | Any | Not Supported | No Vulkan support. |
| Web (WebGPU) | Any | Not Supported | Would require dedicated port. |

### Platform-Specific Shader Configuration

The module automatically adjusts shader parameters based on the target platform:

| Parameter | Desktop | Metal (macOS) | Mobile |
|-----------|---------|---------------|--------|
| Dispatch Local Size | 256 | 128 | 64 |
| Tile Size | 16 | 16 | 8 |
| Tile Splat Capacity | 1024 | 1024 | 128 |

These values are configured in `shaders/includes/platform_compat.glsl`.

### GPU Feature Requirements

**Required Features:**
- Vulkan 1.1 or later
- Compute shader support
- Storage buffer support
- Atomic operations

**Recommended Features:**
- Subgroup operations (GL_KHR_shader_subgroup_basic, GL_KHR_shader_subgroup_ballot)
- Subgroup vote operations (GL_KHR_shader_subgroup_vote) for early-exit optimization

When subgroup operations are unavailable, the module falls back to per-thread atomics with reduced performance.

---

## Performance Cost Guide

### Pipeline Settings Impact

| Setting | Default | Performance Impact | Visual Impact | Memory Impact |
|---------|---------|-------------------|---------------|---------------|
| `max_splats_per_frame` | - | Linear with count | Direct quality | Linear |
| `tile_size` | 16 | Smaller = more dispatch overhead | Affects edge quality | Tile buffer scales |
| `tile_splat_capacity` | 1024 | Higher = handles dense scenes | Prevents overflow artifacts | Per-tile buffer |
| `sh_bands` | 3 (SH3) | SH0 is ~4x cheaper than SH3 | Color/view-dependent quality | 3-48 floats/splat |
| `quantization` | 16-bit | 30% faster load, 2x compression | Minor precision loss | 50% position storage |
| `streaming_enabled` | true | Enables large scene support | LOD transitions visible | Chunked allocation |
| `painterly_mode` | false | 20-40% additional cost | Artistic brush stroke style | Extra render pass |
| `gpu_culling` | true | 2-3x faster frustum culling | None | Minimal |
| `radix_sort` | true | Fastest for >10K splats | Required for correct blending | Temporary buffers |

### Spherical Harmonics (SH) Band Configuration

| SH Level | Coefficients | Floats/Splat | Memory Multiplier | Visual Quality |
|----------|--------------|--------------|-------------------|----------------|
| SH0 (DC) | 1 | 3 | 0.0625x | Base color only, no view dependence |
| SH1 | 4 | 12 | 0.25x | Basic view-dependent color |
| SH2 | 9 | 27 | 0.5625x | Enhanced view-dependent effects |
| SH3 | 16 | 48 | 1.0x (baseline) | Full quality, all view-dependent effects |

**SH Configuration Guidance:**
- Use SH0 for distant LODs or performance-critical scenarios
- Use SH1-SH2 for balanced quality/performance
- Use SH3 for highest quality close-up rendering
- Progressive loading: Start with SH0, stream higher bands as available

### Tile Renderer Settings

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| `tile_size` | 8-32 | 16 | Screen-space tile dimensions in pixels |
| `max_splats_per_tile` | 128-1024 | 1024 | GPU workgroup limit for per-tile sorting |
| `binning_group_size` | - | 256 | Compute shader workgroup size |

**Tile Size Considerations:**
- Smaller tiles (8): Better for sparse scenes, more dispatch overhead
- Larger tiles (32): More efficient dispatches, but overflow risk in dense regions
- Default (16): Balanced for most scenes

### Quantization Settings

| Setting | Default | Range | Impact |
|---------|---------|-------|--------|
| `per_chunk_quantization` | false | bool | Enables ~4x position compression |
| `position_bits` | 16 | 8-24 | 8=4x compress, 16=2x compress, 24=full precision |
| `scale_bits` | 12 | 8-16 | Scale quantization precision |
| `chunk_size` | 256-8192 | adaptive | Smaller chunks = better precision |

### LOD (Level of Detail) Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | true | Enable distance-based LOD |
| `num_levels` | 4 | Number of LOD levels (2-8) |
| `max_distance` | 100.0 | Maximum render distance |
| `splat_skip_enabled` | true | Skip splats at distance (2^LOD factor) |
| `sh_reduction_enabled` | true | Reduce SH bands at distance |
| `opacity_fade_enabled` | true | Fade opacity at max distance |

**LOD Level Mapping:**
- LOD 0: Full detail (SH3, all splats)
- LOD 1: SH2, every 2nd splat
- LOD 2: SH1, every 4th splat
- LOD 3+: SH0, every 8th+ splat

---

## Quality Presets

### Built-in Hardware Presets

| Preset | Max Splats | GPU Memory | Target Use Case |
|--------|------------|------------|-----------------|
| `rtx_3090_1080p` | 5,000,000 | 12 GB | High-end desktop, 1080p |
| `rtx_3090_4k` | 3,500,000 | 12 GB | High-end desktop, 4K |
| `desktop_1080p` | 1,200,000 | 1 GB | Mid-range desktop, 1080p |
| `desktop_4k` | 700,000 | 768 MB | Mid-range desktop, 4K |
| `steam_deck` | 300,000 | 256 MB | Handheld/integrated GPU |

### Recommended Settings by Hardware Tier

#### Ultra (RTX 3090/4090, 8+ GB VRAM)
```gdscript
# Project Settings
rendering/gaussian_splatting/rendering/sh_bands = 3
rendering/gaussian_splatting/lod/enabled = true
rendering/gaussian_splatting/lod/num_levels = 4
# Quality tier preset: "rtx_3090_1080p" or "rtx_3090_4k"
```

- Max splats: 3-5 million
- Full SH3 quality
- All features enabled
- Painterly mode supported

#### High (RTX 3070/3080, RX 6800, 6+ GB VRAM)
```gdscript
rendering/gaussian_splatting/rendering/sh_bands = 2
rendering/gaussian_splatting/lod/enabled = true
rendering/gaussian_splatting/lod/num_levels = 4
# Quality tier preset: "desktop_1080p"
```

- Max splats: 1-1.5 million
- SH2 for quality/performance balance
- LOD enabled for large scenes
- Painterly mode optional

#### Medium (GTX 1660, RX 5600, 4+ GB VRAM)
```gdscript
rendering/gaussian_splatting/rendering/sh_bands = 1
rendering/gaussian_splatting/lod/enabled = true
rendering/gaussian_splatting/lod/num_levels = 6
rendering/gaussian_splatting/streaming/sh_progressive_load = true
```

- Max splats: 500K-800K
- SH1 default, progressive load to SH2
- Aggressive LOD culling
- Consider quantization for large scenes

#### Low (Integrated GPU, Steam Deck, 2+ GB shared)
```gdscript
rendering/gaussian_splatting/rendering/sh_bands = 0
rendering/gaussian_splatting/lod/enabled = true
rendering/gaussian_splatting/lod/num_levels = 8
rendering/gaussian_splatting/quantization/per_chunk_quantization = true
# Quality tier preset: "steam_deck"
```

- Max splats: 200K-300K
- SH0 only (DC color)
- Maximum LOD reduction
- Quantization enabled
- Painterly mode disabled

---

## Performance Optimization Tips

### General Recommendations

1. **Profile First**: Use the built-in performance monitors to identify bottlenecks
2. **Match Resolution**: Higher resolutions require proportionally fewer splats
3. **LOD Aggressively**: Distance-based culling provides the best performance gains
4. **Stream Large Scenes**: Enable streaming for scenes exceeding 1M splats

### Common Bottlenecks

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Low FPS, high GPU usage | Too many visible splats | Enable LOD, reduce max_splats |
| Stuttering during movement | Streaming/LOD transitions | Increase load_ahead_factor |
| Memory warnings | Exceeding VRAM budget | Enable quantization, reduce SH bands |
| Tile overflow artifacts | Dense scene regions | Increase tile_splat_capacity or reduce tile_size |
| Slow load times | Large uncompressed data | Enable quantization, use .spz format |

### Performance Monitoring

Access performance metrics via:
```gdscript
var renderer = gaussian_splat_node.get_renderer()
print("Visible splats: ", renderer.get_visible_splat_count())
print("Tile assignment: ", renderer.get_tile_assignment_time(), " ms")
print("Rasterization: ", renderer.get_rasterization_time(), " ms")
print("GPU frame time: ", renderer.get_last_gpu_frame_time_ms(), " ms")
```

For pass-by-pass monitor semantics (`gpu_time_cull_ms`, `gpu_time_sort_ms`,
`gpu_time_binning_ms`, `gpu_time_prefix_ms`, `gpu_time_raster_ms`,
`gpu_time_resolve_ms`) and freshness interpretation, see:
`docs/timing_metrics_reference.md`.

---

## Known Limitations

1. **Mobile Platforms**: Real-time Gaussian splatting is not feasible on current mobile GPUs due to compute and memory constraints.

2. **WebGPU**: Would require a dedicated port with significant shader modifications.

3. **macOS/MoltenVK**: Performance overhead from Vulkan-to-Metal translation. Native Metal implementation would provide better performance.

4. **Subgroup Operations**: When unavailable, atomic operations fall back to per-thread execution with ~20-30% performance reduction.

5. **Very Dense Scenes**: Scenes with >10M splats require careful LOD configuration and streaming setup.

---

## References

- [Platform Compatibility Shader](../shaders/includes/platform_compat.glsl)
- [SH Configuration](../renderer/sh_config.h)
- [Quality Tier Presets](../core/quality_tier_config.cpp)
- [LOD Configuration](../lod/lod_config.h)
- [Quantization Configuration](../renderer/quantization_config.h)
