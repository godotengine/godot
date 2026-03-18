# Recommended Reading Order

Related docs: [ARCHITECTURE](ARCHITECTURE.md), [ABBREVIATIONS](ABBREVIATIONS.md), [README](README.md)

For new contributors to understand the Gaussian Splatting module:

## Level 1: Core Concepts (Start Here)
1. `README.md` - Overview and runtime data flow
2. `core/gaussian_data.h` - Data structures (Gaussian, PackedGaussian)
3. `nodes/gaussian_splat_node_3d.h` - Main node API

## Level 2: Rendering Pipeline
4. `renderer/gaussian_splat_renderer.h` - Renderer interface (implements `IRenderer`)
5. `renderer/render_pipeline_stages.cpp` - Cull -> Sort -> Raster -> Composite
6. `renderer/tile_renderer.cpp` - Tile-based rasterization

## Level 3: Streaming System
7. `core/gaussian_streaming.h` - Streaming state structs (T8)
8. `core/gaussian_streaming.cpp` - Chunk management, VRAM budget
9. `renderer/gpu_memory_stream.cpp` - GPU upload mechanics
10. `MEMORY_SUBSYSTEM.md` - Resident vs streaming memory paths and budget flow

## Level 4: GPU Sorting
11. `renderer/gpu_sorter.h` - Sorter interface
12. `renderer/gpu_sorting_constants.h` - Algorithm thresholds
13. `renderer/gpu_sorter.cpp` - Bitonic, Radix, OneSweep implementations

## Level 5: Advanced Topics
14. `interfaces/` - Dependency-inverted interfaces (see `renderer_interfaces.h` for `IRenderer`)
15. `lod/` - Level-of-detail management
16. `painterly/` - Artistic rendering effects

If you are investigating refactor outcomes, scan `render_*_orchestrator.*` (T9) after Level 2 to see where long-method responsibilities were split.
