# Gaussian Splatting Module Architecture

Related docs: [MEMORY_SUBSYSTEM](MEMORY_SUBSYSTEM.md), [READING_ORDER](READING_ORDER.md), [ABBREVIATIONS](ABBREVIATIONS.md), [README](README.md)

## Module Entry Points

| Entry Point | File | Purpose |
|-------------|------|---------|
| `register_types.cpp` | `register_types.cpp` | Registers all classes with Godot |
| `GaussianSplatManager` | `core/gaussian_splat_manager.cpp` | Singleton managing GPU resources and global config |
| `GaussianSplatNode3D` | `nodes/gaussian_splat_node_3d.cpp` | Main scene node for splat rendering |
| `GaussianSplatRenderer` | `renderer/gaussian_splat_renderer.cpp` | Core rendering pipeline (implements `IRenderer`) |

## Subsystem Map

### Core (`core/`)
- `gaussian_data.*` - Splat data storage (positions, colors, SH coefficients)
- `gaussian_streaming.*` - Chunk-based streaming with VRAM budget management (T8 refactor)
- `gaussian_splat_manager.*` - Global singleton, device management, config registry
- `gaussian_splat_scene_director.*` - Multi-instance coordination and registry
- `gaussian_splat_world.*` - World-scale owner for streaming assets

### Renderer (`renderer/`)
- `gaussian_splat_renderer.*` - Main render orchestration (T9 delegating to orchestrators)
- `render_pipeline_stages.*` - Cull -> Sort -> Raster -> Composite stage runner
- `render_*_orchestrator.*` - Focused subsystems for culling, sorting, streaming, output, and diagnostics (T9 long-method cleanup)
- `gpu_sorter.*` - GPU sorting (Bitonic, Radix, OneSweep)
- `gpu_memory_stream.*` - Triple-buffered GPU uploads
- `tile_renderer.*` - Tile-based rasterization

### Nodes (`nodes/`)
- `gaussian_splat_node_3d.*` - Primary scene node
- `gaussian_splat_container.*` - Multi-splat container
- `gaussian_splat_world_3d.*` - World-scale streaming entry point
- `gaussian_splat_dynamic_instance_3d.*` - Lightweight instance registration

### Interfaces (`interfaces/`)
- `renderer_interfaces.h` - `IRenderer` contract and provider
- `rasterizer_interfaces.h`, `gpu_sorting_pipeline_interfaces.h`, `output_compositor_interfaces.h` - Dependency-inverted rendering components

### Memory Subsystem (shared)
- `renderer/gpu_buffer_manager.*` - Resident buffers for non-streaming data
- `renderer/gpu_memory_stream.*` - Triple-buffered uploads + pooling for streaming
- `core/gaussian_streaming.*` - VRAM budget regulation and eviction policy
- See [MEMORY_SUBSYSTEM](MEMORY_SUBSYSTEM.md) for the full budget/config flow

## Data Flow

1. `GaussianSplatNode3D` registers data and instance transforms through `GaussianSplatSceneDirector` (`core/gaussian_splat_scene_director.*`).
2. `GaussianData` and `GaussianSplatAsset` feed the `GaussianStreamingSystem` (`core/gaussian_streaming.*`) for visibility, eviction, and upload decisions.
3. `GaussianSplatRenderer` coordinates GPU residency via `GPUMemoryStream` (`renderer/gpu_memory_stream.*`) and prepares sorting inputs.
4. `RenderPipelineStages` (`renderer/render_pipeline_stages.*`) executes cull -> sort -> raster -> composite, delegating to the orchestrators from T9.
5. `TileRenderer` (`renderer/tile_renderer.*`) writes color/depth targets, and `RenderOutputOrchestrator` (`renderer/render_output_orchestrator.*`) composites into the active viewport or `RenderSceneBuffersRD`.

## State Structs (from T8 Refactor)

The streaming system in `core/gaussian_streaming.h` uses state structs and companion classes for clear separation:
- `VisibilityState` - Chunk culling, camera tracking, LOD blending
- `EvictionState` - LRU eviction, hysteresis tracking
- `StreamingUploadPipeline` - Async pack threads, upload bandwidth (extracted to `core/streaming_upload_pipeline.h`)
- `BudgetState` - VRAM regulation, loaded chunk tracking

See [READING_ORDER](READING_ORDER.md) for a guided walkthrough and [ABBREVIATIONS](ABBREVIATIONS.md) for naming conventions.
