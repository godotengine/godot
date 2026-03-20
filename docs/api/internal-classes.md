# Internal Classes Index

Classes listed below are registered with Godot's ClassDB but serve as internal
infrastructure. They do not have dedicated API reference pages because they are
either used exclusively by higher-level components, expose no scripting-facing
methods, or are abstract base types. Each entry notes the registration site in
`register_types.cpp` and a one-line purpose summary.

---

## Core Infrastructure

| Class | Description | Registration |
|-------|-------------|-------------|
| **GaussianSplatSceneDirector** | Singleton coordinator that manages multi-instance rendering, wind mode, and per-world scene orchestration across all active splat nodes. | `register_types.cpp:89` |
| **VRAMBudgetRegulator** | Monitors GPU memory pressure and throttles chunk loading to stay within a configurable VRAM budget. Used internally by `GaussianStreamingSystem`. | `register_types.cpp:74` |

## Rendering Internals

| Class | Description | Registration |
|-------|-------------|-------------|
| **GPUBufferManager** | Manages allocation, resizing, and lifetime of RenderingDevice GPU buffers consumed by the splat renderer. | `register_types.cpp:87` |
| **ColorGradingResource** | Resource holding exposure, contrast, saturation, temperature, tint, and hue-shift parameters for post-process color grading of splats. | `register_types.cpp:90` |

## GPU Sorting Implementations

| Class | Description | Registration |
|-------|-------------|-------------|
| **IGPUSorter** | Abstract base class defining the GPU sorting interface for depth-order sorting of visible Gaussians. | `register_types.cpp:125` |
| **BitonicSort** | Bitonic merge sort implementation of `IGPUSorter`, suited for small-to-medium splat counts with predictable dispatch cost. | `register_types.cpp:126` |
| **RadixSort** | Radix sort implementation of `IGPUSorter`, offering linear-time performance for large splat counts. | `register_types.cpp:127` |
| **OneSweepSort** | Single-pass radix sort (OneSweep algorithm) implementation of `IGPUSorter`, providing high throughput with reduced synchronization barriers. | `register_types.cpp:128` |

## Culling

| Class | Description | Registration |
|-------|-------------|-------------|
| **ClusterCuller** | GPU-accelerated two-level hierarchical culler implementing LiteGS-style coarse culling with cluster bounding spheres and per-cluster visibility testing. | `register_types.cpp:131` |

## IO Abstractions

| Class | Description | Registration |
|-------|-------------|-------------|
| **IGaussianLoader** | Abstract base class for Gaussian data loaders. Concrete implementations include `PLYLoader` and `SPZLoader`. | `register_types.cpp:122` |

## Animation and Persistence (v0.6.0)

| Class | Description | Registration |
|-------|-------------|-------------|
| **GaussianSplatting::GaussianAnimationStateMachine** | State machine resource that drives per-splat keyframe animation of position, color, opacity, scale, and rotation properties. | `register_types.cpp:134` |
| **GaussianSplatting::GaussianSceneSerializer** | Binary scene serializer using a chunked file format (magic `GSCF`) for saving and loading full Gaussian scenes with animation data. | `register_types.cpp:135` |
| **GaussianSplatting::GaussianIncrementalSaver** | Tracks per-splat changes and writes incremental delta files to avoid full scene re-serialization during editing sessions. | `register_types.cpp:136` |

## Asset Management (v0.7.0)

| Class | Description | Registration |
|-------|-------------|-------------|
| **AssetDependencyManager** | Tracks inter-asset dependencies using collision-resistant hashing, enabling dependency-aware loading, cache invalidation, and hot-reload workflows. | `register_types.cpp:139` |

---

## Cross-Reference

The following classes **do** have full API reference pages (XML doc_classes or
dedicated documentation):

- `GaussianData`
- `GaussianSplatAsset`
- `GaussianMemoryStream`
- `GaussianSplatContainer`
- `GaussianSplatDebugHUD`
- `GaussianSplatDynamicInstance3D`
- `GaussianSplatManager`
- `GaussianSplatNode3D`
- `GaussianSplatRenderer`
- `GaussianSplatWorld`
- `GaussianSplatWorld3D`
- `GaussianStreamingSystem`
- `PainterlyMaterial`
- `PLYLoader`
- `SPZLoader`
