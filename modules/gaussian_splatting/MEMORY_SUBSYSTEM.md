# Memory Subsystem Guide

Related docs: [ARCHITECTURE](ARCHITECTURE.md), [READING_ORDER](READING_ORDER.md), [ABBREVIATIONS](ABBREVIATIONS.md), [README](README.md)

This module has two distinct GPU memory paths that share data but serve different runtime modes. The goal is to keep budget logic centralized while allowing each path to manage its own buffers.

## High-Level Layout

```
Resident path (non-streaming)
  GaussianSplatRenderer
    -> GPUBufferManager (double-buffered resident storage)

Streaming path
  GaussianSplatRenderer
    -> GaussianStreamingSystem (visibility + budget)
        -> GaussianMemoryStream (triple-buffered uploads + pool)
```

## Components and Responsibilities

### GPUBufferManager (resident data)
- **Files**: `renderer/gpu_buffer_manager.h`, `renderer/gpu_buffer_manager.cpp`
- **Role**: Allocates and manages resident GPU buffers used when streaming is not active.
- **Buffers**: Double-buffered Gaussian data + sort keys + indices.
- **Memory tracking**: Provides `get_memory_usage_mb()` as a size estimate, but does **not** enforce budgets.

### GaussianMemoryStream (streamed uploads)
- **Files**: `renderer/gpu_memory_stream.h`, `renderer/gpu_memory_stream.cpp`
- **Role**: Triple-buffered upload path with a suballocation pool for streaming chunks.
- **Buffers**: Three GPU buffers + pooled suballocations for reuse.
- **Memory tracking**: Reports allocated/used MB and efficiency; does **not** decide budgets.

### GaussianStreamingSystem + VRAMBudgetRegulator (budgeting)
- **Files**: `core/gaussian_streaming.h`, `core/gaussian_streaming.cpp`
- **Role**: Owns VRAM budget policy and eviction/LOD decisions. This is the **only** place that regulates VRAM budgets.
- **Key structs**: `VRAMBudgetConfig`, `VRAMBudgetRegulator`, `BudgetState`.

## Budget Configuration Flow

1. **Defaults** are defined in ProjectSettings via `core/gaussian_splat_manager.cpp`.
2. **Tier presets** apply caps through `QualityTierConfig` and `GaussianSplatNode3D::_apply_quality_tier_limits`.
3. **Per-node overrides** are assembled in `GaussianSplatNode3D::_apply_renderer_settings` and passed into the streaming system via `ConfigOverrides`.
4. **Streaming system** applies overrides to the `VRAMBudgetRegulator` and drives eviction based on usage.

This flow prevents duplication: only the streaming system enforces VRAM budget policy, while buffer managers expose usage stats.

## When to Use Which Path

- **Resident path** (`GPUBufferManager`): small/medium datasets, no streaming, lower per-frame overhead.
- **Streaming path** (`GaussianMemoryStream` + `GaussianStreamingSystem`): large datasets, dynamic loading, budget-aware eviction.

## Debugging and Metrics

- **Budget warnings**: `GaussianStreamingSystem::is_vram_budget_warning_active()`
- **Budget stats**: `GaussianStreamingSystem::get_vram_debug_stats()`
- **Stream usage**: `GaussianMemoryStream::get_allocated_memory_mb()`, `get_used_memory_mb()`, `get_memory_efficiency()`
- **Resident usage**: `GPUBufferManager::get_memory_usage_mb()`

## Notes for Future Refactors

- Avoid moving budget logic into `GPUBufferManager` or `GaussianMemoryStream`; the regulator in `core/gaussian_streaming.*` is the single source of truth.
- If a unified memory subsystem is introduced later, preserve the separation between **budget policy** and **buffer allocation**.
