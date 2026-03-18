# Architecture Overview

This page is the canonical high-level architecture entrypoint.

## Subsystem Map

- Registration/lifecycle: [../../modules/gaussian_splatting/register_types.cpp](../../modules/gaussian_splatting/register_types.cpp)
- Core systems: [../../modules/gaussian_splatting/core/](../../modules/gaussian_splatting/core/)
- Renderer pipeline: [../../modules/gaussian_splatting/renderer/](../../modules/gaussian_splatting/renderer/)
- Nodes/editor integration: [../../modules/gaussian_splatting/nodes/](../../modules/gaussian_splatting/nodes/), [../../modules/gaussian_splatting/editor/](../../modules/gaussian_splatting/editor/)
- IO/import: [../../modules/gaussian_splatting/io/](../../modules/gaussian_splatting/io/)

## Detailed Architecture Docs

- Render pipeline details: [render-pipeline.md](render-pipeline.md)
- Lighting and shadows details: [lighting-system.md](lighting-system.md)
- Module-wide architecture map: [../../modules/gaussian_splatting/ARCHITECTURE.md](../../modules/gaussian_splatting/ARCHITECTURE.md)
- Memory and residency invariants: [../../modules/gaussian_splatting/MEMORY_SUBSYSTEM.md](../../modules/gaussian_splatting/MEMORY_SUBSYSTEM.md)

## Data Flow (High Level)

1. Source asset is imported/loaded.
2. Node and asset state are registered with runtime systems.
3. Visibility, sorting, and raster/composite stages execute.
4. Debug/performance counters are emitted for diagnostics.

## Debugging and Performance

- Timing monitor semantics: [../timing_metrics_reference.md](../timing_metrics_reference.md)
- Recurring render/build issues: [../troubleshooting/recurring-issues.md](../troubleshooting/recurring-issues.md)
