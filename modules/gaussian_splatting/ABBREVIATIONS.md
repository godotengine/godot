# Abbreviation Glossary

Related docs: [ARCHITECTURE](ARCHITECTURE.md), [READING_ORDER](READING_ORDER.md), [README](README.md)

Common prefixes and abbreviations used in the Gaussian Splatting codebase:

## Prefixes

| Prefix | Meaning | Example |
|--------|---------|---------|
| `gs_` | Gaussian Splatting | `gs_logger`, `gs_device_utils` |
| `GS_` | Gaussian Splatting (macros/constants) | `GS_LOG_DEBUG`, `GS_MAX_SPLATS` |
| `sh_` | Spherical Harmonics | `sh_dc`, `sh_degree`, `sh_coefficients` |
| `SH_` | Spherical Harmonics (constants) | `SH_BAND_COUNT` |
| `rd_` | RenderingDevice | `rd->draw_list_begin()` |
| `RD_` | RenderingDevice (types) | `RD::UNIFORM_TYPE_STORAGE_BUFFER` |
| `p_` | Parameter (function argument) | `p_camera_transform` |
| `r_` | Reference output parameter | `r_result` |
| `_` | Private member/method | `_load_chunk()`, `_visible_count` |

## Common Abbreviations

| Abbrev | Full Term |
|--------|-----------|
| LOD | Level of Detail |
| VRAM | Video RAM (GPU memory) |
| LRU | Least Recently Used |
| SH | Spherical Harmonics |
| DC | DC term (zero-order SH coefficient) |
| AABB | Axis-Aligned Bounding Box |
| PLY | Polygon File Format (point cloud format) |
| SPZ | Compressed splat format |

## Struct/Class Naming

| Pattern | Meaning |
|---------|---------|
| `*State` | Grouped state variables (e.g., `VisibilityState`) |
| `*Config` | Configuration settings (e.g., `LODBlendConfig`) |
| `*Orchestrator` | Coordination logic (e.g., `RenderDataOrchestrator`) |
| `*Helper` | Utility functions extracted from main class |
| `I*` | Interface (e.g., `IRenderer`, `IFrameStateProvider`) |
