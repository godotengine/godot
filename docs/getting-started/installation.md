# Installation

_Last updated: 2026-02-25_

## Purpose
Build a custom Godot editor with `gaussian_splatting` and verify the sample project boots.

## Usage
| Requirement | Details |
| --- | --- |
| Python | 3.10 or newer |
| SCons | 4.5 or newer |
| Compiler | Platform C++ toolchain compatible with Godot 4.5 |
| Linux packages | Install the Linux package set listed in [../BUILDING.md](../BUILDING.md) before running `scons` |
| GPU | Vulkan 1.2 or newer for runtime rendering |

Use [../BUILDING.md](../BUILDING.md) for platform package setup.

| Task | Run from | Command |
| --- | --- | --- |
| Build editor (Linux) | repository root | `scons platform=linuxbsd target=editor dev_build=yes -j$(nproc)` |
| Build editor (Windows) | repository root | `scons platform=windows target=editor dev_build=yes -j10` |
| Build editor (macOS arm64) | repository root | `scons platform=macos target=editor dev_build=yes arch=arm64 -j8` |
| Launch sample project (Linux dev build) | repository root | `./bin/godot.linuxbsd.editor.dev.x86_64 --path tests/examples/godot/test_project` |
| Headless boot check (Linux dev build) | repository root | `./bin/godot.linuxbsd.editor.dev.x86_64 --headless --path tests/examples/godot/test_project --quit` |

## API
| Item | Reference |
| --- | --- |
| `GaussianSplatNode3D` registration | `modules/gaussian_splatting/register_types.cpp:82` |
| `GaussianSplatDynamicInstance3D` registration | `modules/gaussian_splatting/register_types.cpp:84` |
| `GaussianSplatWorld3D` registration | `modules/gaussian_splatting/register_types.cpp:85` |
| `GaussianSplatManager` singleton registration | `modules/gaussian_splatting/register_types.cpp:100` |

## Examples
```gdscript
var node := GaussianSplatNode3D.new()
```

## Troubleshooting
| Symptom | Action | Source |
| --- | --- | --- |
| `GaussianSplatNode3D` is missing in the editor. | Rebuild the editor from repository root, then relaunch the binary from `bin/`. | `modules/gaussian_splatting/register_types.cpp:82` |
| Linux launch command cannot find executable. | Use the `dev_build=yes` binary name (`godot.linuxbsd.editor.dev.x86_64`). | [docs/getting-started/quick-start.md](quick-start.md) |
| Linux build logs show `WARNING: wayland-scanner not found`. | Install `libwayland-bin` (and `wayland-protocols`) from [docs/BUILDING.md](../BUILDING.md) and rebuild if you need Wayland backend support. | [docs/BUILDING.md](../BUILDING.md) |
| `GaussianData.load_from_file()` is unresolved in scripts. | Use the bound API exactly as declared and bound. | `modules/gaussian_splatting/core/gaussian_data.h:447`, `modules/gaussian_splatting/core/gaussian_data.cpp:174` |
