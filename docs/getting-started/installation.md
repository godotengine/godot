# Installation

_Last updated: 2026-02-25_

!!! info "Scope"
    For people who need prerequisite, toolchain, or binary-selection details before following the canonical [First Run](quick-start.md) path.
    This page covers setup requirements and how to obtain a module-enabled editor.
    It complements [Build from Source](../BUILDING.md).

## Purpose
Prepare a module-enabled Godot editor before you begin the canonical first-run flow.

## Usage
| Requirement | Details |
| --- | --- |
| Python | 3.10 or newer |
| SCons | 4.5 or newer |
| Compiler | Platform C++ toolchain compatible with Godot 4.5 |
| Linux packages | Install the Linux package set listed in [../BUILDING.md](../BUILDING.md) before running `scons` |
| GPU | Vulkan 1.2 or newer for runtime rendering |

Use [../BUILDING.md](../BUILDING.md) for platform package setup and local build commands.

| Option | When to use it | Next page |
| --- | --- | --- |
| Reuse an existing module-enabled editor | You already have a binary built from this fork and only need to point docs commands at it. | [First Run](quick-start.md) |
| Build an editor locally | You need a fresh binary from this checkout. | [Build from Source](../BUILDING.md) |
| Build a test-enabled editor | You plan to run guard, QA, or runtime validation lanes. | [Build / Test / CI Command Reference](../reference/build-test-ci.md) |

## Verification

Once you have a module-enabled editor, confirm it opens the sample project before continuing:

```bash
$GODOT_BINARY --headless --path tests/examples/godot/test_project --quit
```

```powershell
& $env:GODOT_BINARY --headless --path .\tests\examples\godot\test_project --quit
```

Then continue with [First Run](quick-start.md).

## API
| Item | Reference |
| --- | --- |
| `GaussianSplatNode3D` registration | `modules/gaussian_splatting/register_types.cpp:77` |
| `GaussianSplatDynamicInstance3D` registration | `modules/gaussian_splatting/register_types.cpp:80` |
| `GaussianSplatWorld3D` registration | `modules/gaussian_splatting/register_types.cpp:81` |
| `GaussianSplatManager` singleton registration | `modules/gaussian_splatting/register_types.cpp:92-100` |

## Examples
```gdscript
var node := GaussianSplatNode3D.new()
```

## Troubleshooting
| Symptom | Action | Source |
| --- | --- | --- |
| `GaussianSplatNode3D` is missing in the editor. | Rebuild the editor from repository root, then relaunch the binary from `bin/`. | `modules/gaussian_splatting/register_types.cpp:77` |
| Linux launch command cannot find executable. | Use the `dev_build=yes` binary name shown in [Build from Source](../BUILDING.md). | [docs/BUILDING.md](../BUILDING.md) |
| Linux build logs show `WARNING: wayland-scanner not found`. | Install `libwayland-bin` (and `wayland-protocols`) from [Build from Source](../BUILDING.md) and rebuild if you need Wayland backend support. | [docs/BUILDING.md](../BUILDING.md) |
| `GaussianData.load_from_file()` is unresolved in scripts. | Use the bound API exactly as declared and bound. | `modules/gaussian_splatting/core/gaussian_data.h:447`, `modules/gaussian_splatting/core/gaussian_data.cpp:174` |
