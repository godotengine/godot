# Installation

Use this page when you need prerequisites, toolchain setup, or an editor built from this fork before [First Run](quick-start.md).

## What You Need

| Requirement | Details |
| --- | --- |
| Python | 3.10 or newer |
| SCons | 4.5 or newer |
| Compiler | Platform C++ toolchain compatible with Godot 4.5 |
| Linux packages | Install the Linux package set listed in [Build from Source](../BUILDING.md) before running `scons` |
| GPU | Vulkan 1.2 or newer for runtime rendering |

## Choose a Path

| Option | When to use it | Next step |
| --- | --- | --- |
| Reuse an editor built from this fork | You already have a working binary and only need to point the docs commands at it. | [First Run](quick-start.md) |
| Build an editor locally | You need a fresh binary from this checkout. | [Build from Source](../BUILDING.md) |
| Build an editor for validation | You plan to run guard, QA, or runtime validation commands. | [Build / Test / CI Command Reference](../reference/build-test-ci.md) |

## Verify the Editor

Once you have an editor built from this fork, confirm it opens the sample project before continuing:

```bash
$GODOT_BINARY --headless --path tests/examples/godot/test_project --quit
```

```powershell
& $env:GODOT_BINARY --headless --path .\tests\examples\godot\test_project --quit
```

Then continue with [First Run](quick-start.md).
