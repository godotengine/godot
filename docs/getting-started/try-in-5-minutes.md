# Try in 5 Minutes

This is the shortest honest path to a visible result: download the Linux nightly editor, open the sample project, and confirm the public evaluator in the viewport. The Windows release path exists in the workflow, but a public Windows release has not landed yet, so Windows users should use [Build from Source](../BUILDING.md) for now. macOS still starts with [Build from Source](../BUILDING.md).

## 1. Get the Linux Nightly Editor

Open the repository releases page and download the newest Linux nightly editor archive:

- [GitHub Releases](https://github.com/klausi3D/godotGS/releases)

If you are on Windows, stop here and use [Build from Source](../BUILDING.md). The Windows release path is already wired into the workflow, but it is not yet visible on Releases. If you are on macOS, stop here and use [Build from Source](../BUILDING.md).

## 2. Open the Project

Launch the Linux editor you downloaded, then point `GODOT_BINARY` at it and open the sample project:

```bash
export GODOT_BINARY=/absolute/path/to/godot.linuxbsd.editor.dev.x86_64
$GODOT_BINARY --path tests/examples/godot/test_project
```

If you do not already have a binary on your path, use the editor you downloaded from Releases.

## 3. Verify the Public Evaluator

Press Play. The sample project opens `res://scenes/public_evaluator.tscn` by default.

You should see:

- a visible splat in the viewport
- the sample project remains open and interactive

## If It Fails

- Read [Public Evaluator](quick-start.md) for the slower canonical flow.
- Check [Recurring Issues](../troubleshooting/recurring-issues.md).
- On macOS, stop using this page and build an editor from this fork with [Build from Source](../BUILDING.md) before retrying.
