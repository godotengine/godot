# Try in 5 Minutes

This is the shortest honest path to a visible result: download the latest nightly editor for your platform, open the sample project, and confirm the public evaluator in the viewport. macOS still starts with [Build from Source](../BUILDING.md).

## 1. Get the Nightly Editor

Open the [GitHub Releases](https://github.com/klausi3D/godotGS/releases) page, pick the most recent `nightly-YYYYMMDD` entry at the top of the list, and download the archive that matches your platform:

- **Linux:** `godotgs-linux-x86_64-<date>.tar.xz`
- **Windows:** `godotgs-windows-x86_64-<date>.zip` (contains both the GUI editor and the console wrapper — pick whichever fits your workflow)
- **macOS:** no published binary; stop here and use [Build from Source](../BUILDING.md).

See the [Downloads page](downloads.md) for verification and integrity-check details.

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
