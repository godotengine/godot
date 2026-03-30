# Try in 5 Minutes

This is the shortest honest path to a visible result: download the Linux nightly editor, open the sample project, and confirm a splat in the viewport.
If you are on Windows or macOS, use [Build from Source](../BUILDING.md) first and come back here after you have an editor from this fork.

## 1. Get the Linux Nightly Editor

Open the repository releases page and download the newest Linux nightly editor archive:

- [GitHub Releases](https://github.com/klausi3D/godotGS/releases)

If you are on Windows or macOS, stop here and use [Build from Source](../BUILDING.md).

## 2. Prepare the Sample Project

From repository root, prepare the sample asset used by the test project:

```bash
python3 tests/runtime/prepare_synthetic_assets.py --quiet
```

You should end up with the synthetic starter asset in the sample project's fixture folder.

## 3. Open the Project

Launch the Linux editor you downloaded, then point `GODOT_BINARY` at it and open the sample project:

```bash
export GODOT_BINARY=/absolute/path/to/godot.linuxbsd.editor.dev.x86_64
$GODOT_BINARY --path tests/examples/godot/test_project
```

If you do not already have a binary on your path, use the editor you downloaded from Releases.

## 4. Verify a Visible Splat

1. Open `res://scenes/benchmark_unified.tscn`.
2. If needed, add `GaussianSplatNode3D`.
3. Set `PLY File Path` to `res://tests/fixtures/test_splats.ply`.
4. Press `F6` to play the scene.

You should see:

- a visible splat in the viewport
- the sample project remains open and interactive

## If It Fails

- Read [First Run](quick-start.md) for the slower canonical flow.
- Check [Recurring Issues](../troubleshooting/recurring-issues.md).
- On Windows or macOS, stop using this page and build an editor from this fork with [Build from Source](../BUILDING.md) before retrying.
