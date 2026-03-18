# User Quickstart (Artists / Non-Programmers)

Goal: get a first visible splat in a few minutes.

Run commands from repository root (`godotgs/`).

## 1) Choose Your Editor Binary

If you already have a module-built editor binary, use it:

```bash
export GODOT_BINARY=/absolute/path/to/your/godot-editor
```

```powershell
$env:GODOT_BINARY="C:\absolute\path\to\your\godot-editor.exe"
```

If you do not have one, build it now:

```bash
# Linux
scons platform=linuxbsd target=editor dev_build=yes -j$(nproc)

# Windows (Developer Command Prompt)
scons platform=windows target=editor dev_build=yes -j10

# macOS (Apple Silicon)
scons platform=macos target=editor dev_build=yes arch=arm64 -j8
```

Then set:

```bash
export GODOT_BINARY=./bin/<your-editor-binary>
```

```powershell
$env:GODOT_BINARY=".\bin\<your-editor-binary>.exe"
```

You should see:
- a file in `bin/` for your platform
- the build includes the in-tree `modules/gaussian_splatting` module

## 2) Generate Synthetic Test Assets

```bash
python3 tests/runtime/prepare_synthetic_assets.py --quiet
```

```powershell
python .\tests\runtime\prepare_synthetic_assets.py --quiet
```

You should see:
- `tests/examples/godot/test_project/tests/fixtures/test_splats.ply` exists

## 3) Open the Sample Project

```bash
# Bash (Linux/macOS)
$GODOT_BINARY --path tests/examples/godot/test_project
```

```powershell
# PowerShell (Windows)
& $env:GODOT_BINARY --path .\tests\examples\godot\test_project
```

You should see:
- the Godot editor opens
- the project path is `tests/examples/godot/test_project`

## 4) Make the First Splat Visible

1. Open `res://scenes/benchmark_unified.tscn`.
2. If no splat node is present, add `GaussianSplatNode3D`.
3. In Inspector, set `PLY File Path` to `res://tests/fixtures/test_splats.ply`.
4. Press `F6` (Play Scene).

You should see:
- a visible splat/point-cloud style object in the viewport

## Next

- Quick start (getting-started lane): [../getting-started/quick-start.md](../getting-started/quick-start.md)
- Troubleshooting: [../troubleshooting/recurring-issues.md](../troubleshooting/recurring-issues.md)
- Artist docs: [manual/concepts.md](manual/concepts.md)
