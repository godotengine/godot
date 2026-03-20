# Quick Start

_Last updated: 2026-03-16_

Purpose: first visible splat with minimal setup.

Run commands from repository root (`godotgs/`).

## 1) Get a Module-Built Editor

Already built one? Set it directly:

```bash
export GODOT_BINARY=/absolute/path/to/your/godot-editor
```

```powershell
$env:GODOT_BINARY="C:\absolute\path\to\your\godot-editor.exe"
```

Need to build now?

```bash
# Linux
scons platform=linuxbsd target=editor dev_build=yes -j$(nproc)

# Windows (Developer Command Prompt)
scons platform=windows target=editor dev_build=yes -j10

# macOS (Apple Silicon)
scons platform=macos target=editor dev_build=yes arch=arm64 -j8
```

Then:

```bash
export GODOT_BINARY=./bin/<your-editor-binary>
```

```powershell
$env:GODOT_BINARY=".\bin\<your-editor-binary>.exe"
```

You should see:
- an editor binary in `bin/`
- the binary was built from this fork root and includes `modules/gaussian_splatting`

## 2) Generate Synthetic Starter Assets

```bash
python3 tests/runtime/prepare_synthetic_assets.py --quiet
```

```powershell
python .\tests\runtime\prepare_synthetic_assets.py --quiet
```

You should see:
- `tests/examples/godot/test_project/tests/fixtures/test_splats.ply` in the project

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
- Godot editor running with `tests/examples/godot/test_project`

## 4) Render Your First Splat

1. Open `res://scenes/benchmark_unified.tscn`.
2. If needed, add `GaussianSplatNode3D`.
3. Set `PLY File Path` to `res://tests/fixtures/test_splats.ply`.
4. Press `F6` (Play Scene).

You should see:
- a visible splat in the viewport

## Need Help?

- Artist-specific resources: [../user/quickstart.md](../user/quickstart.md)
- Installation details: [installation.md](installation.md)
- Recurring fixes: [../troubleshooting/recurring-issues.md](../troubleshooting/recurring-issues.md)
