# Build from Source

Build a Godot editor that includes `modules/gaussian_splatting` from this repository checkout.

## Before you build

Common prerequisites:

- Python 3.10 or newer
- SCons 4.5 or newer

Platform prerequisites:

- Linux: install the package set used by release CI in `.github/workflows/release_builds.yml`
  - `build-essential`
  - `scons`
  - `pkg-config`
  - `libx11-dev`
  - `libxcursor-dev`
  - `libxinerama-dev`
  - `libxi-dev`
  - `libxrandr-dev`
  - `libgl1-mesa-dev`
  - `libglu1-mesa-dev`
  - `libasound2-dev`
  - `libpulse-dev`
  - `libudev-dev`
  - `libvulkan-dev`
  - `mesa-vulkan-drivers`
- Windows: Visual Studio Build Tools or Visual Studio with the C++ workload installed, with the MSVC environment already set up. The CI shape is `ilammy/msvc-dev-cmd@v1`, and the docs in `modules/gaussian_splatting/tests/README.md` assume the Visual Studio build tools are on `PATH`.
- macOS: Xcode Command Line Tools or a full Xcode toolchain with Apple Clang available, plus a Vulkan SDK that provides MoltenVK. If SCons cannot find it automatically, pass `vulkan_sdk_path=<path-to-vulkan-sdk>` as the macOS detector suggests.

Run all build commands from repository root.

## Build Commands

### Linux

```bash
scons platform=linuxbsd target=editor dev_build=yes -j"$(nproc)"
```

### Windows

```powershell
scons platform=windows target=editor dev_build=yes -j10
```

### macOS Apple Silicon

```bash
scons platform=macos target=editor dev_build=yes arch=arm64 -j8
```

### macOS Intel

```bash
scons platform=macos target=editor dev_build=yes arch=x86_64 -j8
```

If you want a test-enabled editor build, append `tests=yes` to the same command.

## Smoke Test

Use the built editor to open the sample project from this repository. The command should exit cleanly after the project loads.

### Linux

```bash
./bin/godot.linuxbsd.editor.dev.x86_64 --headless --path tests/examples/godot/test_project --quit
```

### Windows

```powershell
.\bin\godot.windows.editor.dev.x86_64.exe --headless --path .\tests\examples\godot\test_project --quit
```

### macOS Apple Silicon

```bash
./bin/godot.macos.editor.dev.arm64 --headless --path tests/examples/godot/test_project --quit
```

### macOS Intel

```bash
./bin/godot.macos.editor.dev.x86_64 --headless --path tests/examples/godot/test_project --quit
```

## Output Naming

- `dev_build=yes` adds a `.dev` segment to the binary name.
- Windows example: `bin/godot.windows.editor.dev.x86_64.exe`
- Linux example: `bin/godot.linuxbsd.editor.dev.x86_64`
- macOS Apple Silicon example: `bin/godot.macos.editor.dev.arm64`
- macOS Intel example: `bin/godot.macos.editor.dev.x86_64`

## Next Pages

- [First Run](getting-started/quick-start.md) for the canonical install / first visible result path.
- [Build / Test / CI Command Reference](reference/build-test-ci.md) for test runners and CI entrypoints.
- [Versioned Docs Site](development/docs-site.md) for docs publishing commands.
- [Recurring Issues](troubleshooting/recurring-issues.md) for build failures and known fixes.
