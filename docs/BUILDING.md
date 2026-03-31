# Build from Source

Build a Godot editor that includes `modules/gaussian_splatting` from this repository checkout.

## Before you build

| Requirement | Details |
| --- | --- |
| Python | 3.10 or newer |
| SCons | 4.5 or newer |
| Compiler | Platform C++ toolchain compatible with Godot 4.5 |
| Engine-level platform packages | Follow the repository-root [engine build notes](../BUILDING.md) for platform package and compiler setup |

Run all build commands from repository root.

## Build Commands

=== "Linux"

    ```bash
    scons platform=linuxbsd target=editor dev_build=yes -j$(nproc)
    ```

=== "Windows"

    ```powershell
    scons platform=windows target=editor dev_build=yes -j10
    ```

=== "macOS (Apple Silicon)"

    ```bash
    scons platform=macos target=editor dev_build=yes arch=arm64 -j8
    ```

For a test-enabled editor build:

```bash
scons platform=<platform> target=editor dev_build=yes tests=yes -j<jobs>
```

## Output Naming

- `dev_build=yes` adds a `.dev` segment to the binary name.
- Windows example: `bin/godot.windows.editor.dev.x86_64.exe`
- Linux example: `bin/godot.linuxbsd.editor.dev.x86_64`

## Next Pages

- [First Run](getting-started/quick-start.md) for the canonical install / first visible result path.
- [Build / Test / CI Command Reference](reference/build-test-ci.md) for test runners and CI entrypoints.
- [Versioned Docs Site](development/docs-site.md) for docs publishing commands.
- [Recurring Issues](troubleshooting/recurring-issues.md) for build failures and known fixes.
