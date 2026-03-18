# Setting Up Run Configurations in Rider

Since Rider isn't detecting the configurations automatically, here's how to set them up manually.
Examples below show Windows paths; on Linux/macOS use the same commands with platform-appropriate paths and `platform=<linuxbsd|macos>` in SCons.

## Method 1: Import Existing Configurations

1. In Rider, go to **Run** → **Edit Configurations**
2. Click the **+** button → **Import Run Configuration**
3. Navigate to `.run` folder and select each `.run.xml` file

## Method 2: Manual Setup (Recommended)

### 1. Build Module Configuration

1. Click **Run** → **Edit Configurations**
2. Click **+** → **Shell Script**
3. Configure:
   - **Name**: `Build Module`
   - **Script path**: `scons`
   - **Script options**: `platform=<windows|linuxbsd|macos> target=editor dev_build=yes -j8 modules/gaussian_splatting`
   - **Working directory**: `C:\Projects\godotgs`
   - **Execute in terminal**:  Checked

### 2. Debug Godot Editor Configuration

1. Click **+** → **Native Application** (or **C++ Application**)
2. Configure:
   - **Name**: `Debug Godot Editor`
   - **Executable**: `C:\Projects\godotgs\bin\godot.windows.editor.x86_64.exe`
   - **Program arguments**: `--path "C:\Projects\godotgs\tests\examples\godot\test_project" --verbose`
   - **Working directory**: `C:\Projects\godotgs`
   - **Environment variables**: Click `...` → Add → `GODOT_DEV=1`
   - **Before launch**: Click **+** → **Run Another Configuration** → Select `Build Module`

### 3. Quick Build Configuration (Alternative)

1. Click **+** → **Shell Script**
2. Configure:
   - **Name**: `Quick Build`
   - **Script path**: `C:\Projects\godotgs\modules\gaussian_splatting\build_module.bat`
   - **Working directory**: `C:\Projects\godotgs\modules\gaussian_splatting`
   - **Execute in terminal**:  Checked

### 4. Run Tests Configuration

1. Click **+** → **Native Application**
2. Configure:
   - **Name**: `Run Tests`
   - **Executable**: `python3` (or `python` on Windows)
   - **Program arguments**: `tests/ci/run_module_tests.py`
   - **Working directory**: `C:\Projects\godotgs`

## Method 3: Using External Tools (Simple Alternative)

If run configurations are problematic, use External Tools:

1. **File** → **Settings** → **Tools** → **External Tools**
2. Click **+** to add new tool

### Build Tool:
- **Name**: `Build GS Module`
- **Program**: `scons`
- **Arguments**: `platform=<windows|linuxbsd|macos> target=editor dev_build=yes -j8 modules/gaussian_splatting`
- **Working directory**: `$ProjectFileDir$/../..`

### Run Godot Tool:
- **Name**: `Run Godot Editor`
- **Program**: `$ProjectFileDir$/../../bin/godot.windows.editor.x86_64.exe`
- **Arguments**: `--path "$ProjectFileDir$/../../tests/examples/godot/test_project"`
- **Working directory**: `$ProjectFileDir$/../..`

Then access via: **Tools** → **External Tools** → Your tool name

## Quick Terminal Commands

If all else fails, you can always use Rider's built-in terminal:

```bash
# Build
cd ../..
scons platform=<windows|linuxbsd|macos> target=editor dev_build=yes -j8 modules/gaussian_splatting

# Run
bin/<godot-editor-binary> --path tests/examples/godot/test_project

# Run module guards/tests (cross-platform canonical command)
python3 tests/ci/run_module_tests.py
```

## Debugging Setup

For debugging to work:

1. **Build with debug symbols**:
   ```
   scons platform=<windows|linuxbsd|macos> target=editor dev_build=yes debug_symbols=yes modules/gaussian_splatting
   ```

2. **Attach debugger**:
   - Start Godot normally
   - In Rider: **Run** → **Attach to Process**
   - Select `godot.windows.editor.x86_64.exe`
   - Choose **Native** debugger

3. **Set breakpoints**: Click in the gutter next to any line in your `.cpp` files

## Verification

To verify everything works:

1. Try building: Select `Build Module` → Click Run (green arrow)
2. Set a breakpoint in `io/ply_loader.cpp` at the start of `load_file()`
3. Debug: Select `Debug Godot Editor` → Click Debug (bug icon)
4. In Godot, try loading a PLY file - breakpoint should hit

## Troubleshooting

- **"Configuration not found"**: Create manually using Method 2
- **"Cannot start process"**: Check all paths are correct
- **"Breakpoints not hit"**: Ensure `debug_symbols=yes` in build
- **"SCons not found"**: Use full path to your `scons` executable

## Tips

- Pin frequently used configurations to the toolbar
- Use Shift+F10 to run last configuration
- Use Shift+F9 to debug last configuration
- Use Alt+Shift+F10 to select and run any configuration
