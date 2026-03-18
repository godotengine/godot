# JetBrains Rider IDE Setup for Gaussian Splatting Module

This guide helps you set up JetBrains Rider for developing the Godot Gaussian Splatting module with full IntelliSense, debugging, and code navigation.

## Quick Start

1. **Open the Module in Rider**
   - Launch JetBrains Rider
   - File → Open → Select the `CMakeLists.txt` file in this directory
   - Rider will load the project with full C++ support

2. **Build the Module**
   - Use the pre-configured run configuration: **Build Module**
   - Or press `Ctrl+Shift+F10` and select "Build Module"
   - This runs SCons with the correct parameters

3. **Debug the Module**
   - Set breakpoints in any `.cpp` file
   - Use the **Debug Godot Editor** configuration
   - Press `Shift+F9` to start debugging

## Project Structure in Rider

The CMakeLists.txt organizes the code into logical groups:

```
├── Core                    # Core data structures and registration
├── Rendering              # GPU rendering pipeline
│   ├── Renderer           # Main rendering system
│   ├── Painterly          # Artistic rendering features
│   └── Compute            # Compute shader infrastructure
├── Data Management        # Asset and resource handling
│   ├── IO                 # PLY loading and file I/O
│   ├── Assets             # Asset management system
│   └── Persistence        # Save/load functionality
├── Scene Integration      # Godot scene system integration
│   ├── Nodes              # GaussianSplatNode3D
│   └── Animation          # Animation support
├── Systems                # Supporting systems
│   ├── LOD                # Level of detail
│   ├── Interfaces         # Runtime bridge contracts
│   └── Logger             # Diagnostics and logging
├── Editor                 # Editor-only features
├── Resources              # Non-code resources
│   ├── Shaders            # GLSL shaders
│   └── Scripts            # GDScript files
└── Documentation          # Markdown documentation
```

## IntelliSense Configuration

IntelliSense should work automatically, but if you have issues:

1. **Invalidate Caches**: File → Invalidate Caches → Restart
2. **Check CMake Output**: View → Tool Windows → CMake → Check for errors
3. **Verify Include Paths**: Settings → Build, Execution, Deployment → CMake

## Run Configurations

Pre-configured run configurations are available:

### Build Module
- **What**: Builds the module using SCons
- **When**: After making code changes
- **Shortcut**: `Ctrl+F9`

### Debug Godot Editor
- **What**: Launches Godot editor with debugger attached
- **When**: Testing and debugging your module
- **Shortcut**: `Shift+F9`

### Run Tests
- **What**: Runs module unit tests
- **When**: Validating changes
- **Shortcut**: `Shift+F10`

## Code Style

The `.clang-format` file enforces Godot's code style:
- **Format File**: `Ctrl+Alt+L`
- **Format on Save**: Settings → Tools → Actions on Save → Reformat code

## Debugging Tips

### Setting Breakpoints
1. Click in the gutter next to any line number
2. Red dot appears indicating breakpoint
3. Conditional breakpoints: Right-click → Edit Breakpoint

### Viewing Variables
- **Variables Window**: Shows local variables when paused
- **Watches Window**: Add custom expressions to monitor
- **Evaluate Expression**: `Alt+F8` while debugging

### Common Debugging Scenarios

#### Debug PLY Loading
```cpp
// Set breakpoint in io/ply_loader.cpp
Error PLYLoader::load_file(const String &p_path) {
    // Breakpoint here to debug loading issues
}
```

#### Debug Rendering
```cpp
// Set breakpoint in renderer/gaussian_splat_renderer.cpp
void GaussianSplatRenderer::render_gaussians(RenderDataRD *p_render_data) {
    // Breakpoint here to debug rendering
}
```

#### Debug Node Creation
```cpp
// Set breakpoint in nodes/gaussian_splat_node_3d.cpp
void GaussianSplatNode3D::_notification(int p_what) {
    if (p_what == NOTIFICATION_READY) {
        // Breakpoint here for node setup
    }
}
```

## Performance Profiling

### CPU Profiling
1. Run → Profile 'Debug Godot Editor'
2. Use sampling profiler for hot spots
3. Analyze call tree for bottlenecks

### Memory Profiling
1. Run → Profile → Memory
2. Monitor allocations in GPU buffer manager
3. Check for memory leaks in streaming system

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Build Module | `Ctrl+F9` |
| Debug | `Shift+F9` |
| Run | `Shift+F10` |
| Find Usages | `Alt+F7` |
| Go to Definition | `Ctrl+B` |
| Go to Implementation | `Ctrl+Alt+B` |
| Find in Path | `Ctrl+Shift+F` |
| Replace in Path | `Ctrl+Shift+R` |
| Refactor → Rename | `Shift+F6` |
| Generate Code | `Alt+Insert` |
| Quick Documentation | `Ctrl+Q` |
| Parameter Info | `Ctrl+P` |
| Format Code | `Ctrl+Alt+L` |
| Optimize Imports | `Ctrl+Alt+O` |

## Troubleshooting

### IntelliSense Not Working
```bash
# Regenerate CMake project
File → Reload CMake Project

# Or manually:
cd modules/gaussian_splatting
rm -rf cmake-build-*
# Reopen in Rider
```

### Build Errors
```bash
# Clean build
cd <repo-root>
scons -c
scons platform=<windows|linuxbsd|macos> target=editor dev_build=yes modules/gaussian_splatting
```

### Debugger Not Attaching
1. Ensure debug symbols: `debug_symbols=yes` in SCons
2. Check firewall isn't blocking debugger
3. Verify executable path in run configuration

### Slow Performance
1. Increase Rider memory: Help → Change Memory Settings
2. Exclude build directories: Right-click `bin/` → Mark as → Excluded
3. Disable unnecessary plugins: Settings → Plugins

## Advanced Features

### Custom Build Scripts
Add to CMakeLists.txt:
```cmake
add_custom_target(my_custom_build
    COMMAND your_command_here
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
```

### External Tools
Settings → Tools → External Tools → Add:
- **Name**: Format Shaders
- **Program**: `clang-format`
- **Arguments**: `-i $FilePath$`
- **Working directory**: `$FileDir$`

### File Watchers
Settings → Tools → File Watchers → Add:
- Watch `.glsl` files
- Run shader compiler on change
- Auto-reload in editor

## Resources

- [Rider Documentation](https://www.jetbrains.com/help/rider/)
- [Godot Development](https://docs.godotengine.org/en/stable/contributing/development/index.html)
- [CMake Documentation](https://cmake.org/documentation/)
- [SCons with Godot](https://docs.godotengine.org/en/stable/contributing/development/compiling/index.html)

## Getting Help

- **Rider Issues**: File → Help → Submit Feedback
- **Module Issues**: Create issue on GitHub
- **Godot Questions**: https://godotengine.org/community

## Tips for Productive Development

1. **Use Live Templates**: Settings → Editor → Live Templates
   - Create templates for common Godot patterns
   - Example: `gdc` → `GDCLASS($CLASS$, $BASE$);`

2. **Enable GPU Debugging**: For shader development
   - Install RenderDoc
   - Configure as external tool

3. **Use TODO Comments**: View → Tool Windows → TODO
   - Track work items directly in code
   - `// TODO: Optimize this loop`

4. **Version Control Integration**:
   - Git integration works out of the box
   - Use annotate/blame: Right-click → Git → Annotate

5. **Database Tools**: For PLY file inspection
   - View → Tool Windows → Database
   - Add PLY files as data sources

---

Happy coding! 🚀
