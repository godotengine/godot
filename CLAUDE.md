# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a fork of the Godot Engine (v4.6) with VisionOS XR support. The current branch `apple/visionos-xr` contains the VisionOS platform port and XR interface implementation for Apple Vision Pro.

## Build Commands

The build system uses SCons (Python-based). Build commands follow iOS patterns but use `visionos` as the platform.

**Build for device (arm64):**
```bash
scons platform=visionos arch=arm64 target=template_debug
scons platform=visionos arch=arm64 target=template_release
```

**Build for simulator (x86_64):**
```bash
scons platform=visionos arch=x86_64 target=template_debug simulator=yes
```

**Common build options:**
- `target=editor` - Build the editor (not available for visionOS)
- `target=template_debug` - Debug export template
- `target=template_release` - Release export template
- `dev_build=yes` - Development build with extra debugging
- `compiledb=yes` - Generate compile_commands.json for IDE support
- `-j$(nproc)` - Parallel compilation

**VisionOS-specific constraints:**
- Uses Metal rendering exclusively (Vulkan and OpenGL3 are disabled)
- LTO disabled by default to avoid slow Xcode linking
- Simulator builds don't support Metal rendering

## Code Quality Tools

**Pre-commit hooks (run all):**
```bash
pre-commit run --all-files
```

**Run clang-tidy (manual, expensive):**
```bash
pre-commit run --hook-stage manual clang-tidy
```

**Individual tools:**
- C++: clang-format (automatic)
- Python: ruff (check + format), mypy (type checking)
- Spelling: codespell

## Testing

Tests are built into the engine binary using the doctest framework.

**Run tests:**
```bash
./bin/godot.* --test
./bin/godot.* --test --test-case="*specific_test*"
```

**Create new test:**
```bash
python tests/create_test.py
```

## Architecture

### Platform Abstraction

- `platform/visionos/` - VisionOS platform implementation (Swift, Objective-C++)
- `drivers/apple_embedded/` - Abstract Apple Embedded platform (shared with iOS)
- `drivers/apple/` - Shared Apple code (macOS, iOS, visionOS)

### VisionOS XR System

The XR implementation is in `modules/visionos_xr/`:

- `VisionOSXRInterface` - Main XR interface inheriting from `XRInterface`
- Uses ARKit for spatial world tracking (`ar_world_tracking_provider_t`)
- Uses CompositorServices for stereoscopic rendering (`cp_layer_renderer_t`)
- Manages device anchors, frame timing, and drawable presentation
- Has a nested `RenderThread` class for thread-safe rendering operations

**Key patterns:**
- AR session and device anchors must be queried per-thread (not shareable)
- Frame updates are split between main thread (process) and render thread
- Metal command buffers are used for presenting to the compositor

### Module System

Modules in `modules/` follow this structure:
- `config.py` - Build configuration
- `register_types.h/cpp` - Type registration with the engine
- `doc_classes/` - XML class documentation
- `SCsub` - SCons build rules

### Rendering Pipeline

- `servers/rendering/` - Rendering server abstraction
- `drivers/metal/` - Metal rendering driver (mandatory for visionOS)
- Shaders compiled via `gles3_builders.py` and `glsl_builders.py`

### XR Server Architecture

- `servers/xr/xr_interface.h` - Base XR interface class
- `servers/xr/xr_positional_tracker.h` - Positional tracking abstraction
- Platform implementations register with `XRServer` singleton

## Key Files

- `SConstruct` - Main build file
- `methods.py` - Core build utilities
- `platform/visionos/detect.py` - visionOS build detection and configuration
- `modules/visionos_xr/visionos_xr_interface.h` - Main VisionOS XR interface header

## Important Guidelines

- Never compile or recompile the project unless explicitly requested by the user.

## Commit Message Format

Follow Godot conventions:
- First line: < 72 chars, imperative mood, capital letter
- Prefix with area if applicable: "VisionOS: Fix frame timing issue"
- Extended description wrapped at 80 chars after blank line
