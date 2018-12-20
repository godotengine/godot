# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [3.0] - 2018-01-29

### Added

- Physically-based renderer using OpenGL ES 3.0.
  - Uses the Disney PBR model, with clearcoat, sheen and anisotropy parameters available.
  - Uses a forward renderer, supporting multi-sample anti-aliasing (MSAA).
  - Parallax occlusion mapping.
  - Reflection probes.
  - Screen-space reflections.
  - Real-time global illumination using voxel cone tracing (GIProbe).
  - Proximity fade and distance fade (useful for creating soft particles and various effects).
  - [Lightmapper](https://godotengine.org/article/introducing-new-last-minute-lightmapper) for lower-end desktop and mobile platforms, as an alternative to GIProbe.
- New SpatialMaterial resource, replacing FixedMaterial.
  - Multiple passes can now be specified (with an optional "grow" property), allowing for effects such as cel shading.
- Brand new 3D post-processing system.
  - Depth of field (near and far).
  - Fog, supporting light transmittance, sun-oriented fog, depth fog and height fog.
  - Tonemapping and Auto-exposure.
  - Screen-space ambient occlusion.
  - Multi-stage glow and bloom, supporting optional bicubic upscaling for better quality.
  - Color grading and various adjustments.
- Rewritten audio engine from scratch.
  - Supports audio routing with arbitrary number of channels, including Area-based audio redirection ([video](https://youtu.be/K2XOBaJ5OQ0)).
  - More than a dozen of audio effects included.
- Rewritten 3D physics using [Bullet](http://bulletphysics.org/).
- UDP-based high-level networking API using [ENet](http://enet.bespin.org/).
- IPv6 support for all of the engine's networking APIs.
- Visual scripting.
- Rewritten import system.
  - Assets are now referenced with their source files, then imported in a transparent manner by the engine.
  - Imported assets are now cached in a `.import` directory, making distribution and versioning easier.
  - Support for ETC2 compression.
  - Support for uncompressed Targa (.tga) textures, allowing for faster importing.
- Rewritten export system.
  - GPU-based texture compression can now be tweaked per-target.
  - Support for exporting resource packs to build DLC / content addons.
- Improved GDScript.
  - Pattern matching using the `match` keyword.
  - `$` shorthand for `get_node()`.
  - Setters and getters for node properties.
  - Underscores in number literals are now allowed for improved readability (for example,`1_000_000`).
  - Improved performance (+20% to +40%, based on various benchmarks).
- [Feature tags](http://docs.godotengine.org/en/latest/getting_started/workflow/export/feature_tags.html) in the Project Settings, for custom per-platform settings.
- Full support for the [glTF 2.0](https://www.khronos.org/gltf/) 3D interchange format.
- Freelook and fly navigation to the 3D editor.
- Built-in editor logging (logging standard output to a file), disabled by default.
- Improved, more intuitive file chooser in the editor.
- Smoothed out 3D editor zooming, panning and movement.
- Toggleable rendering information box in the 3D editor viewport.
  - FPS display can also be enabled in the editor viewport.
- Ability to render the 3D editor viewport at half resolution to achieve better performance.
- GDNative for binding languages like C++ to Godot as dynamic libraries.
  - Community bindings for [D](https://github.com/GodotNativeTools/godot-d), [Nim](https://github.com/pragmagic/godot-nim) and [Python](https://github.com/touilleMan/godot-python) are available.
- Editor settings and export templates are now versioned, making it easier to use several Godot versions on the same system.
- Optional soft shadows for 2D rendering.
- HDR sky support.
- Ability to toggle V-Sync while the project is running.
- Panorama sky support (sphere maps).
- Support for WebM videos (VP8/VP9 with Vorbis/Opus).
- Exporting to HTML5 using WebAssembly.
- C# support using Mono.
  - The Mono module is disabled by default, and needs to be compiled in at build-time.
  - The latest Mono version (5.4) can be used, fully supporting C# 7.0.
- Support for rasterizing SVG to images on-the-fly, using the nanosvg library.
  - Editor icons are now in SVG format, making them better-looking at non-integer scales.
  - Due to the library used, only simpler SVGs are well-supported, more complex SVGs may not render correctly.
- Support for oversampling DynamicFonts, keeping them sharp when scaled to high resolutions.
- Improved StyleBoxFlat.
  - Border widths can now be set per-corner.
  - Support for anti-aliased rounded and beveled corners.
  - Support for soft drop shadows.
- VeryLoDPI (75%) and MiDPI (150%) scaling modes for the editor.
- Improved internationalization support for projects.
  - Language changes are now effective without reloading the current scene.
- Implemented missing features in the HTML5 platform.
  - Cursor style changes.
  - Cursor capturing and hiding.
- Improved styling and presentation of HTML5 exports.
  - A spinner is now displayed during loading.
- Rewritten the 2D and 3D particle systems.
  - Particles are now GPU-based, allowing their use in much higher quantities than before.
  - Meshes can now be used as particles.
  - Particles can now be emitted from a mesh's shape.
  - Properties can now be modified over time using an editable curve.
  - Custom particle shaders can now be used.
- New editor theme, with customizable base color, highlight color and contrast.
  - A light editor theme option is now available, with icons suited to light backgrounds.
  - Alternative dark gray and Arc colors are available out of the box.
- New adaptive text editor theme, adjusting automatically based on the editor colors.
- Support for macOS trackpad gestures in the editor.
- Exporting to macOS now creates a `.dmg` disk image if exporting from an editor running on macOS.
  - Signing the macOS export now is possible if running macOS (requires a valid code signing certificate).
- Exporting to Windows now changes the exported project's icon using `rcedit` (requires WINE if exporting from Linux or macOS).
- Improved build system.
  - Support for compiling using Visual Studio 2017.
  - [SCons](http://scons.org/) 3.0 and Python 3 are now supported (SCons 2.5 and Python 2.7 still work).
  - Link-time optimization can now be enabled by passing `use_lto=yes` to the SCons command line.
    - Produces faster and sometimes smaller binaries.
    - Currently only supported with GCC and MSVC.
  - Added a progress percentage when compiling Godot.
  - `.zip` archives are automatically created when compiling HTML5 export templates.
- Easier and more powerful way to create editor plugins with EditorPlugin and related APIs.

### Changed

- Increased the default low-processor-usage mode FPS limit (60 → 125).
  - This makes the editor smoother and more responsive.
- Increased the default 3D editor camera's field of view (55 → 70).
- Increased the default 3D Camera node's field of view (65 → 70).
- Changed the default editor font (Droid Sans → [Noto Sans](https://www.google.com/get/noto/)).
- Changed the default script editor font (Source Code Pro → [Hack](http://sourcefoundry.org/hack/))
- Renamed `engine.cfg` to `project.godot`.
  - This allows users to open a project by double-clicking the file if Godot is associated to `.godot` files.
- Some methods from the `OS` singleton were moved to the new `Engine` singleton.
- Switched from [GLEW](http://glew.sourceforge.net/) to [GLAD](http://glad.dav1d.de/) for OpenGL wrapping.
- Changed the SCons build flag for simple logs (`colored=yes` → `verbose=no`).
- The HTML5 platform now uses WebGL 2.0 (instead of 1.0).
- Redesigned the Godot logo to be more legible at small sizes.

### Deprecated

- `opacity` and `self_opacity` are replaced by `modulate` and `self_modulate` in all 2D nodes, allowing for full color changes in addition to opacity changes.

### Removed

- Skybox support.
  - Replaced with panorama skies, which are easier to import.
- Opus audio codec support.
  - This is due to the way the new audio engine is designed.
- HTML5 export using asm.js.
  - Only WebAssembly is supported now, since all browsers supporting WebGL 2.0 also support WebAssembly.

[3.0]: https://github.com/godotengine/godot/compare/2.1-stable...3.0-stable
