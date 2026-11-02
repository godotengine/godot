# WebGPU rendering driver

This opt-in backend ports `dwalter/godotwebgpu` at
`f329e39ce8db7acaa5c9d6628a530fb769969228` to the 2dog Godot 4.7.2 fork.
It supports browser WebGPU and optional native Dawn on Windows, Linux and macOS.
Existing rendering defaults and other backends remain available.

This experimental backend supports the Forward+ and Mobile renderers. Forward+
binds a compact scene on WebGPU (4 lightmaps, 2 VoxelGIs, lightmap data in a
uniform buffer) so it fits Safari's 44 sampled textures and Chrome on Metal's 10
storage buffers per stage. It falls back to Mobile with one warning on adapters
with fewer than 40 sampled textures or 10 storage buffers per stage. Without
`texture-formats-tier2` read/write storage (Firefox, lavapipe), Forward+ runs
with SDFGI and VoxelGI disabled and the raster depth of field, warning once per
feature a scene asks for; SSIL is disabled only on adapters with tier1 but not
tier2. Android, double-precision and multithreaded browser runtimes
reject WebGPU explicitly.
Headless resource imports and exports remain supported for WebGPU projects.
Read/write storage textures require adapter-native format support; unsupported
formats and overlapping writable storage-buffer bindings fail explicitly.

## Dependencies and builds

SCons prepares [checksum-pinned upstream sources](dependencies.md) when
`webgpu=yes`. Local Tint, SPIRV-Tools and Emscripten compatibility changes are
kept as patches. Compiler source selection is shared with the host Tint CLI.

For the .NET browser host, keep Emscripten **3.1.56** and build from the outer
2dog checkout with the Godot worktree substituted below:

```text
uv run scons -C <godot-checkout> platform=web target=template_debug library_type=static_library module_mono_enabled=yes disable_crash_handler=yes disable_path_overrides=no threads=no lto=none webgpu=yes
```

`webgpu_port=auto` uses the prepared Emdawnwebgpu port. Its legacy JavaScript
uses the Emscripten 3.x stack dependency names; newer SDKs use the original
JavaScript. Tint also retains a locale-independent float parser for older libc++.
An explicit local `.py` port or `webgpu_port=emdawnwebgpu` may be supplied for
other toolchains; the remote port requires Emscripten 4.0.10 or later.

On Windows, keep the Emscripten cache on the SDK's drive to avoid the 3.1.56
system-library builder's cross-drive path limitation. Use separate native and
.NET link caches and a writable `WasmCachePath` for the browser host.

For desktop builds, use `webgpu=yes webgpu_native_path=<boundary-sdk>`.
[Native setup](native/README.md) describes the separately pinned Dawn SDK,
shared C API boundary, sidecar libraries and platform commands. The boundary
isolates Dawn's compiler symbols from Godot's patched Tint.

## Runtime selection

Select `--rendering-driver webgpu`, which defaults to Forward+, and optionally
`--rendering-method mobile`. Use the OpenGL driver when selecting `gl_compatibility`.

A custom browser shell must set `renderingDriver: 'webgpu'` before Engine
initialization. The final emcc link must use the prepared Dawn port and its
Closure externs with Emscripten 3.1.56.

The driver keeps shader transformations and renderer workarounds behind WebGPU
capabilities. Browser readback is asynchronous; native readback waits for GPU
completion. Native HWND, Xlib/Wayland and normal/embedded CAMetalLayer surfaces
use the same driver-selection mechanism as the other backends.

## Portable shaders and baking

WebGPU shader containers carry compressed WGSL, typed Tint inspector metadata,
and SPIR-V for unsupported specialization expressions. Supported scalar
specialization constants remain WGSL overrides, so changing them only creates a
new pipeline. Composite/select specialization expressions retain the SPIR-V
retranslation fallback. Loading a current container avoids SPIR-V-to-WGSL
conversion; adapter-specific format and storage-access rewrites
still run on the selected device. Older SPIR-V-only containers remain readable.

Desktop editors built with `webgpu=yes` include the WebGPU shader baker. To add
only the baker to an editor using another renderer, build with
`webgpu=no webgpu_shader_baker=yes`; no native Dawn SDK is needed. Enable
`shader_baker/enabled` in the web export preset and select WebGPU in the project's
web rendering-driver setting. Shader baking requires an editor running a
RenderingDevice renderer, such as Vulkan, Metal or WebGPU. Run the editor with
the web export's `rendering_method.web` to bake the renderer used by WebGPU exports. Runtime-generated
shaders and variants absent from the export continue to compile normally.
The WebGPU baker skips multiview and input-attachment variants enabled by the host editor.

The varying-location compaction and adapter-aware lightmap budget are adapted
from [V-Sekai-fire/entities-godot at 3a5f1a4](https://github.com/V-Sekai-fire/entities-godot/tree/3a5f1a4baa38c246c5e7c7f3ef79ebe306c96281).
That branch also informed the WGSL container and structured metadata design.
Its PointCoord emulation, loop-exit repair and depth-sampling lowering also inform
the local Tint patches. Point coordinates read `(0.5, 0.5)` because WebGPU points
cover one pixel. Tint internal compiler errors return shader diagnostics through
a compiler-only exception boundary; all compiler frames support unwinding.
Non-comparison 2D depth sampling uses texel loads. A depth-specific framebuffer
copy populates the float backbuffer used by `hint_depth_texture`. The lowering
emulates linear-clamp sampling and recognizes Godot's nearest samplers;
implicit LOD and gradients use mip zero, matching Godot's single-mip scene
depth. Arbitrary RD depth samplers with other addressing modes need explicit
texel loads.

Our existing compiler pipeline, native support and push-constant paging remain
in use. Compaction preserves vertex-input and color-output locations and keeps
matrix and array varyings aligned across stages.

## Verification

[Port checks](tests/README.md) cover real compiler transformations,
resource reflection, SDK configuration, JavaScript engine startup and native
render/readback. Optional upstream diagnostics and demos are fetched separately.

The port retains the four-argument `RenderingDevice.draw_list_draw` binding;
the first-instance overload is internal, preserving existing GodotSharp method
hashes. WebGPU has no ray tracing, and unsupported multiview paths are skipped.
Physical-device macOS and Linux validation depends on available hardware.
