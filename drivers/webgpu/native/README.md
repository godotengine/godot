# Native Dawn dependency

Desktop WebGPU is opt-in on Windows, Linux (X11 and Wayland), and macOS
(including the embedded Metal-layer display server). Existing rendering
defaults and Vulkan, D3D12, Metal, and OpenGL selection are unchanged.

The native API was built against Dawn release **v20261014.133826**, commit
`f9ac6c61ff0a88ddec67db7f2d0455b4125ab69d`:
<https://github.com/google/dawn/releases/tag/v20261014.133826>.
Use the release archive for the target OS and architecture, or build the same
Dawn revision with its native backends and monolithic `webgpu_dawn` target.
Newer headers may need API adjustments; the browser port remains separately
pinned and does not use this SDK.

## Build the boundary SDK

Dawn's static release includes its own Tint and SPIRV-Tools. Godot uses a
separately patched Tint, so the two implementations must live in different
symbol namespaces. This small shared library exports the WebGPU C API and two
native initialization helpers while hiding all Dawn C++ implementation symbols.
Do not link the Dawn static library directly into Godot.

With CMake, a C++20 compiler, and the unpacked Dawn release:

```text
uv run --with cmake cmake -S drivers/webgpu/native -B dawn-boundary-build -DDawn_DIR=<Dawn-release>/lib/cmake/Dawn -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<boundary-sdk>
uv run --with cmake cmake --build dawn-boundary-build --config Release
uv run --with cmake cmake --install dawn-boundary-build --config Release
```

Linux release archives use `lib64/cmake/Dawn`. Select a generator and
architecture matching the Godot build. Visual Studio 2026 needs CMake 4.2 or
newer (for example `uv run --with cmake==4.4.4 cmake ... -G "Visual Studio 18 2026" -A x64`).
The resulting SDK contains headers, the runtime library, and a Windows import
library. Retain Dawn's license and third-party notices when distributing it.

For Windows D3D12, supply matching architecture `dxcompiler.dll` and `dxil.dll`
from the Windows SDK's redistributable compiler alongside the boundary DLL
in `<boundary-sdk>/bin`. SCons stages these when present. The initialization
helper searches beside the boundary DLL and in Windows System32 using absolute
paths, without changing the process-wide DLL search policy.

## Build and select WebGPU

Add `webgpu=yes webgpu_native_path=<boundary-sdk>` to a normal desktop SCons
invocation. For example, from the Godot checkout:

```text
uv run scons platform=windows target=template_debug library_type=shared_library module_mono_enabled=yes webgpu=yes webgpu_native_path=<boundary-sdk>
uv run scons platform=linuxbsd target=template_debug webgpu=yes webgpu_native_path=<boundary-sdk>
uv run scons platform=macos target=template_debug webgpu=yes webgpu_native_path=<boundary-sdk>
```

SCons copies the runtime into `bin/`. Keep it beside the executable or libgodot
when deploying. Native platforms load Dawn from that directory only when WebGPU
is selected. WebGPU requires macOS 13 or newer on macOS and the Visual C++
Redistributable on Windows. Existing renderers retain their original runtime
requirements. Build the boundary with
`-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0` on macOS. App bundles
and 2dog platform packages still need their normal staging/signing integration.

Select `--rendering-driver webgpu` (Forward+ unless `--rendering-method mobile` is
given), or choose `webgpu` in `rendering/rendering_device/driver`. Native backend
selection is controlled by `rendering/rendering_device/webgpu/native_backend`:
`auto` (default), `d3d12`, `vulkan`, `metal`, or `d3d11`. Availability depends on
the Dawn build and host. Windows `--gpu-luid` is passed to Dawn's D3D adapter
selection and retains 2dog's automatic fallback behavior.

Native fences wait for actual queue completion and native texture readback uses
Godot's synchronous staging-buffer path. The browser retains asynchronous
readback. WebGPU-specific texture formats, storage usages, and subpass limits
are selected at runtime so the other drivers in the same binary retain their
original behavior.

## Local verification

Windows x64 shared libgodot passes two engine lifetimes, lit-cube pixel readback,
and L8/LA8 texture round trips using existing GodotSharp bindings, with WebGPU,
Vulkan Mobile, Vulkan Forward+, and OpenGL Compatibility. Linux X11 and Wayland
are exercised with `drivers/webgpu/tests/native_smoke`; the WSL host uses llvmpipe Vulkan
software rendering. macOS integration is implemented but has not been built or
run locally. Native GPU interop, export packaging, release builds, and the wider
renderer feature set remain outside these smoke checks.

Run the dependency configuration checks with:

```text
uv run python -m unittest discover -s drivers/webgpu/tests -p test_native_config.py -v
```

To run the native scene, launch a template with path overrides enabled and pass
`--path drivers/webgpu/tests/native_smoke --rendering-driver webgpu --rendering-method mobile`.
For 2dog templates with path overrides disabled, place the three smoke project
files beside the executable instead.
