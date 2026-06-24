# Godot ↔ UWP In-Process Embedding — Integration Guide

> **Purpose:** the complete, reproducible recipe for this integration. Use it to
> (re)build the setup on a **new Godot engine version**, a **new Godot project**,
> or a **new UWP application** — each part is independent and lists every change
> with the reasoning, so the same design can be applied from scratch.
>
> **Canonical implementations to copy from:**
> - Engine: `UWP_Export\godot` (Godot 4.6.4 + patches)
> - UWP app: `UWP_Export\GodotUWPSample\GodotUWPSample` (namespace `Godot.Uwp.Embedding`)
> - Bundled Godot project: `…\GodotUWPSample\GodotProject` (spinning-cube smoke test)

---

## Architecture (what you are building)

```
┌────────────────────────── UWP app (AppContainer) ──────────────────────────┐
│  XAML SwapChainPanel ◄─────────────── ISwapChainPanelNative::SetSwapChain  │
│        │ pointer/size/DPI events            ▲ (UI-thread hop via dispatcher)│
│        ▼                                    │                               │
│  C# host layer                         godot.dll (in-process)              │
│  ├── GodotEngineHost ─ work queue ───► ├── godot_uwp_* flat C ABI          │
│  ├── input injection ────────────────► ├── DisplayServerEmbeddedWin        │
│  └── JSON message bus ◄──────────────► │     (no HWND, driver "embedded")  │
│      (UWPHost singleton)               ├── D3D12 CreateSwapChainForComposition
│                                        └── Main::setup/start/iteration      │
│                                            on ONE dedicated engine thread   │
└─────────────────────────────────────────────────────────────────────────────┘
```

Principles that make it work inside the AppContainer sandbox:
1. **No HWND anywhere** — rendering goes to a composition swap chain bound to the
   panel; input is injected from XAML events. (HWND reparenting is impossible in UWP.)
2. **One dedicated engine thread** owns the entire engine lifecycle; the UI thread
   only enqueues work. The single exception is `SetSwapChain`, which is UI-thread
   affine and reached via a host-provided synchronous dispatcher.
3. **No exception may cross a native→managed callback** (.NET Native fail-fasts).
4. The host adapts the sandbox around the engine (writable `user://`, CWD for the
   Agility SDK) instead of patching engine internals.

---

# Part 1 — Godot engine changes

Apply to a clean engine checkout (4.6+; the same patches should apply to 4.7 with
minor adjustment). Everything is in `platform/windows` plus two D3D12 driver files
— **no core engine changes**.

## 1.1 New file: `platform/windows/display_server_embedded_win.{h,cpp}`

Windowless display server, registered as display driver **`embedded`**.

- **Subclass `DisplayServerHeadless`** (header-only, all-stubs) — you only override
  what an embedded panel needs (~450 lines instead of ~1300).
- Constructor (mirrors `platform/macos/display_server_embedded.mm`, upstream's
  LibGodot reference):
  1. take initial size/scale from the host stash (`GodotUwpEmbedState`),
  2. `memnew(RenderingContextDriverD3D12)` + `initialize()`,
  3. `window_create(MAIN_WINDOW_ID, &wpd)` with `wpd.window = nullptr` and
     `wpd.swap_chain_panel_native = <host panel IUnknown*>`,
  4. `window_set_size` / `window_set_vsync_mode`, seed the surface composition scale,
  5. `RenderingDevice::initialize` + `screen_create` + `RendererCompositorRD::make_current()`.
- Re-hook `Input::set_event_dispatch_function` to your own static (the headless
  base's dispatch is private to it) and route events to per-window callbacks —
  that is what delivers input to Godot `Window` nodes.
- Host-facing methods (called on the engine thread by the C ABI):
  `host_resize`, `host_set_composition_scale`, `host_inject_mouse_button/motion/wheel`,
  `host_inject_key` (Windows VK → Godot Key via `KeyMappingWindows::get_keysym`).
- Overridden queries report the panel as one screen: `screen_get_size` =
  `window_get_size` = panel size; `screen_get_dpi` = `96 × composition_scale`;
  `window_get_native_handle` returns 0; clipboard is a process-local string.
- `process_events()` = `Input::flush_buffered_events()`.
- Registration: `register_create_function("embedded", create_func, get_rendering_drivers_func)`
  with drivers = `{"d3d12"}` only (Vulkan needs an HWND surface — excluded by design).

## 1.2 New file: `platform/windows/godot_uwp_embed.{h,cpp}` — the flat C ABI

The host P/Invokes only these (all `__cdecl`, `__declspec(dllexport)`):

| Export | Purpose |
|---|---|
| `godot_uwp_set_log_callback(cb)` | every print/warning/error line → host (install first) |
| `godot_uwp_set_swap_chain_panel(IUnknown*)` | panel pointer, AddRef'd; **before setup** |
| `godot_uwp_set_ui_dispatcher(fn)` | synchronous run-on-UI-thread hop for `SetSwapChain` |
| `godot_uwp_engine_setup(argc, argv)` | `new OS_Windows` + host logger + `Main::setup(...)` |
| `godot_uwp_engine_start()` | `Main::setup2()` → **register UWPHost singleton** → `Main::start()` → `main_loop->initialize()` |
| `godot_uwp_engine_iteration()` | `process_events()` + `Main::iteration()`; returns 1 = quit |
| `godot_uwp_engine_shutdown()` | finalize main loop, remove singleton, `Main::cleanup()` |
| `godot_uwp_notify_panel_resize(w, h)` | pre-setup: stashed; post: `host_resize` |
| `godot_uwp_set_composition_scale(sx, sy)` | same pattern as resize |
| `godot_uwp_inject_mouse_button/motion/wheel`, `godot_uwp_inject_key` | input |
| `godot_uwp_set_host_message_callback(cb)` | engine→host bus channel |
| `godot_uwp_set_call_return(json)` | host's synchronous reply (valid only inside the callback) |
| `godot_uwp_call_engine(method, argsJson, &ret)` | host→engine bus (engine thread only!) |
| `godot_uwp_free_string(str)` | frees `ret` (engine-side `memalloc`/`memfree` pair) |

Key implementation notes:
- Mirrors upstream `libgodot_windows.cpp`: the ABI itself creates `OS_Windows` and
  drives `Main::*` directly — install the host logger (a `Logger` subclass
  forwarding `logv` to the callback) **before** `Main::setup` so setup failures are visible.
- **Threading contract:** every function is engine-thread affine, except the
  config setters used before setup and the UI dispatcher (which the engine calls
  FROM the engine thread, expecting the work to run ON the UI thread synchronously).
- Pre-setup values (panel/size/scale) go in a `GodotUwpEmbedState` namespace; the
  display server consumes them in its constructor.

## 1.3 New file: `platform/windows/uwp_host.{h,cpp}` — GDScript message-bus singleton

`class UWPHost : public Object` with `_bind_methods` exposing:

- `send_to_host(method: String, args) -> Variant` — JSON-encode args (always an
  array; `"[]"` when empty), invoke the host callback synchronously, return the
  optional reply set via `godot_uwp_set_call_return` (parsed back to a Variant).
- `register_handler(method, callable)` / `unregister_handler(method)` — handlers
  the host invokes through `godot_uwp_call_engine` (JSON array → argument list).
- `has_host() -> bool` — lets project scripts detect the embedding.

**Registration timing is load-bearing** (in `godot_uwp_engine_start`):
after `Main::setup2()` (ClassDB alive), **before `Main::start()`** (autoload
`_ready` must already see it):

```cpp
GDREGISTER_CLASS(UWPHost);
uwp_host_singleton = memnew(UWPHost);
Engine::get_singleton()->add_singleton(Engine::Singleton("UWPHost", uwp_host_singleton));
```

## 1.4 Modified: `drivers/d3d12/rendering_context_driver_d3d12.{h,cpp}`

- `WindowPlatformData`: add `IUnknown *swap_chain_panel_native;`
  ⚠ **must stay trivially constructible** — it is used inside anonymous unions in
  `display_server_windows.cpp`; a default member initializer breaks the build (C2280).
- `Surface`: add `ComPtr<IUnknown> swap_chain_panel_native;` +
  `float composition_scale_x/y = 1.0f;`
- New method `surface_set_composition_scale(SurfaceID, sx, sy)` → stores scale,
  sets `needs_resize` (forces the swap-chain transform to be reapplied).
- Static `EmbedUiDispatchFunc embed_ui_dispatch;` — the host's UI-thread hop,
  set by the ABI; used by the device driver for `SetSwapChain`.
- `surface_create` copies the panel pointer from the platform data.

## 1.5 Modified: `drivers/d3d12/rendering_device_driver_d3d12.cpp` (`swap_chain_create`)

- `bool panel_mode = surface->swap_chain_panel_native != nullptr;`
- panel_mode forces **`CreateSwapChainForComposition`** (independent of
  transparency/DCOMP), skips `MakeWindowAssociation` and the DComp-HWND block.
- Use `DXGI_ALPHA_MODE_IGNORE` for the panel (not PREMULTIPLIED) unless the project
  requests transparency — premultiplied alpha makes the compositor blend the scene
  with the page background and washes it out.
- After creation: QI the panel for `ISwapChainPanelNative` and call
  `SetSwapChain(swapchain)` **via `embed_ui_dispatch`** (UI-thread affine).
  Declare the interface locally with **both IIDs** — the same engine then serves
  UWP *and* WinUI3 XAML hosts:
  - UWP: `f92f19d2-3ade-45a6-a20c-f6f1ea90554b`
  - WinUI3: `63aad0b8-7c24-40ff-85a8-640d944cc325`
- **DPI (critical, symptoms = only the top-left of the scene visible):** XAML
  composes panel buffers in DIPs. After creation AND after every `ResizeBuffers`,
  apply the inverse scale:
  ```cpp
  DXGI_MATRIX_3X2_F m = {}; m._11 = 1/scale_x; m._22 = 1/scale_y;
  swap_chain->SetMatrixTransform(&m);   // IDXGISwapChain2+
  ```

## 1.6 Modified: glue

- `platform/windows/SCsub`: when `library_type == "shared_library"` →
  `env.Append(CPPDEFINES=["GODOT_UWP_EMBED_ENABLED"])` and compile the three new cpp
  files (`display_server_embedded_win.cpp`, `godot_uwp_embed.cpp`, `uwp_host.cpp`).
- `platform/windows/os_windows.cpp`: after `DisplayServerWindows::register_windows_driver()`
  add `DisplayServerEmbeddedWin::register_embedded_driver();` (guarded by the define).
- `platform/windows/display_server_windows.cpp`: set the new
  `wpd.d3d12.swap_chain_panel_native = nullptr;` where the union is filled.

## 1.7 Building the engine

```powershell
# Python + SCons (beware: the Windows Store python alias can shadow real installs)
python misc\scripts\install_d3d12_sdk_windows.py     # once: Agility SDK, NIR, PIX

scons platform=windows target=template_release arch=x86_64 `
      library_type=shared_library debug_symbols=yes disable_path_overrides=no
```

- **`disable_path_overrides=no` is required** — release templates otherwise reject
  the `--path` argument (`.pck` via `--main-pack` and project folders both need it).
- Output: `bin\godot.windows.template_release.x86_64.dll` (+ PDB for native debugging).
- Verify exports: `dumpbin /exports godot...dll | findstr godot_uwp`.

---

# Part 2 — Godot project (optional: talking to the host)

A project needs **no changes** just to render in the panel — drop in any project (or
`.pck`) and it runs. Part 2 only applies if the project wants to exchange data with
the host over the message bus. It stays **fully cross-platform** — the host path
activates only when the engine exposes the `UWPHost` singleton.

## 2.1 A minimal host bridge (autoload)

```gdscript
extends Node

var _host: Object = null

func _ready() -> void:
    if Engine.has_singleton("UWPHost"):           # absent in the editor → no-op
        _host = Engine.get_singleton("UWPHost")
        _host.register_handler("ping", _on_ping)  # host can call back into GDScript
        _host.send_to_host("scene_ready", [])      # fire a message at the host

# Host -> engine: invoked via godot_uwp_call_engine("ping", args_json).
func _on_ping(args) -> String:
    return "pong"

# Engine -> host: blocks until the host replies (synchronous), or fire-and-forget.
func ask_host(method: String, args: Array) -> Variant:
    return _host.send_to_host(method, args) if _host else null
```

Rules learned the hard way:
- Always guard with `Engine.has_singleton("UWPHost")` so the same project runs
  unmodified in the editor.
- If you wire the bridge from a factory that `new()`s a node **without adding it to
  the tree**, hook up in `_init` (not `_ready` — it never fires off-tree).
- A handler registered with `register_handler` whose reply the host reads with
  `godot_uwp_call_engine` runs on the engine thread synchronously — keep it cheap.

## 2.2 Bundling the project for the UWP app — prefer a `.pck`

| Why `.pck` | Loose project folder problems (all hit in practice) |
|---|---|
| One opaque file | The **MRT resource indexer** parses package sub-paths; a folder whose name collides with a resource qualifier (e.g. a 2-letter language code) triggers `0x80073B0C` at XAML resource lookup |
| No path limits | `.godot\shader_cache\<64-char-hash>\…` exceeds **MAX_PATH** during AppX layout copy (DEP1000) |
| 1 file deploy | thousands of file copies per deploy |

- Export with the **Windows Desktop** preset → *Export PCK/ZIP only* → `Assets\project.pck`.
- **Engine version must match the editor line** that exported the pck.
- Keep desktop texture formats enabled (S3TC/BPTC) — the D3D12 renderer does not
  accept ETC2-only imports.
- **GDExtension DLLs cannot load from inside a pck.** Ship each extension's release
  DLL (+ its native dependencies) loose in the package root — Godot's loader falls
  back to "file name next to the executable".

---

# Part 3 — UWP application

Old-style UWP (C# / .NET Native), namespace `Godot.Uwp.Embedding`. For a new app,
copy the whole `Godot\` folder + the MainPage wiring + the csproj fragments.

## 3.1 Project configuration (`.csproj`)

- `TargetPlatformVersion` = an SDK actually installed; Platform **x64**.
- **`UseDotNetNativeToolchain=true` for BOTH configurations.** CoreCLR-hosted Debug
  UWP apps may not launch on current Windows builds (the runtime shim can misroute
  them to desktop .NET Framework). Debug/.NET Native (chk) launches **under F5** (VS
  enables package debug mode → no PLM activation timeout; bare launches of the slow
  chk build can get killed mid-startup).
- Content items (all `PreserveNewest`):
  - `godot.dll` (Link from the engine's `bin\`), `D3D12Core.dll`, `d3d12SDKLayers.dll`
  - the project `.pck` (preferred) — or the `GodotProject\` folder with
    `.godot\shader_cache`, `.godot\editor`, `.godot\exported`, `build\` **excluded**
    (`.godot\imported` must stay for folder mode)
  - any GDExtension DLLs at the package root (`<Link>` to strip the folder)
- No extra NuGet packages needed: the bus uses **`Windows.Data.Json`** (built-in).

## 3.2 C# host layer (`Godot\` folder)

| File | Role / key invariants |
|---|---|
| `GodotNative.cs` | P/Invoke surface, 1:1 with the C ABI. Manual UTF-8 marshalling (`Marshal.PtrToStringUTF8` doesn't exist on .NET Native). All callback delegates pinned in static fields. |
| `GodotEngineHost.cs` | THE threading core. Dedicated engine thread runs setup→start→iteration loop→shutdown; `ConcurrentQueue<Action>` for host→engine work (`Post`/`Invoke`/`InvokeAsync`); frame pacing; **Pause/Resume** for the app lifecycle; log file in `LocalState\Logs`; `CallEngineRaw` + `SetHostMessageHandler` for the bus. |
| `EngineMessageReceiver.cs` | Engine→host. Raises `OnMessage(method, argsJson)` on the captured UI context; plus `RegisterSyncHandler(method, fn)` for handlers that answer **inline on the engine thread** (rules: fast, no XAML, never block on the UI thread — deadlock). |
| `EngineMessageSender.cs` | Host→engine. `Post(method, argsJson)` fire-and-forget; `Call`/`CallAsync(method, argsJson)` for a return value. |

AppContainer adaptations inside `GodotEngineHost.EngineThreadProc` (before setup):

```csharp
// user:// must be writable → Godot resolves it via APPDATA
Environment.SetEnvironmentVariable("APPDATA",      ApplicationData.Current.LocalFolder.Path);
Environment.SetEnvironmentVariable("LOCALAPPDATA", ApplicationData.Current.LocalFolder.Path);
// Agility SDK probes ".\x86_64" then ".\" relative to the CWD (System32 under UWP!)
Directory.SetCurrentDirectory(Package.Current.InstalledLocation.Path);
```

Engine args: `--display-driver embedded --rendering-driver d3d12
--rendering-method forward_plus --path <pkg>\GodotProject` (or `--main-pack <pkg>\Assets\project.pck`),
plus `--verbose` while developing. Force the rendering method so a project set to
`gl_compatibility` still uses the D3D12 RenderingDevice path.

**Iron rule:** no exception may escape `OnNativeLog`, `OnNativeUiDispatch`, or
`OnNativeHostMessage` — a managed exception crossing the reverse-P/Invoke boundary
fail-fasts .NET Native (`SharedLibrary.dll`, code 0x1007). Wrap everything.

## 3.3 MainPage wiring

- XAML: a bare `SwapChainPanel` — **UWP's panel does not allow `Background`**
  (XamlParseException; WinUI3's does — porting trap).
- `Loaded`: compute `ActualWidth/Height × CompositionScaleX/Y` (physical px),
  `Marshal.GetIUnknownForObject(panel)` (the engine QIs the right interface
  itself), register bus handlers **before** `host.Start(...)`.
- `SizeChanged` + `CompositionScaleChanged` → `host.ConfigurePanel(...)` — this is
  the live-resolution path: backbuffer `ResizeBuffers` + Godot window resize +
  DPI transform reapplication all follow automatically.
- Pointer events → `InjectMouse*` (positions × composition scale);
  `CoreWindow.KeyDown/KeyUp` → `InjectKey` (VirtualKey is the Win32 VK code).
- **Lifecycle:** `Application.Suspending → host.Pause()`, `Resuming → host.Resume()`
  — a suspended UWP app that keeps presenting is terminated by PLM (silently, no
  crash event).

## 3.4 Deploy / debug workflow

```powershell
# the registered package LOCKS the layout → always unregister before rebuilding
Get-AppxPackage -Name "<identity>" | Remove-AppxPackage
msbuild GodotUWPSample\GodotUWPSample\GodotUWPSample.csproj /restore /p:Configuration=Release /p:Platform=x64
# the deployable layout is the ilc\ subfolder (.NET Native output), NOT the bin root
Add-AppxPackage -Register "GodotUWPSample\GodotUWPSample\bin\x64\Release\ilc\AppxManifest.xml"
explorer.exe "shell:AppsFolder\<PFN>!App"
```

- Logs: `%LOCALAPPDATA%\Packages\<PFN>\LocalState\` → `boot.log` (App ctor +
  unhandled-exception trap) and `Logs\godot_*.log` (engine + `[Bus]` traffic).
- Debug-config runs only via **F5**; debug the engine C++ by attaching the VS
  **native** debugger to the running Release app (`godot.dll` has full PDBs).
- Crash before any log? Register cdb as the PLM debugger via
  `IPackageDebugSettings::EnableDebugging` (CLSID `B1AEC16F-2383-4852-B0E9-8F0B1DC66B4D`),
  with cdb copied OUT of the WindowsApps folder, `-logo <file>`.
- Debug/.NET Native needs the `Microsoft.NET.Native.Framework.Debug.2.2` appx once
  (in the `runtime.win10-x64.microsoft.net.native.sharedlibrary` NuGet, `tools\SharedLibrary\chk\Native\`).

## 3.5 Message-bus shape

The transport is just `(method, argsJson)` in both directions — the *vocabulary* is
yours to define per app.

| Direction | API | Notes |
|---|---|---|
| engine→host (async) | GDScript `UWPHost.send_to_host(method, args)`; host receives via `EngineMessageReceiver.OnMessage` | host returns nothing synchronously; reply later via the sender if desired |
| engine→host (sync) | same `send_to_host`, but host has a `RegisterSyncHandler(method, …)` | GDScript **blocks** until the host returns a JSON document inline |
| host→engine | `EngineMessageSender.Post/Call/CallAsync(method, argsJson)` → a handler registered with `UWPHost.register_handler` | `Call` blocks for the return value |

The sample's `MainPage` simply logs every incoming message (`[Bus]` lines). A real
app classifies `method`/args and responds.

---

# Quick checklist for a brand-new integration

1. ☐ Engine: copy the 3 new file pairs + the D3D12/glue edits (Part 1) into the new
   engine tree; build with `library_type=shared_library disable_path_overrides=no`;
   verify the `godot_uwp_*` exports.
2. ☐ Project: drop in any project (or export a version-matched `.pck`); add a host
   bridge (Part 2) only if you need host↔engine messaging; ship GDExtension DLLs loose.
3. ☐ UWP app: copy the `Godot\` folder + MainPage wiring + csproj fragments (Part 3);
   point the Content items at the new engine DLL and the project/pck.
4. ☐ First boot: watch `LocalState\Logs\godot_*.log` — expect `display server:
   embedded`, `D3D12 … Using Device`, `Engine running`.
5. ☐ Visual checks: full scene visible (DPI transform correct), live window resize,
   click-through input.
