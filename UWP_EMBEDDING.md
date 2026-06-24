# Godot → UWP in-process embedding

This fork adds the ability to run the Godot engine **in-process inside a UWP
(AppContainer) app** and render into a XAML `SwapChainPanel` — no child window,
no separate process, no frame streaming. The engine's D3D12 swap chain is created
for composition and bound directly to the panel; all input is injected from the
host, and a small JSON message bus lets host and GDScript talk.

> Targets Windows / **D3D12** only (the embedded display server is
> RenderingDevice-based; Vulkan needs an HWND surface and is not used here).
> Built and developed against this branch (Godot 4.x). All additions are gated
> behind `GODOT_UWP_EMBED_ENABLED`, which is defined only for
> `library_type=shared_library` builds — normal editor/template builds are
> completely unaffected.

A ready-to-run sample UWP app lives in
[`platform/windows/uwp_sample/`](platform/windows/uwp_sample/).

---

## What was added

**New files (all under `platform/windows/`):**

| File | Purpose |
|---|---|
| `display_server_embedded_win.{h,cpp}` | A windowless `DisplayServer` registered as the **`embedded`** driver. Subclasses `DisplayServerHeadless`; creates the D3D12 rendering context/device with **no HWND**; reports the host panel as a single screen; turns host-injected events into Godot `InputEvent`s. |
| `godot_uwp_embed.{h,cpp}` | The flat `extern "C"` ABI the C# host P/Invokes — engine lifecycle, panel/size/DPI, input injection, and the message bus. Drives `Main::setup/setup2/start/iteration` directly (LibGodot pattern). |
| `uwp_host.{h,cpp}` | The GDScript-visible **`UWPHost`** singleton (`send_to_host`, `register_handler`, `has_host`) backing the bus. |

**Modified engine files:**

- `platform/windows/SCsub` — compiles the three new files and defines
  `GODOT_UWP_EMBED_ENABLED` when `library_type=shared_library`.
- `platform/windows/os_windows.cpp` — registers the `embedded` display driver.
- `platform/windows/display_server_windows.cpp` — initializes the new
  `swap_chain_panel_native` field on the D3D12 window platform data.
- `drivers/d3d12/rendering_context_driver_d3d12.{h,cpp}` — the surface carries an
  optional `ISwapChainPanelNative` and a composition scale; adds
  `surface_set_composition_scale()` and the host UI-thread dispatcher hook.
- `drivers/d3d12/rendering_device_driver_d3d12.cpp` — when a panel is set,
  `swap_chain_create` uses `CreateSwapChainForComposition`, binds the swap chain
  to the panel via `ISwapChainPanelNative::SetSwapChain` (UWP IID
  `f92f19d2-…` **and** WinUI3 IID `63aad0b8-…` both supported), and applies
  `IDXGISwapChain2::SetMatrixTransform(1/dpiScale)` so the DIP-composited buffer
  maps 1:1 to physical pixels.

---

## Building the engine DLL

```powershell
# Requires Python 3.x + SCons + MSVC (Visual Studio). If the Windows Store
# "python" alias shadows your real install, prepend the real Python dir to PATH.

# One-time: download the D3D12 build deps (Agility SDK, Mesa NIR, PIX).
python misc\scripts\install_d3d12_sdk_windows.py

# Build the engine as a shared library (this is what enables the embedding).
scons platform=windows target=template_release arch=x86_64 ^
      library_type=shared_library debug_symbols=yes disable_path_overrides=no
```

- **`library_type=shared_library`** is required — it produces `godot.dll` with the
  exported C ABI and enables `GODOT_UWP_EMBED_ENABLED`.
- **`disable_path_overrides=no`** is required — template builds otherwise reject
  the `--path` / `--main-pack` argument the host uses to load a project.
- Output: `bin/godot.windows.template_release.x86_64.dll` (+ `D3D12Core.dll`,
  `d3d12SDKLayers.dll`, and a `.pdb` for native debugging).
- Verify the ABI: `dumpbin /exports bin\godot.windows.template_release.x86_64.dll | findstr godot_uwp`.

For a debuggable engine, build `target=template_debug` instead.

---

## Using it from a UWP app (in brief)

1. Build `godot.dll` (above) and ship it, plus `D3D12Core.dll` /
   `d3d12SDKLayers.dll`, in your UWP package.
2. From the UWP app, on a **dedicated thread**, P/Invoke the `godot_uwp_*` ABI:
   set the log callback, the panel (`godot_uwp_set_swap_chain_panel`, passing the
   `SwapChainPanel`'s `IUnknown`), the UI dispatcher, the initial size/scale, then
   `godot_uwp_engine_setup(["--display-driver","embedded","--rendering-driver","d3d12","--path",<project>])`,
   `godot_uwp_engine_start()`, and pump `godot_uwp_engine_iteration()` each frame.
3. Forward XAML pointer/keyboard/size/DPI events through the
   `godot_uwp_inject_*` / `godot_uwp_notify_panel_resize` / `…set_composition_scale`
   entry points.
4. (Optional) Use the message bus: install
   `godot_uwp_set_host_message_callback`, reply with `godot_uwp_set_call_return`,
   and call into GDScript with `godot_uwp_call_engine`. On the GDScript side use
   the `UWPHost` singleton.

The sample in [`platform/windows/uwp_sample/`](platform/windows/uwp_sample/) is a
complete, working implementation of all of the above (a C# host layer + a cube
test project). See its `README.md` and the bundled
`EMBEDDING_ARCHITECTURE.md` / `INTEGRATION_GUIDE.md` for the full host-side recipe,
the threading model, the AppContainer adaptations, and the deploy/debug workflow.

---

## The C ABI

```c
// lifecycle
int  godot_uwp_engine_setup(int argc, char **argv);
int  godot_uwp_engine_start(void);
int  godot_uwp_engine_iteration(void);   // returns 1 when the engine wants to quit
void godot_uwp_engine_shutdown(void);
// configuration (before setup)
void godot_uwp_set_log_callback(GodotUwpLogCallback);
void godot_uwp_set_swap_chain_panel(void *panel_iunknown);
void godot_uwp_set_ui_dispatcher(GodotUwpUiDispatchFunc);
void godot_uwp_notify_panel_resize(int width_px, int height_px);
void godot_uwp_set_composition_scale(float scale_x, float scale_y);
// input injection
void godot_uwp_inject_mouse_button(int button, int pressed, float x, float y, int double_click);
void godot_uwp_inject_mouse_motion(float x, float y, float rel_x, float rel_y);
void godot_uwp_inject_mouse_wheel(float x, float y, float delta_x, float delta_y);
void godot_uwp_inject_key(unsigned int win_vk, int pressed, int echo, unsigned int unicode);
// host <-> engine JSON message bus
void godot_uwp_set_host_message_callback(GodotUwpHostMsgCallback);
void godot_uwp_set_call_return(const char *json);
int  godot_uwp_call_engine(const char *method, const char *args_json, char **ret_json);
void godot_uwp_free_string(char *str);
```

---

## Notes & limitations

- **D3D12 only.** The embedded display server registers only the `d3d12`
  rendering driver.
- **Audio:** WASAPI init fails in the AppContainer and falls back to the dummy
  (silent) driver. AppContainer-friendly audio activation is future work.
- **Store submission** would need a `WINAPI_FAMILY_APP` compliance pass over the
  Windows platform / drivers; sideloaded deployment does not.
