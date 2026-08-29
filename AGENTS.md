# VERTEX GAME ENGINE — Repository Memory

Persistent context for working on the Godot 4.8-dev source tree being transformed
into **Vertex Game Engine**. Branch: `vertex-engine`.

## Core principle
Modify the existing Godot source directly. Do NOT build a separate engine. Do NOT
blindly global-replace "Godot" in internal class names/namespaces/APIs. Preserve all
required Godot + third-party copyright/license notices.

## Build status (IMPORTANT limitation)
- Toolchain `scons` is NOT installed in this sandbox; platform deps (X11, GL, Vulkan
  headers, ALSA, Pulse) are missing. A full engine binary build cannot be verified here.
- New code must stay **structurally** buildable: correct includes, register_types,
  SCsub, config.py following existing module conventions.
- Verification done here = structural (grep, file presence, convention checks).
  A real build must be run by the user in a configured environment.

## Engine version / branding system
- Source of truth for the displayed name: `version.py` -> read by `methods.get_version_info()`
  -> generated into `core/version_generated.gen.h` by `core/core_builders.py:version_info_builder`.
- Macros: `GODOT_VERSION_SHORT_NAME`, `GODOT_VERSION_NAME`, `GODOT_VERSION_WEBSITE`.
  (These macro NAMES are internal identifiers — NOT renamed. Only their VALUES change.)
- `short_name` lowercased/capitalized = config dir name (`~/.config/...`). Changing it
  re-routes editor config to a Vertex-named folder. This is intended for branding but
  note it. See `core/os/os.cpp:292 get_godot_dir_name()` (Win capitalize override too).

## Splash system
- `main/splash.png` (export boot) + `main/splash_editor.png` (editor) -> embedded via
  `main/main_builders.py` into `main/splash.gen.h` / `splash_editor.gen.h`.
- BG colors: boot `Color(0.14,0.14,0.14)`, editor `Color(0.125,0.145,0.192)`.
- `main/main.cpp:~3940` picks editor vs export splash. `no_editor_splash` SCsub option.

## Module system (how new Vertex features integrate without replacing engine)
- Each `modules/<name>/` needs: `config.py` (can_build/configure/get_doc_classes),
  `SCsub`, `register_types.{h,cpp}` with `initialize_<name>_module` /
  `uninitialize_<name>_module(ModuleInitializationLevel)`. Auto-detected by SCons.
- Pattern reference: `modules/jsonrpc` (simple), `modules/gltf` (complex + editor).
- Class registration: `GDREGISTER_CLASS(T)` at MODULE_INITIALIZATION_LEVEL_SCENE.

## Key file inventory (where each feature area lives)
1. Branding: `version.py`, `core/core_builders.py`, `main/main_builders.py`,
   `main/splash.png`, `main/splash_editor.png`, `main/app_icon.png`, `editor/editor_node.cpp`,
   `editor/project_manager/project_manager.cpp`, `editor/themes/`, `editor/icons/`.
2. UI redesign: `editor/themes/` (theme_modern/classic, editor_theme*), `editor/editor_node.cpp`,
   `editor/gui/`, `editor/docks/`, `editor/inspector/`, `editor/file_system/`.
3. Mobile editor: `platform/android/editor/`, `editor/editor_node.cpp`, `editor/themes/editor_scale.cpp`,
   `core/os/` input, `platform/android/android_input_handler.cpp`. NEW module: `modules/vertex_mobile_editor`.
4. Android compat: `platform/android/java/app/config.gradle` (minSdk/targetSdk/compileSdk),
   `editor/export/android_sdk_manager.h` DEFAULT_MIN/TARGET, `platform/android/detect.py`.
   NOTE: current minSdk=24. Lowering to 21 blocked by AndroidX/jetifier/NDK; documented as limitation.
5. Low-end perf: `servers/rendering/`, `scene/main/viewport.cpp`, `core/config/project_settings.cpp`,
   NEW module `modules/vertex_perf` (adaptive quality, budgets, object pooling).
6. Performance profiles: NEW module `modules/vertex_perf` + project manager + project settings.
7. Project optimizer: NEW module `modules/vertex_optimizer` (profiler analysis + recommendations).
8. Splash: `main/main_builders.py`, `main/splash*.png`, `main/main.cpp`.
9. Export branding: `editor/export/`, project settings boot_splash, Android `splash_branding_image.webp`.
10. AI assistant: NEW module `modules/vertex_ai` (architecture; confirm-before-destructive).

## Conventions
- Code style: see `.clang-format`. Tabs, Allman braces, copyright header preserved.
- New modules keep the standard Godot copyright header (legal requirement).
- Commits: add `Co-authored-by: openhands <openhands@all-hands.dev>`.

## Progress log
- [done] vertex-engine branch created.
- [done] Branding: version.py (name/short_name), config dirs (vertex/Vertex),
  splash PNGs + app icon (Vertex V logo), editor titles/strings, export
  'Powered by Vertex' settings. Godot/third-party notices preserved.
- [done] Module: vertex_perf — VertexPerformanceProfile (presets Ultra Low /
  Low / Balanced / High / Ultra / Custom + adaptive quality + budgets) and
  VertexPerformanceManager singleton (VertexPerformance).
- [done] Module: vertex_optimizer — VertexProjectOptimizer (VertexOptimizer)
  + VertexOptimizationReport. Analyzes assets + live counters; safe-only
  auto-opts with dry-run.
- [done] Module: vertex_ai — VertexAIAssistant (VertexAI), VertexAICommand,
  VertexAIContext. Command registry, context gathering, confirm-before-
  destructive, pluggable backend Callable, built-in commands return plans.
- [done] Module: vertex_mobile_editor — VertexMobileSettings + editor plugin
  (TOOLS_ENABLED) with touch-friendly layout panel.
- [done] Module: vertex_benchmarks — VertexBenchmarkRunner +
  VertexBenchmarkResult (FPS/frame/draw/mem/startup collection).
- [done] Project Manager wiring: new projects seeded with
  Vertex/Performance/* keys.
- [done] Android: documented minSdk 24 limitation in config.gradle.
- See git log on `vertex-engine` for incremental commits.

## Singleton names (slash-free to avoid project-setting path semantics)
- VertexPerformance  -> VertexPerformanceManager
- VertexOptimizer     -> VertexProjectOptimizer
- VertexAI            -> VertexAIAssistant
Project-setting KEYS use slashes (Vertex/Performance/*, Vertex/export/*).
