# Vertex Game Engine

Vertex Game Engine is a heavily customized, optimized, professional
**Godot-based** engine. It is *not* a from-scratch reimplementation: it modifies
and extends the existing Godot 4.8 source directly, preserving Godot's
architecture and all third-party licenses (see `LICENSE.txt`, `COPYRIGHT.txt`,
`AUTHORS.md`).

The user-facing identity is **Vertex Game Engine** ("Vertex"). Internal class
names, namespaces, and technical identifiers are intentionally *not* renamed, to
keep binary/source compatibility with existing Godot modules and tooling.

## What changed (this branch: `vertex-engine`)

### Branding
- `version.py`: short_name `vertex`, name `Vertex Game Engine`.
- Config/cache directories now use the Vertex identity (`vertex` on most OSes,
  `Vertex` on Windows) instead of `godot`.
- Boot splash (`main/splash.png`), editor splash (`main/splash_editor.png`), and
  app icon (`main/app_icon.png`) replaced with a clean centered **V** logo on a
  dark background (see `.vertex_assets/generate_vertex_branding.py`).
- Editor/Project Manager window titles and user-facing strings updated.

### New engine modules (auto-detected by SCons, no engine rewrite)
- **vertex_perf** — `VertexPerformanceProfile` (presets: Ultra Low / Low /
  Balanced / High / Ultra / Custom) + `VertexPerformanceManager` singleton with
  adaptive quality that scales 3D render scale down on sustained frame-time
  spikes and recovers when comfortable.
- **vertex_optimizer** — `VertexProjectOptimizer` analyzes textures/assets/
  shaders/particles/draw calls/memory/physics and produces recommendations;
  safe automatic optimizations only (dry-run by default, never deletes files).
- **vertex_ai** — `VertexAIAssistant` architecture: command registry, project
  context gathering, confirm-before-destructive execution, pluggable LLM
  backend (Callable; no network calls from the engine). Built-in commands:
  create_player, create_scene, fix_error, optimize_project,
  reduce_memory_usage, explain_node, create_animation, create_ui,
  optimize_shader.
- **vertex_mobile_editor** — `VertexMobileSettings` + an editor plugin
  (TOOLS_ENABLED only) with a touch-friendly layout panel (large touch targets,
  compact toolbar, collapsible panels, virtual-keyboard assist).
- **vertex_benchmarks** — `VertexBenchmarkRunner` collects FPS / frame time /
  draw calls / memory / startup time across named workloads.

### Project Manager & export
- New projects are seeded with `Vertex/Performance/*` keys (profile preset,
  target platform, low-end optimization, adaptive quality).
- Export branding settings: `Vertex/export/show_powered_by_vertex`,
  `Vertex/export/attribution`, `Vertex/export/splash_fade_seconds`.

### Android
- `minSdk` remains 24 (documented): Godot 4.8's AndroidX/Jetifier + NDK toolchain
  effectively require it; lowering to API 21 needs toolchain/dependency changes.
- Adaptive performance for low-end Android is provided at runtime by
  `vertex_perf`, not by hard promises of "zero lag".

## Building

Build exactly like Godot (SCons). The new modules are auto-detected under
`modules/`. Example (Linux):

```
scons platform=linuxbsd target=editor
```

> **Note:** These changes were made in a sandbox without a full Godot build
> toolchain (SCons + X11/GL/Vulkan headers were unavailable), so a binary
> build could not be run there. All new code follows existing Godot module
> conventions (config.py / SCsub / register_types) and passes structural
> checks; **run a real `scons` build in a configured environment** to produce
> the editor binary and confirm. See `AGENTS.md` for the full file inventory
> and the build-environment limitation.

## Preserving licenses

Godot Engine copyright and the MIT License are preserved in all files. The new
Vertex modules keep the standard Godot copyright header. No third-party notices
were removed.

## Next steps (continued work)
- Add the in-editor UI selectors for performance profile / target platform /
  low-end optimization in the Project Manager (data model already wired).
- Implement the full responsive mobile editor layout (scene tree, inspector,
  pinch-zoom) on top of `vertex_mobile_editor`.
- Wire a real LLM backend into `VertexAIAssistant.backend`.
- Render the "Powered by Vertex" attribution on the exported boot splash.
- Run the benchmark workloads after each major change to track regressions.
