# Editor Startup Investigation (2026-04-16)

## Summary

Investigated a reported regression where the Godot editor in this fork
became "slow to start up and noticeably laggy/unresponsive in the last
1-2 days." Working from the benchmark snapshot captured at
`.tmp-editor-bench.json` (cold run: `[Servers] Rendering = 21.30s`,
`Main::Setup2 = 24.85s`), measured master HEAD (`b93ba6ef92`) against
the suspected baseline `7c9a815f03` (streaming tier 2 / #229) on
Windows, dev_build, debug_symbols, with a full rebuild for each.

**No startup regression was found between the two commits.** Cold
startup cost is essentially identical (within <1%), and the dominant
cost is shader compilation during Godot's `rendering_server->init()` —
engine-internal work that the Gaussian Splatting module does not
contribute to.

## Benchmark data

Measurements taken with:

```
bin/godot.windows.editor.dev.x86_64.console.exe \
  --path tests/examples/godot/test_project \
  --benchmark --quit-after 1
```

| Phase                  | #229 cold | HEAD cold | HEAD warm (2nd run) |
|------------------------|-----------|-----------|---------------------|
| `[Servers] Rendering`  | 21.44s    | 20.72s    | 3.10s               |
| `[Servers]` (total)    | 23.35s    | 22.57s    | 5.03s               |
| `[Startup] Main::Setup2` | 24.67s  | 24.30s    | 6.32s               |

Cold / warm distinction: "cold" = shader cache (`AppData\Roaming\Godot\
app_userdata\GaussianSplattingTest\shader_cache`) wiped before run;
"warm" = cache from a prior successful run present.

The user's bench file (`.tmp-editor-bench.json`) at 21.30s rendering
matches the cold profile almost exactly. HEAD is actually very slightly
faster than #229 on both cold and warm runs, so the observation that
the editor became *slower* in the last 1-2 days is not supported by
measurement on the two reference commits.

## Where the 21s cold window actually goes

With `--verbose --benchmark` against a wiped cache, every shader
emits a `Shader '<name>' ... SHA256: ...` / `Shader cache miss for
<name>/...` line in order of compilation. Cross-referencing the
benchmark closing line (`- Rendering: 20626.087 msec.`) with the
shader log shows that the following shaders compile **before** the
Rendering benchmark closes (all engine-internal Godot shaders, not
ours):

- Canvas: `CanvasSdfShaderRD`, `CanvasShaderRD`, `CanvasOcclusionShaderRD`
- Scene skeleton/particles: `SkeletonShaderRD`, `SortShaderRD`,
  `ParticlesShaderRD`, `ParticlesCopyShaderRD`
- Forward+ core: `ClusterRenderShaderRD`, `ClusterStoreShaderRD`,
  `ClusterDebugShaderRD`, `SceneForwardClusteredShaderRD` (4 groups,
  3 variants each on the hot path)
- PBR/Normals: `BestFitNormalShaderRD`, `IntegrateDfgShaderRD`
- Resolve/TAA/FSR2/FSR: `ResolveShaderRD`, `TaaResolveShaderRD`,
  `Fsr2*ShaderRD` (8 stages), `FsrUpscaleShaderRD`
- Screen-space effects: `SsEffectsDownsampleShaderRD`, `Ssil*ShaderRD`
  (4), `Ssao*ShaderRD` (4), `ScreenSpaceReflection*ShaderRD` (3)
- SSS/Sky/GI: `SubsurfaceScatteringShaderRD`, `SkyShaderRD`,
  `VoxelGi*ShaderRD` (2), `Sdfgi*ShaderRD` (5), `GiShaderRD`
- Volumetric: `VolumetricFog*ShaderRD` (2 × 4 groups)
- Post: `BokehDofShaderRD`, `Copy*ShaderRD` (2), `Cubemap*ShaderRD`
  (3), `SpecularMergeShaderRD`, `ShadowFrustumShaderRD`,
  `MotionVectorsShaderRD`, `LuminanceReduceShaderRD`, `Smaa*ShaderRD`
  (3), `TonemapShaderRD`, `VrsShaderRD`, `BlitShaderRD`

That is ~50 engine shaders compiling serially under dev_build with
debug_symbols, which is the known-slow mode for shader compilation
because the GLSL compiler runs at low optimization inside an
unoptimized Godot. At roughly 300-500 ms per shader on cold cache,
20-21s is consistent.

**Our module's shaders compile *after* the Rendering benchmark
closes**, during scene load / first render_scene_instance — the
verbose output confirms `SobelOutlineShaderRD`, `BrushAccumulateShaderRD`,
`GaussianSplatShaderRD`, `TileBinningShaderRD`, `TileRasterizerShaderRD`,
`TileRasterizerComputeShaderRD`, `TilePrefixScanShaderRD`,
`TileResolveShaderRD`, `FrustumCullShaderRD`, `DepthComputeShaderRD`,
`InstanceCountClampShaderRD`, `InstanceChunkDispatchShaderRD` all
print after `- Rendering: 20626.087 msec.`. These compiles land in
`[Startup] Load Game` / `Main::Start`, not in `[Servers] Rendering`.

## What does not appear to be happening

- No heavy work in `register_types.cpp` at `MODULE_INITIALIZATION_LEVEL_SERVERS`
  (the module doesn't hook that level at all — only SCENE and EDITOR).
- No filesystem scans, no eager RID allocation, no forced shader
  precompile in our `GaussianSplatManager::initialize_module()` — it's
  just `GLOBAL_DEF` calls and a `gs::sorting_settings::*` registration.
- No static constructors / `__attribute__((constructor))` in the module
  that would fire before `main()` and block rendering server init.
- `servers/rendering/renderer_rd/` has had **zero** non-trivial changes
  from us in the last 30 days (one `WARN_PRINT_ONCE` and an
  include-hook style fix). The `GaussianSplatStorage` ctor is
  trivial.

## Why the user may perceive a recent slowdown

Three plausible explanations, none a code regression in the last 1-2
days:

1. **Shader cache thrash from commit-switching.** Godot keys shader
   caches by source SHA256 + driver id. If the user has been switching
   between branches with shader-source churn
   (`shaders/includes/gs_render_params.glsl`, `tile_binning.glsl`,
   `tile_rasterizer_compute.glsl`, `tile_resolve.glsl`,
   `gs_directional_shadow.glsl`, `gs_culling_utils.glsl` all changed
   between #229 and HEAD), each switch forces a cold window on our
   shaders. Godot engine shaders stay cached across our switches, but
   ~12 of our custom shaders recompile per switch.

2. **Stuck headless-import zombies.** When I landed on this worktree,
   two `bin/godot.windows.editor.dev.x86_64(.console).exe --headless
   --import --path tests/examples/godot/test_project` processes
   (PIDs 42020 and 63672) were still running and were holding a lock
   on the editor binary (`Zugriff verweigert` on `scons` link step).
   A stuck headless import on the same project blocks later editor
   launches from acquiring the project or completes the import on top
   of the new run, each of which looks like "the editor is frozen"
   from the user's side. Killing both unlocked the build and editor
   launch. Worth checking `Task-Manager` for leftover
   `godot.windows.editor.dev.x86_64*.exe` processes next time this
   gets reported.

3. **dev_build overhead amplified by project state.** With an empty
   `.godot/imported/` (which this worktree's test project had), the
   first editor open has to import every resource under the project —
   and PRs #229 and #240 introduced 8 synthetic `.ply` fixtures under
   `tests/examples/godot/test_project/tests/fixtures/` (sizes up to
   1.6 MB). Those imports are the first thing
   `EditorFileSystem` will do after startup benchmarks close,
   contributing to a laggy-feeling post-splash experience. The
   benchmark file does not capture this because import happens after
   `Main::Setup2`. If this is the observed slowness, it's one-time
   per fresh worktree, not a progressive regression.

## What was changed in this commit

Removed leftover `[DIAG-*]` `print_line` blocks that were introduced
under the "corridor black screen investigation" (added as part of PR
#229, at `a7e950de79`). These use a `static int _diag_N = 0; if
(++... <= N || ... % 60 == 0) print_line(...)` pattern — classic
temporary-debug style — and fire on the per-frame render path
(`render_scene_instance`, `_select_backend_plan`,
`_record_raster_metrics`, `_dispatch_raster`, etc.). They are *not*
the cause of the 21s cold startup, but they make the stdout of a
running editor noisy forever (one line per second roughly, per
diagnostic) and they survived past the investigation they were
created for.

Files touched:

- `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp` —
  removed `[DIAG-RESIDENT]`, `[DIAG-RSI]`, `[DIAG-ROUTE]`.
- `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp` —
  removed `[DIAG-RASTER]`, `[DIAG-DISPATCH]`, `[DIAG-RRESULT]`.
- `modules/gaussian_splatting/interfaces/output_compositor.cpp` —
  removed `[DIAG-COPY-DEVICE]`, `[DIAG-COMPOSITOR]`,
  `[DIAG-COMPOSITOR2]`; also the now-unused `godot_rd` local.
- `modules/gaussian_splatting/core/gaussian_streaming.cpp` — removed
  `[DIAG-SYNC-LOAD]` (pack/success variants) and `[DIAG-UPLOAD]`.

The module already has a gated debug infrastructure
(`enable_pipeline_trace`, `enable_frame_logging`, 38+ instrumentation
points via `render_debug_state_orchestrator`, and a `gs_debug_trace`
data-flow system). All of it is off by default and is the
supported way to reproduce what the DIAG prints were giving. See
`CLAUDE.md#Debugging Rendering Issues` for how to enable.

## What is still open

- The **warm** `[Servers] Rendering = 3.1s` is higher than a vanilla
  Godot editor on the same hardware (expected <1s). Roughly 2s of
  overhead is attributable to having the Gaussian Splatting module
  linked in — I did not nail down whether this is shader-cache IO,
  `RendererCompositorRD` storage setup, or something the dev_build
  asserts amplify. For normal use this is fine and it's not a
  recent regression, but if the user wants startup tighter, the next
  thing to profile is the `RendererCompositorRD::RendererCompositorRD()`
  ctor and `scene->init()` call.

- **Editor-time FPS floor**. During `--quit-after 300` I observed
  the editor running at 7-9 FPS while previewing
  `public_evaluator.tscn`. This is a dev_build symptom (scene
  previews with a `GaussianSplatWorld3D` ride the full pipeline even
  when just 5 splats are visible); it is not a new regression to my
  knowledge but is worth flagging if the user considers 8 FPS in the
  editor to be the "laggy/unresponsive" symptom. Next steps if it is:
  capture a frame with RenderDoc while the editor is idle on that
  scene and look at the per-dispatch cost.
