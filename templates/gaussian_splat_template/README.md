# Gaussian Splat Template Project

_Last updated: 2025-11-27_

This directory contains a self-contained Godot project configured for the GodotGS Gaussian splatting engine. Import `project.godot` in the editor to explore a fully wired scene that demonstrates best practices for using `GaussianSplatNode3D`.

## Features

- **Preconfigured project settings** for Forward+ rendering, Vulkan threading, and Gaussian splatting budgets.
- **Autoload bootstrap** (`GaussianBootstrap`) that verifies engine capabilities and exposes module-wide statistics.
- **Template scene** (`scenes/main.tscn`) with:
  - Orbit-ready camera rig and directional light.
  - Ground reference mesh and painterly-enabled `GaussianSplatNode3D` pointing at `assets/template_splats.ply`.
  - Canvas-based performance overlay with node, renderer, and global metrics.
- **Input map** tuned for navigation (WASD, Space, C, Shift, RMB orbit, MMB pan, mouse wheel zoom).

## Scene hierarchy

```
GaussianTemplate (Node3D, `scripts/main_scene.gd`)
├── GaussianSplatNode3D (asset + rendering options prefilled)
├── CameraRig (OrbitCameraRig script)
│   └── Camera3D
├── DirectionalLight3D
├── Ground (MeshInstance3D)
└── CanvasLayer
    └── PerformanceOverlay (PackedScene)
```

## GaussianSplatNode3D inspector defaults

The template applies the recommended inspector values programmatically and in the packed scene:

- **Asset**: `ply_file_path = res://assets/template_splats.ply`, `auto_load = true`.
- **Quality**: `preset = Balanced`, `lod_bias = 1.0`, `max_render_distance = 150m`, `max_splat_count = 750,000`.
- **Painterly**: enabled with edge threshold `0.25`, stroke opacity `0.85`, stroke width `1.1`, color variation `0.12`, temporal blend `0.35`, seed `1337`.
- **Rendering**: update when visible, cast shadows, frustum and occlusion culling on, opacity `1.0`.
- **Debug**: inspector preview and LOD spheres enabled, other overlays off by default, debug draw mode `Points`.

These values mirror the guidance in the Gaussian Splatting inspector documentation and can be tweaked safely to suit your project.

## Getting started

1. Open the Godot project manager and import `project.godot` from this folder.
2. Press **F5** to run the template scene. You should see the sample splat cloud rendered with painterly shading.
3. Use the navigation controls listed in the on-screen overlay to orbit and inspect the splats.
4. Duplicate `scenes/main.tscn` to bootstrap new levels or swap `default_ply_path` in `scripts/main_scene.gd` to point at your own `.ply`/`.gsf` assets.

For more details on Gaussian splatting workflows, review the documentation in `docs/getting-started/` and `docs/artist_pipeline.md`.
