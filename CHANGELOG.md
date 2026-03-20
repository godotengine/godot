# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Versioned GitHub Pages docs pipeline with media LFS support
- Single-process benchmark orchestrator for unified scene benchmarking
- Visual metrics to synthetic benchmark scenes
- Shader dependency guard and async readback test hardening
- Tile shared-memory raster contract enforcement in editor
- RWLock migration for gaussian core data
- Streaming queue pressure controller and scan throttle
- LOD distance-based chunk reduction (Octree-GS approach)
- Multi-asset atlas registration for streaming
- Per-chunk quantization (Unity-inspired 4x compression)
- Animation state machine with keyframe interpolation and clip blending
- Color grading bake/restore workflow on GaussianSplatNode3D
- Documentation: streaming feature guide, animation feature guide, expanded performance presets

### Changed

- Benchmark scenes unified under single-process orchestrator
- Streaming path uses shared `gs::settings` helpers (ISSUE-018)
- Removed friend-only access in streaming orchestrator (ISSUE-019)
- Instance sort cache preserved across strict-mode toggles

### Fixed

- Duplicated mike version/alias in docs deploy workflow
- GDScript const compatibility in benchmark scripts
- Missing `ColorGradingResource` include and `batches_completed` getter
- Trailing whitespace in sorting pipeline
- CPU fallback and OneSweep support in sorting validation
