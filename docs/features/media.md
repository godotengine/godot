# Version-Controlled Media

## Purpose

Define how to include images and videos in docs content while keeping media artifacts versioned with the repository.

## Folder Conventions

| Media type | Path |
| --- | --- |
| Images | `docs/assets/images/` |
| Videos | `docs/assets/videos/` |

## Example: Image

![Gaussian pipeline preview](../assets/images/gaussian-pipeline.svg)

## Example: Video

<video controls preload="metadata" width="640">
  <source src="../assets/videos/gaussian-demo.webm" type="video/webm">
  <source src="../assets/videos/gaussian-demo.mp4" type="video/mp4">
  Your browser does not support embedded video playback.
</video>

## Authoring Rules

- Keep docs media under `docs/assets/` so each tag/version serves matching assets.
- Prefer `.webm` + `.mp4` pairs for compatibility.
- Use descriptive file names tied to feature/use case.
- For large video files, use Git LFS tracking (`*.mp4`, `*.webm`).
- Keep total media footprint within CI budget gates.

## Git LFS Setup

```bash
git lfs install
git lfs track "*.mp4" "*.webm"
```
