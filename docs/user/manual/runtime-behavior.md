# User Manual: Runtime Behavior

This page explains what artists usually observe while navigating a splat scene, and which controls are safest to touch first.

## What You Should Expect Visually

- First load can take longer while data streams and GPU resources warm up.
- Camera movement should feel stable after load settles.
- Some distant detail popping can happen as streaming updates visible chunks.
- Transparent splat content can shift slightly as sort order updates.

## Common Confusion Points

- "It looked better in one scene than another": light setup and exposure strongly change perceived splat quality.
- "Parts are missing": often `Render Distance` or max splat budget is too low.
- "It stutters only at scene start": this is often startup/streaming work, not a permanent runtime state.
- "Editor view and play mode differ": post-processing and camera setup can differ between scenes.

## Which Setting Knobs Are Safe to Tweak First

1. Preset (`Balanced`, then move toward performance or quality)
2. Max splat count (raise/lower visible density)
3. Render distance (trade distant detail for stability)
4. Only then advanced streaming/sorting options

Quick references:
- Presets: [performance-presets.md](performance-presets.md)
- Workflows: [workflows.md](workflows.md)

## When to Use Troubleshooting Docs

Use troubleshooting pages when behavior is persistent, not a one-time warmup:

- repeated heavy flicker or mis-ordered transparency
- splats consistently fail to appear
- shader/pipeline errors, or runtime starts with no visible result

Start here:
- [../../troubleshooting/recurring-issues.md](../../troubleshooting/recurring-issues.md)

## Deeper Architecture (Engineers)

For internals and stage-level details:
- [../../architecture/render-pipeline.md](../../architecture/render-pipeline.md)
- [../../architecture/overview.md](../../architecture/overview.md)
