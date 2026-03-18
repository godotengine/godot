# User Manual: Lighting Behavior

This page describes what lighting usually looks like on splats and which controls give the safest, fastest results.

## What You Should Expect Visually

- Splats react to direct lights, but shading is not identical to mesh materials.
- Shadowed areas should read clearly, with strength controlled by project lighting settings.
- Scene readability depends on both direct lighting and indirect/environment contribution.
- Lighting changes should be obvious after adjusting light intensity, direction, or shadow strength.

## Common Confusion Points

- "Everything looks flat": usually too little direct light or no meaningful light direction.
- "Shadows are too harsh": shadow strength is likely too high for the scene style.
- "Shadows seem missing": the light may not cast shadows, or receiver settings are too weak.
- "One level looks correct, another does not": each scene can have different environment and post-processing setup.

## Which Setting Knobs Are Safe to Tweak First

1. `direct_light_scale` (first control for stronger/weaker light response)
2. `indirect_sh_scale` (ambient fill amount)
3. `shadow_strength` (how heavy shadows read)
4. Keep bias controls for later fine-tuning

Reference:
- [../../reference/project-settings.md](../../reference/project-settings.md)

## When to Use Troubleshooting Docs

Use troubleshooting docs when lighting remains broken after basic adjustments:

- persistent incorrect shadow behavior
- severe flicker or unstable lighting from frame to frame
- startup/runtime shader errors affecting light or shadow output

Start here:
- [../../troubleshooting/recurring-issues.md](../../troubleshooting/recurring-issues.md)
- [../../troubleshooting/quick-reference.md](../../troubleshooting/quick-reference.md)

## Deeper Architecture (Engineers)

For shader/path internals and constraints:
- [../../architecture/lighting-system.md](../../architecture/lighting-system.md)
- [../../architecture/render-pipeline.md](../../architecture/render-pipeline.md)
