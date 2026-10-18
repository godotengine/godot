# Mobile Forward+ smoke-test project

This project is a small runtime/editor smoke-test harness for the Forward+ renderer on Vulkan-capable Android devices. It is intentionally separate from the engine source so it cannot silently change the renderer of existing projects.

## Run

1. Open this folder with a build of Nexo-3D.
2. Confirm the renderer in Project Settings is **Forward+**.
3. Export to an Android device whose Vulkan driver supports the engine's required Forward+ features.
4. Capture startup logs and test camera movement, PBR lighting, directional shadows, and sustained rendering.

## Important

- The project requests Forward+ for both the default and mobile feature tag. This requests the renderer; it does not add missing Vulkan feature support or guarantee every Android GPU can run it.
- If startup fails or the device's Vulkan driver is unstable, switch the mobile renderer override back to `mobile` and retest.
- This harness is not a claim that the engine has been optimized, compiled, or validated on Android. Those require engine builds and physical-device tests.
- Keep the Mobile renderer available as a fallback until device coverage and frame-time measurements justify broader use.

## Acceptance checklist

- [ ] Opens in the editor without project-setting warnings.
- [ ] Starts on a supported Android Vulkan device.
- [ ] PBR material and directional shadow render correctly.
- [ ] No crash or device loss during a 15-minute loop.
- [ ] CPU frame time, GPU frame time, memory use, and device temperature recorded.
- [ ] Compared against the Mobile renderer on the same device and scene.
