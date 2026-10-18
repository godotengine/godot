# Mobile Forward+ smoke-test project

This project is a runtime/editor smoke-test harness for the Forward+ renderer on Vulkan-capable Android devices. The engine source also now prefers Forward+ as the Android mobile renderer default when the Forward+ renderer is compiled into that Android build. This default-selection change does not itself optimize shaders or guarantee compatibility on every device.

## Run

1. Open this folder with a build of Nexo-3D.
2. Confirm the renderer in Project Settings is **Forward+**.
3. Export to an Android device whose Vulkan driver supports the engine's required Forward+ features.
4. Capture startup logs and test PBR lighting, directional shadows, and clustered local-light rendering (eight colored OmniLight3D sources), then sustained rendering. The HUD prints renderer name, average FPS, approximate frame time, and elapsed time; the same samples are written to the engine log.

## Important

- The project requests Forward+ for both the default and mobile feature tag. This requests the renderer; it does not add missing Vulkan feature support or guarantee every Android GPU can run it.
- If startup fails or the device's Vulkan driver is unstable, switch the mobile renderer override back to `mobile` and retest.
- The default-renderer change is not a renderer optimization by itself. No Android build or physical-device run has yet been verified from this commit. Vulkan driver compatibility and performance still require engine builds and physical-device tests.
- Keep the Mobile renderer available as a fallback until device coverage and frame-time measurements justify broader use.

## Acceptance checklist

- [ ] Opens in the editor without project-setting warnings.
- [ ] Starts on a supported Android Vulkan device.
- [ ] PBR material and directional shadow render correctly.
- [ ] No crash or device loss during a 15-minute loop.
- [ ] CPU frame time, GPU frame time, memory use, and device temperature recorded.
- [ ] Compared against the Mobile renderer on the same device and scene.
