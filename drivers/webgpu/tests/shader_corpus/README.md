# Shader corpus

GLSL fixtures and their SPIR-V for the SPIR-V → WGSL conversion used by the WebGPU backend.
`../preprocessing_tests/run_tests.mjs` converts every fixture with `bin/tint_convert_cli`
and checks the transformations listed there; it fails when the converter has not been built.

## Fixtures

| Shader | Stage | Key features |
|--------|-------|--------------|
| basic_vertex | vert | Push constants, UBO, gl_Position |
| basic_fragment | frag | Combined samplers, push constants |
| compute_particles | comp | SSBO read/write, push constants |
| camera_struct | comp | Camera structs passed by value, current and previous-frame matrices |
| shadow_pass | vert | Minimal push constant MVP |
| depth_sampling | frag | sampler2DShadow, depth comparison |
| storage_image | comp | image2D with r8 format |
| multi_texture | frag | Six samplers including samplerCube |
| spec_constants | frag | Specialization constants (bool/int/float) |
| per_stage_overrides_vert, per_stage_overrides_frag | vert, frag | Different specialization constants per stage |
| chained_spec_ops | comp | Chained OpSpecConstantOp |
| many_overrides | comp | 24 specialization constants |
| readonly_ssbo | comp | restrict readonly buffer |
| memory_barrier | comp | memoryBarrierShared() and barrier() |
| texture_array | frag | Texture array with constant and dynamic indices |
| sampler_array | frag | Array of combined samplers |
| sampler_params | frag | One helper sampling two combined samplers |
| sampler_control_flow | comp | 2D and array sampler loads consumed after an inlined branch |

The SPIR-V is SPIR-V 1.3 for a Vulkan 1.1 client, the targets Godot's glslang uses for
WebGPU shaders. `compile_fixtures.sh` regenerates it with `glslangValidator`.

## Engine shader dumps

`validate_spirv_dump.mjs` converts every SPIR-V file in a directory, such as the stages
written by a WebGPU engine run with `GODOT_DUMP_SPIRV=<directory>`:

```text
node drivers/webgpu/tests/shader_corpus/validate_spirv_dump.mjs <directory>
node drivers/webgpu/tests/shader_corpus/validate_spirv_dump.mjs <directory> --update-baseline
```

`expected_failures.json` lists dumped variants the Mobile renderer does not use; only
other failures fail the run. `--update-baseline` rewrites the list and keeps the failure
categories of shaders that still fail.
