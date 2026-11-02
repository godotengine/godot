# Tint Patches for Godot WebGPU

These patches modify the pinned upstream Tint source for Godot's WebGPU backend.
Preparation extracts the locked source subset and applies these patches in order.
See [dependency preparation](../../../drivers/webgpu/dependencies.md).

## Applying

From the repository root:

```text
uv run --no-project python drivers/webgpu/dependencies.py
uv run --no-project python drivers/webgpu/dependencies.py --verify
```

## Patch Summary

| Patch | Files | Group | Description |
|-------|-------|-------|-------------|
| 0000 | atomics.cc | Atomic access | Preserve matrix/vector element type traversal from the original snapshot |
| 0001 | validate.cc | UBO layout | `SetSkipBlockLayout(true)` + improved error messages |
| 0002 | validator.h, validator.cc, reader.cc | Spec constants | `kAllowStructMemberSizeMismatch` capability |
| 0003 | decompose_strided_array.cc | Spec constants | Skip padding when stride < element size |
| 0004 | shader_io.cc | Point size | Accept non-constant `point_size` stores |
| 0005 | ir_to_program.cc | Spec constants | `@size` emission guard + capability |
| 0006 | parse_num.cc | Vendoring | Replace `absl::from_chars` with `std::from_chars` |
| 0007 | validator.cc | UBO layout | Extend `kAllowStructMemberSizeMismatch` to the struct TOTAL-size check (spec-constant-sized arrays leave `Size()` at its unfolded value) |
| 0008 | parse_num.cc | Emscripten 3.x | Locale-independent floating-point parsing when libc++ deletes the `from_chars` overloads |
| 0009 | parse_num.cc | macOS | Reuse locale-independent floating-point parsing below the macOS 26 availability of `from_chars` |
| 0010 | inspector.h, inspector.cc | Resource reflection | Expose all module resource declarations, including unused bindings whose dimensions Godot retains in layouts |
| 0011 | parser.cc | Point coordinates | Emulate PointCoord as a private vec2(0.5), including unused inputs and component access |
| 0012 | lower.cc, texture.cc | Control flow and depth | Repair loop exits crossing switches; lower non-comparison depth reads to texel loads; restore Godot's marked depth-copy source type |
| 0013 | ice.h, ice.cc | Error recovery | Throw internal compiler errors to the C wrapper; compiler sources are built with exception unwinding |

## Logical Groups

**Group A — UBO Layout (0001, 0007)**: Godot uses C++ struct packing for uniform buffers,
not std140/std430. Always necessary.

**Group B — Specialization Constants (0002, 0003, 0005)**: Godot's specialization
constants can change struct/array sizes at runtime, creating size mismatches that
Tint's IR validator and lowering passes don't expect. These patches relax validation
and prevent invalid WGSL output.

**Group C — Point Size (0004)**: Godot shaders pass through `gl_PointSize` with
non-constant values. Tint strips point_size during lowering but validates the stored
value first. This patch relaxes that validation. Could potentially be moved to
`spirv_preprocess.cpp` or proposed upstream.

**Group D — Vendoring (0006, 0008, 0009)**: Replaces the Abseil dependency with C++17
`std::from_chars`, with a locale-independent fallback where libc++ lacks the floating-point
overloads (Emscripten 3.x, macOS before 26). Always necessary when vendoring without Abseil.

## Upstream Source

Patches 0011 and 0012 adapt the PointCoord, loop-exit and depth-sampling work in
[V-Sekai-fire/entities-godot](https://github.com/V-Sekai-fire/entities-godot/tree/3a5f1a4baa38c246c5e7c7f3ef79ebe306c96281)
to this pinned Tint revision. Depth sampling emulates linear-clamp reads at mip
zero for implicit LOD/bias/gradients; explicit levels are retained.

Tint is extracted from [Dawn](https://dawn.googlesource.com/dawn) at the exact
revision in `drivers/webgpu/dependencies.json`. The patch stack reproduces the
previous snapshot, including the atomic aggregate fix recorded during the move
to upstream references.
