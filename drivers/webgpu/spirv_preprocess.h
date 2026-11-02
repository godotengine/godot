/**************************************************************************/
/*  spirv_preprocess.h                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/templates/hash_map.h"
#include "core/templates/vector.h"

#include <cstdint>

// SPIR-V rewrites that adapt Godot's Vulkan-flavoured SPIR-V for Tint and WebGPU.
// Each pass returns its input unchanged (and reports an error) when the module is malformed.
namespace spirv_preprocess {

// Push constants become a read-only storage buffer at this group and binding. Every other binding is doubled
// by split_combined_samplers, so this is the highest binding WebGPU allows (maxBindingsPerBindGroup is 1000).
static constexpr uint32_t PUSH_CONSTANT_SET = 3;
static constexpr uint32_t PUSH_CONSTANT_BINDING = 999;

// Runs every pass below in the order the WebGPU driver needs before Tint.
Vector<uint8_t> run_pipeline(const Vector<uint8_t> &p_bytes);

// SPIRV-Tools passes: preserve scalar specialization constants, fold constant expressions, and inline
// shader calls. Unsupported composite specialization operations fall back to patched default values.
Vector<uint8_t> freeze_and_inline(const Vector<uint8_t> &p_bytes);

// Rewrite OpCopyLogical (SPIR-V 1.4+) to the equivalent OpCopyObject.
Vector<uint8_t> rewrite_copy_logical(const Vector<uint8_t> &p_bytes);

// Rewrite OpTerminateInvocation to OpKill, which Tint translates to discard.
Vector<uint8_t> rewrite_terminate_invocation(const Vector<uint8_t> &p_bytes);

// Classify runtime float bits without OpIsNan/OpIsInf, which Tint's SPIR-V reader cannot import.
Vector<uint8_t> rewrite_float_classification(const Vector<uint8_t> &p_bytes);

// Reserve binding ranges shared by all stages before scalarizing their descriptor arrays.
using BindingMap = HashMap<uint64_t, uint32_t>; // (set << 32 | original binding) -> reserved binding.
Vector<uint8_t> remap_bindings(const Vector<uint8_t> &p_bytes, const BindingMap &p_bindings);

// Lower fixed-size descriptor arrays to individual bindings, using switches for dynamic indices.
Vector<uint8_t> flatten_binding_arrays(const Vector<uint8_t> &p_bytes);

// Split combined image samplers. Every binding N is doubled; a split sampler takes 2N and its image 2N+1.
// Combined samplers passed to functions must have been inlined by freeze_and_inline.
Vector<uint8_t> split_combined_samplers(const Vector<uint8_t> &p_bytes);

// Turn push-constant variables into read-only storage buffers at PUSH_CONSTANT_SET/PUSH_CONSTANT_BINDING.
// Runs after split_combined_samplers so that binding is not doubled.
Vector<uint8_t> convert_push_constants_to_storage_buffers(const Vector<uint8_t> &p_bytes);

// Set OpTypeImage depth=2 (unknown) to depth=1, which WGSL can express as a depth texture.
Vector<uint8_t> fix_depth2_images(const Vector<uint8_t> &p_bytes);

// Names a feature WGSL cannot express (16-bit types, image atomics, formatless storage images), or nullptr.
// Variants using one exist for other devices; Tint would fail on them, some with internal errors.
const char *unsupported_feature(const Vector<uint8_t> &p_bytes);

// Negate gl_Position.y before every vertex-shader return; Godot's SPIR-V targets Vulkan's flipped clip space.
Vector<uint8_t> negate_position_y(const Vector<uint8_t> &p_bytes);

// Strip Restrict, Volatile and InputAttachmentIndex decorations, which Tint does not support.
Vector<uint8_t> strip_restrict_decoration(const Vector<uint8_t> &p_bytes);

// Remove OpMemoryBarrier, which Tint does not support; workgroup barriers come from OpControlBarrier.
Vector<uint8_t> strip_memory_barrier(const Vector<uint8_t> &p_bytes);

// Replace infinite and NaN float constants with +/-FLT_MAX (or DBL_MAX); Tint rejects non-finite literals.
Vector<uint8_t> fix_nonfinite_literals(const Vector<uint8_t> &p_bytes);

// WGSL storage buffers cannot be write-only. Remove NonReadable only from StorageBuffer variables and their
// block members, retaining image access.
Vector<uint8_t> normalize_storage_buffer_access(const Vector<uint8_t> &p_bytes);

// Decorate StorageBuffer variables that are never written as NonWritable so Tint emits var<storage, read>.
Vector<uint8_t> infer_readonly_storage(const Vector<uint8_t> &p_bytes);

} // namespace spirv_preprocess
