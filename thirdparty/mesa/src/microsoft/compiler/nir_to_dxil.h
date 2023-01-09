/*
 * Copyright © Microsoft Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef NIR_TO_DXIL_H
#define NIR_TO_DXIL_H

#include <stdbool.h>

#include "nir.h"
#include "dxil_versions.h"

typedef struct GodotNirCallbacks GodotNirCallbacks;

#ifdef __cplusplus
extern "C" {
#endif

struct blob;

/* Controls how resource decls/accesses are handled. Common to all:
 *   Images, textures, and samplers map to D3D UAV, SRV, and sampler types
 *   Shared is lowered to explicit I/O and then to a DXIL-specific intrinsic for 4-byte indices instead of byte addressing
 *   Input/output are lowered to dedicated intrinsics
 */
enum dxil_environment {
   /* In the GL environment:
    *   Samplers/textures are lowered, vars/intrinsics use binding to refer to them; dynamic array indexing not yet supported
    *     The lowering done by mesa/st assigns bindings from 0 -> N
    *   All other resource variables have driver_location set instead, assigned from 0 -> N
    *   UBOs may or may not have interface variables, and are declared from ubo_binding_offset -> num_ubos; no dynamic indexing yet
    *   SSBOs may or may not have interface variables, and are declared from from 0 -> num_ssbos; no dynamic indexing yet
    *   Images are *not* lowered, so that dynamic indexing can deterministically get a base binding via the deref chain
    *   No immediate constant buffer, or scratch
    */
   DXIL_ENVIRONMENT_GL,
   /* In the CL environment:
    *   Shader kind is always KERNEL
    *   All resources use binding for identification
    *   Samplers/textures/images are lowered; dynamic indexing not supported by spec
    *   UBOs are arrays of uints in the NIR
    *   SSBOs are implicitly declared via num_kernel_globals
    *   Variables of shader_temp are used to declare an immediate constant buffer, with load_ptr_dxil intrinsics to access it
    *   Scratch is supported and lowered to DXIL-specific intrinsics for scalar 32-bit access
    */
   DXIL_ENVIRONMENT_CL,
   /* In the Vulkan environment:
    *   All resources use binding / descriptor_set for identification
    *   Samplers/textures/images are not lowered
    *     Deref chains are walked to emit the DXIL handle to the resource; dynamic indexing supported
    *   UBOs/SSBOs are struct variables in the NIR, accessed via vulkan_resource_index/load_vulkan_descriptor; dynamic indexing supported
    *   Read-only SSBOs, as declared in the SPIR-V, are bound as raw buffer SRVs instead of UAVs
    *   No immediate constant buffer or scratch
    */
   DXIL_ENVIRONMENT_VULKAN,
};

struct nir_to_dxil_options {
   bool interpolate_at_vertex;
   bool lower_int16;
   bool disable_math_refactoring;
   bool no_ubo0;
   bool last_ubo_is_not_arrayed;
   unsigned provoking_vertex;
   unsigned num_kernel_globals;
   unsigned input_clip_size;
   enum dxil_environment environment;
   enum dxil_shader_model shader_model_max;
   enum dxil_validator_version validator_version_max;
   const GodotNirCallbacks *godot_nir_callbacks;
};

typedef void (*dxil_msg_callback)(void *priv, const char *msg);

struct dxil_logger {
   void *priv;
   dxil_msg_callback log;
};

bool
nir_to_dxil(struct nir_shader *s, const struct nir_to_dxil_options *opts,
            const struct dxil_logger *logger, struct blob *blob);

const nir_shader_compiler_options*
dxil_get_nir_compiler_options(void);

#ifdef __cplusplus
}
#endif

#endif
