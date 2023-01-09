/*
 * Copyright © 2015 Intel Corporation
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Jason Ekstrand (jason@jlekstrand.net)
 *
 */

#ifndef _NIR_SPIRV_H_
#define _NIR_SPIRV_H_

#include "util/disk_cache.h"
#include "compiler/nir/nir.h"
#include "compiler/shader_info.h"

#ifdef __cplusplus
extern "C" {
#endif

struct nir_spirv_specialization {
   uint32_t id;
   nir_const_value value;
   bool defined_on_module;
};

enum nir_spirv_debug_level {
   NIR_SPIRV_DEBUG_LEVEL_INVALID = -1,
   NIR_SPIRV_DEBUG_LEVEL_INFO,
   NIR_SPIRV_DEBUG_LEVEL_WARNING,
   NIR_SPIRV_DEBUG_LEVEL_ERROR,
};

enum nir_spirv_execution_environment {
   NIR_SPIRV_VULKAN = 0,
   NIR_SPIRV_OPENCL,
   NIR_SPIRV_OPENGL,
};

struct spirv_to_nir_options {
   enum nir_spirv_execution_environment environment;

   /* Whether to keep ViewIndex as an input instead of rewriting to a sysval.
    */
   bool view_index_is_input;

   /* Create a nir library. */
   bool create_library;

   /* Whether to use nir_intrinsic_deref_buffer_array_length intrinsic instead
    * of nir_intrinsic_get_ssbo_size to lower OpArrayLength.
    */
   bool use_deref_buffer_array_length;

   /* Initial value for shader_info::float_controls_execution_mode,
    * indicates hardware requirements rather than shader author intent
    */
   uint16_t float_controls_execution_mode;

   /* Initial subgroup size.  This may be overwritten for CL kernels */
   enum gl_subgroup_size subgroup_size;

   /* True if RelaxedPrecision-decorated ALU result values should be performed
    * with 16-bit math.
    */
   bool mediump_16bit_alu;

   /* When mediump_16bit_alu is set, determines whether nir_op_fddx/fddy can be
    * performed in 16-bit math.
    */
   bool mediump_16bit_derivatives;

   struct spirv_supported_capabilities caps;

   /* Address format for various kinds of pointers. */
   nir_address_format ubo_addr_format;
   nir_address_format ssbo_addr_format;
   nir_address_format phys_ssbo_addr_format;
   nir_address_format push_const_addr_format;
   nir_address_format shared_addr_format;
   nir_address_format task_payload_addr_format;
   nir_address_format global_addr_format;
   nir_address_format temp_addr_format;
   nir_address_format constant_addr_format;

   const nir_shader *clc_shader;

   struct {
      void (*func)(void *private_data,
                   enum nir_spirv_debug_level level,
                   size_t spirv_offset,
                   const char *message);
      void *private_data;
   } debug;

   /* Force texture sampling to be non-uniform. */
   bool force_tex_non_uniform;
};

bool gl_spirv_validation(const uint32_t *words, size_t word_count,
                         struct nir_spirv_specialization *spec, unsigned num_spec,
                         gl_shader_stage stage, const char *entry_point_name);

nir_shader *spirv_to_nir(const uint32_t *words, size_t word_count,
                         struct nir_spirv_specialization *specializations,
                         unsigned num_specializations,
                         gl_shader_stage stage, const char *entry_point_name,
                         const struct spirv_to_nir_options *options,
                         const nir_shader_compiler_options *nir_options);

bool nir_can_find_libclc(unsigned ptr_bit_size);

nir_shader *
nir_load_libclc_shader(unsigned ptr_bit_size,
                       struct disk_cache *disk_cache,
                       const struct spirv_to_nir_options *spirv_options,
                       const nir_shader_compiler_options *nir_options);

bool nir_lower_libclc(nir_shader *shader, const nir_shader *clc_shader);

#ifdef __cplusplus
}
#endif

#endif /* _NIR_SPIRV_H_ */
