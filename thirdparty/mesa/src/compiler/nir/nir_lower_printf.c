/*
 * Copyright © 2020 Microsoft Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"

#include "util/u_math.h"

static bool
lower_printf_instr(nir_builder *b, nir_instr *instr, void *_options)
{
   const nir_lower_printf_options *options = _options;
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *prntf = nir_instr_as_intrinsic(instr);
   if (prntf->intrinsic != nir_intrinsic_printf)
      return false;

   nir_ssa_def *fmt_str_id = prntf->src[0].ssa;
   nir_deref_instr *args = nir_src_as_deref(prntf->src[1]);
   assert(args->deref_type == nir_deref_type_var);

   const unsigned ptr_bit_size = nir_get_ptr_bitsize(b->shader);

   /* Atomic add a buffer size counter to determine where to write.  If
    * overflowed, return -1, otherwise, store the arguments and return 0.
    */
   b->cursor = nir_before_instr(&prntf->instr);
   nir_ssa_def *buffer_addr = nir_load_printf_buffer_address(b, ptr_bit_size);
   nir_deref_instr *buffer =
      nir_build_deref_cast(b, buffer_addr, nir_var_mem_global,
                           glsl_array_type(glsl_uint8_t_type(), 0, 4), 0);

   /* Align the struct size to 4 */
   assert(glsl_type_is_struct_or_ifc(args->type));
   int args_size = align(glsl_get_cl_size(args->type), 4);
   assert(fmt_str_id->bit_size == 32);
   int fmt_str_id_size = 4;

   /* Increment the counter at the beginning of the buffer */
   const unsigned counter_size = 4;
   nir_deref_instr *counter = nir_build_deref_array_imm(b, buffer, 0);
   counter = nir_build_deref_cast(b, &counter->dest.ssa,
                                  nir_var_mem_global,
                                  glsl_uint_type(), 0);
   counter->cast.align_mul = 4;
   nir_ssa_def *offset =
      nir_deref_atomic_add(b, 32, &counter->dest.ssa,
                           nir_imm_int(b, fmt_str_id_size + args_size));

   /* Check if we're still in-bounds */
   const unsigned default_buffer_size = 1024 * 1024;
   unsigned buffer_size = (options && options->max_buffer_size) ?
                          options->max_buffer_size : default_buffer_size;
   int max_valid_offset =
      buffer_size - args_size - fmt_str_id_size - counter_size;
   nir_push_if(b, nir_ilt(b, offset, nir_imm_int(b, max_valid_offset)));

   nir_ssa_def *printf_succ_val = nir_imm_int(b, 0);

   /* Write the format string ID */
   nir_ssa_def *fmt_str_id_offset =
      nir_i2iN(b, offset, ptr_bit_size);
   nir_deref_instr *fmt_str_id_deref =
      nir_build_deref_array(b, buffer, fmt_str_id_offset);
   fmt_str_id_deref = nir_build_deref_cast(b, &fmt_str_id_deref->dest.ssa,
                                           nir_var_mem_global,
                                           glsl_uint_type(), 0);
   fmt_str_id_deref->cast.align_mul = 4;
   nir_store_deref(b, fmt_str_id_deref, fmt_str_id, ~0);

   /* Write the format args */
   for (unsigned i = 0; i < glsl_get_length(args->type); ++i) {
      nir_deref_instr *arg_deref = nir_build_deref_struct(b, args, i);
      nir_ssa_def *arg = nir_load_deref(b, arg_deref);
      const struct glsl_type *arg_type = arg_deref->type;

      /* Clang does promotion of arguments to their "native" size. That means
       * that any floats have been converted to doubles for the call to
       * printf. Since doubles are optional, some drivers might not support
       * them. For those drivers, convert them back to float before writing.
       * Copy prop and other optimizations should remove all hints of doubles.
       */
      if (glsl_get_base_type(arg_type) == GLSL_TYPE_DOUBLE &&
          options && options->treat_doubles_as_floats) {
         arg = nir_f2f32(b, arg);
         arg_type = glsl_float_type();
      }

      unsigned field_offset = glsl_get_struct_field_offset(args->type, i);
      nir_ssa_def *arg_offset =
         nir_i2iN(b, nir_iadd_imm(b, offset,
                                 fmt_str_id_size + field_offset),
                 ptr_bit_size);
      nir_deref_instr *dst_arg_deref =
         nir_build_deref_array(b, buffer, arg_offset);
      dst_arg_deref = nir_build_deref_cast(b, &dst_arg_deref->dest.ssa,
                                           nir_var_mem_global, arg_type, 0);
      assert(field_offset % 4 == 0);
      dst_arg_deref->cast.align_mul = 4;
      nir_store_deref(b, dst_arg_deref, arg, ~0);
   }

   nir_push_else(b, NULL);
   nir_ssa_def *printf_fail_val = nir_imm_int(b, -1);
   nir_pop_if(b, NULL);

   nir_ssa_def *ret_val = nir_if_phi(b, printf_succ_val, printf_fail_val);
   nir_ssa_def_rewrite_uses(&prntf->dest.ssa, ret_val);
   nir_instr_remove(&prntf->instr);

   return true;
}

bool
nir_lower_printf(nir_shader *nir, const nir_lower_printf_options *options)
{
   return nir_shader_instructions_pass(nir, lower_printf_instr,
                                       nir_metadata_none,
                                       (void *)options);
}
