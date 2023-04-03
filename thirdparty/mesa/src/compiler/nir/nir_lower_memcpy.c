/*
 * Copyright © 2020 Intel Corporation
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
 */

#include "nir_builder.h"

#include <string.h>

/** Returns the type to use for a copy of the given size.
 *
 * The actual type doesn't matter here all that much as we're just going to do
 * a load/store on it and never any arithmetic.
 */
static const struct glsl_type *
copy_type_for_byte_size(unsigned size)
{
   switch (size) {
   case 1:  return glsl_vector_type(GLSL_TYPE_UINT8, 1);
   case 2:  return glsl_vector_type(GLSL_TYPE_UINT16, 1);
   case 4:  return glsl_vector_type(GLSL_TYPE_UINT, 1);
   case 8:  return glsl_vector_type(GLSL_TYPE_UINT, 2);
   case 16: return glsl_vector_type(GLSL_TYPE_UINT, 4);
   default:
      unreachable("Unsupported size");
   }
}

static nir_ssa_def *
memcpy_load_deref_elem(nir_builder *b, nir_deref_instr *parent,
                       nir_ssa_def *index)
{
   nir_deref_instr *deref;

   index = nir_i2iN(b, index, nir_dest_bit_size(parent->dest));
   assert(parent->deref_type == nir_deref_type_cast);
   deref = nir_build_deref_ptr_as_array(b, parent, index);

   return nir_load_deref(b, deref);
}

static nir_ssa_def *
memcpy_load_deref_elem_imm(nir_builder *b, nir_deref_instr *parent,
                           uint64_t index)
{
   nir_ssa_def *idx = nir_imm_intN_t(b, index, parent->dest.ssa.bit_size);
   return memcpy_load_deref_elem(b, parent, idx);
}

static void
memcpy_store_deref_elem(nir_builder *b, nir_deref_instr *parent,
                        nir_ssa_def *index, nir_ssa_def *value)
{
   nir_deref_instr *deref;

   index = nir_i2iN(b, index, nir_dest_bit_size(parent->dest));
   assert(parent->deref_type == nir_deref_type_cast);
   deref = nir_build_deref_ptr_as_array(b, parent, index);
   nir_store_deref(b, deref, value, ~0);
}

static void
memcpy_store_deref_elem_imm(nir_builder *b, nir_deref_instr *parent,
                            uint64_t index, nir_ssa_def *value)
{
   nir_ssa_def *idx = nir_imm_intN_t(b, index, parent->dest.ssa.bit_size);
   memcpy_store_deref_elem(b, parent, idx, value);
}

static bool
lower_memcpy_impl(nir_function_impl *impl)
{
   nir_builder b;
   nir_builder_init(&b, impl);

   bool found_const_memcpy = false, found_non_const_memcpy = false;

   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *cpy = nir_instr_as_intrinsic(instr);
         if (cpy->intrinsic != nir_intrinsic_memcpy_deref)
            continue;

         b.cursor = nir_instr_remove(&cpy->instr);

         nir_deref_instr *dst = nir_src_as_deref(cpy->src[0]);
         nir_deref_instr *src = nir_src_as_deref(cpy->src[1]);
         if (nir_src_is_const(cpy->src[2])) {
            found_const_memcpy = true;
            uint64_t size = nir_src_as_uint(cpy->src[2]);
            uint64_t offset = 0;
            while (offset < size) {
               uint64_t remaining = size - offset;
               /* Find the largest chunk size power-of-two (MSB in remaining)
                * and limit our chunk to 16B (a vec4). It's important to do as
                * many 16B chunks as possible first so that the index
                * computation is correct for
                * memcpy_(load|store)_deref_elem_imm.
                */
               unsigned copy_size = 1u << MIN2(util_last_bit64(remaining) - 1, 4);
               const struct glsl_type *copy_type =
                  copy_type_for_byte_size(copy_size);

               nir_deref_instr *copy_dst =
                  nir_build_deref_cast(&b, &dst->dest.ssa, dst->modes,
                                       copy_type, copy_size);
               nir_deref_instr *copy_src =
                  nir_build_deref_cast(&b, &src->dest.ssa, src->modes,
                                       copy_type, copy_size);

               uint64_t index = offset / copy_size;
               nir_ssa_def *value =
                  memcpy_load_deref_elem_imm(&b, copy_src, index);
               memcpy_store_deref_elem_imm(&b, copy_dst, index, value);
               offset += copy_size;
            }
         } else {
            found_non_const_memcpy = true;
            assert(cpy->src[2].is_ssa);
            nir_ssa_def *size = cpy->src[2].ssa;

            /* In this case, we don't have any idea what the size is so we
             * emit a loop which copies one byte at a time.
             */
            nir_deref_instr *copy_dst =
               nir_build_deref_cast(&b, &dst->dest.ssa, dst->modes,
                                    glsl_uint8_t_type(), 1);
            nir_deref_instr *copy_src =
               nir_build_deref_cast(&b, &src->dest.ssa, src->modes,
                                    glsl_uint8_t_type(), 1);

            nir_variable *i = nir_local_variable_create(impl,
               glsl_uintN_t_type(size->bit_size), NULL);
            nir_store_var(&b, i, nir_imm_intN_t(&b, 0, size->bit_size), ~0);
            nir_push_loop(&b);
            {
               nir_ssa_def *index = nir_load_var(&b, i);
               nir_push_if(&b, nir_uge(&b, index, size));
               {
                  nir_jump(&b, nir_jump_break);
               }
               nir_pop_if(&b, NULL);

               nir_ssa_def *value =
                  memcpy_load_deref_elem(&b, copy_src, index);
               memcpy_store_deref_elem(&b, copy_dst, index, value);
               nir_store_var(&b, i, nir_iadd_imm(&b, index, 1),  ~0);
            }
            nir_pop_loop(&b, NULL);
         }
      }
   }

   if (found_non_const_memcpy) {
      nir_metadata_preserve(impl, nir_metadata_none);
   } else if (found_const_memcpy) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return found_const_memcpy || found_non_const_memcpy;
}

bool
nir_lower_memcpy(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl && lower_memcpy_impl(function->impl))
         progress = true;
   }

   return progress;
}
