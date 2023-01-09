/*
 * Copyright (C) 2005-2007  Brian Paul   All Rights Reserved.
 * Copyright (C) 2008  VMware, Inc.   All Rights Reserved.
 * Copyright © 2014 Intel Corporation
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

#include "nir/nir.h"
#include "nir_builder.h"

static void
lower_tex_src_to_offset(nir_builder *b,
                        nir_tex_instr *instr, unsigned src_idx)
{
   nir_ssa_def *index = NULL;
   unsigned base_index = 0;
   unsigned array_elements = 1;
   nir_tex_src *src = &instr->src[src_idx];
   bool is_sampler = src->src_type == nir_tex_src_sampler_deref;

   /* We compute first the offsets */
   nir_deref_instr *deref = nir_instr_as_deref(src->src.ssa->parent_instr);
   while (deref->deref_type != nir_deref_type_var) {
      assert(deref->parent.is_ssa);
      nir_deref_instr *parent =
         nir_instr_as_deref(deref->parent.ssa->parent_instr);

      assert(deref->deref_type == nir_deref_type_array);

      if (nir_src_is_const(deref->arr.index) && index == NULL) {
         /* We're still building a direct index */
         unsigned index_in_array = nir_src_as_uint(deref->arr.index);

         /* Section 5.11 (Out-of-Bounds Accesses) of the GLSL 4.60 spec says:
          *
          *    In the subsections described above for array, vector, matrix and
          *    structure accesses, any out-of-bounds access produced undefined
          *    behavior.... Out-of-bounds reads return undefined values, which
          *    include values from other variables of the active program or zero.
          *
          * Robustness extensions suggest to return zero on out-of-bounds
          * accesses, however it's not applicable to the arrays of samplers,
          * so just clamp the index.
          *
          * Otherwise instr->sampler_index or instr->texture_index would be out
          * of bounds, and they are used as an index to arrays of driver state.
          */
         if (index_in_array < glsl_array_size(parent->type)) {
            base_index += index_in_array * array_elements;
         } else {
            base_index = glsl_array_size(parent->type) - 1;
         }
      } else {
         if (index == NULL) {
            /* We used to be direct but not anymore */
            index = nir_imm_int(b, base_index);
            base_index = 0;
         }

         index = nir_iadd(b, index,
                          nir_imul(b, nir_imm_int(b, array_elements),
                                   nir_ssa_for_src(b, deref->arr.index, 1)));
      }

      array_elements *= glsl_get_length(parent->type);

      deref = parent;
   }

   if (index)
      index = nir_umin(b, index, nir_imm_int(b, array_elements - 1));

   /* We hit the deref_var.  This is the end of the line */
   assert(deref->deref_type == nir_deref_type_var);

   base_index += deref->var->data.binding;

   /* We have the offsets, we apply them, rewriting the source or removing
    * instr if needed
    */
   if (index) {
      nir_instr_rewrite_src(&instr->instr, &src->src,
                            nir_src_for_ssa(index));

      src->src_type = is_sampler ?
         nir_tex_src_sampler_offset :
         nir_tex_src_texture_offset;
   } else {
      nir_tex_instr_remove_src(instr, src_idx);
   }

   if (is_sampler) {
      instr->sampler_index = base_index;
   } else {
      instr->texture_index = base_index;
   }
}

static bool
lower_sampler(nir_builder *b, nir_instr *instr_, UNUSED void *cb_data)
{
   if (instr_->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *instr = nir_instr_as_tex(instr_);

   int texture_idx =
      nir_tex_instr_src_index(instr, nir_tex_src_texture_deref);

   if (texture_idx >= 0) {
      b->cursor = nir_before_instr(&instr->instr);

      lower_tex_src_to_offset(b, instr, texture_idx);
   }

   int sampler_idx =
      nir_tex_instr_src_index(instr, nir_tex_src_sampler_deref);

   if (sampler_idx >= 0) {
      lower_tex_src_to_offset(b, instr, sampler_idx);
   }

   if (texture_idx < 0 && sampler_idx < 0)
      return false;

   return true;
}

bool
nir_lower_samplers(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, lower_sampler,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       NULL);
}
