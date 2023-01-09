/*
 * Copyright © 2019 Intel Corporation
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

#include "nir.h"
#include "nir_builder.h"

static void
build_write_masked_store(nir_builder *b, nir_deref_instr *vec_deref,
                         nir_ssa_def *value, unsigned component)
{
   assert(value->num_components == 1);
   unsigned num_components = glsl_get_components(vec_deref->type);
   assert(num_components > 1 && num_components <= NIR_MAX_VEC_COMPONENTS);

   nir_ssa_def *u = nir_ssa_undef(b, 1, value->bit_size);
   nir_ssa_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < num_components; i++)
      comps[i] = (i == component) ? value : u;

   nir_ssa_def *vec = nir_vec(b, comps, num_components);
   nir_store_deref(b, vec_deref, vec, (1u << component));
}

static void
build_write_masked_stores(nir_builder *b, nir_deref_instr *vec_deref,
                          nir_ssa_def *value, nir_ssa_def *index,
                          unsigned start, unsigned end)
{
   if (start == end - 1) {
      build_write_masked_store(b, vec_deref, value, start);
   } else {
      unsigned mid = start + (end - start) / 2;
      nir_push_if(b, nir_ilt(b, index, nir_imm_int(b, mid)));
      build_write_masked_stores(b, vec_deref, value, index, start, mid);
      nir_push_else(b, NULL);
      build_write_masked_stores(b, vec_deref, value, index, mid, end);
      nir_pop_if(b, NULL);
   }
}

static bool
nir_lower_array_deref_of_vec_impl(nir_function_impl *impl,
                                  nir_variable_mode modes,
                                  nir_lower_array_deref_of_vec_options options)
{
   bool progress = false;

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         assert(intrin->intrinsic != nir_intrinsic_copy_deref);

         if (intrin->intrinsic != nir_intrinsic_load_deref &&
             intrin->intrinsic != nir_intrinsic_interp_deref_at_centroid &&
             intrin->intrinsic != nir_intrinsic_interp_deref_at_sample &&
             intrin->intrinsic != nir_intrinsic_interp_deref_at_offset &&
             intrin->intrinsic != nir_intrinsic_interp_deref_at_vertex &&
             intrin->intrinsic != nir_intrinsic_store_deref)
            continue;

         nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);

         /* We choose to be conservative here.  If the deref contains any
          * modes which weren't specified, we bail and don't bother lowering.
          */
         if (!nir_deref_mode_must_be(deref, modes))
            continue;

         /* We only care about array derefs that act on vectors */
         if (deref->deref_type != nir_deref_type_array)
            continue;

         nir_deref_instr *vec_deref = nir_deref_instr_parent(deref);
         if (!glsl_type_is_vector(vec_deref->type))
            continue;

         assert(intrin->num_components == 1);
         unsigned num_components = glsl_get_components(vec_deref->type);
         assert(num_components > 1 && num_components <= NIR_MAX_VEC_COMPONENTS);

         b.cursor = nir_after_instr(&intrin->instr);

         if (intrin->intrinsic == nir_intrinsic_store_deref) {
            assert(intrin->src[1].is_ssa);
            nir_ssa_def *value = intrin->src[1].ssa;

            if (nir_src_is_const(deref->arr.index)) {
               if (!(options & nir_lower_direct_array_deref_of_vec_store))
                  continue;

               unsigned index = nir_src_as_uint(deref->arr.index);
               /* If index is OOB, we throw the old store away and don't
                * replace it with anything.
                */
               if (index < num_components)
                  build_write_masked_store(&b, vec_deref, value, index);
            } else {
               if (!(options & nir_lower_indirect_array_deref_of_vec_store))
                  continue;

               nir_ssa_def *index = nir_ssa_for_src(&b, deref->arr.index, 1);
               build_write_masked_stores(&b, vec_deref, value, index,
                                         0, num_components);
            }
            nir_instr_remove(&intrin->instr);

            progress = true;
         } else {
            if (nir_src_is_const(deref->arr.index)) {
               if (!(options & nir_lower_direct_array_deref_of_vec_load))
                  continue;
            } else {
               if (!(options & nir_lower_indirect_array_deref_of_vec_load))
                  continue;
            }

            /* Turn the load into a vector load */
            nir_instr_rewrite_src(&intrin->instr, &intrin->src[0],
                                  nir_src_for_ssa(&vec_deref->dest.ssa));
            intrin->dest.ssa.num_components = num_components;
            intrin->num_components = num_components;

            nir_ssa_def *index = nir_ssa_for_src(&b, deref->arr.index, 1);
            nir_ssa_def *scalar =
               nir_vector_extract(&b, &intrin->dest.ssa, index);
            if (scalar->parent_instr->type == nir_instr_type_ssa_undef) {
               nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                                        scalar);
               nir_instr_remove(&intrin->instr);
            } else {
               nir_ssa_def_rewrite_uses_after(&intrin->dest.ssa,
                                              scalar,
                                              scalar->parent_instr);
            }
            progress = true;
         }
      }
   }

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

/* Lowers away array dereferences on vectors
 *
 * These are allowed on certain variable types such as SSBOs and TCS outputs.
 * However, not everyone can actually handle them everywhere.  There are also
 * cases where we want to lower them for performance reasons.
 *
 * This patch assumes that copy_deref instructions have already been lowered.
 */
bool
nir_lower_array_deref_of_vec(nir_shader *shader, nir_variable_mode modes,
                             nir_lower_array_deref_of_vec_options options)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl &&
          nir_lower_array_deref_of_vec_impl(function->impl, modes, options))
         progress = true;
   }

   return progress;
}
