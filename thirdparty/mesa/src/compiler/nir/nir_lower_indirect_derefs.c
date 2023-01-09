/*
 * Copyright © 2016 Intel Corporation
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
#include "nir_deref.h"

static void
emit_load_store_deref(nir_builder *b, nir_intrinsic_instr *orig_instr,
                      nir_deref_instr *parent,
                      nir_deref_instr **deref_arr,
                      nir_ssa_def **dest, nir_ssa_def *src);

static void
emit_indirect_load_store_deref(nir_builder *b, nir_intrinsic_instr *orig_instr,
                               nir_deref_instr *parent,
                               nir_deref_instr **deref_arr,
                               int start, int end,
                               nir_ssa_def **dest, nir_ssa_def *src)
{
   assert(start < end);
   if (start == end - 1) {
      emit_load_store_deref(b, orig_instr,
                            nir_build_deref_array_imm(b, parent, start),
                            deref_arr + 1, dest, src);
   } else {
      int mid = start + (end - start) / 2;

      nir_ssa_def *then_dest, *else_dest;

      nir_deref_instr *deref = *deref_arr;
      assert(deref->deref_type == nir_deref_type_array);

      nir_push_if(b, nir_ilt(b, deref->arr.index.ssa, nir_imm_intN_t(b, mid, parent->dest.ssa.bit_size)));
      emit_indirect_load_store_deref(b, orig_instr, parent, deref_arr,
                                     start, mid, &then_dest, src);
      nir_push_else(b, NULL);
      emit_indirect_load_store_deref(b, orig_instr, parent, deref_arr,
                                     mid, end, &else_dest, src);
      nir_pop_if(b, NULL);

      if (src == NULL)
         *dest = nir_if_phi(b, then_dest, else_dest);
   }
}

static void
emit_load_store_deref(nir_builder *b, nir_intrinsic_instr *orig_instr,
                      nir_deref_instr *parent,
                      nir_deref_instr **deref_arr,
                      nir_ssa_def **dest, nir_ssa_def *src)
{
   for (; *deref_arr; deref_arr++) {
      nir_deref_instr *deref = *deref_arr;
      if (deref->deref_type == nir_deref_type_array &&
          !nir_src_is_const(deref->arr.index)) {
         int length = glsl_get_length(parent->type);

         emit_indirect_load_store_deref(b, orig_instr, parent, deref_arr,
                                        0, length, dest, src);
         return;
      }

      parent = nir_build_deref_follower(b, parent, deref);
   }

   /* We reached the end of the deref chain.  Emit the instruction */
   assert(*deref_arr == NULL);

   if (src == NULL) {
      /* This is a load instruction */
      nir_intrinsic_instr *load =
         nir_intrinsic_instr_create(b->shader, orig_instr->intrinsic);
      load->num_components = orig_instr->num_components;

      load->src[0] = nir_src_for_ssa(&parent->dest.ssa);

      /* Copy over any other sources.  This is needed for interp_deref_at */
      for (unsigned i = 1;
           i < nir_intrinsic_infos[orig_instr->intrinsic].num_srcs; i++)
         nir_src_copy(&load->src[i], &orig_instr->src[i], &load->instr);

      nir_ssa_dest_init(&load->instr, &load->dest,
                        orig_instr->dest.ssa.num_components,
                        orig_instr->dest.ssa.bit_size, NULL);
      nir_builder_instr_insert(b, &load->instr);
      *dest = &load->dest.ssa;
   } else {
      assert(orig_instr->intrinsic == nir_intrinsic_store_deref);
      nir_store_deref(b, parent, src, nir_intrinsic_write_mask(orig_instr));
   }
}

static bool
lower_indirect_derefs_block(nir_block *block, nir_builder *b,
                            nir_variable_mode modes,
                            const struct set *vars,
                            uint32_t max_lower_array_len)
{
   bool progress = false;

   nir_foreach_instr_safe(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      if (intrin->intrinsic != nir_intrinsic_load_deref &&
          intrin->intrinsic != nir_intrinsic_interp_deref_at_centroid &&
          intrin->intrinsic != nir_intrinsic_interp_deref_at_sample &&
          intrin->intrinsic != nir_intrinsic_interp_deref_at_offset &&
          intrin->intrinsic != nir_intrinsic_interp_deref_at_vertex &&
          intrin->intrinsic != nir_intrinsic_store_deref)
         continue;

      nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);

      /* Walk the deref chain back to the base and look for indirects */
      uint32_t indirect_array_len = 1;
      bool has_indirect = false;
      nir_deref_instr *base = deref;
      while (base && base->deref_type != nir_deref_type_var) {
         nir_deref_instr *parent = nir_deref_instr_parent(base);
         if (base->deref_type == nir_deref_type_array &&
             !nir_src_is_const(base->arr.index)) {
            indirect_array_len *= glsl_get_length(parent->type);
            has_indirect = true;
         }

         base = parent;
      }

      if (!has_indirect || !base || indirect_array_len > max_lower_array_len)
         continue;

      /* Only lower variables whose mode is in the mask, or compact
       * array variables.  (We can't handle indirects on tightly packed
       * scalar arrays, so we need to lower them regardless.)
       */
      if (!(modes & base->var->data.mode) && !base->var->data.compact)
         continue;

      if (vars && !_mesa_set_search(vars, base->var))
         continue;

      b->cursor = nir_instr_remove(&intrin->instr);

      nir_deref_path path;
      nir_deref_path_init(&path, deref, NULL);
      assert(path.path[0] == base);

      if (intrin->intrinsic == nir_intrinsic_store_deref) {
         assert(intrin->src[1].is_ssa);
         emit_load_store_deref(b, intrin, base, &path.path[1],
                               NULL, intrin->src[1].ssa);
      } else {
         nir_ssa_def *result;
         emit_load_store_deref(b, intrin, base, &path.path[1],
                               &result, NULL);
         nir_ssa_def_rewrite_uses(&intrin->dest.ssa, result);
      }

      nir_deref_path_finish(&path);

      progress = true;
   }

   return progress;
}

static bool
lower_indirects_impl(nir_function_impl *impl, nir_variable_mode modes,
                     const struct set *vars, uint32_t max_lower_array_len)
{
   nir_builder builder;
   nir_builder_init(&builder, impl);
   bool progress = false;

   nir_foreach_block_safe(block, impl) {
      progress |= lower_indirect_derefs_block(block, &builder, modes, vars,
                                              max_lower_array_len);
   }

   if (progress)
      nir_metadata_preserve(impl, nir_metadata_none);
   else
      nir_metadata_preserve(impl, nir_metadata_all);

   return progress;
}

/** Lowers indirect variable loads/stores to direct loads/stores.
 *
 * The pass works by replacing any indirect load or store with an if-ladder
 * that does a binary search on the array index.
 */
bool
nir_lower_indirect_derefs(nir_shader *shader, nir_variable_mode modes,
                          uint32_t max_lower_array_len)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl) {
         progress = lower_indirects_impl(function->impl, modes, NULL,
                                         max_lower_array_len) || progress;
      }
   }

   return progress;
}

/** Lowers indirects on any variables in the given set */
bool
nir_lower_indirect_var_derefs(nir_shader *shader, const struct set *vars)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl) {
         progress = lower_indirects_impl(function->impl, nir_var_uniform,
                                         vars, UINT_MAX) || progress;
      }
   }

   return progress;
}
