/*
 * Copyright © 2019 Google, Inc.
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "nir.h"
#include "nir_vla.h"

/* Lowering for amul instructions, for drivers that support imul24.
 * This pass will analyze indirect derefs, and convert corresponding
 * amul instructions to either imul or imul24, depending on the
 * required range.
 *
 * 1) Analyze the uniform variables and build a table of UBOs and SSBOs
 *    that are either too large, or might be too large (unknown size)
 *    for imul24
 *
 * 2) Loop thru looking at all the intrinsics, finding dereferences of
 *    large variables, and recursively replacing all amul instructions
 *    used with imul
 *
 * 3) Finally loop again thru all instructions replacing any remaining
 *    amul with imul24.  At this point any remaining amul instructions
 *    are not involved in calculating an offset into a large variable,
 *    thanks to the 2nd step, so they can be safely replace with imul24.
 *
 * Using two passes over all the instructions lets us handle the case
 * where, due to CSE, an amul is used to calculate an offset into both
 * a large and small variable.
 */

typedef struct {
   nir_shader *shader;

   int (*type_size)(const struct glsl_type *, bool);

   /* Tables of UBOs and SSBOs mapping driver_location/base whether
    * they are too large to use imul24:
    */
   bool *large_ubos;
   bool *large_ssbos;

   /* for cases that we cannot determine UBO/SSBO index, track if *any*
    * UBO/SSBO is too large for imul24:
    */
   bool has_large_ubo;
   bool has_large_ssbo;

   unsigned max_slot;

   bool progress;
} lower_state;

/* Lower 'amul's in offset src of large variables to 'imul': */
static bool
lower_large_src(nir_src *src, void *s)
{
   lower_state *state = s;

   assert(src->is_ssa);

   nir_instr *parent = src->ssa->parent_instr;

   /* No need to visit instructions we've already visited.. this also
    * avoids infinite recursion when phi's are involved:
    */
   if (parent->pass_flags)
      return false;

   nir_foreach_src(parent, lower_large_src, state);

   if (parent->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(parent);
      if (alu->op == nir_op_amul) {
         alu->op = nir_op_imul;
         state->progress = true;
      }
   }

   parent->pass_flags = 1;

   return true;
}

static bool
large_ubo(lower_state *state, nir_src src)
{
   if (!nir_src_is_const(src))
      return state->has_large_ubo;
   unsigned idx = nir_src_as_uint(src);
   assert(idx < state->shader->info.num_ubos);
   return state->large_ubos[idx];
}

static bool
large_ssbo(lower_state *state, nir_src src)
{
   if (!nir_src_is_const(src))
      return state->has_large_ssbo;
   unsigned idx = nir_src_as_uint(src);
   assert(idx < state->shader->info.num_ssbos);
   return state->large_ssbos[idx];
}

static void
lower_intrinsic(lower_state *state, nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ubo:
      //# src[] = { buffer_index, offset }.
      if (large_ubo(state, intr->src[0]))
         lower_large_src(&intr->src[1], state);
      return;

   case nir_intrinsic_load_ssbo:
      //# src[] = { buffer_index, offset }.
      if (large_ssbo(state, intr->src[0]))
         lower_large_src(&intr->src[1], state);
      return;

   case nir_intrinsic_store_ssbo:
      //# src[] = { value, block_index, offset }
      if (large_ssbo(state, intr->src[1]))
         lower_large_src(&intr->src[2], state);
      return;

   case nir_intrinsic_ssbo_atomic_add:
   case nir_intrinsic_ssbo_atomic_imin:
   case nir_intrinsic_ssbo_atomic_umin:
   case nir_intrinsic_ssbo_atomic_imax:
   case nir_intrinsic_ssbo_atomic_umax:
   case nir_intrinsic_ssbo_atomic_and:
   case nir_intrinsic_ssbo_atomic_or:
   case nir_intrinsic_ssbo_atomic_xor:
   case nir_intrinsic_ssbo_atomic_exchange:
   case nir_intrinsic_ssbo_atomic_comp_swap:
   case nir_intrinsic_ssbo_atomic_fadd:
   case nir_intrinsic_ssbo_atomic_fmin:
   case nir_intrinsic_ssbo_atomic_fmax:
   case nir_intrinsic_ssbo_atomic_fcomp_swap:
      /* 0: SSBO index
       * 1: offset
       */
      if (large_ssbo(state, intr->src[0]))
         lower_large_src(&intr->src[1], state);
      return;

   case nir_intrinsic_global_atomic_add:
   case nir_intrinsic_global_atomic_imin:
   case nir_intrinsic_global_atomic_umin:
   case nir_intrinsic_global_atomic_imax:
   case nir_intrinsic_global_atomic_umax:
   case nir_intrinsic_global_atomic_and:
   case nir_intrinsic_global_atomic_or:
   case nir_intrinsic_global_atomic_xor:
   case nir_intrinsic_global_atomic_exchange:
   case nir_intrinsic_global_atomic_comp_swap:
   case nir_intrinsic_global_atomic_fadd:
   case nir_intrinsic_global_atomic_fmin:
   case nir_intrinsic_global_atomic_fmax:
   case nir_intrinsic_global_atomic_fcomp_swap:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_load_global:
      /* just assume we that 24b is not sufficient: */
      lower_large_src(&intr->src[0], state);
      return;

   case nir_intrinsic_store_global:
      /* just assume we that 24b is not sufficient: */
      lower_large_src(&intr->src[1], state);
      return;

   /* These should all be small enough to unconditionally use imul24: */
   case nir_intrinsic_shared_atomic_add:
   case nir_intrinsic_shared_atomic_imin:
   case nir_intrinsic_shared_atomic_umin:
   case nir_intrinsic_shared_atomic_imax:
   case nir_intrinsic_shared_atomic_umax:
   case nir_intrinsic_shared_atomic_and:
   case nir_intrinsic_shared_atomic_or:
   case nir_intrinsic_shared_atomic_xor:
   case nir_intrinsic_shared_atomic_exchange:
   case nir_intrinsic_shared_atomic_comp_swap:
   case nir_intrinsic_shared_atomic_fadd:
   case nir_intrinsic_shared_atomic_fmin:
   case nir_intrinsic_shared_atomic_fmax:
   case nir_intrinsic_shared_atomic_fcomp_swap:
   case nir_intrinsic_load_uniform:
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_output:
   case nir_intrinsic_store_output:
   default:
      return;
   }
}

static void
lower_instr(lower_state *state, nir_instr *instr)
{
   if (instr->type == nir_instr_type_intrinsic) {
      lower_intrinsic(state, nir_instr_as_intrinsic(instr));
   }
}

static bool
is_large(lower_state *state, nir_variable *var)
{
   const struct glsl_type *type = glsl_without_array(var->type);
   unsigned size = state->type_size(type, false);

   /* if size is not known (ie. VLA) then assume the worst: */
   if (!size)
      return true;

   return size >= (1 << 23);
}

bool
nir_lower_amul(nir_shader *shader,
               int (*type_size)(const struct glsl_type *, bool))
{
   assert(shader->options->has_imul24);
   assert(type_size);

   NIR_VLA_FILL(bool, large_ubos, shader->info.num_ubos, 0);
   NIR_VLA_FILL(bool, large_ssbos, shader->info.num_ssbos, 0);

   lower_state state = {
      .shader = shader,
      .type_size = type_size,
      .large_ubos = large_ubos,
      .large_ssbos = large_ssbos,
   };

   /* Figure out which UBOs or SSBOs are large enough to be
    * disqualified from imul24:
    */
   nir_foreach_variable_in_shader (var, shader) {
      if (var->data.mode == nir_var_mem_ubo) {
         if (is_large(&state, var)) {
            state.has_large_ubo = true;
            unsigned size = MAX2(1, glsl_array_size(var->type));
            for (unsigned i = 0; i < size; i++)
               state.large_ubos[var->data.binding + i] = true;
         }
      } else if (var->data.mode == nir_var_mem_ssbo) {
         if (is_large(&state, var)) {
            state.has_large_ssbo = true;
            unsigned size = MAX2(1, glsl_array_size(var->type));
            for (unsigned i = 0; i < size; i++)
               state.large_ssbos[var->data.binding + i] = true;
         }
      }
   }

   /* clear pass flags: */
   nir_foreach_function(function, shader) {
      nir_function_impl *impl = function->impl;
      if (!impl)
         continue;

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            instr->pass_flags = 0;
         }
      }
   }

   nir_foreach_function(function, shader) {
      nir_function_impl *impl = function->impl;

      if (!impl)
         continue;

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            lower_instr(&state, instr);
         }
      }
   }

   /* At this point, all 'amul's used in calculating an offset into
    * a large variable have been replaced with 'imul'.  So remaining
    * 'amul's can be replaced with 'imul24':
    *
    * Note the exception for 64b (such as load/store_global where
    * address size is 64b) as imul24 cannot have 64b bitsize
    */
   nir_foreach_function(function, shader) {
      nir_function_impl *impl = function->impl;

      if (!impl)
         continue;

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;

            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op != nir_op_amul)
               continue;

            if (nir_dest_bit_size(alu->dest.dest) <= 32)
               alu->op = nir_op_imul24;
            else
               alu->op = nir_op_imul;

            state.progress |= true;
         }
      }

      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);

   }

   return state.progress;
}
