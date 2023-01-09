/*
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Jason Ekstrand (jason@jlekstrand.net)
 *
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_constant_expressions.h"
#include "nir_deref.h"
#include <math.h>

/*
 * Implements SSA-based constant folding.
 */

struct constant_fold_state {
   bool has_load_constant;
   bool has_indirect_load_const;
};

static bool
try_fold_alu(nir_builder *b, nir_alu_instr *alu)
{
   nir_const_value src[NIR_MAX_VEC_COMPONENTS][NIR_MAX_VEC_COMPONENTS];

   if (!alu->dest.dest.is_ssa)
      return false;

   /* In the case that any outputs/inputs have unsized types, then we need to
    * guess the bit-size. In this case, the validator ensures that all
    * bit-sizes match so we can just take the bit-size from first
    * output/input with an unsized type. If all the outputs/inputs are sized
    * then we don't need to guess the bit-size at all because the code we
    * generate for constant opcodes in this case already knows the sizes of
    * the types involved and does not need the provided bit-size for anything
    * (although it still requires to receive a valid bit-size).
    */
   unsigned bit_size = 0;
   if (!nir_alu_type_get_type_size(nir_op_infos[alu->op].output_type))
      bit_size = alu->dest.dest.ssa.bit_size;

   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      if (!alu->src[i].src.is_ssa)
         return false;

      if (bit_size == 0 &&
          !nir_alu_type_get_type_size(nir_op_infos[alu->op].input_types[i]))
         bit_size = alu->src[i].src.ssa->bit_size;

      nir_instr *src_instr = alu->src[i].src.ssa->parent_instr;

      if (src_instr->type != nir_instr_type_load_const)
         return false;
      nir_load_const_instr* load_const = nir_instr_as_load_const(src_instr);

      for (unsigned j = 0; j < nir_ssa_alu_instr_src_components(alu, i);
           j++) {
         src[i][j] = load_const->value[alu->src[i].swizzle[j]];
      }

      /* We shouldn't have any source modifiers in the optimization loop. */
      assert(!alu->src[i].abs && !alu->src[i].negate);
   }

   if (bit_size == 0)
      bit_size = 32;

   /* We shouldn't have any saturate modifiers in the optimization loop. */
   assert(!alu->dest.saturate);

   nir_const_value dest[NIR_MAX_VEC_COMPONENTS];
   nir_const_value *srcs[NIR_MAX_VEC_COMPONENTS];
   memset(dest, 0, sizeof(dest));
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
      srcs[i] = src[i];
   nir_eval_const_opcode(alu->op, dest, alu->dest.dest.ssa.num_components,
                         bit_size, srcs,
                         b->shader->info.float_controls_execution_mode);

   b->cursor = nir_before_instr(&alu->instr);
   nir_ssa_def *imm = nir_build_imm(b, alu->dest.dest.ssa.num_components,
                                       alu->dest.dest.ssa.bit_size,
                                       dest);
   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, imm);
   nir_instr_remove(&alu->instr);
   nir_instr_free(&alu->instr);

   return true;
}

static nir_const_value *
const_value_for_deref(nir_deref_instr *deref)
{
   if (!nir_deref_mode_is(deref, nir_var_mem_constant))
      return NULL;

   nir_deref_path path;
   nir_deref_path_init(&path, deref, NULL);
   if (path.path[0]->deref_type != nir_deref_type_var)
      goto fail;

   nir_variable *var = path.path[0]->var;
   assert(var->data.mode == nir_var_mem_constant);
   if (var->constant_initializer == NULL)
      goto fail;

   nir_constant *c = var->constant_initializer;
   nir_const_value *v = NULL; /* Vector value for array-deref-of-vec */

   for (unsigned i = 1; path.path[i] != NULL; i++) {
      nir_deref_instr *p = path.path[i];
      switch (p->deref_type) {
      case nir_deref_type_var:
         unreachable("Deref paths can only start with a var deref");

      case nir_deref_type_array: {
         assert(v == NULL);
         if (!nir_src_is_const(p->arr.index))
            goto fail;

         uint64_t idx = nir_src_as_uint(p->arr.index);
         if (c->num_elements > 0) {
            assert(glsl_type_is_array(path.path[i-1]->type));
            if (idx >= c->num_elements)
               goto fail;
            c = c->elements[idx];
         } else {
            assert(glsl_type_is_vector(path.path[i-1]->type));
            assert(glsl_type_is_scalar(p->type));
            if (idx >= NIR_MAX_VEC_COMPONENTS)
               goto fail;
            v = &c->values[idx];
         }
         break;
      }

      case nir_deref_type_struct:
         assert(glsl_type_is_struct(path.path[i-1]->type));
         assert(v == NULL && c->num_elements > 0);
         if (p->strct.index >= c->num_elements)
            goto fail;
         c = c->elements[p->strct.index];
         break;

      default:
         goto fail;
      }
   }

   /* We have to have ended at a vector */
   assert(c->num_elements == 0);
   return v ? v : c->values;

fail:
   nir_deref_path_finish(&path);
   return NULL;
}

static bool
try_fold_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin,
                   struct constant_fold_state *state)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_demote_if:
   case nir_intrinsic_discard_if:
   case nir_intrinsic_terminate_if:
      if (nir_src_is_const(intrin->src[0])) {
         if (nir_src_as_bool(intrin->src[0])) {
            b->cursor = nir_before_instr(&intrin->instr);
            nir_intrinsic_op op;
            switch (intrin->intrinsic) {
            case nir_intrinsic_discard_if:
               op = nir_intrinsic_discard;
               break;
            case nir_intrinsic_demote_if:
               op = nir_intrinsic_demote;
               break;
            case nir_intrinsic_terminate_if:
               op = nir_intrinsic_terminate;
               break;
            default:
               unreachable("invalid intrinsic");
            }
            nir_intrinsic_instr *new_instr =
               nir_intrinsic_instr_create(b->shader, op);
            nir_builder_instr_insert(b, &new_instr->instr);
         }
         nir_instr_remove(&intrin->instr);
         return true;
      }
      return false;

   case nir_intrinsic_load_deref: {
      nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
      nir_const_value *v = const_value_for_deref(deref);
      if (v) {
         b->cursor = nir_before_instr(&intrin->instr);
         nir_ssa_def *val = nir_build_imm(b, intrin->dest.ssa.num_components,
                                             intrin->dest.ssa.bit_size, v);
         nir_ssa_def_rewrite_uses(&intrin->dest.ssa, val);
         nir_instr_remove(&intrin->instr);
         return true;
      }
      return false;
   }

   case nir_intrinsic_load_constant: {
      state->has_load_constant = true;

      if (!nir_src_is_const(intrin->src[0])) {
         state->has_indirect_load_const = true;
         return false;
      }

      unsigned offset = nir_src_as_uint(intrin->src[0]);
      unsigned base = nir_intrinsic_base(intrin);
      unsigned range = nir_intrinsic_range(intrin);
      assert(base + range <= b->shader->constant_data_size);

      b->cursor = nir_before_instr(&intrin->instr);
      nir_ssa_def *val;
      if (offset >= range) {
         val = nir_ssa_undef(b, intrin->dest.ssa.num_components,
                                intrin->dest.ssa.bit_size);
      } else {
         nir_const_value imm[NIR_MAX_VEC_COMPONENTS];
         memset(imm, 0, sizeof(imm));
         uint8_t *data = (uint8_t*)b->shader->constant_data + base;
         for (unsigned i = 0; i < intrin->num_components; i++) {
            unsigned bytes = intrin->dest.ssa.bit_size / 8;
            bytes = MIN2(bytes, range - offset);

            memcpy(&imm[i].u64, data + offset, bytes);
            offset += bytes;
         }
         val = nir_build_imm(b, intrin->dest.ssa.num_components,
                                intrin->dest.ssa.bit_size, imm);
      }
      nir_ssa_def_rewrite_uses(&intrin->dest.ssa, val);
      nir_instr_remove(&intrin->instr);
      return true;
   }

   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_quad_swizzle_amd:
   case nir_intrinsic_masked_swizzle_amd:
      /* All of these have the data payload in the first source.  They may
       * have a second source with a shuffle index but that doesn't matter if
       * the data is constant.
       */
      if (nir_src_is_const(intrin->src[0])) {
         nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                                  intrin->src[0].ssa);
         nir_instr_remove(&intrin->instr);
         return true;
      }
      return false;

   case nir_intrinsic_vote_feq:
   case nir_intrinsic_vote_ieq:
      if (nir_src_is_const(intrin->src[0])) {
         b->cursor = nir_before_instr(&intrin->instr);
         nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                                  nir_imm_true(b));
         nir_instr_remove(&intrin->instr);
         return true;
      }
      return false;

   default:
      return false;
   }
}

static bool
try_fold_txb_to_tex(nir_builder *b, nir_tex_instr *tex)
{
   assert(tex->op == nir_texop_txb);

   const int bias_idx = nir_tex_instr_src_index(tex, nir_tex_src_bias);

   /* nir_to_tgsi_lower_tex mangles many kinds of texture instructions,
    * including txb, into invalid states.  It removes the special
    * parameters and appends the values to the texture coordinate.
    */
   if (bias_idx < 0)
      return false;

   if (nir_src_is_const(tex->src[bias_idx].src) &&
       nir_src_as_float(tex->src[bias_idx].src) == 0.0) {
      nir_tex_instr_remove_src(tex, bias_idx);
      tex->op = nir_texop_tex;
      return true;
   }

   return false;
}

static bool
try_fold_tex_offset(nir_tex_instr *tex, unsigned *index,
                    nir_tex_src_type src_type)
{
   const int src_idx = nir_tex_instr_src_index(tex, src_type);
   if (src_idx < 0)
      return false;

   if (!nir_src_is_const(tex->src[src_idx].src))
      return false;

   *index += nir_src_as_uint(tex->src[src_idx].src);
   nir_tex_instr_remove_src(tex, src_idx);

   return true;
}

static bool
try_fold_tex(nir_builder *b, nir_tex_instr *tex)
{
   bool progress = false;

   progress |= try_fold_tex_offset(tex, &tex->texture_index,
                                   nir_tex_src_texture_offset);
   progress |= try_fold_tex_offset(tex, &tex->sampler_index,
                                   nir_tex_src_sampler_offset);

   /* txb with a bias of constant zero is just tex. */
   if (tex->op == nir_texop_txb)
      progress |= try_fold_txb_to_tex(b, tex);

   return progress;
}

static bool
try_fold_instr(nir_builder *b, nir_instr *instr, void *_state)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      return try_fold_alu(b, nir_instr_as_alu(instr));
   case nir_instr_type_intrinsic:
      return try_fold_intrinsic(b, nir_instr_as_intrinsic(instr), _state);
   case nir_instr_type_tex:
      return try_fold_tex(b, nir_instr_as_tex(instr));
   default:
      /* Don't know how to constant fold */
      return false;
   }
}

bool
nir_opt_constant_folding(nir_shader *shader)
{
   struct constant_fold_state state;
   state.has_load_constant = false;
   state.has_indirect_load_const = false;

   bool progress = nir_shader_instructions_pass(shader, try_fold_instr,
                                                nir_metadata_block_index |
                                                nir_metadata_dominance,
                                                &state);

   /* This doesn't free the constant data if there are no constant loads because
    * the data might still be used but the loads have been lowered to load_ubo
    */
   if (state.has_load_constant && !state.has_indirect_load_const &&
       shader->constant_data_size) {
      ralloc_free(shader->constant_data);
      shader->constant_data = NULL;
      shader->constant_data_size = 0;
   }

   return progress;
}
