/*
 * Copyright © 2021 Valve Corporation
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
 *    Timur Kristóf
 *
 */

#include "nir.h"
#include "nir_builder.h"

typedef struct
{
   struct hash_table *range_ht;
   const nir_opt_offsets_options *options;
} opt_offsets_state;

static nir_ssa_scalar
try_extract_const_addition(nir_builder *b, nir_ssa_scalar val, opt_offsets_state *state, unsigned *out_const, uint32_t max)
{
   val = nir_ssa_scalar_chase_movs(val);

   if (!nir_ssa_scalar_is_alu(val))
      return val;

   nir_alu_instr *alu = nir_instr_as_alu(val.def->parent_instr);
   if (alu->op != nir_op_iadd ||
       !alu->src[0].src.is_ssa ||
       !alu->src[1].src.is_ssa ||
       alu->src[0].negate || alu->src[0].abs ||
       alu->src[1].negate || alu->src[1].abs)
      return val;

   nir_ssa_scalar src[2] = {
      {alu->src[0].src.ssa, alu->src[0].swizzle[val.comp]},
      {alu->src[1].src.ssa, alu->src[1].swizzle[val.comp]},
   };

   /* Make sure that we aren't taking out an addition that could trigger
    * unsigned wrapping in a way that would change the semantics of the load.
    * Ignored for ints-as-floats (lower_bitops is a proxy for that), where
    * unsigned wrapping doesn't make sense.
    */
   if (!alu->no_unsigned_wrap && !b->shader->options->lower_bitops) {
      if (!state->range_ht) {
         /* Cache for nir_unsigned_upper_bound */
         state->range_ht = _mesa_pointer_hash_table_create(NULL);
      }

      /* Check if there can really be an unsigned wrap. */
      uint32_t ub0 = nir_unsigned_upper_bound(b->shader, state->range_ht, src[0], NULL);
      uint32_t ub1 = nir_unsigned_upper_bound(b->shader, state->range_ht, src[1], NULL);

      if ((UINT32_MAX - ub0) < ub1)
         return val;

      /* We proved that unsigned wrap won't be possible, so we can set the flag too. */
      alu->no_unsigned_wrap = true;
   }

   for (unsigned i = 0; i < 2; ++i) {
      src[i] = nir_ssa_scalar_chase_movs(src[i]);
      if (nir_ssa_scalar_is_const(src[i])) {
         uint32_t offset = nir_ssa_scalar_as_uint(src[i]);
         if (offset + *out_const <= max) {
            *out_const += offset;
            return try_extract_const_addition(b, src[1 - i], state, out_const, max);
         }
      }
   }

   uint32_t orig_offset = *out_const;
   src[0] = try_extract_const_addition(b, src[0], state, out_const, max);
   src[1] = try_extract_const_addition(b, src[1], state, out_const, max);
   if (*out_const == orig_offset)
      return val;

   b->cursor = nir_before_instr(&alu->instr);
   nir_ssa_def *r =
          nir_iadd(b, nir_channel(b, src[0].def, src[0].comp),
                   nir_channel(b, src[1].def, src[1].comp));
   return nir_get_ssa_scalar(r, 0);
}

static bool
try_fold_load_store(nir_builder *b,
                    nir_intrinsic_instr *intrin,
                    opt_offsets_state *state,
                    unsigned offset_src_idx,
                    uint32_t max)
{
   /* Assume that BASE is the constant offset of a load/store.
    * Try to constant-fold additions to the offset source
    * into the actual const offset of the instruction.
    */

   unsigned off_const = nir_intrinsic_base(intrin);
   nir_src *off_src = &intrin->src[offset_src_idx];
   nir_ssa_def *replace_src = NULL;

   if (!off_src->is_ssa || off_src->ssa->bit_size != 32)
      return false;

   if (!nir_src_is_const(*off_src)) {
      uint32_t add_offset = 0;
      nir_ssa_scalar val = {.def = off_src->ssa, .comp = 0};
      val = try_extract_const_addition(b, val, state, &add_offset, max - off_const);
      if (add_offset == 0)
         return false;
      off_const += add_offset;
      b->cursor = nir_before_instr(&intrin->instr);
      replace_src = nir_channel(b, val.def, val.comp);
   } else if (nir_src_as_uint(*off_src) && off_const + nir_src_as_uint(*off_src) <= max) {
      off_const += nir_src_as_uint(*off_src);
      b->cursor = nir_before_instr(&intrin->instr);
      replace_src = nir_imm_zero(b, off_src->ssa->num_components, off_src->ssa->bit_size);
   }

   if (!replace_src)
      return false;

   nir_instr_rewrite_src(&intrin->instr, &intrin->src[offset_src_idx], nir_src_for_ssa(replace_src));

   assert(off_const <= max);
   nir_intrinsic_set_base(intrin, off_const);
   return true;
}

static bool
try_fold_shared2(nir_builder *b,
                    nir_intrinsic_instr *intrin,
                    opt_offsets_state *state,
                    unsigned offset_src_idx)
{
   unsigned comp_size = (intrin->intrinsic == nir_intrinsic_load_shared2_amd ?
                         intrin->dest.ssa.bit_size : intrin->src[0].ssa->bit_size) / 8;
   unsigned stride = (nir_intrinsic_st64(intrin) ? 64 : 1) * comp_size;
   unsigned offset0 = nir_intrinsic_offset0(intrin) * stride;
   unsigned offset1 = nir_intrinsic_offset1(intrin) * stride;
   nir_src *off_src = &intrin->src[offset_src_idx];

   if (!nir_src_is_const(*off_src))
      return false;

   unsigned const_offset = nir_src_as_uint(*off_src);
   offset0 += const_offset;
   offset1 += const_offset;
   bool st64 = offset0 % (64 * comp_size) == 0 && offset1 % (64 * comp_size) == 0;
   stride = (st64 ? 64 : 1) * comp_size;
   if (const_offset % stride || offset0 > 255 * stride || offset1 > 255 * stride)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   nir_instr_rewrite_src(&intrin->instr, off_src, nir_src_for_ssa(nir_imm_zero(b, 1, 32)));
   nir_intrinsic_set_offset0(intrin, offset0 / stride);
   nir_intrinsic_set_offset1(intrin, offset1 / stride);
   nir_intrinsic_set_st64(intrin, st64);

   return true;
}

static bool
process_instr(nir_builder *b, nir_instr *instr, void *s)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   opt_offsets_state *state = (opt_offsets_state *) s;
   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   switch (intrin->intrinsic) {
   case nir_intrinsic_load_uniform:
      return try_fold_load_store(b, intrin, state, 0, state->options->uniform_max);
   case nir_intrinsic_load_ubo_vec4:
      return try_fold_load_store(b, intrin, state, 1, state->options->ubo_vec4_max);
   case nir_intrinsic_load_shared:
   case nir_intrinsic_load_shared_ir3:
      return try_fold_load_store(b, intrin, state, 0, state->options->shared_max);
   case nir_intrinsic_store_shared:
   case nir_intrinsic_store_shared_ir3:
      return try_fold_load_store(b, intrin, state, 1, state->options->shared_max);
   case nir_intrinsic_load_shared2_amd:
      return try_fold_shared2(b, intrin, state, 0);
   case nir_intrinsic_store_shared2_amd:
      return try_fold_shared2(b, intrin, state, 1);
   case nir_intrinsic_load_buffer_amd:
      return try_fold_load_store(b, intrin, state, 1, state->options->buffer_max);
   case nir_intrinsic_store_buffer_amd:
      return try_fold_load_store(b, intrin, state, 2, state->options->buffer_max);
   default:
      return false;
   }

   unreachable("Can't reach here.");
}

bool
nir_opt_offsets(nir_shader *shader, const nir_opt_offsets_options *options)
{
   opt_offsets_state state;
   state.range_ht = NULL;
   state.options = options;

   bool p = nir_shader_instructions_pass(shader, process_instr,
                                         nir_metadata_block_index |
                                         nir_metadata_dominance,
                                         &state);

   if (state.range_ht)
      _mesa_hash_table_destroy(state.range_ht, NULL);


   return p;
}
