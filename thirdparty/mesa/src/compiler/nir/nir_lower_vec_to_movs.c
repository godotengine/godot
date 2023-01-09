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

struct vec_to_movs_data {
   nir_instr_writemask_filter_cb cb;
   const void *data;
};

/*
 * Implements a simple pass that lowers vecN instructions to a series of
 * moves with partial writes.
 */

static bool
src_matches_dest_reg(nir_dest *dest, nir_src *src)
{
   if (dest->is_ssa || src->is_ssa)
      return false;

   return (dest->reg.reg == src->reg.reg &&
           dest->reg.base_offset == src->reg.base_offset &&
           !dest->reg.indirect &&
           !src->reg.indirect);
}

/**
 * For a given starting writemask channel and corresponding source index in
 * the vec instruction, insert a MOV to the vec instruction's dest of all the
 * writemask channels that get read from the same src reg.
 *
 * Returns the writemask of our MOV, so the parent loop calling this knows
 * which ones have been processed.
 */
static unsigned
insert_mov(nir_alu_instr *vec, unsigned start_idx, nir_shader *shader)
{
   assert(start_idx < nir_op_infos[vec->op].num_inputs);

   /* No sense generating a MOV from undef, we can just leave the dst channel undef. */
   if (nir_src_is_undef(vec->src[start_idx].src))
      return 1 << start_idx;

   nir_alu_instr *mov = nir_alu_instr_create(shader, nir_op_mov);
   nir_alu_src_copy(&mov->src[0], &vec->src[start_idx], mov);
   nir_alu_dest_copy(&mov->dest, &vec->dest, mov);

   mov->dest.write_mask = (1u << start_idx);
   mov->src[0].swizzle[start_idx] = vec->src[start_idx].swizzle[0];
   mov->src[0].negate = vec->src[start_idx].negate;
   mov->src[0].abs = vec->src[start_idx].abs;

   for (unsigned i = start_idx + 1; i < 4; i++) {
      if (!(vec->dest.write_mask & (1 << i)))
         continue;

      if (nir_srcs_equal(vec->src[i].src, vec->src[start_idx].src) &&
          vec->src[i].negate == vec->src[start_idx].negate &&
          vec->src[i].abs == vec->src[start_idx].abs) {
         mov->dest.write_mask |= (1 << i);
         mov->src[0].swizzle[i] = vec->src[i].swizzle[0];
      }
   }

   unsigned channels_handled = mov->dest.write_mask;

   /* In some situations (if the vecN is involved in a phi-web), we can end
    * up with a mov from a register to itself.  Some of those channels may end
    * up doing nothing and there's no reason to have them as part of the mov.
    */
   if (src_matches_dest_reg(&mov->dest.dest, &mov->src[0].src) &&
       !mov->src[0].abs && !mov->src[0].negate) {
      for (unsigned i = 0; i < 4; i++) {
         if (mov->src[0].swizzle[i] == i) {
            mov->dest.write_mask &= ~(1 << i);
         }
      }
   }

   /* Only emit the instruction if it actually does something */
   if (mov->dest.write_mask) {
      nir_instr_insert_before(&vec->instr, &mov->instr);
   } else {
      nir_instr_free(&mov->instr);
   }

   return channels_handled;
}

static bool
has_replicated_dest(nir_alu_instr *alu)
{
   return alu->op == nir_op_fdot2_replicated ||
          alu->op == nir_op_fdot3_replicated ||
          alu->op == nir_op_fdot4_replicated ||
          alu->op == nir_op_fdph_replicated;
}

/* Attempts to coalesce the "move" from the given source of the vec to the
 * destination of the instruction generating the value. If, for whatever
 * reason, we cannot coalesce the mmove, it does nothing and returns 0.  We
 * can then call insert_mov as normal.
 */
static unsigned
try_coalesce(nir_alu_instr *vec, unsigned start_idx, void *_data)
{
   struct vec_to_movs_data *data = _data;

   assert(start_idx < nir_op_infos[vec->op].num_inputs);

   /* We will only even try if the source is SSA */
   if (!vec->src[start_idx].src.is_ssa)
      return 0;

   assert(vec->src[start_idx].src.ssa);

   /* If we are going to do a reswizzle, then the vecN operation must be the
    * only use of the source value.  We also can't have any source modifiers.
    */
   nir_foreach_use(src, vec->src[start_idx].src.ssa) {
      if (src->parent_instr != &vec->instr)
         return 0;

      nir_alu_src *alu_src = exec_node_data(nir_alu_src, src, src);
      if (alu_src->abs || alu_src->negate)
         return 0;
   }

   if (!list_is_empty(&vec->src[start_idx].src.ssa->if_uses))
      return 0;

   if (vec->src[start_idx].src.ssa->parent_instr->type != nir_instr_type_alu)
      return 0;

   nir_alu_instr *src_alu =
      nir_instr_as_alu(vec->src[start_idx].src.ssa->parent_instr);

   if (has_replicated_dest(src_alu)) {
      /* The fdot instruction is special: It replicates its result to all
       * components.  This means that we can always rewrite its destination
       * and we don't need to swizzle anything.
       */
   } else {
      /* We only care about being able to re-swizzle the instruction if it is
       * something that we can reswizzle.  It must be per-component.  The one
       * exception to this is the fdotN instructions which implicitly splat
       * their result out to all channels.
       */
      if (nir_op_infos[src_alu->op].output_size != 0)
         return 0;

      /* If we are going to reswizzle the instruction, we can't have any
       * non-per-component sources either.
       */
      for (unsigned j = 0; j < nir_op_infos[src_alu->op].num_inputs; j++)
         if (nir_op_infos[src_alu->op].input_sizes[j] != 0)
            return 0;
   }

   /* Stash off all of the ALU instruction's swizzles. */
   uint8_t swizzles[4][4];
   for (unsigned j = 0; j < nir_op_infos[src_alu->op].num_inputs; j++)
      for (unsigned i = 0; i < 4; i++)
         swizzles[j][i] = src_alu->src[j].swizzle[i];

   /* Generate the final write mask */
   unsigned write_mask = 0;
   for (unsigned i = start_idx; i < 4; i++) {
      if (!(vec->dest.write_mask & (1 << i)))
         continue;

      if (!vec->src[i].src.is_ssa ||
          vec->src[i].src.ssa != &src_alu->dest.dest.ssa)
         continue;

      write_mask |= 1 << i;
   }

   /* If the instruction would be vectorized but the backend
    * doesn't support vectorizing this op, abort. */
   if (data->cb && !data->cb(&src_alu->instr, write_mask, data->data))
      return 0;

   for (unsigned i = start_idx; i < 4; i++) {
      if (!(write_mask & (1 << i)))
         continue;

      /* At this point, the given vec source matches up with the ALU
       * instruction so we can re-swizzle that component to match.
       */
      if (has_replicated_dest(src_alu)) {
         /* Since the destination is a single replicated value, we don't need
          * to do any reswizzling
          */
      } else {
         for (unsigned j = 0; j < nir_op_infos[src_alu->op].num_inputs; j++)
            src_alu->src[j].swizzle[i] = swizzles[j][vec->src[i].swizzle[0]];
      }

      /* Clear the no longer needed vec source */
      nir_instr_rewrite_src(&vec->instr, &vec->src[i].src, NIR_SRC_INIT);
   }

   nir_instr_rewrite_dest(&src_alu->instr, &src_alu->dest.dest, vec->dest.dest);
   src_alu->dest.write_mask = write_mask;

   return write_mask;
}

static bool
nir_lower_vec_to_movs_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *vec = nir_instr_as_alu(instr);

   switch (vec->op) {
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
      break;
   default:
      return false;
   }

   bool vec_had_ssa_dest = vec->dest.dest.is_ssa;
   if (vec->dest.dest.is_ssa) {
      /* Since we insert multiple MOVs, we have a register destination. */
      nir_register *reg = nir_local_reg_create(b->impl);
      reg->num_components = vec->dest.dest.ssa.num_components;
      reg->bit_size = vec->dest.dest.ssa.bit_size;

      nir_ssa_def_rewrite_uses_src(&vec->dest.dest.ssa, nir_src_for_reg(reg));

      nir_instr_rewrite_dest(&vec->instr, &vec->dest.dest,
                             nir_dest_for_reg(reg));
   }

   unsigned finished_write_mask = 0;

   /* First, emit a MOV for all the src channels that are in the
    * destination reg, in case other values we're populating in the dest
    * might overwrite them.
    */
   for (unsigned i = 0; i < 4; i++) {
      if (!(vec->dest.write_mask & (1 << i)))
         continue;

      if (src_matches_dest_reg(&vec->dest.dest, &vec->src[i].src)) {
         finished_write_mask |= insert_mov(vec, i, b->shader);
         break;
      }
   }

   /* Now, emit MOVs for all the other src channels. */
   for (unsigned i = 0; i < 4; i++) {
      if (!(vec->dest.write_mask & (1 << i)))
         continue;

      /* Coalescing moves the register writes from the vec up to the ALU
       * instruction in the source.  We can only do this if the original
       * vecN had an SSA destination.
       */
      if (vec_had_ssa_dest && !(finished_write_mask & (1 << i)))
         finished_write_mask |= try_coalesce(vec, i, data);

      if (!(finished_write_mask & (1 << i)))
         finished_write_mask |= insert_mov(vec, i, b->shader);
   }

   nir_instr_remove(&vec->instr);
   nir_instr_free(&vec->instr);

   return true;
}

bool
nir_lower_vec_to_movs(nir_shader *shader, nir_instr_writemask_filter_cb cb,
                      const void *_data)
{
   struct vec_to_movs_data data = {
      .cb = cb,
      .data = _data,
   };

   return nir_shader_instructions_pass(shader,
                                       nir_lower_vec_to_movs_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       &data);
}
