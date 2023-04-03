/*
 * Copyright © 2017 Intel Corporation
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
#include "util/u_math.h"

/**
 * \file nir_opt_intrinsics.c
 */

static nir_intrinsic_instr *
lower_subgroups_64bit_split_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin,
                                      unsigned int component)
{
   nir_ssa_def *comp;
   if (component == 0)
      comp = nir_unpack_64_2x32_split_x(b, intrin->src[0].ssa);
   else
      comp = nir_unpack_64_2x32_split_y(b, intrin->src[0].ssa);

   nir_intrinsic_instr *intr = nir_intrinsic_instr_create(b->shader, intrin->intrinsic);
   nir_ssa_dest_init(&intr->instr, &intr->dest, 1, 32, NULL);
   intr->const_index[0] = intrin->const_index[0];
   intr->const_index[1] = intrin->const_index[1];
   intr->src[0] = nir_src_for_ssa(comp);
   if (nir_intrinsic_infos[intrin->intrinsic].num_srcs == 2)
      nir_src_copy(&intr->src[1], &intrin->src[1], &intr->instr);

   intr->num_components = 1;
   nir_builder_instr_insert(b, &intr->instr);
   return intr;
}

static nir_ssa_def *
lower_subgroup_op_to_32bit(nir_builder *b, nir_intrinsic_instr *intrin)
{
   assert(intrin->src[0].ssa->bit_size == 64);
   nir_intrinsic_instr *intr_x = lower_subgroups_64bit_split_intrinsic(b, intrin, 0);
   nir_intrinsic_instr *intr_y = lower_subgroups_64bit_split_intrinsic(b, intrin, 1);
   return nir_pack_64_2x32_split(b, &intr_x->dest.ssa, &intr_y->dest.ssa);
}

static nir_ssa_def *
ballot_type_to_uint(nir_builder *b, nir_ssa_def *value,
                    const nir_lower_subgroups_options *options)
{
   /* Only the new-style SPIR-V subgroup instructions take a ballot result as
    * an argument, so we only use this on uvec4 types.
    */
   assert(value->num_components == 4 && value->bit_size == 32);

   return nir_extract_bits(b, &value, 1, 0, options->ballot_components,
                           options->ballot_bit_size);
}

static nir_ssa_def *
uint_to_ballot_type(nir_builder *b, nir_ssa_def *value,
                    unsigned num_components, unsigned bit_size)
{
   assert(util_is_power_of_two_nonzero(num_components));
   assert(util_is_power_of_two_nonzero(value->num_components));

   unsigned total_bits = bit_size * num_components;

   /* If the source doesn't have enough bits, zero-pad */
   if (total_bits > value->bit_size * value->num_components)
      value = nir_pad_vector_imm_int(b, value, 0, total_bits / value->bit_size);

   value = nir_bitcast_vector(b, value, bit_size);

   /* If the source has too many components, truncate.  This can happen if,
    * for instance, we're implementing GL_ARB_shader_ballot or
    * VK_EXT_shader_subgroup_ballot which have 64-bit ballot values on an
    * architecture with a native 128-bit uvec4 ballot.  This comes up in Zink
    * for OpenGL on Vulkan.  It's the job of the driver calling this lowering
    * pass to ensure that it's restricted subgroup sizes sufficiently that we
    * have enough ballot bits.
    */
   if (value->num_components > num_components)
      value = nir_trim_vector(b, value, num_components);

   return value;
}

static nir_ssa_def *
lower_subgroup_op_to_scalar(nir_builder *b, nir_intrinsic_instr *intrin,
                            bool lower_to_32bit)
{
   /* This is safe to call on scalar things but it would be silly */
   assert(intrin->dest.ssa.num_components > 1);

   nir_ssa_def *value = nir_ssa_for_src(b, intrin->src[0],
                                           intrin->num_components);
   nir_ssa_def *reads[NIR_MAX_VEC_COMPONENTS];

   for (unsigned i = 0; i < intrin->num_components; i++) {
      nir_intrinsic_instr *chan_intrin =
         nir_intrinsic_instr_create(b->shader, intrin->intrinsic);
      nir_ssa_dest_init(&chan_intrin->instr, &chan_intrin->dest,
                        1, intrin->dest.ssa.bit_size, NULL);
      chan_intrin->num_components = 1;

      /* value */
      chan_intrin->src[0] = nir_src_for_ssa(nir_channel(b, value, i));
      /* invocation */
      if (nir_intrinsic_infos[intrin->intrinsic].num_srcs > 1) {
         assert(nir_intrinsic_infos[intrin->intrinsic].num_srcs == 2);
         nir_src_copy(&chan_intrin->src[1], &intrin->src[1], &chan_intrin->instr);
      }

      chan_intrin->const_index[0] = intrin->const_index[0];
      chan_intrin->const_index[1] = intrin->const_index[1];

      if (lower_to_32bit && chan_intrin->src[0].ssa->bit_size == 64) {
         reads[i] = lower_subgroup_op_to_32bit(b, chan_intrin);
      } else {
         nir_builder_instr_insert(b, &chan_intrin->instr);
         reads[i] = &chan_intrin->dest.ssa;
      }
   }

   return nir_vec(b, reads, intrin->num_components);
}

static nir_ssa_def *
lower_vote_eq_to_scalar(nir_builder *b, nir_intrinsic_instr *intrin)
{
   assert(intrin->src[0].is_ssa);
   nir_ssa_def *value = intrin->src[0].ssa;

   nir_ssa_def *result = NULL;
   for (unsigned i = 0; i < intrin->num_components; i++) {
      nir_intrinsic_instr *chan_intrin =
         nir_intrinsic_instr_create(b->shader, intrin->intrinsic);
      nir_ssa_dest_init(&chan_intrin->instr, &chan_intrin->dest,
                        1, intrin->dest.ssa.bit_size, NULL);
      chan_intrin->num_components = 1;
      chan_intrin->src[0] = nir_src_for_ssa(nir_channel(b, value, i));
      nir_builder_instr_insert(b, &chan_intrin->instr);

      if (result) {
         result = nir_iand(b, result, &chan_intrin->dest.ssa);
      } else {
         result = &chan_intrin->dest.ssa;
      }
   }

   return result;
}

static nir_ssa_def *
lower_vote_eq(nir_builder *b, nir_intrinsic_instr *intrin)
{
   assert(intrin->src[0].is_ssa);
   nir_ssa_def *value = intrin->src[0].ssa;

   /* We have to implicitly lower to scalar */
   nir_ssa_def *all_eq = NULL;
   for (unsigned i = 0; i < intrin->num_components; i++) {
      nir_ssa_def *rfi = nir_read_first_invocation(b, nir_channel(b, value, i));

      nir_ssa_def *is_eq;
      if (intrin->intrinsic == nir_intrinsic_vote_feq) {
         is_eq = nir_feq(b, rfi, nir_channel(b, value, i));
      } else {
         is_eq = nir_ieq(b, rfi, nir_channel(b, value, i));
      }

      if (all_eq == NULL) {
         all_eq = is_eq;
      } else {
         all_eq = nir_iand(b, all_eq, is_eq);
      }
   }

   return nir_vote_all(b, 1, all_eq);
}

static nir_ssa_def *
lower_shuffle_to_swizzle(nir_builder *b, nir_intrinsic_instr *intrin,
                         const nir_lower_subgroups_options *options)
{
   unsigned mask = nir_src_as_uint(intrin->src[1]);

   if (mask >= 32)
      return NULL;

   nir_intrinsic_instr *swizzle = nir_intrinsic_instr_create(
      b->shader, nir_intrinsic_masked_swizzle_amd);
   swizzle->num_components = intrin->num_components;
   nir_src_copy(&swizzle->src[0], &intrin->src[0], &swizzle->instr);
   nir_intrinsic_set_swizzle_mask(swizzle, (mask << 10) | 0x1f);
   nir_ssa_dest_init(&swizzle->instr, &swizzle->dest,
                     intrin->dest.ssa.num_components,
                     intrin->dest.ssa.bit_size, NULL);

   if (options->lower_to_scalar && swizzle->num_components > 1) {
      return lower_subgroup_op_to_scalar(b, swizzle, options->lower_shuffle_to_32bit);
   } else if (options->lower_shuffle_to_32bit && swizzle->src[0].ssa->bit_size == 64) {
      return lower_subgroup_op_to_32bit(b, swizzle);
   } else {
      nir_builder_instr_insert(b, &swizzle->instr);
      return &swizzle->dest.ssa;
   }
}

/* Lowers "specialized" shuffles to a generic nir_intrinsic_shuffle. */

static nir_ssa_def *
lower_to_shuffle(nir_builder *b, nir_intrinsic_instr *intrin,
                 const nir_lower_subgroups_options *options)
{
   if (intrin->intrinsic == nir_intrinsic_shuffle_xor &&
       options->lower_shuffle_to_swizzle_amd &&
       nir_src_is_const(intrin->src[1])) {
      nir_ssa_def *result =
         lower_shuffle_to_swizzle(b, intrin, options);
      if (result)
         return result;
   }

   nir_ssa_def *index = nir_load_subgroup_invocation(b);
   bool is_shuffle = false;
   switch (intrin->intrinsic) {
   case nir_intrinsic_shuffle_xor:
      assert(intrin->src[1].is_ssa);
      index = nir_ixor(b, index, intrin->src[1].ssa);
      is_shuffle = true;
      break;
   case nir_intrinsic_shuffle_up:
      assert(intrin->src[1].is_ssa);
      index = nir_isub(b, index, intrin->src[1].ssa);
      is_shuffle = true;
      break;
   case nir_intrinsic_shuffle_down:
      assert(intrin->src[1].is_ssa);
      index = nir_iadd(b, index, intrin->src[1].ssa);
      is_shuffle = true;
      break;
   case nir_intrinsic_quad_broadcast:
      assert(intrin->src[1].is_ssa);
      index = nir_ior(b, nir_iand(b, index, nir_imm_int(b, ~0x3)),
                         intrin->src[1].ssa);
      break;
   case nir_intrinsic_quad_swap_horizontal:
      /* For Quad operations, subgroups are divided into quads where
       * (invocation % 4) is the index to a square arranged as follows:
       *
       *    +---+---+
       *    | 0 | 1 |
       *    +---+---+
       *    | 2 | 3 |
       *    +---+---+
       */
      index = nir_ixor(b, index, nir_imm_int(b, 0x1));
      break;
   case nir_intrinsic_quad_swap_vertical:
      index = nir_ixor(b, index, nir_imm_int(b, 0x2));
      break;
   case nir_intrinsic_quad_swap_diagonal:
      index = nir_ixor(b, index, nir_imm_int(b, 0x3));
      break;
   default:
      unreachable("Invalid intrinsic");
   }

   nir_intrinsic_instr *shuffle =
      nir_intrinsic_instr_create(b->shader, nir_intrinsic_shuffle);
   shuffle->num_components = intrin->num_components;
   nir_src_copy(&shuffle->src[0], &intrin->src[0], &shuffle->instr);
   shuffle->src[1] = nir_src_for_ssa(index);
   nir_ssa_dest_init(&shuffle->instr, &shuffle->dest,
                     intrin->dest.ssa.num_components,
                     intrin->dest.ssa.bit_size, NULL);

   bool lower_to_32bit = options->lower_shuffle_to_32bit && is_shuffle;
   if (options->lower_to_scalar && shuffle->num_components > 1) {
      return lower_subgroup_op_to_scalar(b, shuffle, lower_to_32bit);
   } else if (lower_to_32bit && shuffle->src[0].ssa->bit_size == 64) {
      return lower_subgroup_op_to_32bit(b, shuffle);
   } else {
      nir_builder_instr_insert(b, &shuffle->instr);
      return &shuffle->dest.ssa;
   }
}

static const struct glsl_type *
glsl_type_for_ssa(nir_ssa_def *def)
{
   const struct glsl_type *comp_type = def->bit_size == 1 ? glsl_bool_type() :
      glsl_uintN_t_type(def->bit_size);
   return glsl_replace_vector_type(comp_type, def->num_components);
}

/* Lower nir_intrinsic_shuffle to a waterfall loop + nir_read_invocation.
 */
static nir_ssa_def *
lower_shuffle(nir_builder *b, nir_intrinsic_instr *intrin)
{
   assert(intrin->src[0].is_ssa);
   assert(intrin->src[1].is_ssa);
   nir_ssa_def *val = intrin->src[0].ssa;
   nir_ssa_def *id = intrin->src[1].ssa;

   /* The loop is something like:
    *
    * while (true) {
    *    first_id = readFirstInvocation(gl_SubgroupInvocationID);
    *    first_val = readFirstInvocation(val);
    *    first_result = readInvocation(val, readFirstInvocation(id));
    *    if (id == first_id)
    *       result = first_val;
    *    if (elect()) {
    *       if (id > gl_SubgroupInvocationID) {
    *          result = first_result;
    *       }
    *       break;
    *    }
    * }
    *
    * The idea is to guarantee, on each iteration of the loop, that anything
    * reading from first_id gets the correct value, so that we can then kill
    * it off by breaking out of the loop. Before doing that we also have to
    * ensure that first_id invocation gets the correct value. It only won't be
    * assigned the correct value already if the invocation it's reading from
    * isn't already killed off, that is, if it's later than its own ID.
    * Invocations where id <= gl_SubgroupInvocationID will be assigned their
    * result in the first if, and invocations where id >
    * gl_SubgroupInvocationID will be assigned their result in the second if.
    *
    * We do this more complicated loop rather than looping over all id's
    * explicitly because at this point we don't know the "actual" subgroup
    * size and at the moment there's no way to get at it, which means we may
    * loop over always-inactive invocations.
    */

   nir_ssa_def *subgroup_id = nir_load_subgroup_invocation(b);

   nir_variable *result =
      nir_local_variable_create(b->impl, glsl_type_for_ssa(val), "result");

   nir_loop *loop = nir_push_loop(b); {
      nir_ssa_def *first_id = nir_read_first_invocation(b, subgroup_id);
      nir_ssa_def *first_val = nir_read_first_invocation(b, val);
      nir_ssa_def *first_result =
         nir_read_invocation(b, val, nir_read_first_invocation(b, id));

      nir_if *nif = nir_push_if(b, nir_ieq(b, id, first_id)); {
         nir_store_var(b, result, first_val, BITFIELD_MASK(val->num_components));
      } nir_pop_if(b, nif);

      nir_if *nif2 = nir_push_if(b, nir_elect(b, 1)); {
         nir_if *nif3 = nir_push_if(b, nir_ult(b, subgroup_id, id)); {
            nir_store_var(b, result, first_result, BITFIELD_MASK(val->num_components));
         } nir_pop_if(b, nif3);

         nir_jump(b, nir_jump_break);
      } nir_pop_if(b, nif2);
   } nir_pop_loop(b, loop);

   return nir_load_var(b, result);
}

static bool
lower_subgroups_filter(const nir_instr *instr, const void *_options)
{
   return instr->type == nir_instr_type_intrinsic;
}

/* Return a ballot-mask-sized value which represents "val" sign-extended and
 * then shifted left by "shift". Only particular values for "val" are
 * supported, see below.
 */
static nir_ssa_def *
build_ballot_imm_ishl(nir_builder *b, int64_t val, nir_ssa_def *shift,
                      const nir_lower_subgroups_options *options)
{
   /* This only works if all the high bits are the same as bit 1. */
   assert((val >> 2) == (val & 0x2 ? -1 : 0));

   /* First compute the result assuming one ballot component. */
   nir_ssa_def *result =
      nir_ishl(b, nir_imm_intN_t(b, val, options->ballot_bit_size), shift);

   if (options->ballot_components == 1)
      return result;

   /* Fix up the result when there is > 1 component. The idea is that nir_ishl
    * masks out the high bits of the shift value already, so in case there's
    * more than one component the component which 1 would be shifted into
    * already has the right value and all we have to do is fixup the other
    * components. Components below it should always be 0, and components above
    * it must be either 0 or ~0 because of the assert above. For example, if
    * the target ballot size is 2 x uint32, and we're shifting 1 by 33, then
    * we'll feed 33 into ishl, which will mask it off to get 1, so we'll
    * compute a single-component result of 2, which is correct for the second
    * component, but the first component needs to be 0, which we get by
    * comparing the high bits of the shift with 0 and selecting the original
    * answer or 0 for the first component (and something similar with the
    * second component). This idea is generalized here for any component count
    */
   nir_const_value min_shift[4];
   for (unsigned i = 0; i < options->ballot_components; i++)
      min_shift[i] = nir_const_value_for_int(i * options->ballot_bit_size, 32);
   nir_ssa_def *min_shift_val = nir_build_imm(b, options->ballot_components, 32, min_shift);

   nir_const_value max_shift[4];
   for (unsigned i = 0; i < options->ballot_components; i++)
      max_shift[i] = nir_const_value_for_int((i + 1) * options->ballot_bit_size, 32);
   nir_ssa_def *max_shift_val = nir_build_imm(b, options->ballot_components, 32, max_shift);

   return nir_bcsel(b, nir_ult(b, shift, max_shift_val),
                    nir_bcsel(b, nir_ult(b, shift, min_shift_val),
                              nir_imm_intN_t(b, val >> 63, result->bit_size),
                              result),
                    nir_imm_intN_t(b, 0, result->bit_size));
}

static nir_ssa_def *
build_subgroup_eq_mask(nir_builder *b,
                       const nir_lower_subgroups_options *options)
{
   nir_ssa_def *subgroup_idx = nir_load_subgroup_invocation(b);

   return build_ballot_imm_ishl(b, 1, subgroup_idx, options);
}

static nir_ssa_def *
build_subgroup_ge_mask(nir_builder *b,
                       const nir_lower_subgroups_options *options)
{
   nir_ssa_def *subgroup_idx = nir_load_subgroup_invocation(b);

   return build_ballot_imm_ishl(b, ~0ull, subgroup_idx, options);
}

static nir_ssa_def *
build_subgroup_gt_mask(nir_builder *b,
                       const nir_lower_subgroups_options *options)
{
   nir_ssa_def *subgroup_idx = nir_load_subgroup_invocation(b);

   return build_ballot_imm_ishl(b, ~1ull, subgroup_idx, options);
}

/* Return a mask which is 1 for threads up to the run-time subgroup size, i.e.
 * 1 for the entire subgroup. SPIR-V requires us to return 0 for indices at or
 * above the subgroup size for the masks, but gt_mask and ge_mask make them 1
 * so we have to "and" with this mask.
 */
static nir_ssa_def *
build_subgroup_mask(nir_builder *b,
                    const nir_lower_subgroups_options *options)
{
   nir_ssa_def *subgroup_size = nir_load_subgroup_size(b);

   /* First compute the result assuming one ballot component. */
   nir_ssa_def *result =
      nir_ushr(b, nir_imm_intN_t(b, ~0ull, options->ballot_bit_size),
                  nir_isub_imm(b, options->ballot_bit_size,
                               subgroup_size));

   /* Since the subgroup size and ballot bitsize are both powers of two, there
    * are two possible cases to consider:
    *
    * (1) The subgroup size is less than the ballot bitsize. We need to return
    * "result" in the first component and 0 in every other component.
    * (2) The subgroup size is a multiple of the ballot bitsize. We need to
    * return ~0 if the subgroup size divided by the ballot bitsize is less
    * than or equal to the index in the vector and 0 otherwise. For example,
    * with a target ballot type of 4 x uint32 and subgroup_size = 64 we'd need
    * to return { ~0, ~0, 0, 0 }.
    *
    * In case (2) it turns out that "result" will be ~0, because
    * "ballot_bit_size - subgroup_size" is also a multiple of
    * "ballot_bit_size" and since nir_ushr masks the shift value it will
    * shifted by 0. This means that the first component can just be "result"
    * in all cases.  The other components will also get the correct value in
    * case (1) if we just use the rule in case (2), so we'll get the correct
    * result if we just follow (2) and then replace the first component with
    * "result". 
    */ 
   nir_const_value min_idx[4];
   for (unsigned i = 0; i < options->ballot_components; i++)
      min_idx[i] = nir_const_value_for_int(i * options->ballot_bit_size, 32);
   nir_ssa_def *min_idx_val = nir_build_imm(b, options->ballot_components, 32, min_idx);

   nir_ssa_def *result_extended =
      nir_pad_vector_imm_int(b, result, ~0ull, options->ballot_components);

   return nir_bcsel(b, nir_ult(b, min_idx_val, subgroup_size),
                    result_extended, nir_imm_intN_t(b, 0, options->ballot_bit_size));
}

static nir_ssa_def *
vec_bit_count(nir_builder *b, nir_ssa_def *value)
{
   nir_ssa_def *vec_result = nir_bit_count(b, value);
   nir_ssa_def *result = nir_channel(b, vec_result, 0);
   for (unsigned i = 1; i < value->num_components; i++)
      result = nir_iadd(b, result, nir_channel(b, vec_result, i));
   return result;
}

static nir_ssa_def *
vec_find_lsb(nir_builder *b, nir_ssa_def *value)
{
   nir_ssa_def *vec_result = nir_find_lsb(b, value);
   nir_ssa_def *result = nir_imm_int(b, -1);
   for (int i = value->num_components - 1; i >= 0; i--) {
      nir_ssa_def *channel = nir_channel(b, vec_result, i);
      /* result = channel >= 0 ? (i * bitsize + channel) : result */
      result = nir_bcsel(b, nir_ige(b, channel, nir_imm_int(b, 0)),
                         nir_iadd_imm(b, channel, i * value->bit_size),
                         result);
   }
   return result;
}

static nir_ssa_def *
vec_find_msb(nir_builder *b, nir_ssa_def *value)
{
   nir_ssa_def *vec_result = nir_ufind_msb(b, value);
   nir_ssa_def *result = nir_imm_int(b, -1);
   for (unsigned i = 0; i < value->num_components; i++) {
      nir_ssa_def *channel = nir_channel(b, vec_result, i);
      /* result = channel >= 0 ? (i * bitsize + channel) : result */
      result = nir_bcsel(b, nir_ige(b, channel, nir_imm_int(b, 0)),
                         nir_iadd_imm(b, channel, i * value->bit_size),
                         result);
   }
   return result;
}

static nir_ssa_def *
lower_dynamic_quad_broadcast(nir_builder *b, nir_intrinsic_instr *intrin,
                             const nir_lower_subgroups_options *options)
{
   if (!options->lower_quad_broadcast_dynamic_to_const)
      return lower_to_shuffle(b, intrin, options);

   nir_ssa_def *dst = NULL;

   for (unsigned i = 0; i < 4; ++i) {
      nir_intrinsic_instr *qbcst =
         nir_intrinsic_instr_create(b->shader, nir_intrinsic_quad_broadcast);

      qbcst->num_components = intrin->num_components;
      qbcst->src[1] = nir_src_for_ssa(nir_imm_int(b, i));
      nir_src_copy(&qbcst->src[0], &intrin->src[0], &qbcst->instr);
      nir_ssa_dest_init(&qbcst->instr, &qbcst->dest,
                        intrin->dest.ssa.num_components,
                        intrin->dest.ssa.bit_size, NULL);

      nir_ssa_def *qbcst_dst = NULL;

      if (options->lower_to_scalar && qbcst->num_components > 1) {
         qbcst_dst = lower_subgroup_op_to_scalar(b, qbcst, false);
      } else {
         nir_builder_instr_insert(b, &qbcst->instr);
         qbcst_dst = &qbcst->dest.ssa;
      }

      if (i)
         dst = nir_bcsel(b, nir_ieq(b, intrin->src[1].ssa,
                                    nir_src_for_ssa(nir_imm_int(b, i)).ssa),
                         qbcst_dst, dst);
      else
         dst = qbcst_dst;
   }

   return dst;
}

static nir_ssa_def *
lower_read_invocation_to_cond(nir_builder *b, nir_intrinsic_instr *intrin)
{
   return nir_read_invocation_cond_ir3(b, intrin->dest.ssa.bit_size,
                                       intrin->src[0].ssa,
                                       nir_ieq(b, intrin->src[1].ssa,
                                               nir_load_subgroup_invocation(b)));
}

static nir_ssa_def *
lower_subgroups_instr(nir_builder *b, nir_instr *instr, void *_options)
{
   const nir_lower_subgroups_options *options = _options;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   switch (intrin->intrinsic) {
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
      if (options->lower_vote_trivial)
         return nir_ssa_for_src(b, intrin->src[0], 1);
      break;

   case nir_intrinsic_vote_feq:
   case nir_intrinsic_vote_ieq:
      if (options->lower_vote_trivial)
         return nir_imm_true(b);

      if (options->lower_vote_eq)
         return lower_vote_eq(b, intrin);

      if (options->lower_to_scalar && intrin->num_components > 1)
         return lower_vote_eq_to_scalar(b, intrin);
      break;

   case nir_intrinsic_load_subgroup_size:
      if (options->subgroup_size)
         return nir_imm_int(b, options->subgroup_size);
      break;

   case nir_intrinsic_read_invocation:
      if (options->lower_to_scalar && intrin->num_components > 1)
         return lower_subgroup_op_to_scalar(b, intrin, false);

      if (options->lower_read_invocation_to_cond)
         return lower_read_invocation_to_cond(b, intrin);

      break;

   case nir_intrinsic_read_first_invocation:
      if (options->lower_to_scalar && intrin->num_components > 1)
         return lower_subgroup_op_to_scalar(b, intrin, false);
      break;

   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_gt_mask:
   case nir_intrinsic_load_subgroup_le_mask:
   case nir_intrinsic_load_subgroup_lt_mask: {
      if (!options->lower_subgroup_masks)
         return NULL;

      nir_ssa_def *val;
      switch (intrin->intrinsic) {
      case nir_intrinsic_load_subgroup_eq_mask:
         val = build_subgroup_eq_mask(b, options);
         break;
      case nir_intrinsic_load_subgroup_ge_mask:
         val = nir_iand(b, build_subgroup_ge_mask(b, options),
                           build_subgroup_mask(b, options));
         break;
      case nir_intrinsic_load_subgroup_gt_mask:
         val = nir_iand(b, build_subgroup_gt_mask(b, options),
                           build_subgroup_mask(b, options));
         break;
      case nir_intrinsic_load_subgroup_le_mask:
         val = nir_inot(b, build_subgroup_gt_mask(b, options));
         break;
      case nir_intrinsic_load_subgroup_lt_mask:
         val = nir_inot(b, build_subgroup_ge_mask(b, options));
         break;
      default:
         unreachable("you seriously can't tell this is unreachable?");
      }

      return uint_to_ballot_type(b, val,
                                 intrin->dest.ssa.num_components,
                                 intrin->dest.ssa.bit_size);
   }

   case nir_intrinsic_ballot: {
      if (intrin->dest.ssa.num_components == options->ballot_components &&
          intrin->dest.ssa.bit_size == options->ballot_bit_size)
         return NULL;

      nir_ssa_def *ballot =
         nir_ballot(b, options->ballot_components, options->ballot_bit_size,
                    intrin->src[0].ssa);

      return uint_to_ballot_type(b, ballot,
                                 intrin->dest.ssa.num_components,
                                 intrin->dest.ssa.bit_size);
   }

   case nir_intrinsic_ballot_bitfield_extract:
   case nir_intrinsic_ballot_bit_count_reduce:
   case nir_intrinsic_ballot_find_lsb:
   case nir_intrinsic_ballot_find_msb: {
      assert(intrin->src[0].is_ssa);
      nir_ssa_def *int_val = ballot_type_to_uint(b, intrin->src[0].ssa,
                                                 options);

      if (intrin->intrinsic != nir_intrinsic_ballot_bitfield_extract &&
          intrin->intrinsic != nir_intrinsic_ballot_find_lsb) {
         /* For OpGroupNonUniformBallotFindMSB, the SPIR-V Spec says:
          *
          *    "Find the most significant bit set to 1 in Value, considering
          *    only the bits in Value required to represent all bits of the
          *    group’s invocations.  If none of the considered bits is set to
          *    1, the result is undefined."
          *
          * It has similar text for the other three.  This means that, in case
          * the subgroup size is less than 32, we have to mask off the unused
          * bits.  If the subgroup size is fixed and greater than or equal to
          * 32, the mask will be 0xffffffff and nir_opt_algebraic will delete
          * the iand.
          *
          * We only have to worry about this for BitCount and FindMSB because
          * FindLSB counts from the bottom and BitfieldExtract selects
          * individual bits.  In either case, if run outside the range of
          * valid bits, we hit the undefined results case and we can return
          * anything we want.
          */
         int_val = nir_iand(b, int_val, build_subgroup_mask(b, options));
      }

      switch (intrin->intrinsic) {
      case nir_intrinsic_ballot_bitfield_extract: {
         assert(intrin->src[1].is_ssa);
         nir_ssa_def *idx = intrin->src[1].ssa;
         if (int_val->num_components > 1) {
            /* idx will be truncated by nir_ushr, so we just need to select
             * the right component using the bits of idx that are truncated in
             * the shift.
             */
            int_val =
               nir_vector_extract(b, int_val,
                                  nir_udiv_imm(b, idx, int_val->bit_size));
         }

         return nir_test_mask(b, nir_ushr(b, int_val, idx), 1);
      }
      case nir_intrinsic_ballot_bit_count_reduce:
         return vec_bit_count(b, int_val);
      case nir_intrinsic_ballot_find_lsb:
         return vec_find_lsb(b, int_val);
      case nir_intrinsic_ballot_find_msb:
         return vec_find_msb(b, int_val);
      default:
         unreachable("you seriously can't tell this is unreachable?");
      }
   }

   case nir_intrinsic_ballot_bit_count_exclusive:
   case nir_intrinsic_ballot_bit_count_inclusive: {
      nir_ssa_def *mask;
      if (intrin->intrinsic == nir_intrinsic_ballot_bit_count_inclusive) {
         mask = nir_inot(b, build_subgroup_gt_mask(b, options));
      } else {
         mask = nir_inot(b, build_subgroup_ge_mask(b, options));
      }

      assert(intrin->src[0].is_ssa);
      nir_ssa_def *int_val = ballot_type_to_uint(b, intrin->src[0].ssa,
                                                 options);

      return vec_bit_count(b, nir_iand(b, int_val, mask));
   }

   case nir_intrinsic_elect: {
      if (!options->lower_elect)
         return NULL;

      return nir_ieq(b, nir_load_subgroup_invocation(b), nir_first_invocation(b));
   }

   case nir_intrinsic_shuffle:
      if (options->lower_shuffle)
         return lower_shuffle(b, intrin);
      else if (options->lower_to_scalar && intrin->num_components > 1)
         return lower_subgroup_op_to_scalar(b, intrin, options->lower_shuffle_to_32bit);
      else if (options->lower_shuffle_to_32bit && intrin->src[0].ssa->bit_size == 64)
         return lower_subgroup_op_to_32bit(b, intrin);
      break;
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
      if (options->lower_relative_shuffle)
         return lower_to_shuffle(b, intrin, options);
      else if (options->lower_to_scalar && intrin->num_components > 1)
         return lower_subgroup_op_to_scalar(b, intrin, options->lower_shuffle_to_32bit);
      else if (options->lower_shuffle_to_32bit && intrin->src[0].ssa->bit_size == 64)
         return lower_subgroup_op_to_32bit(b, intrin);
      break;

   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
      if (options->lower_quad ||
          (options->lower_quad_broadcast_dynamic &&
           intrin->intrinsic == nir_intrinsic_quad_broadcast &&
           !nir_src_is_const(intrin->src[1])))
         return lower_dynamic_quad_broadcast(b, intrin, options);
      else if (options->lower_to_scalar && intrin->num_components > 1)
         return lower_subgroup_op_to_scalar(b, intrin, false);
      break;

   case nir_intrinsic_reduce: {
      nir_ssa_def *ret = NULL;
      /* A cluster size greater than the subgroup size is implemention defined */
      if (options->subgroup_size &&
          nir_intrinsic_cluster_size(intrin) >= options->subgroup_size) {
         nir_intrinsic_set_cluster_size(intrin, 0);
         ret = NIR_LOWER_INSTR_PROGRESS;
      }
      if (options->lower_to_scalar && intrin->num_components > 1)
         ret = lower_subgroup_op_to_scalar(b, intrin, false);
      return ret;
   }
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan:
      if (options->lower_to_scalar && intrin->num_components > 1)
         return lower_subgroup_op_to_scalar(b, intrin, false);
      break;

   default:
      break;
   }

   return NULL;
}

bool
nir_lower_subgroups(nir_shader *shader,
                    const nir_lower_subgroups_options *options)
{
   return nir_shader_lower_instructions(shader,
                                        lower_subgroups_filter,
                                        lower_subgroups_instr,
                                        (void *)options);
}
