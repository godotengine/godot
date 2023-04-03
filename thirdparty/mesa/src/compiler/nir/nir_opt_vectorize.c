/*
 * Copyright © 2015 Connor Abbott
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
 */

/**
 * nir_opt_vectorize() aims to vectorize ALU instructions.
 *
 * The default vectorization width is 4.
 * If desired, a callback function which returns the max vectorization width
 * per instruction can be provided.
 *
 * The max vectorization width must be a power of 2.
 */

#include "nir.h"
#include "nir_vla.h"
#include "nir_builder.h"
#include "util/u_dynarray.h"

#define HASH(hash, data) XXH32(&data, sizeof(data), hash)

static uint32_t
hash_src(uint32_t hash, const nir_src *src)
{
   assert(src->is_ssa);
   void *hash_data = nir_src_is_const(*src) ? NULL : src->ssa;

   return HASH(hash, hash_data);
}

static uint32_t
hash_alu_src(uint32_t hash, const nir_alu_src *src,
             uint32_t num_components, uint32_t max_vec)
{
   assert(!src->abs && !src->negate);

   /* hash whether a swizzle accesses elements beyond the maximum
    * vectorization factor:
    * For example accesses to .x and .y are considered different variables
    * compared to accesses to .z and .w for 16-bit vec2.
    */
   uint32_t swizzle = (src->swizzle[0] & ~(max_vec - 1));
   hash = HASH(hash, swizzle);

   return hash_src(hash, &src->src);
}

static uint32_t
hash_instr(const void *data)
{
   const nir_instr *instr = (nir_instr *) data;
   assert(instr->type == nir_instr_type_alu);
   nir_alu_instr *alu = nir_instr_as_alu(instr);

   uint32_t hash = HASH(0, alu->op);
   hash = HASH(hash, alu->dest.dest.ssa.bit_size);

   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++)
      hash = hash_alu_src(hash, &alu->src[i],
                          alu->dest.dest.ssa.num_components,
                          instr->pass_flags);

   return hash;
}

static bool
srcs_equal(const nir_src *src1, const nir_src *src2)
{
   assert(src1->is_ssa);
   assert(src2->is_ssa);

   return src1->ssa == src2->ssa ||
          (nir_src_is_const(*src1) && nir_src_is_const(*src2));
}

static bool
alu_srcs_equal(const nir_alu_src *src1, const nir_alu_src *src2,
               uint32_t max_vec)
{
   assert(!src1->abs);
   assert(!src1->negate);
   assert(!src2->abs);
   assert(!src2->negate);

   uint32_t mask = ~(max_vec - 1);
   if ((src1->swizzle[0] & mask) != (src2->swizzle[0] & mask))
      return false;

   return srcs_equal(&src1->src, &src2->src);
}

static bool
instrs_equal(const void *data1, const void *data2)
{
   const nir_instr *instr1 = (nir_instr *) data1;
   const nir_instr *instr2 = (nir_instr *) data2;
   assert(instr1->type == nir_instr_type_alu);
   assert(instr2->type == nir_instr_type_alu);

   nir_alu_instr *alu1 = nir_instr_as_alu(instr1);
   nir_alu_instr *alu2 = nir_instr_as_alu(instr2);

   if (alu1->op != alu2->op)
      return false;

   if (alu1->dest.dest.ssa.bit_size != alu2->dest.dest.ssa.bit_size)
      return false;

   for (unsigned i = 0; i < nir_op_infos[alu1->op].num_inputs; i++) {
      if (!alu_srcs_equal(&alu1->src[i], &alu2->src[i], instr1->pass_flags))
         return false;
   }

   return true;
}

static bool
instr_can_rewrite(nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);

      /* Don't try and vectorize mov's. Either they'll be handled by copy
       * prop, or they're actually necessary and trying to vectorize them
       * would result in fighting with copy prop.
       */
      if (alu->op == nir_op_mov)
         return false;

      /* no need to hash instructions which are already vectorized */
      if (alu->dest.dest.ssa.num_components >= instr->pass_flags)
         return false;

      if (nir_op_infos[alu->op].output_size != 0)
         return false;

      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         if (nir_op_infos[alu->op].input_sizes[i] != 0)
            return false;

         /* don't hash instructions which are already swizzled
          * outside of max_components: these should better be scalarized */
         uint32_t mask = ~(instr->pass_flags - 1);
         for (unsigned j = 1; j < alu->dest.dest.ssa.num_components; j++) {
            if ((alu->src[i].swizzle[0] & mask) != (alu->src[i].swizzle[j] & mask))
               return false;
         }
      }

      return true;
   }

   /* TODO support phi nodes */
   default:
      break;
   }

   return false;
}

/*
 * Tries to combine two instructions whose sources are different components of
 * the same instructions into one vectorized instruction. Note that instr1
 * should dominate instr2.
 */
static nir_instr *
instr_try_combine(struct set *instr_set, nir_instr *instr1, nir_instr *instr2)
{
   assert(instr1->type == nir_instr_type_alu);
   assert(instr2->type == nir_instr_type_alu);
   nir_alu_instr *alu1 = nir_instr_as_alu(instr1);
   nir_alu_instr *alu2 = nir_instr_as_alu(instr2);

   assert(alu1->dest.dest.ssa.bit_size == alu2->dest.dest.ssa.bit_size);
   unsigned alu1_components = alu1->dest.dest.ssa.num_components;
   unsigned alu2_components = alu2->dest.dest.ssa.num_components;
   unsigned total_components = alu1_components + alu2_components;

   assert(instr1->pass_flags == instr2->pass_flags);
   if (total_components > instr1->pass_flags)
      return NULL;

   nir_builder b;
   nir_builder_init(&b, nir_cf_node_get_function(&instr1->block->cf_node));
   b.cursor = nir_after_instr(instr1);

   nir_alu_instr *new_alu = nir_alu_instr_create(b.shader, alu1->op);
   nir_ssa_dest_init(&new_alu->instr, &new_alu->dest.dest,
                     total_components, alu1->dest.dest.ssa.bit_size, NULL);
   new_alu->dest.write_mask = (1 << total_components) - 1;
   new_alu->instr.pass_flags = alu1->instr.pass_flags;

   /* If either channel is exact, we have to preserve it even if it's
    * not optimal for other channels.
    */
   new_alu->exact = alu1->exact || alu2->exact;

   /* If all channels don't wrap, we can say that the whole vector doesn't
    * wrap.
    */
   new_alu->no_signed_wrap = alu1->no_signed_wrap && alu2->no_signed_wrap;
   new_alu->no_unsigned_wrap = alu1->no_unsigned_wrap && alu2->no_unsigned_wrap;

   for (unsigned i = 0; i < nir_op_infos[alu1->op].num_inputs; i++) {
      /* handle constant merging case */
      if (alu1->src[i].src.ssa != alu2->src[i].src.ssa) {
         nir_const_value *c1 = nir_src_as_const_value(alu1->src[i].src);
         nir_const_value *c2 = nir_src_as_const_value(alu2->src[i].src);
         assert(c1 && c2);
         nir_const_value value[NIR_MAX_VEC_COMPONENTS];
         unsigned bit_size = alu1->src[i].src.ssa->bit_size;

         for (unsigned j = 0; j < total_components; j++) {
            value[j].u64 = j < alu1_components ?
                              c1[alu1->src[i].swizzle[j]].u64 :
                              c2[alu2->src[i].swizzle[j - alu1_components]].u64;
         }
         nir_ssa_def *def = nir_build_imm(&b, total_components, bit_size, value);

         new_alu->src[i].src = nir_src_for_ssa(def);
         for (unsigned j = 0; j < total_components; j++)
            new_alu->src[i].swizzle[j] = j;
         continue;
      }

      new_alu->src[i].src = alu1->src[i].src;

      for (unsigned j = 0; j < alu1_components; j++)
         new_alu->src[i].swizzle[j] = alu1->src[i].swizzle[j];

      for (unsigned j = 0; j < alu2_components; j++) {
         new_alu->src[i].swizzle[j + alu1_components] =
            alu2->src[i].swizzle[j];
      }
   }

   nir_builder_instr_insert(&b, &new_alu->instr);

   /* update all ALU uses */
   nir_foreach_use_safe(src, &alu1->dest.dest.ssa) {
      nir_instr *user_instr = src->parent_instr;
      if (user_instr->type == nir_instr_type_alu) {
         /* Check if user is found in the hashset */
         struct set_entry *entry = _mesa_set_search(instr_set, user_instr);

         /* For ALU instructions, rewrite the source directly to avoid a
          * round-trip through copy propagation.
          */
         nir_instr_rewrite_src(user_instr, src,
                               nir_src_for_ssa(&new_alu->dest.dest.ssa));

         /* Rehash user if it was found in the hashset */
         if (entry && entry->key == user_instr) {
            _mesa_set_remove(instr_set, entry);
            _mesa_set_add(instr_set, user_instr);
         }
      }
   }

   nir_foreach_use_safe(src, &alu2->dest.dest.ssa) {
      if (src->parent_instr->type == nir_instr_type_alu) {
         /* For ALU instructions, rewrite the source directly to avoid a
          * round-trip through copy propagation.
          */
         nir_instr_rewrite_src(src->parent_instr, src,
                               nir_src_for_ssa(&new_alu->dest.dest.ssa));

         nir_alu_src *alu_src = container_of(src, nir_alu_src, src);
         nir_alu_instr *use = nir_instr_as_alu(src->parent_instr);
         unsigned components = nir_ssa_alu_instr_src_components(use, alu_src - use->src);
         for (unsigned i = 0; i < components; i++)
            alu_src->swizzle[i] += alu1_components;
      }
   }

   /* update all other uses if there are any */
   unsigned swiz[NIR_MAX_VEC_COMPONENTS];

   if (!nir_ssa_def_is_unused(&alu1->dest.dest.ssa)) {
      for (unsigned i = 0; i < alu1_components; i++)
         swiz[i] = i;
      nir_ssa_def *new_alu1 = nir_swizzle(&b, &new_alu->dest.dest.ssa, swiz,
                                          alu1_components);
      nir_ssa_def_rewrite_uses(&alu1->dest.dest.ssa, new_alu1);
   }

   if (!nir_ssa_def_is_unused(&alu2->dest.dest.ssa)) {
      for (unsigned i = 0; i < alu2_components; i++)
         swiz[i] = i + alu1_components;
      nir_ssa_def *new_alu2 = nir_swizzle(&b, &new_alu->dest.dest.ssa, swiz,
                                          alu2_components);
      nir_ssa_def_rewrite_uses(&alu2->dest.dest.ssa, new_alu2);
   }

   nir_instr_remove(instr1);
   nir_instr_remove(instr2);

   return &new_alu->instr;
}

static struct set *
vec_instr_set_create(void)
{
   return _mesa_set_create(NULL, hash_instr, instrs_equal);
}

static void
vec_instr_set_destroy(struct set *instr_set)
{
   _mesa_set_destroy(instr_set, NULL);
}

static bool
vec_instr_set_add_or_rewrite(struct set *instr_set, nir_instr *instr,
                             nir_vectorize_cb filter, void *data)
{
   /* set max vector to instr pass flags: this is used to hash swizzles */
   instr->pass_flags = filter ? filter(instr, data) : 4;
   assert(util_is_power_of_two_or_zero(instr->pass_flags));

   if (!instr_can_rewrite(instr))
      return false;

   struct set_entry *entry = _mesa_set_search(instr_set, instr);
   if (entry) {
      nir_instr *old_instr = (nir_instr *) entry->key;
      _mesa_set_remove(instr_set, entry);
      nir_instr *new_instr = instr_try_combine(instr_set, old_instr, instr);
      if (new_instr) {
         if (instr_can_rewrite(new_instr))
            _mesa_set_add(instr_set, new_instr);
         return true;
      }
   }

   _mesa_set_add(instr_set, instr);
   return false;
}

static bool
vectorize_block(nir_block *block, struct set *instr_set,
                nir_vectorize_cb filter, void *data)
{
   bool progress = false;

   nir_foreach_instr_safe(instr, block) {
      if (vec_instr_set_add_or_rewrite(instr_set, instr, filter, data))
         progress = true;
   }

   for (unsigned i = 0; i < block->num_dom_children; i++) {
      nir_block *child = block->dom_children[i];
      progress |= vectorize_block(child, instr_set, filter, data);
   }

   nir_foreach_instr_reverse(instr, block) {
      if (instr_can_rewrite(instr))
         _mesa_set_remove_key(instr_set, instr);
   }

   return progress;
}

static bool
nir_opt_vectorize_impl(nir_function_impl *impl,
                       nir_vectorize_cb filter, void *data)
{
   struct set *instr_set = vec_instr_set_create();

   nir_metadata_require(impl, nir_metadata_dominance);

   bool progress = vectorize_block(nir_start_block(impl), instr_set,
                                   filter, data);

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   vec_instr_set_destroy(instr_set);
   return progress;
}

bool
nir_opt_vectorize(nir_shader *shader, nir_vectorize_cb filter,
                  void *data)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_opt_vectorize_impl(function->impl, filter, data);
   }

   return progress;
}
