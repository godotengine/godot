/*
 * Copyright © Microsoft Corporation
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "dxil_nir.h"

#include "nir_builder.h"
#include "nir_deref.h"
#include "nir_to_dxil.h"
#include "util/u_math.h"
#include "vulkan/vulkan_core.h"

static void
cl_type_size_align(const struct glsl_type *type, unsigned *size,
                   unsigned *align)
{
   *size = glsl_get_cl_size(type);
   *align = glsl_get_cl_alignment(type);
}

static void
extract_comps_from_vec32(nir_builder *b, nir_ssa_def *vec32,
                         unsigned dst_bit_size,
                         nir_ssa_def **dst_comps,
                         unsigned num_dst_comps)
{
   unsigned step = DIV_ROUND_UP(dst_bit_size, 32);
   unsigned comps_per32b = 32 / dst_bit_size;
   nir_ssa_def *tmp;

   for (unsigned i = 0; i < vec32->num_components; i += step) {
      switch (dst_bit_size) {
      case 64:
         tmp = nir_pack_64_2x32_split(b, nir_channel(b, vec32, i),
                                         nir_channel(b, vec32, i + 1));
         dst_comps[i / 2] = tmp;
         break;
      case 32:
         dst_comps[i] = nir_channel(b, vec32, i);
         break;
      case 16:
      case 8: {
         unsigned dst_offs = i * comps_per32b;

         tmp = nir_unpack_bits(b, nir_channel(b, vec32, i), dst_bit_size);
         for (unsigned j = 0; j < comps_per32b && dst_offs + j < num_dst_comps; j++)
            dst_comps[dst_offs + j] = nir_channel(b, tmp, j);
         }

         break;
      }
   }
}

static nir_ssa_def *
load_comps_to_vec32(nir_builder *b, unsigned src_bit_size,
                    nir_ssa_def **src_comps, unsigned num_src_comps)
{
   unsigned num_vec32comps = DIV_ROUND_UP(num_src_comps * src_bit_size, 32);
   unsigned step = DIV_ROUND_UP(src_bit_size, 32);
   unsigned comps_per32b = 32 / src_bit_size;
   nir_ssa_def *vec32comps[4];

   for (unsigned i = 0; i < num_vec32comps; i += step) {
      switch (src_bit_size) {
      case 64:
         vec32comps[i] = nir_unpack_64_2x32_split_x(b, src_comps[i / 2]);
         vec32comps[i + 1] = nir_unpack_64_2x32_split_y(b, src_comps[i / 2]);
         break;
      case 32:
         vec32comps[i] = src_comps[i];
         break;
      case 16:
      case 8: {
         unsigned src_offs = i * comps_per32b;

         vec32comps[i] = nir_u2u32(b, src_comps[src_offs]);
         for (unsigned j = 1; j < comps_per32b && src_offs + j < num_src_comps; j++) {
            nir_ssa_def *tmp = nir_ishl(b, nir_u2u32(b, src_comps[src_offs + j]),
                                           nir_imm_int(b, j * src_bit_size));
            vec32comps[i] = nir_ior(b, vec32comps[i], tmp);
         }
         break;
      }
      }
   }

   return nir_vec(b, vec32comps, num_vec32comps);
}

static nir_ssa_def *
build_load_ptr_dxil(nir_builder *b, nir_deref_instr *deref, nir_ssa_def *idx)
{
   return nir_load_ptr_dxil(b, 1, 32, &deref->dest.ssa, idx);
}

static bool
lower_load_deref(nir_builder *b, nir_intrinsic_instr *intr)
{
   assert(intr->dest.is_ssa);

   b->cursor = nir_before_instr(&intr->instr);

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   if (!nir_deref_mode_is(deref, nir_var_shader_temp))
      return false;
   nir_ssa_def *ptr = nir_u2u32(b, nir_build_deref_offset(b, deref, cl_type_size_align));
   nir_ssa_def *offset = nir_iand(b, ptr, nir_inot(b, nir_imm_int(b, 3)));

   assert(intr->dest.is_ssa);
   unsigned num_components = nir_dest_num_components(intr->dest);
   unsigned bit_size = nir_dest_bit_size(intr->dest);
   unsigned load_size = MAX2(32, bit_size);
   unsigned num_bits = num_components * bit_size;
   nir_ssa_def *comps[NIR_MAX_VEC_COMPONENTS];
   unsigned comp_idx = 0;

   nir_deref_path path;
   nir_deref_path_init(&path, deref, NULL);
   nir_ssa_def *base_idx = nir_ishr(b, offset, nir_imm_int(b, 2 /* log2(32 / 8) */));

   /* Split loads into 32-bit chunks */
   for (unsigned i = 0; i < num_bits; i += load_size) {
      unsigned subload_num_bits = MIN2(num_bits - i, load_size);
      nir_ssa_def *idx = nir_iadd(b, base_idx, nir_imm_int(b, i / 32));
      nir_ssa_def *vec32 = build_load_ptr_dxil(b, path.path[0], idx);

      if (load_size == 64) {
         idx = nir_iadd(b, idx, nir_imm_int(b, 1));
         vec32 = nir_vec2(b, vec32,
                             build_load_ptr_dxil(b, path.path[0], idx));
      }

      /* If we have 2 bytes or less to load we need to adjust the u32 value so
       * we can always extract the LSB.
       */
      if (subload_num_bits <= 16) {
         nir_ssa_def *shift = nir_imul(b, nir_iand(b, ptr, nir_imm_int(b, 3)),
                                          nir_imm_int(b, 8));
         vec32 = nir_ushr(b, vec32, shift);
      }

      /* And now comes the pack/unpack step to match the original type. */
      extract_comps_from_vec32(b, vec32, bit_size, &comps[comp_idx],
                               subload_num_bits / bit_size);
      comp_idx += subload_num_bits / bit_size;
   }

   nir_deref_path_finish(&path);
   assert(comp_idx == num_components);
   nir_ssa_def *result = nir_vec(b, comps, num_components);
   nir_ssa_def_rewrite_uses(&intr->dest.ssa, result);
   nir_instr_remove(&intr->instr);
   return true;
}

static nir_ssa_def *
ubo_load_select_32b_comps(nir_builder *b, nir_ssa_def *vec32,
                          nir_ssa_def *offset, unsigned num_bytes)
{
   assert(num_bytes == 16 || num_bytes == 12 || num_bytes == 8 ||
          num_bytes == 4 || num_bytes == 3 || num_bytes == 2 ||
          num_bytes == 1);
   assert(vec32->num_components == 4);

   /* 16 and 12 byte types are always aligned on 16 bytes. */
   if (num_bytes > 8)
      return vec32;

   nir_ssa_def *comps[4];
   nir_ssa_def *cond;

   for (unsigned i = 0; i < 4; i++)
      comps[i] = nir_channel(b, vec32, i);

   /* If we have 8bytes or less to load, select which half the vec4 should
    * be used.
    */
   cond = nir_ine(b, nir_iand(b, offset, nir_imm_int(b, 0x8)),
                                 nir_imm_int(b, 0));

   comps[0] = nir_bcsel(b, cond, comps[2], comps[0]);
   comps[1] = nir_bcsel(b, cond, comps[3], comps[1]);

   /* Thanks to the CL alignment constraints, if we want 8 bytes we're done. */
   if (num_bytes == 8)
      return nir_vec(b, comps, 2);

   /* 4 bytes or less needed, select which of the 32bit component should be
    * used and return it. The sub-32bit split is handled in
    * extract_comps_from_vec32().
    */
   cond = nir_ine(b, nir_iand(b, offset, nir_imm_int(b, 0x4)),
                                 nir_imm_int(b, 0));
   return nir_bcsel(b, cond, comps[1], comps[0]);
}

nir_ssa_def *
build_load_ubo_dxil(nir_builder *b, nir_ssa_def *buffer,
                    nir_ssa_def *offset, unsigned num_components,
                    unsigned bit_size)
{
   nir_ssa_def *idx = nir_ushr(b, offset, nir_imm_int(b, 4));
   nir_ssa_def *comps[NIR_MAX_VEC_COMPONENTS];
   unsigned num_bits = num_components * bit_size;
   unsigned comp_idx = 0;

   /* We need to split loads in 16byte chunks because that's the
    * granularity of cBufferLoadLegacy().
    */
   for (unsigned i = 0; i < num_bits; i += (16 * 8)) {
      /* For each 16byte chunk (or smaller) we generate a 32bit ubo vec
       * load.
       */
      unsigned subload_num_bits = MIN2(num_bits - i, 16 * 8);
      nir_ssa_def *vec32 =
         nir_load_ubo_dxil(b, 4, 32, buffer, nir_iadd(b, idx, nir_imm_int(b, i / (16 * 8))));

      /* First re-arrange the vec32 to account for intra 16-byte offset. */
      vec32 = ubo_load_select_32b_comps(b, vec32, offset, subload_num_bits / 8);

      /* If we have 2 bytes or less to load we need to adjust the u32 value so
       * we can always extract the LSB.
       */
      if (subload_num_bits <= 16) {
         nir_ssa_def *shift = nir_imul(b, nir_iand(b, offset,
                                                      nir_imm_int(b, 3)),
                                          nir_imm_int(b, 8));
         vec32 = nir_ushr(b, vec32, shift);
      }

      /* And now comes the pack/unpack step to match the original type. */
      extract_comps_from_vec32(b, vec32, bit_size, &comps[comp_idx],
                               subload_num_bits / bit_size);
      comp_idx += subload_num_bits / bit_size;
   }

   assert(comp_idx == num_components);
   return nir_vec(b, comps, num_components);
}

static bool
lower_load_ssbo(nir_builder *b, nir_intrinsic_instr *intr)
{
   assert(intr->dest.is_ssa);
   assert(intr->src[0].is_ssa);
   assert(intr->src[1].is_ssa);

   b->cursor = nir_before_instr(&intr->instr);

   nir_ssa_def *buffer = intr->src[0].ssa;
   nir_ssa_def *offset = nir_iand(b, intr->src[1].ssa, nir_imm_int(b, ~3));
   enum gl_access_qualifier access = nir_intrinsic_access(intr);
   unsigned bit_size = nir_dest_bit_size(intr->dest);
   unsigned num_components = nir_dest_num_components(intr->dest);
   unsigned num_bits = num_components * bit_size;

   nir_ssa_def *comps[NIR_MAX_VEC_COMPONENTS];
   unsigned comp_idx = 0;

   /* We need to split loads in 16byte chunks because that's the optimal
    * granularity of bufferLoad(). Minimum alignment is 4byte, which saves
    * from us from extra complexity to extract >= 32 bit components.
    */
   for (unsigned i = 0; i < num_bits; i += 4 * 32) {
      /* For each 16byte chunk (or smaller) we generate a 32bit ssbo vec
       * load.
       */
      unsigned subload_num_bits = MIN2(num_bits - i, 4 * 32);

      /* The number of components to store depends on the number of bytes. */
      nir_ssa_def *vec32 =
         nir_load_ssbo(b, DIV_ROUND_UP(subload_num_bits, 32), 32,
                       buffer, nir_iadd(b, offset, nir_imm_int(b, i / 8)),
                       .align_mul = 4,
                       .align_offset = 0,
                       .access = access);

      /* If we have 2 bytes or less to load we need to adjust the u32 value so
       * we can always extract the LSB.
       */
      if (subload_num_bits <= 16) {
         nir_ssa_def *shift = nir_imul(b, nir_iand(b, intr->src[1].ssa, nir_imm_int(b, 3)),
                                          nir_imm_int(b, 8));
         vec32 = nir_ushr(b, vec32, shift);
      }

      /* And now comes the pack/unpack step to match the original type. */
      extract_comps_from_vec32(b, vec32, bit_size, &comps[comp_idx],
                               subload_num_bits / bit_size);
      comp_idx += subload_num_bits / bit_size;
   }

   assert(comp_idx == num_components);
   nir_ssa_def *result = nir_vec(b, comps, num_components);
   nir_ssa_def_rewrite_uses(&intr->dest.ssa, result);
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_store_ssbo(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);

   assert(intr->src[0].is_ssa);
   assert(intr->src[1].is_ssa);
   assert(intr->src[2].is_ssa);

   nir_ssa_def *val = intr->src[0].ssa;
   nir_ssa_def *buffer = intr->src[1].ssa;
   nir_ssa_def *offset = nir_iand(b, intr->src[2].ssa, nir_imm_int(b, ~3));

   unsigned bit_size = val->bit_size;
   unsigned num_components = val->num_components;
   unsigned num_bits = num_components * bit_size;

   nir_ssa_def *comps[NIR_MAX_VEC_COMPONENTS] = { 0 };
   unsigned comp_idx = 0;

   unsigned write_mask = nir_intrinsic_write_mask(intr);
   for (unsigned i = 0; i < num_components; i++)
      if (write_mask & (1 << i))
         comps[i] = nir_channel(b, val, i);

   /* We split stores in 16byte chunks because that's the optimal granularity
    * of bufferStore(). Minimum alignment is 4byte, which saves from us from
    * extra complexity to store >= 32 bit components.
    */
   unsigned bit_offset = 0;
   while (true) {
      /* Skip over holes in the write mask */
      while (comp_idx < num_components && comps[comp_idx] == NULL) {
         comp_idx++;
         bit_offset += bit_size;
      }
      if (comp_idx >= num_components)
         break;

      /* For each 16byte chunk (or smaller) we generate a 32bit ssbo vec
       * store. If a component is skipped by the write mask, do a smaller
       * sub-store
       */
      unsigned num_src_comps_stored = 0, substore_num_bits = 0;
      while(num_src_comps_stored + comp_idx < num_components &&
            substore_num_bits + bit_offset < num_bits &&
            substore_num_bits < 4 * 32 &&
            comps[comp_idx + num_src_comps_stored]) {
         ++num_src_comps_stored;
         substore_num_bits += bit_size;
      }
      nir_ssa_def *local_offset = nir_iadd(b, offset, nir_imm_int(b, bit_offset / 8));
      nir_ssa_def *vec32 = load_comps_to_vec32(b, bit_size, &comps[comp_idx],
                                               num_src_comps_stored);
      nir_intrinsic_instr *store;

      if (substore_num_bits < 32) {
         nir_ssa_def *mask = nir_imm_int(b, (1 << substore_num_bits) - 1);

        /* If we have 16 bits or less to store we need to place them
         * correctly in the u32 component. Anything greater than 16 bits
         * (including uchar3) is naturally aligned on 32bits.
         */
         if (substore_num_bits <= 16) {
            nir_ssa_def *pos = nir_iand(b, intr->src[2].ssa, nir_imm_int(b, 3));
            nir_ssa_def *shift = nir_imul_imm(b, pos, 8);

            vec32 = nir_ishl(b, vec32, shift);
            mask = nir_ishl(b, mask, shift);
         }

         store = nir_intrinsic_instr_create(b->shader,
                                            nir_intrinsic_store_ssbo_masked_dxil);
         store->src[0] = nir_src_for_ssa(vec32);
         store->src[1] = nir_src_for_ssa(nir_inot(b, mask));
         store->src[2] = nir_src_for_ssa(buffer);
         store->src[3] = nir_src_for_ssa(local_offset);
      } else {
         store = nir_intrinsic_instr_create(b->shader,
                                            nir_intrinsic_store_ssbo);
         store->src[0] = nir_src_for_ssa(vec32);
         store->src[1] = nir_src_for_ssa(buffer);
         store->src[2] = nir_src_for_ssa(local_offset);

         nir_intrinsic_set_align(store, 4, 0);
      }

      /* The number of components to store depends on the number of bits. */
      store->num_components = DIV_ROUND_UP(substore_num_bits, 32);
      nir_builder_instr_insert(b, &store->instr);
      comp_idx += num_src_comps_stored;
      bit_offset += substore_num_bits;

      if (nir_intrinsic_has_write_mask(store))
         nir_intrinsic_set_write_mask(store, (1 << store->num_components) - 1);
   }

   nir_instr_remove(&intr->instr);
   return true;
}

static void
lower_load_vec32(nir_builder *b, nir_ssa_def *index, unsigned num_comps, nir_ssa_def **comps, nir_intrinsic_op op)
{
   for (unsigned i = 0; i < num_comps; i++) {
      nir_intrinsic_instr *load =
         nir_intrinsic_instr_create(b->shader, op);

      load->num_components = 1;
      load->src[0] = nir_src_for_ssa(nir_iadd(b, index, nir_imm_int(b, i)));
      nir_ssa_dest_init(&load->instr, &load->dest, 1, 32, NULL);
      nir_builder_instr_insert(b, &load->instr);
      comps[i] = &load->dest.ssa;
   }
}

static bool
lower_32b_offset_load(nir_builder *b, nir_intrinsic_instr *intr)
{
   assert(intr->dest.is_ssa);
   unsigned bit_size = nir_dest_bit_size(intr->dest);
   unsigned num_components = nir_dest_num_components(intr->dest);
   unsigned num_bits = num_components * bit_size;

   b->cursor = nir_before_instr(&intr->instr);
   nir_intrinsic_op op = intr->intrinsic;

   assert(intr->src[0].is_ssa);
   nir_ssa_def *offset = intr->src[0].ssa;
   if (op == nir_intrinsic_load_shared) {
      offset = nir_iadd(b, offset, nir_imm_int(b, nir_intrinsic_base(intr)));
      op = nir_intrinsic_load_shared_dxil;
   } else {
      offset = nir_u2u32(b, offset);
      op = nir_intrinsic_load_scratch_dxil;
   }
   nir_ssa_def *index = nir_ushr(b, offset, nir_imm_int(b, 2));
   nir_ssa_def *comps[NIR_MAX_VEC_COMPONENTS];
   nir_ssa_def *comps_32bit[NIR_MAX_VEC_COMPONENTS * 2];

   /* We need to split loads in 32-bit accesses because the buffer
    * is an i32 array and DXIL does not support type casts.
    */
   unsigned num_32bit_comps = DIV_ROUND_UP(num_bits, 32);
   lower_load_vec32(b, index, num_32bit_comps, comps_32bit, op);
   unsigned num_comps_per_pass = MIN2(num_32bit_comps, 4);

   for (unsigned i = 0; i < num_32bit_comps; i += num_comps_per_pass) {
      unsigned num_vec32_comps = MIN2(num_32bit_comps - i, 4);
      unsigned num_dest_comps = num_vec32_comps * 32 / bit_size;
      nir_ssa_def *vec32 = nir_vec(b, &comps_32bit[i], num_vec32_comps);

      /* If we have 16 bits or less to load we need to adjust the u32 value so
       * we can always extract the LSB.
       */
      if (num_bits <= 16) {
         nir_ssa_def *shift =
            nir_imul(b, nir_iand(b, offset, nir_imm_int(b, 3)),
                        nir_imm_int(b, 8));
         vec32 = nir_ushr(b, vec32, shift);
      }

      /* And now comes the pack/unpack step to match the original type. */
      unsigned dest_index = i * 32 / bit_size;
      extract_comps_from_vec32(b, vec32, bit_size, &comps[dest_index], num_dest_comps);
   }

   nir_ssa_def *result = nir_vec(b, comps, num_components);
   nir_ssa_def_rewrite_uses(&intr->dest.ssa, result);
   nir_instr_remove(&intr->instr);

   return true;
}

static void
lower_store_vec32(nir_builder *b, nir_ssa_def *index, nir_ssa_def *vec32, nir_intrinsic_op op)
{

   for (unsigned i = 0; i < vec32->num_components; i++) {
      nir_intrinsic_instr *store =
         nir_intrinsic_instr_create(b->shader, op);

      store->src[0] = nir_src_for_ssa(nir_channel(b, vec32, i));
      store->src[1] = nir_src_for_ssa(nir_iadd(b, index, nir_imm_int(b, i)));
      store->num_components = 1;
      nir_builder_instr_insert(b, &store->instr);
   }
}

static void
lower_masked_store_vec32(nir_builder *b, nir_ssa_def *offset, nir_ssa_def *index,
                         nir_ssa_def *vec32, unsigned num_bits, nir_intrinsic_op op)
{
   nir_ssa_def *mask = nir_imm_int(b, (1 << num_bits) - 1);

   /* If we have 16 bits or less to store we need to place them correctly in
    * the u32 component. Anything greater than 16 bits (including uchar3) is
    * naturally aligned on 32bits.
    */
   if (num_bits <= 16) {
      nir_ssa_def *shift =
         nir_imul_imm(b, nir_iand(b, offset, nir_imm_int(b, 3)), 8);

      vec32 = nir_ishl(b, vec32, shift);
      mask = nir_ishl(b, mask, shift);
   }

   if (op == nir_intrinsic_store_shared_dxil) {
      /* Use the dedicated masked intrinsic */
      nir_store_shared_masked_dxil(b, vec32, nir_inot(b, mask), index);
   } else {
      /* For scratch, since we don't need atomics, just generate the read-modify-write in NIR */
      nir_ssa_def *load = nir_load_scratch_dxil(b, 1, 32, index);

      nir_ssa_def *new_val = nir_ior(b, vec32,
                                     nir_iand(b,
                                              nir_inot(b, mask),
                                              load));

      lower_store_vec32(b, index, new_val, op);
   }
}

static bool
lower_32b_offset_store(nir_builder *b, nir_intrinsic_instr *intr)
{
   assert(intr->src[0].is_ssa);
   unsigned num_components = nir_src_num_components(intr->src[0]);
   unsigned bit_size = nir_src_bit_size(intr->src[0]);
   unsigned num_bits = num_components * bit_size;

   b->cursor = nir_before_instr(&intr->instr);
   nir_intrinsic_op op = intr->intrinsic;

   nir_ssa_def *offset = intr->src[1].ssa;
   if (op == nir_intrinsic_store_shared) {
      offset = nir_iadd(b, offset, nir_imm_int(b, nir_intrinsic_base(intr)));
      op = nir_intrinsic_store_shared_dxil;
   } else {
      offset = nir_u2u32(b, offset);
      op = nir_intrinsic_store_scratch_dxil;
   }
   nir_ssa_def *comps[NIR_MAX_VEC_COMPONENTS];

   unsigned comp_idx = 0;
   for (unsigned i = 0; i < num_components; i++)
      comps[i] = nir_channel(b, intr->src[0].ssa, i);

   for (unsigned i = 0; i < num_bits; i += 4 * 32) {
      /* For each 4byte chunk (or smaller) we generate a 32bit scalar store.
       */
      unsigned substore_num_bits = MIN2(num_bits - i, 4 * 32);
      nir_ssa_def *local_offset = nir_iadd(b, offset, nir_imm_int(b, i / 8));
      nir_ssa_def *vec32 = load_comps_to_vec32(b, bit_size, &comps[comp_idx],
                                               substore_num_bits / bit_size);
      nir_ssa_def *index = nir_ushr(b, local_offset, nir_imm_int(b, 2));

      /* For anything less than 32bits we need to use the masked version of the
       * intrinsic to preserve data living in the same 32bit slot.
       */
      if (num_bits < 32) {
         lower_masked_store_vec32(b, local_offset, index, vec32, num_bits, op);
      } else {
         lower_store_vec32(b, index, vec32, op);
      }

      comp_idx += substore_num_bits / bit_size;
   }

   nir_instr_remove(&intr->instr);

   return true;
}

static void
ubo_to_temp_patch_deref_mode(nir_deref_instr *deref)
{
   deref->modes = nir_var_shader_temp;
   nir_foreach_use(use_src, &deref->dest.ssa) {
      if (use_src->parent_instr->type != nir_instr_type_deref)
         continue;

      nir_deref_instr *parent = nir_instr_as_deref(use_src->parent_instr);
      ubo_to_temp_patch_deref_mode(parent);
   }
}

static void
ubo_to_temp_update_entry(nir_deref_instr *deref, struct hash_entry *he)
{
   assert(nir_deref_mode_is(deref, nir_var_mem_constant));
   assert(deref->dest.is_ssa);
   assert(he->data);

   nir_foreach_use(use_src, &deref->dest.ssa) {
      if (use_src->parent_instr->type == nir_instr_type_deref) {
         ubo_to_temp_update_entry(nir_instr_as_deref(use_src->parent_instr), he);
      } else if (use_src->parent_instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(use_src->parent_instr);
         if (intr->intrinsic != nir_intrinsic_load_deref)
            he->data = NULL;
      } else {
         he->data = NULL;
      }

      if (!he->data)
         break;
   }
}

bool
dxil_nir_lower_ubo_to_temp(nir_shader *nir)
{
   struct hash_table *ubo_to_temp = _mesa_pointer_hash_table_create(NULL);
   bool progress = false;

   /* First pass: collect all UBO accesses that could be turned into
    * shader temp accesses.
    */
   foreach_list_typed(nir_function, func, node, &nir->functions) {
      if (!func->is_entrypoint)
         continue;
      assert(func->impl);

      nir_foreach_block(block, func->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (!nir_deref_mode_is(deref, nir_var_mem_constant) ||
                deref->deref_type != nir_deref_type_var)
                  continue;

            struct hash_entry *he =
               _mesa_hash_table_search(ubo_to_temp, deref->var);

            if (!he)
               he = _mesa_hash_table_insert(ubo_to_temp, deref->var, deref->var);

            if (!he->data)
               continue;

            ubo_to_temp_update_entry(deref, he);
         }
      }
   }

   hash_table_foreach(ubo_to_temp, he) {
      nir_variable *var = he->data;

      if (!var)
         continue;

      /* Change the variable mode. */
      var->data.mode = nir_var_shader_temp;

      /* Make sure the variable has a name.
       * DXIL variables must have names.
       */
      if (!var->name)
         var->name = ralloc_asprintf(nir, "global_%d", exec_list_length(&nir->variables));

      progress = true;
   }
   _mesa_hash_table_destroy(ubo_to_temp, NULL);

   /* Second pass: patch all derefs that were accessing the converted UBOs
    * variables.
    */
   foreach_list_typed(nir_function, func, node, &nir->functions) {
      if (!func->is_entrypoint)
         continue;
      assert(func->impl);

      nir_foreach_block(block, func->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (nir_deref_mode_is(deref, nir_var_mem_constant) &&
                deref->deref_type == nir_deref_type_var &&
                deref->var->data.mode == nir_var_shader_temp)
               ubo_to_temp_patch_deref_mode(deref);
         }
      }
   }

   return progress;
}

static bool
lower_load_ubo(nir_builder *b, nir_intrinsic_instr *intr)
{
   assert(intr->dest.is_ssa);
   assert(intr->src[0].is_ssa);
   assert(intr->src[1].is_ssa);

   b->cursor = nir_before_instr(&intr->instr);

   nir_ssa_def *result =
      build_load_ubo_dxil(b, intr->src[0].ssa, intr->src[1].ssa,
                             nir_dest_num_components(intr->dest),
                             nir_dest_bit_size(intr->dest));

   nir_ssa_def_rewrite_uses(&intr->dest.ssa, result);
   nir_instr_remove(&intr->instr);
   return true;
}

bool
dxil_nir_lower_loads_stores_to_dxil(nir_shader *nir)
{
   bool progress = false;

   foreach_list_typed(nir_function, func, node, &nir->functions) {
      if (!func->is_entrypoint)
         continue;
      assert(func->impl);

      nir_builder b;
      nir_builder_init(&b, func->impl);

      nir_foreach_block(block, func->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

            switch (intr->intrinsic) {
            case nir_intrinsic_load_deref:
               progress |= lower_load_deref(&b, intr);
               break;
            case nir_intrinsic_load_shared:
            case nir_intrinsic_load_scratch:
               progress |= lower_32b_offset_load(&b, intr);
               break;
            case nir_intrinsic_load_ssbo:
               progress |= lower_load_ssbo(&b, intr);
               break;
            case nir_intrinsic_load_ubo:
               progress |= lower_load_ubo(&b, intr);
               break;
            case nir_intrinsic_store_shared:
            case nir_intrinsic_store_scratch:
               progress |= lower_32b_offset_store(&b, intr);
               break;
            case nir_intrinsic_store_ssbo:
               progress |= lower_store_ssbo(&b, intr);
               break;
            default:
               break;
            }
         }
      }
   }

   return progress;
}

static bool
lower_shared_atomic(nir_builder *b, nir_intrinsic_instr *intr,
                    nir_intrinsic_op dxil_op)
{
   b->cursor = nir_before_instr(&intr->instr);

   assert(intr->src[0].is_ssa);
   nir_ssa_def *offset =
      nir_iadd(b, intr->src[0].ssa, nir_imm_int(b, nir_intrinsic_base(intr)));
   nir_ssa_def *index = nir_ushr(b, offset, nir_imm_int(b, 2));

   nir_intrinsic_instr *atomic = nir_intrinsic_instr_create(b->shader, dxil_op);
   atomic->src[0] = nir_src_for_ssa(index);
   assert(intr->src[1].is_ssa);
   atomic->src[1] = nir_src_for_ssa(intr->src[1].ssa);
   if (dxil_op == nir_intrinsic_shared_atomic_comp_swap_dxil) {
      assert(intr->src[2].is_ssa);
      atomic->src[2] = nir_src_for_ssa(intr->src[2].ssa);
   }
   atomic->num_components = 0;
   nir_ssa_dest_init(&atomic->instr, &atomic->dest, 1, 32, NULL);

   nir_builder_instr_insert(b, &atomic->instr);
   nir_ssa_def_rewrite_uses(&intr->dest.ssa, &atomic->dest.ssa);
   nir_instr_remove(&intr->instr);
   return true;
}

bool
dxil_nir_lower_atomics_to_dxil(nir_shader *nir)
{
   bool progress = false;

   foreach_list_typed(nir_function, func, node, &nir->functions) {
      if (!func->is_entrypoint)
         continue;
      assert(func->impl);

      nir_builder b;
      nir_builder_init(&b, func->impl);

      nir_foreach_block(block, func->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

            switch (intr->intrinsic) {

#define ATOMIC(op)                                                            \
  case nir_intrinsic_shared_atomic_##op:                                     \
     progress |= lower_shared_atomic(&b, intr,                                \
                                     nir_intrinsic_shared_atomic_##op##_dxil); \
     break

            ATOMIC(add);
            ATOMIC(imin);
            ATOMIC(umin);
            ATOMIC(imax);
            ATOMIC(umax);
            ATOMIC(and);
            ATOMIC(or);
            ATOMIC(xor);
            ATOMIC(exchange);
            ATOMIC(comp_swap);

#undef ATOMIC
            default:
               break;
            }
         }
      }
   }

   return progress;
}

static bool
lower_deref_ssbo(nir_builder *b, nir_deref_instr *deref)
{
   assert(nir_deref_mode_is(deref, nir_var_mem_ssbo));
   assert(deref->deref_type == nir_deref_type_var ||
          deref->deref_type == nir_deref_type_cast);
   nir_variable *var = deref->var;

   b->cursor = nir_before_instr(&deref->instr);

   if (deref->deref_type == nir_deref_type_var) {
      /* We turn all deref_var into deref_cast and build a pointer value based on
       * the var binding which encodes the UAV id.
       */
      nir_ssa_def *ptr = nir_imm_int64(b, (uint64_t)var->data.binding << 32);
      nir_deref_instr *deref_cast =
         nir_build_deref_cast(b, ptr, nir_var_mem_ssbo, deref->type,
                              glsl_get_explicit_stride(var->type));
      nir_ssa_def_rewrite_uses(&deref->dest.ssa,
                               &deref_cast->dest.ssa);
      nir_instr_remove(&deref->instr);

      deref = deref_cast;
      return true;
   }
   return false;
}

bool
dxil_nir_lower_deref_ssbo(nir_shader *nir)
{
   bool progress = false;

   foreach_list_typed(nir_function, func, node, &nir->functions) {
      if (!func->is_entrypoint)
         continue;
      assert(func->impl);

      nir_builder b;
      nir_builder_init(&b, func->impl);

      nir_foreach_block(block, func->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);

            if (!nir_deref_mode_is(deref, nir_var_mem_ssbo) ||
                (deref->deref_type != nir_deref_type_var &&
                 deref->deref_type != nir_deref_type_cast))
               continue;

            progress |= lower_deref_ssbo(&b, deref);
         }
      }
   }

   return progress;
}

static bool
lower_alu_deref_srcs(nir_builder *b, nir_alu_instr *alu)
{
   const nir_op_info *info = &nir_op_infos[alu->op];
   bool progress = false;

   b->cursor = nir_before_instr(&alu->instr);

   for (unsigned i = 0; i < info->num_inputs; i++) {
      nir_deref_instr *deref = nir_src_as_deref(alu->src[i].src);

      if (!deref)
         continue;

      nir_deref_path path;
      nir_deref_path_init(&path, deref, NULL);
      nir_deref_instr *root_deref = path.path[0];
      nir_deref_path_finish(&path);

      if (root_deref->deref_type != nir_deref_type_cast)
         continue;

      nir_ssa_def *ptr =
         nir_iadd(b, root_deref->parent.ssa,
                     nir_build_deref_offset(b, deref, cl_type_size_align));
      nir_instr_rewrite_src(&alu->instr, &alu->src[i].src, nir_src_for_ssa(ptr));
      progress = true;
   }

   return progress;
}

bool
dxil_nir_opt_alu_deref_srcs(nir_shader *nir)
{
   bool progress = false;

   foreach_list_typed(nir_function, func, node, &nir->functions) {
      if (!func->is_entrypoint)
         continue;
      assert(func->impl);

      nir_builder b;
      nir_builder_init(&b, func->impl);

      nir_foreach_block(block, func->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;

            nir_alu_instr *alu = nir_instr_as_alu(instr);
            progress |= lower_alu_deref_srcs(&b, alu);
         }
      }
   }

   return progress;
}

static void
cast_phi(nir_builder *b, nir_phi_instr *phi, unsigned new_bit_size)
{
   nir_phi_instr *lowered = nir_phi_instr_create(b->shader);
   int num_components = 0;
   int old_bit_size = phi->dest.ssa.bit_size;

   nir_foreach_phi_src(src, phi) {
      assert(num_components == 0 || num_components == src->src.ssa->num_components);
      num_components = src->src.ssa->num_components;

      b->cursor = nir_after_instr_and_phis(src->src.ssa->parent_instr);

      nir_ssa_def *cast = nir_u2uN(b, src->src.ssa, new_bit_size);

      nir_phi_instr_add_src(lowered, src->pred, nir_src_for_ssa(cast));
   }

   nir_ssa_dest_init(&lowered->instr, &lowered->dest,
                     num_components, new_bit_size, NULL);

   b->cursor = nir_before_instr(&phi->instr);
   nir_builder_instr_insert(b, &lowered->instr);

   b->cursor = nir_after_phis(nir_cursor_current_block(b->cursor));
   nir_ssa_def *result = nir_u2uN(b, &lowered->dest.ssa, old_bit_size);

   nir_ssa_def_rewrite_uses(&phi->dest.ssa, result);
   nir_instr_remove(&phi->instr);
}

static bool
upcast_phi_impl(nir_function_impl *impl, unsigned min_bit_size)
{
   nir_builder b;
   nir_builder_init(&b, impl);
   bool progress = false;

   nir_foreach_block_reverse(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_phi)
            continue;

         nir_phi_instr *phi = nir_instr_as_phi(instr);
         assert(phi->dest.is_ssa);

         if (phi->dest.ssa.bit_size == 1 ||
             phi->dest.ssa.bit_size >= min_bit_size)
            continue;

         cast_phi(&b, phi, min_bit_size);
         progress = true;
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

bool
dxil_nir_lower_upcast_phis(nir_shader *shader, unsigned min_bit_size)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= upcast_phi_impl(function->impl, min_bit_size);
   }

   return progress;
}

struct dxil_nir_split_clip_cull_distance_params {
   nir_variable *new_var[2];
   nir_shader *shader;
};

/* In GLSL and SPIR-V, clip and cull distance are arrays of floats (with a limit of 8).
 * In DXIL, clip and cull distances are up to 2 float4s combined.
 * Coming from GLSL, we can request this 2 float4 format, but coming from SPIR-V,
 * we can't, and have to accept a "compact" array of scalar floats.
 *
 * To help emitting a valid input signature for this case, split the variables so that they
 * match what we need to put in the signature (e.g. { float clip[4]; float clip1; float cull[3]; })
 */
static bool
dxil_nir_split_clip_cull_distance_instr(nir_builder *b,
                                        nir_instr *instr,
                                        void *cb_data)
{
   struct dxil_nir_split_clip_cull_distance_params *params = cb_data;

   if (instr->type != nir_instr_type_deref)
      return false;

   nir_deref_instr *deref = nir_instr_as_deref(instr);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (!var ||
       var->data.location < VARYING_SLOT_CLIP_DIST0 ||
       var->data.location > VARYING_SLOT_CULL_DIST1 ||
       !var->data.compact)
      return false;

   unsigned new_var_idx = var->data.mode == nir_var_shader_in ? 0 : 1;
   nir_variable *new_var = params->new_var[new_var_idx];

   /* The location should only be inside clip distance, because clip
    * and cull should've been merged by nir_lower_clip_cull_distance_arrays()
    */
   assert(var->data.location == VARYING_SLOT_CLIP_DIST0 ||
          var->data.location == VARYING_SLOT_CLIP_DIST1);

   /* The deref chain to the clip/cull variables should be simple, just the
    * var and an array with a constant index, otherwise more lowering/optimization
    * might be needed before this pass, e.g. copy prop, lower_io_to_temporaries,
    * split_var_copies, and/or lower_var_copies. In the case of arrayed I/O like
    * inputs to the tessellation or geometry stages, there might be a second level
    * of array index.
    */
   assert(deref->deref_type == nir_deref_type_var ||
          deref->deref_type == nir_deref_type_array);

   b->cursor = nir_before_instr(instr);
   unsigned arrayed_io_length = 0;
   const struct glsl_type *old_type = var->type;
   if (nir_is_arrayed_io(var, b->shader->info.stage)) {
      arrayed_io_length = glsl_array_size(old_type);
      old_type = glsl_get_array_element(old_type);
   }
   if (!new_var) {
      /* Update lengths for new and old vars */
      int old_length = glsl_array_size(old_type);
      int new_length = (old_length + var->data.location_frac) - 4;
      old_length -= new_length;

      /* The existing variable fits in the float4 */
      if (new_length <= 0)
         return false;

      new_var = nir_variable_clone(var, params->shader);
      nir_shader_add_variable(params->shader, new_var);
      assert(glsl_get_base_type(glsl_get_array_element(old_type)) == GLSL_TYPE_FLOAT);
      var->type = glsl_array_type(glsl_float_type(), old_length, 0);
      new_var->type = glsl_array_type(glsl_float_type(), new_length, 0);
      if (arrayed_io_length) {
         var->type = glsl_array_type(var->type, arrayed_io_length, 0);
         new_var->type = glsl_array_type(new_var->type, arrayed_io_length, 0);
      }
      new_var->data.location++;
      new_var->data.location_frac = 0;
      params->new_var[new_var_idx] = new_var;
   }

   /* Update the type for derefs of the old var */
   if (deref->deref_type == nir_deref_type_var) {
      deref->type = var->type;
      return false;
   }

   if (glsl_type_is_array(deref->type)) {
      assert(arrayed_io_length > 0);
      deref->type = glsl_get_array_element(var->type);
      return false;
   }

   assert(glsl_get_base_type(deref->type) == GLSL_TYPE_FLOAT);

   nir_const_value *index = nir_src_as_const_value(deref->arr.index);
   assert(index);

   /* Treat this array as a vector starting at the component index in location_frac,
    * so if location_frac is 1 and index is 0, then it's accessing the 'y' component
    * of the vector. If index + location_frac is >= 4, there's no component there,
    * so we need to add a new variable and adjust the index.
    */
   unsigned total_index = index->u32 + var->data.location_frac;
   if (total_index < 4)
      return false;

   nir_deref_instr *new_var_deref = nir_build_deref_var(b, new_var);
   nir_deref_instr *new_intermediate_deref = new_var_deref;
   if (arrayed_io_length) {
      nir_deref_instr *parent = nir_src_as_deref(deref->parent);
      assert(parent->deref_type == nir_deref_type_array);
      new_intermediate_deref = nir_build_deref_array(b, new_intermediate_deref, parent->arr.index.ssa);
   }
   nir_deref_instr *new_array_deref = nir_build_deref_array(b, new_intermediate_deref, nir_imm_int(b, total_index % 4));
   nir_ssa_def_rewrite_uses(&deref->dest.ssa, &new_array_deref->dest.ssa);
   return true;
}

bool
dxil_nir_split_clip_cull_distance(nir_shader *shader)
{
   struct dxil_nir_split_clip_cull_distance_params params = {
      .new_var = { NULL, NULL },
      .shader = shader,
   };
   nir_shader_instructions_pass(shader,
                                dxil_nir_split_clip_cull_distance_instr,
                                nir_metadata_block_index |
                                nir_metadata_dominance |
                                nir_metadata_loop_analysis,
                                &params);
   return params.new_var[0] != NULL || params.new_var[1] != NULL;
}

static bool
dxil_nir_lower_double_math_instr(nir_builder *b,
                                 nir_instr *instr,
                                 UNUSED void *cb_data)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);

   /* TODO: See if we can apply this explicitly to packs/unpacks that are then
    * used as a double. As-is, if we had an app explicitly do a 64bit integer op,
    * then try to bitcast to double (not expressible in HLSL, but it is in other
    * source languages), this would unpack the integer and repack as a double, when
    * we probably want to just send the bitcast through to the backend.
    */

   b->cursor = nir_before_instr(&alu->instr);

   bool progress = false;
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i) {
      if (nir_alu_type_get_base_type(nir_op_infos[alu->op].input_types[i]) == nir_type_float &&
          alu->src[i].src.ssa->bit_size == 64) {
         unsigned num_components = nir_op_infos[alu->op].input_sizes[i];
         if (!num_components)
            num_components = alu->dest.dest.ssa.num_components;
         nir_ssa_def *components[NIR_MAX_VEC_COMPONENTS];
         for (unsigned c = 0; c < num_components; ++c) {
            nir_ssa_def *packed_double = nir_channel(b, alu->src[i].src.ssa, alu->src[i].swizzle[c]);
            nir_ssa_def *unpacked_double = nir_unpack_64_2x32(b, packed_double);
            components[c] = nir_pack_double_2x32_dxil(b, unpacked_double);
            alu->src[i].swizzle[c] = c;
         }
         nir_instr_rewrite_src_ssa(instr, &alu->src[i].src, nir_vec(b, components, num_components));
         progress = true;
      }
   }

   if (nir_alu_type_get_base_type(nir_op_infos[alu->op].output_type) == nir_type_float &&
       alu->dest.dest.ssa.bit_size == 64) {
      b->cursor = nir_after_instr(&alu->instr);
      nir_ssa_def *components[NIR_MAX_VEC_COMPONENTS];
      for (unsigned c = 0; c < alu->dest.dest.ssa.num_components; ++c) {
         nir_ssa_def *packed_double = nir_channel(b, &alu->dest.dest.ssa, c);
         nir_ssa_def *unpacked_double = nir_unpack_double_2x32_dxil(b, packed_double);
         components[c] = nir_pack_64_2x32(b, unpacked_double);
      }
      nir_ssa_def *repacked_dvec = nir_vec(b, components, alu->dest.dest.ssa.num_components);
      nir_ssa_def_rewrite_uses_after(&alu->dest.dest.ssa, repacked_dvec, repacked_dvec->parent_instr);
      progress = true;
   }

   return progress;
}

bool
dxil_nir_lower_double_math(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader,
                                       dxil_nir_lower_double_math_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance |
                                       nir_metadata_loop_analysis,
                                       NULL);
}

typedef struct {
   gl_system_value *values;
   uint32_t count;
} zero_system_values_state;

static bool
lower_system_value_to_zero_filter(const nir_instr* instr, const void* cb_state)
{
   if (instr->type != nir_instr_type_intrinsic) {
      return false;
   }

   nir_intrinsic_instr* intrin = nir_instr_as_intrinsic(instr);

   /* All the intrinsics we care about are loads */
   if (!nir_intrinsic_infos[intrin->intrinsic].has_dest)
      return false;

   assert(intrin->dest.is_ssa);

   zero_system_values_state* state = (zero_system_values_state*)cb_state;
   for (uint32_t i = 0; i < state->count; ++i) {
      gl_system_value value = state->values[i];
      nir_intrinsic_op value_op = nir_intrinsic_from_system_value(value);

      if (intrin->intrinsic == value_op) {
         return true;
      } else if (intrin->intrinsic == nir_intrinsic_load_deref) {
         nir_deref_instr* deref = nir_src_as_deref(intrin->src[0]);
         if (!nir_deref_mode_is(deref, nir_var_system_value))
            return false;

         nir_variable* var = deref->var;
         if (var->data.location == value) {
            return true;
         }
      }
   }

   return false;
}

static nir_ssa_def*
lower_system_value_to_zero_instr(nir_builder* b, nir_instr* instr, void* _state)
{
   return nir_imm_int(b, 0);
}

bool
dxil_nir_lower_system_values_to_zero(nir_shader* shader,
                                     gl_system_value* system_values,
                                     uint32_t count)
{
   zero_system_values_state state = { system_values, count };
   return nir_shader_lower_instructions(shader,
      lower_system_value_to_zero_filter,
      lower_system_value_to_zero_instr,
      &state);
}

static void
lower_load_local_group_size(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_after_instr(&intr->instr);

   nir_const_value v[3] = {
      nir_const_value_for_int(b->shader->info.workgroup_size[0], 32),
      nir_const_value_for_int(b->shader->info.workgroup_size[1], 32),
      nir_const_value_for_int(b->shader->info.workgroup_size[2], 32)
   };
   nir_ssa_def *size = nir_build_imm(b, 3, 32, v);
   nir_ssa_def_rewrite_uses(&intr->dest.ssa, size);
   nir_instr_remove(&intr->instr);
}

static bool
lower_system_values_impl(nir_builder *b, nir_instr *instr, void *_state)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_workgroup_size:
      lower_load_local_group_size(b, intr);
      return true;
   default:
      return false;
   }
}

bool
dxil_nir_lower_system_values(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, lower_system_values_impl,
      nir_metadata_block_index | nir_metadata_dominance | nir_metadata_loop_analysis, NULL);
}

static const struct glsl_type *
get_bare_samplers_for_type(const struct glsl_type *type, bool is_shadow)
{
   const struct glsl_type *base_sampler_type =
      is_shadow ?
      glsl_bare_shadow_sampler_type() : glsl_bare_sampler_type();
   return glsl_type_wrap_in_arrays(base_sampler_type, type);
}

static const struct glsl_type *
get_textures_for_sampler_type(const struct glsl_type *type)
{
   return glsl_type_wrap_in_arrays(
      glsl_sampler_type_to_texture(
         glsl_without_array(type)), type);
}

static bool
redirect_sampler_derefs(struct nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);

   int sampler_idx = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   if (sampler_idx == -1) {
      /* No sampler deref - does this instruction even need a sampler? If not,
       * sampler_index doesn't necessarily point to a sampler, so early-out.
       */
      if (!nir_tex_instr_need_sampler(tex))
         return false;

      /* No derefs but needs a sampler, must be using indices */
      nir_variable *bare_sampler = _mesa_hash_table_u64_search(data, tex->sampler_index);

      /* Already have a bare sampler here */
      if (bare_sampler)
         return false;

      nir_variable *old_sampler = NULL;
      nir_foreach_variable_with_modes(var, b->shader, nir_var_uniform) {
         if (var->data.binding <= tex->sampler_index &&
             var->data.binding + glsl_type_get_sampler_count(var->type) >
                tex->sampler_index) {

            /* Already have a bare sampler for this binding and it is of the
             * correct type, add it to the table */
            if (glsl_type_is_bare_sampler(glsl_without_array(var->type)) &&
                glsl_sampler_type_is_shadow(glsl_without_array(var->type)) ==
                   tex->is_shadow) {
               _mesa_hash_table_u64_insert(data, tex->sampler_index, var);
               return false;
            }

            old_sampler = var;
         }
      }

      assert(old_sampler);

      /* Clone the original sampler to a bare sampler of the correct type */
      bare_sampler = nir_variable_clone(old_sampler, b->shader);
      nir_shader_add_variable(b->shader, bare_sampler);

      bare_sampler->type =
         get_bare_samplers_for_type(old_sampler->type, tex->is_shadow);
      _mesa_hash_table_u64_insert(data, tex->sampler_index, bare_sampler);
      return true;
   }

   /* Using derefs, means we have to rewrite the deref chain in addition to cloning */
   nir_deref_instr *final_deref = nir_src_as_deref(tex->src[sampler_idx].src);
   nir_deref_path path;
   nir_deref_path_init(&path, final_deref, NULL);

   nir_deref_instr *old_tail = path.path[0];
   assert(old_tail->deref_type == nir_deref_type_var);
   nir_variable *old_var = old_tail->var;
   if (glsl_type_is_bare_sampler(glsl_without_array(old_var->type)) &&
       glsl_sampler_type_is_shadow(glsl_without_array(old_var->type)) ==
          tex->is_shadow) {
      nir_deref_path_finish(&path);
      return false;
   }

   uint64_t var_key = ((uint64_t)old_var->data.descriptor_set << 32) |
                      old_var->data.binding;
   nir_variable *new_var = _mesa_hash_table_u64_search(data, var_key);
   if (!new_var) {
      new_var = nir_variable_clone(old_var, b->shader);
      nir_shader_add_variable(b->shader, new_var);
      new_var->type = 
         get_bare_samplers_for_type(old_var->type, tex->is_shadow);
      _mesa_hash_table_u64_insert(data, var_key, new_var);
   }

   b->cursor = nir_after_instr(&old_tail->instr);
   nir_deref_instr *new_tail = nir_build_deref_var(b, new_var);

   for (unsigned i = 1; path.path[i]; ++i) {
      b->cursor = nir_after_instr(&path.path[i]->instr);
      new_tail = nir_build_deref_follower(b, new_tail, path.path[i]);
   }

   nir_deref_path_finish(&path);
   nir_instr_rewrite_src_ssa(&tex->instr, &tex->src[sampler_idx].src, &new_tail->dest.ssa);
   return true;
}

static bool
redirect_texture_derefs(struct nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);

   int texture_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (texture_idx == -1) {
      /* No derefs, must be using indices */
      nir_variable *bare_sampler = _mesa_hash_table_u64_search(data, tex->texture_index);

      /* Already have a texture here */
      if (bare_sampler)
         return false;

      nir_variable *typed_sampler = NULL;
      nir_foreach_variable_with_modes(var, b->shader, nir_var_uniform) {
         if (var->data.binding <= tex->texture_index &&
             var->data.binding + glsl_type_get_texture_count(var->type) > tex->texture_index) {
            /* Already have a texture for this binding, add it to the table */
            _mesa_hash_table_u64_insert(data, tex->texture_index, var);
            return false;
         }

         if (var->data.binding <= tex->texture_index &&
             var->data.binding + glsl_type_get_sampler_count(var->type) > tex->texture_index &&
             !glsl_type_is_bare_sampler(glsl_without_array(var->type))) {
            typed_sampler = var;
         }
      }

      /* Clone the typed sampler to a texture and we're done */
      assert(typed_sampler);
      bare_sampler = nir_variable_clone(typed_sampler, b->shader);
      bare_sampler->type = get_textures_for_sampler_type(typed_sampler->type);
      nir_shader_add_variable(b->shader, bare_sampler);
      _mesa_hash_table_u64_insert(data, tex->texture_index, bare_sampler);
      return true;
   }

   /* Using derefs, means we have to rewrite the deref chain in addition to cloning */
   nir_deref_instr *final_deref = nir_src_as_deref(tex->src[texture_idx].src);
   nir_deref_path path;
   nir_deref_path_init(&path, final_deref, NULL);

   nir_deref_instr *old_tail = path.path[0];
   assert(old_tail->deref_type == nir_deref_type_var);
   nir_variable *old_var = old_tail->var;
   if (glsl_type_is_texture(glsl_without_array(old_var->type)) ||
       glsl_type_is_image(glsl_without_array(old_var->type))) {
      nir_deref_path_finish(&path);
      return false;
   }

   uint64_t var_key = ((uint64_t)old_var->data.descriptor_set << 32) |
                      old_var->data.binding;
   nir_variable *new_var = _mesa_hash_table_u64_search(data, var_key);
   if (!new_var) {
      new_var = nir_variable_clone(old_var, b->shader);
      new_var->type = get_textures_for_sampler_type(old_var->type);
      nir_shader_add_variable(b->shader, new_var);
      _mesa_hash_table_u64_insert(data, var_key, new_var);
   }

   b->cursor = nir_after_instr(&old_tail->instr);
   nir_deref_instr *new_tail = nir_build_deref_var(b, new_var);

   for (unsigned i = 1; path.path[i]; ++i) {
      b->cursor = nir_after_instr(&path.path[i]->instr);
      new_tail = nir_build_deref_follower(b, new_tail, path.path[i]);
   }

   nir_deref_path_finish(&path);
   nir_instr_rewrite_src_ssa(&tex->instr, &tex->src[texture_idx].src, &new_tail->dest.ssa);

   return true;
}

bool
dxil_nir_split_typed_samplers(nir_shader *nir)
{
   struct hash_table_u64 *hash_table = _mesa_hash_table_u64_create(NULL);

   bool progress = nir_shader_instructions_pass(nir, redirect_sampler_derefs,
      nir_metadata_block_index | nir_metadata_dominance | nir_metadata_loop_analysis, hash_table);

   _mesa_hash_table_u64_clear(hash_table);

   progress |= nir_shader_instructions_pass(nir, redirect_texture_derefs,
      nir_metadata_block_index | nir_metadata_dominance | nir_metadata_loop_analysis, hash_table);

   _mesa_hash_table_u64_destroy(hash_table);
   return progress;
}


static bool
lower_sysval_to_load_input_impl(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   gl_system_value sysval = SYSTEM_VALUE_MAX;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_front_face:
      sysval = SYSTEM_VALUE_FRONT_FACE;
      break;
   case nir_intrinsic_load_instance_id:
      sysval = SYSTEM_VALUE_INSTANCE_ID;
      break;
   case nir_intrinsic_load_vertex_id_zero_base:
      sysval = SYSTEM_VALUE_VERTEX_ID_ZERO_BASE;
      break;
   default:
      return false;
   }

   nir_variable **sysval_vars = (nir_variable **)data;
   nir_variable *var = sysval_vars[sysval];
   assert(var);

   const nir_alu_type dest_type = (sysval == SYSTEM_VALUE_FRONT_FACE)
      ? nir_type_uint32 : nir_get_nir_type_for_glsl_type(var->type);
   const unsigned bit_size = (sysval == SYSTEM_VALUE_FRONT_FACE)
      ? 32 : intr->dest.ssa.bit_size;

   b->cursor = nir_before_instr(instr);
   nir_ssa_def *result = nir_build_load_input(b, intr->dest.ssa.num_components, bit_size, nir_imm_int(b, 0),
      .base = var->data.driver_location, .dest_type = dest_type);

   /* The nir_type_uint32 is really a nir_type_bool32, but that type is very
    * inconvenient at this point during compilation.  Convert to
    * nir_type_bool1 by comparing with zero.
    */
   if (sysval == SYSTEM_VALUE_FRONT_FACE)
      result = nir_ine_imm(b, result, 0);

   nir_ssa_def_rewrite_uses(&intr->dest.ssa, result);
   return true;
}

bool
dxil_nir_lower_sysval_to_load_input(nir_shader *s, nir_variable **sysval_vars)
{
   return nir_shader_instructions_pass(s, lower_sysval_to_load_input_impl,
      nir_metadata_block_index | nir_metadata_dominance, sysval_vars);
}

/* Comparison function to sort io values so that first come normal varyings,
 * then system values, and then system generated values.
 */
static int
variable_location_cmp(const nir_variable* a, const nir_variable* b)
{
   // Sort by stream, driver_location, location, location_frac, then index
   unsigned a_location = a->data.location;
   if (a_location >= VARYING_SLOT_PATCH0)
      a_location -= VARYING_SLOT_PATCH0;
   unsigned b_location = b->data.location;
   if (b_location >= VARYING_SLOT_PATCH0)
      b_location -= VARYING_SLOT_PATCH0;
   unsigned a_stream = a->data.stream & ~NIR_STREAM_PACKED;
   unsigned b_stream = b->data.stream & ~NIR_STREAM_PACKED;
   return a_stream != b_stream ?
            a_stream - b_stream :
            a->data.driver_location != b->data.driver_location ?
               a->data.driver_location - b->data.driver_location :
               a_location !=  b_location ?
                  a_location - b_location :
                  a->data.location_frac != b->data.location_frac ?
                     a->data.location_frac - b->data.location_frac :
                     a->data.index - b->data.index;
}

/* Order varyings according to driver location */
uint64_t
dxil_sort_by_driver_location(nir_shader* s, nir_variable_mode modes)
{
   nir_sort_variables_with_modes(s, variable_location_cmp, modes);

   uint64_t result = 0;
   nir_foreach_variable_with_modes(var, s, modes) {
      result |= 1ull << var->data.location;
   }
   return result;
}

/* Sort PS outputs so that color outputs come first */
void
dxil_sort_ps_outputs(nir_shader* s)
{
   nir_foreach_variable_with_modes_safe(var, s, nir_var_shader_out) {
      /* We use the driver_location here to avoid introducing a new
       * struct or member variable here. The true, updated driver location
       * will be written below, after sorting */
      switch (var->data.location) {
      case FRAG_RESULT_DEPTH:
         var->data.driver_location = 1;
         break;
      case FRAG_RESULT_STENCIL:
         var->data.driver_location = 2;
         break;
      case FRAG_RESULT_SAMPLE_MASK:
         var->data.driver_location = 3;
         break;
      default:
         var->data.driver_location = 0;
      }
   }

   nir_sort_variables_with_modes(s, variable_location_cmp,
                                 nir_var_shader_out);

   unsigned driver_loc = 0;
   nir_foreach_variable_with_modes(var, s, nir_var_shader_out) {
      var->data.driver_location = driver_loc++;
   }
}

enum dxil_sysvalue_type {
   DXIL_NO_SYSVALUE = 0,
   DXIL_USED_SYSVALUE,
   DXIL_SYSVALUE,
   DXIL_GENERATED_SYSVALUE
};

static enum dxil_sysvalue_type
nir_var_to_dxil_sysvalue_type(nir_variable *var, uint64_t other_stage_mask)
{
   switch (var->data.location) {
   case VARYING_SLOT_FACE:
      return DXIL_GENERATED_SYSVALUE;
   case VARYING_SLOT_POS:
   case VARYING_SLOT_PRIMITIVE_ID:
   case VARYING_SLOT_CLIP_DIST0:
   case VARYING_SLOT_CLIP_DIST1:
   case VARYING_SLOT_PSIZ:
   case VARYING_SLOT_TESS_LEVEL_INNER:
   case VARYING_SLOT_TESS_LEVEL_OUTER:
   case VARYING_SLOT_VIEWPORT:
   case VARYING_SLOT_LAYER:
   case VARYING_SLOT_VIEW_INDEX:
      if (!((1ull << var->data.location) & other_stage_mask))
         return DXIL_SYSVALUE;
      return DXIL_USED_SYSVALUE;
   default:
      return DXIL_NO_SYSVALUE;
   }
}

/* Order between stage values so that normal varyings come first,
 * then sysvalues and then system generated values.
 */
uint64_t
dxil_reassign_driver_locations(nir_shader* s, nir_variable_mode modes,
   uint64_t other_stage_mask)
{
   nir_foreach_variable_with_modes_safe(var, s, modes) {
      /* We use the driver_location here to avoid introducing a new
       * struct or member variable here. The true, updated driver location
       * will be written below, after sorting */
      var->data.driver_location = nir_var_to_dxil_sysvalue_type(var, other_stage_mask);
   }

   nir_sort_variables_with_modes(s, variable_location_cmp, modes);

   uint64_t result = 0;
   unsigned driver_loc = 0, driver_patch_loc = 0;
   nir_foreach_variable_with_modes(var, s, modes) {
      if (var->data.location < 64)
         result |= 1ull << var->data.location;
      /* Overlap patches with non-patch */
      var->data.driver_location = var->data.patch ?
         driver_patch_loc++ : driver_loc++;
   }
   return result;
}

static bool
lower_ubo_array_one_to_static(struct nir_builder *b, nir_instr *inst,
                              void *cb_data)
{
   if (inst->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(inst);

   if (intrin->intrinsic != nir_intrinsic_load_vulkan_descriptor)
      return false;

   nir_variable *var =
      nir_get_binding_variable(b->shader, nir_chase_binding(intrin->src[0]));

   if (!var)
      return false;

   if (!glsl_type_is_array(var->type) || glsl_array_size(var->type) != 1)
      return false;

   nir_intrinsic_instr *index = nir_src_as_intrinsic(intrin->src[0]);
   /* We currently do not support reindex */
   assert(index && index->intrinsic == nir_intrinsic_vulkan_resource_index);

   if (nir_src_is_const(index->src[0]) && nir_src_as_uint(index->src[0]) == 0)
      return false;

   if (nir_intrinsic_desc_type(index) != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
      return false;

   b->cursor = nir_instr_remove(&index->instr);

   // Indexing out of bounds on array of UBOs is considered undefined
   // behavior. Therefore, we just hardcode all the index to 0.
   uint8_t bit_size = index->dest.ssa.bit_size;
   nir_ssa_def *zero = nir_imm_intN_t(b, 0, bit_size);
   nir_ssa_def *dest =
      nir_vulkan_resource_index(b, index->num_components, bit_size, zero,
                                .desc_set = nir_intrinsic_desc_set(index),
                                .binding = nir_intrinsic_binding(index),
                                .desc_type = nir_intrinsic_desc_type(index));

   nir_ssa_def_rewrite_uses(&index->dest.ssa, dest);

   return true;
}

bool
dxil_nir_lower_ubo_array_one_to_static(nir_shader *s)
{
   bool progress = nir_shader_instructions_pass(
      s, lower_ubo_array_one_to_static, nir_metadata_none, NULL);

   return progress;
}

static bool
is_fquantize2f16(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   return alu->op == nir_op_fquantize2f16;
}

static nir_ssa_def *
lower_fquantize2f16(struct nir_builder *b, nir_instr *instr, void *data)
{
   /*
    * SpvOpQuantizeToF16 documentation says:
    *
    * "
    * If Value is an infinity, the result is the same infinity.
    * If Value is a NaN, the result is a NaN, but not necessarily the same NaN.
    * If Value is positive with a magnitude too large to represent as a 16-bit
    * floating-point value, the result is positive infinity. If Value is negative
    * with a magnitude too large to represent as a 16-bit floating-point value,
    * the result is negative infinity. If the magnitude of Value is too small to
    * represent as a normalized 16-bit floating-point value, the result may be
    * either +0 or -0.
    * "
    *
    * which we turn into:
    *
    *   if (val < MIN_FLOAT16)
    *      return -INFINITY;
    *   else if (val > MAX_FLOAT16)
    *      return -INFINITY;
    *   else if (fabs(val) < SMALLEST_NORMALIZED_FLOAT16 && sign(val) != 0)
    *      return -0.0f;
    *   else if (fabs(val) < SMALLEST_NORMALIZED_FLOAT16 && sign(val) == 0)
    *      return +0.0f;
    *   else
    *      return round(val);
    */
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_ssa_def *src =
      nir_ssa_for_src(b, alu->src[0].src, nir_src_num_components(alu->src[0].src));

   nir_ssa_def *neg_inf_cond =
      nir_flt(b, src, nir_imm_float(b, -65504.0f));
   nir_ssa_def *pos_inf_cond =
      nir_flt(b, nir_imm_float(b, 65504.0f), src);
   nir_ssa_def *zero_cond =
      nir_flt(b, nir_fabs(b, src), nir_imm_float(b, ldexpf(1.0, -14)));
   nir_ssa_def *zero = nir_iand_imm(b, src, 1 << 31);
   nir_ssa_def *round = nir_iand_imm(b, src, ~BITFIELD_MASK(13));

   nir_ssa_def *res =
      nir_bcsel(b, neg_inf_cond, nir_imm_float(b, -INFINITY), round);
   res = nir_bcsel(b, pos_inf_cond, nir_imm_float(b, INFINITY), res);
   res = nir_bcsel(b, zero_cond, zero, res);
   return res;
}

bool
dxil_nir_lower_fquantize2f16(nir_shader *s)
{
   return nir_shader_lower_instructions(s, is_fquantize2f16, lower_fquantize2f16, NULL);
}

static bool
fix_io_uint_deref_types(struct nir_builder *builder, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_deref)
      return false;

   nir_deref_instr *deref = nir_instr_as_deref(instr);
   nir_variable *var =
      deref->deref_type == nir_deref_type_var ? deref->var : NULL;

   if (var == data) {
      deref->type = var->type;
      return true;
   }

   return false;
}

static bool
fix_io_uint_type(nir_shader *s, nir_variable_mode modes, int slot)
{
   nir_variable *fixed_var = NULL;
   nir_foreach_variable_with_modes(var, s, modes) {
      if (var->data.location == slot) {
         if (var->type == glsl_uint_type())
            return false;

         assert(var->type == glsl_int_type());
         var->type = glsl_uint_type();
         fixed_var = var;
         break;
      }
   }

   assert(fixed_var);

   return nir_shader_instructions_pass(s, fix_io_uint_deref_types,
                                       nir_metadata_all, fixed_var);
}

bool
dxil_nir_fix_io_uint_type(nir_shader *s, uint64_t in_mask, uint64_t out_mask)
{
   if (!(s->info.outputs_written & out_mask) &&
       !(s->info.inputs_read & in_mask))
      return false;

   bool progress = false;

   while (in_mask) {
      int slot = u_bit_scan64(&in_mask);
      progress |= (s->info.inputs_read & (1ull << slot)) &&
                  fix_io_uint_type(s, nir_var_shader_in, slot);
   }

   while (out_mask) {
      int slot = u_bit_scan64(&out_mask);
      progress |= (s->info.outputs_written & (1ull << slot)) &&
                  fix_io_uint_type(s, nir_var_shader_out, slot);
   }

   return progress;
}

struct remove_after_discard_state {
   struct nir_block *active_block;
};

static bool
remove_after_discard(struct nir_builder *builder, nir_instr *instr,
                      void *cb_data)
{
   struct remove_after_discard_state *state = cb_data;
   if (instr->block == state->active_block) {
      nir_instr_remove_v(instr);
      return true;
   }

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   if (intr->intrinsic != nir_intrinsic_discard &&
       intr->intrinsic != nir_intrinsic_terminate &&
       intr->intrinsic != nir_intrinsic_discard_if &&
       intr->intrinsic != nir_intrinsic_terminate_if)
      return false;

   state->active_block = instr->block;

   return false;
}

static bool
lower_kill(struct nir_builder *builder, nir_instr *instr, void *_cb_data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   if (intr->intrinsic != nir_intrinsic_discard &&
       intr->intrinsic != nir_intrinsic_terminate &&
       intr->intrinsic != nir_intrinsic_discard_if &&
       intr->intrinsic != nir_intrinsic_terminate_if)
      return false;

   builder->cursor = nir_instr_remove(instr);
   if (intr->intrinsic == nir_intrinsic_discard ||
       intr->intrinsic == nir_intrinsic_terminate) {
      nir_demote(builder);
   } else {
      assert(intr->src[0].is_ssa);
      nir_demote_if(builder, intr->src[0].ssa);
   }

   nir_jump(builder, nir_jump_return);

   return true;
}

bool
dxil_nir_lower_discard_and_terminate(nir_shader *s)
{
   if (s->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   // This pass only works if all functions have been inlined
   assert(exec_list_length(&s->functions) == 1);
   struct remove_after_discard_state state;
   state.active_block = NULL;
   nir_shader_instructions_pass(s, remove_after_discard, nir_metadata_none,
                                &state);
   return nir_shader_instructions_pass(s, lower_kill, nir_metadata_none,
                                       NULL);
}

static bool
update_writes(struct nir_builder *b, nir_instr *instr, void *_state)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics io = nir_intrinsic_io_semantics(intr);
   if (io.location != VARYING_SLOT_POS)
      return false;

   nir_ssa_def *src = intr->src[0].ssa;
   unsigned write_mask = nir_intrinsic_write_mask(intr);
   if (src->num_components == 4 && write_mask == 0xf)
      return false;

   b->cursor = nir_before_instr(instr);
   unsigned first_comp = nir_intrinsic_component(intr);
   nir_ssa_def *channels[4] = { NULL, NULL, NULL, NULL };
   assert(first_comp + src->num_components <= ARRAY_SIZE(channels));
   for (unsigned i = 0; i < src->num_components; ++i)
      if (write_mask & (1 << i))
         channels[i + first_comp] = nir_channel(b, src, i);
   for (unsigned i = 0; i < 4; ++i)
      if (!channels[i])
         channels[i] = nir_imm_intN_t(b, 0, src->bit_size);

   intr->num_components = 4;
   nir_instr_rewrite_src_ssa(instr, &intr->src[0], nir_vec(b, channels, 4));
   nir_intrinsic_set_component(intr, 0);
   nir_intrinsic_set_write_mask(intr, 0xf);
   return true;
}

bool
dxil_nir_ensure_position_writes(nir_shader *s)
{
   if (s->info.stage != MESA_SHADER_VERTEX &&
       s->info.stage != MESA_SHADER_GEOMETRY &&
       s->info.stage != MESA_SHADER_TESS_EVAL)
      return false;
   if ((s->info.outputs_written & VARYING_BIT_POS) == 0)
      return false;

   return nir_shader_instructions_pass(s, update_writes,
                                       nir_metadata_block_index | nir_metadata_dominance,
                                       NULL);
}

static bool
is_sample_pos(const nir_instr *instr, const void *_data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return intr->intrinsic == nir_intrinsic_load_sample_pos;
}

static nir_ssa_def *
lower_sample_pos(nir_builder *b, nir_instr *instr, void *_data)
{
   return nir_load_sample_pos_from_id(b, 32, nir_load_sample_id(b));
}

bool
dxil_nir_lower_sample_pos(nir_shader *s)
{
   return nir_shader_lower_instructions(s, is_sample_pos, lower_sample_pos, NULL);
}

static bool
lower_subgroup_id(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_subgroup_id)
      return false;

   nir_ssa_def **subgroup_id = (nir_ssa_def **)data;
   if (*subgroup_id == NULL) {
      nir_variable *subgroup_id_counter = nir_variable_create(b->shader, nir_var_mem_shared, glsl_uint_type(), "dxil_SubgroupID_counter");
      nir_variable *subgroup_id_local = nir_local_variable_create(b->impl, glsl_uint_type(), "dxil_SubgroupID_local");
      b->cursor = nir_before_block(nir_start_block(b->impl));
      nir_store_var(b, subgroup_id_local, nir_imm_int(b, 0), 1);

      nir_deref_instr *counter_deref = nir_build_deref_var(b, subgroup_id_counter);
      nir_ssa_def *tid = nir_load_local_invocation_index(b);
      nir_if *nif = nir_push_if(b, nir_ieq_imm(b, tid, 0));
      nir_store_deref(b, counter_deref, nir_imm_int(b, 0), 1);
      nir_pop_if(b, nif);

      nir_scoped_memory_barrier(b, NIR_SCOPE_WORKGROUP, NIR_MEMORY_ACQ_REL, nir_var_mem_shared);

      nif = nir_push_if(b, nir_elect(b, 1));
      nir_ssa_def *subgroup_id_first_thread = nir_deref_atomic_add(b, 32, &counter_deref->dest.ssa, nir_imm_int(b, 1));
      nir_store_var(b, subgroup_id_local, subgroup_id_first_thread, 1);
      nir_pop_if(b, nif);

      nir_ssa_def *subgroup_id_loaded = nir_load_var(b, subgroup_id_local);
      *subgroup_id = nir_read_first_invocation(b, subgroup_id_loaded);
   }
   nir_ssa_def_rewrite_uses(&intr->dest.ssa, *subgroup_id);
   return true;
}

bool
dxil_nir_lower_subgroup_id(nir_shader *s)
{
   nir_ssa_def *subgroup_id = NULL;
   return nir_shader_instructions_pass(s, lower_subgroup_id, nir_metadata_none, &subgroup_id);
}

static bool
lower_num_subgroups(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_num_subgroups)
      return false;

   b->cursor = nir_before_instr(instr);
   nir_ssa_def *subgroup_size = nir_load_subgroup_size(b);
   nir_ssa_def *size_minus_one = nir_iadd_imm(b, subgroup_size, -1);
   nir_ssa_def *workgroup_size_vec = nir_load_workgroup_size(b);
   nir_ssa_def *workgroup_size = nir_imul(b, nir_channel(b, workgroup_size_vec, 0),
                                             nir_imul(b, nir_channel(b, workgroup_size_vec, 1),
                                                         nir_channel(b, workgroup_size_vec, 2)));
   nir_ssa_def *ret = nir_idiv(b, nir_iadd(b, workgroup_size, size_minus_one), subgroup_size);
   nir_ssa_def_rewrite_uses(&intr->dest.ssa, ret);
   return true;
}

bool
dxil_nir_lower_num_subgroups(nir_shader *s)
{
   return nir_shader_instructions_pass(s, lower_num_subgroups,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance |
                                       nir_metadata_loop_analysis, NULL);
}
