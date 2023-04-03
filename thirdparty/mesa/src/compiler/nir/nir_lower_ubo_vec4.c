/*
 * Copyright © 2020 Google LLC
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

/* Lowers nir_intrinsic_load_ubo() to nir_intrinsic_load_ubo_vec4() taking an
 * offset in vec4 units.  This is a fairly common mode of UBO addressing for
 * hardware to have, and it gives NIR a chance to optimize the addressing math
 * and CSE the loads.
 *
 * This pass handles lowering for loads that straddle a vec4 alignment
 * boundary.  We try to minimize the extra loads we generate for that case,
 * and are ensured non-straddling loads with:
 *
 * - std140 (GLSL 1.40, GLSL ES)
 * - Vulkan "Extended Layout" (the baseline for UBOs)
 *
 * but not:
 *
 * - GLSL 4.30's new packed mode (enabled by PIPE_CAP_LOAD_CONSTBUF) where
 *   vec3 arrays are packed tightly.
 *
 * - PackedDriverUniformStorage in GL (enabled by PIPE_CAP_PACKED_UNIFORMS)
 *   combined with nir_lower_uniforms_to_ubo, where values in the default
 *   uniform block are packed tightly.
 *
 * - Vulkan's scalarBlockLayout optional feature:
 *
 *   "A member is defined to improperly straddle if either of the following are
 *    true:
 *
 *    • It is a vector with total size less than or equal to 16 bytes, and has
 *      Offset decorations placing its first byte at F and its last byte at L
 *      where floor(F / 16) != floor(L / 16).
 *    • It is a vector with total size greater than 16 bytes and has its Offset
 *      decorations placing its first byte at a non-integer multiple of 16.
 *
 *    [...]
 *
 *    Unless the scalarBlockLayout feature is enabled on the device:
 *
 *    • Vectors must not improperly straddle, as defined above."
 */

#include "nir.h"
#include "nir_builder.h"

static bool
nir_lower_ubo_vec4_filter(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   return nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_ubo;
}

static nir_intrinsic_instr *
create_load(nir_builder *b, nir_ssa_def *block, nir_ssa_def *offset,
            unsigned bit_size, unsigned num_components)
{
   nir_ssa_def *def = nir_load_ubo_vec4(b, num_components, bit_size, block, offset);
   return nir_instr_as_intrinsic(def->parent_instr);
}

static nir_ssa_def *
nir_lower_ubo_vec4_lower(nir_builder *b, nir_instr *instr, void *data)
{
   b->cursor = nir_before_instr(instr);

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   nir_ssa_def *byte_offset = nir_ssa_for_src(b, intr->src[1], 1);
   nir_ssa_def *vec4_offset = nir_ushr_imm(b, byte_offset, 4);

   unsigned align_mul = nir_intrinsic_align_mul(intr);
   unsigned align_offset = nir_intrinsic_align_offset(intr);

   int chan_size_bytes = intr->dest.ssa.bit_size / 8;
   int chans_per_vec4 = 16 / chan_size_bytes;

   /* We don't care if someone figured out that things are aligned beyond
    * vec4.
    */
   align_mul = MIN2(align_mul, 16);
   align_offset &= 15;
   assert(align_offset % chan_size_bytes == 0);

   unsigned num_components = intr->num_components;
   bool aligned_mul = (align_mul == 16 &&
                       align_offset +  chan_size_bytes * num_components <= 16);
   if (!aligned_mul)
      num_components = chans_per_vec4;

   nir_intrinsic_instr *load = create_load(b, intr->src[0].ssa, vec4_offset,
                                           intr->dest.ssa.bit_size,
                                           num_components);

   nir_intrinsic_set_access(load, nir_intrinsic_access(intr));

   nir_ssa_def *result = &load->dest.ssa;

   int align_chan_offset = align_offset / chan_size_bytes;
   if (aligned_mul) {
      /* For an aligned load, just ask the backend to load from the known
       * offset's component.
       */
      nir_intrinsic_set_component(load, align_chan_offset);
   } else if (intr->num_components == 1) {
      /* If we're loading a single component, that component alone won't
       * straddle a vec4 boundary so we can do this with a single UBO load.
       */
      nir_ssa_def *component =
         nir_iand_imm(b,
                      nir_udiv_imm(b, byte_offset, chan_size_bytes),
                      chans_per_vec4 - 1);

      result = nir_vector_extract(b, result, component);
   } else if (align_mul == 8 &&
              align_offset + chan_size_bytes * intr->num_components <= 8) {
      /* Special case: Loading small vectors from offset % 8 == 0 can be done
       * with just one load and one bcsel.
       */
      nir_component_mask_t low_channels =
         BITSET_MASK(intr->num_components) << (align_chan_offset);
      nir_component_mask_t high_channels =
         low_channels << (8 / chan_size_bytes);
      result = nir_bcsel(b, nir_test_mask(b, byte_offset, 8),
                            nir_channels(b, result, high_channels),
                            nir_channels(b, result, low_channels));
   } else {
      /* General fallback case: Per-result-channel bcsel-based extraction
       * from two separate vec4 loads.
       */
      assert(num_components == 4);
      nir_ssa_def *next_vec4_offset = nir_iadd_imm(b, vec4_offset, 1);
      nir_intrinsic_instr *next_load = create_load(b, intr->src[0].ssa, next_vec4_offset,
                                                   intr->dest.ssa.bit_size,
                                                   num_components);

      nir_ssa_def *channels[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < intr->num_components; i++) {
         nir_ssa_def *chan_byte_offset = nir_iadd_imm(b, byte_offset, i * chan_size_bytes);

         nir_ssa_def *chan_vec4_offset = nir_ushr_imm(b, chan_byte_offset, 4);

         nir_ssa_def *component =
            nir_iand_imm(b,
                         nir_udiv_imm(b, chan_byte_offset, chan_size_bytes),
                         chans_per_vec4 - 1);

         channels[i] = nir_vector_extract(b,
                                          nir_bcsel(b,
                                                    nir_ieq(b,
                                                            chan_vec4_offset,
                                                            vec4_offset),
                                                    &load->dest.ssa,
                                                    &next_load->dest.ssa),
                                          component);
      }

      result = nir_vec(b, channels, intr->num_components);
   }

   return result;
}

bool
nir_lower_ubo_vec4(nir_shader *shader)
{
   return nir_shader_lower_instructions(shader,
                                        nir_lower_ubo_vec4_filter,
                                        nir_lower_ubo_vec4_lower,
                                        NULL);
}
