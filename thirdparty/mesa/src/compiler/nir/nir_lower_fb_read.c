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
#include "nir_builder.h"

/* Lowing for fragment shader load_output.
 *
 * This pass supports the blend_equation_advanced, where a fragment
 * shader loads the output (fragcolor) to read the current framebuffer.
 * It does this by lowering the output read to a txf_ms_fb instruction.
 * This instruction works similarly to a normal txf_ms except without
 * taking a texture source argument.  (The driver backend is expected
 * to wire this up to a free texture slot which is configured to read
 * from the framebuffer.)
 *
 * This should be run after lower_wpos_ytransform, because the tex
 * coordinates should be the physical fragcoord, not the logical
 * y-flipped coord.
 *
 * Note that this pass explicitly does *not* add a sampler uniform
 * (as txf_ms_fb does not reference a texture).  The driver backend
 * is going to want nif->info.num_textures to include the count of
 * number of textures *not* including the one it inserts to sample
 * from the framebuffer, so it more easily knows where to insert the
 * hidden texture to read from the fb.
 */

static bool
nir_lower_fb_read_instr(nir_builder *b, nir_instr *instr, UNUSED void *cb_data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_output)
      return false;

   /* TODO KHR_blend_equation_advanced is limited to non-MRT
    * scenarios.. but possible there are other extensions
    * where this pass would be useful that do support MRT?
    *
    * I guess for now I'll leave that as an exercise for the
    * reader.
    */
   if (nir_intrinsic_base(intr) != 0 || nir_src_as_uint(intr->src[0]) != 0)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_ssa_def *fragcoord = nir_load_frag_coord(b);
   nir_ssa_def *sampid = nir_load_sample_id(b);

   fragcoord = nir_f2i32(b, fragcoord);

   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = nir_texop_txf_ms_fb;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->src[0].src_type = nir_tex_src_coord;
   tex->src[0].src = nir_src_for_ssa(nir_channels(b, fragcoord, 0x3));
   tex->src[1].src_type = nir_tex_src_ms_index;
   tex->src[1].src = nir_src_for_ssa(sampid);

   nir_ssa_dest_init(&tex->instr, &tex->dest, 4, 32, NULL);
   nir_builder_instr_insert(b, &tex->instr);

   nir_ssa_def_rewrite_uses(&intr->dest.ssa, &tex->dest.ssa);

   return true;
}

bool
nir_lower_fb_read(nir_shader *shader)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);

   return nir_shader_instructions_pass(shader, nir_lower_fb_read_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       NULL);
}
