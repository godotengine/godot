/*
 * Copyright © 2021 Intel Corporation
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

/**
 * This lowering pass supports (as configured via nir_lower_image_options)
 * image related conversions:
 *   + cube array size lowering. The size operation is converted from cube
 *     size to a 2d-array with the z component divided by 6.
 */

#include "nir.h"
#include "nir_builder.h"

static void
lower_cube_size(nir_builder *b, nir_intrinsic_instr *intrin)
{
   assert(nir_intrinsic_image_dim(intrin) == GLSL_SAMPLER_DIM_CUBE);

   b->cursor = nir_before_instr(&intrin->instr);

   nir_intrinsic_instr *_2darray_size =
      nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intrin->instr));
   nir_intrinsic_set_image_dim(_2darray_size, GLSL_SAMPLER_DIM_2D);
   nir_intrinsic_set_image_array(_2darray_size, true);
   nir_builder_instr_insert(b, &_2darray_size->instr);

   nir_ssa_def *size = nir_instr_ssa_def(&_2darray_size->instr);
   nir_ssa_scalar comps[NIR_MAX_VEC_COMPONENTS] = { 0 };
   unsigned coord_comps = intrin->dest.ssa.num_components;
   for (unsigned c = 0; c < coord_comps; c++) {
      if (c == 2) {
         comps[2] = nir_get_ssa_scalar(nir_idiv(b, nir_channel(b, size, 2), nir_imm_int(b, 6)), 0);
      } else {
         comps[c] = nir_get_ssa_scalar(size, c);
      }
   }

   nir_ssa_def *vec = nir_vec_scalars(b, comps, intrin->dest.ssa.num_components);
   nir_ssa_def_rewrite_uses(&intrin->dest.ssa, vec);
   nir_instr_remove(&intrin->instr);
   nir_instr_free(&intrin->instr);
}

static void
lower_image_to_fragment_mask_load(nir_builder *b, nir_intrinsic_instr *intrin)
{
   b->cursor = nir_before_instr(&intrin->instr);

   nir_intrinsic_op fmask_op;
   switch (intrin->intrinsic) {
   case nir_intrinsic_image_load:
      fmask_op = nir_intrinsic_image_fragment_mask_load_amd;
      break;
   case nir_intrinsic_image_deref_load:
      fmask_op = nir_intrinsic_image_deref_fragment_mask_load_amd;
      break;
   case nir_intrinsic_bindless_image_load:
      fmask_op = nir_intrinsic_bindless_image_fragment_mask_load_amd;
      break;
   default:
      unreachable("bad intrinsic");
      break;
   }

   nir_ssa_def *fmask =
      nir_image_fragment_mask_load_amd(b, intrin->src[0].ssa, intrin->src[1].ssa,
                                       .image_dim = nir_intrinsic_image_dim(intrin),
                                       .image_array = nir_intrinsic_image_array(intrin),
                                       .format = nir_intrinsic_format(intrin),
                                       .access = nir_intrinsic_access(intrin));

   /* fix intrinsic op */
   nir_intrinsic_instr *fmask_load = nir_instr_as_intrinsic(fmask->parent_instr);
   fmask_load->intrinsic = fmask_op;

   /* extract real color buffer index from fmask buffer */
   nir_ssa_def *sample_index_old = intrin->src[2].ssa;
   nir_ssa_def *fmask_offset = nir_ishl_imm(b, sample_index_old, 2);
   nir_ssa_def *fmask_width = nir_imm_int(b, 4);
   nir_ssa_def *sample_index_new = nir_ubfe(b, fmask, fmask_offset, fmask_width);

   /* fix color buffer load */
   nir_instr_rewrite_src_ssa(&intrin->instr, &intrin->src[2], sample_index_new);

   /* Mark uses fmask to prevent lower this intrinsic again. */
   enum gl_access_qualifier access = nir_intrinsic_access(intrin);
   nir_intrinsic_set_access(intrin, access | ACCESS_FMASK_LOWERED_AMD);
}

static void
lower_image_samples_identical_to_fragment_mask_load(nir_builder *b, nir_intrinsic_instr *intrin)
{
   b->cursor = nir_before_instr(&intrin->instr);

   nir_intrinsic_instr *fmask_load =
      nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intrin->instr));

   switch (intrin->intrinsic) {
   case nir_intrinsic_image_samples_identical:
      fmask_load->intrinsic = nir_intrinsic_image_fragment_mask_load_amd;
      break;
   case nir_intrinsic_image_deref_samples_identical:
      fmask_load->intrinsic = nir_intrinsic_image_deref_fragment_mask_load_amd;
      break;
   case nir_intrinsic_bindless_image_samples_identical:
      fmask_load->intrinsic = nir_intrinsic_bindless_image_fragment_mask_load_amd;
      break;
   default:
      unreachable("bad intrinsic");
      break;
   }

   nir_ssa_dest_init(&fmask_load->instr, &fmask_load->dest, 1, 32, NULL);
   nir_builder_instr_insert(b, &fmask_load->instr);

   nir_ssa_def *samples_identical = nir_ieq_imm(b, &fmask_load->dest.ssa, 0);
   nir_ssa_def_rewrite_uses(&intrin->dest.ssa, samples_identical);

   nir_instr_remove(&intrin->instr);
   nir_instr_free(&intrin->instr);
}

static bool
lower_image_instr(nir_builder *b, nir_instr *instr, void *state)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_lower_image_options *options = state;
   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   switch (intrin->intrinsic) {
   case nir_intrinsic_image_size:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_bindless_image_size:
      if (options->lower_cube_size &&
          nir_intrinsic_image_dim(intrin) == GLSL_SAMPLER_DIM_CUBE) {
         lower_cube_size(b, intrin);
         return true;
      }
      return false;

   case nir_intrinsic_image_load:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_bindless_image_load:
      if (options->lower_to_fragment_mask_load_amd &&
          nir_intrinsic_image_dim(intrin) == GLSL_SAMPLER_DIM_MS &&
          /* Don't lower again. */
          !(nir_intrinsic_access(intrin) & ACCESS_FMASK_LOWERED_AMD)) {
         lower_image_to_fragment_mask_load(b, intrin);
         return true;
      }
      return false;

   case nir_intrinsic_image_samples_identical:
   case nir_intrinsic_image_deref_samples_identical:
   case nir_intrinsic_bindless_image_samples_identical:
      if (options->lower_to_fragment_mask_load_amd &&
          nir_intrinsic_image_dim(intrin) == GLSL_SAMPLER_DIM_MS) {
         lower_image_samples_identical_to_fragment_mask_load(b, intrin);
         return true;
      }
      return false;

   default:
      return false;
   }
}

bool
nir_lower_image(nir_shader *nir, const nir_lower_image_options *options)
{
   return nir_shader_instructions_pass(nir, lower_image_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance, (void*)options);
}
