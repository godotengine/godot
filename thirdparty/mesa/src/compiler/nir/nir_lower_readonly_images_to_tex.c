/*
 * Copyright © 2020 Intel Corporation
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

static const struct glsl_type *
get_texture_type_for_image(const struct glsl_type *type)
{
   if (glsl_type_is_array(type)) {
      const struct glsl_type *elem_type =
         get_texture_type_for_image(glsl_get_array_element(type));
      return glsl_array_type(elem_type, glsl_get_length(type), 0 /*explicit size*/);
   }

   assert((glsl_type_is_image(type)));
   return glsl_texture_type(glsl_get_sampler_dim(type),
                            glsl_sampler_type_is_array(type),
                            glsl_get_sampler_result_type(type));
}

static bool
replace_image_type_with_texture(nir_deref_instr *deref)
{
   const struct glsl_type *type = deref->type;

   /* If we've already chased up the deref chain this far from a different intrinsic, we're done */
   if (!glsl_type_is_image(glsl_without_array(type)))
      return false;

   deref->type = get_texture_type_for_image(type);
   deref->modes = nir_var_uniform;
   if (deref->deref_type == nir_deref_type_var) {
      type = deref->var->type;
      if (glsl_type_is_image(glsl_without_array(type))) {
         deref->var->type = get_texture_type_for_image(type);
         deref->var->data.mode = nir_var_uniform;
         memset(&deref->var->data.sampler, 0, sizeof(deref->var->data.sampler));
      }
   } else {
      nir_deref_instr *parent = nir_deref_instr_parent(deref);
      if (parent)
         replace_image_type_with_texture(parent);
   }

   return true;
}

struct readonly_image_lower_options {
   bool per_variable;
};

static bool
lower_readonly_image_instr_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin,
                                     const struct readonly_image_lower_options *options)
{
   if (intrin->intrinsic != nir_intrinsic_image_deref_load &&
       intrin->intrinsic != nir_intrinsic_image_deref_size)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   nir_variable *var = nir_deref_instr_get_variable(deref);

   /* In CL 1.2, images are required to be either read-only or
    * write-only.  We can always translate the read-only image ops to
    * texture ops.  In CL 2.0 (and an extension), the ability is added
    * to have read-write images but sampling (with a sampler) is only
    * allowed on read-only images.  As long as we only lower read-only
    * images to texture ops, everything should stay consistent.
    */
   enum gl_access_qualifier access = 0;
   if (options->per_variable) {
      if (var)
         access = var->data.access;
   } else {
      access = nir_intrinsic_access(intrin);
   }
   if (!(access & ACCESS_NON_WRITEABLE))
      return false;

   unsigned num_srcs;
   nir_texop texop;
   switch (intrin->intrinsic) {
   case nir_intrinsic_image_deref_load:
      texop = nir_texop_txf;
      num_srcs = 3;
      break;
   case nir_intrinsic_image_deref_size:
      texop = nir_texop_txs;
      num_srcs = 2;
      break;
   default:
      unreachable("Unsupported intrinsic");
   }

   b->cursor = nir_before_instr(&intrin->instr);

   nir_tex_instr *tex = nir_tex_instr_create(b->shader, num_srcs);
   tex->op = texop;

   tex->sampler_dim = glsl_get_sampler_dim(deref->type);
   tex->is_array = glsl_sampler_type_is_array(deref->type);
   tex->is_shadow = false;

   unsigned coord_components =
      glsl_get_sampler_dim_coordinate_components(tex->sampler_dim);
   if (glsl_sampler_type_is_array(deref->type))
      coord_components++;

   tex->src[0].src_type = nir_tex_src_texture_deref;
   tex->src[0].src = nir_src_for_ssa(&deref->dest.ssa);

   if (options->per_variable) {
      assert(nir_deref_instr_get_variable(deref));
      replace_image_type_with_texture(deref);
   }

   switch (intrin->intrinsic) {
   case nir_intrinsic_image_deref_load: {
      assert(intrin->src[1].is_ssa);
      nir_ssa_def *coord =
         nir_trim_vector(b, intrin->src[1].ssa, coord_components);
      tex->src[1].src_type = nir_tex_src_coord;
      tex->src[1].src = nir_src_for_ssa(coord);
      tex->coord_components = coord_components;

      assert(intrin->src[3].is_ssa);
      nir_ssa_def *lod = intrin->src[3].ssa;
      tex->src[2].src_type = nir_tex_src_lod;
      tex->src[2].src = nir_src_for_ssa(lod);

      assert(num_srcs == 3);

      tex->dest_type = nir_intrinsic_dest_type(intrin);
      nir_ssa_dest_init(&tex->instr, &tex->dest, 4, 32, NULL);
      break;
   }

   case nir_intrinsic_image_deref_size: {
      assert(intrin->src[1].is_ssa);
      nir_ssa_def *lod = intrin->src[1].ssa;
      tex->src[1].src_type = nir_tex_src_lod;
      tex->src[1].src = nir_src_for_ssa(lod);

      assert(num_srcs == 2);

      tex->dest_type = nir_type_uint32;
      nir_ssa_dest_init(&tex->instr, &tex->dest,
                        coord_components, 32, NULL);
      break;
   }

   default:
      unreachable("Unsupported intrinsic");
   }

   nir_builder_instr_insert(b, &tex->instr);

   nir_ssa_def *res = nir_trim_vector(b, &tex->dest.ssa,
                                      intrin->dest.ssa.num_components);

   nir_ssa_def_rewrite_uses(&intrin->dest.ssa, res);
   nir_instr_remove(&intrin->instr);

   return true;
}

static bool
lower_readonly_image_instr_tex(nir_builder *b, nir_tex_instr *tex,
                               const struct readonly_image_lower_options *options)
{
   int deref_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (deref_idx == -1)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(tex->src[deref_idx].src);
   if (options->per_variable) {
      assert(nir_deref_instr_get_variable(deref));
      return replace_image_type_with_texture(deref);
   }

   return false;
}

static bool
lower_readonly_image_instr(nir_builder *b, nir_instr *instr, void *context)
{
   struct readonly_image_lower_options *options = (struct readonly_image_lower_options *)context;

   switch (instr->type) {
   case nir_instr_type_intrinsic:
      return lower_readonly_image_instr_intrinsic(b, nir_instr_as_intrinsic(instr), options);
   case nir_instr_type_tex:
      return lower_readonly_image_instr_tex(b, nir_instr_as_tex(instr), options);
   default:
      return false;
   }
}

/** Lowers image ops to texture ops for read-only images
 *
 * If per_variable is set:
 * - Variable access is used to indicate read-only instead of intrinsic access
 * - Variable/deref types will be changed from image types to sampler types
 *
 * per_variable should not be set for OpenCL, because all image types will be
 * void-returning, and there is no corresponding valid sampler type, and it
 * will collide with the "bare" sampler type.
 */
bool
nir_lower_readonly_images_to_tex(nir_shader *shader, bool per_variable)
{
   struct readonly_image_lower_options options = { per_variable };
   return nir_shader_instructions_pass(shader, lower_readonly_image_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       &options);
}
