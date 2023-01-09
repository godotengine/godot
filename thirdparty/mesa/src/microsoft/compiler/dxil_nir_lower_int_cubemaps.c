/*
 * Copyright © Microsoft Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "dxil_nir_lower_int_cubemaps.h"

#include "nir_builder.h"
#include "nir_builtin_builder.h"

static bool
type_needs_lowering(const struct glsl_type *type, bool lower_samplers)
{
   type = glsl_without_array(type);
   if (!glsl_type_is_image(type) && !glsl_type_is_sampler(type))
      return false;
   if (glsl_get_sampler_dim(type) != GLSL_SAMPLER_DIM_CUBE)
      return false;
   if (glsl_type_is_image(type))
      return true;
   return lower_samplers && glsl_base_type_is_integer(glsl_get_sampler_result_type(type));
}

static bool
lower_int_cubmap_to_array_filter(const nir_instr *instr,
                                 const void *options)
{
   bool lower_samplers = *(bool *)options;
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_image_atomic_add:
      case nir_intrinsic_image_atomic_and:
      case nir_intrinsic_image_atomic_comp_swap:
      case nir_intrinsic_image_atomic_dec_wrap:
      case nir_intrinsic_image_atomic_exchange:
      case nir_intrinsic_image_atomic_fadd:
      case nir_intrinsic_image_atomic_fmax:
      case nir_intrinsic_image_atomic_fmin:
      case nir_intrinsic_image_atomic_imax:
      case nir_intrinsic_image_atomic_imin:
      case nir_intrinsic_image_atomic_inc_wrap:
      case nir_intrinsic_image_atomic_or:
      case nir_intrinsic_image_atomic_umax:
      case nir_intrinsic_image_atomic_umin:
      case nir_intrinsic_image_atomic_xor:
      case nir_intrinsic_image_load:
      case nir_intrinsic_image_size:
      case nir_intrinsic_image_store:
      case nir_intrinsic_image_deref_atomic_add:
      case nir_intrinsic_image_deref_atomic_and:
      case nir_intrinsic_image_deref_atomic_comp_swap:
      case nir_intrinsic_image_deref_atomic_dec_wrap:
      case nir_intrinsic_image_deref_atomic_exchange:
      case nir_intrinsic_image_deref_atomic_fadd:
      case nir_intrinsic_image_deref_atomic_fmax:
      case nir_intrinsic_image_deref_atomic_fmin:
      case nir_intrinsic_image_deref_atomic_imax:
      case nir_intrinsic_image_deref_atomic_imin:
      case nir_intrinsic_image_deref_atomic_inc_wrap:
      case nir_intrinsic_image_deref_atomic_or:
      case nir_intrinsic_image_deref_atomic_umax:
      case nir_intrinsic_image_deref_atomic_umin:
      case nir_intrinsic_image_deref_atomic_xor:
      case nir_intrinsic_image_deref_load:
      case nir_intrinsic_image_deref_size:
      case nir_intrinsic_image_deref_store:
         return nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_CUBE;
      default:
         return false;
      }
   } else if (instr->type == nir_instr_type_deref) {
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      return type_needs_lowering(deref->type, lower_samplers);
   } else if (instr->type == nir_instr_type_tex && lower_samplers) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE)
         return false;

      switch (tex->op) {
      case nir_texop_tex:
      case nir_texop_txb:
      case nir_texop_txd:
      case nir_texop_txl:
      case nir_texop_txs:
      case nir_texop_lod:
      case nir_texop_tg4:
         break;
      default:
         return false;
      }

      int sampler_deref = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
      assert(sampler_deref >= 0);
      nir_deref_instr *deref = nir_instr_as_deref(tex->src[sampler_deref].src.ssa->parent_instr);
      nir_variable *cube = nir_deref_instr_get_variable(deref);
      return glsl_base_type_is_integer(glsl_get_sampler_result_type(cube->type));
   }

   return false;
}

typedef struct {
   bool image;
   nir_ssa_def *rx;
   nir_ssa_def *ry;
   nir_ssa_def *rz;
   nir_ssa_def *arx;
   nir_ssa_def *ary;
   nir_ssa_def *arz;
   nir_ssa_def *array;
} coord_t;


/* This is taken from from sp_tex_sample:convert_cube */
static nir_ssa_def *
evaluate_face_x(nir_builder *b, coord_t *coord)
{
   nir_ssa_def *sign = nir_fsign(b, coord->rx);
   nir_ssa_def *positive = nir_fge(b, coord->rx, nir_imm_float(b, 0.0));
   nir_ssa_def *ima = nir_fdiv(b, nir_imm_float(b, -0.5), coord->arx);

   nir_ssa_def *x = nir_fadd(b, nir_fmul(b, nir_fmul(b, sign, ima), coord->rz), nir_imm_float(b, 0.5));
   nir_ssa_def *y = nir_fadd(b, nir_fmul(b, ima, coord->ry), nir_imm_float(b, 0.5));
   nir_ssa_def *face = nir_bcsel(b, positive, nir_imm_float(b, 0.0), nir_imm_float(b, 1.0));

   if (coord->array)
      face = nir_fadd(b, face, coord->array);

   return coord->image ?
      nir_vec4(b, x,y, face, nir_ssa_undef(b, 1, 32)) :
      nir_vec3(b, x,y, face);
}

static nir_ssa_def *
evaluate_face_y(nir_builder *b, coord_t *coord)
{
   nir_ssa_def *sign = nir_fsign(b, coord->ry);
   nir_ssa_def *positive = nir_fge(b, coord->ry, nir_imm_float(b, 0.0));
   nir_ssa_def *ima = nir_fdiv(b, nir_imm_float(b, 0.5), coord->ary);

   nir_ssa_def *x = nir_fadd(b, nir_fmul(b, ima, coord->rx), nir_imm_float(b, 0.5));
   nir_ssa_def *y = nir_fadd(b, nir_fmul(b, nir_fmul(b, sign, ima), coord->rz), nir_imm_float(b, 0.5));
   nir_ssa_def *face = nir_bcsel(b, positive, nir_imm_float(b, 2.0), nir_imm_float(b, 3.0));

   if (coord->array)
      face = nir_fadd(b, face, coord->array);
   
   return coord->image ?
      nir_vec4(b, x,y, face, nir_ssa_undef(b, 1, 32)) :
      nir_vec3(b, x,y, face);
}

static nir_ssa_def *
evaluate_face_z(nir_builder *b, coord_t *coord)
{
   nir_ssa_def *sign = nir_fsign(b, coord->rz);
   nir_ssa_def *positive = nir_fge(b, coord->rz, nir_imm_float(b, 0.0));
   nir_ssa_def *ima = nir_fdiv(b, nir_imm_float(b, -0.5), coord->arz);

   nir_ssa_def *x = nir_fadd(b, nir_fmul(b, nir_fmul(b, sign, ima), nir_fneg(b, coord->rx)), nir_imm_float(b, 0.5));
   nir_ssa_def *y = nir_fadd(b, nir_fmul(b, ima, coord->ry), nir_imm_float(b, 0.5));
   nir_ssa_def *face = nir_bcsel(b, positive, nir_imm_float(b, 4.0), nir_imm_float(b, 5.0));

   if (coord->array)
      face = nir_fadd(b, face, coord->array);
   
   return coord->image ?
      nir_vec4(b, x,y, face, nir_ssa_undef(b, 1, 32)) :
      nir_vec3(b, x,y, face);
}

static nir_ssa_def *
create_array_tex_from_cube_tex(nir_builder *b, nir_tex_instr *tex, nir_ssa_def *coord, nir_texop op)
{
   nir_tex_instr *array_tex;

   array_tex = nir_tex_instr_create(b->shader, tex->num_srcs);
   array_tex->op = op;
   array_tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   array_tex->is_array = true;
   array_tex->is_shadow = tex->is_shadow;
   array_tex->is_new_style_shadow = tex->is_new_style_shadow;
   array_tex->texture_index = tex->texture_index;
   array_tex->sampler_index = tex->sampler_index;
   array_tex->dest_type = tex->dest_type;
   array_tex->coord_components = 3;

   nir_src coord_src = nir_src_for_ssa(coord);
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      nir_src *psrc = (tex->src[i].src_type == nir_tex_src_coord) ?
                         &coord_src : &tex->src[i].src;

      nir_src_copy(&array_tex->src[i].src, psrc, &array_tex->instr);
      array_tex->src[i].src_type = tex->src[i].src_type;
   }

   nir_ssa_dest_init(&array_tex->instr, &array_tex->dest,
                     nir_tex_instr_dest_size(array_tex), 32, NULL);
   nir_builder_instr_insert(b, &array_tex->instr);
   return &array_tex->dest.ssa;
}

static nir_ssa_def *
handle_cube_edge(nir_builder *b, nir_ssa_def *x, nir_ssa_def *y, nir_ssa_def *face, nir_ssa_def *array_slice_cube_base, nir_ssa_def *tex_size)
{
   enum cube_remap
   {
      cube_remap_zero = 0,
      cube_remap_x,
      cube_remap_y,
      cube_remap_tex_size,
      cube_remap_tex_size_minus_x,
      cube_remap_tex_size_minus_y,

      cube_remap_size,
   };

   struct cube_remap_table
   {
      enum cube_remap remap_x;
      enum cube_remap remap_y;
      uint32_t        remap_face;
   };

   static const struct cube_remap_table cube_remap_neg_x[6] =
   {
       {cube_remap_tex_size,         cube_remap_y,         4},
       {cube_remap_tex_size,         cube_remap_y,         5},
       {cube_remap_y,                cube_remap_zero,      1},
       {cube_remap_tex_size_minus_y, cube_remap_tex_size,  1},
       {cube_remap_tex_size,         cube_remap_y,         1},
       {cube_remap_tex_size,         cube_remap_y,         0},
   };

   static const struct cube_remap_table cube_remap_pos_x[6] =
   {
       {cube_remap_zero,             cube_remap_y,         5},
       {cube_remap_zero,             cube_remap_y,         4},
       {cube_remap_tex_size_minus_y, cube_remap_zero,      0},
       {cube_remap_y,                cube_remap_tex_size,  0},
       {cube_remap_zero,             cube_remap_y,         0},
       {cube_remap_zero,             cube_remap_y,         1},
   };

   static const struct cube_remap_table cube_remap_neg_y[6] =
   {
       {cube_remap_tex_size,         cube_remap_tex_size_minus_x, 2},
       {cube_remap_zero,             cube_remap_x,                2},
       {cube_remap_tex_size_minus_x, cube_remap_zero,             5},
       {cube_remap_x,                cube_remap_tex_size,         4},
       {cube_remap_x,                cube_remap_tex_size,         2},
       {cube_remap_tex_size_minus_x, cube_remap_zero,             2},
   };

   static const struct cube_remap_table cube_remap_pos_y[6] =
   {
       {cube_remap_tex_size,         cube_remap_x,                   3},
       {cube_remap_zero,             cube_remap_tex_size_minus_x,    3},
       {cube_remap_x,                cube_remap_zero,                4},
       {cube_remap_tex_size_minus_x, cube_remap_tex_size,            5},
       {cube_remap_x,                cube_remap_zero,                3},
       {cube_remap_tex_size_minus_x, cube_remap_tex_size,            3},
   };

   static const struct cube_remap_table* remap_tables[4] = {
      cube_remap_neg_x,
      cube_remap_pos_x,
      cube_remap_neg_y,
      cube_remap_pos_y
   };

   nir_ssa_def *zero = nir_imm_int(b, 0);
   
   /* Doesn't matter since the texture is square */
   tex_size = nir_channel(b, tex_size, 0);

   nir_ssa_def *x_on = nir_iand(b, nir_ige(b, x, zero), nir_ige(b, tex_size, x));
   nir_ssa_def *y_on = nir_iand(b, nir_ige(b, y, zero), nir_ige(b, tex_size, y));
   nir_ssa_def *one_on = nir_ixor(b, x_on, y_on);

   /* If the sample did not fall off the face in either dimension, then set output = input */
   nir_ssa_def *x_result = x;
   nir_ssa_def *y_result = y;
   nir_ssa_def *face_result = face;

   /* otherwise, if the sample fell off the face in either the X or the Y direction, remap to the new face */
   nir_ssa_def *remap_predicates[4] =
   {
      nir_iand(b, one_on, nir_ilt(b, x, zero)),
      nir_iand(b, one_on, nir_ilt(b, tex_size, x)),
      nir_iand(b, one_on, nir_ilt(b, y, zero)),
      nir_iand(b, one_on, nir_ilt(b, tex_size, y)),
   };

   nir_ssa_def *remap_array[cube_remap_size];

   remap_array[cube_remap_zero] = zero;
   remap_array[cube_remap_x] = x;
   remap_array[cube_remap_y] = y;
   remap_array[cube_remap_tex_size] = tex_size;
   remap_array[cube_remap_tex_size_minus_x] = nir_isub(b, tex_size, x);
   remap_array[cube_remap_tex_size_minus_y] = nir_isub(b, tex_size, y);

   /* For each possible way the sample could have fallen off */
   for (unsigned i = 0; i < 4; i++) {
      const struct cube_remap_table* remap_table = remap_tables[i];

      /* For each possible original face */
      for (unsigned j = 0; j < 6; j++) {
         nir_ssa_def *predicate = nir_iand(b, remap_predicates[i], nir_ieq(b, face, nir_imm_int(b, j)));

         x_result = nir_bcsel(b, predicate, remap_array[remap_table[j].remap_x], x_result);
         y_result = nir_bcsel(b, predicate, remap_array[remap_table[j].remap_y], y_result);
         face_result = nir_bcsel(b, predicate, remap_array[remap_table[j].remap_face], face_result);
      }
   }

   return nir_vec3(b, x_result, y_result, nir_iadd(b, face_result, array_slice_cube_base));
}

static nir_ssa_def *
handle_cube_gather(nir_builder *b, nir_tex_instr *tex, nir_ssa_def *coord)
{
   nir_ssa_def *tex_size = nir_get_texture_size(b, tex);

   /* nir_get_texture_size puts the cursor before the tex op */
   b->cursor = nir_after_instr(coord->parent_instr);

   nir_ssa_def *const_05 = nir_imm_float(b, 0.5f);
   nir_ssa_def *texel_coords = nir_fmul(b, nir_channels(b, coord, 3),
      nir_i2f32(b, nir_channels(b, tex_size, 3)));

   nir_ssa_def *x_orig = nir_channel(b, texel_coords, 0);
   nir_ssa_def *y_orig = nir_channel(b, texel_coords, 1);

   nir_ssa_def *x_pos = nir_f2i32(b, nir_fadd(b, x_orig, const_05));
   nir_ssa_def *x_neg = nir_f2i32(b, nir_fsub(b, x_orig, const_05));
   nir_ssa_def *y_pos = nir_f2i32(b, nir_fadd(b, y_orig, const_05));
   nir_ssa_def *y_neg = nir_f2i32(b, nir_fsub(b, y_orig, const_05));
   nir_ssa_def *coords[4][2] = {
      { x_neg, y_pos },
      { x_pos, y_pos },
      { x_pos, y_neg },
      { x_neg, y_neg },
   };

   nir_ssa_def *array_slice_2d = nir_f2i32(b, nir_channel(b, coord, 2));
   nir_ssa_def *face = nir_imod(b, array_slice_2d, nir_imm_int(b, 6));
   nir_ssa_def *array_slice_cube_base = nir_isub(b, array_slice_2d, face);

   nir_ssa_def *channels[4];
   for (unsigned i = 0; i < 4; ++i) {
      nir_ssa_def *final_coord = handle_cube_edge(b, coords[i][0], coords[i][1], face, array_slice_cube_base, tex_size);
      nir_ssa_def *sampled_val = create_array_tex_from_cube_tex(b, tex, final_coord, nir_texop_txf);
      channels[i] = nir_channel(b, sampled_val, tex->component);
   }

   return nir_vec(b, channels, 4);
}

static nir_ssa_def *
lower_cube_coords(nir_builder *b, nir_ssa_def *coord, bool is_array, bool is_image)
{
   coord_t coords;
   coords.image = is_image;
   coords.rx = nir_channel(b, coord, 0);
   coords.ry = nir_channel(b, coord, 1);
   coords.rz = nir_channel(b, coord, 2);
   coords.arx = nir_fabs(b, coords.rx);
   coords.ary = nir_fabs(b, coords.ry);
   coords.arz = nir_fabs(b, coords.rz);
   coords.array = NULL;
   if (is_array)
      coords.array = nir_fmul(b, nir_channel(b, coord, 3), nir_imm_float(b, 6.0f));

   nir_ssa_def *use_face_x = nir_iand(b,
                                      nir_fge(b, coords.arx, coords.ary),
                                      nir_fge(b, coords.arx, coords.arz));

   nir_if *use_face_x_if = nir_push_if(b, use_face_x);
   nir_ssa_def *face_x_coord = evaluate_face_x(b, &coords);
   nir_if *use_face_x_else = nir_push_else(b, use_face_x_if);

   nir_ssa_def *use_face_y = nir_iand(b,
                                      nir_fge(b, coords.ary, coords.arx),
                                      nir_fge(b, coords.ary, coords.arz));

   nir_if *use_face_y_if = nir_push_if(b, use_face_y);
   nir_ssa_def *face_y_coord = evaluate_face_y(b, &coords);
   nir_if *use_face_y_else = nir_push_else(b, use_face_y_if);

   nir_ssa_def *face_z_coord = evaluate_face_z(b, &coords);

   nir_pop_if(b, use_face_y_else);
   nir_ssa_def *face_y_or_z_coord = nir_if_phi(b, face_y_coord, face_z_coord);
   nir_pop_if(b, use_face_x_else);

   // This contains in xy the normalized sample coordinates, and in z the face index
   nir_ssa_def *coord_and_face = nir_if_phi(b, face_x_coord, face_y_or_z_coord);

   return coord_and_face;
}

static nir_ssa_def *
lower_cube_sample(nir_builder *b, nir_tex_instr *tex)
{
   int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   assert(coord_index >= 0);

   /* Evaluate the face and the xy coordinates for a 2D tex op */
   nir_ssa_def *coord = tex->src[coord_index].src.ssa;
   nir_ssa_def *coord_and_face = lower_cube_coords(b, coord, tex->is_array, false);

   if (tex->op == nir_texop_tg4)
      return handle_cube_gather(b, tex, coord_and_face);
   else
      return create_array_tex_from_cube_tex(b, tex, coord_and_face, tex->op);
}

static nir_ssa_def *
lower_cube_image_load_store_atomic(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_intrinsic_set_image_array(intr, true);
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);

   return NIR_LOWER_INSTR_PROGRESS;
}

static nir_ssa_def *
lower_cube_txs(nir_builder *b, nir_tex_instr *tex)
{
   b->cursor = nir_after_instr(&tex->instr);
   if (!tex->is_array)
      return nir_channels(b, &tex->dest.ssa, 3);

   nir_ssa_def *array_dim = nir_channel(b, &tex->dest.ssa, 2);
   nir_ssa_def *cube_array_dim = nir_idiv(b, array_dim, nir_imm_int(b, 6));
   return nir_vec3(b, nir_channel(b, &tex->dest.ssa, 0),
                      nir_channel(b, &tex->dest.ssa, 1),
                      cube_array_dim);
}

static nir_ssa_def *
lower_cube_image_size(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_after_instr(&intr->instr);
   if (!nir_intrinsic_image_array(intr))
      return nir_channels(b, &intr->dest.ssa, 3);

   nir_ssa_def *array_dim = nir_channel(b, &intr->dest.ssa, 2);
   nir_ssa_def *cube_array_dim = nir_idiv(b, array_dim, nir_imm_int(b, 6));
   return nir_vec3(b, nir_channel(b, &intr->dest.ssa, 0),
                      nir_channel(b, &intr->dest.ssa, 1),
                      cube_array_dim);
}

static const struct glsl_type *
make_2darray_sampler_from_cubemap(const struct glsl_type *type)
{
   return  glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_CUBE ?
            glsl_sampler_type(
               GLSL_SAMPLER_DIM_2D,
               false, true,
               glsl_get_sampler_result_type(type)) : type;
}

static const struct glsl_type *
make_2darray_image_from_cubemap(const struct glsl_type *type)
{
   return  glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_CUBE ?
            glsl_image_type(
               GLSL_SAMPLER_DIM_2D,
               true,
               glsl_get_sampler_result_type(type)) : type;
}

static const struct glsl_type *
make_2darray_from_cubemap_with_array(const struct glsl_type *type, bool is_image)
{
   if (glsl_type_is_array(type)) {
      const struct glsl_type *new_type = glsl_without_array(type);
      return new_type != type ? glsl_array_type(make_2darray_from_cubemap_with_array(glsl_without_array(type), is_image),
                                                glsl_get_length(type), 0) : type;
   } else if (is_image)
      return make_2darray_image_from_cubemap(type);
   else
      return make_2darray_sampler_from_cubemap(type);
}

static nir_ssa_def *
lower_int_cubemap_to_array_tex(nir_builder *b, nir_tex_instr *tex)
{
   switch (tex->op) {
   case nir_texop_tex:
   case nir_texop_txb:
   case nir_texop_txd:
   case nir_texop_txl:
   case nir_texop_lod:
   case nir_texop_tg4:
      return lower_cube_sample(b, tex);
   case nir_texop_txs:
      return lower_cube_txs(b, tex);
   default:
      unreachable("Unsupported cupe map texture operation");
   }
}

static nir_ssa_def *
lower_cube_image_intrinsic(nir_builder *b, nir_intrinsic_instr *intr)
{
   if (intr->intrinsic == nir_intrinsic_image_size ||
      intr->intrinsic == nir_intrinsic_image_deref_size)
      return lower_cube_image_size(b, intr);
   else
      return lower_cube_image_load_store_atomic(b, intr);
}

static nir_ssa_def *
lower_cube_image_deref(nir_builder *b, nir_deref_instr *deref)
{
   deref->type = make_2darray_from_cubemap_with_array(
      deref->type,
      glsl_type_is_image(glsl_without_array(deref->type)));
   return NIR_LOWER_INSTR_PROGRESS;
}

static nir_ssa_def *
lower_int_cubmap_to_array_impl(nir_builder *b, nir_instr *instr,
                               void *options)
{
   bool lower_samplers = *(bool *)options;
   if (instr->type == nir_instr_type_tex && lower_samplers)
      return lower_int_cubemap_to_array_tex(b, nir_instr_as_tex(instr));
   else if (instr->type == nir_instr_type_intrinsic)
      return lower_cube_image_intrinsic(b, nir_instr_as_intrinsic(instr));
   else if (instr->type == nir_instr_type_deref)
      return lower_cube_image_deref(b, nir_instr_as_deref(instr));
   return NULL;
}

bool
dxil_nir_lower_int_cubemaps(nir_shader *s, bool lower_samplers)
{
   bool result =
         nir_shader_lower_instructions(s,
                                       lower_int_cubmap_to_array_filter,
                                       lower_int_cubmap_to_array_impl,
                                       &lower_samplers);

   if (result) {
      nir_foreach_variable_with_modes_safe(var, s, nir_var_uniform | nir_var_image) {
         if (!type_needs_lowering(var->type, lower_samplers))
            continue;
         bool is_image = glsl_type_is_image(glsl_without_array(var->type));
         var->type = make_2darray_from_cubemap_with_array(var->type, is_image);
      }
   }
   return result;

}
