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

#include "dxil_nir_lower_int_samplers.h"
#include "nir_builder.h"
#include "nir_builtin_builder.h"

static bool
lower_sample_to_txf_for_integer_tex_filter(const nir_instr *instr,
                                           UNUSED const void *_options)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_tex &&
       tex->op != nir_texop_txb &&
       tex->op != nir_texop_txl &&
       tex->op != nir_texop_txd)
      return false;

   return (tex->dest_type & (nir_type_int | nir_type_uint));
}

static nir_ssa_def *
dx_get_texture_lod(nir_builder *b, nir_tex_instr *tex)
{
   nir_tex_instr *tql;

   unsigned num_srcs = 0;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_coord ||
          tex->src[i].src_type == nir_tex_src_texture_deref ||
          tex->src[i].src_type == nir_tex_src_sampler_deref ||
          tex->src[i].src_type == nir_tex_src_texture_offset ||
          tex->src[i].src_type == nir_tex_src_sampler_offset ||
          tex->src[i].src_type == nir_tex_src_texture_handle ||
          tex->src[i].src_type == nir_tex_src_sampler_handle)
         num_srcs++;
   }

   tql = nir_tex_instr_create(b->shader, num_srcs);
   tql->op = nir_texop_lod;
   unsigned coord_components = tex->coord_components;
   if (tex->is_array)
      --coord_components;

   tql->coord_components = coord_components;
   tql->sampler_dim = tex->sampler_dim;
   tql->is_shadow = tex->is_shadow;
   tql->is_new_style_shadow = tex->is_new_style_shadow;
   tql->texture_index = tex->texture_index;
   tql->sampler_index = tex->sampler_index;
   tql->dest_type = nir_type_float32;

   /* The coordinate needs special handling because we might have
    * to strip the array index. Don't clutter the code  with an additional
    * check for is_array though, in the worst case we create an additional
    * move the the optimization will remove later again. */
   int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_ssa_def *ssa_src = nir_channels(b, tex->src[coord_index].src.ssa,
                                       (1 << coord_components) - 1);
   nir_src src = nir_src_for_ssa(ssa_src);
   nir_src_copy(&tql->src[0].src, &src, &tql->instr);
   tql->src[0].src_type = nir_tex_src_coord;

   unsigned idx = 1;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_texture_deref ||
          tex->src[i].src_type == nir_tex_src_sampler_deref ||
          tex->src[i].src_type == nir_tex_src_texture_offset ||
          tex->src[i].src_type == nir_tex_src_sampler_offset ||
          tex->src[i].src_type == nir_tex_src_texture_handle ||
          tex->src[i].src_type == nir_tex_src_sampler_handle) {
         nir_src_copy(&tql->src[idx].src, &tex->src[i].src, &tql->instr);
         tql->src[idx].src_type = tex->src[i].src_type;
         idx++;
      }
   }

   nir_ssa_dest_init(&tql->instr, &tql->dest, 2, 32, NULL);
   nir_builder_instr_insert(b, &tql->instr);

   /* DirectX LOD only has a value in x channel */
   return nir_channel(b, &tql->dest.ssa, 0);
}

typedef struct {
   nir_ssa_def *coords;
   nir_ssa_def *use_border_color;
} wrap_result_t;

typedef struct {
   nir_ssa_def *lod;
   nir_ssa_def *size;
   int ncoord_comp;
   wrap_result_t wrap[3];
} wrap_lower_param_t;

static void
wrap_clamp_to_edge(nir_builder *b, wrap_result_t *wrap_params, nir_ssa_def *size)
{
   /* clamp(coord, 0, size - 1) */
   wrap_params->coords = nir_fmin(b, nir_fsub(b, size, nir_imm_float(b, 1.0f)),
                                  nir_fmax(b, wrap_params->coords, nir_imm_float(b, 0.0f)));
}

static void
wrap_repeat(nir_builder *b, wrap_result_t *wrap_params, nir_ssa_def *size)
{
   /* mod(coord, size)
    * This instruction must be exact, otherwise certain sizes result in
    * incorrect sampling */
   wrap_params->coords = nir_fmod(b, wrap_params->coords, size);
   nir_instr_as_alu(wrap_params->coords->parent_instr)->exact = true;
}

static nir_ssa_def *
mirror(nir_builder *b, nir_ssa_def *coord)
{
   /* coord if >= 0, otherwise -(1 + coord) */
   return nir_bcsel(b, nir_fge(b, coord, nir_imm_float(b, 0.0f)), coord,
                    nir_fneg(b, nir_fadd(b, nir_imm_float(b, 1.0f), coord)));
}

static void
wrap_mirror_repeat(nir_builder *b, wrap_result_t *wrap_params, nir_ssa_def *size)
{
   /* (size − 1) − mirror(mod(coord, 2 * size) − size) */
   nir_ssa_def *coord_mod2size = nir_fmod(b, wrap_params->coords, nir_fmul(b, nir_imm_float(b, 2.0f), size));
   nir_instr_as_alu(coord_mod2size->parent_instr)->exact = true;
   nir_ssa_def *a = nir_fsub(b, coord_mod2size, size);
   wrap_params->coords = nir_fsub(b, nir_fsub(b, size, nir_imm_float(b, 1.0f)), mirror(b, a));
}

static void
wrap_mirror_clamp_to_edge(nir_builder *b, wrap_result_t *wrap_params, nir_ssa_def *size)
{
   /* clamp(mirror(coord), 0, size - 1) */
   wrap_params->coords = nir_fmin(b, nir_fsub(b, size, nir_imm_float(b, 1.0f)),
                                  nir_fmax(b, mirror(b, wrap_params->coords), nir_imm_float(b, 0.0f)));
}

static void
wrap_clamp(nir_builder *b, wrap_result_t *wrap_params, nir_ssa_def *size)
{
   nir_ssa_def *is_low = nir_flt(b, wrap_params->coords, nir_imm_float(b, 0.0));
   nir_ssa_def *is_high = nir_fge(b, wrap_params->coords, size);
   wrap_params->use_border_color = nir_ior(b, is_low, is_high);
}

static void
wrap_mirror_clamp(nir_builder *b, wrap_result_t *wrap_params, nir_ssa_def *size)
{
   /* We have to take care of the boundaries */
   nir_ssa_def *is_low = nir_flt(b, wrap_params->coords, nir_fmul(b, size, nir_imm_float(b, -1.0)));
   nir_ssa_def *is_high = nir_flt(b, nir_fmul(b, size, nir_imm_float(b, 2.0)), wrap_params->coords);
   wrap_params->use_border_color = nir_ior(b, is_low, is_high);

   /* Within the boundaries this acts like mirror_repeat */
   wrap_mirror_repeat(b, wrap_params, size);

}

static wrap_result_t
wrap_coords(nir_builder *b, nir_ssa_def *coords, enum pipe_tex_wrap wrap,
            nir_ssa_def *size)
{
   wrap_result_t result = {coords, nir_imm_false(b)};

   switch (wrap) {
   case PIPE_TEX_WRAP_CLAMP_TO_EDGE:
      wrap_clamp_to_edge(b, &result, size);
      break;
   case PIPE_TEX_WRAP_REPEAT:
      wrap_repeat(b, &result, size);
      break;
   case PIPE_TEX_WRAP_MIRROR_REPEAT:
      wrap_mirror_repeat(b, &result, size);
      break;
   case PIPE_TEX_WRAP_MIRROR_CLAMP:
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE:
      wrap_mirror_clamp_to_edge(b, &result, size);
      break;
   case PIPE_TEX_WRAP_CLAMP:
   case PIPE_TEX_WRAP_CLAMP_TO_BORDER:
      wrap_clamp(b, &result, size);
      break;
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_BORDER:
      wrap_mirror_clamp(b, &result, size);
      break;
   }
   return result;
}

static nir_ssa_def *
load_bordercolor(nir_builder *b, nir_tex_instr *tex, dxil_wrap_sampler_state *active_state,
                 const dxil_texture_swizzle_state *tex_swizzle)
{
   int ndest_comp = nir_dest_num_components(tex->dest);

   unsigned swizzle[4] = {
      tex_swizzle->swizzle_r,
      tex_swizzle->swizzle_g,
      tex_swizzle->swizzle_b,
      tex_swizzle->swizzle_a
   };

   /* Avoid any possible float conversion issues */
   uint32_t border_color[4];
   memcpy(border_color, active_state->border_color, sizeof(border_color));
   STATIC_ASSERT(sizeof(border_color) == sizeof(active_state->border_color));

   nir_const_value const_value[4];
   for (int i = 0; i < ndest_comp; ++i) {
      switch (swizzle[i]) {
      case PIPE_SWIZZLE_0:
         const_value[i] = nir_const_value_for_uint(0, 32);
         break;
      case PIPE_SWIZZLE_1:
         const_value[i] = nir_const_value_for_uint(1, 32);
         break;
      case PIPE_SWIZZLE_X:
      case PIPE_SWIZZLE_Y:
      case PIPE_SWIZZLE_Z:
      case PIPE_SWIZZLE_W:
         const_value[i] = nir_const_value_for_uint(border_color[swizzle[i]], 32);
         break;
      default:
         unreachable("Unexpected swizzle value");
      }
   }

   return nir_build_imm(b, ndest_comp, 32, const_value);
}

static nir_tex_instr *
create_txf_from_tex(nir_builder *b, nir_tex_instr *tex)
{
   nir_tex_instr *txf;

   unsigned num_srcs = 0;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_texture_deref ||
          tex->src[i].src_type == nir_tex_src_texture_offset ||
          tex->src[i].src_type == nir_tex_src_texture_handle)
         num_srcs++;
   }

   txf = nir_tex_instr_create(b->shader, num_srcs);
   txf->op = nir_texop_txf;
   txf->coord_components = tex->coord_components;
   txf->sampler_dim = tex->sampler_dim;
   txf->is_array = tex->is_array;
   txf->is_shadow = tex->is_shadow;
   txf->is_new_style_shadow = tex->is_new_style_shadow;
   txf->texture_index = tex->texture_index;
   txf->sampler_index = tex->sampler_index;
   txf->dest_type = tex->dest_type;

   unsigned idx = 0;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_texture_deref ||
          tex->src[i].src_type == nir_tex_src_texture_offset ||
          tex->src[i].src_type == nir_tex_src_texture_handle) {
         nir_src_copy(&txf->src[idx].src, &tex->src[i].src, &txf->instr);
         txf->src[idx].src_type = tex->src[i].src_type;
         idx++;
      }
   }

   nir_ssa_dest_init(&txf->instr, &txf->dest,
                     nir_tex_instr_dest_size(txf), 32, NULL);
   nir_builder_instr_insert(b, &txf->instr);

   return txf;
}

static nir_ssa_def *
load_texel(nir_builder *b, nir_tex_instr *tex, wrap_lower_param_t *params)
{
   nir_ssa_def *texcoord = NULL;

   /* Put coordinates back together */
   switch (tex->coord_components) {
   case 1:
      texcoord = params->wrap[0].coords;
      break;
   case 2:
      texcoord = nir_vec2(b, params->wrap[0].coords, params->wrap[1].coords);
      break;
   case 3:
      texcoord = nir_vec3(b, params->wrap[0].coords, params->wrap[1].coords, params->wrap[2].coords);
      break;
   default:
      ;
   }

   texcoord = nir_f2i32(b, texcoord);

   nir_tex_instr *load = create_txf_from_tex(b, tex);
   nir_tex_instr_add_src(load, nir_tex_src_lod, nir_src_for_ssa(params->lod));
   nir_tex_instr_add_src(load, nir_tex_src_coord, nir_src_for_ssa(texcoord));
   b->cursor = nir_after_instr(&load->instr);
   return &load->dest.ssa;
}

typedef struct {
   dxil_wrap_sampler_state *aws;
   float max_bias;
   nir_ssa_def *size;
   int ncoord_comp;
} lod_params;

static nir_ssa_def *
evalute_active_lod(nir_builder *b, nir_tex_instr *tex, lod_params *params)
{
   static nir_ssa_def *lod = NULL;

   /* Later we use min_lod for clamping the LOD to a legal value */
   float min_lod = MAX2(params->aws->min_lod, 0.0f);

   /* Evaluate the LOD to be used for the texel fetch */
   if (unlikely(tex->op == nir_texop_txl)) {
      int lod_index = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      /* if we have an explicite LOD, take it */
      lod = tex->src[lod_index].src.ssa;
   } else if (unlikely(tex->op == nir_texop_txd)) {
      int ddx_index = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
      int ddy_index = nir_tex_instr_src_index(tex, nir_tex_src_ddy);
      assert(ddx_index >= 0 && ddy_index >= 0);

      nir_ssa_def *grad = nir_fmax(b,
                                   tex->src[ddx_index].src.ssa,
                                   tex->src[ddy_index].src.ssa);

      nir_ssa_def *r = nir_fmul(b, grad, nir_i2f32(b, params->size));
      nir_ssa_def *rho = nir_channel(b, r, 0);
      for (int i = 1; i < params->ncoord_comp; ++i)
         rho = nir_fmax(b, rho, nir_channel(b, r, i));
      lod = nir_flog2(b, rho);
   } else if (b->shader->info.stage == MESA_SHADER_FRAGMENT){
      lod = dx_get_texture_lod(b, tex);
   } else {
      /* Only fragment shaders provide the gradient information to evaluate a LOD,
       * so force 0 otherwise */
      lod = nir_imm_float(b, 0.0);
   }

   /* Evaluate bias according to OpenGL (4.6 (Compatibility  Profile) October 22, 2019),
    * sec. 8.14.1, eq. (8.9)
    *
    *    lod' = lambda + CLAMP(bias_texobj + bias_texunit + bias_shader)
    *
    * bias_texobj is the value of TEXTURE_LOD_BIAS for the bound texture object. ...
    * bias_textunt is the value of TEXTURE_LOD_BIAS for the current texture unit, ...
    * bias shader is the value of the optional bias parameter in the texture
    * lookup functions available to fragment shaders. ... The sum of these values
    * is clamped to the range [−bias_max, bias_max] where bias_max is the value
    * of the implementation defined constant MAX_TEXTURE_LOD_BIAS.
    * In core contexts the value bias_texunit is dropped from above equation.
    *
    * Gallium provides the value lod_bias as the sum of bias_texobj and bias_texunit
    * in compatibility contexts and as bias_texobj in core contexts, hence the
    * implementation here is the same in both cases.
    */
   nir_ssa_def *lod_bias = nir_imm_float(b, params->aws->lod_bias);

   if (unlikely(tex->op == nir_texop_txb)) {
      int bias_index = nir_tex_instr_src_index(tex, nir_tex_src_bias);
      lod_bias = nir_fadd(b, lod_bias, tex->src[bias_index].src.ssa);
   }

   lod = nir_fadd(b, lod, nir_fclamp(b, lod_bias,
                                     nir_imm_float(b, -params->max_bias),
                                     nir_imm_float(b, params->max_bias)));

   /* Clamp lod according to ibid. eq. (8.10) */
   lod = nir_fmax(b, lod, nir_imm_float(b, min_lod));

   /* If the max lod is > max_bias = log2(max_texture_size), the lod will be clamped
    * by the number of levels, no need to clamp it againt the max_lod first. */
   if (params->aws->max_lod <= params->max_bias)
      lod = nir_fmin(b, lod, nir_imm_float(b, params->aws->max_lod));

   /* Pick nearest LOD */
   lod = nir_f2i32(b, nir_fround_even(b, lod));

   /* cap actual lod by number of available levels */
   return nir_imin(b, lod, nir_imm_int(b, params->aws->last_level));
}

typedef struct {
   dxil_wrap_sampler_state *wrap_states;
   dxil_texture_swizzle_state *tex_swizzles;
   float max_bias;
} sampler_states;


static nir_ssa_def *
lower_sample_to_txf_for_integer_tex_impl(nir_builder *b, nir_instr *instr,
                                         void *options)
{
   sampler_states *states = (sampler_states *)options;
   wrap_lower_param_t params = {0};

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   dxil_wrap_sampler_state *active_wrap_state = &states->wrap_states[tex->sampler_index];

   b->cursor = nir_before_instr(instr);

   int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_ssa_def *old_coord = tex->src[coord_index].src.ssa;
   params.ncoord_comp = tex->coord_components;
   if (tex->is_array)
      params.ncoord_comp -= 1;

   /* This helper to get the texture size always uses LOD 0, and DirectX doesn't support
    * giving another LOD when querying the texture size */
   nir_ssa_def *size0 = nir_get_texture_size(b, tex);

   params.lod = nir_imm_int(b, 0);

   if (active_wrap_state->last_level > 0) {
      lod_params p = {
         .aws = active_wrap_state,
         .max_bias = states->max_bias,
         .size = size0,
         .ncoord_comp = params.ncoord_comp
      };
      params.lod = evalute_active_lod(b, tex, &p);

      /* Evaluate actual level size*/
      params.size = nir_i2f32(b, nir_imax(b, nir_ishr(b, size0, params.lod),
                                             nir_imm_int(b, 1)));
   } else {
      params.size = nir_i2f32(b, size0);
   }

   nir_ssa_def *new_coord = old_coord;
   if (!active_wrap_state->is_nonnormalized_coords) {
      /* Evaluate the integer lookup coordinates for the requested LOD, don't touch the
       * array index */
      if (!tex->is_array) {
         new_coord = nir_fmul(b, params.size, old_coord);
      } else {
         nir_ssa_def *array_index = nir_channel(b, old_coord, params.ncoord_comp);
         int mask = (1 << params.ncoord_comp) - 1;
         nir_ssa_def *coord = nir_fmul(b, nir_channels(b, params.size, mask),
                                          nir_channels(b, old_coord, mask));
         switch (params.ncoord_comp) {
         case 1:
            new_coord = nir_vec2(b, coord, array_index);
            break;
         case 2:
            new_coord = nir_vec3(b, nir_channel(b, coord, 0),
                                    nir_channel(b, coord, 1),
                                    array_index);
            break;
         default:
            unreachable("unsupported number of non-array coordinates");
         }
      }
   }

   nir_ssa_def *coord_help[3];
   for (int i = 0; i < params.ncoord_comp; ++i)
      coord_help[i] = nir_ffloor(b, nir_channel(b, new_coord, i));

   // Note: array index needs to be rounded to nearest before clamp rather than floored
   if (tex->is_array)
      coord_help[params.ncoord_comp] = nir_fround_even(b, nir_channel(b, new_coord, params.ncoord_comp));

   /* Correct the texture coordinates for the offsets. */
   int offset_index = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   if (offset_index >= 0) {
      nir_ssa_def *offset = tex->src[offset_index].src.ssa;
      for (int i = 0; i < params.ncoord_comp; ++i)
         coord_help[i] = nir_fadd(b, coord_help[i], nir_i2f32(b, nir_channel(b, offset, i)));
   }

   nir_ssa_def *use_border_color = nir_imm_false(b);

   if (!active_wrap_state->skip_boundary_conditions) {

      for (int i = 0; i < params.ncoord_comp; ++i) {
         params.wrap[i] = wrap_coords(b, coord_help[i], active_wrap_state->wrap[i], nir_channel(b, params.size, i));
         use_border_color = nir_ior(b, use_border_color, params.wrap[i].use_border_color);
      }

      if (tex->is_array)
         params.wrap[params.ncoord_comp] =
               wrap_coords(b, coord_help[params.ncoord_comp],
                           PIPE_TEX_WRAP_CLAMP_TO_EDGE,
                           nir_i2f32(b, nir_channel(b, size0, params.ncoord_comp)));
   } else {
      /* When we emulate a cube map by using a texture array, the coordinates are always
       * in range, and we don't have to take care of boundary conditions */
      for (unsigned i = 0; i < 3; ++i) {
         params.wrap[i].coords = coord_help[i];
         params.wrap[i].use_border_color = nir_imm_false(b);
      }
   }

   const dxil_texture_swizzle_state one2one = {
     PIPE_SWIZZLE_X,  PIPE_SWIZZLE_Y,  PIPE_SWIZZLE_Z, PIPE_SWIZZLE_W
   };

   nir_if *border_if = nir_push_if(b, use_border_color);
   const dxil_texture_swizzle_state *swizzle = states->tex_swizzles ?
                                                 &states->tex_swizzles[tex->sampler_index]:
                                                 &one2one;

   nir_ssa_def *border_color = load_bordercolor(b, tex, active_wrap_state, swizzle);
   nir_if *border_else = nir_push_else(b, border_if);
   nir_ssa_def *sampler_color = load_texel(b, tex, &params);
   nir_pop_if(b, border_else);

   return nir_if_phi(b, border_color, sampler_color);
}

/* Sampling from integer textures is not allowed in DirectX, so we have
 * to use texel fetches. For this we have to scale the coordiantes
 * to be integer based, and evaluate the LOD the texel fetch has to be
 * applied on, and take care of the boundary conditions .
 */
bool
dxil_lower_sample_to_txf_for_integer_tex(nir_shader *s,
                                         dxil_wrap_sampler_state *wrap_states,
                                         dxil_texture_swizzle_state *tex_swizzles,
                                         float max_bias)
{
   sampler_states states = {wrap_states, tex_swizzles, max_bias};

   bool result =
         nir_shader_lower_instructions(s,
                                       lower_sample_to_txf_for_integer_tex_filter,
                                       lower_sample_to_txf_for_integer_tex_impl,
                                       &states);
   return result;
}
