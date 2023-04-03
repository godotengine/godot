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

#include "nir_vulkan.h"
#include "vulkan/util/vk_format.h"
#include "vulkan/runtime/vk_ycbcr_conversion.h"
#include <math.h>

static nir_ssa_def *
y_range(nir_builder *b,
        nir_ssa_def *y_channel,
        int bpc,
        VkSamplerYcbcrRange range)
{
   switch (range) {
   case VK_SAMPLER_YCBCR_RANGE_ITU_FULL:
      return y_channel;
   case VK_SAMPLER_YCBCR_RANGE_ITU_NARROW:
      return nir_fmul(b,
                      nir_fadd(b,
                               nir_fmul(b, y_channel,
                                        nir_imm_float(b, pow(2, bpc) - 1)),
                               nir_imm_float(b, -16.0f * pow(2, bpc - 8))),
                      nir_frcp(b, nir_imm_float(b, 219.0f * pow(2, bpc - 8))));
   default:
      unreachable("missing Ycbcr range");
      return NULL;
   }
}

static nir_ssa_def *
chroma_range(nir_builder *b,
             nir_ssa_def *chroma_channel,
             int bpc,
             VkSamplerYcbcrRange range)
{
   switch (range) {
   case VK_SAMPLER_YCBCR_RANGE_ITU_FULL:
      return nir_fadd(b, chroma_channel,
                      nir_imm_float(b, -pow(2, bpc - 1) / (pow(2, bpc) - 1.0f)));
   case VK_SAMPLER_YCBCR_RANGE_ITU_NARROW:
      return nir_fmul(b,
                      nir_fadd(b,
                               nir_fmul(b, chroma_channel,
                                        nir_imm_float(b, pow(2, bpc) - 1)),
                               nir_imm_float(b, -128.0f * pow(2, bpc - 8))),
                      nir_frcp(b, nir_imm_float(b, 224.0f * pow(2, bpc - 8))));
   default:
      unreachable("missing Ycbcr range");
      return NULL;
   }
}

typedef struct nir_const_value_3_4 {
   nir_const_value v[3][4];
} nir_const_value_3_4;

static const nir_const_value_3_4 *
ycbcr_model_to_rgb_matrix(VkSamplerYcbcrModelConversion model)
{
   switch (model) {
   case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601: {
      static const nir_const_value_3_4 bt601 = { {
         { { .f32 =  1.402f             }, { .f32 = 1.0f }, { .f32 =  0.0f               }, { .f32 = 0.0f } },
         { { .f32 = -0.714136286201022f }, { .f32 = 1.0f }, { .f32 = -0.344136286201022f }, { .f32 = 0.0f } },
         { { .f32 =  0.0f               }, { .f32 = 1.0f }, { .f32 =  1.772f             }, { .f32 = 0.0f } },
      } };

      return &bt601;
   }
   case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709: {
      static const nir_const_value_3_4 bt709 = { {
         { { .f32 =  1.5748031496063f   }, { .f32 = 1.0f }, { .f32 =  0.0f               }, { .f32 = 0.0f } },
         { { .f32 = -0.468125209181067f }, { .f32 = 1.0f }, { .f32 = -0.187327487470334f }, { .f32 = 0.0f } },
         { { .f32 =  0.0f               }, { .f32 = 1.0f }, { .f32 =  1.85563184264242f  }, { .f32 = 0.0f } },
      } };

      return &bt709;
   }
   case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020: {
      static const nir_const_value_3_4 bt2020 = { {
         { { .f32 =  1.4746f            }, { .f32 = 1.0f }, { .f32 =  0.0f               }, { .f32 = 0.0f } },
         { { .f32 = -0.571353126843658f }, { .f32 = 1.0f }, { .f32 = -0.164553126843658f }, { .f32 = 0.0f } },
         { { .f32 =  0.0f               }, { .f32 = 1.0f }, { .f32 =  1.8814f            }, { .f32 = 0.0f } },
      } };

      return &bt2020;
   }
   default:
      unreachable("missing Ycbcr model");
      return NULL;
   }
}

nir_ssa_def *
nir_convert_ycbcr_to_rgb(nir_builder *b,
                         VkSamplerYcbcrModelConversion model,
                         VkSamplerYcbcrRange range,
                         nir_ssa_def *raw_channels,
                         uint32_t *bpcs)
{
   nir_ssa_def *expanded_channels =
      nir_vec4(b,
               chroma_range(b, nir_channel(b, raw_channels, 0), bpcs[0], range),
               y_range(b, nir_channel(b, raw_channels, 1), bpcs[1], range),
               chroma_range(b, nir_channel(b, raw_channels, 2), bpcs[2], range),
               nir_channel(b, raw_channels, 3));

   if (model == VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY)
      return expanded_channels;

   const nir_const_value_3_4 *conversion_matrix =
      ycbcr_model_to_rgb_matrix(model);

   nir_ssa_def *converted_channels[] = {
      nir_fdot(b, expanded_channels, nir_build_imm(b, 4, 32, conversion_matrix->v[0])),
      nir_fdot(b, expanded_channels, nir_build_imm(b, 4, 32, conversion_matrix->v[1])),
      nir_fdot(b, expanded_channels, nir_build_imm(b, 4, 32, conversion_matrix->v[2]))
   };

   return nir_vec4(b,
                   converted_channels[0], converted_channels[1],
                   converted_channels[2], nir_channel(b, raw_channels, 3));
}

struct ycbcr_state {
   nir_builder *builder;
   nir_ssa_def *image_size;
   nir_tex_instr *origin_tex;
   nir_deref_instr *tex_deref;
   const struct vk_ycbcr_conversion *conversion;
   const struct vk_format_ycbcr_info *format_ycbcr_info;
};

/* TODO: we should probably replace this with a push constant/uniform. */
static nir_ssa_def *
get_texture_size(struct ycbcr_state *state, nir_deref_instr *texture)
{
   if (state->image_size)
      return state->image_size;

   nir_builder *b = state->builder;
   const struct glsl_type *type = texture->type;
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 1);

   tex->op = nir_texop_txs;
   tex->sampler_dim = glsl_get_sampler_dim(type);
   tex->is_array = glsl_sampler_type_is_array(type);
   tex->is_shadow = glsl_sampler_type_is_shadow(type);
   tex->dest_type = nir_type_int32;

   tex->src[0].src_type = nir_tex_src_texture_deref;
   tex->src[0].src = nir_src_for_ssa(&texture->dest.ssa);

   nir_ssa_dest_init(&tex->instr, &tex->dest,
                     nir_tex_instr_dest_size(tex), 32, NULL);
   nir_builder_instr_insert(b, &tex->instr);

   state->image_size = nir_i2f32(b, &tex->dest.ssa);

   return state->image_size;
}

static nir_ssa_def *
implicit_downsampled_coord(nir_builder *b,
                           nir_ssa_def *value,
                           nir_ssa_def *max_value,
                           int div_scale)
{
   return nir_fadd(b,
                   value,
                   nir_fdiv(b,
                            nir_imm_float(b, 1.0f),
                            nir_fmul(b,
                                     nir_imm_float(b, div_scale),
                                     max_value)));
}

static nir_ssa_def *
implicit_downsampled_coords(struct ycbcr_state *state,
                            nir_ssa_def *old_coords,
                            const struct vk_format_ycbcr_plane *format_plane)
{
   nir_builder *b = state->builder;
   const struct vk_ycbcr_conversion *conversion = state->conversion;
   nir_ssa_def *image_size = get_texture_size(state, state->tex_deref);
   nir_ssa_def *comp[4] = { NULL, };
   int c;

   for (c = 0; c < ARRAY_SIZE(conversion->chroma_offsets); c++) {
      if (format_plane->denominator_scales[c] > 1 &&
          conversion->chroma_offsets[c] == VK_CHROMA_LOCATION_COSITED_EVEN) {
         comp[c] = implicit_downsampled_coord(b,
                                              nir_channel(b, old_coords, c),
                                              nir_channel(b, image_size, c),
                                              format_plane->denominator_scales[c]);
      } else {
         comp[c] = nir_channel(b, old_coords, c);
      }
   }

   /* Leave other coordinates untouched */
   for (; c < old_coords->num_components; c++)
      comp[c] = nir_channel(b, old_coords, c);

   return nir_vec(b, comp, old_coords->num_components);
}

static nir_ssa_def *
create_plane_tex_instr_implicit(struct ycbcr_state *state,
                                uint32_t plane)
{
   nir_builder *b = state->builder;
   const struct vk_ycbcr_conversion *conversion = state->conversion;
   const struct vk_format_ycbcr_plane *format_plane =
      &state->format_ycbcr_info->planes[plane];
   nir_tex_instr *old_tex = state->origin_tex;
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, old_tex->num_srcs + 1);

   for (uint32_t i = 0; i < old_tex->num_srcs; i++) {
      tex->src[i].src_type = old_tex->src[i].src_type;

      switch (old_tex->src[i].src_type) {
      case nir_tex_src_coord:
         if (format_plane->has_chroma && conversion->chroma_reconstruction) {
            assert(old_tex->src[i].src.is_ssa);
            tex->src[i].src =
               nir_src_for_ssa(implicit_downsampled_coords(state,
                                                           old_tex->src[i].src.ssa,
                                                           format_plane));
            break;
         }
         FALLTHROUGH;
      default:
         nir_src_copy(&tex->src[i].src, &old_tex->src[i].src, &tex->instr);
         break;
      }
   }
   tex->src[tex->num_srcs - 1].src = nir_src_for_ssa(nir_imm_int(b, plane));
   tex->src[tex->num_srcs - 1].src_type = nir_tex_src_plane;

   tex->sampler_dim = old_tex->sampler_dim;
   tex->dest_type = old_tex->dest_type;

   tex->op = old_tex->op;
   tex->coord_components = old_tex->coord_components;
   tex->is_new_style_shadow = old_tex->is_new_style_shadow;
   tex->component = old_tex->component;

   tex->texture_index = old_tex->texture_index;
   tex->sampler_index = old_tex->sampler_index;
   tex->is_array = old_tex->is_array;

   nir_ssa_dest_init(&tex->instr, &tex->dest,
                     old_tex->dest.ssa.num_components,
                     nir_dest_bit_size(old_tex->dest), NULL);
   nir_builder_instr_insert(b, &tex->instr);

   return &tex->dest.ssa;
}

static unsigned
swizzle_to_component(VkComponentSwizzle swizzle)
{
   switch (swizzle) {
   case VK_COMPONENT_SWIZZLE_R:
      return 0;
   case VK_COMPONENT_SWIZZLE_G:
      return 1;
   case VK_COMPONENT_SWIZZLE_B:
      return 2;
   case VK_COMPONENT_SWIZZLE_A:
      return 3;
   default:
      unreachable("invalid channel");
      return 0;
   }
}

struct lower_ycbcr_tex_state {
   nir_vk_ycbcr_conversion_lookup_cb cb;
   const void *cb_data;
};

static bool
lower_ycbcr_tex_instr(nir_builder *b, nir_instr *instr, void *_state)
{
   const struct lower_ycbcr_tex_state *state = _state;

   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);

   /* For the following instructions, we don't apply any change and let the
    * instruction apply to the first plane.
    */
   if (tex->op == nir_texop_txs ||
       tex->op == nir_texop_query_levels ||
       tex->op == nir_texop_lod)
      return false;

   int deref_src_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   assert(deref_src_idx >= 0);
   nir_deref_instr *deref = nir_src_as_deref(tex->src[deref_src_idx].src);

   nir_variable *var = nir_deref_instr_get_variable(deref);
   uint32_t set = var->data.descriptor_set;
   uint32_t binding = var->data.binding;

   assert(tex->texture_index == 0);
   unsigned array_index = 0;
   if (deref->deref_type != nir_deref_type_var) {
      assert(deref->deref_type == nir_deref_type_array);
      if (!nir_src_is_const(deref->arr.index))
         return false;
      array_index = nir_src_as_uint(deref->arr.index);
   }

   const struct vk_ycbcr_conversion *conversion =
      state->cb(state->cb_data, set, binding, array_index);
   if (conversion == NULL)
      return false;

   const struct vk_format_ycbcr_info *format_ycbcr_info =
      vk_format_get_ycbcr_info(conversion->format);

   /* This can happen if the driver hasn't done a good job of filtering on
    * sampler creation and lets through a VkYcbcrConversion object which isn't
    * actually YCbCr.  We're supposed to ignore those.
    */
   if (format_ycbcr_info == NULL)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   VkFormat y_format = VK_FORMAT_UNDEFINED;
   for (uint32_t p = 0; p < format_ycbcr_info->n_planes; p++) {
      if (!format_ycbcr_info->planes[p].has_chroma)
         y_format = format_ycbcr_info->planes[p].format;
   }
   assert(y_format != VK_FORMAT_UNDEFINED);
   const struct util_format_description *y_format_desc =
      util_format_description(vk_format_to_pipe_format(y_format));
   uint8_t y_bpc = y_format_desc->channel[0].size;

   /* |ycbcr_comp| holds components in the order : Cr-Y-Cb */
   nir_ssa_def *zero = nir_imm_float(b, 0.0f);
   nir_ssa_def *one = nir_imm_float(b, 1.0f);
   /* Use extra 2 channels for following swizzle */
   nir_ssa_def *ycbcr_comp[5] = { zero, zero, zero, one, zero };

   uint8_t ycbcr_bpcs[5];
   memset(ycbcr_bpcs, y_bpc, sizeof(ycbcr_bpcs));

   /* Go through all the planes and gather the samples into a |ycbcr_comp|
    * while applying a swizzle required by the spec:
    *
    *    R, G, B should respectively map to Cr, Y, Cb
    */
   for (uint32_t p = 0; p < format_ycbcr_info->n_planes; p++) {
      const struct vk_format_ycbcr_plane *format_plane =
         &format_ycbcr_info->planes[p];

      struct ycbcr_state tex_state = {
         .builder = b,
         .origin_tex = tex,
         .tex_deref = deref,
         .conversion = conversion,
         .format_ycbcr_info = format_ycbcr_info,
      };
      nir_ssa_def *plane_sample = create_plane_tex_instr_implicit(&tex_state, p);

      for (uint32_t pc = 0; pc < 4; pc++) {
         VkComponentSwizzle ycbcr_swizzle = format_plane->ycbcr_swizzle[pc];
         if (ycbcr_swizzle == VK_COMPONENT_SWIZZLE_ZERO)
            continue;

         unsigned ycbcr_component = swizzle_to_component(ycbcr_swizzle);
         ycbcr_comp[ycbcr_component] = nir_channel(b, plane_sample, pc);

         /* Also compute the number of bits for each component. */
         const struct util_format_description *plane_format_desc =
            util_format_description(vk_format_to_pipe_format(format_plane->format));
         ycbcr_bpcs[ycbcr_component] = plane_format_desc->channel[pc].size;
      }
   }

   /* Now remaps components to the order specified by the conversion. */
   nir_ssa_def *swizzled_comp[4] = { NULL, };
   uint32_t swizzled_bpcs[4] = { 0, };

   for (uint32_t i = 0; i < ARRAY_SIZE(conversion->mapping); i++) {
      /* Maps to components in |ycbcr_comp| */
      static const uint32_t swizzle_mapping[] = {
         [VK_COMPONENT_SWIZZLE_ZERO] = 4,
         [VK_COMPONENT_SWIZZLE_ONE]  = 3,
         [VK_COMPONENT_SWIZZLE_R]    = 0,
         [VK_COMPONENT_SWIZZLE_G]    = 1,
         [VK_COMPONENT_SWIZZLE_B]    = 2,
         [VK_COMPONENT_SWIZZLE_A]    = 3,
      };
      const VkComponentSwizzle m = conversion->mapping[i];

      if (m == VK_COMPONENT_SWIZZLE_IDENTITY) {
         swizzled_comp[i] = ycbcr_comp[i];
         swizzled_bpcs[i] = ycbcr_bpcs[i];
      } else {
         swizzled_comp[i] = ycbcr_comp[swizzle_mapping[m]];
         swizzled_bpcs[i] = ycbcr_bpcs[swizzle_mapping[m]];
      }
   }

   nir_ssa_def *result = nir_vec(b, swizzled_comp, 4);
   if (conversion->ycbcr_model != VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY) {
      result = nir_convert_ycbcr_to_rgb(b, conversion->ycbcr_model,
                                           conversion->ycbcr_range,
                                           result,
                                           swizzled_bpcs);
   }

   nir_ssa_def_rewrite_uses(&tex->dest.ssa, result);
   nir_instr_remove(&tex->instr);

   return true;
}

bool nir_vk_lower_ycbcr_tex(nir_shader *nir,
                            nir_vk_ycbcr_conversion_lookup_cb cb,
                            const void *cb_data)
{
   struct lower_ycbcr_tex_state state = {
      .cb = cb,
      .cb_data = cb_data,
   };

   return nir_shader_instructions_pass(nir, lower_ycbcr_tex_instr,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance,
                                       &state);
}
