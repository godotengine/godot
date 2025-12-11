/**************************************************************************/
/*  tone_mapper.cpp                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "tone_mapper.h"
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

ToneMapper::ToneMapper(bool p_use_mobile_version) {
	using_mobile_version = p_use_mobile_version;
	if (using_mobile_version) {
		// Initialize tonemapper
		Vector<String> tonemap_modes;
		tonemap_modes.push_back("\n");
		tonemap_modes.push_back("\n#define USE_1D_LUT\n");
		tonemap_modes.push_back("\n#define SUBPASS\n");
		tonemap_modes.push_back("\n#define SUBPASS\n#define USE_1D_LUT\n");

		// multiview versions of our shaders
		tonemap_modes.push_back("\n#define USE_MULTIVIEW\n");
		tonemap_modes.push_back("\n#define USE_MULTIVIEW\n#define USE_1D_LUT\n");
		tonemap_modes.push_back("\n#define USE_MULTIVIEW\n#define SUBPASS\n");
		tonemap_modes.push_back("\n#define USE_MULTIVIEW\n#define SUBPASS\n#define USE_1D_LUT\n");

		tonemap_mobile.shader.initialize(tonemap_modes);

		if (!RendererCompositorRD::get_singleton()->is_xr_enabled()) {
			tonemap_mobile.shader.set_variant_enabled(TONEMAP_MOBILE_MODE_NORMAL_MULTIVIEW, false);
			tonemap_mobile.shader.set_variant_enabled(TONEMAP_MOBILE_MODE_1D_LUT_MULTIVIEW, false);
			tonemap_mobile.shader.set_variant_enabled(TONEMAP_MOBILE_MODE_SUBPASS_MULTIVIEW, false);
			tonemap_mobile.shader.set_variant_enabled(TONEMAP_MOBILE_MODE_SUBPASS_1D_LUT_MULTIVIEW, false);
		}

		tonemap_mobile.shader_version = tonemap_mobile.shader.version_create();

		for (int i = 0; i < TONEMAP_MODE_MAX; i++) {
			if (tonemap_mobile.shader.is_variant_enabled(i)) {
				tonemap_mobile.pipelines[i].setup(tonemap_mobile.shader.version_get_shader(tonemap_mobile.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
			} else {
				tonemap_mobile.pipelines[i].clear();
			}
		}

	} else {
		// Initialize tonemapper
		Vector<String> tonemap_modes;
		tonemap_modes.push_back("\n");
		tonemap_modes.push_back("\n#define USE_GLOW_FILTER_BICUBIC\n");
		tonemap_modes.push_back("\n#define USE_1D_LUT\n");
		tonemap_modes.push_back("\n#define USE_GLOW_FILTER_BICUBIC\n#define USE_1D_LUT\n");

		// multiview versions of our shaders
		tonemap_modes.push_back("\n#define USE_MULTIVIEW\n");
		tonemap_modes.push_back("\n#define USE_MULTIVIEW\n#define USE_GLOW_FILTER_BICUBIC\n");
		tonemap_modes.push_back("\n#define USE_MULTIVIEW\n#define USE_1D_LUT\n");
		tonemap_modes.push_back("\n#define USE_MULTIVIEW\n#define USE_GLOW_FILTER_BICUBIC\n#define USE_1D_LUT\n");

		tonemap.shader.initialize(tonemap_modes);

		if (!RendererCompositorRD::get_singleton()->is_xr_enabled()) {
			tonemap.shader.set_variant_enabled(TONEMAP_MODE_NORMAL_MULTIVIEW, false);
			tonemap.shader.set_variant_enabled(TONEMAP_MODE_BICUBIC_GLOW_FILTER_MULTIVIEW, false);
			tonemap.shader.set_variant_enabled(TONEMAP_MODE_1D_LUT_MULTIVIEW, false);
			tonemap.shader.set_variant_enabled(TONEMAP_MODE_BICUBIC_GLOW_FILTER_1D_LUT_MULTIVIEW, false);
		}

		tonemap.shader_version = tonemap.shader.version_create();

		for (int i = 0; i < TONEMAP_MODE_MAX; i++) {
			if (tonemap.shader.is_variant_enabled(i)) {
				tonemap.pipelines[i].setup(tonemap.shader.version_get_shader(tonemap.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
			} else {
				tonemap.pipelines[i].clear();
			}
		}
	}
}

ToneMapper::~ToneMapper() {
	if (using_mobile_version) {
		tonemap_mobile.shader.version_free(tonemap_mobile.shader_version);
	} else {
		tonemap.shader.version_free(tonemap.shader_version);
	}
}

void ToneMapper::tonemapper(RID p_source_color, RID p_dst_framebuffer, const TonemapSettings &p_settings) {
	ERR_FAIL_COND_MSG(using_mobile_version, "Can't use the non mobile version of the tonemapper with the Mobile renderer.");
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&tonemap.push_constant, 0, sizeof(TonemapPushConstant));

	tonemap.push_constant.flags |= p_settings.use_bcs ? TONEMAP_FLAG_USE_BCS : 0;
	tonemap.push_constant.bcs[0] = p_settings.brightness;
	tonemap.push_constant.bcs[1] = p_settings.contrast;
	tonemap.push_constant.bcs[2] = p_settings.saturation;

	tonemap.push_constant.flags |= p_settings.use_glow ? TONEMAP_FLAG_USE_GLOW : 0;
	tonemap.push_constant.glow_intensity = p_settings.glow_intensity;
	tonemap.push_constant.glow_map_strength = p_settings.glow_map_strength;
	tonemap.push_constant.glow_levels[0] = p_settings.glow_levels[0]; // clean this up to just pass by pointer or something
	tonemap.push_constant.glow_levels[1] = p_settings.glow_levels[1];
	tonemap.push_constant.glow_levels[2] = p_settings.glow_levels[2];
	tonemap.push_constant.glow_levels[3] = p_settings.glow_levels[3];
	tonemap.push_constant.glow_levels[4] = p_settings.glow_levels[4];
	tonemap.push_constant.glow_levels[5] = p_settings.glow_levels[5];
	tonemap.push_constant.glow_levels[6] = p_settings.glow_levels[6];
	tonemap.push_constant.glow_texture_size[0] = p_settings.glow_texture_size.x;
	tonemap.push_constant.glow_texture_size[1] = p_settings.glow_texture_size.y;
	tonemap.push_constant.glow_mode = p_settings.glow_mode;

	int mode = p_settings.glow_use_bicubic_upscale ? TONEMAP_MODE_BICUBIC_GLOW_FILTER : TONEMAP_MODE_NORMAL;
	if (p_settings.use_1d_color_correction) {
		mode += 2;
	}

	tonemap.push_constant.tonemapper = p_settings.tonemap_mode;
	tonemap.push_constant.tonemapper_params[0] = p_settings.tonemapper_params[0];
	tonemap.push_constant.tonemapper_params[1] = p_settings.tonemapper_params[1];
	tonemap.push_constant.tonemapper_params[2] = p_settings.tonemapper_params[2];
	tonemap.push_constant.tonemapper_params[3] = p_settings.tonemapper_params[3];
	tonemap.push_constant.flags |= p_settings.use_auto_exposure ? TONEMAP_FLAG_USE_AUTO_EXPOSURE : 0;
	tonemap.push_constant.exposure = p_settings.exposure;
	tonemap.push_constant.white = p_settings.white;
	tonemap.push_constant.auto_exposure_scale = p_settings.auto_exposure_scale;
	tonemap.push_constant.luminance_multiplier = p_settings.luminance_multiplier;

	tonemap.push_constant.flags |= p_settings.use_color_correction ? TONEMAP_FLAG_USE_COLOR_CORRECTION : 0;

	tonemap.push_constant.flags |= p_settings.use_fxaa ? TONEMAP_FLAG_USE_FXAA : 0;
	if (p_settings.debanding_mode == TonemapSettings::DEBANDING_MODE_8_BIT) {
		tonemap.push_constant.flags |= TONEMAP_FLAG_USE_8_BIT_DEBANDING;
	}
	tonemap.push_constant.pixel_size[0] = 1.0 / p_settings.texture_size.x;
	tonemap.push_constant.pixel_size[1] = 1.0 / p_settings.texture_size.y;

	tonemap.push_constant.flags |= p_settings.convert_to_srgb ? TONEMAP_FLAG_CONVERT_TO_SRGB : 0;

	if (p_settings.view_count > 1) {
		// Use USE_MULTIVIEW versions
		mode += 4;
	}

	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	RID default_mipmap_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_color(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_color }));

	RD::Uniform u_exposure_texture;
	u_exposure_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u_exposure_texture.binding = 0;
	u_exposure_texture.append_id(default_sampler);
	u_exposure_texture.append_id(p_settings.exposure_texture);

	RD::Uniform u_glow_texture;
	u_glow_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u_glow_texture.binding = 0;
	u_glow_texture.append_id(default_mipmap_sampler);
	u_glow_texture.append_id(p_settings.glow_texture);

	RD::Uniform u_glow_map;
	u_glow_map.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u_glow_map.binding = 1;
	u_glow_map.append_id(default_mipmap_sampler);
	u_glow_map.append_id(p_settings.glow_map);

	RD::Uniform u_color_correction_texture;
	u_color_correction_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u_color_correction_texture.binding = 0;
	u_color_correction_texture.append_id(default_sampler);
	u_color_correction_texture.append_id(p_settings.color_correction_texture);

	RID shader = tonemap.shader.version_get_shader(tonemap.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dst_framebuffer);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, tonemap.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dst_framebuffer), false, RD::get_singleton()->draw_list_get_current_pass()));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_color), 0);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 1, u_exposure_texture), 1);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 2, u_glow_texture, u_glow_map), 2);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 3, u_color_correction_texture), 3);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &tonemap.push_constant, sizeof(TonemapPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
}

void ToneMapper::tonemapper_mobile(RID p_source_color, RID p_dst_framebuffer, const TonemapSettings &p_settings) {
	ERR_FAIL_COND_MSG(!using_mobile_version, "Can't use the mobile version of the tonemapper with the clustered renderer.");
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&tonemap_mobile.push_constant, 0, sizeof(TonemapPushConstantMobile));

	tonemap_mobile.push_constant.bcs[0] = p_settings.brightness;
	tonemap_mobile.push_constant.bcs[1] = p_settings.contrast;
	tonemap_mobile.push_constant.bcs[2] = p_settings.saturation;

	tonemap_mobile.push_constant.src_pixel_size[0] = 1.0 / p_settings.texture_size.x;
	tonemap_mobile.push_constant.src_pixel_size[1] = 1.0 / p_settings.texture_size.y;
	tonemap_mobile.push_constant.dest_pixel_size[0] = 1.0 / p_settings.dest_texture_size.x;
	tonemap_mobile.push_constant.dest_pixel_size[1] = 1.0 / p_settings.dest_texture_size.y;
	tonemap_mobile.push_constant.glow_intensity = p_settings.glow_intensity;
	tonemap_mobile.push_constant.glow_map_strength = p_settings.glow_map_strength;

	tonemap_mobile.push_constant.exposure = p_settings.exposure;
	tonemap_mobile.push_constant.white = p_settings.white;
	tonemap_mobile.push_constant.luminance_multiplier = p_settings.luminance_multiplier;

	tonemap_mobile.push_constant.tonemapper_params[0] = p_settings.tonemapper_params[0];
	tonemap_mobile.push_constant.tonemapper_params[1] = p_settings.tonemapper_params[1];
	tonemap_mobile.push_constant.tonemapper_params[2] = p_settings.tonemapper_params[2];
	tonemap_mobile.push_constant.tonemapper_params[3] = p_settings.tonemapper_params[3];

	uint32_t spec_constant = 0;
	spec_constant |= p_settings.use_bcs ? TONEMAP_MOBILE_FLAG_USE_BCS : 0;
	spec_constant |= p_settings.use_glow ? TONEMAP_MOBILE_FLAG_USE_GLOW : 0;
	spec_constant |= p_settings.glow_map_strength > 0.01 ? TONEMAP_MOBILE_FLAG_USE_GLOW_MAP : 0;
	spec_constant |= p_settings.use_color_correction ? TONEMAP_MOBILE_FLAG_USE_COLOR_CORRECTION : 0;
	spec_constant |= p_settings.use_fxaa ? TONEMAP_MOBILE_FLAG_USE_FXAA : 0;
	spec_constant |= p_settings.debanding_mode == TonemapSettings::DEBANDING_MODE_8_BIT ? TONEMAP_MOBILE_FLAG_USE_8_BIT_DEBANDING : 0;
	spec_constant |= p_settings.debanding_mode == TonemapSettings::DEBANDING_MODE_10_BIT ? TONEMAP_MOBILE_FLAG_USE_10_BIT_DEBANDING : 0;
	spec_constant |= p_settings.convert_to_srgb ? TONEMAP_MOBILE_FLAG_CONVERT_TO_SRGB : 0;
	spec_constant |= p_settings.tonemap_mode == RS::ENV_TONE_MAPPER_LINEAR ? TONEMAP_MOBILE_FLAG_TONEMAPPER_LINEAR : 0;
	spec_constant |= p_settings.tonemap_mode == RS::ENV_TONE_MAPPER_REINHARD ? TONEMAP_MOBILE_FLAG_TONEMAPPER_REINHARD : 0;
	spec_constant |= p_settings.tonemap_mode == RS::ENV_TONE_MAPPER_FILMIC ? TONEMAP_MOBILE_FLAG_TONEMAPPER_FILMIC : 0;
	spec_constant |= p_settings.tonemap_mode == RS::ENV_TONE_MAPPER_ACES ? TONEMAP_MOBILE_FLAG_TONEMAPPER_ACES : 0;
	spec_constant |= p_settings.tonemap_mode == RS::ENV_TONE_MAPPER_AGX ? TONEMAP_MOBILE_FLAG_TONEMAPPER_AGX : 0;
	spec_constant |= p_settings.glow_mode == RS::ENV_GLOW_BLEND_MODE_ADDITIVE ? TONEMAP_MOBILE_FLAG_GLOW_MODE_ADD : 0;
	spec_constant |= p_settings.glow_mode == RS::ENV_GLOW_BLEND_MODE_SCREEN ? TONEMAP_MOBILE_FLAG_GLOW_MODE_SCREEN : 0;
	spec_constant |= p_settings.glow_mode == RS::ENV_GLOW_BLEND_MODE_SOFTLIGHT ? TONEMAP_MOBILE_FLAG_GLOW_MODE_SOFTLIGHT : 0;
	spec_constant |= p_settings.glow_mode == RS::ENV_GLOW_BLEND_MODE_REPLACE ? TONEMAP_MOBILE_FLAG_GLOW_MODE_REPLACE : 0;
	spec_constant |= p_settings.glow_mode == RS::ENV_GLOW_BLEND_MODE_MIX ? TONEMAP_MOBILE_FLAG_GLOW_MODE_MIX : 0;

	int mode = p_settings.use_1d_color_correction ? TONEMAP_MOBILE_MODE_1D_LUT : TONEMAP_MOBILE_MODE_NORMAL;

	if (p_settings.view_count > 1) {
		// Use USE_MULTIVIEW versions
		mode += 4;
	}

	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	RID default_mipmap_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_color(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_color }));

	RD::Uniform u_glow_texture;
	u_glow_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u_glow_texture.binding = 0;
	u_glow_texture.append_id(default_mipmap_sampler);
	u_glow_texture.append_id(p_settings.glow_texture);

	RD::Uniform u_glow_map;
	u_glow_map.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u_glow_map.binding = 1;
	u_glow_map.append_id(default_mipmap_sampler);
	u_glow_map.append_id(p_settings.glow_map);

	RD::Uniform u_color_correction_texture;
	u_color_correction_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u_color_correction_texture.binding = 0;
	u_color_correction_texture.append_id(default_sampler);
	u_color_correction_texture.append_id(p_settings.color_correction_texture);

	RID shader = tonemap_mobile.shader.version_get_shader(tonemap_mobile.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dst_framebuffer);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, tonemap_mobile.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dst_framebuffer), false, RD::get_singleton()->draw_list_get_current_pass(), spec_constant));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_color), 0);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 1, u_glow_texture, u_glow_map), 1);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 2, u_color_correction_texture), 2);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &tonemap_mobile.push_constant, sizeof(TonemapPushConstantMobile));
	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
}

void ToneMapper::tonemapper_subpass(RD::DrawListID p_subpass_draw_list, RID p_source_color, RD::FramebufferFormatID p_dst_format_id, const TonemapSettings &p_settings) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	ERR_FAIL_COND_MSG(p_settings.use_glow, "Glow is not supported when using subpasses.");

	memset(&tonemap_mobile.push_constant, 0, sizeof(TonemapPushConstantMobile));

	tonemap_mobile.push_constant.bcs[0] = p_settings.brightness;
	tonemap_mobile.push_constant.bcs[1] = p_settings.contrast;
	tonemap_mobile.push_constant.bcs[2] = p_settings.saturation;

	tonemap_mobile.push_constant.src_pixel_size[0] = 1.0 / p_settings.texture_size.x;
	tonemap_mobile.push_constant.src_pixel_size[1] = 1.0 / p_settings.texture_size.y;
	tonemap_mobile.push_constant.glow_intensity = p_settings.glow_intensity;
	tonemap_mobile.push_constant.glow_map_strength = p_settings.glow_map_strength;

	tonemap_mobile.push_constant.exposure = p_settings.exposure;
	tonemap_mobile.push_constant.white = p_settings.white;
	tonemap_mobile.push_constant.luminance_multiplier = p_settings.luminance_multiplier;

	tonemap_mobile.push_constant.tonemapper_params[0] = p_settings.tonemapper_params[0];
	tonemap_mobile.push_constant.tonemapper_params[1] = p_settings.tonemapper_params[1];
	tonemap_mobile.push_constant.tonemapper_params[2] = p_settings.tonemapper_params[2];
	tonemap_mobile.push_constant.tonemapper_params[3] = p_settings.tonemapper_params[3];

	uint32_t spec_constant = TONEMAP_MOBILE_ADRENO_BUG;
	spec_constant |= p_settings.use_bcs ? TONEMAP_MOBILE_FLAG_USE_BCS : 0;
	//spec_constant |= p_settings.use_glow ? TONEMAP_MOBILE_FLAG_USE_GLOW : 0;
	//spec_constant |= p_settings.glow_map_strength > 0.01 ? TONEMAP_MOBILE_FLAG_USE_GLOW_MAP : 0;
	//spec_constant |= p_settings.use_color_correction ? TONEMAP_MOBILE_FLAG_USE_COLOR_CORRECTION : 0;
	//spec_constant |= p_settings.use_fxaa ? TONEMAP_MOBILE_FLAG_USE_FXAA : 0;
	spec_constant |= p_settings.debanding_mode == TonemapSettings::DEBANDING_MODE_8_BIT ? TONEMAP_MOBILE_FLAG_USE_8_BIT_DEBANDING : 0;
	spec_constant |= p_settings.convert_to_srgb ? TONEMAP_MOBILE_FLAG_CONVERT_TO_SRGB : 0;
	spec_constant |= p_settings.tonemap_mode == RS::ENV_TONE_MAPPER_LINEAR ? TONEMAP_MOBILE_FLAG_TONEMAPPER_LINEAR : 0;
	spec_constant |= p_settings.tonemap_mode == RS::ENV_TONE_MAPPER_REINHARD ? TONEMAP_MOBILE_FLAG_TONEMAPPER_REINHARD : 0;
	spec_constant |= p_settings.tonemap_mode == RS::ENV_TONE_MAPPER_FILMIC ? TONEMAP_MOBILE_FLAG_TONEMAPPER_FILMIC : 0;
	spec_constant |= p_settings.tonemap_mode == RS::ENV_TONE_MAPPER_ACES ? TONEMAP_MOBILE_FLAG_TONEMAPPER_ACES : 0;
	spec_constant |= p_settings.tonemap_mode == RS::ENV_TONE_MAPPER_AGX ? TONEMAP_MOBILE_FLAG_TONEMAPPER_AGX : 0;
	//spec_constant |= p_settings.glow_mode == RS::ENV_GLOW_BLEND_MODE_ADDITIVE ? TONEMAP_MOBILE_FLAG_GLOW_MODE_ADD : 0;
	//spec_constant |= p_settings.glow_mode == RS::ENV_GLOW_BLEND_MODE_SCREEN ? TONEMAP_MOBILE_FLAG_GLOW_MODE_SCREEN : 0;
	//spec_constant |= p_settings.glow_mode == RS::ENV_GLOW_BLEND_MODE_SOFTLIGHT ? TONEMAP_MOBILE_FLAG_GLOW_MODE_SOFTLIGHT : 0;
	//spec_constant |= p_settings.glow_mode == RS::ENV_GLOW_BLEND_MODE_REPLACE ? TONEMAP_MOBILE_FLAG_GLOW_MODE_REPLACE : 0;
	//spec_constant |= p_settings.glow_mode == RS::ENV_GLOW_BLEND_MODE_MIX ? TONEMAP_MOBILE_FLAG_GLOW_MODE_MIX : 0;

	int mode = p_settings.use_1d_color_correction ? TONEMAP_MOBILE_MODE_SUBPASS_1D_LUT : TONEMAP_MOBILE_MODE_SUBPASS;
	if (p_settings.view_count > 1) {
		// Use USE_MULTIVIEW versions
		mode += 4;
	}

	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	RID default_mipmap_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_color;
	u_source_color.uniform_type = RD::UNIFORM_TYPE_INPUT_ATTACHMENT;
	u_source_color.binding = 0;
	u_source_color.append_id(p_source_color);

	RD::Uniform u_glow_texture;
	u_glow_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u_glow_texture.binding = 0;
	u_glow_texture.append_id(default_mipmap_sampler);
	u_glow_texture.append_id(p_settings.glow_texture);

	RD::Uniform u_glow_map;
	u_glow_map.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u_glow_map.binding = 1;
	u_glow_map.append_id(default_mipmap_sampler);
	u_glow_map.append_id(p_settings.glow_map);

	RD::Uniform u_color_correction_texture;
	u_color_correction_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u_color_correction_texture.binding = 0;
	u_color_correction_texture.append_id(default_sampler);
	u_color_correction_texture.append_id(p_settings.color_correction_texture);

	RID shader = tonemap_mobile.shader.version_get_shader(tonemap_mobile.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::get_singleton()->draw_list_bind_render_pipeline(p_subpass_draw_list, tonemap_mobile.pipelines[mode].get_render_pipeline(RD::INVALID_ID, p_dst_format_id, false, RD::get_singleton()->draw_list_get_current_pass(), spec_constant));
	RD::get_singleton()->draw_list_bind_uniform_set(p_subpass_draw_list, uniform_set_cache->get_cache(shader, 0, u_source_color), 0);
	RD::get_singleton()->draw_list_bind_uniform_set(p_subpass_draw_list, uniform_set_cache->get_cache(shader, 1, u_glow_texture, u_glow_map), 1); // should be set to a default texture, it's ignored
	RD::get_singleton()->draw_list_bind_uniform_set(p_subpass_draw_list, uniform_set_cache->get_cache(shader, 2, u_color_correction_texture), 2);

	RD::get_singleton()->draw_list_set_push_constant(p_subpass_draw_list, &tonemap_mobile.push_constant, sizeof(TonemapPushConstantMobile));
	RD::get_singleton()->draw_list_draw(p_subpass_draw_list, false, 1u, 3u);
}
