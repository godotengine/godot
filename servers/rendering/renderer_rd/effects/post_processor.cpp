/*************************************************************************/
/*  post_processor.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "post_processor.h"
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

PostProcessor::PostProcessor() {
	{
		// Initialize PostProcessor
		Vector<String> tonemap_modes;
		tonemap_modes.push_back("\n");
		tonemap_modes.push_back("\n#define USE_GLOW_FILTER_BICUBIC\n");
		tonemap_modes.push_back("\n#define USE_1D_LUT\n");
		tonemap_modes.push_back("\n#define USE_GLOW_FILTER_BICUBIC\n#define USE_1D_LUT\n");
		tonemap_modes.push_back("\n#define SUBPASS\n");
		tonemap_modes.push_back("\n#define SUBPASS\n#define USE_1D_LUT\n");

		// multiview versions of our shaders
		tonemap_modes.push_back("\n#define MULTIVIEW\n");
		tonemap_modes.push_back("\n#define MULTIVIEW\n#define USE_GLOW_FILTER_BICUBIC\n");
		tonemap_modes.push_back("\n#define MULTIVIEW\n#define USE_1D_LUT\n");
		tonemap_modes.push_back("\n#define MULTIVIEW\n#define USE_GLOW_FILTER_BICUBIC\n#define USE_1D_LUT\n");
		tonemap_modes.push_back("\n#define MULTIVIEW\n#define SUBPASS\n");
		tonemap_modes.push_back("\n#define MULTIVIEW\n#define SUBPASS\n#define USE_1D_LUT\n");

		post_process.shader.initialize(tonemap_modes);

		if (!RendererCompositorRD::singleton->is_xr_enabled()) {
			post_process.shader.set_variant_enabled(TONEMAP_MODE_NORMAL_MULTIVIEW, false);
			post_process.shader.set_variant_enabled(TONEMAP_MODE_BICUBIC_GLOW_FILTER_MULTIVIEW, false);
			post_process.shader.set_variant_enabled(TONEMAP_MODE_1D_LUT_MULTIVIEW, false);
			post_process.shader.set_variant_enabled(TONEMAP_MODE_BICUBIC_GLOW_FILTER_1D_LUT_MULTIVIEW, false);
			post_process.shader.set_variant_enabled(TONEMAP_MODE_SUBPASS_MULTIVIEW, false);
			post_process.shader.set_variant_enabled(TONEMAP_MODE_SUBPASS_1D_LUT_MULTIVIEW, false);
		}

		post_process.shader_version = post_process.shader.version_create();

		for (int i = 0; i < TONEMAP_MODE_MAX; i++) {
			if (post_process.shader.is_variant_enabled(i)) {
				post_process.pipelines[i].setup(post_process.shader.version_get_shader(post_process.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
			} else {
				post_process.pipelines[i].clear();
			}
		}
	}
}

PostProcessor::~PostProcessor() {
	post_process.shader.version_free(post_process.shader_version);
}

void PostProcessor::postprocessor(RID p_source_color, RID p_dst_framebuffer, const PostProcessSettings &p_settings) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&post_process.push_constant, 0, sizeof(PostProcessPushConstant));

	post_process.push_constant.use_bcs = p_settings.use_bcs;
	post_process.push_constant.bcs[0] = p_settings.brightness;
	post_process.push_constant.bcs[1] = p_settings.contrast;
	post_process.push_constant.bcs[2] = p_settings.saturation;

	post_process.push_constant.use_glow = p_settings.use_glow;
	post_process.push_constant.glow_intensity = p_settings.glow_intensity;
	post_process.push_constant.glow_map_strength = p_settings.glow_map_strength;
	post_process.push_constant.glow_levels[0] = p_settings.glow_levels[0]; // clean this up to just pass by pointer or something
	post_process.push_constant.glow_levels[1] = p_settings.glow_levels[1];
	post_process.push_constant.glow_levels[2] = p_settings.glow_levels[2];
	post_process.push_constant.glow_levels[3] = p_settings.glow_levels[3];
	post_process.push_constant.glow_levels[4] = p_settings.glow_levels[4];
	post_process.push_constant.glow_levels[5] = p_settings.glow_levels[5];
	post_process.push_constant.glow_levels[6] = p_settings.glow_levels[6];
	post_process.push_constant.glow_texture_size[0] = p_settings.glow_texture_size.x;
	post_process.push_constant.glow_texture_size[1] = p_settings.glow_texture_size.y;
	post_process.push_constant.glow_mode = p_settings.glow_mode;
	post_process.push_constant.use_linearize = p_settings.use_linearize;
	post_process.push_constant.use_tonemap = p_settings.use_tonemap;

	int mode = p_settings.glow_use_bicubic_upscale ? TONEMAP_MODE_BICUBIC_GLOW_FILTER : TONEMAP_MODE_NORMAL;
	if (p_settings.use_1d_color_correction) {
		mode += 2;
	}

	post_process.push_constant.PostProcessor = p_settings.tonemap_mode;
	post_process.push_constant.use_auto_exposure = p_settings.use_auto_exposure;
	post_process.push_constant.exposure = p_settings.exposure;
	post_process.push_constant.white = p_settings.white;
	post_process.push_constant.auto_exposure_scale = p_settings.auto_exposure_scale;
	post_process.push_constant.luminance_multiplier = p_settings.luminance_multiplier;

	post_process.push_constant.use_color_correction = p_settings.use_color_correction;

	post_process.push_constant.use_fxaa = p_settings.use_fxaa;
	post_process.push_constant.use_debanding = p_settings.use_debanding;
	post_process.push_constant.pixel_size[0] = 1.0 / p_settings.texture_size.x;
	post_process.push_constant.pixel_size[1] = 1.0 / p_settings.texture_size.y;

	if (p_settings.view_count > 1) {
		// Use MULTIVIEW versions
		mode += 6;
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

	RID shader = post_process.shader.version_get_shader(post_process.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dst_framebuffer, RD::INITIAL_ACTION_DROP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_DROP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, post_process.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dst_framebuffer), false, RD::get_singleton()->draw_list_get_current_pass()));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_color), 0);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 1, u_exposure_texture), 1);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 2, u_glow_texture, u_glow_map), 2);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 3, u_color_correction_texture), 3);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &post_process.push_constant, sizeof(PostProcessPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void PostProcessor::postprocessor(RD::DrawListID p_subpass_draw_list, RID p_source_color, RD::FramebufferFormatID p_dst_format_id, const PostProcessSettings &p_settings) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&post_process.push_constant, 0, sizeof(PostProcessPushConstant));

	post_process.push_constant.use_bcs = p_settings.use_bcs;
	post_process.push_constant.bcs[0] = p_settings.brightness;
	post_process.push_constant.bcs[1] = p_settings.contrast;
	post_process.push_constant.bcs[2] = p_settings.saturation;

	ERR_FAIL_COND_MSG(p_settings.use_glow, "Glow is not supported when using subpasses.");
	post_process.push_constant.use_glow = p_settings.use_glow;

	int mode = p_settings.use_1d_color_correction ? TONEMAP_MODE_SUBPASS_1D_LUT : TONEMAP_MODE_SUBPASS;
	if (p_settings.view_count > 1) {
		// Use MULTIVIEW versions
		mode += 6;
	}

	post_process.push_constant.PostProcessor = p_settings.tonemap_mode;
	post_process.push_constant.use_auto_exposure = p_settings.use_auto_exposure;
	post_process.push_constant.exposure = p_settings.exposure;
	post_process.push_constant.white = p_settings.white;
	post_process.push_constant.auto_exposure_scale = p_settings.auto_exposure_scale;

	post_process.push_constant.use_color_correction = p_settings.use_color_correction;

	post_process.push_constant.use_debanding = p_settings.use_debanding;
	post_process.push_constant.luminance_multiplier = p_settings.luminance_multiplier;

	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	RID default_mipmap_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_color;
	u_source_color.uniform_type = RD::UNIFORM_TYPE_INPUT_ATTACHMENT;
	u_source_color.binding = 0;
	u_source_color.append_id(p_source_color);

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

	RID shader = post_process.shader.version_get_shader(post_process.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::get_singleton()->draw_list_bind_render_pipeline(p_subpass_draw_list, post_process.pipelines[mode].get_render_pipeline(RD::INVALID_ID, p_dst_format_id, false, RD::get_singleton()->draw_list_get_current_pass()));
	RD::get_singleton()->draw_list_bind_uniform_set(p_subpass_draw_list, uniform_set_cache->get_cache(shader, 0, u_source_color), 0);
	RD::get_singleton()->draw_list_bind_uniform_set(p_subpass_draw_list, uniform_set_cache->get_cache(shader, 1, u_exposure_texture), 1); // should be set to a default texture, it's ignored
	RD::get_singleton()->draw_list_bind_uniform_set(p_subpass_draw_list, uniform_set_cache->get_cache(shader, 2, u_glow_texture, u_glow_map), 2); // should be set to a default texture, it's ignored
	RD::get_singleton()->draw_list_bind_uniform_set(p_subpass_draw_list, uniform_set_cache->get_cache(shader, 3, u_color_correction_texture), 3);
	RD::get_singleton()->draw_list_bind_index_array(p_subpass_draw_list, material_storage->get_quad_index_array());

	RD::get_singleton()->draw_list_set_push_constant(p_subpass_draw_list, &post_process.push_constant, sizeof(PostProcessPushConstant));
	RD::get_singleton()->draw_list_draw(p_subpass_draw_list, true);
}
