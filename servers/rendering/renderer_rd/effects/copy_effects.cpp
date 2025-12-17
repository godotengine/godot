/**************************************************************************/
/*  copy_effects.cpp                                                      */
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

#include "copy_effects.h"
#include "core/config/project_settings.h"
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "thirdparty/misc/cubemap_coeffs.h"

using namespace RendererRD;

CopyEffects *CopyEffects::singleton = nullptr;

CopyEffects *CopyEffects::get_singleton() {
	return singleton;
}

CopyEffects::CopyEffects(BitField<RasterEffects> p_raster_effects) {
	singleton = this;
	raster_effects = p_raster_effects;

	if (raster_effects.has_flag(RASTER_EFFECT_GAUSSIAN_BLUR)) {
		// init blur shader (on compute use copy shader)

		Vector<String> blur_modes;
		blur_modes.push_back("\n#define MODE_MIPMAP\n"); // BLUR_MIPMAP
		blur_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n"); // BLUR_MODE_GAUSSIAN_BLUR
		blur_modes.push_back("\n#define MODE_GLOW_GATHER\n"); // BLUR_MODE_GAUSSIAN_GLOW_GATHER
		blur_modes.push_back("\n#define MODE_GLOW_DOWNSAMPLE\n"); // BLUR_MODE_GAUSSIAN_GLOW_DOWNSAMPLE
		blur_modes.push_back("\n#define MODE_GLOW_UPSAMPLE\n"); // BLUR_MODE_GAUSSIAN_GLOW_UPSAMPLE
		blur_modes.push_back("\n#define MODE_COPY\n"); // BLUR_MODE_COPY
		blur_modes.push_back("\n#define MODE_SET_COLOR\n"); // BLUR_MODE_SET_COLOR

		blur_raster.shader.initialize(blur_modes);
		memset(&blur_raster.push_constant, 0, sizeof(BlurRasterPushConstant));
		blur_raster.shader_version = blur_raster.shader.version_create();

		for (int i = 0; i < BLUR_MODE_MAX; i++) {
			blur_raster.pipelines[i].setup(blur_raster.shader.version_get_shader(blur_raster.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
		}

		RD::SamplerState sampler_state;
		sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
		sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
		sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_BORDER;
		sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_BORDER;
		sampler_state.border_color = RD::SAMPLER_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;

		blur_raster.glow_sampler = RD::get_singleton()->sampler_create(sampler_state);

	} else {
		// not used in clustered
		for (int i = 0; i < BLUR_MODE_MAX; i++) {
			blur_raster.pipelines[i].clear();
		}
	}

	{
		Vector<String> copy_modes;
		copy_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n");
		copy_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n#define DST_IMAGE_8BIT\n");
		copy_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n#define MODE_GLOW\n");
		copy_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n#define MODE_GLOW\n#define GLOW_USE_AUTO_EXPOSURE\n");
		copy_modes.push_back("\n#define MODE_SIMPLE_COPY\n");
		copy_modes.push_back("\n#define MODE_SIMPLE_COPY\n#define DST_IMAGE_8BIT\n");
		copy_modes.push_back("\n#define MODE_SIMPLE_COPY_DEPTH\n");
		copy_modes.push_back("\n#define MODE_SET_COLOR\n");
		copy_modes.push_back("\n#define MODE_SET_COLOR\n#define DST_IMAGE_8BIT\n");
		copy_modes.push_back("\n#define MODE_MIPMAP\n");
		copy_modes.push_back("\n#define MODE_LINEARIZE_DEPTH_COPY\n");
		copy_modes.push_back("\n#define MODE_OCTMAP_TO_PANORAMA\n");
		copy_modes.push_back("\n#define MODE_OCTMAP_ARRAY_TO_PANORAMA\n");

		copy.shader.initialize(copy_modes);
		memset(&copy.push_constant, 0, sizeof(CopyPushConstant));

		copy.shader_version = copy.shader.version_create();

		for (int i = 0; i < COPY_MODE_MAX; i++) {
			if (copy.shader.is_variant_enabled(i)) {
				copy.pipelines[i].create_compute_pipeline(copy.shader.version_get_shader(copy.shader_version, i));
			}
		}
	}

	{
		Vector<String> copy_modes;
		copy_modes.push_back("\n"); // COPY_TO_FB_COPY
		copy_modes.push_back("\n#define MODE_PANORAMA_TO_DP\n"); // COPY_TO_FB_COPY_PANORAMA_TO_DP
		copy_modes.push_back("\n#define MODE_TWO_SOURCES\n"); // COPY_TO_FB_COPY2
		copy_modes.push_back("\n#define MODE_SET_COLOR\n"); // COPY_TO_FB_SET_COLOR
		copy_modes.push_back("\n#define USE_MULTIVIEW\n"); // COPY_TO_FB_MULTIVIEW
		copy_modes.push_back("\n#define USE_MULTIVIEW\n#define MODE_TWO_SOURCES\n"); // COPY_TO_FB_MULTIVIEW_WITH_DEPTH

		copy_to_fb.shader.initialize(copy_modes);

		if (!RendererCompositorRD::get_singleton()->is_xr_enabled()) {
			copy_to_fb.shader.set_variant_enabled(COPY_TO_FB_MULTIVIEW, false);
			copy_to_fb.shader.set_variant_enabled(COPY_TO_FB_MULTIVIEW_WITH_DEPTH, false);
		}

		copy_to_fb.shader_version = copy_to_fb.shader.version_create();

		//use additive

		for (int i = 0; i < COPY_TO_FB_MAX; i++) {
			if (copy_to_fb.shader.is_variant_enabled(i)) {
				copy_to_fb.pipelines[i].setup(copy_to_fb.shader.version_get_shader(copy_to_fb.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
			} else {
				copy_to_fb.pipelines[i].clear();
			}
		}
	}

	{
		// Initialize copier
		Vector<String> copy_modes;
		copy_modes.push_back("\n");

		cube_to_dp.shader.initialize(copy_modes);

		cube_to_dp.shader_version = cube_to_dp.shader.version_create();
		RID shader = cube_to_dp.shader.version_get_shader(cube_to_dp.shader_version, 0);
		RD::PipelineDepthStencilState dss;
		dss.enable_depth_test = true;
		dss.depth_compare_operator = RD::COMPARE_OP_ALWAYS;
		dss.enable_depth_write = true;
		cube_to_dp.pipeline.setup(shader, RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), dss, RD::PipelineColorBlendState(), 0);
	}

	{
		// Initialize cubemap to octmap copier.
		cube_to_octmap.shader.initialize({ "" });
		cube_to_octmap.shader_version = cube_to_octmap.shader.version_create();
		RID shader = cube_to_octmap.shader.version_get_shader(cube_to_octmap.shader_version, 0);
		cube_to_octmap.pipeline.setup(shader, RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled());
	}

	{
		// Initialize octmap downsampler.
		if (raster_effects.has_flag(RASTER_EFFECT_OCTMAP)) {
			octmap_downsampler.raster_shader.initialize({ "", "\n#define USE_HIGH_QUALITY\n" });
			octmap_downsampler.shader_version = octmap_downsampler.raster_shader.version_create();
			for (int i = 0; i < DOWNSAMPLER_MODE_MAX; i++) {
				octmap_downsampler.raster_pipelines[i].setup(octmap_downsampler.raster_shader.version_get_shader(octmap_downsampler.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
			}
		} else {
			octmap_downsampler.compute_shader.initialize({ "", "\n#define USE_HIGH_QUALITY\n" });
			octmap_downsampler.shader_version = octmap_downsampler.compute_shader.version_create();
			for (int i = 0; i < DOWNSAMPLER_MODE_MAX; i++) {
				octmap_downsampler.compute_pipelines[i].create_compute_pipeline(octmap_downsampler.compute_shader.version_get_shader(octmap_downsampler.shader_version, i));
			}
		}
	}

	{
		// Initialize cubemap filter
		filter.use_high_quality = GLOBAL_GET("rendering/reflections/sky_reflections/fast_filter_high_quality");

		Vector<String> cubemap_filter_modes;
		cubemap_filter_modes.push_back("\n#define USE_HIGH_QUALITY\n");
		cubemap_filter_modes.push_back("\n#define USE_LOW_QUALITY\n");
		cubemap_filter_modes.push_back("\n#define USE_HIGH_QUALITY\n#define USE_TEXTURE_ARRAY\n");
		cubemap_filter_modes.push_back("\n#define USE_LOW_QUALITY\n#define USE_TEXTURE_ARRAY\n");

		if (filter.use_high_quality) {
			filter.coefficient_buffer = RD::get_singleton()->storage_buffer_create(sizeof(high_quality_coeffs));
			RD::get_singleton()->buffer_update(filter.coefficient_buffer, 0, sizeof(high_quality_coeffs), &high_quality_coeffs[0]);
		} else {
			filter.coefficient_buffer = RD::get_singleton()->storage_buffer_create(sizeof(low_quality_coeffs));
			RD::get_singleton()->buffer_update(filter.coefficient_buffer, 0, sizeof(low_quality_coeffs), &low_quality_coeffs[0]);
		}

		if (raster_effects.has_flag(RASTER_EFFECT_OCTMAP)) {
			filter.raster_shader.initialize(cubemap_filter_modes);

			// array variants are not supported in raster
			filter.raster_shader.set_variant_enabled(FILTER_MODE_HIGH_QUALITY_ARRAY, false);
			filter.raster_shader.set_variant_enabled(FILTER_MODE_LOW_QUALITY_ARRAY, false);

			filter.shader_version = filter.raster_shader.version_create();

			for (int i = 0; i < FILTER_MODE_MAX; i++) {
				if (filter.raster_shader.is_variant_enabled(i)) {
					filter.raster_pipelines[i].setup(filter.raster_shader.version_get_shader(filter.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
				} else {
					filter.raster_pipelines[i].clear();
				}
			}

			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 0;
				u.append_id(filter.coefficient_buffer);
				uniforms.push_back(u);
			}
			filter.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, filter.raster_shader.version_get_shader(filter.shader_version, filter.use_high_quality ? 0 : 1), 1);
		} else {
			filter.compute_shader.initialize(cubemap_filter_modes);
			filter.shader_version = filter.compute_shader.version_create();

			for (int i = 0; i < FILTER_MODE_MAX; i++) {
				filter.compute_pipelines[i].create_compute_pipeline(filter.compute_shader.version_get_shader(filter.shader_version, i));
				filter.raster_pipelines[i].clear();
			}

			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 0;
				u.append_id(filter.coefficient_buffer);
				uniforms.push_back(u);
			}
			filter.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, filter.compute_shader.version_get_shader(filter.shader_version, filter.use_high_quality ? 0 : 1), 1);
		}
	}

	{
		// Initialize roughness
		Vector<String> cubemap_roughness_modes;
		cubemap_roughness_modes.push_back("");

		if (raster_effects.has_flag(RASTER_EFFECT_OCTMAP)) {
			roughness.raster_shader.initialize(cubemap_roughness_modes);

			roughness.shader_version = roughness.raster_shader.version_create();

			roughness.raster_pipeline.setup(roughness.raster_shader.version_get_shader(roughness.shader_version, 0), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);

		} else {
			roughness.compute_shader.initialize(cubemap_roughness_modes);

			roughness.shader_version = roughness.compute_shader.version_create();

			roughness.compute_pipeline.create_compute_pipeline(roughness.compute_shader.version_get_shader(roughness.shader_version, 0));
			roughness.raster_pipeline.clear();
		}
	}

	{
		Vector<String> specular_modes;
		specular_modes.push_back("\n#define MODE_MERGE\n"); // SPECULAR_MERGE_ADD
		specular_modes.push_back("\n#define MODE_MERGE\n#define MODE_SSR\n"); // SPECULAR_MERGE_SSR
		specular_modes.push_back("\n"); // SPECULAR_MERGE_ADDITIVE_ADD
		specular_modes.push_back("\n#define MODE_SSR\n"); // SPECULAR_MERGE_ADDITIVE_SSR

		specular_modes.push_back("\n#define USE_MULTIVIEW\n#define MODE_MERGE\n"); // SPECULAR_MERGE_ADD_MULTIVIEW
		specular_modes.push_back("\n#define USE_MULTIVIEW\n#define MODE_MERGE\n#define MODE_SSR\n"); // SPECULAR_MERGE_SSR_MULTIVIEW
		specular_modes.push_back("\n#define USE_MULTIVIEW\n"); // SPECULAR_MERGE_ADDITIVE_ADD_MULTIVIEW
		specular_modes.push_back("\n#define USE_MULTIVIEW\n#define MODE_SSR\n"); // SPECULAR_MERGE_ADDITIVE_SSR_MULTIVIEW

		specular_merge.shader.initialize(specular_modes);

		if (!RendererCompositorRD::get_singleton()->is_xr_enabled()) {
			specular_merge.shader.set_variant_enabled(SPECULAR_MERGE_ADD_MULTIVIEW, false);
			specular_merge.shader.set_variant_enabled(SPECULAR_MERGE_SSR_MULTIVIEW, false);
			specular_merge.shader.set_variant_enabled(SPECULAR_MERGE_ADDITIVE_ADD_MULTIVIEW, false);
			specular_merge.shader.set_variant_enabled(SPECULAR_MERGE_ADDITIVE_SSR_MULTIVIEW, false);
		}

		specular_merge.shader_version = specular_merge.shader.version_create();

		//use additive

		RD::PipelineColorBlendState::Attachment ba;
		ba.enable_blend = true;
		ba.src_color_blend_factor = RD::BLEND_FACTOR_ONE;
		ba.dst_color_blend_factor = RD::BLEND_FACTOR_ONE;
		ba.src_alpha_blend_factor = RD::BLEND_FACTOR_ZERO;
		ba.dst_alpha_blend_factor = RD::BLEND_FACTOR_ZERO;
		ba.color_blend_op = RD::BLEND_OP_ADD;
		ba.alpha_blend_op = RD::BLEND_OP_ADD;

		RD::PipelineColorBlendState blend_additive;
		blend_additive.attachments.push_back(ba);

		for (int i = 0; i < SPECULAR_MERGE_MAX; i++) {
			if (specular_merge.shader.is_variant_enabled(i)) {
				RD::PipelineColorBlendState blend_state;
				if (i == SPECULAR_MERGE_ADDITIVE_ADD || i == SPECULAR_MERGE_ADDITIVE_SSR || i == SPECULAR_MERGE_ADDITIVE_ADD_MULTIVIEW || i == SPECULAR_MERGE_ADDITIVE_SSR_MULTIVIEW) {
					blend_state = blend_additive;
				} else {
					blend_state = RD::PipelineColorBlendState::create_disabled();
				}
				specular_merge.pipelines[i].setup(specular_merge.shader.version_get_shader(specular_merge.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), blend_state, 0);
			}
		}
	}
}

CopyEffects::~CopyEffects() {
	for (int i = 0; i < COPY_MODE_MAX; i++) {
		copy.pipelines[i].free();
	}

	if (raster_effects.has_flag(RASTER_EFFECT_GAUSSIAN_BLUR)) {
		blur_raster.shader.version_free(blur_raster.shader_version);
		RD::get_singleton()->free_rid(blur_raster.glow_sampler);
	}

	if (raster_effects.has_flag(RASTER_EFFECT_OCTMAP)) {
		octmap_downsampler.raster_shader.version_free(octmap_downsampler.shader_version);
		filter.raster_shader.version_free(filter.shader_version);
		roughness.raster_shader.version_free(roughness.shader_version);
	} else {
		// PipelineDeferredRD always needs to be freed before its corresponding shader since the pipeline may not have finished compiling before the shader is freed. This
		// ensures that we wait on the pipeline compilation before we free it.
		for (int i = 0; i < DOWNSAMPLER_MODE_MAX; i++) {
			octmap_downsampler.compute_pipelines[i].free();
		}
		octmap_downsampler.compute_shader.version_free(octmap_downsampler.shader_version);

		for (int i = 0; i < FILTER_MODE_MAX; i++) {
			filter.compute_pipelines[i].free();
		}
		filter.compute_shader.version_free(filter.shader_version);

		roughness.compute_pipeline.free();
		roughness.compute_shader.version_free(roughness.shader_version);
	}

	copy.shader.version_free(copy.shader_version);
	specular_merge.shader.version_free(specular_merge.shader_version);

	RD::get_singleton()->free_rid(filter.coefficient_buffer);

	if (RD::get_singleton()->uniform_set_is_valid(filter.image_uniform_set)) {
		RD::get_singleton()->free_rid(filter.image_uniform_set);
	}

	if (RD::get_singleton()->uniform_set_is_valid(filter.uniform_set)) {
		RD::get_singleton()->free_rid(filter.uniform_set);
	}

	copy_to_fb.shader.version_free(copy_to_fb.shader_version);
	cube_to_dp.shader.version_free(cube_to_dp.shader_version);
	cube_to_octmap.shader.version_free(cube_to_octmap.shader_version);

	singleton = nullptr;
}

void CopyEffects::copy_to_rect(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_rect, bool p_flip_y, bool p_force_luminance, bool p_all_source, bool p_8_bit_dst, bool p_alpha_to_one, bool p_sanitize_inf_nan) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));
	if (p_flip_y) {
		copy.push_constant.flags |= COPY_FLAG_FLIP_Y;
	}

	if (p_force_luminance) {
		copy.push_constant.flags |= COPY_FLAG_FORCE_LUMINANCE;
	}

	if (p_all_source) {
		copy.push_constant.flags |= COPY_FLAG_ALL_SOURCE;
	}

	if (p_alpha_to_one) {
		copy.push_constant.flags |= COPY_FLAG_ALPHA_TO_ONE;
	}

	if (p_sanitize_inf_nan) {
		copy.push_constant.flags |= COPY_FLAG_SANITIZE_INF_NAN;
	}

	copy.push_constant.section[0] = p_rect.position.x;
	copy.push_constant.section[1] = p_rect.position.y;
	copy.push_constant.section[2] = p_rect.size.width;
	copy.push_constant.section[3] = p_rect.size.height;
	copy.push_constant.target[0] = p_rect.position.x;
	copy.push_constant.target[1] = p_rect.position.y;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_dest_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_dest_texture);

	CopyMode mode = p_8_bit_dst ? COPY_MODE_SIMPLY_COPY_8BIT : COPY_MODE_SIMPLY_COPY;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode].get_rid());
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_rect.size.width, p_rect.size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::copy_octmap_to_panorama(RID p_source_octmap, RID p_dest_panorama, const Size2i &p_panorama_size, float p_lod, bool p_is_array, const Size2 &p_source_octmap_border_size) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_panorama_size.width;
	copy.push_constant.section[3] = p_panorama_size.height;
	copy.push_constant.target[0] = 0;
	copy.push_constant.target[1] = 0;
	copy.push_constant.camera_z_far = p_lod;
	copy.push_constant.octmap_border_size[0] = p_source_octmap_border_size.x;
	copy.push_constant.octmap_border_size[1] = p_source_octmap_border_size.y;

	// TODO, if this is needed at the copy stage, then we need to pass in the multiplier.
	copy.push_constant.luminance_multiplier = raster_effects.has_flag(RASTER_EFFECT_COPY) ? 2.0 : 1.0;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_octmap(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_octmap }));
	RD::Uniform u_dest_panorama(RD::UNIFORM_TYPE_IMAGE, 0, p_dest_panorama);

	CopyMode mode = p_is_array ? COPY_MODE_OCTMAP_ARRAY_TO_PANORAMA : COPY_MODE_OCTMAP_TO_PANORAMA;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode].get_rid());
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_octmap), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_dest_panorama), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_panorama_size.width, p_panorama_size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::copy_depth_to_rect(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_rect, bool p_flip_y) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));
	if (p_flip_y) {
		copy.push_constant.flags |= COPY_FLAG_FLIP_Y;
	}

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_rect.size.width;
	copy.push_constant.section[3] = p_rect.size.height;
	copy.push_constant.target[0] = p_rect.position.x;
	copy.push_constant.target[1] = p_rect.position.y;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_dest_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_dest_texture);

	CopyMode mode = COPY_MODE_SIMPLY_COPY_DEPTH;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode].get_rid());
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_rect.size.width, p_rect.size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::copy_depth_to_rect_and_linearize(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_rect, bool p_flip_y, float p_z_near, float p_z_far) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));
	if (p_flip_y) {
		copy.push_constant.flags |= COPY_FLAG_FLIP_Y;
	}

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_rect.size.width;
	copy.push_constant.section[3] = p_rect.size.height;
	copy.push_constant.target[0] = p_rect.position.x;
	copy.push_constant.target[1] = p_rect.position.y;
	copy.push_constant.camera_z_far = p_z_far;
	copy.push_constant.camera_z_near = p_z_near;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_dest_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_dest_texture);

	CopyMode mode = COPY_MODE_LINEARIZE_DEPTH;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode].get_rid());
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_rect.size.width, p_rect.size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::copy_to_atlas_fb(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2 &p_uv_rect, RD::DrawListID p_draw_list, bool p_flip_y, bool p_panorama) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy_to_fb.push_constant, 0, sizeof(CopyToFbPushConstant));

	copy_to_fb.push_constant.flags |= COPY_TO_FB_FLAG_USE_SECTION;
	copy_to_fb.push_constant.section[0] = p_uv_rect.position.x;
	copy_to_fb.push_constant.section[1] = p_uv_rect.position.y;
	copy_to_fb.push_constant.section[2] = p_uv_rect.size.x;
	copy_to_fb.push_constant.section[3] = p_uv_rect.size.y;

	if (p_flip_y) {
		copy_to_fb.push_constant.flags |= COPY_TO_FB_FLAG_FLIP_Y;
	}

	copy_to_fb.push_constant.luminance_multiplier = 1.0;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));

	CopyToFBMode mode = p_panorama ? COPY_TO_FB_COPY_PANORAMA_TO_DP : COPY_TO_FB_COPY;
	RID shader = copy_to_fb.shader.version_get_shader(copy_to_fb.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = p_draw_list;
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, copy_to_fb.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &copy_to_fb.push_constant, sizeof(CopyToFbPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
}

void CopyEffects::copy_to_fb_rect(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2i &p_rect, bool p_flip_y, bool p_force_luminance, bool p_alpha_to_zero, bool p_srgb, RID p_secondary, bool p_multiview, bool p_alpha_to_one, bool p_linear, bool p_normal, const Rect2 &p_src_rect, float p_linear_luminance_multiplier) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy_to_fb.push_constant, 0, sizeof(CopyToFbPushConstant));
	copy_to_fb.push_constant.luminance_multiplier = 1.0;

	if (p_flip_y) {
		copy_to_fb.push_constant.flags |= COPY_TO_FB_FLAG_FLIP_Y;
	}
	if (p_force_luminance) {
		copy_to_fb.push_constant.flags |= COPY_TO_FB_FLAG_FORCE_LUMINANCE;
	}
	if (p_alpha_to_zero) {
		copy_to_fb.push_constant.flags |= COPY_TO_FB_FLAG_ALPHA_TO_ZERO;
	}
	if (p_srgb) {
		copy_to_fb.push_constant.flags |= COPY_TO_FB_FLAG_SRGB;
	}
	if (p_alpha_to_one) {
		copy_to_fb.push_constant.flags |= COPY_TO_FB_FLAG_ALPHA_TO_ONE;
	}
	if (p_linear) {
		copy_to_fb.push_constant.flags |= COPY_TO_FB_FLAG_LINEAR;
		copy_to_fb.push_constant.luminance_multiplier = p_linear_luminance_multiplier;
	}

	if (p_normal) {
		copy_to_fb.push_constant.flags |= COPY_TO_FB_FLAG_NORMAL;
	}

	if (p_src_rect != Rect2()) {
		copy_to_fb.push_constant.section[0] = p_src_rect.position.x;
		copy_to_fb.push_constant.section[1] = p_src_rect.position.y;
		copy_to_fb.push_constant.section[2] = p_src_rect.size.x;
		copy_to_fb.push_constant.section[3] = p_src_rect.size.y;
		copy_to_fb.push_constant.flags |= COPY_TO_FB_FLAG_USE_SRC_SECTION;
	}

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));

	CopyToFBMode mode;
	if (p_multiview) {
		mode = p_secondary.is_valid() ? COPY_TO_FB_MULTIVIEW_WITH_DEPTH : COPY_TO_FB_MULTIVIEW;
	} else {
		mode = p_secondary.is_valid() ? COPY_TO_FB_COPY2 : COPY_TO_FB_COPY;
	}

	RID shader = copy_to_fb.shader.version_get_shader(copy_to_fb.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::DRAW_DEFAULT_ALL, Vector<Color>(), 1.0f, 0, p_rect);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, copy_to_fb.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	if (p_secondary.is_valid()) {
		// TODO may need to do this differently when reading from depth buffer for multiview
		RD::Uniform u_secondary(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_secondary }));
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 1, u_secondary), 1);
	}
	RD::get_singleton()->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &copy_to_fb.push_constant, sizeof(CopyToFbPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::copy_to_drawlist(RD::DrawListID p_draw_list, RD::FramebufferFormatID p_fb_format, RID p_source_rd_texture, bool p_linear, float p_linear_luminance_multiplier) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy_to_fb.push_constant, 0, sizeof(CopyToFbPushConstant));
	copy_to_fb.push_constant.luminance_multiplier = 1.0;

	if (p_linear) {
		copy_to_fb.push_constant.flags |= COPY_TO_FB_FLAG_LINEAR;
		copy_to_fb.push_constant.luminance_multiplier = p_linear_luminance_multiplier;
	}

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));

	// Multiview not supported here!
	CopyToFBMode mode = COPY_TO_FB_COPY;

	RID shader = copy_to_fb.shader.version_get_shader(copy_to_fb.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, copy_to_fb.pipelines[mode].get_render_pipeline(RD::INVALID_ID, p_fb_format));
	RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(p_draw_list, material_storage->get_quad_index_array());
	RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &copy_to_fb.push_constant, sizeof(CopyToFbPushConstant));
	RD::get_singleton()->draw_list_draw(p_draw_list, true);
}

void CopyEffects::copy_raster(RID p_source_texture, RID p_dest_framebuffer) {
	ERR_FAIL_COND_MSG(!raster_effects.has_flag(RASTER_EFFECT_COPY), "Can't use the raster version of the copy.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&blur_raster.push_constant, 0, sizeof(BlurRasterPushConstant));

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_texture }));

	RID shader = blur_raster.shader.version_get_shader(blur_raster.shader_version, BLUR_MODE_COPY);
	ERR_FAIL_COND(shader.is_null());

	// Just copy it back (we use our blur raster shader here)..
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur_raster.pipelines[BLUR_MODE_COPY].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_texture), 0);
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur_raster.push_constant, sizeof(BlurRasterPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::gaussian_blur(RID p_source_rd_texture, RID p_texture, const Rect2i &p_region, const Size2i &p_size, bool p_8bit_dst) {
	ERR_FAIL_COND_MSG(raster_effects.has_flag(RASTER_EFFECT_GAUSSIAN_BLUR), "Can't use the compute version of the gaussian blur.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));

	copy.push_constant.section[0] = p_region.position.x;
	copy.push_constant.section[1] = p_region.position.y;
	copy.push_constant.target[0] = p_region.position.x;
	copy.push_constant.target[1] = p_region.position.y;
	copy.push_constant.section[2] = p_size.width;
	copy.push_constant.section[3] = p_size.height;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_texture);

	CopyMode mode = p_8bit_dst ? COPY_MODE_GAUSSIAN_COPY_8BIT : COPY_MODE_GAUSSIAN_COPY;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode].get_rid());
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_texture), 3);

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_region.size.width, p_region.size.height, 1);

	RD::get_singleton()->compute_list_end();
}

void CopyEffects::gaussian_blur_raster(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_region, const Size2i &p_size) {
	ERR_FAIL_COND_MSG(!raster_effects.has_flag(RASTER_EFFECT_GAUSSIAN_BLUR), "Can't use the raster version of the gaussian blur.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	RID dest_framebuffer = FramebufferCacheRD::get_singleton()->get_cache(p_dest_texture);

	memset(&blur_raster.push_constant, 0, sizeof(BlurRasterPushConstant));

	BlurRasterMode blur_mode = BLUR_MODE_GAUSSIAN_BLUR;

	blur_raster.push_constant.dest_pixel_size[0] = 1.0 / float(p_size.x);
	blur_raster.push_constant.dest_pixel_size[1] = 1.0 / float(p_size.y);

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));

	RID shader = blur_raster.shader.version_get_shader(blur_raster.shader_version, blur_mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(dest_framebuffer);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur_raster.pipelines[blur_mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur_raster.push_constant, sizeof(BlurRasterPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::gaussian_glow(RID p_source_rd_texture, RID p_back_texture, const Size2i &p_size, float p_strength, bool p_first_pass, float p_luminance_cap, float p_exposure, float p_bloom, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, RID p_auto_exposure, float p_auto_exposure_scale) {
	ERR_FAIL_COND_MSG(raster_effects.has_flag(RASTER_EFFECT_GAUSSIAN_BLUR), "Can't use the compute version of the gaussian glow.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));

	CopyMode copy_mode = p_first_pass && p_auto_exposure.is_valid() ? COPY_MODE_GAUSSIAN_GLOW_AUTO_EXPOSURE : COPY_MODE_GAUSSIAN_GLOW;
	uint32_t base_flags = 0;

	copy.push_constant.section[2] = p_size.x;
	copy.push_constant.section[3] = p_size.y;

	copy.push_constant.glow_strength = p_strength;
	copy.push_constant.glow_bloom = p_bloom;
	copy.push_constant.glow_hdr_threshold = p_hdr_bleed_threshold;
	copy.push_constant.glow_hdr_scale = p_hdr_bleed_scale;
	copy.push_constant.glow_exposure = p_exposure;
	copy.push_constant.glow_white = 0; //actually unused
	copy.push_constant.glow_luminance_cap = p_luminance_cap;

	copy.push_constant.glow_auto_exposure_scale = p_auto_exposure_scale; //unused also

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_back_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_back_texture);

	RID shader = copy.shader.version_get_shader(copy.shader_version, copy_mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[copy_mode].get_rid());
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_back_texture), 3);
	if (p_auto_exposure.is_valid() && p_first_pass) {
		RD::Uniform u_auto_exposure(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_auto_exposure }));
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_auto_exposure), 1);
	}

	copy.push_constant.flags = base_flags | (p_first_pass ? COPY_FLAG_GLOW_FIRST_PASS : 0);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_size.width, p_size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::gaussian_glow_downsample_raster(RID p_source_rd_texture, RID p_dest_texture, float p_luminance_multiplier, const Size2i &p_size, float p_strength, bool p_first_pass, float p_luminance_cap, float p_exposure, float p_bloom, float p_hdr_bleed_threshold, float p_hdr_bleed_scale) {
	ERR_FAIL_COND_MSG(!raster_effects.has_flag(RASTER_EFFECT_GAUSSIAN_BLUR), "Can't use the raster version of the gaussian glow.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	RID dest_framebuffer = FramebufferCacheRD::get_singleton()->get_cache(p_dest_texture);

	memset(&blur_raster.push_constant, 0, sizeof(BlurRasterPushConstant));

	BlurRasterMode blur_mode = p_first_pass ? BLUR_MODE_GAUSSIAN_GLOW_GATHER : BLUR_MODE_GAUSSIAN_GLOW_DOWNSAMPLE;

	blur_raster.push_constant.source_pixel_size[0] = 1.0 / float(p_size.x);
	blur_raster.push_constant.source_pixel_size[1] = 1.0 / float(p_size.y);

	blur_raster.push_constant.glow_strength = p_strength;
	blur_raster.push_constant.glow_bloom = p_bloom;
	blur_raster.push_constant.glow_hdr_threshold = p_hdr_bleed_threshold;
	blur_raster.push_constant.glow_hdr_scale = p_hdr_bleed_scale;
	blur_raster.push_constant.glow_exposure = p_exposure;
	blur_raster.push_constant.glow_white = 0; //actually unused
	blur_raster.push_constant.glow_luminance_cap = p_luminance_cap;

	blur_raster.push_constant.luminance_multiplier = p_luminance_multiplier;

	// setup our uniforms
	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ blur_raster.glow_sampler, p_source_rd_texture }));

	RID shader = blur_raster.shader.version_get_shader(blur_raster.shader_version, blur_mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(dest_framebuffer);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur_raster.pipelines[blur_mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur_raster.push_constant, sizeof(BlurRasterPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::gaussian_glow_upsample_raster(RID p_source_rd_texture, RID p_dest_texture, RID p_blend_texture, float p_luminance_multiplier, const Size2i &p_source_size, const Size2i &p_dest_size, float p_level, float p_base_strength, bool p_use_debanding) {
	ERR_FAIL_COND_MSG(!raster_effects.has_flag(RASTER_EFFECT_GAUSSIAN_BLUR), "Can't use the raster version of the gaussian glow.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	RID dest_framebuffer = FramebufferCacheRD::get_singleton()->get_cache(p_dest_texture);

	memset(&blur_raster.push_constant, 0, sizeof(BlurRasterPushConstant));

	BlurRasterMode blur_mode = BLUR_MODE_GAUSSIAN_GLOW_UPSAMPLE;

	blur_raster.push_constant.source_pixel_size[0] = 1.0 / float(p_source_size.x);
	blur_raster.push_constant.source_pixel_size[1] = 1.0 / float(p_source_size.y);
	blur_raster.push_constant.dest_pixel_size[0] = 1.0 / float(p_dest_size.x);
	blur_raster.push_constant.dest_pixel_size[1] = 1.0 / float(p_dest_size.y);
	blur_raster.push_constant.luminance_multiplier = p_luminance_multiplier;
	blur_raster.push_constant.level = p_level * 0.5;
	blur_raster.push_constant.glow_strength = p_base_strength;

	uint32_t spec_constant = p_use_debanding ? 1 : 0;
	spec_constant |= p_level > 0.01 ? 2 : 0;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_blend_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_blend_texture }));

	RID shader = blur_raster.shader.version_get_shader(blur_raster.shader_version, blur_mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(dest_framebuffer);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur_raster.pipelines[blur_mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(dest_framebuffer), false, 0, spec_constant));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 1, u_blend_rd_texture), 1);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur_raster.push_constant, sizeof(BlurRasterPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::make_mipmap(RID p_source_rd_texture, RID p_dest_texture, const Size2i &p_size) {
	ERR_FAIL_COND_MSG(raster_effects.has_flag(RASTER_EFFECT_COPY), "Can't use the compute version of the make_mipmap shader.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_size.width;
	copy.push_constant.section[3] = p_size.height;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_dest_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_dest_texture);

	CopyMode mode = COPY_MODE_MIPMAP;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode].get_rid());
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_size.width, p_size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::make_mipmap_raster(RID p_source_rd_texture, RID p_dest_texture, const Size2i &p_size) {
	ERR_FAIL_COND_MSG(!raster_effects.has_flag(RASTER_EFFECT_COPY), "Can't use the raster version of mipmap.");

	RID dest_framebuffer = FramebufferCacheRD::get_singleton()->get_cache(p_dest_texture);

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&blur_raster.push_constant, 0, sizeof(BlurRasterPushConstant));

	BlurRasterMode mode = BLUR_MIPMAP;

	blur_raster.push_constant.dest_pixel_size[0] = 1.0 / float(p_size.x);
	blur_raster.push_constant.dest_pixel_size[1] = 1.0 / float(p_size.y);

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));

	RID shader = blur_raster.shader.version_get_shader(blur_raster.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(dest_framebuffer);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur_raster.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur_raster.push_constant, sizeof(BlurRasterPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::set_color(RID p_dest_texture, const Color &p_color, const Rect2i &p_region, bool p_8bit_dst) {
	ERR_FAIL_COND_MSG(raster_effects.has_flag(RASTER_EFFECT_COPY), "Can't use the compute version of the set_color shader.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_region.size.width;
	copy.push_constant.section[3] = p_region.size.height;
	copy.push_constant.target[0] = p_region.position.x;
	copy.push_constant.target[1] = p_region.position.y;
	copy.push_constant.set_color[0] = p_color.r;
	copy.push_constant.set_color[1] = p_color.g;
	copy.push_constant.set_color[2] = p_color.b;
	copy.push_constant.set_color[3] = p_color.a;

	// setup our uniforms
	RD::Uniform u_dest_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_dest_texture);

	CopyMode mode = p_8bit_dst ? COPY_MODE_SET_COLOR_8BIT : COPY_MODE_SET_COLOR;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode].get_rid());
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_region.size.width, p_region.size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::set_color_raster(RID p_dest_texture, const Color &p_color, const Rect2i &p_region) {
	ERR_FAIL_COND_MSG(!raster_effects.has_flag(RASTER_EFFECT_COPY), "Can't use the raster version of the set_color shader.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy_to_fb.push_constant, 0, sizeof(CopyToFbPushConstant));

	copy_to_fb.push_constant.set_color[0] = p_color.r;
	copy_to_fb.push_constant.set_color[1] = p_color.g;
	copy_to_fb.push_constant.set_color[2] = p_color.b;
	copy_to_fb.push_constant.set_color[3] = p_color.a;

	RID dest_framebuffer = FramebufferCacheRD::get_singleton()->get_cache(p_dest_texture);

	CopyToFBMode mode = COPY_TO_FB_SET_COLOR;

	RID shader = copy_to_fb.shader.version_get_shader(copy_to_fb.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(dest_framebuffer, RD::DRAW_DEFAULT_ALL, Vector<Color>(), 1.0f, 0, p_region);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, copy_to_fb.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &copy_to_fb.push_constant, sizeof(CopyToFbPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::copy_cubemap_to_dp(RID p_source_rd_texture, RID p_dst_framebuffer, const Rect2 &p_rect, const Vector2 &p_dst_size, float p_z_near, float p_z_far, bool p_dp_flip) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	Rect2i screen_rect;
	float atlas_width = p_dst_size.width / p_rect.size.width;
	float atlas_height = p_dst_size.height / p_rect.size.height;
	screen_rect.position.x = (int32_t)(Math::round(p_rect.position.x * atlas_width));
	screen_rect.position.y = (int32_t)(Math::round(p_rect.position.y * atlas_height));
	screen_rect.size.width = (int32_t)(Math::round(p_dst_size.width));
	screen_rect.size.height = (int32_t)(Math::round(p_dst_size.height));

	CopyToDPPushConstant push_constant;
	push_constant.z_far = p_z_far;
	push_constant.z_near = p_z_near;
	push_constant.texel_size[0] = 1.0f / p_dst_size.width;
	push_constant.texel_size[1] = 1.0f / p_dst_size.height;
	push_constant.texel_size[0] *= p_dp_flip ? -1.0f : 1.0f; // Encode dp flip as x size sign

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));

	RID shader = cube_to_dp.shader.version_get_shader(cube_to_dp.shader_version, 0);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dst_framebuffer, RD::DRAW_DEFAULT_ALL, Vector<Color>(), 1.0f, 0, screen_rect);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, cube_to_dp.pipeline.get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dst_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(CopyToDPPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::copy_cubemap_to_octmap(RID p_source_rd_texture, RID p_dst_framebuffer, float p_border_size) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);
	RID shader = cube_to_octmap.shader.version_get_shader(cube_to_octmap.shader_version, 0);
	ERR_FAIL_COND(shader.is_null());

	cube_to_octmap.push_constant.border_size = 1.0f - p_border_size * 2.0f;

	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dst_framebuffer);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, cube_to_octmap.pipeline.get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dst_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &cube_to_octmap.push_constant, sizeof(CopyToOctmapPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::octmap_downsample(RID p_source_octmap, RID p_dest_octmap, const Size2i &p_size, bool p_use_filter_quality, float p_border_size) {
	ERR_FAIL_COND_MSG(raster_effects.has_flag(RASTER_EFFECT_OCTMAP), "Can't use compute based octmap downsample.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	octmap_downsampler.push_constant.size = p_size.x;
	octmap_downsampler.push_constant.border_size = 1.0f - p_border_size * 2.0f;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_octmap(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_octmap }));
	RD::Uniform u_dest_octmap(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_dest_octmap }));

	RID shader = octmap_downsampler.compute_shader.version_get_shader(octmap_downsampler.shader_version, 0);
	ERR_FAIL_COND(shader.is_null());

	int pipeline_index = (!p_use_filter_quality || filter.use_high_quality) ? DOWNSAMPLER_MODE_HIGH_QUALITY : DOWNSAMPLER_MODE_LOW_QUALITY;
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, octmap_downsampler.compute_pipelines[pipeline_index].get_rid());
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_octmap), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_dest_octmap), 1);

	int x_groups = Math::division_round_up(p_size.x, 8);
	int y_groups = Math::division_round_up(p_size.y, 8);

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &octmap_downsampler.push_constant, sizeof(OctmapDownsamplerPushConstant));

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);

	RD::get_singleton()->compute_list_end();
}

void CopyEffects::octmap_downsample_raster(RID p_source_octmap, RID p_dest_framebuffer, const Size2i &p_size, bool p_use_filter_quality, float p_border_size) {
	ERR_FAIL_COND_MSG(!raster_effects.has_flag(RASTER_EFFECT_OCTMAP), "Can't use raster based octmap downsample.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	octmap_downsampler.push_constant.size = p_size.x;
	octmap_downsampler.push_constant.border_size = 1.0f - p_border_size * 2.0f;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_cubemap(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_octmap }));

	RID shader = octmap_downsampler.raster_shader.version_get_shader(octmap_downsampler.shader_version, 0);
	ERR_FAIL_COND(shader.is_null());

	int pipeline_index = (!p_use_filter_quality || filter.use_high_quality) ? DOWNSAMPLER_MODE_HIGH_QUALITY : DOWNSAMPLER_MODE_LOW_QUALITY;
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::DRAW_IGNORE_COLOR_ALL);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, octmap_downsampler.raster_pipelines[pipeline_index].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_cubemap), 0);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &octmap_downsampler.push_constant, sizeof(OctmapDownsamplerPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
}

static constexpr int _compute_dispatch_size(bool p_use_array) {
	constexpr int SIZE = 320;
	constexpr int GROUP = 64;
	constexpr int LEVELS = 6; // One less than Sky::REAL_TIME_ROUGHNESS_LAYERS.
	int size = 0;
	if (p_use_array) {
		size = SIZE * SIZE * LEVELS;
	} else {
		int dim = SIZE;
		for (int i = 0; i < LEVELS && dim >= 2; i++) {
			size += dim * dim;
			dim >>= 1;
		}
	}

	return (size + GROUP - 1) / GROUP;
}

void CopyEffects::octmap_filter(RID p_source_octmap, const Vector<RID> &p_dest_octmap, bool p_use_array, float p_border_size) {
	ERR_FAIL_COND_MSG(raster_effects.has_flag(RASTER_EFFECT_OCTMAP), "Can't use compute based octmap filter.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	OctmapFilterPushConstant push_constant;
	push_constant.border_size[0] = p_border_size;
	push_constant.border_size[1] = 1.0f - p_border_size * 2.0f;
	push_constant.size = 320;

	Vector<RD::Uniform> uniforms;
	for (int i = 0; i < p_dest_octmap.size(); i++) {
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
		u.binding = i;
		u.append_id(p_dest_octmap[i]);
		uniforms.push_back(u);
	}
	if (RD::get_singleton()->uniform_set_is_valid(filter.image_uniform_set)) {
		RD::get_singleton()->free_rid(filter.image_uniform_set);
	}
	filter.image_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, filter.compute_shader.version_get_shader(filter.shader_version, 0), 2);

	// setup our uniforms
	RID default_mipmap_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_octmap(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_mipmap_sampler, p_source_octmap }));

	int mode = p_use_array ? FILTER_MODE_HIGH_QUALITY_ARRAY : FILTER_MODE_HIGH_QUALITY;
	mode = filter.use_high_quality ? mode : mode + 1;

	RID shader = filter.compute_shader.version_get_shader(filter.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, filter.compute_pipelines[mode].get_rid());
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_octmap), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, filter.uniform_set, 1);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, filter.image_uniform_set, 2);

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(OctmapFilterPushConstant));

	RD::get_singleton()->compute_list_dispatch(compute_list, _compute_dispatch_size(p_use_array), 1, 1);

	RD::get_singleton()->compute_list_end();
}

void CopyEffects::octmap_filter_raster(RID p_source_octmap, RID p_dest_framebuffer, uint32_t p_mip_level, float p_border_size) {
	ERR_FAIL_COND_MSG(!raster_effects.has_flag(RASTER_EFFECT_OCTMAP), "Can't use raster based octmap filter.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	OctmapFilterRasterPushConstant push_constant;
	push_constant.border_size[0] = p_border_size;
	push_constant.border_size[1] = 1.0f - p_border_size * 2.0f;
	push_constant.mip_level = p_mip_level;

	// setup our uniforms
	RID default_mipmap_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_octmap(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_mipmap_sampler, p_source_octmap }));

	OctmapFilterMode mode = filter.use_high_quality ? FILTER_MODE_HIGH_QUALITY : FILTER_MODE_LOW_QUALITY;

	RID shader = filter.raster_shader.version_get_shader(filter.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::DRAW_IGNORE_COLOR_ALL);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, filter.raster_pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_octmap), 0);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, filter.uniform_set, 1);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(OctmapFilterRasterPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::octmap_roughness(RID p_source_rd_texture, RID p_dest_texture, uint32_t p_sample_count, float p_roughness, uint32_t p_source_size, uint32_t p_dest_size, float p_border_size) {
	ERR_FAIL_COND_MSG(raster_effects.has_flag(RASTER_EFFECT_OCTMAP), "Can't use compute based octmap roughness.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&roughness.push_constant, 0, sizeof(OctmapRoughnessPushConstant));

	// Remap to perceptual-roughness^2 to create more detail in lower mips and match the mapping of octmap_filter.
	roughness.push_constant.roughness = p_roughness * p_roughness;
	roughness.push_constant.sample_count = p_sample_count;
	roughness.push_constant.source_size = p_source_size;
	roughness.push_constant.dest_size = p_dest_size;
	roughness.push_constant.use_direct_write = p_roughness == 0.0;
	roughness.push_constant.border_size[0] = p_border_size;
	roughness.push_constant.border_size[1] = 1.0f - p_border_size * 2.0;

	// setup our uniforms
	RID default_mipmap_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_mipmap_sampler, p_source_rd_texture }));
	RD::Uniform u_dest_texture(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_dest_texture }));

	RID shader = roughness.compute_shader.version_get_shader(roughness.shader_version, 0);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, roughness.compute_pipeline.get_rid());

	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_dest_texture), 1);

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &roughness.push_constant, sizeof(OctmapRoughnessPushConstant));

	int x_groups = (p_dest_size + 7) / 8;
	int y_groups = x_groups;

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);

	RD::get_singleton()->compute_list_end();
}

void CopyEffects::octmap_roughness_raster(RID p_source_rd_texture, RID p_dest_framebuffer, uint32_t p_sample_count, float p_roughness, uint32_t p_source_size, uint32_t p_dest_size, float p_border_size) {
	ERR_FAIL_COND_MSG(!raster_effects.has_flag(RASTER_EFFECT_OCTMAP), "Can't use raster based octmap roughness.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&roughness.push_constant, 0, sizeof(OctmapRoughnessPushConstant));

	roughness.push_constant.roughness = p_roughness * p_roughness; // Shader expects roughness, not perceptual roughness, so multiply before passing in.
	roughness.push_constant.sample_count = p_sample_count;
	roughness.push_constant.source_size = p_source_size;
	roughness.push_constant.dest_size = p_dest_size;
	roughness.push_constant.use_direct_write = p_roughness == 0.0;
	roughness.push_constant.border_size[0] = p_border_size;
	roughness.push_constant.border_size[1] = 1.0f - p_border_size * 2.0;

	// Setup our uniforms.
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));

	RID shader = roughness.raster_shader.version_get_shader(roughness.shader_version, 0);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::DRAW_IGNORE_COLOR_ALL);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, roughness.raster_pipeline.get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &roughness.push_constant, sizeof(OctmapRoughnessPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::merge_specular(RID p_dest_framebuffer, RID p_specular, RID p_base, RID p_reflection, uint32_t p_view_count) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::get_singleton()->draw_command_begin_label("Merge Specular");

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer);

	int mode;
	if (p_reflection.is_valid()) {
		if (p_base.is_valid()) {
			mode = SPECULAR_MERGE_SSR;
		} else {
			mode = SPECULAR_MERGE_ADDITIVE_SSR;
		}
	} else {
		if (p_base.is_valid()) {
			mode = SPECULAR_MERGE_ADD;
		} else {
			mode = SPECULAR_MERGE_ADDITIVE_ADD;
		}
	}

	if (p_view_count > 1) {
		mode += SPECULAR_MERGE_ADD_MULTIVIEW;
	}

	RID shader = specular_merge.shader.version_get_shader(specular_merge.shader_version, mode);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, specular_merge.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));

	if (p_base.is_valid()) {
		RD::Uniform u_base(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_base }));
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 2, u_base), 2);
	}

	RD::Uniform u_specular(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_specular }));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_specular), 0);

	if (p_reflection.is_valid()) {
		RD::Uniform u_reflection(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_reflection }));
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 1, u_reflection), 1);
	}

	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();

	RD::get_singleton()->draw_command_end_label();
}
