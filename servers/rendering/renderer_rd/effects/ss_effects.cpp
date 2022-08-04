/*************************************************************************/
/*  ss_effects.cpp                                                       */
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

#include "ss_effects.h"

#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

SSEffects *SSEffects::singleton = nullptr;

static _FORCE_INLINE_ void store_camera(const Projection &p_mtx, float *p_array) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			p_array[i * 4 + j] = p_mtx.matrix[i][j];
		}
	}
}

SSEffects::SSEffects() {
	singleton = this;

	{
		// Initialize depth buffer for screen space effects
		Vector<String> downsampler_modes;
		downsampler_modes.push_back("\n");
		downsampler_modes.push_back("\n#define USE_HALF_SIZE\n");
		downsampler_modes.push_back("\n#define GENERATE_MIPS\n");
		downsampler_modes.push_back("\n#define GENERATE_MIPS\n#define USE_HALF_SIZE\n");
		downsampler_modes.push_back("\n#define USE_HALF_BUFFERS\n");
		downsampler_modes.push_back("\n#define USE_HALF_BUFFERS\n#define USE_HALF_SIZE\n");
		downsampler_modes.push_back("\n#define GENERATE_MIPS\n#define GENERATE_FULL_MIPS");

		ss_effects.downsample_shader.initialize(downsampler_modes);

		ss_effects.downsample_shader_version = ss_effects.downsample_shader.version_create();

		for (int i = 0; i < SS_EFFECTS_MAX; i++) {
			ss_effects.pipelines[i] = RD::get_singleton()->compute_pipeline_create(ss_effects.downsample_shader.version_get_shader(ss_effects.downsample_shader_version, i));
		}

		ss_effects.gather_constants_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(SSEffectsGatherConstants));
		SSEffectsGatherConstants gather_constants;

		const int sub_pass_count = 5;
		for (int pass = 0; pass < 4; pass++) {
			for (int subPass = 0; subPass < sub_pass_count; subPass++) {
				int a = pass;
				int b = subPass;

				int spmap[5]{ 0, 1, 4, 3, 2 };
				b = spmap[subPass];

				float ca, sa;
				float angle0 = (float(a) + float(b) / float(sub_pass_count)) * Math_PI * 0.5f;

				ca = Math::cos(angle0);
				sa = Math::sin(angle0);

				float scale = 1.0f + (a - 1.5f + (b - (sub_pass_count - 1.0f) * 0.5f) / float(sub_pass_count)) * 0.07f;

				gather_constants.rotation_matrices[pass * 20 + subPass * 4 + 0] = scale * ca;
				gather_constants.rotation_matrices[pass * 20 + subPass * 4 + 1] = scale * -sa;
				gather_constants.rotation_matrices[pass * 20 + subPass * 4 + 2] = -scale * sa;
				gather_constants.rotation_matrices[pass * 20 + subPass * 4 + 3] = -scale * ca;
			}
		}

		RD::get_singleton()->buffer_update(ss_effects.gather_constants_buffer, 0, sizeof(SSEffectsGatherConstants), &gather_constants);
	}

	// Initialize Screen Space Indirect Lighting (SSIL)

	{
		Vector<String> ssil_modes;
		ssil_modes.push_back("\n");
		ssil_modes.push_back("\n#define SSIL_BASE\n");
		ssil_modes.push_back("\n#define ADAPTIVE\n");

		ssil.gather_shader.initialize(ssil_modes);

		ssil.gather_shader_version = ssil.gather_shader.version_create();

		for (int i = SSIL_GATHER; i <= SSIL_GATHER_ADAPTIVE; i++) {
			ssil.pipelines[i] = RD::get_singleton()->compute_pipeline_create(ssil.gather_shader.version_get_shader(ssil.gather_shader_version, i));
		}
		ssil.projection_uniform_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(SSILProjectionUniforms));
	}

	{
		Vector<String> ssil_modes;
		ssil_modes.push_back("\n#define GENERATE_MAP\n");
		ssil_modes.push_back("\n#define PROCESS_MAPA\n");
		ssil_modes.push_back("\n#define PROCESS_MAPB\n");

		ssil.importance_map_shader.initialize(ssil_modes);

		ssil.importance_map_shader_version = ssil.importance_map_shader.version_create();

		for (int i = SSIL_GENERATE_IMPORTANCE_MAP; i <= SSIL_PROCESS_IMPORTANCE_MAPB; i++) {
			ssil.pipelines[i] = RD::get_singleton()->compute_pipeline_create(ssil.importance_map_shader.version_get_shader(ssil.importance_map_shader_version, i - SSIL_GENERATE_IMPORTANCE_MAP));
		}
		ssil.importance_map_load_counter = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t));
		int zero[1] = { 0 };
		RD::get_singleton()->buffer_update(ssil.importance_map_load_counter, 0, sizeof(uint32_t), &zero);
		RD::get_singleton()->set_resource_name(ssil.importance_map_load_counter, "Importance Map Load Counter");

		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 0;
			u.append_id(ssil.importance_map_load_counter);
			uniforms.push_back(u);
		}
		ssil.counter_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, ssil.importance_map_shader.version_get_shader(ssil.importance_map_shader_version, 2), 2);
		RD::get_singleton()->set_resource_name(ssil.counter_uniform_set, "Load Counter Uniform Set");
	}

	{
		Vector<String> ssil_modes;
		ssil_modes.push_back("\n#define MODE_NON_SMART\n");
		ssil_modes.push_back("\n#define MODE_SMART\n");
		ssil_modes.push_back("\n#define MODE_WIDE\n");

		ssil.blur_shader.initialize(ssil_modes);

		ssil.blur_shader_version = ssil.blur_shader.version_create();
		for (int i = SSIL_BLUR_PASS; i <= SSIL_BLUR_PASS_WIDE; i++) {
			ssil.pipelines[i] = RD::get_singleton()->compute_pipeline_create(ssil.blur_shader.version_get_shader(ssil.blur_shader_version, i - SSIL_BLUR_PASS));
		}
	}

	{
		Vector<String> ssil_modes;
		ssil_modes.push_back("\n#define MODE_NON_SMART\n");
		ssil_modes.push_back("\n#define MODE_SMART\n");
		ssil_modes.push_back("\n#define MODE_HALF\n");

		ssil.interleave_shader.initialize(ssil_modes);

		ssil.interleave_shader_version = ssil.interleave_shader.version_create();
		for (int i = SSIL_INTERLEAVE; i <= SSIL_INTERLEAVE_HALF; i++) {
			ssil.pipelines[i] = RD::get_singleton()->compute_pipeline_create(ssil.interleave_shader.version_get_shader(ssil.interleave_shader_version, i - SSIL_INTERLEAVE));
		}
	}

	{
		// Initialize Screen Space Ambient Occlusion (SSAO)

		RD::SamplerState sampler;
		sampler.mag_filter = RD::SAMPLER_FILTER_NEAREST;
		sampler.min_filter = RD::SAMPLER_FILTER_NEAREST;
		sampler.mip_filter = RD::SAMPLER_FILTER_NEAREST;
		sampler.repeat_u = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
		sampler.repeat_v = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
		sampler.repeat_w = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
		sampler.max_lod = 4;

		uint32_t pipeline = 0;
		{
			Vector<String> ssao_modes;

			ssao_modes.push_back("\n");
			ssao_modes.push_back("\n#define SSAO_BASE\n");
			ssao_modes.push_back("\n#define ADAPTIVE\n");

			ssao.gather_shader.initialize(ssao_modes);

			ssao.gather_shader_version = ssao.gather_shader.version_create();

			for (int i = 0; i <= SSAO_GATHER_ADAPTIVE; i++) {
				ssao.pipelines[pipeline] = RD::get_singleton()->compute_pipeline_create(ssao.gather_shader.version_get_shader(ssao.gather_shader_version, i));
				pipeline++;
			}
		}

		{
			Vector<String> ssao_modes;
			ssao_modes.push_back("\n#define GENERATE_MAP\n");
			ssao_modes.push_back("\n#define PROCESS_MAPA\n");
			ssao_modes.push_back("\n#define PROCESS_MAPB\n");

			ssao.importance_map_shader.initialize(ssao_modes);

			ssao.importance_map_shader_version = ssao.importance_map_shader.version_create();

			for (int i = SSAO_GENERATE_IMPORTANCE_MAP; i <= SSAO_PROCESS_IMPORTANCE_MAPB; i++) {
				ssao.pipelines[pipeline] = RD::get_singleton()->compute_pipeline_create(ssao.importance_map_shader.version_get_shader(ssao.importance_map_shader_version, i - SSAO_GENERATE_IMPORTANCE_MAP));

				pipeline++;
			}

			ssao.importance_map_load_counter = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t));
			int zero[1] = { 0 };
			RD::get_singleton()->buffer_update(ssao.importance_map_load_counter, 0, sizeof(uint32_t), &zero);
			RD::get_singleton()->set_resource_name(ssao.importance_map_load_counter, "Importance Map Load Counter");

			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 0;
				u.append_id(ssao.importance_map_load_counter);
				uniforms.push_back(u);
			}
			ssao.counter_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, ssao.importance_map_shader.version_get_shader(ssao.importance_map_shader_version, 2), 2);
			RD::get_singleton()->set_resource_name(ssao.counter_uniform_set, "Load Counter Uniform Set");
		}

		{
			Vector<String> ssao_modes;
			ssao_modes.push_back("\n#define MODE_NON_SMART\n");
			ssao_modes.push_back("\n#define MODE_SMART\n");
			ssao_modes.push_back("\n#define MODE_WIDE\n");

			ssao.blur_shader.initialize(ssao_modes);

			ssao.blur_shader_version = ssao.blur_shader.version_create();

			for (int i = SSAO_BLUR_PASS; i <= SSAO_BLUR_PASS_WIDE; i++) {
				ssao.pipelines[pipeline] = RD::get_singleton()->compute_pipeline_create(ssao.blur_shader.version_get_shader(ssao.blur_shader_version, i - SSAO_BLUR_PASS));

				pipeline++;
			}
		}

		{
			Vector<String> ssao_modes;
			ssao_modes.push_back("\n#define MODE_NON_SMART\n");
			ssao_modes.push_back("\n#define MODE_SMART\n");
			ssao_modes.push_back("\n#define MODE_HALF\n");

			ssao.interleave_shader.initialize(ssao_modes);

			ssao.interleave_shader_version = ssao.interleave_shader.version_create();
			for (int i = SSAO_INTERLEAVE; i <= SSAO_INTERLEAVE_HALF; i++) {
				ssao.pipelines[pipeline] = RD::get_singleton()->compute_pipeline_create(ssao.interleave_shader.version_get_shader(ssao.interleave_shader_version, i - SSAO_INTERLEAVE));
				RD::get_singleton()->set_resource_name(ssao.pipelines[pipeline], "Interleave Pipeline " + itos(i));
				pipeline++;
			}
		}

		ERR_FAIL_COND(pipeline != SSAO_MAX);

		ss_effects.mirror_sampler = RD::get_singleton()->sampler_create(sampler);
	}

	{
		// Screen Space Reflections

		Vector<RD::PipelineSpecializationConstant> specialization_constants;

		{
			RD::PipelineSpecializationConstant sc;
			sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
			sc.constant_id = 0; // SSR_USE_FULL_PROJECTION_MATRIX
			sc.bool_value = false;
			specialization_constants.push_back(sc);
		}

		{
			Vector<String> ssr_scale_modes;
			ssr_scale_modes.push_back("\n");

			ssr_scale.shader.initialize(ssr_scale_modes);
			ssr_scale.shader_version = ssr_scale.shader.version_create();

			for (int v = 0; v < SSR_VARIATIONS; v++) {
				specialization_constants.ptrw()[0].bool_value = (v & SSR_MULTIVIEW) ? true : false;
				ssr_scale.pipelines[v] = RD::get_singleton()->compute_pipeline_create(ssr_scale.shader.version_get_shader(ssr_scale.shader_version, 0), specialization_constants);
			}
		}

		{
			Vector<String> ssr_modes;
			ssr_modes.push_back("\n"); // SCREEN_SPACE_REFLECTION_NORMAL
			ssr_modes.push_back("\n#define MODE_ROUGH\n"); // SCREEN_SPACE_REFLECTION_ROUGH

			ssr.shader.initialize(ssr_modes);
			ssr.shader_version = ssr.shader.version_create();

			for (int v = 0; v < SSR_VARIATIONS; v++) {
				specialization_constants.ptrw()[0].bool_value = (v & SSR_MULTIVIEW) ? true : false;
				for (int i = 0; i < SCREEN_SPACE_REFLECTION_MAX; i++) {
					ssr.pipelines[v][i] = RD::get_singleton()->compute_pipeline_create(ssr.shader.version_get_shader(ssr.shader_version, i), specialization_constants);
				}
			}
		}

		{
			Vector<String> ssr_filter_modes;
			ssr_filter_modes.push_back("\n"); // SCREEN_SPACE_REFLECTION_FILTER_HORIZONTAL
			ssr_filter_modes.push_back("\n#define VERTICAL_PASS\n"); // SCREEN_SPACE_REFLECTION_FILTER_VERTICAL

			ssr_filter.shader.initialize(ssr_filter_modes);
			ssr_filter.shader_version = ssr_filter.shader.version_create();

			for (int v = 0; v < SSR_VARIATIONS; v++) {
				specialization_constants.ptrw()[0].bool_value = (v & SSR_MULTIVIEW) ? true : false;
				for (int i = 0; i < SCREEN_SPACE_REFLECTION_FILTER_MAX; i++) {
					ssr_filter.pipelines[v][i] = RD::get_singleton()->compute_pipeline_create(ssr_filter.shader.version_get_shader(ssr_filter.shader_version, i), specialization_constants);
				}
			}
		}
	}

	// Subsurface scattering
	{
		Vector<String> sss_modes;
		sss_modes.push_back("\n#define USE_11_SAMPLES\n");
		sss_modes.push_back("\n#define USE_17_SAMPLES\n");
		sss_modes.push_back("\n#define USE_25_SAMPLES\n");

		sss.shader.initialize(sss_modes);

		sss.shader_version = sss.shader.version_create();

		for (int i = 0; i < sss_modes.size(); i++) {
			sss.pipelines[i] = RD::get_singleton()->compute_pipeline_create(sss.shader.version_get_shader(sss.shader_version, i));
		}
	}
}

SSEffects::~SSEffects() {
	{
		// Cleanup SS Reflections
		ssr.shader.version_free(ssr.shader_version);
		ssr_filter.shader.version_free(ssr_filter.shader_version);
		ssr_scale.shader.version_free(ssr_scale.shader_version);

		if (ssr.ubo.is_valid()) {
			RD::get_singleton()->free(ssr.ubo);
		}
	}

	{
		// Cleanup SS downsampler
		ss_effects.downsample_shader.version_free(ss_effects.downsample_shader_version);

		RD::get_singleton()->free(ss_effects.mirror_sampler);
		RD::get_singleton()->free(ss_effects.gather_constants_buffer);
	}

	{
		// Cleanup SSIL
		ssil.blur_shader.version_free(ssil.blur_shader_version);
		ssil.gather_shader.version_free(ssil.gather_shader_version);
		ssil.interleave_shader.version_free(ssil.interleave_shader_version);
		ssil.importance_map_shader.version_free(ssil.importance_map_shader_version);

		RD::get_singleton()->free(ssil.importance_map_load_counter);
		RD::get_singleton()->free(ssil.projection_uniform_buffer);
	}

	{
		// Cleanup SSAO
		ssao.blur_shader.version_free(ssao.blur_shader_version);
		ssao.gather_shader.version_free(ssao.gather_shader_version);
		ssao.interleave_shader.version_free(ssao.interleave_shader_version);
		ssao.importance_map_shader.version_free(ssao.importance_map_shader_version);

		RD::get_singleton()->free(ssao.importance_map_load_counter);
	}

	{
		// Cleanup Subsurface scattering
		sss.shader.version_free(sss.shader_version);
	}

	singleton = nullptr;
}

/* SS Downsampler */

void SSEffects::downsample_depth(RID p_depth_buffer, const Vector<RID> &p_depth_mipmaps, RS::EnvironmentSSAOQuality p_ssao_quality, RS::EnvironmentSSILQuality p_ssil_quality, bool p_invalidate_uniform_set, bool p_ssao_half_size, bool p_ssil_half_size, Size2i p_full_screen_size, const Projection &p_projection) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	// Downsample and deinterleave the depth buffer for SSAO and SSIL
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	int downsample_mode = SS_EFFECTS_DOWNSAMPLE;
	bool use_mips = p_ssao_quality > RS::ENV_SSAO_QUALITY_MEDIUM || p_ssil_quality > RS::ENV_SSIL_QUALITY_MEDIUM;

	if (p_ssao_quality == RS::ENV_SSAO_QUALITY_VERY_LOW && p_ssil_quality == RS::ENV_SSIL_QUALITY_VERY_LOW) {
		downsample_mode = SS_EFFECTS_DOWNSAMPLE_HALF;
	} else if (use_mips) {
		downsample_mode = SS_EFFECTS_DOWNSAMPLE_MIPMAP;
	}

	bool use_half_size = false;
	bool use_full_mips = false;

	if (p_ssao_half_size && p_ssil_half_size) {
		downsample_mode++;
		use_half_size = true;
	} else if (p_ssao_half_size != p_ssil_half_size) {
		if (use_mips) {
			downsample_mode = SS_EFFECTS_DOWNSAMPLE_FULL_MIPS;
			use_full_mips = true;
		} else {
			// Only need the first two mipmaps, but the cost to generate the next two is trivial
			// TODO investigate the benefit of a shader version to generate only 2 mips
			downsample_mode = SS_EFFECTS_DOWNSAMPLE_MIPMAP;
			use_mips = true;
		}
	}

	int depth_index = use_half_size ? 1 : 0;

	RD::get_singleton()->draw_command_begin_label("Downsample Depth");
	if (p_invalidate_uniform_set || use_full_mips != ss_effects.used_full_mips_last_frame || use_half_size != ss_effects.used_half_size_last_frame || use_mips != ss_effects.used_mips_last_frame) {
		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 0;
			u.append_id(p_depth_mipmaps[depth_index + 1]);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 1;
			u.append_id(p_depth_mipmaps[depth_index + 2]);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 2;
			u.append_id(p_depth_mipmaps[depth_index + 3]);
			uniforms.push_back(u);
		}
		if (use_full_mips) {
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 3;
			u.append_id(p_depth_mipmaps[4]);
			uniforms.push_back(u);
		}
		ss_effects.downsample_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, ss_effects.downsample_shader.version_get_shader(ss_effects.downsample_shader_version, use_full_mips ? 6 : 2), 2);
	}

	float depth_linearize_mul = -p_projection.matrix[3][2];
	float depth_linearize_add = p_projection.matrix[2][2];
	if (depth_linearize_mul * depth_linearize_add < 0) {
		depth_linearize_add = -depth_linearize_add;
	}

	ss_effects.downsample_push_constant.orthogonal = p_projection.is_orthogonal();
	ss_effects.downsample_push_constant.z_near = depth_linearize_mul;
	ss_effects.downsample_push_constant.z_far = depth_linearize_add;
	if (ss_effects.downsample_push_constant.orthogonal) {
		ss_effects.downsample_push_constant.z_near = p_projection.get_z_near();
		ss_effects.downsample_push_constant.z_far = p_projection.get_z_far();
	}
	ss_effects.downsample_push_constant.pixel_size[0] = 1.0 / p_full_screen_size.x;
	ss_effects.downsample_push_constant.pixel_size[1] = 1.0 / p_full_screen_size.y;
	ss_effects.downsample_push_constant.radius_sq = 1.0;

	RID shader = ss_effects.downsample_shader.version_get_shader(ss_effects.downsample_shader_version, downsample_mode);
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_depth_buffer(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_depth_buffer }));
	RD::Uniform u_depth_mipmaps(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_depth_mipmaps[depth_index + 0] }));

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ss_effects.pipelines[downsample_mode]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_depth_buffer), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_depth_mipmaps), 1);
	if (use_mips) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, ss_effects.downsample_uniform_set, 2);
	}
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &ss_effects.downsample_push_constant, sizeof(SSEffectsDownsamplePushConstant));

	Size2i size(MAX(1, p_full_screen_size.x >> (use_half_size ? 2 : 1)), MAX(1, p_full_screen_size.y >> (use_half_size ? 2 : 1)));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, size.x, size.y, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);
	RD::get_singleton()->draw_command_end_label();

	RD::get_singleton()->compute_list_end(RD::BARRIER_MASK_COMPUTE);

	ss_effects.used_full_mips_last_frame = use_full_mips;
	ss_effects.used_half_size_last_frame = use_half_size;
}

/* SSIL */

void SSEffects::gather_ssil(RD::ComputeListID p_compute_list, const Vector<RID> p_ssil_slices, const Vector<RID> p_edges_slices, const SSILSettings &p_settings, bool p_adaptive_base_pass, RID p_gather_uniform_set, RID p_importance_map_uniform_set, RID p_projection_uniform_set) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);

	RD::get_singleton()->compute_list_bind_uniform_set(p_compute_list, p_gather_uniform_set, 0);
	if ((p_settings.quality == RS::ENV_SSIL_QUALITY_ULTRA) && !p_adaptive_base_pass) {
		RD::get_singleton()->compute_list_bind_uniform_set(p_compute_list, p_importance_map_uniform_set, 1);
	}
	RD::get_singleton()->compute_list_bind_uniform_set(p_compute_list, p_projection_uniform_set, 3);

	RID shader = ssil.gather_shader.version_get_shader(ssil.gather_shader_version, 0);

	for (int i = 0; i < 4; i++) {
		if ((p_settings.quality == RS::ENV_SSIL_QUALITY_VERY_LOW) && ((i == 1) || (i == 2))) {
			continue;
		}

		RD::Uniform u_ssil_slice(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssil_slices[i] }));
		RD::Uniform u_edges_slice(RD::UNIFORM_TYPE_IMAGE, 1, Vector<RID>({ p_edges_slices[i] }));

		ssil.gather_push_constant.pass_coord_offset[0] = i % 2;
		ssil.gather_push_constant.pass_coord_offset[1] = i / 2;
		ssil.gather_push_constant.pass_uv_offset[0] = ((i % 2) - 0.0) / p_settings.full_screen_size.x;
		ssil.gather_push_constant.pass_uv_offset[1] = ((i / 2) - 0.0) / p_settings.full_screen_size.y;
		ssil.gather_push_constant.pass = i;
		RD::get_singleton()->compute_list_bind_uniform_set(p_compute_list, uniform_set_cache->get_cache(shader, 2, u_ssil_slice, u_edges_slice), 2);
		RD::get_singleton()->compute_list_set_push_constant(p_compute_list, &ssil.gather_push_constant, sizeof(SSILGatherPushConstant));

		Size2i size = Size2i(p_settings.full_screen_size.x >> (p_settings.half_size ? 2 : 1), p_settings.full_screen_size.y >> (p_settings.half_size ? 2 : 1));

		RD::get_singleton()->compute_list_dispatch_threads(p_compute_list, size.x, size.y, 1);
	}
	RD::get_singleton()->compute_list_add_barrier(p_compute_list);
}

void SSEffects::ssil_allocate_buffers(SSILRenderBuffers &p_ssil_buffers, const SSILSettings &p_settings, RID p_linear_depth) {
	if (p_ssil_buffers.half_size != p_settings.half_size) {
		ssil_free(p_ssil_buffers);
	}

	if (p_settings.half_size) {
		p_ssil_buffers.buffer_width = (p_settings.full_screen_size.x + 3) / 4;
		p_ssil_buffers.buffer_height = (p_settings.full_screen_size.y + 3) / 4;
		p_ssil_buffers.half_buffer_width = (p_settings.full_screen_size.x + 7) / 8;
		p_ssil_buffers.half_buffer_height = (p_settings.full_screen_size.y + 7) / 8;
	} else {
		p_ssil_buffers.buffer_width = (p_settings.full_screen_size.x + 1) / 2;
		p_ssil_buffers.buffer_height = (p_settings.full_screen_size.y + 1) / 2;
		p_ssil_buffers.half_buffer_width = (p_settings.full_screen_size.x + 3) / 4;
		p_ssil_buffers.half_buffer_height = (p_settings.full_screen_size.y + 3) / 4;
	}

	if (p_ssil_buffers.ssil_final.is_null()) {
		{
			p_ssil_buffers.depth_texture_view = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_linear_depth, 0, p_settings.half_size ? 1 : 0, 4, RD::TEXTURE_SLICE_2D_ARRAY);
		}
		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			tf.width = p_settings.full_screen_size.x;
			tf.height = p_settings.full_screen_size.y;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
			p_ssil_buffers.ssil_final = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(p_ssil_buffers.ssil_final, "SSIL texture");
			RD::get_singleton()->texture_clear(p_ssil_buffers.ssil_final, Color(0, 0, 0, 0), 0, 1, 0, 1);
			if (p_ssil_buffers.last_frame.is_null()) {
				tf.mipmaps = 6;
				p_ssil_buffers.last_frame = RD::get_singleton()->texture_create(tf, RD::TextureView());
				RD::get_singleton()->set_resource_name(p_ssil_buffers.last_frame, "Last Frame Radiance");
				RD::get_singleton()->texture_clear(p_ssil_buffers.last_frame, Color(0, 0, 0, 0), 0, tf.mipmaps, 0, 1);
				for (uint32_t i = 0; i < 6; i++) {
					RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_ssil_buffers.last_frame, 0, i);
					p_ssil_buffers.last_frame_slices.push_back(slice);
					RD::get_singleton()->set_resource_name(slice, "Last Frame Radiance Mip " + itos(i) + " ");
				}
			}
		}
		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
			tf.width = p_ssil_buffers.buffer_width;
			tf.height = p_ssil_buffers.buffer_height;
			tf.array_layers = 4;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			p_ssil_buffers.deinterleaved = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(p_ssil_buffers.deinterleaved, "SSIL deinterleaved buffer");
			for (uint32_t i = 0; i < 4; i++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_ssil_buffers.deinterleaved, i, 0);
				p_ssil_buffers.deinterleaved_slices.push_back(slice);
				RD::get_singleton()->set_resource_name(slice, "SSIL deinterleaved buffer array " + itos(i) + " ");
			}
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
			tf.width = p_ssil_buffers.buffer_width;
			tf.height = p_ssil_buffers.buffer_height;
			tf.array_layers = 4;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			p_ssil_buffers.pong = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(p_ssil_buffers.pong, "SSIL deinterleaved pong buffer");
			for (uint32_t i = 0; i < 4; i++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_ssil_buffers.pong, i, 0);
				p_ssil_buffers.pong_slices.push_back(slice);
				RD::get_singleton()->set_resource_name(slice, "SSIL deinterleaved buffer pong array " + itos(i) + " ");
			}
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
			tf.width = p_ssil_buffers.buffer_width;
			tf.height = p_ssil_buffers.buffer_height;
			tf.array_layers = 4;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			p_ssil_buffers.edges = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(p_ssil_buffers.edges, "SSIL edges buffer");
			for (uint32_t i = 0; i < 4; i++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_ssil_buffers.edges, i, 0);
				p_ssil_buffers.edges_slices.push_back(slice);
				RD::get_singleton()->set_resource_name(slice, "SSIL edges buffer slice " + itos(i) + " ");
			}
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = p_ssil_buffers.half_buffer_width;
			tf.height = p_ssil_buffers.half_buffer_height;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			p_ssil_buffers.importance_map[0] = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(p_ssil_buffers.importance_map[0], "SSIL Importance Map");
			p_ssil_buffers.importance_map[1] = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(p_ssil_buffers.importance_map[1], "SSIL Importance Map Pong");
		}
		p_ssil_buffers.half_size = p_settings.half_size;
	}
}

void SSEffects::screen_space_indirect_lighting(SSILRenderBuffers &p_ssil_buffers, RID p_normal_buffer, const Projection &p_projection, const Projection &p_last_projection, const SSILSettings &p_settings) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	RD::get_singleton()->draw_command_begin_label("Process Screen Space Indirect Lighting");
	//Store projection info before starting the compute list
	SSILProjectionUniforms projection_uniforms;
	store_camera(p_last_projection, projection_uniforms.inv_last_frame_projection_matrix);

	RD::get_singleton()->buffer_update(ssil.projection_uniform_buffer, 0, sizeof(SSILProjectionUniforms), &projection_uniforms);

	memset(&ssil.gather_push_constant, 0, sizeof(SSILGatherPushConstant));

	RID shader = ssil.gather_shader.version_get_shader(ssil.gather_shader_version, 0);
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	RID default_mipmap_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	{
		RD::get_singleton()->draw_command_begin_label("Gather Samples");
		ssil.gather_push_constant.screen_size[0] = p_settings.full_screen_size.x;
		ssil.gather_push_constant.screen_size[1] = p_settings.full_screen_size.y;

		ssil.gather_push_constant.half_screen_pixel_size[0] = 1.0 / p_ssil_buffers.buffer_width;
		ssil.gather_push_constant.half_screen_pixel_size[1] = 1.0 / p_ssil_buffers.buffer_height;
		float tan_half_fov_x = 1.0 / p_projection.matrix[0][0];
		float tan_half_fov_y = 1.0 / p_projection.matrix[1][1];
		ssil.gather_push_constant.NDC_to_view_mul[0] = tan_half_fov_x * 2.0;
		ssil.gather_push_constant.NDC_to_view_mul[1] = tan_half_fov_y * -2.0;
		ssil.gather_push_constant.NDC_to_view_add[0] = tan_half_fov_x * -1.0;
		ssil.gather_push_constant.NDC_to_view_add[1] = tan_half_fov_y;
		ssil.gather_push_constant.z_near = p_projection.get_z_near();
		ssil.gather_push_constant.z_far = p_projection.get_z_far();
		ssil.gather_push_constant.is_orthogonal = p_projection.is_orthogonal();

		ssil.gather_push_constant.half_screen_pixel_size_x025[0] = ssil.gather_push_constant.half_screen_pixel_size[0] * 0.25;
		ssil.gather_push_constant.half_screen_pixel_size_x025[1] = ssil.gather_push_constant.half_screen_pixel_size[1] * 0.25;

		ssil.gather_push_constant.radius = p_settings.radius;
		float radius_near_limit = (p_settings.radius * 1.2f);
		if (p_settings.quality <= RS::ENV_SSIL_QUALITY_LOW) {
			radius_near_limit *= 1.50f;

			if (p_settings.quality == RS::ENV_SSIL_QUALITY_VERY_LOW) {
				ssil.gather_push_constant.radius *= 0.8f;
			}
		}
		radius_near_limit /= tan_half_fov_y;
		ssil.gather_push_constant.intensity = p_settings.intensity * Math_PI;
		ssil.gather_push_constant.fade_out_mul = -1.0 / (p_settings.fadeout_to - p_settings.fadeout_from);
		ssil.gather_push_constant.fade_out_add = p_settings.fadeout_from / (p_settings.fadeout_to - p_settings.fadeout_from) + 1.0;
		ssil.gather_push_constant.inv_radius_near_limit = 1.0f / radius_near_limit;
		ssil.gather_push_constant.neg_inv_radius = -1.0 / ssil.gather_push_constant.radius;
		ssil.gather_push_constant.normal_rejection_amount = p_settings.normal_rejection;

		ssil.gather_push_constant.load_counter_avg_div = 9.0 / float((p_ssil_buffers.half_buffer_width) * (p_ssil_buffers.half_buffer_height) * 255);
		ssil.gather_push_constant.adaptive_sample_limit = p_settings.adaptive_target;

		ssil.gather_push_constant.quality = MAX(0, p_settings.quality - 1);
		ssil.gather_push_constant.size_multiplier = p_settings.half_size ? 2 : 1;

		if (p_ssil_buffers.projection_uniform_set.is_null()) {
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
				u.binding = 0;
				u.append_id(default_mipmap_sampler);
				u.append_id(p_ssil_buffers.last_frame);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
				u.binding = 1;
				u.append_id(ssil.projection_uniform_buffer);
				uniforms.push_back(u);
			}
			p_ssil_buffers.projection_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, ssil.gather_shader.version_get_shader(ssil.gather_shader_version, 0), 3);
		}

		if (p_ssil_buffers.gather_uniform_set.is_null()) {
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
				u.binding = 0;
				u.append_id(default_sampler);
				u.append_id(p_ssil_buffers.depth_texture_view);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 1;
				u.append_id(p_normal_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
				u.binding = 2;
				u.append_id(ss_effects.gather_constants_buffer);
				uniforms.push_back(u);
			}
			p_ssil_buffers.gather_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, ssil.gather_shader.version_get_shader(ssil.gather_shader_version, 0), 0);
		}

		if (p_ssil_buffers.importance_map_uniform_set.is_null()) {
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 0;
				u.append_id(p_ssil_buffers.pong);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
				u.binding = 1;
				u.append_id(default_sampler);
				u.append_id(p_ssil_buffers.importance_map[0]);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 2;
				u.append_id(ssil.importance_map_load_counter);
				uniforms.push_back(u);
			}
			p_ssil_buffers.importance_map_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, ssil.gather_shader.version_get_shader(ssil.gather_shader_version, 2), 1);
		}

		if (p_settings.quality == RS::ENV_SSIL_QUALITY_ULTRA) {
			RD::get_singleton()->draw_command_begin_label("Generate Importance Map");
			ssil.importance_map_push_constant.half_screen_pixel_size[0] = 1.0 / p_ssil_buffers.buffer_width;
			ssil.importance_map_push_constant.half_screen_pixel_size[1] = 1.0 / p_ssil_buffers.buffer_height;
			ssil.importance_map_push_constant.intensity = p_settings.intensity * Math_PI;
			//base pass
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssil.pipelines[SSIL_GATHER_BASE]);
			gather_ssil(compute_list, p_ssil_buffers.pong_slices, p_ssil_buffers.edges_slices, p_settings, true, p_ssil_buffers.gather_uniform_set, p_ssil_buffers.importance_map_uniform_set, p_ssil_buffers.projection_uniform_set);

			//generate importance map
			RD::Uniform u_ssil_pong_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssil_buffers.pong }));
			RD::Uniform u_importance_map(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssil_buffers.importance_map[0] }));

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssil.pipelines[SSIL_GENERATE_IMPORTANCE_MAP]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_ssil_pong_with_sampler), 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_importance_map), 1);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssil.importance_map_push_constant, sizeof(SSILImportanceMapPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_ssil_buffers.half_buffer_width, p_ssil_buffers.half_buffer_height, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);

			// process Importance Map A
			RD::Uniform u_importance_map_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssil_buffers.importance_map[0] }));
			RD::Uniform u_importance_map_pong(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssil_buffers.importance_map[1] }));

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssil.pipelines[SSIL_PROCESS_IMPORTANCE_MAPA]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_importance_map_with_sampler), 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_importance_map_pong), 1);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssil.importance_map_push_constant, sizeof(SSILImportanceMapPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_ssil_buffers.half_buffer_width, p_ssil_buffers.half_buffer_height, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);

			// process Importance Map B
			RD::Uniform u_importance_map_pong_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssil_buffers.importance_map[1] }));

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssil.pipelines[SSIL_PROCESS_IMPORTANCE_MAPB]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_importance_map_pong_with_sampler), 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_importance_map), 1);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, ssil.counter_uniform_set, 2);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssil.importance_map_push_constant, sizeof(SSILImportanceMapPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_ssil_buffers.half_buffer_width, p_ssil_buffers.half_buffer_height, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);

			RD::get_singleton()->draw_command_end_label(); // Importance Map

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssil.pipelines[SSIL_GATHER_ADAPTIVE]);
		} else {
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssil.pipelines[SSIL_GATHER]);
		}

		gather_ssil(compute_list, p_ssil_buffers.deinterleaved_slices, p_ssil_buffers.edges_slices, p_settings, false, p_ssil_buffers.gather_uniform_set, p_ssil_buffers.importance_map_uniform_set, p_ssil_buffers.projection_uniform_set);
		RD::get_singleton()->draw_command_end_label(); //Gather
	}

	{
		RD::get_singleton()->draw_command_begin_label("Edge Aware Blur");
		ssil.blur_push_constant.edge_sharpness = 1.0 - p_settings.sharpness;
		ssil.blur_push_constant.half_screen_pixel_size[0] = 1.0 / p_ssil_buffers.buffer_width;
		ssil.blur_push_constant.half_screen_pixel_size[1] = 1.0 / p_ssil_buffers.buffer_height;

		int blur_passes = p_settings.quality > RS::ENV_SSIL_QUALITY_VERY_LOW ? p_settings.blur_passes : 1;

		shader = ssil.blur_shader.version_get_shader(ssil.blur_shader_version, 0);

		for (int pass = 0; pass < blur_passes; pass++) {
			int blur_pipeline = SSIL_BLUR_PASS;
			if (p_settings.quality > RS::ENV_SSIL_QUALITY_VERY_LOW) {
				blur_pipeline = SSIL_BLUR_PASS_SMART;
				if (pass < blur_passes - 2) {
					blur_pipeline = SSIL_BLUR_PASS_WIDE;
				}
			}

			for (int i = 0; i < 4; i++) {
				if ((p_settings.quality == RS::ENV_SSIL_QUALITY_VERY_LOW) && ((i == 1) || (i == 2))) {
					continue;
				}

				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssil.pipelines[blur_pipeline]);
				if (pass % 2 == 0) {
					if (p_settings.quality == RS::ENV_SSIL_QUALITY_VERY_LOW) {
						RD::Uniform u_ssil_slice(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssil_buffers.deinterleaved_slices[i] }));
						RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_ssil_slice), 0);
					} else {
						RD::Uniform u_ssil_slice(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ ss_effects.mirror_sampler, p_ssil_buffers.deinterleaved_slices[i] }));
						RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_ssil_slice), 0);
					}

					RD::Uniform u_ssil_pong_slice(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssil_buffers.pong_slices[i] }));
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_ssil_pong_slice), 1);
				} else {
					if (p_settings.quality == RS::ENV_SSIL_QUALITY_VERY_LOW) {
						RD::Uniform u_ssil_pong_slice(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssil_buffers.pong_slices[i] }));
						RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_ssil_pong_slice), 0);
					} else {
						RD::Uniform u_ssil_pong_slice(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ ss_effects.mirror_sampler, p_ssil_buffers.pong_slices[i] }));
						RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_ssil_pong_slice), 0);
					}

					RD::Uniform u_ssil_slice(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssil_buffers.deinterleaved_slices[i] }));
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_ssil_slice), 1);
				}

				RD::Uniform u_edges_slice(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssil_buffers.edges_slices[i] }));
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 2, u_edges_slice), 2);

				RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssil.blur_push_constant, sizeof(SSILBlurPushConstant));

				int x_groups = (p_settings.full_screen_size.x >> (p_settings.half_size ? 2 : 1));
				int y_groups = (p_settings.full_screen_size.y >> (p_settings.half_size ? 2 : 1));

				RD::get_singleton()->compute_list_dispatch_threads(compute_list, x_groups, y_groups, 1);
				if (p_settings.quality > RS::ENV_SSIL_QUALITY_VERY_LOW) {
					RD::get_singleton()->compute_list_add_barrier(compute_list);
				}
			}
		}

		RD::get_singleton()->draw_command_end_label(); // Blur
	}

	{
		RD::get_singleton()->draw_command_begin_label("Interleave Buffers");
		ssil.interleave_push_constant.inv_sharpness = 1.0 - p_settings.sharpness;
		ssil.interleave_push_constant.pixel_size[0] = 1.0 / p_settings.full_screen_size.x;
		ssil.interleave_push_constant.pixel_size[1] = 1.0 / p_settings.full_screen_size.y;
		ssil.interleave_push_constant.size_modifier = uint32_t(p_settings.half_size ? 4 : 2);

		int interleave_pipeline = SSIL_INTERLEAVE_HALF;
		if (p_settings.quality == RS::ENV_SSIL_QUALITY_LOW) {
			interleave_pipeline = SSIL_INTERLEAVE;
		} else if (p_settings.quality >= RS::ENV_SSIL_QUALITY_MEDIUM) {
			interleave_pipeline = SSIL_INTERLEAVE_SMART;
		}

		shader = ssil.interleave_shader.version_get_shader(ssil.interleave_shader_version, 0);

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssil.pipelines[interleave_pipeline]);

		RD::Uniform u_destination(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssil_buffers.ssil_final }));
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_destination), 0);

		if (p_settings.quality > RS::ENV_SSIL_QUALITY_VERY_LOW && p_settings.blur_passes % 2 == 0) {
			RD::Uniform u_ssil(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssil_buffers.deinterleaved }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_ssil), 1);
		} else {
			RD::Uniform u_ssil_pong(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssil_buffers.pong }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_ssil_pong), 1);
		}

		RD::Uniform u_edges(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssil_buffers.edges }));
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 2, u_edges), 2);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssil.interleave_push_constant, sizeof(SSILInterleavePushConstant));

		RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_settings.full_screen_size.x, p_settings.full_screen_size.y, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);
		RD::get_singleton()->draw_command_end_label(); // Interleave
	}

	RD::get_singleton()->draw_command_end_label(); // SSIL

	RD::get_singleton()->compute_list_end(RD::BARRIER_MASK_NO_BARRIER);

	int zero[1] = { 0 };
	RD::get_singleton()->buffer_update(ssil.importance_map_load_counter, 0, sizeof(uint32_t), &zero, 0); //no barrier
}

void SSEffects::ssil_free(SSILRenderBuffers &p_ssil_buffers) {
	if (p_ssil_buffers.ssil_final.is_valid()) {
		RD::get_singleton()->free(p_ssil_buffers.ssil_final);
		RD::get_singleton()->free(p_ssil_buffers.deinterleaved);
		RD::get_singleton()->free(p_ssil_buffers.pong);
		RD::get_singleton()->free(p_ssil_buffers.edges);
		RD::get_singleton()->free(p_ssil_buffers.importance_map[0]);
		RD::get_singleton()->free(p_ssil_buffers.importance_map[1]);
		RD::get_singleton()->free(p_ssil_buffers.last_frame);

		p_ssil_buffers.ssil_final = RID();
		p_ssil_buffers.deinterleaved = RID();
		p_ssil_buffers.pong = RID();
		p_ssil_buffers.edges = RID();
		p_ssil_buffers.deinterleaved_slices.clear();
		p_ssil_buffers.pong_slices.clear();
		p_ssil_buffers.edges_slices.clear();
		p_ssil_buffers.importance_map[0] = RID();
		p_ssil_buffers.importance_map[1] = RID();
		p_ssil_buffers.last_frame = RID();
		p_ssil_buffers.last_frame_slices.clear();

		p_ssil_buffers.gather_uniform_set = RID();
		p_ssil_buffers.importance_map_uniform_set = RID();
		p_ssil_buffers.projection_uniform_set = RID();
	}
}

/* SSAO */

void SSEffects::gather_ssao(RD::ComputeListID p_compute_list, const Vector<RID> p_ao_slices, const SSAOSettings &p_settings, bool p_adaptive_base_pass, RID p_gather_uniform_set, RID p_importance_map_uniform_set) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);

	RD::get_singleton()->compute_list_bind_uniform_set(p_compute_list, p_gather_uniform_set, 0);
	if ((p_settings.quality == RS::ENV_SSAO_QUALITY_ULTRA) && !p_adaptive_base_pass) {
		RD::get_singleton()->compute_list_bind_uniform_set(p_compute_list, p_importance_map_uniform_set, 0);
	}

	RID shader = ssao.gather_shader.version_get_shader(ssao.gather_shader_version, 1); //

	for (int i = 0; i < 4; i++) {
		if ((p_settings.quality == RS::ENV_SSAO_QUALITY_VERY_LOW) && ((i == 1) || (i == 2))) {
			continue;
		}

		RD::Uniform u_ao_slice(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ao_slices[i] }));

		ssao.gather_push_constant.pass_coord_offset[0] = i % 2;
		ssao.gather_push_constant.pass_coord_offset[1] = i / 2;
		ssao.gather_push_constant.pass_uv_offset[0] = ((i % 2) - 0.0) / p_settings.full_screen_size.x;
		ssao.gather_push_constant.pass_uv_offset[1] = ((i / 2) - 0.0) / p_settings.full_screen_size.y;
		ssao.gather_push_constant.pass = i;
		RD::get_singleton()->compute_list_bind_uniform_set(p_compute_list, uniform_set_cache->get_cache(shader, 2, u_ao_slice), 2);
		RD::get_singleton()->compute_list_set_push_constant(p_compute_list, &ssao.gather_push_constant, sizeof(SSAOGatherPushConstant));

		Size2i size = Size2i(p_settings.full_screen_size.x >> (p_settings.half_size ? 2 : 1), p_settings.full_screen_size.y >> (p_settings.half_size ? 2 : 1));

		RD::get_singleton()->compute_list_dispatch_threads(p_compute_list, size.x, size.y, 1);
	}
	RD::get_singleton()->compute_list_add_barrier(p_compute_list);
}

void SSEffects::ssao_allocate_buffers(SSAORenderBuffers &p_ssao_buffers, const SSAOSettings &p_settings, RID p_linear_depth) {
	if (p_ssao_buffers.half_size != p_settings.half_size) {
		ssao_free(p_ssao_buffers);
	}

	if (p_settings.half_size) {
		p_ssao_buffers.buffer_width = (p_settings.full_screen_size.x + 3) / 4;
		p_ssao_buffers.buffer_height = (p_settings.full_screen_size.y + 3) / 4;
		p_ssao_buffers.half_buffer_width = (p_settings.full_screen_size.x + 7) / 8;
		p_ssao_buffers.half_buffer_height = (p_settings.full_screen_size.y + 7) / 8;
	} else {
		p_ssao_buffers.buffer_width = (p_settings.full_screen_size.x + 1) / 2;
		p_ssao_buffers.buffer_height = (p_settings.full_screen_size.y + 1) / 2;
		p_ssao_buffers.half_buffer_width = (p_settings.full_screen_size.x + 3) / 4;
		p_ssao_buffers.half_buffer_height = (p_settings.full_screen_size.y + 3) / 4;
	}

	if (p_ssao_buffers.ao_deinterleaved.is_null()) {
		{
			p_ssao_buffers.depth_texture_view = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_linear_depth, 0, p_settings.half_size ? 1 : 0, 4, RD::TEXTURE_SLICE_2D_ARRAY);
		}
		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8G8_UNORM;
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
			tf.width = p_ssao_buffers.buffer_width;
			tf.height = p_ssao_buffers.buffer_height;
			tf.array_layers = 4;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			p_ssao_buffers.ao_deinterleaved = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(p_ssao_buffers.ao_deinterleaved, "SSAO De-interleaved Array");
			for (uint32_t i = 0; i < 4; i++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_ssao_buffers.ao_deinterleaved, i, 0);
				p_ssao_buffers.ao_deinterleaved_slices.push_back(slice);
				RD::get_singleton()->set_resource_name(slice, "SSAO De-interleaved Array Layer " + itos(i) + " ");
			}
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8G8_UNORM;
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
			tf.width = p_ssao_buffers.buffer_width;
			tf.height = p_ssao_buffers.buffer_height;
			tf.array_layers = 4;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			p_ssao_buffers.ao_pong = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(p_ssao_buffers.ao_pong, "SSAO De-interleaved Array Pong");
			for (uint32_t i = 0; i < 4; i++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_ssao_buffers.ao_pong, i, 0);
				p_ssao_buffers.ao_pong_slices.push_back(slice);
				RD::get_singleton()->set_resource_name(slice, "SSAO De-interleaved Array Layer " + itos(i) + " Pong");
			}
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = p_ssao_buffers.buffer_width;
			tf.height = p_ssao_buffers.buffer_height;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			p_ssao_buffers.importance_map[0] = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(p_ssao_buffers.importance_map[0], "SSAO Importance Map");
			p_ssao_buffers.importance_map[1] = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(p_ssao_buffers.importance_map[1], "SSAO Importance Map Pong");
		}
		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = p_settings.full_screen_size.x;
			tf.height = p_settings.full_screen_size.y;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			p_ssao_buffers.ao_final = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(p_ssao_buffers.ao_final, "SSAO Final");
		}
		p_ssao_buffers.half_size = p_settings.half_size;
	}
}

void SSEffects::generate_ssao(SSAORenderBuffers &p_ssao_buffers, RID p_normal_buffer, const Projection &p_projection, const SSAOSettings &p_settings) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	memset(&ssao.gather_push_constant, 0, sizeof(SSAOGatherPushConstant));
	/* FIRST PASS */

	RID shader = ssao.gather_shader.version_get_shader(ssao.gather_shader_version, 0);
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::get_singleton()->draw_command_begin_label("Process Screen Space Ambient Occlusion");
	/* SECOND PASS */
	// Sample SSAO
	{
		RD::get_singleton()->draw_command_begin_label("Gather Samples");
		ssao.gather_push_constant.screen_size[0] = p_settings.full_screen_size.x;
		ssao.gather_push_constant.screen_size[1] = p_settings.full_screen_size.y;

		ssao.gather_push_constant.half_screen_pixel_size[0] = 1.0 / p_ssao_buffers.buffer_width;
		ssao.gather_push_constant.half_screen_pixel_size[1] = 1.0 / p_ssao_buffers.buffer_height;
		float tan_half_fov_x = 1.0 / p_projection.matrix[0][0];
		float tan_half_fov_y = 1.0 / p_projection.matrix[1][1];
		ssao.gather_push_constant.NDC_to_view_mul[0] = tan_half_fov_x * 2.0;
		ssao.gather_push_constant.NDC_to_view_mul[1] = tan_half_fov_y * -2.0;
		ssao.gather_push_constant.NDC_to_view_add[0] = tan_half_fov_x * -1.0;
		ssao.gather_push_constant.NDC_to_view_add[1] = tan_half_fov_y;
		ssao.gather_push_constant.is_orthogonal = p_projection.is_orthogonal();

		ssao.gather_push_constant.half_screen_pixel_size_x025[0] = ssao.gather_push_constant.half_screen_pixel_size[0] * 0.25;
		ssao.gather_push_constant.half_screen_pixel_size_x025[1] = ssao.gather_push_constant.half_screen_pixel_size[1] * 0.25;

		ssao.gather_push_constant.radius = p_settings.radius;
		float radius_near_limit = (p_settings.radius * 1.2f);
		if (p_settings.quality <= RS::ENV_SSAO_QUALITY_LOW) {
			radius_near_limit *= 1.50f;

			if (p_settings.quality == RS::ENV_SSAO_QUALITY_VERY_LOW) {
				ssao.gather_push_constant.radius *= 0.8f;
			}
		}
		radius_near_limit /= tan_half_fov_y;
		ssao.gather_push_constant.intensity = p_settings.intensity;
		ssao.gather_push_constant.shadow_power = p_settings.power;
		ssao.gather_push_constant.shadow_clamp = 0.98;
		ssao.gather_push_constant.fade_out_mul = -1.0 / (p_settings.fadeout_to - p_settings.fadeout_from);
		ssao.gather_push_constant.fade_out_add = p_settings.fadeout_from / (p_settings.fadeout_to - p_settings.fadeout_from) + 1.0;
		ssao.gather_push_constant.horizon_angle_threshold = p_settings.horizon;
		ssao.gather_push_constant.inv_radius_near_limit = 1.0f / radius_near_limit;
		ssao.gather_push_constant.neg_inv_radius = -1.0 / ssao.gather_push_constant.radius;

		ssao.gather_push_constant.load_counter_avg_div = 9.0 / float((p_ssao_buffers.half_buffer_width) * (p_ssao_buffers.half_buffer_height) * 255);
		ssao.gather_push_constant.adaptive_sample_limit = p_settings.adaptive_target;

		ssao.gather_push_constant.detail_intensity = p_settings.detail;
		ssao.gather_push_constant.quality = MAX(0, p_settings.quality - 1);
		ssao.gather_push_constant.size_multiplier = p_settings.half_size ? 2 : 1;

		if (p_ssao_buffers.gather_uniform_set.is_null()) {
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
				u.binding = 0;
				u.append_id(default_sampler);
				u.append_id(p_ssao_buffers.depth_texture_view);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 1;
				u.append_id(p_normal_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
				u.binding = 2;
				u.append_id(ss_effects.gather_constants_buffer);
				uniforms.push_back(u);
			}
			p_ssao_buffers.gather_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, shader, 0);
			RD::get_singleton()->set_resource_name(p_ssao_buffers.gather_uniform_set, "SSAO Gather Uniform Set");
		}

		if (p_ssao_buffers.importance_map_uniform_set.is_null()) {
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 0;
				u.append_id(p_ssao_buffers.ao_pong);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
				u.binding = 1;
				u.append_id(default_sampler);
				u.append_id(p_ssao_buffers.importance_map[0]);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 2;
				u.append_id(ssao.importance_map_load_counter);
				uniforms.push_back(u);
			}
			p_ssao_buffers.importance_map_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, ssao.gather_shader.version_get_shader(ssao.gather_shader_version, 2), 1);
			RD::get_singleton()->set_resource_name(p_ssao_buffers.importance_map_uniform_set, "SSAO Importance Map Uniform Set");
		}

		if (p_settings.quality == RS::ENV_SSAO_QUALITY_ULTRA) {
			RD::get_singleton()->draw_command_begin_label("Generate Importance Map");
			ssao.importance_map_push_constant.half_screen_pixel_size[0] = 1.0 / p_ssao_buffers.buffer_width;
			ssao.importance_map_push_constant.half_screen_pixel_size[1] = 1.0 / p_ssao_buffers.buffer_height;
			ssao.importance_map_push_constant.intensity = p_settings.intensity;
			ssao.importance_map_push_constant.power = p_settings.power;

			//base pass
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[SSAO_GATHER_BASE]);
			gather_ssao(compute_list, p_ssao_buffers.ao_pong_slices, p_settings, true, p_ssao_buffers.gather_uniform_set, RID());

			//generate importance map
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[SSAO_GENERATE_IMPORTANCE_MAP]);

			RD::Uniform u_ao_pong_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssao_buffers.ao_pong }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_ao_pong_with_sampler), 0);

			RD::Uniform u_importance_map(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssao_buffers.importance_map[0] }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_importance_map), 1);

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssao.importance_map_push_constant, sizeof(SSAOImportanceMapPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_ssao_buffers.half_buffer_width, p_ssao_buffers.half_buffer_height, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);

			//process importance map A
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[SSAO_PROCESS_IMPORTANCE_MAPA]);

			RD::Uniform u_importance_map_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssao_buffers.importance_map[0] }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_importance_map_with_sampler), 0);

			RD::Uniform u_importance_map_pong(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssao_buffers.importance_map[1] }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_importance_map_pong), 1);

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssao.importance_map_push_constant, sizeof(SSAOImportanceMapPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_ssao_buffers.half_buffer_width, p_ssao_buffers.half_buffer_height, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);

			//process Importance Map B
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[SSAO_PROCESS_IMPORTANCE_MAPB]);

			RD::Uniform u_importance_map_pong_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssao_buffers.importance_map[1] }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_importance_map_pong_with_sampler), 0);

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_importance_map), 1);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, ssao.counter_uniform_set, 2);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssao.importance_map_push_constant, sizeof(SSAOImportanceMapPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_ssao_buffers.half_buffer_width, p_ssao_buffers.half_buffer_height, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[SSAO_GATHER_ADAPTIVE]);
			RD::get_singleton()->draw_command_end_label(); // Importance Map
		} else {
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[SSAO_GATHER]);
		}

		gather_ssao(compute_list, p_ssao_buffers.ao_deinterleaved_slices, p_settings, false, p_ssao_buffers.gather_uniform_set, p_ssao_buffers.importance_map_uniform_set);
		RD::get_singleton()->draw_command_end_label(); // Gather SSAO
	}

	//	/* THIRD PASS */
	//	// Blur
	//
	{
		RD::get_singleton()->draw_command_begin_label("Edge Aware Blur");
		ssao.blur_push_constant.edge_sharpness = 1.0 - p_settings.sharpness;
		ssao.blur_push_constant.half_screen_pixel_size[0] = 1.0 / p_ssao_buffers.buffer_width;
		ssao.blur_push_constant.half_screen_pixel_size[1] = 1.0 / p_ssao_buffers.buffer_height;

		int blur_passes = p_settings.quality > RS::ENV_SSAO_QUALITY_VERY_LOW ? p_settings.blur_passes : 1;

		shader = ssao.blur_shader.version_get_shader(ssao.blur_shader_version, 0);

		for (int pass = 0; pass < blur_passes; pass++) {
			int blur_pipeline = SSAO_BLUR_PASS;
			if (p_settings.quality > RS::ENV_SSAO_QUALITY_VERY_LOW) {
				blur_pipeline = SSAO_BLUR_PASS_SMART;
				if (pass < blur_passes - 2) {
					blur_pipeline = SSAO_BLUR_PASS_WIDE;
				} else {
					blur_pipeline = SSAO_BLUR_PASS_SMART;
				}
			}

			for (int i = 0; i < 4; i++) {
				if ((p_settings.quality == RS::ENV_SSAO_QUALITY_VERY_LOW) && ((i == 1) || (i == 2))) {
					continue;
				}

				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[blur_pipeline]);
				if (pass % 2 == 0) {
					if (p_settings.quality == RS::ENV_SSAO_QUALITY_VERY_LOW) {
						RD::Uniform u_ao_slices_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssao_buffers.ao_deinterleaved_slices[i] }));
						RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_ao_slices_with_sampler), 0);
					} else {
						RD::Uniform u_ao_slices_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ ss_effects.mirror_sampler, p_ssao_buffers.ao_deinterleaved_slices[i] }));
						RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_ao_slices_with_sampler), 0);
					}

					RD::Uniform u_ao_pong_slices(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssao_buffers.ao_pong_slices[i] }));
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_ao_pong_slices), 1);
				} else {
					if (p_settings.quality == RS::ENV_SSAO_QUALITY_VERY_LOW) {
						RD::Uniform u_ao_pong_slices_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssao_buffers.ao_pong_slices[i] }));
						RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_ao_pong_slices_with_sampler), 0);
					} else {
						RD::Uniform u_ao_pong_slices_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ ss_effects.mirror_sampler, p_ssao_buffers.ao_pong_slices[i] }));
						RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_ao_pong_slices_with_sampler), 0);
					}

					RD::Uniform u_ao_slices(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssao_buffers.ao_deinterleaved_slices[i] }));
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_ao_slices), 1);
				}
				RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssao.blur_push_constant, sizeof(SSAOBlurPushConstant));

				Size2i size(p_settings.full_screen_size.x >> (p_settings.half_size ? 2 : 1), p_settings.full_screen_size.y >> (p_settings.half_size ? 2 : 1));
				RD::get_singleton()->compute_list_dispatch_threads(compute_list, size.x, size.y, 1);
			}

			if (p_settings.quality > RS::ENV_SSAO_QUALITY_VERY_LOW) {
				RD::get_singleton()->compute_list_add_barrier(compute_list);
			}
		}
		RD::get_singleton()->draw_command_end_label(); // Blur
	}

	/* FOURTH PASS */
	// Interleave buffers
	// back to full size
	{
		RD::get_singleton()->draw_command_begin_label("Interleave Buffers");
		ssao.interleave_push_constant.inv_sharpness = 1.0 - p_settings.sharpness;
		ssao.interleave_push_constant.pixel_size[0] = 1.0 / p_settings.full_screen_size.x;
		ssao.interleave_push_constant.pixel_size[1] = 1.0 / p_settings.full_screen_size.y;
		ssao.interleave_push_constant.size_modifier = uint32_t(p_settings.half_size ? 4 : 2);

		shader = ssao.interleave_shader.version_get_shader(ssao.interleave_shader_version, 0);

		int interleave_pipeline = SSAO_INTERLEAVE_HALF;
		if (p_settings.quality == RS::ENV_SSAO_QUALITY_LOW) {
			interleave_pipeline = SSAO_INTERLEAVE;
		} else if (p_settings.quality >= RS::ENV_SSAO_QUALITY_MEDIUM) {
			interleave_pipeline = SSAO_INTERLEAVE_SMART;
		}

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[interleave_pipeline]);

		RD::Uniform u_upscale_buffer(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssao_buffers.ao_final }));
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_upscale_buffer), 0);

		if (p_settings.quality > RS::ENV_SSAO_QUALITY_VERY_LOW && p_settings.blur_passes % 2 == 0) {
			RD::Uniform u_ao(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssao_buffers.ao_deinterleaved }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_ao), 1);
		} else {
			RD::Uniform u_ao(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_ssao_buffers.ao_pong }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_ao), 1);
		}

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssao.interleave_push_constant, sizeof(SSAOInterleavePushConstant));

		RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_settings.full_screen_size.x, p_settings.full_screen_size.y, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);
		RD::get_singleton()->draw_command_end_label(); // Interleave
	}
	RD::get_singleton()->draw_command_end_label(); //SSAO
	RD::get_singleton()->compute_list_end(RD::BARRIER_MASK_NO_BARRIER); //wait for upcoming transfer

	int zero[1] = { 0 };
	RD::get_singleton()->buffer_update(ssao.importance_map_load_counter, 0, sizeof(uint32_t), &zero, 0); //no barrier
}

void SSEffects::ssao_free(SSAORenderBuffers &p_ssao_buffers) {
	if (p_ssao_buffers.ao_final.is_valid()) {
		RD::get_singleton()->free(p_ssao_buffers.ao_deinterleaved);
		RD::get_singleton()->free(p_ssao_buffers.ao_pong);
		RD::get_singleton()->free(p_ssao_buffers.ao_final);

		RD::get_singleton()->free(p_ssao_buffers.importance_map[0]);
		RD::get_singleton()->free(p_ssao_buffers.importance_map[1]);

		p_ssao_buffers.ao_deinterleaved = RID();
		p_ssao_buffers.ao_pong = RID();
		p_ssao_buffers.ao_final = RID();
		p_ssao_buffers.importance_map[0] = RID();
		p_ssao_buffers.importance_map[1] = RID();
		p_ssao_buffers.ao_deinterleaved_slices.clear();
		p_ssao_buffers.ao_pong_slices.clear();

		p_ssao_buffers.gather_uniform_set = RID();
		p_ssao_buffers.importance_map_uniform_set = RID();
	}
}

/* Screen Space Reflection */

void SSEffects::ssr_allocate_buffers(SSRRenderBuffers &p_ssr_buffers, const RenderingDevice::DataFormat p_color_format, RenderingServer::EnvironmentSSRRoughnessQuality p_roughness_quality, const Size2i &p_screen_size, const uint32_t p_view_count) {
	// As we are processing one view at a time, we can reuse buffers, only our output needs to have layers for each view.

	if (p_ssr_buffers.depth_scaled.is_null()) {
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		tf.width = p_screen_size.x;
		tf.height = p_screen_size.y;
		tf.texture_type = RD::TEXTURE_TYPE_2D;
		tf.array_layers = 1;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

		p_ssr_buffers.depth_scaled = RD::get_singleton()->texture_create(tf, RD::TextureView());
		RD::get_singleton()->set_resource_name(p_ssr_buffers.depth_scaled, "SSR Depth Scaled");

		tf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;

		p_ssr_buffers.normal_scaled = RD::get_singleton()->texture_create(tf, RD::TextureView());
		RD::get_singleton()->set_resource_name(p_ssr_buffers.normal_scaled, "SSR Normal Scaled");
	}

	if (p_roughness_quality != RS::ENV_SSR_ROUGHNESS_QUALITY_DISABLED && !p_ssr_buffers.blur_radius[0].is_valid()) {
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R8_UNORM;
		tf.width = p_screen_size.x;
		tf.height = p_screen_size.y;
		tf.texture_type = RD::TEXTURE_TYPE_2D;
		tf.array_layers = 1;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;

		p_ssr_buffers.blur_radius[0] = RD::get_singleton()->texture_create(tf, RD::TextureView());
		RD::get_singleton()->set_resource_name(p_ssr_buffers.blur_radius[0], "SSR Blur Radius 0");
		p_ssr_buffers.blur_radius[1] = RD::get_singleton()->texture_create(tf, RD::TextureView());
		RD::get_singleton()->set_resource_name(p_ssr_buffers.blur_radius[1], "SSR Blur Radius 1");
	}

	if (p_ssr_buffers.intermediate.is_null()) {
		RD::TextureFormat tf;
		tf.format = p_color_format;
		tf.width = p_screen_size.x;
		tf.height = p_screen_size.y;
		tf.texture_type = RD::TEXTURE_TYPE_2D;
		tf.array_layers = 1;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

		p_ssr_buffers.intermediate = RD::get_singleton()->texture_create(tf, RD::TextureView());
		RD::get_singleton()->set_resource_name(p_ssr_buffers.intermediate, "SSR Intermediate");

		if (p_view_count > 1) {
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
			tf.array_layers = p_view_count;
		} else {
			tf.texture_type = RD::TEXTURE_TYPE_2D;
			tf.array_layers = 1;
		}

		p_ssr_buffers.output = RD::get_singleton()->texture_create(tf, RD::TextureView());
		RD::get_singleton()->set_resource_name(p_ssr_buffers.output, "SSR Output");

		for (uint32_t v = 0; v < p_view_count; v++) {
			p_ssr_buffers.output_slices[v] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_ssr_buffers.output, v, 0);
		}
	}
}

void SSEffects::screen_space_reflection(SSRRenderBuffers &p_ssr_buffers, const RID *p_diffuse_slices, const RID *p_normal_roughness_slices, RenderingServer::EnvironmentSSRRoughnessQuality p_roughness_quality, const RID *p_metallic_slices, const Color &p_metallic_mask, const RID *p_depth_slices, const Size2i &p_screen_size, int p_max_steps, float p_fade_in, float p_fade_out, float p_tolerance, const uint32_t p_view_count, const Projection *p_projections, const Vector3 *p_eye_offsets) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	{
		// Store some scene data in a UBO, in the near future we will use a UBO shared with other shaders
		ScreenSpaceReflectionSceneData scene_data;

		if (ssr.ubo.is_null()) {
			ssr.ubo = RD::get_singleton()->uniform_buffer_create(sizeof(ScreenSpaceReflectionSceneData));
		}

		for (uint32_t v = 0; v < p_view_count; v++) {
			store_camera(p_projections[v], scene_data.projection[v]);
			store_camera(p_projections[v].inverse(), scene_data.inv_projection[v]);
			scene_data.eye_offset[v][0] = p_eye_offsets[v].x;
			scene_data.eye_offset[v][1] = p_eye_offsets[v].y;
			scene_data.eye_offset[v][2] = p_eye_offsets[v].z;
			scene_data.eye_offset[v][3] = 0.0;
		}

		RD::get_singleton()->buffer_update(ssr.ubo, 0, sizeof(ScreenSpaceReflectionSceneData), &scene_data, RD::BARRIER_MASK_COMPUTE);
	}

	uint32_t pipeline_specialization = 0;
	if (p_view_count > 1) {
		pipeline_specialization |= SSR_MULTIVIEW;
	}

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	for (uint32_t v = 0; v < p_view_count; v++) {
		RD::get_singleton()->draw_command_begin_label(String("SSR View ") + itos(v));

		{ //scale color and depth to half
			RD::get_singleton()->draw_command_begin_label("SSR Scale");

			ScreenSpaceReflectionScalePushConstant push_constant;
			push_constant.view_index = v;
			push_constant.camera_z_far = p_projections[v].get_z_far();
			push_constant.camera_z_near = p_projections[v].get_z_near();
			push_constant.orthogonal = p_projections[v].is_orthogonal();
			push_constant.filter = false; //enabling causes arctifacts
			push_constant.screen_size[0] = p_screen_size.x;
			push_constant.screen_size[1] = p_screen_size.y;

			RID shader = ssr_scale.shader.version_get_shader(ssr_scale.shader_version, 0);

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssr_scale.pipelines[pipeline_specialization]);

			RD::Uniform u_diffuse(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_diffuse_slices[v] }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_diffuse), 0);

			RD::Uniform u_depth(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_depth_slices[v] }));
			RD::Uniform u_normal_roughness(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, Vector<RID>({ default_sampler, p_normal_roughness_slices[v] }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_depth, u_normal_roughness), 1);

			RD::Uniform u_output_blur(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssr_buffers.output_slices[v] }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 2, u_output_blur), 2);

			RD::Uniform u_scale_depth(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssr_buffers.depth_scaled }));
			RD::Uniform u_scale_normal(RD::UNIFORM_TYPE_IMAGE, 1, Vector<RID>({ p_ssr_buffers.normal_scaled }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_scale_depth, u_scale_normal), 3);

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(ScreenSpaceReflectionScalePushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_screen_size.width, p_screen_size.height, 1);

			RD::get_singleton()->compute_list_add_barrier(compute_list);

			RD::get_singleton()->draw_command_end_label();
		}

		{
			RD::get_singleton()->draw_command_begin_label("SSR main");

			ScreenSpaceReflectionPushConstant push_constant;
			push_constant.view_index = v;
			push_constant.camera_z_far = p_projections[v].get_z_far();
			push_constant.camera_z_near = p_projections[v].get_z_near();
			push_constant.orthogonal = p_projections[v].is_orthogonal();
			push_constant.screen_size[0] = p_screen_size.x;
			push_constant.screen_size[1] = p_screen_size.y;
			push_constant.curve_fade_in = p_fade_in;
			push_constant.distance_fade = p_fade_out;
			push_constant.num_steps = p_max_steps;
			push_constant.depth_tolerance = p_tolerance;
			push_constant.use_half_res = true;
			push_constant.proj_info[0] = -2.0f / (p_screen_size.width * p_projections[v].matrix[0][0]);
			push_constant.proj_info[1] = -2.0f / (p_screen_size.height * p_projections[v].matrix[1][1]);
			push_constant.proj_info[2] = (1.0f - p_projections[v].matrix[0][2]) / p_projections[v].matrix[0][0];
			push_constant.proj_info[3] = (1.0f + p_projections[v].matrix[1][2]) / p_projections[v].matrix[1][1];
			push_constant.metallic_mask[0] = CLAMP(p_metallic_mask.r * 255.0, 0, 255);
			push_constant.metallic_mask[1] = CLAMP(p_metallic_mask.g * 255.0, 0, 255);
			push_constant.metallic_mask[2] = CLAMP(p_metallic_mask.b * 255.0, 0, 255);
			push_constant.metallic_mask[3] = CLAMP(p_metallic_mask.a * 255.0, 0, 255);

			ScreenSpaceReflectionMode mode = (p_roughness_quality != RS::ENV_SSR_ROUGHNESS_QUALITY_DISABLED) ? SCREEN_SPACE_REFLECTION_ROUGH : SCREEN_SPACE_REFLECTION_NORMAL;
			RID shader = ssr.shader.version_get_shader(ssr.shader_version, mode);

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssr.pipelines[pipeline_specialization][mode]);

			RD::Uniform u_scene_data(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, ssr.ubo);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 4, u_scene_data), 4);

			RD::Uniform u_output_blur(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssr_buffers.output_slices[v] }));
			RD::Uniform u_scale_depth(RD::UNIFORM_TYPE_IMAGE, 1, Vector<RID>({ p_ssr_buffers.depth_scaled }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_output_blur, u_scale_depth), 0);

			if (p_roughness_quality != RS::ENV_SSR_ROUGHNESS_QUALITY_DISABLED) {
				RD::Uniform u_intermediate(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssr_buffers.intermediate }));
				RD::Uniform u_blur_radius(RD::UNIFORM_TYPE_IMAGE, 1, Vector<RID>({ p_ssr_buffers.blur_radius[0] }));
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_intermediate, u_blur_radius), 1);
			} else {
				RD::Uniform u_intermediate(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssr_buffers.intermediate }));
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_intermediate), 1);
			}

			RD::Uniform u_scale_normal(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssr_buffers.normal_scaled }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 2, u_scale_normal), 2);

			RD::Uniform u_metallic(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_metallic_slices[v] }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_metallic), 3);

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(ScreenSpaceReflectionPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_screen_size.width, p_screen_size.height, 1);

			RD::get_singleton()->draw_command_end_label();
		}

		if (p_roughness_quality != RS::ENV_SSR_ROUGHNESS_QUALITY_DISABLED) {
			RD::get_singleton()->draw_command_begin_label("SSR filter");
			//blur

			RD::get_singleton()->compute_list_add_barrier(compute_list);

			ScreenSpaceReflectionFilterPushConstant push_constant;
			push_constant.view_index = v;
			push_constant.orthogonal = p_projections[v].is_orthogonal();
			push_constant.edge_tolerance = Math::sin(Math::deg_to_rad(15.0));
			push_constant.proj_info[0] = -2.0f / (p_screen_size.width * p_projections[v].matrix[0][0]);
			push_constant.proj_info[1] = -2.0f / (p_screen_size.height * p_projections[v].matrix[1][1]);
			push_constant.proj_info[2] = (1.0f - p_projections[v].matrix[0][2]) / p_projections[v].matrix[0][0];
			push_constant.proj_info[3] = (1.0f + p_projections[v].matrix[1][2]) / p_projections[v].matrix[1][1];
			push_constant.vertical = 0;
			if (p_roughness_quality == RS::ENV_SSR_ROUGHNESS_QUALITY_LOW) {
				push_constant.steps = p_max_steps / 3;
				push_constant.increment = 3;
			} else if (p_roughness_quality == RS::ENV_SSR_ROUGHNESS_QUALITY_MEDIUM) {
				push_constant.steps = p_max_steps / 2;
				push_constant.increment = 2;
			} else {
				push_constant.steps = p_max_steps;
				push_constant.increment = 1;
			}

			push_constant.screen_size[0] = p_screen_size.width;
			push_constant.screen_size[1] = p_screen_size.height;

			// Horizontal pass

			SSRReflectionMode mode = SCREEN_SPACE_REFLECTION_FILTER_HORIZONTAL;

			RID shader = ssr_filter.shader.version_get_shader(ssr_filter.shader_version, mode);

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssr_filter.pipelines[pipeline_specialization][mode]);

			RD::Uniform u_scene_data(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, ssr.ubo);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 4, u_scene_data), 4);

			RD::Uniform u_intermediate(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssr_buffers.intermediate }));
			RD::Uniform u_blur_radius(RD::UNIFORM_TYPE_IMAGE, 1, Vector<RID>({ p_ssr_buffers.blur_radius[0] }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_intermediate, u_blur_radius), 0);

			RD::Uniform u_scale_normal(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssr_buffers.normal_scaled }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_scale_normal), 1);

			RD::Uniform u_output_blur(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssr_buffers.output_slices[v] }));
			RD::Uniform u_blur_radius2(RD::UNIFORM_TYPE_IMAGE, 1, Vector<RID>({ p_ssr_buffers.blur_radius[1] }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 2, u_output_blur, u_blur_radius2), 2);

			RD::Uniform u_scale_depth(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_ssr_buffers.depth_scaled }));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_scale_depth), 3);

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(ScreenSpaceReflectionFilterPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_screen_size.width, p_screen_size.height, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);

			// Vertical pass

			mode = SCREEN_SPACE_REFLECTION_FILTER_VERTICAL;
			shader = ssr_filter.shader.version_get_shader(ssr_filter.shader_version, mode);

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssr_filter.pipelines[pipeline_specialization][mode]);

			push_constant.vertical = 1;

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_output_blur, u_blur_radius2), 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_scale_normal), 1);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 2, u_intermediate), 2);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_scale_depth), 3);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 4, u_scene_data), 4);

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(ScreenSpaceReflectionFilterPushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_screen_size.width, p_screen_size.height, 1);

			if (v != p_view_count - 1) {
				RD::get_singleton()->compute_list_add_barrier(compute_list);
			}

			RD::get_singleton()->draw_command_end_label();
		}

		RD::get_singleton()->draw_command_end_label();
	}

	RD::get_singleton()->compute_list_end();
}

void SSEffects::ssr_free(SSRRenderBuffers &p_ssr_buffers) {
	for (uint32_t v = 0; v < RendererSceneRender::MAX_RENDER_VIEWS; v++) {
		p_ssr_buffers.output_slices[v] = RID();
	}

	if (p_ssr_buffers.output.is_valid()) {
		RD::get_singleton()->free(p_ssr_buffers.output);
		p_ssr_buffers.output = RID();
	}

	if (p_ssr_buffers.intermediate.is_valid()) {
		RD::get_singleton()->free(p_ssr_buffers.intermediate);
		p_ssr_buffers.intermediate = RID();
	}

	if (p_ssr_buffers.blur_radius[0].is_valid()) {
		RD::get_singleton()->free(p_ssr_buffers.blur_radius[0]);
		RD::get_singleton()->free(p_ssr_buffers.blur_radius[1]);
		p_ssr_buffers.blur_radius[0] = RID();
		p_ssr_buffers.blur_radius[1] = RID();
	}

	if (p_ssr_buffers.depth_scaled.is_valid()) {
		RD::get_singleton()->free(p_ssr_buffers.depth_scaled);
		p_ssr_buffers.depth_scaled = RID();
		RD::get_singleton()->free(p_ssr_buffers.normal_scaled);
		p_ssr_buffers.normal_scaled = RID();
	}
}

/* Subsurface scattering */

void SSEffects::sub_surface_scattering(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_diffuse, RID p_depth, const Projection &p_camera, const Size2i &p_screen_size, float p_scale, float p_depth_scale, RenderingServer::SubSurfaceScatteringQuality p_quality) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	// Our intermediate buffer is only created if we haven't created it already.
	RD::DataFormat format = p_render_buffers->get_base_data_format();
	uint32_t usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
	uint32_t layers = 1; // We only need one layer, we're handling one view at a time
	uint32_t mipmaps = 1; // Image::get_image_required_mipmaps(p_screen_size.x, p_screen_size.y, Image::FORMAT_RGBAH);
	RID intermediate = p_render_buffers->create_texture(SNAME("SSR"), SNAME("intermediate"), format, usage_bits, RD::TEXTURE_SAMPLES_1, p_screen_size, layers, mipmaps);

	Plane p = p_camera.xform4(Plane(1, 0, -1, 1));
	p.normal /= p.d;
	float unit_size = p.normal.x;

	{ //scale color and depth to half
		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

		sss.push_constant.camera_z_far = p_camera.get_z_far();
		sss.push_constant.camera_z_near = p_camera.get_z_near();
		sss.push_constant.orthogonal = p_camera.is_orthogonal();
		sss.push_constant.unit_size = unit_size;
		sss.push_constant.screen_size[0] = p_screen_size.x;
		sss.push_constant.screen_size[1] = p_screen_size.y;
		sss.push_constant.vertical = false;
		sss.push_constant.scale = p_scale;
		sss.push_constant.depth_scale = p_depth_scale;

		RID shader = sss.shader.version_get_shader(sss.shader_version, p_quality - 1);
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sss.pipelines[p_quality - 1]);

		RD::Uniform u_diffuse_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_diffuse }));
		RD::Uniform u_diffuse(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_diffuse }));
		RD::Uniform u_intermediate_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, intermediate }));
		RD::Uniform u_intermediate(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ intermediate }));
		RD::Uniform u_depth_with_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_depth }));

		// horizontal

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_diffuse_with_sampler), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_intermediate), 1);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 2, u_depth_with_sampler), 2);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &sss.push_constant, sizeof(SubSurfaceScatteringPushConstant));

		RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_screen_size.width, p_screen_size.height, 1);

		RD::get_singleton()->compute_list_add_barrier(compute_list);

		// vertical

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_intermediate_with_sampler), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_diffuse), 1);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 2, u_depth_with_sampler), 2);

		sss.push_constant.vertical = true;
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &sss.push_constant, sizeof(SubSurfaceScatteringPushConstant));

		RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_screen_size.width, p_screen_size.height, 1);

		RD::get_singleton()->compute_list_end();
	}
}
