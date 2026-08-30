/**************************************************************************/
/*  gi_hddagi_screen_probes.cpp                                           */
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

#include "gi.h"

#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/rendering_server_globals.h"

using namespace RendererRD;

void GI::disable_hddagi_screen_probes(Ref<RenderSceneBuffersRD> p_render_buffers) {
	if (p_render_buffers.is_null()) {
		return;
	}

	p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBES);
	if (p_render_buffers->has_custom_data(RB_SCOPE_GI)) {
		Ref<RenderBuffersGI> rbgi = p_render_buffers->get_custom_data(RB_SCOPE_GI);
		if (rbgi.is_valid() && rbgi->screen_probe_scene_data_ubo.is_valid()) {
			RD::get_singleton()->free_rid(rbgi->screen_probe_scene_data_ubo);
			rbgi->screen_probe_scene_data_ubo = RID();
		}
	}
}

void GI::process_hddagi_screen_probes(Ref<RenderSceneBuffersRD> p_render_buffers, const RID *p_normal_roughness_slices, const RID *p_hiz_slices, Size2i p_hiz_size, uint32_t p_hiz_mip_count, bool p_detail_trace, RID p_environment, uint32_t p_view_count, Size2i p_gi_size, const Projection *p_projections, const Transform3D &p_cam_transform, float p_exposure_normalization, float p_ibl_exposure_normalization, int p_probe_size, float p_normal_bias) {
	ERR_FAIL_COND(p_render_buffers.is_null());
	if (p_view_count == 0 || p_view_count > 2 || p_gi_size.x <= 0 || p_gi_size.y <= 0 || p_normal_roughness_slices == nullptr || p_projections == nullptr) {
		disable_hddagi_screen_probes(p_render_buffers);
		ERR_FAIL_MSG("HDDAGI screen probes require valid render data for one or two views.");
	}

	Ref<RenderBuffersGI> rbgi;
	if (p_render_buffers->has_custom_data(RB_SCOPE_GI)) {
		rbgi = p_render_buffers->get_custom_data(RB_SCOPE_GI);
	}
	Ref<HDDAGI> hddagi;
	if (p_render_buffers->has_custom_data(RB_SCOPE_HDDAGI)) {
		hddagi = p_render_buffers->get_custom_data(RB_SCOPE_HDDAGI);
	}
	if (rbgi.is_null() || hddagi.is_null() || hddagi->cascades.is_empty() || !hddagi_ubo.is_valid() ||
			!hddagi->voxel_bits_tex.is_valid() || !hddagi->voxel_region_tex.is_valid() || !hddagi->voxel_light_tex.is_valid() ||
			!hddagi->voxel_light_neighbour_data.is_valid() || !hddagi->voxel_disocclusion_tex.is_valid() || !hddagi->lightprobe_specular_tex.is_valid()) {
		disable_hddagi_screen_probes(p_render_buffers);
		return;
	}
	if (!hddagi_shader.screen_probe_available) {
		disable_hddagi_screen_probes(p_render_buffers);
		WARN_PRINT_ONCE("HDDAGI screen probes are unavailable because their compute shaders could not be created.");
		return;
	}
	if (!p_render_buffers->has_texture(RB_SCOPE_GI, RB_TEX_AMBIENT_U32)) {
		disable_hddagi_screen_probes(p_render_buffers);
		return;
	}

	const Size2i internal_size = p_render_buffers->get_internal_size();
	const int probe_size = CLAMP(p_probe_size, 1, 32);
	const Size2i probe_atlas_size((p_gi_size.x + probe_size - 1) / probe_size, (p_gi_size.y + probe_size - 1) / probe_size);
	const float normal_bias = CLAMP(p_normal_bias, -8.0f, 8.0f);
	const uint32_t candidate_count = uint32_t(CLAMP(GLOBAL_GET_CACHED(int, "rendering/global_illumination/hddagi/screen_probe_candidate_count"), 1, 8));
	const bool guided_sampling = GLOBAL_GET_CACHED(bool, "rendering/global_illumination/hddagi/screen_probe_guided_sampling");

	bool resources_valid = p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE) &&
			p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE) &&
			p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE);
	if (resources_valid) {
		const RD::TextureFormat surface_format = p_render_buffers->get_texture_format(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE);
		const RD::TextureFormat raw_format = p_render_buffers->get_texture_format(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE);
		const RD::TextureFormat resolved_format = p_render_buffers->get_texture_format(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE);
		resources_valid = surface_format.format == RD::DATA_FORMAT_R32G32B32A32_UINT && surface_format.width == uint32_t(probe_atlas_size.x) && surface_format.height == uint32_t(probe_atlas_size.y) && surface_format.array_layers == p_view_count &&
				raw_format.format == RD::DATA_FORMAT_R16G16B16A16_SFLOAT && raw_format.width == uint32_t(probe_atlas_size.x) && raw_format.height == uint32_t(probe_atlas_size.y) && raw_format.array_layers == p_view_count &&
				resolved_format.format == RD::DATA_FORMAT_R16G16B16A16_SFLOAT && resolved_format.width == uint32_t(p_gi_size.x) && resolved_format.height == uint32_t(p_gi_size.y) && resolved_format.array_layers == p_view_count &&
				p_render_buffers->get_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE).is_valid() &&
				p_render_buffers->get_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE).is_valid() &&
				p_render_buffers->get_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE).is_valid();
	}
	if (!resources_valid) {
		p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBES);
		const uint32_t storage_usage = RD::TEXTURE_USAGE_STORAGE_BIT;
		const uint32_t radiance_usage = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
		const RID surface = p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, RD::DATA_FORMAT_R32G32B32A32_UINT, storage_usage, RD::TEXTURE_SAMPLES_1, probe_atlas_size, p_view_count);
		const RID raw_radiance = p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, radiance_usage, RD::TEXTURE_SAMPLES_1, probe_atlas_size, p_view_count);
		const RID resolved_radiance = p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, radiance_usage, RD::TEXTURE_SAMPLES_1, p_gi_size, p_view_count);
		if (!surface.is_valid() || !raw_radiance.is_valid() || !resolved_radiance.is_valid()) {
			disable_hddagi_screen_probes(p_render_buffers);
			WARN_PRINT_ONCE("HDDAGI screen probe textures could not be allocated.");
			return;
		}
	}

	if (rbgi->screen_probe_scene_data_ubo.is_null()) {
		rbgi->screen_probe_scene_data_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(ScreenProbeSceneData));
	}
	if (rbgi->screen_probe_scene_data_ubo.is_null()) {
		disable_hddagi_screen_probes(p_render_buffers);
		WARN_PRINT_ONCE("The HDDAGI screen probe scene buffer could not be allocated.");
		return;
	}

	ScreenProbeSceneData scene_data = {};
	Projection correction;
	correction.set_depth_correction(true);
	for (uint32_t v = 0; v < p_view_count; v++) {
		const Projection projection = correction * p_projections[v];
		RendererRD::MaterialStorage::store_camera(projection.inverse(), scene_data.inv_projection[v]);
		RendererRD::MaterialStorage::store_camera(projection, scene_data.projection[v]);
	}
	RendererRD::MaterialStorage::store_transform(p_cam_transform, scene_data.cam_transform);
	Basis radiance_transform = p_cam_transform.basis;
	if (p_environment.is_valid()) {
		radiance_transform = RendererSceneRenderRD::get_singleton()->environment_get_sky_orientation(p_environment).inverse() * p_cam_transform.basis;
	}
	RendererRD::MaterialStorage::store_transform_3x3(radiance_transform, scene_data.radiance_inverse_xform);
	RD::get_singleton()->buffer_update(rbgi->screen_probe_scene_data_ubo, 0, sizeof(ScreenProbeSceneData), &scene_data);

	HDDAGIShader::ScreenProbePushConstant push_constant = {};
	push_constant.gi_size[0] = p_gi_size.x;
	push_constant.gi_size[1] = p_gi_size.y;
	push_constant.screen_size[0] = internal_size.x;
	push_constant.screen_size[1] = internal_size.y;
	push_constant.probe_size = probe_size;
	push_constant.frame_index = uint32_t(RSG::rasterizer->get_frame_number());
	push_constant.normal_bias = normal_bias;
	push_constant.candidate_count = candidate_count;
	push_constant.sky_mode = HDDAGIShader::SCREEN_PROBE_SKY_DISABLED;
	push_constant.flags = guided_sampling ? HDDAGIShader::ScreenProbePushConstant::FLAG_GUIDED_SAMPLING : 0u;

	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();
	RendererSceneRenderRD *scene_render = RendererSceneRenderRD::get_singleton();
	if (texture_storage == nullptr || material_storage == nullptr || scene_render == nullptr) {
		disable_hddagi_screen_probes(p_render_buffers);
		return;
	}

	RID sky_texture = texture_storage->texture_rd_get_default(sky && sky->sky_use_octmap_array ? RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_2D_ARRAY_BLACK : RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
	if (hddagi->reads_sky && p_environment.is_valid()) {
		const RSE::EnvironmentBG background = scene_render->environment_get_background(p_environment);
		if (background == RSE::ENV_BG_CLEAR_COLOR || background == RSE::ENV_BG_COLOR) {
			Color color = scene_render->environment_get_bg_color(p_environment);
			if (background == RSE::ENV_BG_CLEAR_COLOR) {
				color = RSG::texture_storage->get_default_clear_color().srgb_to_linear();
			}
			push_constant.sky_color[0] = color.r;
			push_constant.sky_color[1] = color.g;
			push_constant.sky_color[2] = color.b;
			push_constant.sky_energy = scene_render->environment_get_bg_energy_multiplier(p_environment) * scene_render->environment_get_bg_intensity(p_environment) * p_exposure_normalization;
			push_constant.sky_mode = HDDAGIShader::SCREEN_PROBE_SKY_COLOR;
		} else if (background == RSE::ENV_BG_SKY && sky) {
			const RID sky_rid = scene_render->environment_get_sky(p_environment);
			if (sky_rid.is_valid()) {
				const RID environment_radiance = sky->sky_get_radiance_texture_rd(sky_rid);
				if (environment_radiance.is_valid()) {
					sky_texture = environment_radiance;
					push_constant.sky_color[3] = sky->sky_get_uv_border_size(sky_rid);
					push_constant.sky_energy = scene_render->environment_get_bg_energy_multiplier(p_environment) * p_ibl_exposure_normalization;
					push_constant.sky_mode = HDDAGIShader::SCREEN_PROBE_SKY_TEXTURE;
				}
			}
		}
	}

	bool detail_trace = p_detail_trace && p_hiz_slices != nullptr && p_hiz_size == internal_size && p_hiz_mip_count > 0;
	for (uint32_t v = 0; v < p_view_count && detail_trace; v++) {
		detail_trace = p_hiz_slices[v].is_valid();
	}
	if (detail_trace) {
		push_constant.flags |= HDDAGIShader::ScreenProbePushConstant::FLAG_DETAIL_TRACE;
		push_constant.detail_trace_mip_count = p_hiz_mip_count;
	}

	const RID nearest_sampler = material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	const RID linear_sampler = material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	const RID linear_mipmap_sampler = material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	const RID detail_hiz_fallback = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
	if (!nearest_sampler.is_valid() || !linear_sampler.is_valid() || !linear_mipmap_sampler.is_valid() || !sky_texture.is_valid() || !detail_hiz_fallback.is_valid()) {
		disable_hddagi_screen_probes(p_render_buffers);
		return;
	}

	RID surface_sets[2];
	RID trace_sets[2];
	RID trace_sky_sets[2];
	RID resolve_sets[2];
	RID apply_sets[2];
	for (uint32_t v = 0; v < p_view_count; v++) {
		const RID depth = p_render_buffers->get_depth_texture(v);
		const RID normal_roughness = p_normal_roughness_slices[v];
		const RID surface = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, v, 0);
		const RID raw_radiance = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE, v, 0);
		const RID resolved_radiance = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE, v, 0);
		const RID ambient = p_render_buffers->get_texture_slice(RB_SCOPE_GI, RB_TEX_AMBIENT_U32, v, 0);
		if (!depth.is_valid() || !normal_roughness.is_valid() || !surface.is_valid() || !raw_radiance.is_valid() || !resolved_radiance.is_valid() || !ambient.is_valid()) {
			disable_hddagi_screen_probes(p_render_buffers);
			return;
		}

		surface_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
				hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_SURFACE], 0,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, depth),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, normal_roughness),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 2, nearest_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, surface));
		trace_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
				hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_TRACE], 0,
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, surface),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, raw_radiance),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, hddagi->voxel_bits_tex),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, hddagi->voxel_region_tex),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 4, hddagi->voxel_light_tex),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 5, linear_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, hddagi->voxel_light_neighbour_data),
				RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, hddagi_ubo),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 8, hddagi->voxel_disocclusion_tex),
				RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 9, rbgi->screen_probe_scene_data_ubo),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 10, detail_trace ? p_hiz_slices[v] : detail_hiz_fallback),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 11, normal_roughness));
		trace_sky_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
				hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_TRACE], 1,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, sky_texture),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, linear_mipmap_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, hddagi->lightprobe_specular_tex));
		resolve_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
				hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_RESOLVE], 0,
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, surface),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, raw_radiance),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, depth),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, normal_roughness),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, nearest_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, resolved_radiance),
				RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 6, rbgi->screen_probe_scene_data_ubo));
		apply_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
				hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_APPLY], 0,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, resolved_radiance),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, nearest_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, ambient));

		if (!RD::get_singleton()->uniform_set_is_valid(surface_sets[v]) || !RD::get_singleton()->uniform_set_is_valid(trace_sets[v]) ||
				!RD::get_singleton()->uniform_set_is_valid(trace_sky_sets[v]) || !RD::get_singleton()->uniform_set_is_valid(resolve_sets[v]) ||
				!RD::get_singleton()->uniform_set_is_valid(apply_sets[v])) {
			disable_hddagi_screen_probes(p_render_buffers);
			WARN_PRINT_ONCE("HDDAGI screen probe resources could not be bound.");
			return;
		}
	}

	RD::get_singleton()->draw_command_begin_label("HDDAGI Screen Probes");
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	for (uint32_t v = 0; v < p_view_count; v++) {
		push_constant.view_index = v;

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_SURFACE]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, surface_sets[v], 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, probe_atlas_size.x, probe_atlas_size.y, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_TRACE]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, trace_sets[v], 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, trace_sky_sets[v], 1);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, probe_atlas_size.x, probe_atlas_size.y, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_RESOLVE]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, resolve_sets[v], 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_gi_size.x, p_gi_size.y, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_APPLY]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, apply_sets[v], 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_gi_size.x, p_gi_size.y, 1);
	}
	RD::get_singleton()->compute_list_end();
	RD::get_singleton()->draw_command_end_label();
}
