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

#include "core/config/project_settings.h"
#include "core/templates/hashfuncs.h"
#include "core/templates/span.h"
#include "servers/rendering/renderer_rd/environment/gi.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/rendering_server_globals.h"

using namespace RendererRD;

static constexpr uint32_t SCREEN_PROBE_HISTORY_SEQUENCE_MASK = 0x00ffffffu;
static constexpr uint32_t SCREEN_PROBE_CONFIG_SEED = 0x44475203u;
static constexpr int SCREEN_PROBE_DIRECTIONAL_SIZE = 8;

namespace {

struct ScreenProbeTexture {
	StringName name;
	RD::DataFormat format;
	Size2i size;
	uint32_t layers;
};

bool screen_probe_textures_match(const Ref<RenderSceneBuffersRD> &p_buffers, const StringName &p_scope, Span<ScreenProbeTexture> p_textures) {
	for (const ScreenProbeTexture &texture : p_textures) {
		if (!p_buffers->has_texture(p_scope, texture.name)) {
			return false;
		}
		const RD::TextureFormat format = p_buffers->get_texture_format(p_scope, texture.name);
		if (format.format != texture.format || format.width != uint32_t(texture.size.x) || format.height != uint32_t(texture.size.y) || format.array_layers != texture.layers || !p_buffers->get_texture(p_scope, texture.name).is_valid()) {
			return false;
		}
	}
	return true;
}

bool create_screen_probe_textures(const Ref<RenderSceneBuffersRD> &p_buffers, const StringName &p_scope, Span<ScreenProbeTexture> p_textures) {
	for (const ScreenProbeTexture &texture : p_textures) {
		RID rid = p_buffers->create_texture(p_scope, texture.name, texture.format, RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT, RD::TEXTURE_SAMPLES_1, texture.size, texture.layers);
		if (rid.is_null()) {
			return false;
		}
	}
	return true;
}

void clear_screen_probe_svgf(const Ref<RenderSceneBuffersRD> &p_render_buffers, GI::RenderBuffersGI *p_gi) {
	p_gi->screen_probe_svgf.clear();
	p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER);
}

RID get_screen_probe_apply_uniform_set(RID p_shader, RID p_radiance, RID p_sampler, RID p_ambient_output, RID p_base_ambient) {
	return UniformSetCacheRD::get_singleton()->get_cache(
			p_shader, 0,
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, p_radiance),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, p_sampler),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, p_ambient_output),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, p_base_ambient));
}

} // namespace

void GI::disable_hddagi_screen_probes(Ref<RenderSceneBuffersRD> p_render_buffers) {
	if (p_render_buffers->has_custom_data(RB_SCOPE_GI)) {
		Ref<RenderBuffersGI> rbgi = p_render_buffers->get_custom_data(RB_SCOPE_GI);
		if (rbgi->screen_probe_scene_data_ubo.is_valid()) {
			RD::get_singleton()->free_rid(rbgi->screen_probe_scene_data_ubo);
			rbgi->screen_probe_scene_data_ubo = RID();
		}
		rbgi->screen_probe_svgf.clear();
	}
	p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER);
	p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBES);

	if (!p_render_buffers->has_custom_data(RB_SCOPE_HDDAGI)) {
		return;
	}
	Ref<HDDAGI> hddagi = p_render_buffers->get_custom_data(RB_SCOPE_HDDAGI);
	hddagi->screen_probe_history_initialized = false;
}

void GI::process_hddagi_screen_probes(const RenderDataRD *p_render_data, const RID *p_normal_roughness_slices, const RID *p_hiz_slices, Size2i p_hiz_size, uint32_t p_hiz_mip_count, bool p_detail_trace, Size2i p_gi_size, float p_exposure_normalization, float p_ibl_exposure_normalization, float p_normal_bias) {
	const Ref<RenderSceneBuffersRD> &render_buffers = p_render_data->render_buffers;
	const RenderSceneDataRD *camera = p_render_data->scene_data;
	const uint32_t view_count = camera->view_count;
	RD *rd = RD::get_singleton();
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	if (view_count == 0 || view_count > RendererSceneRender::MAX_RENDER_VIEWS || p_gi_size.x <= 0 || p_gi_size.y <= 0 || p_normal_roughness_slices == nullptr) {
		disable_hddagi_screen_probes(render_buffers);
		ERR_FAIL_MSG("HDDAGI screen probes require valid render data for one or two views.");
	}

	if (!render_buffers->has_custom_data(RB_SCOPE_GI) || !render_buffers->has_custom_data(RB_SCOPE_HDDAGI) || hddagi_ubo.is_null()) {
		disable_hddagi_screen_probes(render_buffers);
		return;
	}
	Ref<RenderBuffersGI> rbgi = render_buffers->get_custom_data(RB_SCOPE_GI);
	Ref<HDDAGI> hddagi = render_buffers->get_custom_data(RB_SCOPE_HDDAGI);
	if (hddagi->cascades.is_empty() || !hddagi->voxel_bits_tex.is_valid() || !hddagi->voxel_region_tex.is_valid() || !hddagi->voxel_data.is_valid() || !hddagi->voxel_page_table.is_valid()) {
		disable_hddagi_screen_probes(render_buffers);
		return;
	}
	if (!hddagi_shader.screen_probe_available) {
		disable_hddagi_screen_probes(render_buffers);
		WARN_PRINT_ONCE("HDDAGI screen probes are unavailable because their compute shaders could not be created.");
		return;
	}

	const RID *probe_shaders = hddagi_shader.screen_probe_shader_version;
	const RID *probe_pipelines = hddagi_shader.screen_probe_pipeline;

	if (!render_buffers->has_texture(RB_SCOPE_GI, RB_TEX_AMBIENT_U32)) {
		disable_hddagi_screen_probes(render_buffers);
		return;
	}

	const Size2i internal_size = render_buffers->get_internal_size();
	const bool uniform_placement_jitter = GLOBAL_GET_CACHED(bool, "rendering/global_illumination/hddagi/screen_probe_uniform_placement_jitter");
	const int probe_size = CLAMP(GLOBAL_GET_CACHED(int, "rendering/global_illumination/hddagi/screen_probe_spacing"), 16, 32);
	const Size2i probe_atlas_size((p_gi_size.x + probe_size - 1) / probe_size, (p_gi_size.y + probe_size - 1) / probe_size);
	const uint64_t maximum_texture_size = rd->limit_get(RD::LIMIT_MAX_TEXTURE_SIZE_2D);
	const uint32_t adaptive_capacity = uint32_t(probe_atlas_size.x * probe_atlas_size.y) / 2;
	const int adaptive_rows = (adaptive_capacity + probe_atlas_size.x - 1) / probe_atlas_size.x;
	const bool adaptive = GLOBAL_GET_CACHED(bool, "rendering/global_illumination/hddagi/screen_probe_directional_adaptive") &&
			adaptive_capacity > 0 && uint64_t(probe_atlas_size.y + adaptive_rows) * (SCREEN_PROBE_DIRECTIONAL_SIZE + 2) <= maximum_texture_size;
	const Size2i physical_probe_size(probe_atlas_size.x, probe_atlas_size.y + (adaptive ? adaptive_rows : 0));
	const Size2i directional_atlas_size = physical_probe_size * SCREEN_PROBE_DIRECTIONAL_SIZE;
	const Size2i irradiance_atlas_size = physical_probe_size * (SCREEN_PROBE_DIRECTIONAL_SIZE + 2);
	if (uint64_t(irradiance_atlas_size.x) > maximum_texture_size || uint64_t(irradiance_atlas_size.y) > maximum_texture_size) {
		disable_hddagi_screen_probes(render_buffers);
		WARN_PRINT_ONCE("HDDAGI Directional Gather exceeds the RenderingDevice texture size limit.");
		return;
	}
	const uint32_t candidate_count = CLAMP(GLOBAL_GET_CACHED(int, "rendering/global_illumination/hddagi/screen_probe_candidate_count"), 1, 8);
	const int trace_variant = (adaptive ? HDDAGIShader::DIRECTIONAL_TRACE_ADAPTIVE : 0) |
			(candidate_count > 1 ? HDDAGIShader::DIRECTIONAL_TRACE_MULTISAMPLE : 0);
	const int uniform_trace_mode = int(HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_TRACE) + trace_variant;
	const int importance_trace_mode = uniform_trace_mode + HDDAGIShader::DIRECTIONAL_TRACE_IMPORTANCE;
	const bool importance_requested = GLOBAL_GET_CACHED(bool, "rendering/global_illumination/hddagi/screen_probe_importance_sampling");
	const bool importance_sampling = importance_requested && hddagi->get_lightprobe_specular_texture().is_valid() && probe_pipelines[importance_trace_mode].is_valid();
	if (importance_requested && !importance_sampling) {
		WARN_PRINT_ONCE("HDDAGI importance sampling is unavailable; using uniform directional sampling.");
	}
	const int directional_trace_mode = importance_sampling ? importance_trace_mode : uniform_trace_mode;
	const Size2i resolve_size = internal_size;
	const uint32_t detail_trace_quality = uint32_t(CLAMP(GLOBAL_GET_CACHED(int, "rendering/global_illumination/hddagi/screen_probe_detail_trace_quality"), 0, 2));
	const uint32_t detail_trace_quality_flags = detail_trace_quality << HDDAGIShader::ScreenProbePushConstant::FLAG_DETAIL_TRACE_QUALITY_SHIFT;
	const HDDAGIScreenProbeSVGF::Quality svgf_quality = static_cast<HDDAGIScreenProbeSVGF::Quality>(CLAMP(GLOBAL_GET_CACHED(int, "rendering/global_illumination/hddagi/screen_probe_denoiser_quality"), 0, int(HDDAGIScreenProbeSVGF::QUALITY_MAX) - 1));

	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();
	RendererSceneRenderRD *scene_render = RendererSceneRenderRD::get_singleton();

	bool hiz_available = p_hiz_slices != nullptr && p_hiz_size == internal_size && p_hiz_mip_count > 0;
	for (uint32_t v = 0; v < view_count && hiz_available; v++) {
		hiz_available = p_hiz_slices[v].is_valid();
	}
	const bool detail_trace = p_detail_trace && hiz_available;
	const uint32_t detail_trace_mip_count = detail_trace ? CLAMP(p_hiz_mip_count, 1u, 16u) : 0u;
	uint32_t transport_configuration = hash_murmur3_one_64(p_render_data->environment.get_id());
	transport_configuration = hash_murmur3_one_32(hddagi->reads_sky ? 1u : 0u, transport_configuration);
	transport_configuration = hash_murmur3_one_32(importance_sampling ? 1u : 0u, transport_configuration);

	transport_configuration = hash_murmur3_one_float(hddagi->y_mult, transport_configuration);

	transport_configuration = hash_murmur3_one_float(p_exposure_normalization, transport_configuration);
	transport_configuration = hash_murmur3_one_float(p_ibl_exposure_normalization, transport_configuration);
	transport_configuration = hash_murmur3_one_32(hddagi->cascades.size(), transport_configuration);
	for (const HDDAGI::Cascade &cascade : hddagi->cascades) {
		transport_configuration = hash_murmur3_one_float(cascade.baked_exposure_normalization, transport_configuration);
	}
	transport_configuration = hash_murmur3_one_32(hddagi->version, transport_configuration);
	if (p_render_data->environment.is_valid()) {
		const RSE::EnvironmentBG background = scene_render->environment_get_background(p_render_data->environment);
		transport_configuration = hash_murmur3_one_32(uint32_t(background), transport_configuration);
		if (background == RSE::ENV_BG_SKY && sky) {
			const RID sky_rid = scene_render->environment_get_sky(p_render_data->environment);
			transport_configuration = hash_murmur3_one_64(sky_rid.get_id(), transport_configuration);
			if (sky_rid.is_valid()) {
				transport_configuration = hash_murmur3_one_64(sky->sky_get_radiance_texture_rd(sky_rid).get_id(), transport_configuration);
				transport_configuration = hash_murmur3_one_float(sky->sky_get_uv_border_size(sky_rid), transport_configuration);
			}
		}
	}
	transport_configuration = hash_fmix32(transport_configuration);

	Projection raster_correction;
	raster_correction.set_depth_correction(true);
	raster_correction.add_jitter_offset(camera->taa_jitter);
	Projection temporal_correction;
	temporal_correction.set_depth_correction(true);

	bool camera_cut = false;
	if (hddagi->screen_probe_history_initialized) {
		const float translation = hddagi->screen_probe_previous_cam_transform.origin.distance_to(camera->cam_transform.origin);
		const float translation_limit = MAX(4.0f, hddagi->min_cell_size * 16.0f);
		const float rotation = hddagi->screen_probe_previous_cam_transform.basis.get_rotation_quaternion().angle_to(camera->cam_transform.basis.get_rotation_quaternion());
		camera_cut = translation > translation_limit || rotation > Math::deg_to_rad(55.0f);
		for (uint32_t v = 0; v < view_count && !camera_cut; v++) {
			const Projection temporal_projection = temporal_correction * camera->view_projection[v];
			if (temporal_projection.is_orthogonal() != hddagi->screen_probe_previous_temporal_projection[v].is_orthogonal()) {
				camera_cut = true;
				break;
			}
			for (int column = 0; column < 4 && !camera_cut; column++) {
				for (int row = 0; row < 4; row++) {
					if (Math::abs(temporal_projection.columns[column][row] - hddagi->screen_probe_previous_temporal_projection[v].columns[column][row]) > 0.35f) {
						camera_cut = true;
						break;
					}
				}
			}
		}
	}

	uint32_t screen_probe_sky_mode = HDDAGIShader::SCREEN_PROBE_SKY_DISABLED;
	float screen_probe_sky_energy = 0.0f;
	float screen_probe_sky_color[4] = {};
	RID sky_texture = texture_storage->texture_rd_get_default(sky && sky->sky_use_octmap_array ? RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_2D_ARRAY_BLACK : RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
	if (hddagi->reads_sky && p_render_data->environment.is_valid()) {
		const RSE::EnvironmentBG background = scene_render->environment_get_background(p_render_data->environment);
		if (background == RSE::ENV_BG_CLEAR_COLOR || background == RSE::ENV_BG_COLOR) {
			const Color color = get_solid_background_radiance(p_render_data->environment, background);
			screen_probe_sky_color[0] = color.r;
			screen_probe_sky_color[1] = color.g;
			screen_probe_sky_color[2] = color.b;
			screen_probe_sky_energy = p_exposure_normalization;
			screen_probe_sky_mode = HDDAGIShader::SCREEN_PROBE_SKY_COLOR;
		} else if (background == RSE::ENV_BG_SKY && sky) {
			const RID sky_rid = scene_render->environment_get_sky(p_render_data->environment);
			if (sky_rid.is_valid()) {
				const RID environment_radiance = sky->sky_get_radiance_texture_rd(sky_rid);
				if (environment_radiance.is_valid()) {
					sky_texture = environment_radiance;
					screen_probe_sky_color[3] = sky->sky_get_uv_border_size(sky_rid);
					screen_probe_sky_energy = scene_render->environment_get_bg_energy_multiplier(p_render_data->environment) * p_ibl_exposure_normalization;
					screen_probe_sky_mode = HDDAGIShader::SCREEN_PROBE_SKY_TEXTURE;
				}
			}
		}
	}

	if (rbgi->screen_probe_scene_data_ubo.is_null()) {
		rbgi->screen_probe_scene_data_ubo = rd->uniform_buffer_create(sizeof(ScreenProbeSceneData));
	}
	if (rbgi->screen_probe_scene_data_ubo.is_null()) {
		disable_hddagi_screen_probes(render_buffers);
		WARN_PRINT_ONCE("The HDDAGI screen probe scene buffer could not be allocated.");
		return;
	}
	const RID nearest_sampler = material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	const RID linear_sampler = material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	const RID linear_mipmap_sampler = material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	const RID detail_hiz_fallback = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
	if (!nearest_sampler.is_valid() || !linear_sampler.is_valid() || !linear_mipmap_sampler.is_valid() || !sky_texture.is_valid() || !detail_hiz_fallback.is_valid()) {
		disable_hddagi_screen_probes(render_buffers);
		return;
	}

	uint32_t configuration = hash_murmur3_one_32(SCREEN_PROBE_CONFIG_SEED);

	configuration = hash_murmur3_one_32(probe_size, configuration);
	configuration = hash_murmur3_one_32(adaptive, configuration);
	configuration = hash_murmur3_one_32(candidate_count, configuration);
	configuration = hash_murmur3_one_32(uniform_placement_jitter ? 1u : 0u, configuration);
	configuration = hash_murmur3_one_32(detail_trace ? 1u : 0u, configuration);
	configuration = hash_murmur3_one_32(detail_trace_quality, configuration);
	configuration = hash_murmur3_one_32(transport_configuration, configuration);
	configuration = hash_murmur3_one_float(p_normal_bias, configuration);
	configuration = hash_fmix32(configuration);

	const uint32_t history_layers = view_count * 2u;
	const Size2i receiver_size = uniform_placement_jitter ? Size2i(probe_atlas_size.x * 2, probe_atlas_size.y * 4) : Size2i(1, 1);
	const ScreenProbeTexture directional_textures[] = {
		{ RB_TEX_HDDAGI_SCREEN_PROBE_RECEIVER_LUMINANCE, RD::DATA_FORMAT_R16G16_SFLOAT, receiver_size, view_count },
		{ RB_TEX_HDDAGI_SCREEN_PROBE_RECEIVER_GEOMETRY, RD::DATA_FORMAT_R32G32B32A32_UINT, receiver_size, view_count },
		{ RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, RD::DATA_FORMAT_R32G32B32A32_UINT, physical_probe_size, history_layers },
		{ RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, resolve_size, view_count },
		{ RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_RADIANCE, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, directional_atlas_size, history_layers },
		{ RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_HISTORY_AGE, RD::DATA_FORMAT_R8_UINT, physical_probe_size, history_layers },
		{ RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_LIGHTING_RESPONSE, RD::DATA_FORMAT_R8_UNORM, physical_probe_size, history_layers },
		{ RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_LIGHTING_STATISTICS, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, physical_probe_size, history_layers },
		{ RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_IRRADIANCE, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, irradiance_atlas_size, view_count },
	};
	const ScreenProbeTexture adaptive_textures[] = {
		{ RB_TEX_HDDAGI_SCREEN_PROBE_ADAPTIVE_MARK, RD::DATA_FORMAT_R8_UINT, probe_atlas_size, view_count },
		{ RB_TEX_HDDAGI_SCREEN_PROBE_ADAPTIVE_TILES, RD::DATA_FORMAT_R32G32_UINT, probe_atlas_size, history_layers },
		{ RB_TEX_HDDAGI_SCREEN_PROBE_ADAPTIVE_COUNTER, RD::DATA_FORMAT_R32_UINT, Size2i(1, 1), view_count },
	};
	const bool resources_recreated = !screen_probe_textures_match(render_buffers, RB_SCOPE_HDDAGI_SCREEN_PROBES, directional_textures) ||
			(adaptive && !screen_probe_textures_match(render_buffers, RB_SCOPE_HDDAGI_SCREEN_PROBES, adaptive_textures));
	if (resources_recreated) {
		render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBES);
		if (!create_screen_probe_textures(render_buffers, RB_SCOPE_HDDAGI_SCREEN_PROBES, directional_textures) ||
				(adaptive && !create_screen_probe_textures(render_buffers, RB_SCOPE_HDDAGI_SCREEN_PROBES, adaptive_textures))) {
			disable_hddagi_screen_probes(render_buffers);
			WARN_PRINT_ONCE("HDDAGI Directional Gather textures could not be allocated.");
			return;
		}
		const uint32_t texture_usage = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
		RD::TextureFormat directional_ambient_format;
		directional_ambient_format.format = RD::DATA_FORMAT_R32_UINT;
		directional_ambient_format.width = internal_size.x;
		directional_ambient_format.height = internal_size.y;
		directional_ambient_format.depth = 1;
		directional_ambient_format.array_layers = view_count;
		directional_ambient_format.texture_type = view_count > 1 ? RD::TEXTURE_TYPE_2D_ARRAY : RD::TEXTURE_TYPE_2D;
		directional_ambient_format.shareable_formats.push_back(RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32);
		directional_ambient_format.shareable_formats.push_back(RD::DATA_FORMAT_R32_UINT);
		directional_ambient_format.usage_bits = texture_usage;
		const RID directional_ambient_u32 = render_buffers->create_texture_from_format(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT_U32, directional_ambient_format);
		if (directional_ambient_u32.is_valid()) {
			RD::TextureView directional_ambient_view;
			directional_ambient_view.format_override = RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32;
			render_buffers->create_texture_view(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT_U32, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT, directional_ambient_view);
		}
	}

	RID view_velocity[RendererSceneRender::MAX_RENDER_VIEWS];
	bool all_views_have_velocity = render_buffers->has_velocity_buffer(false);
	if (all_views_have_velocity) {
		for (uint32_t v = 0; v < view_count; v++) {
			const RID velocity = render_buffers->get_velocity_buffer(false, v);
			if (rd->texture_is_valid(velocity) && rd->texture_size(velocity) == internal_size) {
				view_velocity[v] = velocity;
			} else {
				all_views_have_velocity = false;
			}
		}
	}

	const HDDAGIShader::ScreenProbeMode svgf_resolve_mode = adaptive ? HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_RESOLVE_SVGF_ADAPTIVE : HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_RESOLVE_SVGF;
	const HDDAGIShader::ScreenProbeMode svgf_apply_mode = HDDAGIShader::SCREEN_PROBE_MODE_APPLY_SVGF;
	bool svgf_active = all_views_have_velocity && probe_pipelines[svgf_resolve_mode].is_valid() && probe_pipelines[svgf_apply_mode].is_valid();
	bool svgf_resources_recreated = false;
	RID resolved_radiance[RendererSceneRender::MAX_RENDER_VIEWS];
	RID svgf_normal_roughness[RendererSceneRender::MAX_RENDER_VIEWS];
	RID svgf_view_z[RendererSceneRender::MAX_RENDER_VIEWS];
	RID svgf_output[RendererSceneRender::MAX_RENDER_VIEWS];
	if (!svgf_active) {
		clear_screen_probe_svgf(render_buffers, rbgi.ptr());
		WARN_PRINT_ONCE("HDDAGI screen-probe SVGF requires supported compute resources and valid forward-pass velocity textures; using the unfiltered signal.");
	} else {
		const ScreenProbeTexture svgf_textures[] = {
			{ RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_NORMAL_ROUGHNESS, RD::DATA_FORMAT_R8G8B8A8_UNORM, resolve_size, view_count },
			{ RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_VIEW_Z, RD::DATA_FORMAT_R32_SFLOAT, resolve_size, history_layers },
			{ RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_OUTPUT, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, resolve_size, view_count },
		};
		svgf_resources_recreated = !screen_probe_textures_match(render_buffers, RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, svgf_textures);
		if (svgf_resources_recreated) {
			clear_screen_probe_svgf(render_buffers, rbgi.ptr());
			svgf_active = create_screen_probe_textures(render_buffers, RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, svgf_textures);
		}
		for (uint32_t v = 0; v < view_count && svgf_active; v++) {
			svgf_normal_roughness[v] = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_NORMAL_ROUGHNESS, v, 0);
			svgf_output[v] = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_OUTPUT, v, 0);
			svgf_active = svgf_normal_roughness[v].is_valid() && svgf_output[v].is_valid();
		}
		if (!svgf_active) {
			clear_screen_probe_svgf(render_buffers, rbgi.ptr());
			WARN_PRINT_ONCE("HDDAGI screen-probe SVGF textures could not be allocated; using the unfiltered signal.");
		}
	}
	const bool history_configuration_valid = !resources_recreated &&
			hddagi->screen_probe_history_gi_size == p_gi_size && hddagi->screen_probe_history_screen_size == internal_size &&
			hddagi->screen_probe_history_view_count == view_count && hddagi->screen_probe_history_configuration == configuration;
	const bool camera_history_valid = hddagi->screen_probe_history_initialized && !camera_cut;
	const HDDAGIShader::ScreenProbeMode raw_apply_mode = HDDAGIShader::SCREEN_PROBE_MODE_APPLY;
	const bool history_valid = history_configuration_valid && camera_history_valid;
	const bool svgf_history_valid = svgf_active && history_valid && !svgf_resources_recreated;
	const uint32_t current_history_slot = history_valid ? (hddagi->screen_probe_history_slot ^ 1u) : 0u;
	const uint32_t previous_history_slot = current_history_slot ^ 1u;
	const uint32_t history_sequence = history_valid ? ((hddagi->screen_probe_history_sequence + 1u) & SCREEN_PROBE_HISTORY_SEQUENCE_MASK) : 0u;

	RENDER_TIMESTAMP("HDDAGI Directional Screen Probe Begin");
	ScreenProbeSceneData scene_data = {};
	const real_t receiver_radius = hddagi->cascades[0].cell_size * 4096.0;
	const bool reset_receivers = !history_valid || camera->cam_transform.origin.distance_squared_to(hddagi->screen_probe_receiver_origin) > receiver_radius * receiver_radius;
	if (reset_receivers) {
		hddagi->screen_probe_receiver_origin = camera->cam_transform.origin;
	}
	const Vector3 receiver_camera_offset = camera->cam_transform.origin - hddagi->screen_probe_receiver_origin;
	for (int i = 0; i < 3; i++) {
		scene_data.receiver_camera_offset[i] = receiver_camera_offset[i];
	}
	scene_data.receiver_camera_offset[3] = reset_receivers ? 1.0f : 0.0f;
	const Transform3D previous_cam_transform = camera_history_valid ? hddagi->screen_probe_previous_cam_transform : camera->cam_transform;
	const Transform3D previous_cam_inv_transform = previous_cam_transform.affine_inverse();
	const Vector2 previous_jitter = camera_history_valid ? hddagi->screen_probe_previous_taa_jitter : camera->taa_jitter;
	scene_data.taa_jitter_uv[0] = camera->taa_jitter.x * 0.5f;
	scene_data.taa_jitter_uv[1] = camera->taa_jitter.y * 0.5f;
	scene_data.taa_jitter_uv[2] = previous_jitter.x * 0.5f;
	scene_data.taa_jitter_uv[3] = previous_jitter.y * 0.5f;
	for (uint32_t v = 0; v < view_count; v++) {
		const Projection raster_projection = raster_correction * camera->view_projection[v];
		const Projection previous_raster_projection = camera_history_valid ? hddagi->screen_probe_previous_projection[v] : raster_projection;
		RendererRD::MaterialStorage::store_camera(raster_projection.inverse(), scene_data.inv_projection[v]);
		RendererRD::MaterialStorage::store_camera(raster_projection, scene_data.projection[v]);
		RendererRD::MaterialStorage::store_camera(previous_raster_projection.inverse(), scene_data.previous_inv_projection[v]);
		RendererRD::MaterialStorage::store_camera(previous_raster_projection, scene_data.previous_projection[v]);
	}
	RendererRD::MaterialStorage::store_transform(camera->cam_transform, scene_data.cam_transform);
	RendererRD::MaterialStorage::store_transform(previous_cam_inv_transform, scene_data.previous_cam_inv_transform);
	RendererRD::MaterialStorage::store_transform(previous_cam_transform, scene_data.previous_cam_transform);
	Basis radiance_transform = camera->cam_transform.basis;
	if (p_render_data->environment.is_valid()) {
		radiance_transform = scene_render->environment_get_sky_orientation(p_render_data->environment).inverse() * camera->cam_transform.basis;
	}
	RendererRD::MaterialStorage::store_transform_3x3(radiance_transform, scene_data.radiance_inverse_xform);
	RENDER_TIMESTAMP("HDDAGI Directional Screen Probe Scene Upload Begin");
	rd->buffer_update(rbgi->screen_probe_scene_data_ubo, 0, sizeof(ScreenProbeSceneData), &scene_data);
	RENDER_TIMESTAMP("HDDAGI Directional Screen Probe Scene Upload End");

	HDDAGIShader::ScreenProbePushConstant push_constant = {};
	push_constant.gi_size[0] = p_gi_size.x;
	push_constant.gi_size[1] = p_gi_size.y;
	push_constant.screen_size[0] = internal_size.x;
	push_constant.screen_size[1] = internal_size.y;
	push_constant.probe_size = probe_size;
	push_constant.candidate_count = candidate_count;
	push_constant.frame_index = history_sequence;
	push_constant.flags = (detail_trace ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_DETAIL_TRACE) : 0u) |
			(history_valid ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_DIRECTIONAL_HISTORY_VALID) : 0u) |
			(uniform_placement_jitter ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_UNIFORM_PLACEMENT_JITTER) : 0u) |
			detail_trace_quality_flags;
	push_constant.normal_bias = p_normal_bias;
	push_constant.sky_mode = screen_probe_sky_mode;
	push_constant.sky_energy = screen_probe_sky_energy;
	push_constant.detail_trace_mip_count = detail_trace_mip_count;
	for (uint32_t i = 0; i < 4; i++) {
		push_constant.sky_color[i] = screen_probe_sky_color[i];
	}

	RID surface_sets[RendererSceneRender::MAX_RENDER_VIEWS];
	RID adaptive_mark_sets[RendererSceneRender::MAX_RENDER_VIEWS];
	RID adaptive_spawn_sets[RendererSceneRender::MAX_RENDER_VIEWS];
	RID directional_trace_sets[RendererSceneRender::MAX_RENDER_VIEWS];
	RID directional_trace_sky_sets[RendererSceneRender::MAX_RENDER_VIEWS];
	RID directional_irradiance_sets[RendererSceneRender::MAX_RENDER_VIEWS];
	RID resolve_sets[RendererSceneRender::MAX_RENDER_VIEWS];
	RID svgf_resolve_sets[RendererSceneRender::MAX_RENDER_VIEWS];
	RID apply_sets[RendererSceneRender::MAX_RENDER_VIEWS];
	RID svgf_apply_sets[RendererSceneRender::MAX_RENDER_VIEWS];
	uint32_t view_flags[RendererSceneRender::MAX_RENDER_VIEWS] = {};
	const HDDAGIShader::ScreenProbeMode directional_irradiance_mode = HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_IRRADIANCE;
	const HDDAGIShader::ScreenProbeMode resolve_mode = adaptive ? HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_RESOLVE_ADAPTIVE : HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_RESOLVE;
	bool svgf_sets_valid = true;
	RD::TextureView packed_ambient_view;
	packed_ambient_view.format_override = RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32;
	for (uint32_t v = 0; v < view_count; v++) {
		const RID depth = render_buffers->get_depth_texture(v);
		const RID normal_roughness = p_normal_roughness_slices[v];
		const bool motion_valid = view_velocity[v].is_valid();
		const RID velocity = motion_valid ? view_velocity[v] : texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
		view_flags[v] = push_constant.flags | (motion_valid ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_DIRECTIONAL_MOTION_VALID) : 0u);
		const uint32_t surface_layer = v * 2u + current_history_slot;
		const uint32_t previous_surface_layer = v * 2u + previous_history_slot;
		const RID surface = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, surface_layer, 0);
		const RID previous_surface = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, previous_surface_layer, 0);
		resolved_radiance[v] = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE, v, 0);
		const RID base_ambient = render_buffers->get_texture_slice_view(RB_SCOPE_GI, RB_TEX_AMBIENT_U32, v, 0, 1, 1, packed_ambient_view);
		const RID ambient_output = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT_U32, v, 0);
		if (!depth.is_valid() || !normal_roughness.is_valid() || !surface.is_valid() || !previous_surface.is_valid() || !resolved_radiance[v].is_valid() || !base_ambient.is_valid() || !ambient_output.is_valid()) {
			disable_hddagi_screen_probes(render_buffers);
			return;
		}

		surface_sets[v] = uniform_set_cache->get_cache(
				probe_shaders[HDDAGIShader::SCREEN_PROBE_MODE_SURFACE], 0,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, depth),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, normal_roughness),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 2, nearest_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, surface));

		const uint32_t current_directional_layer = v * 2u + current_history_slot;
		const uint32_t previous_directional_layer = v * 2u + previous_history_slot;
		const RID directional_radiance = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_RADIANCE, current_directional_layer, 0);
		const RID previous_directional_radiance = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_RADIANCE, previous_directional_layer, 0);
		const RID directional_history_age = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_HISTORY_AGE, current_directional_layer, 0);
		const RID previous_directional_history_age = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_HISTORY_AGE, previous_directional_layer, 0);
		const RID directional_lighting_response = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_LIGHTING_RESPONSE, current_directional_layer, 0);
		const RID previous_directional_lighting_response = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_LIGHTING_RESPONSE, previous_directional_layer, 0);
		const RID directional_lighting_statistics = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_LIGHTING_STATISTICS, current_directional_layer, 0);
		const RID previous_directional_lighting_statistics = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_LIGHTING_STATISTICS, previous_directional_layer, 0);
		const RID receiver_luminance = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RECEIVER_LUMINANCE, v, 0);
		const RID receiver_geometry = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RECEIVER_GEOMETRY, v, 0);
		const RID directional_irradiance = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_IRRADIANCE, v, 0);
		if (!directional_radiance.is_valid() || !previous_directional_radiance.is_valid() || !directional_history_age.is_valid() || !previous_directional_history_age.is_valid() || !directional_lighting_response.is_valid() || !previous_directional_lighting_response.is_valid() || !directional_lighting_statistics.is_valid() || !previous_directional_lighting_statistics.is_valid() || !receiver_luminance.is_valid() || !receiver_geometry.is_valid() || !directional_irradiance.is_valid()) {
			disable_hddagi_screen_probes(render_buffers);
			return;
		}

		RID adaptive_tiles;
		RID previous_adaptive_tiles;
		if (adaptive) {
			const RID mark = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_ADAPTIVE_MARK, v, 0);
			const RID counter = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_ADAPTIVE_COUNTER, v, 0);
			adaptive_tiles = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_ADAPTIVE_TILES, surface_layer, 0);
			previous_adaptive_tiles = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_ADAPTIVE_TILES, previous_surface_layer, 0);
			LocalVector<RD::Uniform> placement_uniforms = {
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, depth),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, normal_roughness),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 2, nearest_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, surface),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, mark),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, counter),
				RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, rbgi->screen_probe_scene_data_ubo),
			};
			adaptive_mark_sets[v] = uniform_set_cache->get_cache_vec(probe_shaders[HDDAGIShader::SCREEN_PROBE_MODE_ADAPTIVE_MARK], 0, placement_uniforms);
			placement_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, adaptive_tiles));
			placement_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 8, previous_surface));
			placement_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 9, previous_adaptive_tiles));
			adaptive_spawn_sets[v] = uniform_set_cache->get_cache_vec(probe_shaders[HDDAGIShader::SCREEN_PROBE_MODE_ADAPTIVE_SPAWN], 0, placement_uniforms);
		}

		LocalVector<RD::Uniform> directional_trace_uniforms = {
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, surface),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, directional_radiance),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, hddagi->voxel_bits_tex),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, hddagi->voxel_region_tex),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, hddagi->voxel_data),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 5, linear_sampler),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, hddagi->voxel_page_table),
			RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, hddagi_ubo),
			RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 9, rbgi->screen_probe_scene_data_ubo),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 10, detail_trace ? p_hiz_slices[v] : detail_hiz_fallback),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 11, normal_roughness),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 12, previous_directional_radiance),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 13, previous_surface),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 14, previous_directional_history_age),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 15, directional_history_age),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 17, velocity),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 23, previous_directional_lighting_response),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 24, directional_lighting_response),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 25, previous_directional_lighting_statistics),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 26, directional_lighting_statistics),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 28, receiver_luminance),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 29, receiver_geometry),
		};
		if (adaptive) {
			directional_trace_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 16, previous_adaptive_tiles));
		}
		if (importance_sampling) {
			directional_trace_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 27, hddagi->get_lightprobe_specular_texture()));
		}
		directional_trace_sets[v] = uniform_set_cache->get_cache_vec(probe_shaders[directional_trace_mode], 0, directional_trace_uniforms);
		directional_trace_sky_sets[v] = uniform_set_cache->get_cache(
				probe_shaders[directional_trace_mode], 1,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, sky_texture),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, linear_mipmap_sampler));

		directional_irradiance_sets[v] = uniform_set_cache->get_cache(
				probe_shaders[directional_irradiance_mode], 0,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, directional_radiance),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, directional_irradiance),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 2, nearest_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, surface),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 4, velocity),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, directional_history_age),
				RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, rbgi->screen_probe_scene_data_ubo));

		LocalVector<RD::Uniform> resolve_uniforms = {
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, surface),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, directional_irradiance),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, depth),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, normal_roughness),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, linear_sampler),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, resolved_radiance[v]),
			RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, rbgi->screen_probe_scene_data_ubo),
		};

		resolve_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 8, base_ambient));
		resolve_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 18, hddagi->voxel_data));
		resolve_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 19, hddagi->voxel_page_table));
		resolve_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 20, hddagi_ubo));
		if (adaptive) {
			resolve_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 13, adaptive_tiles));
		}
		resolve_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 14, directional_lighting_response));
		resolve_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 15, receiver_luminance));
		resolve_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 16, receiver_geometry));

		resolve_sets[v] = uniform_set_cache->get_cache_vec(probe_shaders[resolve_mode], 0, resolve_uniforms);
		if (svgf_active) {
			svgf_view_z[v] = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_VIEW_Z, surface_layer, 0);
			resolve_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, svgf_normal_roughness[v]));
			resolve_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 11, svgf_view_z[v]));
			svgf_resolve_sets[v] = uniform_set_cache->get_cache_vec(probe_shaders[svgf_resolve_mode], 0, resolve_uniforms);
		}
		apply_sets[v] = get_screen_probe_apply_uniform_set(probe_shaders[raw_apply_mode], resolved_radiance[v], nearest_sampler, ambient_output, base_ambient);
		if (svgf_active) {
			svgf_apply_sets[v] = get_screen_probe_apply_uniform_set(probe_shaders[svgf_apply_mode], svgf_output[v], nearest_sampler, ambient_output, base_ambient);
		}

		bool sets_valid = rd->uniform_set_is_valid(surface_sets[v]) && rd->uniform_set_is_valid(resolve_sets[v]) && rd->uniform_set_is_valid(apply_sets[v]);

		if (svgf_active) {
			svgf_sets_valid = svgf_sets_valid && rd->uniform_set_is_valid(svgf_resolve_sets[v]) && rd->uniform_set_is_valid(svgf_apply_sets[v]);
		}

		if (adaptive) {
			sets_valid = sets_valid && rd->uniform_set_is_valid(adaptive_mark_sets[v]) && rd->uniform_set_is_valid(adaptive_spawn_sets[v]);
		}
		sets_valid = sets_valid && rd->uniform_set_is_valid(directional_trace_sets[v]) && rd->uniform_set_is_valid(directional_trace_sky_sets[v]) && rd->uniform_set_is_valid(directional_irradiance_sets[v]);

		if (!sets_valid) {
			disable_hddagi_screen_probes(render_buffers);
			WARN_PRINT_ONCE("HDDAGI screen probe resources could not be bound.");
			return;
		}
	}
	if (svgf_active && !svgf_sets_valid) {
		clear_screen_probe_svgf(render_buffers, rbgi.ptr());
		svgf_active = false;
		WARN_PRINT_ONCE("HDDAGI screen-probe SVGF resources could not be bound; using the unfiltered signal.");
	}

	rd->draw_command_begin_label("HDDAGI Screen Probes");
	RD::ComputeListID compute_list = rd->compute_list_begin();

	for (uint32_t v = 0; v < view_count; v++) {
		push_constant.flags = view_flags[v];
		push_constant.view_index = v;

		RENDER_TIMESTAMP("HDDAGI Directional Screen Probe Surface");
		rd->compute_list_bind_compute_pipeline(compute_list, probe_pipelines[HDDAGIShader::SCREEN_PROBE_MODE_SURFACE]);
		rd->compute_list_bind_uniform_set(compute_list, surface_sets[v], 0);
		rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		rd->compute_list_dispatch(compute_list, probe_atlas_size.x, probe_atlas_size.y, 1);
		rd->compute_list_add_barrier(compute_list);

		if (adaptive) {
			RENDER_TIMESTAMP("HDDAGI Directional Adaptive Probe Mark");
			rd->compute_list_bind_compute_pipeline(compute_list, probe_pipelines[HDDAGIShader::SCREEN_PROBE_MODE_ADAPTIVE_MARK]);
			rd->compute_list_bind_uniform_set(compute_list, adaptive_mark_sets[v], 0);
			rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			rd->compute_list_dispatch(compute_list, probe_atlas_size.x, probe_atlas_size.y, 1);
			rd->compute_list_add_barrier(compute_list);
			RENDER_TIMESTAMP("HDDAGI Directional Adaptive Probe Spawn");
			rd->compute_list_bind_compute_pipeline(compute_list, probe_pipelines[HDDAGIShader::SCREEN_PROBE_MODE_ADAPTIVE_SPAWN]);
			rd->compute_list_bind_uniform_set(compute_list, adaptive_spawn_sets[v], 0);
			rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			rd->compute_list_dispatch(compute_list, probe_atlas_size.x, probe_atlas_size.y, 1);
			rd->compute_list_add_barrier(compute_list);
		}

		RENDER_TIMESTAMP("HDDAGI Directional Screen Probe Trace");
		rd->compute_list_bind_compute_pipeline(compute_list, probe_pipelines[directional_trace_mode]);

		rd->compute_list_bind_uniform_set(compute_list, directional_trace_sets[v], 0);
		rd->compute_list_bind_uniform_set(compute_list, directional_trace_sky_sets[v], 1);

		rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		rd->compute_list_dispatch_threads(compute_list, directional_atlas_size.x, directional_atlas_size.y, 1);
		rd->compute_list_add_barrier(compute_list);

		RENDER_TIMESTAMP("HDDAGI Directional Screen Probe Irradiance");
		rd->compute_list_bind_compute_pipeline(compute_list, probe_pipelines[directional_irradiance_mode]);
		rd->compute_list_bind_uniform_set(compute_list, directional_irradiance_sets[v], 0);
		rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		rd->compute_list_dispatch_threads(compute_list, directional_atlas_size.x, directional_atlas_size.y, 1);
		rd->compute_list_add_barrier(compute_list);

		RENDER_TIMESTAMP("HDDAGI Directional Screen Probe Resolve");
		const HDDAGIShader::ScreenProbeMode active_resolve_mode = svgf_active ? svgf_resolve_mode : resolve_mode;
		rd->compute_list_bind_compute_pipeline(compute_list, probe_pipelines[active_resolve_mode]);

		rd->compute_list_bind_uniform_set(compute_list, svgf_active ? svgf_resolve_sets[v] : resolve_sets[v], 0);

		HDDAGIShader::ScreenProbePushConstant resolve_push_constant = push_constant;

		resolve_push_constant.history_depth_tolerance = 0.02f;
		resolve_push_constant.spatial_depth_tolerance_scale = 0.02f;
		resolve_push_constant.history_normal_threshold = 0.35f;

		rd->compute_list_set_push_constant(compute_list, &resolve_push_constant, sizeof(resolve_push_constant));
		rd->compute_list_dispatch_threads(compute_list, resolve_size.x, resolve_size.y, 1);
		rd->compute_list_add_barrier(compute_list);

		if (!svgf_active) {
			RENDER_TIMESTAMP("HDDAGI Screen Probe Apply");

			rd->compute_list_bind_compute_pipeline(compute_list, probe_pipelines[raw_apply_mode]);
			rd->compute_list_bind_uniform_set(compute_list, apply_sets[v], 0);
			rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			rd->compute_list_dispatch_threads(compute_list, resolve_size.x, resolve_size.y, 1);
			if (RSG::utilities->capturing_timestamps) {
				rd->compute_list_add_barrier(compute_list);
				RENDER_TIMESTAMP("HDDAGI Screen Probe Apply End");
			}
		}
	}
	rd->compute_list_end();
	if (svgf_active) {
		bool svgf_succeeded[RendererSceneRender::MAX_RENDER_VIEWS] = {};
		bool svgf_failed = false;
		for (uint32_t v = 0; v < view_count; v++) {
			HDDAGIScreenProbeSVGF::FrameSettings svgf_frame;
			svgf_frame.projection = camera->view_projection[v];
			svgf_frame.previous_projection = svgf_history_valid ? hddagi->screen_probe_previous_svgf_projection[v] : camera->view_projection[v];
			svgf_frame.camera_transform = camera->cam_transform;
			svgf_frame.previous_camera_transform = svgf_history_valid ? hddagi->screen_probe_previous_cam_transform : camera->cam_transform;
			svgf_frame.taa_jitter = camera->taa_jitter;
			svgf_frame.previous_taa_jitter = svgf_history_valid ? hddagi->screen_probe_previous_taa_jitter : camera->taa_jitter;
			svgf_frame.size = resolve_size;
			svgf_frame.quality = svgf_quality;
			svgf_frame.history_valid = svgf_history_valid;

			HDDAGIScreenProbeSVGF::Resources svgf_resources;
			svgf_resources.motion_vectors = view_velocity[v];
			svgf_resources.normal_roughness = svgf_normal_roughness[v];
			svgf_resources.view_z = svgf_view_z[v];
			svgf_resources.previous_view_z = render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_VIEW_Z, v * 2u + previous_history_slot, 0);
			svgf_resources.radiance_hit_distance = resolved_radiance[v];
			svgf_resources.output_radiance_hit_distance = svgf_output[v];
			const Error denoise_error = rbgi->screen_probe_svgf.denoise(v, svgf_frame, svgf_resources);
			svgf_succeeded[v] = denoise_error == OK;
			if (denoise_error != OK) {
				svgf_failed = true;
				WARN_PRINT_ONCE(vformat("HDDAGI screen-probe SVGF failed with error %d; using the unfiltered signal.", denoise_error));
			}
		}
		if (svgf_failed) {
			rbgi->screen_probe_svgf.clear();
		}

		compute_list = rd->compute_list_begin();
		for (uint32_t v = 0; v < view_count; v++) {
			push_constant.view_index = v;
			const HDDAGIShader::ScreenProbeMode apply_mode = svgf_succeeded[v] ? svgf_apply_mode : raw_apply_mode;
			RENDER_TIMESTAMP("HDDAGI Screen Probe Apply");

			rd->compute_list_bind_compute_pipeline(compute_list, probe_pipelines[apply_mode]);
			rd->compute_list_bind_uniform_set(compute_list, svgf_succeeded[v] ? svgf_apply_sets[v] : apply_sets[v], 0);
			rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			rd->compute_list_dispatch_threads(compute_list, resolve_size.x, resolve_size.y, 1);
			if (RSG::utilities->capturing_timestamps) {
				rd->compute_list_add_barrier(compute_list);
				RENDER_TIMESTAMP("HDDAGI Screen Probe Apply End");
			}
		}
		rd->compute_list_end();
	}

	rd->draw_command_end_label();
	RENDER_TIMESTAMP("HDDAGI Directional Screen Probe End");

	hddagi->screen_probe_history_initialized = true;
	hddagi->screen_probe_history_gi_size = p_gi_size;
	hddagi->screen_probe_history_screen_size = internal_size;
	hddagi->screen_probe_history_view_count = view_count;
	hddagi->screen_probe_history_configuration = configuration;
	hddagi->screen_probe_history_slot = current_history_slot;
	hddagi->screen_probe_history_sequence = history_sequence;
	for (uint32_t v = 0; v < view_count; v++) {
		hddagi->screen_probe_previous_projection[v] = raster_correction * camera->view_projection[v];
		hddagi->screen_probe_previous_temporal_projection[v] = temporal_correction * camera->view_projection[v];
		hddagi->screen_probe_previous_svgf_projection[v] = camera->view_projection[v];
	}
	hddagi->screen_probe_previous_taa_jitter = camera->taa_jitter;
	hddagi->screen_probe_previous_cam_transform = camera->cam_transform;
}
