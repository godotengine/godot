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
#include "servers/rendering/renderer_rd/environment/gi.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/rendering_server_globals.h"

using namespace RendererRD;

static constexpr uint32_t SCREEN_PROBE_HISTORY_SEQUENCE_MASK = 0x00ffffffu;
static constexpr float SCREEN_PROBE_IRRADIANCE_CACHE_MINIMUM_CELL_SIZE = 0.001f;
static constexpr float SCREEN_PROBE_SVGF_DENOISING_RANGE = 500000.0f;
static constexpr float SCREEN_PROBE_SVGF_SCENE_TO_SIGNAL_SCALE = 1.0f / 512.0f;
static constexpr float SCREEN_PROBE_SVGF_INPUT_RADIANCE_MAX = 128.0f;
static constexpr float SCREEN_PROBE_SVGF_INPUT_HIT_DISTANCE_MAX = 65504.0f;
static constexpr int SCREEN_PROBE_DIRECTIONAL_SIZE = 8;
static constexpr uint32_t SCREEN_PROBE_DIRECTIONAL_FILTER_PASS_COUNT = 3u;
static constexpr uint32_t SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE = 8u;
static constexpr uint32_t SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT = 8u;
static constexpr uint32_t SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE = 8u;
static constexpr uint32_t SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CAPACITY_DIVISOR = 2u;
static constexpr uint32_t SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_WORKGROUP_SIZE = 8u;
static constexpr float SCREEN_PROBE_SPECULAR_FULL_AUTHORITY_ROUGHNESS = 0.30f;
static constexpr float SCREEN_PROBE_SPECULAR_FALLBACK_ROUGHNESS = 0.40f;
static constexpr float SCREEN_PROBE_SPECULAR_RADIANCE_MAX = 40.0f;
static constexpr float SCREEN_PROBE_SPECULAR_MIN_GGX_ALPHA = 0.001f;
static constexpr float SCREEN_PROBE_SPECULAR_DENOISING_RANGE = 65504.0f;

void GI::disable_hddagi_screen_probes(Ref<RenderSceneBuffersRD> p_render_buffers) {
	if (p_render_buffers.is_null()) {
		return;
	}

	Ref<RenderBuffersGI> rbgi;
	if (p_render_buffers->has_custom_data(RB_SCOPE_GI)) {
		rbgi = p_render_buffers->get_custom_data(RB_SCOPE_GI);
		if (rbgi.is_valid()) {
			rbgi->hddagi_specular_reflection_valid = false;
			rbgi->screen_probe_debug_montage_valid = false;
			rbgi->screen_probe_debug_svgf_output_valid = false;
			rbgi->screen_probe_debug_hiz_valid = false;
			rbgi->screen_probe_debug_directional_valid = false;
			rbgi->screen_probe_debug_radiance_scale = 1.0f;
			rbgi->screen_probe_debug_surface_layer_stride = 1;
			rbgi->screen_probe_debug_surface_history_slot = 0;
			rbgi->screen_probe_debug_hiz_mip_count = 0;
			if (rbgi->screen_probe_scene_data_ubo.is_valid()) {
				RD::get_singleton()->free_rid(rbgi->screen_probe_scene_data_ubo);
				rbgi->screen_probe_scene_data_ubo = RID();
			}
			rbgi->screen_probe_svgf.clear();
			rbgi->screen_probe_specular_svgf.clear();
			rbgi->screen_probe_irradiance_cache.clear();
		}
	}
	p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER);
	p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS);
	p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBE_DEBUG);
	p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBES);

	if (!p_render_buffers->has_custom_data(RB_SCOPE_HDDAGI)) {
		return;
	}
	Ref<HDDAGI> hddagi = p_render_buffers->get_custom_data(RB_SCOPE_HDDAGI);
	if (hddagi.is_null()) {
		return;
	}

	hddagi->screen_probe_history_initialized = false;
	hddagi->screen_probe_history_probe_size = 0;
	hddagi->screen_probe_history_gi_size = Size2i();
	hddagi->screen_probe_history_screen_size = Size2i();
	hddagi->screen_probe_history_view_count = 0;
	hddagi->screen_probe_history_configuration = 0;
	hddagi->screen_probe_history_slot = 0;
	hddagi->screen_probe_history_sequence = 0;
	hddagi->screen_probe_directional_allocation_failed = false;
	hddagi->screen_probe_previous_camera_valid = false;
	hddagi->screen_probe_previous_exposure_normalization = 1.0f;
}

void GI::process_hddagi_screen_probes(Ref<RenderSceneBuffersRD> p_render_buffers, const RID *p_normal_roughness_slices, const RID *p_hiz_slices, const RID *p_previous_screen_color_slices, const RID *p_previous_screen_depth_slices, bool p_previous_screen_color_valid, Size2i p_hiz_size, uint32_t p_hiz_mip_count, bool p_detail_trace, RID p_environment, uint32_t p_view_count, Size2i p_gi_size, const Projection *p_projections, const Vector3 *p_eye_offsets, const Vector2 &p_taa_jitter, const Transform3D &p_cam_transform, float p_exposure_normalization, float p_ibl_exposure_normalization, int p_probe_size, float p_normal_bias, RSE::EnvironmentHDDAGIScreenProbeMode p_mode, bool p_debug_montage) {
	ERR_FAIL_COND(p_render_buffers.is_null());
	ERR_FAIL_INDEX(int(p_mode), int(RSE::ENV_HDDAGI_SCREEN_PROBE_MODE_MAX));
	if (p_view_count == 0 || p_view_count > 2 || p_gi_size.x <= 0 || p_gi_size.y <= 0 || p_normal_roughness_slices == nullptr || p_projections == nullptr || p_eye_offsets == nullptr) {
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
			!hddagi->voxel_light_neighbour_data.is_valid() || !hddagi->voxel_disocclusion_tex.is_valid() ||
			!hddagi->lightprobe_specular_tex.is_valid()) {
		disable_hddagi_screen_probes(p_render_buffers);
		return;
	}
	rbgi->screen_probe_debug_montage_valid = false;
	rbgi->screen_probe_debug_svgf_output_valid = false;
	rbgi->screen_probe_debug_hiz_valid = false;
	rbgi->screen_probe_debug_directional_valid = false;
	rbgi->screen_probe_debug_radiance_scale = 1.0f;
	rbgi->screen_probe_debug_surface_layer_stride = 1;
	rbgi->screen_probe_debug_surface_history_slot = 0;
	rbgi->screen_probe_debug_hiz_mip_count = 0;
	if (!hddagi_shader.screen_probe_available) {
		disable_hddagi_screen_probes(p_render_buffers);
		WARN_PRINT_ONCE("HDDAGI screen probes are unavailable because their compute shaders could not be created.");
		return;
	}
	if (!p_render_buffers->has_texture(RB_SCOPE_GI, RB_TEX_AMBIENT_U32)) {
		disable_hddagi_screen_probes(p_render_buffers);
		return;
	}
	rbgi->hddagi_specular_reflection_valid = false;

	const Size2i internal_size = p_render_buffers->get_internal_size();
	if (hddagi->screen_probe_requested_mode != p_mode) {
		hddagi->screen_probe_requested_mode = p_mode;
		hddagi->screen_probe_directional_allocation_failed = false;
	}
	const bool directional_requested = p_mode == RSE::ENV_HDDAGI_SCREEN_PROBE_MODE_DIRECTIONAL_GATHER;
	const int stochastic_probe_size = CLAMP(p_probe_size, 1, 32);
	const float gi_resolution_scale = 0.5f * (float(p_gi_size.x) / float(MAX(internal_size.x, 1)) + float(p_gi_size.y) / float(MAX(internal_size.y, 1)));
	const int directional_probe_size = CLAMP(int(Math::round(16.0f * gi_resolution_scale)), 1, 32);
	int probe_size = directional_requested ? directional_probe_size : stochastic_probe_size;
	Size2i probe_atlas_size((p_gi_size.x + probe_size - 1) / probe_size, (p_gi_size.y + probe_size - 1) / probe_size);
	auto get_directional_probe_atlas_size = [](const Size2i &p_base_size) {
		const uint64_t base_probe_count = uint64_t(p_base_size.x) * uint64_t(p_base_size.y);
		const uint64_t adaptive_capacity = base_probe_count / SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CAPACITY_DIVISOR;
		const uint64_t adaptive_rows = adaptive_capacity > 0u ? (adaptive_capacity + uint64_t(p_base_size.x) - 1u) / uint64_t(p_base_size.x) : 0u;
		return Size2i(p_base_size.x, p_base_size.y + int(adaptive_rows));
	};
	Size2i directional_probe_atlas_size = get_directional_probe_atlas_size(probe_atlas_size);
	Size2i directional_atlas_size(directional_probe_atlas_size.x * SCREEN_PROBE_DIRECTIONAL_SIZE, directional_probe_atlas_size.y * SCREEN_PROBE_DIRECTIONAL_SIZE);
	const uint64_t maximum_texture_size = RD::get_singleton()->limit_get(RD::LIMIT_MAX_TEXTURE_SIZE_2D);
	bool directional_gather = directional_requested && !hddagi->screen_probe_directional_allocation_failed &&
			uint64_t(directional_probe_atlas_size.x) <= maximum_texture_size && uint64_t(directional_probe_atlas_size.y) <= maximum_texture_size &&
			uint64_t(directional_atlas_size.x) <= maximum_texture_size && uint64_t(directional_atlas_size.y) <= maximum_texture_size;
	if (directional_requested && !hddagi->screen_probe_directional_allocation_failed && !directional_gather) {
		WARN_PRINT_ONCE(vformat("HDDAGI Directional Gather requires a %dx%d direction atlas, exceeding this RenderingDevice's %d texel 2D limit; using Stochastic Integrated mode.", directional_atlas_size.x, directional_atlas_size.y, maximum_texture_size));
	}
	if (directional_gather) {
		auto screen_probe_mode_is_ready = [&](HDDAGIShader::ScreenProbeMode p_shader_mode) {
			return hddagi_shader.screen_probe_shader_version[p_shader_mode].is_valid() && hddagi_shader.screen_probe_pipeline[p_shader_mode].is_valid();
		};
		directional_gather = screen_probe_mode_is_ready(HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_ADAPTIVE_MARK) &&
				screen_probe_mode_is_ready(HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_ADAPTIVE_SPAWN) &&
				screen_probe_mode_is_ready(HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_TRACE) &&
				screen_probe_mode_is_ready(HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_FILTER) &&
				screen_probe_mode_is_ready(HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_IRRADIANCE) &&
				screen_probe_mode_is_ready(HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_RESOLVE);
		if (!directional_gather) {
			WARN_PRINT_ONCE("One or more HDDAGI Directional Gather shader variants failed to initialize; using Stochastic Integrated mode.");
		}
	}
	if (directional_requested && !directional_gather) {
		probe_size = stochastic_probe_size;
		probe_atlas_size = Size2i((p_gi_size.x + probe_size - 1) / probe_size, (p_gi_size.y + probe_size - 1) / probe_size);
		directional_probe_atlas_size = get_directional_probe_atlas_size(probe_atlas_size);
		directional_atlas_size = Size2i(directional_probe_atlas_size.x * SCREEN_PROBE_DIRECTIONAL_SIZE, directional_probe_atlas_size.y * SCREEN_PROBE_DIRECTIONAL_SIZE);
	}
	const Size2i resolve_size = directional_gather ? internal_size : p_gi_size;
	const float normal_bias = CLAMP(p_normal_bias, -8.0f, 8.0f);
	const uint32_t candidate_count = directional_gather ? 1u : uint32_t(CLAMP(GLOBAL_GET_CACHED(int, "rendering/global_illumination/hddagi/screen_probe_candidate_count"), 1, 8));
	const bool guided_sampling = GLOBAL_GET_CACHED(bool, "rendering/global_illumination/hddagi/screen_probe_guided_sampling");
	const int irradiance_cache_setting = directional_gather ? 0 : GLOBAL_GET_CACHED(int, "rendering/global_illumination/hddagi/screen_probe_radiance_cache");
	const float irradiance_cache_minimum_cell_size = MAX(GLOBAL_GET_CACHED(float, "rendering/global_illumination/hddagi/screen_probe_radiance_cache_minimum_cell_size"), SCREEN_PROBE_IRRADIANCE_CACHE_MINIMUM_CELL_SIZE);
	const bool irradiance_cache_multibounce = !directional_gather && hddagi->screen_probe_radiance_cache_multibounce_active;
	const int denoiser_setting = GLOBAL_GET_CACHED(int, "rendering/global_illumination/hddagi/screen_probe_denoiser");
	const HDDAGIScreenProbeSVGF::Quality svgf_quality = static_cast<HDDAGIScreenProbeSVGF::Quality>(CLAMP(GLOBAL_GET_CACHED(int, "rendering/global_illumination/hddagi/screen_probe_denoiser_quality"), 0, int(HDDAGIScreenProbeSVGF::QUALITY_MAX) - 1));

	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();
	RendererSceneRenderRD *scene_render = RendererSceneRenderRD::get_singleton();
	if (texture_storage == nullptr || material_storage == nullptr || scene_render == nullptr) {
		disable_hddagi_screen_probes(p_render_buffers);
		return;
	}

	bool hiz_available = p_hiz_slices != nullptr && p_hiz_size == internal_size && p_hiz_mip_count > 0;
	for (uint32_t v = 0; v < p_view_count && hiz_available; v++) {
		hiz_available = p_hiz_slices[v].is_valid();
	}
	const bool detail_trace = p_detail_trace && hiz_available;
	const uint32_t detail_trace_mip_count = detail_trace ? CLAMP(p_hiz_mip_count, 1u, 16u) : 0u;
	bool specular_reflections = GLOBAL_GET_CACHED(bool, "rendering/global_illumination/hddagi/screen_probe_specular_reflections") &&
			p_render_buffers->has_texture(RB_SCOPE_GI, RB_TEX_REFLECTION_U32);
	if (specular_reflections &&
			(!hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_SPECULAR_TRACE].is_valid() || !hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_SPECULAR_TRACE].is_valid() ||
					!hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_SPECULAR_APPLY].is_valid() || !hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_SPECULAR_APPLY].is_valid())) {
		specular_reflections = false;
		WARN_PRINT_ONCE("HDDAGI specular reflections are unavailable because required resources could not be created.");
	}
	const bool specular_detail_trace = specular_reflections && hiz_available;
	bool specular_screen_radiance_available = specular_detail_trace && p_previous_screen_color_valid && p_previous_screen_color_slices != nullptr && p_previous_screen_depth_slices != nullptr;
	for (uint32_t v = 0; v < p_view_count && specular_screen_radiance_available; v++) {
		specular_screen_radiance_available = p_previous_screen_color_slices[v].is_valid() && p_previous_screen_depth_slices[v].is_valid() &&
				RD::get_singleton()->texture_size(p_previous_screen_color_slices[v]) == internal_size && RD::get_singleton()->texture_size(p_previous_screen_depth_slices[v]) == internal_size;
	}
	if (!specular_reflections) {
		p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS);
		rbgi->screen_probe_specular_svgf.clear();
	}

	uint32_t transport_configuration = hash_murmur3_one_64(p_environment.get_id());
	transport_configuration = hash_murmur3_one_32(hddagi->reads_sky ? 1u : 0u, transport_configuration);
	transport_configuration = hash_murmur3_one_float(hddagi->energy, transport_configuration);
	transport_configuration = hash_murmur3_one_float(hddagi->y_mult, transport_configuration);
	transport_configuration = hash_murmur3_one_float(p_exposure_normalization, transport_configuration);
	transport_configuration = hash_murmur3_one_float(p_ibl_exposure_normalization, transport_configuration);
	transport_configuration = hash_murmur3_one_32(hddagi->cascades.size(), transport_configuration);
	for (const HDDAGI::Cascade &cascade : hddagi->cascades) {
		transport_configuration = hash_murmur3_one_float(cascade.baked_exposure_normalization, transport_configuration);
	}
	transport_configuration = hash_murmur3_one_32(hddagi->version, transport_configuration);
	if (p_environment.is_valid()) {
		const RSE::EnvironmentBG background = scene_render->environment_get_background(p_environment);
		transport_configuration = hash_murmur3_one_32(uint32_t(background), transport_configuration);
		transport_configuration = hash_murmur3_one_float(scene_render->environment_get_bg_energy_multiplier(p_environment), transport_configuration);
		transport_configuration = hash_murmur3_one_float(scene_render->environment_get_bg_intensity(p_environment), transport_configuration);
		const Basis sky_orientation = scene_render->environment_get_sky_orientation(p_environment);
		for (int row = 0; row < 3; row++) {
			for (int column = 0; column < 3; column++) {
				transport_configuration = hash_murmur3_one_float(sky_orientation[row][column], transport_configuration);
			}
		}
		if (background == RSE::ENV_BG_CLEAR_COLOR || background == RSE::ENV_BG_COLOR) {
			const Color color = background == RSE::ENV_BG_CLEAR_COLOR ? RSG::texture_storage->get_default_clear_color() : scene_render->environment_get_bg_color(p_environment);
			transport_configuration = hash_murmur3_one_float(color.r, transport_configuration);
			transport_configuration = hash_murmur3_one_float(color.g, transport_configuration);
			transport_configuration = hash_murmur3_one_float(color.b, transport_configuration);
			transport_configuration = hash_murmur3_one_float(color.a, transport_configuration);
		} else if (background == RSE::ENV_BG_SKY && sky) {
			const RID sky_rid = scene_render->environment_get_sky(p_environment);
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
	raster_correction.add_jitter_offset(p_taa_jitter);
	Projection temporal_correction;
	temporal_correction.set_depth_correction(true);

	bool camera_cut = false;
	if (hddagi->screen_probe_previous_camera_valid) {
		const float translation = hddagi->screen_probe_previous_cam_transform.origin.distance_to(p_cam_transform.origin);
		const float translation_limit = MAX(4.0f, hddagi->min_cell_size * 16.0f);
		const float rotation = hddagi->screen_probe_previous_cam_transform.basis.get_rotation_quaternion().angle_to(p_cam_transform.basis.get_rotation_quaternion());
		camera_cut = translation > translation_limit || rotation > Math::deg_to_rad(55.0f);
		for (uint32_t v = 0; v < p_view_count && !camera_cut; v++) {
			const Projection temporal_projection = temporal_correction * p_projections[v];
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
	if (hddagi->reads_sky && p_environment.is_valid()) {
		const RSE::EnvironmentBG background = scene_render->environment_get_background(p_environment);
		if (background == RSE::ENV_BG_CLEAR_COLOR || background == RSE::ENV_BG_COLOR) {
			Color color = scene_render->environment_get_bg_color(p_environment);
			if (background == RSE::ENV_BG_CLEAR_COLOR) {
				color = RSG::texture_storage->get_default_clear_color().srgb_to_linear();
			}
			screen_probe_sky_color[0] = color.r;
			screen_probe_sky_color[1] = color.g;
			screen_probe_sky_color[2] = color.b;
			screen_probe_sky_energy = scene_render->environment_get_bg_energy_multiplier(p_environment) * scene_render->environment_get_bg_intensity(p_environment) * p_exposure_normalization;
			screen_probe_sky_mode = HDDAGIShader::SCREEN_PROBE_SKY_COLOR;
		} else if (background == RSE::ENV_BG_SKY && sky) {
			const RID sky_rid = scene_render->environment_get_sky(p_environment);
			if (sky_rid.is_valid()) {
				const RID environment_radiance = sky->sky_get_radiance_texture_rd(sky_rid);
				if (environment_radiance.is_valid()) {
					sky_texture = environment_radiance;
					screen_probe_sky_color[3] = sky->sky_get_uv_border_size(sky_rid);
					screen_probe_sky_energy = scene_render->environment_get_bg_energy_multiplier(p_environment) * p_ibl_exposure_normalization;
					screen_probe_sky_mode = HDDAGIShader::SCREEN_PROBE_SKY_TEXTURE;
				}
			}
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

	const RID nearest_sampler = material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	const RID linear_sampler = material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	const RID linear_mipmap_sampler = material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	const RID detail_hiz_fallback = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
	if (!nearest_sampler.is_valid() || !linear_sampler.is_valid() || !linear_mipmap_sampler.is_valid() || !sky_texture.is_valid() || !detail_hiz_fallback.is_valid()) {
		disable_hddagi_screen_probes(p_render_buffers);
		return;
	}

	bool irradiance_cache_active = false;
	HDDAGIScreenProbeIrradianceCache::DispatchInfo irradiance_cache_dispatch_info;
	RID irradiance_cache_maintenance_sets[HDDAGIShader::SCREEN_PROBE_IRRADIANCE_CACHE_MAX];
	RID irradiance_cache_trace_set;
	RID irradiance_cache_multibounce_set;
	RID irradiance_cache_multibounce_resources_set;
	RID irradiance_cache_multibounce_sky_set;
	auto clear_irradiance_cache = [&](const String &p_reason) {
		rbgi->screen_probe_irradiance_cache.clear();
		WARN_PRINT_ONCE("HDDAGI screen-probe irradiance cache is unavailable: " + p_reason);
	};
	if (irradiance_cache_setting == 1) {
		auto screen_probe_mode_is_ready = [&](HDDAGIShader::ScreenProbeMode p_screen_probe_mode) {
			return hddagi_shader.screen_probe_shader_version[p_screen_probe_mode].is_valid() && hddagi_shader.screen_probe_pipeline[p_screen_probe_mode].is_valid();
		};
		bool modes_ready = HDDAGIScreenProbeIrradianceCache::is_supported();
		for (uint32_t mode = 0; mode < HDDAGIShader::SCREEN_PROBE_IRRADIANCE_CACHE_MAX; mode++) {
			modes_ready = modes_ready && hddagi_shader.screen_probe_irradiance_cache_shader_version[mode].is_valid() && hddagi_shader.screen_probe_irradiance_cache_pipeline[mode].is_valid();
		}
		modes_ready = modes_ready && screen_probe_mode_is_ready(HDDAGIShader::SCREEN_PROBE_MODE_TRACE_IRRADIANCE_CACHE);
		if (irradiance_cache_multibounce) {
			modes_ready = modes_ready && screen_probe_mode_is_ready(HDDAGIShader::SCREEN_PROBE_MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE) && hddagi->voxel_albedo_data.is_valid();
		}

		if (!modes_ready) {
			clear_irradiance_cache("the required compute resources could not be created");
		} else {
			uint32_t irradiance_cache_history_key = hash_murmur3_one_32(transport_configuration);
			irradiance_cache_history_key = hash_murmur3_one_32(0x49430101u, irradiance_cache_history_key);
			irradiance_cache_history_key = hash_murmur3_one_32(irradiance_cache_multibounce ? 1u : 0u, irradiance_cache_history_key);
			irradiance_cache_history_key = hash_murmur3_one_float(irradiance_cache_minimum_cell_size, irradiance_cache_history_key);
			irradiance_cache_history_key = hash_fmix32(irradiance_cache_history_key);

			HDDAGIScreenProbeIrradianceCache::FrameSettings cache_frame;
			cache_frame.camera_position = p_cam_transform.origin;
			cache_frame.history_key = irradiance_cache_history_key;
			cache_frame.minimum_cell_size = irradiance_cache_minimum_cell_size;
			cache_frame.sky_mode = screen_probe_sky_mode;
			cache_frame.sky_energy = screen_probe_sky_energy;
			for (uint32_t i = 0; i < 4; i++) {
				cache_frame.sky_color[i] = screen_probe_sky_color[i];
			}
			cache_frame.reset_history = camera_cut;
			cache_frame.multibounce = irradiance_cache_multibounce;
			const Error frame_error = rbgi->screen_probe_irradiance_cache.prepare_frame(cache_frame, irradiance_cache_dispatch_info);
			if (frame_error != OK) {
				clear_irradiance_cache(vformat("frame preparation failed with error %d", frame_error));
			} else {
				bool sets_valid = true;
				for (uint32_t mode = 0; mode < HDDAGIShader::SCREEN_PROBE_IRRADIANCE_CACHE_MAX; mode++) {
					irradiance_cache_maintenance_sets[mode] = rbgi->screen_probe_irradiance_cache.get_uniform_set(hddagi_shader.screen_probe_irradiance_cache_shader_version[mode], 0);
					sets_valid = sets_valid && RD::get_singleton()->uniform_set_is_valid(irradiance_cache_maintenance_sets[mode]);
				}
				auto get_query_set = [&](HDDAGIShader::ScreenProbeMode p_screen_probe_mode) {
					return rbgi->screen_probe_irradiance_cache.get_uniform_set(hddagi_shader.screen_probe_shader_version[p_screen_probe_mode], 2);
				};
				irradiance_cache_trace_set = get_query_set(HDDAGIShader::SCREEN_PROBE_MODE_TRACE_IRRADIANCE_CACHE);
				sets_valid = sets_valid && RD::get_singleton()->uniform_set_is_valid(irradiance_cache_trace_set);
				if (irradiance_cache_multibounce) {
					irradiance_cache_multibounce_set = get_query_set(HDDAGIShader::SCREEN_PROBE_MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE);
					const HDDAGIShader::ScreenProbeMode update_mode = HDDAGIShader::SCREEN_PROBE_MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE;
					irradiance_cache_multibounce_resources_set = UniformSetCacheRD::get_singleton()->get_cache(
							hddagi_shader.screen_probe_shader_version[update_mode], 0,
							RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, hddagi->voxel_bits_tex),
							RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, hddagi->voxel_region_tex),
							RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 4, hddagi->voxel_light_tex),
							RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 5, linear_sampler),
							RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, hddagi->voxel_light_neighbour_data),
							RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, hddagi_ubo),
							RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 8, hddagi->voxel_disocclusion_tex),
							RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 9, rbgi->screen_probe_scene_data_ubo),
							RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 10, hddagi->voxel_albedo_data));
					irradiance_cache_multibounce_sky_set = UniformSetCacheRD::get_singleton()->get_cache(
							hddagi_shader.screen_probe_shader_version[update_mode], 1,
							RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, sky_texture),
							RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, linear_mipmap_sampler));
					sets_valid = sets_valid && RD::get_singleton()->uniform_set_is_valid(irradiance_cache_multibounce_set) &&
							RD::get_singleton()->uniform_set_is_valid(irradiance_cache_multibounce_resources_set) &&
							RD::get_singleton()->uniform_set_is_valid(irradiance_cache_multibounce_sky_set);
				}
				if (sets_valid) {
					irradiance_cache_active = true;
				} else {
					clear_irradiance_cache("the shader resources could not be bound");
				}
			}
		}
	} else {
		rbgi->screen_probe_irradiance_cache.clear();
		if (irradiance_cache_setting != 0) {
			WARN_PRINT_ONCE(vformat("Unknown HDDAGI screen-probe irradiance cache setting %d; using Disabled.", irradiance_cache_setting));
		}
	}

	uint32_t configuration = hash_murmur3_one_32(directional_gather ? 0x44475202u : 0u);
	if (directional_gather) {
		configuration = hash_murmur3_one_32(uint32_t(SCREEN_PROBE_DIRECTIONAL_SIZE), configuration);
		configuration = hash_murmur3_one_32(SCREEN_PROBE_DIRECTIONAL_FILTER_PASS_COUNT, configuration);
		configuration = hash_murmur3_one_32(SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE, configuration);
		configuration = hash_murmur3_one_32(SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT, configuration);
		configuration = hash_murmur3_one_32(SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE, configuration);
		configuration = hash_murmur3_one_32(SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CAPACITY_DIVISOR, configuration);
		configuration = hash_murmur3_one_32(uint32_t((uint64_t(probe_atlas_size.x) * uint64_t(probe_atlas_size.y)) / SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CAPACITY_DIVISOR), configuration);
		configuration = hash_murmur3_one_32(uint32_t(directional_probe_atlas_size.y), configuration);
	}
	configuration = hash_murmur3_one_32(candidate_count, configuration);
	configuration = hash_murmur3_one_32(guided_sampling ? 1u : 0u, configuration);
	configuration = hash_murmur3_one_32(detail_trace ? 1u : 0u, configuration);
	configuration = hash_murmur3_one_32(transport_configuration, configuration);
	configuration = hash_murmur3_one_32(irradiance_cache_active ? 1u : 0u, configuration);
	if (irradiance_cache_active) {
		configuration = hash_murmur3_one_32(irradiance_cache_multibounce ? 1u : 0u, configuration);
		configuration = hash_murmur3_one_float(irradiance_cache_minimum_cell_size, configuration);
	}
	configuration = hash_murmur3_one_float(normal_bias, configuration);
	configuration = hash_fmix32(configuration);

	const uint32_t surface_layers = directional_gather ? p_view_count * 2u : p_view_count;
	const RD::DataFormat radiance_format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
	auto texture_matches = [&](const StringName &p_name, RD::DataFormat p_format, const Size2i &p_size, uint32_t p_layers) {
		if (!p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, p_name)) {
			return false;
		}
		const RD::TextureFormat format = p_render_buffers->get_texture_format(RB_SCOPE_HDDAGI_SCREEN_PROBES, p_name);
		return format.format == p_format && format.width == uint32_t(p_size.x) && format.height == uint32_t(p_size.y) && format.array_layers == p_layers &&
				p_render_buffers->get_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, p_name).is_valid();
	};
	const uint32_t directional_history_layers = p_view_count * 2u;
	const bool has_directional_resources = p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_RADIANCE) ||
			p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_HISTORY_AGE) ||
			p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_FILTER_SCRATCH) ||
			p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_IRRADIANCE) ||
			p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_TRACE_COUNT) ||
			p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MARK) ||
			p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_TILE_DATA) ||
			p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_COUNTER) ||
			p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT_U32) ||
			p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT);
	auto directional_textures_match = [&]() {
		return texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_RADIANCE, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, directional_atlas_size, directional_history_layers) &&
				texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_HISTORY_AGE, RD::DATA_FORMAT_R8_UINT, directional_probe_atlas_size, directional_history_layers) &&
				texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_FILTER_SCRATCH, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, directional_atlas_size, directional_history_layers) &&
				texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_IRRADIANCE, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, directional_atlas_size, p_view_count) &&
				texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_TRACE_COUNT, RD::DATA_FORMAT_R8_UINT, directional_atlas_size, p_view_count) &&
				texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MARK, RD::DATA_FORMAT_R8_UINT, probe_atlas_size, p_view_count) &&
				texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_TILE_DATA, RD::DATA_FORMAT_R32G32B32A32_UINT, probe_atlas_size, directional_history_layers) &&
				texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_COUNTER, RD::DATA_FORMAT_R32_UINT, Size2i(1, 1), p_view_count) &&
				texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT_U32, RD::DATA_FORMAT_R32_UINT, internal_size, p_view_count) &&
				p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT) &&
				p_render_buffers->get_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT).is_valid();
	};
	const Size2i surface_size = directional_gather ? directional_probe_atlas_size : probe_atlas_size;
	bool resources_valid = texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, RD::DATA_FORMAT_R32G32B32A32_UINT, surface_size, surface_layers) &&
			texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE, radiance_format, resolve_size, p_view_count) &&
			(directional_gather ? (!p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE) && directional_textures_match()) : (texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE, radiance_format, probe_atlas_size, p_view_count) && !has_directional_resources));
	const bool resources_recreated = !resources_valid;
	if (!resources_valid) {
		p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBES);
		const uint32_t texture_usage = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
		p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, RD::DATA_FORMAT_R32G32B32A32_UINT, texture_usage, RD::TEXTURE_SAMPLES_1, surface_size, surface_layers);
		p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE, radiance_format, texture_usage, RD::TEXTURE_SAMPLES_1, resolve_size, p_view_count);
		if (directional_gather) {
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_RADIANCE, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, texture_usage, RD::TEXTURE_SAMPLES_1, directional_atlas_size, directional_history_layers);
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_HISTORY_AGE, RD::DATA_FORMAT_R8_UINT, texture_usage, RD::TEXTURE_SAMPLES_1, directional_probe_atlas_size, directional_history_layers);
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_FILTER_SCRATCH, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, texture_usage, RD::TEXTURE_SAMPLES_1, directional_atlas_size, directional_history_layers);
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_IRRADIANCE, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, texture_usage, RD::TEXTURE_SAMPLES_1, directional_atlas_size, p_view_count);
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_TRACE_COUNT, RD::DATA_FORMAT_R8_UINT, texture_usage, RD::TEXTURE_SAMPLES_1, directional_atlas_size, p_view_count);
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MARK, RD::DATA_FORMAT_R8_UINT, texture_usage, RD::TEXTURE_SAMPLES_1, probe_atlas_size, p_view_count);
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_TILE_DATA, RD::DATA_FORMAT_R32G32B32A32_UINT, texture_usage, RD::TEXTURE_SAMPLES_1, probe_atlas_size, directional_history_layers);
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_COUNTER, RD::DATA_FORMAT_R32_UINT, texture_usage, RD::TEXTURE_SAMPLES_1, Size2i(1, 1), p_view_count);

			RD::TextureFormat directional_ambient_format;
			directional_ambient_format.format = RD::DATA_FORMAT_R32_UINT;
			directional_ambient_format.width = internal_size.x;
			directional_ambient_format.height = internal_size.y;
			directional_ambient_format.depth = 1;
			directional_ambient_format.array_layers = p_view_count;
			directional_ambient_format.texture_type = p_view_count > 1 ? RD::TEXTURE_TYPE_2D_ARRAY : RD::TEXTURE_TYPE_2D;
			directional_ambient_format.shareable_formats.push_back(RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32);
			directional_ambient_format.shareable_formats.push_back(RD::DATA_FORMAT_R32_UINT);
			directional_ambient_format.usage_bits = texture_usage;
			const RID directional_ambient_u32 = p_render_buffers->create_texture_from_format(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT_U32, directional_ambient_format);
			if (directional_ambient_u32.is_valid()) {
				RD::TextureView directional_ambient_view;
				directional_ambient_view.format_override = RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32;
				p_render_buffers->create_texture_view(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT_U32, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT, directional_ambient_view);
			}
		} else {
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE, radiance_format, texture_usage, RD::TEXTURE_SAMPLES_1, probe_atlas_size, p_view_count);
		}
		resources_valid = texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, RD::DATA_FORMAT_R32G32B32A32_UINT, surface_size, surface_layers) &&
				texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE, radiance_format, resolve_size, p_view_count) &&
				(directional_gather ? directional_textures_match() : texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE, radiance_format, probe_atlas_size, p_view_count));
	}
	if (!resources_valid) {
		disable_hddagi_screen_probes(p_render_buffers);
		hddagi->screen_probe_directional_allocation_failed = directional_gather;
		if (directional_gather) {
			WARN_PRINT_ONCE("HDDAGI Directional Gather textures could not be allocated; using Stochastic Integrated mode until Directional Gather is reselected.");
		} else {
			WARN_PRINT_ONCE("HDDAGI screen probe textures could not be allocated.");
		}
		return;
	}

	RID trace_debug_slices[2];
	bool debug_montage = false;
	if (p_debug_montage) {
		auto debug_texture_matches = [&]() {
			if (!p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DEBUG, RB_TEX_HDDAGI_SCREEN_PROBE_TRACE_DEBUG)) {
				return false;
			}
			const RD::TextureFormat format = p_render_buffers->get_texture_format(RB_SCOPE_HDDAGI_SCREEN_PROBE_DEBUG, RB_TEX_HDDAGI_SCREEN_PROBE_TRACE_DEBUG);
			return format.format == RD::DATA_FORMAT_R32_UINT && format.width == uint32_t(probe_atlas_size.x) && format.height == uint32_t(probe_atlas_size.y) &&
					format.mipmaps == 1 && format.array_layers == p_view_count &&
					p_render_buffers->get_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DEBUG, RB_TEX_HDDAGI_SCREEN_PROBE_TRACE_DEBUG).is_valid();
		};
		if (!debug_texture_matches()) {
			p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBE_DEBUG);
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DEBUG, RB_TEX_HDDAGI_SCREEN_PROBE_TRACE_DEBUG, RD::DATA_FORMAT_R32_UINT,
					RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT, RD::TEXTURE_SAMPLES_1, probe_atlas_size, p_view_count);
		}
		debug_montage = debug_texture_matches();
		for (uint32_t v = 0; v < p_view_count && debug_montage; v++) {
			trace_debug_slices[v] = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBE_DEBUG, RB_TEX_HDDAGI_SCREEN_PROBE_TRACE_DEBUG, v, 0);
			debug_montage = trace_debug_slices[v].is_valid();
		}
		if (debug_montage && directional_gather) {
			RD::get_singleton()->texture_clear(p_render_buffers->get_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DEBUG, RB_TEX_HDDAGI_SCREEN_PROBE_TRACE_DEBUG), Color(), 0, 1, 0, p_view_count);
		} else if (!debug_montage) {
			p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBE_DEBUG);
			WARN_PRINT_ONCE("HDDAGI screen-probe debug data could not be allocated.");
		}
	} else {
		p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBE_DEBUG);
	}

	bool specular_resources_recreated = false;
	if (specular_reflections) {
		auto specular_texture_matches = [&](const StringName &p_name, RD::DataFormat p_format) {
			if (!p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, p_name)) {
				return false;
			}
			const RD::TextureFormat format = p_render_buffers->get_texture_format(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, p_name);
			return format.format == p_format && format.width == uint32_t(p_gi_size.x) && format.height == uint32_t(p_gi_size.y) && format.mipmaps == 1 && format.array_layers == p_view_count &&
					p_render_buffers->get_texture(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, p_name).is_valid();
		};
		bool specular_resources_valid = specular_texture_matches(RB_TEX_HDDAGI_SPECULAR_RAW, RD::DATA_FORMAT_R16G16B16A16_SFLOAT) &&
				specular_texture_matches(RB_TEX_HDDAGI_SPECULAR_NORMAL_ROUGHNESS, RD::DATA_FORMAT_R8G8B8A8_UNORM) &&
				specular_texture_matches(RB_TEX_HDDAGI_SPECULAR_VIEW_Z, RD::DATA_FORMAT_R32_SFLOAT) &&
				specular_texture_matches(RB_TEX_HDDAGI_SPECULAR_MOTION, RD::DATA_FORMAT_R16G16B16A16_SFLOAT) &&
				specular_texture_matches(RB_TEX_HDDAGI_SPECULAR_DENOISED, RD::DATA_FORMAT_R16G16B16A16_SFLOAT);
		if (!specular_resources_valid) {
			specular_resources_recreated = true;
			p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS);
			rbgi->screen_probe_specular_svgf.clear();
			const uint32_t usage = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, RB_TEX_HDDAGI_SPECULAR_RAW, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage, RD::TEXTURE_SAMPLES_1, p_gi_size, p_view_count);
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, RB_TEX_HDDAGI_SPECULAR_NORMAL_ROUGHNESS, RD::DATA_FORMAT_R8G8B8A8_UNORM, usage, RD::TEXTURE_SAMPLES_1, p_gi_size, p_view_count);
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, RB_TEX_HDDAGI_SPECULAR_VIEW_Z, RD::DATA_FORMAT_R32_SFLOAT, usage, RD::TEXTURE_SAMPLES_1, p_gi_size, p_view_count);
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, RB_TEX_HDDAGI_SPECULAR_MOTION, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage, RD::TEXTURE_SAMPLES_1, p_gi_size, p_view_count);
			p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, RB_TEX_HDDAGI_SPECULAR_DENOISED, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage, RD::TEXTURE_SAMPLES_1, p_gi_size, p_view_count);
			specular_resources_valid = specular_texture_matches(RB_TEX_HDDAGI_SPECULAR_RAW, RD::DATA_FORMAT_R16G16B16A16_SFLOAT) &&
					specular_texture_matches(RB_TEX_HDDAGI_SPECULAR_NORMAL_ROUGHNESS, RD::DATA_FORMAT_R8G8B8A8_UNORM) &&
					specular_texture_matches(RB_TEX_HDDAGI_SPECULAR_VIEW_Z, RD::DATA_FORMAT_R32_SFLOAT) &&
					specular_texture_matches(RB_TEX_HDDAGI_SPECULAR_MOTION, RD::DATA_FORMAT_R16G16B16A16_SFLOAT) &&
					specular_texture_matches(RB_TEX_HDDAGI_SPECULAR_DENOISED, RD::DATA_FORMAT_R16G16B16A16_SFLOAT);
		}
		if (!specular_resources_valid) {
			p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS);
			rbgi->screen_probe_specular_svgf.clear();
			specular_reflections = false;
			WARN_PRINT_ONCE("HDDAGI specular reflection textures could not be allocated.");
		}
	}

	bool svgf_active = false;
	bool svgf_resources_recreated = false;
	RID svgf_velocity[2];
	RID svgf_input[2];
	RID svgf_normal_roughness[2];
	RID svgf_view_z[2];
	RID svgf_motion[2];
	RID svgf_output[2];
	auto clear_svgf = [&]() {
		rbgi->screen_probe_svgf.clear();
		p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER);
	};
	if (denoiser_setting == 1) {
		const HDDAGIShader::ScreenProbeMode svgf_resolve_mode = directional_gather ? HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_RESOLVE_SVGF : HDDAGIShader::SCREEN_PROBE_MODE_RESOLVE_SVGF;
		bool svgf_ready = HDDAGIScreenProbeSVGF::is_supported() &&
				hddagi_shader.screen_probe_shader_version[svgf_resolve_mode].is_valid() && hddagi_shader.screen_probe_pipeline[svgf_resolve_mode].is_valid() &&
				hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_APPLY_SVGF].is_valid() && hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_APPLY_SVGF].is_valid() &&
				p_render_buffers->has_velocity_buffer(false);
		for (uint32_t v = 0; v < p_view_count && svgf_ready; v++) {
			svgf_velocity[v] = p_render_buffers->get_velocity_buffer(false, v);
			svgf_ready = svgf_velocity[v].is_valid() && RD::get_singleton()->texture_is_valid(svgf_velocity[v]) && RD::get_singleton()->texture_size(svgf_velocity[v]) == internal_size;
		}
		if (!svgf_ready) {
			clear_svgf();
			WARN_PRINT_ONCE("HDDAGI screen-probe SVGF requires supported compute resources and valid forward-pass velocity textures; using the unfiltered signal.");
		} else {
			auto svgf_texture_matches = [&](const StringName &p_name, RD::DataFormat p_format) {
				if (!p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, p_name)) {
					return false;
				}
				const RD::TextureFormat format = p_render_buffers->get_texture_format(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, p_name);
				return format.format == p_format && format.width == uint32_t(resolve_size.x) && format.height == uint32_t(resolve_size.y) && format.array_layers == p_view_count &&
						p_render_buffers->get_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, p_name).is_valid();
			};
			bool svgf_resources_valid = svgf_texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_INPUT, RD::DATA_FORMAT_R16G16B16A16_SFLOAT) &&
					svgf_texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_NORMAL_ROUGHNESS, RD::DATA_FORMAT_R8G8B8A8_UNORM) &&
					svgf_texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_VIEW_Z, RD::DATA_FORMAT_R32_SFLOAT) &&
					svgf_texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_MOTION, RD::DATA_FORMAT_R16G16B16A16_SFLOAT) &&
					svgf_texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_OUTPUT, RD::DATA_FORMAT_R16G16B16A16_SFLOAT);
			if (!svgf_resources_valid) {
				svgf_resources_recreated = true;
				clear_svgf();
				const uint32_t svgf_texture_usage = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
				p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_INPUT, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, svgf_texture_usage, RD::TEXTURE_SAMPLES_1, resolve_size, p_view_count);
				p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_NORMAL_ROUGHNESS, RD::DATA_FORMAT_R8G8B8A8_UNORM, svgf_texture_usage, RD::TEXTURE_SAMPLES_1, resolve_size, p_view_count);
				p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_VIEW_Z, RD::DATA_FORMAT_R32_SFLOAT, svgf_texture_usage, RD::TEXTURE_SAMPLES_1, resolve_size, p_view_count);
				p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_MOTION, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, svgf_texture_usage, RD::TEXTURE_SAMPLES_1, resolve_size, p_view_count);
				p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_OUTPUT, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, svgf_texture_usage, RD::TEXTURE_SAMPLES_1, resolve_size, p_view_count);
				svgf_resources_valid = svgf_texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_INPUT, RD::DATA_FORMAT_R16G16B16A16_SFLOAT) &&
						svgf_texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_NORMAL_ROUGHNESS, RD::DATA_FORMAT_R8G8B8A8_UNORM) &&
						svgf_texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_VIEW_Z, RD::DATA_FORMAT_R32_SFLOAT) &&
						svgf_texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_MOTION, RD::DATA_FORMAT_R16G16B16A16_SFLOAT) &&
						svgf_texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_OUTPUT, RD::DATA_FORMAT_R16G16B16A16_SFLOAT);
			}
			for (uint32_t v = 0; v < p_view_count && svgf_resources_valid; v++) {
				svgf_input[v] = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_INPUT, v, 0);
				svgf_normal_roughness[v] = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_NORMAL_ROUGHNESS, v, 0);
				svgf_view_z[v] = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_VIEW_Z, v, 0);
				svgf_motion[v] = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_MOTION, v, 0);
				svgf_output[v] = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_OUTPUT, v, 0);
				svgf_resources_valid = svgf_input[v].is_valid() && svgf_normal_roughness[v].is_valid() && svgf_view_z[v].is_valid() && svgf_motion[v].is_valid() && svgf_output[v].is_valid();
			}
			if (svgf_resources_valid) {
				svgf_active = true;
			} else {
				clear_svgf();
				WARN_PRINT_ONCE("HDDAGI screen-probe SVGF textures could not be allocated; using the unfiltered signal.");
			}
		}
	} else {
		clear_svgf();
		if (denoiser_setting != 0) {
			WARN_PRINT_ONCE(vformat("Unknown HDDAGI screen-probe denoiser setting %d; using Disabled.", denoiser_setting));
		}
	}
	RID specular_velocity[2];
	bool specular_motion_valid = specular_reflections && p_render_buffers->has_velocity_buffer(false);
	for (uint32_t v = 0; v < p_view_count && specular_motion_valid; v++) {
		specular_velocity[v] = p_render_buffers->get_velocity_buffer(false, v);
		specular_motion_valid = specular_velocity[v].is_valid() && RD::get_singleton()->texture_is_valid(specular_velocity[v]) && RD::get_singleton()->texture_size(specular_velocity[v]) == internal_size;
	}
	const bool specular_svgf_active = specular_reflections && denoiser_setting == 1 && HDDAGIScreenProbeSVGF::is_supported() && specular_motion_valid;
	if (!specular_svgf_active) {
		rbgi->screen_probe_specular_svgf.clear();
	}

	const bool history_configuration_valid = !resources_recreated && hddagi->screen_probe_history_initialized && hddagi->screen_probe_history_probe_size == probe_size &&
			hddagi->screen_probe_history_gi_size == p_gi_size && hddagi->screen_probe_history_screen_size == internal_size &&
			hddagi->screen_probe_history_view_count == p_view_count && hddagi->screen_probe_history_configuration == configuration;
	const bool camera_history_valid = hddagi->screen_probe_previous_camera_valid && !camera_cut;
	const bool common_history_valid = history_configuration_valid && camera_history_valid;
	const bool history_valid = directional_gather && common_history_valid;
	const bool svgf_history_valid = svgf_active && common_history_valid && !svgf_resources_recreated;
	const bool specular_history_valid = specular_reflections && common_history_valid && !specular_resources_recreated;
	const bool specular_screen_radiance_valid = specular_reflections && specular_screen_radiance_available && camera_history_valid;
	const bool reset_history = resources_recreated || !history_configuration_valid || camera_cut;
	const uint32_t current_history_slot = reset_history ? 0u : (hddagi->screen_probe_history_slot ^ 1u);
	const uint32_t previous_history_slot = current_history_slot ^ 1u;
	const uint32_t history_sequence = reset_history ? 0u : ((hddagi->screen_probe_history_sequence + 1u) & SCREEN_PROBE_HISTORY_SEQUENCE_MASK);

	ScreenProbeSceneData scene_data = {};
	const Transform3D previous_cam_inv_transform = camera_history_valid ? hddagi->screen_probe_previous_cam_transform.affine_inverse() : p_cam_transform.affine_inverse();
	for (uint32_t v = 0; v < p_view_count; v++) {
		const Projection raster_projection = raster_correction * p_projections[v];
		const Projection previous_raster_projection = camera_history_valid ? hddagi->screen_probe_previous_projection[v] : raster_projection;
		const Projection temporal_projection = temporal_correction * p_projections[v];
		const Projection previous_temporal_projection = camera_history_valid ? hddagi->screen_probe_previous_temporal_projection[v] : temporal_projection;
		RendererRD::MaterialStorage::store_camera(raster_projection.inverse(), scene_data.inv_projection[v]);
		RendererRD::MaterialStorage::store_camera(raster_projection, scene_data.projection[v]);
		RendererRD::MaterialStorage::store_camera(previous_raster_projection.inverse(), scene_data.previous_inv_projection[v]);
		RendererRD::MaterialStorage::store_camera(temporal_projection, scene_data.temporal_projection[v]);
		RendererRD::MaterialStorage::store_camera(previous_temporal_projection, scene_data.previous_temporal_projection[v]);
	}
	RendererRD::MaterialStorage::store_transform(p_cam_transform, scene_data.cam_transform);
	RendererRD::MaterialStorage::store_transform(previous_cam_inv_transform, scene_data.previous_cam_inv_transform);
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
	push_constant.frame_index = directional_gather ? history_sequence : uint32_t(RSG::rasterizer->get_frame_number());
	push_constant.flags = (detail_trace ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_DETAIL_TRACE) : 0u) |
			(guided_sampling ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_GUIDED_SAMPLING) : 0u) |
			(directional_gather ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_DIRECTIONAL_SURFACE_FOOTPRINT) : 0u) |
			((directional_gather && history_valid) ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_DIRECTIONAL_HISTORY_VALID) : 0u) |
			(directional_gather ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_DIRECTIONAL_ADAPTIVE) : 0u);
	push_constant.normal_bias = normal_bias;
	push_constant.candidate_count = candidate_count;
	push_constant.sky_mode = screen_probe_sky_mode;
	push_constant.sky_energy = screen_probe_sky_energy;
	push_constant.detail_trace_mip_count = detail_trace_mip_count;
	for (uint32_t i = 0; i < 4; i++) {
		push_constant.sky_color[i] = screen_probe_sky_color[i];
	}
	HDDAGIShader::ScreenProbeSVGFPreparePushConstant svgf_prepare_push_constant = {};
	svgf_prepare_push_constant.base = push_constant;
	svgf_prepare_push_constant.denoising_range = SCREEN_PROBE_SVGF_DENOISING_RANGE;
	svgf_prepare_push_constant.scene_to_svgf_scale = SCREEN_PROBE_SVGF_SCENE_TO_SIGNAL_SCALE;
	svgf_prepare_push_constant.input_radiance_max = SCREEN_PROBE_SVGF_INPUT_RADIANCE_MAX;
	svgf_prepare_push_constant.input_hit_distance_max = SCREEN_PROBE_SVGF_INPUT_HIT_DISTANCE_MAX;
	HDDAGIShader::ScreenProbeSpecularPushConstant specular_push_constant = {};
	specular_push_constant.base = push_constant;
	specular_push_constant.base.flags = (specular_detail_trace ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_DETAIL_TRACE) : 0u) |
			(specular_motion_valid ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_SPECULAR_MOTION_VALID) : 0u) |
			(specular_screen_radiance_valid ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_SPECULAR_SCREEN_RADIANCE_VALID) : 0u);
	specular_push_constant.base.detail_trace_mip_count = specular_detail_trace ? CLAMP(p_hiz_mip_count, 1u, 16u) : 0u;
	specular_push_constant.tuning[0] = SCREEN_PROBE_SPECULAR_FULL_AUTHORITY_ROUGHNESS;
	specular_push_constant.tuning[1] = SCREEN_PROBE_SPECULAR_FALLBACK_ROUGHNESS;
	specular_push_constant.tuning[2] = SCREEN_PROBE_SPECULAR_RADIANCE_MAX;
	specular_push_constant.tuning[3] = SCREEN_PROBE_SPECULAR_MIN_GGX_ALPHA;
	specular_push_constant.eye_offset_exposure[3] = specular_screen_radiance_valid ? p_exposure_normalization / MAX(hddagi->screen_probe_previous_exposure_normalization, 0.001f) : 1.0f;

	RID surface_sets[2];
	RID trace_sets[2];
	RID trace_sky_sets[2];
	RID directional_adaptive_mark_sets[2];
	RID directional_adaptive_spawn_sets[2];
	RID directional_trace_sets[2];
	RID directional_trace_sky_sets[2];
	RID directional_filter_sets[2][SCREEN_PROBE_DIRECTIONAL_FILTER_PASS_COUNT];
	RID directional_irradiance_sets[2];
	RID trace_debug_sets[2];
	RID resolve_sets[2];
	RID svgf_resolve_sets[2];
	RID apply_sets[2];
	RID svgf_apply_sets[2];
	RID specular_trace_sets[2];
	RID specular_trace_sky_sets[2];
	RID specular_apply_sets[2];
	RID specular_denoised_apply_sets[2];
	RID specular_raw[2];
	RID specular_normal_roughness[2];
	RID specular_view_z[2];
	RID specular_motion[2];
	RID specular_denoised[2];
	uint32_t view_flags[2] = {};
	const HDDAGIShader::ScreenProbeMode production_trace_mode = irradiance_cache_active ? HDDAGIShader::SCREEN_PROBE_MODE_TRACE_IRRADIANCE_CACHE : HDDAGIShader::SCREEN_PROBE_MODE_TRACE;
	HDDAGIShader::ScreenProbeMode trace_mode = production_trace_mode;
	if (debug_montage && !directional_gather) {
		const HDDAGIShader::ScreenProbeMode debug_mode = irradiance_cache_active ? HDDAGIShader::SCREEN_PROBE_MODE_TRACE_IRRADIANCE_CACHE_DEBUG : HDDAGIShader::SCREEN_PROBE_MODE_TRACE_DEBUG;
		bool debug_mode_valid = hddagi_shader.screen_probe_shader_version[debug_mode].is_valid() && hddagi_shader.screen_probe_pipeline[debug_mode].is_valid();
		RID debug_irradiance_cache_set;
		if (debug_mode_valid && irradiance_cache_active) {
			debug_irradiance_cache_set = rbgi->screen_probe_irradiance_cache.get_uniform_set(hddagi_shader.screen_probe_shader_version[debug_mode], 2);
			debug_mode_valid = RD::get_singleton()->uniform_set_is_valid(debug_irradiance_cache_set);
		}
		if (debug_mode_valid) {
			trace_mode = debug_mode;
			if (irradiance_cache_active) {
				irradiance_cache_trace_set = debug_irradiance_cache_set;
			}
		} else {
			debug_montage = false;
			p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBE_DEBUG);
			WARN_PRINT_ONCE("HDDAGI screen-probe debug shaders could not be created.");
		}
	}
	const HDDAGIShader::ScreenProbeMode resolve_mode = directional_gather ? HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_RESOLVE : HDDAGIShader::SCREEN_PROBE_MODE_RESOLVE;
	const HDDAGIShader::ScreenProbeMode svgf_resolve_mode = directional_gather ? HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_RESOLVE_SVGF : HDDAGIShader::SCREEN_PROBE_MODE_RESOLVE_SVGF;
	bool pipelines_valid = hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_SURFACE].is_valid() &&
			hddagi_shader.screen_probe_pipeline[resolve_mode].is_valid() && hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_APPLY].is_valid();
	if (svgf_active) {
		pipelines_valid = pipelines_valid && hddagi_shader.screen_probe_pipeline[svgf_resolve_mode].is_valid() && hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_APPLY_SVGF].is_valid();
	}
	if (directional_gather) {
		pipelines_valid = pipelines_valid && hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_ADAPTIVE_MARK].is_valid() &&
				hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_ADAPTIVE_SPAWN].is_valid() &&
				hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_TRACE].is_valid() &&
				hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_FILTER].is_valid() &&
				hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_IRRADIANCE].is_valid();
	} else {
		pipelines_valid = pipelines_valid && hddagi_shader.screen_probe_pipeline[trace_mode].is_valid();
	}
	if (!pipelines_valid) {
		disable_hddagi_screen_probes(p_render_buffers);
		return;
	}
	bool svgf_sets_valid = true;
	bool specular_sets_valid = specular_reflections;
	RD::TextureView packed_ambient_view;
	packed_ambient_view.format_override = RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32;
	for (uint32_t v = 0; v < p_view_count; v++) {
		const RID depth = p_render_buffers->get_depth_texture(v);
		const RID normal_roughness = p_normal_roughness_slices[v];
		RID directional_velocity = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
		bool directional_motion_valid = false;
		if (directional_gather && p_render_buffers->has_velocity_buffer(false)) {
			const RID velocity = p_render_buffers->get_velocity_buffer(false, v);
			if (velocity.is_valid() && RD::get_singleton()->texture_is_valid(velocity) && RD::get_singleton()->texture_size(velocity) == internal_size) {
				directional_velocity = velocity;
				directional_motion_valid = true;
			}
		}
		view_flags[v] = push_constant.flags | (directional_motion_valid ? uint32_t(HDDAGIShader::ScreenProbePushConstant::FLAG_DIRECTIONAL_MOTION_VALID) : 0u);
		const bool surface_history_ping_pong = directional_gather;
		const uint32_t surface_layer = surface_history_ping_pong ? v * 2u + current_history_slot : v;
		const uint32_t previous_surface_layer = surface_history_ping_pong ? v * 2u + previous_history_slot : v;
		const RID surface = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, surface_layer, 0);
		const RID previous_surface = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, previous_surface_layer, 0);
		RID raw_radiance;
		const RID resolved_radiance = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE, v, 0);
		const RID base_ambient = p_render_buffers->get_texture_slice_view(RB_SCOPE_GI, RB_TEX_AMBIENT_U32, v, 0, 1, 1, packed_ambient_view);
		const RID ambient_output = p_render_buffers->get_texture_slice(directional_gather ? RB_SCOPE_HDDAGI_SCREEN_PROBES : RB_SCOPE_GI, directional_gather ? RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_AMBIENT_U32 : RB_TEX_AMBIENT_U32, v, 0);
		if (!depth.is_valid() || !normal_roughness.is_valid() || !surface.is_valid() || !previous_surface.is_valid() || !resolved_radiance.is_valid() || !base_ambient.is_valid() || !ambient_output.is_valid()) {
			disable_hddagi_screen_probes(p_render_buffers);
			return;
		}

		surface_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
				hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_SURFACE], 0,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, depth),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, normal_roughness),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 2, nearest_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, surface));
		RID directional_adaptive_tile_data;
		if (directional_gather) {
			const uint32_t current_directional_layer = v * 2u + current_history_slot;
			const uint32_t previous_directional_layer = v * 2u + previous_history_slot;
			const RID directional_radiance = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_RADIANCE, current_directional_layer, 0);
			const RID previous_directional_radiance = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_RADIANCE, previous_directional_layer, 0);
			const RID directional_history_age = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_HISTORY_AGE, current_directional_layer, 0);
			const RID previous_directional_history_age = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_HISTORY_AGE, previous_directional_layer, 0);
			RID directional_filter_scratch[2] = {
				p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_FILTER_SCRATCH, current_directional_layer, 0),
				p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_FILTER_SCRATCH, previous_directional_layer, 0),
			};
			const RID directional_irradiance = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_IRRADIANCE, v, 0);
			const RID directional_trace_count = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_TRACE_COUNT, v, 0);
			const RID directional_adaptive_mark = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MARK, v, 0);
			directional_adaptive_tile_data = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_TILE_DATA, current_directional_layer, 0);
			const RID previous_directional_adaptive_tile_data = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_TILE_DATA, previous_directional_layer, 0);
			const RID directional_adaptive_counter = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_COUNTER, v, 0);
			if (!directional_radiance.is_valid() || !previous_directional_radiance.is_valid() || !directional_history_age.is_valid() || !previous_directional_history_age.is_valid() ||
					!directional_filter_scratch[0].is_valid() || !directional_filter_scratch[1].is_valid() || !directional_irradiance.is_valid() || !directional_trace_count.is_valid() ||
					!directional_adaptive_mark.is_valid() || !directional_adaptive_tile_data.is_valid() || !previous_directional_adaptive_tile_data.is_valid() || !directional_adaptive_counter.is_valid()) {
				disable_hddagi_screen_probes(p_render_buffers);
				return;
			}
			raw_radiance = directional_irradiance;

			auto get_directional_adaptive_set = [&](HDDAGIShader::ScreenProbeMode p_screen_probe_mode) {
				return UniformSetCacheRD::get_singleton()->get_cache(
						hddagi_shader.screen_probe_shader_version[p_screen_probe_mode], 0,
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, depth),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, normal_roughness),
						RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 2, nearest_sampler),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, surface),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, directional_adaptive_mark),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, directional_adaptive_tile_data),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, directional_adaptive_counter),
						RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, rbgi->screen_probe_scene_data_ubo),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 8, previous_surface),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 9, previous_directional_adaptive_tile_data));
			};
			directional_adaptive_mark_sets[v] = get_directional_adaptive_set(HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_ADAPTIVE_MARK);
			directional_adaptive_spawn_sets[v] = get_directional_adaptive_set(HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_ADAPTIVE_SPAWN);

			const HDDAGIShader::ScreenProbeMode directional_trace_mode = HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_TRACE;
			directional_trace_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
					hddagi_shader.screen_probe_shader_version[directional_trace_mode], 0,
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, surface),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, directional_radiance),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, hddagi->voxel_bits_tex),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, hddagi->voxel_region_tex),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 4, hddagi->voxel_light_tex),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 5, linear_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, hddagi->voxel_light_neighbour_data),
					RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, hddagi_ubo),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 8, hddagi->voxel_disocclusion_tex),
					RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 9, rbgi->screen_probe_scene_data_ubo),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 10, detail_trace ? p_hiz_slices[v] : detail_hiz_fallback),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 11, normal_roughness),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 12, previous_directional_radiance),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 13, previous_surface),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 14, previous_directional_history_age),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 15, directional_history_age),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 16, previous_directional_adaptive_tile_data),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 17, directional_velocity),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 18, depth),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 19, directional_filter_scratch[1]),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 20, directional_trace_count));
			directional_trace_sky_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
					hddagi_shader.screen_probe_shader_version[directional_trace_mode], 1,
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, sky_texture),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, linear_mipmap_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, hddagi->lightprobe_specular_tex));

			RID directional_filter_source = directional_radiance;
			for (uint32_t pass = 0; pass < SCREEN_PROBE_DIRECTIONAL_FILTER_PASS_COUNT; pass++) {
				const uint32_t scratch_index = (SCREEN_PROBE_DIRECTIONAL_FILTER_PASS_COUNT - 1u - pass) & 1u;
				const RID directional_filter_destination = directional_filter_scratch[scratch_index];
				directional_filter_sets[v][pass] = UniformSetCacheRD::get_singleton()->get_cache(
						hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_FILTER], 0,
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, surface),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, directional_filter_source),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, directional_filter_destination),
						RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 3, nearest_sampler),
						RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 4, rbgi->screen_probe_scene_data_ubo),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, directional_trace_count));
				directional_filter_source = directional_filter_destination;
			}
			directional_irradiance_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
					hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_IRRADIANCE], 0,
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, directional_filter_source),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, directional_irradiance),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 2, nearest_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, surface),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, directional_trace_count));
		} else {
			raw_radiance = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE, v, 0);
			if (!raw_radiance.is_valid()) {
				disable_hddagi_screen_probes(p_render_buffers);
				return;
			}
		}
		if (!directional_gather) {
			trace_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
					hddagi_shader.screen_probe_shader_version[trace_mode], 0,
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
					hddagi_shader.screen_probe_shader_version[trace_mode], 1,
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, sky_texture),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, linear_mipmap_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, hddagi->lightprobe_specular_tex));
		}
		if (debug_montage && !directional_gather) {
			trace_debug_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
					hddagi_shader.screen_probe_shader_version[trace_mode], 3,
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, trace_debug_slices[v]));
		}
		if (directional_gather) {
			resolve_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
					hddagi_shader.screen_probe_shader_version[resolve_mode], 0,
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, surface),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, raw_radiance),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, depth),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, normal_roughness),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, linear_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, resolved_radiance),
					RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, rbgi->screen_probe_scene_data_ubo),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 8, base_ambient),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 13, directional_adaptive_tile_data));
		} else {
			resolve_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
					hddagi_shader.screen_probe_shader_version[resolve_mode], 0,
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, surface),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, raw_radiance),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, depth),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, normal_roughness),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, nearest_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, resolved_radiance),
					RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, rbgi->screen_probe_scene_data_ubo));
		}
		if (svgf_active) {
			if (directional_gather) {
				svgf_resolve_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
						hddagi_shader.screen_probe_shader_version[svgf_resolve_mode], 0,
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, surface),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, raw_radiance),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, depth),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, normal_roughness),
						RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, linear_sampler),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, resolved_radiance),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, svgf_normal_roughness[v]),
						RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, rbgi->screen_probe_scene_data_ubo),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 8, base_ambient),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 9, svgf_velocity[v]),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 10, svgf_input[v]),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 11, svgf_view_z[v]),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 12, svgf_motion[v]),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 13, directional_adaptive_tile_data));
			} else {
				svgf_resolve_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
						hddagi_shader.screen_probe_shader_version[svgf_resolve_mode], 0,
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, surface),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, raw_radiance),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, depth),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, normal_roughness),
						RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, nearest_sampler),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, resolved_radiance),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, svgf_normal_roughness[v]),
						RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, rbgi->screen_probe_scene_data_ubo),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 9, svgf_velocity[v]),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 10, svgf_input[v]),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 11, svgf_view_z[v]),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 12, svgf_motion[v]));
			}
		}
		const RID apply_base_ambient = directional_gather ? base_ambient : texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
		apply_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
				hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_APPLY], 0,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, resolved_radiance),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, nearest_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, ambient_output),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, apply_base_ambient));
		if (svgf_active) {
			svgf_apply_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
					hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_APPLY_SVGF], 0,
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, svgf_output[v]),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, nearest_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, ambient_output),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, apply_base_ambient));
		}
		if (specular_reflections) {
			specular_raw[v] = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, RB_TEX_HDDAGI_SPECULAR_RAW, v, 0);
			specular_normal_roughness[v] = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, RB_TEX_HDDAGI_SPECULAR_NORMAL_ROUGHNESS, v, 0);
			specular_view_z[v] = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, RB_TEX_HDDAGI_SPECULAR_VIEW_Z, v, 0);
			specular_motion[v] = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, RB_TEX_HDDAGI_SPECULAR_MOTION, v, 0);
			specular_denoised[v] = p_render_buffers->get_texture_slice(RB_SCOPE_HDDAGI_SPECULAR_REFLECTIONS, RB_TEX_HDDAGI_SPECULAR_DENOISED, v, 0);
			const RID velocity = specular_motion_valid ? specular_velocity[v] : detail_hiz_fallback;
			const RID previous_color = specular_screen_radiance_valid ? p_previous_screen_color_slices[v] : detail_hiz_fallback;
			const RID previous_depth = specular_screen_radiance_valid ? p_previous_screen_depth_slices[v] : detail_hiz_fallback;
			const HDDAGIShader::ScreenProbeMode specular_trace_mode = HDDAGIShader::SCREEN_PROBE_MODE_SPECULAR_TRACE;
			specular_trace_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
					hddagi_shader.screen_probe_shader_version[specular_trace_mode], 0,
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, surface),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, specular_raw[v]),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, hddagi->voxel_bits_tex),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, hddagi->voxel_region_tex),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 4, hddagi->voxel_light_tex),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 5, linear_mipmap_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, hddagi->voxel_light_neighbour_data),
					RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, hddagi_ubo),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 8, hddagi->voxel_disocclusion_tex),
					RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 9, rbgi->screen_probe_scene_data_ubo),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 10, specular_detail_trace ? p_hiz_slices[v] : detail_hiz_fallback),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 11, normal_roughness),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 12, depth),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 13, velocity),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 14, specular_normal_roughness[v]),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 15, specular_view_z[v]),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 16, specular_motion[v]),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 17, previous_color),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 18, previous_depth));
			specular_trace_sky_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
					hddagi_shader.screen_probe_shader_version[specular_trace_mode], 1,
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, sky_texture),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, linear_mipmap_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, hddagi->lightprobe_specular_tex));
			specular_apply_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
					hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_SPECULAR_APPLY], 0,
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, specular_raw[v]),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, nearest_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, p_render_buffers->get_texture_slice(RB_SCOPE_GI, RB_TEX_REFLECTION_U32, v, 0)),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, specular_normal_roughness[v]));
			if (specular_svgf_active) {
				specular_denoised_apply_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
						hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_SPECULAR_APPLY], 0,
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, specular_denoised[v]),
						RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, nearest_sampler),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, p_render_buffers->get_texture_slice(RB_SCOPE_GI, RB_TEX_REFLECTION_U32, v, 0)),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, specular_normal_roughness[v]));
			}
			specular_sets_valid = specular_sets_valid && specular_raw[v].is_valid() && specular_normal_roughness[v].is_valid() && specular_view_z[v].is_valid() && specular_motion[v].is_valid() && specular_denoised[v].is_valid() &&
					RD::get_singleton()->uniform_set_is_valid(specular_trace_sets[v]) && RD::get_singleton()->uniform_set_is_valid(specular_trace_sky_sets[v]) && RD::get_singleton()->uniform_set_is_valid(specular_apply_sets[v]) &&
					(!specular_svgf_active || RD::get_singleton()->uniform_set_is_valid(specular_denoised_apply_sets[v]));
		}

		bool sets_valid = RD::get_singleton()->uniform_set_is_valid(surface_sets[v]) && RD::get_singleton()->uniform_set_is_valid(resolve_sets[v]) && RD::get_singleton()->uniform_set_is_valid(apply_sets[v]);
		if (svgf_active) {
			svgf_sets_valid = svgf_sets_valid && RD::get_singleton()->uniform_set_is_valid(svgf_resolve_sets[v]) && RD::get_singleton()->uniform_set_is_valid(svgf_apply_sets[v]);
		}
		if (directional_gather) {
			sets_valid = sets_valid && RD::get_singleton()->uniform_set_is_valid(directional_adaptive_mark_sets[v]) &&
					RD::get_singleton()->uniform_set_is_valid(directional_adaptive_spawn_sets[v]) && RD::get_singleton()->uniform_set_is_valid(directional_trace_sets[v]) &&
					RD::get_singleton()->uniform_set_is_valid(directional_trace_sky_sets[v]) && RD::get_singleton()->uniform_set_is_valid(directional_irradiance_sets[v]);
			for (uint32_t pass = 0; pass < SCREEN_PROBE_DIRECTIONAL_FILTER_PASS_COUNT; pass++) {
				sets_valid = sets_valid && RD::get_singleton()->uniform_set_is_valid(directional_filter_sets[v][pass]);
			}
		} else {
			sets_valid = sets_valid && RD::get_singleton()->uniform_set_is_valid(trace_sets[v]) && RD::get_singleton()->uniform_set_is_valid(trace_sky_sets[v]);
		}
		sets_valid = sets_valid && (!debug_montage || directional_gather || RD::get_singleton()->uniform_set_is_valid(trace_debug_sets[v]));
		if (!sets_valid) {
			disable_hddagi_screen_probes(p_render_buffers);
			WARN_PRINT_ONCE("HDDAGI screen probe resources could not be bound.");
			return;
		}
	}
	if (svgf_active && !svgf_sets_valid) {
		clear_svgf();
		svgf_active = false;
		WARN_PRINT_ONCE("HDDAGI screen-probe SVGF resources could not be bound; using the unfiltered signal.");
	}
	if (specular_reflections && !specular_sets_valid) {
		specular_reflections = false;
		rbgi->screen_probe_specular_svgf.clear();
		WARN_PRINT_ONCE("HDDAGI specular reflection resources could not be bound.");
	}

	RD::get_singleton()->draw_command_begin_label("HDDAGI Screen Probes");
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	if (irradiance_cache_active) {
		RENDER_TIMESTAMP("HDDAGI Screen Probe Irradiance Cache Maintenance");
		auto dispatch_irradiance_cache = [&](HDDAGIShader::ScreenProbeIrradianceCacheMode p_irradiance_cache_mode, uint32_t p_thread_count) {
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_irradiance_cache_pipeline[p_irradiance_cache_mode]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, irradiance_cache_maintenance_sets[p_irradiance_cache_mode], 0);
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_thread_count, 1, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);
		};
		if (irradiance_cache_dispatch_info.clear_required) {
			dispatch_irradiance_cache(HDDAGIShader::SCREEN_PROBE_IRRADIANCE_CACHE_CLEAR, irradiance_cache_dispatch_info.grid_count);
			rbgi->screen_probe_irradiance_cache.mark_clear_recorded();
		}
		dispatch_irradiance_cache(HDDAGIShader::SCREEN_PROBE_IRRADIANCE_CACHE_AGE, irradiance_cache_dispatch_info.capacity);
		if (irradiance_cache_multibounce) {
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, irradiance_cache_multibounce_resources_set, 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, irradiance_cache_multibounce_sky_set, 1);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, irradiance_cache_multibounce_set, 2);
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, irradiance_cache_dispatch_info.capacity, 1, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);
			dispatch_irradiance_cache(HDDAGIShader::SCREEN_PROBE_IRRADIANCE_CACHE_AGE, irradiance_cache_dispatch_info.capacity);
		}
		dispatch_irradiance_cache(HDDAGIShader::SCREEN_PROBE_IRRADIANCE_CACHE_PROCESS, irradiance_cache_dispatch_info.max_requests);
		dispatch_irradiance_cache(HDDAGIShader::SCREEN_PROBE_IRRADIANCE_CACHE_RESOLVE, irradiance_cache_dispatch_info.capacity);
		dispatch_irradiance_cache(HDDAGIShader::SCREEN_PROBE_IRRADIANCE_CACHE_RESET, 1u);
	}
	for (uint32_t v = 0; v < p_view_count; v++) {
		push_constant.flags = view_flags[v];
		push_constant.view_index = v;

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_SURFACE]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, surface_sets[v], 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, directional_gather ? directional_probe_atlas_size.x : probe_atlas_size.x, directional_gather ? directional_probe_atlas_size.y : probe_atlas_size.y, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		if (directional_gather) {
			auto dispatch_directional_adaptive = [&](HDDAGIShader::ScreenProbeMode p_screen_probe_mode, RID p_uniform_set, const char *p_timestamp) {
				RENDER_TIMESTAMP(p_timestamp);
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[p_screen_probe_mode]);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, p_uniform_set, 0);
				RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
				RD::get_singleton()->compute_list_dispatch_threads(compute_list, probe_atlas_size.x * SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_WORKGROUP_SIZE, probe_atlas_size.y * SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_WORKGROUP_SIZE, 1);
				RD::get_singleton()->compute_list_add_barrier(compute_list);
			};
			dispatch_directional_adaptive(HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_ADAPTIVE_MARK, directional_adaptive_mark_sets[v], "HDDAGI Directional Adaptive Probe Mark");
			dispatch_directional_adaptive(HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_ADAPTIVE_SPAWN, directional_adaptive_spawn_sets[v], "HDDAGI Directional Adaptive Probe Spawn");

			RENDER_TIMESTAMP("HDDAGI Directional Screen Probe Trace");
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_TRACE]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, directional_trace_sets[v], 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, directional_trace_sky_sets[v], 1);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, directional_atlas_size.x, directional_atlas_size.y, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);

			HDDAGIShader::ScreenProbePushConstant directional_filter_push_constant = push_constant;
			directional_filter_push_constant.history_depth_tolerance = 0.02f;
			directional_filter_push_constant.spatial_depth_tolerance_scale = 0.02f;
			directional_filter_push_constant.history_normal_threshold = 0.35f;
			for (uint32_t pass = 0; pass < SCREEN_PROBE_DIRECTIONAL_FILTER_PASS_COUNT; pass++) {
				RENDER_TIMESTAMP("HDDAGI Directional Screen Probe Filter");
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_FILTER]);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, directional_filter_sets[v][pass], 0);
				RD::get_singleton()->compute_list_set_push_constant(compute_list, &directional_filter_push_constant, sizeof(directional_filter_push_constant));
				RD::get_singleton()->compute_list_dispatch_threads(compute_list, directional_atlas_size.x, directional_atlas_size.y, 1);
				RD::get_singleton()->compute_list_add_barrier(compute_list);
			}

			RENDER_TIMESTAMP("HDDAGI Directional Screen Probe Irradiance");
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_IRRADIANCE]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, directional_irradiance_sets[v], 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, directional_atlas_size.x, directional_atlas_size.y, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);
		} else {
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[trace_mode]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, trace_sets[v], 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, trace_sky_sets[v], 1);
			if (irradiance_cache_active) {
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, irradiance_cache_trace_set, 2);
			}
			if (debug_montage) {
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, trace_debug_sets[v], 3);
			}
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, probe_atlas_size.x, probe_atlas_size.y, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);
		}

		const HDDAGIShader::ScreenProbeMode active_resolve_mode = svgf_active ? svgf_resolve_mode : resolve_mode;
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[active_resolve_mode]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, svgf_active ? svgf_resolve_sets[v] : resolve_sets[v], 0);
		HDDAGIShader::ScreenProbePushConstant resolve_push_constant = push_constant;
		if (directional_gather) {
			resolve_push_constant.history_depth_tolerance = 0.02f;
			resolve_push_constant.spatial_depth_tolerance_scale = 0.02f;
			resolve_push_constant.history_normal_threshold = 0.35f;
		}
		if (svgf_active) {
			svgf_prepare_push_constant.base = resolve_push_constant;
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &svgf_prepare_push_constant, sizeof(svgf_prepare_push_constant));
		} else {
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &resolve_push_constant, sizeof(resolve_push_constant));
		}
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, resolve_size.x, resolve_size.y, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		if (!svgf_active) {
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_APPLY]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, apply_sets[v], 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, resolve_size.x, resolve_size.y, 1);
		}
		if (specular_reflections) {
			specular_push_constant.base.view_index = v;
			specular_push_constant.eye_offset_exposure[0] = p_eye_offsets[v].x;
			specular_push_constant.eye_offset_exposure[1] = p_eye_offsets[v].y;
			specular_push_constant.eye_offset_exposure[2] = p_eye_offsets[v].z;
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_SPECULAR_TRACE]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, specular_trace_sets[v], 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, specular_trace_sky_sets[v], 1);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &specular_push_constant, sizeof(specular_push_constant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_gi_size.x, p_gi_size.y, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);
		}
	}
	RD::get_singleton()->compute_list_end();
	bool svgf_all_views_succeeded = false;
	if (svgf_active) {
		bool svgf_succeeded[2] = {};
		bool svgf_failed = false;
		for (uint32_t v = 0; v < p_view_count; v++) {
			HDDAGIScreenProbeSVGF::FrameSettings svgf_frame;
			svgf_frame.projection = p_projections[v];
			svgf_frame.previous_projection = svgf_history_valid ? hddagi->screen_probe_previous_svgf_projection[v] : p_projections[v];
			svgf_frame.camera_transform = p_cam_transform;
			svgf_frame.previous_camera_transform = svgf_history_valid ? hddagi->screen_probe_previous_cam_transform : p_cam_transform;
			svgf_frame.taa_jitter = p_taa_jitter;
			svgf_frame.previous_taa_jitter = svgf_history_valid ? hddagi->screen_probe_previous_taa_jitter : p_taa_jitter;
			svgf_frame.size = resolve_size;
			svgf_frame.denoising_range = SCREEN_PROBE_SVGF_DENOISING_RANGE;
			svgf_frame.quality = svgf_quality;
			svgf_frame.history_valid = svgf_history_valid;

			HDDAGIScreenProbeSVGF::Resources svgf_resources;
			svgf_resources.motion_vectors = svgf_motion[v];
			svgf_resources.normal_roughness = svgf_normal_roughness[v];
			svgf_resources.view_z = svgf_view_z[v];
			svgf_resources.diffuse_radiance_hit_distance = svgf_input[v];
			svgf_resources.output_diffuse_radiance_hit_distance = svgf_output[v];
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
		svgf_all_views_succeeded = !svgf_failed;

		compute_list = RD::get_singleton()->compute_list_begin();
		for (uint32_t v = 0; v < p_view_count; v++) {
			push_constant.view_index = v;
			const HDDAGIShader::ScreenProbeMode apply_mode = svgf_succeeded[v] ? HDDAGIShader::SCREEN_PROBE_MODE_APPLY_SVGF : HDDAGIShader::SCREEN_PROBE_MODE_APPLY;
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[apply_mode]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, svgf_succeeded[v] ? svgf_apply_sets[v] : apply_sets[v], 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, resolve_size.x, resolve_size.y, 1);
		}
		RD::get_singleton()->compute_list_end();
	}
	if (specular_reflections) {
		bool specular_svgf_succeeded[2] = {};
		bool specular_svgf_failed = false;
		if (specular_svgf_active) {
			for (uint32_t v = 0; v < p_view_count; v++) {
				HDDAGIScreenProbeSVGF::FrameSettings frame;
				frame.projection = p_projections[v];
				frame.previous_projection = specular_history_valid ? hddagi->screen_probe_previous_svgf_projection[v] : p_projections[v];
				frame.camera_transform = p_cam_transform;
				frame.previous_camera_transform = specular_history_valid ? hddagi->screen_probe_previous_cam_transform : p_cam_transform;
				frame.taa_jitter = p_taa_jitter;
				frame.previous_taa_jitter = specular_history_valid ? hddagi->screen_probe_previous_taa_jitter : p_taa_jitter;
				frame.size = p_gi_size;
				frame.denoising_range = SCREEN_PROBE_SPECULAR_DENOISING_RANGE;
				frame.quality = svgf_quality;
				frame.history_valid = specular_history_valid;
				frame.specular = true;
				frame.specular_full_resolution = p_gi_size == internal_size;

				HDDAGIScreenProbeSVGF::Resources resources;
				resources.motion_vectors = specular_motion[v];
				resources.normal_roughness = specular_normal_roughness[v];
				resources.view_z = specular_view_z[v];
				resources.diffuse_radiance_hit_distance = specular_raw[v];
				resources.output_diffuse_radiance_hit_distance = specular_denoised[v];
				const Error error = rbgi->screen_probe_specular_svgf.denoise(v, frame, resources);
				specular_svgf_succeeded[v] = error == OK;
				if (error != OK) {
					specular_svgf_failed = true;
					WARN_PRINT_ONCE(vformat("HDDAGI specular SVGF failed with error %d; using the unfiltered signal.", error));
				}
			}
			if (specular_svgf_failed) {
				rbgi->screen_probe_specular_svgf.clear();
			}
		}

		compute_list = RD::get_singleton()->compute_list_begin();
		for (uint32_t v = 0; v < p_view_count; v++) {
			specular_push_constant.base.view_index = v;
			specular_push_constant.eye_offset_exposure[0] = p_eye_offsets[v].x;
			specular_push_constant.eye_offset_exposure[1] = p_eye_offsets[v].y;
			specular_push_constant.eye_offset_exposure[2] = p_eye_offsets[v].z;
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_SPECULAR_APPLY]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, specular_svgf_succeeded[v] ? specular_denoised_apply_sets[v] : specular_apply_sets[v], 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &specular_push_constant, sizeof(specular_push_constant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_gi_size.x, p_gi_size.y, 1);
		}
		RD::get_singleton()->compute_list_end();
		rbgi->hddagi_specular_reflection_valid = true;
	}
	rbgi->screen_probe_debug_montage_valid = debug_montage;
	rbgi->screen_probe_debug_svgf_output_valid = debug_montage && svgf_all_views_succeeded;
	rbgi->screen_probe_debug_hiz_valid = debug_montage && detail_trace;
	rbgi->screen_probe_debug_directional_valid = debug_montage && directional_gather;
	rbgi->screen_probe_debug_radiance_scale = rbgi->screen_probe_debug_svgf_output_valid ? 1.0f / SCREEN_PROBE_SVGF_SCENE_TO_SIGNAL_SCALE : 1.0f;
	rbgi->screen_probe_debug_surface_layer_stride = directional_gather ? 2u : 1u;
	rbgi->screen_probe_debug_surface_history_slot = rbgi->screen_probe_debug_surface_layer_stride == 2u ? current_history_slot : 0u;
	rbgi->screen_probe_debug_hiz_mip_count = rbgi->screen_probe_debug_hiz_valid ? detail_trace_mip_count : 0u;
	RD::get_singleton()->draw_command_end_label();

	hddagi->screen_probe_history_initialized = true;
	hddagi->screen_probe_history_probe_size = probe_size;
	hddagi->screen_probe_history_gi_size = p_gi_size;
	hddagi->screen_probe_history_screen_size = internal_size;
	hddagi->screen_probe_history_view_count = p_view_count;
	hddagi->screen_probe_history_configuration = configuration;
	hddagi->screen_probe_history_slot = current_history_slot;
	hddagi->screen_probe_history_sequence = history_sequence;
	for (uint32_t v = 0; v < p_view_count; v++) {
		hddagi->screen_probe_previous_projection[v] = raster_correction * p_projections[v];
		hddagi->screen_probe_previous_temporal_projection[v] = temporal_correction * p_projections[v];
		hddagi->screen_probe_previous_svgf_projection[v] = p_projections[v];
	}
	hddagi->screen_probe_previous_taa_jitter = p_taa_jitter;
	hddagi->screen_probe_previous_cam_transform = p_cam_transform;
	hddagi->screen_probe_previous_exposure_normalization = p_exposure_normalization;
	hddagi->screen_probe_previous_camera_valid = true;
}
