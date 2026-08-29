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

#include "core/config/project_settings.h"
#include "core/templates/hashfuncs.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/rendering_server_globals.h"

using namespace RendererRD;

static constexpr float SCREEN_PROBE_IRRADIANCE_CACHE_MINIMUM_CELL_SIZE = 0.001f;
static constexpr float SCREEN_PROBE_SVGF_DENOISING_RANGE = 500000.0f;
static constexpr float SCREEN_PROBE_SVGF_SCENE_TO_SIGNAL_SCALE = 1.0f / 512.0f;
static constexpr float SCREEN_PROBE_SVGF_INPUT_RADIANCE_MAX = 128.0f;
static constexpr float SCREEN_PROBE_SVGF_INPUT_HIT_DISTANCE_MAX = 65504.0f;

void GI::disable_hddagi_screen_probes(Ref<RenderSceneBuffersRD> p_render_buffers) {
	if (p_render_buffers.is_null()) {
		return;
	}

	Ref<RenderBuffersGI> rbgi;
	if (p_render_buffers->has_custom_data(RB_SCOPE_GI)) {
		rbgi = p_render_buffers->get_custom_data(RB_SCOPE_GI);
		if (rbgi.is_valid()) {
			if (rbgi->screen_probe_scene_data_ubo.is_valid()) {
				RD::get_singleton()->free_rid(rbgi->screen_probe_scene_data_ubo);
				rbgi->screen_probe_scene_data_ubo = RID();
			}
			rbgi->screen_probe_svgf.clear();
			rbgi->screen_probe_irradiance_cache.clear();
		}
	}
	p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER);
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
	hddagi->screen_probe_previous_camera_valid = false;
}

void GI::process_hddagi_screen_probes(Ref<RenderSceneBuffersRD> p_render_buffers, const RID *p_normal_roughness_slices, const RID *p_hiz_slices, Size2i p_hiz_size, uint32_t p_hiz_mip_count, bool p_detail_trace, RID p_environment, uint32_t p_view_count, Size2i p_gi_size, const Projection *p_projections, const Vector2 &p_taa_jitter, const Transform3D &p_cam_transform, float p_exposure_normalization, float p_ibl_exposure_normalization, int p_probe_size, float p_normal_bias) {
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
			!hddagi->voxel_light_neighbour_data.is_valid() || !hddagi->voxel_disocclusion_tex.is_valid() ||
			!hddagi->lightprobe_specular_tex.is_valid()) {
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
	const int irradiance_cache_setting = GLOBAL_GET_CACHED(int, "rendering/global_illumination/hddagi/screen_probe_radiance_cache");
	const float irradiance_cache_minimum_cell_size = MAX(GLOBAL_GET_CACHED(float, "rendering/global_illumination/hddagi/screen_probe_radiance_cache_minimum_cell_size"), SCREEN_PROBE_IRRADIANCE_CACHE_MINIMUM_CELL_SIZE);
	const bool irradiance_cache_multibounce = hddagi->screen_probe_radiance_cache_multibounce_active;
	const int denoiser_setting = GLOBAL_GET_CACHED(int, "rendering/global_illumination/hddagi/screen_probe_denoiser");
	const HDDAGIScreenProbeSVGF::Quality svgf_quality = static_cast<HDDAGIScreenProbeSVGF::Quality>(CLAMP(GLOBAL_GET_CACHED(int, "rendering/global_illumination/hddagi/screen_probe_denoiser_quality"), 0, int(HDDAGIScreenProbeSVGF::QUALITY_MAX) - 1));

	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();
	RendererSceneRenderRD *scene_render = RendererSceneRenderRD::get_singleton();
	if (texture_storage == nullptr || material_storage == nullptr || scene_render == nullptr) {
		disable_hddagi_screen_probes(p_render_buffers);
		return;
	}

	bool detail_trace = p_detail_trace && p_hiz_slices != nullptr && p_hiz_size == internal_size && p_hiz_mip_count > 0;
	for (uint32_t v = 0; v < p_view_count && detail_trace; v++) {
		detail_trace = p_hiz_slices[v].is_valid();
	}
	const uint32_t detail_trace_mip_count = detail_trace ? CLAMP(p_hiz_mip_count, 1u, 16u) : 0u;

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

	Projection correction;
	correction.set_depth_correction(true);

	bool camera_cut = false;
	if (hddagi->screen_probe_previous_camera_valid) {
		const float translation = hddagi->screen_probe_previous_cam_transform.origin.distance_to(p_cam_transform.origin);
		const float translation_limit = MAX(4.0f, hddagi->min_cell_size * 16.0f);
		const float rotation = hddagi->screen_probe_previous_cam_transform.basis.get_rotation_quaternion().angle_to(p_cam_transform.basis.get_rotation_quaternion());
		camera_cut = translation > translation_limit || rotation > Math::deg_to_rad(55.0f);
		for (uint32_t v = 0; v < p_view_count && !camera_cut; v++) {
			if (p_projections[v].is_orthogonal() != hddagi->screen_probe_previous_svgf_projection[v].is_orthogonal()) {
				camera_cut = true;
				break;
			}
			for (int column = 0; column < 4 && !camera_cut; column++) {
				for (int row = 0; row < 4; row++) {
					if (Math::abs(p_projections[v].columns[column][row] - hddagi->screen_probe_previous_svgf_projection[v].columns[column][row]) > 0.35f) {
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
		auto screen_probe_mode_is_ready = [&](HDDAGIShader::ScreenProbeMode p_mode) {
			return hddagi_shader.screen_probe_shader_version[p_mode].is_valid() && hddagi_shader.screen_probe_pipeline[p_mode].is_valid();
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
				auto get_query_set = [&](HDDAGIShader::ScreenProbeMode p_mode) {
					return rbgi->screen_probe_irradiance_cache.get_uniform_set(hddagi_shader.screen_probe_shader_version[p_mode], 2);
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

	uint32_t configuration = hash_murmur3_one_32(candidate_count);
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

	const RD::DataFormat radiance_format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
	auto texture_matches = [&](const StringName &p_name, RD::DataFormat p_format, const Size2i &p_size, uint32_t p_layers) {
		if (!p_render_buffers->has_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, p_name)) {
			return false;
		}
		const RD::TextureFormat format = p_render_buffers->get_texture_format(RB_SCOPE_HDDAGI_SCREEN_PROBES, p_name);
		return format.format == p_format && format.width == uint32_t(p_size.x) && format.height == uint32_t(p_size.y) && format.array_layers == p_layers &&
				p_render_buffers->get_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, p_name).is_valid();
	};
	bool resources_valid = texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, RD::DATA_FORMAT_R32G32B32A32_UINT, probe_atlas_size, p_view_count) &&
			texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE, radiance_format, probe_atlas_size, p_view_count) &&
			texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE, radiance_format, p_gi_size, p_view_count);
	const bool resources_recreated = !resources_valid;
	if (!resources_valid) {
		p_render_buffers->clear_context(RB_SCOPE_HDDAGI_SCREEN_PROBES);
		const uint32_t texture_usage = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
		p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, RD::DATA_FORMAT_R32G32B32A32_UINT, texture_usage, RD::TEXTURE_SAMPLES_1, probe_atlas_size, p_view_count);
		p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE, radiance_format, texture_usage, RD::TEXTURE_SAMPLES_1, probe_atlas_size, p_view_count);
		p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBES, RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE, radiance_format, texture_usage, RD::TEXTURE_SAMPLES_1, p_gi_size, p_view_count);
		resources_valid = texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_SURFACE, RD::DATA_FORMAT_R32G32B32A32_UINT, probe_atlas_size, p_view_count) &&
				texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_RAW_RADIANCE, radiance_format, probe_atlas_size, p_view_count) &&
				texture_matches(RB_TEX_HDDAGI_SCREEN_PROBE_RESOLVED_RADIANCE, radiance_format, p_gi_size, p_view_count);
	}
	if (!resources_valid) {
		disable_hddagi_screen_probes(p_render_buffers);
		WARN_PRINT_ONCE("HDDAGI screen probe textures could not be allocated.");
		return;
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
		const HDDAGIShader::ScreenProbeMode svgf_resolve_mode = HDDAGIShader::SCREEN_PROBE_MODE_RESOLVE_SVGF;
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
				return format.format == p_format && format.width == uint32_t(p_gi_size.x) && format.height == uint32_t(p_gi_size.y) && format.array_layers == p_view_count &&
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
				p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_INPUT, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, svgf_texture_usage, RD::TEXTURE_SAMPLES_1, p_gi_size, p_view_count);
				p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_NORMAL_ROUGHNESS, RD::DATA_FORMAT_R8G8B8A8_UNORM, svgf_texture_usage, RD::TEXTURE_SAMPLES_1, p_gi_size, p_view_count);
				p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_VIEW_Z, RD::DATA_FORMAT_R32_SFLOAT, svgf_texture_usage, RD::TEXTURE_SAMPLES_1, p_gi_size, p_view_count);
				p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_MOTION, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, svgf_texture_usage, RD::TEXTURE_SAMPLES_1, p_gi_size, p_view_count);
				p_render_buffers->create_texture(RB_SCOPE_HDDAGI_SCREEN_PROBE_DENOISER, RB_TEX_HDDAGI_SCREEN_PROBE_DENOISER_OUTPUT, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, svgf_texture_usage, RD::TEXTURE_SAMPLES_1, p_gi_size, p_view_count);
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

	const bool history_configuration_valid = !resources_recreated && hddagi->screen_probe_history_initialized && hddagi->screen_probe_history_probe_size == probe_size &&
			hddagi->screen_probe_history_gi_size == p_gi_size && hddagi->screen_probe_history_screen_size == internal_size &&
			hddagi->screen_probe_history_view_count == p_view_count && hddagi->screen_probe_history_configuration == configuration;
	const bool common_history_valid = history_configuration_valid && hddagi->screen_probe_previous_camera_valid && !camera_cut;
	const bool svgf_history_valid = svgf_active && common_history_valid && !svgf_resources_recreated;

	ScreenProbeSceneData scene_data = {};
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
	push_constant.flags = (detail_trace ? HDDAGIShader::ScreenProbePushConstant::FLAG_DETAIL_TRACE : 0u) |
			(guided_sampling ? HDDAGIShader::ScreenProbePushConstant::FLAG_GUIDED_SAMPLING : 0u);
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

	RID surface_sets[2];
	RID trace_sets[2];
	RID trace_sky_sets[2];
	RID resolve_sets[2];
	RID svgf_resolve_sets[2];
	RID apply_sets[2];
	RID svgf_apply_sets[2];
	const HDDAGIShader::ScreenProbeMode trace_mode = irradiance_cache_active ? HDDAGIShader::SCREEN_PROBE_MODE_TRACE_IRRADIANCE_CACHE : HDDAGIShader::SCREEN_PROBE_MODE_TRACE;
	const HDDAGIShader::ScreenProbeMode resolve_mode = HDDAGIShader::SCREEN_PROBE_MODE_RESOLVE;
	const HDDAGIShader::ScreenProbeMode svgf_resolve_mode = HDDAGIShader::SCREEN_PROBE_MODE_RESOLVE_SVGF;
	bool pipelines_valid = hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_SURFACE].is_valid() &&
			hddagi_shader.screen_probe_pipeline[resolve_mode].is_valid() && hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_APPLY].is_valid();
	if (svgf_active) {
		pipelines_valid = pipelines_valid && hddagi_shader.screen_probe_pipeline[svgf_resolve_mode].is_valid() && hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_APPLY_SVGF].is_valid();
	}
	pipelines_valid = pipelines_valid && hddagi_shader.screen_probe_pipeline[trace_mode].is_valid();
	if (!pipelines_valid) {
		disable_hddagi_screen_probes(p_render_buffers);
		return;
	}
	bool svgf_sets_valid = true;
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
		resolve_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
				hddagi_shader.screen_probe_shader_version[resolve_mode], 0,
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, surface),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, raw_radiance),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, depth),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, normal_roughness),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, nearest_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, resolved_radiance),
				RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, rbgi->screen_probe_scene_data_ubo));
		if (svgf_active) {
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
		apply_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
				hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_APPLY], 0,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, resolved_radiance),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, nearest_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, ambient));
		if (svgf_active) {
			svgf_apply_sets[v] = UniformSetCacheRD::get_singleton()->get_cache(
					hddagi_shader.screen_probe_shader_version[HDDAGIShader::SCREEN_PROBE_MODE_APPLY_SVGF], 0,
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, svgf_output[v]),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, nearest_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, ambient));
		}

		bool sets_valid = RD::get_singleton()->uniform_set_is_valid(surface_sets[v]) && RD::get_singleton()->uniform_set_is_valid(trace_sets[v]) &&
				RD::get_singleton()->uniform_set_is_valid(trace_sky_sets[v]) && RD::get_singleton()->uniform_set_is_valid(resolve_sets[v]) &&
				RD::get_singleton()->uniform_set_is_valid(apply_sets[v]);
		if (svgf_active) {
			svgf_sets_valid = svgf_sets_valid && RD::get_singleton()->uniform_set_is_valid(svgf_resolve_sets[v]) && RD::get_singleton()->uniform_set_is_valid(svgf_apply_sets[v]);
		}
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

	RD::get_singleton()->draw_command_begin_label("HDDAGI Screen Probes");
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	if (irradiance_cache_active) {
		RENDER_TIMESTAMP("HDDAGI Screen Probe Irradiance Cache Maintenance");
		auto dispatch_irradiance_cache = [&](HDDAGIShader::ScreenProbeIrradianceCacheMode p_mode, uint32_t p_thread_count) {
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_irradiance_cache_pipeline[p_mode]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, irradiance_cache_maintenance_sets[p_mode], 0);
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
		push_constant.view_index = v;

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_SURFACE]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, surface_sets[v], 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, probe_atlas_size.x, probe_atlas_size.y, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[trace_mode]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, trace_sets[v], 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, trace_sky_sets[v], 1);
		if (irradiance_cache_active) {
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, irradiance_cache_trace_set, 2);
		}
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, probe_atlas_size.x, probe_atlas_size.y, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		const HDDAGIShader::ScreenProbeMode active_resolve_mode = svgf_active ? svgf_resolve_mode : resolve_mode;
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[active_resolve_mode]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, svgf_active ? svgf_resolve_sets[v] : resolve_sets[v], 0);
		if (svgf_active) {
			svgf_prepare_push_constant.base = push_constant;
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &svgf_prepare_push_constant, sizeof(svgf_prepare_push_constant));
		} else {
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		}
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_gi_size.x, p_gi_size.y, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		if (!svgf_active) {
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[HDDAGIShader::SCREEN_PROBE_MODE_APPLY]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, apply_sets[v], 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_gi_size.x, p_gi_size.y, 1);
		}
	}
	RD::get_singleton()->compute_list_end();
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
			svgf_frame.size = p_gi_size;
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

		compute_list = RD::get_singleton()->compute_list_begin();
		for (uint32_t v = 0; v < p_view_count; v++) {
			push_constant.view_index = v;
			const HDDAGIShader::ScreenProbeMode apply_mode = svgf_succeeded[v] ? HDDAGIShader::SCREEN_PROBE_MODE_APPLY_SVGF : HDDAGIShader::SCREEN_PROBE_MODE_APPLY;
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, hddagi_shader.screen_probe_pipeline[apply_mode]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, svgf_succeeded[v] ? svgf_apply_sets[v] : apply_sets[v], 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_gi_size.x, p_gi_size.y, 1);
		}
		RD::get_singleton()->compute_list_end();
	}
	RD::get_singleton()->draw_command_end_label();

	hddagi->screen_probe_history_initialized = true;
	hddagi->screen_probe_history_probe_size = probe_size;
	hddagi->screen_probe_history_gi_size = p_gi_size;
	hddagi->screen_probe_history_screen_size = internal_size;
	hddagi->screen_probe_history_view_count = p_view_count;
	hddagi->screen_probe_history_configuration = configuration;
	for (uint32_t v = 0; v < p_view_count; v++) {
		hddagi->screen_probe_previous_svgf_projection[v] = p_projections[v];
	}
	hddagi->screen_probe_previous_taa_jitter = p_taa_jitter;
	hddagi->screen_probe_previous_cam_transform = p_cam_transform;
	hddagi->screen_probe_previous_camera_valid = true;
}
