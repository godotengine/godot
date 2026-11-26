/**************************************************************************/
/*  hddagi_screen_probe_svgf.cpp                                          */
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

#include "hddagi_screen_probe_svgf.h"

#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "servers/rendering/renderer_rd/shaders/environment/hddagi_screen_probe_svgf.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/renderer_scene_render.h"
#include "servers/rendering/rendering_server_globals.h"
#include "servers/rendering/storage/utilities.h"

namespace {

enum SVGFMode {
	SVGF_MODE_TEMPORAL,
	SVGF_MODE_HISTORY_FIX,
	SVGF_MODE_ATROUS_DIFFUSE_0,
	SVGF_MODE_ATROUS_DIFFUSE_1,
	SVGF_MODE_ATROUS_DIFFUSE_2,
	SVGF_MODE_ATROUS_DIFFUSE_3,
	SVGF_MODE_ATROUS_DIFFUSE_4,
	SVGF_MODE_SPECULAR_RECONSTRUCT,
	SVGF_MODE_SPECULAR_TEMPORAL,
	SVGF_MODE_SPECULAR_SPATIAL,
	SVGF_MODE_MAX,
};

static constexpr uint32_t MAX_ATROUS_ITERATION_COUNT = 5u;
static constexpr uint32_t SHARED_ATROUS_ITERATION_COUNT = 2u;
static constexpr uint32_t DIFFUSE_ATROUS_GROUP_SIZE_X = 16u;
static constexpr uint32_t DIFFUSE_ATROUS_GROUP_SIZE_Y = 8u;
static_assert(SVGF_MODE_ATROUS_DIFFUSE_4 - SVGF_MODE_ATROUS_DIFFUSE_0 + 1 == MAX_ATROUS_ITERATION_COUNT);
static constexpr uint32_t HISTORY_FIX_FRAME_COUNT = 3u;
static constexpr uint32_t HISTORY_FIX_BASE_PIXEL_STRIDE = 14u;

struct SVGFDenoisePreset {
	float view_z_limit;
	float normal_threshold;
	float depth_tolerance;
	float luminance_phi;
	float normal_power;
	float plane_threshold_scale;
	float variance_floor;
};

static constexpr SVGFDenoisePreset SVGF_DIFFUSE_PRESET = { 500000.0f, 0.9f, 0.005f, 2.0f, 8.0f, 0.005f, 4.0e-5f };
static constexpr SVGFDenoisePreset SVGF_SPECULAR_PRESET = { 65504.0f, 0.95f, 0.0025f, 2.5f, 12.0f, 0.003f, 4.0e-5f };

static constexpr uint32_t SVGF_CONTROL_HISTORY_VALID = 1u << 0;
static constexpr uint32_t SVGF_CONTROL_SPECULAR_FULL_RESOLUTION = 1u << 2;
static constexpr uint32_t SVGF_CONTROL_UNFILTERED_MIRROR = 1u << 3;
static constexpr uint32_t SVGF_CONTROL_LIGHTS_CHANGED = 1u << 4;

struct alignas(16) SVGFFrameData {
	float current_inv_projection[16] = {};
	float previous_inv_projection[16] = {};
	float current_view_to_previous_view[16] = {};
	float current_view_to_world[16] = {};
	float previous_view_to_world[16] = {};
	float taa_jitter[4] = {};
};

struct alignas(16) SVGFPushConstant {
	uint32_t control[4] = {};
	float tuning[4] = {};
	float atrous[4] = {};
};

} // namespace

struct HDDAGIScreenProbeSVGF::Implementation {
	struct ViewContext {
		Size2i size;
		RID history_signal[2];
		RID history_moments[2];
		RID history_normal_roughness[2];
		RID filter_scratch;
		RID frame_data_buffer;
		uint32_t history_slot = 0;
		bool history_initialized = false;
		bool previous_lights_changed = false;
	};

	HddagiScreenProbeSvgfShaderRD shader;
	RID shader_version;
	RID pipelines[SVGF_MODE_MAX];
	ViewContext views[RendererSceneRender::MAX_RENDER_VIEWS];
	bool shader_initialized = false;
	Error shader_status = ERR_CANT_CREATE;

	Error initialize_shader(bool p_specular) {
		if (shader_initialized) {
			return shader_status;
		}

		RenderingDevice *rd = RenderingDevice::get_singleton();
		ERR_FAIL_NULL_V(rd, ERR_UNAVAILABLE);
		ERR_FAIL_COND_V_MSG(!HDDAGIScreenProbeSVGF::is_supported(), ERR_UNAVAILABLE, "The RenderingDevice cannot run the HDDAGI screen-probe SVGF shader.");

		Vector<String> modes;
		modes.push_back("\n#define MODE_TEMPORAL\n");
		modes.push_back("\n#define MODE_HISTORY_FIX\n");
		for (uint32_t iteration = 0; iteration < MAX_ATROUS_ITERATION_COUNT; iteration++) {
			modes.push_back(vformat("\n#define MODE_ATROUS\n#define ATROUS_STEP_SIZE %d\n", 1u << iteration));
		}
		modes.push_back("\n#define MODE_SPECULAR_RECONSTRUCT\n");
		modes.push_back("\n#define MODE_TEMPORAL\n#define SVGF_SPECULAR\n");
		modes.push_back("\n#define MODE_SPECULAR_SPATIAL\n");
		shader.initialize(modes);
		for (uint32_t mode = 0; mode < SVGF_MODE_MAX; mode++) {
			shader.set_variant_enabled(mode, (mode >= SVGF_MODE_SPECULAR_RECONSTRUCT) == p_specular);
		}
		shader_version = shader.version_create();
		shader_initialized = true;
		ERR_FAIL_COND_V_MSG(shader_version.is_null(), ERR_CANT_CREATE, "Failed to create the HDDAGI screen-probe SVGF shader version.");

		for (uint32_t mode = 0; mode < SVGF_MODE_MAX; mode++) {
			if ((mode >= SVGF_MODE_SPECULAR_RECONSTRUCT) != p_specular) {
				continue;
			}
			const RID mode_shader = shader.version_get_shader(shader_version, mode);
			if (mode_shader.is_null()) {
				return ERR_CANT_CREATE;
			}
			pipelines[mode] = rd->compute_pipeline_create(mode_shader);
			if (pipelines[mode].is_null()) {
				return ERR_CANT_CREATE;
			}
		}

		shader_status = OK;
		return OK;
	}

	RID create_texture(const Size2i &p_size, RD::DataFormat p_format, const String &p_name) const {
		RenderingDevice *rd = RenderingDevice::get_singleton();
		RD::TextureFormat texture_format;
		texture_format.width = p_size.x;
		texture_format.height = p_size.y;
		texture_format.format = p_format;
		texture_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		RID texture = rd->texture_create(texture_format, RD::TextureView());
		if (texture.is_valid()) {
			rd->set_resource_name(texture, p_name);
		}
		return texture;
	}

	void free_view(ViewContext &r_view) const {
		RenderingDevice *rd = RenderingDevice::get_singleton();
		if (rd != nullptr) {
			const RID resources[] = {
				r_view.history_signal[0],
				r_view.history_signal[1],
				r_view.history_moments[0],
				r_view.history_moments[1],
				r_view.history_normal_roughness[0],
				r_view.history_normal_roughness[1],
				r_view.filter_scratch,
				r_view.frame_data_buffer,
			};
			for (const RID &resource : resources) {
				if (resource.is_valid()) {
					rd->free_rid(resource);
				}
			}
		}
		r_view = ViewContext();
	}

	void clear_views() {
		for (ViewContext &view : views) {
			if (view.size != Size2i()) {
				free_view(view);
			}
		}
	}

	Error ensure_view(uint32_t p_view_id, const FrameSettings &p_frame) {
		RenderingDevice *rd = RenderingDevice::get_singleton();
		ViewContext &view = views[p_view_id];
		if (view.size == p_frame.size) {
			return OK;
		}
		if (view.size != Size2i()) {
			free_view(view);
		}
		view.size = p_frame.size;
		for (uint32_t slot = 0; slot < 2; slot++) {
			view.history_signal[slot] = create_texture(p_frame.size, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, vformat("HDDAGI SVGF View %d History Signal %d", p_view_id, slot));
			view.history_moments[slot] = create_texture(p_frame.size, RD::DATA_FORMAT_R16G16_SFLOAT, vformat("HDDAGI SVGF View %d History Moments %d", p_view_id, slot));
			view.history_normal_roughness[slot] = create_texture(p_frame.size, RD::DATA_FORMAT_R8G8B8A8_UNORM, vformat("HDDAGI SVGF View %d History Normal Roughness %d", p_view_id, slot));
		}
		if (!p_frame.specular) {
			view.filter_scratch = create_texture(p_frame.size, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, vformat("HDDAGI SVGF View %d Filter Scratch", p_view_id));
		}
		view.frame_data_buffer = rd->uniform_buffer_create(sizeof(SVGFFrameData));
		if (view.frame_data_buffer.is_valid()) {
			rd->set_resource_name(view.frame_data_buffer, vformat("HDDAGI SVGF View %d Frame Data", p_view_id));
		}

		bool resources_valid = (p_frame.specular || view.filter_scratch.is_valid()) && view.frame_data_buffer.is_valid();
		for (uint32_t slot = 0; slot < 2; slot++) {
			resources_valid = resources_valid && view.history_signal[slot].is_valid() && view.history_moments[slot].is_valid() &&
					view.history_normal_roughness[slot].is_valid();
		}
		if (!resources_valid) {
			free_view(view);
			return ERR_OUT_OF_MEMORY;
		}

		return OK;
	}

	Error denoise(uint32_t p_view_id, const HDDAGIScreenProbeSVGF::FrameSettings &p_frame, const HDDAGIScreenProbeSVGF::Resources &p_resources) {
		ERR_FAIL_COND_V_MSG(!p_resources.is_valid(), ERR_INVALID_PARAMETER, "The HDDAGI screen-probe SVGF resources are incomplete or alias each other.");
		ERR_FAIL_COND_V(p_view_id >= RendererSceneRender::MAX_RENDER_VIEWS, ERR_INVALID_PARAMETER);

		Error error = initialize_shader(p_frame.specular);
		if (error != OK) {
			return error;
		}
		error = ensure_view(p_view_id, p_frame);
		if (error != OK) {
			return error;
		}

		ViewContext &view = views[p_view_id];
		SVGFFrameData frame_data;
		Projection raster_correction;
		raster_correction.set_depth_correction(true);
		raster_correction.add_jitter_offset(p_frame.taa_jitter);
		const Projection current_inv_projection = (raster_correction * p_frame.projection).inverse();
		Projection previous_raster_correction;
		previous_raster_correction.set_depth_correction(true);
		previous_raster_correction.add_jitter_offset(p_frame.previous_taa_jitter);
		const Projection previous_inv_projection = (previous_raster_correction * p_frame.previous_projection).inverse();
		const Transform3D current_view_to_previous_view = p_frame.previous_camera_transform.affine_inverse() * p_frame.camera_transform;
		RendererRD::MaterialStorage::store_camera(current_inv_projection, frame_data.current_inv_projection);
		RendererRD::MaterialStorage::store_camera(previous_inv_projection, frame_data.previous_inv_projection);
		RendererRD::MaterialStorage::store_transform(current_view_to_previous_view, frame_data.current_view_to_previous_view);
		RendererRD::MaterialStorage::store_transform(p_frame.camera_transform, frame_data.current_view_to_world);
		RendererRD::MaterialStorage::store_transform(p_frame.previous_camera_transform, frame_data.previous_view_to_world);
		frame_data.taa_jitter[0] = p_frame.taa_jitter.x;
		frame_data.taa_jitter[1] = p_frame.taa_jitter.y;
		frame_data.taa_jitter[2] = p_frame.previous_taa_jitter.x;
		frame_data.taa_jitter[3] = p_frame.previous_taa_jitter.y;

		RenderingDevice *rd = RenderingDevice::get_singleton();
		ERR_FAIL_NULL_V(rd, ERR_UNAVAILABLE);
		RENDER_TIMESTAMP(p_frame.specular ? "HDDAGI GGX Scene Upload Begin" : "HDDAGI SVGF Scene Upload Begin");
		error = rd->buffer_update(view.frame_data_buffer, 0, sizeof(frame_data), &frame_data);
		if (error != OK) {
			return error;
		}
		RENDER_TIMESTAMP(p_frame.specular ? "HDDAGI GGX Scene Upload End" : "HDDAGI SVGF Scene Upload End");

		// SubViewports can skip global frames, so advance history after successful local submissions.
		const uint32_t previous_slot = view.history_slot;
		const uint32_t current_slot = previous_slot ^ 1u;
		const bool history_valid = p_frame.history_valid && view.history_initialized;
		const uint32_t iteration_count = p_frame.specular ? 1u : HDDAGIScreenProbeSVGF::get_atrous_iteration_count(p_frame.quality);

		SVGFPushConstant push_constant = {};
		const bool unfiltered_mirror = p_frame.specular && (p_frame.specular_full_resolution || (p_frame.taa_jitter == Vector2() && p_frame.previous_taa_jitter == Vector2()));
		push_constant.control[0] = (history_valid ? SVGF_CONTROL_HISTORY_VALID : 0u) | (p_frame.specular_full_resolution ? SVGF_CONTROL_SPECULAR_FULL_RESOLUTION : 0u) | (unfiltered_mirror ? SVGF_CONTROL_UNFILTERED_MIRROR : 0u) | ((p_frame.lights_changed || view.previous_lights_changed) ? SVGF_CONTROL_LIGHTS_CHANGED : 0u);
		push_constant.control[3] = (p_frame.projection.is_orthogonal() ? 1u : 0u) | (p_frame.previous_projection.is_orthogonal() ? 2u : 0u);
		const SVGFDenoisePreset &preset = p_frame.specular ? SVGF_SPECULAR_PRESET : SVGF_DIFFUSE_PRESET;
		push_constant.tuning[0] = preset.view_z_limit;
		push_constant.tuning[1] = preset.normal_threshold;
		push_constant.tuning[2] = preset.depth_tolerance;
		push_constant.tuning[3] = preset.luminance_phi;
		push_constant.atrous[0] = preset.normal_power;
		push_constant.atrous[1] = preset.plane_threshold_scale;
		push_constant.atrous[2] = preset.variance_floor;
		const int64_t max_history_frames = p_frame.specular
				? GLOBAL_GET_CACHED(int64_t, "rendering/global_illumination/hddagi/screen_probe_specular_max_history_frames")
				: GLOBAL_GET_CACHED(int64_t, "rendering/global_illumination/hddagi/screen_probe_diffuse_max_history_frames");
		push_constant.atrous[3] = float(CLAMP(max_history_frames, 1, 255));

		const RID nearest_sampler = RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
		UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
		ERR_FAIL_NULL_V(uniform_set_cache, ERR_UNAVAILABLE);
		const SVGFMode temporal_mode = p_frame.specular ? SVGF_MODE_SPECULAR_TEMPORAL : SVGF_MODE_TEMPORAL;
		const RID temporal_shader = shader.version_get_shader(shader_version, temporal_mode);
		const bool specular_reconstruction = p_frame.specular && p_frame.quality != HDDAGIScreenProbeSVGF::QUALITY_LOW;
		RID specular_reconstruct_set;
		RID temporal_signal_input = p_resources.radiance_hit_distance;
		if (specular_reconstruction) {
			const RID specular_reconstruct_shader = shader.version_get_shader(shader_version, SVGF_MODE_SPECULAR_RECONSTRUCT);
			specular_reconstruct_set = uniform_set_cache->get_cache(
					specular_reconstruct_shader, 0,
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, p_resources.radiance_hit_distance),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, p_resources.normal_roughness),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, p_resources.view_z),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 3, nearest_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, p_resources.output_radiance_hit_distance));
			temporal_signal_input = p_resources.output_radiance_hit_distance;
		}

		const RID temporal_signal_destination = p_frame.specular ? view.history_signal[current_slot] : view.filter_scratch;
		const RID temporal_moments_destination = p_frame.specular ? view.history_moments[current_slot] : p_resources.output_radiance_hit_distance;
		const RID temporal_set = uniform_set_cache->get_cache(
				temporal_shader, 0,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, temporal_signal_input),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, p_resources.normal_roughness),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, p_resources.view_z),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, p_resources.motion_vectors),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 4, view.history_signal[previous_slot]),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 5, view.history_moments[previous_slot]),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 6, view.history_normal_roughness[previous_slot]),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 7, p_resources.previous_view_z),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 8, nearest_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 9, temporal_signal_destination),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 10, temporal_moments_destination),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 11, view.history_normal_roughness[current_slot]));
		const RID temporal_frame_data_set = uniform_set_cache->get_cache(
				temporal_shader, 1,
				RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, view.frame_data_buffer));
		RID history_fix_set;
		RID history_fix_frame_data_set;
		if (!p_frame.specular) {
			const RID history_fix_shader = shader.version_get_shader(shader_version, SVGF_MODE_HISTORY_FIX);
			history_fix_set = uniform_set_cache->get_cache(
					history_fix_shader, 0,
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, temporal_signal_destination),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, temporal_moments_destination),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, view.history_normal_roughness[current_slot]),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, p_resources.view_z),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, nearest_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, view.history_signal[current_slot]),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, view.history_moments[current_slot]));
			history_fix_frame_data_set = uniform_set_cache->get_cache(
					history_fix_shader, 1,
					RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, view.frame_data_buffer));
		}

		RID atrous_sets[MAX_ATROUS_ITERATION_COUNT];
		RID signal_source = view.history_signal[current_slot];
		const bool first_iteration_writes_output = (iteration_count & 1u) != 0u;
		for (uint32_t iteration = 0; iteration < iteration_count; iteration++) {
			const uint32_t atrous_mode = p_frame.specular ? uint32_t(SVGF_MODE_SPECULAR_SPATIAL) : SVGF_MODE_ATROUS_DIFFUSE_0 + iteration;
			const RID atrous_shader = shader.version_get_shader(shader_version, atrous_mode);
			const bool write_output = ((iteration & 1u) == 0u) == first_iteration_writes_output;
			const RID signal_destination = write_output ? p_resources.output_radiance_hit_distance : view.filter_scratch;
			auto get_spatial_set = [&](auto... p_extra_guides) {
				return uniform_set_cache->get_cache(
						atrous_shader, 0,
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, signal_source),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, view.history_moments[current_slot]),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, view.history_normal_roughness[current_slot]),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, p_resources.view_z),
						RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, nearest_sampler),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, signal_destination), p_extra_guides...);
			};
			if (p_frame.specular) {
				atrous_sets[iteration] = get_spatial_set(
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 6, p_resources.radiance_hit_distance),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 7, p_resources.normal_roughness),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 8, p_resources.motion_vectors));
			} else {
				atrous_sets[iteration] = get_spatial_set();
			}
			signal_source = signal_destination;
		}

		if (!temporal_set.is_valid() || !temporal_frame_data_set.is_valid() || (specular_reconstruction && !specular_reconstruct_set.is_valid()) || (!p_frame.specular && (!history_fix_set.is_valid() || !history_fix_frame_data_set.is_valid()))) {
			return ERR_CANT_CREATE;
		}
		for (uint32_t iteration = 0; iteration < iteration_count; iteration++) {
			if (!atrous_sets[iteration].is_valid()) {
				return ERR_CANT_CREATE;
			}
		}

		rd->draw_command_begin_label("HDDAGI Screen Probe SVGF");
		RD::ComputeListID compute_list = rd->compute_list_begin();

		if (specular_reconstruction) {
			push_constant.control[2] = uint32_t(p_frame.quality) << 2u;
			RENDER_TIMESTAMP("HDDAGI GGX Reconstruction");
			rd->compute_list_bind_compute_pipeline(compute_list, pipelines[SVGF_MODE_SPECULAR_RECONSTRUCT]);
			rd->compute_list_bind_uniform_set(compute_list, specular_reconstruct_set, 0);
			rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			rd->compute_list_dispatch_threads(compute_list, p_frame.size.x, p_frame.size.y, 1);
			rd->compute_list_add_barrier(compute_list);
		}
		RENDER_TIMESTAMP(p_frame.specular ? "HDDAGI GGX Temporal" : "HDDAGI SVGF Temporal");
		rd->compute_list_bind_compute_pipeline(compute_list, pipelines[temporal_mode]);
		rd->compute_list_bind_uniform_set(compute_list, temporal_set, 0);
		rd->compute_list_bind_uniform_set(compute_list, temporal_frame_data_set, 1);
		rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		rd->compute_list_dispatch_threads(compute_list, p_frame.size.x, p_frame.size.y, 1);
		rd->compute_list_add_barrier(compute_list);

		if (!p_frame.specular) {
			RENDER_TIMESTAMP("HDDAGI SVGF History Fix");
			push_constant.control[1] = HISTORY_FIX_BASE_PIXEL_STRIDE;
			push_constant.control[2] = HISTORY_FIX_FRAME_COUNT;
			rd->compute_list_bind_compute_pipeline(compute_list, pipelines[SVGF_MODE_HISTORY_FIX]);
			rd->compute_list_bind_uniform_set(compute_list, history_fix_set, 0);
			rd->compute_list_bind_uniform_set(compute_list, history_fix_frame_data_set, 1);
			rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			rd->compute_list_dispatch_threads(compute_list, p_frame.size.x, p_frame.size.y, 1);
			rd->compute_list_add_barrier(compute_list);
		}

		for (uint32_t iteration = 0; iteration < iteration_count; iteration++) {
			RENDER_TIMESTAMP(p_frame.specular ? String("HDDAGI GGX Spatial") : "HDDAGI SVGF A-Trous Step " + itos(uint64_t(1) << iteration));
			push_constant.control[1] = 1u << iteration;
			push_constant.control[2] = (iteration == 0u ? 1u : 0u) | (iteration + 1u == iteration_count ? 2u : 0u) | (p_frame.specular ? uint32_t(p_frame.quality) << 2u : 0u);
			const uint32_t atrous_mode = p_frame.specular ? uint32_t(SVGF_MODE_SPECULAR_SPATIAL) : SVGF_MODE_ATROUS_DIFFUSE_0 + iteration;
			rd->compute_list_bind_compute_pipeline(compute_list, pipelines[atrous_mode]);
			rd->compute_list_bind_uniform_set(compute_list, atrous_sets[iteration], 0);
			rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			if (!p_frame.specular && iteration < SHARED_ATROUS_ITERATION_COUNT) {
				const uint32_t tile_width = DIFFUSE_ATROUS_GROUP_SIZE_X << iteration;
				const uint32_t tile_height = DIFFUSE_ATROUS_GROUP_SIZE_Y << iteration;
				rd->compute_list_dispatch_threads(compute_list, Math::division_round_up(uint32_t(p_frame.size.x), tile_width) * tile_width,
						Math::division_round_up(uint32_t(p_frame.size.y), tile_height) * tile_height, 1);
			} else {
				rd->compute_list_dispatch_threads(compute_list, p_frame.size.x, p_frame.size.y, 1);
			}
			if (iteration + 1u < iteration_count) {
				rd->compute_list_add_barrier(compute_list);
			}
		}
		rd->compute_list_end();
		RENDER_TIMESTAMP(p_frame.specular ? "HDDAGI GGX Denoise End" : "HDDAGI SVGF End");

		rd->draw_command_end_label();

		view.previous_lights_changed = p_frame.lights_changed;
		view.history_initialized = true;
		view.history_slot = current_slot;
		return OK;
	}

	~Implementation() {
		clear_views();
		if (shader_version.is_valid()) {
			shader.version_free(shader_version);
		}
	}
};

bool HDDAGIScreenProbeSVGF::is_supported() {
	RenderingDevice *rd = RenderingDevice::get_singleton();
	if (rd == nullptr) {
		return false;
	}
	const uint32_t sampled_storage_usage = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
	return rd->texture_is_format_supported_for_usage(RD::DATA_FORMAT_R16G16B16A16_SFLOAT, sampled_storage_usage) &&
			rd->texture_is_format_supported_for_usage(RD::DATA_FORMAT_R8G8B8A8_UNORM, sampled_storage_usage) &&
			rd->texture_is_format_supported_for_usage(RD::DATA_FORMAT_R32_SFLOAT, sampled_storage_usage);
}

Error HDDAGIScreenProbeSVGF::denoise(uint32_t p_view_id, const FrameSettings &p_frame, const Resources &p_resources) {
	return implementation->denoise(p_view_id, p_frame, p_resources);
}

void HDDAGIScreenProbeSVGF::clear() {
	implementation->clear_views();
}

HDDAGIScreenProbeSVGF::HDDAGIScreenProbeSVGF() {
	implementation = memnew(Implementation);
}

HDDAGIScreenProbeSVGF::~HDDAGIScreenProbeSVGF() {
	memdelete(implementation);
}
