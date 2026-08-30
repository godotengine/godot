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

#include "core/math/math_funcs.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/renderer_rd/shaders/environment/hddagi_screen_probe_svgf.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

namespace {

enum SVGFMode {
	SVGF_MODE_TEMPORAL,
	SVGF_MODE_HISTORY_FIX,
	SVGF_MODE_ATROUS,
	SVGF_MODE_MAX,
};

static constexpr uint32_t MAX_ATROUS_ITERATION_COUNT = 4u;
static constexpr uint32_t HISTORY_FIX_FRAME_COUNT = 3u;
static constexpr uint32_t HISTORY_FIX_BASE_PIXEL_STRIDE = 14u;
static constexpr uint32_t SAMPLED_TEXTURE_COUNT = 8u;
static constexpr uint32_t STORAGE_IMAGE_COUNT = 4u;

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

static_assert(sizeof(SVGFFrameData) == 336u, "SVGF frame uniform ABI must remain 336 bytes.");
static_assert(sizeof(SVGFPushConstant) == 48u, "SVGF push-constant ABI must remain 48 bytes.");

} // namespace

struct HDDAGIScreenProbeSVGF::Implementation {
	struct ViewContext {
		Size2i size;
		RID history_signal[2];
		RID history_moments[2];
		RID history_normal_roughness[2];
		RID history_view_z[2];
		RID filter_scratch;
		RID frame_data_buffer;
		uint32_t history_slot = 0;
		bool history_initialized = false;
	};

	HddagiScreenProbeSvgfShaderRD shader;
	RID shader_version;
	RID pipelines[SVGF_MODE_MAX];
	RID nearest_sampler;
	HashMap<uint32_t, ViewContext *> views;
	bool shader_initialized = false;

	Error initialize_shader() {
		if (shader_initialized) {
			for (uint32_t mode = 0; mode < SVGF_MODE_MAX; mode++) {
				if (!pipelines[mode].is_valid()) {
					return ERR_CANT_CREATE;
				}
			}
			return nearest_sampler.is_valid() ? OK : ERR_CANT_CREATE;
		}

		RenderingDevice *rd = RenderingDevice::get_singleton();
		ERR_FAIL_NULL_V(rd, ERR_UNAVAILABLE);
		ERR_FAIL_COND_V_MSG(!HDDAGIScreenProbeSVGF::is_supported(), ERR_UNAVAILABLE, "The RenderingDevice cannot run the HDDAGI screen-probe SVGF shader.");

		Vector<String> modes;
		modes.push_back("\n#define MODE_TEMPORAL\n");
		modes.push_back("\n#define MODE_HISTORY_FIX\n");
		modes.push_back("\n#define MODE_ATROUS\n");
		shader.initialize(modes);
		shader_version = shader.version_create();
		shader_initialized = true;
		ERR_FAIL_COND_V_MSG(shader_version.is_null(), ERR_CANT_CREATE, "Failed to create the HDDAGI screen-probe SVGF shader version.");

		for (uint32_t mode = 0; mode < SVGF_MODE_MAX; mode++) {
			const RID mode_shader = shader.version_get_shader(shader_version, mode);
			if (mode_shader.is_null()) {
				return ERR_CANT_CREATE;
			}
			pipelines[mode] = rd->compute_pipeline_create(mode_shader);
			if (pipelines[mode].is_null()) {
				return ERR_CANT_CREATE;
			}
		}

		RD::SamplerState sampler_state;
		sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
		sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
		sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
		sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
		sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
		sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
		nearest_sampler = rd->sampler_create(sampler_state);
		if (nearest_sampler.is_null()) {
			return ERR_CANT_CREATE;
		}
		rd->set_resource_name(nearest_sampler, "HDDAGI Screen Probe SVGF Nearest Sampler");
		return OK;
	}

	RID create_texture(const Size2i &p_size, RD::DataFormat p_format, const String &p_name) const {
		RenderingDevice *rd = RenderingDevice::get_singleton();
		ERR_FAIL_NULL_V(rd, RID());
		RD::TextureFormat texture_format;
		texture_format.texture_type = RD::TEXTURE_TYPE_2D;
		texture_format.width = p_size.x;
		texture_format.height = p_size.y;
		texture_format.depth = 1;
		texture_format.array_layers = 1;
		texture_format.mipmaps = 1;
		texture_format.format = p_format;
		texture_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		RID texture = rd->texture_create(texture_format, RD::TextureView());
		if (texture.is_valid()) {
			rd->set_resource_name(texture, p_name);
		}
		return texture;
	}

	void free_view(ViewContext *p_view) const {
		if (p_view == nullptr) {
			return;
		}
		RenderingDevice *rd = RenderingDevice::get_singleton();
		if (rd != nullptr) {
			const RID resources[] = {
				p_view->history_signal[0],
				p_view->history_signal[1],
				p_view->history_moments[0],
				p_view->history_moments[1],
				p_view->history_normal_roughness[0],
				p_view->history_normal_roughness[1],
				p_view->history_view_z[0],
				p_view->history_view_z[1],
				p_view->filter_scratch,
				p_view->frame_data_buffer,
			};
			for (const RID &resource : resources) {
				if (resource.is_valid()) {
					rd->free_rid(resource);
				}
			}
		}
		memdelete(p_view);
	}

	void clear_views() {
		for (const KeyValue<uint32_t, ViewContext *> &E : views) {
			free_view(E.value);
		}
		views.clear();
	}

	Error create_view(uint32_t p_view_id, const Size2i &p_size, ViewContext *&r_view) {
		ERR_FAIL_COND_V_MSG(p_size.x <= 0 || p_size.y <= 0, ERR_INVALID_PARAMETER, "SVGF view dimensions must be positive.");
		RenderingDevice *rd = RenderingDevice::get_singleton();
		ERR_FAIL_NULL_V(rd, ERR_UNAVAILABLE);

		ViewContext *view = memnew(ViewContext);
		view->size = p_size;
		for (uint32_t slot = 0; slot < 2; slot++) {
			view->history_signal[slot] = create_texture(p_size, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, vformat("HDDAGI SVGF View %d History Signal %d", p_view_id, slot));
			view->history_moments[slot] = create_texture(p_size, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, vformat("HDDAGI SVGF View %d History Moments and Surface Offset %d", p_view_id, slot));
			view->history_normal_roughness[slot] = create_texture(p_size, RD::DATA_FORMAT_R8G8B8A8_UNORM, vformat("HDDAGI SVGF View %d History Normal Roughness %d", p_view_id, slot));
			view->history_view_z[slot] = create_texture(p_size, RD::DATA_FORMAT_R32_SFLOAT, vformat("HDDAGI SVGF View %d History ViewZ %d", p_view_id, slot));
		}
		view->filter_scratch = create_texture(p_size, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, vformat("HDDAGI SVGF View %d Filter Scratch", p_view_id));
		view->frame_data_buffer = rd->uniform_buffer_create(sizeof(SVGFFrameData));
		if (view->frame_data_buffer.is_valid()) {
			rd->set_resource_name(view->frame_data_buffer, vformat("HDDAGI SVGF View %d Frame Data", p_view_id));
		}

		bool resources_valid = view->filter_scratch.is_valid() && view->frame_data_buffer.is_valid();
		for (uint32_t slot = 0; slot < 2; slot++) {
			resources_valid = resources_valid && view->history_signal[slot].is_valid() && view->history_moments[slot].is_valid() &&
					view->history_normal_roughness[slot].is_valid() && view->history_view_z[slot].is_valid();
		}
		if (!resources_valid) {
			free_view(view);
			return ERR_OUT_OF_MEMORY;
		}

		r_view = view;
		return OK;
	}

	Error ensure_view(uint32_t p_view_id, const Size2i &p_size, ViewContext *&r_view) {
		ViewContext **view_ptr = views.getptr(p_view_id);
		if (view_ptr != nullptr && (*view_ptr)->size == p_size) {
			r_view = *view_ptr;
			return OK;
		}
		if (view_ptr != nullptr) {
			free_view(*view_ptr);
			views.erase(p_view_id);
		}

		ViewContext *view = nullptr;
		const Error error = create_view(p_view_id, p_size, view);
		if (error != OK) {
			return error;
		}
		views.insert(p_view_id, view);
		r_view = view;
		return OK;
	}

	Error validate_texture(RID p_texture, const Size2i &p_size, RD::DataFormat p_format, uint32_t p_required_usage, const String &p_name) const {
		RenderingDevice *rd = RenderingDevice::get_singleton();
		ERR_FAIL_NULL_V(rd, ERR_UNAVAILABLE);
		ERR_FAIL_COND_V_MSG(!rd->texture_is_valid(p_texture), ERR_INVALID_PARAMETER, vformat("SVGF resource '%s' is not a valid RenderingDevice texture.", p_name));
		const RD::TextureFormat texture_format = rd->texture_get_format(p_texture);
		ERR_FAIL_COND_V_MSG(texture_format.format != p_format, ERR_INVALID_PARAMETER,
				vformat("SVGF resource '%s' has format %d, expected %d.", p_name, uint32_t(texture_format.format), uint32_t(p_format)));
		ERR_FAIL_COND_V_MSG((texture_format.usage_bits & p_required_usage) != p_required_usage, ERR_INVALID_PARAMETER,
				vformat("SVGF resource '%s' is missing required texture usage bits 0x%x.", p_name, p_required_usage));
		ERR_FAIL_COND_V_MSG(rd->texture_size(p_texture) != p_size, ERR_INVALID_PARAMETER,
				vformat("SVGF resource '%s' has size %s, expected %s.", p_name, rd->texture_size(p_texture), p_size));
		return OK;
	}

	Error validate_resources(const HDDAGIScreenProbeSVGF::FrameSettings &p_frame, const HDDAGIScreenProbeSVGF::Resources &p_resources) const {
		ERR_FAIL_COND_V_MSG(!p_resources.is_valid(), ERR_INVALID_PARAMETER, "The HDDAGI screen-probe SVGF resources are incomplete or alias each other.");
		Error error = validate_texture(p_resources.motion_vectors, p_frame.size, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, RD::TEXTURE_USAGE_SAMPLING_BIT, "motion_vectors");
		if (error == OK) {
			error = validate_texture(p_resources.normal_roughness, p_frame.size, RD::DATA_FORMAT_R8G8B8A8_UNORM, RD::TEXTURE_USAGE_SAMPLING_BIT, "normal_roughness");
		}
		if (error == OK) {
			error = validate_texture(p_resources.view_z, p_frame.size, RD::DATA_FORMAT_R32_SFLOAT, RD::TEXTURE_USAGE_SAMPLING_BIT, "view_z");
		}
		if (error == OK) {
			error = validate_texture(p_resources.diffuse_radiance_hit_distance, p_frame.size, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, RD::TEXTURE_USAGE_SAMPLING_BIT, "diffuse_radiance_hit_distance");
		}
		if (error == OK) {
			error = validate_texture(p_resources.output_diffuse_radiance_hit_distance, p_frame.size, RD::DATA_FORMAT_R16G16B16A16_SFLOAT,
					RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT, "output_diffuse_radiance_hit_distance");
		}
		return error;
	}

	Error denoise(uint32_t p_view_id, const HDDAGIScreenProbeSVGF::FrameSettings &p_frame, const HDDAGIScreenProbeSVGF::Resources &p_resources) {
		ERR_FAIL_COND_V_MSG(p_frame.size.x <= 0 || p_frame.size.y <= 0, ERR_INVALID_PARAMETER, "The HDDAGI screen-probe SVGF size must be positive.");
		ERR_FAIL_COND_V_MSG(!Math::is_finite(p_frame.denoising_range) || p_frame.denoising_range <= 0.0f, ERR_INVALID_PARAMETER, "The HDDAGI screen-probe SVGF denoising range must be finite and positive.");
		ERR_FAIL_INDEX_V(int(p_frame.quality), int(HDDAGIScreenProbeSVGF::QUALITY_MAX), ERR_INVALID_PARAMETER);
		ERR_FAIL_COND_V_MSG(!p_frame.camera_transform.is_finite() || !p_frame.previous_camera_transform.is_finite() || !p_frame.taa_jitter.is_finite() || !p_frame.previous_taa_jitter.is_finite(),
				ERR_INVALID_PARAMETER, "The HDDAGI screen-probe SVGF camera transforms and jitter must be finite.");

		Error error = initialize_shader();
		if (error != OK) {
			return error;
		}
		error = validate_resources(p_frame, p_resources);
		if (error != OK) {
			return error;
		}

		ViewContext *view = nullptr;
		error = ensure_view(p_view_id, p_frame.size, view);
		if (error != OK) {
			return error;
		}

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
		error = rd->buffer_update(view->frame_data_buffer, 0, sizeof(frame_data), &frame_data);
		if (error != OK) {
			return error;
		}

		// SubViewports can skip global frames, so advance history after successful local submissions.
		const uint32_t previous_slot = view->history_slot;
		const uint32_t current_slot = previous_slot ^ 1u;
		const bool history_valid = p_frame.history_valid && view->history_initialized;
		const uint32_t iteration_count = p_frame.specular ? 1u : HDDAGIScreenProbeSVGF::get_atrous_iteration_count(p_frame.quality);

		SVGFPushConstant push_constant = {};
		push_constant.control[0] = (history_valid ? 1u : 0u) | (p_frame.specular ? 2u : 0u) |
				(p_frame.specular && p_frame.specular_full_resolution ? 4u : 0u);
		push_constant.control[3] = (p_frame.projection.is_orthogonal() ? 1u : 0u) | (p_frame.previous_projection.is_orthogonal() ? 2u : 0u);
		push_constant.tuning[0] = p_frame.denoising_range;
		push_constant.tuning[1] = p_frame.specular ? 0.95f : 0.9f;
		push_constant.tuning[2] = p_frame.specular ? 0.0025f : 0.005f;
		push_constant.tuning[3] = p_frame.specular ? 2.5f : 2.0f;
		push_constant.atrous[0] = p_frame.specular ? 12.0f : 8.0f;
		push_constant.atrous[1] = p_frame.specular ? 0.003f : 0.005f;
		// Scale luminance moments separately to avoid FP16 underflow in the scene / 512 radiance domain.
		push_constant.atrous[2] = 4.0e-5f;
		push_constant.atrous[3] = p_frame.specular ? 12.0f : 30.0f;

		UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
		ERR_FAIL_NULL_V(uniform_set_cache, ERR_UNAVAILABLE);
		const RID temporal_shader = shader.version_get_shader(shader_version, SVGF_MODE_TEMPORAL);
		const RID history_fix_shader = shader.version_get_shader(shader_version, SVGF_MODE_HISTORY_FIX);
		const RID atrous_shader = shader.version_get_shader(shader_version, SVGF_MODE_ATROUS);
		ERR_FAIL_COND_V(temporal_shader.is_null() || history_fix_shader.is_null() || atrous_shader.is_null(), ERR_CANT_CREATE);

		const RID temporal_signal_destination = p_frame.specular ? view->history_signal[current_slot] : view->filter_scratch;
		const RID temporal_moments_destination = p_frame.specular ? view->history_moments[current_slot] : p_resources.output_diffuse_radiance_hit_distance;
		const RID temporal_set = uniform_set_cache->get_cache(
				temporal_shader, 0,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, p_resources.diffuse_radiance_hit_distance),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, p_resources.normal_roughness),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, p_resources.view_z),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, p_resources.motion_vectors),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 4, view->history_signal[previous_slot]),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 5, view->history_moments[previous_slot]),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 6, view->history_normal_roughness[previous_slot]),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 7, view->history_view_z[previous_slot]),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 8, nearest_sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 9, temporal_signal_destination),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 10, temporal_moments_destination),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 11, view->history_normal_roughness[current_slot]),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 12, view->history_view_z[current_slot]));
		const RID temporal_frame_data_set = uniform_set_cache->get_cache(
				temporal_shader, 1,
				RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, view->frame_data_buffer));
		RID history_fix_set;
		RID history_fix_frame_data_set;
		if (!p_frame.specular) {
			history_fix_set = uniform_set_cache->get_cache(
					history_fix_shader, 0,
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, view->filter_scratch),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, p_resources.output_diffuse_radiance_hit_distance),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, view->history_normal_roughness[current_slot]),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, view->history_view_z[current_slot]),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, nearest_sampler),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, view->history_signal[current_slot]),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, view->history_moments[current_slot]));
			history_fix_frame_data_set = uniform_set_cache->get_cache(
					history_fix_shader, 1,
					RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, view->frame_data_buffer));
		}
		RID atrous_sets[MAX_ATROUS_ITERATION_COUNT];
		RID signal_source = view->history_signal[current_slot];
		const bool first_iteration_writes_output = (iteration_count & 1u) != 0u;
		for (uint32_t iteration = 0; iteration < iteration_count; iteration++) {
			const bool write_output = ((iteration & 1u) == 0u) == first_iteration_writes_output;
			const RID signal_destination = write_output ? p_resources.output_diffuse_radiance_hit_distance : view->filter_scratch;
			LocalVector<RD::Uniform> atrous_uniforms;
			atrous_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, signal_source));
			atrous_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, view->history_moments[current_slot]));
			atrous_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, view->history_normal_roughness[current_slot]));
			atrous_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, p_resources.view_z));
			atrous_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, nearest_sampler));
			atrous_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, signal_destination));
			atrous_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 6, p_resources.diffuse_radiance_hit_distance));
			atrous_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 7, p_resources.normal_roughness));
			atrous_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 8, p_resources.motion_vectors));
			atrous_sets[iteration] = uniform_set_cache->get_cache_vec(atrous_shader, 0, atrous_uniforms);
			signal_source = signal_destination;
		}

		if (!temporal_set.is_valid() || !temporal_frame_data_set.is_valid() ||
				(!p_frame.specular && (!history_fix_set.is_valid() || !history_fix_frame_data_set.is_valid())) ||
				signal_source != p_resources.output_diffuse_radiance_hit_distance) {
			return ERR_CANT_CREATE;
		}
		for (uint32_t iteration = 0; iteration < iteration_count; iteration++) {
			if (!atrous_sets[iteration].is_valid()) {
				return ERR_CANT_CREATE;
			}
		}

		rd->draw_command_begin_label("HDDAGI Screen Probe SVGF");
		RD::ComputeListID compute_list = rd->compute_list_begin();
		rd->compute_list_bind_compute_pipeline(compute_list, pipelines[SVGF_MODE_TEMPORAL]);
		rd->compute_list_bind_uniform_set(compute_list, temporal_set, 0);
		rd->compute_list_bind_uniform_set(compute_list, temporal_frame_data_set, 1);
		rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		rd->compute_list_dispatch_threads(compute_list, p_frame.size.x, p_frame.size.y, 1);
		rd->compute_list_add_barrier(compute_list);

		if (!p_frame.specular) {
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
			push_constant.control[1] = 1u << iteration;
			push_constant.control[2] = (iteration == 0u ? 1u : 0u) | (iteration + 1u == iteration_count ? 2u : 0u);
			rd->compute_list_bind_compute_pipeline(compute_list, pipelines[SVGF_MODE_ATROUS]);
			rd->compute_list_bind_uniform_set(compute_list, atrous_sets[iteration], 0);
			rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
			rd->compute_list_dispatch_threads(compute_list, p_frame.size.x, p_frame.size.y, 1);
			if (iteration + 1u < iteration_count) {
				rd->compute_list_add_barrier(compute_list);
			}
		}
		rd->compute_list_end();
		rd->draw_command_end_label();

		view->history_initialized = true;
		view->history_slot = current_slot;
		return OK;
	}

	void shutdown() {
		clear_views();
		RenderingDevice *rd = RenderingDevice::get_singleton();
		if (rd != nullptr && nearest_sampler.is_valid()) {
			rd->free_rid(nearest_sampler);
		}
		nearest_sampler = RID();
		if (shader_initialized && shader_version.is_valid()) {
			shader.version_free(shader_version);
		}
		shader_version = RID();
		for (uint32_t mode = 0; mode < SVGF_MODE_MAX; mode++) {
			pipelines[mode] = RID();
		}
		shader_initialized = false;
	}

	~Implementation() {
		shutdown();
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
			rd->texture_is_format_supported_for_usage(RD::DATA_FORMAT_R32_SFLOAT, sampled_storage_usage) &&
			rd->limit_get(RD::LIMIT_MAX_BOUND_UNIFORM_SETS) >= 2u &&
			rd->limit_get(RD::LIMIT_MAX_TEXTURES_PER_UNIFORM_SET) >= SAMPLED_TEXTURE_COUNT &&
			rd->limit_get(RD::LIMIT_MAX_SAMPLERS_PER_UNIFORM_SET) >= 1u &&
			rd->limit_get(RD::LIMIT_MAX_STORAGE_IMAGES_PER_UNIFORM_SET) >= STORAGE_IMAGE_COUNT &&
			rd->limit_get(RD::LIMIT_MAX_UNIFORM_BUFFERS_PER_UNIFORM_SET) >= 1u &&
			rd->limit_get(RD::LIMIT_MAX_TEXTURES_PER_SHADER_STAGE) >= SAMPLED_TEXTURE_COUNT &&
			rd->limit_get(RD::LIMIT_MAX_SAMPLERS_PER_SHADER_STAGE) >= 1u &&
			rd->limit_get(RD::LIMIT_MAX_STORAGE_IMAGES_PER_SHADER_STAGE) >= STORAGE_IMAGE_COUNT &&
			rd->limit_get(RD::LIMIT_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE) >= 1u &&
			rd->limit_get(RD::LIMIT_MAX_PUSH_CONSTANT_SIZE) >= sizeof(SVGFPushConstant) &&
			rd->limit_get(RD::LIMIT_MAX_UNIFORM_BUFFER_SIZE) >= sizeof(SVGFFrameData);
}

Error HDDAGIScreenProbeSVGF::denoise(uint32_t p_view_id, const FrameSettings &p_frame, const Resources &p_resources) {
	ERR_FAIL_NULL_V(implementation, ERR_UNAVAILABLE);
	return implementation->denoise(p_view_id, p_frame, p_resources);
}

void HDDAGIScreenProbeSVGF::clear() {
	if (implementation != nullptr) {
		implementation->clear_views();
	}
}

HDDAGIScreenProbeSVGF::HDDAGIScreenProbeSVGF() {
	implementation = memnew(Implementation);
}

HDDAGIScreenProbeSVGF::~HDDAGIScreenProbeSVGF() {
	if (implementation != nullptr) {
		memdelete(implementation);
		implementation = nullptr;
	}
}
