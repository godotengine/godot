/**************************************************************************/
/*  hddagi_screen_probe_irradiance_cache.cpp                              */
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

#include "hddagi_screen_probe_irradiance_cache.h"

#include "core/math/math_funcs.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

namespace {

static constexpr uint32_t VEC4_SIZE_BYTES = sizeof(uint32_t) * 4u;
static constexpr uint32_t META_BUFFER_SIZE = sizeof(uint32_t) * 4u;
static constexpr uint32_t GRID_BUFFER_SIZE = sizeof(uint32_t) * 2u * HDDAGIScreenProbeIrradianceCache::GRID_COUNT;
static constexpr uint32_t ENTRY_CELLS_BUFFER_SIZE = VEC4_SIZE_BYTES * HDDAGIScreenProbeIrradianceCache::CAPACITY;
static constexpr uint32_t ENTRY_STATE_BUFFER_SIZE = VEC4_SIZE_BYTES * HDDAGIScreenProbeIrradianceCache::CAPACITY;
static constexpr uint32_t SH_BUFFER_SIZE = VEC4_SIZE_BYTES * HDDAGIScreenProbeIrradianceCache::SH_STRIDE * HDDAGIScreenProbeIrradianceCache::CAPACITY;
static constexpr uint32_t SPATIAL_BUFFER_SIZE = VEC4_SIZE_BYTES * 2u * HDDAGIScreenProbeIrradianceCache::CAPACITY;
static constexpr uint32_t PROPOSALS_BUFFER_SIZE = VEC4_SIZE_BYTES * 2u * HDDAGIScreenProbeIrradianceCache::CAPACITY;
static constexpr uint32_t VOTE_COUNT_BUFFER_SIZE = sizeof(uint32_t) * HDDAGIScreenProbeIrradianceCache::CAPACITY;
static constexpr uint32_t FREE_POOL_BUFFER_SIZE = sizeof(uint32_t) * HDDAGIScreenProbeIrradianceCache::CAPACITY;
static constexpr uint32_t REQUESTS_BUFFER_SIZE = VEC4_SIZE_BYTES * HDDAGIScreenProbeIrradianceCache::REQUEST_STRIDE * HDDAGIScreenProbeIrradianceCache::MAX_REQUESTS;
static constexpr uint32_t ACCUMULATION_BUFFER_SIZE = sizeof(int32_t) * HDDAGIScreenProbeIrradianceCache::ACCUMULATION_STRIDE * HDDAGIScreenProbeIrradianceCache::CAPACITY;
static constexpr uint32_t STORAGE_BUFFER_COUNT = 11u;
static constexpr uint32_t QUERY_UNIFORM_BUFFER_COUNT = 3u;

} // namespace

bool HDDAGIScreenProbeIrradianceCache::is_supported() {
	RenderingDevice *rd = RenderingDevice::get_singleton();
	if (rd == nullptr) {
		return false;
	}

	return rd->limit_get(RenderingDevice::LIMIT_MAX_STORAGE_BUFFERS_PER_UNIFORM_SET) >= STORAGE_BUFFER_COUNT &&
			rd->limit_get(RenderingDevice::LIMIT_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE) >= STORAGE_BUFFER_COUNT &&
			rd->limit_get(RenderingDevice::LIMIT_MAX_BOUND_UNIFORM_SETS) >= 3u &&
			rd->limit_get(RenderingDevice::LIMIT_MAX_UNIFORM_BUFFERS_PER_UNIFORM_SET) >= 1u &&
			rd->limit_get(RenderingDevice::LIMIT_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE) >= QUERY_UNIFORM_BUFFER_COUNT &&
			rd->limit_get(RenderingDevice::LIMIT_MAX_UNIFORM_BUFFER_SIZE) >= sizeof(Parameters);
}

Error HDDAGIScreenProbeIrradianceCache::prepare_frame(const FrameSettings &p_frame, DispatchInfo &r_dispatch_info) {
	frame_prepared = false;
	r_dispatch_info = DispatchInfo();
	ERR_FAIL_COND_V_MSG(!is_supported(), ERR_UNAVAILABLE, "The RenderingDevice cannot bind the HDDAGI screen-probe irradiance cache resources.");

	if (!resources.is_valid()) {
		const Error allocation_error = _allocate_resources();
		if (allocation_error != OK) {
			return allocation_error;
		}
	}

	const bool history_changed = history_key_valid && history_key != p_frame.history_key;
	bool frame_clear_required = clear_required || !history_key_valid || history_changed || p_frame.reset_history;
	if (!frame_clear_required && frame_index == UINT32_MAX) {
		frame_clear_required = true;
	}
	const uint32_t next_frame_index = frame_clear_required ? 0u : frame_index + 1u;
	Parameters parameters;
	const Error parameter_error = _build_parameters(p_frame, next_frame_index, parameters);
	if (parameter_error != OK) {
		return parameter_error;
	}

	RenderingDevice *rd = RenderingDevice::get_singleton();
	ERR_FAIL_NULL_V(rd, ERR_UNAVAILABLE);
	const Error update_error = rd->buffer_update(resources.parameters, 0, sizeof(parameters), &parameters);
	if (update_error != OK) {
		return update_error;
	}

	history_key = p_frame.history_key;
	frame_index = next_frame_index;
	history_key_valid = true;
	clear_required = frame_clear_required;
	frame_prepared = true;
	r_dispatch_info = get_dispatch_info();
	return OK;
}

HDDAGIScreenProbeIrradianceCache::DispatchInfo HDDAGIScreenProbeIrradianceCache::get_dispatch_info() const {
	DispatchInfo dispatch_info;
	dispatch_info.clear_required = is_active() && clear_required;
	return dispatch_info;
}

RID HDDAGIScreenProbeIrradianceCache::get_uniform_set(RID p_shader, uint32_t p_set) const {
	ERR_FAIL_COND_V(!is_active() || p_shader.is_null(), RID());
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL_V(uniform_set_cache, RID());

	return uniform_set_cache->get_cache(
			p_shader, p_set,
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, resources.meta),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, resources.grid),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, resources.entry_cells),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, resources.entry_state),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, resources.sh),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, resources.spatial),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, resources.proposals),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 7, resources.vote_count),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 8, resources.free_pool),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 9, resources.requests),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 10, resources.accumulation),
			RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 11, resources.parameters));
}

void HDDAGIScreenProbeIrradianceCache::mark_clear_recorded() {
	ERR_FAIL_COND(!is_active());
	clear_required = false;
}

bool HDDAGIScreenProbeIrradianceCache::is_active() const {
	return frame_prepared && resources.is_valid() && is_supported();
}

void HDDAGIScreenProbeIrradianceCache::clear() {
	RenderingDevice *rd = RenderingDevice::get_singleton();
	if (rd != nullptr) {
		const RID resource_rids[] = {
			resources.meta,
			resources.grid,
			resources.entry_cells,
			resources.entry_state,
			resources.sh,
			resources.spatial,
			resources.proposals,
			resources.vote_count,
			resources.free_pool,
			resources.requests,
			resources.accumulation,
			resources.parameters,
		};
		for (const RID &rid : resource_rids) {
			if (rid.is_valid()) {
				rd->free_rid(rid);
			}
		}
	}

	resources = Resources();
	history_key = 0;
	frame_index = 0;
	history_key_valid = false;
	clear_required = false;
	frame_prepared = false;
}

Error HDDAGIScreenProbeIrradianceCache::_allocate_resources() {
	clear();
	RenderingDevice *rd = RenderingDevice::get_singleton();
	ERR_FAIL_NULL_V(rd, ERR_UNAVAILABLE);

	const auto create_storage_buffer = [rd](uint32_t p_size_bytes, const char *p_name) {
		RID buffer = rd->storage_buffer_create(p_size_bytes);
		if (buffer.is_valid()) {
			rd->set_resource_name(buffer, p_name);
		}
		return buffer;
	};

	resources.meta = create_storage_buffer(META_BUFFER_SIZE, "HDDAGI Irradiance Cache Meta");
	resources.grid = create_storage_buffer(GRID_BUFFER_SIZE, "HDDAGI Irradiance Cache Grid");
	resources.entry_cells = create_storage_buffer(ENTRY_CELLS_BUFFER_SIZE, "HDDAGI Irradiance Cache Entry Cells");
	resources.entry_state = create_storage_buffer(ENTRY_STATE_BUFFER_SIZE, "HDDAGI Irradiance Cache Entry State");
	resources.sh = create_storage_buffer(SH_BUFFER_SIZE, "HDDAGI Irradiance Cache SH");
	resources.spatial = create_storage_buffer(SPATIAL_BUFFER_SIZE, "HDDAGI Irradiance Cache Spatial");
	resources.proposals = create_storage_buffer(PROPOSALS_BUFFER_SIZE, "HDDAGI Irradiance Cache Proposals");
	resources.vote_count = create_storage_buffer(VOTE_COUNT_BUFFER_SIZE, "HDDAGI Irradiance Cache Vote Count");
	resources.free_pool = create_storage_buffer(FREE_POOL_BUFFER_SIZE, "HDDAGI Irradiance Cache Free Pool");
	resources.requests = create_storage_buffer(REQUESTS_BUFFER_SIZE, "HDDAGI Irradiance Cache Requests");
	resources.accumulation = create_storage_buffer(ACCUMULATION_BUFFER_SIZE, "HDDAGI Irradiance Cache Accumulation");
	resources.parameters = rd->uniform_buffer_create(sizeof(Parameters));
	if (resources.parameters.is_valid()) {
		rd->set_resource_name(resources.parameters, "HDDAGI Irradiance Cache Parameters");
	}

	if (!resources.is_valid()) {
		clear();
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Failed to allocate the HDDAGI screen-probe irradiance cache buffers.");
	}

	clear_required = true;
	return OK;
}

Error HDDAGIScreenProbeIrradianceCache::_build_parameters(const FrameSettings &p_frame, uint32_t p_frame_index, Parameters &r_parameters) const {
	ERR_FAIL_COND_V_MSG(!p_frame.camera_position.is_finite(), ERR_INVALID_PARAMETER, "The HDDAGI screen-probe irradiance cache camera position must be finite.");
	ERR_FAIL_COND_V_MSG(!Math::is_finite(p_frame.minimum_cell_size) || p_frame.minimum_cell_size <= 0.0f, ERR_INVALID_PARAMETER, "The HDDAGI screen-probe irradiance cache minimum cell size must be finite and positive.");
	ERR_FAIL_COND_V_MSG(p_frame.multibounce_update_stride == 0u, ERR_INVALID_PARAMETER, "The HDDAGI screen-probe irradiance cache multi-bounce update stride must be positive.");
	ERR_FAIL_COND_V_MSG(p_frame.sky_mode > 2u || !Math::is_finite(p_frame.sky_energy) || p_frame.sky_energy < 0.0f, ERR_INVALID_PARAMETER, "The HDDAGI screen-probe irradiance cache sky settings are invalid.");
	for (uint32_t i = 0; i < 4; i++) {
		ERR_FAIL_COND_V_MSG(!Math::is_finite(p_frame.sky_color[i]), ERR_INVALID_PARAMETER, "The HDDAGI screen-probe irradiance cache sky color must be finite.");
	}

	r_parameters.camera_position_minimum_cell_size[0] = float(p_frame.camera_position.x);
	r_parameters.camera_position_minimum_cell_size[1] = float(p_frame.camera_position.y);
	r_parameters.camera_position_minimum_cell_size[2] = float(p_frame.camera_position.z);
	r_parameters.camera_position_minimum_cell_size[3] = p_frame.minimum_cell_size;
	r_parameters.control[0] = p_frame_index;
	r_parameters.control[1] = CAPACITY;
	r_parameters.control[2] = MAX_REQUESTS;
	r_parameters.control[3] = GRID_COUNT;
	r_parameters.training[0] = p_frame.multibounce ? 1u : 0u;
	r_parameters.training[1] = p_frame.multibounce_update_stride;
	r_parameters.training[2] = p_frame.sky_mode;
	if (p_frame.sky_mode == 2u) {
		// Texture mode needs only the octahedral border; RGB is sampled.
		r_parameters.sky_color_or_border_energy[0] = p_frame.sky_color[3];
	} else {
		r_parameters.sky_color_or_border_energy[0] = p_frame.sky_color[0];
		r_parameters.sky_color_or_border_energy[1] = p_frame.sky_color[1];
		r_parameters.sky_color_or_border_energy[2] = p_frame.sky_color[2];
	}
	r_parameters.sky_color_or_border_energy[3] = p_frame.sky_energy;

	return OK;
}

HDDAGIScreenProbeIrradianceCache::~HDDAGIScreenProbeIrradianceCache() {
	clear();
}
