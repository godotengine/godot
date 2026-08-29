/**************************************************************************/
/*  hddagi_screen_probe_irradiance_cache.h                                */
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

#pragma once

#include "core/math/vector3.h"
#include "servers/rendering/rendering_device.h"

class HDDAGIScreenProbeIrradianceCache {
public:
	static constexpr uint32_t CASCADE_COUNT = 12u;
	static constexpr uint32_t CASCADE_GRID_SIZE = 32u;
	static constexpr uint32_t CASCADE_GRID_CELL_COUNT = CASCADE_GRID_SIZE * CASCADE_GRID_SIZE * CASCADE_GRID_SIZE;
	static constexpr uint32_t GRID_COUNT = CASCADE_COUNT * CASCADE_GRID_CELL_COUNT;
	static constexpr uint32_t CAPACITY = 16384u;
	static constexpr uint32_t MAX_REQUESTS = 65536u;
	static constexpr uint32_t MULTIBOUNCE_UPDATE_BUDGET = 4096u;
	static constexpr uint32_t MULTIBOUNCE_UPDATE_STRIDE = (CAPACITY + MULTIBOUNCE_UPDATE_BUDGET - 1u) / MULTIBOUNCE_UPDATE_BUDGET;
	static constexpr uint32_t SH_STRIDE = 4u;
	static constexpr uint32_t REQUEST_STRIDE = 3u;
	static constexpr uint32_t ACCUMULATION_STRIDE = 16u;
	static_assert(GRID_COUNT >= CAPACITY && GRID_COUNT >= MAX_REQUESTS, "The clear dispatch must cover every fixed irradiance-cache allocation.");

	struct FrameSettings {
		Vector3 camera_position;
		uint32_t history_key = 0;
		float minimum_cell_size = 0.04f;
		uint32_t multibounce_update_stride = MULTIBOUNCE_UPDATE_STRIDE;
		uint32_t sky_mode = 0;
		float sky_energy = 0.0f;
		float sky_color[4] = {};
		bool reset_history = false;
		bool multibounce = false;
	};

	struct DispatchInfo {
		bool clear_required = false;
		uint32_t grid_count = GRID_COUNT;
		uint32_t capacity = CAPACITY;
		uint32_t max_requests = MAX_REQUESTS;
	};

	static bool is_supported();

	Error prepare_frame(const FrameSettings &p_frame, DispatchInfo &r_dispatch_info);
	DispatchInfo get_dispatch_info() const;
	RID get_uniform_set(RID p_shader, uint32_t p_set) const;

	void mark_clear_recorded();
	bool is_active() const;
	void clear();

	HDDAGIScreenProbeIrradianceCache() = default;
	~HDDAGIScreenProbeIrradianceCache();

	HDDAGIScreenProbeIrradianceCache(const HDDAGIScreenProbeIrradianceCache &) = delete;
	HDDAGIScreenProbeIrradianceCache &operator=(const HDDAGIScreenProbeIrradianceCache &) = delete;

private:
	struct alignas(16) Parameters {
		float camera_position_minimum_cell_size[4] = {};
		uint32_t control[4] = {};
		uint32_t training[4] = {};
		float sky_color_or_border_energy[4] = {};
	};

	static_assert(sizeof(Parameters) == 4u * 16u, "Irradiance cache parameter ABI must match std140.");

	struct Resources {
		RID meta;
		RID grid;
		RID entry_cells;
		RID entry_state;
		RID sh;
		RID spatial;
		RID proposals;
		RID vote_count;
		RID free_pool;
		RID requests;
		RID accumulation;
		RID parameters;

		bool is_valid() const {
			return meta.is_valid() && grid.is_valid() && entry_cells.is_valid() && entry_state.is_valid() &&
					sh.is_valid() && spatial.is_valid() && proposals.is_valid() && vote_count.is_valid() &&
					free_pool.is_valid() && requests.is_valid() && accumulation.is_valid() && parameters.is_valid();
		}
	};

	Resources resources;
	uint32_t history_key = 0;
	uint32_t frame_index = 0;
	bool history_key_valid = false;
	bool clear_required = false;
	bool frame_prepared = false;

	Error _allocate_resources();
	Error _build_parameters(const FrameSettings &p_frame, uint32_t p_frame_index, Parameters &r_parameters) const;
};
