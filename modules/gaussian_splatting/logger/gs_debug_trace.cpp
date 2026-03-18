#include "gs_debug_trace.h"

#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "core/os/mutex.h"
#include "core/os/os.h"
#include "core/templates/vector.h"

namespace GaussianSplatting {

namespace {

constexpr uint32_t DEBUG_EVENT_CAPACITY = 64;
constexpr uint32_t DATA_FLOW_RECENT_WINDOW_CAPACITY = 120;
constexpr float SH_DC_ZERO_EPSILON = 0.0001f;

struct DebugEvent {
	uint64_t timestamp_usec = 0;
	String category;
	String message;
	bool is_error = false;
};

struct DataFlowStats {
	uint64_t pack_sh_samples = 0;
	double sh_dc_sum = 0.0;
	float sh_dc_min = 0.0f;
	uint32_t sh_dc_zero_count = 0;
	Color last_pack_sh_dc;
	float last_pack_opacity = 0.0f;

	uint32_t pack_range_calls = 0;
	uint32_t last_pack_range_count = 0;
	uint32_t last_pack_range_start = 0;
	uint32_t last_pack_range_src_size = 0;
	Color last_pack_range_sh_dc;
	float last_pack_range_opacity = 0.0f;

	uint32_t gaussians_check_calls = 0;
	uint32_t last_asset_id = 0;
	uint32_t last_gaussian_total = 0;
	uint32_t last_chunk_start = 0;
	uint32_t last_chunk_count = 0;
	Color last_gaussian_sh_dc;
	float last_gaussian_opacity = 0.0f;

	uint32_t buffer_mgr_calls = 0;
	uint32_t last_buffer_mgr_total = 0;
	Color last_buffer_mgr_sh_dc;
	float last_buffer_mgr_opacity = 0.0f;

	uint32_t last_diag_chunk = 0;
	uint32_t last_diag_zero_scale = 0;
	uint32_t last_diag_zero_opacity = 0;
	uint32_t last_diag_nan_pos = 0;
	uint32_t last_diag_total = 0;
	bool diag_valid = false;

	uint32_t last_instance_buffer_count = 0;
	uint32_t last_instance_world_count = 0;
	uint32_t last_instance_skip_count = 0;
	uint64_t instance_buffer_updates = 0;

	uint32_t last_instance_visible_clamped = 0;
	uint32_t last_instance_visible_raw = 0;
	uint32_t last_instance_overflow_flag = 0;
	uint64_t instance_count_updates = 0;

	uint32_t last_instance_total = 0;
	uint32_t last_instance_rotation_identity = 0;
	uint32_t last_instance_scale_identity = 0;
	uint32_t last_instance_translation_zero = 0;
	uint32_t last_instance_fully_identity = 0;
	uint64_t instance_flag_updates = 0;
};

struct DataFlowFrameDelta {
	uint64_t frame = 0;
	uint32_t pack_sh_samples = 0;
	uint32_t pack_range_calls = 0;
	uint32_t gaussians_check_calls = 0;
	uint32_t buffer_mgr_calls = 0;
	uint32_t chunk_diagnostics_updates = 0;
	uint32_t instance_buffer_updates = 0;
	uint32_t instance_count_updates = 0;
	uint32_t instance_flag_updates = 0;
};

Mutex debug_mutex;
Vector<DebugEvent> debug_events;
uint32_t debug_event_cursor = 0;
bool debug_event_full = false;
DataFlowStats data_flow_stats;
Vector<DataFlowFrameDelta> recent_data_flow_deltas;
uint32_t recent_data_flow_cursor = 0;
bool recent_data_flow_full = false;
DataFlowFrameDelta active_data_flow_delta;
bool active_data_flow_delta_valid = false;

static bool _delta_has_activity(const DataFlowFrameDelta &p_delta) {
	return p_delta.pack_sh_samples > 0 ||
			p_delta.pack_range_calls > 0 ||
			p_delta.gaussians_check_calls > 0 ||
			p_delta.buffer_mgr_calls > 0 ||
			p_delta.chunk_diagnostics_updates > 0 ||
			p_delta.instance_buffer_updates > 0 ||
			p_delta.instance_count_updates > 0 ||
			p_delta.instance_flag_updates > 0;
}

static void _push_recent_delta_locked(const DataFlowFrameDelta &p_delta) {
	if (!_delta_has_activity(p_delta)) {
		return;
	}
	if (recent_data_flow_deltas.size() < static_cast<int>(DATA_FLOW_RECENT_WINDOW_CAPACITY)) {
		recent_data_flow_deltas.resize(DATA_FLOW_RECENT_WINDOW_CAPACITY);
	}
	recent_data_flow_deltas.write[recent_data_flow_cursor] = p_delta;
	recent_data_flow_cursor = (recent_data_flow_cursor + 1) % DATA_FLOW_RECENT_WINDOW_CAPACITY;
	if (recent_data_flow_cursor == 0) {
		recent_data_flow_full = true;
	}
}

static void _finalize_active_delta_locked() {
	if (!active_data_flow_delta_valid) {
		return;
	}
	_push_recent_delta_locked(active_data_flow_delta);
	active_data_flow_delta = DataFlowFrameDelta();
	active_data_flow_delta_valid = false;
}

static bool _is_enabled() {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	if (!ps) {
		return false;
	}
	return ps->get_setting("rendering/gaussian_splatting/debug/enable_pipeline_trace", false)
			|| ps->get_setting("rendering/gaussian_splatting/debug/enable_data_logging", false);
}

static void _push_event(const String &p_category, const String &p_message, bool p_is_error) {
	MutexLock lock(debug_mutex);
	if (debug_events.size() < static_cast<int>(DEBUG_EVENT_CAPACITY)) {
		debug_events.resize(DEBUG_EVENT_CAPACITY);
	}
	DebugEvent &entry = debug_events.write[debug_event_cursor];
	entry.timestamp_usec = OS::get_singleton()->get_ticks_usec();
	entry.category = p_category;
	entry.message = p_message;
	entry.is_error = p_is_error;
	debug_event_cursor = (debug_event_cursor + 1) % DEBUG_EVENT_CAPACITY;
	if (debug_event_cursor == 0) {
		debug_event_full = true;
	}
}

} // namespace

bool debug_trace_is_enabled() {
	return _is_enabled();
}

void debug_trace_begin_frame(uint64_t p_frame_id) {
	if (!_is_enabled()) {
		return;
	}
	MutexLock lock(debug_mutex);
	if (active_data_flow_delta_valid && active_data_flow_delta.frame == p_frame_id) {
		return;
	}
	_finalize_active_delta_locked();
	active_data_flow_delta = DataFlowFrameDelta();
	active_data_flow_delta.frame = p_frame_id;
	active_data_flow_delta_valid = true;
}

void debug_trace_record_event(const String &p_category, const String &p_message, bool p_is_error) {
	if (!_is_enabled()) {
		return;
	}
	_push_event(p_category, p_message, p_is_error);
}

void debug_trace_record_pack_sh(const Color &p_sh_dc, float p_opacity) {
	if (!_is_enabled()) {
		return;
	}
	const float magnitude = Math::sqrt(p_sh_dc.r * p_sh_dc.r + p_sh_dc.g * p_sh_dc.g + p_sh_dc.b * p_sh_dc.b);
	MutexLock lock(debug_mutex);
	data_flow_stats.pack_sh_samples++;
	data_flow_stats.sh_dc_sum += magnitude;
	data_flow_stats.sh_dc_min = (data_flow_stats.pack_sh_samples == 1)
			? magnitude
			: MIN(data_flow_stats.sh_dc_min, magnitude);
	if (magnitude < SH_DC_ZERO_EPSILON) {
		data_flow_stats.sh_dc_zero_count++;
	}
	data_flow_stats.last_pack_sh_dc = p_sh_dc;
	data_flow_stats.last_pack_opacity = p_opacity;
	if (active_data_flow_delta_valid) {
		active_data_flow_delta.pack_sh_samples++;
	}
}

void debug_trace_record_pack_range(uint32_t p_count, uint32_t p_start, uint32_t p_src_size,
		const Color &p_sh_dc, float p_opacity) {
	if (!_is_enabled()) {
		return;
	}
	MutexLock lock(debug_mutex);
	data_flow_stats.pack_range_calls++;
	data_flow_stats.last_pack_range_count = p_count;
	data_flow_stats.last_pack_range_start = p_start;
	data_flow_stats.last_pack_range_src_size = p_src_size;
	data_flow_stats.last_pack_range_sh_dc = p_sh_dc;
	data_flow_stats.last_pack_range_opacity = p_opacity;
	if (active_data_flow_delta_valid) {
		active_data_flow_delta.pack_range_calls++;
	}
}

void debug_trace_record_gaussians_check(uint32_t p_asset_id, uint32_t p_total, uint32_t p_chunk_start, uint32_t p_chunk_count,
		const Color &p_sh_dc, float p_opacity) {
	if (!_is_enabled()) {
		return;
	}
	MutexLock lock(debug_mutex);
	data_flow_stats.gaussians_check_calls++;
	data_flow_stats.last_asset_id = p_asset_id;
	data_flow_stats.last_gaussian_total = p_total;
	data_flow_stats.last_chunk_start = p_chunk_start;
	data_flow_stats.last_chunk_count = p_chunk_count;
	data_flow_stats.last_gaussian_sh_dc = p_sh_dc;
	data_flow_stats.last_gaussian_opacity = p_opacity;
	if (active_data_flow_delta_valid) {
		active_data_flow_delta.gaussians_check_calls++;
	}
}

void debug_trace_record_buffer_mgr(uint32_t p_total, const Color &p_sh_dc, float p_opacity) {
	if (!_is_enabled()) {
		return;
	}
	MutexLock lock(debug_mutex);
	data_flow_stats.buffer_mgr_calls++;
	data_flow_stats.last_buffer_mgr_total = p_total;
	data_flow_stats.last_buffer_mgr_sh_dc = p_sh_dc;
	data_flow_stats.last_buffer_mgr_opacity = p_opacity;
	if (active_data_flow_delta_valid) {
		active_data_flow_delta.buffer_mgr_calls++;
	}
}

void debug_trace_record_chunk_diagnostics(uint32_t p_chunk_idx, uint32_t p_zero_scale, uint32_t p_zero_opacity,
		uint32_t p_nan_pos, uint32_t p_total) {
	if (!_is_enabled()) {
		return;
	}
	MutexLock lock(debug_mutex);
	data_flow_stats.last_diag_chunk = p_chunk_idx;
	data_flow_stats.last_diag_zero_scale = p_zero_scale;
	data_flow_stats.last_diag_zero_opacity = p_zero_opacity;
	data_flow_stats.last_diag_nan_pos = p_nan_pos;
	data_flow_stats.last_diag_total = p_total;
	data_flow_stats.diag_valid = true;
	if (active_data_flow_delta_valid) {
		active_data_flow_delta.chunk_diagnostics_updates++;
	}
}

void debug_trace_record_instance_buffer(uint32_t p_out_count, uint32_t p_world_count, uint32_t p_skip_count) {
	if (!_is_enabled()) {
		return;
	}
	MutexLock lock(debug_mutex);
	data_flow_stats.last_instance_buffer_count = p_out_count;
	data_flow_stats.last_instance_world_count = p_world_count;
	data_flow_stats.last_instance_skip_count = p_skip_count;
	data_flow_stats.instance_buffer_updates++;
	if (active_data_flow_delta_valid) {
		active_data_flow_delta.instance_buffer_updates++;
	}
}

void debug_trace_record_instance_counts(uint32_t p_visible_clamped, uint32_t p_visible_raw, uint32_t p_overflow_flag) {
	if (!_is_enabled()) {
		return;
	}
	MutexLock lock(debug_mutex);
	data_flow_stats.last_instance_visible_clamped = p_visible_clamped;
	data_flow_stats.last_instance_visible_raw = p_visible_raw;
	data_flow_stats.last_instance_overflow_flag = p_overflow_flag;
	data_flow_stats.instance_count_updates++;
	if (active_data_flow_delta_valid) {
		active_data_flow_delta.instance_count_updates++;
	}
}

void debug_trace_record_instance_flags(uint32_t p_total, uint32_t p_rotation_identity, uint32_t p_scale_identity,
		uint32_t p_translation_zero, uint32_t p_fully_identity) {
	if (!_is_enabled()) {
		return;
	}
	MutexLock lock(debug_mutex);
	data_flow_stats.last_instance_total = p_total;
	data_flow_stats.last_instance_rotation_identity = p_rotation_identity;
	data_flow_stats.last_instance_scale_identity = p_scale_identity;
	data_flow_stats.last_instance_translation_zero = p_translation_zero;
	data_flow_stats.last_instance_fully_identity = p_fully_identity;
	data_flow_stats.instance_flag_updates++;
	if (active_data_flow_delta_valid) {
		active_data_flow_delta.instance_flag_updates++;
	}
}

Dictionary debug_trace_get_data_flow_snapshot() {
	MutexLock lock(debug_mutex);
	if (data_flow_stats.pack_sh_samples == 0 &&
			data_flow_stats.pack_range_calls == 0 &&
			data_flow_stats.gaussians_check_calls == 0 &&
			data_flow_stats.buffer_mgr_calls == 0 &&
			!data_flow_stats.diag_valid &&
			data_flow_stats.instance_buffer_updates == 0 &&
			data_flow_stats.instance_count_updates == 0 &&
			data_flow_stats.instance_flag_updates == 0) {
		return Dictionary();
	}
	Dictionary snapshot;
	snapshot["pack_sh_samples"] = static_cast<int64_t>(data_flow_stats.pack_sh_samples);
	snapshot["sh_dc_min"] = data_flow_stats.pack_sh_samples > 0 ? data_flow_stats.sh_dc_min : 0.0f;
	snapshot["sh_dc_avg"] = data_flow_stats.pack_sh_samples > 0
			? data_flow_stats.sh_dc_sum / static_cast<double>(data_flow_stats.pack_sh_samples)
			: 0.0;
	snapshot["sh_dc_zero_count"] = static_cast<int64_t>(data_flow_stats.sh_dc_zero_count);
	Dictionary last_pack_sh;
	last_pack_sh["sh_dc"] = data_flow_stats.last_pack_sh_dc;
	last_pack_sh["opacity"] = data_flow_stats.last_pack_opacity;
	snapshot["last_pack_sh"] = last_pack_sh;

	Dictionary pack_range;
	pack_range["calls"] = data_flow_stats.pack_range_calls;
	pack_range["count"] = data_flow_stats.last_pack_range_count;
	pack_range["start"] = data_flow_stats.last_pack_range_start;
	pack_range["src_size"] = data_flow_stats.last_pack_range_src_size;
	pack_range["sh_dc"] = data_flow_stats.last_pack_range_sh_dc;
	pack_range["opacity"] = data_flow_stats.last_pack_range_opacity;
	snapshot["pack_range"] = pack_range;

	Dictionary gaussians_check;
	gaussians_check["calls"] = data_flow_stats.gaussians_check_calls;
	gaussians_check["asset_id"] = data_flow_stats.last_asset_id;
	gaussians_check["total"] = data_flow_stats.last_gaussian_total;
	gaussians_check["chunk_start"] = data_flow_stats.last_chunk_start;
	gaussians_check["chunk_count"] = data_flow_stats.last_chunk_count;
	gaussians_check["sh_dc"] = data_flow_stats.last_gaussian_sh_dc;
	gaussians_check["opacity"] = data_flow_stats.last_gaussian_opacity;
	snapshot["gaussians_check"] = gaussians_check;

	Dictionary buffer_mgr;
	buffer_mgr["calls"] = data_flow_stats.buffer_mgr_calls;
	buffer_mgr["total"] = data_flow_stats.last_buffer_mgr_total;
	buffer_mgr["sh_dc"] = data_flow_stats.last_buffer_mgr_sh_dc;
	buffer_mgr["opacity"] = data_flow_stats.last_buffer_mgr_opacity;
	snapshot["buffer_mgr"] = buffer_mgr;

	if (data_flow_stats.diag_valid) {
		Dictionary diagnostics;
		diagnostics["chunk"] = data_flow_stats.last_diag_chunk;
		diagnostics["zero_scale"] = data_flow_stats.last_diag_zero_scale;
		diagnostics["zero_opacity"] = data_flow_stats.last_diag_zero_opacity;
		diagnostics["nan_pos"] = data_flow_stats.last_diag_nan_pos;
		diagnostics["total"] = data_flow_stats.last_diag_total;
		snapshot["chunk_diagnostics"] = diagnostics;
	}

	if (data_flow_stats.instance_buffer_updates > 0) {
		Dictionary instance_buffer;
		instance_buffer["updates"] = static_cast<int64_t>(data_flow_stats.instance_buffer_updates);
		instance_buffer["out_count"] = data_flow_stats.last_instance_buffer_count;
		instance_buffer["world_count"] = data_flow_stats.last_instance_world_count;
		instance_buffer["skip_count"] = data_flow_stats.last_instance_skip_count;
		snapshot["instance_buffer"] = instance_buffer;
	}

	if (data_flow_stats.instance_count_updates > 0) {
		Dictionary instance_counts;
		instance_counts["updates"] = static_cast<int64_t>(data_flow_stats.instance_count_updates);
		instance_counts["visible_clamped"] = data_flow_stats.last_instance_visible_clamped;
		instance_counts["visible_raw"] = data_flow_stats.last_instance_visible_raw;
		instance_counts["overflow_flag"] = data_flow_stats.last_instance_overflow_flag;
		snapshot["instance_counts"] = instance_counts;
	}

	if (data_flow_stats.instance_flag_updates > 0) {
		Dictionary instance_flags;
		instance_flags["updates"] = static_cast<int64_t>(data_flow_stats.instance_flag_updates);
		instance_flags["total"] = data_flow_stats.last_instance_total;
		instance_flags["rotation_identity"] = data_flow_stats.last_instance_rotation_identity;
		instance_flags["scale_identity"] = data_flow_stats.last_instance_scale_identity;
		instance_flags["translation_zero"] = data_flow_stats.last_instance_translation_zero;
		instance_flags["fully_identity"] = data_flow_stats.last_instance_fully_identity;
		snapshot["instance_flags"] = instance_flags;
	}

	Dictionary recent_window;
	recent_window["capacity"] = static_cast<int64_t>(DATA_FLOW_RECENT_WINDOW_CAPACITY);
	const uint32_t window_count = recent_data_flow_full ? DATA_FLOW_RECENT_WINDOW_CAPACITY : recent_data_flow_cursor;
	const uint32_t window_start = recent_data_flow_full ? recent_data_flow_cursor : 0;
	Array frame_deltas;
	Vector<DataFlowFrameDelta> window_deltas;
	uint64_t total_pack_sh_samples = 0;
	uint64_t total_pack_range_calls = 0;
	uint64_t total_gaussians_check_calls = 0;
	uint64_t total_buffer_mgr_calls = 0;
	uint64_t total_chunk_diagnostics_updates = 0;
	uint64_t total_instance_buffer_updates = 0;
	uint64_t total_instance_count_updates = 0;
	uint64_t total_instance_flag_updates = 0;

	for (uint32_t i = 0; i < window_count; i++) {
		const uint32_t idx = (window_start + i) % DATA_FLOW_RECENT_WINDOW_CAPACITY;
		const DataFlowFrameDelta &delta = recent_data_flow_deltas[idx];
		if (!_delta_has_activity(delta)) {
			continue;
		}
		window_deltas.push_back(delta);
	}

	if (active_data_flow_delta_valid && _delta_has_activity(active_data_flow_delta)) {
		// Keep the exported recent window hard-bounded to declared capacity.
		if (window_deltas.size() >= static_cast<int>(DATA_FLOW_RECENT_WINDOW_CAPACITY)) {
			window_deltas.remove_at(0);
		}
		window_deltas.push_back(active_data_flow_delta);
	}

	for (int i = 0; i < window_deltas.size(); i++) {
		const DataFlowFrameDelta &delta = window_deltas[i];
		total_pack_sh_samples += delta.pack_sh_samples;
		total_pack_range_calls += delta.pack_range_calls;
		total_gaussians_check_calls += delta.gaussians_check_calls;
		total_buffer_mgr_calls += delta.buffer_mgr_calls;
		total_chunk_diagnostics_updates += delta.chunk_diagnostics_updates;
		total_instance_buffer_updates += delta.instance_buffer_updates;
		total_instance_count_updates += delta.instance_count_updates;
		total_instance_flag_updates += delta.instance_flag_updates;
		Dictionary frame_delta;
		frame_delta["frame"] = static_cast<int64_t>(delta.frame);
		frame_delta["pack_sh_samples"] = static_cast<int64_t>(delta.pack_sh_samples);
		frame_delta["pack_range_calls"] = static_cast<int64_t>(delta.pack_range_calls);
		frame_delta["gaussians_check_calls"] = static_cast<int64_t>(delta.gaussians_check_calls);
		frame_delta["buffer_mgr_calls"] = static_cast<int64_t>(delta.buffer_mgr_calls);
		frame_delta["chunk_diagnostics_updates"] = static_cast<int64_t>(delta.chunk_diagnostics_updates);
		frame_delta["instance_buffer_updates"] = static_cast<int64_t>(delta.instance_buffer_updates);
		frame_delta["instance_count_updates"] = static_cast<int64_t>(delta.instance_count_updates);
		frame_delta["instance_flag_updates"] = static_cast<int64_t>(delta.instance_flag_updates);
		frame_deltas.push_back(frame_delta);
	}

	recent_window["frames_recorded"] = static_cast<int64_t>(frame_deltas.size());
	recent_window["pack_sh_samples"] = static_cast<int64_t>(total_pack_sh_samples);
	recent_window["pack_range_calls"] = static_cast<int64_t>(total_pack_range_calls);
	recent_window["gaussians_check_calls"] = static_cast<int64_t>(total_gaussians_check_calls);
	recent_window["buffer_mgr_calls"] = static_cast<int64_t>(total_buffer_mgr_calls);
	recent_window["chunk_diagnostics_updates"] = static_cast<int64_t>(total_chunk_diagnostics_updates);
	recent_window["instance_buffer_updates"] = static_cast<int64_t>(total_instance_buffer_updates);
	recent_window["instance_count_updates"] = static_cast<int64_t>(total_instance_count_updates);
	recent_window["instance_flag_updates"] = static_cast<int64_t>(total_instance_flag_updates);
	recent_window["frame_deltas"] = frame_deltas;
	snapshot["recent_window"] = recent_window;

	return snapshot;
}

Array debug_trace_get_recent_events() {
	MutexLock lock(debug_mutex);
	Array out;
	if (debug_events.is_empty()) {
		return out;
	}
	const uint32_t count = debug_event_full ? DEBUG_EVENT_CAPACITY : debug_event_cursor;
	if (count == 0) {
		return out;
	}
	const uint32_t start = debug_event_full ? debug_event_cursor : 0;
	out.resize(count);
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t idx = (start + i) % DEBUG_EVENT_CAPACITY;
		const DebugEvent &entry = debug_events[idx];
		Dictionary dict;
		dict["timestamp_usec"] = static_cast<int64_t>(entry.timestamp_usec);
		dict["category"] = entry.category;
		dict["message"] = entry.message;
		dict["is_error"] = entry.is_error;
		out[i] = dict;
	}
	return out;
}

} // namespace GaussianSplatting
