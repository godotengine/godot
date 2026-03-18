/**
 * tile_render_prefix_scan.cpp — TileRenderer::TilePrefixScanStage method implementations.
 *
 * Companion .cpp for tile_renderer.h / tile_render_stages.h.
 * Contains the three-pass tile prefix scan (histogram, hierarchical sum,
 * finalize), CPU emergency fallback, overflow mode selection, and the
 * update_global_tile_ranges orchestrator.
 *
 * Pattern 10 (Flyweight + GPU resource cache): prefix uniform sets are
 * cached flyweight references into shared GPU resources.
 */

#include "tile_renderer.h"
#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/math_funcs.h"
#include "core/object/callable_method_pointer.h"
#include "servers/rendering/rendering_device.h"
#include "core/templates/hash_map.h"
#include "servers/rendering/renderer_rd/storage_rd/light_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"

#include "gpu_debug_utils.h"
#include "gpu_performance_monitor.h"
#include "gpu_sorter.h"
#include "gpu_sorting_config.h"
#include "quantization_config.h"
#include "resource_owner_mismatch_contract.h"
#include "../logger/gs_logger.h"
#include "gaussian_gpu_layout.h"
#include "pipeline_io_contracts.h"
#include "shader_compilation_helper.h"
#include "sh_config.h"
#include "tile_prefix_scan_utils.h"
#include "../interfaces/sync_policy.h"
#include "../shaders/tile_resolve.glsl.gen.h"

using GaussianSplatting::PassColors;
using GaussianSplatting::ScopedGpuMarker;
using GaussianSplatting::ScopedGpuMarkerEx;

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

static void _release_uniform_set(RenderingDevice *p_device, RID &p_uniform_set) {
	if (!p_uniform_set.is_valid()) {
		return;
	}
	if (p_device && p_device->uniform_set_is_valid(p_uniform_set)) {
		p_device->free(p_uniform_set);
	}
	p_uniform_set = RID();
}

} // namespace

TileRenderer::TilePrefixScanStage::PrefixParams TileRenderer::TilePrefixScanStage::build_prefix_params() const {
    PrefixParams prefix_params;
    prefix_params.total_tiles = owner.grid_state.total_tiles;
    prefix_params.workgroup_stride = GaussianSplatting::kTilePrefixPassLocalSize;
    prefix_params.total_workgroups = GaussianSplatting::tile_prefix_compute_total_workgroups(owner.grid_state.total_tiles);
    prefix_params.global_sort_capacity = owner._get_effective_overlap_capacity();
    return prefix_params;
}

TileRenderer::TilePrefixScanStage::PrefixOverflowMode TileRenderer::TilePrefixScanStage::decide_overflow_mode(
        bool p_allow_sync_readback, bool p_used_cpu_fallback) const {
    if (p_used_cpu_fallback) {
        return PrefixOverflowMode::CPU_EMERGENCY;
    }
    if (p_allow_sync_readback || g_gpu_sorting_config.debug_validate_prefix) {
        return PrefixOverflowMode::DETERMINISTIC_SYNC_READBACK;
    }
    return PrefixOverflowMode::ASYNC_ESTIMATE;
}

bool TileRenderer::TilePrefixScanStage::dispatch_pass1(const PrefixDispatchContext &p_context) const {
    if (!p_context.device || p_context.compute_list == RD::INVALID_ID || p_context.workgroup_count == 0u) {
        return false;
    }
    if (!p_context.buffer_uniform_set.is_valid() || !p_context.param_uniform_set.is_valid() ||
            !owner.shader_resources.tile_prefix_pipeline_pass1.is_valid()) {
        return false;
    }
    p_context.device->compute_list_bind_compute_pipeline(p_context.compute_list, owner.shader_resources.tile_prefix_pipeline_pass1);
    p_context.device->compute_list_bind_uniform_set(p_context.compute_list, p_context.buffer_uniform_set, 0);
    p_context.device->compute_list_bind_uniform_set(p_context.compute_list, p_context.param_uniform_set, 1);
    p_context.device->compute_list_dispatch(p_context.compute_list, p_context.workgroup_count, 1, 1);
    return true;
}

bool TileRenderer::TilePrefixScanStage::dispatch_pass2_command(const PrefixDispatchContext &p_context,
        const GaussianSplatting::TilePrefixPass2ControlLayout &p_control) const {
    if (!p_context.device || p_context.compute_list == RD::INVALID_ID || p_context.workgroup_count == 0u) {
        return false;
    }
    if (!p_context.buffer_uniform_set.is_valid() || !p_context.param_uniform_set.is_valid() ||
            !owner.shader_resources.tile_prefix_pipeline_pass2.is_valid()) {
        return false;
    }
    const uint32_t dispatch_x = GaussianSplatting::tile_prefix_compute_dispatch_groups(p_context.workgroup_count);
    if (dispatch_x == 0u) {
        return false;
    }
    p_context.device->compute_list_bind_compute_pipeline(p_context.compute_list, owner.shader_resources.tile_prefix_pipeline_pass2);
    p_context.device->compute_list_bind_uniform_set(p_context.compute_list, p_context.buffer_uniform_set, 0);
    p_context.device->compute_list_bind_uniform_set(p_context.compute_list, p_context.param_uniform_set, 1);
    p_context.device->compute_list_set_push_constant(p_context.compute_list, &p_control, sizeof(p_control));
    p_context.device->compute_list_dispatch(p_context.compute_list, dispatch_x, 1, 1);
    return true;
}

bool TileRenderer::TilePrefixScanStage::dispatch_pass2_hierarchical(const PrefixDispatchContext &p_context) const {
    if (!p_context.device || p_context.compute_list == RD::INVALID_ID || p_context.workgroup_count == 0u) {
        return false;
    }

    bool source_is_offsets = false;
    for (uint64_t stride64 = 1u; stride64 < uint64_t(p_context.workgroup_count); stride64 <<= 1u) {
        GaussianSplatting::TilePrefixPass2ControlLayout control;
        control.operation = GaussianSplatting::TILE_PREFIX_PASS2_OP_INCLUSIVE_STEP;
        control.source_buffer = source_is_offsets
                ? GaussianSplatting::TILE_PREFIX_PASS2_SOURCE_WG_OFFSETS
                : GaussianSplatting::TILE_PREFIX_PASS2_SOURCE_WG_SUMS;
        control.stride = uint32_t(stride64);
        if (!dispatch_pass2_command(p_context, control)) {
            return false;
        }
        p_context.device->compute_list_add_barrier(p_context.compute_list);
        source_is_offsets = !source_is_offsets;
    }

    GaussianSplatting::TilePrefixPass2ControlLayout exclusive_control;
    exclusive_control.operation = GaussianSplatting::TILE_PREFIX_PASS2_OP_EXCLUSIVE_SHIFT;
    exclusive_control.source_buffer = source_is_offsets
            ? GaussianSplatting::TILE_PREFIX_PASS2_SOURCE_WG_OFFSETS
            : GaussianSplatting::TILE_PREFIX_PASS2_SOURCE_WG_SUMS;
    exclusive_control.stride = 0u;
    if (!dispatch_pass2_command(p_context, exclusive_control)) {
        return false;
    }

    if (source_is_offsets) {
        p_context.device->compute_list_add_barrier(p_context.compute_list);
        GaussianSplatting::TilePrefixPass2ControlLayout copy_control;
        copy_control.operation = GaussianSplatting::TILE_PREFIX_PASS2_OP_COPY;
        copy_control.source_buffer = GaussianSplatting::TILE_PREFIX_PASS2_SOURCE_WG_SUMS;
        copy_control.stride = 0u;
        if (!dispatch_pass2_command(p_context, copy_control)) {
            return false;
        }
    }

    return true;
}

bool TileRenderer::TilePrefixScanStage::dispatch_pass3(const PrefixDispatchContext &p_context) const {
    if (!p_context.device || p_context.compute_list == RD::INVALID_ID || p_context.workgroup_count == 0u) {
        return false;
    }
    if (!p_context.buffer_uniform_set.is_valid() || !p_context.param_uniform_set.is_valid() ||
            !owner.shader_resources.tile_prefix_pipeline_pass3.is_valid()) {
        return false;
    }
    p_context.device->compute_list_bind_compute_pipeline(p_context.compute_list, owner.shader_resources.tile_prefix_pipeline_pass3);
    p_context.device->compute_list_bind_uniform_set(p_context.compute_list, p_context.buffer_uniform_set, 0);
    p_context.device->compute_list_bind_uniform_set(p_context.compute_list, p_context.param_uniform_set, 1);
    p_context.device->compute_list_dispatch(p_context.compute_list, p_context.workgroup_count, 1, 1);
    return true;
}

RID TileRenderer::TilePrefixScanStage::create_prefix_param_uniform_set(RenderingDevice *p_device, const PrefixParams &p_params) {
    ERR_FAIL_NULL_V(p_device, RID());
    if (!owner._ensure_prefix_param_buffer(p_device, sizeof(PrefixParams))) {
        return RID();
    }
    const RID param_buffer = owner.uniform_buffers.prefix_param_uniform_buffer;
    const bool params_changed = !cached_prefix_params_valid ||
            std::memcmp(&cached_prefix_params, &p_params, sizeof(PrefixParams)) != 0;
    const bool buffer_changed = cached_prefix_param_buffer != param_buffer;
    if (params_changed || buffer_changed) {
        p_device->buffer_update(param_buffer, 0, sizeof(PrefixParams), &p_params);
        cached_prefix_params = p_params;
        cached_prefix_params_valid = true;
        cached_prefix_param_buffer = param_buffer;
    }

    Vector<RD::Uniform> uniforms;
    RD::Uniform params_uniform;
    params_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
    params_uniform.binding = 0;
    params_uniform.append_id(param_buffer);
    uniforms.push_back(params_uniform);

    RID prefix_param_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_prefix_shader, 1);
    if (prefix_param_uniform_set.is_valid()) {
        p_device->set_resource_name(prefix_param_uniform_set, "GS_TileRenderer_PrefixParamSet");
    }
    return prefix_param_uniform_set;
}

bool TileRenderer::TilePrefixScanStage::run_cpu_prefix_fallback(RenderingDevice *p_device, const PrefixParams &p_params,
        uint32_t &r_record_count, uint32_t &r_raw_record_count) {
    ERR_FAIL_NULL_V(p_device, false);
    if (p_params.total_tiles == 0u) {
        r_record_count = 0u;
        r_raw_record_count = 0u;
        return true;
    }

    const uint64_t counts_bytes = uint64_t(p_params.total_tiles) * sizeof(uint32_t);
    Vector<uint8_t> counts_data = p_device->buffer_get_data(owner.global_sort_resources.get_tile_counts_buffer(), 0, counts_bytes);
    if (uint64_t(counts_data.size()) != counts_bytes) {
        GS_LOG_ERROR_DEFAULT(vformat("[TileRenderer] CPU prefix fallback failed to read tile counts (expected=%s bytes, got=%d)",
                String::num_uint64(counts_bytes), counts_data.size()));
        return false;
    }

    Vector<uint32_t> tile_counts_words;
    tile_counts_words.resize(p_params.total_tiles);
    std::memcpy(tile_counts_words.ptrw(), counts_data.ptr(), size_t(counts_bytes));

    Vector<uint32_t> tile_ranges_words;
    tile_ranges_words.resize(p_params.total_tiles * 2u);

    GaussianSplatting::TilePrefixCpuScanResult cpu_result;
    if (!GaussianSplatting::compute_tile_prefix_cpu(tile_counts_words.ptr(),
                p_params.total_tiles, tile_ranges_words.ptrw(), p_params.global_sort_capacity, cpu_result)) {
        GS_LOG_ERROR_DEFAULT("[TileRenderer] CPU prefix fallback failed while building ranges");
        return false;
    }

    if (cpu_result.raw_total_saturated) {
        WARN_PRINT_ONCE("[TileRenderer] CPU prefix fallback saturated total overlap count to 32-bit max");
    }

    const uint64_t tile_ranges_bytes = uint64_t(tile_ranges_words.size()) * sizeof(uint32_t);
    p_device->buffer_update(owner.global_sort_resources.tile_ranges_buffer, 0, tile_ranges_bytes, tile_ranges_words.ptr());

    p_device->buffer_update(owner.global_sort_resources.prefix_total_buffer, 0, sizeof(uint32_t), &cpu_result.raw_total);

    if (owner.global_sort_resources.indirect_dispatch_buffer.is_valid()) {
        p_device->buffer_update(owner.global_sort_resources.indirect_dispatch_buffer, 0,
                sizeof(GaussianSplatting::IndirectDispatchLayout), &cpu_result.indirect_dispatch);
    }

    r_raw_record_count = cpu_result.raw_total;
    r_record_count = cpu_result.indirect_dispatch.element_count;

    owner.perf_metrics.profiling_cached_overlap_total = r_raw_record_count;
    owner.diagnostics.last_overlap_record_budget_effective = owner._get_effective_overlap_capacity();

    owner.async_readback.overflow_state.pending_readback = false;
    owner.async_readback.overflow_state.requested_frame_serial = 0;
    owner.async_readback.overflow_state.overflow_detected = cpu_result.indirect_dispatch.overflow_flag != 0u;
    owner.async_readback.overflow_state.last_unclamped_total = cpu_result.raw_total;
    owner.async_readback.overflow_state.first_frame_complete = true;

    return true;
}

bool TileRenderer::TilePrefixScanStage::update_global_tile_ranges(const RID &p_gaussian_buffer, const RID &p_sorted_indices,
        RenderingDevice *p_device,
		uint32_t &r_record_count, uint32_t &r_raw_record_count, bool p_allow_sync_readback) {
	RenderingDevice *device = p_device ? p_device : owner._get_resource_device();
	if (!device || owner.grid_state.total_tiles == 0) {
		return false;
	}
	if (!owner.global_sort_resources.get_tile_counts_buffer().is_valid() || !owner.global_sort_resources.tile_ranges_buffer.is_valid() ||
			!owner.global_sort_resources.wg_sums_buffer.is_valid() || !owner.global_sort_resources.wg_offsets_buffer.is_valid()) {
		return false;
	}
	if (owner.global_sort_resources.tile_buffer_tiles != owner.grid_state.total_tiles) {
		return false;
	}

	PrefixParams prefix_params = build_prefix_params();
	uint32_t wg_count = prefix_params.total_workgroups;
	if (wg_count == 0u) {
		r_record_count = 0u;
		r_raw_record_count = 0u;
		return true;
	}

	const uint64_t max_dispatch_groups_x_u64 = MAX<uint64_t>(1u,
			device->limit_get(RenderingDevice::LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X));
	const uint32_t max_dispatch_groups_x = max_dispatch_groups_x_u64 > uint64_t(UINT32_MAX)
			? UINT32_MAX
			: uint32_t(max_dispatch_groups_x_u64);
	const GaussianSplatting::TilePrefixDispatchCounts dispatch_counts =
			GaussianSplatting::tile_prefix_compute_dispatch_counts(wg_count);
	const bool pass1_exceeds_limit =
			GaussianSplatting::tile_prefix_pass1_requires_cpu_fallback(wg_count, max_dispatch_groups_x);
	const bool pass2_exceeds_limit =
			GaussianSplatting::tile_prefix_pass2_requires_cpu_fallback(wg_count, max_dispatch_groups_x);
	const bool pass3_exceeds_limit =
			GaussianSplatting::tile_prefix_pass3_requires_cpu_fallback(wg_count, max_dispatch_groups_x);
	if (pass1_exceeds_limit || pass2_exceeds_limit || pass3_exceeds_limit) {
		WARN_PRINT_ONCE(vformat("[TileRenderer] Prefix GPU dispatch exceeds device limit; using CPU emergency fallback "
				"(workgroups=%u pass1_x=%u pass2_x=%u pass3_x=%u max_dispatch_x=%u exceeded_passes={%u,%u,%u})",
				wg_count,
				dispatch_counts.pass1_dispatch_x,
				dispatch_counts.pass2_dispatch_x,
				dispatch_counts.pass3_dispatch_x,
				max_dispatch_groups_x,
				pass1_exceeds_limit ? 1u : 0u,
				pass2_exceeds_limit ? 1u : 0u,
				pass3_exceeds_limit ? 1u : 0u));
		return run_cpu_prefix_fallback(device, prefix_params, r_record_count, r_raw_record_count);
	}

	RID prefix_param_uniform_set = create_prefix_param_uniform_set(device, prefix_params);
	if (!prefix_param_uniform_set.is_valid()) {
		return false;
	}

	auto free_prefix_param_set = [&]() {
		if (prefix_param_uniform_set.is_valid() && device->uniform_set_is_valid(prefix_param_uniform_set)) {
			device->free(prefix_param_uniform_set);
		}
		prefix_param_uniform_set = RID();
	};

	RID prefix_buffer_uniform_set = acquire_prefix_uniform_set(device);
	if (!prefix_buffer_uniform_set.is_valid()) {
		free_prefix_param_set();
		return false;
	}

	const PrefixOverflowMode overflow_mode = decide_overflow_mode(p_allow_sync_readback, false);
	const bool allow_sync_readback = overflow_mode == PrefixOverflowMode::DETERMINISTIC_SYNC_READBACK;

	// Measure prefix scan time (CPU wall-clock including GPU sync when enabled).
	uint64_t prefix_start_usec = allow_sync_readback ? OS::get_singleton()->get_ticks_usec() : 0;

	uint32_t prefix_timestamp_base = device->get_captured_timestamps_count();
	String prefix_label = "TilePrefix_" + String::num_uint64(owner.frame_state.current_frame_serial);
	device->capture_timestamp(prefix_label + String("_Begin"));
	ScopedGpuMarker prefix_marker(device, "GS_TilePrefix", PassColors::PREFIX);

	RD::ComputeListID prefix_list = device->compute_list_begin();
	if (prefix_list == RD::INVALID_ID) {
		ERR_PRINT_ONCE("[TileRenderer] Failed to begin prefix scan compute list");
		free_prefix_param_set();
		return false;
	}

	PrefixDispatchContext dispatch_context;
	dispatch_context.device = device;
	dispatch_context.compute_list = prefix_list;
	dispatch_context.buffer_uniform_set = prefix_buffer_uniform_set;
	dispatch_context.param_uniform_set = prefix_param_uniform_set;
	dispatch_context.workgroup_count = wg_count;

	// Pass 1: Per-workgroup histogram - compute local prefix sums within each workgroup.
	{
		ScopedGpuMarkerEx pass1_marker(device, "GS_PrefixScan_Pass1_Histogram", PassColors::PREFIX);
		if (!dispatch_pass1(dispatch_context)) {
			device->compute_list_end();
			free_prefix_param_set();
			return false;
		}
	}
	device->compute_list_add_barrier(prefix_list);

	// Pass 2: Hierarchical workgroup-prefix scan over wg_sums.
	{
		ScopedGpuMarkerEx pass2_marker(device, "GS_PrefixScan_Pass2_WGSumHierarchical", PassColors::PREFIX);
		if (!dispatch_pass2_hierarchical(dispatch_context)) {
			device->compute_list_end();
			free_prefix_param_set();
			return false;
		}
	}
	device->compute_list_add_barrier(prefix_list);

	// Pass 3: Write total + ranges - add offsets to tile ranges.
	{
		ScopedGpuMarkerEx pass3_marker(device, "GS_PrefixScan_Pass3_Finalize", PassColors::PREFIX);
		if (!dispatch_pass3(dispatch_context)) {
			device->compute_list_end();
			free_prefix_param_set();
			return false;
		}
	}
	device->compute_list_add_barrier(prefix_list);

	device->compute_list_end();
	owner._queue_submission(device, false);

	device->capture_timestamp(prefix_label + String("_End"));
	owner.timing_state.prefix_timestamp.device = device;
	owner.timing_state.prefix_timestamp.start_index = prefix_timestamp_base;
	owner.timing_state.prefix_timestamp.end_index = prefix_timestamp_base + 1;
	owner.timing_state.prefix_timestamp.label = prefix_label;

	if (allow_sync_readback) {
		owner._flush_pending_submission(true);
		uint64_t prefix_end_usec = OS::get_singleton()->get_ticks_usec();
		owner.timing_state.last_prefix_gpu_ms = float(prefix_end_usec - prefix_start_usec) / 1000.0f;
	} else {
		owner.timing_state.last_prefix_gpu_ms = 0.0f;
	}

	struct IndirectDispatchReadback {
		uint32_t dispatch_xyz[3];
		uint32_t element_count;
		uint32_t overflow_flag;
		uint32_t unclamped_total;
	};

	uint32_t total_records = 0;
	if (!allow_sync_readback) {
		if (owner.async_readback.overflow_state.pending_readback) {
			const uint64_t request_frame = owner.async_readback.overflow_state.requested_frame_serial;
			if (request_frame > 0 && owner.frame_state.current_frame_serial > request_frame + 8u) {
				WARN_PRINT_ONCE("[TileRenderer] Async overlap readback timed out; retrying request");
				owner.async_readback.overflow_state.pending_readback = false;
				owner.async_readback.overflow_state.requested_frame_serial = 0;
			}
		}

		// GPU-driven path: issue async readback for overflow detection (one frame latency).
		// The indirect buffer layout matches IndirectDispatchLayout (element_count onwards).
		if (!owner.async_readback.overflow_state.pending_readback &&
				owner.global_sort_resources.indirect_dispatch_buffer.is_valid()) {
			const uint64_t request_frame_serial = owner.frame_state.current_frame_serial;
			owner.async_readback.overflow_state.pending_readback = true;
			owner.async_readback.overflow_state.requested_frame_serial = request_frame_serial;
			Callable callback = callable_mp(&owner, &TileRenderer::_on_overflow_flag_readback).bind(int64_t(request_frame_serial));
			Error err = device->buffer_get_data_async(
					owner.global_sort_resources.indirect_dispatch_buffer,
					callback,
					GaussianSplatting::kIndirectDispatchElementCountOffset,
					GaussianSplatting::kIndirectDispatchElementCountReadbackSize);
			if (err != OK) {
				owner.async_readback.overflow_state.pending_readback = false;
				owner.async_readback.overflow_state.requested_frame_serial = 0;
			}
		}

		// Async estimate: use effective capacity as the CPU-side record count.
		// The GPU prefix scan already wrote the true clamped element_count into the
		// indirect dispatch buffer — this value only drives CPU-side diagnostics
		// and buffer sizing decisions, not GPU dispatch.
		const uint32_t effective_capacity = owner._get_effective_overlap_capacity();
		total_records = effective_capacity > 0 ? effective_capacity : 1u;
		uint32_t fallback_unclamped = owner.async_readback.overflow_state.last_unclamped_total;
		if (fallback_unclamped == 0) {
			fallback_unclamped = total_records;
		}

		// DEBUG: Log async path overlap stats (gated by debug flag and throttled)
		static int async_log_counter = 0;
		const int log_interval = owner.diagnostics.debug_frame_log_frequency;
		if (owner.diagnostics.debug_tile_pipeline_logs_enabled && log_interval > 0 &&
				(++async_log_counter == 1 || (async_log_counter % log_interval == 0))) {
			GS_LOG_RENDERER_DEBUG(vformat("[TileRenderer-ASYNC] estimate=%d unclamped=%d capped=%d effective_capacity=%d overflow=%s pending=%s",
					fallback_unclamped,
					owner.async_readback.overflow_state.last_unclamped_total,
					total_records,
					effective_capacity,
					owner.async_readback.overflow_state.overflow_detected ? "YES" : "no",
					owner.async_readback.overflow_state.pending_readback ? "yes" : "no"));
		}

		owner.perf_metrics.profiling_cached_overlap_total = fallback_unclamped;
		r_record_count = total_records;
		r_raw_record_count = fallback_unclamped;
		owner.diagnostics.last_overlap_record_budget_effective = effective_capacity;
		free_prefix_param_set();
		return true;
	}

	// Read back the total overlap count from the GPU (enabled for validation or explicit readback).
	// Cached/estimated values are used by default to avoid per-frame GPU/CPU stalls.
	Vector<uint8_t> total_bytes = device->buffer_get_data(owner.global_sort_resources.prefix_total_buffer, 0, sizeof(uint32_t));
	if (total_bytes.size() < sizeof(uint32_t)) {
		GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to read global overlap total");
		free_prefix_param_set();
		return false;
	}
	const uint32_t *total_ptr = reinterpret_cast<const uint32_t *>(total_bytes.ptr());
	uint64_t total_records64 = uint64_t(*total_ptr);
	if (total_records64 > UINT32_MAX) {
		WARN_PRINT_ONCE(vformat("[TileRenderer] Overlap total exceeds 32-bit range; truncating to %u (raw=%s)",
				UINT32_MAX, String::num_uint64(total_records64)));
	}
	total_records = total_records64 > UINT32_MAX ? UINT32_MAX : uint32_t(total_records64);

	uint32_t raw_total_records = total_records;
	if (owner.global_sort_resources.indirect_dispatch_buffer.is_valid()) {
		Vector<uint8_t> indirect_bytes = device->buffer_get_data(owner.global_sort_resources.indirect_dispatch_buffer, 0,
				sizeof(IndirectDispatchReadback));
		if (indirect_bytes.size() >= int(sizeof(IndirectDispatchReadback))) {
			const IndirectDispatchReadback *indirect = reinterpret_cast<const IndirectDispatchReadback *>(indirect_bytes.ptr());
			total_records = indirect->element_count;
			raw_total_records = indirect->unclamped_total;
			// DEBUG: Log indirect buffer values
			static int indirect_debug_counter = 0;
			const int indirect_log_interval = owner.diagnostics.debug_frame_log_frequency;
			if (owner.diagnostics.debug_tile_pipeline_logs_enabled && indirect_log_interval > 0 &&
					(++indirect_debug_counter == 1 || (indirect_debug_counter % indirect_log_interval == 0))) {
				GS_LOG_RENDERER_DEBUG(vformat("[INDIRECT-DEBUG] element_count=%d unclamped=%d overflow=%d dispatch=(%d,%d,%d)",
						indirect->element_count, indirect->unclamped_total, indirect->overflow_flag,
						indirect->dispatch_xyz[0], indirect->dispatch_xyz[1], indirect->dispatch_xyz[2]));
			}
		}
	}

	owner.perf_metrics.profiling_cached_overlap_total = raw_total_records;

	// DEBUG: Sample center tile ranges every 10 frames to catch intermittent issues
	static int tile_range_debug_counter = 0;
	const int tile_range_log_interval = owner.diagnostics.debug_frame_log_frequency;
	if (owner.diagnostics.debug_tile_pipeline_logs_enabled && tile_range_log_interval > 0 &&
			(++tile_range_debug_counter == 1 || (tile_range_debug_counter % tile_range_log_interval == 0)) &&
			owner.grid_state.total_tiles > 0) {
		// Sample a tile near the center of the screen
		uint32_t center_tile = owner.grid_state.total_tiles / 2;
		uint32_t sample_count = MIN(owner.grid_state.total_tiles - center_tile, 4u);
		uint64_t offset = uint64_t(center_tile) * sizeof(uint32_t) * 2u;
		Vector<uint8_t> sample_bytes = device->buffer_get_data(owner.global_sort_resources.tile_ranges_buffer, offset,
				sample_count * sizeof(uint32_t) * 2u);
		if (sample_bytes.size() >= int(sample_count * sizeof(uint32_t) * 2u)) {
			const uint32_t *ranges = reinterpret_cast<const uint32_t *>(sample_bytes.ptr());
			uint32_t total_sampled = 0;
			for (uint32_t i = 0; i < sample_count; i++) {
				total_sampled += ranges[i * 2 + 1]; // count
			}
			GS_LOG_RENDERER_DEBUG(vformat("[TILE-RANGES] center_tile=%d ranges: [%d,%d] [%d,%d] [%d,%d] [%d,%d] element_count=%d",
					center_tile,
					ranges[0], ranges[1], ranges[2], ranges[3], ranges[4], ranges[5], ranges[6], ranges[7],
					total_records));
		}
	}

	// One-time validation: verify prefix sum correctness.
	static bool prefix_validated = false;
	if (!prefix_validated && owner.grid_state.total_tiles > 1) {
		prefix_validated = true;
		// Read first few ranges to verify prefix[i] == prefix[i-1] + count[i-1]
		uint32_t check_count = MIN(owner.grid_state.total_tiles, 16u);
		Vector<uint8_t> check_bytes = device->buffer_get_data(owner.global_sort_resources.tile_ranges_buffer, 0,
				check_count * sizeof(uint32_t) * 2u);
		if (check_bytes.size() >= int(check_count * sizeof(uint32_t) * 2u)) {
			const uint32_t *ranges = reinterpret_cast<const uint32_t *>(check_bytes.ptr());
			bool valid = true;
			uint32_t expected_prefix = 0;
			for (uint32_t i = 0; i < check_count && valid; i++) {
				uint32_t prefix = ranges[i * 2 + 0];
				uint32_t count = ranges[i * 2 + 1];
				if (prefix != expected_prefix) {
					GS_LOG_ERROR_DEFAULT(vformat("[TileRenderer] GPU prefix sum INVALID: tile %d prefix=%d expected=%d", i, prefix, expected_prefix));
					valid = false;
				}
				expected_prefix += count;
			}
            if (owner.diagnostics.debug_tile_pipeline_logs_enabled && valid) {
                GS_LOG_RENDERER_DEBUG(vformat("[TileRenderer] GPU prefix sum VALIDATED: first %d tiles correct, total_records=%d", check_count, total_records));
            }
		}
	}

#ifdef DEBUG_ENABLED
	if (g_gpu_sorting_config.debug_validate_prefix) {
		const uint64_t counts_bytes = uint64_t(owner.grid_state.total_tiles) * sizeof(uint32_t);
		Vector<uint8_t> counts_data = device->buffer_get_data(owner.global_sort_resources.get_tile_counts_buffer(), 0, counts_bytes);
		if (counts_data.size() == counts_bytes) {
			const uint32_t *counts = reinterpret_cast<const uint32_t *>(counts_data.ptr());
			uint64_t cpu_total64 = 0;
			for (uint32_t i = 0; i < owner.grid_state.total_tiles; i++) {
				cpu_total64 += counts[i];
			}
			if (cpu_total64 != uint64_t(total_records)) {
				GS_LOG_ERROR_DEFAULT(vformat("[TileRenderer] Prefix validation mismatch: GPU total=%d CPU total=%s (tiles=%d)",
						total_records,
						String::num_uint64(cpu_total64),
						owner.grid_state.total_tiles));
			}
		} else {
			GS_LOG_ERROR_DEFAULT("[TileRenderer] Prefix validation failed: unable to read tile counts buffer");
		}
	}
#endif

	// Preserve the raw total before capping so callers can decide on resizing/retries.
	r_raw_record_count = raw_total_records;

	// Enforce the user-configurable overlap budget so emit/sort never write past buffer capacity.
	// The budget is controlled via ProjectSettings: rendering/gaussian_splatting/gpu_sorting/max_overlap_records
	// Default: 100 million (~1.2 GB VRAM). Users can increase for large scenes or decrease for VRAM-limited GPUs.
	uint64_t max_records = uint64_t(UINT32_MAX);
	const uint32_t effective_capacity = owner._get_effective_overlap_capacity();
	if (effective_capacity > 0) {
		max_records = MIN<uint64_t>(max_records, uint64_t(effective_capacity));
	}
	// DEBUG: Log overlap stats (throttled to once per second at 60 FPS)
	static int overlap_stats_log_counter = 0;
	const int overlap_log_interval = owner.diagnostics.debug_frame_log_frequency;
	if (owner.diagnostics.debug_tile_pipeline_logs_enabled && overlap_log_interval > 0 &&
			(++overlap_stats_log_counter == 1 || (overlap_stats_log_counter % overlap_log_interval == 0))) {
		GS_LOG_RENDERER_DEBUG(vformat("[TileRenderer] Overlap stats: raw=%s sort_capacity=%s effective_overlap_budget=%s configured_overlap_budget=%d",
				String::num_uint64(raw_total_records),
				String::num_uint64(owner.global_sort_resources.capacity),
				String::num_uint64(max_records),
				g_gpu_sorting_config.max_overlap_records));
	}

	if (uint64_t(raw_total_records) > max_records) {
		// Emit a warning that the overlap budget was exceeded - this causes visual cutoff.
		// Users should increase max_overlap_records in ProjectSettings if this happens frequently.
		static int budget_exceeded_log_counter = 0;
		if (++budget_exceeded_log_counter % 60 == 1) {
			WARN_PRINT_ED(vformat("[TileRenderer] Overlap record budget exceeded: %s records requested, capped to %s. "
					"Increase 'rendering/gaussian_splatting/gpu_sorting/max_overlap_records' (currently %d) for large/close scenes.",
					String::num_uint64(raw_total_records),
					String::num_uint64(max_records),
					g_gpu_sorting_config.max_overlap_records));
		}
		total_records = uint32_t(max_records);
	}

	r_record_count = total_records;
	owner.diagnostics.last_overlap_record_budget_effective = uint32_t(MIN<uint64_t>(max_records, uint64_t(UINT32_MAX)));
	owner.perf_metrics.profiling_cached_overlap_total = raw_total_records;

	free_prefix_param_set();
	return true;
}

uint64_t TileRenderer::TilePrefixScanStage::dispatch_prefix(uint32_t p_dispatch_x, RID p_pipeline, RID p_buffer_uniform_set,
        RID p_param_uniform_set, RenderingDevice *p_submission_device, bool p_requires_sync) {
	if (!p_pipeline.is_valid() || p_dispatch_x == 0) {
		return 0;
	}

	RenderingDevice *submission_device = p_submission_device;
	if (!submission_device) {
		return 0;
	}

	RD::ComputeListID compute_list = submission_device->compute_list_begin();
	if (compute_list == RD::INVALID_ID) {
		ERR_PRINT_ONCE("[TileRenderer] Failed to begin prefix dispatch compute list");
		return 0;
	}

	submission_device->compute_list_bind_compute_pipeline(compute_list, p_pipeline);
	submission_device->compute_list_bind_uniform_set(compute_list, p_buffer_uniform_set, 0);
	submission_device->compute_list_bind_uniform_set(compute_list, p_param_uniform_set, 1);
	submission_device->compute_list_dispatch(compute_list, p_dispatch_x, 1, 1);
	submission_device->compute_list_end();

	owner._queue_submission(submission_device, p_requires_sync);

	return 0;
}

RID TileRenderer::TilePrefixScanStage::acquire_prefix_uniform_set(RenderingDevice *p_device) {
	ERR_FAIL_NULL_V(p_device, RID());
	ERR_FAIL_COND_V(!owner.global_sort_resources.get_tile_counts_buffer().is_valid(), RID());
	ERR_FAIL_COND_V(!owner.global_sort_resources.tile_ranges_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.global_sort_resources.prefix_total_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.global_sort_resources.indirect_dispatch_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.global_sort_resources.wg_sums_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.global_sort_resources.wg_offsets_buffer.is_valid(), RID());

	RID tile_counts_buffer = owner.global_sort_resources.get_tile_counts_buffer();
	const bool cache_deps_match = cached_generation == owner.descriptor_generation &&
			cached_binning_prefix_device == p_device;
	if (!cache_deps_match) {
		if (cached_binning_prefix_uniform_set.is_valid()) {
			RenderingDevice *owner_device = cached_binning_prefix_device ? cached_binning_prefix_device : p_device;
			_release_uniform_set(owner_device, cached_binning_prefix_uniform_set);
		}
		if (cached_binning_prefix_uniform_set_alt.is_valid()) {
			RenderingDevice *owner_device = cached_binning_prefix_device ? cached_binning_prefix_device : p_device;
			_release_uniform_set(owner_device, cached_binning_prefix_uniform_set_alt);
		}
		cached_binning_prefix_uniform_set = RID();
		cached_binning_prefix_uniform_set_alt = RID();
		cached_binning_prefix_device = nullptr;
		cached_binning_prefix_tile_counts = RID();
		cached_binning_prefix_tile_counts_alt = RID();
	}

	if (cache_deps_match && cached_binning_prefix_uniform_set.is_valid() &&
			cached_binning_prefix_tile_counts == tile_counts_buffer) {
		return cached_binning_prefix_uniform_set;
	}
	if (cache_deps_match && cached_binning_prefix_uniform_set_alt.is_valid() &&
			cached_binning_prefix_tile_counts_alt == tile_counts_buffer) {
		return cached_binning_prefix_uniform_set_alt;
	}

	Vector<RD::Uniform> uniforms;

	RD::Uniform counts_uniform;
	counts_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	counts_uniform.binding = 0;
	counts_uniform.append_id(owner.global_sort_resources.get_tile_counts_buffer());
	uniforms.push_back(counts_uniform);

	RD::Uniform ranges_uniform;
	ranges_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	ranges_uniform.binding = 1;
	ranges_uniform.append_id(owner.global_sort_resources.tile_ranges_buffer);
	uniforms.push_back(ranges_uniform);

	RD::Uniform sums_uniform;
	sums_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	sums_uniform.binding = 2;
	sums_uniform.append_id(owner.global_sort_resources.wg_sums_buffer);
	uniforms.push_back(sums_uniform);

	RD::Uniform offsets_uniform;
	offsets_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	offsets_uniform.binding = 3;
	offsets_uniform.append_id(owner.global_sort_resources.wg_offsets_buffer);
	uniforms.push_back(offsets_uniform);

	RD::Uniform total_uniform;
	total_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	total_uniform.binding = 4;
	total_uniform.append_id(owner.global_sort_resources.prefix_total_buffer);
	uniforms.push_back(total_uniform);

	RD::Uniform indirect_uniform;
	indirect_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	indirect_uniform.binding = 5;
	indirect_uniform.append_id(owner.global_sort_resources.indirect_dispatch_buffer);
	uniforms.push_back(indirect_uniform);

	RID new_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_prefix_shader, 0);
	ERR_FAIL_COND_V_MSG(!new_uniform_set.is_valid(), RID(),
			"[TileRenderer] Failed to create global prefix uniform set (tile_prefix_scan.glsl)");
	p_device->set_resource_name(new_uniform_set, "GS_TileRenderer_BinningPrefixSet");

	RID *target_set = &cached_binning_prefix_uniform_set;
	RID *target_counts = &cached_binning_prefix_tile_counts;
	if (cached_binning_prefix_uniform_set.is_valid()) {
		if (!cached_binning_prefix_uniform_set_alt.is_valid()) {
			target_set = &cached_binning_prefix_uniform_set_alt;
			target_counts = &cached_binning_prefix_tile_counts_alt;
		}
	}
	*target_set = new_uniform_set;
	*target_counts = tile_counts_buffer;

	cached_binning_prefix_device = p_device;
	cached_generation = owner.descriptor_generation;
	return *target_set;
}
