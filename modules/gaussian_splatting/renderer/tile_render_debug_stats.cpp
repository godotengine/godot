/**
 * tile_render_debug_stats.cpp — TileRenderer::TileRendererDebugStats method implementations.
 *
 * Companion .cpp for tile_renderer.h / tile_render_stages.h.
 * Contains all debug statistics, GPU counter readback, overflow tracking,
 * and splat audit methods for the TileRendererDebugStats inner class.
 *
 * Pattern 11 (RAII for RIDs): debug_counter_buffer, overflow_statistics_buffer,
 * and debug_splat_audit_buffer are owned RIDs created/freed by this stage.
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

static constexpr uint32_t kDebugSplatAuditInvalidIndex = 0xFFFFFFFFu;
static constexpr uint32_t kSplatAuditFlagProjected = 1u << 0u;
static constexpr uint32_t kSplatAuditFlagInViewport = 1u << 1u;
static constexpr uint32_t kSplatAuditFlagIterated = 1u << 2u;
static constexpr uint32_t kSplatAuditFlagContributed = 1u << 3u;
static constexpr uint32_t kSplatAuditFlagAlphaSkipped = 1u << 4u;

struct DebugSplatAuditHeader {
	uint32_t enabled = 0;
	uint32_t sample_count = 0;
	uint32_t frame_id = 0;
	uint32_t reserved = 0;
};

struct DebugSplatAuditEntry {
	uint32_t global_idx = kDebugSplatAuditInvalidIndex;
	uint32_t expected_x = 0;
	uint32_t expected_y = 0;
	uint32_t flags = 0;
};

struct DebugSplatAuditBuffer {
	DebugSplatAuditHeader header;
	DebugSplatAuditEntry entries[TileRenderer::DEBUG_SPLAT_AUDIT_MAX_SAMPLES];
};

static_assert(sizeof(DebugSplatAuditHeader) == 16, "DebugSplatAuditHeader must be 16 bytes");
static_assert(sizeof(DebugSplatAuditEntry) == 16, "DebugSplatAuditEntry must be 16 bytes");
static_assert(sizeof(DebugSplatAuditBuffer) == sizeof(DebugSplatAuditHeader) +
		sizeof(DebugSplatAuditEntry) * TileRenderer::DEBUG_SPLAT_AUDIT_MAX_SAMPLES,
		"DebugSplatAuditBuffer size mismatch");

} // namespace

void TileRenderer::TileRendererDebugStats::on_debug_counters_readback(const Vector<uint8_t> &p_data) {
    debug_counter_readback.pending = false;
    const size_t debug_bytes = sizeof(DebugCounterSnapshot);
    if ((size_t)p_data.size() < debug_bytes) {
        return;
    }
    const DebugCounterSnapshot *ptr = reinterpret_cast<const DebugCounterSnapshot *>(p_data.ptr());
    cached_debug_counters = *ptr;
    cached_debug_frame_serial = debug_counter_readback.requested_frame_serial;

}

void TileRenderer::TileRendererDebugStats::on_overflow_stats_readback(const Vector<uint8_t> &p_data) {
    overflow_stats_readback.pending = false;
    const size_t overflow_bytes = sizeof(OverflowStatsSnapshot);
    if ((size_t)p_data.size() < overflow_bytes) {
        return;
    }
    const OverflowStatsSnapshot *ptr = reinterpret_cast<const OverflowStatsSnapshot *>(p_data.ptr());
    cached_overflow_stats = *ptr;
    cached_overflow_frame_serial = overflow_stats_readback.requested_frame_serial;
}

void TileRenderer::TileRendererDebugStats::on_splat_audit_readback(const Vector<uint8_t> &p_data) {
    splat_audit_readback.pending = false;
    const size_t audit_bytes = sizeof(DebugSplatAuditBuffer);
    if ((size_t)p_data.size() < audit_bytes) {
        return;
    }

    const DebugSplatAuditBuffer *buffer = reinterpret_cast<const DebugSplatAuditBuffer *>(p_data.ptr());
    SplatAuditSnapshot snapshot;
    snapshot.frame_serial = splat_audit_readback.requested_frame_serial;
    snapshot.sample_count = buffer->header.sample_count;
    snapshot.valid = (buffer->header.enabled != 0u && buffer->header.sample_count > 0u);

    const uint32_t sample_count = MIN<uint32_t>(buffer->header.sample_count,
            TileRenderer::DEBUG_SPLAT_AUDIT_MAX_SAMPLES);
    for (uint32_t i = 0; i < sample_count; ++i) {
        const DebugSplatAuditEntry &entry = buffer->entries[i];
        const uint32_t flags = entry.flags;
        if ((flags & kSplatAuditFlagProjected) != 0u) {
            snapshot.projected_count++;
        }
        if ((flags & kSplatAuditFlagInViewport) != 0u) {
            snapshot.in_viewport_count++;
        }
        if ((flags & kSplatAuditFlagIterated) != 0u) {
            snapshot.iterated_count++;
        }
        if ((flags & kSplatAuditFlagContributed) != 0u) {
            snapshot.contributed_count++;
        }
        if ((flags & kSplatAuditFlagAlphaSkipped) != 0u) {
            snapshot.alpha_skipped_count++;
        }

        const bool missing_iterated = (flags & kSplatAuditFlagProjected) != 0u &&
                (flags & kSplatAuditFlagInViewport) != 0u &&
                (flags & kSplatAuditFlagIterated) == 0u;
        if (missing_iterated) {
            snapshot.missing_iterated_count++;
        }

        const bool missing_contrib = (flags & kSplatAuditFlagIterated) != 0u &&
                (flags & kSplatAuditFlagContributed) == 0u &&
                (flags & kSplatAuditFlagAlphaSkipped) == 0u;
        if (missing_contrib) {
            snapshot.missing_contrib_count++;
        }

        if ((missing_iterated || missing_contrib) && snapshot.first_mismatch_flags == 0u) {
            snapshot.first_mismatch_global_idx = entry.global_idx;
            snapshot.first_mismatch_expected_x = entry.expected_x;
            snapshot.first_mismatch_expected_y = entry.expected_y;
            snapshot.first_mismatch_flags = flags;
        }
    }

    cached_splat_audit_snapshot = snapshot;
    cached_splat_audit_frame_serial = splat_audit_readback.requested_frame_serial;
}

void TileRenderer::TileRendererDebugStats::create_buffers(RenderingDevice *p_device) {
	ERR_FAIL_NULL(p_device);

	// Create debug counters buffer (TileDebugCounterSnapshot layout)
	Vector<uint8_t> debug_counter_data;
	debug_counter_data.resize(sizeof(DebugCounterSnapshot));
	debug_counter_data.fill(0);
    debug_counter_buffer = p_device->storage_buffer_create(debug_counter_data.size(), debug_counter_data);

    if (!debug_counter_buffer.is_valid()) {
        GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to allocate debug counter buffer");
        debug_counter_owner.clear();
    } else {
        p_device->set_resource_name(debug_counter_buffer, "GS_TileRenderer_DebugCounterBuffer");
        debug_counter_owner.set(p_device);
    }

    // Create overflow statistics buffer (OverflowStatsSnapshot).
    // Includes overflow counters + sampled rasterizer diagnostics (enabled via debug_show_splat_coverage).
    Vector<uint8_t> stats_data;
    stats_data.resize(sizeof(OverflowStatsSnapshot));
    stats_data.fill(0);
    overflow_statistics_buffer = p_device->storage_buffer_create(stats_data.size(), stats_data);

    if (!overflow_statistics_buffer.is_valid()) {
        GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to allocate overflow statistics buffer");
        overflow_stats_owner.clear();
    } else {
        p_device->set_resource_name(overflow_statistics_buffer, "GS_TileRenderer_OverflowStatsBuffer");
        overflow_stats_owner.set(p_device);
    }

    Vector<uint8_t> audit_data;
    audit_data.resize(sizeof(DebugSplatAuditBuffer));
    audit_data.fill(0);
    debug_splat_audit_buffer = p_device->storage_buffer_create(audit_data.size(), audit_data);
    if (!debug_splat_audit_buffer.is_valid()) {
        GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to allocate splat audit buffer");
        debug_splat_audit_owner.clear();
    } else {
        p_device->set_resource_name(debug_splat_audit_buffer, "GS_TileRenderer_SplatAuditBuffer");
        debug_splat_audit_owner.set(p_device);
    }

    owner._invalidate_descriptor_cache();
}

void TileRenderer::TileRendererDebugStats::free_buffers(RenderingDevice *p_device) {
	if (!p_device) {
		debug_counter_buffer = RID();
		overflow_statistics_buffer = RID();
		debug_splat_audit_buffer = RID();
		debug_counter_owner.clear();
		overflow_stats_owner.clear();
		debug_splat_audit_owner.clear();
		return;
	}

	auto safe_free_debug_buffer = [&](RID &p_rid, BufferOwnership &p_owner) {
		RenderingDevice *owner_device = p_owner.device ? p_owner.device : p_device;
		if (p_rid.is_valid() && owner_device && owner_device->buffer_is_valid(p_rid)) {
			owner_device->free(p_rid);
		}
		p_rid = RID();
		p_owner.clear();
	};

	if (debug_counter_buffer.is_valid()) {
		safe_free_debug_buffer(debug_counter_buffer, debug_counter_owner);
	}
	if (debug_splat_audit_buffer.is_valid()) {
		safe_free_debug_buffer(debug_splat_audit_buffer, debug_splat_audit_owner);
	}
	if (overflow_statistics_buffer.is_valid()) {
		safe_free_debug_buffer(overflow_statistics_buffer, overflow_stats_owner);
	}
}

void TileRenderer::TileRendererDebugStats::clear_counters(RenderingDevice *p_device) {
	if (!p_device) {
		return;
	}
	// Clear debug counters at frame start for per-frame tracking
	if (debug_counter_buffer.is_valid()) {
		p_device->buffer_clear(debug_counter_buffer, 0, sizeof(DebugCounterSnapshot));
	}
	// Clear overflow stats as well
	if (overflow_statistics_buffer.is_valid()) {
		p_device->buffer_clear(overflow_statistics_buffer, 0, sizeof(OverflowStatsSnapshot));
	}
}

void TileRenderer::TileRendererDebugStats::update_splat_audit_buffer(RenderingDevice *p_device, const RenderParams &p_params,
        uint64_t p_frame_serial) {
	if (!p_device || !debug_splat_audit_buffer.is_valid()) {
		return;
	}

	DebugSplatAuditBuffer buffer;
	buffer.header.enabled = p_params.debug_enable_splat_audit ? 1u : 0u;
	buffer.header.sample_count = 0u;
	buffer.header.frame_id = static_cast<uint32_t>(p_frame_serial & 0xFFFFFFFFu);
	buffer.header.reserved = 0u;

	for (uint32_t i = 0; i < TileRenderer::DEBUG_SPLAT_AUDIT_MAX_SAMPLES; ++i) {
		buffer.entries[i].global_idx = kDebugSplatAuditInvalidIndex;
		buffer.entries[i].expected_x = 0;
		buffer.entries[i].expected_y = 0;
		buffer.entries[i].flags = 0;
	}

	if (p_params.debug_enable_splat_audit && p_params.splat_count > 0) {
		uint32_t requested_samples = p_params.debug_splat_audit_sample_count;
		if (requested_samples == 0) {
			requested_samples = TileRenderer::DEBUG_SPLAT_AUDIT_MAX_SAMPLES;
		}
		uint32_t available = MIN<uint32_t>(p_params.splat_count, TileRenderer::DEBUG_SPLAT_AUDIT_MAX_SAMPLES);
		uint32_t sample_count = MIN<uint32_t>(requested_samples, available);
		buffer.header.sample_count = sample_count;

		uint32_t step = MAX<uint32_t>(1u, p_params.splat_count / MAX<uint32_t>(sample_count, 1u));
		uint32_t start = static_cast<uint32_t>((p_frame_serial * 2654435761u) % p_params.splat_count);
		for (uint32_t i = 0; i < sample_count; ++i) {
			uint32_t idx = (start + i * step) % p_params.splat_count;
			buffer.entries[i].global_idx = idx;
		}
	}

	Vector<uint8_t> data;
	data.resize(sizeof(DebugSplatAuditBuffer));
	std::memcpy(data.ptrw(), &buffer, sizeof(DebugSplatAuditBuffer));
	p_device->buffer_update(debug_splat_audit_buffer, 0, data.size(), data.ptr());
}

void TileRenderer::TileRendererDebugStats::_log_readback_state_if_needed(uint64_t p_frame_serial, bool p_log_enabled) const {
	// DEBUG: Log readback state every 30 frames
	static int dump_call_counter = 0;
	if (p_log_enabled && (++dump_call_counter % 30 == 1)) {
		GS_LOG_RENDERER_DEBUG(vformat("[DEBUG-READBACK] frame=%d cached_debug=%d cached_overflow=%d pending_debug=%d pending_overflow=%d",
				int(p_frame_serial), int(cached_debug_frame_serial), int(cached_overflow_frame_serial),
				debug_counter_readback.pending ? 1 : 0, overflow_stats_readback.pending ? 1 : 0));
	}
}

void TileRenderer::TileRendererDebugStats::_schedule_debug_readbacks(RenderingDevice *p_device, uint64_t p_frame_serial,
		bool p_has_debug_buffer, bool p_has_overflow_buffer) {
	static constexpr uint64_t kDebugReadbackDelayFrames = 2;
	if (p_has_debug_buffer && p_frame_serial > 0 && (p_frame_serial > cached_debug_frame_serial + kDebugReadbackDelayFrames)) {
		const size_t debug_bytes = sizeof(DebugCounterSnapshot);
		if (!debug_counter_readback.pending) {
			debug_counter_readback.requested_frame_serial = p_frame_serial;
			Callable callback = callable_mp(&owner, &TileRenderer::_on_debug_counters_readback);
			Error err = p_device->buffer_get_data_async(debug_counter_buffer, callback, 0, debug_bytes);
			if (err == OK) {
				debug_counter_readback.pending = true;
			}
		}
	}

	if (p_has_overflow_buffer && p_frame_serial > 0 && (p_frame_serial > cached_overflow_frame_serial + kDebugReadbackDelayFrames)) {
		const size_t overflow_bytes = sizeof(OverflowStatsSnapshot);
		if (!overflow_stats_readback.pending) {
			overflow_stats_readback.requested_frame_serial = p_frame_serial;
			Callable callback = callable_mp(&owner, &TileRenderer::_on_overflow_stats_readback);
			Error err = p_device->buffer_get_data_async(overflow_statistics_buffer, callback, 0, overflow_bytes);
			if (err == OK) {
				overflow_stats_readback.pending = true;
			}
		}
	}
}

void TileRenderer::TileRendererDebugStats::_log_debug_counters(uint64_t p_frame_serial, bool p_log_enabled,
		bool p_has_debug_data, bool p_has_overflow_data) {
	if (!p_log_enabled || !p_has_debug_data) {
		return;
	}
	(void)p_has_overflow_data;

	// GPU counters exposed via Custom Performance Monitors
	// See Editor -> Debugger -> Monitors for real-time statistics
	const DebugCounterSnapshot &snapshot = cached_debug_counters;
	if (owner.render_settings.enable_sh_amortization) {
		const uint32_t hits = snapshot.sh_cache_hits;
		const uint32_t updates = snapshot.sh_cache_updates;
		const uint32_t forced = snapshot.sh_cache_forced_updates;
		const uint32_t total = hits + updates;
		const float hit_rate = total > 0 ? (100.0f * float(hits) / float(total)) : 0.0f;
		GS_LOG_RENDERER_DEBUG(vformat("[SH-CACHE] frame=%d divisor=%u hits=%u updates=%u forced=%u hit_rate=%.1f%%",
				int(p_frame_serial),
				owner.render_settings.sh_amortization_divisor,
				hits,
				updates,
				forced,
				hit_rate));
	}
	// GPU timing exposed via Custom Performance Monitors
	// See Editor -> Debugger -> Monitors for real-time GPU timing graphs

	const uint32_t overlap_records = owner.diagnostics.last_overlap_record_count;
	const bool tighter_bounds = owner.render_settings.enable_tighter_bounds;
	const bool toggled = last_logged_valid && (tighter_bounds != last_logged_tighter_bounds);
	if (toggled) {
		GS_LOG_RENDERER_DEBUG(vformat("[GPU-COUNTERS-BASELINE] frame=%d tighter_bounds=%s tile_extent_reject=%d overlap_records=%u",
				int(last_logged_frame_serial),
				last_logged_tighter_bounds ? "YES" : "no",
				int(last_logged_debug_counters.tile_extent_reject),
				last_logged_overlap_records));
	}
	if (last_logged_valid) {
		const int32_t delta_tile_extent = int32_t(snapshot.tile_extent_reject) - int32_t(last_logged_debug_counters.tile_extent_reject);
		const int32_t delta_overlap = int32_t(overlap_records) - int32_t(last_logged_overlap_records);
		const String toggle_note = toggled ? " (toggled)" : "";
		const float overlap_pct = (last_logged_overlap_records > 0)
				? (100.0f * float(delta_overlap) / float(last_logged_overlap_records))
				: 0.0f;
		const String delta_te_str = (delta_tile_extent >= 0 ? "+" : "") + String::num_int64(delta_tile_extent);
		const String delta_ov_str = (delta_overlap >= 0 ? "+" : "") + String::num_int64(delta_overlap);
		GS_LOG_RENDERER_DEBUG("[GPU-COUNTERS-DELTA] frame=" + itos(p_frame_serial) +
				" tighter_bounds=" + (tighter_bounds ? "YES" : "no") + toggle_note +
				" tile_extent_reject=" + itos(snapshot.tile_extent_reject) + " (delta " + delta_te_str + ")" +
				" overlap_records=" + itos(overlap_records) + " (delta " + delta_ov_str + ", " + String::num(overlap_pct, 1) + "%)");
	} else {
		GS_LOG_RENDERER_DEBUG("[GPU-COUNTERS-DELTA] frame=" + itos(p_frame_serial) +
				" tighter_bounds=" + (tighter_bounds ? "YES" : "no") +
				" tile_extent_reject=" + itos(snapshot.tile_extent_reject) +
				" overlap_records=" + itos(overlap_records));
	}
	last_logged_debug_counters = snapshot;
	last_logged_overlap_records = overlap_records;
	last_logged_tighter_bounds = tighter_bounds;
	last_logged_frame_serial = p_frame_serial;
	last_logged_valid = true;
}

void TileRenderer::TileRendererDebugStats::_log_overflow_stats(uint64_t p_frame_serial, bool p_log_enabled, bool p_has_overflow_data) {
	if (!p_log_enabled || !p_has_overflow_data) {
		return;
	}

	const OverflowStatsSnapshot &stats = cached_overflow_stats;

	GS_LOG_RENDERER_DEBUG(vformat("[GPU-OVERFLOW] frame=%d tiles=%d clamped=%d aggregated=%d",
			int(p_frame_serial),
			stats.overflow_tile_count,
			stats.overflow_splats_clamped,
			stats.overflow_splats_aggregated));

	if (stats.raster_sample_count > 0) {
		float avg_alpha = float(stats.raster_alpha_sum_q10) / float(stats.raster_sample_count) / 1024.0f;
		uint32_t no_contrib_tiles = stats.raster_sample_count > stats.raster_has_depth
				? (stats.raster_sample_count - stats.raster_has_depth)
				: 0;
		GS_LOG_RENDERER_DEBUG(vformat("[GPU-RASTER] samples=%d iter=%d contrib=%d has_depth=%d no_contrib=%d avg_alpha=%.4f rejects:{sorted=%d gaussian=%d idx_mismatch=%d base0=%d nan=%d weight=%d alpha=%d} breaks:{remain=%d final=%d subgroup=%d}",
				stats.raster_sample_count,
				stats.raster_splats_iterated,
				stats.raster_splats_contributed,
				stats.raster_has_depth,
				no_contrib_tiles,
				avg_alpha,
				stats.raster_reject_sorted_idx_oob,
				stats.raster_reject_gaussian_idx_oob,
				stats.raster_reject_index_mismatch,
				stats.raster_reject_base_opacity,
				stats.raster_reject_nan_inf,
				stats.raster_reject_weight,
				stats.raster_reject_alpha,
				stats.raster_break_remaining_alpha,
				stats.raster_break_final_alpha,
				stats.raster_break_subgroup_early_exit));
	}

	// Note: overflow buffer is cleared at frame start via _clear_debug_counters(),
	// not here, so that get_overflow_stats() can read it for auto-tune feedback.
}

void TileRenderer::TileRendererDebugStats::dump_gpu_debug_counters(RenderingDevice *p_device, const RenderParams &p_params,
        uint64_t p_frame_serial) {
	if (!p_params.debug_dump_gpu_counters) {
		return;
	}

	RenderingDevice *device = p_device;
	const bool log_enabled = p_params.debug_enable_gpu_counter_logs;
	_log_readback_state_if_needed(p_frame_serial, log_enabled);
	const bool has_debug_buffer = device && debug_counter_buffer.is_valid();
	const bool has_overflow_buffer = device && overflow_statistics_buffer.is_valid();
	if (!has_debug_buffer && !has_overflow_buffer) {
		return;
	}
	_schedule_debug_readbacks(device, p_frame_serial, has_debug_buffer, has_overflow_buffer);

	const bool has_debug_data = has_debug_buffer && cached_debug_frame_serial > 0;
	const bool has_overflow_data = has_overflow_buffer && cached_overflow_frame_serial > 0;
	if (!has_debug_data && !has_overflow_data) {
		return;
	}
	_log_debug_counters(p_frame_serial, log_enabled, has_debug_data, has_overflow_data);
	_log_overflow_stats(p_frame_serial, log_enabled, has_overflow_data);
}

TileRenderer::DebugCounterSnapshot TileRenderer::TileRendererDebugStats::get_debug_counters(RenderingDevice *p_device,
        uint64_t p_frame_serial) const {
    DebugCounterSnapshot snapshot = cached_debug_counters;
    if (!owner.diagnostics.debug_binning_counters_enabled) {
        return snapshot;
    }
    if (!p_device || !debug_counter_buffer.is_valid()) {
        return snapshot;
    }
    static constexpr uint64_t kDebugReadbackDelayFrames = 2;
    if (p_frame_serial > 0 && (p_frame_serial > cached_debug_frame_serial + kDebugReadbackDelayFrames)) {
        const size_t debug_bytes = sizeof(DebugCounterSnapshot);
        if (!debug_counter_readback.pending) {
            debug_counter_readback.requested_frame_serial = p_frame_serial;
            Callable callback = callable_mp(&owner, &TileRenderer::_on_debug_counters_readback);
            Error err = p_device->buffer_get_data_async(debug_counter_buffer, callback, 0, debug_bytes);
            if (err == OK) {
                debug_counter_readback.pending = true;
            }
        }
    }
    return snapshot;
}

TileRenderer::OverflowStatsSnapshot TileRenderer::TileRendererDebugStats::get_overflow_stats(RenderingDevice *p_device,
        uint64_t p_frame_serial) const {
    OverflowStatsSnapshot stats = cached_overflow_stats;
    if (!p_device || !overflow_statistics_buffer.is_valid()) {
        return stats;
    }
    static constexpr uint64_t kDebugReadbackDelayFrames = 2;
    if (p_frame_serial > 0 && (p_frame_serial > cached_overflow_frame_serial + kDebugReadbackDelayFrames)) {
        const size_t overflow_bytes = sizeof(OverflowStatsSnapshot);
        if (!overflow_stats_readback.pending) {
            overflow_stats_readback.requested_frame_serial = p_frame_serial;
            Callable callback = callable_mp(&owner, &TileRenderer::_on_overflow_stats_readback);
            Error err = p_device->buffer_get_data_async(overflow_statistics_buffer, callback, 0, overflow_bytes);
            if (err == OK) {
                overflow_stats_readback.pending = true;
            }
        }
    }
    return stats;
}

TileRenderer::SplatAuditSnapshot TileRenderer::TileRendererDebugStats::get_splat_audit_snapshot(RenderingDevice *p_device,
        uint64_t p_frame_serial) const {
    SplatAuditSnapshot snapshot = cached_splat_audit_snapshot;
    if (!p_device || !debug_splat_audit_buffer.is_valid()) {
        return snapshot;
    }
    static constexpr uint64_t kDebugReadbackDelayFrames = 2;
    if (p_frame_serial > 0 && (p_frame_serial > cached_splat_audit_frame_serial + kDebugReadbackDelayFrames)) {
        const size_t audit_bytes = sizeof(DebugSplatAuditBuffer);
        if (!splat_audit_readback.pending) {
            splat_audit_readback.requested_frame_serial = p_frame_serial;
            Callable callback = callable_mp(&owner, &TileRenderer::_on_splat_audit_readback);
            Error err = p_device->buffer_get_data_async(debug_splat_audit_buffer, callback, 0, audit_bytes);
            if (err == OK) {
                splat_audit_readback.pending = true;
            }
        }
    }
    return snapshot;
}
