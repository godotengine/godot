#include "streaming_queue_pressure_controller.h"

#include "core/math/math_funcs.h"
#include "core/string/ustring.h"

namespace {

static void _set_error(String *r_error, const String &p_message) {
    if (r_error) {
        *r_error = p_message;
    }
}

} // namespace

StreamingQueuePressureController::ScanBudgetResult StreamingQueuePressureController::compute_candidate_scan_budget(
        const ScanBudgetInput &p_input) {
    ScanBudgetResult result;
    result.scan_budget = p_input.base_scan_budget;
    result.throttle_active = false;
    result.effective_queue_depth = p_input.observed_queue_depth;

    if (p_input.base_scan_budget == 0 || !p_input.throttle_enabled) {
        return result;
    }

    if (p_input.enqueue_headroom != UINT32_MAX) {
        if (p_input.enqueue_headroom == 0) {
            result.effective_queue_depth = MAX(result.effective_queue_depth,
                    p_input.throttle_min_queue_depth + ENQUEUE_HEADROOM_TARGET);
        } else if (p_input.enqueue_headroom < ENQUEUE_HEADROOM_TARGET) {
            const uint32_t synthetic_depth = p_input.throttle_min_queue_depth +
                    (ENQUEUE_HEADROOM_TARGET - p_input.enqueue_headroom);
            result.effective_queue_depth = MAX(result.effective_queue_depth, synthetic_depth);
        }
    }

    result.throttle_active = result.effective_queue_depth >= p_input.throttle_min_queue_depth;
    if (!result.throttle_active) {
        return result;
    }

    const uint32_t throttle_cap = MAX<uint32_t>(1u, p_input.throttle_scan_cap);
    const uint32_t throttle_budget_remaining = p_input.scanned_this_frame >= throttle_cap
            ? 0
            : (throttle_cap - p_input.scanned_this_frame);
    uint32_t scan_budget = MIN(p_input.base_scan_budget, throttle_budget_remaining);
    if (scan_budget == 0) {
        result.scan_budget = 0;
        return result;
    }

    if (result.effective_queue_depth > p_input.throttle_min_queue_depth) {
        const uint32_t depth_excess = result.effective_queue_depth - p_input.throttle_min_queue_depth;
        scan_budget = MAX<uint32_t>(1u, scan_budget / (depth_excess + 1u));
    }

    if (p_input.enqueue_headroom != UINT32_MAX) {
        if (p_input.enqueue_headroom == 0) {
            scan_budget = 1;
        } else if (p_input.enqueue_headroom < ENQUEUE_HEADROOM_TARGET) {
            const uint32_t enqueue_limited_budget = MAX<uint32_t>(1u,
                    (throttle_cap * p_input.enqueue_headroom + ENQUEUE_HEADROOM_TARGET - 1u) / ENQUEUE_HEADROOM_TARGET);
            scan_budget = MIN(scan_budget, enqueue_limited_budget);
        }
    }

    result.scan_budget = scan_budget;
    return result;
}

StreamingQueuePressureController::PressureSummary StreamingQueuePressureController::summarize(
        const PressureSample &p_sample) {
    PressureSummary summary;
    const bool pack_queue_backlog = p_sample.pack_queue_depth > 0;
    const bool upload_queue_backlog = p_sample.upload_queue_depth > 0;
    const bool sync_queue_backlog = p_sample.sync_fallback_queue_depth > 0;

    summary.pack_source_active = pack_queue_backlog || p_sample.pack_inflight_saturated;
    summary.upload_source_active = upload_queue_backlog || p_sample.upload_frame_cap_hit ||
            p_sample.upload_bandwidth_cap_hit || p_sample.chunk_load_cap_hit;
    summary.sync_source_active = sync_queue_backlog || p_sample.sync_backpressure;
    summary.backlog_depth = MAX(p_sample.sync_fallback_queue_depth,
            MAX(p_sample.pack_queue_depth, p_sample.upload_queue_depth));

    const bool queue_backlog_active = pack_queue_backlog || upload_queue_backlog || sync_queue_backlog;
    summary.cap_active = p_sample.pack_inflight_saturated ||
            p_sample.upload_frame_cap_hit ||
            p_sample.upload_bandwidth_cap_hit ||
            p_sample.chunk_load_cap_hit ||
            p_sample.vram_chunk_cap_hit ||
            p_sample.sync_backpressure;
    summary.active = queue_backlog_active || summary.cap_active;

    const uint32_t active_sources = uint32_t(summary.pack_source_active) +
            uint32_t(summary.upload_source_active) +
            uint32_t(summary.sync_source_active);
    if (active_sources > 1) {
        summary.source = SOURCE_COMBINED;
    } else if (summary.pack_source_active) {
        summary.source = SOURCE_PACK;
    } else if (summary.upload_source_active) {
        summary.source = SOURCE_UPLOAD;
    } else if (summary.sync_source_active) {
        summary.source = SOURCE_SYNC;
    } else if (summary.active) {
        summary.source = SOURCE_CAP;
    } else {
        summary.source = SOURCE_NONE;
    }

    if (!summary.active) {
        summary.reason = REASON_NONE;
    } else if (queue_backlog_active && summary.cap_active) {
        summary.reason = REASON_QUEUE_AND_CAPS;
    } else if (queue_backlog_active) {
        if (pack_queue_backlog && !upload_queue_backlog && !sync_queue_backlog) {
            summary.reason = REASON_PACK_QUEUE_BACKLOG;
        } else if (upload_queue_backlog && !pack_queue_backlog && !sync_queue_backlog) {
            summary.reason = REASON_UPLOAD_QUEUE_BACKLOG;
        } else if (sync_queue_backlog && !pack_queue_backlog && !upload_queue_backlog) {
            summary.reason = REASON_SYNC_QUEUE_BACKLOG;
        } else {
            summary.reason = REASON_QUEUE_BACKLOG;
        }
    } else if (p_sample.upload_frame_cap_hit &&
            !p_sample.upload_bandwidth_cap_hit &&
            !p_sample.chunk_load_cap_hit &&
            !p_sample.vram_chunk_cap_hit) {
        summary.reason = REASON_UPLOAD_FRAME_CAP;
    } else if (p_sample.upload_bandwidth_cap_hit &&
            !p_sample.upload_frame_cap_hit &&
            !p_sample.chunk_load_cap_hit &&
            !p_sample.vram_chunk_cap_hit) {
        summary.reason = REASON_UPLOAD_BANDWIDTH_CAP;
    } else if (p_sample.chunk_load_cap_hit &&
            !p_sample.upload_frame_cap_hit &&
            !p_sample.upload_bandwidth_cap_hit &&
            !p_sample.vram_chunk_cap_hit) {
        summary.reason = REASON_CHUNK_LOAD_CAP;
    } else if (p_sample.vram_chunk_cap_hit &&
            !p_sample.upload_frame_cap_hit &&
            !p_sample.upload_bandwidth_cap_hit &&
            !p_sample.chunk_load_cap_hit) {
        summary.reason = REASON_VRAM_CHUNK_CAP;
    } else if (p_sample.pack_inflight_saturated &&
            !p_sample.upload_frame_cap_hit &&
            !p_sample.upload_bandwidth_cap_hit &&
            !p_sample.chunk_load_cap_hit &&
            !p_sample.vram_chunk_cap_hit) {
        summary.reason = REASON_PACK_INFLIGHT_CAP;
    } else if (p_sample.sync_backpressure &&
            !p_sample.upload_frame_cap_hit &&
            !p_sample.upload_bandwidth_cap_hit &&
            !p_sample.chunk_load_cap_hit &&
            !p_sample.vram_chunk_cap_hit) {
        summary.reason = REASON_SYNC_FALLBACK_PRESSURE;
    } else {
        summary.reason = REASON_CAP_COMBINED;
    }

    return summary;
}

void StreamingQueuePressureController::reset_latched_state(bool &r_active, String &r_source, String &r_reason) {
    r_active = false;
    r_source = SOURCE_NONE;
    r_reason = REASON_NONE;
}

void StreamingQueuePressureController::mark_latched_state(bool &r_active, String &r_source, String &r_reason,
        const char *p_source, const char *p_reason) {
    r_active = true;
    const String source_token = p_source ? String(p_source) : String(SOURCE_NONE);
    const String reason_token = p_reason ? String(p_reason) : String(REASON_NONE);

    r_source = is_known_source(source_token) ? source_token : String(SOURCE_CAP);
    r_reason = is_known_reason(reason_token) ? reason_token : String(REASON_CAP_COMBINED);
    if (r_source == SOURCE_NONE) {
        r_source = SOURCE_CAP;
    }
    if (r_reason == REASON_NONE) {
        r_reason = REASON_CAP_COMBINED;
    }
}

void StreamingQueuePressureController::latch_summary(const PressureSummary &p_summary,
        bool &r_active, String &r_source, String &r_reason) {
    if (p_summary.active) {
        r_active = true;
        if (p_summary.source != SOURCE_NONE) {
            r_source = p_summary.source;
        }
        if (p_summary.reason != REASON_NONE) {
            r_reason = p_summary.reason;
        }
    }

    if (!r_active) {
        r_source = SOURCE_NONE;
        r_reason = REASON_NONE;
    }
}

bool StreamingQueuePressureController::is_known_source(const String &p_source) {
    return p_source == SOURCE_NONE ||
            p_source == SOURCE_PACK ||
            p_source == SOURCE_UPLOAD ||
            p_source == SOURCE_SYNC ||
            p_source == SOURCE_CAP ||
            p_source == SOURCE_COMBINED;
}

bool StreamingQueuePressureController::is_known_reason(const String &p_reason) {
    return p_reason == REASON_NONE ||
            p_reason == REASON_QUEUE_BACKLOG ||
            p_reason == REASON_PACK_QUEUE_BACKLOG ||
            p_reason == REASON_UPLOAD_QUEUE_BACKLOG ||
            p_reason == REASON_SYNC_QUEUE_BACKLOG ||
            p_reason == REASON_QUEUE_AND_CAPS ||
            p_reason == REASON_UPLOAD_FRAME_CAP ||
            p_reason == REASON_UPLOAD_BANDWIDTH_CAP ||
            p_reason == REASON_UPLOAD_CAP_COMBINED ||
            p_reason == REASON_CHUNK_LOAD_CAP ||
            p_reason == REASON_VRAM_CHUNK_CAP ||
            p_reason == REASON_PACK_INFLIGHT_CAP ||
            p_reason == REASON_SYNC_FALLBACK_PRESSURE ||
            p_reason == REASON_SYNC_QUEUE_CAP ||
            p_reason == REASON_CAP_COMBINED;
}

bool StreamingQueuePressureController::validate_summary_invariants(const PressureSummary &p_summary,
        const PressureSample &p_sample, String *r_error) {
    const bool pack_queue_backlog = p_sample.pack_queue_depth > 0;
    const bool upload_queue_backlog = p_sample.upload_queue_depth > 0;
    const bool sync_queue_backlog = p_sample.sync_fallback_queue_depth > 0;

    const bool expected_pack_source = pack_queue_backlog || p_sample.pack_inflight_saturated;
    const bool expected_upload_source = upload_queue_backlog || p_sample.upload_frame_cap_hit ||
            p_sample.upload_bandwidth_cap_hit || p_sample.chunk_load_cap_hit;
    const bool expected_sync_source = sync_queue_backlog || p_sample.sync_backpressure;
    const bool expected_cap_active = p_sample.pack_inflight_saturated ||
            p_sample.upload_frame_cap_hit ||
            p_sample.upload_bandwidth_cap_hit ||
            p_sample.chunk_load_cap_hit ||
            p_sample.vram_chunk_cap_hit ||
            p_sample.sync_backpressure;
    const bool expected_active = pack_queue_backlog || upload_queue_backlog || sync_queue_backlog || expected_cap_active;
    const uint32_t expected_backlog_depth = MAX(p_sample.sync_fallback_queue_depth,
            MAX(p_sample.pack_queue_depth, p_sample.upload_queue_depth));

    if (p_summary.pack_source_active != expected_pack_source) {
        _set_error(r_error, "pack source activity must match queue/cap inputs");
        return false;
    }
    if (p_summary.upload_source_active != expected_upload_source) {
        _set_error(r_error, "upload source activity must match queue/cap inputs");
        return false;
    }
    if (p_summary.sync_source_active != expected_sync_source) {
        _set_error(r_error, "sync source activity must match queue/cap inputs");
        return false;
    }
    if (p_summary.cap_active != expected_cap_active) {
        _set_error(r_error, "cap activity must match cap inputs");
        return false;
    }
    if (p_summary.active != expected_active) {
        _set_error(r_error, "active flag must match queue/cap aggregate");
        return false;
    }
    if (p_summary.backlog_depth != expected_backlog_depth) {
        _set_error(r_error, "backlog depth must equal max(pack, upload, sync)");
        return false;
    }
    if (!is_known_source(p_summary.source)) {
        _set_error(r_error, "summary source token is unknown");
        return false;
    }
    if (!is_known_reason(p_summary.reason)) {
        _set_error(r_error, "summary reason token is unknown");
        return false;
    }
    if (!p_summary.active) {
        if (p_summary.source != SOURCE_NONE || p_summary.reason != REASON_NONE) {
            _set_error(r_error, "inactive summary must use source=none and reason=none");
            return false;
        }
    } else {
        if (p_summary.source == SOURCE_NONE || p_summary.reason == REASON_NONE) {
            _set_error(r_error, "active summary must expose non-none source and reason");
            return false;
        }
    }

    return true;
}

bool StreamingQueuePressureController::validate_latched_state_invariants(bool p_active,
        const String &p_source, const String &p_reason, String *r_error) {
    if (!is_known_source(p_source)) {
        _set_error(r_error, "latched source token is unknown");
        return false;
    }
    if (!is_known_reason(p_reason)) {
        _set_error(r_error, "latched reason token is unknown");
        return false;
    }
    if (!p_active) {
        if (p_source != SOURCE_NONE || p_reason != REASON_NONE) {
            _set_error(r_error, "inactive latch must use source=none and reason=none");
            return false;
        }
        return true;
    }
    if (p_source == SOURCE_NONE || p_reason == REASON_NONE) {
        _set_error(r_error, "active latch must expose non-none source and reason");
        return false;
    }
    return true;
}
