#include "gpu_performance_monitor.h"

#include "core/math/math_defs.h"
#include "core/os/os.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/rendering_device.h"

// FrameCompletionTracker implementation

void FrameCompletionTracker::set_rendering_device(RenderingDevice *p_device) {
    rendering_device = p_device;
}

bool FrameCompletionTracker::poll_frame_completion(uint64_t &r_frame_index, uint64_t &r_time_usec) const {
    if (!rendering_device) {
        return false;
    }

    // Get current captured frame from RenderingDevice.
    // Note: get_captured_timestamps_frame() returns a buffer index (0-2) that wraps,
    // so we convert it to a monotonic counter to avoid false stall detection on wrap.
    uint64_t captured_frame = rendering_device->get_captured_timestamps_frame();
    r_time_usec = OS::get_singleton()->get_ticks_usec();

    // Convert wrapping buffer index to monotonic counter
    if (last_captured_frame == UINT64_MAX) {
        // First call - initialize
        last_captured_frame = captured_frame;
        monotonic_frame_counter = captured_frame;
    } else if (captured_frame != last_captured_frame) {
        // Frame changed - increment monotonic counter
        // Handle wrap-around: if captured_frame < last_captured_frame, we wrapped
        if (captured_frame < last_captured_frame) {
            // Wrapped from 2 to 0 (or similar) - advance by the wrap distance
            // Assuming 3-buffer system (0, 1, 2)
            monotonic_frame_counter += (3 - last_captured_frame) + captured_frame;
        } else {
            monotonic_frame_counter += (captured_frame - last_captured_frame);
        }
        last_captured_frame = captured_frame;
    }

    r_frame_index = monotonic_frame_counter;
    return true;
}

// GPUPerformanceMonitor implementation

GPUPerformanceMonitor::GPUPerformanceMonitor() {
    frame_tracker = &internal_frame_tracker;
}

void GPUPerformanceMonitor::set_frame_tracker(FrameCompletionTracker *p_tracker) {
    MutexLock lock(metrics_mutex);
    frame_tracker = p_tracker ? p_tracker : &internal_frame_tracker;
}

void GPUPerformanceMonitor::set_rendering_device(RenderingDevice *p_device) {
    MutexLock lock(metrics_mutex);
    internal_frame_tracker.set_rendering_device(p_device);
    if (!frame_tracker) {
        frame_tracker = &internal_frame_tracker;
    }
}

void GPUPerformanceMonitor::record_submission(uint64_t frame_id, uint64_t frame_index) {
    MutexLock lock(metrics_mutex);
    FrameMetrics &frame = metrics[frame_id];
    frame.frame_id = frame_id;
    if (frame_index > frame.submit_frame_index) {
        frame.submit_frame_index = frame_index;
    }
    frame.last_observed_frame = frame_index;
    frame.submit_time_usec = OS::get_singleton()->get_ticks_usec();
    _prune_old_frames(frame_id);
}

void GPUPerformanceMonitor::record_completion(uint64_t frame_id, uint64_t frame_index) {
    MutexLock lock(metrics_mutex);
    FrameMetrics &frame = metrics[frame_id];
    frame.frame_id = frame_id;
    if (frame_index > frame.complete_frame_index) {
        frame.complete_frame_index = frame_index;
    }
    frame.last_observed_frame = frame_index;
}

float GPUPerformanceMonitor::get_cached_completion_rate() {
    MutexLock lock(metrics_mutex);

    uint64_t active_frames = 0;
    uint64_t completed_frames = 0;
    for (const KeyValue<uint64_t, FrameMetrics> &E : metrics) {
        if (E.value.submit_frame_index != 0) {
            active_frames++;
            if (E.value.complete_frame_index >= E.value.submit_frame_index) {
                completed_frames++;
            }
        }
    }

    if (active_frames == 0) {
        cached_completion_rate = 0.0f;
        return cached_completion_rate;
    }

    cached_completion_rate = float(completed_frames) / float(active_frames);
    cached_completion_rate = CLAMP(cached_completion_rate, 0.0f, 1.0f);
    return cached_completion_rate;
}

void GPUPerformanceMonitor::detect_pipeline_stalls(RID timeline_semaphore) {
    (void)timeline_semaphore; // Parameter retained for API compatibility but ignored

    MutexLock lock(metrics_mutex);

    if (!frame_tracker) {
        return;
    }

    uint64_t frame_index = 0;
    uint64_t poll_usec = 0;
    if (!frame_tracker->poll_frame_completion(frame_index, poll_usec)) {
        return;
    }

    if (last_poll_usec == 0) {
        last_poll_usec = poll_usec;
        last_frame_index = frame_index;
        return;
    }

    const bool frame_advanced = frame_index > last_frame_index;

    for (KeyValue<uint64_t, FrameMetrics> &E : metrics) {
        FrameMetrics &frame = E.value;
        if (frame.submit_frame_index == 0) {
            continue;
        }
        // Skip already completed frames
        if (frame.complete_frame_index >= frame.submit_frame_index) {
            continue;
        }

        // Check if frame completed (frame tracker advanced past submission)
        if (frame_advanced && frame.submit_frame_index <= frame_index) {
            frame.complete_frame_index = frame_index;
        } else {
            // Frame not yet complete - check for stall using time-based threshold
            // Only count as stall if we've exceeded the threshold since submission
            if (frame.submit_time_usec > 0) {
                uint64_t elapsed_usec = poll_usec - frame.submit_time_usec;
                if (elapsed_usec > STALL_THRESHOLD_USEC) {
                    // Count one stall per detection cycle, not cumulative time
                    frame.stall_count += 1;
                    frame.total_stall_ns += (elapsed_usec - STALL_THRESHOLD_USEC) * 1000;
                }
            }
        }
        frame.last_observed_frame = frame_index;
    }

    last_frame_index = frame_index;
    last_poll_usec = poll_usec;
}

GPUPerformanceMonitor::FrameMetrics GPUPerformanceMonitor::get_frame_metrics_nonblocking(uint64_t frame_id) const {
    MutexLock lock(metrics_mutex);
    if (const FrameMetrics *frame = metrics.getptr(frame_id)) {
        return *frame;
    }
    return FrameMetrics();
}

GPUPerformanceMonitor::SummaryMetrics GPUPerformanceMonitor::get_summary_metrics() const {
    MutexLock lock(metrics_mutex);
    SummaryMetrics summary;
    summary.last_frame_index = last_frame_index;
    for (const KeyValue<uint64_t, FrameMetrics> &E : metrics) {
        const FrameMetrics &frame = E.value;
        if (frame.submit_frame_index == 0) {
            continue;
        }
        if (frame.complete_frame_index >= frame.submit_frame_index) {
            summary.completed_frames++;
        } else {
            summary.inflight_frames++;
        }
        if (frame.stall_count > 0) {
            summary.stalled_frames++;
            summary.stall_count += frame.stall_count;
            summary.total_stall_ns += frame.total_stall_ns;
        }
    }
    return summary;
}

void GPUPerformanceMonitor::_prune_old_frames(uint64_t p_current_frame_id) {
    // Note: Called with mutex already held by caller

    const uint64_t max_age = 240;
    const uint64_t soft_limit = 256;  // Start pruning earlier
    const uint64_t hard_limit = 512;

    // Always prune if we exceed soft limit, or prune by age regardless
    bool force_prune = metrics.size() > hard_limit;
    bool should_prune = metrics.size() > soft_limit;

    LocalVector<uint64_t> to_remove;
    to_remove.reserve(metrics.size() / 4);

    for (const KeyValue<uint64_t, FrameMetrics> &E : metrics) {
        const FrameMetrics &frame = E.value;
        bool is_old = E.key + max_age < p_current_frame_id;

        // Prune zero-value frames (never submitted) if old
        if (frame.submit_frame_index == 0 && is_old) {
            to_remove.push_back(E.key);
            continue;
        }

        // Prune completed frames if old, or force prune if at hard limit
        bool completed = frame.complete_frame_index >= frame.submit_frame_index;
        if (completed && (is_old || force_prune)) {
            to_remove.push_back(E.key);
            continue;
        }

        // At soft limit, also prune incomplete but very old frames
        if (should_prune && is_old && frame.submit_frame_index != 0) {
            to_remove.push_back(E.key);
        }
    }

    for (uint64_t frame_id : to_remove) {
        metrics.erase(frame_id);
    }
}
