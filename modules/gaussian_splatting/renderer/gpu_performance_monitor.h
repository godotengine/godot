#ifndef GPU_PERFORMANCE_MONITOR_H
#define GPU_PERFORMANCE_MONITOR_H

#include "core/os/mutex.h"
#include "core/templates/hash_map.h"

class RenderingDevice;

// Tracks frame completion using monotonic frame IDs.
// Note: Godot lacks true timeline semaphore support, so this tracks frame
// completion based on internal frame counting rather than GPU timeline values.
class FrameCompletionTracker {
public:
    void set_rendering_device(RenderingDevice *p_device);
    RenderingDevice *get_rendering_device() const { return rendering_device; }

    // Polls the current completed frame index and timestamp.
    // Returns false if rendering device is unavailable.
    // r_frame_index: monotonic frame counter (not GPU timeline value)
    // r_time_usec: current CPU timestamp in microseconds
    bool poll_frame_completion(uint64_t &r_frame_index, uint64_t &r_time_usec) const;

    // Deprecated: Use poll_frame_completion instead. The RID parameter is ignored.
    bool poll_timeline_value(RID p_timeline_semaphore, uint64_t &r_value, uint64_t &r_time_usec) const {
        (void)p_timeline_semaphore;
        return poll_frame_completion(r_value, r_time_usec);
    }

private:
    RenderingDevice *rendering_device = nullptr;
    mutable uint64_t monotonic_frame_counter = 0;
    mutable uint64_t last_captured_frame = UINT64_MAX;
};

// Deprecated alias for backwards compatibility
using TimelineSemaphoreManager = FrameCompletionTracker;

class GPUPerformanceMonitor {
public:
    struct FrameMetrics {
        uint64_t frame_id = 0;
        uint64_t submit_frame_index = 0;  // Renamed from submit_timeline_value for clarity
        uint64_t complete_frame_index = 0; // Renamed from complete_timeline_value for clarity
        float completion_rate = 0.0f;     // Renamed from gpu_utilization for accuracy
        uint32_t stall_count = 0;
        uint64_t total_stall_ns = 0;
        uint64_t last_observed_frame = 0; // Renamed from last_observed_timeline
        uint64_t submit_time_usec = 0;    // Added: timestamp when frame was submitted

        // Deprecated accessors for backwards compatibility
        uint64_t get_submit_timeline_value() const { return submit_frame_index; }
        uint64_t get_complete_timeline_value() const { return complete_frame_index; }
        float get_gpu_utilization() const { return completion_rate; }
    };

    struct SummaryMetrics {
        uint32_t inflight_frames = 0;
        uint32_t completed_frames = 0;
        uint32_t stalled_frames = 0;
        uint32_t stall_count = 0;
        uint64_t total_stall_ns = 0;
        uint64_t last_frame_index = 0;    // Renamed from last_timeline_value

        // Deprecated accessor for backwards compatibility
        uint64_t get_last_timeline_value() const { return last_frame_index; }
    };

    // Stall detection threshold: 16ms (60fps frame budget)
    static constexpr uint64_t STALL_THRESHOLD_USEC = 16000;

    GPUPerformanceMonitor();

    void set_frame_tracker(FrameCompletionTracker *p_tracker);
    // Deprecated: Use set_frame_tracker instead
    void set_timeline_manager(TimelineSemaphoreManager *p_manager) { set_frame_tracker(p_manager); }
    void set_rendering_device(RenderingDevice *p_device);

    void record_submission(uint64_t frame_id, uint64_t frame_index);
    void record_completion(uint64_t frame_id, uint64_t frame_index);

    // Returns cached completion rate (ratio of completed to submitted frames).
    // Thread-safe: protected by mutex.
    float get_cached_completion_rate();
    // Deprecated: Use get_cached_completion_rate instead. Name was misleading.
    float get_gpu_utilization_async() { return get_cached_completion_rate(); }

    // Detects frames that have exceeded the stall threshold without completion.
    // The RID parameter is ignored (retained for API compatibility).
    void detect_pipeline_stalls(RID timeline_semaphore = RID());

    FrameMetrics get_frame_metrics_nonblocking(uint64_t frame_id) const;
    SummaryMetrics get_summary_metrics() const;

private:
    mutable Mutex metrics_mutex;
    FrameCompletionTracker internal_frame_tracker;
    FrameCompletionTracker *frame_tracker = nullptr;
    HashMap<uint64_t, FrameMetrics> metrics;
    float cached_completion_rate = 0.0f;
    uint64_t last_poll_usec = 0;
    uint64_t last_frame_index = 0;

    void _prune_old_frames(uint64_t p_current_frame_id);
};

#endif // GPU_PERFORMANCE_MONITOR_H
