#ifndef STREAMING_RUNTIME_STATE_H
#define STREAMING_RUNTIME_STATE_H

#include "core/string/ustring.h"
#include "core/templates/hash_set.h"
#include "core/variant/dictionary.h"
#include "streaming_asset_types.h"
#include "streaming_vram_regulator.h"
#include <cstdint>

namespace GaussianStreamingTypes {

struct FrameData {
    LocalVector<uint32_t> visible_chunks;
    uint64_t frame_number = 0;
};

struct BudgetState {
    Ref<VRAMBudgetRegulator> vram_regulator;
    uint32_t loaded_chunks_count = 0;
    uint64_t vram_usage = 0;
    uint64_t evicted_bytes_total = 0;
    uint32_t chunks_loaded_this_frame = 0;
    bool vram_chunk_cap_hit_this_frame = false;

    Dictionary get_vram_debug_stats() const;
    bool is_vram_budget_warning_active() const;
    uint32_t get_effective_max_chunks() const;
};

struct SchedulerState {
    static constexpr uint32_t DEFAULT_PREFETCH_LOADS_PER_FRAME = 6;
    static constexpr uint32_t MAX_PREFETCH_LOADS_PER_FRAME = 64;
    static constexpr uint32_t DEFAULT_SYNC_FALLBACK_LOADS_PER_FRAME = 1;
    static constexpr uint32_t MAX_SYNC_FALLBACK_LOADS_PER_FRAME = 8;
    static constexpr uint32_t DEFAULT_SYNC_FALLBACK_QUEUE_SIZE = 2048;
    static constexpr uint32_t SYNC_FALLBACK_QUEUE_COMPACT_MIN_PREFIX = 128;
    static constexpr bool DEFAULT_QUEUE_PRESSURE_CANDIDATE_SCAN_THROTTLE_ENABLED = false;
    static constexpr uint32_t DEFAULT_QUEUE_PRESSURE_CANDIDATE_SCAN_THROTTLE_MIN_QUEUE_DEPTH = 1;
    static constexpr uint32_t DEFAULT_QUEUE_PRESSURE_VISIBLE_SCAN_BUDGET = 1024;
    static constexpr uint32_t DEFAULT_QUEUE_PRESSURE_PREFETCH_SCAN_BUDGET = 1024;
    uint32_t max_visible_chunk_scan_per_frame = 4096;
    uint32_t max_prefetch_chunk_scan_per_frame = 4096;
    uint32_t max_prefetch_loads_per_frame = DEFAULT_PREFETCH_LOADS_PER_FRAME;
    uint32_t max_sync_fallback_loads_per_frame = DEFAULT_SYNC_FALLBACK_LOADS_PER_FRAME;
    uint32_t max_sync_fallback_queue_size = DEFAULT_SYNC_FALLBACK_QUEUE_SIZE;
    bool queue_pressure_candidate_scan_throttle_enabled =
            DEFAULT_QUEUE_PRESSURE_CANDIDATE_SCAN_THROTTLE_ENABLED;
    uint32_t queue_pressure_candidate_scan_throttle_min_queue_depth =
            DEFAULT_QUEUE_PRESSURE_CANDIDATE_SCAN_THROTTLE_MIN_QUEUE_DEPTH;
    uint32_t queue_pressure_candidate_scan_throttle_visible_scan_cap =
            DEFAULT_QUEUE_PRESSURE_VISIBLE_SCAN_BUDGET;
    uint32_t queue_pressure_candidate_scan_throttle_prefetch_scan_cap =
            DEFAULT_QUEUE_PRESSURE_PREFETCH_SCAN_BUDGET;
    uint32_t visible_scan_cursor = 0;
    uint32_t prefetch_scan_cursor = 0;
    uint32_t last_visible_scan_count = 0;
    uint32_t last_visible_scan_budget_effective = 0;
    uint32_t last_load_candidate_count = 0;
    uint32_t last_non_primary_scan_count = 0;
    uint32_t last_prefetch_scan_count = 0;
    uint32_t last_prefetch_scan_budget_effective = 0;
    uint32_t last_prefetch_candidate_count = 0;
    uint32_t last_prefetch_upload_pending_skip_count = 0;
    uint32_t last_prefetch_enqueued_count = 0;
    uint32_t last_prefetch_enqueue_headroom_stall_count = 0;
    uint32_t last_sync_fallback_queue_depth = 0;
    uint32_t last_sync_fallback_enqueued_count = 0;
    uint32_t last_sync_fallback_drained_count = 0;
    uint32_t last_sync_fallback_dropped_count = 0;
    uint32_t last_sync_fallback_stalled_count = 0;
    uint32_t prefetch_loads_remaining_this_frame = DEFAULT_PREFETCH_LOADS_PER_FRAME;
    uint32_t prefetch_scan_budget_remaining_this_frame = 0;
    bool queue_pressure_candidate_scan_throttle_active = false;
    uint32_t queue_pressure_candidate_scan_throttle_queue_depth = 0;
    bool force_sync_fallback_due_to_async_stall = false;
    LocalVector<uint64_t> sync_fallback_chunk_load_queue;
    HashSet<uint64_t> sync_fallback_chunk_load_set;
    uint32_t sync_fallback_chunk_load_queue_read_idx = 0;
    double last_update_cpu_ms = 0.0;
    double last_visibility_cpu_ms = 0.0;
    double last_load_cpu_ms = 0.0;
    double last_build_visible_cpu_ms = 0.0;
    double last_prefetch_cpu_ms = 0.0;
    double last_sync_fallback_cpu_ms = 0.0;
    double last_cpu_total_attributed_ms = 0.0;
    double last_cpu_unattributed_ms = 0.0;
};

struct DiagnosticsState {
    static constexpr uint32_t STALL_THRESHOLD_FRAMES = 30;
    static constexpr uint32_t LOG_INTERVAL_FRAMES = 120;

    uint32_t init_invalid_frames = 0;
    uint32_t culling_empty_frames = 0;
    uint32_t scheduler_stall_frames = 0;
    uint32_t upload_stall_frames = 0;
    uint32_t sync_fallback_stall_frames = 0;
    uint32_t queue_pressure_frames = 0;
    uint32_t vram_cap_hit_frames = 0;
    uint64_t visible_evict_fallback_attempts = 0;
    uint64_t visible_evict_fallback_successes = 0;

    uint32_t last_total_chunks = 0;
    uint32_t last_visible_chunks = 0;
    uint32_t last_loaded_chunks = 0;

    uint64_t invariant_slot_ownership_violations = 0;
    uint64_t invariant_upload_lifecycle_violations = 0;
    uint64_t invariant_generation_violations = 0;
    uint64_t integrity_mismatch_count = 0;
    String last_invariant_context;
    String last_invariant_message;
    String last_integrity_mismatch_message;

    String active_category = "ok";
    String active_reason = "healthy";
    String active_fingerprint = "ok";
    String last_logged_fingerprint;
    uint64_t last_fingerprint_log_frame = 0;
};

struct PrimaryChunkLayoutMetrics {
    bool spatial_partition_enabled = false;
    uint32_t source_index_count = 0;
    float avg_chunk_radius_ratio = 0.0f;
    float max_chunk_radius_ratio = 0.0f;
    float bounds_volume_ratio = 0.0f;

    void reset() {
        spatial_partition_enabled = false;
        source_index_count = 0;
        avg_chunk_radius_ratio = 0.0f;
        max_chunk_radius_ratio = 0.0f;
        bounds_volume_ratio = 0.0f;
    }
};

} // namespace GaussianStreamingTypes

#endif // STREAMING_RUNTIME_STATE_H
