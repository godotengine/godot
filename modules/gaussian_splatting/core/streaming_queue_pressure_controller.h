#ifndef STREAMING_QUEUE_PRESSURE_CONTROLLER_H
#define STREAMING_QUEUE_PRESSURE_CONTROLLER_H

#include "core/string/ustring.h"
#include <cstdint>

class StreamingQueuePressureController {
public:
    // Invariants:
    // 1) inactive state must serialize as source=none, reason=none.
    // 2) active state must expose a known source and reason token.
    // 3) summary source flags must match sampled queue/cap inputs.
    struct ScanBudgetInput {
        uint32_t base_scan_budget = 0;
        bool throttle_enabled = false;
        uint32_t throttle_min_queue_depth = 0;
        uint32_t observed_queue_depth = 0;
        uint32_t throttle_scan_cap = 0;
        uint32_t scanned_this_frame = 0;
        uint32_t enqueue_headroom = UINT32_MAX;
    };

    struct ScanBudgetResult {
        uint32_t scan_budget = 0;
        bool throttle_active = false;
        uint32_t effective_queue_depth = 0;
    };

    struct PressureSample {
        uint32_t pack_queue_depth = 0;
        uint32_t upload_queue_depth = 0;
        uint32_t sync_fallback_queue_depth = 0;
        bool pack_inflight_saturated = false;
        bool upload_frame_cap_hit = false;
        bool upload_bandwidth_cap_hit = false;
        bool chunk_load_cap_hit = false;
        bool vram_chunk_cap_hit = false;
        bool sync_backpressure = false;
    };

    struct PressureSummary {
        bool active = false;
        bool cap_active = false;
        bool pack_source_active = false;
        bool upload_source_active = false;
        bool sync_source_active = false;
        uint32_t backlog_depth = 0;
        String source = "none";
        String reason = "none";
    };

    static constexpr uint32_t ENQUEUE_HEADROOM_TARGET = 4u;

    static constexpr const char *SOURCE_NONE = "none";
    static constexpr const char *SOURCE_PACK = "pack";
    static constexpr const char *SOURCE_UPLOAD = "upload";
    static constexpr const char *SOURCE_SYNC = "sync";
    static constexpr const char *SOURCE_CAP = "cap";
    static constexpr const char *SOURCE_COMBINED = "combined";

    static constexpr const char *REASON_NONE = "none";
    static constexpr const char *REASON_QUEUE_BACKLOG = "queue_backlog";
    static constexpr const char *REASON_PACK_QUEUE_BACKLOG = "pack_queue_backlog";
    static constexpr const char *REASON_UPLOAD_QUEUE_BACKLOG = "upload_queue_backlog";
    static constexpr const char *REASON_SYNC_QUEUE_BACKLOG = "sync_queue_backlog";
    static constexpr const char *REASON_QUEUE_AND_CAPS = "queue_and_caps";
    static constexpr const char *REASON_UPLOAD_FRAME_CAP = "upload_frame_cap";
    static constexpr const char *REASON_UPLOAD_BANDWIDTH_CAP = "upload_bandwidth_cap";
    static constexpr const char *REASON_UPLOAD_CAP_COMBINED = "upload_cap_combined";
    static constexpr const char *REASON_CHUNK_LOAD_CAP = "chunk_load_cap";
    static constexpr const char *REASON_VRAM_CHUNK_CAP = "vram_chunk_cap";
    static constexpr const char *REASON_PACK_INFLIGHT_CAP = "pack_inflight_cap";
    static constexpr const char *REASON_SYNC_FALLBACK_PRESSURE = "sync_fallback_pressure";
    static constexpr const char *REASON_SYNC_QUEUE_CAP = "sync_queue_cap";
    static constexpr const char *REASON_CAP_COMBINED = "cap_combined";

    static ScanBudgetResult compute_candidate_scan_budget(const ScanBudgetInput &p_input);
    static PressureSummary summarize(const PressureSample &p_sample);

    static void reset_latched_state(bool &r_active, String &r_source, String &r_reason);
    static void mark_latched_state(bool &r_active, String &r_source, String &r_reason,
            const char *p_source, const char *p_reason);
    static void latch_summary(const PressureSummary &p_summary, bool &r_active, String &r_source, String &r_reason);

    static bool is_known_source(const String &p_source);
    static bool is_known_reason(const String &p_reason);
    static bool validate_summary_invariants(const PressureSummary &p_summary,
            const PressureSample &p_sample, String *r_error = nullptr);
    static bool validate_latched_state_invariants(bool p_active,
            const String &p_source, const String &p_reason, String *r_error = nullptr);
};

#endif // STREAMING_QUEUE_PRESSURE_CONTROLLER_H
