#ifndef STREAMING_VISIBILITY_CONTROLLER_H
#define STREAMING_VISIBILITY_CONTROLLER_H

#include "core/math/aabb.h"
#include "core/math/plane.h"
#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "streaming_vram_regulator.h"
#include <cstdint>

class GaussianStreamingSystem;

class StreamingVisibilityController {
    friend class GaussianStreamingSystem;

public:
    struct ChunkCullingStats {
        uint32_t total_chunks = 0;
        uint32_t visible_chunks = 0;
        uint32_t frustum_culled_chunks = 0;
        uint32_t loaded_chunks = 0;

        void reset() {
            total_chunks = 0;
            visible_chunks = 0;
            frustum_culled_chunks = 0;
            loaded_chunks = 0;
        }
    };

    struct CameraVelocityTracker {
        Vector3 last_position;
        Vector3 velocity;
        bool has_previous_position = false;

        void update(const Vector3 &current_pos, float delta_time);
        Vector3 predict_position(const Vector3 &current_pos, float lookahead_distance) const;
    };

    enum class ZeroVisibleRecoveryMode : uint8_t {
        STARTUP_ONLY = 0,
        PERSISTENT = 1,
    };

    struct ZeroVisibleRecoveryState {
        ZeroVisibleRecoveryMode mode = ZeroVisibleRecoveryMode::PERSISTENT;
        uint32_t persistent_trigger_frames = 16;
        uint32_t persistent_cooldown_frames = 30;
        uint32_t stall_log_interval_frames = 120;
        uint32_t zero_visible_consecutive_frames = 0;
        uint64_t last_recovery_frame = UINT64_MAX;
        uint64_t last_stall_log_frame = UINT64_MAX;
        uint64_t recoveries_triggered = 0;
        uint64_t stall_detections = 0;
    };

    void reset_runtime_state();
    void clear_visible_state();
    void update_camera_tracking(const Vector3 &camera_pos, float frame_delta_seconds);
    void load_zero_visible_recovery_config_from_project_settings();
    void handle_zero_visible_chunk_recovery(GaussianStreamingSystem &system);
    void update_chunk_visibility(GaussianStreamingSystem &system, const Transform3D &camera_transform, const Projection &projection);
    bool is_chunk_in_frustum(const AABB &bounds, const Vector<Plane> &frustum_planes) const;
    void update_culling_config_from_project_settings();
    void load_prefetch_config_from_project_settings();
    void load_lod_blend_config_from_project_settings();
    float calculate_lod_blend_factor(float distance, float lod_distance) const;
    void update_chunk_lod_blend_factors(GaussianStreamingSystem &system, const Vector3 &camera_pos);
    void update_chunk_lod_parameters(GaussianStreamingSystem &system, const Vector3 &camera_pos);
    uint32_t prefetch_chunks_at_predicted_position(GaussianStreamingSystem &system, const Vector3 &predicted_pos,
            uint32_t available_slots, uint32_t load_budget, uint32_t max_scan_budget);
    float get_visible_count_change_ratio() const;
    float get_effective_count_change_ratio(uint32_t visible_chunks_evicted_this_frame) const;

private:
    uint32_t get_prefetch_limit(GaussianStreamingSystem &system, uint32_t available_slots, uint32_t load_budget) const;
    void collect_prefetch_candidates(GaussianStreamingSystem &system, const Vector3 &predicted_pos,
            uint32_t max_prefetch, uint32_t max_scan_budget, LocalVector<uint32_t> &out_candidates) const;
    uint32_t schedule_prefetch_loads(GaussianStreamingSystem &system, const Vector3 &predicted_pos,
            const LocalVector<uint32_t> &candidates, uint32_t available_slots, uint32_t load_budget);

    bool chunk_frustum_culling_enabled = true;
    float chunk_frustum_padding = 1.5f;
    float chunk_radius_multiplier = 1.0f;
    ChunkCullingStats culling_stats;
    LocalVector<uint32_t> visible_chunk_indices;
    CameraVelocityTracker camera_tracker;
    bool predictive_prefetch_enabled = true;
    float prefetch_lookahead_distance = 10.0f;
    LODBlendConfig lod_blend_config;
    float current_lod_blend_factor = 1.0f;
    int global_sh_band_level = 3;
    uint32_t lod_transitions_this_frame = 0;
    uint32_t prev_visible_count = 0;
    ZeroVisibleRecoveryState zero_visible_recovery;
};

#endif // STREAMING_VISIBILITY_CONTROLLER_H
