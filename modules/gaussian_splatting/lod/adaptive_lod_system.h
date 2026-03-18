#ifndef ADAPTIVE_LOD_SYSTEM_H
#define ADAPTIVE_LOD_SYSTEM_H

#include "scene/3d/camera_3d.h"
#include "core/math/math_defs.h"
#include "core/templates/vector.h"
#include "core/templates/local_vector.h"
#include "../core/gaussian_data.h"
#include "hierarchical_splat_structure.h"

namespace GaussianSplatting {

class AdaptiveLODSystem {
public:
    enum LODStrategy {
        DISTANCE_BASED,      // Simple distance culling
        IMPORTANCE_BASED,    // Visual importance metric
        BUDGET_BASED,        // Fixed splat budget
        HYBRID,              // Combination of all
        ADAPTIVE_QUALITY     // Dynamic quality based on framerate
    };

    struct LODConfig {
        // Distance thresholds for LOD levels
        float lod0_distance = 10.0f;   // Full detail
        float lod1_distance = 50.0f;   // 50% splats
        float lod2_distance = 100.0f;  // 25% splats
        float lod3_distance = 200.0f;  // 10% splats
        float cull_distance = 500.0f;  // Beyond this, cull completely
        float far_lod_keep_ratio = 0.1f; // Deterministic retention ratio for far LOD sampling

        // Budget constraints
        uint32_t max_splats_per_frame = 500000;
        uint32_t min_splats_per_frame = 10000;

        // Importance thresholds
        float importance_threshold = 0.1f;
        float size_cull_threshold = 0.5f;  // Cull splats smaller than N pixels

        // Quality settings
        float lod_bias = 1.0f;  // Global LOD bias (lower = more aggressive)
        bool smooth_transitions = true;
        float transition_time = 0.25f;  // Seconds for LOD transitions

        // Adaptive quality
        float target_framerate = 60.0f;
        float quality_adjustment_rate = 0.1f;
        bool enable_temporal_coherence = true;
        bool enable_painterly_mode = false;
    };

    struct LODSelection {
        LocalVector<uint32_t> visible_indices;
        LocalVector<float> lod_weights;     // For smooth transitions
        LocalVector<uint8_t> lod_levels;    // LOD level per splat
        LocalVector<PainterlyMetadata> painterly_metadata; // Painterly parameters per splat
        LocalVector<uint32_t> painterly_seeds;             // Active painterly seeds
        LocalVector<uint32_t> painterly_prev_seeds;        // Previous seeds for blending
        LocalVector<float> painterly_blend_weights;        // Blend weights between seeds

        struct Statistics {
            uint32_t total_tested = 0;
            uint32_t frustum_culled = 0;
            uint32_t distance_culled = 0;
            uint32_t size_culled = 0;
            uint32_t importance_culled = 0;
            uint32_t budget_culled = 0;

            uint32_t lod0_count = 0;
            uint32_t lod1_count = 0;
            uint32_t lod2_count = 0;
            uint32_t lod3_count = 0;

            uint32_t temporal_added = 0;
            uint32_t temporal_removed = 0;
            uint32_t temporal_retained = 0;
            float temporal_visibility_churn_ratio = 0.0f;

            float selection_time_ms = 0.0f;
        } stats;
    };

    struct ImportanceMetrics {
        float size_factor;       // Screen space size
        float opacity_factor;    // Alpha contribution
        float velocity_factor;   // Motion importance
        float saliency_factor;  // Visual saliency
        float temporal_factor;  // Temporal coherence

        float compute_total() const {
            return size_factor * 0.3f +
                   opacity_factor * 0.2f +
                   velocity_factor * 0.2f +
                   saliency_factor * 0.2f +
                   temporal_factor * 0.1f;
        }
    };

public:
    AdaptiveLODSystem();
    ~AdaptiveLODSystem();

    // Initialize with configuration
    void initialize(const LODConfig& p_config);

    // Main LOD selection method
    LODSelection select_lod_splats(
        const Vector<GaussianData>& all_splats,
        const Camera3D* camera,
        HierarchicalSplatStructure* spatial_structure,
        LODStrategy strategy = HYBRID
    );

    // Individual selection strategies
    LODSelection select_distance_based(
        const Vector<GaussianData>& splats,
        const Vector3& camera_pos,
        const LODConfig& p_config
    );

    LODSelection select_importance_based(
        const Vector<GaussianData>& splats,
        const Camera3D* camera,
        const LODConfig& p_config
    );

    LODSelection select_budget_based(
        const Vector<GaussianData>& splats,
        const Camera3D* camera,
        uint32_t budget
    );

    // Hybrid selection combining multiple strategies
    LODSelection select_hybrid(
        const Vector<GaussianData>& splats,
        const Camera3D* camera,
        HierarchicalSplatStructure* spatial_structure,
        float p_effective_lod_bias
    );

    // Adaptive quality based on performance
    void update_adaptive_quality(float current_framerate);
    float get_adaptive_lod_bias() const { return adaptive_lod_bias; }

    // Importance calculation
    ImportanceMetrics calculate_importance(
        const GaussianData& splat,
        const Camera3D* camera,
        const Vector3& prev_camera_pos
    ) const;

    // Screen space calculations
    float compute_screen_size(
        const GaussianData& splat,
        const Camera3D* camera
    ) const;

    bool is_splat_visible(
        const GaussianData& splat,
        const Frustum& frustum,
        float min_size = 0.5f
    ) const;

    // Temporal coherence
    void update_temporal_cache(LODSelection& selection);
    float compute_temporal_importance(uint32_t splat_index) const;

    // LOD level determination
    uint8_t compute_lod_level(
        float distance,
        float importance,
        const LODConfig& p_config
    ) const;

    // Smooth LOD transitions
    float compute_transition_weight(
        float distance,
        float lod_boundary,
        float transition_width = 5.0f
    ) const;

    // Configuration
    void set_config(const LODConfig& p_config) { this->config = p_config; }
    const LODConfig& get_config() const { return config; }

    // Statistics
    struct FrameStatistics {
        float avg_selection_time_ms;
        float avg_splats_rendered;
        float avg_cull_ratio;
        float avg_temporal_churn_ratio;
        uint32_t frame_count;

        void update(const LODSelection::Statistics& frame_stats);
        void reset();
    };

    const FrameStatistics& get_frame_statistics() const { return frame_stats; }

private:
    LODConfig config;
    LODStrategy current_strategy;

    // Adaptive quality state
    float adaptive_lod_bias;
    float quality_trend;
    uint32_t low_fps_frames;
    uint32_t high_fps_frames;

    // Temporal coherence cache
    struct TemporalCache {
        LocalVector<uint32_t> previous_visible;
        LocalVector<uint8_t> visibility_history;  // Per-splat visibility count
        uint32_t frame_number;
    } temporal_cache;

    // Performance monitoring
    FrameStatistics frame_stats;

    // Helper methods
    void sort_by_importance(
        LocalVector<std::pair<float, uint32_t>>& importance_pairs,
        uint32_t max_count
    );

    void apply_budget_constraint(
        LODSelection& selection,
        uint32_t budget
    );

    void apply_smooth_transitions(
        LODSelection& selection,
        const TemporalCache& cache
    );

    bool frustum_contains_point(
        const Frustum& frustum,
        const Vector3& point
    ) const;
};

// Inline implementations for performance
inline float AdaptiveLODSystem::compute_transition_weight(
    float distance,
    float lod_boundary,
    float transition_width) const {

    float delta = distance - lod_boundary;
    if (delta < -transition_width) return 1.0f;
    if (delta > transition_width) return 0.0f;

    // Smooth transition using cosine
    float t = (delta + transition_width) / (2.0f * transition_width);
    return 0.5f + 0.5f * cos(Math::PI * t);
}

inline uint8_t AdaptiveLODSystem::compute_lod_level(
    float distance,
    float importance,
    const LODConfig& p_config) const {

    // Adjust distance threshold based on importance
    float adjusted_distance = distance / (importance + 0.1f);

    if (adjusted_distance < p_config.lod0_distance) return 0;
    if (adjusted_distance < p_config.lod1_distance) return 1;
    if (adjusted_distance < p_config.lod2_distance) return 2;
    if (adjusted_distance < p_config.lod3_distance) return 3;

    return 4;  // Culled
}

} // namespace GaussianSplatting

#endif // ADAPTIVE_LOD_SYSTEM_H
