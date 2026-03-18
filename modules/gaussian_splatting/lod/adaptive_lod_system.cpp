/**
 * Adaptive LOD (Level of Detail) System for Gaussian Splatting
 * =============================================================
 *
 * ALGORITHM OVERVIEW
 * ------------------
 * This system dynamically selects which Gaussian splats to render based on their
 * visual importance, enabling real-time performance with millions of splats.
 *
 * LOD SELECTION STRATEGIES
 * ------------------------
 * 1. DISTANCE_BASED: Simple distance thresholds with LOD levels (0-3)
 * 2. IMPORTANCE_BASED: Visual importance (screen-size, opacity, view angle)
 * 3. BUDGET_BASED: Fixed splat count with distance priority
 * 4. HYBRID (default): Hierarchical spatial culling with importance fallback
 * 5. ADAPTIVE_QUALITY: Performance-driven dynamic FPS-based adjustment
 *
 * TEMPORAL COHERENCE: Visibility history reduces LOD popping with smooth transitions.
 *
 * KEY DATA STRUCTURES
 * -------------------
 * - LODSelection: Visible indices, LOD weights, and levels
 * - LODConfig: Distance thresholds, budget limits, quality settings
 * - TemporalCache: Per-splat visibility history
 * - ImportanceMetrics: Multi-factor importance calculation
 */

#include "adaptive_lod_system.h"
#include "core/os/os.h"
#include "core/math/math_funcs.h"
#include "core/math/plane.h"
#include "core/templates/hash_set.h"
#include <algorithm>
#include <cstdint>

namespace {

static inline uint32_t _mix_hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static inline uint32_t _hash_signed_coord(int32_t v) {
    return _mix_hash(static_cast<uint32_t>(v) ^ 0x9e3779b9U);
}

static inline bool _keep_far_lod_sample(
    const GaussianSplatting::GaussianData &splat,
    const GaussianSplatting::AdaptiveLODSystem::LODConfig &config) {

    const float retain_ratio = CLAMP(config.far_lod_keep_ratio, 0.01f, 1.0f);
    const float cell_size = MAX(1.0f, config.lod3_distance * 0.1f);

    const int32_t qx = static_cast<int32_t>(Math::floor(splat.position.x / cell_size));
    const int32_t qy = static_cast<int32_t>(Math::floor(splat.position.y / cell_size));
    const int32_t qz = static_cast<int32_t>(Math::floor(splat.position.z / cell_size));

    uint32_t hash = 2166136261U;
    hash = _mix_hash(hash ^ _hash_signed_coord(qx));
    hash = _mix_hash(hash ^ _hash_signed_coord(qy));
    hash = _mix_hash(hash ^ _hash_signed_coord(qz));
    hash = _mix_hash(hash ^ splat.index);

    const float normalized = static_cast<float>(hash) / static_cast<float>(UINT32_MAX);
    return normalized <= retain_ratio;
}

} // namespace

namespace GaussianSplatting {

AdaptiveLODSystem::AdaptiveLODSystem()
    : current_strategy(HYBRID),
      adaptive_lod_bias(1.0f),
      quality_trend(0.0f),
      low_fps_frames(0),
      high_fps_frames(0) {

    temporal_cache.frame_number = 0;
    frame_stats.reset();
}

AdaptiveLODSystem::~AdaptiveLODSystem() {
}

void AdaptiveLODSystem::initialize(const LODConfig& p_config) {
    config = p_config;
    adaptive_lod_bias = p_config.lod_bias;
}

AdaptiveLODSystem::LODSelection AdaptiveLODSystem::select_lod_splats(
    const Vector<GaussianData>& all_splats,
    const Camera3D* camera,
    HierarchicalSplatStructure* spatial_structure,
    LODStrategy strategy) {

    uint64_t start_time = OS::get_singleton()->get_ticks_usec();

    LODSelection selection;
    current_strategy = strategy;

    const bool hierarchy_available = spatial_structure && spatial_structure->get_root();
    const bool can_select_from_hierarchy_only =
        hierarchy_available && (strategy == HYBRID || strategy == ADAPTIVE_QUALITY);

    if (!camera) {
        return selection;
    }

    if (all_splats.is_empty() && !can_select_from_hierarchy_only) {
        return selection;
    }

    // Choose selection strategy
    switch (strategy) {
        case DISTANCE_BASED:
            selection = select_distance_based(
                all_splats,
                camera->get_global_transform().origin,
                config
            );
            break;

        case IMPORTANCE_BASED:
            selection = select_importance_based(all_splats, camera, config);
            break;

        case BUDGET_BASED:
            selection = select_budget_based(
                all_splats,
                camera,
                config.max_splats_per_frame
            );
            break;

        case HYBRID:
            selection = select_hybrid(all_splats, camera, spatial_structure, config.lod_bias);
            break;

        case ADAPTIVE_QUALITY:
            // Adjust LOD bias based on performance without mutating shared config.
            selection = select_hybrid(all_splats, camera, spatial_structure, adaptive_lod_bias);
            break;
    }

    // Apply temporal coherence if enabled
    if (config.enable_temporal_coherence) {
        apply_smooth_transitions(selection, temporal_cache);
        update_temporal_cache(selection);
    }

    // Update statistics
    selection.stats.selection_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;
    frame_stats.update(selection.stats);

    return selection;
}

AdaptiveLODSystem::LODSelection AdaptiveLODSystem::select_distance_based(
    const Vector<GaussianData>& splats,
    const Vector3& camera_pos,
    const LODConfig& p_config) {

    LODSelection selection;
    selection.stats.total_tested = splats.size();

    // Reserve space
    selection.visible_indices.reserve(splats.size() / 2);
    selection.lod_weights.reserve(splats.size() / 2);
    selection.lod_levels.reserve(splats.size() / 2);

    for (uint32_t i = 0; i < splats.size(); i++) {
        const GaussianData& splat = splats[i];
        float distance = (splat.position - camera_pos).length();

        // Distance culling
        if (distance > p_config.cull_distance) {
            selection.stats.distance_culled++;
            continue;
        }

        // Compute LOD level
        uint8_t lod_level = 0;
        float lod_weight = 1.0f;

        if (distance < p_config.lod0_distance) {
            lod_level = 0;
            selection.stats.lod0_count++;
        } else if (distance < p_config.lod1_distance) {
            lod_level = 1;
            lod_weight = compute_transition_weight(distance, p_config.lod0_distance);
            selection.stats.lod1_count++;
        } else if (distance < p_config.lod2_distance) {
            lod_level = 2;
            lod_weight = compute_transition_weight(distance, p_config.lod1_distance) * 0.5f;
            selection.stats.lod2_count++;
        } else if (distance < p_config.lod3_distance) {
            lod_level = 3;
            lod_weight = compute_transition_weight(distance, p_config.lod2_distance) * 0.25f;
            selection.stats.lod3_count++;
        } else {
            // Deterministic spatial sampling for far LOD keeps temporal stability under camera motion.
            if (!_keep_far_lod_sample(splat, p_config)) {
                continue;
            }
            lod_level = 3;
            lod_weight = CLAMP(p_config.far_lod_keep_ratio, 0.01f, 1.0f);
            selection.stats.lod3_count++;
        }

        selection.visible_indices.push_back(i);
        selection.lod_weights.push_back(lod_weight);
        selection.lod_levels.push_back(lod_level);
        if (p_config.enable_painterly_mode) {
            selection.painterly_metadata.push_back(splats[i].painterly);
            selection.painterly_seeds.push_back(splats[i].painterly.temporal_seed);
            selection.painterly_prev_seeds.push_back(splats[i].painterly.temporal_seed);
            selection.painterly_blend_weights.push_back(1.0f);
        }
    }

    return selection;
}

AdaptiveLODSystem::LODSelection AdaptiveLODSystem::select_importance_based(
    const Vector<GaussianData>& splats,
    const Camera3D* camera,
    const LODConfig& p_config) {

    LODSelection selection;
    selection.stats.total_tested = splats.size();

    Vector3 camera_pos = camera->get_global_transform().origin;
    Vector3 camera_forward = -camera->get_global_transform().basis.get_column(2);

    // Calculate importance for all splats
    LocalVector<std::pair<float, uint32_t>> importance_pairs;
    importance_pairs.reserve(splats.size());

    for (uint32_t i = 0; i < splats.size(); i++) {
        const GaussianData& splat = splats[i];

        // Basic importance calculation
        float distance = (splat.position - camera_pos).length();
        if (distance > p_config.cull_distance) {
            selection.stats.distance_culled++;
            continue;
        }

        // Screen space size
        float screen_size = compute_screen_size(splat, camera);
        if (screen_size < p_config.size_cull_threshold) {
            selection.stats.size_culled++;
            continue;
        }

        // View angle factor (splats in front are more important)
        Vector3 to_splat = (splat.position - camera_pos).normalized();
        float view_factor = MAX(0.0f, to_splat.dot(camera_forward));

        // Opacity factor
        float opacity_factor = splat.color.a;

        // Combined importance
        float importance = (screen_size * 2.0f +
                          view_factor * 1.5f +
                          opacity_factor * 1.0f) / distance;

        if (importance > p_config.importance_threshold) {
            importance_pairs.push_back({importance, i});
        } else {
            selection.stats.importance_culled++;
        }
    }

    // Sort by importance and select top N
    sort_by_importance(importance_pairs, p_config.max_splats_per_frame);

    // Add selected splats
    selection.visible_indices.reserve(importance_pairs.size());
    selection.lod_weights.reserve(importance_pairs.size());
    selection.lod_levels.reserve(importance_pairs.size());

    for (const auto& pair : importance_pairs) {
        uint32_t idx = pair.second;
        float importance = pair.first;
        float distance = (splats[idx].position - camera_pos).length();

        uint8_t lod_level = compute_lod_level(distance, importance, p_config);
        float lod_weight = MIN(1.0f, importance * 2.0f);

        selection.visible_indices.push_back(idx);
        selection.lod_weights.push_back(lod_weight);
        selection.lod_levels.push_back(lod_level);
        if (p_config.enable_painterly_mode) {
            selection.painterly_metadata.push_back(splats[idx].painterly);
            selection.painterly_seeds.push_back(splats[idx].painterly.temporal_seed);
            selection.painterly_prev_seeds.push_back(splats[idx].painterly.temporal_seed);
            selection.painterly_blend_weights.push_back(1.0f);
        }

        // Update LOD statistics
        switch (lod_level) {
            case 0: selection.stats.lod0_count++; break;
            case 1: selection.stats.lod1_count++; break;
            case 2: selection.stats.lod2_count++; break;
            case 3: selection.stats.lod3_count++; break;
        }
    }

    return selection;
}

AdaptiveLODSystem::LODSelection AdaptiveLODSystem::select_budget_based(
    const Vector<GaussianData>& splats,
    const Camera3D* camera,
    uint32_t budget) {

    LODSelection selection;
    selection.stats.total_tested = splats.size();

    Vector3 camera_pos = camera->get_global_transform().origin;
    Frustum frustum = camera->get_frustum();

    // Create distance-sorted list
    LocalVector<std::pair<float, uint32_t>> distance_pairs;
    distance_pairs.reserve(splats.size());

    for (uint32_t i = 0; i < splats.size(); i++) {
        const GaussianData& splat = splats[i];

        // Frustum culling
        if (!frustum_contains_point(frustum, splat.position)) {
            selection.stats.frustum_culled++;
            continue;
        }

        float distance = (splat.position - camera_pos).length();
        if (distance < config.cull_distance) {
            distance_pairs.push_back({distance, i});
        } else {
            selection.stats.distance_culled++;
        }
    }

    // Sort by distance
    std::partial_sort(
        distance_pairs.ptr(),
        distance_pairs.ptr() + MIN(budget, distance_pairs.size()),
        distance_pairs.ptr() + distance_pairs.size(),
        [](const auto& a, const auto& b) { return a.first < b.first; }
    );

    // Select up to budget
    uint32_t selected_count = MIN(budget, distance_pairs.size());
    selection.visible_indices.reserve(selected_count);
    selection.lod_weights.reserve(selected_count);
    selection.lod_levels.reserve(selected_count);

    for (uint32_t i = 0; i < selected_count; i++) {
        uint32_t idx = distance_pairs[i].second;
        float distance = distance_pairs[i].first;

        uint8_t lod_level = 0;
        if (distance > config.lod2_distance) lod_level = 2;
        else if (distance > config.lod1_distance) lod_level = 1;
        else if (distance > config.lod0_distance) lod_level = 0;

        float lod_weight = 1.0f / (1.0f + lod_level * 0.5f);

        selection.visible_indices.push_back(idx);
        selection.lod_weights.push_back(lod_weight);
        selection.lod_levels.push_back(lod_level);
        if (config.enable_painterly_mode) {
            selection.painterly_metadata.push_back(splats[idx].painterly);
            selection.painterly_seeds.push_back(splats[idx].painterly.temporal_seed);
            selection.painterly_prev_seeds.push_back(splats[idx].painterly.temporal_seed);
            selection.painterly_blend_weights.push_back(1.0f);
        }

        // Update statistics
        switch (lod_level) {
            case 0: selection.stats.lod0_count++; break;
            case 1: selection.stats.lod1_count++; break;
            case 2: selection.stats.lod2_count++; break;
            case 3: selection.stats.lod3_count++; break;
        }
    }

    selection.stats.budget_culled = distance_pairs.size() - selected_count;

    return selection;
}

AdaptiveLODSystem::LODSelection AdaptiveLODSystem::select_hybrid(
    const Vector<GaussianData>& splats,
    const Camera3D* camera,
    HierarchicalSplatStructure* spatial_structure,
    float p_effective_lod_bias) {

    LODSelection selection;

    if (spatial_structure && spatial_structure->get_root()) {
        // Use hierarchical culling for efficiency
        Vector3 camera_pos = camera->get_global_transform().origin;
        Frustum frustum = camera->get_frustum();

        auto query_result = spatial_structure->query_visible_splats(
            frustum,
            camera_pos,
            p_effective_lod_bias,
            config.max_splats_per_frame
        );

        // Convert query result to LOD selection
        selection.visible_indices = query_result.visible_indices;
        selection.lod_weights = query_result.lod_weights;

        if (selection.visible_indices.size() != selection.lod_weights.size()) {
            const uint32_t safe_size = MIN(selection.visible_indices.size(), selection.lod_weights.size());
            ERR_PRINT(vformat(
                "Adaptive hybrid LOD cardinality mismatch (indices=%d, weights=%d). Truncating to %d.",
                selection.visible_indices.size(),
                selection.lod_weights.size(),
                safe_size));
            selection.visible_indices.resize(safe_size);
            selection.lod_weights.resize(safe_size);
        }

        // Assign LOD levels based on weights
        const uint32_t selected_count = selection.visible_indices.size();
        selection.lod_levels.resize(selected_count);
        for (uint32_t i = 0; i < selected_count; i++) {
            float weight = selection.lod_weights[i];
            if (weight > 0.75f) {
                selection.lod_levels[i] = 0;
                selection.stats.lod0_count++;
            } else if (weight > 0.5f) {
                selection.lod_levels[i] = 1;
                selection.stats.lod1_count++;
            } else if (weight > 0.25f) {
                selection.lod_levels[i] = 2;
                selection.stats.lod2_count++;
            } else {
                selection.lod_levels[i] = 3;
                selection.stats.lod3_count++;
            }
        }

        selection.stats.total_tested = spatial_structure->get_total_splats();
        selection.stats.frustum_culled =
            uint32_t(query_result.culled_percentage * selection.stats.total_tested / 100.0f);

    } else {
        // Fallback to importance-based selection
        selection = select_importance_based(splats, camera, config);
    }

    // Apply budget constraint if needed
    if (selection.visible_indices.size() > config.max_splats_per_frame) {
        apply_budget_constraint(selection, config.max_splats_per_frame);
    }

    return selection;
}

void AdaptiveLODSystem::update_adaptive_quality(float current_framerate) {
    float target = config.target_framerate;
    float delta = current_framerate - target;

    // Update quality trend
    quality_trend = quality_trend * 0.9f + delta * 0.1f;

    // Count consecutive low/high FPS frames
    if (current_framerate < target * 0.9f) {
        low_fps_frames++;
        high_fps_frames = 0;
    } else if (current_framerate > target * 1.1f) {
        high_fps_frames++;
        low_fps_frames = 0;
    } else {
        low_fps_frames = (low_fps_frames > 0) ? (low_fps_frames - 1) : 0;
        high_fps_frames = (high_fps_frames > 0) ? (high_fps_frames - 1) : 0;
    }

    // Adjust LOD bias
    if (low_fps_frames > 5) {
        // Reduce quality for better performance
        adaptive_lod_bias *= (1.0f - config.quality_adjustment_rate);
        adaptive_lod_bias = MAX(0.25f, adaptive_lod_bias);
        low_fps_frames = 0;
    } else if (high_fps_frames > 10) {
        // Increase quality if we have headroom
        adaptive_lod_bias *= (1.0f + config.quality_adjustment_rate * 0.5f);
        adaptive_lod_bias = MIN(2.0f, adaptive_lod_bias);
        high_fps_frames = 0;
    }
}

AdaptiveLODSystem::ImportanceMetrics AdaptiveLODSystem::calculate_importance(
    const GaussianData& splat,
    const Camera3D* camera,
    const Vector3& prev_camera_pos) const {

    ImportanceMetrics metrics = {};

    Vector3 camera_pos = camera->get_global_transform().origin;
    Vector3 to_splat = splat.position - camera_pos;
    float distance = to_splat.length();

    // Screen space size
    metrics.size_factor = compute_screen_size(splat, camera);

    // Opacity
    metrics.opacity_factor = splat.color.a;

    // Velocity (camera motion)
    Vector3 camera_velocity = camera_pos - prev_camera_pos;
    Vector3 relative_velocity = to_splat.normalized().cross(camera_velocity);
    metrics.velocity_factor = MIN(1.0f, relative_velocity.length() * 0.1f);

    // Saliency (simplified - based on color contrast)
    // In production, this would use more sophisticated saliency detection
    metrics.saliency_factor = 0.5f;

    // Temporal coherence
    metrics.temporal_factor = compute_temporal_importance(splat.index);

    return metrics;
}

float AdaptiveLODSystem::compute_screen_size(
    const GaussianData& splat,
    const Camera3D* camera) const {

    // Get camera parameters
    Transform3D cam_to_world_transform = camera->get_camera_transform();
    Vector3 camera_pos = cam_to_world_transform.origin;

    // Distance to splat
    float distance = (splat.position - camera_pos).length();
    if (distance < 0.001f) return 1000.0f;  // Very close

    // Estimate splat radius from covariance
    float splat_radius = splat.compute_radius();

    // Project to screen space (simplified)
    float fov = camera->get_fov();
    float screen_height = 1080.0f;  // Assumed screen height
    float projection_factor = screen_height / (2.0f * tan(Math::deg_to_rad(fov * 0.5f)));

    float screen_size = (splat_radius * projection_factor) / distance;

    return screen_size;
}

bool AdaptiveLODSystem::is_splat_visible(
    const GaussianData& splat,
    const Frustum& frustum,
    float min_size) const {

    // Frustum test
    if (!frustum_contains_point(frustum, splat.position)) {
        return false;
    }

    // Size test (would need camera for accurate test)
    // For now, just return true if passed frustum test
    return true;
}

void AdaptiveLODSystem::update_temporal_cache(LODSelection& selection) {
    const uint32_t previous_count = temporal_cache.previous_visible.size();
    const uint32_t current_count = selection.visible_indices.size();

    if (previous_count == 0) {
        selection.stats.temporal_added = current_count;
        selection.stats.temporal_removed = 0;
        selection.stats.temporal_retained = 0;
        selection.stats.temporal_visibility_churn_ratio = 0.0f;
    } else {
        HashSet<uint32_t> previous_visible_set;
        for (uint32_t idx : temporal_cache.previous_visible) {
            previous_visible_set.insert(idx);
        }

        uint32_t retained_count = 0;
        uint32_t added_count = 0;
        for (uint32_t idx : selection.visible_indices) {
            if (previous_visible_set.has(idx)) {
                retained_count++;
                previous_visible_set.erase(idx);
            } else {
                added_count++;
            }
        }

        const uint32_t removed_count = previous_visible_set.size();
        const uint32_t churn_count = added_count + removed_count;
        const uint32_t churn_denominator = MAX((uint32_t)1, MAX(previous_count, current_count));

        selection.stats.temporal_added = added_count;
        selection.stats.temporal_removed = removed_count;
        selection.stats.temporal_retained = retained_count;
        selection.stats.temporal_visibility_churn_ratio = float(churn_count) / float(churn_denominator);
    }

    temporal_cache.previous_visible = selection.visible_indices;
    temporal_cache.frame_number++;

    // Update visibility history
    if (temporal_cache.visibility_history.size() < selection.stats.total_tested) {
        temporal_cache.visibility_history.resize(selection.stats.total_tested);
    }

    // Decay old visibility
    for (uint32_t i = 0; i < temporal_cache.visibility_history.size(); i++) {
        if (temporal_cache.visibility_history[i] > 0) {
            temporal_cache.visibility_history[i]--;
        }
    }

    // Mark current visible splats
    for (uint32_t idx : selection.visible_indices) {
        temporal_cache.visibility_history[idx] = MIN(255, temporal_cache.visibility_history[idx] + 2);
    }
}

float AdaptiveLODSystem::compute_temporal_importance(uint32_t splat_index) const {
    if (splat_index >= temporal_cache.visibility_history.size()) {
        return 0.5f;
    }

    uint8_t history = temporal_cache.visibility_history[splat_index];
    return float(history) / 255.0f;
}

void AdaptiveLODSystem::sort_by_importance(
    LocalVector<std::pair<float, uint32_t>>& importance_pairs,
    uint32_t max_count) {

    if (importance_pairs.size() <= max_count) {
        // Sort all
        std::sort(
            importance_pairs.ptr(),
            importance_pairs.ptr() + importance_pairs.size(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
    } else {
        // Partial sort for efficiency
        std::partial_sort(
            importance_pairs.ptr(),
            importance_pairs.ptr() + max_count,
            importance_pairs.ptr() + importance_pairs.size(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
        importance_pairs.resize(max_count);
    }
}

void AdaptiveLODSystem::apply_budget_constraint(
    LODSelection& selection,
    uint32_t budget) {

    if (selection.visible_indices.size() <= budget) {
        return;
    }

    // Remove lowest importance splats
    // For now, just truncate (in production, would sort by importance first)
    uint32_t to_remove = selection.visible_indices.size() - budget;
    selection.visible_indices.resize(budget);
    selection.lod_weights.resize(budget);
    selection.lod_levels.resize(budget);
    if (selection.painterly_metadata.size() > budget) {
        selection.painterly_metadata.resize(budget);
    }
    if (selection.painterly_seeds.size() > budget) {
        selection.painterly_seeds.resize(budget);
    }
    if (selection.painterly_prev_seeds.size() > budget) {
        selection.painterly_prev_seeds.resize(budget);
    }
    if (selection.painterly_blend_weights.size() > budget) {
        selection.painterly_blend_weights.resize(budget);
    }
    selection.stats.budget_culled += to_remove;
}

void AdaptiveLODSystem::apply_smooth_transitions(
    LODSelection& selection,
    const TemporalCache& cache) {

    if (!config.smooth_transitions) {
        return;
    }

    // Blend LOD weights based on temporal coherence
    for (uint32_t i = 0; i < selection.visible_indices.size(); i++) {
        uint32_t idx = selection.visible_indices[i];
        float temporal_factor = compute_temporal_importance(idx);

        // Smooth weight transition
        selection.lod_weights[i] = selection.lod_weights[i] * 0.7f + temporal_factor * 0.3f;
    }
}

bool AdaptiveLODSystem::frustum_contains_point(
    const Frustum& frustum,
    const Vector3& point) const {

    if (frustum.plane_count == 0 && frustum.planes.is_empty()) {
        return true;
    }

    const Plane* planes_ptr = frustum.planes_ptr;
    uint32_t plane_count = frustum.plane_count;

    if (!planes_ptr) {
        planes_ptr = frustum.planes.ptr();
        plane_count = frustum.planes.size();
    }

    for (uint32_t i = 0; i < plane_count; i++) {
        if (planes_ptr[i].distance_to(point) >= 0.0f) {
            return false;
        }
    }

    return true;
}

void AdaptiveLODSystem::FrameStatistics::update(const LODSelection::Statistics& frame_stats) {
    frame_count++;

    float alpha = 0.1f;  // Exponential moving average factor
    avg_selection_time_ms = avg_selection_time_ms * (1.0f - alpha) + frame_stats.selection_time_ms * alpha;

    uint32_t total_rendered = frame_stats.lod0_count + frame_stats.lod1_count +
                             frame_stats.lod2_count + frame_stats.lod3_count;
    avg_splats_rendered = avg_splats_rendered * (1.0f - alpha) + total_rendered * alpha;

    if (frame_stats.total_tested > 0) {
        float cull_ratio = 1.0f - (float(total_rendered) / frame_stats.total_tested);
        avg_cull_ratio = avg_cull_ratio * (1.0f - alpha) + cull_ratio * alpha;
    }
    avg_temporal_churn_ratio = avg_temporal_churn_ratio * (1.0f - alpha) +
            frame_stats.temporal_visibility_churn_ratio * alpha;
}

void AdaptiveLODSystem::FrameStatistics::reset() {
    avg_selection_time_ms = 0.0f;
    avg_splats_rendered = 0.0f;
    avg_cull_ratio = 0.0f;
    avg_temporal_churn_ratio = 0.0f;
    frame_count = 0;
}

} // namespace GaussianSplatting
