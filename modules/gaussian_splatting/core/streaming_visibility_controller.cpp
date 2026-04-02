#include "streaming_visibility_controller.h"

#include "gaussian_streaming.h"
#include "gs_project_settings.h"
#include "streaming_queue_pressure_controller.h"
#include "../lod/lod_config.h"
#include "../logger/gs_logger.h"
#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include <cfloat>

namespace {
uint32_t _compute_visibility_async_enqueue_headroom(
        uint32_t queued_chunk_loads_this_frame,
        uint32_t max_chunk_loads_per_frame,
        uint32_t pack_jobs_in_flight,
        uint32_t max_pack_jobs_in_flight) {
    uint32_t frame_headroom = UINT32_MAX;
    if (max_chunk_loads_per_frame > 0) {
        frame_headroom = queued_chunk_loads_this_frame < max_chunk_loads_per_frame
                ? (max_chunk_loads_per_frame - queued_chunk_loads_this_frame)
                : 0;
    }

    uint32_t flight_headroom = UINT32_MAX;
    if (max_pack_jobs_in_flight > 0) {
        flight_headroom = pack_jobs_in_flight < max_pack_jobs_in_flight
                ? (max_pack_jobs_in_flight - pack_jobs_in_flight)
                : 0;
    }

    return MIN(frame_headroom, flight_headroom);
}
} // namespace

void StreamingVisibilityController::reset_runtime_state() {
    clear_visible_state();
    camera_tracker = CameraVelocityTracker();
    current_lod_blend_factor = 1.0f;
    lod_transitions_this_frame = 0;
    prev_visible_count = 0;
    zero_visible_recovery.zero_visible_consecutive_frames = 0;
    zero_visible_recovery.last_recovery_frame = UINT64_MAX;
    zero_visible_recovery.last_stall_log_frame = UINT64_MAX;
    zero_visible_recovery.recoveries_triggered = 0;
    zero_visible_recovery.stall_detections = 0;
}

void StreamingVisibilityController::clear_visible_state() {
    visible_chunk_indices.clear();
    culling_stats.reset();
}

void StreamingVisibilityController::update_camera_tracking(const Vector3 &camera_pos, float frame_delta_seconds) {
    camera_tracker.update(camera_pos, frame_delta_seconds);
}

void StreamingVisibilityController::load_zero_visible_recovery_config_from_project_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    const StringName recovery_mode_path = "rendering/gaussian_splatting/streaming/zero_visible_recovery_mode";
    if (ps->has_setting(recovery_mode_path)) {
        Variant mode_value = ps->get_setting_with_override(recovery_mode_path);
        if (mode_value.get_type() == Variant::BOOL) {
            zero_visible_recovery.mode = mode_value.operator bool()
                    ? ZeroVisibleRecoveryMode::PERSISTENT
                    : ZeroVisibleRecoveryMode::STARTUP_ONLY;
        } else if (mode_value.get_type() == Variant::INT) {
            const int64_t mode_int = mode_value.operator int64_t();
            zero_visible_recovery.mode = mode_int <= 0
                    ? ZeroVisibleRecoveryMode::STARTUP_ONLY
                    : ZeroVisibleRecoveryMode::PERSISTENT;
        } else if (mode_value.get_type() == Variant::STRING) {
            const String mode_text = String(mode_value).to_lower();
            if (mode_text == "startup_only" || mode_text == "startup-only" || mode_text == "startup") {
                zero_visible_recovery.mode = ZeroVisibleRecoveryMode::STARTUP_ONLY;
            } else if (mode_text == "persistent" || mode_text == "always") {
                zero_visible_recovery.mode = ZeroVisibleRecoveryMode::PERSISTENT;
            }
        }
    }

    zero_visible_recovery.persistent_trigger_frames = MAX(
            gs::settings::get_uint(ps, "rendering/gaussian_splatting/streaming/zero_visible_recovery_trigger_frames",
                    zero_visible_recovery.persistent_trigger_frames),
            uint32_t(1));
    zero_visible_recovery.persistent_cooldown_frames = gs::settings::get_uint(ps,
            "rendering/gaussian_splatting/streaming/zero_visible_recovery_cooldown_frames",
            zero_visible_recovery.persistent_cooldown_frames);
    zero_visible_recovery.stall_log_interval_frames = MAX(
            gs::settings::get_uint(ps, "rendering/gaussian_splatting/streaming/zero_visible_recovery_log_interval_frames",
                    zero_visible_recovery.stall_log_interval_frames),
            uint32_t(1));
}

void StreamingVisibilityController::handle_zero_visible_chunk_recovery(GaussianStreamingSystem &system) {
    const uint32_t total_chunks = culling_stats.total_chunks;
    if (!chunk_frustum_culling_enabled || total_chunks == 0) {
        zero_visible_recovery.zero_visible_consecutive_frames = 0;
        return;
    }

    if (culling_stats.visible_chunks != 0) {
        zero_visible_recovery.zero_visible_consecutive_frames = 0;
        return;
    }

    zero_visible_recovery.zero_visible_consecutive_frames++;

    const uint32_t zero_visible_frames = zero_visible_recovery.zero_visible_consecutive_frames;
    const bool persistent_mode = zero_visible_recovery.mode == ZeroVisibleRecoveryMode::PERSISTENT;
    const bool startup_guard_match = system.budget.loaded_chunks_count == 0 && zero_visible_frames == 1;
    const bool persistent_trigger = persistent_mode &&
            zero_visible_frames >= zero_visible_recovery.persistent_trigger_frames;
    const uint64_t cooldown_frames = zero_visible_recovery.persistent_cooldown_frames;
    const bool cooldown_ready =
            zero_visible_recovery.last_recovery_frame == UINT64_MAX ||
            system.total_frame_count >= zero_visible_recovery.last_recovery_frame + cooldown_frames;
    const bool should_force_visible = startup_guard_match || (persistent_trigger && cooldown_ready);

    const bool should_log_stall = startup_guard_match || persistent_trigger;
    if (should_log_stall &&
            (zero_visible_recovery.last_stall_log_frame == UINT64_MAX ||
                    system.total_frame_count >= zero_visible_recovery.last_stall_log_frame +
                            zero_visible_recovery.stall_log_interval_frames)) {
        zero_visible_recovery.last_stall_log_frame = system.total_frame_count;
        zero_visible_recovery.stall_detections++;
        WARN_PRINT(vformat("[Streaming] Zero-visible chunk stall detected (frame=%d mode=%s zero_visible_frames=%d total_chunks=%d loaded_chunks=%d culling=%s).",
                system.total_frame_count,
                persistent_mode ? "persistent" : "startup_only",
                zero_visible_frames,
                total_chunks,
                system.budget.loaded_chunks_count,
                chunk_frustum_culling_enabled ? "on" : "off"));
    }

    if (!should_force_visible) {
        return;
    }

    for (uint32_t i = 0; i < system.chunks.size(); i++) {
        system.chunks[i].is_visible = true;
    }
    visible_chunk_indices.clear();
    visible_chunk_indices.reserve(system.chunks.size());
    for (uint32_t i = 0; i < system.chunks.size(); i++) {
        visible_chunk_indices.push_back(i);
    }
    culling_stats.visible_chunks = total_chunks;
    culling_stats.frustum_culled_chunks = 0;
    zero_visible_recovery.last_recovery_frame = system.total_frame_count;
    zero_visible_recovery.recoveries_triggered++;

    if (system.debug_logging_enabled) {
        GS_LOG_STREAMING_INFO(vformat("[Streaming] Zero-visible recovery activated (frame=%d mode=%s zero_visible_frames=%d forced_visible_chunks=%d).",
                system.total_frame_count,
                persistent_mode ? "persistent" : "startup_only",
                zero_visible_frames,
                total_chunks));
    }
}

void StreamingVisibilityController::update_chunk_visibility(
        GaussianStreamingSystem &system,
        const Transform3D &camera_transform,
        const Projection &projection) {
    culling_stats.reset();
    culling_stats.total_chunks = system.chunks.size();
    visible_chunk_indices.clear();
    visible_chunk_indices.reserve(culling_stats.total_chunks);

    Vector3 camera_pos = camera_transform.origin;
    Vector<Plane> frustum_planes = projection.get_projection_planes(camera_transform);

    for (uint32_t i = 0; i < system.chunks.size(); i++) {
        system.chunks[i].distance = camera_pos.distance_to(system.chunks[i].center);

        if (chunk_frustum_culling_enabled) {
            AABB padded_bounds = system.chunks[i].bounds;
            if (chunk_radius_multiplier > 1.0f && system.chunks[i].max_radius > 0.0f) {
                float radius_pad = system.chunks[i].max_radius * (chunk_radius_multiplier - 1.0f);
                Vector3 radius_vec(radius_pad, radius_pad, radius_pad);
                padded_bounds.position -= radius_vec;
                padded_bounds.size += radius_vec * 2.0f;
            }
            Vector3 padding_vec = padded_bounds.size * (chunk_frustum_padding - 1.0f) * 0.5f;
            padded_bounds.position -= padding_vec;
            padded_bounds.size += padding_vec * 2.0f;

            system.chunks[i].is_visible = is_chunk_in_frustum(padded_bounds, frustum_planes);

            if (!system.chunks[i].is_visible) {
                culling_stats.frustum_culled_chunks++;
            }
        } else {
            system.chunks[i].is_visible = true;
        }

        if (system.chunks[i].is_visible) {
            culling_stats.visible_chunks++;
            visible_chunk_indices.push_back(i);
        }
        if (system.chunks[i].is_loaded) {
            culling_stats.loaded_chunks++;
        }
    }

    static uint64_t log_counter = 0;
    if (system.debug_logging_enabled && (++log_counter % 300) == 0 && chunk_frustum_culling_enabled) {
        GS_LOG_STREAMING_DEBUG(vformat("[Streaming] Chunk culling: %d total, %d visible, %d culled (%.1f%% reduction)",
                culling_stats.total_chunks,
                culling_stats.visible_chunks,
                culling_stats.frustum_culled_chunks,
                culling_stats.total_chunks > 0 ? (culling_stats.frustum_culled_chunks * 100.0f / culling_stats.total_chunks) : 0.0f));
    }
}

bool StreamingVisibilityController::is_chunk_in_frustum(const AABB &bounds, const Vector<Plane> &frustum_planes) const {
    for (int i = 0; i < frustum_planes.size(); i++) {
        const Plane &plane = frustum_planes[i];
        Vector3 negative_vertex;
        negative_vertex.x = (plane.normal.x >= 0) ? bounds.position.x : (bounds.position.x + bounds.size.x);
        negative_vertex.y = (plane.normal.y >= 0) ? bounds.position.y : (bounds.position.y + bounds.size.y);
        negative_vertex.z = (plane.normal.z >= 0) ? bounds.position.z : (bounds.position.z + bounds.size.z);
        if (plane.distance_to(negative_vertex) > 0) {
            return false;
        }
    }
    return true;
}

void StreamingVisibilityController::update_culling_config_from_project_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    if (ps->has_setting("rendering/gaussian_splatting/streaming/chunk_frustum_culling_enabled")) {
        Variant value = ps->get_setting_with_override("rendering/gaussian_splatting/streaming/chunk_frustum_culling_enabled");
        if (value.get_type() == Variant::BOOL) {
            chunk_frustum_culling_enabled = value.operator bool();
        }
    }

    if (ps->has_setting("rendering/gaussian_splatting/streaming/chunk_frustum_padding")) {
        Variant value = ps->get_setting_with_override("rendering/gaussian_splatting/streaming/chunk_frustum_padding");
        if (value.get_type() == Variant::FLOAT) {
            chunk_frustum_padding = MAX(1.0f, (float)value.operator double());
        } else if (value.get_type() == Variant::INT) {
            chunk_frustum_padding = MAX(1.0f, (float)value.operator int64_t());
        }
    }
}

void StreamingVisibilityController::load_prefetch_config_from_project_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    if (ps->has_setting("rendering/gaussian_splatting/streaming/predictive_prefetch_enabled")) {
        Variant value = ps->get_setting_with_override("rendering/gaussian_splatting/streaming/predictive_prefetch_enabled");
        if (value.get_type() == Variant::BOOL) {
            predictive_prefetch_enabled = value.operator bool();
        }
    }

    if (ps->has_setting("rendering/gaussian_splatting/streaming/prefetch_lookahead_distance")) {
        Variant value = ps->get_setting_with_override("rendering/gaussian_splatting/streaming/prefetch_lookahead_distance");
        if (value.get_type() == Variant::FLOAT) {
            prefetch_lookahead_distance = MAX(0.0f, (float)value.operator double());
        } else if (value.get_type() == Variant::INT) {
            prefetch_lookahead_distance = MAX(0.0f, (float)value.operator int64_t());
        }
    }
}

void StreamingVisibilityController::load_lod_blend_config_from_project_settings() {
    lod_blend_config = LODBlendConfig::load_from_project_settings();
}

float StreamingVisibilityController::calculate_lod_blend_factor(float distance, float lod_distance) const {
    if (!lod_blend_config.blend_enabled) {
        return 1.0f;
    }
    float blend_distance = lod_blend_config.blend_distance;
    if (blend_distance <= 0.0f) {
        return 1.0f;
    }

    float lower_bound = lod_distance - blend_distance;
    float upper_bound = lod_distance + blend_distance;
    if (distance <= lower_bound) {
        return 1.0f;
    } else if (distance >= upper_bound) {
        return 0.0f;
    } else {
        float t = (distance - lower_bound) / (upper_bound - lower_bound);
        return 1.0f - (t * t * (3.0f - 2.0f * t));
    }
}

void StreamingVisibilityController::update_chunk_lod_blend_factors(GaussianStreamingSystem &system, const Vector3 &camera_pos) {
    (void)camera_pos;
    if (!lod_blend_config.blend_enabled) {
        for (uint32_t i = 0; i < system.chunks.size(); i++) {
            system.chunks[i].lod_blend_factor = 1.0f;
        }
        current_lod_blend_factor = 1.0f;
        return;
    }

    float lod_mult = system.budget.vram_regulator.is_valid() ? system.budget.vram_regulator->get_lod_distance_multiplier() : 1.0f;
    float base_lod_distance = GaussianStreamingSystem::STREAMING_LOAD_DISTANCE_BASE / lod_mult;

    float weighted_blend_sum = 0.0f;
    float total_weight = 0.0f;

    for (uint32_t i = 0; i < system.chunks.size(); i++) {
        GaussianStreamingSystem::StreamingChunk &chunk = system.chunks[i];
        float hysteresis = lod_blend_config.hysteresis_zone;
        float effective_distance = chunk.distance;

        if (Math::abs(effective_distance - chunk.previous_distance) > hysteresis) {
            chunk.previous_distance = effective_distance;
        } else {
            effective_distance = chunk.previous_distance;
        }

        chunk.lod_blend_factor = calculate_lod_blend_factor(effective_distance, base_lod_distance);

        if (chunk.is_visible && chunk.is_loaded) {
            float weight = float(chunk.count);
            weighted_blend_sum += chunk.lod_blend_factor * weight;
            total_weight += weight;
        }
    }

    current_lod_blend_factor = total_weight > 0.0f ? (weighted_blend_sum / total_weight) : 1.0f;
}

void StreamingVisibilityController::update_chunk_lod_parameters(GaussianStreamingSystem &system, const Vector3 &camera_pos) {
	    (void)camera_pos;
	    lod_transitions_this_frame = 0;

	    const LODConfig &lod_config = system._get_lod_config();
	    const auto mark_resident_chunk_meta_if_lod_changed = [&](uint32_t chunk_idx,
	                                                             const GaussianStreamingSystem::StreamingChunk &chunk,
                                                             uint32_t prev_effective_count,
                                                             uint32_t prev_lod_level,
                                                             int prev_sh_band_level) {
        if (chunk.effective_count == prev_effective_count &&
                chunk.current_lod_level == prev_lod_level &&
                chunk.sh_band_level == prev_sh_band_level) {
            return;
        }

        const bool resident = chunk.is_loaded && !chunk.upload_pending && chunk.buffer_slot != UINT32_MAX;
        if (resident) {
            system.global_atlas_registry.mark_chunk_meta_dirty(system, chunk_idx);
        }
    };

    if (!lod_config.enabled) {
        for (uint32_t i = 0; i < system.chunks.size(); i++) {
            GaussianStreamingSystem::StreamingChunk &chunk = system.chunks[i];
            const uint32_t prev_effective_count = chunk.effective_count;
            const uint32_t prev_lod_level = chunk.current_lod_level;
            const uint32_t prev_target_lod_level = chunk.target_lod_level;
            const int prev_sh_band_level = chunk.sh_band_level;
            chunk.current_lod_level = 0;
            chunk.target_lod_level = 0;
            chunk.sh_band_level = 3;
            chunk.splat_skip_factor = 1;
            chunk.opacity_multiplier = 1.0f;
            chunk.effective_count = chunk.count;
            if (prev_lod_level != chunk.current_lod_level ||
                    prev_target_lod_level != chunk.target_lod_level) {
                lod_transitions_this_frame++;
            }

            mark_resident_chunk_meta_if_lod_changed(i, chunk, prev_effective_count, prev_lod_level, prev_sh_band_level);
        }
        return;
    }

    for (uint32_t i = 0; i < system.chunks.size(); i++) {
        GaussianStreamingSystem::StreamingChunk &chunk = system.chunks[i];
        const uint32_t prev_effective_count = chunk.effective_count;
        const uint32_t prev_lod_level = chunk.current_lod_level;
        const uint32_t prev_target_lod_level = chunk.target_lod_level;
        const int prev_sh_band_level = chunk.sh_band_level;

        float distance = chunk.distance;
        int lod_level = lod_config.calculate_lod_level(distance);
        if (prev_lod_level != uint32_t(lod_level)) {
            lod_transitions_this_frame++;
        }

        chunk.target_lod_level = lod_level;
        chunk.current_lod_level = lod_level;
        if (prev_lod_level != chunk.current_lod_level ||
                prev_target_lod_level != chunk.target_lod_level) {
            lod_transitions_this_frame++;
        }

        chunk.sh_band_level = lod_config.get_sh_band_for_lod(lod_level);
        chunk.splat_skip_factor = lod_config.get_splat_skip_factor(lod_level);
        chunk.opacity_multiplier = lod_config.get_opacity_multiplier(distance);
        chunk.effective_count = chunk.count / chunk.splat_skip_factor;
        if (chunk.effective_count == 0 && chunk.count > 0) {
            chunk.effective_count = 1;
        }

        mark_resident_chunk_meta_if_lod_changed(i, chunk, prev_effective_count, prev_lod_level, prev_sh_band_level);
    }
}

uint32_t StreamingVisibilityController::prefetch_chunks_at_predicted_position(
        GaussianStreamingSystem &system,
        const Vector3 &predicted_pos,
        uint32_t available_slots,
        uint32_t load_budget,
        uint32_t max_scan_budget) {
    uint32_t max_prefetch = get_prefetch_limit(system, available_slots, load_budget);
    if (max_prefetch == 0) {
        return 0;
    }

    LocalVector<uint32_t> candidates;
    collect_prefetch_candidates(system, predicted_pos, max_prefetch, max_scan_budget, candidates);
    return schedule_prefetch_loads(system, predicted_pos, candidates, available_slots, load_budget);
}

uint32_t StreamingVisibilityController::get_prefetch_limit(
        GaussianStreamingSystem &system, uint32_t available_slots, uint32_t load_budget) const {
    if (camera_tracker.velocity.length_squared() < 0.01f) {
        return 0;
    }
    if (available_slots == 0 || load_budget == 0) {
        return 0;
    }
    if (system.budget.vram_regulator.is_valid() &&
            !system.budget.vram_regulator->can_load_more_chunks(system.budget.loaded_chunks_count)) {
        return 0;
    }
    if (system.scheduler.max_prefetch_loads_per_frame == 0) {
        return 0;
    }
    return MIN(system.scheduler.max_prefetch_loads_per_frame, MIN(available_slots, load_budget));
}

void StreamingVisibilityController::collect_prefetch_candidates(
        GaussianStreamingSystem &system, const Vector3 &predicted_pos,
        uint32_t max_prefetch, uint32_t max_scan_budget, LocalVector<uint32_t> &out_candidates) const {
    struct ChunkCandidate {
        uint32_t idx = UINT32_MAX;
        float distance_sq = FLT_MAX;
    };

    LocalVector<ChunkCandidate> closest;
    closest.resize(max_prefetch);
    for (uint32_t i = 0; i < max_prefetch; i++) {
        closest[i].idx = UINT32_MAX;
        closest[i].distance_sq = FLT_MAX;
    }

    float prefetch_threshold_sq = prefetch_lookahead_distance * prefetch_lookahead_distance * 2.25f;

    out_candidates.clear();
    out_candidates.resize(max_prefetch);
    for (uint32_t i = 0; i < max_prefetch; i++) {
        out_candidates[i] = UINT32_MAX;
    }

    const uint32_t chunk_count = system.chunks.size();
    if (chunk_count == 0 || max_prefetch == 0) {
        return;
    }

    uint32_t scan_budget = 0;
    if (system.scheduler.max_prefetch_chunk_scan_per_frame == 0) {
        scan_budget = chunk_count;
    } else {
        const uint32_t remaining_scan_budget = system.scheduler.prefetch_scan_budget_remaining_this_frame;
        scan_budget = MIN(chunk_count, remaining_scan_budget);
    }
    if (max_scan_budget != UINT32_MAX) {
        scan_budget = MIN(scan_budget, max_scan_budget);
    }
    uint32_t pack_queue_depth = 0;
    uint32_t upload_queue_depth = 0;
    system.upload_pipeline.get_pending_queue_depths_cached(pack_queue_depth, upload_queue_depth);
    const uint32_t sync_fallback_queue_depth = system._get_sync_fallback_queue_depth();
    const uint32_t observed_throttle_queue_depth =
            MAX(sync_fallback_queue_depth, MAX(pack_queue_depth, upload_queue_depth));
    const bool can_async_pack = system.upload_pipeline.async_pack_enabled && system.upload_pipeline.pack_thread_running.load();
    uint32_t enqueue_headroom = UINT32_MAX;
    if (can_async_pack) {
        enqueue_headroom = _compute_visibility_async_enqueue_headroom(
                system.upload_pipeline.queued_chunk_loads_this_frame,
                system.upload_pipeline.max_chunk_loads_per_frame,
                system.upload_pipeline.pack_jobs_in_flight.load(std::memory_order_relaxed),
                system.upload_pipeline.max_pack_jobs_in_flight);
    }
    StreamingQueuePressureController::ScanBudgetInput throttle_input;
    throttle_input.base_scan_budget = scan_budget;
    throttle_input.throttle_enabled = system.scheduler.queue_pressure_candidate_scan_throttle_enabled;
    throttle_input.throttle_min_queue_depth = system.scheduler.queue_pressure_candidate_scan_throttle_min_queue_depth;
    throttle_input.observed_queue_depth = observed_throttle_queue_depth;
    throttle_input.throttle_scan_cap = system.scheduler.queue_pressure_candidate_scan_throttle_prefetch_scan_cap;
    throttle_input.scanned_this_frame = system.scheduler.last_prefetch_scan_count;
    throttle_input.enqueue_headroom = enqueue_headroom;
    const StreamingQueuePressureController::ScanBudgetResult throttle_result =
            StreamingQueuePressureController::compute_candidate_scan_budget(throttle_input);
    scan_budget = throttle_result.scan_budget;
    system.scheduler.queue_pressure_candidate_scan_throttle_active =
            system.scheduler.queue_pressure_candidate_scan_throttle_active || throttle_result.throttle_active;
    system.scheduler.queue_pressure_candidate_scan_throttle_queue_depth =
            MAX(system.scheduler.queue_pressure_candidate_scan_throttle_queue_depth,
                    throttle_result.effective_queue_depth);
    system.scheduler.last_prefetch_scan_budget_effective =
            MAX(system.scheduler.last_prefetch_scan_budget_effective, scan_budget);
    if (scan_budget == 0) {
        return;
    }
    if (system.scheduler.prefetch_scan_cursor >= chunk_count) {
        system.scheduler.prefetch_scan_cursor = 0;
    }

    uint32_t farthest_idx = 0;
    uint32_t valid_count = 0;

    for (uint32_t scanned = 0; scanned < scan_budget; scanned++) {
        const uint32_t chunk_idx = (system.scheduler.prefetch_scan_cursor + scanned) % chunk_count;
        const auto &chunk = system.chunks[chunk_idx];
        if (chunk.is_loaded) {
            continue;
        }
        if (chunk.upload_pending) {
            system.scheduler.last_prefetch_upload_pending_skip_count++;
            continue;
        }

        float dist_sq = (predicted_pos - chunk.center).length_squared();
        if (dist_sq >= prefetch_threshold_sq) {
            continue;
        }

        if (valid_count < max_prefetch) {
            closest[valid_count].idx = chunk_idx;
            closest[valid_count].distance_sq = dist_sq;
            if (dist_sq > closest[farthest_idx].distance_sq) {
                farthest_idx = valid_count;
            }
            valid_count++;
        } else if (dist_sq < closest[farthest_idx].distance_sq) {
            closest[farthest_idx].idx = chunk_idx;
            closest[farthest_idx].distance_sq = dist_sq;
            for (uint32_t j = 0; j < max_prefetch; j++) {
                if (closest[j].distance_sq > closest[farthest_idx].distance_sq) {
                    farthest_idx = j;
                }
            }
        }
    }
    system.scheduler.prefetch_scan_cursor = (system.scheduler.prefetch_scan_cursor + scan_budget) % chunk_count;
    if (system.scheduler.max_prefetch_chunk_scan_per_frame > 0) {
        system.scheduler.prefetch_scan_budget_remaining_this_frame =
                scan_budget >= system.scheduler.prefetch_scan_budget_remaining_this_frame
                ? 0
                : (system.scheduler.prefetch_scan_budget_remaining_this_frame - scan_budget);
    }
    system.scheduler.last_prefetch_scan_count += scan_budget;
    system.scheduler.last_prefetch_scan_budget_effective =
            MAX(system.scheduler.last_prefetch_scan_budget_effective, scan_budget);
    uint32_t candidate_count = 0;
    for (uint32_t i = 0; i < max_prefetch; i++) {
        out_candidates[i] = closest[i].idx;
        if (closest[i].idx != UINT32_MAX) {
            candidate_count++;
        }
    }
    system.scheduler.last_prefetch_candidate_count += candidate_count;
}

uint32_t StreamingVisibilityController::schedule_prefetch_loads(
        GaussianStreamingSystem &system, const Vector3 &predicted_pos,
        const LocalVector<uint32_t> &candidates, uint32_t available_slots, uint32_t load_budget) {
    uint32_t remaining_slots = available_slots;
    uint32_t remaining_budget = load_budget;
    uint32_t queued_count = 0;
    const bool can_async_pack = system.upload_pipeline.async_pack_enabled && system.upload_pipeline.pack_thread_running.load();

    for (uint32_t i = 0; i < candidates.size(); i++) {
        if (candidates[i] == UINT32_MAX) {
            break;
        }

        if (remaining_slots == 0 || remaining_budget == 0) {
            break;
        }
        if (can_async_pack &&
                _compute_visibility_async_enqueue_headroom(
                        system.upload_pipeline.queued_chunk_loads_this_frame,
                        system.upload_pipeline.max_chunk_loads_per_frame,
                        system.upload_pipeline.pack_jobs_in_flight.load(std::memory_order_relaxed),
                        system.upload_pipeline.max_pack_jobs_in_flight) == 0) {
            system.scheduler.last_prefetch_enqueue_headroom_stall_count++;
            break;
        }

        if (system.budget.vram_regulator.is_valid() &&
                !system.budget.vram_regulator->can_load_more_chunks(system.budget.loaded_chunks_count)) {
            break;
        }

        const bool queued = system._enqueue_chunk_load_request(GaussianStreamingSystem::PRIMARY_ASSET_ID, candidates[i], can_async_pack);
        if (!queued) {
            if (can_async_pack &&
                    _compute_visibility_async_enqueue_headroom(
                            system.upload_pipeline.queued_chunk_loads_this_frame,
                            system.upload_pipeline.max_chunk_loads_per_frame,
                            system.upload_pipeline.pack_jobs_in_flight.load(std::memory_order_relaxed),
                            system.upload_pipeline.max_pack_jobs_in_flight) == 0) {
                system.scheduler.last_prefetch_enqueue_headroom_stall_count++;
                break;
            }
            continue;
        }

        queued_count++;
        remaining_slots = remaining_slots > 0 ? (remaining_slots - 1) : 0;
        if (remaining_budget != UINT32_MAX) {
            remaining_budget = remaining_budget > 0 ? (remaining_budget - 1) : 0;
        }

        GS_LOG_STREAMING_DEBUG(vformat("[Streaming] Prefetched chunk %d at predicted position (%.1f, %.1f, %.1f)",
                candidates[i], predicted_pos.x, predicted_pos.y, predicted_pos.z));
    }

    return queued_count;
}

float StreamingVisibilityController::get_visible_count_change_ratio() const {
    const uint32_t current = visible_chunk_indices.size();
    const uint32_t prev = prev_visible_count;
    if (prev == 0) {
        return current > 0 ? 1.0f : 0.0f;
    }
    return Math::abs(float(current) - float(prev)) / float(prev);
}

float StreamingVisibilityController::get_effective_count_change_ratio(uint32_t visible_chunks_evicted_this_frame) const {
    float vis_ratio = get_visible_count_change_ratio();
    const uint32_t visible = visible_chunk_indices.size();
    if (visible == 0) {
        return vis_ratio;
    }
    float evict_ratio = float(visible_chunks_evicted_this_frame) / float(visible);
    return MAX(vis_ratio, evict_ratio);
}

void StreamingVisibilityController::CameraVelocityTracker::update(const Vector3 &current_pos, float delta_time) {
    if (!has_previous_position) {
        last_position = current_pos;
        velocity = Vector3();
        has_previous_position = true;
        return;
    }

    if (delta_time > 0.0001f) {
        velocity = (current_pos - last_position) / delta_time;
    }
    last_position = current_pos;
}

Vector3 StreamingVisibilityController::CameraVelocityTracker::predict_position(const Vector3 &current_pos, float lookahead_distance) const {
    if (!has_previous_position || velocity.length_squared() < 0.0001f) {
        return current_pos;
    }

    float speed = velocity.length();
    if (speed < 0.0001f) {
        return current_pos;
    }

    Vector3 direction = velocity.normalized();
    return current_pos + direction * lookahead_distance;
}
