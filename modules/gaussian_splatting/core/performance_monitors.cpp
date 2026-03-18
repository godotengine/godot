#include "performance_monitors.h"
#include "../renderer/tile_renderer.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../renderer/gaussian_gpu_layout.h"
#include "core/math/math_funcs.h"
#include "main/performance.h"

GaussianSplattingPerformanceMonitors *GaussianSplattingPerformanceMonitors::singleton = nullptr;

static bool _renderer_has_streaming_data(const GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        return false;
    }
    const GaussianSplatRenderer::SceneState &scene_state = p_renderer->get_scene_state();
    if (scene_state.gaussian_data.is_valid() && scene_state.gaussian_data->get_count() > 0) {
        return true;
    }
    return scene_state.active_asset.is_valid();
}

static float _sanitize_ms(float p_value) {
    if (p_value < 0.0f) {
        return 0.0f;
    }
    if (Math::is_nan(p_value) || Math::is_inf(p_value)) {
        return 0.0f;
    }
    return p_value;
}

static float _prefer_direct_or_fallback(float p_direct_ms, float p_fallback_ms) {
    const float direct = _sanitize_ms(p_direct_ms);
    if (direct > 0.0f) {
        return direct;
    }
    return _sanitize_ms(p_fallback_ms);
}

GaussianSplattingPerformanceMonitors::GaussianSplattingPerformanceMonitors() {
}

GaussianSplattingPerformanceMonitors::~GaussianSplattingPerformanceMonitors() {
    cleanup_monitors();
}

GaussianSplattingPerformanceMonitors *GaussianSplattingPerformanceMonitors::get_singleton() {
    return singleton;
}

void GaussianSplattingPerformanceMonitors::create_singleton() {
    if (!singleton) {
        singleton = memnew(GaussianSplattingPerformanceMonitors);
        singleton->initialize_monitors();
    }
}

void GaussianSplattingPerformanceMonitors::destroy_singleton() {
    if (singleton) {
        memdelete(singleton);
        singleton = nullptr;
    }
}

void GaussianSplattingPerformanceMonitors::register_renderer(TileRenderer *p_renderer) {
    ERR_FAIL_NULL(p_renderer);
    if (!monitors_registered) {
        initialize_monitors();
    }

    // Add to tracking list if not already registered
    if (!registered_renderers.has(p_renderer)) {
        registered_renderers.push_back(p_renderer);
    }

    // Always prefer the most recently registered renderer so monitor values
    // follow the currently active viewport/session.
    active_renderer = p_renderer;
}

void GaussianSplattingPerformanceMonitors::unregister_renderer(TileRenderer *p_renderer) {
    // Remove from tracking list
    registered_renderers.erase(p_renderer);

    // If this was the active renderer, switch to another if available
    if (active_renderer == p_renderer) {
        active_renderer = registered_renderers.size() > 0 ? registered_renderers[0] : nullptr;
    }
}

void GaussianSplattingPerformanceMonitors::register_splat_renderer(GaussianSplatRenderer *p_renderer) {
    ERR_FAIL_NULL(p_renderer);
    if (!monitors_registered) {
        initialize_monitors();
    }

    // Add to tracking list if not already registered
    if (!registered_splat_renderers.has(p_renderer)) {
        registered_splat_renderers.push_back(p_renderer);
    }

    // Always prefer the most recently registered renderer so monitor values
    // follow the currently active viewport/session.
    active_splat_renderer = p_renderer;
}

void GaussianSplattingPerformanceMonitors::unregister_splat_renderer(GaussianSplatRenderer *p_renderer) {
    // Remove from tracking list
    registered_splat_renderers.erase(p_renderer);

    // If this was the active renderer, switch to another if available
    if (active_splat_renderer == p_renderer) {
        active_splat_renderer = registered_splat_renderers.size() > 0 ? registered_splat_renderers[0] : nullptr;
    }
}

GaussianSplatRenderer *GaussianSplattingPerformanceMonitors::_get_active_splat_renderer(bool p_require_streaming) const {
    if (active_splat_renderer) {
        if (!p_require_streaming) {
            return active_splat_renderer;
        }
        if (active_splat_renderer->get_streaming_state().current_streaming_system.is_valid() &&
                _renderer_has_streaming_data(active_splat_renderer)) {
            return active_splat_renderer;
        }
    }

    if (!p_require_streaming) {
        return registered_splat_renderers.size() > 0 ? registered_splat_renderers[0] : nullptr;
    }

    GaussianSplatRenderer *fallback = nullptr;
    for (GaussianSplatRenderer *renderer : registered_splat_renderers) {
        if (!renderer || !renderer->get_streaming_state().current_streaming_system.is_valid()) {
            continue;
        }
        if (_renderer_has_streaming_data(renderer)) {
            return renderer;
        }
        if (!fallback) {
            fallback = renderer;
        }
    }

    return fallback;
}

bool GaussianSplattingPerformanceMonitors::_is_telemetry_active() const {
    return active_renderer != nullptr || active_splat_renderer != nullptr ||
            !registered_renderers.is_empty() || !registered_splat_renderers.is_empty();
}

Dictionary GaussianSplattingPerformanceMonitors::_get_streaming_analytics() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) {
        return Dictionary();
    }
    if (!renderer->get_streaming_state().current_streaming_system.is_valid()) {
        return Dictionary();
    }
    return renderer->get_streaming_state().current_streaming_system->get_streaming_analytics();
}

void GaussianSplattingPerformanceMonitors::_register_monitor_definitions(Performance *p_perf) {
    struct MonitorDefinition {
        const char *name;
        Callable callable;
    };

    const MonitorDefinition monitor_definitions[] = {
        // GPU Timing Monitors
        { "gaussian_splatting/gpu_time_frame_ms",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_gpu_frame_time_ms) },
        { "gaussian_splatting/gpu_time_cull_ms",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_gpu_cull_time_ms) },
        { "gaussian_splatting/gpu_time_sort_ms",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_gpu_sort_time_ms) },
        { "gaussian_splatting/gpu_time_binning_ms",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_gpu_binning_time_ms) },
        { "gaussian_splatting/gpu_time_prefix_ms",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_gpu_prefix_time_ms) },
        { "gaussian_splatting/gpu_time_raster_ms",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_gpu_raster_time_ms) },
        { "gaussian_splatting/gpu_time_resolve_ms",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_gpu_resolve_time_ms) },

        // CPU Timing Monitors
        { "gaussian_splatting/telemetry_active",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_telemetry_active) },
        { "gaussian_splatting/cpu_setup_time_ms",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_cpu_setup_time_ms) },

        // Projection Statistics Monitors
        { "gaussian_splatting/visible_splats",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_visible_splat_count) },
        { "gaussian_splatting/total_processed",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_total_processed) },
        { "gaussian_splatting/projection_success_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_projection_success_count) },
        { "gaussian_splatting/projection_success_rate_pct",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_projection_success_rate_pct) },

        // Rejection Statistics Monitors
        { "gaussian_splatting/clip_reject_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_clip_reject_count) },
        { "gaussian_splatting/radius_reject_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_radius_reject_count) },
        { "gaussian_splatting/viewport_reject_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_viewport_reject_count) },

        // Quality Monitors
        { "gaussian_splatting/extreme_aspect_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_extreme_aspect_count) },
        { "gaussian_splatting/index_mismatch_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_index_mismatch_count) },

        // SH Cache Monitors
        { "gaussian_splatting/sh_cache_hits",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_sh_cache_hits) },
        { "gaussian_splatting/sh_cache_updates",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_sh_cache_updates) },
        { "gaussian_splatting/sh_cache_hit_rate_pct",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_sh_cache_hit_rate_pct) },

        // Overflow Statistics Monitors
        { "gaussian_splatting/overflow_tile_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_overflow_tile_count) },
        { "gaussian_splatting/clamped_records",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_clamped_records) },
        { "gaussian_splatting/aggregated_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_aggregated_count) },

        // Rendering Configuration Monitors
        { "gaussian_splatting/tile_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_tile_count) },

        // VRAM Budget Regulation Monitors (Phase 1)
        { "gaussian_splatting/vram_current_usage_mb",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_vram_current_usage_mb) },
        { "gaussian_splatting/vram_budget_mb",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_vram_budget_mb) },
        { "gaussian_splatting/vram_usage_percent",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_vram_usage_percent) },
        { "gaussian_splatting/vram_current_max_chunks",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_vram_current_max_chunks) },
        { "gaussian_splatting/vram_loaded_chunks",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_vram_loaded_chunks) },
        { "gaussian_splatting/vram_evicted_this_frame",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_vram_evicted_this_frame) },
        { "gaussian_splatting/vram_loaded_this_frame",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_vram_loaded_this_frame) },
        { "gaussian_splatting/vram_budget_warning_active",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_vram_budget_warning_active) },
        { "gaussian_splatting/vram_regulation_adjustments",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_vram_regulation_adjustments) },
        { "gaussian_splatting/vram_thrashing_events",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_vram_thrashing_events) },

        // Streaming Core Monitors (Phase 1)
        { "gaussian_splatting/streaming_monitor_ready",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_monitor_ready) },
        { "gaussian_splatting/streaming_runtime_capacity_zero",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_runtime_capacity_zero) },
        { "gaussian_splatting/streaming_runtime_buffer_invalid",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_runtime_buffer_invalid) },
        { "gaussian_splatting/streaming_invalid_camera_inputs",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_invalid_camera_inputs) },
        { "gaussian_splatting/streaming_total_chunks",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_total_chunks) },
        { "gaussian_splatting/streaming_visible_chunks",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_visible_chunks) },
        { "gaussian_splatting/streaming_loaded_chunks",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_loaded_chunks) },
        { "gaussian_splatting/streaming_frustum_culled_chunks",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_frustum_culled_chunks) },
        { "gaussian_splatting/streaming_vram_usage_mb",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_vram_usage_mb) },
        { "gaussian_splatting/streaming_chunks_loaded_this_frame",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_chunks_loaded_this_frame) },
        { "gaussian_splatting/streaming_chunks_evicted_this_frame",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_chunks_evicted_this_frame) },
        { "gaussian_splatting/streaming_visible_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_visible_count) },
        { "gaussian_splatting/streaming_buffer_capacity_splats",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_buffer_capacity_splats) },
        { "gaussian_splatting/streaming_effective_splat_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_effective_splat_count) },
        { "gaussian_splatting/streaming_visible_change_ratio",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_visible_change_ratio) },
        { "gaussian_splatting/streaming_lod_blend_factor",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_lod_blend_factor) },
        { "gaussian_splatting/streaming_sh_band_level",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_sh_band_level) },
        { "gaussian_splatting/streaming_bytes_uploaded_mb",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_bytes_uploaded_mb) },
        { "gaussian_splatting/streaming_buffer_switches",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_buffer_switches) },
        { "gaussian_splatting/streaming_effective_upload_cap_mb_per_frame",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_effective_upload_cap_mb_per_frame) },
        { "gaussian_splatting/streaming_effective_upload_cap_mb_per_slice",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_effective_upload_cap_mb_per_slice) },
        { "gaussian_splatting/streaming_effective_upload_cap_mb_per_second",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_effective_upload_cap_mb_per_second) },
        { "gaussian_splatting/streaming_effective_vram_budget_mb",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_effective_vram_budget_mb) },
        { "gaussian_splatting/streaming_effective_vram_max_chunks",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_effective_vram_max_chunks) },
        { "gaussian_splatting/streaming_upload_frame_cap_hit",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_upload_frame_cap_hit) },
        { "gaussian_splatting/streaming_upload_bandwidth_cap_hit",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_upload_bandwidth_cap_hit) },
        { "gaussian_splatting/streaming_chunk_load_cap_hit",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_chunk_load_cap_hit) },
        { "gaussian_splatting/streaming_vram_chunk_cap_hit",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_vram_chunk_cap_hit) },
        { "gaussian_splatting/streaming_queue_pressure_active",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_streaming_queue_pressure_active) },

        // LOD System Monitors (Phase 2)
        { "gaussian_splatting/lod_current_level",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_current_level) },
        { "gaussian_splatting/lod_distance_multiplier",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_distance_multiplier) },
        { "gaussian_splatting/lod_target_distance",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_target_distance) },
        { "gaussian_splatting/lod_hysteresis_zone",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_hysteresis_zone) },
        { "gaussian_splatting/lod_blend_distance",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_blend_distance) },
        { "gaussian_splatting/lod_transitions_this_frame",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_transitions_this_frame) },
        { "gaussian_splatting/lod_splat_skip_factor",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_splat_skip_factor) },
        { "gaussian_splatting/lod_opacity_multiplier",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_opacity_multiplier) },
        { "gaussian_splatting/lod_effective_count_after_skip",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_effective_count_after_skip) },
        { "gaussian_splatting/lod_chunk_blend_factors_avg",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_chunk_blend_factors_avg) },
        { "gaussian_splatting/lod_chunks_in_transition",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_chunks_in_transition) },
        { "gaussian_splatting/lod_quality_degradation_active",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_quality_degradation_active) },

        // GPU Memory Stream Monitors (Phase 2)
        { "gaussian_splatting/memory_stream_total_bytes_uploaded_mb",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_memory_stream_total_bytes_uploaded_mb) },
        { "gaussian_splatting/memory_stream_total_bytes_downloaded_mb",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_memory_stream_total_bytes_downloaded_mb) },
        { "gaussian_splatting/memory_stream_buffer_switches",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_memory_stream_buffer_switches) },
        { "gaussian_splatting/memory_stream_stalls",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_memory_stream_stalls) },
        { "gaussian_splatting/memory_stream_stall_percent",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_memory_stream_stall_percent) },
        { "gaussian_splatting/memory_stream_pool_hits",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_memory_stream_pool_hits) },
        { "gaussian_splatting/memory_stream_pool_misses",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_memory_stream_pool_misses) },
        { "gaussian_splatting/memory_stream_pool_hit_rate_pct",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_memory_stream_pool_hit_rate_pct) },
        { "gaussian_splatting/memory_stream_peak_memory_mb",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_memory_stream_peak_memory_mb) },
        { "gaussian_splatting/memory_stream_defrag_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_memory_stream_defrag_count) },

        // Chunk Management Monitors (Phase 3)
        { "gaussian_splatting/chunk_prefetch_hits",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_chunk_prefetch_hits) },
        { "gaussian_splatting/chunk_prefetch_misses",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_chunk_prefetch_misses) },
        { "gaussian_splatting/chunk_prefetch_efficiency_pct",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_chunk_prefetch_efficiency_pct) },
        { "gaussian_splatting/chunk_camera_velocity",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_chunk_camera_velocity) },
        { "gaussian_splatting/chunk_average_load_time_ms",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_chunk_average_load_time_ms) },
        { "gaussian_splatting/chunk_upload_queue_depth",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_chunk_upload_queue_depth) },
        { "gaussian_splatting/chunk_pack_jobs_in_flight",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_chunk_pack_jobs_in_flight) },
        { "gaussian_splatting/chunk_total_capacity_mb",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_chunk_total_capacity_mb) },

        // Pack/Upload Timing Monitors (Phase 4.5)
        { "gaussian_splatting/pack_avg_time_ms",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_pack_avg_time_ms) },
        { "gaussian_splatting/pack_max_time_ms",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_pack_max_time_ms) },
        { "gaussian_splatting/pack_jobs_completed",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_pack_jobs_completed) },
        { "gaussian_splatting/upload_mb_this_frame",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_upload_mb_this_frame) },
        { "gaussian_splatting/upload_chunks_this_frame",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_upload_chunks_this_frame) },

        // Advanced LOD Analytics Monitors (Phase 4)
        { "gaussian_splatting/lod_min_chunk_distance",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_min_chunk_distance) },
        { "gaussian_splatting/lod_max_chunk_distance",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_max_chunk_distance) },
        { "gaussian_splatting/lod_avg_chunk_distance",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_avg_chunk_distance) },
        { "gaussian_splatting/lod_reduction_ratio_pct",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_reduction_ratio_pct) },
        { "gaussian_splatting/lod_level_0_chunk_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_level_0_chunk_count) },
        { "gaussian_splatting/lod_sh_band_3_chunk_count",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_lod_sh_band_3_chunk_count) },

        // Compression Analytics Monitors (Phase 5)
        { "gaussian_splatting/sh_compression_raw_mb",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_sh_compression_raw_mb) },
        { "gaussian_splatting/sh_compression_compressed_mb",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_sh_compression_compressed_mb) },
        { "gaussian_splatting/sh_compression_ratio_pct",
                callable_mp(this, &GaussianSplattingPerformanceMonitors::_get_sh_compression_ratio_pct) },
    };

    registered_monitor_ids.clear();
    const Vector<Variant> no_args;
    for (const MonitorDefinition &def : monitor_definitions) {
        const StringName monitor_id = StringName(def.name);
        if (p_perf->has_custom_monitor(monitor_id)) {
            p_perf->remove_custom_monitor(monitor_id);
        }
        p_perf->add_custom_monitor(monitor_id, def.callable, no_args);
        registered_monitor_ids.push_back(monitor_id);
    }
}

void GaussianSplattingPerformanceMonitors::initialize_monitors() {
    if (monitors_registered) {
        return;
    }

    Performance *perf = Performance::get_singleton();
    if (!perf) {
        return;
    }
    _register_monitor_definitions(perf);

    monitors_registered = true;
}

void GaussianSplattingPerformanceMonitors::cleanup_monitors() {
    Performance *perf = Performance::get_singleton();
    if (perf) {
        for (const StringName &monitor_id : registered_monitor_ids) {
            if (perf->has_custom_monitor(monitor_id)) {
                perf->remove_custom_monitor(monitor_id);
            }
        }
    }

    registered_monitor_ids.clear();
    monitors_registered = false;
    active_renderer = nullptr;
    active_splat_renderer = nullptr;
    registered_renderers.clear();
    registered_splat_renderers.clear();
}

// ============================================================================
// GPU Timing Monitor Getters
// ============================================================================

float GaussianSplattingPerformanceMonitors::_get_gpu_frame_time_ms() const {
    const float direct = active_renderer ? active_renderer->get_last_gpu_frame_time_ms() : 0.0f;
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(false);
    const float fallback = renderer ? renderer->get_performance_state().metrics.gpu_frame_time_ms : 0.0f;
    return _prefer_direct_or_fallback(direct, fallback);
}

float GaussianSplattingPerformanceMonitors::_get_gpu_cull_time_ms() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(false);
    if (!renderer) {
        return 0.0f;
    }

    const GaussianSplatRenderer::DebugState &debug_state = renderer->get_debug_state();
    if (debug_state.last_stage_metrics_valid) {
        return _sanitize_ms(debug_state.last_stage_metrics.cull.cull_time_ms);
    }

    return _sanitize_ms(renderer->get_performance_state().metrics.culling_time_ms);
}

float GaussianSplattingPerformanceMonitors::_get_gpu_sort_time_ms() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(false);
    if (!renderer) {
        return 0.0f;
    }

    const GaussianSplatRenderer::DebugState &debug_state = renderer->get_debug_state();
    if (debug_state.last_stage_metrics_valid) {
        return _sanitize_ms(debug_state.last_stage_metrics.sort.sort_time_ms);
    }

    return _sanitize_ms(renderer->get_frame_state().sort_time_ms);
}

float GaussianSplattingPerformanceMonitors::_get_gpu_binning_time_ms() const {
    const float direct = active_renderer ? active_renderer->get_last_gpu_binning_time_ms() : 0.0f;
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(false);
    const float fallback = renderer ? renderer->get_performance_state().metrics.gpu_tile_binning_time_ms : 0.0f;
    return _prefer_direct_or_fallback(direct, fallback);
}

float GaussianSplattingPerformanceMonitors::_get_gpu_prefix_time_ms() const {
    const float direct = active_renderer ? active_renderer->get_last_gpu_prefix_time_ms() : 0.0f;
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(false);
    const float fallback = renderer ? renderer->get_performance_state().metrics.gpu_tile_prefix_time_ms : 0.0f;
    return _prefer_direct_or_fallback(direct, fallback);
}

float GaussianSplattingPerformanceMonitors::_get_gpu_raster_time_ms() const {
    const float direct = active_renderer ? active_renderer->get_last_gpu_raster_time_ms() : 0.0f;
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(false);
    const float fallback = renderer ? renderer->get_performance_state().metrics.gpu_tile_raster_time_ms : 0.0f;
    return _prefer_direct_or_fallback(direct, fallback);
}

float GaussianSplattingPerformanceMonitors::_get_gpu_resolve_time_ms() const {
    const float direct = active_renderer ? active_renderer->get_last_gpu_resolve_time_ms() : 0.0f;
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(false);
    const float fallback = renderer ? renderer->get_performance_state().metrics.gpu_tile_resolve_time_ms : 0.0f;
    return _prefer_direct_or_fallback(direct, fallback);
}

// ============================================================================
// CPU Timing Monitor Getters
// ============================================================================

float GaussianSplattingPerformanceMonitors::_get_cpu_setup_time_ms() const {
    return active_renderer ? active_renderer->get_last_setup_cpu_ms() : 0.0f;
}

int GaussianSplattingPerformanceMonitors::_get_telemetry_active() const {
    return _is_telemetry_active() ? 1 : 0;
}

String GaussianSplattingPerformanceMonitors::_get_route_uid() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(false);
    if (!renderer) {
        return String();
    }
    return renderer->get_debug_state().route_uid;
}

String GaussianSplattingPerformanceMonitors::_get_sort_route_uid() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(false);
    if (!renderer) {
        return String();
    }
    return renderer->get_debug_state().sort_route_uid;
}

// ============================================================================
// Projection Statistics Monitor Getters
// ============================================================================

int GaussianSplattingPerformanceMonitors::_get_visible_splat_count() const {
    // Try TileRenderer first (GPU counter readback from binning pass)
    if (active_renderer) {
        int count = active_renderer->get_visible_splat_count();
        if (count > 0) {
            return count;
        }
    }
    // Fall back to GaussianSplatRenderer (streaming/sorting pipeline count)
    // This is essential for streaming mode where TileRenderer counters may not update
    if (active_splat_renderer) {
        return static_cast<int>(active_splat_renderer->get_visible_splat_count());
    }
    return 0;
}

int GaussianSplattingPerformanceMonitors::_get_total_processed() const {
    // Try TileRenderer first (GPU counter readback from binning pass)
    if (active_renderer) {
        int count = active_renderer->get_total_processed();
        if (count > 0) {
            return count;
        }
    }
    // Fall back to GaussianSplatRenderer frame-scoped metrics.
    // Do not use asset cardinality here because it is not a per-frame processed workload.
    if (active_splat_renderer) {
        const GaussianSplatRenderer::PerformanceMetrics &metrics = active_splat_renderer->get_performance_state().metrics;
        uint32_t total_processed = metrics.visible_after_culling;
        if (metrics.rendered_splat_count > total_processed) {
            total_processed = metrics.rendered_splat_count;
        }
        if (total_processed > 0u) {
            return static_cast<int>(total_processed);
        }
        const uint32_t visible_count = active_splat_renderer->get_visible_splat_count();
        if (visible_count > 0u) {
            return static_cast<int>(visible_count);
        }
    }
    return 0;
}

int GaussianSplattingPerformanceMonitors::_get_projection_success_count() const {
    // Try TileRenderer first (GPU counter readback from binning pass)
    if (active_renderer) {
        int count = active_renderer->get_projection_success_count();
        if (count > 0) {
            return count;
        }
    }
    // Fall back to GaussianSplatRenderer visible count (projection success = visible after projection)
    if (active_splat_renderer) {
        const GaussianSplatRenderer::PerformanceMetrics &metrics = active_splat_renderer->get_performance_state().metrics;
        if (metrics.rendered_splat_count > 0u) {
            return static_cast<int>(metrics.rendered_splat_count);
        }
        return static_cast<int>(active_splat_renderer->get_visible_splat_count());
    }
    return 0;
}

float GaussianSplattingPerformanceMonitors::_get_projection_success_rate_pct() const {
    if (active_renderer) {
        const int total_processed = active_renderer->get_total_processed();
        if (total_processed > 0) {
            const int success_count = active_renderer->get_projection_success_count();
            const float ratio = float(success_count) / float(total_processed);
            return CLAMP(ratio * 100.0f, 0.0f, 100.0f);
        }

        const float direct_rate = active_renderer->get_projection_success_rate_pct();
        if (direct_rate > 0.0f) {
            return direct_rate;
        }
    }

    if (active_splat_renderer) {
        const GaussianSplatRenderer::PerformanceMetrics &metrics = active_splat_renderer->get_performance_state().metrics;
        uint32_t success_count = metrics.rendered_splat_count;
        if (success_count == 0u) {
            success_count = active_splat_renderer->get_visible_splat_count();
        }

        uint32_t total_processed = metrics.visible_after_culling;
        if (metrics.rendered_splat_count > total_processed) {
            total_processed = metrics.rendered_splat_count;
        }

        if (total_processed > 0u) {
            const float ratio = float(success_count) / float(total_processed);
            return CLAMP(ratio * 100.0f, 0.0f, 100.0f);
        }
    }

    return 0.0f;
}

int GaussianSplattingPerformanceMonitors::_get_clip_reject_count() const {
    return active_renderer ? active_renderer->get_clip_bounds_reject_count() : 0;
}

int GaussianSplattingPerformanceMonitors::_get_radius_reject_count() const {
    return active_renderer ? active_renderer->get_radius_reject_count() : 0;
}

int GaussianSplattingPerformanceMonitors::_get_viewport_reject_count() const {
    return active_renderer ? active_renderer->get_viewport_bounds_reject_count() : 0;
}

// ============================================================================
// Quality Monitor Getters
// ============================================================================

int GaussianSplattingPerformanceMonitors::_get_extreme_aspect_count() const {
    return active_renderer ? active_renderer->get_extreme_aspect_count() : 0;
}

int GaussianSplattingPerformanceMonitors::_get_index_mismatch_count() const {
    return active_renderer ? active_renderer->get_index_mismatch_count() : 0;
}

// ============================================================================
// SH Cache Monitor Getters
// ============================================================================

int GaussianSplattingPerformanceMonitors::_get_sh_cache_hits() const {
    return active_renderer ? active_renderer->get_sh_cache_hits() : 0;
}

int GaussianSplattingPerformanceMonitors::_get_sh_cache_updates() const {
    return active_renderer ? active_renderer->get_sh_cache_updates() : 0;
}

float GaussianSplattingPerformanceMonitors::_get_sh_cache_hit_rate_pct() const {
    return active_renderer ? active_renderer->get_sh_cache_hit_rate_pct() : 0.0f;
}

// ============================================================================
// Overflow Monitor Getters
// ============================================================================

int GaussianSplattingPerformanceMonitors::_get_overflow_tile_count() const {
    return active_renderer ? active_renderer->get_overflow_tile_count() : 0;
}

int GaussianSplattingPerformanceMonitors::_get_clamped_records() const {
    return active_renderer ? active_renderer->get_clamped_records() : 0;
}

int GaussianSplattingPerformanceMonitors::_get_aggregated_count() const {
    return active_renderer ? active_renderer->get_aggregated_count() : 0;
}

// ============================================================================
// Configuration Monitor Getters
// ============================================================================

int GaussianSplattingPerformanceMonitors::_get_tile_count() const {
    return active_renderer ? (int)active_renderer->get_tile_count() : 0;
}

// ============================================================================
// VRAM Budget Regulation Monitor Getters (Phase 1)
// ============================================================================

float GaussianSplattingPerformanceMonitors::_get_vram_current_usage_mb() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_vram_debug_stats();
    return stats.has("current_usage_bytes") ? (float)stats["current_usage_bytes"] / (1024.0f * 1024.0f) : 0.0f;
}

float GaussianSplattingPerformanceMonitors::_get_vram_budget_mb() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_vram_debug_stats();
    return stats.has("budget_bytes") ? (float)stats["budget_bytes"] / (1024.0f * 1024.0f) : 0.0f;
}

float GaussianSplattingPerformanceMonitors::_get_vram_usage_percent() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_vram_debug_stats();
    return stats.has("usage_percent") ? (float)stats["usage_percent"] : 0.0f;
}

int GaussianSplattingPerformanceMonitors::_get_vram_current_max_chunks() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_vram_debug_stats();
    return stats.has("current_max_chunks") ? (int)stats["current_max_chunks"] : 0;
}

int GaussianSplattingPerformanceMonitors::_get_vram_loaded_chunks() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_vram_debug_stats();
    return stats.has("loaded_chunks") ? (int)stats["loaded_chunks"] : 0;
}

int GaussianSplattingPerformanceMonitors::_get_vram_evicted_this_frame() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_vram_debug_stats();
    return stats.has("evicted_this_frame") ? (int)stats["evicted_this_frame"] : 0;
}

int GaussianSplattingPerformanceMonitors::_get_vram_loaded_this_frame() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_vram_debug_stats();
    return stats.has("loaded_this_frame") ? (int)stats["loaded_this_frame"] : 0;
}

int GaussianSplattingPerformanceMonitors::_get_vram_budget_warning_active() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_vram_debug_stats();
    return stats.has("budget_warning_active") ? ((bool)stats["budget_warning_active"] ? 1 : 0) : 0;
}

int GaussianSplattingPerformanceMonitors::_get_vram_regulation_adjustments() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_vram_debug_stats();
    return stats.has("regulation_adjustments") ? (int)stats["regulation_adjustments"] : 0;
}

int GaussianSplattingPerformanceMonitors::_get_vram_thrashing_events() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_vram_debug_stats();
    return stats.has("thrashing_events") ? (int)stats["thrashing_events"] : 0;
}

// ============================================================================
// Streaming Core Monitor Getters (Phase 1)
// ============================================================================

int GaussianSplattingPerformanceMonitors::_get_streaming_monitor_ready() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) {
        return 0;
    }
    if (!renderer->get_streaming_state().current_streaming_system.is_valid()) {
        return 0;
    }
    return renderer->get_streaming_state().current_streaming_system->is_runtime_ready() ? 1 : 0;
}

int GaussianSplattingPerformanceMonitors::_get_streaming_runtime_capacity_zero() const {
    Dictionary analytics = _get_streaming_analytics();
    if (!analytics.has("runtime_capacity_zero")) {
        return 0;
    }
    return bool(analytics["runtime_capacity_zero"]) ? 1 : 0;
}

int GaussianSplattingPerformanceMonitors::_get_streaming_runtime_buffer_invalid() const {
    Dictionary analytics = _get_streaming_analytics();
    if (!analytics.has("runtime_buffer_invalid")) {
        return 0;
    }
    return bool(analytics["runtime_buffer_invalid"]) ? 1 : 0;
}

int GaussianSplattingPerformanceMonitors::_get_streaming_invalid_camera_inputs() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) {
        return 0;
    }
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_chunk_culling_stats();
    if (!stats.has("invalid_camera_input_events")) {
        return 0;
    }
    return int(stats["invalid_camera_input_events"]);
}

int GaussianSplattingPerformanceMonitors::_get_streaming_total_chunks() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_chunk_culling_stats();
    return stats.has("total_chunks") ? (int)stats["total_chunks"] : 0;
}

int GaussianSplattingPerformanceMonitors::_get_streaming_visible_chunks() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_chunk_culling_stats();
    return stats.has("visible_chunks") ? (int)stats["visible_chunks"] : 0;
}

int GaussianSplattingPerformanceMonitors::_get_streaming_loaded_chunks() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_chunk_culling_stats();
    return stats.has("loaded_chunks") ? (int)stats["loaded_chunks"] : 0;
}

int GaussianSplattingPerformanceMonitors::_get_streaming_frustum_culled_chunks() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_chunk_culling_stats();
    return stats.has("frustum_culled_chunks") ? (int)stats["frustum_culled_chunks"] : 0;
}

float GaussianSplattingPerformanceMonitors::_get_streaming_vram_usage_mb() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    uint64_t vram_bytes = renderer->get_streaming_state().current_streaming_system->get_vram_usage();
    return (float)vram_bytes / (1024.0f * 1024.0f);
}

int GaussianSplattingPerformanceMonitors::_get_streaming_chunks_loaded_this_frame() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    return (int)renderer->get_streaming_state().current_streaming_system->get_chunks_loaded_this_frame();
}

int GaussianSplattingPerformanceMonitors::_get_streaming_chunks_evicted_this_frame() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    return (int)renderer->get_streaming_state().current_streaming_system->get_chunks_evicted_this_frame();
}

int GaussianSplattingPerformanceMonitors::_get_streaming_visible_count() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    return (int)renderer->get_streaming_state().current_streaming_system->get_visible_count();
}

int GaussianSplattingPerformanceMonitors::_get_streaming_buffer_capacity_splats() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    return (int)renderer->get_streaming_state().current_streaming_system->get_buffer_capacity_splats();
}

int GaussianSplattingPerformanceMonitors::_get_streaming_effective_splat_count() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    return (int)renderer->get_streaming_state().current_streaming_system->get_effective_splat_count();
}

float GaussianSplattingPerformanceMonitors::_get_streaming_visible_change_ratio() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    return renderer->get_streaming_state().current_streaming_system->get_visible_chunk_change_ratio();
}

float GaussianSplattingPerformanceMonitors::_get_streaming_lod_blend_factor() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    return renderer->get_streaming_state().current_streaming_system->get_global_lod_blend_factor();
}

int GaussianSplattingPerformanceMonitors::_get_streaming_sh_band_level() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    return renderer->get_streaming_state().current_streaming_system->get_global_sh_band_level();
}

float GaussianSplattingPerformanceMonitors::_get_streaming_bytes_uploaded_mb() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    if (!renderer->get_streaming_state().memory_stream.is_valid()) return 0.0f;
    StreamingStats stats = renderer->get_streaming_state().memory_stream->get_stats();
    return (float)stats.total_bytes_uploaded / (1024.0f * 1024.0f);
}

int GaussianSplattingPerformanceMonitors::_get_streaming_buffer_switches() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    if (!renderer->get_streaming_state().memory_stream.is_valid()) return 0;
    StreamingStats stats = renderer->get_streaming_state().memory_stream->get_stats();
    return (int)stats.buffer_switches;
}

float GaussianSplattingPerformanceMonitors::_get_streaming_effective_upload_cap_mb_per_frame() const {
    Dictionary analytics = _get_streaming_analytics();
    return analytics.has("effective_upload_cap_mb_per_frame")
            ? (float)analytics["effective_upload_cap_mb_per_frame"]
            : 0.0f;
}

float GaussianSplattingPerformanceMonitors::_get_streaming_effective_upload_cap_mb_per_slice() const {
    Dictionary analytics = _get_streaming_analytics();
    return analytics.has("effective_upload_cap_mb_per_slice")
            ? (float)analytics["effective_upload_cap_mb_per_slice"]
            : 0.0f;
}

float GaussianSplattingPerformanceMonitors::_get_streaming_effective_upload_cap_mb_per_second() const {
    Dictionary analytics = _get_streaming_analytics();
    return analytics.has("effective_upload_cap_mb_per_second")
            ? (float)analytics["effective_upload_cap_mb_per_second"]
            : 0.0f;
}

float GaussianSplattingPerformanceMonitors::_get_streaming_effective_vram_budget_mb() const {
    Dictionary analytics = _get_streaming_analytics();
    return analytics.has("effective_vram_budget_mb")
            ? (float)analytics["effective_vram_budget_mb"]
            : 0.0f;
}

int GaussianSplattingPerformanceMonitors::_get_streaming_effective_vram_max_chunks() const {
    Dictionary analytics = _get_streaming_analytics();
    return analytics.has("effective_vram_max_chunks")
            ? (int)analytics["effective_vram_max_chunks"]
            : 0;
}

int GaussianSplattingPerformanceMonitors::_get_streaming_upload_frame_cap_hit() const {
    Dictionary analytics = _get_streaming_analytics();
    if (!analytics.has("upload_frame_cap_hit")) {
        return 0;
    }
    return bool(analytics["upload_frame_cap_hit"]) ? 1 : 0;
}

int GaussianSplattingPerformanceMonitors::_get_streaming_upload_bandwidth_cap_hit() const {
    Dictionary analytics = _get_streaming_analytics();
    if (!analytics.has("upload_bandwidth_cap_hit")) {
        return 0;
    }
    return bool(analytics["upload_bandwidth_cap_hit"]) ? 1 : 0;
}

int GaussianSplattingPerformanceMonitors::_get_streaming_chunk_load_cap_hit() const {
    Dictionary analytics = _get_streaming_analytics();
    if (!analytics.has("chunk_load_cap_hit")) {
        return 0;
    }
    return bool(analytics["chunk_load_cap_hit"]) ? 1 : 0;
}

int GaussianSplattingPerformanceMonitors::_get_streaming_vram_chunk_cap_hit() const {
    Dictionary analytics = _get_streaming_analytics();
    if (!analytics.has("vram_chunk_cap_hit")) {
        return 0;
    }
    return bool(analytics["vram_chunk_cap_hit"]) ? 1 : 0;
}

int GaussianSplattingPerformanceMonitors::_get_streaming_queue_pressure_active() const {
    Dictionary analytics = _get_streaming_analytics();
    if (!analytics.has("queue_pressure_active")) {
        return 0;
    }
    return bool(analytics["queue_pressure_active"]) ? 1 : 0;
}

// ============================================================================
// LOD System Monitor Getters (Phase 2)
// ============================================================================

int GaussianSplattingPerformanceMonitors::_get_lod_current_level() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_lod_debug_stats();
    return stats.has("current_lod_level") ? (int)stats["current_lod_level"] : 0;
}

float GaussianSplattingPerformanceMonitors::_get_lod_distance_multiplier() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    Ref<VRAMBudgetRegulator> regulator = renderer->get_streaming_state().current_streaming_system->get_vram_regulator();
    return regulator.is_valid() ? regulator->get_lod_distance_multiplier() : 1.0f;
}

float GaussianSplattingPerformanceMonitors::_get_lod_target_distance() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_lod_debug_stats();
    return stats.has("lod_target_distance") ? (float)stats["lod_target_distance"] : 0.0f;
}

float GaussianSplattingPerformanceMonitors::_get_lod_hysteresis_zone() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    return renderer->get_streaming_state().current_streaming_system->get_lod_hysteresis_zone();
}

float GaussianSplattingPerformanceMonitors::_get_lod_blend_distance() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    return renderer->get_streaming_state().current_streaming_system->get_lod_blend_distance();
}

int GaussianSplattingPerformanceMonitors::_get_lod_transitions_this_frame() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_lod_debug_stats();
    return stats.has("transitions_this_frame") ? (int)stats["transitions_this_frame"] : 0;
}

int GaussianSplattingPerformanceMonitors::_get_lod_splat_skip_factor() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 1;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_lod_debug_stats();
    return stats.has("max_splat_skip_factor") ? (int)stats["max_splat_skip_factor"] : 1;
}

float GaussianSplattingPerformanceMonitors::_get_lod_opacity_multiplier() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 1.0f;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_lod_debug_stats();
    return stats.has("min_opacity_multiplier") ? (float)stats["min_opacity_multiplier"] : 1.0f;
}

int GaussianSplattingPerformanceMonitors::_get_lod_effective_count_after_skip() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    return (int)renderer->get_streaming_state().current_streaming_system->get_effective_splat_count();
}

float GaussianSplattingPerformanceMonitors::_get_lod_chunk_blend_factors_avg() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    return renderer->get_streaming_state().current_streaming_system->get_global_lod_blend_factor();
}

int GaussianSplattingPerformanceMonitors::_get_lod_chunks_in_transition() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system->get_lod_debug_stats();
    return stats.has("chunks_in_transition") ? (int)stats["chunks_in_transition"] : 0;
}

int GaussianSplattingPerformanceMonitors::_get_lod_quality_degradation_active() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Ref<VRAMBudgetRegulator> regulator = renderer->get_streaming_state().current_streaming_system->get_vram_regulator();
    if (!regulator.is_valid()) return 0;
    return (regulator->get_lod_distance_multiplier() > 1.0f) ? 1 : 0;
}

// ============================================================================
// GPU Memory Stream Monitor Getters (Phase 2)
// ============================================================================

float GaussianSplattingPerformanceMonitors::_get_memory_stream_total_bytes_uploaded_mb() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    if (!renderer->get_streaming_state().memory_stream.is_valid()) return 0.0f;
    StreamingStats stats = renderer->get_streaming_state().memory_stream->get_stats();
    return (float)stats.total_bytes_uploaded / (1024.0f * 1024.0f);
}

float GaussianSplattingPerformanceMonitors::_get_memory_stream_total_bytes_downloaded_mb() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    if (renderer->get_streaming_state().memory_stream.is_valid()) {
        StreamingStats stats = renderer->get_streaming_state().memory_stream->get_stats();
        if (stats.total_bytes_downloaded > 0) {
            return (float)stats.total_bytes_downloaded / (1024.0f * 1024.0f);
        }
    }

    Dictionary analytics = _get_streaming_analytics();
    if (analytics.has("evicted_bytes_total_mb")) {
        return (float)analytics["evicted_bytes_total_mb"];
    }
    return 0.0f;
}

int GaussianSplattingPerformanceMonitors::_get_memory_stream_buffer_switches() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    if (!renderer->get_streaming_state().memory_stream.is_valid()) return 0;
    StreamingStats stats = renderer->get_streaming_state().memory_stream->get_stats();
    return (int)stats.buffer_switches;
}

int GaussianSplattingPerformanceMonitors::_get_memory_stream_stalls() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    if (!renderer->get_streaming_state().memory_stream.is_valid()) return 0;
    StreamingStats stats = renderer->get_streaming_state().memory_stream->get_stats();
    return (int)stats.stalls;
}

float GaussianSplattingPerformanceMonitors::_get_memory_stream_stall_percent() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    if (!renderer->get_streaming_state().memory_stream.is_valid()) return 0.0f;
    StreamingStats stats = renderer->get_streaming_state().memory_stream->get_stats();
    if (stats.total_frames == 0) return 0.0f;
    return 100.0f * (float)stats.stalls / (float)stats.total_frames;
}

int GaussianSplattingPerformanceMonitors::_get_memory_stream_pool_hits() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    if (!renderer->get_streaming_state().memory_stream.is_valid()) return 0;
    StreamingStats stats = renderer->get_streaming_state().memory_stream->get_stats();
    return (int)stats.pool_hits;
}

int GaussianSplattingPerformanceMonitors::_get_memory_stream_pool_misses() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    if (!renderer->get_streaming_state().memory_stream.is_valid()) return 0;
    StreamingStats stats = renderer->get_streaming_state().memory_stream->get_stats();
    return (int)stats.pool_misses;
}

float GaussianSplattingPerformanceMonitors::_get_memory_stream_pool_hit_rate_pct() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    if (!renderer->get_streaming_state().memory_stream.is_valid()) return 0.0f;
    StreamingStats stats = renderer->get_streaming_state().memory_stream->get_stats();
    uint32_t total = stats.pool_hits + stats.pool_misses;
    if (total == 0) return 0.0f;
    return 100.0f * (float)stats.pool_hits / (float)total;
}

float GaussianSplattingPerformanceMonitors::_get_memory_stream_peak_memory_mb() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    if (!renderer->get_streaming_state().memory_stream.is_valid()) return 0.0f;
    StreamingStats stats = renderer->get_streaming_state().memory_stream->get_stats();
    return stats.peak_memory_mb;
}

int GaussianSplattingPerformanceMonitors::_get_memory_stream_defrag_count() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    if (!renderer->get_streaming_state().memory_stream.is_valid()) return 0;
    StreamingStats stats = renderer->get_streaming_state().memory_stream->get_stats();
    return (int)stats.defrag_count;
}

// ============================================================================
// Chunk Management Monitor Getters (Phase 3)
// ============================================================================

int GaussianSplattingPerformanceMonitors::_get_chunk_prefetch_hits() const {
    Dictionary analytics = _get_streaming_analytics();
    if (analytics.has("prefetch_hits")) {
        return (int)analytics["prefetch_hits"];
    }
    if (analytics.has("scheduler_prefetch_candidates")) {
        return (int)analytics["scheduler_prefetch_candidates"];
    }
    return 0;
}

int GaussianSplattingPerformanceMonitors::_get_chunk_prefetch_misses() const {
    Dictionary analytics = _get_streaming_analytics();
    if (analytics.has("prefetch_misses")) {
        return (int)analytics["prefetch_misses"];
    }
    return 0;
}

float GaussianSplattingPerformanceMonitors::_get_chunk_prefetch_efficiency_pct() const {
    int hits = _get_chunk_prefetch_hits();
    int misses = _get_chunk_prefetch_misses();
    int total = hits + misses;
    if (total == 0) return 0.0f;
    return 100.0f * (float)hits / (float)total;
}

float GaussianSplattingPerformanceMonitors::_get_chunk_camera_velocity() const {
    Dictionary analytics = _get_streaming_analytics();
    return analytics.has("camera_velocity") ? (float)analytics["camera_velocity"] : 0.0f;
}

float GaussianSplattingPerformanceMonitors::_get_chunk_average_load_time_ms() const {
    Dictionary analytics = _get_streaming_analytics();
    if (analytics.has("avg_chunk_load_time_ms")) {
        return (float)analytics["avg_chunk_load_time_ms"];
    }
    if (analytics.has("pack_avg_ms")) {
        return (float)analytics["pack_avg_ms"];
    }
    return 0.0f;
}

int GaussianSplattingPerformanceMonitors::_get_chunk_upload_queue_depth() const {
    Dictionary analytics = _get_streaming_analytics();
    if (analytics.has("pending_uploads")) {
        return (int)analytics["pending_uploads"];
    }
    if (analytics.has("scheduler_upload_queue_depth")) {
        return (int)analytics["scheduler_upload_queue_depth"];
    }
    return 0;
}

int GaussianSplattingPerformanceMonitors::_get_chunk_pack_jobs_in_flight() const {
    Dictionary analytics = _get_streaming_analytics();
    return analytics.has("pack_jobs_in_flight") ? (int)analytics["pack_jobs_in_flight"] : 0;
}

float GaussianSplattingPerformanceMonitors::_get_chunk_total_capacity_mb() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    if (!renderer->get_streaming_state().current_streaming_system.is_valid()) return 0.0f;
    uint32_t capacity = renderer->get_streaming_state().current_streaming_system->get_buffer_capacity_splats();
    return (float)(capacity * sizeof(PackedGaussian)) / (1024.0f * 1024.0f);
}

// Pack/Upload Timing Monitor Getters (Phase 4.5)
float GaussianSplattingPerformanceMonitors::_get_pack_avg_time_ms() const {
    Dictionary analytics = _get_streaming_analytics();
    return analytics.has("pack_avg_ms") ? (float)analytics["pack_avg_ms"] : 0.0f;
}

float GaussianSplattingPerformanceMonitors::_get_pack_max_time_ms() const {
    Dictionary analytics = _get_streaming_analytics();
    return analytics.has("pack_max_ms") ? (float)analytics["pack_max_ms"] : 0.0f;
}

int GaussianSplattingPerformanceMonitors::_get_pack_jobs_completed() const {
    Dictionary analytics = _get_streaming_analytics();
    return analytics.has("pack_jobs_completed") ? (int)analytics["pack_jobs_completed"] : 0;
}

float GaussianSplattingPerformanceMonitors::_get_upload_mb_this_frame() const {
    Dictionary analytics = _get_streaming_analytics();
    return analytics.has("upload_mb_this_frame") ? (float)analytics["upload_mb_this_frame"] : 0.0f;
}

int GaussianSplattingPerformanceMonitors::_get_upload_chunks_this_frame() const {
    Dictionary analytics = _get_streaming_analytics();
    return analytics.has("upload_chunks_this_frame") ? (int)analytics["upload_chunks_this_frame"] : 0;
}

// Advanced LOD Analytics Monitor Getters (Phase 4)
float GaussianSplattingPerformanceMonitors::_get_lod_min_chunk_distance() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system.is_valid() ?
            renderer->get_streaming_state().current_streaming_system->get_lod_debug_stats() :
            Dictionary();
    return stats.has("min_distance") ? (float)stats["min_distance"] : 0.0f;
}

float GaussianSplattingPerformanceMonitors::_get_lod_max_chunk_distance() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system.is_valid() ?
            renderer->get_streaming_state().current_streaming_system->get_lod_debug_stats() :
            Dictionary();
    return stats.has("max_distance") ? (float)stats["max_distance"] : 0.0f;
}

float GaussianSplattingPerformanceMonitors::_get_lod_avg_chunk_distance() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system.is_valid() ?
            renderer->get_streaming_state().current_streaming_system->get_lod_debug_stats() :
            Dictionary();
    return stats.has("avg_distance") ? (float)stats["avg_distance"] : 0.0f;
}

float GaussianSplattingPerformanceMonitors::_get_lod_reduction_ratio_pct() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system.is_valid() ?
            renderer->get_streaming_state().current_streaming_system->get_lod_debug_stats() :
            Dictionary();
    if (stats.has("reduction_ratio")) {
        return 100.0f * (float)stats["reduction_ratio"];
    }
    return 0.0f;
}

int GaussianSplattingPerformanceMonitors::_get_lod_level_0_chunk_count() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system.is_valid() ?
            renderer->get_streaming_state().current_streaming_system->get_lod_debug_stats() :
            Dictionary();
    if (stats.has("lod_distribution")) {
        Array lod_dist = stats["lod_distribution"];
        return lod_dist.size() > 0 ? (int)lod_dist[0] : 0;
    }
    return 0;
}

int GaussianSplattingPerformanceMonitors::_get_lod_sh_band_3_chunk_count() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0;
    Dictionary stats = renderer->get_streaming_state().current_streaming_system.is_valid() ?
            renderer->get_streaming_state().current_streaming_system->get_lod_debug_stats() :
            Dictionary();
    if (stats.has("sh_band_distribution")) {
        Array sh_dist = stats["sh_band_distribution"];
        return sh_dist.size() > 3 ? (int)sh_dist[3] : 0;
    }
    return 0;
}

// Compression Analytics Monitor Getters (Phase 5)
float GaussianSplattingPerformanceMonitors::_get_sh_compression_raw_mb() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    if (!renderer->get_streaming_state().current_streaming_system.is_valid()) return 0.0f;
    SHCompressionMetrics metrics = renderer->get_streaming_state().current_streaming_system->get_total_sh_metrics();
    return (float)metrics.raw_bytes / (1024.0f * 1024.0f);
}

float GaussianSplattingPerformanceMonitors::_get_sh_compression_compressed_mb() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    if (!renderer->get_streaming_state().current_streaming_system.is_valid()) return 0.0f;
    SHCompressionMetrics metrics = renderer->get_streaming_state().current_streaming_system->get_total_sh_metrics();
    return (float)metrics.compressed_bytes / (1024.0f * 1024.0f);
}

float GaussianSplattingPerformanceMonitors::_get_sh_compression_ratio_pct() const {
    GaussianSplatRenderer *renderer = _get_active_splat_renderer(true);
    if (!renderer) return 0.0f;
    if (!renderer->get_streaming_state().current_streaming_system.is_valid()) return 0.0f;
    SHCompressionMetrics metrics = renderer->get_streaming_state().current_streaming_system->get_total_sh_metrics();
    if (metrics.raw_bytes > 0) {
        return 100.0f * (float)metrics.compressed_bytes / (float)metrics.raw_bytes;
    }
    return 0.0f;
}
