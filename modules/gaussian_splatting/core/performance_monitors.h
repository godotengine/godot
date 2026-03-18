#ifndef GAUSSIAN_SPLATTING_PERFORMANCE_MONITORS_H
#define GAUSSIAN_SPLATTING_PERFORMANCE_MONITORS_H

#include "core/object/object.h"
#include "main/performance.h"
#include "core/templates/local_vector.h"
#include "core/string/string_name.h"
#include "core/variant/dictionary.h"

class TileRenderer;
class GaussianSplatRenderer;

/**
 * @class GaussianSplattingPerformanceMonitors
 * @brief Manages Custom Performance Monitors for the Gaussian Splatting module
 *
 * This class registers custom performance monitors with Godot's Performance singleton,
 * exposing GPU timings and rendering statistics to the editor debugger with zero overhead
 * when the debugger is closed.
 *
 * Monitors are organized into categories:
 * - GPU Timings: Frame, cull, binning, prefix, sort, raster, resolve times
 * - Statistics: Visible splats, projection success rates, rejection counts
 * - Quality: Extreme aspect ratios, SH cache hit rates
 * - Overflow: Tile overflow statistics
 *
 * @note Thread Safety: This class assumes single-threaded access from the main thread.
 * All renderer registration/unregistration and monitor polling occurs on the main thread
 * as part of Godot's normal rendering and debugger update cycles. The internal renderer
 * tracking lists (LocalVector) are not protected by mutexes. If future Godot versions
 * introduce multi-threaded debugger polling, mutex protection should be added.
 *
 * @note Inherits from Object to enable callable_mp() for performance monitor callbacks.
 * This is required by Godot's Performance singleton which expects Object-based Callables.
 */
class GaussianSplattingPerformanceMonitors : public Object {
    GDCLASS(GaussianSplattingPerformanceMonitors, Object);

public:
    static GaussianSplattingPerformanceMonitors *get_singleton();
    static void create_singleton();
    static void destroy_singleton();

    /**
     * Register a TileRenderer to be monitored.
     * The renderer must remain valid for the lifetime of monitoring.
     */
    void register_renderer(TileRenderer *p_renderer);

    /**
     * Unregister a TileRenderer when it's destroyed.
     */
    void unregister_renderer(TileRenderer *p_renderer);

    /**
     * Register a GaussianSplatRenderer to be monitored for streaming/LOD metrics.
     * The renderer must remain valid for the lifetime of monitoring.
     */
    void register_splat_renderer(GaussianSplatRenderer *p_renderer);

    /**
     * Unregister a GaussianSplatRenderer when it's destroyed.
     */
    void unregister_splat_renderer(GaussianSplatRenderer *p_renderer);

    /**
     * Initialize all custom performance monitors.
     * Called once during module initialization.
     */
    void initialize_monitors();

    /**
     * Cleanup all custom performance monitors.
     * Called during module shutdown.
     */
    void cleanup_monitors();

    // Destructor must be public for memdelete() to access it
    ~GaussianSplattingPerformanceMonitors();

private:
    GaussianSplattingPerformanceMonitors();

    static GaussianSplattingPerformanceMonitors *singleton;

    // Track all registered renderers to handle multi-viewport scenarios
    // When active renderer is destroyed, we automatically switch to another
    LocalVector<TileRenderer *> registered_renderers;
    LocalVector<GaussianSplatRenderer *> registered_splat_renderers;

    // Currently active renderer used for monitor queries
    TileRenderer *active_renderer = nullptr;
    GaussianSplatRenderer *active_splat_renderer = nullptr;
    LocalVector<StringName> registered_monitor_ids;

    bool monitors_registered = false;

    GaussianSplatRenderer *_get_active_splat_renderer(bool p_require_streaming) const;
    bool _is_telemetry_active() const;
    void _register_monitor_definitions(Performance *p_perf);
    Dictionary _get_streaming_analytics() const;

    // Monitor getter callbacks (these are called by the Performance singleton)
    float _get_gpu_frame_time_ms() const;
    float _get_gpu_cull_time_ms() const;
    float _get_gpu_sort_time_ms() const;
    float _get_gpu_binning_time_ms() const;
    float _get_gpu_prefix_time_ms() const;
    float _get_gpu_raster_time_ms() const;
    float _get_gpu_resolve_time_ms() const;
    int _get_telemetry_active() const;
    float _get_cpu_setup_time_ms() const;
    String _get_route_uid() const;
    String _get_sort_route_uid() const;

    int _get_visible_splat_count() const;
    int _get_total_processed() const;
    int _get_projection_success_count() const;
    float _get_projection_success_rate_pct() const;
    int _get_clip_reject_count() const;
    int _get_radius_reject_count() const;
    int _get_viewport_reject_count() const;
    int _get_extreme_aspect_count() const;
    int _get_index_mismatch_count() const;

    int _get_sh_cache_hits() const;
    int _get_sh_cache_updates() const;
    float _get_sh_cache_hit_rate_pct() const;

    int _get_overflow_tile_count() const;
    int _get_clamped_records() const;
    int _get_aggregated_count() const;

    int _get_tile_count() const;

    // VRAM Budget Regulation Monitors (Phase 1)
    float _get_vram_current_usage_mb() const;
    float _get_vram_budget_mb() const;
    float _get_vram_usage_percent() const;
    int _get_vram_current_max_chunks() const;
    int _get_vram_loaded_chunks() const;
    int _get_vram_evicted_this_frame() const;
    int _get_vram_loaded_this_frame() const;
    int _get_vram_budget_warning_active() const;
    int _get_vram_regulation_adjustments() const;
    int _get_vram_thrashing_events() const;

    // Streaming Core Monitors (Phase 1)
    int _get_streaming_monitor_ready() const;
    int _get_streaming_runtime_capacity_zero() const;
    int _get_streaming_runtime_buffer_invalid() const;
    int _get_streaming_invalid_camera_inputs() const;
    int _get_streaming_total_chunks() const;
    int _get_streaming_visible_chunks() const;
    int _get_streaming_loaded_chunks() const;
    int _get_streaming_frustum_culled_chunks() const;
    float _get_streaming_vram_usage_mb() const;
    int _get_streaming_chunks_loaded_this_frame() const;
    int _get_streaming_chunks_evicted_this_frame() const;
    int _get_streaming_visible_count() const;
    int _get_streaming_buffer_capacity_splats() const;
    int _get_streaming_effective_splat_count() const;
    float _get_streaming_visible_change_ratio() const;
    float _get_streaming_lod_blend_factor() const;
    int _get_streaming_sh_band_level() const;
    float _get_streaming_bytes_uploaded_mb() const;
    int _get_streaming_buffer_switches() const;
    float _get_streaming_effective_upload_cap_mb_per_frame() const;
    float _get_streaming_effective_upload_cap_mb_per_slice() const;
    float _get_streaming_effective_upload_cap_mb_per_second() const;
    float _get_streaming_effective_vram_budget_mb() const;
    int _get_streaming_effective_vram_max_chunks() const;
    int _get_streaming_upload_frame_cap_hit() const;
    int _get_streaming_upload_bandwidth_cap_hit() const;
    int _get_streaming_chunk_load_cap_hit() const;
    int _get_streaming_vram_chunk_cap_hit() const;
    int _get_streaming_queue_pressure_active() const;

    // LOD System Monitors (Phase 2)
    int _get_lod_current_level() const;
    float _get_lod_distance_multiplier() const;
    float _get_lod_target_distance() const;
    float _get_lod_hysteresis_zone() const;
    float _get_lod_blend_distance() const;
    int _get_lod_transitions_this_frame() const;
    int _get_lod_splat_skip_factor() const;
    float _get_lod_opacity_multiplier() const;
    int _get_lod_effective_count_after_skip() const;
    float _get_lod_chunk_blend_factors_avg() const;
    int _get_lod_chunks_in_transition() const;
    int _get_lod_quality_degradation_active() const;

    // GPU Memory Stream Monitors (Phase 2)
    float _get_memory_stream_total_bytes_uploaded_mb() const;
    float _get_memory_stream_total_bytes_downloaded_mb() const;
    int _get_memory_stream_buffer_switches() const;
    int _get_memory_stream_stalls() const;
    float _get_memory_stream_stall_percent() const;
    int _get_memory_stream_pool_hits() const;
    int _get_memory_stream_pool_misses() const;
    float _get_memory_stream_pool_hit_rate_pct() const;
    float _get_memory_stream_peak_memory_mb() const;
    int _get_memory_stream_defrag_count() const;

    // Chunk Management Monitors (Phase 3)
    int _get_chunk_prefetch_hits() const;
    int _get_chunk_prefetch_misses() const;
    float _get_chunk_prefetch_efficiency_pct() const;
    float _get_chunk_camera_velocity() const;
    float _get_chunk_average_load_time_ms() const;
    int _get_chunk_upload_queue_depth() const;
    int _get_chunk_pack_jobs_in_flight() const;
    float _get_chunk_total_capacity_mb() const;

    // Pack/Upload Timing Monitors (Phase 4.5)
    float _get_pack_avg_time_ms() const;
    float _get_pack_max_time_ms() const;
    int _get_pack_jobs_completed() const;
    float _get_upload_mb_this_frame() const;
    int _get_upload_chunks_this_frame() const;

    // Advanced LOD Analytics Monitors (Phase 4)
    float _get_lod_min_chunk_distance() const;
    float _get_lod_max_chunk_distance() const;
    float _get_lod_avg_chunk_distance() const;
    float _get_lod_reduction_ratio_pct() const;
    int _get_lod_level_0_chunk_count() const;
    int _get_lod_sh_band_3_chunk_count() const;

    // Compression Analytics Monitors (Phase 5)
    float _get_sh_compression_raw_mb() const;
    float _get_sh_compression_compressed_mb() const;
    float _get_sh_compression_ratio_pct() const;
};

#endif // GAUSSIAN_SPLATTING_PERFORMANCE_MONITORS_H
