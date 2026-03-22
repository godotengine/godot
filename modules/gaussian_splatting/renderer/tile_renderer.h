#ifndef TILE_RENDERER_H
#define TILE_RENDERER_H

#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2i.h"
#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "servers/rendering/rendering_device.h"

#include "shader_compilation_types.h"

#include <cstdint>
#include <functional>
#include <memory>

#include "../core/gaussian_data.h"
#include "gpu_sorter.h"
#include "tile_render_types.h"
#include "tile_render_resources.h"
#include "tile_prefix_scan_utils.h"

class GPUPerformanceMonitor;
class IGPUSorter;
struct TileRenderParamsGPU;
class ShaderCompilationManager;
class RenderDeviceManager;

class TileRenderer : public RefCounted {
    GDCLASS(TileRenderer, RefCounted);

public:
    static constexpr int DEFAULT_TILE_SIZE = 16;  // Default tile size for performance; reduce if overflow artifacts appear
    static constexpr int MAX_SPLATS_PER_TILE = 1024;  // Shared cache window; tiles can rasterize beyond this via global-buffer fallback
    static constexpr uint32_t BINNING_GROUP_SIZE = 256;
    static constexpr int DENSITY_BUCKET_COUNT = 5;
    static constexpr uint32_t DEBUG_SPLAT_AUDIT_MAX_SAMPLES = 64;
    static constexpr uint32_t MAX_OMNI_LIGHTS = 8;
    static constexpr uint32_t MAX_SPOT_LIGHTS = 8;

    using AdaptiveSettings = GaussianSplatting::TileAdaptiveSettings;
    using DebugCounterSnapshot = GaussianSplatting::TileDebugCounterSnapshot;
    using OverflowStatsSnapshot = GaussianSplatting::TileOverflowStatsSnapshot;
    using SplatAuditSnapshot = GaussianSplatting::TileSplatAuditSnapshot;
    using DensityMetrics = GaussianSplatting::TileDensityMetrics;
    using RenderStats = GaussianSplatting::TileRenderStats;
    using RenderParams = GaussianSplatting::TileRenderParams;
    using TimestampRange = GaussianSplatting::TileTimestampRange;
    using ResolveDebugMode = GaussianSplatting::TileResolveDebugMode;
    using TileTimingState = GaussianSplatting::TileTimingState;
    using TilePerformanceMetrics = GaussianSplatting::TilePerformanceMetrics;
    using TileDiagnosticsState = GaussianSplatting::TileDiagnosticsState;
    using TileConfigState = GaussianSplatting::TileConfigState;
    using TileGridState = GaussianSplatting::TileGridState;
    using TileRenderSettings = GaussianSplatting::TileRenderSettings;
    using TileFrameState = GaussianSplatting::TileFrameState;
    using BufferOwnership = GaussianSplatting::BufferOwnership;
    using TileRenderTargets = GaussianSplatting::TileRenderTargets;
    using TileDeviceContext = GaussianSplatting::TileDeviceContext;
    using TileShaderResources = GaussianSplatting::TileShaderResources;
    using TileGlobalSortResources = GaussianSplatting::TileGlobalSortResources;
    using TileUniformBuffers = GaussianSplatting::TileUniformBuffers;
    using TileProjectionBuffers = GaussianSplatting::TileProjectionBuffers;
    using TileSHCacheBuffers = GaussianSplatting::TileSHCacheBuffers;
    using TileSubpixelHistoryBuffers = GaussianSplatting::TileSubpixelHistoryBuffers;
    using TileSubpixelVisibilityBuffers = GaussianSplatting::TileSubpixelVisibilityBuffers;
    using TileResourceController = GaussianSplatting::TileResourceController;
    using ShaderVariant = TileShaderCompilation::ShaderVariant;

    struct AdaptiveOverlapBudgetRuntimeState {
        uint32_t suggested_budget_records = 0u;
        uint32_t low_utilization_frames = 0u;
        uint32_t recent_raw_usage_records[16] = {};
        uint32_t recent_raw_usage_count = 0u;
        uint32_t recent_raw_usage_write_index = 0u;
        uint32_t recent_raw_usage_peak = 0u;
    };

    static constexpr ResolveDebugMode RESOLVE_DEBUG_NONE = GaussianSplatting::RESOLVE_DEBUG_NONE;
    static constexpr ResolveDebugMode RESOLVE_DEBUG_INPUT = GaussianSplatting::RESOLVE_DEBUG_INPUT;
    static constexpr ResolveDebugMode RESOLVE_DEBUG_OUTPUT = GaussianSplatting::RESOLVE_DEBUG_OUTPUT;

    TileRenderer();
    ~TileRenderer();

    Error initialize(RenderingDevice *p_rendering_device, const Vector2i &p_initial_viewport = Vector2i(), int p_tile_size = -1,
            RD::DataFormat p_format = RD::DATA_FORMAT_MAX, RenderingDevice *p_submission_device = nullptr);
    void cleanup();

    RID render(RenderingDevice *p_rendering_device, const RenderParams &p_params);
    Error resize(const Vector2i &p_size, RD::DataFormat p_format = RD::DATA_FORMAT_MAX);
    Error resize(uint32_t width, uint32_t height, RD::DataFormat p_format = RD::DATA_FORMAT_MAX) {
        return resize(Vector2i(width, height), p_format);
    }

    void set_output_format(RD::DataFormat p_format);
    RD::DataFormat get_output_format() const { return config_state.output_format; }
    void set_output_invalidation_callback(std::function<void()> p_callback);
    void clear_output_invalidation_callback();

    RID get_output_texture() const;
    RID get_depth_texture() const;
    RenderingDevice *get_output_texture_owner() const;
    RenderingDevice *get_depth_texture_owner() const;
    bool has_depth_output() const { return render_targets.depth_texture.is_valid(); }
    bool is_depth_copy_compatible() const { return render_targets.depth_texture_copy_compatible; }
    RID get_debug_counter_buffer() const { return debug_stats.debug_counter_buffer; }
    DebugCounterSnapshot get_debug_counters() const;
    OverflowStatsSnapshot get_overflow_stats() const;
    uint64_t get_overflow_stats_frame_serial() const { return debug_stats.cached_overflow_frame_serial; }
    SplatAuditSnapshot get_splat_audit_snapshot() const;

    // Note: tile_raster_pipeline is created lazily during first render (needs framebuffer format)
    // So we only check compute pipelines here. The raster shader must be valid though.
    bool is_initialized() const { return device_context.resource_rd != nullptr; }
    RID get_tile_binning_pipeline() const { return shader_resources.tile_binning_pipeline; }
    RID get_tile_raster_pipeline() const { return shader_resources.tile_raster_pipeline; }
    uint64_t get_shader_defines_hash() const { return shader_resources.shader_defines_hash; }

    int get_tile_size() const { return config_state.tile_size; }
    Vector2i get_tile_grid_size() const { return Vector2i(grid_state.tiles_x, grid_state.tiles_y); }
    int get_tile_splat_capacity() const { return config_state.effective_splat_capacity; }

    void set_performance_monitor(GPUPerformanceMonitor *p_monitor) { device_context.performance_monitor = p_monitor; }
    void set_contract_main_device(RenderingDevice *p_main_device) { resource_controller.set_contract_main_device(p_main_device); }
    void set_device_manager(RenderDeviceManager *p_device_manager);
    void track_output_resources(const RID &p_color_output, RenderingDevice *p_color_device,
            const RID &p_depth_output, RenderingDevice *p_depth_device);
    void clear_output_resource_tracking();
    struct RenderResult {
        RID output_texture;
        RID depth_texture;
        RenderingDevice *output_owner = nullptr;
        RenderingDevice *depth_owner = nullptr;
        bool success = false;
        bool has_depth = false;
        bool depth_copy_compatible = false;
    };
    RenderResult render_with_contract(RenderingDevice *p_device, const RenderParams &p_params);
    void set_gpu_timestamp_capture_enabled(bool p_enabled) { gpu_timestamp_capture_enabled = p_enabled; }
    bool is_gpu_timestamp_capture_enabled() const { return gpu_timestamp_capture_enabled; }
    void set_frame_serial(uint64_t p_frame_serial) { frame_state.current_frame_serial = p_frame_serial; }
    void set_resolve_debug_mode(ResolveDebugMode p_mode) { render_settings.resolve_debug_mode = p_mode; }
    ResolveDebugMode get_resolve_debug_mode() const { return render_settings.resolve_debug_mode; }
    void set_resolve_debug_visualize_tiles(bool p_enabled);
    bool is_resolve_debug_visualize_tiles_enabled() const { return diagnostics.resolve_debug_visualize_tiles; }
    void set_resolve_use_texel_fetch(bool p_enabled);
    bool is_resolve_use_texel_fetch_enabled() const { return diagnostics.resolve_use_texel_fetch_sampling; }
    void set_debug_binning_counters_enabled(bool p_enabled);
    bool is_debug_binning_counters_enabled() const { return diagnostics.debug_binning_counters_enabled; }

    void set_adaptive_settings(const AdaptiveSettings &p_settings);
    AdaptiveSettings get_adaptive_settings() const { return adaptive_controller.get_settings(); }
    AdaptiveOverlapBudgetRuntimeState &access_adaptive_overlap_budget_runtime_state();
    const AdaptiveOverlapBudgetRuntimeState *get_adaptive_overlap_budget_runtime_state_ptr() const;
    void clear_adaptive_overlap_budget_runtime_state();

    float get_tile_assignment_time() const { return perf_metrics.tile_assignment_ms; }
    float get_rasterization_time() const { return perf_metrics.rasterization_ms; }
    uint64_t get_sort_sync_fallback_count() const { return perf_metrics.sort_sync_fallback_count; }
    float get_last_submission_cpu_ms() const { return timing_state.last_submission_cpu_ms; }
    float get_last_gpu_frame_time_ms() const { return timing_state.last_frame_gpu_ms; }
    float get_last_gpu_binning_time_ms() const { return timing_state.last_binning_gpu_ms; }
    float get_last_gpu_raster_time_ms() const { return timing_state.last_raster_gpu_ms; }
    float get_last_gpu_prefix_time_ms() const { return timing_state.last_prefix_gpu_ms; }
    float get_last_gpu_resolve_time_ms() const { return timing_state.last_resolve_gpu_ms; }
    float get_last_setup_cpu_ms() const { return timing_state.last_setup_cpu_ms; }
    uint64_t get_gpu_timing_frame_serial() const { return timing_state.gpu_timing_frame_serial; }
    uint64_t get_gpu_timing_frames_behind() const { return timing_state.gpu_timing_frames_behind; }
    void resolve_gpu_timestamps_async();
    uint32_t get_tile_count() const { return grid_state.total_tiles; }

    RenderStats get_last_render_stats() const { return diagnostics.last_render_stats; }
    const Vector<uint32_t> &get_tile_density_snapshot() const { return diagnostics.tile_density_snapshot; }
    SortingMetrics get_sorter_metrics() const;

    void set_debug_log_resolve(bool p_enabled);
    bool get_debug_log_resolve() const { return diagnostics.debug_log_resolve; }

    // Performance monitor getters for statistics
    int get_visible_splat_count() const { return debug_stats.cached_debug_counters.success_count; }
    int get_total_processed() const { return debug_stats.cached_debug_counters.total_processed; }
    int get_projection_success_count() const { return debug_stats.cached_debug_counters.success_count; }
    float get_projection_success_rate_pct() const;
    int get_clip_bounds_reject_count() const { return debug_stats.cached_debug_counters.clip_bounds_reject; }
    int get_radius_reject_count() const { return debug_stats.cached_debug_counters.radius_reject; }
    int get_viewport_bounds_reject_count() const { return debug_stats.cached_debug_counters.viewport_bounds_reject; }
    int get_extreme_aspect_count() const { return debug_stats.cached_debug_counters.extreme_conic_count; }
    int get_index_mismatch_count() const { return debug_stats.cached_debug_counters.index_mismatch_count; }
    int get_sh_cache_hits() const { return debug_stats.cached_debug_counters.sh_cache_hits; }
    int get_sh_cache_updates() const { return debug_stats.cached_debug_counters.sh_cache_updates; }
    float get_sh_cache_hit_rate_pct() const;
    int get_overflow_tile_count() const { return debug_stats.cached_overflow_stats.overflow_tile_count; }
    int get_clamped_records() const { return debug_stats.cached_overflow_stats.overflow_splats_clamped; }
    int get_aggregated_count() const { return debug_stats.cached_overflow_stats.overflow_splats_aggregated; }

protected:
    static void _bind_methods();

private:
    class RenderFrameExecutor;

    // These struct friends are internal data types whose cleanup methods need private
    // TileRenderer helpers (e.g., device access, resource freeing). They are tightly
    // coupled by design as part of TileRenderer's decomposed state.
    friend struct GaussianSplatting::TileRenderTargets;
    friend struct GaussianSplatting::TileShaderResources;
    friend struct GaussianSplatting::TileGlobalSortResources;
    friend struct GaussianSplatting::TileUniformBuffers;
    friend struct GaussianSplatting::TileProjectionBuffers;
    friend struct GaussianSplatting::TileSHCacheBuffers;
    friend struct GaussianSplatting::TileSubpixelHistoryBuffers;
    friend struct GaussianSplatting::TileSubpixelVisibilityBuffers;
    // ShaderCompilationManager is the compilation subsystem for TileRenderer's shader
    // pipelines. It accesses numerous private shader sources, compilation helpers, and
    // device management methods that are specific to shader setup and not appropriate
    // for general public API exposure.
    friend class ShaderCompilationManager;

    Error _ensure_resources(const Vector2i &p_size, int p_tile_size, RD::DataFormat p_format);
    int _compute_adaptive_tile_size(int p_requested_tile_size, const Vector2i &p_size);
    void _update_tile_dimensions(const Vector2i &p_size);
    void _create_aux_buffers();
    void _create_output_texture(const Vector2i &p_size, RD::DataFormat p_format);
    Error _compile_tile_shaders();
    void _detect_subgroup_support(RenderingDevice *p_device);
    bool _check_pipeline_validity(RenderingDevice *p_device, bool p_want_compute_raster) const;
    void _initialize_shader_sources();
    void _free_existing_pipelines(RenderingDevice *p_shader_owner);
    Error _compile_binning_shaders(RenderingDevice *p_device);
    Error _compile_prefix_shaders(RenderingDevice *p_device);
    Error _compile_raster_shaders(RenderingDevice *p_device, bool p_want_compute_raster);
    void _clear_debug_counters();
    void _update_splat_audit_buffer(const RenderParams &p_params);
    uint32_t _get_projection_stride() const;
    uint64_t _compute_projection_buffer_bytes(uint32_t p_visible_count, uint32_t &r_capacity);
    void _destroy_output_textures();
	    uint64_t _dispatch_tile_binning(uint32_t gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
	            RID p_lighting_uniform_set, RenderingDevice *p_submission_device, bool p_requires_sync);
	    uint64_t _dispatch_tile_binning_count(uint32_t gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
	            RID p_lighting_uniform_set, RenderingDevice *p_submission_device, bool p_requires_sync);
	    uint64_t _dispatch_tile_rasterizer(uint32_t gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
	            RenderingDevice *p_submission_device);
	    uint64_t _dispatch_tile_rasterizer_compute(uint32_t gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
	            RID p_image_uniform_set, RenderingDevice *p_submission_device);
    void _dispatch_tile_resolve(const Vector2i &p_viewport, int p_tile_size, bool p_output_is_premultiplied,
            const RenderParams &p_params);
    void _queue_submission(RenderingDevice *p_device, bool p_requires_sync = false);
    void _flush_pending_submission(bool p_block);
    void _dump_gpu_debug_counters(const RenderParams &p_params);
    bool _resolve_texture_owner(const char *p_label, const RID &p_texture, RenderingDevice *&r_owner,
            RenderingDevice *p_main_device, RenderDeviceManager *p_manager, bool p_log_errors);
    void _collect_render_statistics();
    void _reset_timestamp_tracking();
    void _resolve_timestamp_range(TimestampRange &p_range, float &r_duration_ms);
    uint64_t _compute_shader_defines_hash() const;

    // GPU timestamp helper structs (must be declared before methods that use them)
    struct GpuTimestampStageTimes {
        double begin_ns = -1.0;
        double end_ns = -1.0;

        bool is_complete() const { return begin_ns >= 0.0 && end_ns >= 0.0; }
    };
    struct GpuTimestampFrameStages {
        GpuTimestampStageTimes binning;
        GpuTimestampStageTimes raster;
        GpuTimestampStageTimes overlap_count;
        GpuTimestampStageTimes prefix;
        GpuTimestampStageTimes resolve;
        GpuTimestampStageTimes total;
    };
    struct GpuTimestampDurations {
        uint64_t serial = UINT64_MAX;
        double bin_ms = 0.0;
        double raster_ms = 0.0;
        double count_ms = 0.0;
        double prefix_ms = 0.0;
        double resolve_ms = 0.0;
        double total_ms = 0.0;
        bool has_data = false;
    };

    void _parse_timestamps_into_frame_map(RenderingDevice *p_device, uint32_t p_available,
            HashMap<uint64_t, GpuTimestampFrameStages> &r_frames) const;
    GpuTimestampDurations _compute_stage_durations(const HashMap<uint64_t, GpuTimestampFrameStages> &p_frames) const;
    void _update_timing_metrics(const GpuTimestampDurations &p_durations);
    void _issue_async_tile_count_readback(RenderingDevice *p_device, uint32_t p_counts_bytes);
    void _compute_density_metrics(const uint32_t *p_counts, uint32_t p_process_tiles, uint32_t p_total_tiles,
            uint32_t p_effective_capacity, uint32_t *p_density_write,
            RenderStats &r_stats, DensityMetrics &r_density_metrics);
    RenderStats _build_render_stats_from_cached_counts();
    void _update_adaptive_state(const RenderStats &p_stats);
    static bool _is_main_rendering_device(RenderingDevice *p_device);
    RenderingDevice *_get_contract_main_device() const {
        return resource_controller.get_contract_main_device();
    }
	RenderingDevice *_get_resource_device() const;
	RenderingDevice *_get_submission_device();
	RenderingDevice *_acquire_submission_device();
	void _invalidate_descriptor_cache();
	uint64_t descriptor_generation = 0; // Monotonic counter; incremented by _invalidate_descriptor_cache().
	bool _ensure_param_uniform_buffer(RenderingDevice *p_device);
	RID _get_default_state_uniform(RenderingDevice *p_device);
	std::function<void()> output_invalidation_callback;
	bool _ensure_prefix_param_buffer(RenderingDevice *p_device, uint32_t p_size);
	uint32_t _clamp_overlap_record_budget(uint32_t p_requested) const;
	uint32_t _get_effective_overlap_capacity(uint32_t p_capacity_hint = 0) const;
    SortKeyConfig _get_effective_sort_key_config() const;
    Vector<String> _build_common_shader_defines(bool p_include_dispatch_group) const;
    Vector<String> _build_binning_shader_defines() const;
    Vector<String> _build_raster_shader_defines() const;
    void _ensure_global_projection_buffer(uint32_t p_visible_count);
    void _ensure_global_sort_resources(uint32_t p_visible_count);
	bool _update_global_tile_ranges(const RID &p_gaussian_buffer, const RID &p_sorted_indices, RenderingDevice *p_device,
			uint32_t &r_record_count, uint32_t &r_raw_record_count, bool p_allow_sync_readback);
    bool _validate_tile_grid(const char *p_context) const;
    bool _validate_resource_owner(const RID &p_resource, const BufferOwnership &p_owner, RenderingDevice *p_device,
            const char *p_label) const;
    bool _validate_local_owner(const RID &p_resource, RenderingDevice *p_owner_device, RenderingDevice *p_expected_device,
            const char *p_label) const;
    // Cross-device ownership guard (ISSUE-002): verifies ALL resources in a
    // uniform set belong to p_rd before binding. Returns false if any resource
    // is owned by a different device (prevents GPU hang from cross-device binding).
    static bool _verify_texture_device_ownership(RenderingDevice *p_rd, RID p_resource, const char *p_label);
    static bool _verify_buffer_device_ownership(RenderingDevice *p_rd, RID p_resource, const char *p_label);
    struct RasterDecision {
        bool use_compute = false;
        String reason;
    };

    RasterDecision _evaluate_raster_path(RenderingDevice *p_device) const;
    void _log_raster_path_decision(const RasterDecision &p_decision);

private:
    #include "tile_render_stages.h"
    #include "tile_render_async_readback.h"
    #include "tile_render_adaptive_controller.h"

    // Buffers
    TileRendererDebugStats debug_stats;
    TileRenderParamsBuilder params_builder;
    TilePrefixScanStage prefix_scan_stage;
    TileBinningStage binning_stage;
    TileRasterizerStage raster_stage;
    TileResolveStage resolve_stage;
    TileAdaptiveController adaptive_controller;
    TileAsyncReadback async_readback;
    TileGlobalSortResources global_sort_resources;
    TileUniformBuffers uniform_buffers;
    RID last_param_uniform_buffer;
    uint32_t last_param_hash = 0;
    bool last_param_hash_valid = false;
    TileProjectionBuffers projection_buffers;
    TileSHCacheBuffers sh_cache_buffers;
    TileSubpixelHistoryBuffers subpixel_history_buffers;
    TileSubpixelVisibilityBuffers subpixel_visibility_buffers;
    std::unique_ptr<ShaderCompilationManager> shader_compilation_manager;
    TileShaderResources shader_resources;
    HashMap<uint64_t, bool> subgroup_support_cache;
    TileTimingState timing_state;
    TilePerformanceMetrics perf_metrics;
    TileDiagnosticsState diagnostics;
    TileConfigState config_state;
    TileGridState grid_state;
    TileRenderSettings render_settings;
    AdaptiveOverlapBudgetRuntimeState adaptive_overlap_budget_runtime_state;
    bool adaptive_overlap_budget_runtime_state_initialized = false;
    TileResourceController resource_controller;
    struct InstancePipelineBindings {
        RID instance_buffer;
        RID splat_ref_buffer;
        RID quantization_buffer;
        RID indirect_count_buffer;
        RID indirect_dispatch_buffer;
    };
    InstancePipelineBindings instance_pipeline_buffers;
    bool sh_cache_needs_full_update = false;
    TileFrameState frame_state;
    TileDeviceContext device_context;
    bool gpu_timestamp_capture_enabled = true;
    TileRenderTargets render_targets;

    // Cache keys for binning/raster uniform sets live in stage structs.
    // Depth buffer for post-processing compatibility (Issue #128) lives in TileRenderTargets.
    // projection buffer state lives in TileProjectionBuffers.

    // Descriptor caching

    // Performance metrics

    void _on_overflow_flag_readback(const Vector<uint8_t> &p_data, int64_t p_request_frame_serial);
    void _on_debug_counters_readback(const Vector<uint8_t> &p_data);
    void _on_overflow_stats_readback(const Vector<uint8_t> &p_data);
    void _on_splat_audit_readback(const Vector<uint8_t> &p_data);
    void _on_tile_counts_readback(const Vector<uint8_t> &p_data, int64_t p_request_frame_serial);
};

#endif // TILE_RENDERER_H
