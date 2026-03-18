// Debug state definitions for GaussianSplatRenderer.
// Included inside the class declaration to keep debug types grouped.
//
// See also: render_types/render_debug_types.h for standalone types.

    // Import SplatAuditSummary from namespace for class-internal use
    using SplatAuditSummary = GaussianRenderDebug::SplatAuditSummary;

    // Debug state types (public for orchestrator access)
    struct DebugConfig {
        bool gpu_sort_readback_enabled = false; // Enable for debug readback; default off to avoid stalls
        bool dump_gpu_counters = false;
        bool enable_pipeline_trace = false;
        bool enable_state_guardrails = false;
        bool enable_splat_audit = false;
        bool enable_all_debug = false;
        bool enable_frame_logging = false;
        bool enable_frame_logging_verbose = false;
        int frame_log_frequency = 300;
        bool enable_sort_path_logs = false;
        bool enable_tile_logs = false;
        bool enable_tile_pipeline_logs = false;
        bool enable_tile_dispatch_logs = false;
        bool enable_gpu_counter_logs = false;
        bool enable_binning_counters = false;
        bool enable_cull_counters = false;
        bool enable_autotune_logs = false;
        bool enable_data_logging = false;
        bool enable_cull_guardrails = false;
        int splat_audit_sample_count = 64;
        int cull_guardrail_min_visible = 256;
        float cull_guardrail_position_epsilon = 0.05f;
        float cull_guardrail_rotation_epsilon = 0.01f;
        float cull_guardrail_drop_ratio = 0.75f;
        float overlay_opacity = 0.3f;
        float heatmap_exponent = 1.0f;
        GaussianSplatting::ComputeRasterPolicy compute_raster_policy_override = GaussianSplatting::ComputeRasterPolicy::Default;
        Vector2 hud_origin = Vector2(16.0f, 16.0f);
        float hud_scale = 1.0f;
    };

    struct DebugState {
        bool show_tile_bounds = false;
        bool show_splat_coverage = false;
        bool show_overflow_tiles = false;
        bool show_projection_issues = false;
        bool show_white_albedo = false;
        bool show_shadow_opacity = false;
        bool show_tile_grid = false; // DISABLED for perf
        bool show_density_heatmap = false;
        bool show_performance_hud = false;
        bool show_resolve_input = false;
        bool show_resolve_output = false;
        bool show_residency_hud = false;
        bool show_device_boundaries = false;
        bool show_texture_states = false;
        DebugPreviewMode preview_mode = DEBUG_PREVIEW_OFF;
        bool overlay_dirty = false;
        bool hud_dirty = false;
        uint64_t overlay_version = 0;
        uint64_t hud_version = 0;
        Vector<String> hud_lines;
        Vector<uint32_t> tile_density_cache;
        int tile_density_width = 0;
        int tile_density_height = 0;
        uint32_t tile_density_peak = 0;
        float tile_density_average = 0.0f;
        float last_tile_assignment_ms = 0.0f;
        float last_tile_rasterization_ms = 0.0f;
        float last_render_time_ms = 0.0f;
        float last_sort_time_ms = 0.0f;
        String route_uid;
        String sort_route_uid;
        StageMetrics last_stage_metrics;
        bool last_stage_metrics_valid = false;
        SplatAuditSummary splat_audit;
        Vector<PipelineEvent> pipeline_events;
        uint64_t pipeline_events_frame = 0;
        bool pipeline_events_valid = false;
        uint64_t pipeline_trace_generation = 0;
    };
