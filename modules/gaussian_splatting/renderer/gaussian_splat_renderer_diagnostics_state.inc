// Diagnostics state definitions for GaussianSplatRenderer.
// Included inside the class declaration to keep diagnostics types grouped.
//
// See also: render_types/render_frame_types.h for standalone types.

    // Import types from namespace for class-internal use
    using RuntimeErrorStatistics = GaussianRenderFrame::RuntimeErrorStatistics;
    using TextureTraceEntry = GaussianRenderFrame::TextureTraceEntry;
    using CrossDeviceOperation = GaussianRenderFrame::CrossDeviceOperation;
    using FrameTimingSample = GaussianRenderFrame::FrameTimingSample;

    struct DiagnosticsState {
        RuntimeErrorStatistics runtime_error_statistics;
        ErrorRecoveryStateMachine recovery_state_machine;
        Vector<TextureTraceEntry> texture_allocation_trace;
        Vector<CrossDeviceOperation> cross_device_operations;
        Vector<FrameTimingSample> frame_timing_history;
        Dictionary last_telemetry_snapshot;
        Dictionary last_production_metrics;
        Dictionary last_production_metrics_validation;
        Dictionary last_perf_gate_result;
        Vector<Dictionary> production_metrics_summaries;
        uint64_t production_metrics_window_start_frame = 0;
        uint64_t production_metrics_window_start_usec = 0;
        uint32_t production_metrics_window_frames = 0;
        double production_metrics_frame_ms_sum = 0.0;
        double production_metrics_cull_ms_sum = 0.0;
        double production_metrics_sort_ms_sum = 0.0;
        double production_metrics_raster_ms_sum = 0.0;
        double production_metrics_composite_ms_sum = 0.0;
        double production_metrics_stage_total_ms_sum = 0.0;
        double production_metrics_frame_ms_peak = 0.0;
        double production_metrics_stage_ms_peak = 0.0;
        uint32_t production_metrics_visible_peak = 0;
        uint64_t production_metrics_visible_sum = 0;
        uint32_t production_metrics_perf_gate_checks = 0;
        uint32_t production_metrics_perf_gate_failures = 0;
        uint64_t production_metrics_invalid_count = 0;
        bool runtime_diagnostics_requested = false;
    };
