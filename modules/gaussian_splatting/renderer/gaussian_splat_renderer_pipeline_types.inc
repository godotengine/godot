// Pipeline helper type definitions for GaussianSplatRenderer.
// Included inside the class declaration to keep pipeline types grouped.
//
// See also: render_types/render_frame_types.h for standalone frame types.

    struct StageMetrics {
        CullStageOutput cull;
        StageResult cull_result;
        SortStageOutput sort;
        StageResult sort_result;
        RasterStageOutput raster;
        StageResult raster_result;
        float composite_time_ms = 0.0f;
        bool composite_executed = false;
        StageResult composite_result;
        StageIO cull_io;
        StageIO sort_io;
        StageIO raster_io;
        StageIO composite_io;
    };

    struct PipelineEvent {
        String stage;
        String message;
        String route_uid;
        uint32_t input_count = 0;
        uint32_t output_count = 0;
        bool is_error = false;
        RenderFallbackReason fallback_reason = RenderFallbackReason::NONE;
    };
