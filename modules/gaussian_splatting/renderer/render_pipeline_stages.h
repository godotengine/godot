#ifndef GAUSSIAN_RENDER_PIPELINE_STAGES_H
#define GAUSSIAN_RENDER_PIPELINE_STAGES_H

#include "gaussian_splat_renderer.h"

#include <memory>

class RenderDebugStateOrchestrator;
class RenderDiagnosticsOrchestrator;

class RenderPipelineStages {
public:
	explicit RenderPipelineStages(GaussianSplatRenderer *p_renderer);
	~RenderPipelineStages();

	using StageResult = GaussianSplatRenderer::StageResult;
	using DataSourcePlan = GaussianSplatRenderer::DataSourcePlan;
	using RenderFramePlan = GaussianSplatRenderer::RenderFramePlan;
	using RenderFrameContext = GaussianSplatRenderer::RenderFrameContext;
	using RenderFallbackReason = GaussianSplatRenderer::RenderFallbackReason;
	using StageMetrics = GaussianSplatRenderer::StageMetrics;
	using SceneState = GaussianSplatRenderer::SceneState;
	using StreamingState = GaussianSplatRenderer::StreamingState;
	using SortingState = GaussianSplatRenderer::SortingState;
	using InstanceBackendPolicy = GaussianSplatRenderer::InstanceBackendPolicy;
	using InstancePipelineBuffers = GaussianSplatRenderer::InstancePipelineBuffers;
	using ResourceState = GaussianSplatRenderer::ResourceState;
	using SubsystemState = GaussianSplatRenderer::SubsystemState;
	using PerformanceMetrics = GaussianSplatRenderer::PerformanceMetrics;
	using PipelineFeatureSet = ::PipelineFeatureSet;

	// Frame planning static helpers (moved from GaussianSplatRenderer)
	static DataSourcePlan build_data_source_plan(const SceneState &p_scene_state,
			const StreamingState &p_streaming_state,
			const SortingState &p_sorting_state,
			const InstancePipelineBuffers *p_instance_buffers,
			InstanceBackendPolicy p_instance_backend_policy,
			const ResourceState &p_resource_state,
			const SubsystemState &p_subsystem_state);
	static void apply_data_source_plan(const DataSourcePlan &p_plan, PerformanceMetrics &p_metrics,
			const ResourceState &p_resource_state);
	static RenderFramePlan build_frame_plan(const SceneState &p_scene_state,
			const StreamingState &p_streaming_state,
			const SortingState &p_sorting_state,
			const InstancePipelineBuffers *p_instance_buffers,
			InstanceBackendPolicy p_instance_backend_policy,
			const ResourceState &p_resource_state,
			const SubsystemState &p_subsystem_state,
			const PipelineFeatureSet *p_pipeline_features,
			bool p_has_render_data,
			const String &p_cull_skip_reason,
			const String &p_sort_skip_reason,
			RenderFallbackReason p_cull_skip_reason_code,
			RenderFallbackReason p_sort_skip_reason_code,
			bool p_set_skip_metrics,
			bool p_clear_cull_state_on_skip);

	// Frame context preparation and entry (moved from GaussianSplatRenderer)
	void prepare_frame_context(RenderDataRD *p_render_data, const Transform3D &p_world_to_camera_transform,
			const Projection &p_projection, const Projection &p_render_projection, bool p_defer_render_buffers_commit,
			RenderFrameContext &r_context);
	void execute_frame_entry(const RenderFrameContext &p_frame_context,
			bool p_has_render_data, const String &p_cull_skip_reason, const String &p_sort_skip_reason,
			RenderFallbackReason p_cull_skip_reason_code, RenderFallbackReason p_sort_skip_reason_code,
			bool p_set_skip_metrics, bool p_clear_cull_state_on_skip);

	StageResult execute_cull_stage(const GaussianSplatRenderer::CullStageInput &p_input,
			GaussianSplatRenderer::CullStageOutput &r_output);
	StageResult execute_sort_stage(const GaussianSplatRenderer::SortStageInput &p_input,
			GaussianSplatRenderer::SortStageOutput &r_output);
	void render_sorted_splats_with_context(const GaussianSplatRenderer::RenderFrameContext &p_context);
	void reset_render_state_for_frame(const GaussianSplatRenderer::IFrameStateView *p_state_view = nullptr,
			GaussianSplatRenderer::IFrameMutationAccess *p_mutation_access = nullptr);
	void set_debug_state_orchestrator(RenderDebugStateOrchestrator *p_debug_state_orchestrator);
	void set_diagnostics_orchestrator(RenderDiagnosticsOrchestrator *p_diagnostics_orchestrator);

private:
	struct CullStage;
	struct SortStage;
	struct RasterStage;
	struct CompositeStage;
	struct RasterCompositeStage;

	GaussianSplatRenderer *renderer = nullptr;
	RenderDebugStateOrchestrator *debug_state_orchestrator = nullptr;
	RenderDiagnosticsOrchestrator *diagnostics_orchestrator = nullptr;
	std::unique_ptr<CullStage> cull_stage;
	std::unique_ptr<SortStage> sort_stage;
	std::unique_ptr<RasterStage> raster_stage;
	std::unique_ptr<CompositeStage> composite_stage;
	std::unique_ptr<RasterCompositeStage> raster_composite_stage;

	GaussianSplatRenderer::CullStageOutput cull_for_view(const Transform3D &p_world_to_camera_transform,
			const Projection &p_projection, const Size2i &p_viewport_size);
	GaussianSplatRenderer::SortStageSummary sort_for_view(const Transform3D &p_world_to_camera_transform,
			GaussianSplatRenderer::IndexDomain p_input_domain);
	void reset_debug_overlay_metrics(float p_sort_ms);
	void store_stage_metrics(const GaussianSplatRenderer::StageMetrics &p_metrics);
	void clear_stage_metrics();
	void increment_frame_counter();
	void finalize_frame_metrics(uint64_t p_frame_start_usec);

	bool execute_raster_composite_pipeline(const GaussianSplatRenderer::RenderFrameContext &p_context,
			uint64_t p_frame_start_usec, GaussianSplatRenderer::RasterStageOutput &r_raster_output,
			StageResult &r_raster_result, StageResult &r_composite_result, float &r_composite_time_ms,
			bool &r_composite_executed);

	void log_stage_result(const char *p_stage_label, const StageResult &p_result) const;
};

#endif
