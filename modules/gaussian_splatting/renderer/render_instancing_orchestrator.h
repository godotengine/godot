#ifndef GAUSSIAN_RENDER_INSTANCING_ORCHESTRATOR_H
#define GAUSSIAN_RENDER_INSTANCING_ORCHESTRATOR_H

#include "gaussian_splat_renderer.h"

#include <cstdint>
#include <functional>

class RenderInstancingOrchestrator {
public:
	using PrepareRenderFrameContextFn = std::function<void(RenderDataRD *, const Transform3D &, const Projection &,
			const Projection &, bool, GaussianSplatRenderer::RenderFrameContext &)>;
	using RenderSortedSplatsFn = std::function<void(RenderDataRD *, const Transform3D &, const Projection &,
			const Projection &, bool)>;

	struct Dependencies {
		GaussianSplatRenderer *renderer = nullptr;
		OutputCompositor *output_compositor = nullptr;
		RenderPipelineStages *pipeline_stages = nullptr;
		PrepareRenderFrameContextFn prepare_render_frame_context;
		RenderSortedSplatsFn render_sorted_splats;
	};

	explicit RenderInstancingOrchestrator(const Dependencies &p_dependencies);

	void render_instanced(RenderDataRD *p_render_data,
			const GaussianSplatManager::SharedDynamicAssetHandle &p_handle,
			const Transform3D &p_world_to_camera_transform, const Projection &p_projection, const Projection &p_render_projection,
			const LocalVector<Transform3D> &p_instance_transforms);

	enum class InstanceReadinessFailureMode : uint8_t {
		NONE = 0,
		INSTANCE_BACKEND_CONTRACT_UNAVAILABLE,
		INSTANCE_PIPELINE_BUFFERS_UNAVAILABLE,
		INSTANCE_PIPELINE_BUFFERS_INVALID,
	};

	struct InstanceReadinessResult {
		bool ready = false;
		InstanceReadinessFailureMode failure_mode = InstanceReadinessFailureMode::NONE;
	};

	static InstanceReadinessResult evaluate_instance_pipeline_readiness(bool p_streaming_system_ready,
			bool p_has_instance_pipeline_buffers, const GaussianRenderPipeline::InstancePipelineBuffers &p_buffers);

private:
	void _warn_instanced_readiness_failure_once(InstanceReadinessFailureMode p_failure_mode) const;

	GaussianSplatRenderer *renderer = nullptr;
	OutputCompositor *output_compositor = nullptr;
	RenderPipelineStages *pipeline_stages = nullptr;
	PrepareRenderFrameContextFn prepare_render_frame_context;
	RenderSortedSplatsFn render_sorted_splats;
};

#endif
