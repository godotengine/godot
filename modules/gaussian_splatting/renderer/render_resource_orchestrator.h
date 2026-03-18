#ifndef GAUSSIAN_RENDER_RESOURCE_ORCHESTRATOR_H
#define GAUSSIAN_RENDER_RESOURCE_ORCHESTRATOR_H

#include "gaussian_splat_renderer.h"

class RenderResourceOrchestrator {
public:
	RenderResourceOrchestrator(GaussianSplatRenderer *p_renderer, GaussianSplatRenderer::DeviceState *p_device_state);

	GaussianSplatRenderer::PipelineState &get_pipeline_state() { return pipeline_state; }
	const GaussianSplatRenderer::PipelineState &get_pipeline_state() const { return pipeline_state; }
	GaussianSplatRenderer::ResourceState &get_resource_state() { return resource_state; }
	const GaussianSplatRenderer::ResourceState &get_resource_state() const { return resource_state; }

	void initialize_shaders();
	void create_gpu_resources_safe();
	RID load_graphics_shader(const Vector<String> &p_vertex_paths, const Vector<String> &p_fragment_paths);
	void update_gpu_pass_metrics_from_tile_renderer();

private:
	GaussianSplatRenderer::PipelineState pipeline_state;
	GaussianSplatRenderer::ResourceState resource_state;
	GaussianSplatRenderer *renderer = nullptr;
	GaussianSplatRenderer::DeviceState *device_state = nullptr;
};

#endif
