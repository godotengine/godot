#ifndef GAUSSIAN_RENDER_RESOURCE_ORCHESTRATOR_H
#define GAUSSIAN_RENDER_RESOURCE_ORCHESTRATOR_H

#include "gaussian_splat_renderer.h"

class RenderResourceOrchestrator {
public:
	struct RuntimePorts {
		bool (GaussianSplatRenderer::*ensure_rendering_device)(const char *p_context) = &GaussianSplatRenderer::ensure_rendering_device;
		RenderingDevice *(GaussianSplatRenderer::*get_submission_device)() = &GaussianSplatRenderer::get_submission_device;
		RenderingDevice *(GaussianSplatRenderer::*get_main_rendering_device)() const = &GaussianSplatRenderer::get_main_rendering_device;
		void (GaussianSplatRenderer::*refresh_gpu_sorter)(const char *p_context) = &GaussianSplatRenderer::refresh_gpu_sorter;
		void (GaussianSplatRenderer::*track_resource_owner)(const RID &p_rid, RenderingDevice *p_device, bool p_owned, const char *p_label) = &GaussianSplatRenderer::track_resource_owner;
		void (GaussianSplatRenderer::*free_owned_resource)(RenderingDevice *p_fallback_device, RID &p_rid) = &GaussianSplatRenderer::free_owned_resource;
	};

	struct Dependencies {
		GaussianSplatRenderer *renderer = nullptr;
		GaussianSplatRenderer::DeviceState *device_state = nullptr;
		const GaussianSplatRenderer::PerformanceSettings *performance_settings = nullptr;
		const GaussianSplatRenderer::PainterlyConfig *painterly_config = nullptr;
		const GaussianSplatRenderer::DebugConfig *debug_config = nullptr;
		GaussianSplatRenderer::TestDataState *test_data_state = nullptr;
		GaussianSplatRenderer::TileRendererState *tile_renderer_state = nullptr;
		GaussianSplatRenderer::SubsystemState *subsystem_state = nullptr;
		PipelineFeatureSet *pipeline_features_effective = nullptr;
		String *pipeline_features_warning_cache = nullptr;
		RuntimePorts runtime_ports;
	};

	explicit RenderResourceOrchestrator(const Dependencies &p_dependencies);

	GaussianSplatRenderer::PipelineState &get_pipeline_state() { return pipeline_state; }
	const GaussianSplatRenderer::PipelineState &get_pipeline_state() const { return pipeline_state; }
	GaussianSplatRenderer::ResourceState &get_resource_state() { return resource_state; }
	const GaussianSplatRenderer::ResourceState &get_resource_state() const { return resource_state; }

	void initialize_shaders();
	void create_gpu_resources_safe();
	RID load_graphics_shader(const Vector<String> &p_vertex_paths, const Vector<String> &p_fragment_paths);
	void update_gpu_pass_metrics_from_tile_renderer();
	void update_pipeline_features(RenderingDevice *p_device);

private:
	GaussianSplatRenderer::PipelineState pipeline_state;
	GaussianSplatRenderer::ResourceState resource_state;
	GaussianSplatRenderer *renderer = nullptr;
	GaussianSplatRenderer::DeviceState *device_state = nullptr;
	const GaussianSplatRenderer::PerformanceSettings *performance_settings = nullptr;
	const GaussianSplatRenderer::PainterlyConfig *painterly_config = nullptr;
	const GaussianSplatRenderer::DebugConfig *debug_config = nullptr;
	GaussianSplatRenderer::TestDataState *test_data_state = nullptr;
	GaussianSplatRenderer::TileRendererState *tile_renderer_state = nullptr;
	GaussianSplatRenderer::SubsystemState *subsystem_state = nullptr;
	PipelineFeatureSet *pipeline_features_effective = nullptr;
	String *pipeline_features_warning_cache = nullptr;
	RuntimePorts runtime_ports;
};

#endif
