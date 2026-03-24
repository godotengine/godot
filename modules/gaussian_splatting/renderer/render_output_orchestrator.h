#ifndef GAUSSIAN_RENDER_OUTPUT_ORCHESTRATOR_H
#define GAUSSIAN_RENDER_OUTPUT_ORCHESTRATOR_H

#include "gaussian_splat_renderer.h"

#include <functional>

class RenderOutputOrchestrator {
public:
	struct RuntimePorts {
		std::function<void()> create_gpu_resources;
		bool (GaussianSplatRenderer::*ensure_rendering_device)(const char *p_context) = &GaussianSplatRenderer::ensure_rendering_device;
		RD::TextureFormat (GaussianSplatRenderer::*get_texture_format)(RenderingDevice *p_device, RID p_texture) const = &GaussianSplatRenderer::get_texture_format;
		void (GaussianSplatRenderer::*set_active_viewport_format)(RD::DataFormat p_format, const char *p_context) = &GaussianSplatRenderer::set_active_viewport_format;
		void (GaussianSplatRenderer::*set_manual_viewport_format)(RD::DataFormat p_format, const char *p_context) = &GaussianSplatRenderer::set_manual_viewport_format;
	};

	struct Dependencies {
		GaussianSplatRenderer *renderer = nullptr;
		OutputCompositor *output_compositor = nullptr;
		PainterlyRenderer *painterly_renderer = nullptr;
		GPUCuller *gpu_culler = nullptr;
		RuntimePorts runtime_ports;
	};

	explicit RenderOutputOrchestrator(const Dependencies &p_dependencies);

	bool copy_final_texture_to_target(RID p_render_target, const Size2i &p_viewport_size);
	void commit_to_render_buffers(RenderDataRD *p_render_data);
	bool render_for_view(const Transform3D &p_world_to_camera_transform, const Projection &p_cam_projection,
			RID p_render_target, const Size2i &p_viewport_size);
	bool was_last_viewport_copy_successful() const;
	Size2i get_last_viewport_copy_source_size() const;
	Size2i get_last_viewport_copy_dest_size() const;
#ifdef TESTS_ENABLED
	bool test_copy_final_output(RID p_source, RID p_destination, const Size2i &p_viewport_size);
#endif

private:
	GaussianSplatRenderer *renderer = nullptr;
	OutputCompositor *output_compositor = nullptr;
	PainterlyRenderer *painterly_renderer = nullptr;
	GPUCuller *gpu_culler = nullptr;
	RuntimePorts runtime_ports;
};

#endif
