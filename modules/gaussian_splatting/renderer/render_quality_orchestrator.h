#ifndef GAUSSIAN_RENDER_QUALITY_ORCHESTRATOR_H
#define GAUSSIAN_RENDER_QUALITY_ORCHESTRATOR_H

#include "gaussian_splat_renderer.h"

#include <functional>

// Manages quality settings (LOD, culling parameters, presets) and
// executes the GPU culling pass. Merged from the former
// RenderQualityOrchestrator + RenderCullingOrchestrator (ISSUE-016).
class RenderQualityOrchestrator {
public:
	struct RuntimePorts {
		void (GaussianSplatRenderer::*refresh_gpu_sorter)(const char *p_context) = &GaussianSplatRenderer::refresh_gpu_sorter;
		void (GaussianSplatRenderer::*track_resource_owner)(const RID &p_rid, RenderingDevice *p_device,
				bool p_owned, const char *p_label) = &GaussianSplatRenderer::track_resource_owner;
		GaussianSplatRenderer::StreamingState &(GaussianSplatRenderer::*get_streaming_state_mut)() =
				static_cast<GaussianSplatRenderer::StreamingState &(GaussianSplatRenderer::*)()>(
						&GaussianSplatRenderer::get_streaming_state);
		const GaussianSplatRenderer::StreamingState &(GaussianSplatRenderer::*get_streaming_state_view)() const =
				static_cast<const GaussianSplatRenderer::StreamingState &(GaussianSplatRenderer::*)() const>(
						&GaussianSplatRenderer::get_streaming_state);
	};

	struct Dependencies {
		GaussianSplatRenderer *renderer = nullptr;
		GPUCuller *gpu_culler = nullptr;
		GaussianSplatRenderer::TestDataState *test_data_state = nullptr;
		RuntimePorts runtime_ports;
	};

	explicit RenderQualityOrchestrator(const Dependencies &p_dependencies);

	// Quality / LOD settings
	void set_lod_enabled(bool p_enabled);
	void set_lod_bias(float p_bias);
	void set_lod_min_screen_size(float p_pixels);
	void set_lod_max_distance(float p_distance);
	void set_importance_cull_threshold(float p_threshold);
	void set_cull_radius_multiplier(float p_multiplier);
	void set_cull_frustum_plane_slack(float p_slack);
	void set_cull_near_tolerance(float p_tolerance);
	void set_cull_far_tolerance(float p_tolerance);
	void set_tiny_splat_screen_radius(float p_pixels);
	void set_opacity_aware_culling(bool p_enabled);
	void set_visibility_threshold(float p_threshold);
	void set_distance_cull_enabled(bool p_enabled);
	void set_distance_cull_start(float p_distance);
	void set_distance_cull_max_rate(float p_rate);
	void set_overflow_autotune_enabled(bool p_enabled);
	void set_max_splats(int p_count);
	void set_frustum_culling(bool p_enabled);
	void set_async_upload_enabled(bool p_enabled);
	void set_quality_preset(const String &p_preset);
	String get_quality_preset() const;
	bool get_async_upload_enabled() const;

	// GPU culling pass (absorbed from RenderCullingOrchestrator)
	GaussianRenderState::CullStageOutput cull_for_view(const Transform3D &p_world_to_camera_transform,
			const Projection &p_projection, const Size2i &p_viewport_size);

	GaussianSplatRenderer::PerformanceSettings &get_performance_settings() { return performance_settings; }
	const GaussianSplatRenderer::PerformanceSettings &get_performance_settings() const { return performance_settings; }

private:
	GaussianSplatRenderer::PerformanceSettings performance_settings;
	GaussianSplatRenderer *renderer = nullptr;
	GPUCuller *gpu_culler = nullptr;
	GaussianSplatRenderer::TestDataState *test_data_state = nullptr;
	RuntimePorts runtime_ports;
};

#endif
