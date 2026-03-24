#ifndef GAUSSIAN_RENDER_STREAMING_ORCHESTRATOR_H
#define GAUSSIAN_RENDER_STREAMING_ORCHESTRATOR_H

#include "gaussian_splat_renderer.h"
#include <cstdint>

class RenderDataOrchestrator;
class RenderDeviceOrchestrator;
class GaussianSplatSceneDirector;

struct RenderStreamingOrchestratorDependencies {
	GaussianSplatRenderer *renderer = nullptr;
	RenderDataOrchestrator *data_orchestrator = nullptr;
	RenderDeviceOrchestrator *device_orchestrator = nullptr;

	struct RuntimePorts {
		bool (GaussianSplatRenderer::*ensure_rendering_device)(const char *p_context) = &GaussianSplatRenderer::ensure_rendering_device;
		Projection (GaussianSplatRenderer::*build_cull_projection)(RenderDataRD *p_render_data, const Projection &p_projection) const = &GaussianSplatRenderer::build_cull_projection;
		bool (GaussianSplatRenderer::*validate_cull_projection_contract)(RenderDataRD *p_render_data, const Projection &p_projection,
				const Projection &p_cull_projection, const char *p_context) = &GaussianSplatRenderer::validate_cull_projection_contract;
		void (GaussianSplatRenderer::*clear_instance_pipeline_buffers)() = &GaussianSplatRenderer::clear_instance_pipeline_buffers;
		bool (GaussianSplatRenderer::*update_instance_buffer)(LocalVector<InstanceDataGPU> &p_instances) = &GaussianSplatRenderer::update_instance_buffer;
		void (GaussianSplatRenderer::*run_cull_sort_pipeline_frame)(RenderDataRD *p_render_data, const Transform3D &p_world_to_camera_transform,
				const Projection &p_projection, const Projection &p_render_projection, RenderSceneBuffersRD *p_render_buffers,
				bool p_has_render_data, const String &p_cull_skip_reason, const String &p_sort_skip_reason,
				GaussianSplatRenderer::RenderFallbackReason p_cull_skip_reason_code,
				GaussianSplatRenderer::RenderFallbackReason p_sort_skip_reason_code,
				bool p_set_skip_metrics, bool p_clear_cull_state_on_skip) = &GaussianSplatRenderer::run_cull_sort_pipeline_frame;
		float (GaussianSplatRenderer::*get_cull_radius_multiplier)() const = &GaussianSplatRenderer::get_cull_radius_multiplier;
		float (GaussianSplatRenderer::*get_cull_frustum_plane_slack)() const = &GaussianSplatRenderer::get_cull_frustum_plane_slack;
	};

	RuntimePorts runtime_ports;
};

class RenderStreamingOrchestrator {
public:
	explicit RenderStreamingOrchestrator(const RenderStreamingOrchestratorDependencies &p_dependencies);

	bool ensure_instance_streaming_system();
	void sync_instance_pipeline_assets(GaussianStreamingSystem *p_streaming_system);
	bool render_streaming_frame(RenderDataRD *p_render_data, const Transform3D &p_camera_to_world_transform,
			const Transform3D &p_world_to_camera_transform, const Projection &p_projection,
			const Projection &p_render_projection, RenderSceneBuffersRD *p_render_buffers,
			bool p_allow_runtime_fallback_instance = false);
	void tick_streaming_only(const Transform3D &p_camera_to_world_transform, const Projection &p_projection);
	bool should_throttle_streaming_rebuild(uint32_t p_chunks_loaded, uint32_t p_chunks_evicted,
			uint32_t p_visible_evicted, uint64_t p_current_frame);

	struct VisibleLODSelection {
		uint32_t residency_request_count = 0;
		bool has_instances() const { return !instances.is_empty(); }
		const LocalVector<InstanceDataGPU> &get_instances() const { return instances; }
		LocalVector<InstanceDataGPU> &_internal_get_instances() { return instances; }

	private:
		LocalVector<InstanceDataGPU> instances;
	};

private:
	GaussianSplatRenderer *renderer = nullptr;
	RenderDataOrchestrator *data_orchestrator = nullptr;
	RenderDeviceOrchestrator *device_orchestrator = nullptr;
	RenderStreamingOrchestratorDependencies::RuntimePorts runtime_ports;

	HashSet<uint32_t> instance_pipeline_assets;
	HashSet<uint32_t> instance_pipeline_assets_next;
	LocalVector<uint32_t> instance_pipeline_assets_to_remove;
	LocalVector<InstanceAssetRegistration> instance_pipeline_assets_cache;
	LocalVector<InstanceDataGPU> instance_pipeline_instance_cache;
	HashMap<uint32_t, uint32_t> instance_pipeline_lod_mask_cache;
	HashMap<uint32_t, uint32_t> instance_pipeline_asset_versions;
	uint32_t instance_pipeline_empty_asset_sync_streak = 0;

	uint64_t static_layout_cache_revision = 0;
	Vector<GaussianStreamingSystem::ChunkLayoutHint> static_layout_cache_hints;
	bool static_layout_cache_valid = false;
	bool static_layout_warned_non_contiguous = false;
	uint64_t static_layout_fallback_total = 0;
	uint64_t static_layout_fallback_io_total = 0;
	uint64_t static_layout_fallback_primary_total = 0;
	HashMap<String, uint64_t> static_layout_fallback_reason_counts;
	HashMap<String, uint64_t> static_layout_fallback_category_counts;
	String static_layout_fallback_last_usage = "none";
	String static_layout_fallback_last_reason = "none";
	String static_layout_fallback_last_reason_category = "other";
	String static_layout_fallback_last_context = "none";
	String static_layout_fallback_last_detail = "none";
	uint32_t static_layout_bound_asset_id = UINT32_MAX;
	Vector<GaussianStreamingSystem::ChunkLayoutHint> primary_layout_cache_hints;
	Vector<uint32_t> primary_layout_cache_source_indices;
	bool primary_layout_cache_valid = false;
	bool primary_layout_warned_invalid = false;

	VisibleLODSelection visible_lod_selection;

	const VisibleLODSelection &produce_visible_lod_selection(GaussianSplatSceneDirector *p_director,
			GaussianStreamingSystem *p_streaming_system, const Vector3 &p_camera_origin);
	void consume_visible_lod_selection_for_residency(const VisibleLODSelection &p_selection,
			GaussianStreamingSystem *p_streaming_system, bool p_trace_enabled);
};

#endif
