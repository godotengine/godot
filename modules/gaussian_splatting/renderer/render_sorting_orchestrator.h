#ifndef GAUSSIAN_RENDER_SORTING_ORCHESTRATOR_H
#define GAUSSIAN_RENDER_SORTING_ORCHESTRATOR_H

#include "gaussian_splat_renderer.h"
#include "core/math/projection.h"
#include "core/templates/local_vector.h"
#include "core/variant/variant.h"
#include "render_types/render_state_types.h"
#include "rendering_error.h"

#include <functional>

class GPUCuller;
class GPUSortingPipeline;

class RenderSortingOrchestrator {
public:
	using CullForViewFn = std::function<GaussianRenderState::CullStageOutput(const Transform3D &, const Projection &, const Size2i &)>;
	using RecordRenderingErrorFn = std::function<void(const RenderingError &)>;
	using EnsureRenderingDeviceFn = std::function<bool(const char *)>;

	struct Dependencies {
		GaussianSplatRenderer *renderer = nullptr;
		GPUCuller *gpu_culler = nullptr;
		GPUSortingPipeline *sorting_pipeline = nullptr;
		GaussianSplatRenderer::PerformanceSettings *performance_settings = nullptr;
		GaussianSplatRenderer::TestDataState *test_data_state = nullptr;
		GaussianSplatRenderer::DeviceState *device_state = nullptr;
		CullForViewFn cull_for_view;
		RecordRenderingErrorFn record_rendering_error;
		EnsureRenderingDeviceFn ensure_rendering_device;
	};

	explicit RenderSortingOrchestrator(const Dependencies &p_dependencies);

	void refresh_gpu_sorter(const char *p_context);
	void initialize_sorting();
	Array run_sort_benchmark(const PackedInt32Array &p_sizes);
	void benchmark_sorting_performance();
	GaussianRenderState::SortStageSummary sort_gaussians_for_view(const Transform3D &p_world_to_camera_transform,
			GaussianRenderState::IndexDomain p_input_domain = GaussianRenderState::IndexDomain::UNKNOWN);
	void force_sort_for_view(const Transform3D &p_world_to_camera_transform);

	// Sort cache methods (merged from RenderSortCacheOrchestrator)
	void set_static_sort_cache_enabled(bool p_enabled);
	void invalidate_static_chunk_caches(bool p_free_rids);
	bool try_reuse_instance_sort_cache(const Transform3D &p_world_to_camera_transform,
			uint64_t p_content_generation, uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count,
			uint32_t &r_sorted_count);
	void update_instance_sort_cache(const Transform3D &p_world_to_camera_transform,
			uint64_t p_content_generation, uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count,
			uint32_t p_sorted_count);

	const GaussianRenderState::SortingState &get_sorting_state() const { return sorting_state; }
	GaussianRenderState::SortingState &access_sorting_state_mutable() { return sorting_state; }

private:
	// Instance sort cache (merged from RenderSortCacheOrchestrator)
	struct InstanceSortCache {
		Vector3 camera_direction = Vector3();
		Vector3 camera_position = Vector3();
		uint64_t content_generation = 0;
		uint32_t max_visible_splats = 0;
		uint32_t visible_chunk_count = 0;
		uint32_t sorted_count = 0;
		bool valid = false;
	};
	InstanceSortCache instance_sort_cache;

	// CPU sort scratch buffers (reused across frames to avoid allocations)
	struct CpuSortEntry {
		float depth = 0.0f;
		uint32_t index = 0;
		uint32_t source_index = 0;
	};
	LocalVector<uint32_t> cpu_sort_original_indices_scratch;
	LocalVector<float> cpu_sort_original_distances_scratch;
	LocalVector<float> cpu_sort_original_importance_scratch;
	LocalVector<CpuSortEntry> cpu_sort_entries_scratch;

	GaussianRenderState::SortingState sorting_state;
	GaussianSplatRenderer *renderer = nullptr;
	GPUCuller *gpu_culler = nullptr;
	GPUSortingPipeline *sorting_pipeline = nullptr;
	GaussianSplatRenderer::PerformanceSettings *performance_settings = nullptr;
	GaussianSplatRenderer::TestDataState *test_data_state = nullptr;
	GaussianSplatRenderer::DeviceState *device_state = nullptr;
	CullForViewFn cull_for_view;
	RecordRenderingErrorFn record_rendering_error;
	EnsureRenderingDeviceFn ensure_rendering_device_fn;
	Transform3D cached_world_to_camera;
	Transform3D cached_camera_to_world;
	bool cached_camera_to_world_valid = false;
	bool runtime_override_tracking_initialized = false;
	bool last_force_cpu_override = false;
	int last_force_algorithm_override = -1;

	const Transform3D &_get_camera_to_world_cached(const Transform3D &p_world_to_camera_transform);
	bool _try_reuse_instance_sort_cache_with_camera(const Transform3D &p_camera_to_world,
			uint64_t p_content_generation, uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count,
			uint32_t &r_sorted_count);
	void _update_instance_sort_cache_with_camera(const Transform3D &p_camera_to_world,
			uint64_t p_content_generation, uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count,
			uint32_t p_sorted_count);
};

#endif
