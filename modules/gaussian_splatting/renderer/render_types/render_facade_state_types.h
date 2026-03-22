/**
 * @file render_facade_state_types.h
 * @brief Remaining GaussianSplatRenderer facade state carrier types.
 */

#ifndef GAUSSIAN_RENDER_FACADE_STATE_TYPES_H
#define GAUSSIAN_RENDER_FACADE_STATE_TYPES_H

#include "core/math/color.h"
#include "core/math/vector3.h"
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/rid.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "../gaussian_gpu_layout.h"
#include "../gpu_buffer_manager.h"
#include "../gpu_performance_monitor.h"
#include <cstdint>
#include <memory>

class RendererSceneRenderRD;
class TileRenderer;
class RenderDeviceManager;
class DebugOverlaySystem;
class InteractiveStateManager;
class TileRasterizer;
class GPUCuller;
class OutputCompositor;
class GPUSortingPipeline;
class OverflowAutoTuner;
class PainterlyRenderer;
class PainterlyMaterialManager;
class GsShadowBlitShaderRD;

namespace GaussianRenderFacadeState {

struct DeviceState {
	RendererSceneRenderRD *scene_render = nullptr;
	RenderingDevice *rd = nullptr;
	bool reported_missing_submission_device = false;
	bool reported_missing_render_device = false;
};

struct ResourceState {
	bool gpu_resources_initialized = false;
	bool gpu_initialization_pending = false;
	Ref<GPUBufferManager> buffer_manager;
	bool buffer_manager_initialized = false;
	GPUBufferManager::DeferredDeletionQueue deletion_queue;
	RID instance_buffer;
	uint32_t instance_buffer_capacity = 0;
	RID instance_visible_chunk_buffer;
	uint32_t instance_visible_chunk_capacity = 0;
	RID instance_splat_ref_buffer;
	uint32_t instance_splat_ref_capacity = 0;
	RID instance_counter_buffer;
	RID instance_chunk_dispatch_buffer;
	RID instance_indirect_count_buffer;
	RID instance_count_buffer;
	uint64_t instance_pipeline_content_generation = 0;
};

struct TestDataState {
	LocalVector<Vector3> positions;
	LocalVector<Color> colors;
	LocalVector<Vector3> scales;
	LocalVector<Object *> mesh_instances;
	RID vertex_buffer;
	RID position_buffer;
	RID scale_buffer;
	RID rotation_buffer;
	RID sh_buffer;
	uint64_t content_generation = 0;
	uint64_t uploaded_generation = 0;
	uint32_t uploaded_count = 0;
};

struct TileRendererState {
	Ref<TileRenderer> renderer;
	GPUPerformanceMonitor gpu_performance_monitor;
	bool init_failed = false;
};

struct SubsystemState {
	Ref<RenderDeviceManager> device_manager;
	Ref<DebugOverlaySystem> debug_overlay_system;
	Ref<InteractiveStateManager> interactive_state_manager;
	Ref<TileRasterizer> rasterizer;
	Ref<GPUCuller> gpu_culler;
	Ref<class OutputCompositor> output_compositor;
	Ref<class GPUSortingPipeline> sorting_pipeline;
	Ref<class OverflowAutoTuner> overflow_auto_tuner;
	Ref<PainterlyRenderer> painterly_renderer;
	Ref<class PainterlyMaterialManager> painterly_material_manager;
};

struct ShadowBlitState {
	bool shader_source_initialized = false;
	uint64_t device_id = 0;
	std::unique_ptr<GsShadowBlitShaderRD> shader_source;
	RID shader;
	PipelineCacheRD pipeline_cache;
	RID sampler;
	GaussianSplatting::BufferOwnership sampler_owner;

	void clear(RenderingDevice *p_device);
};

} // namespace GaussianRenderFacadeState

#endif // GAUSSIAN_RENDER_FACADE_STATE_TYPES_H
