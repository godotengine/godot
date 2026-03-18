#ifndef GAUSSIAN_RENDER_DATA_ORCHESTRATOR_H
#define GAUSSIAN_RENDER_DATA_ORCHESTRATOR_H

#include "../core/gaussian_streaming.h"
#include "../interfaces/gpu_culler.h"
#include "render_types/render_state_types.h"

#include <functional>

class GaussianSplatRenderer;
class RenderingDevice;

class RenderDataOrchestrator {
public:
	using ReleaseSharedDynamicAssetFn = std::function<void()>;
	using AcquireRenderingDeviceFn = std::function<RenderingDevice *()>;
	using InvalidateStaticChunkCachesFn = std::function<void(bool)>;

	RenderDataOrchestrator(GaussianSplatRenderer *p_renderer,
			ReleaseSharedDynamicAssetFn p_release_shared_dynamic_asset,
			AcquireRenderingDeviceFn p_acquire_rendering_device,
			InvalidateStaticChunkCachesFn p_invalidate_static_chunk_caches);

	Error set_gaussian_data(const Ref<::GaussianData> &p_data);
	void set_gaussian_asset(const Ref<GaussianSplatAsset> &p_asset);
	Error update_gpu_buffers_with_real_data();
	void set_static_chunks(const Vector<StaticChunk> &p_chunks);
	void clear_static_chunks();
	void set_streaming_config_overrides(const GaussianStreamingSystem::ConfigOverrides &p_overrides);
	const GaussianRenderState::SceneState &get_scene_state() const { return scene_state; }
	GaussianRenderState::SceneState &access_scene_state_mutable() { return scene_state; }
	const GaussianRenderState::StreamingState &get_streaming_state() const { return streaming_state; }
	GaussianRenderState::StreamingState &access_streaming_state_mutable() { return streaming_state; }
	const GaussianStreamingSystem::ConfigOverrides &get_streaming_config_overrides() const { return streaming_config_overrides; }

private:
	GaussianSplatRenderer *renderer = nullptr;
	GaussianRenderState::SceneState scene_state;
	GaussianRenderState::StreamingState streaming_state;
	GaussianStreamingSystem::ConfigOverrides streaming_config_overrides;
	ReleaseSharedDynamicAssetFn release_shared_dynamic_asset;
	AcquireRenderingDeviceFn acquire_rendering_device;
	InvalidateStaticChunkCachesFn invalidate_static_chunk_caches;
};

#endif
