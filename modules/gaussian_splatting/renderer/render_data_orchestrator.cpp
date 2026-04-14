#include "render_data_orchestrator.h"

#include "gaussian_splat_renderer.h"
#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "../interfaces/gpu_sorting_pipeline.h"
#include "../logger/gs_logger.h"

static bool _is_data_orchestrator_log_enabled(const GaussianSplatRenderer::DebugConfig *p_debug_config) {
	if (!p_debug_config) {
		return false;
	}
	return p_debug_config->enable_data_logging ||
			p_debug_config->enable_frame_logging ||
			p_debug_config->enable_all_debug;
}

RenderDataOrchestrator::RenderDataOrchestrator(const Dependencies &p_dependencies) :
		renderer(p_dependencies.renderer),
		debug_config(p_dependencies.debug_config),
		performance_settings(p_dependencies.performance_settings),
		culling_config(p_dependencies.culling_config),
		release_shared_dynamic_asset(p_dependencies.release_shared_dynamic_asset),
		acquire_rendering_device(p_dependencies.acquire_rendering_device),
		invalidate_static_chunk_caches(p_dependencies.invalidate_static_chunk_caches),
		runtime_ports(p_dependencies.runtime_ports) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(debug_config);
	ERR_FAIL_NULL(performance_settings);
	ERR_FAIL_NULL(culling_config);
	ERR_FAIL_COND_MSG(!release_shared_dynamic_asset, "RenderDataOrchestrator requires release callback.");
	ERR_FAIL_COND_MSG(!acquire_rendering_device, "RenderDataOrchestrator requires device acquisition callback.");
	ERR_FAIL_COND_MSG(!invalidate_static_chunk_caches, "RenderDataOrchestrator requires cache invalidation callback.");
	ERR_FAIL_COND_MSG(!runtime_ports.invalidate_cached_render, "RenderDataOrchestrator requires invalidate_cached_render runtime port.");

	GaussianSplatRenderer::FrameStateProvider init_state_provider(renderer);
	const GaussianSplatRenderer::IFrameStateView &init_state_view = init_state_provider;

	streaming_state.memory_stream.instantiate();
	if (streaming_state.memory_stream.is_valid()) {
		streaming_state.memory_stream->set_device_manager(init_state_view.get_subsystem_state_view().device_manager);
	}
	streaming_state.gpu_gaussian_cache.clear();
	streaming_state.gpu_gaussian_cache_buffer = RID();
	streaming_state.gpu_gaussian_cache_start = 0;
	streaming_state.gpu_gaussian_cache_count = 0;
	streaming_state.gpu_gaussian_cache_frame = 0;
	streaming_state.gpu_gaussian_cache_valid = false;
}

void RenderDataOrchestrator::set_gaussian_asset(const Ref<GaussianSplatAsset> &p_asset) {
	ERR_FAIL_COND_MSG(!release_shared_dynamic_asset, "RenderDataOrchestrator missing release callback.");
	if (scene_state.active_asset == p_asset) {
		return;
	}

	if (p_asset.is_null()) {
		scene_state.active_asset.unref();
		release_shared_dynamic_asset();
		return;
	}

	scene_state.active_asset = p_asset;

	if (scene_state.active_asset->get_asset_type() != GaussianSplatAsset::ASSET_TYPE_DYNAMIC) {
		release_shared_dynamic_asset();
	}
}

Error RenderDataOrchestrator::set_gaussian_data(const Ref<::GaussianData> &p_data) {
	const bool log_enabled = _is_data_orchestrator_log_enabled(debug_config);
	if (log_enabled) {
		GS_LOG_RENDERER_DEBUG(vformat("[RDO-SET-DATA] ENTER count=%d", p_data.is_valid() ? p_data->get_count() : -1));
	}
	ObjectID incoming_data_id = p_data.is_valid() ? p_data->get_instance_id() : ObjectID();
	GaussianSplatRenderer::FrameStateProvider state_provider(renderer);
	GaussianSplatRenderer::IFrameMutationAccess &state_mut = state_provider;
	const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
	const GaussianSplatRenderer::SubsystemState &subsystem_state_view = state_view.get_subsystem_state_view();
	GaussianSplatRenderer::SubsystemState &subsystem_state = state_mut.get_subsystem_state_mut();
	GaussianSplatRenderer::PerformanceState &performance_state = state_mut.get_performance_state_mut();
	GaussianSplatRenderer::SortingState &sorting_state = state_mut.get_sorting_state_mut();
	GaussianSplatRenderer::FrameState &frame_state = state_mut.get_frame_state_mut();
	GaussianSplatRenderer::ResourceState &resource_state = state_mut.get_resource_state_mut();

	if (incoming_data_id != streaming_state.registered_gaussian_data_id) {
		release_shared_dynamic_asset();
	}

	if (streaming_state.registered_gaussian_buffer.is_valid() &&
			incoming_data_id != streaming_state.registered_gaussian_data_id) {
		renderer->forget_resource_owner(streaming_state.registered_gaussian_buffer);
		GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
		if (manager) {
			manager->unregister_gaussian_buffer(streaming_state.registered_gaussian_buffer);
		}
		streaming_state.registered_gaussian_buffer = RID();
		streaming_state.registered_gaussian_data_id = ObjectID();
	}

	scene_state.gaussian_data = p_data;
	// Scene data identity changed: never reuse last-frame color/depth outputs.
	(renderer->*runtime_ports.invalidate_cached_render)();

	subsystem_state.gpu_culler->get_state().hierarchical_structure_dirty = true;

	performance_state.metrics.data_source = GaussianSplatRenderer::SplatDataSource::kSourceNone;
	performance_state.metrics.using_real_data = false;
	performance_state.metrics.data_source_error = String();
	performance_state.metrics.uploaded_splat_count = 0;
	performance_state.metrics.raster_path = "unknown";

	if (!scene_state.gaussian_data.is_valid()) {
		// Clearing data resets streaming state and metrics.
		streaming_state.use_streamed_data = false;
		streaming_state.cached_streamed_gaussians.clear();
		streaming_state.cached_streamed_indices.clear();
		streaming_state.cached_streamed_source_indices.clear();
		streaming_state.cached_streamed_sh_limits.clear();
		streaming_state.cached_streamed_index_lookup.clear();
		streaming_state.current_stream_gpu_buffer = RID();
		streaming_state.streaming_gpu_splat_count = 0;
		streaming_state.streaming_gpu_total_capacity = 0;
		streaming_state.streamed_indices_generation = 0;
		streaming_state.streamed_indices_are_local = false;
		streaming_state.cached_streamed_indices_valid = false;
		streaming_state.current_streaming_system.unref();
		sorting_state.sorted_splat_count = 0;
		frame_state.visible_splat_count.store(0, std::memory_order_release);
		clear_static_chunks();
		return OK;
	}

	performance_state.metrics.data_source = GaussianSplatRenderer::SplatDataSource::kSourceCpuData;
	performance_state.metrics.data_source_error = String();
	performance_state.metrics.raster_path = "unknown";

	Error status = OK;

	bool buf_valid = resource_state.buffer_manager.is_valid();
	bool buf_init = resource_state.buffer_manager_initialized;
	if (log_enabled) {
		GS_LOG_RENDERER_DEBUG(vformat("[RenderDataOrch] buffer_manager valid=%s initialized=%s",
				buf_valid ? "yes" : "no", buf_init ? "yes" : "no"));
		GS_LOG_RENDERER_DEBUG(vformat("[RDO-SET-DATA] buf_valid=%s buf_init=%s",
				buf_valid ? "yes" : "no", buf_init ? "yes" : "no"));
	}
	if (buf_valid && buf_init) {
		resource_state.buffer_manager->clear_gaussian_data();
	}

	if (log_enabled) {
		GS_LOG_RENDERER_DEBUG("[RenderDataOrch] status before update_gpu_buffers: " + itos(status));
	}
	if (status == OK) {
		Error init_err = update_gpu_buffers_with_real_data();
		if (log_enabled) {
			GS_LOG_RENDERER_DEBUG(vformat("[SET-DATA-DBG] update_gpu_buffers_with_real_data returned init_err=%d", init_err));
		}
		if (init_err != OK) {
			performance_state.metrics.using_real_data = false;
			status = init_err;
		}
	}
	if (log_enabled) {
		GS_LOG_RENDERER_DEBUG(vformat("[SET-DATA-DBG] After update, status=%d", status));
	}

	if (status == OK && scene_state.gaussian_data.is_valid()) {
		if (log_enabled) {
			GS_LOG_RENDERER_DEBUG("[SET-DATA-DBG] Entering shared dynamic asset handling");
		}
		bool using_shared_dynamic = false;

		if (scene_state.active_asset.is_valid() &&
				scene_state.active_asset->get_asset_type() == GaussianSplatAsset::ASSET_TYPE_DYNAMIC) {
			if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
				streaming_state.shared_dynamic_asset_handle =
						manager->acquire_dynamic_asset(scene_state.active_asset,
							scene_state.gaussian_data,
							state_view.get_rendering_device());
				if (streaming_state.shared_dynamic_asset_handle.is_valid()) {
					using_shared_dynamic = true;
					streaming_state.registered_gaussian_buffer =
						streaming_state.shared_dynamic_asset_handle.gaussian_buffer;
					streaming_state.registered_gaussian_data_id =
						scene_state.gaussian_data->get_instance_id();
				} else {
					streaming_state.shared_dynamic_asset_handle = GaussianSplatManager::SharedDynamicAssetHandle();
				}
			}
		} else {
			streaming_state.shared_dynamic_asset_handle = GaussianSplatManager::SharedDynamicAssetHandle();
		}

		if (!using_shared_dynamic) {
			if (sorting_state.sort_indices_external) {
				sorting_state.sort_indices_external = false;
				sorting_state.sort_buffers_pipeline_managed = false;
				if (subsystem_state_view.sorting_pipeline.is_valid()) {
					subsystem_state_view.sorting_pipeline->clear_external_sort_indices();
				}
			}

			bool needs_registration = !streaming_state.registered_gaussian_buffer.is_valid() ||
					streaming_state.registered_gaussian_data_id !=
					scene_state.gaussian_data->get_instance_id();

			if (needs_registration) {
				if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
					RID shared_buffer = manager->register_gaussian_buffer(scene_state.gaussian_data);
					if (shared_buffer.is_valid()) {
						streaming_state.registered_gaussian_buffer = shared_buffer;
						streaming_state.registered_gaussian_data_id =
							scene_state.gaussian_data->get_instance_id();
					} else {
						streaming_state.registered_gaussian_buffer = RID();
						streaming_state.registered_gaussian_data_id = ObjectID();
					}
				}
			}
		}
	} else {
		release_shared_dynamic_asset();
		streaming_state.current_streaming_system.unref();
	}

	if (status != OK) {
		GS_LOG_ERROR_DEFAULT(vformat("[Hello Splat] Failed to activate gaussian data on GPU: %d", status));
	}

	return status;
}

Error RenderDataOrchestrator::update_gpu_buffers_with_real_data() {
	const bool log_enabled = _is_data_orchestrator_log_enabled(debug_config);
	GaussianSplatRenderer::FrameStateProvider state_provider(renderer);
	GaussianSplatRenderer::IFrameMutationAccess &state_mut = state_provider;
	GaussianSplatRenderer::FrameState &frame_state = state_mut.get_frame_state_mut();
	GaussianSplatRenderer::SortingState &sorting_state = state_mut.get_sorting_state_mut();
	if (log_enabled) {
		GS_LOG_STREAMING_DEBUG("[STREAM-INIT] update_gpu_buffers_with_real_data CALLED");
		GS_LOG_STREAMING_DEBUG("[RenderDataOrch] update_gpu_buffers_with_real_data ENTERED");
	}
	ERR_FAIL_COND_V_MSG(!scene_state.gaussian_data.is_valid(), ERR_INVALID_PARAMETER,
			"Gaussian data must be valid before uploading");

	const int real_count = scene_state.gaussian_data->get_count();
	if (log_enabled) {
		GS_LOG_STREAMING_DEBUG(vformat("[RenderDataOrch] real_count=%d", real_count));
	}
	if (real_count == 0) {
		if (log_enabled) {
			GS_LOG_STREAMING_DEBUG("[RenderDataOrch] early return: real_count==0");
		}
		frame_state.visible_splat_count.store(0, std::memory_order_release);
		streaming_state.current_streaming_system.unref();
		streaming_state.use_streamed_data = false;
		streaming_state.cached_streamed_gaussians.clear();
		streaming_state.cached_streamed_indices.clear();
		streaming_state.cached_streamed_source_indices.clear();
		streaming_state.cached_streamed_sh_limits.clear();
		streaming_state.cached_streamed_index_lookup.clear();
		streaming_state.current_stream_gpu_buffer = RID();
		streaming_state.streaming_gpu_splat_count = 0;
		streaming_state.streaming_gpu_total_capacity = 0;
		streaming_state.streamed_indices_generation = 0;
		streaming_state.streamed_indices_are_local = false;
		streaming_state.cached_streamed_indices_valid = false;
		sorting_state.sorted_splat_count = 0;
		return OK;
	}

	const GaussianSplatRenderer::RuntimeFidelityPolicy runtime_policy =
			renderer->build_runtime_fidelity_policy(scene_state, *performance_settings);
	if (runtime_policy.prefer_resident_backend) {
		streaming_state.current_streaming_system.unref();
		streaming_state.use_streamed_data = false;
		streaming_state.cached_streamed_gaussians.clear();
		streaming_state.cached_streamed_indices.clear();
		streaming_state.cached_streamed_source_indices.clear();
		streaming_state.cached_streamed_sh_limits.clear();
		streaming_state.cached_streamed_index_lookup.clear();
		streaming_state.current_stream_gpu_buffer = RID();
		streaming_state.streaming_gpu_splat_count = 0;
		streaming_state.streaming_gpu_total_capacity = 0;
		streaming_state.streamed_indices_generation = 0;
		streaming_state.streamed_indices_are_local = false;
		streaming_state.cached_streamed_indices_valid = false;
		sorting_state.sorted_splat_count = 0;
		renderer->clear_instance_pipeline_buffers();
		frame_state.visible_splat_count.store(0, std::memory_order_release);
		return OK;
	}

	if (log_enabled) {
		GS_LOG_STREAMING_DEBUG(vformat("[RenderDataOrch] memory_stream valid=%s",
				streaming_state.memory_stream.is_valid() ? "yes" : "no"));
	}
	if (!streaming_state.memory_stream.is_valid()) {
		if (log_enabled) {
			GS_LOG_STREAMING_DEBUG("[RenderDataOrch] early return: memory stream not available");
		}
		GS_LOG_WARN_DEFAULT("[GPU Streaming] Memory stream not available; skipping real data upload");
		streaming_state.current_streaming_system.unref();
		return ERR_UNCONFIGURED;
	}

	RenderingDevice *rd_ptr = acquire_rendering_device();
	if (log_enabled) {
		GS_LOG_STREAMING_DEBUG(vformat("[RenderDataOrch] rd_ptr=%s", rd_ptr ? "valid" : "null"));
	}
	if (!rd_ptr) {
		GS_LOG_WARN_DEFAULT("[GPU Streaming] RenderingDevice unavailable; skipping real data upload");
		streaming_state.current_streaming_system.unref();
		return ERR_UNCONFIGURED;
	}

	// Reset existing streaming resources before re-initializing.
	streaming_state.memory_stream->shutdown();
	streaming_state.current_streaming_system.unref();
	streaming_state.cached_streamed_gaussians.clear();
	streaming_state.cached_streamed_indices.clear();
	streaming_state.cached_streamed_source_indices.clear();
	streaming_state.cached_streamed_sh_limits.clear();
	streaming_state.cached_streamed_index_lookup.clear();
	streaming_state.use_streamed_data = false;
	streaming_state.current_stream_gpu_buffer = RID();
	streaming_state.streaming_gpu_splat_count = 0;
	streaming_state.streaming_gpu_total_capacity = 0;
	streaming_state.streamed_indices_generation = 0;
	streaming_state.streamed_indices_are_local = false;
	streaming_state.cached_streamed_indices_valid = false;
	sorting_state.sorted_splat_count = 0;

	const uint32_t max_streamed = runtime_policy.runtime_budget_splats;
	if (max_streamed == 0) {
		GS_LOG_WARN_DEFAULT("[GPU Streaming] Max splat budget is zero; skipping real data upload");
		return ERR_PARAMETER_RANGE_ERROR;
	}

	Error err = streaming_state.memory_stream->initialize(rd_ptr, max_streamed, 64);
	if (err != OK) {
		GS_LOG_ERROR_DEFAULT(vformat("[GPU Streaming] Memory stream initialization failed: %d", err));
		return err;
	}

	Ref<GaussianStreamingSystem> streaming_system;
	streaming_system.instantiate();
	streaming_system->set_config_overrides(streaming_config_overrides);
	streaming_system->set_chunk_radius_multiplier(
			culling_config->cull_radius_multiplier * culling_config->cull_frustum_plane_slack);
	streaming_system->initialize_with_device(scene_state.gaussian_data, rd_ptr);
	if (streaming_state.pending_payload_source.is_valid()) {
		streaming_system->set_chunk_payload_source(0 /* PRIMARY_ASSET_ID */,
				streaming_state.pending_payload_source);
		// NOTE: detach_source_data deferred — needed for visibility and
		// per-frame chunk source index resolution in streaming pipelines.
		// streaming_system->detach_source_data(0 /* PRIMARY_ASSET_ID */);
	}
	if (log_enabled) {
		GS_LOG_STREAMING_DEBUG(vformat("[RenderDataOrch] streaming_system after initialize: chunks=%d",
				streaming_state.current_streaming_system.is_valid() ? 0 : -1));
	}
	streaming_system->attach_memory_stream(streaming_state.memory_stream);
	streaming_state.current_streaming_system = streaming_system;
	if (log_enabled) {
		GS_LOG_STREAMING_DEBUG(vformat("[STREAM-INIT] Streaming system CREATED valid=%s",
				streaming_state.current_streaming_system.is_valid() ? "yes" : "no"));
		GS_LOG_STREAMING_DEBUG("[STREAM-INIT] Resetting visible_splat_count to 0 until first streaming visibility pass.");
	}
	frame_state.visible_splat_count.store(0, std::memory_order_release);
	if (log_enabled) {
		GS_LOG_STREAMING_DEBUG(vformat("[RenderDataOrch] Streaming system created, max_streamed=%d", max_streamed));
	}

	if (log_enabled) {
		GS_LOG_STREAMING_DEBUG("[STREAM-INIT] update_gpu_buffers_with_real_data returning OK");
	}
	return OK;
}

void RenderDataOrchestrator::set_static_chunks(const Vector<StaticChunk> &p_chunks) {
	GaussianSplatRenderer::FrameStateProvider state_provider(renderer);
	GaussianSplatRenderer::IFrameMutationAccess &state_mut = state_provider;
	GaussianSplatRenderer::SubsystemState &subsystem_state = state_mut.get_subsystem_state_mut();
	invalidate_static_chunk_caches(true);
	auto &cull_state = subsystem_state.gpu_culler->get_state();
	cull_state.static_chunks = p_chunks;
	cull_state.static_chunks_revision++;
	if (cull_state.static_chunks_revision == 0) {
		cull_state.static_chunks_revision = 1;
	}
	cull_state.visible_static_chunk_indices.clear();
}

void RenderDataOrchestrator::clear_static_chunks() {
	GaussianSplatRenderer::FrameStateProvider state_provider(renderer);
	GaussianSplatRenderer::IFrameMutationAccess &state_mut = state_provider;
	GaussianSplatRenderer::SubsystemState &subsystem_state = state_mut.get_subsystem_state_mut();
	invalidate_static_chunk_caches(true);
	auto &cull_state = subsystem_state.gpu_culler->get_state();
	cull_state.static_chunks.clear();
	cull_state.static_chunks_revision++;
	if (cull_state.static_chunks_revision == 0) {
		cull_state.static_chunks_revision = 1;
	}
	cull_state.visible_static_chunk_indices.clear();
}

void RenderDataOrchestrator::set_streaming_config_overrides(
		const GaussianStreamingSystem::ConfigOverrides &p_overrides) {
	streaming_config_overrides = p_overrides;
	if (streaming_state.current_streaming_system.is_valid()) {
		streaming_state.current_streaming_system->set_config_overrides(streaming_config_overrides);
	}
}

Error GaussianSplatRenderer::set_gaussian_data(const Ref<::GaussianData> &p_data) {
	const auto &debug_config = get_debug_config();
	const bool log_enabled = debug_config.enable_data_logging ||
			debug_config.enable_frame_logging ||
			debug_config.enable_all_debug;
	if (log_enabled) {
		GS_LOG_RENDERER_DEBUG(vformat("[GSR-SET-DATA] ENTER count=%d orchestrator=%s",
				p_data.is_valid() ? p_data->get_count() : -1,
				data_orchestrator ? "valid" : "null"));
	}

	RenderingServer *rs = RenderingServer::get_singleton();
	bool dispatch_submitted = false;
	uint64_t dispatched_request_id = 0;
	if (rs && !rs->is_on_render_thread()) {
		if (_dispatch_call_on_render_thread_blocking(
					callable_mp(this, &GaussianSplatRenderer::_set_gaussian_data_on_render_thread).bind(p_data),
					&dispatch_submitted,
					true,
					&dispatched_request_id)) {
			if (log_enabled) {
				GS_LOG_RENDERER_DEBUG(vformat("[GSR-SET-DATA] EXIT err=%d (render-thread dispatch)",
						render_thread_dispatcher ? int(render_thread_dispatcher->get_latest_data_result()) : int(ERR_UNAVAILABLE)));
			}
			return render_thread_dispatcher ? render_thread_dispatcher->get_latest_data_result() : ERR_UNAVAILABLE;
		}
		if (dispatch_submitted) {
			if (render_thread_dispatcher) {
				render_thread_dispatcher->promote_latest_data_request_id(dispatched_request_id);
			}
			if (log_enabled) {
				GS_LOG_RENDERER_WARN("[GSR-SET-DATA] Render-thread dispatch timed out after submit; skipping unsafe local fallback");
			}
			return ERR_BUSY;
		}
	}

	Error err = data_orchestrator->set_gaussian_data(p_data);
	if (render_thread_dispatcher) {
		const uint64_t request_id_floor = render_thread_dispatcher->get_next_request_id();
		render_thread_dispatcher->promote_latest_data_request_id(request_id_floor);
	}
	if (log_enabled) {
		GS_LOG_RENDERER_DEBUG(vformat("[GSR-SET-DATA] EXIT err=%d", err));
	}
	return err;
}

void GaussianSplatRenderer::_set_gaussian_data_on_render_thread(const Ref<::GaussianData> &p_data, uint64_t p_request_id) {
	const uint64_t latest_request_id =
			render_thread_dispatcher ? render_thread_dispatcher->get_latest_data_request_id() : 0;
	if (p_request_id < latest_request_id) {
		_notify_render_thread_dispatch_completed(p_request_id);
		return;
	}
	if (render_thread_dispatcher) {
		render_thread_dispatcher->promote_latest_data_request_id(p_request_id);
	}
	// Re-read after promotion because another request may have advanced the latest ID
	// between our initial stale check and the dispatcher update above. If that happened,
	// this request must still exit as stale and avoid overwriting newer set-data state.
	const uint64_t previous_request_id =
			render_thread_dispatcher ? render_thread_dispatcher->get_latest_data_request_id() : p_request_id;
	if (p_request_id < previous_request_id) {
		_notify_render_thread_dispatch_completed(p_request_id);
		return;
	}
	const Error set_data_result = data_orchestrator->set_gaussian_data(p_data);
	if (render_thread_dispatcher) {
		render_thread_dispatcher->set_latest_data_result(set_data_result);
	}
	_notify_render_thread_dispatch_completed(p_request_id);
}

Error GaussianSplatRenderer::_update_gpu_buffers_with_real_data() {
	return data_orchestrator->update_gpu_buffers_with_real_data();
}

void GaussianSplatRenderer::set_static_chunks(const Vector<StaticChunk> &p_chunks) {
	data_orchestrator->set_static_chunks(p_chunks);
}

void GaussianSplatRenderer::clear_static_chunks() {
	data_orchestrator->clear_static_chunks();
}

void GaussianSplatRenderer::set_streaming_config_overrides(
		const GaussianStreamingSystem::ConfigOverrides &p_overrides) {
	data_orchestrator->set_streaming_config_overrides(p_overrides);
}

GaussianStreamingSystem::ConfigOverrides GaussianSplatRenderer::get_streaming_config_overrides() const {
	return data_orchestrator ? data_orchestrator->get_streaming_config_overrides() : GaussianStreamingSystem::ConfigOverrides();
}

GaussianSplatRenderer::WorldSubmissionRuntimeStateSnapshot GaussianSplatRenderer::snapshot_world_submission_runtime_state() const {
	WorldSubmissionRuntimeStateSnapshot snapshot;
	snapshot.valid = true;
	snapshot.gaussian_data = get_scene_state().gaussian_data;
	if (subsystem_state.gpu_culler.is_valid()) {
		snapshot.static_chunks = subsystem_state.gpu_culler->get_state().static_chunks;
	}
	snapshot.lod_enabled = get_lod_enabled();
	snapshot.lod_bias = get_lod_bias();
	snapshot.lod_max_distance = get_lod_max_distance();
	snapshot.frustum_culling = get_frustum_culling();
	snapshot.async_upload_enabled = get_async_upload_enabled();
	snapshot.opacity_multiplier = get_opacity_multiplier();
	snapshot.max_splats = get_max_splats();
	snapshot.streaming_overrides = get_streaming_config_overrides();
	snapshot.has_active_world_submission = world_submission_contract_active;
	snapshot.has_desired_residency_hint = world_submission_has_residency_hint;
	snapshot.desired_residency_hint = world_submission_residency_hint;
	return snapshot;
}

Error GaussianSplatRenderer::restore_world_submission_runtime_state(const WorldSubmissionRuntimeStateSnapshot &p_snapshot) {
	if (!p_snapshot.valid) {
		return ERR_INVALID_PARAMETER;
	}

	set_lod_enabled(p_snapshot.lod_enabled);
	set_lod_bias(p_snapshot.lod_bias);
	set_lod_max_distance(p_snapshot.lod_max_distance);
	set_frustum_culling(p_snapshot.frustum_culling);
	set_async_upload_enabled(p_snapshot.async_upload_enabled);
	set_opacity_multiplier(p_snapshot.opacity_multiplier);
	set_streaming_config_overrides(p_snapshot.streaming_overrides);
	set_max_splats(MAX(1, p_snapshot.max_splats));

	const bool has_renderable_data =
			p_snapshot.gaussian_data.is_valid() && p_snapshot.gaussian_data->get_count() > 0;
	const bool has_active_world_submission = p_snapshot.has_active_world_submission && has_renderable_data;
	const bool has_residency_hint = has_active_world_submission && p_snapshot.has_desired_residency_hint;

	if (p_snapshot.gaussian_data.is_valid()) {
		const auto &resource_state = get_resource_state();
		if (!resource_state.gpu_resources_initialized && !resource_state.gpu_initialization_pending) {
			initialize();
		}
	}

	const Error err = set_gaussian_data(p_snapshot.gaussian_data);
	if (err != OK) {
		set_gaussian_data(Ref<GaussianData>());
		clear_static_chunks();
		world_submission_contract_active = false;
		world_submission_has_residency_hint = false;
		world_submission_residency_hint = 0;
		return err;
	}

	if (has_renderable_data) {
		set_static_chunks(p_snapshot.static_chunks);
	} else {
		clear_static_chunks();
	}

	world_submission_contract_active = has_active_world_submission;
	world_submission_has_residency_hint = has_residency_hint;
	world_submission_residency_hint = has_residency_hint ? p_snapshot.desired_residency_hint : 0;
	return OK;
}

Error GaussianSplatRenderer::apply_world_submission_contract(const WorldSubmissionContract &p_contract) {
	// World submissions are built from a synchronous read of the project
	// route policy in the world node (see gaussian_splat_world_3d.cpp).
	// The renderer's cached route policy is otherwise refreshed lazily via
	// the ProjectSettings settings_changed signal, which dispatches one
	// frame later. Invalidate the cache eagerly here so that any same-turn
	// build_frame_backend_plan() observes the same route policy the
	// submission was constructed with, instead of a stale prior value.
	_mark_streaming_route_policy_dirty();
	_refresh_streaming_route_policy_cache();
	set_lod_enabled(p_contract.lod_enabled);
	set_lod_bias(p_contract.lod_bias);
	set_lod_max_distance(p_contract.lod_max_distance);
	set_frustum_culling(p_contract.frustum_culling);
	set_async_upload_enabled(p_contract.async_upload_enabled);
	set_opacity_multiplier(p_contract.opacity_multiplier);
	set_streaming_config_overrides(p_contract.streaming_overrides);
	set_max_splats(MAX(1, p_contract.max_splats));

	const bool has_renderable_data =
			p_contract.gaussian_data.is_valid() && p_contract.gaussian_data->get_count() > 0;

	if (p_contract.gaussian_data.is_valid()) {
		const auto &resource_state = get_resource_state();
		if (!resource_state.gpu_resources_initialized && !resource_state.gpu_initialization_pending) {
			initialize();
		}
	}

	const Error err = set_gaussian_data(p_contract.gaussian_data);
	if (err != OK) {
		clear_static_chunks();
		world_submission_contract_active = false;
		world_submission_has_residency_hint = false;
		world_submission_residency_hint = 0;
		return err;
	}

	if (!has_renderable_data) {
		world_submission_contract_active = false;
		world_submission_has_residency_hint = false;
		world_submission_residency_hint = 0;
		if (p_contract.debug_label.is_empty()) {
			WARN_PRINT("[GaussianSplatRenderer] World submission has zero splats; renderer will stay disconnected.");
		} else {
			WARN_PRINT(vformat("[GaussianSplatRenderer] World submission '%s' has zero splats; renderer will stay disconnected.",
					p_contract.debug_label));
		}
		clear_static_chunks();
		return OK;
	}

	world_submission_contract_active = true;
	world_submission_has_residency_hint = p_contract.has_desired_residency_hint;
	world_submission_residency_hint = p_contract.has_desired_residency_hint ? p_contract.desired_residency_hint : 0;
	data_orchestrator->access_streaming_state_mutable().pending_payload_source = p_contract.payload_source;
	set_static_chunks(p_contract.static_chunks);
	return OK;
}

void GaussianSplatRenderer::clear_world_submission_contract() {
	world_submission_contract_active = false;
	world_submission_has_residency_hint = false;
	world_submission_residency_hint = 0;
	data_orchestrator->access_streaming_state_mutable().pending_payload_source.unref();
	set_gaussian_data(Ref<GaussianData>());
	clear_static_chunks();
}
