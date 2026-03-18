#include "render_instancing_orchestrator.h"

#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/os/os.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "instance_pipeline_contract.h"
#include "render_pipeline_stages.h"
#include "../interfaces/output_compositor.h"
#include <cstdint>

namespace {

static void _reset_output_cache_after_readiness_failure(OutputCompositor::OutputCacheState &r_output_cache) {
	r_output_cache.has_valid_render = false;
	r_output_cache.last_viewport_copy_success = false;
	r_output_cache.last_viewport_copy_source_size = Size2i();
	r_output_cache.last_viewport_copy_dest_size = Size2i();
	r_output_cache.render_buffers_commit_pending = false;
	r_output_cache.pending_render_buffers_size = Size2i();
	r_output_cache.pending_painterly_commit = false;
}

} // namespace

RenderInstancingOrchestrator::RenderInstancingOrchestrator(GaussianSplatRenderer *p_renderer,
		OutputCompositor *p_output_compositor, RenderPipelineStages *p_pipeline_stages,
		PrepareRenderFrameContextFn p_prepare_render_frame_context, RenderSortedSplatsFn p_render_sorted_splats) :
		renderer(p_renderer),
		output_compositor(p_output_compositor),
		pipeline_stages(p_pipeline_stages),
		prepare_render_frame_context(p_prepare_render_frame_context),
		render_sorted_splats(p_render_sorted_splats) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(output_compositor);
	ERR_FAIL_NULL(pipeline_stages);
	ERR_FAIL_COND_MSG(!prepare_render_frame_context, "RenderInstancingOrchestrator requires a frame context callback.");
	ERR_FAIL_COND_MSG(!render_sorted_splats, "RenderInstancingOrchestrator requires a render callback.");
}

RenderInstancingOrchestrator::InstanceReadinessResult RenderInstancingOrchestrator::evaluate_instance_pipeline_readiness(
		bool p_streaming_system_ready, bool p_has_instance_pipeline_buffers,
		const GaussianRenderPipeline::InstancePipelineBuffers &p_buffers) {
	InstanceReadinessResult result;
	if (!p_streaming_system_ready) {
		result.failure_mode = InstanceReadinessFailureMode::STREAMING_SYSTEM_UNAVAILABLE;
		return result;
	}
	if (!p_has_instance_pipeline_buffers) {
		result.failure_mode = InstanceReadinessFailureMode::INSTANCE_PIPELINE_BUFFERS_UNAVAILABLE;
		return result;
	}
	if (!GaussianSplatting::InstancePipelineContract::has_cull_buffers(p_buffers) ||
			!GaussianSplatting::InstancePipelineContract::has_sort_buffers(p_buffers) ||
			!GaussianSplatting::InstancePipelineContract::has_raster_buffers(p_buffers)) {
		result.failure_mode = InstanceReadinessFailureMode::INSTANCE_PIPELINE_BUFFERS_INVALID;
		return result;
	}

	result.ready = true;
	return result;
}

void RenderInstancingOrchestrator::_warn_instanced_readiness_failure_once(
		InstanceReadinessFailureMode p_failure_mode) const {
	switch (p_failure_mode) {
		case InstanceReadinessFailureMode::NONE:
			return;
		case InstanceReadinessFailureMode::STREAMING_SYSTEM_UNAVAILABLE:
			WARN_PRINT_ONCE("[GaussianSplatRenderer] Instanced render skipped: streaming system unavailable.");
			return;
		case InstanceReadinessFailureMode::INSTANCE_PIPELINE_BUFFERS_UNAVAILABLE:
			WARN_PRINT_ONCE("[GaussianSplatRenderer] Instanced render skipped: instance pipeline buffers are unavailable.");
			return;
		case InstanceReadinessFailureMode::INSTANCE_PIPELINE_BUFFERS_INVALID:
			WARN_PRINT_ONCE("[GaussianSplatRenderer] Instanced render skipped: required instance pipeline buffers are invalid.");
			return;
	}
}

void RenderInstancingOrchestrator::render_instanced(RenderDataRD *p_render_data,
			const GaussianSplatManager::SharedDynamicAssetHandle &p_handle,
			const Transform3D &p_world_to_camera_transform, const Projection &p_projection, const Projection &p_render_projection,
			const LocalVector<Transform3D> &p_instance_transforms) {
	(void)p_handle;
	ERR_FAIL_NULL_MSG(renderer, "RenderInstancingOrchestrator requires a renderer.");
	ERR_FAIL_NULL_MSG(pipeline_stages, "RenderInstancingOrchestrator requires pipeline stages.");
	ERR_FAIL_COND_MSG(!output_compositor, "OutputCompositor not initialized");
	auto &output_cache = output_compositor->get_cache_state();

	bool defer_commit = p_render_data && p_render_data->render_buffers.is_valid();

	if (p_instance_transforms.is_empty()) {
		render_sorted_splats(p_render_data, p_world_to_camera_transform, p_projection, p_render_projection, defer_commit);
		return;
	}

	const GaussianSplatRenderer::StreamingState &streaming_state = renderer->get_streaming_state();
	const bool streaming_system_ready = streaming_state.current_streaming_system.is_valid();
	const InstanceReadinessResult readiness = evaluate_instance_pipeline_readiness(
			streaming_system_ready, renderer->has_instance_pipeline_buffers(),
			renderer->get_instance_pipeline_buffers());
	if (!readiness.ready) {
		_warn_instanced_readiness_failure_once(readiness.failure_mode);
		GaussianSplatRenderer::FrameState &frame_state = renderer->get_frame_state();
		GaussianSplatRenderer::SortingState &sorting_state = renderer->get_sorting_state();
		GaussianSplatRenderer::PerformanceMetrics &metrics = renderer->get_performance_state().metrics;
		frame_state.render_time_ms = 0.0f;
		frame_state.sort_time_ms = 0.0f;
		metrics.rendered_splat_count = 0;
		metrics.sort_submission_time_ms = 0.0f;
		metrics.sort_wait_time_ms = 0.0f;
		metrics.sort_input_build_time_ms = 0.0f;
		metrics.async_sort_used = false;
		metrics.async_sort_waited = false;
		metrics.async_overlap_efficiency = 0.0f;
		frame_state.visible_splat_count.store(0, std::memory_order_release);
		sorting_state.sorted_splat_count = 0;
		renderer->get_debug_state().last_stage_metrics = GaussianSplatRenderer::StageMetrics();
		renderer->get_debug_state().last_stage_metrics_valid = false;
		_reset_output_cache_after_readiness_failure(output_cache);
		return;
	}

	uint64_t instanced_frame_start_usec = OS::get_singleton()->get_ticks_usec();
	auto &metrics = renderer->get_performance_state().metrics;
	uint64_t frames_before_instances = metrics.total_frames_rendered;
	float avg_before_instances = metrics.avg_frame_time_ms;
	float peak_before_instances = metrics.peak_frame_time_ms;

	double accumulated_render_time_ms = 0.0;
	uint64_t accumulated_rendered_splats = 0;
	uint64_t aggregated_visible_splats = 0;
	bool any_render_performed = false;
	bool any_viewport_copy = false;
	Size2i last_copy_source_size;
	Size2i last_copy_dest_size;

	const int instance_count = p_instance_transforms.size();
	bool allow_deferred_commit = defer_commit && instance_count == 1;

	for (int instance_index = 0; instance_index < instance_count; instance_index++) {
		const Transform3D &instance_xform = p_instance_transforms[instance_index];

		// Combine the camera view transform with the instance transform to keep splats in world space.
		Transform3D instance_view_transform = p_world_to_camera_transform * instance_xform;

		bool instance_defer_commit = allow_deferred_commit && (instance_index == instance_count - 1);

		GaussianSplatRenderer::StageMetrics stage_metrics{};
		GaussianSplatRenderer::RenderFrameContext frame_context;
		prepare_render_frame_context(p_render_data, instance_view_transform, p_projection, p_render_projection,
				instance_defer_commit, frame_context);
		frame_context.metrics = &stage_metrics;
		GaussianSplatRenderer::FrameStateProvider frame_provider(renderer, &frame_context.deps);
		frame_context.state_provider = &frame_provider;
		GaussianSplatRenderer::RenderFramePlan frame_plan = GaussianSplatRenderer::build_frame_plan(
				renderer->get_scene_state(), renderer->get_streaming_state(), renderer->get_sorting_state(),
				renderer->get_resource_state(), renderer->get_subsystem_state(), frame_provider.get_pipeline_features(),
				true, String(), String(), GaussianSplatRenderer::RenderFallbackReason::NONE,
				GaussianSplatRenderer::RenderFallbackReason::NONE, false, false);
		frame_context.deps.frame_plan = &frame_plan;
		DEV_ASSERT(frame_context.deps.frame_plan);
		ERR_FAIL_COND(!frame_context.deps.validate());

		GaussianSplatRenderer::CullStageInput cull_input{
			frame_context.frame_id,
			frame_context.world_to_camera_transform,
			frame_context.cull_projection,
			frame_context.viewport_size,
			frame_context.metrics,
			&frame_provider
		};
		GaussianSplatRenderer::CullStageOutput cull_output;
		pipeline_stages->execute_cull_stage(cull_input, cull_output);
		GaussianSplatRenderer::SortStageOutput sort_output;
		if (cull_output.has_visible) {
			GaussianSplatRenderer::SortStageInput sort_input{
				frame_context.frame_id,
				frame_context.world_to_camera_transform,
				cull_output.visible_count,
				frame_context.metrics,
				&frame_provider,
				cull_output.visible_domain
			};
			pipeline_stages->execute_sort_stage(sort_input, sort_output);
		}
		frame_context.snapshot.valid = true;
		frame_context.snapshot.cull_visible_domain = cull_output.visible_domain;
		frame_context.snapshot.sorted_index_domain = sort_output.output_domain;
		if (sort_output.output_domain == GaussianSplatRenderer::IndexDomain::SPLAT_REF) {
			frame_context.snapshot.visible_splats = sort_output.sorted_count;
		} else if (sort_output.output_domain == GaussianSplatRenderer::IndexDomain::GAUSSIAN_GLOBAL) {
			frame_context.snapshot.visible_splats =
					(sort_output.sorted_count > 0 ? sort_output.sorted_count : cull_output.visible_count);
		} else if (cull_output.visible_domain == GaussianSplatRenderer::IndexDomain::CHUNK_REF) {
			frame_context.snapshot.visible_splats = sort_output.sorted_count;
		} else {
			frame_context.snapshot.visible_splats = cull_output.visible_count;
		}
		frame_context.snapshot.sorted_splats = sort_output.sorted_count;

		uint64_t before_total_frames = metrics.total_frames_rendered;
		float before_avg_time = metrics.avg_frame_time_ms;
		float before_peak_time = metrics.peak_frame_time_ms;
		uint64_t before_frame_counter = renderer->get_frame_state().frame_counter;

		pipeline_stages->render_sorted_splats_with_context(frame_context);

		accumulated_render_time_ms += renderer->get_frame_state().render_time_ms;
		accumulated_rendered_splats += metrics.rendered_splat_count;
		aggregated_visible_splats += renderer->get_frame_state().visible_splat_count.load(std::memory_order_acquire);
		any_render_performed = any_render_performed || output_cache.has_valid_render;
		if (output_cache.last_viewport_copy_success) {
			any_viewport_copy = true;
			last_copy_source_size = output_cache.last_viewport_copy_source_size;
			last_copy_dest_size = output_cache.last_viewport_copy_dest_size;
		}

		if (instance_count > 1 && instance_index < instance_count - 1) {
			renderer->get_frame_state().frame_counter = before_frame_counter;
			metrics.total_frames_rendered = before_total_frames;
			metrics.avg_frame_time_ms = before_avg_time;
			metrics.peak_frame_time_ms = before_peak_time;
		}
	}

	renderer->get_frame_state().render_time_ms = accumulated_render_time_ms;
	metrics.rendered_splat_count = accumulated_rendered_splats;
	renderer->get_frame_state().visible_splat_count.store(
			static_cast<uint32_t>(MIN<uint64_t>(aggregated_visible_splats, UINT32_MAX)),
			std::memory_order_release);
	output_cache.has_valid_render = any_render_performed;

	if (any_viewport_copy) {
		output_cache.last_viewport_copy_success = true;
		output_cache.last_viewport_copy_source_size = last_copy_source_size;
		output_cache.last_viewport_copy_dest_size = last_copy_dest_size;
	} else {
		output_cache.last_viewport_copy_success = false;
		output_cache.last_viewport_copy_source_size = Size2i();
		output_cache.last_viewport_copy_dest_size = Size2i();
	}

	uint64_t instanced_frame_end_usec = OS::get_singleton()->get_ticks_usec();
	float aggregated_frame_time_ms = (instanced_frame_end_usec - instanced_frame_start_usec) / 1000.0f;
	uint64_t frames_after_instances = metrics.total_frames_rendered;
	if (frames_after_instances == frames_before_instances + 1) {
		float alpha = 0.05f;
		if (frames_before_instances == 0) {
			metrics.avg_frame_time_ms = aggregated_frame_time_ms;
		} else {
			metrics.avg_frame_time_ms =
					avg_before_instances * (1.0f - alpha) + aggregated_frame_time_ms * alpha;
		}
		metrics.peak_frame_time_ms =
				MAX(peak_before_instances, aggregated_frame_time_ms);
	}
}

void GaussianSplatRenderer::render_instanced(RenderDataRD *p_render_data,
		const GaussianSplatManager::SharedDynamicAssetHandle &p_handle,
		const Transform3D &p_world_to_camera_transform, const Projection &p_projection, const Projection &p_render_projection,
		const LocalVector<Transform3D> &p_instance_transforms) {
	instancing_orchestrator->render_instanced(p_render_data, p_handle, p_world_to_camera_transform,
			p_projection, p_render_projection, p_instance_transforms);
}
