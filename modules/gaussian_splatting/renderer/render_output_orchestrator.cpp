#include "render_output_orchestrator.h"

#include "core/error/error_macros.h"
#include "core/string/ustring.h"
#include "core/templates/paged_array.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "../interfaces/output_compositor.h"
#include "../logger/gs_logger.h"

RenderOutputOrchestrator::RenderOutputOrchestrator(const Dependencies &p_dependencies) :
		renderer(p_dependencies.renderer),
		output_compositor(p_dependencies.output_compositor),
		painterly_renderer(p_dependencies.painterly_renderer),
		gpu_culler(p_dependencies.gpu_culler),
		runtime_ports(p_dependencies.runtime_ports) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(output_compositor);
	ERR_FAIL_NULL(gpu_culler);
	ERR_FAIL_COND_MSG(!runtime_ports.create_gpu_resources, "RenderOutputOrchestrator requires resource init callback.");
	ERR_FAIL_COND_MSG(runtime_ports.ensure_rendering_device == nullptr, "RenderOutputOrchestrator requires ensure_rendering_device runtime port.");
	ERR_FAIL_COND_MSG(runtime_ports.get_texture_format == nullptr, "RenderOutputOrchestrator requires get_texture_format runtime port.");
	ERR_FAIL_COND_MSG(runtime_ports.set_active_viewport_format == nullptr, "RenderOutputOrchestrator requires set_active_viewport_format runtime port.");
	ERR_FAIL_COND_MSG(runtime_ports.set_manual_viewport_format == nullptr, "RenderOutputOrchestrator requires set_manual_viewport_format runtime port.");
}

bool RenderOutputOrchestrator::copy_final_texture_to_target(RID p_render_target, const Size2i &p_viewport_size) {
	GaussianSplatRenderer::FrameStateProvider frame_provider(renderer);
	const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;

	OutputCompositor *local_output_compositor = state_view.get_output_compositor();
	PainterlyRenderer *local_painterly_renderer = state_view.get_painterly_renderer();
	if (!local_output_compositor) {
		return false;
	}

	RID final_texture = local_output_compositor->get_final_render_texture();
	if (!final_texture.is_valid() || !p_render_target.is_valid()) {
		return false;
	}

	if (!(renderer->*runtime_ports.ensure_rendering_device)("copy_final_texture_to_target")) {
		return false;
	}

	if (!local_output_compositor->is_initialized() && renderer->get_device_state().rd) {
		local_output_compositor->initialize(renderer->get_device_state().rd);
	}

	Size2i internal_size;
	if (local_painterly_renderer) {
		PainterlyPassGraph *pass_graph = local_painterly_renderer->get_pass_graph();
		if (pass_graph) {
			internal_size = pass_graph->get_internal_size();
		}
	}
	if (internal_size.x > 0 && internal_size.y > 0) {
		local_output_compositor->set_internal_render_size(internal_size);
	}

	OutputCopyParams params;
	params.source_texture = final_texture;
	params.destination_texture = p_render_target;
	params.viewport_size = p_viewport_size;
	params.composite_with_destination = false;
	params.source_is_premultiplied = true;

	OutputCopyResult result = local_output_compositor->copy_to_render_target(params);
	if (!result.success && !result.error.is_empty()) {
		GS_LOG_WARN_DEFAULT(vformat("[OutputCompositor] %s", result.error));
	}
	return result.success;
}

void RenderOutputOrchestrator::commit_to_render_buffers(RenderDataRD *p_render_data) {
	(void)p_render_data;
	// NOTE: Viewport copy is ALREADY handled in OutputCompositor::integrate_final_output(), which is called
	// from render_scene_instance(). That copy writes to get_internal_texture() which is
	// the correct target for Godot's rendering pipeline.
	//
	// Previously we had a DUPLICATE copy here that was writing to a DIFFERENT target
	// (render_target_get_rd_texture vs get_internal_texture) which caused confusion.
	// The copy in OutputCompositor::integrate_final_output uses format 96 (sRGB) with blending enabled,
	// while this was using format 36 (linear) without blending - wrong target!
	//
	// This function now just clears the pending state flags.

	GaussianSplatRenderer::FrameStateProvider frame_provider(renderer);
	const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
	OutputCompositor *local_output_compositor = state_view.get_output_compositor();
	if (local_output_compositor) {
		auto &output_cache = local_output_compositor->get_cache_state();
		output_cache.render_buffers_commit_pending = false;
		output_cache.pending_render_buffers_size = Size2i();
		output_cache.pending_painterly_commit = false;
	}
}

bool RenderOutputOrchestrator::render_for_view(const Transform3D &p_world_to_camera_transform, const Projection &p_cam_projection,
		RID p_render_target, const Size2i &p_viewport_size) {
	GaussianSplatRenderer::FrameStateProvider frame_provider(renderer);
	GaussianSplatRenderer::IFrameMutationAccess &state_mut = frame_provider;
	const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
	auto &frame_state = state_mut.get_frame_state_mut();
	auto &resource_state = state_mut.get_resource_state_mut();
	OutputCompositor *local_output_compositor = state_view.get_output_compositor();
	GPUCuller *local_gpu_culler = state_view.get_gpu_culler();
	PainterlyRenderer *local_painterly_renderer = state_view.get_painterly_renderer();

	// Derive camera_to_world from the world_to_camera (view) matrix
	Transform3D camera_to_world = p_world_to_camera_transform.affine_inverse();
	renderer->get_view_state().last_camera_to_world_transform = camera_to_world;
	renderer->get_view_state().last_camera_projection = p_cam_projection;
	renderer->get_view_state().last_camera_position = camera_to_world.origin;

	if (!resource_state.gpu_resources_initialized || resource_state.gpu_initialization_pending) {
		runtime_ports.create_gpu_resources();
	}

	if (!state_view.get_scene_state().gaussian_data.is_valid() && renderer->get_test_data_state().positions.is_empty()) {
		frame_state.visible_splat_count.store(0, std::memory_order_release);
		return false;
	}

	if (!(renderer->*runtime_ports.ensure_rendering_device)("render_for_view")) {
		return false;
	}

	Size2i viewport_size = p_viewport_size;
	if (viewport_size.x <= 0 || viewport_size.y <= 0) {
		viewport_size = Size2i(1, 1);
	}

	renderer->get_view_state().manual_viewport_override = viewport_size;
	if (local_gpu_culler) {
		local_gpu_culler->get_config().last_cull_viewport_size = viewport_size;
	}

	RD::DataFormat target_format = RD::DATA_FORMAT_MAX;
	RID destination_texture = p_render_target;

	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	if (texture_storage && p_render_target.is_valid()) {
		RID rd_texture = texture_storage->render_target_get_rd_texture(p_render_target);
		if (rd_texture.is_valid()) {
			destination_texture = rd_texture;
		}

		RD::TextureFormat detected_format = (renderer->*runtime_ports.get_texture_format)(
			renderer->get_resource_owner(destination_texture, renderer->get_device_state().rd), destination_texture);
		if (detected_format.format != RD::DATA_FORMAT_MAX) {
			target_format = detected_format.format;
		} else if (texture_storage->render_target_is_using_hdr(p_render_target)) {
			target_format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		}
	} else if (destination_texture.is_valid()) {
		RD::TextureFormat detected_format = (renderer->*runtime_ports.get_texture_format)(
			renderer->get_resource_owner(destination_texture, renderer->get_device_state().rd), destination_texture);
		if (detected_format.format != RD::DATA_FORMAT_MAX) {
			target_format = detected_format.format;
		}
	}

	if (target_format == RD::DATA_FORMAT_MAX && renderer->get_view_state().active_viewport_color_format != RD::DATA_FORMAT_MAX) {
		target_format = renderer->get_view_state().active_viewport_color_format;
	}
	if (target_format == RD::DATA_FORMAT_MAX) {
		target_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	}

	(renderer->*runtime_ports.set_active_viewport_format)(target_format, "render_for_view");
	(renderer->*runtime_ports.set_manual_viewport_format)(target_format, "render_for_view");

	PagedArray<RID> dummy_instances;
	renderer->render_gaussians(nullptr, dummy_instances);

	renderer->get_view_state().manual_viewport_override = Size2i();
	(renderer->*runtime_ports.set_manual_viewport_format)(RD::DATA_FORMAT_MAX, "render_for_view_cleanup");

	if (!local_output_compositor) {
		return false;
	}

	RID final_texture = local_output_compositor->get_final_render_texture();
	if (!final_texture.is_valid()) {
		return false;
	}

	if (!destination_texture.is_valid()) {
		return true;
	}

	if (!local_output_compositor->is_initialized() && renderer->get_device_state().rd) {
		local_output_compositor->initialize(renderer->get_device_state().rd);
	}

	Size2i internal_size;
	if (local_painterly_renderer) {
		PainterlyPassGraph *pass_graph = local_painterly_renderer->get_pass_graph();
		if (pass_graph) {
			internal_size = pass_graph->get_internal_size();
		}
	}
	if (internal_size.x > 0 && internal_size.y > 0) {
		local_output_compositor->set_internal_render_size(internal_size);
	}

	OutputCopyParams params;
	params.source_texture = final_texture;
	params.destination_texture = destination_texture;
	params.viewport_size = viewport_size;
	params.composite_with_destination = false;
	params.source_is_premultiplied = true;

	OutputCopyResult result = local_output_compositor->copy_to_render_target(params);
	if (!result.success && !result.error.is_empty()) {
		GS_LOG_WARN_DEFAULT(vformat("[OutputCompositor] %s", result.error));
	}
	return result.success;
}

bool RenderOutputOrchestrator::was_last_viewport_copy_successful() const {
	GaussianSplatRenderer::FrameStateProvider frame_provider(renderer);
	const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
	OutputCompositor *local_output_compositor = state_view.get_output_compositor();
	if (local_output_compositor) {
		return local_output_compositor->get_last_copy_success();
	}
	return false;
}

Size2i RenderOutputOrchestrator::get_last_viewport_copy_source_size() const {
	GaussianSplatRenderer::FrameStateProvider frame_provider(renderer);
	const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
	OutputCompositor *local_output_compositor = state_view.get_output_compositor();
	if (local_output_compositor) {
		return local_output_compositor->get_last_copy_source_size();
	}
	return Size2i();
}

Size2i RenderOutputOrchestrator::get_last_viewport_copy_dest_size() const {
	GaussianSplatRenderer::FrameStateProvider frame_provider(renderer);
	const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
	OutputCompositor *local_output_compositor = state_view.get_output_compositor();
	if (local_output_compositor) {
		return local_output_compositor->get_last_copy_dest_size();
	}
	return Size2i();
}

#ifdef TESTS_ENABLED
bool RenderOutputOrchestrator::test_copy_final_output(RID p_source, RID p_destination, const Size2i &p_viewport_size) {
	GaussianSplatRenderer::FrameStateProvider frame_provider(renderer);
	const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
	OutputCompositor *local_output_compositor = state_view.get_output_compositor();
	PainterlyRenderer *local_painterly_renderer = state_view.get_painterly_renderer();
	if (!local_output_compositor) {
		return false;
	}

	if (!(renderer->*runtime_ports.ensure_rendering_device)("test_copy_final_output")) {
		return false;
	}

	if (!local_output_compositor->is_initialized() && renderer->get_device_state().rd) {
		local_output_compositor->initialize(renderer->get_device_state().rd);
	}

	Size2i internal_size;
	if (local_painterly_renderer) {
		PainterlyPassGraph *pass_graph = local_painterly_renderer->get_pass_graph();
		if (pass_graph) {
			internal_size = pass_graph->get_internal_size();
		}
	}
	if (internal_size.x > 0 && internal_size.y > 0) {
		local_output_compositor->set_internal_render_size(internal_size);
	}

	OutputCopyParams params;
	params.source_texture = p_source;
	params.destination_texture = p_destination;
	params.viewport_size = p_viewport_size;
	params.composite_with_destination = false;
	params.source_is_premultiplied = true;

	OutputCopyResult result = local_output_compositor->copy_to_render_target(params);
	if (!result.success && !result.error.is_empty()) {
		GS_LOG_WARN_DEFAULT(vformat("[OutputCompositor] %s", result.error));
	}
	return result.success;
}
#endif

bool GaussianSplatRenderer::copy_final_texture_to_target(RID p_render_target, const Size2i &p_viewport_size) {
	return output_orchestrator->copy_final_texture_to_target(p_render_target, p_viewport_size);
}

void GaussianSplatRenderer::commit_to_render_buffers(RenderDataRD *p_render_data) {
	output_orchestrator->commit_to_render_buffers(p_render_data);
}

bool GaussianSplatRenderer::render_for_view(const Transform3D &p_world_to_camera_transform, const Projection &p_cam_projection,
		RID p_render_target, const Size2i &p_viewport_size) {
	return output_orchestrator->render_for_view(p_world_to_camera_transform, p_cam_projection, p_render_target, p_viewport_size);
}

bool GaussianSplatRenderer::was_last_viewport_copy_successful() const {
	return output_orchestrator->was_last_viewport_copy_successful();
}

Size2i GaussianSplatRenderer::get_last_viewport_copy_source_size() const {
	return output_orchestrator->get_last_viewport_copy_source_size();
}

Size2i GaussianSplatRenderer::get_last_viewport_copy_dest_size() const {
	return output_orchestrator->get_last_viewport_copy_dest_size();
}

void GaussianSplatRenderer::set_cached_render_reuse_enabled(bool p_enabled) {
	if (get_subsystem_state().output_compositor.is_valid()) {
		get_subsystem_state().output_compositor->set_cached_render_reuse_enabled(p_enabled);
	}
}

bool GaussianSplatRenderer::is_cached_render_reuse_enabled() const {
	if (get_subsystem_state().output_compositor.is_valid()) {
		return get_subsystem_state().output_compositor->is_cached_render_reuse_enabled();
	}
	return true;
}

void GaussianSplatRenderer::invalidate_cached_render() {
	if (get_subsystem_state().output_compositor.is_valid()) {
		get_subsystem_state().output_compositor->invalidate_cached_render();
	}
}

#ifdef TESTS_ENABLED
bool GaussianSplatRenderer::test_copy_final_output(RID p_source, RID p_destination, const Size2i &p_viewport_size) {
	return output_orchestrator->test_copy_final_output(p_source, p_destination, p_viewport_size);
}
#endif
