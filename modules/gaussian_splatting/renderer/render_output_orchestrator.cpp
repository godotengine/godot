#include "render_output_orchestrator.h"

#include "core/error/error_macros.h"
#include "core/string/ustring.h"
#include "core/templates/paged_array.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "../interfaces/output_compositor.h"
#include "../logger/gs_logger.h"

RenderOutputOrchestrator::RenderOutputOrchestrator(GaussianSplatRenderer *p_renderer, OutputCompositor *p_output_compositor,
		PainterlyRenderer *p_painterly_renderer, GPUCuller *p_gpu_culler,
		CreateGpuResourcesFn p_create_gpu_resources, SetViewportFormatFn p_set_active_viewport_format,
		SetViewportFormatFn p_set_manual_viewport_format, TextureFormatFn p_get_texture_format) :
		renderer(p_renderer),
		output_compositor(p_output_compositor),
		painterly_renderer(p_painterly_renderer),
		gpu_culler(p_gpu_culler),
		create_gpu_resources(p_create_gpu_resources),
		set_active_viewport_format(p_set_active_viewport_format),
		set_manual_viewport_format(p_set_manual_viewport_format),
		get_texture_format(p_get_texture_format) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(output_compositor);
	ERR_FAIL_NULL(gpu_culler);
	ERR_FAIL_COND_MSG(!create_gpu_resources, "RenderOutputOrchestrator requires resource init callback.");
	ERR_FAIL_COND_MSG(!set_active_viewport_format, "RenderOutputOrchestrator requires viewport format callback.");
	ERR_FAIL_COND_MSG(!set_manual_viewport_format, "RenderOutputOrchestrator requires viewport format callback.");
	ERR_FAIL_COND_MSG(!get_texture_format, "RenderOutputOrchestrator requires texture format callback.");
}

bool RenderOutputOrchestrator::copy_final_texture_to_target(RID p_render_target, const Size2i &p_viewport_size) {
	if (!output_compositor) {
		return false;
	}

	RID final_texture = output_compositor->get_final_render_texture();
	if (!final_texture.is_valid() || !p_render_target.is_valid()) {
		return false;
	}

	if (!renderer->ensure_rendering_device("copy_final_texture_to_target")) {
		return false;
	}

	if (!output_compositor->is_initialized() && renderer->get_device_state().rd) {
		output_compositor->initialize(renderer->get_device_state().rd);
	}

	Size2i internal_size;
	if (painterly_renderer) {
		PainterlyPassGraph *pass_graph = painterly_renderer->get_pass_graph();
		if (pass_graph) {
			internal_size = pass_graph->get_internal_size();
		}
	}
	if (internal_size.x > 0 && internal_size.y > 0) {
		output_compositor->set_internal_render_size(internal_size);
	}

	OutputCopyParams params;
	params.source_texture = final_texture;
	params.destination_texture = p_render_target;
	params.viewport_size = p_viewport_size;
	params.composite_with_destination = false;
	params.source_is_premultiplied = true;

	OutputCopyResult result = output_compositor->copy_to_render_target(params);
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

	if (output_compositor) {
		auto &output_cache = output_compositor->get_cache_state();
		output_cache.render_buffers_commit_pending = false;
		output_cache.pending_render_buffers_size = Size2i();
		output_cache.pending_painterly_commit = false;
	}
}

bool RenderOutputOrchestrator::render_for_view(const Transform3D &p_world_to_camera_transform, const Projection &p_cam_projection,
		RID p_render_target, const Size2i &p_viewport_size) {
	// Derive camera_to_world from the world_to_camera (view) matrix
	Transform3D camera_to_world = p_world_to_camera_transform.affine_inverse();
	renderer->get_view_state().last_camera_to_world_transform = camera_to_world;
	renderer->get_view_state().last_camera_projection = p_cam_projection;
	renderer->get_view_state().last_camera_position = camera_to_world.origin;

	if (!renderer->get_resource_state().gpu_resources_initialized || renderer->get_resource_state().gpu_initialization_pending) {
		create_gpu_resources();
	}

	if (!renderer->get_scene_state().gaussian_data.is_valid() && renderer->get_test_data_state().positions.is_empty()) {
		renderer->get_frame_state().visible_splat_count.store(0, std::memory_order_release);
		return false;
	}

	if (!renderer->ensure_rendering_device("render_for_view")) {
		return false;
	}

	Size2i viewport_size = p_viewport_size;
	if (viewport_size.x <= 0 || viewport_size.y <= 0) {
		viewport_size = Size2i(1, 1);
	}

	renderer->get_view_state().manual_viewport_override = viewport_size;
	gpu_culler->get_config().last_cull_viewport_size = viewport_size;

	RD::DataFormat target_format = RD::DATA_FORMAT_MAX;
	RID destination_texture = p_render_target;

	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	if (texture_storage && p_render_target.is_valid()) {
		RID rd_texture = texture_storage->render_target_get_rd_texture(p_render_target);
		if (rd_texture.is_valid()) {
			destination_texture = rd_texture;
		}

		RD::TextureFormat detected_format = get_texture_format(
			renderer->get_resource_owner(destination_texture, renderer->get_device_state().rd), destination_texture);
		if (detected_format.format != RD::DATA_FORMAT_MAX) {
			target_format = detected_format.format;
		} else if (texture_storage->render_target_is_using_hdr(p_render_target)) {
			target_format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		}
	} else if (destination_texture.is_valid()) {
		RD::TextureFormat detected_format = get_texture_format(
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

	set_active_viewport_format(target_format, "render_for_view");
	set_manual_viewport_format(target_format, "render_for_view");

	PagedArray<RID> dummy_instances;
	renderer->render_gaussians(nullptr, dummy_instances);

	renderer->get_view_state().manual_viewport_override = Size2i();
	set_manual_viewport_format(RD::DATA_FORMAT_MAX, "render_for_view_cleanup");

	if (!output_compositor) {
		return false;
	}

	RID final_texture = output_compositor->get_final_render_texture();
	if (!final_texture.is_valid()) {
		return false;
	}

	if (!destination_texture.is_valid()) {
		return true;
	}

	if (!output_compositor->is_initialized() && renderer->get_device_state().rd) {
		output_compositor->initialize(renderer->get_device_state().rd);
	}

	Size2i internal_size;
	if (painterly_renderer) {
		PainterlyPassGraph *pass_graph = painterly_renderer->get_pass_graph();
		if (pass_graph) {
			internal_size = pass_graph->get_internal_size();
		}
	}
	if (internal_size.x > 0 && internal_size.y > 0) {
		output_compositor->set_internal_render_size(internal_size);
	}

	OutputCopyParams params;
	params.source_texture = final_texture;
	params.destination_texture = destination_texture;
	params.viewport_size = viewport_size;
	params.composite_with_destination = false;
	params.source_is_premultiplied = true;

	OutputCopyResult result = output_compositor->copy_to_render_target(params);
	if (!result.success && !result.error.is_empty()) {
		GS_LOG_WARN_DEFAULT(vformat("[OutputCompositor] %s", result.error));
	}
	return result.success;
}

bool RenderOutputOrchestrator::was_last_viewport_copy_successful() const {
	if (output_compositor) {
		return output_compositor->get_last_copy_success();
	}
	return false;
}

Size2i RenderOutputOrchestrator::get_last_viewport_copy_source_size() const {
	if (output_compositor) {
		return output_compositor->get_last_copy_source_size();
	}
	return Size2i();
}

Size2i RenderOutputOrchestrator::get_last_viewport_copy_dest_size() const {
	if (output_compositor) {
		return output_compositor->get_last_copy_dest_size();
	}
	return Size2i();
}

#ifdef TESTS_ENABLED
bool RenderOutputOrchestrator::test_copy_final_output(RID p_source, RID p_destination, const Size2i &p_viewport_size) {
	if (!output_compositor) {
		return false;
	}

	if (!renderer->ensure_rendering_device("test_copy_final_output")) {
		return false;
	}

	if (!output_compositor->is_initialized() && renderer->get_device_state().rd) {
		output_compositor->initialize(renderer->get_device_state().rd);
	}

	Size2i internal_size;
	if (painterly_renderer) {
		PainterlyPassGraph *pass_graph = painterly_renderer->get_pass_graph();
		if (pass_graph) {
			internal_size = pass_graph->get_internal_size();
		}
	}
	if (internal_size.x > 0 && internal_size.y > 0) {
		output_compositor->set_internal_render_size(internal_size);
	}

	OutputCopyParams params;
	params.source_texture = p_source;
	params.destination_texture = p_destination;
	params.viewport_size = p_viewport_size;
	params.composite_with_destination = false;
	params.source_is_premultiplied = true;

	OutputCopyResult result = output_compositor->copy_to_render_target(params);
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
