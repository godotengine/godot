#include "render_frame_context_manager.h"

RenderFrameContextManager::RenderFrameContextManager() {
	reset();
}

void RenderFrameContextManager::reset_frame_state() {
	frame_state.visible_splat_count.store(0, std::memory_order_release);
	frame_state.frame_counter = 0;
	frame_state.sort_time_ms = 0.0f;
	frame_state.render_time_ms = 0.0f;
}

void RenderFrameContextManager::reset_view_state_defaults() {
	view_state.last_camera_to_world_transform = Transform3D();
	view_state.last_camera_projection.set_perspective(60.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
	view_state.last_camera_position = Vector3(0, 0, 5);
	view_state.manual_viewport_override = Size2i();
	view_state.manual_viewport_format_override = RD::DATA_FORMAT_MAX;
	view_state.active_viewport_color_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	view_state.using_scene_data = false;
}

void RenderFrameContextManager::reset() {
	reset_frame_state();
	reset_view_state_defaults();
}
