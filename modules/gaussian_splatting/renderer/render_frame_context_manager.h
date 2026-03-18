#ifndef GAUSSIAN_RENDER_FRAME_CONTEXT_MANAGER_H
#define GAUSSIAN_RENDER_FRAME_CONTEXT_MANAGER_H

#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2i.h"
#include "core/math/vector3.h"
#include "servers/rendering/rendering_device.h"

#include <atomic>

class RenderFrameContextManager {
public:
	struct FrameState {
		std::atomic<uint32_t> visible_splat_count{0};
		uint32_t frame_counter = 0;
		float sort_time_ms = 0.0f;
		float render_time_ms = 0.0f;
	};

	struct ViewState {
		Transform3D last_camera_to_world_transform;
		Projection last_camera_projection;
		Vector3 last_camera_position = Vector3(0, 0, 5);
		Size2i manual_viewport_override = Size2i();
		RD::DataFormat manual_viewport_format_override = RD::DATA_FORMAT_MAX;
		RD::DataFormat active_viewport_color_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		bool using_scene_data = false;
	};

	RenderFrameContextManager();

	FrameState &get_frame_state() { return frame_state; }
	const FrameState &get_frame_state() const { return frame_state; }
	ViewState &get_view_state() { return view_state; }
	const ViewState &get_view_state() const { return view_state; }

	void reset_frame_state();
	void reset_view_state_defaults();
	void reset();

private:
	FrameState frame_state;
	ViewState view_state;
};

#endif
