/**************************************************************************/
/*  visionos_xr_interface.h                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#ifdef VISIONOS_ENABLED

#include "drivers/metal/metal_objects_shared.h"
#include "drivers/metal/rendering_context_driver_metal.h"
#include "drivers/metal/rendering_device_driver_metal.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_server.h"
#include "servers/xr/xr_interface.h"
#include "servers/xr/xr_positional_tracker.h"
#include "servers/xr/xr_vrs.h"

#ifdef __OBJC__
// When compiling as Objective-C++, include the actual headers
#import <ARKit/ARKit.h>
#import <CompositorServices/CompositorServices.h>
#else
// When compiling as C++, use forward declarations for ARKit and CompositorServices types (opaque pointers)
typedef struct ar_world_tracking_provider *ar_world_tracking_provider_t;
typedef struct cp_layer_renderer *cp_layer_renderer_t;
typedef struct cp_layer_renderer_capabilities *cp_layer_renderer_capabilities_t;
typedef struct ar_session *ar_session_t;
typedef struct ar_device_anchor *ar_device_anchor_t;
typedef struct cp_frame *cp_frame_t;
typedef struct cp_drawable *cp_drawable_t;
#endif

class VisionOSXRInterface : public XRInterface {
	GDCLASS(VisionOSXRInterface, XRInterface);

public:
	enum SignalEnum {
		VISIONOS_XR_SIGNAL_SESSION_STARTED,
		VISIONOS_XR_SIGNAL_SESSION_PAUSED,
		VISIONOS_XR_SIGNAL_SESSION_RESUMED,
		VISIONOS_XR_SIGNAL_SESSION_INVALIDATED,
		VISIONOS_XR_SIGNAL_POSE_RECENTERED,
		VISIONOS_XR_SIGNAL_MAX,
	};

private:
	bool initialized = false;
	XRInterface::TrackingStatus tracking_state;

	static RenderingServer *rendering_server;
	static ar_world_tracking_provider_t world_tracking_provider;

	cp_layer_renderer_t layer_renderer = nullptr;
	cp_layer_renderer_capabilities_t layer_renderer_capabilities = nullptr;
	ar_session_t ar_session = nullptr;

	ar_device_anchor_t current_device_anchor = nullptr;
	cp_frame_t current_frame = nullptr;

	// Data and functions only accessible from the rendering thread
	class RenderThread : public Object {
	private:
		bool initialized = false;
		RenderingDevice *rendering_device = nullptr;
		PixelFormats *pixel_formats = nullptr;

		float minimum_supported_near_plane = 0;

		// RenderThread must query the device anchor again,
		// because ar_device_anchor_t objects cannot be safely shared between threads
		ar_device_anchor_t current_device_anchor = nullptr;
		Transform3D origin_from_head;

		cp_frame_t current_frame = nullptr;
		cp_drawable_t current_drawable = nullptr;

		RD::Texture current_color_texture;
		RID current_color_texture_id;
		RD::Texture current_depth_texture;
		RID current_depth_texture_id;
		RD::Texture current_rasterization_rate_map;
		RID current_rasterization_rate_map_id;

	public:
		void initialize();
		void uninitialize();

		void set_minimum_supported_near_plane(float p_minimum_supported_near_plane);
		// p_current_frame should be an cp_frame_t pointer casted to uint64_t
		void set_current_frame(uint64_t p_current_frame);

		void start_frame_update();
		void end_frame_update();

		uint32_t get_view_count();
		Size2 get_render_target_size();
		Transform3D get_camera_transform();
		Transform3D get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform);
		Projection get_projection_for_view(uint32_t p_view, double p_aspect, double p_z_near, double p_z_far);
		Rect2i get_render_region();

		void pre_render();
		Vector<RenderingServerTypes::BlitToScreen> post_draw_viewport(RID p_render_target, const Rect2 &p_screen_rect);
		void encode_present(MTL3::MDCommandBuffer *p_cmd_buffer);
		void end_frame();

		RID get_color_texture();
		RID get_depth_texture();
		RID get_vrs_texture();
	} rt;

	// Head tracker
	Ref<XRPositionalTracker> head_tracker;

	static void _bind_methods();
	static const String name;
	static StringName get_signal_name(SignalEnum p_signal);

	void set_head_pose_from_arkit();

public:
	static Ref<VisionOSXRInterface> find_interface() {
		return XRServer::get_singleton()->find_interface(name);
	}

	VisionOSXRInterface();
	~VisionOSXRInterface();

	void emit_signal_enum(SignalEnum p_signal);

	virtual StringName get_name() const override;
	virtual uint32_t get_capabilities() const override;

	virtual TrackingStatus get_tracking_status() const override;

	virtual bool is_initialized() const override;
	virtual bool initialize() override;
	virtual void uninitialize() override;

	// The LayerRenderer and Capabilities are polled from the app delegate when initializing the VisionOSXRInterface,
	// but they need to be updated when the app backgrounds and foregrounds because they are recreated by visionOS
	void update_layer_renderer(cp_layer_renderer_t p_layer_renderer, cp_layer_renderer_capabilities_t p_layer_renderer_capabilities);

	virtual Dictionary get_system_info() override;
	virtual VRSTextureFormat get_vrs_texture_format() override;

	virtual bool supports_play_area_mode(XRInterface::PlayAreaMode p_mode) override;
	virtual XRInterface::PlayAreaMode get_play_area_mode() const override;
	virtual bool set_play_area_mode(XRInterface::PlayAreaMode p_mode) override;

	virtual void process() override;

	// Render thread methods
	virtual uint32_t get_view_count() override {
		return rt.get_view_count();
	}
	virtual Size2 get_render_target_size() override {
		return rt.get_render_target_size();
	}
	virtual Transform3D get_camera_transform() override {
		return rt.get_camera_transform();
	}
	virtual Transform3D get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform) override {
		return rt.get_transform_for_view(p_view, p_cam_transform);
	}
	virtual Projection get_projection_for_view(uint32_t p_view, double p_aspect, double p_z_near, double p_z_far) override {
		return rt.get_projection_for_view(p_view, p_aspect, p_z_near, p_z_far);
	}
	virtual Rect2i get_render_region() override {
		return rt.get_render_region();
	}
	virtual void pre_render() override {
		rt.pre_render();
	}
	virtual Vector<RenderingServerTypes::BlitToScreen> post_draw_viewport(RID p_render_target, const Rect2 &p_screen_rect) override {
		return rt.post_draw_viewport(p_render_target, p_screen_rect);
	}
	void encode_present(MTL3::MDCommandBuffer *p_cmd_buffer) {
		rt.encode_present(p_cmd_buffer);
	}
	virtual void end_frame() override {
		rt.end_frame();
	}

	virtual RID get_color_texture() override {
		return rt.get_color_texture();
	}
	virtual RID get_depth_texture() override {
		return rt.get_depth_texture();
	}
	virtual RID get_vrs_texture() override {
		return rt.get_vrs_texture();
	}
};

#endif
