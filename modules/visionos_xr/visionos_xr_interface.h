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

#include "drivers/metal/rendering_context_driver_metal.h"
#include "drivers/metal/rendering_device_driver_metal.h"
#include "servers/xr/xr_interface.h"
#include "servers/xr/xr_positional_tracker.h"
#include "servers/xr/xr_vrs.h"

#import <ARKit/ARKit.h>
#import <CompositorServices/CompositorServices.h>

class VisionOSXRInterface : public XRInterface {
	GDCLASS(VisionOSXRInterface, XRInterface);

private:
	bool initialized = false;
	XRInterface::TrackingStatus tracking_state;
	XRPose::TrackingConfidence tracking_confidence = XRPose::XR_TRACKING_CONFIDENCE_NONE;

	cp_layer_renderer_t layer_renderer = nullptr;
	cp_layer_renderer_capabilities_t layer_renderer_capabilities = nullptr;
	ar_session_t ar_session;
	ar_world_tracking_provider_t world_tracking_provider;

	ar_device_anchor_t current_device_anchor;
	cp_frame_t current_frame;
	cp_drawable_t current_drawable;

	RD::Texture current_color_texture;
	RID current_color_texture_id;
	RD::Texture current_depth_texture;
	RID current_depth_texture_id;
	RD::Texture current_rasterization_rate_map;
	RID current_rasterization_rate_map_id;

	// Head tracker
	Ref<XRPositionalTracker> head_tracker;
	simd_float4x4 origin_from_head_simd;

	RenderingDevice *rendering_device = nullptr;
	RenderingDeviceDriverMetal *rendering_device_driver_metal = nullptr;
	PixelFormats *pixel_formats = nullptr;

	static void _bind_methods();

	void set_head_pose_from_arkit(bool p_use_drawable);

	static const String name;

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
	static StringName get_signal_name(SignalEnum p_signal);

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
	virtual Dictionary get_system_info() override;

	virtual bool supports_play_area_mode(XRInterface::PlayAreaMode p_mode) override;
	virtual XRInterface::PlayAreaMode get_play_area_mode() const override;
	virtual bool set_play_area_mode(XRInterface::PlayAreaMode p_mode) override;

	virtual Size2 get_render_target_size() override;
	virtual uint32_t get_view_count() override;

	virtual Transform3D get_camera_transform() override;
	virtual Transform3D get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform) override;
	virtual Projection get_projection_for_view(uint32_t p_view, double p_aspect, double p_z_near, double p_z_far) override;
	virtual Rect2i get_render_region() override;

	virtual void process() override;
	virtual void pre_render() override;
	virtual Vector<BlitToScreen> post_draw_viewport(RID p_render_target, const Rect2 &p_screen_rect) override;
	void encode_present(MDCommandBuffer *p_cmd_buffer);
	virtual void end_frame() override;

	virtual RID get_color_texture() override;
	virtual RID get_depth_texture() override;
	virtual RID get_vrs_texture() override;
	virtual VRSTextureFormat get_vrs_texture_format() override;
};

#endif
