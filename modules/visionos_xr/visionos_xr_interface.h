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

#include "visionos_controller_tracking.h"
#include "visionos_definitions.h"
#include "visionos_hand_tracking.h"

#ifdef __OBJC__
#import <CompositorServices/CompositorServices.h>
#else
typedef struct cp_layer_renderer *cp_layer_renderer_t;
typedef struct cp_layer_renderer_capabilities *cp_layer_renderer_capabilities_t;
typedef struct cp_frame *cp_frame_t;
typedef struct cp_drawable *cp_drawable_t;
typedef struct cp_frame_timing *cp_frame_timing_t;
#endif

class RenderingServer;
namespace MTL3 {
class MDCommandBuffer;
}

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

	enum ImmersionStyle {
		IMMERSION_STYLE_FULL,
		IMMERSION_STYLE_MIXED,
		IMMERSION_STYLE_PROGRESSIVE,
	};

	enum Visibility {
		VISIBILITY_AUTOMATIC,
		VISIBILITY_VISIBLE,
		VISIBILITY_HIDDEN,
	};

private:
	bool initialized = false;
	XRInterface::TrackingStatus tracking_state;

	RenderingServer *rendering_server;

	ar_session_t ar_session = nullptr;

	// Rendering in XR with CompositorServices
	struct CompositorServicesData {
		bool enabled = false;

		cp_layer_renderer_t layer_renderer = nullptr;
		cp_layer_renderer_capabilities_t layer_renderer_capabilities = nullptr;

		ar_world_tracking_provider_t world_tracking_provider = nullptr;

		cp_frame_t current_frame = nullptr;
		cp_frame_timing_t current_timing = nullptr;

		// Head tracker
		Ref<XRPositionalTracker> head_tracker;

		ar_device_anchor_t current_device_anchor = nullptr;

		bool initialize(XRServer *xr_server);
	} cs;

	void set_head_pose_from_arkit();

	// Checks the ARKit authorizations asynchronously
	// and updates the ARKit session if they changed.
	void update_authorizations_async();

	// Time used for pose prediction
	CFTimeInterval get_trackable_anchor_time();

	void update_from_authorizations(ar_authorization_results_t);

	// Hand tracking
	VisionOSHandTracking hands;

	// Controller tracking
	VisionOSControllerTracking controllers;

	// Data and functions only accessible from the rendering thread
	class RenderThread : public Object {
		// Inherit from Object to use callable_mp(), so we declare it as GDCLASS,
		// but this class should not be exposed to GDScript with GDREGISTER_CLASS.
		GDCLASS(RenderThread, Object);

	private:
		bool initialized = false;
		RenderingDevice *rendering_device = nullptr;
		PixelFormats *pixel_formats = nullptr;

		float minimum_supported_near_plane = 0;

		// RenderThread must query the device anchor again,
		// because ar_device_anchor_t objects cannot be safely shared between threads
		ar_device_anchor_t current_device_anchor = nullptr;
		ar_world_tracking_provider_t world_tracking_provider = nullptr;
		Transform3D origin_from_head;

		cp_frame_t current_frame = nullptr;
		cp_drawable_t current_drawable = nullptr;

		RD::Texture current_color_texture;
		RID current_color_texture_id;
		RD::Texture current_depth_texture;
		RID current_depth_texture_id;
		RID current_rasterization_rate_map_id;

		// Cached render target size, set in pre_render() on the render thread
		// and read from the game thread via get_render_target_size().
		SafeNumeric<uint32_t> cached_render_target_width{ 0 };
		SafeNumeric<uint32_t> cached_render_target_height{ 0 };

		// Wraps cp_drawable_encode_present in a drawable render context with a no-op pass,
		// required by Compositor Services when the layer supports progressive immersion.
		// p_command_buffer is an id<MTLCommandBuffer> bridge-cast to void *, since this header
		// is included from non-Objective-C++ translation units.
		static void encode_drawable_no_op_and_present(cp_drawable_t p_drawable, cp_frame_t p_frame, void *p_command_buffer);

	public:
		void initialize();
		void uninitialize();
		void prepare_screen();

		void set_minimum_supported_near_plane(float p_minimum_supported_near_plane);
		// p_current_frame should be an cp_frame_t pointer casted to uint64_t
		void set_current_frame(uint64_t p_current_frame);

		// Expects an ar_world_tracking_provider_t
		void set_world_tracking_provider(uint64_t p_world_tracking_provider);

		// Safe to be called from the game thread
		void start_frame_update();
		void end_frame_update();
		Size2 get_render_target_size();

		// Only safe to be called from the render thread
		uint32_t get_view_count();
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

	static void _bind_methods();
	static const String name;
	static StringName get_signal_name(SignalEnum p_signal);

public:
	static Ref<VisionOSXRInterface> find_interface() {
		return XRServer::get_singleton()->find_interface(name);
	}

	VisionOSXRInterface();
	~VisionOSXRInterface();

	cp_frame_timing_t get_current_timing();

	void emit_signal_enum(SignalEnum p_signal);

	virtual StringName get_name() const override;
	virtual uint32_t get_capabilities() const override;

	virtual TrackingStatus get_tracking_status() const override;
	virtual void trigger_haptic_pulse(const String &p_action_name, const StringName &p_tracker_name, double p_frequency, double p_amplitude, double p_duration_sec, double p_delay_sec = 0) override;

	virtual bool is_initialized() const override;
	virtual bool initialize() override;
	virtual void uninitialize() override;

	// Running the ARKit session.
	// Note that we need to re-run it when the privacy authorizations changed
	// or when a new controller was connected (on visionOS 26 and earlier).
	void run_ar_session();

	// The LayerRenderer and Capabilities are polled from the app delegate when initializing the VisionOSXRInterface,
	// but they need to be updated when the app backgrounds and foregrounds because they are recreated by visionOS
	void update_layer_renderer(cp_layer_renderer_t p_layer_renderer, cp_layer_renderer_capabilities_t p_layer_renderer_capabilities);

	virtual Dictionary get_system_info() override;
	virtual VRSTextureFormat get_vrs_texture_format() override;

	virtual bool supports_play_area_mode(XRInterface::PlayAreaMode p_mode) override;
	virtual XRInterface::PlayAreaMode get_play_area_mode() const override;
	virtual bool set_play_area_mode(XRInterface::PlayAreaMode p_mode) override;

	float get_current_render_quality();
	void set_current_render_quality(float p_render_quality);

	ImmersionStyle get_immersion_style();
	void set_immersion_style(ImmersionStyle p_immersion_style);

	Visibility get_upper_limb_visibility();
	void set_upper_limb_visibility(Visibility p_upper_limb_visibility);

	Visibility get_persistent_system_overlays();
	void set_persistent_system_overlays(Visibility p_persistent_system_overlays);

	// Methods called from the game thread
	virtual void process() override;
	virtual Size2 get_render_target_size() override;

	// Methods only called from the render thread
	virtual uint32_t get_view_count() override {
		return rt.get_view_count();
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

VARIANT_ENUM_CAST(VisionOSXRInterface::ImmersionStyle);
VARIANT_ENUM_CAST(VisionOSXRInterface::Visibility);

#endif // VISIONOS_ENABLED
