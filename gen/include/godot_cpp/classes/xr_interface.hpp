/**************************************************************************/
/*  xr_interface.hpp                                                      */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/projection.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class String;

class XRInterface : public RefCounted {
	GDEXTENSION_CLASS(XRInterface, RefCounted)

public:
	enum Capabilities {
		XR_NONE = 0,
		XR_MONO = 1,
		XR_STEREO = 2,
		XR_QUAD = 4,
		XR_VR = 8,
		XR_AR = 16,
		XR_EXTERNAL = 32,
	};

	enum TrackingStatus {
		XR_NORMAL_TRACKING = 0,
		XR_EXCESSIVE_MOTION = 1,
		XR_INSUFFICIENT_FEATURES = 2,
		XR_UNKNOWN_TRACKING = 3,
		XR_NOT_TRACKING = 4,
	};

	enum PlayAreaMode {
		XR_PLAY_AREA_UNKNOWN = 0,
		XR_PLAY_AREA_3DOF = 1,
		XR_PLAY_AREA_SITTING = 2,
		XR_PLAY_AREA_ROOMSCALE = 3,
		XR_PLAY_AREA_STAGE = 4,
		XR_PLAY_AREA_CUSTOM = 2147483647,
	};

	enum EnvironmentBlendMode {
		XR_ENV_BLEND_MODE_OPAQUE = 0,
		XR_ENV_BLEND_MODE_ADDITIVE = 1,
		XR_ENV_BLEND_MODE_ALPHA_BLEND = 2,
	};

	enum VRSTextureFormat {
		XR_VRS_TEXTURE_FORMAT_UNIFIED = 0,
		XR_VRS_TEXTURE_FORMAT_FRAGMENT_SHADING_RATE = 1,
		XR_VRS_TEXTURE_FORMAT_FRAGMENT_DENSITY_MAP = 2,
	};

	StringName get_name() const;
	uint32_t get_capabilities() const;
	bool is_primary();
	void set_primary(bool p_primary);
	bool is_initialized() const;
	bool initialize();
	void uninitialize();
	Dictionary get_system_info();
	XRInterface::TrackingStatus get_tracking_status() const;
	Vector2 get_render_target_size();
	uint32_t get_view_count();
	void trigger_haptic_pulse(const String &p_action_name, const StringName &p_tracker_name, double p_frequency, double p_amplitude, double p_duration_sec, double p_delay_sec);
	bool supports_play_area_mode(XRInterface::PlayAreaMode p_mode);
	XRInterface::PlayAreaMode get_play_area_mode() const;
	bool set_play_area_mode(XRInterface::PlayAreaMode p_mode);
	PackedVector3Array get_play_area() const;
	bool get_anchor_detection_is_enabled() const;
	void set_anchor_detection_is_enabled(bool p_enable);
	int32_t get_camera_feed_id();
	bool is_passthrough_supported();
	bool is_passthrough_enabled();
	bool start_passthrough();
	void stop_passthrough();
	Transform3D get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform);
	Projection get_projection_for_view(uint32_t p_view, double p_aspect, double p_near, double p_far);
	Array get_supported_environment_blend_modes();
	bool set_environment_blend_mode(XRInterface::EnvironmentBlendMode p_mode);
	XRInterface::EnvironmentBlendMode get_environment_blend_mode() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(XRInterface::Capabilities);
VARIANT_ENUM_CAST(XRInterface::TrackingStatus);
VARIANT_ENUM_CAST(XRInterface::PlayAreaMode);
VARIANT_ENUM_CAST(XRInterface::EnvironmentBlendMode);
VARIANT_ENUM_CAST(XRInterface::VRSTextureFormat);

