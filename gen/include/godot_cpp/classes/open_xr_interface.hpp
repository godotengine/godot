/**************************************************************************/
/*  open_xr_interface.hpp                                                 */
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
#include <godot_cpp/classes/xr_interface.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class String;

class OpenXRInterface : public XRInterface {
	GDEXTENSION_CLASS(OpenXRInterface, XRInterface)

public:
	enum SessionState {
		SESSION_STATE_UNKNOWN = 0,
		SESSION_STATE_IDLE = 1,
		SESSION_STATE_READY = 2,
		SESSION_STATE_SYNCHRONIZED = 3,
		SESSION_STATE_VISIBLE = 4,
		SESSION_STATE_FOCUSED = 5,
		SESSION_STATE_STOPPING = 6,
		SESSION_STATE_LOSS_PENDING = 7,
		SESSION_STATE_EXITING = 8,
	};

	enum Hand {
		HAND_LEFT = 0,
		HAND_RIGHT = 1,
		HAND_MAX = 2,
	};

	enum HandMotionRange {
		HAND_MOTION_RANGE_UNOBSTRUCTED = 0,
		HAND_MOTION_RANGE_CONFORM_TO_CONTROLLER = 1,
		HAND_MOTION_RANGE_MAX = 2,
	};

	enum HandTrackedSource {
		HAND_TRACKED_SOURCE_UNKNOWN = 0,
		HAND_TRACKED_SOURCE_UNOBSTRUCTED = 1,
		HAND_TRACKED_SOURCE_CONTROLLER = 2,
		HAND_TRACKED_SOURCE_MAX = 3,
	};

	enum HandJoints {
		HAND_JOINT_PALM = 0,
		HAND_JOINT_WRIST = 1,
		HAND_JOINT_THUMB_METACARPAL = 2,
		HAND_JOINT_THUMB_PROXIMAL = 3,
		HAND_JOINT_THUMB_DISTAL = 4,
		HAND_JOINT_THUMB_TIP = 5,
		HAND_JOINT_INDEX_METACARPAL = 6,
		HAND_JOINT_INDEX_PROXIMAL = 7,
		HAND_JOINT_INDEX_INTERMEDIATE = 8,
		HAND_JOINT_INDEX_DISTAL = 9,
		HAND_JOINT_INDEX_TIP = 10,
		HAND_JOINT_MIDDLE_METACARPAL = 11,
		HAND_JOINT_MIDDLE_PROXIMAL = 12,
		HAND_JOINT_MIDDLE_INTERMEDIATE = 13,
		HAND_JOINT_MIDDLE_DISTAL = 14,
		HAND_JOINT_MIDDLE_TIP = 15,
		HAND_JOINT_RING_METACARPAL = 16,
		HAND_JOINT_RING_PROXIMAL = 17,
		HAND_JOINT_RING_INTERMEDIATE = 18,
		HAND_JOINT_RING_DISTAL = 19,
		HAND_JOINT_RING_TIP = 20,
		HAND_JOINT_LITTLE_METACARPAL = 21,
		HAND_JOINT_LITTLE_PROXIMAL = 22,
		HAND_JOINT_LITTLE_INTERMEDIATE = 23,
		HAND_JOINT_LITTLE_DISTAL = 24,
		HAND_JOINT_LITTLE_TIP = 25,
		HAND_JOINT_MAX = 26,
	};

	enum PerfSettingsLevel {
		PERF_SETTINGS_LEVEL_POWER_SAVINGS = 0,
		PERF_SETTINGS_LEVEL_SUSTAINED_LOW = 1,
		PERF_SETTINGS_LEVEL_SUSTAINED_HIGH = 2,
		PERF_SETTINGS_LEVEL_BOOST = 3,
	};

	enum PerfSettingsSubDomain {
		PERF_SETTINGS_SUB_DOMAIN_COMPOSITING = 0,
		PERF_SETTINGS_SUB_DOMAIN_RENDERING = 1,
		PERF_SETTINGS_SUB_DOMAIN_THERMAL = 2,
	};

	enum PerfSettingsNotificationLevel {
		PERF_SETTINGS_NOTIF_LEVEL_NORMAL = 0,
		PERF_SETTINGS_NOTIF_LEVEL_WARNING = 1,
		PERF_SETTINGS_NOTIF_LEVEL_IMPAIRED = 2,
	};

	enum HandJointFlags : uint64_t {
		HAND_JOINT_NONE = 0,
		HAND_JOINT_ORIENTATION_VALID = 1,
		HAND_JOINT_ORIENTATION_TRACKED = 2,
		HAND_JOINT_POSITION_VALID = 4,
		HAND_JOINT_POSITION_TRACKED = 8,
		HAND_JOINT_LINEAR_VELOCITY_VALID = 16,
		HAND_JOINT_ANGULAR_VELOCITY_VALID = 32,
	};

	OpenXRInterface::SessionState get_session_state();
	float get_display_refresh_rate() const;
	void set_display_refresh_rate(float p_refresh_rate);
	double get_render_target_size_multiplier() const;
	void set_render_target_size_multiplier(double p_multiplier);
	bool is_foveation_supported() const;
	int32_t get_foveation_level() const;
	void set_foveation_level(int32_t p_foveation_level);
	bool get_foveation_dynamic() const;
	void set_foveation_dynamic(bool p_foveation_dynamic);
	bool is_action_set_active(const String &p_name) const;
	void set_action_set_active(const String &p_name, bool p_active);
	Array get_action_sets() const;
	Array get_available_display_refresh_rates() const;
	void set_motion_range(OpenXRInterface::Hand p_hand, OpenXRInterface::HandMotionRange p_motion_range);
	OpenXRInterface::HandMotionRange get_motion_range(OpenXRInterface::Hand p_hand) const;
	OpenXRInterface::HandTrackedSource get_hand_tracking_source(OpenXRInterface::Hand p_hand) const;
	BitField<OpenXRInterface::HandJointFlags> get_hand_joint_flags(OpenXRInterface::Hand p_hand, OpenXRInterface::HandJoints p_joint) const;
	Quaternion get_hand_joint_rotation(OpenXRInterface::Hand p_hand, OpenXRInterface::HandJoints p_joint) const;
	Vector3 get_hand_joint_position(OpenXRInterface::Hand p_hand, OpenXRInterface::HandJoints p_joint) const;
	float get_hand_joint_radius(OpenXRInterface::Hand p_hand, OpenXRInterface::HandJoints p_joint) const;
	Vector3 get_hand_joint_linear_velocity(OpenXRInterface::Hand p_hand, OpenXRInterface::HandJoints p_joint) const;
	Vector3 get_hand_joint_angular_velocity(OpenXRInterface::Hand p_hand, OpenXRInterface::HandJoints p_joint) const;
	bool is_hand_tracking_supported();
	bool is_hand_interaction_supported() const;
	bool is_eye_gaze_interaction_supported();
	float get_vrs_min_radius() const;
	void set_vrs_min_radius(float p_radius);
	float get_vrs_strength() const;
	void set_vrs_strength(float p_strength);
	void set_cpu_level(OpenXRInterface::PerfSettingsLevel p_level);
	void set_gpu_level(OpenXRInterface::PerfSettingsLevel p_level);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		XRInterface::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(OpenXRInterface::SessionState);
VARIANT_ENUM_CAST(OpenXRInterface::Hand);
VARIANT_ENUM_CAST(OpenXRInterface::HandMotionRange);
VARIANT_ENUM_CAST(OpenXRInterface::HandTrackedSource);
VARIANT_ENUM_CAST(OpenXRInterface::HandJoints);
VARIANT_ENUM_CAST(OpenXRInterface::PerfSettingsLevel);
VARIANT_ENUM_CAST(OpenXRInterface::PerfSettingsSubDomain);
VARIANT_ENUM_CAST(OpenXRInterface::PerfSettingsNotificationLevel);
VARIANT_BITFIELD_CAST(OpenXRInterface::HandJointFlags);

