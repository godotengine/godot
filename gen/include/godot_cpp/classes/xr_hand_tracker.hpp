/**************************************************************************/
/*  xr_hand_tracker.hpp                                                   */
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
#include <godot_cpp/classes/xr_positional_tracker.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class XRHandTracker : public XRPositionalTracker {
	GDEXTENSION_CLASS(XRHandTracker, XRPositionalTracker)

public:
	enum HandTrackingSource {
		HAND_TRACKING_SOURCE_UNKNOWN = 0,
		HAND_TRACKING_SOURCE_UNOBSTRUCTED = 1,
		HAND_TRACKING_SOURCE_CONTROLLER = 2,
		HAND_TRACKING_SOURCE_NOT_TRACKED = 3,
		HAND_TRACKING_SOURCE_MAX = 4,
	};

	enum HandJoint {
		HAND_JOINT_PALM = 0,
		HAND_JOINT_WRIST = 1,
		HAND_JOINT_THUMB_METACARPAL = 2,
		HAND_JOINT_THUMB_PHALANX_PROXIMAL = 3,
		HAND_JOINT_THUMB_PHALANX_DISTAL = 4,
		HAND_JOINT_THUMB_TIP = 5,
		HAND_JOINT_INDEX_FINGER_METACARPAL = 6,
		HAND_JOINT_INDEX_FINGER_PHALANX_PROXIMAL = 7,
		HAND_JOINT_INDEX_FINGER_PHALANX_INTERMEDIATE = 8,
		HAND_JOINT_INDEX_FINGER_PHALANX_DISTAL = 9,
		HAND_JOINT_INDEX_FINGER_TIP = 10,
		HAND_JOINT_MIDDLE_FINGER_METACARPAL = 11,
		HAND_JOINT_MIDDLE_FINGER_PHALANX_PROXIMAL = 12,
		HAND_JOINT_MIDDLE_FINGER_PHALANX_INTERMEDIATE = 13,
		HAND_JOINT_MIDDLE_FINGER_PHALANX_DISTAL = 14,
		HAND_JOINT_MIDDLE_FINGER_TIP = 15,
		HAND_JOINT_RING_FINGER_METACARPAL = 16,
		HAND_JOINT_RING_FINGER_PHALANX_PROXIMAL = 17,
		HAND_JOINT_RING_FINGER_PHALANX_INTERMEDIATE = 18,
		HAND_JOINT_RING_FINGER_PHALANX_DISTAL = 19,
		HAND_JOINT_RING_FINGER_TIP = 20,
		HAND_JOINT_PINKY_FINGER_METACARPAL = 21,
		HAND_JOINT_PINKY_FINGER_PHALANX_PROXIMAL = 22,
		HAND_JOINT_PINKY_FINGER_PHALANX_INTERMEDIATE = 23,
		HAND_JOINT_PINKY_FINGER_PHALANX_DISTAL = 24,
		HAND_JOINT_PINKY_FINGER_TIP = 25,
		HAND_JOINT_MAX = 26,
	};

	enum HandJointFlags : uint64_t {
		HAND_JOINT_FLAG_ORIENTATION_VALID = 1,
		HAND_JOINT_FLAG_ORIENTATION_TRACKED = 2,
		HAND_JOINT_FLAG_POSITION_VALID = 4,
		HAND_JOINT_FLAG_POSITION_TRACKED = 8,
		HAND_JOINT_FLAG_LINEAR_VELOCITY_VALID = 16,
		HAND_JOINT_FLAG_ANGULAR_VELOCITY_VALID = 32,
	};

	void set_has_tracking_data(bool p_has_data);
	bool get_has_tracking_data() const;
	void set_hand_tracking_source(XRHandTracker::HandTrackingSource p_source);
	XRHandTracker::HandTrackingSource get_hand_tracking_source() const;
	void set_hand_joint_flags(XRHandTracker::HandJoint p_joint, BitField<XRHandTracker::HandJointFlags> p_flags);
	BitField<XRHandTracker::HandJointFlags> get_hand_joint_flags(XRHandTracker::HandJoint p_joint) const;
	void set_hand_joint_transform(XRHandTracker::HandJoint p_joint, const Transform3D &p_transform);
	Transform3D get_hand_joint_transform(XRHandTracker::HandJoint p_joint) const;
	void set_hand_joint_radius(XRHandTracker::HandJoint p_joint, float p_radius);
	float get_hand_joint_radius(XRHandTracker::HandJoint p_joint) const;
	void set_hand_joint_linear_velocity(XRHandTracker::HandJoint p_joint, const Vector3 &p_linear_velocity);
	Vector3 get_hand_joint_linear_velocity(XRHandTracker::HandJoint p_joint) const;
	void set_hand_joint_angular_velocity(XRHandTracker::HandJoint p_joint, const Vector3 &p_angular_velocity);
	Vector3 get_hand_joint_angular_velocity(XRHandTracker::HandJoint p_joint) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		XRPositionalTracker::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(XRHandTracker::HandTrackingSource);
VARIANT_ENUM_CAST(XRHandTracker::HandJoint);
VARIANT_BITFIELD_CAST(XRHandTracker::HandJointFlags);

