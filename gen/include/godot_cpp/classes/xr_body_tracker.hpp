/**************************************************************************/
/*  xr_body_tracker.hpp                                                   */
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

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class XRBodyTracker : public XRPositionalTracker {
	GDEXTENSION_CLASS(XRBodyTracker, XRPositionalTracker)

public:
	enum BodyFlags : uint64_t {
		BODY_FLAG_UPPER_BODY_SUPPORTED = 1,
		BODY_FLAG_LOWER_BODY_SUPPORTED = 2,
		BODY_FLAG_HANDS_SUPPORTED = 4,
	};

	enum Joint {
		JOINT_ROOT = 0,
		JOINT_HIPS = 1,
		JOINT_SPINE = 2,
		JOINT_CHEST = 3,
		JOINT_UPPER_CHEST = 4,
		JOINT_NECK = 5,
		JOINT_HEAD = 6,
		JOINT_HEAD_TIP = 7,
		JOINT_LEFT_SHOULDER = 8,
		JOINT_LEFT_UPPER_ARM = 9,
		JOINT_LEFT_LOWER_ARM = 10,
		JOINT_RIGHT_SHOULDER = 11,
		JOINT_RIGHT_UPPER_ARM = 12,
		JOINT_RIGHT_LOWER_ARM = 13,
		JOINT_LEFT_UPPER_LEG = 14,
		JOINT_LEFT_LOWER_LEG = 15,
		JOINT_LEFT_FOOT = 16,
		JOINT_LEFT_TOES = 17,
		JOINT_RIGHT_UPPER_LEG = 18,
		JOINT_RIGHT_LOWER_LEG = 19,
		JOINT_RIGHT_FOOT = 20,
		JOINT_RIGHT_TOES = 21,
		JOINT_LEFT_HAND = 22,
		JOINT_LEFT_PALM = 23,
		JOINT_LEFT_WRIST = 24,
		JOINT_LEFT_THUMB_METACARPAL = 25,
		JOINT_LEFT_THUMB_PHALANX_PROXIMAL = 26,
		JOINT_LEFT_THUMB_PHALANX_DISTAL = 27,
		JOINT_LEFT_THUMB_TIP = 28,
		JOINT_LEFT_INDEX_FINGER_METACARPAL = 29,
		JOINT_LEFT_INDEX_FINGER_PHALANX_PROXIMAL = 30,
		JOINT_LEFT_INDEX_FINGER_PHALANX_INTERMEDIATE = 31,
		JOINT_LEFT_INDEX_FINGER_PHALANX_DISTAL = 32,
		JOINT_LEFT_INDEX_FINGER_TIP = 33,
		JOINT_LEFT_MIDDLE_FINGER_METACARPAL = 34,
		JOINT_LEFT_MIDDLE_FINGER_PHALANX_PROXIMAL = 35,
		JOINT_LEFT_MIDDLE_FINGER_PHALANX_INTERMEDIATE = 36,
		JOINT_LEFT_MIDDLE_FINGER_PHALANX_DISTAL = 37,
		JOINT_LEFT_MIDDLE_FINGER_TIP = 38,
		JOINT_LEFT_RING_FINGER_METACARPAL = 39,
		JOINT_LEFT_RING_FINGER_PHALANX_PROXIMAL = 40,
		JOINT_LEFT_RING_FINGER_PHALANX_INTERMEDIATE = 41,
		JOINT_LEFT_RING_FINGER_PHALANX_DISTAL = 42,
		JOINT_LEFT_RING_FINGER_TIP = 43,
		JOINT_LEFT_PINKY_FINGER_METACARPAL = 44,
		JOINT_LEFT_PINKY_FINGER_PHALANX_PROXIMAL = 45,
		JOINT_LEFT_PINKY_FINGER_PHALANX_INTERMEDIATE = 46,
		JOINT_LEFT_PINKY_FINGER_PHALANX_DISTAL = 47,
		JOINT_LEFT_PINKY_FINGER_TIP = 48,
		JOINT_RIGHT_HAND = 49,
		JOINT_RIGHT_PALM = 50,
		JOINT_RIGHT_WRIST = 51,
		JOINT_RIGHT_THUMB_METACARPAL = 52,
		JOINT_RIGHT_THUMB_PHALANX_PROXIMAL = 53,
		JOINT_RIGHT_THUMB_PHALANX_DISTAL = 54,
		JOINT_RIGHT_THUMB_TIP = 55,
		JOINT_RIGHT_INDEX_FINGER_METACARPAL = 56,
		JOINT_RIGHT_INDEX_FINGER_PHALANX_PROXIMAL = 57,
		JOINT_RIGHT_INDEX_FINGER_PHALANX_INTERMEDIATE = 58,
		JOINT_RIGHT_INDEX_FINGER_PHALANX_DISTAL = 59,
		JOINT_RIGHT_INDEX_FINGER_TIP = 60,
		JOINT_RIGHT_MIDDLE_FINGER_METACARPAL = 61,
		JOINT_RIGHT_MIDDLE_FINGER_PHALANX_PROXIMAL = 62,
		JOINT_RIGHT_MIDDLE_FINGER_PHALANX_INTERMEDIATE = 63,
		JOINT_RIGHT_MIDDLE_FINGER_PHALANX_DISTAL = 64,
		JOINT_RIGHT_MIDDLE_FINGER_TIP = 65,
		JOINT_RIGHT_RING_FINGER_METACARPAL = 66,
		JOINT_RIGHT_RING_FINGER_PHALANX_PROXIMAL = 67,
		JOINT_RIGHT_RING_FINGER_PHALANX_INTERMEDIATE = 68,
		JOINT_RIGHT_RING_FINGER_PHALANX_DISTAL = 69,
		JOINT_RIGHT_RING_FINGER_TIP = 70,
		JOINT_RIGHT_PINKY_FINGER_METACARPAL = 71,
		JOINT_RIGHT_PINKY_FINGER_PHALANX_PROXIMAL = 72,
		JOINT_RIGHT_PINKY_FINGER_PHALANX_INTERMEDIATE = 73,
		JOINT_RIGHT_PINKY_FINGER_PHALANX_DISTAL = 74,
		JOINT_RIGHT_PINKY_FINGER_TIP = 75,
		JOINT_LOWER_CHEST = 76,
		JOINT_LEFT_SCAPULA = 77,
		JOINT_LEFT_WRIST_TWIST = 78,
		JOINT_RIGHT_SCAPULA = 79,
		JOINT_RIGHT_WRIST_TWIST = 80,
		JOINT_LEFT_FOOT_TWIST = 81,
		JOINT_LEFT_HEEL = 82,
		JOINT_LEFT_MIDDLE_FOOT = 83,
		JOINT_RIGHT_FOOT_TWIST = 84,
		JOINT_RIGHT_HEEL = 85,
		JOINT_RIGHT_MIDDLE_FOOT = 86,
		JOINT_MAX = 87,
	};

	enum JointFlags : uint64_t {
		JOINT_FLAG_ORIENTATION_VALID = 1,
		JOINT_FLAG_ORIENTATION_TRACKED = 2,
		JOINT_FLAG_POSITION_VALID = 4,
		JOINT_FLAG_POSITION_TRACKED = 8,
	};

	void set_has_tracking_data(bool p_has_data);
	bool get_has_tracking_data() const;
	void set_body_flags(BitField<XRBodyTracker::BodyFlags> p_flags);
	BitField<XRBodyTracker::BodyFlags> get_body_flags() const;
	void set_joint_flags(XRBodyTracker::Joint p_joint, BitField<XRBodyTracker::JointFlags> p_flags);
	BitField<XRBodyTracker::JointFlags> get_joint_flags(XRBodyTracker::Joint p_joint) const;
	void set_joint_transform(XRBodyTracker::Joint p_joint, const Transform3D &p_transform);
	Transform3D get_joint_transform(XRBodyTracker::Joint p_joint) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		XRPositionalTracker::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_BITFIELD_CAST(XRBodyTracker::BodyFlags);
VARIANT_ENUM_CAST(XRBodyTracker::Joint);
VARIANT_BITFIELD_CAST(XRBodyTracker::JointFlags);

