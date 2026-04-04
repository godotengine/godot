/**************************************************************************/
/*  xr_positional_tracker.hpp                                             */
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
#include <godot_cpp/classes/xr_pose.hpp>
#include <godot_cpp/classes/xr_tracker.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class StringName;
struct Transform3D;
struct Vector3;

class XRPositionalTracker : public XRTracker {
	GDEXTENSION_CLASS(XRPositionalTracker, XRTracker)

public:
	enum TrackerHand {
		TRACKER_HAND_UNKNOWN = 0,
		TRACKER_HAND_LEFT = 1,
		TRACKER_HAND_RIGHT = 2,
		TRACKER_HAND_MAX = 3,
	};

	String get_tracker_profile() const;
	void set_tracker_profile(const String &p_profile);
	XRPositionalTracker::TrackerHand get_tracker_hand() const;
	void set_tracker_hand(XRPositionalTracker::TrackerHand p_hand);
	bool has_pose(const StringName &p_name) const;
	Ref<XRPose> get_pose(const StringName &p_name) const;
	void invalidate_pose(const StringName &p_name);
	void set_pose(const StringName &p_name, const Transform3D &p_transform, const Vector3 &p_linear_velocity, const Vector3 &p_angular_velocity, XRPose::TrackingConfidence p_tracking_confidence);
	Variant get_input(const StringName &p_name) const;
	void set_input(const StringName &p_name, const Variant &p_value);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		XRTracker::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(XRPositionalTracker::TrackerHand);

