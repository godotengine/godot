/**************************************************************************/
/*  xr_pose.hpp                                                           */
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
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class XRPose : public RefCounted {
	GDEXTENSION_CLASS(XRPose, RefCounted)

public:
	enum TrackingConfidence {
		XR_TRACKING_CONFIDENCE_NONE = 0,
		XR_TRACKING_CONFIDENCE_LOW = 1,
		XR_TRACKING_CONFIDENCE_HIGH = 2,
	};

	void set_has_tracking_data(bool p_has_tracking_data);
	bool get_has_tracking_data() const;
	void set_name(const StringName &p_name);
	StringName get_name() const;
	void set_transform(const Transform3D &p_transform);
	Transform3D get_transform() const;
	Transform3D get_adjusted_transform() const;
	void set_linear_velocity(const Vector3 &p_velocity);
	Vector3 get_linear_velocity() const;
	void set_angular_velocity(const Vector3 &p_velocity);
	Vector3 get_angular_velocity() const;
	void set_tracking_confidence(XRPose::TrackingConfidence p_tracking_confidence);
	XRPose::TrackingConfidence get_tracking_confidence() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(XRPose::TrackingConfidence);

