/**************************************************************************/
/*  slider_joint3d.hpp                                                    */
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

#include <godot_cpp/classes/joint3d.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class SliderJoint3D : public Joint3D {
	GDEXTENSION_CLASS(SliderJoint3D, Joint3D)

public:
	enum Param {
		PARAM_LINEAR_LIMIT_UPPER = 0,
		PARAM_LINEAR_LIMIT_LOWER = 1,
		PARAM_LINEAR_LIMIT_SOFTNESS = 2,
		PARAM_LINEAR_LIMIT_RESTITUTION = 3,
		PARAM_LINEAR_LIMIT_DAMPING = 4,
		PARAM_LINEAR_MOTION_SOFTNESS = 5,
		PARAM_LINEAR_MOTION_RESTITUTION = 6,
		PARAM_LINEAR_MOTION_DAMPING = 7,
		PARAM_LINEAR_ORTHOGONAL_SOFTNESS = 8,
		PARAM_LINEAR_ORTHOGONAL_RESTITUTION = 9,
		PARAM_LINEAR_ORTHOGONAL_DAMPING = 10,
		PARAM_ANGULAR_LIMIT_UPPER = 11,
		PARAM_ANGULAR_LIMIT_LOWER = 12,
		PARAM_ANGULAR_LIMIT_SOFTNESS = 13,
		PARAM_ANGULAR_LIMIT_RESTITUTION = 14,
		PARAM_ANGULAR_LIMIT_DAMPING = 15,
		PARAM_ANGULAR_MOTION_SOFTNESS = 16,
		PARAM_ANGULAR_MOTION_RESTITUTION = 17,
		PARAM_ANGULAR_MOTION_DAMPING = 18,
		PARAM_ANGULAR_ORTHOGONAL_SOFTNESS = 19,
		PARAM_ANGULAR_ORTHOGONAL_RESTITUTION = 20,
		PARAM_ANGULAR_ORTHOGONAL_DAMPING = 21,
		PARAM_MAX = 22,
	};

	void set_param(SliderJoint3D::Param p_param, float p_value);
	float get_param(SliderJoint3D::Param p_param) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Joint3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(SliderJoint3D::Param);

