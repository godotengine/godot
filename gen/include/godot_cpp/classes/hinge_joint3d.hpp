/**************************************************************************/
/*  hinge_joint3d.hpp                                                     */
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

class HingeJoint3D : public Joint3D {
	GDEXTENSION_CLASS(HingeJoint3D, Joint3D)

public:
	enum Param {
		PARAM_BIAS = 0,
		PARAM_LIMIT_UPPER = 1,
		PARAM_LIMIT_LOWER = 2,
		PARAM_LIMIT_BIAS = 3,
		PARAM_LIMIT_SOFTNESS = 4,
		PARAM_LIMIT_RELAXATION = 5,
		PARAM_MOTOR_TARGET_VELOCITY = 6,
		PARAM_MOTOR_MAX_IMPULSE = 7,
		PARAM_MAX = 8,
	};

	enum Flag {
		FLAG_USE_LIMIT = 0,
		FLAG_ENABLE_MOTOR = 1,
		FLAG_MAX = 2,
	};

	void set_param(HingeJoint3D::Param p_param, float p_value);
	float get_param(HingeJoint3D::Param p_param) const;
	void set_flag(HingeJoint3D::Flag p_flag, bool p_enabled);
	bool get_flag(HingeJoint3D::Flag p_flag) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Joint3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(HingeJoint3D::Param);
VARIANT_ENUM_CAST(HingeJoint3D::Flag);

