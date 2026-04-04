/**************************************************************************/
/*  generic6_dof_joint3d.hpp                                              */
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

class Generic6DOFJoint3D : public Joint3D {
	GDEXTENSION_CLASS(Generic6DOFJoint3D, Joint3D)

public:
	enum Param {
		PARAM_LINEAR_LOWER_LIMIT = 0,
		PARAM_LINEAR_UPPER_LIMIT = 1,
		PARAM_LINEAR_LIMIT_SOFTNESS = 2,
		PARAM_LINEAR_RESTITUTION = 3,
		PARAM_LINEAR_DAMPING = 4,
		PARAM_LINEAR_MOTOR_TARGET_VELOCITY = 5,
		PARAM_LINEAR_MOTOR_FORCE_LIMIT = 6,
		PARAM_LINEAR_SPRING_STIFFNESS = 7,
		PARAM_LINEAR_SPRING_DAMPING = 8,
		PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT = 9,
		PARAM_ANGULAR_LOWER_LIMIT = 10,
		PARAM_ANGULAR_UPPER_LIMIT = 11,
		PARAM_ANGULAR_LIMIT_SOFTNESS = 12,
		PARAM_ANGULAR_DAMPING = 13,
		PARAM_ANGULAR_RESTITUTION = 14,
		PARAM_ANGULAR_FORCE_LIMIT = 15,
		PARAM_ANGULAR_ERP = 16,
		PARAM_ANGULAR_MOTOR_TARGET_VELOCITY = 17,
		PARAM_ANGULAR_MOTOR_FORCE_LIMIT = 18,
		PARAM_ANGULAR_SPRING_STIFFNESS = 19,
		PARAM_ANGULAR_SPRING_DAMPING = 20,
		PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT = 21,
		PARAM_MAX = 22,
	};

	enum Flag {
		FLAG_ENABLE_LINEAR_LIMIT = 0,
		FLAG_ENABLE_ANGULAR_LIMIT = 1,
		FLAG_ENABLE_LINEAR_SPRING = 3,
		FLAG_ENABLE_ANGULAR_SPRING = 2,
		FLAG_ENABLE_MOTOR = 4,
		FLAG_ENABLE_LINEAR_MOTOR = 5,
		FLAG_MAX = 6,
	};

	void set_param_x(Generic6DOFJoint3D::Param p_param, float p_value);
	float get_param_x(Generic6DOFJoint3D::Param p_param) const;
	void set_param_y(Generic6DOFJoint3D::Param p_param, float p_value);
	float get_param_y(Generic6DOFJoint3D::Param p_param) const;
	void set_param_z(Generic6DOFJoint3D::Param p_param, float p_value);
	float get_param_z(Generic6DOFJoint3D::Param p_param) const;
	void set_flag_x(Generic6DOFJoint3D::Flag p_flag, bool p_value);
	bool get_flag_x(Generic6DOFJoint3D::Flag p_flag) const;
	void set_flag_y(Generic6DOFJoint3D::Flag p_flag, bool p_value);
	bool get_flag_y(Generic6DOFJoint3D::Flag p_flag) const;
	void set_flag_z(Generic6DOFJoint3D::Flag p_flag, bool p_value);
	bool get_flag_z(Generic6DOFJoint3D::Flag p_flag) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Joint3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Generic6DOFJoint3D::Param);
VARIANT_ENUM_CAST(Generic6DOFJoint3D::Flag);

