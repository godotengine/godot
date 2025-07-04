/**************************************************************************/
/*  distance_joint_3d.h                                                   */
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

#include "scene/3d/physics/joints/joint_3d.h"

#include "modules/jolt_physics/jolt_physics_server_3d.h"

class DistanceJoint3D : public Joint3D {
	GDCLASS(DistanceJoint3D, Joint3D);

public:
	enum Param {
		PARAM_LIMITS_SPRING_STIFFNESS = PhysicsServer3D::DISTANCE_JOINT_LIMITS_SPRING_STIFFNESS,
		PARAM_LIMITS_SPRING_DAMPING = PhysicsServer3D::DISTANCE_JOINT_LIMITS_SPRING_DAMPING,
		PARAM_DISTANCE_MIN = PhysicsServer3D::DISTANCE_JOINT_DISTANCE_MIN,
		PARAM_DISTANCE_MAX = PhysicsServer3D::DISTANCE_JOINT_DISTANCE_MAX,
		PARAM_MAX
	};

	enum PointParam {
		POINT_PARAM_A,
		POINT_PARAM_B,
		POINT_PARAM_MAX
	};

protected:
	real_t params[PARAM_MAX];
	Vector3 point_params[POINT_PARAM_MAX];

	void _configure_joint(RID p_joint, PhysicsBody3D *p_body_a, PhysicsBody3D *p_body_b) override;
	static void _bind_methods();
	PhysicsBody3D *_get_body_from_param(PointParam p_param) const;

public:
	void set_param(Param p_param, real_t p_value);
	real_t get_param(Param p_param) const;

	void set_point_param(PointParam p_param, const Vector3 &p_value);
	Vector3 get_point_param(PointParam p_param) const;
	Vector3 get_global_point(PointParam p_param) const;

	PackedStringArray get_configuration_warnings() const override;

	DistanceJoint3D();
};

VARIANT_ENUM_CAST(DistanceJoint3D::Param);
VARIANT_ENUM_CAST(DistanceJoint3D::PointParam);
