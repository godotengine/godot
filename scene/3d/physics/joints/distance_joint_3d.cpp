/**************************************************************************/
/*  distance_joint_3d.cpp                                                 */
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

#include "distance_joint_3d.h"

void DistanceJoint3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &DistanceJoint3D::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &DistanceJoint3D::get_param);
	ClassDB::bind_method(D_METHOD("set_point_param", "point", "value"), &DistanceJoint3D::set_point_param);
	ClassDB::bind_method(D_METHOD("get_point_param", "point"), &DistanceJoint3D::get_point_param);
	ClassDB::bind_method(D_METHOD("get_global_point", "point"), &DistanceJoint3D::get_global_point);

	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "spring/stiffness", PROPERTY_HINT_RANGE, "0,100,or_greater,suffix:N/m"), "set_param", "get_param", PARAM_LIMITS_SPRING_STIFFNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "spring/damping", PROPERTY_HINT_RANGE, "0,2,or_greater"), "set_param", "get_param", PARAM_LIMITS_SPRING_DAMPING);

	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "distance/min", PROPERTY_HINT_RANGE, "0,100,or_greater,suffix:m"), "set_param", "get_param", PARAM_DISTANCE_MIN);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "distance/max", PROPERTY_HINT_RANGE, "0,100,or_greater,suffix:m"), "set_param", "get_param", PARAM_DISTANCE_MAX);

	ADD_PROPERTYI(PropertyInfo(Variant::VECTOR3, "anchor/a"), "set_point_param", "get_point_param", POINT_PARAM_A);
	ADD_PROPERTYI(PropertyInfo(Variant::VECTOR3, "anchor/b"), "set_point_param", "get_point_param", POINT_PARAM_B);

	BIND_ENUM_CONSTANT(PARAM_LIMITS_SPRING_STIFFNESS);
	BIND_ENUM_CONSTANT(PARAM_LIMITS_SPRING_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_DISTANCE_MIN);
	BIND_ENUM_CONSTANT(PARAM_DISTANCE_MAX);
	BIND_ENUM_CONSTANT(PARAM_MAX);

	BIND_ENUM_CONSTANT(POINT_PARAM_A);
	BIND_ENUM_CONSTANT(POINT_PARAM_B);
}

void DistanceJoint3D::set_param(Param p_param, real_t p_value) {
	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params[p_param] = p_value;
	if (is_configured()) {
		PhysicsServer3D::get_singleton()->distance_joint_set_param(get_rid(), PhysicsServer3D::DistanceJointParam(p_param), p_value);
	}
}

real_t DistanceJoint3D::get_param(Param p_param) const {
	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params[p_param];
}

void DistanceJoint3D::set_point_param(PointParam p_param, const Vector3 &p_value) {
	ERR_FAIL_INDEX(p_param, POINT_PARAM_MAX);
	point_params[p_param] = p_value;
	if (is_configured()) {
		_disconnect_signals();
		_update_joint();
	}
}

Vector3 DistanceJoint3D::get_point_param(PointParam p_param) const {
	ERR_FAIL_INDEX_V(p_param, POINT_PARAM_MAX, Vector3());
	return point_params[p_param];
}

Vector3 DistanceJoint3D::get_global_point(PointParam p_param) const {
	const PhysicsBody3D *body = _get_body_from_param(p_param);
	const Vector3 local_point = get_point_param(p_param);
	if (body == nullptr) {
		return to_global(local_point);
	}
	return body->to_global(local_point);
}

PhysicsBody3D *DistanceJoint3D::_get_body_from_param(PointParam p_param) const {
	const NodePath node_path = p_param == POINT_PARAM_A ? get_node_a() : get_node_b();
	Node *node = get_node_or_null(node_path);
	return Object::cast_to<PhysicsBody3D>(node);
}

void DistanceJoint3D::_configure_joint(RID p_joint, PhysicsBody3D *p_body_a, PhysicsBody3D *p_body_b) {
	PhysicsServer3D *physics_server = PhysicsServer3D::get_singleton();
	ERR_FAIL_NULL(physics_server);

	const bool are_bodies_switched = _get_body_from_param(POINT_PARAM_A) == nullptr;

	const Vector3 global_position = are_bodies_switched ? get_global_point(POINT_PARAM_A) : get_global_point(POINT_PARAM_B);
	const Vector3 point_a = get_point_param(POINT_PARAM_A);
	const Vector3 point_b = get_point_param(POINT_PARAM_B);
	const Vector3 p_body_a_point = are_bodies_switched ? point_b : point_a;
	const Vector3 p_body_b_point = are_bodies_switched ? point_a : point_b;

	physics_server->joint_make_distance(
			p_joint,
			p_body_a->get_rid(),
			p_body_a_point,
			p_body_b != nullptr ? p_body_b->get_rid() : RID(),
			p_body_b != nullptr ? p_body_b_point : global_position);

	for (int i = 0; i < PARAM_MAX; i++) {
		physics_server->distance_joint_set_param(p_joint, PhysicsServer3D::DistanceJointParam(i), params[i]);
	}
}

PackedStringArray DistanceJoint3D::get_configuration_warnings() const {
	PackedStringArray warnings = Joint3D::get_configuration_warnings();

	JoltPhysicsServer3D *jolt_physics_server = JoltPhysicsServer3D::get_singleton();
	if (!jolt_physics_server) {
		warnings.push_back(RTR("DistanceJoint3D is only compatible with Jolt Physics. Please change your Physics Engine in Project Settings."));
	}
	return warnings;
}

DistanceJoint3D::DistanceJoint3D() {
	params[PARAM_LIMITS_SPRING_STIFFNESS] = 0.0;
	params[PARAM_LIMITS_SPRING_DAMPING] = 0.0;
	params[PARAM_DISTANCE_MIN] = 0.0;
	params[PARAM_DISTANCE_MAX] = INFINITY;

	point_params[POINT_PARAM_A] = Vector3(0.0, 0.0, 0.0);
	point_params[POINT_PARAM_B] = Vector3(0.0, 0.0, 0.0);
}
