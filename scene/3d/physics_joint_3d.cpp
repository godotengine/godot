/*************************************************************************/
/*  physics_joint_3d.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "physics_joint_3d.h"

#include "scene/scene_string_names.h"

void Joint3D::_disconnect_signals() {
	Node *node_a = get_node_or_null(a);
	PhysicsBody3D *body_a = Object::cast_to<PhysicsBody3D>(node_a);
	if (body_a) {
		body_a->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Joint3D::_body_exit_tree));
	}

	Node *node_b = get_node_or_null(b);
	PhysicsBody3D *body_b = Object::cast_to<PhysicsBody3D>(node_b);
	if (body_b) {
		body_b->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Joint3D::_body_exit_tree));
	}
}

void Joint3D::_body_exit_tree() {
	_disconnect_signals();
	_update_joint(true);
}

void Joint3D::_update_joint(bool p_only_free) {
	if (ba.is_valid() && bb.is_valid()) {
		PhysicsServer3D::get_singleton()->body_remove_collision_exception(ba, bb);
		PhysicsServer3D::get_singleton()->body_remove_collision_exception(bb, ba);
	}

	ba = RID();
	bb = RID();

	configured = false;

	if (p_only_free || !is_inside_tree()) {
		PhysicsServer3D::get_singleton()->joint_clear(joint);
		warning = String();
		update_configuration_warnings();
		return;
	}

	Node *node_a = get_node_or_null(a);
	Node *node_b = get_node_or_null(b);

	PhysicsBody3D *body_a = Object::cast_to<PhysicsBody3D>(node_a);
	PhysicsBody3D *body_b = Object::cast_to<PhysicsBody3D>(node_b);

	if (node_a && !body_a && node_b && !body_b) {
		warning = TTR("Node A and Node B must be PhysicsBody3Ds");
	} else if (node_a && !body_a) {
		warning = TTR("Node A must be a PhysicsBody3D");
	} else if (node_b && !body_b) {
		warning = TTR("Node B must be a PhysicsBody3D");
	} else if (!body_a && !body_b) {
		warning = TTR("Joint is not connected to any PhysicsBody3Ds");
	} else if (body_a == body_b) {
		warning = TTR("Node A and Node B must be different PhysicsBody3Ds");
	} else {
		warning = String();
	}

	update_configuration_warnings();

	if (!warning.is_empty()) {
		PhysicsServer3D::get_singleton()->joint_clear(joint);
		return;
	}

	configured = true;

	if (body_a) {
		_configure_joint(joint, body_a, body_b);
	} else if (body_b) {
		_configure_joint(joint, body_b, nullptr);
	}

	PhysicsServer3D::get_singleton()->joint_set_solver_priority(joint, solver_priority);

	if (body_a) {
		ba = body_a->get_rid();
		body_a->connect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Joint3D::_body_exit_tree));
	}

	if (body_b) {
		bb = body_b->get_rid();
		body_b->connect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Joint3D::_body_exit_tree));
	}

	PhysicsServer3D::get_singleton()->joint_disable_collisions_between_bodies(joint, exclude_from_collision);
}

void Joint3D::set_node_a(const NodePath &p_node_a) {
	if (a == p_node_a) {
		return;
	}

	if (joint.is_valid()) {
		_disconnect_signals();
	}

	a = p_node_a;
	_update_joint();
}

NodePath Joint3D::get_node_a() const {
	return a;
}

void Joint3D::set_node_b(const NodePath &p_node_b) {
	if (b == p_node_b) {
		return;
	}

	if (joint.is_valid()) {
		_disconnect_signals();
	}

	b = p_node_b;
	_update_joint();
}

NodePath Joint3D::get_node_b() const {
	return b;
}

void Joint3D::set_solver_priority(int p_priority) {
	solver_priority = p_priority;
	if (joint.is_valid()) {
		PhysicsServer3D::get_singleton()->joint_set_solver_priority(joint, solver_priority);
	}
}

int Joint3D::get_solver_priority() const {
	return solver_priority;
}

void Joint3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			_update_joint();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (joint.is_valid()) {
				_disconnect_signals();
				_update_joint(true);
			}
		} break;
	}
}

void Joint3D::set_exclude_nodes_from_collision(bool p_enable) {
	if (exclude_from_collision == p_enable) {
		return;
	}
	exclude_from_collision = p_enable;
	_update_joint();
}

bool Joint3D::get_exclude_nodes_from_collision() const {
	return exclude_from_collision;
}

TypedArray<String> Joint3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node3D::get_configuration_warnings();

	if (!warning.is_empty()) {
		warnings.push_back(warning);
	}

	return warnings;
}

void Joint3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_node_a", "node"), &Joint3D::set_node_a);
	ClassDB::bind_method(D_METHOD("get_node_a"), &Joint3D::get_node_a);

	ClassDB::bind_method(D_METHOD("set_node_b", "node"), &Joint3D::set_node_b);
	ClassDB::bind_method(D_METHOD("get_node_b"), &Joint3D::get_node_b);

	ClassDB::bind_method(D_METHOD("set_solver_priority", "priority"), &Joint3D::set_solver_priority);
	ClassDB::bind_method(D_METHOD("get_solver_priority"), &Joint3D::get_solver_priority);

	ClassDB::bind_method(D_METHOD("set_exclude_nodes_from_collision", "enable"), &Joint3D::set_exclude_nodes_from_collision);
	ClassDB::bind_method(D_METHOD("get_exclude_nodes_from_collision"), &Joint3D::get_exclude_nodes_from_collision);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "nodes/node_a", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "PhysicsBody3D"), "set_node_a", "get_node_a");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "nodes/node_b", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "PhysicsBody3D"), "set_node_b", "get_node_b");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "solver/priority", PROPERTY_HINT_RANGE, "1,8,1"), "set_solver_priority", "get_solver_priority");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collision/exclude_nodes"), "set_exclude_nodes_from_collision", "get_exclude_nodes_from_collision");
}

Joint3D::Joint3D() {
	set_notify_transform(true);
	joint = PhysicsServer3D::get_singleton()->joint_create();
}

Joint3D::~Joint3D() {
	PhysicsServer3D::get_singleton()->free(joint);
}

///////////////////////////////////

void PinJoint3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &PinJoint3D::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &PinJoint3D::get_param);

	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "params/bias", PROPERTY_HINT_RANGE, "0.01,0.99,0.01"), "set_param", "get_param", PARAM_BIAS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "params/damping", PROPERTY_HINT_RANGE, "0.01,8.0,0.01"), "set_param", "get_param", PARAM_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "params/impulse_clamp", PROPERTY_HINT_RANGE, "0.0,64.0,0.01"), "set_param", "get_param", PARAM_IMPULSE_CLAMP);

	BIND_ENUM_CONSTANT(PARAM_BIAS);
	BIND_ENUM_CONSTANT(PARAM_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_IMPULSE_CLAMP);
}

void PinJoint3D::set_param(Param p_param, real_t p_value) {
	ERR_FAIL_INDEX(p_param, 3);
	params[p_param] = p_value;
	if (is_configured()) {
		PhysicsServer3D::get_singleton()->pin_joint_set_param(get_joint(), PhysicsServer3D::PinJointParam(p_param), p_value);
	}
}

real_t PinJoint3D::get_param(Param p_param) const {
	ERR_FAIL_INDEX_V(p_param, 3, 0);
	return params[p_param];
}

void PinJoint3D::_configure_joint(RID p_joint, PhysicsBody3D *body_a, PhysicsBody3D *body_b) {
	Vector3 pinpos = get_global_transform().origin;
	Vector3 local_a = body_a->get_global_transform().affine_inverse().xform(pinpos);
	Vector3 local_b;

	if (body_b) {
		local_b = body_b->get_global_transform().affine_inverse().xform(pinpos);
	} else {
		local_b = pinpos;
	}

	PhysicsServer3D::get_singleton()->joint_make_pin(p_joint, body_a->get_rid(), local_a, body_b ? body_b->get_rid() : RID(), local_b);
	for (int i = 0; i < 3; i++) {
		PhysicsServer3D::get_singleton()->pin_joint_set_param(p_joint, PhysicsServer3D::PinJointParam(i), params[i]);
	}
}

PinJoint3D::PinJoint3D() {
	params[PARAM_BIAS] = 0.3;
	params[PARAM_DAMPING] = 1;
	params[PARAM_IMPULSE_CLAMP] = 0;
}

/////////////////////////////////////////////////

///////////////////////////////////

void HingeJoint3D::_set_upper_limit(real_t p_limit) {
	set_param(PARAM_LIMIT_UPPER, Math::deg2rad(p_limit));
}

real_t HingeJoint3D::_get_upper_limit() const {
	return Math::rad2deg(get_param(PARAM_LIMIT_UPPER));
}

void HingeJoint3D::_set_lower_limit(real_t p_limit) {
	set_param(PARAM_LIMIT_LOWER, Math::deg2rad(p_limit));
}

real_t HingeJoint3D::_get_lower_limit() const {
	return Math::rad2deg(get_param(PARAM_LIMIT_LOWER));
}

void HingeJoint3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &HingeJoint3D::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &HingeJoint3D::get_param);

	ClassDB::bind_method(D_METHOD("set_flag", "flag", "enabled"), &HingeJoint3D::set_flag);
	ClassDB::bind_method(D_METHOD("get_flag", "flag"), &HingeJoint3D::get_flag);

	ClassDB::bind_method(D_METHOD("_set_upper_limit", "upper_limit"), &HingeJoint3D::_set_upper_limit);
	ClassDB::bind_method(D_METHOD("_get_upper_limit"), &HingeJoint3D::_get_upper_limit);

	ClassDB::bind_method(D_METHOD("_set_lower_limit", "lower_limit"), &HingeJoint3D::_set_lower_limit);
	ClassDB::bind_method(D_METHOD("_get_lower_limit"), &HingeJoint3D::_get_lower_limit);

	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "params/bias", PROPERTY_HINT_RANGE, "0.00,0.99,0.01"), "set_param", "get_param", PARAM_BIAS);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_limit/enable"), "set_flag", "get_flag", FLAG_USE_LIMIT);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_limit/upper", PROPERTY_HINT_RANGE, "-180,180,0.1"), "_set_upper_limit", "_get_upper_limit");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_limit/lower", PROPERTY_HINT_RANGE, "-180,180,0.1"), "_set_lower_limit", "_get_lower_limit");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit/bias", PROPERTY_HINT_RANGE, "0.01,0.99,0.01"), "set_param", "get_param", PARAM_LIMIT_BIAS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param", "get_param", PARAM_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit/relaxation", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param", "get_param", PARAM_LIMIT_RELAXATION);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "motor/enable"), "set_flag", "get_flag", FLAG_ENABLE_MOTOR);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "motor/target_velocity", PROPERTY_HINT_RANGE, "-200,200,0.01,or_greater,or_lesser"), "set_param", "get_param", PARAM_MOTOR_TARGET_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "motor/max_impulse", PROPERTY_HINT_RANGE, "0.01,1024,0.01"), "set_param", "get_param", PARAM_MOTOR_MAX_IMPULSE);

	BIND_ENUM_CONSTANT(PARAM_BIAS);
	BIND_ENUM_CONSTANT(PARAM_LIMIT_UPPER);
	BIND_ENUM_CONSTANT(PARAM_LIMIT_LOWER);
	BIND_ENUM_CONSTANT(PARAM_LIMIT_BIAS);
	BIND_ENUM_CONSTANT(PARAM_LIMIT_SOFTNESS);
	BIND_ENUM_CONSTANT(PARAM_LIMIT_RELAXATION);
	BIND_ENUM_CONSTANT(PARAM_MOTOR_TARGET_VELOCITY);
	BIND_ENUM_CONSTANT(PARAM_MOTOR_MAX_IMPULSE);
	BIND_ENUM_CONSTANT(PARAM_MAX);

	BIND_ENUM_CONSTANT(FLAG_USE_LIMIT);
	BIND_ENUM_CONSTANT(FLAG_ENABLE_MOTOR);
	BIND_ENUM_CONSTANT(FLAG_MAX);
}

void HingeJoint3D::set_param(Param p_param, real_t p_value) {
	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params[p_param] = p_value;
	if (is_configured()) {
		PhysicsServer3D::get_singleton()->hinge_joint_set_param(get_joint(), PhysicsServer3D::HingeJointParam(p_param), p_value);
	}

	update_gizmo();
}

real_t HingeJoint3D::get_param(Param p_param) const {
	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params[p_param];
}

void HingeJoint3D::set_flag(Flag p_flag, bool p_value) {
	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	flags[p_flag] = p_value;
	if (is_configured()) {
		PhysicsServer3D::get_singleton()->hinge_joint_set_flag(get_joint(), PhysicsServer3D::HingeJointFlag(p_flag), p_value);
	}

	update_gizmo();
}

bool HingeJoint3D::get_flag(Flag p_flag) const {
	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	return flags[p_flag];
}

void HingeJoint3D::_configure_joint(RID p_joint, PhysicsBody3D *body_a, PhysicsBody3D *body_b) {
	Transform gt = get_global_transform();
	Transform ainv = body_a->get_global_transform().affine_inverse();

	Transform local_a = ainv * gt;
	local_a.orthonormalize();
	Transform local_b = gt;

	if (body_b) {
		Transform binv = body_b->get_global_transform().affine_inverse();
		local_b = binv * gt;
	}

	local_b.orthonormalize();

	PhysicsServer3D::get_singleton()->joint_make_hinge(p_joint, body_a->get_rid(), local_a, body_b ? body_b->get_rid() : RID(), local_b);
	for (int i = 0; i < PARAM_MAX; i++) {
		PhysicsServer3D::get_singleton()->hinge_joint_set_param(p_joint, PhysicsServer3D::HingeJointParam(i), params[i]);
	}
	for (int i = 0; i < FLAG_MAX; i++) {
		set_flag(Flag(i), flags[i]);
		PhysicsServer3D::get_singleton()->hinge_joint_set_flag(p_joint, PhysicsServer3D::HingeJointFlag(i), flags[i]);
	}
}

HingeJoint3D::HingeJoint3D() {
	params[PARAM_BIAS] = 0.3;
	params[PARAM_LIMIT_UPPER] = Math_PI * 0.5;
	params[PARAM_LIMIT_LOWER] = -Math_PI * 0.5;
	params[PARAM_LIMIT_BIAS] = 0.3;
	params[PARAM_LIMIT_SOFTNESS] = 0.9;
	params[PARAM_LIMIT_RELAXATION] = 1.0;
	params[PARAM_MOTOR_TARGET_VELOCITY] = 1;
	params[PARAM_MOTOR_MAX_IMPULSE] = 1;

	flags[FLAG_USE_LIMIT] = false;
	flags[FLAG_ENABLE_MOTOR] = false;
}

/////////////////////////////////////////////////

//////////////////////////////////

void SliderJoint3D::_set_upper_limit_angular(real_t p_limit_angular) {
	set_param(PARAM_ANGULAR_LIMIT_UPPER, Math::deg2rad(p_limit_angular));
}

real_t SliderJoint3D::_get_upper_limit_angular() const {
	return Math::rad2deg(get_param(PARAM_ANGULAR_LIMIT_UPPER));
}

void SliderJoint3D::_set_lower_limit_angular(real_t p_limit_angular) {
	set_param(PARAM_ANGULAR_LIMIT_LOWER, Math::deg2rad(p_limit_angular));
}

real_t SliderJoint3D::_get_lower_limit_angular() const {
	return Math::rad2deg(get_param(PARAM_ANGULAR_LIMIT_LOWER));
}

void SliderJoint3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &SliderJoint3D::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &SliderJoint3D::get_param);

	ClassDB::bind_method(D_METHOD("_set_upper_limit_angular", "upper_limit_angular"), &SliderJoint3D::_set_upper_limit_angular);
	ClassDB::bind_method(D_METHOD("_get_upper_limit_angular"), &SliderJoint3D::_get_upper_limit_angular);

	ClassDB::bind_method(D_METHOD("_set_lower_limit_angular", "lower_limit_angular"), &SliderJoint3D::_set_lower_limit_angular);
	ClassDB::bind_method(D_METHOD("_get_lower_limit_angular"), &SliderJoint3D::_get_lower_limit_angular);

	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit/upper_distance", PROPERTY_HINT_RANGE, "-1024,1024,0.01"), "set_param", "get_param", PARAM_LINEAR_LIMIT_UPPER);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit/lower_distance", PROPERTY_HINT_RANGE, "-1024,1024,0.01"), "set_param", "get_param", PARAM_LINEAR_LIMIT_LOWER);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit/restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_LIMIT_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit/damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_LIMIT_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_motion/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_MOTION_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_motion/restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_MOTION_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_motion/damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_MOTION_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_ortho/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_ORTHOGONAL_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_ortho/restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_ORTHOGONAL_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_ortho/damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_ORTHOGONAL_DAMPING);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_limit/upper_angle", PROPERTY_HINT_RANGE, "-180,180,0.1"), "_set_upper_limit_angular", "_get_upper_limit_angular");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_limit/lower_angle", PROPERTY_HINT_RANGE, "-180,180,0.1"), "_set_lower_limit_angular", "_get_lower_limit_angular");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit/restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_LIMIT_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit/damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_LIMIT_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_motion/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_MOTION_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_motion/restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_MOTION_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_motion/damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_MOTION_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_ortho/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_ORTHOGONAL_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_ortho/restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_ORTHOGONAL_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_ortho/damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_ORTHOGONAL_DAMPING);

	BIND_ENUM_CONSTANT(PARAM_LINEAR_LIMIT_UPPER);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_LIMIT_LOWER);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_LIMIT_SOFTNESS);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_LIMIT_RESTITUTION);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_LIMIT_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_MOTION_SOFTNESS);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_MOTION_RESTITUTION);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_MOTION_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_ORTHOGONAL_SOFTNESS);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_ORTHOGONAL_RESTITUTION);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_ORTHOGONAL_DAMPING);

	BIND_ENUM_CONSTANT(PARAM_ANGULAR_LIMIT_UPPER);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_LIMIT_LOWER);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_LIMIT_SOFTNESS);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_LIMIT_RESTITUTION);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_LIMIT_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_MOTION_SOFTNESS);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_MOTION_RESTITUTION);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_MOTION_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_ORTHOGONAL_SOFTNESS);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_ORTHOGONAL_RESTITUTION);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_ORTHOGONAL_DAMPING);

	BIND_ENUM_CONSTANT(PARAM_MAX);
}

void SliderJoint3D::set_param(Param p_param, real_t p_value) {
	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params[p_param] = p_value;
	if (is_configured()) {
		PhysicsServer3D::get_singleton()->slider_joint_set_param(get_joint(), PhysicsServer3D::SliderJointParam(p_param), p_value);
	}
	update_gizmo();
}

real_t SliderJoint3D::get_param(Param p_param) const {
	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params[p_param];
}

void SliderJoint3D::_configure_joint(RID p_joint, PhysicsBody3D *body_a, PhysicsBody3D *body_b) {
	Transform gt = get_global_transform();
	Transform ainv = body_a->get_global_transform().affine_inverse();

	Transform local_a = ainv * gt;
	local_a.orthonormalize();
	Transform local_b = gt;

	if (body_b) {
		Transform binv = body_b->get_global_transform().affine_inverse();
		local_b = binv * gt;
	}

	local_b.orthonormalize();

	PhysicsServer3D::get_singleton()->joint_make_slider(p_joint, body_a->get_rid(), local_a, body_b ? body_b->get_rid() : RID(), local_b);
	for (int i = 0; i < PARAM_MAX; i++) {
		PhysicsServer3D::get_singleton()->slider_joint_set_param(p_joint, PhysicsServer3D::SliderJointParam(i), params[i]);
	}
}

SliderJoint3D::SliderJoint3D() {
	params[PARAM_LINEAR_LIMIT_UPPER] = 1.0;
	params[PARAM_LINEAR_LIMIT_LOWER] = -1.0;
	params[PARAM_LINEAR_LIMIT_SOFTNESS] = 1.0;
	params[PARAM_LINEAR_LIMIT_RESTITUTION] = 0.7;
	params[PARAM_LINEAR_LIMIT_DAMPING] = 1.0;
	params[PARAM_LINEAR_MOTION_SOFTNESS] = 1.0;
	params[PARAM_LINEAR_MOTION_RESTITUTION] = 0.7;
	params[PARAM_LINEAR_MOTION_DAMPING] = 0; //1.0;
	params[PARAM_LINEAR_ORTHOGONAL_SOFTNESS] = 1.0;
	params[PARAM_LINEAR_ORTHOGONAL_RESTITUTION] = 0.7;
	params[PARAM_LINEAR_ORTHOGONAL_DAMPING] = 1.0;

	params[PARAM_ANGULAR_LIMIT_UPPER] = 0;
	params[PARAM_ANGULAR_LIMIT_LOWER] = 0;
	params[PARAM_ANGULAR_LIMIT_SOFTNESS] = 1.0;
	params[PARAM_ANGULAR_LIMIT_RESTITUTION] = 0.7;
	params[PARAM_ANGULAR_LIMIT_DAMPING] = 0; //1.0;
	params[PARAM_ANGULAR_MOTION_SOFTNESS] = 1.0;
	params[PARAM_ANGULAR_MOTION_RESTITUTION] = 0.7;
	params[PARAM_ANGULAR_MOTION_DAMPING] = 1.0;
	params[PARAM_ANGULAR_ORTHOGONAL_SOFTNESS] = 1.0;
	params[PARAM_ANGULAR_ORTHOGONAL_RESTITUTION] = 0.7;
	params[PARAM_ANGULAR_ORTHOGONAL_DAMPING] = 1.0;
}

//////////////////////////////////

void ConeTwistJoint3D::_set_swing_span(real_t p_limit_angular) {
	set_param(PARAM_SWING_SPAN, Math::deg2rad(p_limit_angular));
}

real_t ConeTwistJoint3D::_get_swing_span() const {
	return Math::rad2deg(get_param(PARAM_SWING_SPAN));
}

void ConeTwistJoint3D::_set_twist_span(real_t p_limit_angular) {
	set_param(PARAM_TWIST_SPAN, Math::deg2rad(p_limit_angular));
}

real_t ConeTwistJoint3D::_get_twist_span() const {
	return Math::rad2deg(get_param(PARAM_TWIST_SPAN));
}

void ConeTwistJoint3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &ConeTwistJoint3D::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &ConeTwistJoint3D::get_param);

	ClassDB::bind_method(D_METHOD("_set_swing_span", "swing_span"), &ConeTwistJoint3D::_set_swing_span);
	ClassDB::bind_method(D_METHOD("_get_swing_span"), &ConeTwistJoint3D::_get_swing_span);

	ClassDB::bind_method(D_METHOD("_set_twist_span", "twist_span"), &ConeTwistJoint3D::_set_twist_span);
	ClassDB::bind_method(D_METHOD("_get_twist_span"), &ConeTwistJoint3D::_get_twist_span);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "swing_span", PROPERTY_HINT_RANGE, "-180,180,0.1"), "_set_swing_span", "_get_swing_span");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "twist_span", PROPERTY_HINT_RANGE, "-40000,40000,0.1"), "_set_twist_span", "_get_twist_span");

	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "bias", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_BIAS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "relaxation", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_RELAXATION);

	BIND_ENUM_CONSTANT(PARAM_SWING_SPAN);
	BIND_ENUM_CONSTANT(PARAM_TWIST_SPAN);
	BIND_ENUM_CONSTANT(PARAM_BIAS);
	BIND_ENUM_CONSTANT(PARAM_SOFTNESS);
	BIND_ENUM_CONSTANT(PARAM_RELAXATION);
	BIND_ENUM_CONSTANT(PARAM_MAX);
}

void ConeTwistJoint3D::set_param(Param p_param, real_t p_value) {
	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params[p_param] = p_value;
	if (is_configured()) {
		PhysicsServer3D::get_singleton()->cone_twist_joint_set_param(get_joint(), PhysicsServer3D::ConeTwistJointParam(p_param), p_value);
	}

	update_gizmo();
}

real_t ConeTwistJoint3D::get_param(Param p_param) const {
	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params[p_param];
}

void ConeTwistJoint3D::_configure_joint(RID p_joint, PhysicsBody3D *body_a, PhysicsBody3D *body_b) {
	Transform gt = get_global_transform();
	//Vector3 cone_twistpos = gt.origin;
	//Vector3 cone_twistdir = gt.basis.get_axis(2);

	Transform ainv = body_a->get_global_transform().affine_inverse();

	Transform local_a = ainv * gt;
	local_a.orthonormalize();
	Transform local_b = gt;

	if (body_b) {
		Transform binv = body_b->get_global_transform().affine_inverse();
		local_b = binv * gt;
	}

	local_b.orthonormalize();

	PhysicsServer3D::get_singleton()->joint_make_cone_twist(p_joint, body_a->get_rid(), local_a, body_b ? body_b->get_rid() : RID(), local_b);
	for (int i = 0; i < PARAM_MAX; i++) {
		PhysicsServer3D::get_singleton()->cone_twist_joint_set_param(p_joint, PhysicsServer3D::ConeTwistJointParam(i), params[i]);
	}
}

ConeTwistJoint3D::ConeTwistJoint3D() {
	params[PARAM_SWING_SPAN] = Math_PI * 0.25;
	params[PARAM_TWIST_SPAN] = Math_PI;
	params[PARAM_BIAS] = 0.3;
	params[PARAM_SOFTNESS] = 0.8;
	params[PARAM_RELAXATION] = 1.0;
}

/////////////////////////////////////////////////////////////////////

void Generic6DOFJoint3D::_set_angular_hi_limit_x(real_t p_limit_angular) {
	set_param_x(PARAM_ANGULAR_UPPER_LIMIT, Math::deg2rad(p_limit_angular));
}

real_t Generic6DOFJoint3D::_get_angular_hi_limit_x() const {
	return Math::rad2deg(get_param_x(PARAM_ANGULAR_UPPER_LIMIT));
}

void Generic6DOFJoint3D::_set_angular_lo_limit_x(real_t p_limit_angular) {
	set_param_x(PARAM_ANGULAR_LOWER_LIMIT, Math::deg2rad(p_limit_angular));
}

real_t Generic6DOFJoint3D::_get_angular_lo_limit_x() const {
	return Math::rad2deg(get_param_x(PARAM_ANGULAR_LOWER_LIMIT));
}

void Generic6DOFJoint3D::_set_angular_hi_limit_y(real_t p_limit_angular) {
	set_param_y(PARAM_ANGULAR_UPPER_LIMIT, Math::deg2rad(p_limit_angular));
}

real_t Generic6DOFJoint3D::_get_angular_hi_limit_y() const {
	return Math::rad2deg(get_param_y(PARAM_ANGULAR_UPPER_LIMIT));
}

void Generic6DOFJoint3D::_set_angular_lo_limit_y(real_t p_limit_angular) {
	set_param_y(PARAM_ANGULAR_LOWER_LIMIT, Math::deg2rad(p_limit_angular));
}

real_t Generic6DOFJoint3D::_get_angular_lo_limit_y() const {
	return Math::rad2deg(get_param_y(PARAM_ANGULAR_LOWER_LIMIT));
}

void Generic6DOFJoint3D::_set_angular_hi_limit_z(real_t p_limit_angular) {
	set_param_z(PARAM_ANGULAR_UPPER_LIMIT, Math::deg2rad(p_limit_angular));
}

real_t Generic6DOFJoint3D::_get_angular_hi_limit_z() const {
	return Math::rad2deg(get_param_z(PARAM_ANGULAR_UPPER_LIMIT));
}

void Generic6DOFJoint3D::_set_angular_lo_limit_z(real_t p_limit_angular) {
	set_param_z(PARAM_ANGULAR_LOWER_LIMIT, Math::deg2rad(p_limit_angular));
}

real_t Generic6DOFJoint3D::_get_angular_lo_limit_z() const {
	return Math::rad2deg(get_param_z(PARAM_ANGULAR_LOWER_LIMIT));
}

void Generic6DOFJoint3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_set_angular_hi_limit_x", "angle"), &Generic6DOFJoint3D::_set_angular_hi_limit_x);
	ClassDB::bind_method(D_METHOD("_get_angular_hi_limit_x"), &Generic6DOFJoint3D::_get_angular_hi_limit_x);

	ClassDB::bind_method(D_METHOD("_set_angular_lo_limit_x", "angle"), &Generic6DOFJoint3D::_set_angular_lo_limit_x);
	ClassDB::bind_method(D_METHOD("_get_angular_lo_limit_x"), &Generic6DOFJoint3D::_get_angular_lo_limit_x);

	ClassDB::bind_method(D_METHOD("_set_angular_hi_limit_y", "angle"), &Generic6DOFJoint3D::_set_angular_hi_limit_y);
	ClassDB::bind_method(D_METHOD("_get_angular_hi_limit_y"), &Generic6DOFJoint3D::_get_angular_hi_limit_y);

	ClassDB::bind_method(D_METHOD("_set_angular_lo_limit_y", "angle"), &Generic6DOFJoint3D::_set_angular_lo_limit_y);
	ClassDB::bind_method(D_METHOD("_get_angular_lo_limit_y"), &Generic6DOFJoint3D::_get_angular_lo_limit_y);

	ClassDB::bind_method(D_METHOD("_set_angular_hi_limit_z", "angle"), &Generic6DOFJoint3D::_set_angular_hi_limit_z);
	ClassDB::bind_method(D_METHOD("_get_angular_hi_limit_z"), &Generic6DOFJoint3D::_get_angular_hi_limit_z);

	ClassDB::bind_method(D_METHOD("_set_angular_lo_limit_z", "angle"), &Generic6DOFJoint3D::_set_angular_lo_limit_z);
	ClassDB::bind_method(D_METHOD("_get_angular_lo_limit_z"), &Generic6DOFJoint3D::_get_angular_lo_limit_z);

	ClassDB::bind_method(D_METHOD("set_param_x", "param", "value"), &Generic6DOFJoint3D::set_param_x);
	ClassDB::bind_method(D_METHOD("get_param_x", "param"), &Generic6DOFJoint3D::get_param_x);

	ClassDB::bind_method(D_METHOD("set_param_y", "param", "value"), &Generic6DOFJoint3D::set_param_y);
	ClassDB::bind_method(D_METHOD("get_param_y", "param"), &Generic6DOFJoint3D::get_param_y);

	ClassDB::bind_method(D_METHOD("set_param_z", "param", "value"), &Generic6DOFJoint3D::set_param_z);
	ClassDB::bind_method(D_METHOD("get_param_z", "param"), &Generic6DOFJoint3D::get_param_z);

	ClassDB::bind_method(D_METHOD("set_flag_x", "flag", "value"), &Generic6DOFJoint3D::set_flag_x);
	ClassDB::bind_method(D_METHOD("get_flag_x", "flag"), &Generic6DOFJoint3D::get_flag_x);

	ClassDB::bind_method(D_METHOD("set_flag_y", "flag", "value"), &Generic6DOFJoint3D::set_flag_y);
	ClassDB::bind_method(D_METHOD("get_flag_y", "flag"), &Generic6DOFJoint3D::get_flag_y);

	ClassDB::bind_method(D_METHOD("set_flag_z", "flag", "value"), &Generic6DOFJoint3D::set_flag_z);
	ClassDB::bind_method(D_METHOD("get_flag_z", "flag"), &Generic6DOFJoint3D::get_flag_z);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "linear_limit_x/enabled"), "set_flag_x", "get_flag_x", FLAG_ENABLE_LINEAR_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_x/upper_distance"), "set_param_x", "get_param_x", PARAM_LINEAR_UPPER_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_x/lower_distance"), "set_param_x", "get_param_x", PARAM_LINEAR_LOWER_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_x/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_x", "get_param_x", PARAM_LINEAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_x/restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_x", "get_param_x", PARAM_LINEAR_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_x/damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_x", "get_param_x", PARAM_LINEAR_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "linear_motor_x/enabled"), "set_flag_x", "get_flag_x", FLAG_ENABLE_LINEAR_MOTOR);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_motor_x/target_velocity"), "set_param_x", "get_param_x", PARAM_LINEAR_MOTOR_TARGET_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_motor_x/force_limit"), "set_param_x", "get_param_x", PARAM_LINEAR_MOTOR_FORCE_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "linear_spring_x/enabled"), "set_flag_x", "get_flag_x", FLAG_ENABLE_LINEAR_SPRING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_spring_x/stiffness"), "set_param_x", "get_param_x", PARAM_LINEAR_SPRING_STIFFNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_spring_x/damping"), "set_param_x", "get_param_x", PARAM_LINEAR_SPRING_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_spring_x/equilibrium_point"), "set_param_x", "get_param_x", PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_limit_x/enabled"), "set_flag_x", "get_flag_x", FLAG_ENABLE_ANGULAR_LIMIT);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_limit_x/upper_angle", PROPERTY_HINT_RANGE, "-180,180,0.01"), "_set_angular_hi_limit_x", "_get_angular_hi_limit_x");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_limit_x/lower_angle", PROPERTY_HINT_RANGE, "-180,180,0.01"), "_set_angular_lo_limit_x", "_get_angular_lo_limit_x");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_x/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_x", "get_param_x", PARAM_ANGULAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_x/restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_x", "get_param_x", PARAM_ANGULAR_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_x/damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_x", "get_param_x", PARAM_ANGULAR_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_x/force_limit"), "set_param_x", "get_param_x", PARAM_ANGULAR_FORCE_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_x/erp"), "set_param_x", "get_param_x", PARAM_ANGULAR_ERP);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_motor_x/enabled"), "set_flag_x", "get_flag_x", FLAG_ENABLE_MOTOR);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_motor_x/target_velocity"), "set_param_x", "get_param_x", PARAM_ANGULAR_MOTOR_TARGET_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_motor_x/force_limit"), "set_param_x", "get_param_x", PARAM_ANGULAR_MOTOR_FORCE_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_spring_x/enabled"), "set_flag_x", "get_flag_x", FLAG_ENABLE_ANGULAR_SPRING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_spring_x/stiffness"), "set_param_x", "get_param_x", PARAM_ANGULAR_SPRING_STIFFNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_spring_x/damping"), "set_param_x", "get_param_x", PARAM_ANGULAR_SPRING_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_spring_x/equilibrium_point"), "set_param_x", "get_param_x", PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "linear_limit_y/enabled"), "set_flag_y", "get_flag_y", FLAG_ENABLE_LINEAR_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_y/upper_distance"), "set_param_y", "get_param_y", PARAM_LINEAR_UPPER_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_y/lower_distance"), "set_param_y", "get_param_y", PARAM_LINEAR_LOWER_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_y/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_y", "get_param_y", PARAM_LINEAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_y/restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_y", "get_param_y", PARAM_LINEAR_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_y/damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_y", "get_param_y", PARAM_LINEAR_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "linear_motor_y/enabled"), "set_flag_y", "get_flag_y", FLAG_ENABLE_LINEAR_MOTOR);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_motor_y/target_velocity"), "set_param_y", "get_param_y", PARAM_LINEAR_MOTOR_TARGET_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_motor_y/force_limit"), "set_param_y", "get_param_y", PARAM_LINEAR_MOTOR_FORCE_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "linear_spring_y/enabled"), "set_flag_y", "get_flag_y", FLAG_ENABLE_LINEAR_SPRING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_spring_y/stiffness"), "set_param_y", "get_param_y", PARAM_LINEAR_SPRING_STIFFNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_spring_y/damping"), "set_param_y", "get_param_y", PARAM_LINEAR_SPRING_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_spring_y/equilibrium_point"), "set_param_y", "get_param_y", PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_limit_y/enabled"), "set_flag_y", "get_flag_y", FLAG_ENABLE_ANGULAR_LIMIT);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_limit_y/upper_angle", PROPERTY_HINT_RANGE, "-180,180,0.01"), "_set_angular_hi_limit_y", "_get_angular_hi_limit_y");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_limit_y/lower_angle", PROPERTY_HINT_RANGE, "-180,180,0.01"), "_set_angular_lo_limit_y", "_get_angular_lo_limit_y");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_y/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_y", "get_param_y", PARAM_ANGULAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_y/restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_y", "get_param_y", PARAM_ANGULAR_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_y/damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_y", "get_param_y", PARAM_ANGULAR_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_y/force_limit"), "set_param_y", "get_param_y", PARAM_ANGULAR_FORCE_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_y/erp"), "set_param_y", "get_param_y", PARAM_ANGULAR_ERP);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_motor_y/enabled"), "set_flag_y", "get_flag_y", FLAG_ENABLE_MOTOR);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_motor_y/target_velocity"), "set_param_y", "get_param_y", PARAM_ANGULAR_MOTOR_TARGET_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_motor_y/force_limit"), "set_param_y", "get_param_y", PARAM_ANGULAR_MOTOR_FORCE_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_spring_y/enabled"), "set_flag_y", "get_flag_y", FLAG_ENABLE_ANGULAR_SPRING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_spring_y/stiffness"), "set_param_y", "get_param_y", PARAM_ANGULAR_SPRING_STIFFNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_spring_y/damping"), "set_param_y", "get_param_y", PARAM_ANGULAR_SPRING_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_spring_y/equilibrium_point"), "set_param_y", "get_param_y", PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "linear_limit_z/enabled"), "set_flag_z", "get_flag_z", FLAG_ENABLE_LINEAR_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_z/upper_distance"), "set_param_z", "get_param_z", PARAM_LINEAR_UPPER_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_z/lower_distance"), "set_param_z", "get_param_z", PARAM_LINEAR_LOWER_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_z/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_z", "get_param_z", PARAM_LINEAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_z/restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_z", "get_param_z", PARAM_LINEAR_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_limit_z/damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_z", "get_param_z", PARAM_LINEAR_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "linear_motor_z/enabled"), "set_flag_z", "get_flag_z", FLAG_ENABLE_LINEAR_MOTOR);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_motor_z/target_velocity"), "set_param_z", "get_param_z", PARAM_LINEAR_MOTOR_TARGET_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_motor_z/force_limit"), "set_param_z", "get_param_z", PARAM_LINEAR_MOTOR_FORCE_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "linear_spring_z/enabled"), "set_flag_z", "get_flag_z", FLAG_ENABLE_LINEAR_SPRING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_spring_z/stiffness"), "set_param_z", "get_param_z", PARAM_LINEAR_SPRING_STIFFNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_spring_z/damping"), "set_param_z", "get_param_z", PARAM_LINEAR_SPRING_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "linear_spring_z/equilibrium_point"), "set_param_z", "get_param_z", PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_limit_z/enabled"), "set_flag_z", "get_flag_z", FLAG_ENABLE_ANGULAR_LIMIT);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_limit_z/upper_angle", PROPERTY_HINT_RANGE, "-180,180,0.01"), "_set_angular_hi_limit_z", "_get_angular_hi_limit_z");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_limit_z/lower_angle", PROPERTY_HINT_RANGE, "-180,180,0.01"), "_set_angular_lo_limit_z", "_get_angular_lo_limit_z");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_z/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_z", "get_param_z", PARAM_ANGULAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_z/restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_z", "get_param_z", PARAM_ANGULAR_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_z/damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_z", "get_param_z", PARAM_ANGULAR_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_z/force_limit"), "set_param_z", "get_param_z", PARAM_ANGULAR_FORCE_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_limit_z/erp"), "set_param_z", "get_param_z", PARAM_ANGULAR_ERP);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_motor_z/enabled"), "set_flag_z", "get_flag_z", FLAG_ENABLE_MOTOR);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_motor_z/target_velocity"), "set_param_z", "get_param_z", PARAM_ANGULAR_MOTOR_TARGET_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_motor_z/force_limit"), "set_param_z", "get_param_z", PARAM_ANGULAR_MOTOR_FORCE_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_spring_z/enabled"), "set_flag_z", "get_flag_z", FLAG_ENABLE_ANGULAR_SPRING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_spring_z/stiffness"), "set_param_z", "get_param_z", PARAM_ANGULAR_SPRING_STIFFNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_spring_z/damping"), "set_param_z", "get_param_z", PARAM_ANGULAR_SPRING_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "angular_spring_z/equilibrium_point"), "set_param_z", "get_param_z", PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT);

	BIND_ENUM_CONSTANT(PARAM_LINEAR_LOWER_LIMIT);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_UPPER_LIMIT);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_LIMIT_SOFTNESS);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_RESTITUTION);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_MOTOR_TARGET_VELOCITY);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_MOTOR_FORCE_LIMIT);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_SPRING_STIFFNESS);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_SPRING_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_LOWER_LIMIT);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_UPPER_LIMIT);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_LIMIT_SOFTNESS);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_RESTITUTION);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_FORCE_LIMIT);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_ERP);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_MOTOR_FORCE_LIMIT);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_SPRING_STIFFNESS);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_SPRING_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT);
	BIND_ENUM_CONSTANT(PARAM_MAX);

	BIND_ENUM_CONSTANT(FLAG_ENABLE_LINEAR_LIMIT);
	BIND_ENUM_CONSTANT(FLAG_ENABLE_ANGULAR_LIMIT);
	BIND_ENUM_CONSTANT(FLAG_ENABLE_LINEAR_SPRING);
	BIND_ENUM_CONSTANT(FLAG_ENABLE_ANGULAR_SPRING);
	BIND_ENUM_CONSTANT(FLAG_ENABLE_MOTOR);
	BIND_ENUM_CONSTANT(FLAG_ENABLE_LINEAR_MOTOR);
	BIND_ENUM_CONSTANT(FLAG_MAX);
}

void Generic6DOFJoint3D::set_param_x(Param p_param, real_t p_value) {
	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params_x[p_param] = p_value;
	if (is_configured()) {
		PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(get_joint(), Vector3::AXIS_X, PhysicsServer3D::G6DOFJointAxisParam(p_param), p_value);
	}

	update_gizmo();
}

real_t Generic6DOFJoint3D::get_param_x(Param p_param) const {
	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params_x[p_param];
}

void Generic6DOFJoint3D::set_param_y(Param p_param, real_t p_value) {
	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params_y[p_param] = p_value;
	if (is_configured()) {
		PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(get_joint(), Vector3::AXIS_Y, PhysicsServer3D::G6DOFJointAxisParam(p_param), p_value);
	}
	update_gizmo();
}

real_t Generic6DOFJoint3D::get_param_y(Param p_param) const {
	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params_y[p_param];
}

void Generic6DOFJoint3D::set_param_z(Param p_param, real_t p_value) {
	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params_z[p_param] = p_value;
	if (is_configured()) {
		PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(get_joint(), Vector3::AXIS_Z, PhysicsServer3D::G6DOFJointAxisParam(p_param), p_value);
	}
	update_gizmo();
}

real_t Generic6DOFJoint3D::get_param_z(Param p_param) const {
	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params_z[p_param];
}

void Generic6DOFJoint3D::set_flag_x(Flag p_flag, bool p_enabled) {
	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	flags_x[p_flag] = p_enabled;
	if (is_configured()) {
		PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(get_joint(), Vector3::AXIS_X, PhysicsServer3D::G6DOFJointAxisFlag(p_flag), p_enabled);
	}
	update_gizmo();
}

bool Generic6DOFJoint3D::get_flag_x(Flag p_flag) const {
	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	return flags_x[p_flag];
}

void Generic6DOFJoint3D::set_flag_y(Flag p_flag, bool p_enabled) {
	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	flags_y[p_flag] = p_enabled;
	if (is_configured()) {
		PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(get_joint(), Vector3::AXIS_Y, PhysicsServer3D::G6DOFJointAxisFlag(p_flag), p_enabled);
	}
	update_gizmo();
}

bool Generic6DOFJoint3D::get_flag_y(Flag p_flag) const {
	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	return flags_y[p_flag];
}

void Generic6DOFJoint3D::set_flag_z(Flag p_flag, bool p_enabled) {
	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	flags_z[p_flag] = p_enabled;
	if (is_configured()) {
		PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(get_joint(), Vector3::AXIS_Z, PhysicsServer3D::G6DOFJointAxisFlag(p_flag), p_enabled);
	}
	update_gizmo();
}

bool Generic6DOFJoint3D::get_flag_z(Flag p_flag) const {
	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	return flags_z[p_flag];
}

void Generic6DOFJoint3D::_configure_joint(RID p_joint, PhysicsBody3D *body_a, PhysicsBody3D *body_b) {
	Transform gt = get_global_transform();
	//Vector3 cone_twistpos = gt.origin;
	//Vector3 cone_twistdir = gt.basis.get_axis(2);

	Transform ainv = body_a->get_global_transform().affine_inverse();

	Transform local_a = ainv * gt;
	local_a.orthonormalize();
	Transform local_b = gt;

	if (body_b) {
		Transform binv = body_b->get_global_transform().affine_inverse();
		local_b = binv * gt;
	}

	local_b.orthonormalize();

	PhysicsServer3D::get_singleton()->joint_make_generic_6dof(p_joint, body_a->get_rid(), local_a, body_b ? body_b->get_rid() : RID(), local_b);
	for (int i = 0; i < PARAM_MAX; i++) {
		PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(p_joint, Vector3::AXIS_X, PhysicsServer3D::G6DOFJointAxisParam(i), params_x[i]);
		PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(p_joint, Vector3::AXIS_Y, PhysicsServer3D::G6DOFJointAxisParam(i), params_y[i]);
		PhysicsServer3D::get_singleton()->generic_6dof_joint_set_param(p_joint, Vector3::AXIS_Z, PhysicsServer3D::G6DOFJointAxisParam(i), params_z[i]);
	}
	for (int i = 0; i < FLAG_MAX; i++) {
		PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(p_joint, Vector3::AXIS_X, PhysicsServer3D::G6DOFJointAxisFlag(i), flags_x[i]);
		PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(p_joint, Vector3::AXIS_Y, PhysicsServer3D::G6DOFJointAxisFlag(i), flags_y[i]);
		PhysicsServer3D::get_singleton()->generic_6dof_joint_set_flag(p_joint, Vector3::AXIS_Z, PhysicsServer3D::G6DOFJointAxisFlag(i), flags_z[i]);
	}
}

Generic6DOFJoint3D::Generic6DOFJoint3D() {
	set_param_x(PARAM_LINEAR_LOWER_LIMIT, 0);
	set_param_x(PARAM_LINEAR_UPPER_LIMIT, 0);
	set_param_x(PARAM_LINEAR_LIMIT_SOFTNESS, 0.7);
	set_param_x(PARAM_LINEAR_RESTITUTION, 0.5);
	set_param_x(PARAM_LINEAR_DAMPING, 1.0);
	set_param_x(PARAM_LINEAR_MOTOR_TARGET_VELOCITY, 0);
	set_param_x(PARAM_LINEAR_MOTOR_FORCE_LIMIT, 0);
	set_param_x(PARAM_LINEAR_SPRING_STIFFNESS, 0.01);
	set_param_x(PARAM_LINEAR_SPRING_DAMPING, 0.01);
	set_param_x(PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT, 0.0);
	set_param_x(PARAM_ANGULAR_LOWER_LIMIT, 0);
	set_param_x(PARAM_ANGULAR_UPPER_LIMIT, 0);
	set_param_x(PARAM_ANGULAR_LIMIT_SOFTNESS, 0.5f);
	set_param_x(PARAM_ANGULAR_DAMPING, 1.0f);
	set_param_x(PARAM_ANGULAR_RESTITUTION, 0);
	set_param_x(PARAM_ANGULAR_FORCE_LIMIT, 0);
	set_param_x(PARAM_ANGULAR_ERP, 0.5);
	set_param_x(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, 0);
	set_param_x(PARAM_ANGULAR_MOTOR_FORCE_LIMIT, 300);
	set_param_x(PARAM_ANGULAR_SPRING_STIFFNESS, 0);
	set_param_x(PARAM_ANGULAR_SPRING_DAMPING, 0);
	set_param_x(PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT, 0);

	set_flag_x(FLAG_ENABLE_ANGULAR_LIMIT, true);
	set_flag_x(FLAG_ENABLE_LINEAR_LIMIT, true);
	set_flag_x(FLAG_ENABLE_ANGULAR_SPRING, false);
	set_flag_x(FLAG_ENABLE_LINEAR_SPRING, false);
	set_flag_x(FLAG_ENABLE_MOTOR, false);
	set_flag_x(FLAG_ENABLE_LINEAR_MOTOR, false);

	set_param_y(PARAM_LINEAR_LOWER_LIMIT, 0);
	set_param_y(PARAM_LINEAR_UPPER_LIMIT, 0);
	set_param_y(PARAM_LINEAR_LIMIT_SOFTNESS, 0.7);
	set_param_y(PARAM_LINEAR_RESTITUTION, 0.5);
	set_param_y(PARAM_LINEAR_DAMPING, 1.0);
	set_param_y(PARAM_LINEAR_MOTOR_TARGET_VELOCITY, 0);
	set_param_y(PARAM_LINEAR_MOTOR_FORCE_LIMIT, 0);
	set_param_y(PARAM_LINEAR_SPRING_STIFFNESS, 0.01);
	set_param_y(PARAM_LINEAR_SPRING_DAMPING, 0.01);
	set_param_y(PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT, 0.0);
	set_param_y(PARAM_ANGULAR_LOWER_LIMIT, 0);
	set_param_y(PARAM_ANGULAR_UPPER_LIMIT, 0);
	set_param_y(PARAM_ANGULAR_LIMIT_SOFTNESS, 0.5f);
	set_param_y(PARAM_ANGULAR_DAMPING, 1.0f);
	set_param_y(PARAM_ANGULAR_RESTITUTION, 0);
	set_param_y(PARAM_ANGULAR_FORCE_LIMIT, 0);
	set_param_y(PARAM_ANGULAR_ERP, 0.5);
	set_param_y(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, 0);
	set_param_y(PARAM_ANGULAR_MOTOR_FORCE_LIMIT, 300);
	set_param_y(PARAM_ANGULAR_SPRING_STIFFNESS, 0);
	set_param_y(PARAM_ANGULAR_SPRING_DAMPING, 0);
	set_param_y(PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT, 0);

	set_flag_y(FLAG_ENABLE_ANGULAR_LIMIT, true);
	set_flag_y(FLAG_ENABLE_LINEAR_LIMIT, true);
	set_flag_y(FLAG_ENABLE_ANGULAR_SPRING, false);
	set_flag_y(FLAG_ENABLE_LINEAR_SPRING, false);
	set_flag_y(FLAG_ENABLE_MOTOR, false);
	set_flag_y(FLAG_ENABLE_LINEAR_MOTOR, false);

	set_param_z(PARAM_LINEAR_LOWER_LIMIT, 0);
	set_param_z(PARAM_LINEAR_UPPER_LIMIT, 0);
	set_param_z(PARAM_LINEAR_LIMIT_SOFTNESS, 0.7);
	set_param_z(PARAM_LINEAR_RESTITUTION, 0.5);
	set_param_z(PARAM_LINEAR_DAMPING, 1.0);
	set_param_z(PARAM_LINEAR_MOTOR_TARGET_VELOCITY, 0);
	set_param_z(PARAM_LINEAR_MOTOR_FORCE_LIMIT, 0);
	set_param_z(PARAM_LINEAR_SPRING_STIFFNESS, 0.01);
	set_param_z(PARAM_LINEAR_SPRING_DAMPING, 0.01);
	set_param_z(PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT, 0.0);
	set_param_z(PARAM_ANGULAR_LOWER_LIMIT, 0);
	set_param_z(PARAM_ANGULAR_UPPER_LIMIT, 0);
	set_param_z(PARAM_ANGULAR_LIMIT_SOFTNESS, 0.5f);
	set_param_z(PARAM_ANGULAR_DAMPING, 1.0f);
	set_param_z(PARAM_ANGULAR_RESTITUTION, 0);
	set_param_z(PARAM_ANGULAR_FORCE_LIMIT, 0);
	set_param_z(PARAM_ANGULAR_ERP, 0.5);
	set_param_z(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, 0);
	set_param_z(PARAM_ANGULAR_MOTOR_FORCE_LIMIT, 300);
	set_param_z(PARAM_ANGULAR_SPRING_STIFFNESS, 0);
	set_param_z(PARAM_ANGULAR_SPRING_DAMPING, 0);
	set_param_z(PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT, 0);

	set_flag_z(FLAG_ENABLE_ANGULAR_LIMIT, true);
	set_flag_z(FLAG_ENABLE_LINEAR_LIMIT, true);
	set_flag_z(FLAG_ENABLE_ANGULAR_SPRING, false);
	set_flag_z(FLAG_ENABLE_LINEAR_SPRING, false);
	set_flag_z(FLAG_ENABLE_MOTOR, false);
	set_flag_z(FLAG_ENABLE_LINEAR_MOTOR, false);
}
