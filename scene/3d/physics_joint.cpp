/*************************************************************************/
/*  physics_joint.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "physics_joint.h"

void Joint::_update_joint(bool p_only_free) {

	if (joint.is_valid()) {
		if (ba.is_valid() && bb.is_valid()) {

			if (exclude_from_collision)
				PhysicsServer::get_singleton()->body_add_collision_exception(ba, bb);
			else
				PhysicsServer::get_singleton()->body_remove_collision_exception(ba, bb);
		}

		PhysicsServer::get_singleton()->free(joint);
		joint = RID();
		ba = RID();
		bb = RID();
	}

	if (p_only_free || !is_inside_tree())
		return;

	Node *node_a = has_node(get_node_a()) ? get_node(get_node_a()) : (Node *)NULL;
	Node *node_b = has_node(get_node_b()) ? get_node(get_node_b()) : (Node *)NULL;

	if (!node_a && !node_b)
		return;

	PhysicsBody *body_a = node_a ? node_a->cast_to<PhysicsBody>() : (PhysicsBody *)NULL;
	PhysicsBody *body_b = node_b ? node_b->cast_to<PhysicsBody>() : (PhysicsBody *)NULL;

	if (!body_a && !body_b)
		return;

	if (!body_a) {
		SWAP(body_a, body_b);
	} else if (body_b) {
		//add a collision exception between both
		PhysicsServer::get_singleton()->body_add_collision_exception(body_a->get_rid(), body_b->get_rid());
	}

	joint = _configure_joint(body_a, body_b);

	if (joint.is_valid())
		PhysicsServer::get_singleton()->joint_set_solver_priority(joint, solver_priority);

	if (body_b && joint.is_valid()) {

		ba = body_a->get_rid();
		bb = body_b->get_rid();
		PhysicsServer::get_singleton()->body_add_collision_exception(body_a->get_rid(), body_b->get_rid());
	}
}

void Joint::set_node_a(const NodePath &p_node_a) {

	if (a == p_node_a)
		return;

	a = p_node_a;
	_update_joint();
}

NodePath Joint::get_node_a() const {

	return a;
}

void Joint::set_node_b(const NodePath &p_node_b) {

	if (b == p_node_b)
		return;
	b = p_node_b;
	_update_joint();
}
NodePath Joint::get_node_b() const {

	return b;
}

void Joint::set_solver_priority(int p_priority) {

	solver_priority = p_priority;
	if (joint.is_valid())
		PhysicsServer::get_singleton()->joint_set_solver_priority(joint, solver_priority);
}

int Joint::get_solver_priority() const {

	return solver_priority;
}

void Joint::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_READY: {
			_update_joint();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (joint.is_valid()) {
				_update_joint(true);
				//PhysicsServer::get_singleton()->free(joint);
				joint = RID();
			}
		} break;
	}
}

void Joint::set_exclude_nodes_from_collision(bool p_enable) {

	if (exclude_from_collision == p_enable)
		return;
	exclude_from_collision = p_enable;
	_update_joint();
}

bool Joint::get_exclude_nodes_from_collision() const {

	return exclude_from_collision;
}

void Joint::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_node_a", "node"), &Joint::set_node_a);
	ClassDB::bind_method(D_METHOD("get_node_a"), &Joint::get_node_a);

	ClassDB::bind_method(D_METHOD("set_node_b", "node"), &Joint::set_node_b);
	ClassDB::bind_method(D_METHOD("get_node_b"), &Joint::get_node_b);

	ClassDB::bind_method(D_METHOD("set_solver_priority", "priority"), &Joint::set_solver_priority);
	ClassDB::bind_method(D_METHOD("get_solver_priority"), &Joint::get_solver_priority);

	ClassDB::bind_method(D_METHOD("set_exclude_nodes_from_collision", "enable"), &Joint::set_exclude_nodes_from_collision);
	ClassDB::bind_method(D_METHOD("get_exclude_nodes_from_collision"), &Joint::get_exclude_nodes_from_collision);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "nodes/node_a"), "set_node_a", "get_node_a");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "nodes/node_b"), "set_node_b", "get_node_b");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "solver/priority", PROPERTY_HINT_RANGE, "1,8,1"), "set_solver_priority", "get_solver_priority");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collision/exclude_nodes"), "set_exclude_nodes_from_collision", "get_exclude_nodes_from_collision");
}

Joint::Joint() {

	exclude_from_collision = true;
	solver_priority = 1;
	set_notify_transform(true);
}

///////////////////////////////////

void PinJoint::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &PinJoint::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &PinJoint::get_param);

	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "params/bias", PROPERTY_HINT_RANGE, "0.01,0.99,0.01"), "set_param", "get_param", PARAM_BIAS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "params/damping", PROPERTY_HINT_RANGE, "0.01,8.0,0.01"), "set_param", "get_param", PARAM_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "params/impulse_clamp", PROPERTY_HINT_RANGE, "0.0,64.0,0.01"), "set_param", "get_param", PARAM_IMPULSE_CLAMP);

	BIND_CONSTANT(PARAM_BIAS);
	BIND_CONSTANT(PARAM_DAMPING);
	BIND_CONSTANT(PARAM_IMPULSE_CLAMP);
}

void PinJoint::set_param(Param p_param, float p_value) {

	ERR_FAIL_INDEX(p_param, 3);
	params[p_param] = p_value;
	if (get_joint().is_valid())
		PhysicsServer::get_singleton()->pin_joint_set_param(get_joint(), PhysicsServer::PinJointParam(p_param), p_value);
}
float PinJoint::get_param(Param p_param) const {

	ERR_FAIL_INDEX_V(p_param, 3, 0);
	return params[p_param];
}

RID PinJoint::_configure_joint(PhysicsBody *body_a, PhysicsBody *body_b) {

	Vector3 pinpos = get_global_transform().origin;
	Vector3 local_a = body_a->get_global_transform().affine_inverse().xform(pinpos);
	Vector3 local_b;

	if (body_b)
		local_b = body_b->get_global_transform().affine_inverse().xform(pinpos);
	else
		local_b = pinpos;

	RID j = PhysicsServer::get_singleton()->joint_create_pin(body_a->get_rid(), local_a, body_b ? body_b->get_rid() : RID(), local_b);
	for (int i = 0; i < 3; i++) {
		PhysicsServer::get_singleton()->pin_joint_set_param(j, PhysicsServer::PinJointParam(i), params[i]);
	}
	return j;
}

PinJoint::PinJoint() {

	params[PARAM_BIAS] = 0.3;
	params[PARAM_DAMPING] = 1;
	params[PARAM_IMPULSE_CLAMP] = 0;
}

/////////////////////////////////////////////////

///////////////////////////////////

void HingeJoint::_set_upper_limit(float p_limit) {

	set_param(PARAM_LIMIT_UPPER, Math::deg2rad(p_limit));
}

float HingeJoint::_get_upper_limit() const {

	return Math::rad2deg(get_param(PARAM_LIMIT_UPPER));
}

void HingeJoint::_set_lower_limit(float p_limit) {

	set_param(PARAM_LIMIT_LOWER, Math::deg2rad(p_limit));
}

float HingeJoint::_get_lower_limit() const {

	return Math::rad2deg(get_param(PARAM_LIMIT_LOWER));
}

void HingeJoint::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &HingeJoint::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &HingeJoint::get_param);

	ClassDB::bind_method(D_METHOD("set_flag", "flag", "enabled"), &HingeJoint::set_flag);
	ClassDB::bind_method(D_METHOD("get_flag", "flag"), &HingeJoint::get_flag);

	ClassDB::bind_method(D_METHOD("_set_upper_limit", "upper_limit"), &HingeJoint::_set_upper_limit);
	ClassDB::bind_method(D_METHOD("_get_upper_limit"), &HingeJoint::_get_upper_limit);

	ClassDB::bind_method(D_METHOD("_set_lower_limit", "lower_limit"), &HingeJoint::_set_lower_limit);
	ClassDB::bind_method(D_METHOD("_get_lower_limit"), &HingeJoint::_get_lower_limit);

	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "params/bias", PROPERTY_HINT_RANGE, "0.01,0.99,0.01"), "set_param", "get_param", PARAM_BIAS);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_limit/enable"), "set_flag", "get_flag", FLAG_USE_LIMIT);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_limit/upper", PROPERTY_HINT_RANGE, "-180,180,0.1"), "_set_upper_limit", "_get_upper_limit");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_limit/lower", PROPERTY_HINT_RANGE, "-180,180,0.1"), "_set_lower_limit", "_get_lower_limit");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit/bias", PROPERTY_HINT_RANGE, "0.01,0.99,0.01"), "set_param", "get_param", PARAM_LIMIT_BIAS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param", "get_param", PARAM_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit/relaxation", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param", "get_param", PARAM_LIMIT_RELAXATION);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "motor/enable"), "set_flag", "get_flag", FLAG_ENABLE_MOTOR);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "motor/target_velocity", PROPERTY_HINT_RANGE, "0.01,4096,0.01"), "set_param", "get_param", PARAM_MOTOR_TARGET_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "motor/max_impulse", PROPERTY_HINT_RANGE, "0.01,1024,0.01"), "set_param", "get_param", PARAM_MOTOR_MAX_IMPULSE);

	BIND_CONSTANT(PARAM_BIAS);
	BIND_CONSTANT(PARAM_LIMIT_UPPER);
	BIND_CONSTANT(PARAM_LIMIT_LOWER);
	BIND_CONSTANT(PARAM_LIMIT_BIAS);
	BIND_CONSTANT(PARAM_LIMIT_SOFTNESS);
	BIND_CONSTANT(PARAM_LIMIT_RELAXATION);
	BIND_CONSTANT(PARAM_MOTOR_TARGET_VELOCITY);
	BIND_CONSTANT(PARAM_MOTOR_MAX_IMPULSE);
	BIND_CONSTANT(PARAM_MAX);

	BIND_CONSTANT(FLAG_USE_LIMIT);
	BIND_CONSTANT(FLAG_ENABLE_MOTOR);
	BIND_CONSTANT(FLAG_MAX);
}

void HingeJoint::set_param(Param p_param, float p_value) {

	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params[p_param] = p_value;
	if (get_joint().is_valid())
		PhysicsServer::get_singleton()->hinge_joint_set_param(get_joint(), PhysicsServer::HingeJointParam(p_param), p_value);

	update_gizmo();
}
float HingeJoint::get_param(Param p_param) const {

	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params[p_param];
}

void HingeJoint::set_flag(Flag p_flag, bool p_value) {

	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	flags[p_flag] = p_value;
	if (get_joint().is_valid())
		PhysicsServer::get_singleton()->hinge_joint_set_flag(get_joint(), PhysicsServer::HingeJointFlag(p_flag), p_value);

	update_gizmo();
}
bool HingeJoint::get_flag(Flag p_flag) const {

	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	return flags[p_flag];
}

RID HingeJoint::_configure_joint(PhysicsBody *body_a, PhysicsBody *body_b) {

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

	RID j = PhysicsServer::get_singleton()->joint_create_hinge(body_a->get_rid(), local_a, body_b ? body_b->get_rid() : RID(), local_b);
	for (int i = 0; i < PARAM_MAX; i++) {
		PhysicsServer::get_singleton()->hinge_joint_set_param(j, PhysicsServer::HingeJointParam(i), params[i]);
	}
	for (int i = 0; i < FLAG_MAX; i++) {
		set_flag(Flag(i), flags[i]);
		PhysicsServer::get_singleton()->hinge_joint_set_flag(j, PhysicsServer::HingeJointFlag(i), flags[i]);
	}
	return j;
}

HingeJoint::HingeJoint() {

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

void SliderJoint::_set_upper_limit_angular(float p_limit_angular) {

	set_param(PARAM_ANGULAR_LIMIT_UPPER, Math::deg2rad(p_limit_angular));
}

float SliderJoint::_get_upper_limit_angular() const {

	return Math::rad2deg(get_param(PARAM_ANGULAR_LIMIT_UPPER));
}

void SliderJoint::_set_lower_limit_angular(float p_limit_angular) {

	set_param(PARAM_ANGULAR_LIMIT_LOWER, Math::deg2rad(p_limit_angular));
}

float SliderJoint::_get_lower_limit_angular() const {

	return Math::rad2deg(get_param(PARAM_ANGULAR_LIMIT_LOWER));
}

void SliderJoint::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &SliderJoint::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &SliderJoint::get_param);

	ClassDB::bind_method(D_METHOD("_set_upper_limit_angular", "upper_limit_angular"), &SliderJoint::_set_upper_limit_angular);
	ClassDB::bind_method(D_METHOD("_get_upper_limit_angular"), &SliderJoint::_get_upper_limit_angular);

	ClassDB::bind_method(D_METHOD("_set_lower_limit_angular", "lower_limit_angular"), &SliderJoint::_set_lower_limit_angular);
	ClassDB::bind_method(D_METHOD("_get_lower_limit_angular"), &SliderJoint::_get_lower_limit_angular);

	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit/upper_distance", PROPERTY_HINT_RANGE, "-1024,1024,0.01"), "set_param", "get_param", PARAM_LINEAR_LIMIT_UPPER);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit/lower_distance", PROPERTY_HINT_RANGE, "-1024,1024,0.01"), "set_param", "get_param", PARAM_LINEAR_LIMIT_LOWER);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit/restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_LIMIT_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit/damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_LIMIT_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_motion/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_MOTION_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_motion/restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_MOTION_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_motion/damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_MOTION_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_ortho/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_ORTHOGONAL_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_ortho/restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_ORTHOGONAL_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_ortho/damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"), "set_param", "get_param", PARAM_LINEAR_ORTHOGONAL_DAMPING);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_limit/upper_angle", PROPERTY_HINT_RANGE, "-180,180,0.1"), "_set_upper_limit_angular", "_get_upper_limit_angular");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_limit/lower_angle", PROPERTY_HINT_RANGE, "-180,180,0.1"), "_set_lower_limit_angular", "_get_lower_limit_angular");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit/restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_LIMIT_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit/damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_LIMIT_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_motion/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_MOTION_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_motion/restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_MOTION_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_motion/damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_MOTION_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_ortho/softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_ORTHOGONAL_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_ortho/restitution", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_ORTHOGONAL_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_ortho/damping", PROPERTY_HINT_RANGE, "0,16.0,0.01"), "set_param", "get_param", PARAM_ANGULAR_ORTHOGONAL_DAMPING);

	BIND_CONSTANT(PARAM_LINEAR_LIMIT_UPPER);
	BIND_CONSTANT(PARAM_LINEAR_LIMIT_LOWER);
	BIND_CONSTANT(PARAM_LINEAR_LIMIT_SOFTNESS);
	BIND_CONSTANT(PARAM_LINEAR_LIMIT_RESTITUTION);
	BIND_CONSTANT(PARAM_LINEAR_LIMIT_DAMPING);
	BIND_CONSTANT(PARAM_LINEAR_MOTION_SOFTNESS);
	BIND_CONSTANT(PARAM_LINEAR_MOTION_RESTITUTION);
	BIND_CONSTANT(PARAM_LINEAR_MOTION_DAMPING);
	BIND_CONSTANT(PARAM_LINEAR_ORTHOGONAL_SOFTNESS);
	BIND_CONSTANT(PARAM_LINEAR_ORTHOGONAL_RESTITUTION);
	BIND_CONSTANT(PARAM_LINEAR_ORTHOGONAL_DAMPING);

	BIND_CONSTANT(PARAM_ANGULAR_LIMIT_UPPER);
	BIND_CONSTANT(PARAM_ANGULAR_LIMIT_LOWER);
	BIND_CONSTANT(PARAM_ANGULAR_LIMIT_SOFTNESS);
	BIND_CONSTANT(PARAM_ANGULAR_LIMIT_RESTITUTION);
	BIND_CONSTANT(PARAM_ANGULAR_LIMIT_DAMPING);
	BIND_CONSTANT(PARAM_ANGULAR_MOTION_SOFTNESS);
	BIND_CONSTANT(PARAM_ANGULAR_MOTION_RESTITUTION);
	BIND_CONSTANT(PARAM_ANGULAR_MOTION_DAMPING);
	BIND_CONSTANT(PARAM_ANGULAR_ORTHOGONAL_SOFTNESS);
	BIND_CONSTANT(PARAM_ANGULAR_ORTHOGONAL_RESTITUTION);
	BIND_CONSTANT(PARAM_ANGULAR_ORTHOGONAL_DAMPING);

	BIND_CONSTANT(PARAM_MAX);
}

void SliderJoint::set_param(Param p_param, float p_value) {

	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params[p_param] = p_value;
	if (get_joint().is_valid())
		PhysicsServer::get_singleton()->slider_joint_set_param(get_joint(), PhysicsServer::SliderJointParam(p_param), p_value);
	update_gizmo();
}
float SliderJoint::get_param(Param p_param) const {

	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params[p_param];
}

RID SliderJoint::_configure_joint(PhysicsBody *body_a, PhysicsBody *body_b) {

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

	RID j = PhysicsServer::get_singleton()->joint_create_slider(body_a->get_rid(), local_a, body_b ? body_b->get_rid() : RID(), local_b);
	for (int i = 0; i < PARAM_MAX; i++) {
		PhysicsServer::get_singleton()->slider_joint_set_param(j, PhysicsServer::SliderJointParam(i), params[i]);
	}

	return j;
}

SliderJoint::SliderJoint() {

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

void ConeTwistJoint::_set_swing_span(float p_limit_angular) {

	set_param(PARAM_SWING_SPAN, Math::deg2rad(p_limit_angular));
}

float ConeTwistJoint::_get_swing_span() const {

	return Math::rad2deg(get_param(PARAM_SWING_SPAN));
}

void ConeTwistJoint::_set_twist_span(float p_limit_angular) {

	set_param(PARAM_TWIST_SPAN, Math::deg2rad(p_limit_angular));
}

float ConeTwistJoint::_get_twist_span() const {

	return Math::rad2deg(get_param(PARAM_TWIST_SPAN));
}

void ConeTwistJoint::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &ConeTwistJoint::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &ConeTwistJoint::get_param);

	ClassDB::bind_method(D_METHOD("_set_swing_span", "swing_span"), &ConeTwistJoint::_set_swing_span);
	ClassDB::bind_method(D_METHOD("_get_swing_span"), &ConeTwistJoint::_get_swing_span);

	ClassDB::bind_method(D_METHOD("_set_twist_span", "twist_span"), &ConeTwistJoint::_set_twist_span);
	ClassDB::bind_method(D_METHOD("_get_twist_span"), &ConeTwistJoint::_get_twist_span);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "swing_span", PROPERTY_HINT_RANGE, "-180,180,0.1"), "_set_swing_span", "_get_swing_span");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "twist_span", PROPERTY_HINT_RANGE, "-40000,40000,0.1"), "_set_twist_span", "_get_twist_span");

	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "bias", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_BIAS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "softness", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "relaxation", PROPERTY_HINT_RANGE, "0.01,16.0,0.01"), "set_param", "get_param", PARAM_RELAXATION);

	BIND_CONSTANT(PARAM_SWING_SPAN);
	BIND_CONSTANT(PARAM_TWIST_SPAN);
	BIND_CONSTANT(PARAM_BIAS);
	BIND_CONSTANT(PARAM_SOFTNESS);
	BIND_CONSTANT(PARAM_RELAXATION);
	BIND_CONSTANT(PARAM_MAX);
}

void ConeTwistJoint::set_param(Param p_param, float p_value) {

	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params[p_param] = p_value;
	if (get_joint().is_valid())
		PhysicsServer::get_singleton()->cone_twist_joint_set_param(get_joint(), PhysicsServer::ConeTwistJointParam(p_param), p_value);

	update_gizmo();
}
float ConeTwistJoint::get_param(Param p_param) const {

	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params[p_param];
}

RID ConeTwistJoint::_configure_joint(PhysicsBody *body_a, PhysicsBody *body_b) {

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

	RID j = PhysicsServer::get_singleton()->joint_create_cone_twist(body_a->get_rid(), local_a, body_b ? body_b->get_rid() : RID(), local_b);
	for (int i = 0; i < PARAM_MAX; i++) {
		PhysicsServer::get_singleton()->cone_twist_joint_set_param(j, PhysicsServer::ConeTwistJointParam(i), params[i]);
	}

	return j;
}

ConeTwistJoint::ConeTwistJoint() {

	params[PARAM_SWING_SPAN] = Math_PI * 0.25;
	params[PARAM_TWIST_SPAN] = Math_PI;
	params[PARAM_BIAS] = 0.3;
	params[PARAM_SOFTNESS] = 0.8;
	params[PARAM_RELAXATION] = 1.0;
}

/////////////////////////////////////////////////////////////////////

void Generic6DOFJoint::_set_angular_hi_limit_x(float p_limit_angular) {

	set_param_x(PARAM_ANGULAR_UPPER_LIMIT, Math::deg2rad(p_limit_angular));
}

float Generic6DOFJoint::_get_angular_hi_limit_x() const {

	return Math::rad2deg(get_param_x(PARAM_ANGULAR_UPPER_LIMIT));
}

void Generic6DOFJoint::_set_angular_lo_limit_x(float p_limit_angular) {

	set_param_x(PARAM_ANGULAR_LOWER_LIMIT, Math::deg2rad(p_limit_angular));
}

float Generic6DOFJoint::_get_angular_lo_limit_x() const {

	return Math::rad2deg(get_param_x(PARAM_ANGULAR_LOWER_LIMIT));
}

void Generic6DOFJoint::_set_angular_hi_limit_y(float p_limit_angular) {

	set_param_y(PARAM_ANGULAR_UPPER_LIMIT, Math::deg2rad(p_limit_angular));
}

float Generic6DOFJoint::_get_angular_hi_limit_y() const {

	return Math::rad2deg(get_param_y(PARAM_ANGULAR_UPPER_LIMIT));
}

void Generic6DOFJoint::_set_angular_lo_limit_y(float p_limit_angular) {

	set_param_y(PARAM_ANGULAR_LOWER_LIMIT, Math::deg2rad(p_limit_angular));
}

float Generic6DOFJoint::_get_angular_lo_limit_y() const {

	return Math::rad2deg(get_param_y(PARAM_ANGULAR_LOWER_LIMIT));
}

void Generic6DOFJoint::_set_angular_hi_limit_z(float p_limit_angular) {

	set_param_z(PARAM_ANGULAR_UPPER_LIMIT, Math::deg2rad(p_limit_angular));
}

float Generic6DOFJoint::_get_angular_hi_limit_z() const {

	return Math::rad2deg(get_param_z(PARAM_ANGULAR_UPPER_LIMIT));
}

void Generic6DOFJoint::_set_angular_lo_limit_z(float p_limit_angular) {

	set_param_z(PARAM_ANGULAR_LOWER_LIMIT, Math::deg2rad(p_limit_angular));
}

float Generic6DOFJoint::_get_angular_lo_limit_z() const {

	return Math::rad2deg(get_param_z(PARAM_ANGULAR_LOWER_LIMIT));
}

void Generic6DOFJoint::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_set_angular_hi_limit_x", "angle"), &Generic6DOFJoint::_set_angular_hi_limit_x);
	ClassDB::bind_method(D_METHOD("_get_angular_hi_limit_x"), &Generic6DOFJoint::_get_angular_hi_limit_x);

	ClassDB::bind_method(D_METHOD("_set_angular_lo_limit_x", "angle"), &Generic6DOFJoint::_set_angular_lo_limit_x);
	ClassDB::bind_method(D_METHOD("_get_angular_lo_limit_x"), &Generic6DOFJoint::_get_angular_lo_limit_x);

	ClassDB::bind_method(D_METHOD("_set_angular_hi_limit_y", "angle"), &Generic6DOFJoint::_set_angular_hi_limit_y);
	ClassDB::bind_method(D_METHOD("_get_angular_hi_limit_y"), &Generic6DOFJoint::_get_angular_hi_limit_y);

	ClassDB::bind_method(D_METHOD("_set_angular_lo_limit_y", "angle"), &Generic6DOFJoint::_set_angular_lo_limit_y);
	ClassDB::bind_method(D_METHOD("_get_angular_lo_limit_y"), &Generic6DOFJoint::_get_angular_lo_limit_y);

	ClassDB::bind_method(D_METHOD("_set_angular_hi_limit_z", "angle"), &Generic6DOFJoint::_set_angular_hi_limit_z);
	ClassDB::bind_method(D_METHOD("_get_angular_hi_limit_z"), &Generic6DOFJoint::_get_angular_hi_limit_z);

	ClassDB::bind_method(D_METHOD("_set_angular_lo_limit_z", "angle"), &Generic6DOFJoint::_set_angular_lo_limit_z);
	ClassDB::bind_method(D_METHOD("_get_angular_lo_limit_z"), &Generic6DOFJoint::_get_angular_lo_limit_z);

	ClassDB::bind_method(D_METHOD("set_param_x", "param", "value"), &Generic6DOFJoint::set_param_x);
	ClassDB::bind_method(D_METHOD("get_param_x", "param"), &Generic6DOFJoint::get_param_x);

	ClassDB::bind_method(D_METHOD("set_param_y", "param", "value"), &Generic6DOFJoint::set_param_y);
	ClassDB::bind_method(D_METHOD("get_param_y", "param"), &Generic6DOFJoint::get_param_y);

	ClassDB::bind_method(D_METHOD("set_param_z", "param", "value"), &Generic6DOFJoint::set_param_z);
	ClassDB::bind_method(D_METHOD("get_param_z", "param"), &Generic6DOFJoint::get_param_z);

	ClassDB::bind_method(D_METHOD("set_flag_x", "flag", "value"), &Generic6DOFJoint::set_flag_x);
	ClassDB::bind_method(D_METHOD("get_flag_x", "flag"), &Generic6DOFJoint::get_flag_x);

	ClassDB::bind_method(D_METHOD("set_flag_y", "flag", "value"), &Generic6DOFJoint::set_flag_y);
	ClassDB::bind_method(D_METHOD("get_flag_y", "flag"), &Generic6DOFJoint::get_flag_y);

	ClassDB::bind_method(D_METHOD("set_flag_z", "flag", "value"), &Generic6DOFJoint::set_flag_z);
	ClassDB::bind_method(D_METHOD("get_flag_z", "flag"), &Generic6DOFJoint::get_flag_z);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "linear_limit_x/enabled"), "set_flag_x", "get_flag_x", FLAG_ENABLE_LINEAR_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_x/upper_distance"), "set_param_x", "get_param_x", PARAM_LINEAR_UPPER_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_x/lower_distance"), "set_param_x", "get_param_x", PARAM_LINEAR_LOWER_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_x/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_x", "get_param_x", PARAM_LINEAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_x/restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_x", "get_param_x", PARAM_LINEAR_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_x/damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_x", "get_param_x", PARAM_LINEAR_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_limit_x/enabled"), "set_flag_x", "get_flag_x", FLAG_ENABLE_ANGULAR_LIMIT);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_limit_x/upper_angle", PROPERTY_HINT_RANGE, "-180,180,0.01"), "_set_angular_hi_limit_x", "_get_angular_hi_limit_x");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_limit_x/lower_angle", PROPERTY_HINT_RANGE, "-180,180,0.01"), "_set_angular_lo_limit_x", "_get_angular_lo_limit_x");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_x/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_x", "get_param_x", PARAM_ANGULAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_x/restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_x", "get_param_x", PARAM_ANGULAR_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_x/damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_x", "get_param_x", PARAM_ANGULAR_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_x/force_limit"), "set_param_x", "get_param_x", PARAM_ANGULAR_FORCE_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_x/erp"), "set_param_x", "get_param_x", PARAM_ANGULAR_ERP);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_motor_x/enabled"), "set_flag_x", "get_flag_x", FLAG_ENABLE_MOTOR);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_motor_x/target_velocity"), "set_param_x", "get_param_x", PARAM_ANGULAR_MOTOR_TARGET_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_motor_x/force_limit"), "set_param_x", "get_param_x", PARAM_ANGULAR_MOTOR_FORCE_LIMIT);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "linear_limit_y/enabled"), "set_flag_y", "get_flag_y", FLAG_ENABLE_LINEAR_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_y/upper_distance"), "set_param_y", "get_param_y", PARAM_LINEAR_UPPER_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_y/lower_distance"), "set_param_y", "get_param_y", PARAM_LINEAR_LOWER_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_y/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_y", "get_param_y", PARAM_LINEAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_y/restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_y", "get_param_y", PARAM_LINEAR_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_y/damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_y", "get_param_y", PARAM_LINEAR_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_limit_y/enabled"), "set_flag_y", "get_flag_y", FLAG_ENABLE_ANGULAR_LIMIT);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_limit_y/upper_angle", PROPERTY_HINT_RANGE, "-180,180,0.01"), "_set_angular_hi_limit_y", "_get_angular_hi_limit_y");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_limit_y/lower_angle", PROPERTY_HINT_RANGE, "-180,180,0.01"), "_set_angular_lo_limit_y", "_get_angular_lo_limit_y");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_y/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_y", "get_param_y", PARAM_ANGULAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_y/restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_y", "get_param_y", PARAM_ANGULAR_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_y/damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_y", "get_param_y", PARAM_ANGULAR_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_y/force_limit"), "set_param_y", "get_param_y", PARAM_ANGULAR_FORCE_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_y/erp"), "set_param_y", "get_param_y", PARAM_ANGULAR_ERP);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_motor_y/enabled"), "set_flag_y", "get_flag_y", FLAG_ENABLE_MOTOR);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_motor_y/target_velocity"), "set_param_y", "get_param_y", PARAM_ANGULAR_MOTOR_TARGET_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_motor_y/force_limit"), "set_param_y", "get_param_y", PARAM_ANGULAR_MOTOR_FORCE_LIMIT);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "linear_limit_z/enabled"), "set_flag_z", "get_flag_z", FLAG_ENABLE_LINEAR_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_z/upper_distance"), "set_param_z", "get_param_z", PARAM_LINEAR_UPPER_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_z/lower_distance"), "set_param_z", "get_param_z", PARAM_LINEAR_LOWER_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_z/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_z", "get_param_z", PARAM_LINEAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_z/restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_z", "get_param_z", PARAM_LINEAR_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "linear_limit_z/damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_z", "get_param_z", PARAM_LINEAR_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_limit_z/enabled"), "set_flag_z", "get_flag_z", FLAG_ENABLE_ANGULAR_LIMIT);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_limit_z/upper_angle", PROPERTY_HINT_RANGE, "-180,180,0.01"), "_set_angular_hi_limit_z", "_get_angular_hi_limit_z");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_limit_z/lower_angle", PROPERTY_HINT_RANGE, "-180,180,0.01"), "_set_angular_lo_limit_z", "_get_angular_lo_limit_z");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_z/softness", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_z", "get_param_z", PARAM_ANGULAR_LIMIT_SOFTNESS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_z/restitution", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_z", "get_param_z", PARAM_ANGULAR_RESTITUTION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_z/damping", PROPERTY_HINT_RANGE, "0.01,16,0.01"), "set_param_z", "get_param_z", PARAM_ANGULAR_DAMPING);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_z/force_limit"), "set_param_z", "get_param_z", PARAM_ANGULAR_FORCE_LIMIT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_limit_z/erp"), "set_param_z", "get_param_z", PARAM_ANGULAR_ERP);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "angular_motor_z/enabled"), "set_flag_z", "get_flag_z", FLAG_ENABLE_MOTOR);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_motor_z/target_velocity"), "set_param_z", "get_param_z", PARAM_ANGULAR_MOTOR_TARGET_VELOCITY);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "angular_motor_z/force_limit"), "set_param_z", "get_param_z", PARAM_ANGULAR_MOTOR_FORCE_LIMIT);

	BIND_CONSTANT(PARAM_LINEAR_LOWER_LIMIT);
	BIND_CONSTANT(PARAM_LINEAR_UPPER_LIMIT);
	BIND_CONSTANT(PARAM_LINEAR_LIMIT_SOFTNESS);
	BIND_CONSTANT(PARAM_LINEAR_RESTITUTION);
	BIND_CONSTANT(PARAM_LINEAR_DAMPING);
	BIND_CONSTANT(PARAM_ANGULAR_LOWER_LIMIT);
	BIND_CONSTANT(PARAM_ANGULAR_UPPER_LIMIT);
	BIND_CONSTANT(PARAM_ANGULAR_LIMIT_SOFTNESS);
	BIND_CONSTANT(PARAM_ANGULAR_DAMPING);
	BIND_CONSTANT(PARAM_ANGULAR_RESTITUTION);
	BIND_CONSTANT(PARAM_ANGULAR_FORCE_LIMIT);
	BIND_CONSTANT(PARAM_ANGULAR_ERP);
	BIND_CONSTANT(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY);
	BIND_CONSTANT(PARAM_ANGULAR_MOTOR_FORCE_LIMIT);
	BIND_CONSTANT(PARAM_MAX);

	BIND_CONSTANT(FLAG_ENABLE_LINEAR_LIMIT);
	BIND_CONSTANT(FLAG_ENABLE_ANGULAR_LIMIT);
	BIND_CONSTANT(FLAG_ENABLE_MOTOR);
	BIND_CONSTANT(FLAG_MAX);
}

void Generic6DOFJoint::set_param_x(Param p_param, float p_value) {

	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params_x[p_param] = p_value;
	if (get_joint().is_valid())
		PhysicsServer::get_singleton()->generic_6dof_joint_set_param(get_joint(), Vector3::AXIS_X, PhysicsServer::G6DOFJointAxisParam(p_param), p_value);

	update_gizmo();
}
float Generic6DOFJoint::get_param_x(Param p_param) const {

	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params_x[p_param];
}

void Generic6DOFJoint::set_param_y(Param p_param, float p_value) {

	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params_y[p_param] = p_value;
	if (get_joint().is_valid())
		PhysicsServer::get_singleton()->generic_6dof_joint_set_param(get_joint(), Vector3::AXIS_Y, PhysicsServer::G6DOFJointAxisParam(p_param), p_value);
	update_gizmo();
}
float Generic6DOFJoint::get_param_y(Param p_param) const {

	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params_y[p_param];
}

void Generic6DOFJoint::set_param_z(Param p_param, float p_value) {

	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params_z[p_param] = p_value;
	if (get_joint().is_valid())
		PhysicsServer::get_singleton()->generic_6dof_joint_set_param(get_joint(), Vector3::AXIS_Z, PhysicsServer::G6DOFJointAxisParam(p_param), p_value);
	update_gizmo();
}
float Generic6DOFJoint::get_param_z(Param p_param) const {

	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params_z[p_param];
}

void Generic6DOFJoint::set_flag_x(Flag p_flag, bool p_enabled) {

	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	flags_x[p_flag] = p_enabled;
	if (get_joint().is_valid())
		PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(get_joint(), Vector3::AXIS_X, PhysicsServer::G6DOFJointAxisFlag(p_flag), p_enabled);
	update_gizmo();
}
bool Generic6DOFJoint::get_flag_x(Flag p_flag) const {

	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	return flags_x[p_flag];
}

void Generic6DOFJoint::set_flag_y(Flag p_flag, bool p_enabled) {

	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	flags_y[p_flag] = p_enabled;
	if (get_joint().is_valid())
		PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(get_joint(), Vector3::AXIS_Y, PhysicsServer::G6DOFJointAxisFlag(p_flag), p_enabled);
	update_gizmo();
}
bool Generic6DOFJoint::get_flag_y(Flag p_flag) const {

	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	return flags_y[p_flag];
}

void Generic6DOFJoint::set_flag_z(Flag p_flag, bool p_enabled) {

	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	flags_z[p_flag] = p_enabled;
	if (get_joint().is_valid())
		PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(get_joint(), Vector3::AXIS_Z, PhysicsServer::G6DOFJointAxisFlag(p_flag), p_enabled);
	update_gizmo();
}
bool Generic6DOFJoint::get_flag_z(Flag p_flag) const {

	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	return flags_z[p_flag];
}

RID Generic6DOFJoint::_configure_joint(PhysicsBody *body_a, PhysicsBody *body_b) {

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

	RID j = PhysicsServer::get_singleton()->joint_create_generic_6dof(body_a->get_rid(), local_a, body_b ? body_b->get_rid() : RID(), local_b);
	for (int i = 0; i < PARAM_MAX; i++) {
		PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, Vector3::AXIS_X, PhysicsServer::G6DOFJointAxisParam(i), params_x[i]);
		PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, Vector3::AXIS_Y, PhysicsServer::G6DOFJointAxisParam(i), params_y[i]);
		PhysicsServer::get_singleton()->generic_6dof_joint_set_param(j, Vector3::AXIS_Z, PhysicsServer::G6DOFJointAxisParam(i), params_z[i]);
	}
	for (int i = 0; i < FLAG_MAX; i++) {
		PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(j, Vector3::AXIS_X, PhysicsServer::G6DOFJointAxisFlag(i), flags_x[i]);
		PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(j, Vector3::AXIS_Y, PhysicsServer::G6DOFJointAxisFlag(i), flags_y[i]);
		PhysicsServer::get_singleton()->generic_6dof_joint_set_flag(j, Vector3::AXIS_Z, PhysicsServer::G6DOFJointAxisFlag(i), flags_z[i]);
	}

	return j;
}

Generic6DOFJoint::Generic6DOFJoint() {

	set_param_x(PARAM_LINEAR_LOWER_LIMIT, 0);
	set_param_x(PARAM_LINEAR_UPPER_LIMIT, 0);
	set_param_x(PARAM_LINEAR_LIMIT_SOFTNESS, 0.7);
	set_param_x(PARAM_LINEAR_RESTITUTION, 0.5);
	set_param_x(PARAM_LINEAR_DAMPING, 1.0);
	set_param_x(PARAM_ANGULAR_LOWER_LIMIT, 0);
	set_param_x(PARAM_ANGULAR_UPPER_LIMIT, 0);
	set_param_x(PARAM_ANGULAR_LIMIT_SOFTNESS, 0.5f);
	set_param_x(PARAM_ANGULAR_DAMPING, 1.0f);
	set_param_x(PARAM_ANGULAR_RESTITUTION, 0);
	set_param_x(PARAM_ANGULAR_FORCE_LIMIT, 0);
	set_param_x(PARAM_ANGULAR_ERP, 0.5);
	set_param_x(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, 0);
	set_param_x(PARAM_ANGULAR_MOTOR_FORCE_LIMIT, 300);

	set_flag_x(FLAG_ENABLE_ANGULAR_LIMIT, true);
	set_flag_x(FLAG_ENABLE_LINEAR_LIMIT, true);
	set_flag_x(FLAG_ENABLE_MOTOR, false);

	set_param_y(PARAM_LINEAR_LOWER_LIMIT, 0);
	set_param_y(PARAM_LINEAR_UPPER_LIMIT, 0);
	set_param_y(PARAM_LINEAR_LIMIT_SOFTNESS, 0.7);
	set_param_y(PARAM_LINEAR_RESTITUTION, 0.5);
	set_param_y(PARAM_LINEAR_DAMPING, 1.0);
	set_param_y(PARAM_ANGULAR_LOWER_LIMIT, 0);
	set_param_y(PARAM_ANGULAR_UPPER_LIMIT, 0);
	set_param_y(PARAM_ANGULAR_LIMIT_SOFTNESS, 0.5f);
	set_param_y(PARAM_ANGULAR_DAMPING, 1.0f);
	set_param_y(PARAM_ANGULAR_RESTITUTION, 0);
	set_param_y(PARAM_ANGULAR_FORCE_LIMIT, 0);
	set_param_y(PARAM_ANGULAR_ERP, 0.5);
	set_param_y(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, 0);
	set_param_y(PARAM_ANGULAR_MOTOR_FORCE_LIMIT, 300);

	set_flag_y(FLAG_ENABLE_ANGULAR_LIMIT, true);
	set_flag_y(FLAG_ENABLE_LINEAR_LIMIT, true);
	set_flag_y(FLAG_ENABLE_MOTOR, false);

	set_param_z(PARAM_LINEAR_LOWER_LIMIT, 0);
	set_param_z(PARAM_LINEAR_UPPER_LIMIT, 0);
	set_param_z(PARAM_LINEAR_LIMIT_SOFTNESS, 0.7);
	set_param_z(PARAM_LINEAR_RESTITUTION, 0.5);
	set_param_z(PARAM_LINEAR_DAMPING, 1.0);
	set_param_z(PARAM_ANGULAR_LOWER_LIMIT, 0);
	set_param_z(PARAM_ANGULAR_UPPER_LIMIT, 0);
	set_param_z(PARAM_ANGULAR_LIMIT_SOFTNESS, 0.5f);
	set_param_z(PARAM_ANGULAR_DAMPING, 1.0f);
	set_param_z(PARAM_ANGULAR_RESTITUTION, 0);
	set_param_z(PARAM_ANGULAR_FORCE_LIMIT, 0);
	set_param_z(PARAM_ANGULAR_ERP, 0.5);
	set_param_z(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, 0);
	set_param_z(PARAM_ANGULAR_MOTOR_FORCE_LIMIT, 300);

	set_flag_z(FLAG_ENABLE_ANGULAR_LIMIT, true);
	set_flag_z(FLAG_ENABLE_LINEAR_LIMIT, true);
	set_flag_z(FLAG_ENABLE_MOTOR, false);
}

#if 0

void PhysicsJoint::_set(const String& p_name, const Variant& p_value) {

	if (p_name=="body_A")
		set_body_A(p_value);
	else if (p_name=="body_B")
		set_body_B(p_value);
	else if (p_name=="active")
		set_active(p_value);
	else if (p_name=="no_collision")
		set_disable_collision(p_value);
}
Variant PhysicsJoint::_get(const String& p_name) const {

	if (p_name=="body_A")
		return get_body_A();
	else if (p_name=="body_B")
		return get_body_B();
	else if (p_name=="active")
		return is_active();
	else if (p_name=="no_collision")
		return has_disable_collision();

	return Variant();
}
void PhysicsJoint::_get_property_list( List<PropertyInfo> *p_list) const {


	p_list->push_back( PropertyInfo( Variant::NODE_PATH, "body_A" ) );
	p_list->push_back( PropertyInfo( Variant::NODE_PATH, "body_B" ) );
	p_list->push_back( PropertyInfo( Variant::BOOL, "active" ) );
	p_list->push_back( PropertyInfo( Variant::BOOL, "no_collision" ) );
}
void PhysicsJoint::_notification(int p_what) {


	switch(p_what) {

		case NOTIFICATION_PARENT_CONFIGURED: {

			_connect();
			if (get_root_node()->get_editor() && !indicator.is_valid()) {

				indicator=VisualServer::get_singleton()->poly_create();
				RID mat=VisualServer::get_singleton()->fixed_material_create();
				VisualServer::get_singleton()->material_set_flag( mat, VisualServer::MATERIAL_FLAG_UNSHADED, true );
				VisualServer::get_singleton()->material_set_flag( mat, VisualServer::MATERIAL_FLAG_ONTOP, true );
				VisualServer::get_singleton()->material_set_flag( mat, VisualServer::MATERIAL_FLAG_WIREFRAME, true );
				VisualServer::get_singleton()->material_set_flag( mat, VisualServer::MATERIAL_FLAG_DOUBLE_SIDED, true );
				VisualServer::get_singleton()->material_set_line_width( mat, 3 );

				VisualServer::get_singleton()->poly_set_material(indicator,mat,true);
				_update_indicator();

			}

			if (indicator.is_valid()) {

				indicator_instance=VisualServer::get_singleton()->instance_create(indicator,get_world()->get_scenario());
				VisualServer::get_singleton()->instance_attach_object_instance_ID( indicator_instance,get_instance_ID() );
			}
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (indicator_instance.is_valid()) {

				VisualServer::get_singleton()->instance_set_transform(indicator_instance,get_global_transform());
			}
		} break;
		case NOTIFICATION_EXIT_SCENE: {

			if (indicator_instance.is_valid()) {

				VisualServer::get_singleton()->free(indicator_instance);
			}
			_disconnect();

		} break;

	}
}


RID PhysicsJoint::_get_visual_instance_rid() const {

	return indicator_instance;

}

void PhysicsJoint::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_get_visual_instance_rid"),&PhysicsJoint::_get_visual_instance_rid);
	ClassDB::bind_method(D_METHOD("set_body_A","path"),&PhysicsJoint::set_body_A);
	ClassDB::bind_method(D_METHOD("set_body_B"),&PhysicsJoint::set_body_B);
	ClassDB::bind_method(D_METHOD("get_body_A","path"),&PhysicsJoint::get_body_A);
	ClassDB::bind_method(D_METHOD("get_body_B"),&PhysicsJoint::get_body_B);

	ClassDB::bind_method(D_METHOD("set_active","active"),&PhysicsJoint::set_active);
	ClassDB::bind_method(D_METHOD("is_active"),&PhysicsJoint::is_active);

	ClassDB::bind_method(D_METHOD("set_disable_collision","disable"),&PhysicsJoint::set_disable_collision);
	ClassDB::bind_method(D_METHOD("has_disable_collision"),&PhysicsJoint::has_disable_collision);


	ClassDB::bind_method("reconnect",&PhysicsJoint::reconnect);

	ClassDB::bind_method(D_METHOD("get_rid"),&PhysicsJoint::get_rid);

}

void PhysicsJoint::set_body_A(const NodePath& p_path) {

	_disconnect();
	body_A=p_path;
	_connect();
	_change_notify("body_A");
}
void PhysicsJoint::set_body_B(const NodePath& p_path) {

	_disconnect();
	body_B=p_path;
	_connect();
	_change_notify("body_B");

}
NodePath PhysicsJoint::get_body_A() const {

	return body_A;
}
NodePath PhysicsJoint::get_body_B() const {

	return body_B;
}

void PhysicsJoint::set_active(bool p_active) {

	active=p_active;
	if (is_inside_scene()) {
		PhysicsServer::get_singleton()->joint_set_active(joint,active);
	}
	_change_notify("active");
}

void PhysicsJoint::set_disable_collision(bool p_active) {

	if (no_collision==p_active)
		return;
	_disconnect();
	no_collision=p_active;
	_connect();

	_change_notify("no_collision");
}
bool PhysicsJoint::has_disable_collision() const {

	return no_collision;
}



bool PhysicsJoint::is_active() const {

	return active;
}

void PhysicsJoint::_disconnect() {

	if (!is_inside_scene())
		return;

	if (joint.is_valid())
		PhysicsServer::get_singleton()->free(joint);

	joint=RID();

	Node *nA = get_node(body_A);
	Node *nB = get_node(body_B);

	PhysicsBody *A = nA?nA->cast_to<PhysicsBody>():NULL;
	PhysicsBody *B = nA?nB->cast_to<PhysicsBody>():NULL;

	if (!A ||!B)
		return;

	if (no_collision)
		PhysicsServer::get_singleton()->body_remove_collision_exception(A->get_body(),B->get_body());

}
void PhysicsJoint::_connect() {

	if (!is_inside_scene())
		return;

	ERR_FAIL_COND(joint.is_valid());

	Node *nA = get_node(body_A);
	Node *nB = get_node(body_B);

	PhysicsBody *A = nA?nA->cast_to<PhysicsBody>():NULL;
	PhysicsBody *B = nA?nB->cast_to<PhysicsBody>():NULL;

	if (!A && !B)
		return;

	if (B && !A)
		SWAP(B,A);

	joint = create(A,B);

	if (A<B)
		SWAP(A,B);

	if (no_collision)
		PhysicsServer::get_singleton()->body_add_collision_exception(A->get_body(),B->get_body());



}

void PhysicsJoint::reconnect() {

	_disconnect();
	_connect();

}


RID PhysicsJoint::get_rid() {

	return joint;
}


PhysicsJoint::PhysicsJoint() {

	active=true;
	no_collision=true;
}


PhysicsJoint::~PhysicsJoint() {

	if (indicator.is_valid()) {

		VisualServer::get_singleton()->free(indicator);
	}

}

/* PIN */

void PhysicsJointPin::_update_indicator() {


	VisualServer::get_singleton()->poly_clear(indicator);

	Vector<Color> colors;
	colors.push_back( Color(0.3,0.9,0.2,0.7) );
	colors.push_back( Color(0.3,0.9,0.2,0.7) );

	Vector<Vector3> points;
	points.resize(2);
	points[0]=Vector3(Vector3(-0.2,0,0));
	points[1]=Vector3(Vector3(0.2,0,0));
	VisualServer::get_singleton()->poly_add_primitive(indicator,points,Vector<Vector3>(),colors,Vector<Vector3>());

	points[0]=Vector3(Vector3(0,-0.2,0));
	points[1]=Vector3(Vector3(0,0.2,0));
	VisualServer::get_singleton()->poly_add_primitive(indicator,points,Vector<Vector3>(),colors,Vector<Vector3>());

	points[0]=Vector3(Vector3(0,0,-0.2));
	points[1]=Vector3(Vector3(0,0,0.2));
	VisualServer::get_singleton()->poly_add_primitive(indicator,points,Vector<Vector3>(),colors,Vector<Vector3>());

}

RID PhysicsJointPin::create(PhysicsBody*A,PhysicsBody*B) {

	RID body_A = A->get_body();
	RID body_B = B?B->get_body():RID();

	ERR_FAIL_COND_V( !body_A.is_valid(), RID() );

	Vector3 pin_pos = get_global_transform().origin;

	if (body_B.is_valid())
		return PhysicsServer::get_singleton()->joint_create_double_pin_global(body_A,pin_pos,body_B,pin_pos);
	else
		return PhysicsServer::get_singleton()->joint_create_pin(body_A,A->get_global_transform().xform_inv(pin_pos),pin_pos);
}

PhysicsJointPin::PhysicsJointPin() {


}
#endif
