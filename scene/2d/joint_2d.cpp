/**************************************************************************/
/*  joint_2d.cpp                                                          */
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

#include "joint_2d.h"

#include "physics_body_2d.h"
#include "scene/scene_string_names.h"

void Joint2D::_disconnect_signals() {
	Node *node_a = get_node_or_null(a);
	PhysicsBody2D *body_a = Object::cast_to<PhysicsBody2D>(node_a);
	if (body_a) {
		body_a->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Joint2D::_body_exit_tree));
	}

	Node *node_b = get_node_or_null(b);
	PhysicsBody2D *body_b = Object::cast_to<PhysicsBody2D>(node_b);
	if (body_b) {
		body_b->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Joint2D::_body_exit_tree));
	}
}

void Joint2D::_body_exit_tree() {
	_disconnect_signals();
	_update_joint(true);
	update_configuration_warnings();
}

void Joint2D::_update_joint(bool p_only_free) {
	if (ba.is_valid() && bb.is_valid() && exclude_from_collision) {
		PhysicsServer2D::get_singleton()->joint_disable_collisions_between_bodies(joint, false);
	}

	ba = RID();
	bb = RID();
	configured = false;

	if (p_only_free || !is_inside_tree()) {
		PhysicsServer2D::get_singleton()->joint_clear(joint);
		warning = String();
		return;
	}

	Node *node_a = get_node_or_null(a);
	Node *node_b = get_node_or_null(b);

	PhysicsBody2D *body_a = Object::cast_to<PhysicsBody2D>(node_a);
	PhysicsBody2D *body_b = Object::cast_to<PhysicsBody2D>(node_b);

	bool valid = false;

	if (node_a && !body_a && node_b && !body_b) {
		warning = RTR("Node A and Node B must be PhysicsBody2Ds");
	} else if (node_a && !body_a) {
		warning = RTR("Node A must be a PhysicsBody2D");
	} else if (node_b && !body_b) {
		warning = RTR("Node B must be a PhysicsBody2D");
	} else if (!body_a || !body_b) {
		warning = RTR("Joint is not connected to two PhysicsBody2Ds");
	} else if (body_a == body_b) {
		warning = RTR("Node A and Node B must be different PhysicsBody2Ds");
	} else {
		warning = String();
		valid = true;
	}

	update_configuration_warnings();

	if (!valid) {
		PhysicsServer2D::get_singleton()->joint_clear(joint);
		return;
	}

	if (body_a) {
		body_a->force_update_transform();
	}

	if (body_b) {
		body_b->force_update_transform();
	}

	configured = true;

	_configure_joint(joint, body_a, body_b);

	ERR_FAIL_COND_MSG(!joint.is_valid(), "Failed to configure the joint.");

	PhysicsServer2D::get_singleton()->joint_set_param(joint, PhysicsServer2D::JOINT_PARAM_BIAS, bias);

	ba = body_a->get_rid();
	bb = body_b->get_rid();

	body_a->connect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Joint2D::_body_exit_tree));
	body_b->connect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Joint2D::_body_exit_tree));

	PhysicsServer2D::get_singleton()->joint_disable_collisions_between_bodies(joint, exclude_from_collision);
}

void Joint2D::set_node_a(const NodePath &p_node_a) {
	if (a == p_node_a) {
		return;
	}

	if (is_configured()) {
		_disconnect_signals();
	}

	a = p_node_a;
	if (Engine::get_singleton()->is_editor_hint()) {
		// When in editor, the setter may be called as a result of node rename.
		// It happens before the node actually changes its name, which triggers false warning.
		callable_mp(this, &Joint2D::_update_joint).call_deferred();
	} else {
		_update_joint();
	}
}

NodePath Joint2D::get_node_a() const {
	return a;
}

void Joint2D::set_node_b(const NodePath &p_node_b) {
	if (b == p_node_b) {
		return;
	}

	if (is_configured()) {
		_disconnect_signals();
	}

	b = p_node_b;
	if (Engine::get_singleton()->is_editor_hint()) {
		callable_mp(this, &Joint2D::_update_joint).call_deferred();
	} else {
		_update_joint();
	}
}

NodePath Joint2D::get_node_b() const {
	return b;
}

void Joint2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POST_ENTER_TREE: {
			if (is_configured()) {
				_disconnect_signals();
			}
			_update_joint();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (is_configured()) {
				_disconnect_signals();
			}
			_update_joint(true);
		} break;
	}
}

void Joint2D::set_bias(real_t p_bias) {
	bias = p_bias;
	if (joint.is_valid()) {
		PhysicsServer2D::get_singleton()->joint_set_param(joint, PhysicsServer2D::JOINT_PARAM_BIAS, bias);
	}
}

real_t Joint2D::get_bias() const {
	return bias;
}

void Joint2D::set_exclude_nodes_from_collision(bool p_enable) {
	if (exclude_from_collision == p_enable) {
		return;
	}
	if (is_configured()) {
		_disconnect_signals();
	}
	_update_joint(true);
	exclude_from_collision = p_enable;
	_update_joint();
}

bool Joint2D::get_exclude_nodes_from_collision() const {
	return exclude_from_collision;
}

PackedStringArray Joint2D::get_configuration_warnings() const {
	PackedStringArray warnings = Node2D::get_configuration_warnings();

	if (!warning.is_empty()) {
		warnings.push_back(warning);
	}

	return warnings;
}

void Joint2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_node_a", "node"), &Joint2D::set_node_a);
	ClassDB::bind_method(D_METHOD("get_node_a"), &Joint2D::get_node_a);

	ClassDB::bind_method(D_METHOD("set_node_b", "node"), &Joint2D::set_node_b);
	ClassDB::bind_method(D_METHOD("get_node_b"), &Joint2D::get_node_b);

	ClassDB::bind_method(D_METHOD("set_bias", "bias"), &Joint2D::set_bias);
	ClassDB::bind_method(D_METHOD("get_bias"), &Joint2D::get_bias);

	ClassDB::bind_method(D_METHOD("set_exclude_nodes_from_collision", "enable"), &Joint2D::set_exclude_nodes_from_collision);
	ClassDB::bind_method(D_METHOD("get_exclude_nodes_from_collision"), &Joint2D::get_exclude_nodes_from_collision);

	ClassDB::bind_method(D_METHOD("get_rid"), &Joint2D::get_rid);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "node_a", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "PhysicsBody2D"), "set_node_a", "get_node_a");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "node_b", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "PhysicsBody2D"), "set_node_b", "get_node_b");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bias", PROPERTY_HINT_RANGE, "0,0.9,0.001"), "set_bias", "get_bias");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disable_collision"), "set_exclude_nodes_from_collision", "get_exclude_nodes_from_collision");
}

Joint2D::Joint2D() {
	joint = PhysicsServer2D::get_singleton()->joint_create();
	set_hide_clip_children(true);
}

Joint2D::~Joint2D() {
	ERR_FAIL_NULL(PhysicsServer2D::get_singleton());
	PhysicsServer2D::get_singleton()->free(joint);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void PinJoint2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			if (!is_inside_tree()) {
				break;
			}

			if (!Engine::get_singleton()->is_editor_hint() && !get_tree()->is_debugging_collisions_hint()) {
				break;
			}

			draw_line(Point2(-10, 0), Point2(+10, 0), Color(0.7, 0.6, 0.0, 0.5), 3);
			draw_line(Point2(0, -10), Point2(0, +10), Color(0.7, 0.6, 0.0, 0.5), 3);
		} break;
	}
}

void PinJoint2D::_configure_joint(RID p_joint, PhysicsBody2D *body_a, PhysicsBody2D *body_b) {
	PhysicsServer2D::get_singleton()->joint_make_pin(p_joint, get_global_position(), body_a->get_rid(), body_b ? body_b->get_rid() : RID());
	PhysicsServer2D::get_singleton()->pin_joint_set_param(p_joint, PhysicsServer2D::PIN_JOINT_SOFTNESS, softness);
	PhysicsServer2D::get_singleton()->pin_joint_set_param(p_joint, PhysicsServer2D::PIN_JOINT_LIMIT_UPPER, angular_limit_upper);
	PhysicsServer2D::get_singleton()->pin_joint_set_param(p_joint, PhysicsServer2D::PIN_JOINT_LIMIT_LOWER, angular_limit_lower);
	PhysicsServer2D::get_singleton()->pin_joint_set_param(p_joint, PhysicsServer2D::PIN_JOINT_MOTOR_TARGET_VELOCITY, motor_target_velocity);
	PhysicsServer2D::get_singleton()->pin_joint_set_flag(p_joint, PhysicsServer2D::PIN_JOINT_FLAG_MOTOR_ENABLED, motor_enabled);
	PhysicsServer2D::get_singleton()->pin_joint_set_flag(p_joint, PhysicsServer2D::PIN_JOINT_FLAG_ANGULAR_LIMIT_ENABLED, angular_limit_enabled);
}

void PinJoint2D::set_softness(real_t p_softness) {
	if (softness == p_softness) {
		return;
	}
	softness = p_softness;
	queue_redraw();
	if (is_configured()) {
		PhysicsServer2D::get_singleton()->pin_joint_set_param(get_rid(), PhysicsServer2D::PIN_JOINT_SOFTNESS, p_softness);
	}
}

real_t PinJoint2D::get_softness() const {
	return softness;
}

void PinJoint2D::set_angular_limit_lower(real_t p_angular_limit_lower) {
	if (angular_limit_lower == p_angular_limit_lower) {
		return;
	}
	angular_limit_lower = p_angular_limit_lower;
	queue_redraw();
	if (is_configured()) {
		PhysicsServer2D::get_singleton()->pin_joint_set_param(get_rid(), PhysicsServer2D::PIN_JOINT_LIMIT_LOWER, p_angular_limit_lower);
	}
}

real_t PinJoint2D::get_angular_limit_lower() const {
	return angular_limit_lower;
}

void PinJoint2D::set_angular_limit_upper(real_t p_angular_limit_upper) {
	if (angular_limit_upper == p_angular_limit_upper) {
		return;
	}
	angular_limit_upper = p_angular_limit_upper;
	queue_redraw();
	if (is_configured()) {
		PhysicsServer2D::get_singleton()->pin_joint_set_param(get_rid(), PhysicsServer2D::PIN_JOINT_LIMIT_UPPER, p_angular_limit_upper);
	}
}

real_t PinJoint2D::get_angular_limit_upper() const {
	return angular_limit_upper;
}

void PinJoint2D::set_motor_target_velocity(real_t p_motor_target_velocity) {
	if (motor_target_velocity == p_motor_target_velocity) {
		return;
	}
	motor_target_velocity = p_motor_target_velocity;
	queue_redraw();
	if (is_configured()) {
		PhysicsServer2D::get_singleton()->pin_joint_set_param(get_rid(), PhysicsServer2D::PIN_JOINT_MOTOR_TARGET_VELOCITY, motor_target_velocity);
	}
}

real_t PinJoint2D::get_motor_target_velocity() const {
	return motor_target_velocity;
}

void PinJoint2D::set_motor_enabled(bool p_motor_enabled) {
	if (motor_enabled == p_motor_enabled) {
		return;
	}
	motor_enabled = p_motor_enabled;
	queue_redraw();
	if (is_configured()) {
		PhysicsServer2D::get_singleton()->pin_joint_set_flag(get_rid(), PhysicsServer2D::PIN_JOINT_FLAG_MOTOR_ENABLED, motor_enabled);
	}
}

bool PinJoint2D::is_motor_enabled() const {
	return motor_enabled;
}

void PinJoint2D::set_angular_limit_enabled(bool p_angular_limit_enabled) {
	if (angular_limit_enabled == p_angular_limit_enabled) {
		return;
	}
	angular_limit_enabled = p_angular_limit_enabled;
	queue_redraw();
	if (is_configured()) {
		PhysicsServer2D::get_singleton()->pin_joint_set_flag(get_rid(), PhysicsServer2D::PIN_JOINT_FLAG_ANGULAR_LIMIT_ENABLED, angular_limit_enabled);
	}
}

bool PinJoint2D::is_angular_limit_enabled() const {
	return angular_limit_enabled;
}

void PinJoint2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_softness", "softness"), &PinJoint2D::set_softness);
	ClassDB::bind_method(D_METHOD("get_softness"), &PinJoint2D::get_softness);
	ClassDB::bind_method(D_METHOD("set_angular_limit_lower", "angular_limit_lower"), &PinJoint2D::set_angular_limit_lower);
	ClassDB::bind_method(D_METHOD("get_angular_limit_lower"), &PinJoint2D::get_angular_limit_lower);
	ClassDB::bind_method(D_METHOD("set_angular_limit_upper", "angular_limit_upper"), &PinJoint2D::set_angular_limit_upper);
	ClassDB::bind_method(D_METHOD("get_angular_limit_upper"), &PinJoint2D::get_angular_limit_upper);
	ClassDB::bind_method(D_METHOD("set_motor_target_velocity", "motor_target_velocity"), &PinJoint2D::set_motor_target_velocity);
	ClassDB::bind_method(D_METHOD("get_motor_target_velocity"), &PinJoint2D::get_motor_target_velocity);
	ClassDB::bind_method(D_METHOD("set_motor_enabled", "enabled"), &PinJoint2D::set_motor_enabled);
	ClassDB::bind_method(D_METHOD("is_motor_enabled"), &PinJoint2D::is_motor_enabled);
	ClassDB::bind_method(D_METHOD("set_angular_limit_enabled", "enabled"), &PinJoint2D::set_angular_limit_enabled);
	ClassDB::bind_method(D_METHOD("is_angular_limit_enabled"), &PinJoint2D::is_angular_limit_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "softness", PROPERTY_HINT_RANGE, "0.00,16,0.01,exp"), "set_softness", "get_softness");
	ADD_GROUP("Angular Limit", "angular_limit_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "angular_limit_enabled"), "set_angular_limit_enabled", "is_angular_limit_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_limit_lower", PROPERTY_HINT_RANGE, "-180,180,0.1,radians_as_degrees"), "set_angular_limit_lower", "get_angular_limit_lower");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_limit_upper", PROPERTY_HINT_RANGE, "-180,180,0.1,radians_as_degrees"), "set_angular_limit_upper", "get_angular_limit_upper");
	ADD_GROUP("Motor", "motor_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "motor_enabled"), "set_motor_enabled", "is_motor_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "motor_target_velocity", PROPERTY_HINT_RANGE, U"-200,200,0.01,or_greater,or_less,radians_as_degrees,suffix:\u00B0/s"), "set_motor_target_velocity", "get_motor_target_velocity");
}

PinJoint2D::PinJoint2D() {
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void GrooveJoint2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			if (!is_inside_tree()) {
				break;
			}

			if (!Engine::get_singleton()->is_editor_hint() && !get_tree()->is_debugging_collisions_hint()) {
				break;
			}

			draw_line(Point2(-10, 0), Point2(+10, 0), Color(0.7, 0.6, 0.0, 0.5), 3);
			draw_line(Point2(-10, length), Point2(+10, length), Color(0.7, 0.6, 0.0, 0.5), 3);
			draw_line(Point2(0, 0), Point2(0, length), Color(0.7, 0.6, 0.0, 0.5), 3);
			draw_line(Point2(-10, initial_offset), Point2(+10, initial_offset), Color(0.8, 0.8, 0.9, 0.5), 5);
		} break;
	}
}

void GrooveJoint2D::_configure_joint(RID p_joint, PhysicsBody2D *body_a, PhysicsBody2D *body_b) {
	Transform2D gt = get_global_transform();
	Vector2 groove_A1 = gt.get_origin();
	Vector2 groove_A2 = gt.xform(Vector2(0, length));
	Vector2 anchor_B = gt.xform(Vector2(0, initial_offset));

	PhysicsServer2D::get_singleton()->joint_make_groove(p_joint, groove_A1, groove_A2, anchor_B, body_a->get_rid(), body_b->get_rid());
}

void GrooveJoint2D::set_length(real_t p_length) {
	length = p_length;
	queue_redraw();
}

real_t GrooveJoint2D::get_length() const {
	return length;
}

void GrooveJoint2D::set_initial_offset(real_t p_initial_offset) {
	initial_offset = p_initial_offset;
	queue_redraw();
}

real_t GrooveJoint2D::get_initial_offset() const {
	return initial_offset;
}

void GrooveJoint2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_length", "length"), &GrooveJoint2D::set_length);
	ClassDB::bind_method(D_METHOD("get_length"), &GrooveJoint2D::get_length);
	ClassDB::bind_method(D_METHOD("set_initial_offset", "offset"), &GrooveJoint2D::set_initial_offset);
	ClassDB::bind_method(D_METHOD("get_initial_offset"), &GrooveJoint2D::get_initial_offset);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length", PROPERTY_HINT_RANGE, "1,65535,1,exp,suffix:px"), "set_length", "get_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "initial_offset", PROPERTY_HINT_RANGE, "1,65535,1,exp,suffix:px"), "set_initial_offset", "get_initial_offset");
}

GrooveJoint2D::GrooveJoint2D() {
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void DampedSpringJoint2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			if (!is_inside_tree()) {
				break;
			}

			if (!Engine::get_singleton()->is_editor_hint() && !get_tree()->is_debugging_collisions_hint()) {
				break;
			}

			draw_line(Point2(-10, 0), Point2(+10, 0), Color(0.7, 0.6, 0.0, 0.5), 3);
			draw_line(Point2(-10, length), Point2(+10, length), Color(0.7, 0.6, 0.0, 0.5), 3);
			draw_line(Point2(0, 0), Point2(0, length), Color(0.7, 0.6, 0.0, 0.5), 3);
		} break;
	}
}

void DampedSpringJoint2D::_configure_joint(RID p_joint, PhysicsBody2D *body_a, PhysicsBody2D *body_b) {
	Transform2D gt = get_global_transform();
	Vector2 anchor_A = gt.get_origin();
	Vector2 anchor_B = gt.xform(Vector2(0, length));

	PhysicsServer2D::get_singleton()->joint_make_damped_spring(p_joint, anchor_A, anchor_B, body_a->get_rid(), body_b->get_rid());
	if (rest_length) {
		PhysicsServer2D::get_singleton()->damped_spring_joint_set_param(p_joint, PhysicsServer2D::DAMPED_SPRING_REST_LENGTH, rest_length);
	}
	PhysicsServer2D::get_singleton()->damped_spring_joint_set_param(p_joint, PhysicsServer2D::DAMPED_SPRING_STIFFNESS, stiffness);
	PhysicsServer2D::get_singleton()->damped_spring_joint_set_param(p_joint, PhysicsServer2D::DAMPED_SPRING_DAMPING, damping);
}

void DampedSpringJoint2D::set_length(real_t p_length) {
	length = p_length;
	queue_redraw();
}

real_t DampedSpringJoint2D::get_length() const {
	return length;
}

void DampedSpringJoint2D::set_rest_length(real_t p_rest_length) {
	rest_length = p_rest_length;
	queue_redraw();
	if (is_configured()) {
		PhysicsServer2D::get_singleton()->damped_spring_joint_set_param(get_rid(), PhysicsServer2D::DAMPED_SPRING_REST_LENGTH, p_rest_length ? p_rest_length : length);
	}
}

real_t DampedSpringJoint2D::get_rest_length() const {
	return rest_length;
}

void DampedSpringJoint2D::set_stiffness(real_t p_stiffness) {
	stiffness = p_stiffness;
	queue_redraw();
	if (is_configured()) {
		PhysicsServer2D::get_singleton()->damped_spring_joint_set_param(get_rid(), PhysicsServer2D::DAMPED_SPRING_STIFFNESS, p_stiffness);
	}
}

real_t DampedSpringJoint2D::get_stiffness() const {
	return stiffness;
}

void DampedSpringJoint2D::set_damping(real_t p_damping) {
	damping = p_damping;
	queue_redraw();
	if (is_configured()) {
		PhysicsServer2D::get_singleton()->damped_spring_joint_set_param(get_rid(), PhysicsServer2D::DAMPED_SPRING_DAMPING, p_damping);
	}
}

real_t DampedSpringJoint2D::get_damping() const {
	return damping;
}

void DampedSpringJoint2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_length", "length"), &DampedSpringJoint2D::set_length);
	ClassDB::bind_method(D_METHOD("get_length"), &DampedSpringJoint2D::get_length);
	ClassDB::bind_method(D_METHOD("set_rest_length", "rest_length"), &DampedSpringJoint2D::set_rest_length);
	ClassDB::bind_method(D_METHOD("get_rest_length"), &DampedSpringJoint2D::get_rest_length);
	ClassDB::bind_method(D_METHOD("set_stiffness", "stiffness"), &DampedSpringJoint2D::set_stiffness);
	ClassDB::bind_method(D_METHOD("get_stiffness"), &DampedSpringJoint2D::get_stiffness);
	ClassDB::bind_method(D_METHOD("set_damping", "damping"), &DampedSpringJoint2D::set_damping);
	ClassDB::bind_method(D_METHOD("get_damping"), &DampedSpringJoint2D::get_damping);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length", PROPERTY_HINT_RANGE, "1,65535,1,exp,suffix:px"), "set_length", "get_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rest_length", PROPERTY_HINT_RANGE, "0,65535,1,exp,suffix:px"), "set_rest_length", "get_rest_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stiffness", PROPERTY_HINT_RANGE, "0.1,64,0.1,exp"), "set_stiffness", "get_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping", PROPERTY_HINT_RANGE, "0.01,16,0.01,exp"), "set_damping", "get_damping");
}

DampedSpringJoint2D::DampedSpringJoint2D() {
}
