/**************************************************************************/
/*  look_at_modifier_2d.cpp                                               */
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

#include "look_at_modifier_2d.h"

#include "core/object/class_db.h"
#include "scene/2d/skeleton_2d.h"

static Vector2 _get_bone_forward_vector(Skeleton2D *p_skeleton, int p_bone, const Transform2D &p_bone_global_pose) {
	const Vector2 bone_position = p_bone_global_pose.get_origin();
	const int bone_count = p_skeleton->get_bone_count();
	for (int i = 0; i < bone_count; i++) {
		if (p_skeleton->get_bone_parent(i) == p_bone) {
			const Vector2 child_vector = p_skeleton->get_bone_global_pose(i).get_origin() - bone_position;
			if (!child_vector.is_zero_approx()) {
				return child_vector;
			}
		}
	}
	return p_bone_global_pose.basis_xform(Vector2(1, 0));
}

real_t LookAtModifier2D::_clamp_angle(real_t p_angle) const {
	real_t min_angle = Math::fposmod(constraint_angle_min, (real_t)Math::TAU);
	real_t max_angle = Math::fposmod(constraint_angle_max, (real_t)Math::TAU);
	real_t angle = Math::fposmod(p_angle, (real_t)Math::TAU);
	if (min_angle > max_angle) {
		SWAP(min_angle, max_angle);
	}
	bool is_beyond_bounds = angle < min_angle || angle > max_angle;
	bool is_within_bounds = angle > min_angle && angle < max_angle;
	if ((!constraint_angle_invert && is_beyond_bounds) || (constraint_angle_invert && is_within_bounds)) {
		Vector2 min_vec = Vector2(Math::cos(min_angle), Math::sin(min_angle));
		Vector2 max_vec = Vector2(Math::cos(max_angle), Math::sin(max_angle));
		Vector2 angle_vec = Vector2(Math::cos(angle), Math::sin(angle));
		angle = angle_vec.distance_squared_to(min_vec) <= angle_vec.distance_squared_to(max_vec) ? min_angle : max_angle;
	}
	return angle > Math::PI ? angle - Math::TAU : angle;
}

PackedStringArray LookAtModifier2D::get_configuration_warnings() const {
	PackedStringArray warnings = SkeletonModifier2D::get_configuration_warnings();
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton) {
		return warnings;
	}
	if (bone < 0 || bone >= skeleton->get_bone_count()) {
		warnings.push_back(RTR("Bone index is out of range."));
	}
	if (target_node.is_empty()) {
		warnings.push_back(RTR("Target node is not set."));
	} else if (!Object::cast_to<Node2D>(get_node_or_null(target_node))) {
		warnings.push_back(RTR("Target node must be a Node2D."));
	}
	return warnings;
}

void LookAtModifier2D::_process_modification(double p_delta) {
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton || bone < 0 || bone >= skeleton->get_bone_count()) {
		return;
	}
	Node2D *target = Object::cast_to<Node2D>(get_node_or_null(target_node));
	if (!target || !target->is_inside_tree()) {
		return;
	}
	Transform2D bone_global_pose = skeleton->get_bone_global_pose(bone);
	Vector2 target_position = skeleton->get_global_transform().affine_inverse().xform(target->get_global_position());
	Vector2 to_target = target_position - bone_global_pose.get_origin();
	Vector2 forward_vector = _get_bone_forward_vector(skeleton, bone, bone_global_pose);
	if (to_target.is_zero_approx() || forward_vector.is_zero_approx()) {
		return;
	}
	real_t parent_rotation = 0.0;
	int parent_bone = skeleton->get_bone_parent(bone);
	if (parent_bone >= 0) {
		parent_rotation = skeleton->get_bone_global_pose(parent_bone).get_rotation();
	}
	Vector2 desired_forward = to_target.rotated(additional_rotation);
	real_t rotation_delta = forward_vector.angle_to(desired_forward);
	Transform2D destination = skeleton->get_bone_pose(bone);
	real_t desired_local_rotation = destination.get_rotation() + rotation_delta;
	real_t desired_global_rotation = bone_global_pose.get_rotation() + rotation_delta;
	if (enable_constraint) {
		if (constraint_in_localspace) {
			desired_local_rotation = _clamp_angle(desired_local_rotation);
		} else {
			desired_global_rotation = _clamp_angle(desired_global_rotation);
			desired_local_rotation = desired_global_rotation - parent_rotation;
		}
	}
	destination.set_rotation(desired_local_rotation);
	skeleton->set_bone_pose(bone, destination);
}

void LookAtModifier2D::set_bone(int p_bone) {
	bone = p_bone;
	update_configuration_warnings();
}
int LookAtModifier2D::get_bone() const {
	return bone;
}
void LookAtModifier2D::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_configuration_warnings();
}
NodePath LookAtModifier2D::get_target_node() const {
	return target_node;
}
void LookAtModifier2D::set_additional_rotation(real_t p_rotation) {
	additional_rotation = p_rotation;
}
real_t LookAtModifier2D::get_additional_rotation() const {
	return additional_rotation;
}
void LookAtModifier2D::set_enable_constraint(bool p_enabled) {
	enable_constraint = p_enabled;
}
bool LookAtModifier2D::is_constraint_enabled() const {
	return enable_constraint;
}
void LookAtModifier2D::set_constraint_in_localspace(bool p_enabled) {
	constraint_in_localspace = p_enabled;
}
bool LookAtModifier2D::is_constraint_in_localspace() const {
	return constraint_in_localspace;
}
void LookAtModifier2D::set_constraint_angle_min(real_t p_angle) {
	constraint_angle_min = p_angle;
}
real_t LookAtModifier2D::get_constraint_angle_min() const {
	return constraint_angle_min;
}
void LookAtModifier2D::set_constraint_angle_max(real_t p_angle) {
	constraint_angle_max = p_angle;
}
real_t LookAtModifier2D::get_constraint_angle_max() const {
	return constraint_angle_max;
}
void LookAtModifier2D::set_constraint_angle_invert(bool p_invert) {
	constraint_angle_invert = p_invert;
}
bool LookAtModifier2D::is_constraint_angle_inverted() const {
	return constraint_angle_invert;
}

void LookAtModifier2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bone", "bone"), &LookAtModifier2D::set_bone);
	ClassDB::bind_method(D_METHOD("get_bone"), &LookAtModifier2D::get_bone);
	ClassDB::bind_method(D_METHOD("set_target_node", "target_node"), &LookAtModifier2D::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &LookAtModifier2D::get_target_node);
	ClassDB::bind_method(D_METHOD("set_additional_rotation", "rotation"), &LookAtModifier2D::set_additional_rotation);
	ClassDB::bind_method(D_METHOD("get_additional_rotation"), &LookAtModifier2D::get_additional_rotation);
	ClassDB::bind_method(D_METHOD("set_enable_constraint", "enabled"), &LookAtModifier2D::set_enable_constraint);
	ClassDB::bind_method(D_METHOD("is_constraint_enabled"), &LookAtModifier2D::is_constraint_enabled);
	ClassDB::bind_method(D_METHOD("set_constraint_in_localspace", "enabled"), &LookAtModifier2D::set_constraint_in_localspace);
	ClassDB::bind_method(D_METHOD("is_constraint_in_localspace"), &LookAtModifier2D::is_constraint_in_localspace);
	ClassDB::bind_method(D_METHOD("set_constraint_angle_min", "angle"), &LookAtModifier2D::set_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("get_constraint_angle_min"), &LookAtModifier2D::get_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("set_constraint_angle_max", "angle"), &LookAtModifier2D::set_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("get_constraint_angle_max"), &LookAtModifier2D::get_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("set_constraint_angle_invert", "invert"), &LookAtModifier2D::set_constraint_angle_invert);
	ClassDB::bind_method(D_METHOD("is_constraint_angle_inverted"), &LookAtModifier2D::is_constraint_angle_inverted);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone", PROPERTY_HINT_RANGE, "-1,1024,1,or_greater"), "set_bone", "get_bone");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "additional_rotation", PROPERTY_HINT_RANGE, "-360,360,0.01,radians_as_degrees"), "set_additional_rotation", "get_additional_rotation");
	ADD_GROUP("Constraint", "constraint_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "constraint_enabled"), "set_enable_constraint", "is_constraint_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "constraint_in_localspace"), "set_constraint_in_localspace", "is_constraint_in_localspace");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "constraint_angle_min", PROPERTY_HINT_RANGE, "-360,360,0.01,radians_as_degrees"), "set_constraint_angle_min", "get_constraint_angle_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "constraint_angle_max", PROPERTY_HINT_RANGE, "-360,360,0.01,radians_as_degrees"), "set_constraint_angle_max", "get_constraint_angle_max");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "constraint_angle_invert"), "set_constraint_angle_invert", "is_constraint_angle_inverted");
}
