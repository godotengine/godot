/**************************************************************************/
/*  two_bone_ik_2d.cpp                                                    */
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

#include "two_bone_ik_2d.h"

#include "core/object/class_db.h"
#include "scene/2d/skeleton_2d.h"

PackedStringArray TwoBoneIK2D::get_configuration_warnings() const {
	PackedStringArray warnings = IKModifier2D::get_configuration_warnings();
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton) {
		return warnings;
	}
	const int bone_count = skeleton->get_bone_count();
	if (root_bone < 0 || root_bone >= bone_count) {
		warnings.push_back(RTR("Root bone index is out of range."));
	}
	if (middle_bone < 0 || middle_bone >= bone_count) {
		warnings.push_back(RTR("Middle bone index is out of range."));
	}
	if (tip_bone < 0 || tip_bone >= bone_count) {
		warnings.push_back(RTR("Tip bone index is out of range."));
	}
	if (middle_bone >= 0 && root_bone >= 0 && skeleton->get_bone_parent(middle_bone) != root_bone) {
		warnings.push_back(RTR("Middle bone must be a direct child of root bone."));
	}
	if (tip_bone >= 0 && middle_bone >= 0 && skeleton->get_bone_parent(tip_bone) != middle_bone) {
		warnings.push_back(RTR("Tip bone must be a direct child of middle bone."));
	}
	if (target_node.is_empty()) {
		warnings.push_back(RTR("Target node is not set."));
	} else if (!Object::cast_to<Node2D>(get_node_or_null(target_node))) {
		warnings.push_back(RTR("Target node must be a Node2D."));
	}
	if (pole_node.is_empty()) {
		warnings.push_back(RTR("Pole node is not set."));
	} else if (!Object::cast_to<Node2D>(get_node_or_null(pole_node))) {
		warnings.push_back(RTR("Pole node must be a Node2D."));
	}
	return warnings;
}

void TwoBoneIK2D::_process_modification(double p_delta) {
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	const int bone_count = skeleton->get_bone_count();
	if (root_bone < 0 || root_bone >= bone_count || middle_bone < 0 || middle_bone >= bone_count || tip_bone < 0 || tip_bone >= bone_count) {
		return;
	}
	if (skeleton->get_bone_parent(middle_bone) != root_bone || skeleton->get_bone_parent(tip_bone) != middle_bone) {
		return;
	}

	Node2D *target = Object::cast_to<Node2D>(get_node_or_null(target_node));
	Node2D *pole = Object::cast_to<Node2D>(get_node_or_null(pole_node));
	if (!target || !target->is_inside_tree() || !pole || !pole->is_inside_tree()) {
		return;
	}

	const Transform2D skeleton_global_inverse = skeleton->get_global_transform().affine_inverse();
	const Vector2 target_position_raw = skeleton_global_inverse.xform(target->get_global_position());
	const Vector2 pole_position = skeleton_global_inverse.xform(pole->get_global_position());

	const Transform2D root_global_pose = skeleton->get_bone_global_pose(root_bone);
	const Transform2D middle_global_pose = skeleton->get_bone_global_pose(middle_bone);
	const Transform2D tip_global_pose = skeleton->get_bone_global_pose(tip_bone);

	const Vector2 root_position = root_global_pose.get_origin();
	const Vector2 middle_position = middle_global_pose.get_origin();
	const Vector2 tip_position = tip_global_pose.get_origin();

	const Vector2 current_root_vector = middle_position - root_position;
	const Vector2 current_middle_vector = tip_position - middle_position;
	const real_t root_len = current_root_vector.length();
	const real_t middle_len = current_middle_vector.length();
	if (Math::is_zero_approx(root_len) || Math::is_zero_approx(middle_len)) {
		return;
	}

	Vector2 target_position = target_position_raw;
	Vector2 root_to_target = target_position - root_position;
	real_t target_distance = root_to_target.length();
	if (Math::is_zero_approx(target_distance)) {
		return;
	}
	if (target_minimum_distance > 0.0 && target_distance < target_minimum_distance) {
		target_position = root_position + root_to_target.normalized() * target_minimum_distance;
		root_to_target = target_position - root_position;
		target_distance = target_minimum_distance;
	}
	if (target_maximum_distance > 0.0 && target_distance > target_maximum_distance) {
		target_position = root_position + root_to_target.normalized() * target_maximum_distance;
		root_to_target = target_position - root_position;
		target_distance = target_maximum_distance;
	}

	const Vector2 root_to_target_direction = root_to_target / target_distance;
	const real_t max_reach = root_len + middle_len;
	const real_t min_reach = Math::abs(root_len - middle_len);
	Vector2 solved_middle_position;
	Vector2 solved_tip_position;

	if (target_distance >= max_reach) {
		solved_middle_position = root_position + root_to_target_direction * root_len;
		solved_tip_position = solved_middle_position + root_to_target_direction * middle_len;
	} else {
		if (target_distance < min_reach) {
			target_position = root_position + root_to_target_direction * min_reach;
			root_to_target = target_position - root_position;
			target_distance = min_reach;
		}

		solved_tip_position = target_position;
		const Vector2 chain_direction = root_to_target / target_distance;
		const Vector2 pole_direction = Vector2(-chain_direction.y, chain_direction.x);
		const real_t a = (target_distance * target_distance + root_len * root_len - middle_len * middle_len) / (2.0 * target_distance);
		real_t h_sq = root_len * root_len - a * a;
		if (h_sq < 0.0) {
			h_sq = 0.0;
		}
		const real_t h = Math::sqrt(h_sq);
		const Vector2 det_plus = root_position + chain_direction * a + pole_direction * h;
		const Vector2 det_minus = root_position + chain_direction * a - pole_direction * h;
		const bool plus_is_closer = pole_position.distance_squared_to(det_plus) < pole_position.distance_squared_to(det_minus);
		solved_middle_position = (plus_is_closer != flip_bend_direction) ? det_plus : det_minus;
	}

	Vector2 desired_root_vector = solved_middle_position - root_position;
	if (current_root_vector.is_zero_approx() || desired_root_vector.is_zero_approx()) {
		return;
	}

	Transform2D root_destination = skeleton->get_bone_pose(root_bone);
	root_destination.set_rotation(root_destination.get_rotation() + current_root_vector.angle_to(desired_root_vector));
	skeleton->set_bone_pose(root_bone, root_destination);

	const Vector2 updated_middle_position = skeleton->get_bone_global_pose(middle_bone).get_origin();
	const Vector2 updated_tip_position = skeleton->get_bone_global_pose(tip_bone).get_origin();
	const Vector2 updated_middle_vector = updated_tip_position - updated_middle_position;
	const Vector2 desired_middle_vector = solved_tip_position - updated_middle_position;
	if (updated_middle_vector.is_zero_approx() || desired_middle_vector.is_zero_approx()) {
		return;
	}

	Transform2D middle_destination = skeleton->get_bone_pose(middle_bone);
	middle_destination.set_rotation(middle_destination.get_rotation() + updated_middle_vector.angle_to(desired_middle_vector));
	skeleton->set_bone_pose(middle_bone, middle_destination);
}

void TwoBoneIK2D::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_configuration_warnings();
}
NodePath TwoBoneIK2D::get_target_node() const {
	return target_node;
}
void TwoBoneIK2D::set_pole_node(const NodePath &p_pole_node) {
	pole_node = p_pole_node;
	update_configuration_warnings();
}
NodePath TwoBoneIK2D::get_pole_node() const {
	return pole_node;
}
void TwoBoneIK2D::set_root_bone(int p_bone) {
	root_bone = p_bone;
	update_configuration_warnings();
}
int TwoBoneIK2D::get_root_bone() const {
	return root_bone;
}
void TwoBoneIK2D::set_middle_bone(int p_bone) {
	middle_bone = p_bone;
	update_configuration_warnings();
}
int TwoBoneIK2D::get_middle_bone() const {
	return middle_bone;
}
void TwoBoneIK2D::set_tip_bone(int p_bone) {
	tip_bone = p_bone;
	update_configuration_warnings();
}
int TwoBoneIK2D::get_tip_bone() const {
	return tip_bone;
}
void TwoBoneIK2D::set_flip_bend_direction(bool p_flip) {
	flip_bend_direction = p_flip;
}
bool TwoBoneIK2D::is_bend_direction_flipped() const {
	return flip_bend_direction;
}
void TwoBoneIK2D::set_target_minimum_distance(real_t p_distance) {
	target_minimum_distance = MAX((real_t)0.0, p_distance);
}
real_t TwoBoneIK2D::get_target_minimum_distance() const {
	return target_minimum_distance;
}
void TwoBoneIK2D::set_target_maximum_distance(real_t p_distance) {
	target_maximum_distance = MAX((real_t)0.0, p_distance);
}
real_t TwoBoneIK2D::get_target_maximum_distance() const {
	return target_maximum_distance;
}

void TwoBoneIK2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_node"), &TwoBoneIK2D::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &TwoBoneIK2D::get_target_node);
	ClassDB::bind_method(D_METHOD("set_pole_node", "pole_node"), &TwoBoneIK2D::set_pole_node);
	ClassDB::bind_method(D_METHOD("get_pole_node"), &TwoBoneIK2D::get_pole_node);
	ClassDB::bind_method(D_METHOD("set_root_bone", "root_bone"), &TwoBoneIK2D::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone"), &TwoBoneIK2D::get_root_bone);
	ClassDB::bind_method(D_METHOD("set_middle_bone", "middle_bone"), &TwoBoneIK2D::set_middle_bone);
	ClassDB::bind_method(D_METHOD("get_middle_bone"), &TwoBoneIK2D::get_middle_bone);
	ClassDB::bind_method(D_METHOD("set_tip_bone", "tip_bone"), &TwoBoneIK2D::set_tip_bone);
	ClassDB::bind_method(D_METHOD("get_tip_bone"), &TwoBoneIK2D::get_tip_bone);
	ClassDB::bind_method(D_METHOD("set_flip_bend_direction", "flip"), &TwoBoneIK2D::set_flip_bend_direction);
	ClassDB::bind_method(D_METHOD("is_bend_direction_flipped"), &TwoBoneIK2D::is_bend_direction_flipped);
	ClassDB::bind_method(D_METHOD("set_target_minimum_distance", "distance"), &TwoBoneIK2D::set_target_minimum_distance);
	ClassDB::bind_method(D_METHOD("get_target_minimum_distance"), &TwoBoneIK2D::get_target_minimum_distance);
	ClassDB::bind_method(D_METHOD("set_target_maximum_distance", "distance"), &TwoBoneIK2D::set_target_maximum_distance);
	ClassDB::bind_method(D_METHOD("get_target_maximum_distance"), &TwoBoneIK2D::get_target_maximum_distance);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "pole_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_pole_node", "get_pole_node");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "root_bone", PROPERTY_HINT_RANGE, "-1,1024,1,or_greater"), "set_root_bone", "get_root_bone");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "middle_bone", PROPERTY_HINT_RANGE, "-1,1024,1,or_greater"), "set_middle_bone", "get_middle_bone");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tip_bone", PROPERTY_HINT_RANGE, "-1,1024,1,or_greater"), "set_tip_bone", "get_tip_bone");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_bend_direction"), "set_flip_bend_direction", "is_bend_direction_flipped");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "target_minimum_distance", PROPERTY_HINT_RANGE, "0,1024,0.01,or_greater,suffix:px"), "set_target_minimum_distance", "get_target_minimum_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "target_maximum_distance", PROPERTY_HINT_RANGE, "0,1024,0.01,or_greater,suffix:px"), "set_target_maximum_distance", "get_target_maximum_distance");
}
