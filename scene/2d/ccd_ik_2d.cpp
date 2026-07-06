/**************************************************************************/
/*  ccd_ik_2d.cpp                                                         */
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

#include "ccd_ik_2d.h"

#include "core/object/class_db.h"
#include "core/templates/local_vector.h"
#include "scene/2d/skeleton_2d.h"

PackedStringArray CCDIK2D::get_configuration_warnings() const {
	PackedStringArray warnings = IterateIK2D::get_configuration_warnings();
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton) {
		return warnings;
	}
	if (tip_bone < 0 || tip_bone >= skeleton->get_bone_count()) {
		warnings.push_back(RTR("Tip bone index is out of range."));
	}
	if (root_bone >= skeleton->get_bone_count()) {
		warnings.push_back(RTR("Root bone index is out of range."));
	}
	if (target_node.is_empty()) {
		warnings.push_back(RTR("Target node is not set."));
	} else if (!Object::cast_to<Node2D>(get_node_or_null(target_node))) {
		warnings.push_back(RTR("Target node must be a Node2D."));
	}
	return warnings;
}

void CCDIK2D::_process_modification(double p_delta) {
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton || tip_bone < 0 || tip_bone >= skeleton->get_bone_count()) {
		return;
	}
	Node2D *target = Object::cast_to<Node2D>(get_node_or_null(target_node));
	if (!target || !target->is_inside_tree()) {
		return;
	}

	LocalVector<int> chain;
	for (int bone = tip_bone; bone >= 0; bone = skeleton->get_bone_parent(bone)) {
		chain.push_back(bone);
		if (bone == root_bone) {
			break;
		}
	}
	if (chain.is_empty() || (root_bone >= 0 && chain[chain.size() - 1] != root_bone)) {
		return;
	}
	chain.reverse();

	const Vector2 target_position = skeleton->get_global_transform().affine_inverse().xform(target->get_global_position());
	const real_t tolerance_sq = tolerance * tolerance;
	const int joint_count = (int)chain.size() - 1;

	for (int iteration = 0; iteration < max_iterations; iteration++) {
		Vector2 tip_position = skeleton->get_bone_global_pose(tip_bone).get_origin();
		if (tip_position.distance_squared_to(target_position) <= tolerance_sq) {
			break;
		}
		for (int ancestor = joint_count - 1; ancestor >= 0; ancestor--) {
			for (int i = ancestor; i < joint_count; i++) {
				const int bone = chain[i];
				const Vector2 joint_position = skeleton->get_bone_global_pose(bone).get_origin();
				tip_position = skeleton->get_bone_global_pose(tip_bone).get_origin();
				Vector2 joint_to_tip = tip_position - joint_position;
				Vector2 joint_to_target = target_position - joint_position;
				if (joint_to_tip.is_zero_approx() || joint_to_target.is_zero_approx()) {
					continue;
				}
				Transform2D pose = skeleton->get_bone_pose(bone);
				pose.set_rotation(pose.get_rotation() + joint_to_tip.angle_to(joint_to_target));
				skeleton->set_bone_pose(bone, pose);
			}
		}
	}
}

void CCDIK2D::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_configuration_warnings();
}
NodePath CCDIK2D::get_target_node() const {
	return target_node;
}
void CCDIK2D::set_root_bone(int p_bone) {
	root_bone = p_bone;
	update_configuration_warnings();
}
int CCDIK2D::get_root_bone() const {
	return root_bone;
}
void CCDIK2D::set_tip_bone(int p_bone) {
	tip_bone = p_bone;
	update_configuration_warnings();
}
int CCDIK2D::get_tip_bone() const {
	return tip_bone;
}
void CCDIK2D::set_max_iterations(int p_iterations) {
	max_iterations = MAX(1, p_iterations);
}
int CCDIK2D::get_max_iterations() const {
	return max_iterations;
}
void CCDIK2D::set_tolerance(real_t p_tolerance) {
	tolerance = MAX((real_t)0.0, p_tolerance);
}
real_t CCDIK2D::get_tolerance() const {
	return tolerance;
}

void CCDIK2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_node"), &CCDIK2D::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &CCDIK2D::get_target_node);
	ClassDB::bind_method(D_METHOD("set_root_bone", "root_bone"), &CCDIK2D::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone"), &CCDIK2D::get_root_bone);
	ClassDB::bind_method(D_METHOD("set_tip_bone", "tip_bone"), &CCDIK2D::set_tip_bone);
	ClassDB::bind_method(D_METHOD("get_tip_bone"), &CCDIK2D::get_tip_bone);
	ClassDB::bind_method(D_METHOD("set_max_iterations", "max_iterations"), &CCDIK2D::set_max_iterations);
	ClassDB::bind_method(D_METHOD("get_max_iterations"), &CCDIK2D::get_max_iterations);
	ClassDB::bind_method(D_METHOD("set_tolerance", "tolerance"), &CCDIK2D::set_tolerance);
	ClassDB::bind_method(D_METHOD("get_tolerance"), &CCDIK2D::get_tolerance);
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "root_bone", PROPERTY_HINT_RANGE, "-1,1024,1,or_greater"), "set_root_bone", "get_root_bone");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tip_bone", PROPERTY_HINT_RANGE, "-1,1024,1,or_greater"), "set_tip_bone", "get_tip_bone");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_iterations", PROPERTY_HINT_RANGE, "1,128,1,or_greater"), "set_max_iterations", "get_max_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tolerance", PROPERTY_HINT_RANGE, "0,1024,0.01,or_greater,suffix:px"), "set_tolerance", "get_tolerance");
}
