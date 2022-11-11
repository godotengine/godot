/*************************************************************************/
/*  skeleton_modification_3d_fabrik.cpp                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "scene/3d/skeleton_modification_3d_fabrik.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/skeleton_modification_3d.h"

bool SkeletonModification3DFABRIK::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_")) {
		int fabrik_data_size = fabrik_data_chain.size();
		int which = path.get_slicec('/', 0).substr(6).to_int();
		String what = path.get_slicec('/', 1);
		ERR_FAIL_INDEX_V(which, fabrik_data_size, false);

		if (what == "bone") {
			set_joint_bone(which, p_value);
		} else if (what == "length") {
			set_joint_length(which, p_value);
		} else if (what == "magnet_position") {
			set_joint_magnet(which, p_value);
		} else if (what == "auto_calculate_length") {
			set_joint_auto_calculate_length(which, p_value);
		} else if (what == "tip_node") {
			set_joint_tip_node(which, p_value);
		} else if (what == "tip_bone") {
			set_joint_tip_bone(which, p_value);
		} else if (what == "use_target_basis") {
			set_joint_use_target_basis(which, p_value);
		} else if (what == "roll") {
			set_joint_roll(which, real_t(p_value));
		}
		return true;
	}
	return true;
}

bool SkeletonModification3DFABRIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_")) {
		const int fabrik_data_size = fabrik_data_chain.size();
		int which = path.get_slicec('/', 0).substr(6).to_int();
		String what = path.get_slicec('/', 1);
		ERR_FAIL_INDEX_V(which, fabrik_data_size, false);

		if (what == "bone") {
			r_ret = get_joint_bone(which);
		} else if (what == "length") {
			r_ret = get_joint_length(which);
		} else if (what == "magnet_position") {
			r_ret = get_joint_magnet(which);
		} else if (what == "auto_calculate_length") {
			r_ret = get_joint_auto_calculate_length(which);
		} else if (what == "tip_node") {
			r_ret = get_joint_tip_node(which);
		} else if (what == "tip_bone") {
			r_ret = get_joint_tip_bone(which);
		} else if (what == "use_target_basis") {
			r_ret = get_joint_use_target_basis(which);
		} else if (what == "roll") {
			r_ret = get_joint_roll(which);
		}
		return true;
	}
	return true;
}

void SkeletonModification3DFABRIK::_get_property_list(List<PropertyInfo> *p_list) const {
	for (uint32_t i = 0; i < fabrik_data_chain.size(); i++) {
		String base_string = "joint_" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::STRING, base_string + "bone", get_skeleton() ? PROPERTY_HINT_ENUM : PROPERTY_HINT_NONE, get_bone_name_list(), PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "roll", PROPERTY_HINT_RANGE, "-360,360,0.01,radians", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "auto_calculate_length", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));

		if (!fabrik_data_chain[i].auto_calculate_length) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "length", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		} else {
			p_list->push_back(PropertyInfo(Variant::NODE_PATH, base_string + "tip_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::STRING, base_string + "tip_bone", get_skeleton() ? PROPERTY_HINT_ENUM : PROPERTY_HINT_NONE, get_bone_name_list(), PROPERTY_USAGE_DEFAULT));
		}

		// Cannot apply magnet to the origin of the chain, as it will not do anything.
		if (i > 0) {
			p_list->push_back(PropertyInfo(Variant::VECTOR3, base_string + "magnet_position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
		// Only give the override basis option on the last bone in the chain, so only include it for the last bone.
		if (i == fabrik_data_chain.size() - 1) {
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "use_target_basis", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
	}
}

void SkeletonModification3DFABRIK::chain_backwards() {
	int final_bone_idx = fabrik_data_chain[final_joint_idx].bone_idx;
	Transform3D final_joint_trans = fabrik_transforms[final_joint_idx];
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	// Get the direction the final bone is facing in.
	Vector3 forward_vector = get_bone_rest_forward_vector(final_bone_idx);
	Transform3D final_bone_direction_trans = final_joint_trans.looking_at(target_global_pose.origin, Vector3(0, 1, 0));
	final_bone_direction_trans.basis = global_pose_z_forward_to_bone_forward(forward_vector, final_bone_direction_trans.basis);
	Vector3 direction = final_bone_direction_trans.basis.xform(forward_vector).normalized();

	// If set to override, then use the target's Basis rather than the bone's
	if (fabrik_data_chain[final_joint_idx].use_target_basis) {
		direction = target_global_pose.basis.xform(forward_vector).normalized();
	}

	// set the position of the final joint to the target position
	final_joint_trans.origin = target_global_pose.origin - (direction * fabrik_data_chain[final_joint_idx].length);
	fabrik_transforms[final_joint_idx] = final_joint_trans;

	// for all other joints, move them towards the target
	int i = final_joint_idx;
	while (i >= 1) {
		Transform3D next_bone_trans = fabrik_transforms[i];
		i -= 1;
		Transform3D current_trans = fabrik_transforms[i];

		real_t length = fabrik_data_chain[i].length / (current_trans.origin.distance_to(next_bone_trans.origin));
		current_trans.origin = next_bone_trans.origin.lerp(current_trans.origin, length);

		// Save the result
		fabrik_transforms[i] = current_trans;
	}
}

void SkeletonModification3DFABRIK::chain_forwards() {
	// Set root at the initial position.
	Transform3D root_transform = fabrik_transforms[0];

	root_transform.origin = origin_global_pose.origin;
	fabrik_transforms[0] = origin_global_pose;

	for (uint32_t i = 0; i < fabrik_data_chain.size() - 1; i++) {
		Transform3D current_trans = fabrik_transforms[i];
		Transform3D next_bone_trans = fabrik_transforms[i + 1];

		real_t length = fabrik_data_chain[i].length / (next_bone_trans.origin.distance_to(current_trans.origin));
		next_bone_trans.origin = current_trans.origin.lerp(next_bone_trans.origin, length);

		// Save the result
		fabrik_transforms[i + 1] = next_bone_trans;
	}
}

void SkeletonModification3DFABRIK::chain_apply() {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	int prev_bone_idx = -1;
	Transform3D prev_transform;

	for (uint32_t i = 0; i < fabrik_data_chain.size(); i++) {
		int current_bone_idx = fabrik_data_chain[i].bone_idx;
		if (!_cache_bone(fabrik_data_chain[i].bone_idx, fabrik_data_chain[i].bone_name)) {
			return;
		}
		int bone_idx = fabrik_data_chain[i].bone_idx;
		int par_idx = skeleton->get_bone_parent(bone_idx);
		Transform3D parent_bone_transform;
		bool first = true;
		while (par_idx >= 0 && par_idx != prev_bone_idx) {
			if (first) {
				parent_bone_transform = skeleton->get_bone_pose(par_idx);
				first = false;
			} else {
				parent_bone_transform = skeleton->get_bone_pose(par_idx) * parent_bone_transform;
			}
			par_idx = skeleton->get_bone_parent(par_idx);
		}
		if (i != 0 && par_idx < 0) {
			parent_bone_transform = prev_transform * parent_bone_transform;
		}

		Transform3D current_trans = fabrik_transforms[i];

		// If this is the last bone in the chain...
		if (i == fabrik_data_chain.size() - 1) {
			if (fabrik_data_chain[i].use_target_basis == false) { // Point to target...
				// Get the forward direction that the basis is facing in right now.
				Vector3 forward_vector = get_bone_rest_forward_vector(current_bone_idx);
				// Rotate the bone towards the target:
				current_trans.basis.rotate_to_align(forward_vector, current_trans.origin.direction_to(target_global_pose.origin));
				current_trans.basis.rotate_local(forward_vector, fabrik_data_chain[i].roll);
			} else { // Use the target's Basis...
				current_trans.basis = target_global_pose.basis.orthonormalized().scaled(current_trans.basis.get_scale());
			}
		} else { // every other bone in the chain...
			Transform3D next_trans = fabrik_transforms[i + 1];

			// Get the forward direction that the basis is facing in right now.
			Vector3 forward_vector = get_bone_rest_forward_vector(current_bone_idx);
			// Rotate the bone towards the next bone in the chain:
			current_trans.basis.rotate_to_align(forward_vector, current_trans.origin.direction_to(next_trans.origin));
			current_trans.basis.rotate_local(forward_vector, fabrik_data_chain[i].roll);
		}
		Transform3D new_bone_trans_local = parent_bone_transform.affine_inverse() * current_trans;
		skeleton->set_bone_pose_rotation(current_bone_idx, new_bone_trans_local.basis.get_rotation_quaternion());

		prev_transform = parent_bone_transform * new_bone_trans_local;
		prev_bone_idx = bone_idx;
	}
}

void SkeletonModification3DFABRIK::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	if (!target_node.is_empty()) {
		target_bone = String();
	}
	target_cache = Variant();
}

NodePath SkeletonModification3DFABRIK::get_target_node() const {
	return target_node;
}

void SkeletonModification3DFABRIK::set_target_bone(const String &p_target_bone) {
	target_bone = p_target_bone;
	if (!target_bone.is_empty()) {
		target_node = NodePath();
	}
	target_cache = Variant();
}

String SkeletonModification3DFABRIK::get_target_bone() const {
	return target_bone;
}

int SkeletonModification3DFABRIK::get_joint_count() {
	return fabrik_data_chain.size();
}

void SkeletonModification3DFABRIK::set_joint_count(int p_length) {
	ERR_FAIL_COND(p_length < 0);
	fabrik_data_chain.resize(p_length);
	fabrik_transforms.resize(p_length);
	notify_property_list_changed();
}

real_t SkeletonModification3DFABRIK::get_chain_tolerance() {
	return chain_tolerance;
}

void SkeletonModification3DFABRIK::set_chain_tolerance(real_t p_tolerance) {
	ERR_FAIL_COND_MSG(p_tolerance <= 0, "FABRIK chain tolerance must be more than zero!");
	chain_tolerance = p_tolerance;
}

int SkeletonModification3DFABRIK::get_chain_max_iterations() {
	return chain_max_iterations;
}
void SkeletonModification3DFABRIK::set_chain_max_iterations(int p_iterations) {
	ERR_FAIL_COND_MSG(p_iterations <= 0, "FABRIK chain iterations must be at least one. Set enabled to false to disable the FABRIK chain.");
	chain_max_iterations = p_iterations;
}

// FABRIK joint data functions
String SkeletonModification3DFABRIK::get_joint_bone(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, String());
	return fabrik_data_chain[p_joint_idx].bone_name;
}

void SkeletonModification3DFABRIK::set_joint_bone(int p_joint_idx, String p_bone_name) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].bone_name = p_bone_name;
	fabrik_data_chain[p_joint_idx].bone_idx = UNCACHED_BONE_IDX;
}

real_t SkeletonModification3DFABRIK::get_joint_length(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, -1);
	return fabrik_data_chain[p_joint_idx].length;
}

void SkeletonModification3DFABRIK::set_joint_length(int p_joint_idx, real_t p_bone_length) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ERR_FAIL_COND_MSG(p_bone_length < 0, "FABRIK joint length cannot be less than zero!");
	fabrik_data_chain[p_joint_idx].length = p_bone_length;
}

Vector3 SkeletonModification3DFABRIK::get_joint_magnet(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, Vector3());
	return fabrik_data_chain[p_joint_idx].magnet_position;
}

void SkeletonModification3DFABRIK::set_joint_magnet(int p_joint_idx, Vector3 p_magnet) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].magnet_position = p_magnet;
}

bool SkeletonModification3DFABRIK::get_joint_auto_calculate_length(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return fabrik_data_chain[p_joint_idx].auto_calculate_length;
}

void SkeletonModification3DFABRIK::set_joint_auto_calculate_length(int p_joint_idx, bool p_auto_calculate) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].auto_calculate_length = p_auto_calculate;
	if (get_skeleton()) {
		calculate_fabrik_joint_length(p_joint_idx);
	}
	notify_property_list_changed();
}

void SkeletonModification3DFABRIK::calculate_fabrik_joint_length(int p_joint_idx) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	if (!fabrik_data_chain[p_joint_idx].auto_calculate_length) {
		return;
	}

	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		WARN_PRINT_ONCE("Cannot auto calculate joint length: modification is not properly setup!");
		return;
	}
	ERR_FAIL_INDEX_MSG(fabrik_data_chain[p_joint_idx].bone_idx, skeleton->get_bone_count(),
			"Bone for joint " + itos(p_joint_idx) + " is not set or points to an unknown bone!");

	Transform3D bone_trans = get_bone_transform(fabrik_data_chain[p_joint_idx].bone_idx);

	if (fabrik_data_chain[p_joint_idx].tip_node != NodePath() || fabrik_data_chain[p_joint_idx].tip_bone != String()) { // Use the tip node to update joint length.

		if (!_cache_target(fabrik_data_chain[p_joint_idx].tip_cache, fabrik_data_chain[p_joint_idx].tip_node, fabrik_data_chain[p_joint_idx].tip_bone)) {
			ERR_FAIL_MSG("Tip node for joint " + itos(p_joint_idx) + "is not a Node3D-based node. Cannot calculate length...");
		}

		Transform3D node_trans = get_target_transform(fabrik_data_chain[p_joint_idx].tip_cache);
		fabrik_data_chain[p_joint_idx].length = bone_trans.get_origin().distance_to(node_trans.origin);

	} else { // Use child bone(s) to update joint length, if possible
		Vector<int> bone_children = skeleton->get_bone_children(fabrik_data_chain[p_joint_idx].bone_idx);
		if (bone_children.size() <= 0) {
			ERR_FAIL_MSG("Cannot calculate length for joint " + itos(p_joint_idx) + "joint uses leaf bone. \nPlease manually set the bone length or use a tip node!");
			return;
		}

		real_t final_length = 0;
		for (int i = 0; i < bone_children.size(); i++) {
			final_length += bone_trans.get_basis().xform(skeleton->get_bone_pose_position(bone_children[i])).length();
		}
		fabrik_data_chain[p_joint_idx].length = final_length / bone_children.size();
	}
	notify_property_list_changed();
}

NodePath SkeletonModification3DFABRIK::get_joint_tip_node(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, NodePath());
	return fabrik_data_chain[p_joint_idx].tip_node;
}

void SkeletonModification3DFABRIK::set_joint_tip_node(int p_joint_idx, NodePath p_tip_node) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].tip_node = p_tip_node;
	if (!p_tip_node.is_empty()) {
		fabrik_data_chain[p_joint_idx].tip_bone = String();
	}
	fabrik_data_chain[p_joint_idx].tip_cache = Variant();
}

String SkeletonModification3DFABRIK::get_joint_tip_bone(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, String());
	return fabrik_data_chain[p_joint_idx].tip_bone;
}

void SkeletonModification3DFABRIK::set_joint_tip_bone(int p_joint_idx, String p_tip_bone) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].tip_bone = p_tip_bone;
	if (!p_tip_bone.is_empty()) {
		fabrik_data_chain[p_joint_idx].tip_node = String();
	}
	fabrik_data_chain[p_joint_idx].tip_cache = Variant();
}

bool SkeletonModification3DFABRIK::get_joint_use_target_basis(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return fabrik_data_chain[p_joint_idx].use_target_basis;
}

void SkeletonModification3DFABRIK::set_joint_use_target_basis(int p_joint_idx, bool p_use_target_basis) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].use_target_basis = p_use_target_basis;
}

real_t SkeletonModification3DFABRIK::get_joint_roll(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, 0.0);
	return fabrik_data_chain[p_joint_idx].roll;
}

void SkeletonModification3DFABRIK::set_joint_roll(int p_joint_idx, real_t p_roll) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].roll = p_roll;
}

void SkeletonModification3DFABRIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification3DFABRIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification3DFABRIK::get_target_node);
	ClassDB::bind_method(D_METHOD("set_target_bone", "target_bone_name"), &SkeletonModification3DFABRIK::set_target_bone);
	ClassDB::bind_method(D_METHOD("get_target_bone"), &SkeletonModification3DFABRIK::get_target_bone);
	ClassDB::bind_method(D_METHOD("set_joint_count", "fabrik_chain_length"), &SkeletonModification3DFABRIK::set_joint_count);
	ClassDB::bind_method(D_METHOD("get_joint_count"), &SkeletonModification3DFABRIK::get_joint_count);
	ClassDB::bind_method(D_METHOD("set_chain_tolerance", "tolerance"), &SkeletonModification3DFABRIK::set_chain_tolerance);
	ClassDB::bind_method(D_METHOD("get_chain_tolerance"), &SkeletonModification3DFABRIK::get_chain_tolerance);
	ClassDB::bind_method(D_METHOD("set_chain_max_iterations", "max_iterations"), &SkeletonModification3DFABRIK::set_chain_max_iterations);
	ClassDB::bind_method(D_METHOD("get_chain_max_iterations"), &SkeletonModification3DFABRIK::get_chain_max_iterations);

	// FABRIK joint data functions
	ClassDB::bind_method(D_METHOD("get_joint_bone", "joint_idx"), &SkeletonModification3DFABRIK::get_joint_bone);
	ClassDB::bind_method(D_METHOD("set_joint_bone", "joint_idx", "bone_name"), &SkeletonModification3DFABRIK::set_joint_bone);
	ClassDB::bind_method(D_METHOD("get_joint_length", "joint_idx"), &SkeletonModification3DFABRIK::get_joint_length);
	ClassDB::bind_method(D_METHOD("set_joint_length", "joint_idx", "length"), &SkeletonModification3DFABRIK::set_joint_length);
	ClassDB::bind_method(D_METHOD("get_joint_magnet", "joint_idx"), &SkeletonModification3DFABRIK::get_joint_magnet);
	ClassDB::bind_method(D_METHOD("set_joint_magnet", "joint_idx", "magnet_position"), &SkeletonModification3DFABRIK::set_joint_magnet);
	ClassDB::bind_method(D_METHOD("get_joint_auto_calculate_length", "joint_idx"), &SkeletonModification3DFABRIK::get_joint_auto_calculate_length);
	ClassDB::bind_method(D_METHOD("set_joint_auto_calculate_length", "joint_idx", "auto_calculate_length"), &SkeletonModification3DFABRIK::set_joint_auto_calculate_length);
	ClassDB::bind_method(D_METHOD("calculate_fabrik_joint_length", "joint_idx"), &SkeletonModification3DFABRIK::calculate_fabrik_joint_length);
	ClassDB::bind_method(D_METHOD("get_joint_tip_node", "joint_idx"), &SkeletonModification3DFABRIK::get_joint_tip_node);
	ClassDB::bind_method(D_METHOD("set_joint_tip_node", "joint_idx", "tip_node"), &SkeletonModification3DFABRIK::set_joint_tip_node);
	ClassDB::bind_method(D_METHOD("get_joint_tip_bone", "joint_idx"), &SkeletonModification3DFABRIK::get_joint_tip_bone);
	ClassDB::bind_method(D_METHOD("set_joint_tip_bone", "joint_idx", "tip_bone"), &SkeletonModification3DFABRIK::set_joint_tip_bone);
	ClassDB::bind_method(D_METHOD("get_joint_use_target_basis", "joint_idx"), &SkeletonModification3DFABRIK::get_joint_use_target_basis);
	ClassDB::bind_method(D_METHOD("set_joint_use_target_basis", "joint_idx", "use_target_basis"), &SkeletonModification3DFABRIK::set_joint_use_target_basis);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "target_bone", PROPERTY_HINT_NONE, ""), "set_target_bone", "get_target_bone");
	ADD_ARRAY_COUNT("FABRIK Joint Chain", "joint_count", "set_joint_count", "get_joint_count", "joint_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "chain_tolerance", PROPERTY_HINT_RANGE, "0,100,0.001"), "set_chain_tolerance", "get_chain_tolerance");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "chain_max_iterations", PROPERTY_HINT_RANGE, "1,50,1"), "set_chain_max_iterations", "get_chain_max_iterations");
}

SkeletonModification3DFABRIK::SkeletonModification3DFABRIK() {
}

SkeletonModification3DFABRIK::~SkeletonModification3DFABRIK() {
}

void SkeletonModification3DFABRIK::execute(real_t p_delta) {
	SkeletonModification3D::execute(p_delta);

	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton || !_cache_target(target_cache, target_node, target_bone)) {
		WARN_PRINT_ONCE("FABRIK: Unable to resolve target");
		return;
	}

	// Make sure the transform cache is the correct size
	if (fabrik_transforms.size() != fabrik_data_chain.size()) {
		fabrik_transforms.resize(fabrik_data_chain.size());
	}

	Transform3D origin_global_pose_trans;
	int prev_bone_idx = -1;
	Transform3D prev_transform;

	// Verify that all joints have a valid bone ID, and that all bone lengths are zero or more
	// Also, while we are here, apply magnet positions.
	for (uint32_t i = 0; i < fabrik_data_chain.size(); i++) {
		if (!_cache_bone(fabrik_data_chain[i].bone_idx, fabrik_data_chain[i].bone_name)) {
			WARN_PRINT_ONCE("FABRIK: Failed to cache joint bone " + itos(i));
			return;
		}

		if (fabrik_data_chain[i].length < 0 && fabrik_data_chain[i].auto_calculate_length) {
			calculate_fabrik_joint_length(i);
		}
		int par_idx = skeleton->get_bone_parent(fabrik_data_chain[i].bone_idx);
		Transform3D bone_transform = skeleton->get_bone_pose(fabrik_data_chain[i].bone_idx);
		while (par_idx >= 0 && par_idx != prev_bone_idx) {
			bone_transform = skeleton->get_bone_pose(par_idx) * bone_transform;
			par_idx = skeleton->get_bone_parent(par_idx);
		}
		if (i != 0 && par_idx < 0) {
			bone_transform = prev_transform * bone_transform;
		}
		prev_transform = bone_transform;
		prev_bone_idx = fabrik_data_chain[i].bone_idx;

		ERR_FAIL_COND_MSG(fabrik_data_chain[i].length < 0, "FABRIK Joint " + itos(i) + " has an invalid joint length. Cannot execute!");
		fabrik_transforms[i] = bone_transform;
		if (i == 0) {
			origin_global_pose_trans = bone_transform;
		}

		// Apply magnet positions:
		if (skeleton->get_bone_parent(fabrik_data_chain[i].bone_idx) >= 0) {
			// bone_transform is the global bone pose.
			// global_pose * local_pose.inverse() gives the parent's pose.
			Basis conversion_basis = bone_transform.get_basis() * skeleton->get_bone_pose(fabrik_data_chain[i].bone_idx).get_basis().inverse();
			fabrik_transforms[i].origin += conversion_basis.xform_inv(fabrik_data_chain[i].magnet_position);
		} else {
			fabrik_transforms[i].origin += fabrik_data_chain[i].magnet_position;
		}
	}

	target_global_pose = get_target_transform(target_cache);
	origin_global_pose = origin_global_pose_trans;

	final_joint_idx = fabrik_data_chain.size() - 1;
	real_t target_distance = fabrik_transforms[final_joint_idx].origin.distance_to(target_global_pose.origin);
	chain_iterations = 0;

	while (target_distance > chain_tolerance) {
		chain_backwards();
		chain_forwards();

		// update the target distance
		target_distance = fabrik_transforms[final_joint_idx].origin.distance_to(target_global_pose.origin);

		// update chain iterations
		chain_iterations += 1;
		if (chain_iterations >= chain_max_iterations) {
			break;
		}
	}
	chain_apply();
}

void SkeletonModification3DFABRIK::skeleton_changed(Skeleton3D *skeleton) {
	target_cache = Variant();
	for (uint32_t i = 0; i < fabrik_data_chain.size(); i++) {
		fabrik_data_chain[i].bone_idx = UNCACHED_BONE_IDX;
		fabrik_data_chain[i].tip_cache = Variant();
	}
	SkeletonModification3D::skeleton_changed(skeleton);
}

bool SkeletonModification3DFABRIK::is_bone_property(String p_property_name) const {
	if (p_property_name == "target_bone" || p_property_name.ends_with("/tip_bone") || p_property_name.ends_with("/bone")) {
		return true;
	}
	return SkeletonModification3D::is_bone_property(p_property_name);
}

bool SkeletonModification3DFABRIK::is_property_hidden(String p_property_name) const {
	if (p_property_name.begins_with("joint_") && p_property_name.ends_with("/tip_bone")) {
		const int fabrik_data_size = fabrik_data_chain.size();
		int which = p_property_name.get_slicec('/', 0).substr(6).to_int();
		ERR_FAIL_INDEX_V(which, fabrik_data_size, false);
		if (!fabrik_data_chain[which].tip_node.is_empty()) {
			return true;
		}
	}
	if (p_property_name == "target_bone" && !target_node.is_empty()) {
		return true;
	}
	return SkeletonModification3D::is_property_hidden(p_property_name);
}

PackedStringArray SkeletonModification3DFABRIK::get_configuration_warnings() const {
	PackedStringArray ret = SkeletonModification3D::get_configuration_warnings();
	if (!get_skeleton()) {
		return ret;
	}
	if (!_cache_target(target_cache, target_node, target_bone)) {
		ret.append(vformat("Target %s %s was not found.", target_node.is_empty() ? "bone" : "node", target_node.is_empty() ? target_bone : (String)target_node));
	}
	for (uint32_t i = 0; i < fabrik_data_chain.size(); i++) {
		if (!_cache_bone(fabrik_data_chain[i].bone_idx, fabrik_data_chain[i].bone_name)) {
			ret.append(vformat("Joint %d bone %s was not found.", i, fabrik_data_chain[i].bone_name));
		}
		if (!_cache_target(fabrik_data_chain[i].tip_cache, fabrik_data_chain[i].tip_node, fabrik_data_chain[i].tip_bone)) {
			ret.append(vformat("Joint %d tip %s %s was not found.", i, fabrik_data_chain[i].tip_node.is_empty() ? "bone" : "node", fabrik_data_chain[i].tip_node.is_empty() ? fabrik_data_chain[i].tip_bone : (String)fabrik_data_chain[i].tip_node));
		}
	}
	return ret;
}
