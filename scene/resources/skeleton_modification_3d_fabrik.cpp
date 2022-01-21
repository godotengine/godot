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

#include "scene/resources/skeleton_modification_3d_fabrik.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/skeleton_modification_3d.h"

bool SkeletonModification3DFABRIK::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int fabrik_data_size = fabrik_data_chain.size();
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, fabrik_data_size, false);

		if (what == "bone_name") {
			set_fabrik_joint_bone_name(which, p_value);
		} else if (what == "bone_index") {
			set_fabrik_joint_bone_index(which, p_value);
		} else if (what == "length") {
			set_fabrik_joint_length(which, p_value);
		} else if (what == "magnet_position") {
			set_fabrik_joint_magnet(which, p_value);
		} else if (what == "auto_calculate_length") {
			set_fabrik_joint_auto_calculate_length(which, p_value);
		} else if (what == "use_tip_node") {
			set_fabrik_joint_use_tip_node(which, p_value);
		} else if (what == "tip_node") {
			set_fabrik_joint_tip_node(which, p_value);
		} else if (what == "use_target_basis") {
			set_fabrik_joint_use_target_basis(which, p_value);
		} else if (what == "roll") {
			set_fabrik_joint_roll(which, Math::deg2rad(real_t(p_value)));
		}
		return true;
	}
	return true;
}

bool SkeletonModification3DFABRIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		const int fabrik_data_size = fabrik_data_chain.size();
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, fabrik_data_size, false);

		if (what == "bone_name") {
			r_ret = get_fabrik_joint_bone_name(which);
		} else if (what == "bone_index") {
			r_ret = get_fabrik_joint_bone_index(which);
		} else if (what == "length") {
			r_ret = get_fabrik_joint_length(which);
		} else if (what == "magnet_position") {
			r_ret = get_fabrik_joint_magnet(which);
		} else if (what == "auto_calculate_length") {
			r_ret = get_fabrik_joint_auto_calculate_length(which);
		} else if (what == "use_tip_node") {
			r_ret = get_fabrik_joint_use_tip_node(which);
		} else if (what == "tip_node") {
			r_ret = get_fabrik_joint_tip_node(which);
		} else if (what == "use_target_basis") {
			r_ret = get_fabrik_joint_use_target_basis(which);
		} else if (what == "roll") {
			r_ret = Math::rad2deg(get_fabrik_joint_roll(which));
		}
		return true;
	}
	return true;
}

void SkeletonModification3DFABRIK::_get_property_list(List<PropertyInfo> *p_list) const {
	for (uint32_t i = 0; i < fabrik_data_chain.size(); i++) {
		String base_string = "joint_data/" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::STRING_NAME, base_string + "bone_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::INT, base_string + "bone_index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "roll", PROPERTY_HINT_RANGE, "-360,360,0.01", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "auto_calculate_length", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));

		if (!fabrik_data_chain[i].auto_calculate_length) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "length", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		} else {
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "use_tip_node", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			if (fabrik_data_chain[i].use_tip_node) {
				p_list->push_back(PropertyInfo(Variant::NODE_PATH, base_string + "tip_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D", PROPERTY_USAGE_DEFAULT));
			}
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

void SkeletonModification3DFABRIK::_execute(real_t p_delta) {
	ERR_FAIL_COND_MSG(!stack || !is_setup || stack->skeleton == nullptr,
			"Modification is not setup and therefore cannot execute!");
	if (!enabled) {
		return;
	}

	if (target_node_cache.is_null()) {
		_print_execution_error(true, "Target cache is out of date. Attempting to update...");
		update_target_cache();
		return;
	}

	if (_print_execution_error(fabrik_data_chain.size() <= 1, "FABRIK requires at least two joints to operate. Cannot execute modification!")) {
		return;
	}

	Node3D *node_target = Object::cast_to<Node3D>(ObjectDB::get_instance(target_node_cache));
	if (_print_execution_error(!node_target || !node_target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!")) {
		return;
	}

	// Make sure the transform cache is the correct size
	if (fabrik_transforms.size() != fabrik_data_chain.size()) {
		fabrik_transforms.resize(fabrik_data_chain.size());
	}

	// Verify that all joints have a valid bone ID, and that all bone lengths are zero or more
	// Also, while we are here, apply magnet positions.
	for (uint32_t i = 0; i < fabrik_data_chain.size(); i++) {
		if (_print_execution_error(fabrik_data_chain[i].bone_idx < 0, "FABRIK Joint " + itos(i) + " has an invalid bone ID. Cannot execute!")) {
			return;
		}

		if (fabrik_data_chain[i].length < 0 && fabrik_data_chain[i].auto_calculate_length) {
			fabrik_joint_auto_calculate_length(i);
		}
		if (_print_execution_error(fabrik_data_chain[i].length < 0, "FABRIK Joint " + itos(i) + " has an invalid joint length. Cannot execute!")) {
			return;
		}
		fabrik_transforms[i] = stack->skeleton->get_bone_global_pose(fabrik_data_chain[i].bone_idx);

		// Apply magnet positions:
		if (stack->skeleton->get_bone_parent(fabrik_data_chain[i].bone_idx) >= 0) {
			int parent_bone_idx = stack->skeleton->get_bone_parent(fabrik_data_chain[i].bone_idx);
			Transform3D conversion_transform = (stack->skeleton->get_bone_global_pose(parent_bone_idx));
			fabrik_transforms[i].origin += conversion_transform.basis.xform_inv(fabrik_data_chain[i].magnet_position);
		} else {
			fabrik_transforms[i].origin += fabrik_data_chain[i].magnet_position;
		}
	}
	Transform3D origin_global_pose_trans = stack->skeleton->get_bone_global_pose_no_override(fabrik_data_chain[0].bone_idx);

	target_global_pose = stack->skeleton->world_transform_to_global_pose(node_target->get_global_transform());
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

	execution_error_found = false;
}

void SkeletonModification3DFABRIK::chain_backwards() {
	int final_bone_idx = fabrik_data_chain[final_joint_idx].bone_idx;
	Transform3D final_joint_trans = fabrik_transforms[final_joint_idx];

	// Get the direction the final bone is facing in.
	stack->skeleton->update_bone_rest_forward_vector(final_bone_idx);
	Transform3D final_bone_direction_trans = final_joint_trans.looking_at(target_global_pose.origin, Vector3(0, 1, 0));
	final_bone_direction_trans.basis = stack->skeleton->global_pose_z_forward_to_bone_forward(final_bone_idx, final_bone_direction_trans.basis);
	Vector3 direction = final_bone_direction_trans.basis.xform(stack->skeleton->get_bone_axis_forward_vector(final_bone_idx)).normalized();

	// If set to override, then use the target's Basis rather than the bone's
	if (fabrik_data_chain[final_joint_idx].use_target_basis) {
		direction = target_global_pose.basis.xform(stack->skeleton->get_bone_axis_forward_vector(final_bone_idx)).normalized();
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
	for (uint32_t i = 0; i < fabrik_data_chain.size(); i++) {
		int current_bone_idx = fabrik_data_chain[i].bone_idx;
		Transform3D current_trans = fabrik_transforms[i];

		// If this is the last bone in the chain...
		if (i == fabrik_data_chain.size() - 1) {
			if (fabrik_data_chain[i].use_target_basis == false) { // Point to target...
				// Get the forward direction that the basis is facing in right now.
				stack->skeleton->update_bone_rest_forward_vector(current_bone_idx);
				Vector3 forward_vector = stack->skeleton->get_bone_axis_forward_vector(current_bone_idx);
				// Rotate the bone towards the target:
				current_trans.basis.rotate_to_align(forward_vector, current_trans.origin.direction_to(target_global_pose.origin));
				current_trans.basis.rotate_local(forward_vector, fabrik_data_chain[i].roll);
			} else { // Use the target's Basis...
				current_trans.basis = target_global_pose.basis.orthonormalized().scaled(current_trans.basis.get_scale());
			}
		} else { // every other bone in the chain...
			Transform3D next_trans = fabrik_transforms[i + 1];

			// Get the forward direction that the basis is facing in right now.
			stack->skeleton->update_bone_rest_forward_vector(current_bone_idx);
			Vector3 forward_vector = stack->skeleton->get_bone_axis_forward_vector(current_bone_idx);
			// Rotate the bone towards the next bone in the chain:
			current_trans.basis.rotate_to_align(forward_vector, current_trans.origin.direction_to(next_trans.origin));
			current_trans.basis.rotate_local(forward_vector, fabrik_data_chain[i].roll);
		}
		stack->skeleton->set_bone_local_pose_override(current_bone_idx, stack->skeleton->global_pose_to_local_pose(current_bone_idx, current_trans), stack->strength, true);
	}

	// Update all the bones so the next modification has up-to-date data.
	stack->skeleton->force_update_all_bone_transforms();
}

void SkeletonModification3DFABRIK::_setup_modification(SkeletonModificationStack3D *p_stack) {
	stack = p_stack;
	if (stack != nullptr) {
		is_setup = true;
		execution_error_found = false;
		update_target_cache();

		for (uint32_t i = 0; i < fabrik_data_chain.size(); i++) {
			update_joint_tip_cache(i);
		}
	}
}

void SkeletonModification3DFABRIK::update_target_cache() {
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update target cache: modification is not properly setup!");
		return;
	}
	target_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree() && target_node.is_empty() == false) {
			if (stack->skeleton->has_node(target_node)) {
				Node *node = stack->skeleton->get_node(target_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: node is not in the scene tree!");
				target_node_cache = node->get_instance_id();

				execution_error_found = false;
			}
		}
	}
}

void SkeletonModification3DFABRIK::update_joint_tip_cache(int p_joint_idx) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_MSG(p_joint_idx, bone_chain_size, "FABRIK joint not found");
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update tip cache: modification is not properly setup!");
		return;
	}
	fabrik_data_chain[p_joint_idx].tip_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree() && fabrik_data_chain[p_joint_idx].tip_node.is_empty() == false) {
			if (stack->skeleton->has_node(fabrik_data_chain[p_joint_idx].tip_node)) {
				Node *node = stack->skeleton->get_node(fabrik_data_chain[p_joint_idx].tip_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update tip cache for joint " + itos(p_joint_idx) + ": node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update tip cache for joint " + itos(p_joint_idx) + ": node is not in scene tree!");
				fabrik_data_chain[p_joint_idx].tip_node_cache = node->get_instance_id();

				execution_error_found = false;
			}
		}
	}
}

void SkeletonModification3DFABRIK::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_target_cache();
}

NodePath SkeletonModification3DFABRIK::get_target_node() const {
	return target_node;
}

int SkeletonModification3DFABRIK::get_fabrik_data_chain_length() {
	return fabrik_data_chain.size();
}

void SkeletonModification3DFABRIK::set_fabrik_data_chain_length(int p_length) {
	ERR_FAIL_COND(p_length < 0);
	fabrik_data_chain.resize(p_length);
	fabrik_transforms.resize(p_length);
	execution_error_found = false;
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
String SkeletonModification3DFABRIK::get_fabrik_joint_bone_name(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, String());
	return fabrik_data_chain[p_joint_idx].bone_name;
}

void SkeletonModification3DFABRIK::set_fabrik_joint_bone_name(int p_joint_idx, String p_bone_name) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].bone_name = p_bone_name;

	if (stack) {
		if (stack->skeleton) {
			fabrik_data_chain[p_joint_idx].bone_idx = stack->skeleton->find_bone(p_bone_name);
		}
	}
	execution_error_found = false;
	notify_property_list_changed();
}

int SkeletonModification3DFABRIK::get_fabrik_joint_bone_index(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, -1);
	return fabrik_data_chain[p_joint_idx].bone_idx;
}

void SkeletonModification3DFABRIK::set_fabrik_joint_bone_index(int p_joint_idx, int p_bone_idx) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");
	fabrik_data_chain[p_joint_idx].bone_idx = p_bone_idx;

	if (stack) {
		if (stack->skeleton) {
			fabrik_data_chain[p_joint_idx].bone_name = stack->skeleton->get_bone_name(p_bone_idx);
		}
	}
	execution_error_found = false;
	notify_property_list_changed();
}

real_t SkeletonModification3DFABRIK::get_fabrik_joint_length(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, -1);
	return fabrik_data_chain[p_joint_idx].length;
}

void SkeletonModification3DFABRIK::set_fabrik_joint_length(int p_joint_idx, real_t p_bone_length) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ERR_FAIL_COND_MSG(p_bone_length < 0, "FABRIK joint length cannot be less than zero!");

	if (!is_setup) {
		fabrik_data_chain[p_joint_idx].length = p_bone_length;
		return;
	}

	if (fabrik_data_chain[p_joint_idx].auto_calculate_length) {
		WARN_PRINT("FABRIK Length not set: auto calculate length is enabled for this joint!");
		fabrik_joint_auto_calculate_length(p_joint_idx);
	} else {
		fabrik_data_chain[p_joint_idx].length = p_bone_length;
	}

	execution_error_found = false;
}

Vector3 SkeletonModification3DFABRIK::get_fabrik_joint_magnet(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, Vector3());
	return fabrik_data_chain[p_joint_idx].magnet_position;
}

void SkeletonModification3DFABRIK::set_fabrik_joint_magnet(int p_joint_idx, Vector3 p_magnet) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].magnet_position = p_magnet;
}

bool SkeletonModification3DFABRIK::get_fabrik_joint_auto_calculate_length(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return fabrik_data_chain[p_joint_idx].auto_calculate_length;
}

void SkeletonModification3DFABRIK::set_fabrik_joint_auto_calculate_length(int p_joint_idx, bool p_auto_calculate) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].auto_calculate_length = p_auto_calculate;
	fabrik_joint_auto_calculate_length(p_joint_idx);
	notify_property_list_changed();
}

void SkeletonModification3DFABRIK::fabrik_joint_auto_calculate_length(int p_joint_idx) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	if (!fabrik_data_chain[p_joint_idx].auto_calculate_length) {
		return;
	}

	if (!stack || !stack->skeleton || !is_setup) {
		_print_execution_error(true, "Cannot auto calculate joint length: modification is not properly setup!");
		return;
	}
	ERR_FAIL_INDEX_MSG(fabrik_data_chain[p_joint_idx].bone_idx, stack->skeleton->get_bone_count(),
			"Bone for joint " + itos(p_joint_idx) + " is not set or points to an unknown bone!");

	if (fabrik_data_chain[p_joint_idx].use_tip_node) { // Use the tip node to update joint length.

		update_joint_tip_cache(p_joint_idx);

		Node3D *tip_node = Object::cast_to<Node3D>(ObjectDB::get_instance(fabrik_data_chain[p_joint_idx].tip_node_cache));
		ERR_FAIL_COND_MSG(!tip_node, "Tip node for joint " + itos(p_joint_idx) + "is not a Node3D-based node. Cannot calculate length...");
		ERR_FAIL_COND_MSG(!tip_node->is_inside_tree(), "Tip node for joint " + itos(p_joint_idx) + "is not in the scene tree. Cannot calculate length...");

		Transform3D node_trans = tip_node->get_global_transform();
		node_trans = stack->skeleton->world_transform_to_global_pose(node_trans);
		//node_trans = stack->skeleton->global_pose_to_local_pose(fabrik_data_chain[p_joint_idx].bone_idx, node_trans);
		//fabrik_data_chain[p_joint_idx].length = node_trans.origin.length();

		fabrik_data_chain[p_joint_idx].length = stack->skeleton->get_bone_global_pose(fabrik_data_chain[p_joint_idx].bone_idx).origin.distance_to(node_trans.origin);

	} else { // Use child bone(s) to update joint length, if possible
		Vector<int> bone_children = stack->skeleton->get_bone_children(fabrik_data_chain[p_joint_idx].bone_idx);
		if (bone_children.size() <= 0) {
			ERR_FAIL_MSG("Cannot calculate length for joint " + itos(p_joint_idx) + "joint uses leaf bone. \nPlease manually set the bone length or use a tip node!");
			return;
		}

		Transform3D bone_trans = stack->skeleton->get_bone_global_pose(fabrik_data_chain[p_joint_idx].bone_idx);

		real_t final_length = 0;
		for (int i = 0; i < bone_children.size(); i++) {
			Transform3D child_transform = stack->skeleton->get_bone_global_pose(bone_children[i]);
			final_length += bone_trans.origin.distance_to(child_transform.origin);
			//final_length += stack->skeleton->global_pose_to_local_pose(fabrik_data_chain[p_joint_idx].bone_idx, child_transform).origin.length();
		}
		fabrik_data_chain[p_joint_idx].length = final_length / bone_children.size();
	}
	execution_error_found = false;
	notify_property_list_changed();
}

bool SkeletonModification3DFABRIK::get_fabrik_joint_use_tip_node(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return fabrik_data_chain[p_joint_idx].use_tip_node;
}

void SkeletonModification3DFABRIK::set_fabrik_joint_use_tip_node(int p_joint_idx, bool p_use_tip_node) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].use_tip_node = p_use_tip_node;
	notify_property_list_changed();
}

NodePath SkeletonModification3DFABRIK::get_fabrik_joint_tip_node(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, NodePath());
	return fabrik_data_chain[p_joint_idx].tip_node;
}

void SkeletonModification3DFABRIK::set_fabrik_joint_tip_node(int p_joint_idx, NodePath p_tip_node) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].tip_node = p_tip_node;
	update_joint_tip_cache(p_joint_idx);
}

bool SkeletonModification3DFABRIK::get_fabrik_joint_use_target_basis(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return fabrik_data_chain[p_joint_idx].use_target_basis;
}

void SkeletonModification3DFABRIK::set_fabrik_joint_use_target_basis(int p_joint_idx, bool p_use_target_basis) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].use_target_basis = p_use_target_basis;
}

real_t SkeletonModification3DFABRIK::get_fabrik_joint_roll(int p_joint_idx) const {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, 0.0);
	return fabrik_data_chain[p_joint_idx].roll;
}

void SkeletonModification3DFABRIK::set_fabrik_joint_roll(int p_joint_idx, real_t p_roll) {
	const int bone_chain_size = fabrik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	fabrik_data_chain[p_joint_idx].roll = p_roll;
}

void SkeletonModification3DFABRIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification3DFABRIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification3DFABRIK::get_target_node);
	ClassDB::bind_method(D_METHOD("set_fabrik_data_chain_length", "length"), &SkeletonModification3DFABRIK::set_fabrik_data_chain_length);
	ClassDB::bind_method(D_METHOD("get_fabrik_data_chain_length"), &SkeletonModification3DFABRIK::get_fabrik_data_chain_length);
	ClassDB::bind_method(D_METHOD("set_chain_tolerance", "tolerance"), &SkeletonModification3DFABRIK::set_chain_tolerance);
	ClassDB::bind_method(D_METHOD("get_chain_tolerance"), &SkeletonModification3DFABRIK::get_chain_tolerance);
	ClassDB::bind_method(D_METHOD("set_chain_max_iterations", "max_iterations"), &SkeletonModification3DFABRIK::set_chain_max_iterations);
	ClassDB::bind_method(D_METHOD("get_chain_max_iterations"), &SkeletonModification3DFABRIK::get_chain_max_iterations);

	// FABRIK joint data functions
	ClassDB::bind_method(D_METHOD("get_fabrik_joint_bone_name", "joint_idx"), &SkeletonModification3DFABRIK::get_fabrik_joint_bone_name);
	ClassDB::bind_method(D_METHOD("set_fabrik_joint_bone_name", "joint_idx", "bone_name"), &SkeletonModification3DFABRIK::set_fabrik_joint_bone_name);
	ClassDB::bind_method(D_METHOD("get_fabrik_joint_bone_index", "joint_idx"), &SkeletonModification3DFABRIK::get_fabrik_joint_bone_index);
	ClassDB::bind_method(D_METHOD("set_fabrik_joint_bone_index", "joint_idx", "bone_index"), &SkeletonModification3DFABRIK::set_fabrik_joint_bone_index);
	ClassDB::bind_method(D_METHOD("get_fabrik_joint_length", "joint_idx"), &SkeletonModification3DFABRIK::get_fabrik_joint_length);
	ClassDB::bind_method(D_METHOD("set_fabrik_joint_length", "joint_idx", "length"), &SkeletonModification3DFABRIK::set_fabrik_joint_length);
	ClassDB::bind_method(D_METHOD("get_fabrik_joint_magnet", "joint_idx"), &SkeletonModification3DFABRIK::get_fabrik_joint_magnet);
	ClassDB::bind_method(D_METHOD("set_fabrik_joint_magnet", "joint_idx", "magnet_position"), &SkeletonModification3DFABRIK::set_fabrik_joint_magnet);
	ClassDB::bind_method(D_METHOD("get_fabrik_joint_auto_calculate_length", "joint_idx"), &SkeletonModification3DFABRIK::get_fabrik_joint_auto_calculate_length);
	ClassDB::bind_method(D_METHOD("set_fabrik_joint_auto_calculate_length", "joint_idx", "auto_calculate_length"), &SkeletonModification3DFABRIK::set_fabrik_joint_auto_calculate_length);
	ClassDB::bind_method(D_METHOD("fabrik_joint_auto_calculate_length", "joint_idx"), &SkeletonModification3DFABRIK::fabrik_joint_auto_calculate_length);
	ClassDB::bind_method(D_METHOD("get_fabrik_joint_use_tip_node", "joint_idx"), &SkeletonModification3DFABRIK::get_fabrik_joint_use_tip_node);
	ClassDB::bind_method(D_METHOD("set_fabrik_joint_use_tip_node", "joint_idx", "use_tip_node"), &SkeletonModification3DFABRIK::set_fabrik_joint_use_tip_node);
	ClassDB::bind_method(D_METHOD("get_fabrik_joint_tip_node", "joint_idx"), &SkeletonModification3DFABRIK::get_fabrik_joint_tip_node);
	ClassDB::bind_method(D_METHOD("set_fabrik_joint_tip_node", "joint_idx", "tip_node"), &SkeletonModification3DFABRIK::set_fabrik_joint_tip_node);
	ClassDB::bind_method(D_METHOD("get_fabrik_joint_use_target_basis", "joint_idx"), &SkeletonModification3DFABRIK::get_fabrik_joint_use_target_basis);
	ClassDB::bind_method(D_METHOD("set_fabrik_joint_use_target_basis", "joint_idx", "use_target_basis"), &SkeletonModification3DFABRIK::set_fabrik_joint_use_target_basis);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fabrik_data_chain_length", PROPERTY_HINT_RANGE, "0,100,1"), "set_fabrik_data_chain_length", "get_fabrik_data_chain_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "chain_tolerance", PROPERTY_HINT_RANGE, "0,100,0.001"), "set_chain_tolerance", "get_chain_tolerance");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "chain_max_iterations", PROPERTY_HINT_RANGE, "1,50,1"), "set_chain_max_iterations", "get_chain_max_iterations");
}

SkeletonModification3DFABRIK::SkeletonModification3DFABRIK() {
	stack = nullptr;
	is_setup = false;
	enabled = true;
}

SkeletonModification3DFABRIK::~SkeletonModification3DFABRIK() {
}
