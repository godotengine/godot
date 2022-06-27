/*************************************************************************/
/*  skeleton_modification_3d_ccdik.cpp                                   */
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

#include "scene/resources/skeleton_modification_3d_ccdik.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/skeleton_modification_3d.h"

bool SkeletonModification3DCCDIK::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int ccdik_data_size = ccdik_data_chain.size();
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, ccdik_data_size, false);

		if (what == "bone_name") {
			set_ccdik_joint_bone_name(which, p_value);
		} else if (what == "bone_index") {
			set_ccdik_joint_bone_index(which, p_value);
		} else if (what == "ccdik_axis") {
			set_ccdik_joint_ccdik_axis(which, p_value);
		} else if (what == "enable_joint_constraint") {
			set_ccdik_joint_enable_constraint(which, p_value);
		} else if (what == "joint_constraint_angle_min") {
			set_ccdik_joint_constraint_angle_min(which, Math::deg2rad(real_t(p_value)));
		} else if (what == "joint_constraint_angle_max") {
			set_ccdik_joint_constraint_angle_max(which, Math::deg2rad(real_t(p_value)));
		} else if (what == "joint_constraint_angles_invert") {
			set_ccdik_joint_constraint_invert(which, p_value);
		}
		return true;
	}
	return true;
}

bool SkeletonModification3DCCDIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		const int ccdik_data_size = ccdik_data_chain.size();
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, ccdik_data_size, false);

		if (what == "bone_name") {
			r_ret = get_ccdik_joint_bone_name(which);
		} else if (what == "bone_index") {
			r_ret = get_ccdik_joint_bone_index(which);
		} else if (what == "ccdik_axis") {
			r_ret = get_ccdik_joint_ccdik_axis(which);
		} else if (what == "enable_joint_constraint") {
			r_ret = get_ccdik_joint_enable_constraint(which);
		} else if (what == "joint_constraint_angle_min") {
			r_ret = Math::rad2deg(get_ccdik_joint_constraint_angle_min(which));
		} else if (what == "joint_constraint_angle_max") {
			r_ret = Math::rad2deg(get_ccdik_joint_constraint_angle_max(which));
		} else if (what == "joint_constraint_angles_invert") {
			r_ret = get_ccdik_joint_constraint_invert(which);
		}
		return true;
	}
	return true;
}

void SkeletonModification3DCCDIK::_get_property_list(List<PropertyInfo> *p_list) const {
	for (uint32_t i = 0; i < ccdik_data_chain.size(); i++) {
		String base_string = "joint_data/" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::STRING_NAME, base_string + "bone_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::INT, base_string + "bone_index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));

		p_list->push_back(PropertyInfo(Variant::INT, base_string + "ccdik_axis",
				PROPERTY_HINT_ENUM, "X Axis, Y Axis, Z Axis", PROPERTY_USAGE_DEFAULT));

		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "enable_joint_constraint", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		if (ccdik_data_chain[i].enable_constraint) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "joint_constraint_angle_min", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "joint_constraint_angle_max", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "joint_constraint_angles_invert", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
	}
}

void SkeletonModification3DCCDIK::_execute(real_t p_delta) {
	ERR_FAIL_COND_MSG(!stack || !is_setup || stack->skeleton == nullptr,
			"Modification is not setup and therefore cannot execute!");
	if (!enabled) {
		return;
	}

	if (target_node_cache.is_null()) {
		_print_execution_error(true, "Target cache is out of date. Attempting to update");
		update_target_cache();
		return;
	}
	if (tip_node_cache.is_null()) {
		_print_execution_error(true, "Tip cache is out of date. Attempting to update");
		update_tip_cache();
		return;
	}

	// Reset the local bone overrides for CCDIK affected nodes
	for (uint32_t i = 0; i < ccdik_data_chain.size(); i++) {
		stack->skeleton->set_bone_local_pose_override(ccdik_data_chain[i].bone_idx,
				stack->skeleton->get_bone_local_pose_override(ccdik_data_chain[i].bone_idx),
				0.0, false);
	}

	Node3D *node_target = Object::cast_to<Node3D>(ObjectDB::get_instance(target_node_cache));
	Node3D *node_tip = Object::cast_to<Node3D>(ObjectDB::get_instance(tip_node_cache));

	if (_print_execution_error(!node_target || !node_target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!")) {
		return;
	}
	if (_print_execution_error(!node_tip || !node_tip->is_inside_tree(), "Tip node is not in the scene tree. Cannot execute modification!")) {
		return;
	}

	if (use_high_quality_solve) {
		for (uint32_t i = 0; i < ccdik_data_chain.size(); i++) {
			for (uint32_t j = i; j < ccdik_data_chain.size(); j++) {
				_execute_ccdik_joint(j, node_target, node_tip);
			}
		}
	} else {
		for (uint32_t i = 0; i < ccdik_data_chain.size(); i++) {
			_execute_ccdik_joint(i, node_target, node_tip);
		}
	}

	execution_error_found = false;
}

void SkeletonModification3DCCDIK::_execute_ccdik_joint(int p_joint_idx, Node3D *p_target, Node3D *p_tip) {
	CCDIK_Joint_Data ccdik_data = ccdik_data_chain[p_joint_idx];

	if (_print_execution_error(ccdik_data.bone_idx < 0 || ccdik_data.bone_idx > stack->skeleton->get_bone_count(),
				"CCDIK joint: bone index for joint" + itos(p_joint_idx) + " not found. Cannot execute modification!")) {
		return;
	}

	Transform3D bone_trans = stack->skeleton->global_pose_to_local_pose(ccdik_data.bone_idx, stack->skeleton->get_bone_global_pose(ccdik_data.bone_idx));
	Transform3D tip_trans = stack->skeleton->global_pose_to_local_pose(ccdik_data.bone_idx, stack->skeleton->world_transform_to_global_pose(p_tip->get_global_transform()));
	Transform3D target_trans = stack->skeleton->global_pose_to_local_pose(ccdik_data.bone_idx, stack->skeleton->world_transform_to_global_pose(p_target->get_global_transform()));

	if (tip_trans.origin.distance_to(target_trans.origin) <= 0.01) {
		return;
	}

	// Inspired (and very loosely based on) by the CCDIK algorithm made by Zalo on GitHub (https://github.com/zalo/MathUtilities)
	// Convert the 3D position to a 2D position so we can use Atan2 (via the angle function)
	// to know how much rotation we need on the given axis to place the tip at the target.
	Vector2 tip_pos_2d;
	Vector2 target_pos_2d;
	if (ccdik_data.ccdik_axis == CCDIK_Axes::AXIS_X) {
		tip_pos_2d = Vector2(tip_trans.origin.y, tip_trans.origin.z);
		target_pos_2d = Vector2(target_trans.origin.y, target_trans.origin.z);
		bone_trans.basis.rotate_local(Vector3(1, 0, 0), target_pos_2d.angle() - tip_pos_2d.angle());
	} else if (ccdik_data.ccdik_axis == CCDIK_Axes::AXIS_Y) {
		tip_pos_2d = Vector2(tip_trans.origin.z, tip_trans.origin.x);
		target_pos_2d = Vector2(target_trans.origin.z, target_trans.origin.x);
		bone_trans.basis.rotate_local(Vector3(0, 1, 0), target_pos_2d.angle() - tip_pos_2d.angle());
	} else if (ccdik_data.ccdik_axis == CCDIK_Axes::AXIS_Z) {
		tip_pos_2d = Vector2(tip_trans.origin.x, tip_trans.origin.y);
		target_pos_2d = Vector2(target_trans.origin.x, target_trans.origin.y);
		bone_trans.basis.rotate_local(Vector3(0, 0, 1), target_pos_2d.angle() - tip_pos_2d.angle());
	} else {
		// Should never happen, but...
		ERR_FAIL_MSG("CCDIK joint: Unknown axis vector passed for joint" + itos(p_joint_idx) + ". Cannot execute modification!");
	}

	if (ccdik_data.enable_constraint) {
		Vector3 rotation_axis;
		real_t rotation_angle;
		bone_trans.basis.get_axis_angle(rotation_axis, rotation_angle);

		// Note: When the axis has a negative direction, the angle is OVER 180 degrees and therefore we need to account for this
		// when constraining.
		if (ccdik_data.ccdik_axis == CCDIK_Axes::AXIS_X) {
			if (rotation_axis.x < 0) {
				rotation_angle += Math_PI;
				rotation_axis = Vector3(1, 0, 0);
			}
		} else if (ccdik_data.ccdik_axis == CCDIK_Axes::AXIS_Y) {
			if (rotation_axis.y < 0) {
				rotation_angle += Math_PI;
				rotation_axis = Vector3(0, 1, 0);
			}
		} else if (ccdik_data.ccdik_axis == CCDIK_Axes::AXIS_Z) {
			if (rotation_axis.z < 0) {
				rotation_angle += Math_PI;
				rotation_axis = Vector3(0, 0, 1);
			}
		} else {
			// Should never happen, but...
			ERR_FAIL_MSG("CCDIK joint: Unknown axis vector passed for joint" + itos(p_joint_idx) + ". Cannot execute modification!");
		}
		rotation_angle = clamp_angle(rotation_angle, ccdik_data.constraint_angle_min, ccdik_data.constraint_angle_max, ccdik_data.constraint_angles_invert);

		bone_trans.basis.set_axis_angle(rotation_axis, rotation_angle);
	}

	stack->skeleton->set_bone_local_pose_override(ccdik_data.bone_idx, bone_trans, stack->strength, true);
	stack->skeleton->force_update_bone_children_transforms(ccdik_data.bone_idx);
}

void SkeletonModification3DCCDIK::_setup_modification(SkeletonModificationStack3D *p_stack) {
	stack = p_stack;
	if (stack != nullptr) {
		is_setup = true;
		execution_error_found = false;
		update_target_cache();
		update_tip_cache();
	}
}

void SkeletonModification3DCCDIK::update_target_cache() {
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update target cache: modification is not properly setup!");
		return;
	}

	target_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(target_node)) {
				Node *node = stack->skeleton->get_node(target_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: node is not in scene tree!");
				target_node_cache = node->get_instance_id();

				execution_error_found = false;
			}
		}
	}
}

void SkeletonModification3DCCDIK::update_tip_cache() {
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update tip cache: modification is not properly setup!");
		return;
	}

	tip_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(tip_node)) {
				Node *node = stack->skeleton->get_node(tip_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update tip cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update tip cache: node is not in scene tree!");
				tip_node_cache = node->get_instance_id();

				execution_error_found = false;
			}
		}
	}
}

void SkeletonModification3DCCDIK::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_target_cache();
}

NodePath SkeletonModification3DCCDIK::get_target_node() const {
	return target_node;
}

void SkeletonModification3DCCDIK::set_tip_node(const NodePath &p_tip_node) {
	tip_node = p_tip_node;
	update_tip_cache();
}

NodePath SkeletonModification3DCCDIK::get_tip_node() const {
	return tip_node;
}

void SkeletonModification3DCCDIK::set_use_high_quality_solve(bool p_high_quality) {
	use_high_quality_solve = p_high_quality;
}

bool SkeletonModification3DCCDIK::get_use_high_quality_solve() const {
	return use_high_quality_solve;
}

// CCDIK joint data functions
String SkeletonModification3DCCDIK::get_ccdik_joint_bone_name(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, String());
	return ccdik_data_chain[p_joint_idx].bone_name;
}

void SkeletonModification3DCCDIK::set_ccdik_joint_bone_name(int p_joint_idx, String p_bone_name) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ccdik_data_chain[p_joint_idx].bone_name = p_bone_name;

	if (stack) {
		if (stack->skeleton) {
			ccdik_data_chain[p_joint_idx].bone_idx = stack->skeleton->find_bone(p_bone_name);
		}
	}
	execution_error_found = false;
	notify_property_list_changed();
}

int SkeletonModification3DCCDIK::get_ccdik_joint_bone_index(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, -1);
	return ccdik_data_chain[p_joint_idx].bone_idx;
}

void SkeletonModification3DCCDIK::set_ccdik_joint_bone_index(int p_joint_idx, int p_bone_idx) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");
	ccdik_data_chain[p_joint_idx].bone_idx = p_bone_idx;

	if (stack) {
		if (stack->skeleton) {
			ccdik_data_chain[p_joint_idx].bone_name = stack->skeleton->get_bone_name(p_bone_idx);
		}
	}
	execution_error_found = false;
	notify_property_list_changed();
}

int SkeletonModification3DCCDIK::get_ccdik_joint_ccdik_axis(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, -1);
	return ccdik_data_chain[p_joint_idx].ccdik_axis;
}

void SkeletonModification3DCCDIK::set_ccdik_joint_ccdik_axis(int p_joint_idx, int p_axis) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ERR_FAIL_COND_MSG(p_axis < 0, "CCDIK axis is out of range: The axis mode is too low!");
	ccdik_data_chain[p_joint_idx].ccdik_axis = p_axis;
	notify_property_list_changed();
}

bool SkeletonModification3DCCDIK::get_ccdik_joint_enable_constraint(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return ccdik_data_chain[p_joint_idx].enable_constraint;
}

void SkeletonModification3DCCDIK::set_ccdik_joint_enable_constraint(int p_joint_idx, bool p_enable) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ccdik_data_chain[p_joint_idx].enable_constraint = p_enable;
	notify_property_list_changed();
}

real_t SkeletonModification3DCCDIK::get_ccdik_joint_constraint_angle_min(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return ccdik_data_chain[p_joint_idx].constraint_angle_min;
}

void SkeletonModification3DCCDIK::set_ccdik_joint_constraint_angle_min(int p_joint_idx, real_t p_angle_min) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ccdik_data_chain[p_joint_idx].constraint_angle_min = p_angle_min;
}

real_t SkeletonModification3DCCDIK::get_ccdik_joint_constraint_angle_max(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return ccdik_data_chain[p_joint_idx].constraint_angle_max;
}

void SkeletonModification3DCCDIK::set_ccdik_joint_constraint_angle_max(int p_joint_idx, real_t p_angle_max) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ccdik_data_chain[p_joint_idx].constraint_angle_max = p_angle_max;
}

bool SkeletonModification3DCCDIK::get_ccdik_joint_constraint_invert(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return ccdik_data_chain[p_joint_idx].constraint_angles_invert;
}

void SkeletonModification3DCCDIK::set_ccdik_joint_constraint_invert(int p_joint_idx, bool p_invert) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ccdik_data_chain[p_joint_idx].constraint_angles_invert = p_invert;
}

int SkeletonModification3DCCDIK::get_ccdik_data_chain_length() {
	return ccdik_data_chain.size();
}
void SkeletonModification3DCCDIK::set_ccdik_data_chain_length(int p_length) {
	ERR_FAIL_COND(p_length < 0);
	ccdik_data_chain.resize(p_length);
	execution_error_found = false;
	notify_property_list_changed();
}

void SkeletonModification3DCCDIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification3DCCDIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification3DCCDIK::get_target_node);

	ClassDB::bind_method(D_METHOD("set_tip_node", "tip_nodepath"), &SkeletonModification3DCCDIK::set_tip_node);
	ClassDB::bind_method(D_METHOD("get_tip_node"), &SkeletonModification3DCCDIK::get_tip_node);

	ClassDB::bind_method(D_METHOD("set_use_high_quality_solve", "high_quality_solve"), &SkeletonModification3DCCDIK::set_use_high_quality_solve);
	ClassDB::bind_method(D_METHOD("get_use_high_quality_solve"), &SkeletonModification3DCCDIK::get_use_high_quality_solve);

	// CCDIK joint data functions
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_bone_name", "joint_idx"), &SkeletonModification3DCCDIK::get_ccdik_joint_bone_name);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_bone_name", "joint_idx", "bone_name"), &SkeletonModification3DCCDIK::set_ccdik_joint_bone_name);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_bone_index", "joint_idx"), &SkeletonModification3DCCDIK::get_ccdik_joint_bone_index);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_bone_index", "joint_idx", "bone_index"), &SkeletonModification3DCCDIK::set_ccdik_joint_bone_index);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_ccdik_axis", "joint_idx"), &SkeletonModification3DCCDIK::get_ccdik_joint_ccdik_axis);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_ccdik_axis", "joint_idx", "axis"), &SkeletonModification3DCCDIK::set_ccdik_joint_ccdik_axis);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_enable_joint_constraint", "joint_idx"), &SkeletonModification3DCCDIK::get_ccdik_joint_enable_constraint);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_enable_joint_constraint", "joint_idx", "enable"), &SkeletonModification3DCCDIK::set_ccdik_joint_enable_constraint);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_constraint_angle_min", "joint_idx"), &SkeletonModification3DCCDIK::get_ccdik_joint_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_constraint_angle_min", "joint_idx", "min_angle"), &SkeletonModification3DCCDIK::set_ccdik_joint_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_constraint_angle_max", "joint_idx"), &SkeletonModification3DCCDIK::get_ccdik_joint_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_constraint_angle_max", "joint_idx", "max_angle"), &SkeletonModification3DCCDIK::set_ccdik_joint_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_constraint_invert", "joint_idx"), &SkeletonModification3DCCDIK::get_ccdik_joint_constraint_invert);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_constraint_invert", "joint_idx", "invert"), &SkeletonModification3DCCDIK::set_ccdik_joint_constraint_invert);

	ClassDB::bind_method(D_METHOD("set_ccdik_data_chain_length", "length"), &SkeletonModification3DCCDIK::set_ccdik_data_chain_length);
	ClassDB::bind_method(D_METHOD("get_ccdik_data_chain_length"), &SkeletonModification3DCCDIK::get_ccdik_data_chain_length);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "tip_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_tip_node", "get_tip_node");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "high_quality_solve", PROPERTY_HINT_NONE, ""), "set_use_high_quality_solve", "get_use_high_quality_solve");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ccdik_data_chain_length", PROPERTY_HINT_RANGE, "0,100,1"), "set_ccdik_data_chain_length", "get_ccdik_data_chain_length");
}

SkeletonModification3DCCDIK::SkeletonModification3DCCDIK() {
	stack = nullptr;
	is_setup = false;
	enabled = true;
}

SkeletonModification3DCCDIK::~SkeletonModification3DCCDIK() {
}
