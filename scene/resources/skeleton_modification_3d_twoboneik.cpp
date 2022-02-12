/*************************************************************************/
/*  skeleton_modification_3d_twoboneik.cpp                               */
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

#include "scene/resources/skeleton_modification_3d_twoboneik.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/skeleton_modification_3d.h"

bool SkeletonModification3DTwoBoneIK::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path == "use_tip_node") {
		set_use_tip_node(p_value);
	} else if (path == "tip_node") {
		set_tip_node(p_value);
	} else if (path == "auto_calculate_joint_length") {
		set_auto_calculate_joint_length(p_value);
	} else if (path == "use_pole_node") {
		set_use_pole_node(p_value);
	} else if (path == "pole_node") {
		set_pole_node(p_value);
	} else if (path == "joint_one_length") {
		set_joint_one_length(p_value);
	} else if (path == "joint_two_length") {
		set_joint_two_length(p_value);
	} else if (path == "joint_one/bone_name") {
		set_joint_one_bone_name(p_value);
	} else if (path == "joint_one/bone_idx") {
		set_joint_one_bone_idx(p_value);
	} else if (path == "joint_one/roll") {
		set_joint_one_roll(Math::deg2rad(real_t(p_value)));
	} else if (path == "joint_two/bone_name") {
		set_joint_two_bone_name(p_value);
	} else if (path == "joint_two/bone_idx") {
		set_joint_two_bone_idx(p_value);
	} else if (path == "joint_two/roll") {
		set_joint_two_roll(Math::deg2rad(real_t(p_value)));
	}

	return true;
}

bool SkeletonModification3DTwoBoneIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path == "use_tip_node") {
		r_ret = get_use_tip_node();
	} else if (path == "tip_node") {
		r_ret = get_tip_node();
	} else if (path == "auto_calculate_joint_length") {
		r_ret = get_auto_calculate_joint_length();
	} else if (path == "use_pole_node") {
		r_ret = get_use_pole_node();
	} else if (path == "pole_node") {
		r_ret = get_pole_node();
	} else if (path == "joint_one_length") {
		r_ret = get_joint_one_length();
	} else if (path == "joint_two_length") {
		r_ret = get_joint_two_length();
	} else if (path == "joint_one/bone_name") {
		r_ret = get_joint_one_bone_name();
	} else if (path == "joint_one/bone_idx") {
		r_ret = get_joint_one_bone_idx();
	} else if (path == "joint_one/roll") {
		r_ret = Math::rad2deg(get_joint_one_roll());
	} else if (path == "joint_two/bone_name") {
		r_ret = get_joint_two_bone_name();
	} else if (path == "joint_two/bone_idx") {
		r_ret = get_joint_two_bone_idx();
	} else if (path == "joint_two/roll") {
		r_ret = Math::rad2deg(get_joint_two_roll());
	}

	return true;
}

void SkeletonModification3DTwoBoneIK::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::BOOL, "use_tip_node", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	if (use_tip_node) {
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, "tip_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D", PROPERTY_USAGE_DEFAULT));
	}

	p_list->push_back(PropertyInfo(Variant::BOOL, "auto_calculate_joint_length", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	if (!auto_calculate_joint_length) {
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_one_length", PROPERTY_HINT_RANGE, "-1, 10000, 0.001", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_two_length", PROPERTY_HINT_RANGE, "-1, 10000, 0.001", PROPERTY_USAGE_DEFAULT));
	}

	p_list->push_back(PropertyInfo(Variant::BOOL, "use_pole_node", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	if (use_pole_node) {
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, "pole_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D", PROPERTY_USAGE_DEFAULT));
	}

	p_list->push_back(PropertyInfo(Variant::STRING_NAME, "joint_one/bone_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	p_list->push_back(PropertyInfo(Variant::INT, "joint_one/bone_idx", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_one/roll", PROPERTY_HINT_RANGE, "-360, 360, 0.01", PROPERTY_USAGE_DEFAULT));

	p_list->push_back(PropertyInfo(Variant::STRING_NAME, "joint_two/bone_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	p_list->push_back(PropertyInfo(Variant::INT, "joint_two/bone_idx", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "joint_two/roll", PROPERTY_HINT_RANGE, "-360, 360, 0.01", PROPERTY_USAGE_DEFAULT));
}

void SkeletonModification3DTwoBoneIK::_execute(real_t p_delta) {
	ERR_FAIL_COND_MSG(!stack || !is_setup || stack->skeleton == nullptr,
			"Modification is not setup and therefore cannot execute!");

	if (!enabled) {
		return;
	}

	if (_print_execution_error(joint_one_bone_idx < 0 || joint_two_bone_idx < 0,
				"One (or more) of the bones in the modification have invalid bone indexes. Cannot execute modification!")) {
		return;
	}

	if (target_node_cache.is_null()) {
		_print_execution_error(true, "Target cache is out of date. Attempting to update...");
		update_cache_target();
		return;
	}

	// Update joint lengths (if needed)
	if (auto_calculate_joint_length && (joint_one_length < 0 || joint_two_length < 0)) {
		calculate_joint_lengths();
	}

	// Adopted from the links below:
	// http://theorangeduck.com/page/simple-two-joint
	// https://www.alanzucconi.com/2018/05/02/ik-2d-2/
	// With modifications by TwistedTwigleg
	Node3D *target = Object::cast_to<Node3D>(ObjectDB::get_instance(target_node_cache));
	if (_print_execution_error(!target || !target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!")) {
		return;
	}
	Transform3D target_trans = stack->skeleton->world_transform_to_global_pose(target->get_global_transform());

	Transform3D bone_one_trans;
	Transform3D bone_two_trans;

	// Make the first joint look at the pole, and the second look at the target. That way, the
	// TwoBoneIK solver has to really only handle extension/contraction, which should make it align with the pole.
	if (use_pole_node) {
		if (pole_node_cache.is_null()) {
			_print_execution_error(true, "Pole cache is out of date. Attempting to update...");
			update_cache_pole();
			return;
		}

		Node3D *pole = Object::cast_to<Node3D>(ObjectDB::get_instance(pole_node_cache));
		if (_print_execution_error(!pole || !pole->is_inside_tree(), "Pole node is not in the scene tree. Cannot execute modification!")) {
			return;
		}
		Transform3D pole_trans = stack->skeleton->world_transform_to_global_pose(pole->get_global_transform());

		Transform3D bone_one_local_pos = stack->skeleton->get_bone_local_pose_override(joint_one_bone_idx);
		if (bone_one_local_pos == Transform3D()) {
			bone_one_local_pos = stack->skeleton->get_bone_pose(joint_one_bone_idx);
		}
		Transform3D bone_two_local_pos = stack->skeleton->get_bone_local_pose_override(joint_two_bone_idx);
		if (bone_two_local_pos == Transform3D()) {
			bone_two_local_pos = stack->skeleton->get_bone_pose(joint_two_bone_idx);
		}

		bone_one_trans = stack->skeleton->local_pose_to_global_pose(joint_one_bone_idx, bone_one_local_pos);
		bone_one_trans = bone_one_trans.looking_at(pole_trans.origin, Vector3(0, 1, 0));
		bone_one_trans.basis = stack->skeleton->global_pose_z_forward_to_bone_forward(joint_one_bone_idx, bone_one_trans.basis);
		stack->skeleton->update_bone_rest_forward_vector(joint_one_bone_idx);
		bone_one_trans.basis.rotate_local(stack->skeleton->get_bone_axis_forward_vector(joint_one_bone_idx), joint_one_roll);
		stack->skeleton->set_bone_local_pose_override(joint_one_bone_idx, stack->skeleton->global_pose_to_local_pose(joint_one_bone_idx, bone_one_trans), stack->strength, true);
		stack->skeleton->force_update_bone_children_transforms(joint_one_bone_idx);

		bone_two_trans = stack->skeleton->local_pose_to_global_pose(joint_two_bone_idx, bone_two_local_pos);
		bone_two_trans = bone_two_trans.looking_at(target_trans.origin, Vector3(0, 1, 0));
		bone_two_trans.basis = stack->skeleton->global_pose_z_forward_to_bone_forward(joint_two_bone_idx, bone_two_trans.basis);
		stack->skeleton->update_bone_rest_forward_vector(joint_two_bone_idx);
		bone_two_trans.basis.rotate_local(stack->skeleton->get_bone_axis_forward_vector(joint_two_bone_idx), joint_two_roll);
		stack->skeleton->set_bone_local_pose_override(joint_two_bone_idx, stack->skeleton->global_pose_to_local_pose(joint_two_bone_idx, bone_two_trans), stack->strength, true);
		stack->skeleton->force_update_bone_children_transforms(joint_two_bone_idx);
	} else {
		Transform3D bone_one_local_pos = stack->skeleton->get_bone_local_pose_override(joint_one_bone_idx);
		if (bone_one_local_pos == Transform3D()) {
			bone_one_local_pos = stack->skeleton->get_bone_pose(joint_one_bone_idx);
		}
		Transform3D bone_two_local_pos = stack->skeleton->get_bone_local_pose_override(joint_two_bone_idx);
		if (bone_two_local_pos == Transform3D()) {
			bone_two_local_pos = stack->skeleton->get_bone_pose(joint_two_bone_idx);
		}

		bone_one_trans = stack->skeleton->local_pose_to_global_pose(joint_one_bone_idx, bone_one_local_pos);
		bone_two_trans = stack->skeleton->local_pose_to_global_pose(joint_two_bone_idx, bone_two_local_pos);
	}

	Transform3D bone_two_tip_trans;
	if (use_tip_node) {
		if (tip_node_cache.is_null()) {
			_print_execution_error(true, "Tip cache is out of date. Attempting to update...");
			update_cache_tip();
			return;
		}
		Node3D *tip = Object::cast_to<Node3D>(ObjectDB::get_instance(tip_node_cache));
		if (_print_execution_error(!tip || !tip->is_inside_tree(), "Tip node is not in the scene tree. Cannot execute modification!")) {
			return;
		}
		bone_two_tip_trans = stack->skeleton->world_transform_to_global_pose(tip->get_global_transform());
	} else {
		stack->skeleton->update_bone_rest_forward_vector(joint_two_bone_idx);
		bone_two_tip_trans = bone_two_trans;
		bone_two_tip_trans.origin += bone_two_trans.basis.xform(stack->skeleton->get_bone_axis_forward_vector(joint_two_bone_idx)).normalized() * joint_two_length;
	}

	real_t joint_one_to_target_length = bone_one_trans.origin.distance_to(target_trans.origin);
	if (joint_one_length + joint_two_length < joint_one_to_target_length) {
		// Set the target *just* out of reach to straighten the bones
		joint_one_to_target_length = joint_one_length + joint_two_length + 0.01;
	} else if (joint_one_to_target_length < joint_one_length) {
		// Place the target in reach so the solver doesn't do crazy things
		joint_one_to_target_length = joint_one_length;
	}

	// Get the square lengths for all three sides of the triangle we'll use to calculate the angles
	real_t sqr_one_length = joint_one_length * joint_one_length;
	real_t sqr_two_length = joint_two_length * joint_two_length;
	real_t sqr_three_length = joint_one_to_target_length * joint_one_to_target_length;

	// Calculate the angles for the first joint using the law of cosigns
	real_t ac_ab_0 = Math::acos(CLAMP(bone_two_tip_trans.origin.direction_to(bone_one_trans.origin).dot(bone_two_trans.origin.direction_to(bone_one_trans.origin)), -1, 1));
	real_t ac_at_0 = Math::acos(CLAMP(bone_one_trans.origin.direction_to(bone_two_tip_trans.origin).dot(bone_one_trans.origin.direction_to(target_trans.origin)), -1, 1));
	real_t ac_ab_1 = Math::acos(CLAMP((sqr_two_length - sqr_one_length - sqr_three_length) / (-2.0 * joint_one_length * joint_one_to_target_length), -1, 1));

	// Calculate the angles of rotation. Angle 0 is the extension/contraction axis, while angle 1 is the rotation axis to align the triangle to the target
	Vector3 axis_0 = bone_one_trans.origin.direction_to(bone_two_tip_trans.origin).cross(bone_one_trans.origin.direction_to(bone_two_trans.origin));
	Vector3 axis_1 = bone_one_trans.origin.direction_to(bone_two_tip_trans.origin).cross(bone_one_trans.origin.direction_to(target_trans.origin));

	// Make a quaternion with the delta rotation needed to rotate the first joint into alignment and apply it to the transform.
	Quaternion bone_one_quat = bone_one_trans.basis.get_rotation_quaternion();
	Quaternion rot_0 = Quaternion(bone_one_quat.inverse().xform(axis_0).normalized(), (ac_ab_1 - ac_ab_0));
	Quaternion rot_2 = Quaternion(bone_one_quat.inverse().xform(axis_1).normalized(), ac_at_0);
	bone_one_trans.basis.set_quaternion(bone_one_quat * (rot_0 * rot_2));

	stack->skeleton->update_bone_rest_forward_vector(joint_one_bone_idx);
	bone_one_trans.basis.rotate_local(stack->skeleton->get_bone_axis_forward_vector(joint_one_bone_idx), joint_one_roll);

	// Apply the rotation to the first joint
	bone_one_trans = stack->skeleton->global_pose_to_local_pose(joint_one_bone_idx, bone_one_trans);
	bone_one_trans.origin = Vector3(0, 0, 0);
	stack->skeleton->set_bone_local_pose_override(joint_one_bone_idx, bone_one_trans, stack->strength, true);
	stack->skeleton->force_update_bone_children_transforms(joint_one_bone_idx);

	if (use_pole_node) {
		// Update bone_two_trans so its at the latest position, with the rotation of bone_one_trans taken into account, then look at the target.
		bone_two_trans = stack->skeleton->local_pose_to_global_pose(joint_two_bone_idx, stack->skeleton->get_bone_local_pose_override(joint_two_bone_idx));
		stack->skeleton->update_bone_rest_forward_vector(joint_two_bone_idx);
		Vector3 forward_vector = stack->skeleton->get_bone_axis_forward_vector(joint_two_bone_idx);
		bone_two_trans.basis.rotate_to_align(forward_vector, bone_two_trans.origin.direction_to(target_trans.origin));

		stack->skeleton->update_bone_rest_forward_vector(joint_two_bone_idx);
		bone_two_trans.basis.rotate_local(stack->skeleton->get_bone_axis_forward_vector(joint_two_bone_idx), joint_two_roll);

		bone_two_trans = stack->skeleton->global_pose_to_local_pose(joint_two_bone_idx, bone_two_trans);
		stack->skeleton->set_bone_local_pose_override(joint_two_bone_idx, bone_two_trans, stack->strength, true);
		stack->skeleton->force_update_bone_children_transforms(joint_two_bone_idx);
	} else {
		// Calculate the angles for the second joint using the law of cosigns, make a quaternion with the delta rotation needed to rotate the joint into
		// alignment, and then apply it to the second joint.
		real_t ba_bc_0 = Math::acos(CLAMP(bone_two_trans.origin.direction_to(bone_one_trans.origin).dot(bone_two_trans.origin.direction_to(bone_two_tip_trans.origin)), -1, 1));
		real_t ba_bc_1 = Math::acos(CLAMP((sqr_three_length - sqr_one_length - sqr_two_length) / (-2.0 * joint_one_length * joint_two_length), -1, 1));
		Quaternion bone_two_quat = bone_two_trans.basis.get_rotation_quaternion();
		Quaternion rot_1 = Quaternion(bone_two_quat.inverse().xform(axis_0).normalized(), (ba_bc_1 - ba_bc_0));
		bone_two_trans.basis.set_quaternion(bone_two_quat * rot_1);

		stack->skeleton->update_bone_rest_forward_vector(joint_two_bone_idx);
		bone_two_trans.basis.rotate_local(stack->skeleton->get_bone_axis_forward_vector(joint_two_bone_idx), joint_two_roll);

		bone_two_trans = stack->skeleton->global_pose_to_local_pose(joint_two_bone_idx, bone_two_trans);
		bone_two_trans.origin = Vector3(0, 0, 0);
		stack->skeleton->set_bone_local_pose_override(joint_two_bone_idx, bone_two_trans, stack->strength, true);
		stack->skeleton->force_update_bone_children_transforms(joint_two_bone_idx);
	}
}

void SkeletonModification3DTwoBoneIK::_setup_modification(SkeletonModificationStack3D *p_stack) {
	stack = p_stack;

	if (stack != nullptr) {
		is_setup = true;
		execution_error_found = false;
		update_cache_target();
		update_cache_tip();
	}
}

void SkeletonModification3DTwoBoneIK::update_cache_target() {
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
						"Cannot update target cache: Target node is this modification's skeleton or cannot be found. Cannot execute modification");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: Target node is not in the scene tree. Cannot execute modification!");
				target_node_cache = node->get_instance_id();

				execution_error_found = false;
			}
		}
	}
}

void SkeletonModification3DTwoBoneIK::update_cache_tip() {
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
						"Cannot update tip cache: Tip node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update tip cache: Tip node is not in the scene tree. Cannot execute modification!");
				tip_node_cache = node->get_instance_id();

				execution_error_found = false;
			}
		}
	}
}

void SkeletonModification3DTwoBoneIK::update_cache_pole() {
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update pole cache: modification is not properly setup!");
		return;
	}

	pole_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(pole_node)) {
				Node *node = stack->skeleton->get_node(pole_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update pole cache: Pole node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update pole cache: Pole node is not in the scene tree. Cannot execute modification!");
				pole_node_cache = node->get_instance_id();

				execution_error_found = false;
			}
		}
	}
}

void SkeletonModification3DTwoBoneIK::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_cache_target();
}

NodePath SkeletonModification3DTwoBoneIK::get_target_node() const {
	return target_node;
}

void SkeletonModification3DTwoBoneIK::set_use_tip_node(const bool p_use_tip_node) {
	use_tip_node = p_use_tip_node;
	notify_property_list_changed();
}

bool SkeletonModification3DTwoBoneIK::get_use_tip_node() const {
	return use_tip_node;
}

void SkeletonModification3DTwoBoneIK::set_tip_node(const NodePath &p_tip_node) {
	tip_node = p_tip_node;
	update_cache_tip();
}

NodePath SkeletonModification3DTwoBoneIK::get_tip_node() const {
	return tip_node;
}

void SkeletonModification3DTwoBoneIK::set_use_pole_node(const bool p_use_pole_node) {
	use_pole_node = p_use_pole_node;
	notify_property_list_changed();
}

bool SkeletonModification3DTwoBoneIK::get_use_pole_node() const {
	return use_pole_node;
}

void SkeletonModification3DTwoBoneIK::set_pole_node(const NodePath &p_pole_node) {
	pole_node = p_pole_node;
	update_cache_pole();
}

NodePath SkeletonModification3DTwoBoneIK::get_pole_node() const {
	return pole_node;
}

void SkeletonModification3DTwoBoneIK::set_auto_calculate_joint_length(bool p_calculate) {
	auto_calculate_joint_length = p_calculate;
	if (p_calculate) {
		calculate_joint_lengths();
	}
	notify_property_list_changed();
}

bool SkeletonModification3DTwoBoneIK::get_auto_calculate_joint_length() const {
	return auto_calculate_joint_length;
}

void SkeletonModification3DTwoBoneIK::calculate_joint_lengths() {
	if (!is_setup) {
		return; // fail silently, as we likely just loaded the scene.
	}
	ERR_FAIL_COND_MSG(!stack || stack->skeleton == nullptr,
			"Modification is not setup and therefore cannot calculate joint lengths!");
	ERR_FAIL_COND_MSG(joint_one_bone_idx <= -1 || joint_two_bone_idx <= -1,
			"One of the bones in the TwoBoneIK modification are not set! Cannot calculate joint lengths!");

	Transform3D bone_one_rest_trans = stack->skeleton->get_bone_global_pose(joint_one_bone_idx);
	Transform3D bone_two_rest_trans = stack->skeleton->get_bone_global_pose(joint_two_bone_idx);

	joint_one_length = bone_one_rest_trans.origin.distance_to(bone_two_rest_trans.origin);

	if (use_tip_node) {
		if (tip_node_cache.is_null()) {
			update_cache_tip();
			WARN_PRINT("Tip cache is out of date. Updating...");
		}

		Node3D *tip = Object::cast_to<Node3D>(ObjectDB::get_instance(tip_node_cache));
		if (tip) {
			Transform3D bone_tip_trans = stack->skeleton->world_transform_to_global_pose(tip->get_global_transform());
			joint_two_length = bone_two_rest_trans.origin.distance_to(bone_tip_trans.origin);
		}
	} else {
		// Attempt to use children bones to get the length
		Vector<int> bone_two_children = stack->skeleton->get_bone_children(joint_two_bone_idx);
		if (bone_two_children.size() > 0) {
			joint_two_length = 0;
			for (int i = 0; i < bone_two_children.size(); i++) {
				joint_two_length += bone_two_rest_trans.origin.distance_to(
						stack->skeleton->get_bone_global_pose(bone_two_children[i]).origin);
			}
			joint_two_length = joint_two_length / bone_two_children.size();
		} else {
			WARN_PRINT("TwoBoneIK modification: Cannot auto calculate length for joint 2! Auto setting the length to 1...");
			joint_two_length = 1.0;
		}
	}
	execution_error_found = false;
}

void SkeletonModification3DTwoBoneIK::set_joint_one_bone_name(String p_bone_name) {
	joint_one_bone_name = p_bone_name;
	if (stack && stack->skeleton) {
		joint_one_bone_idx = stack->skeleton->find_bone(p_bone_name);
	}
	execution_error_found = false;
	notify_property_list_changed();
}

String SkeletonModification3DTwoBoneIK::get_joint_one_bone_name() const {
	return joint_one_bone_name;
}

void SkeletonModification3DTwoBoneIK::set_joint_one_bone_idx(int p_bone_idx) {
	joint_one_bone_idx = p_bone_idx;
	if (stack && stack->skeleton) {
		joint_one_bone_name = stack->skeleton->get_bone_name(p_bone_idx);
	}
	execution_error_found = false;
	notify_property_list_changed();
}

int SkeletonModification3DTwoBoneIK::get_joint_one_bone_idx() const {
	return joint_one_bone_idx;
}

void SkeletonModification3DTwoBoneIK::set_joint_one_length(real_t p_length) {
	joint_one_length = p_length;
}

real_t SkeletonModification3DTwoBoneIK::get_joint_one_length() const {
	return joint_one_length;
}

void SkeletonModification3DTwoBoneIK::set_joint_two_bone_name(String p_bone_name) {
	joint_two_bone_name = p_bone_name;
	if (stack && stack->skeleton) {
		joint_two_bone_idx = stack->skeleton->find_bone(p_bone_name);
	}
	execution_error_found = false;
	notify_property_list_changed();
}

String SkeletonModification3DTwoBoneIK::get_joint_two_bone_name() const {
	return joint_two_bone_name;
}

void SkeletonModification3DTwoBoneIK::set_joint_two_bone_idx(int p_bone_idx) {
	joint_two_bone_idx = p_bone_idx;
	if (stack && stack->skeleton) {
		joint_two_bone_name = stack->skeleton->get_bone_name(p_bone_idx);
	}
	execution_error_found = false;
	notify_property_list_changed();
}

int SkeletonModification3DTwoBoneIK::get_joint_two_bone_idx() const {
	return joint_two_bone_idx;
}

void SkeletonModification3DTwoBoneIK::set_joint_two_length(real_t p_length) {
	joint_two_length = p_length;
}

real_t SkeletonModification3DTwoBoneIK::get_joint_two_length() const {
	return joint_two_length;
}

void SkeletonModification3DTwoBoneIK::set_joint_one_roll(real_t p_roll) {
	joint_one_roll = p_roll;
}

real_t SkeletonModification3DTwoBoneIK::get_joint_one_roll() const {
	return joint_one_roll;
}

void SkeletonModification3DTwoBoneIK::set_joint_two_roll(real_t p_roll) {
	joint_two_roll = p_roll;
}

real_t SkeletonModification3DTwoBoneIK::get_joint_two_roll() const {
	return joint_two_roll;
}

void SkeletonModification3DTwoBoneIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification3DTwoBoneIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification3DTwoBoneIK::get_target_node);

	ClassDB::bind_method(D_METHOD("set_use_pole_node", "use_pole_node"), &SkeletonModification3DTwoBoneIK::set_use_pole_node);
	ClassDB::bind_method(D_METHOD("get_use_pole_node"), &SkeletonModification3DTwoBoneIK::get_use_pole_node);
	ClassDB::bind_method(D_METHOD("set_pole_node", "pole_nodepath"), &SkeletonModification3DTwoBoneIK::set_pole_node);
	ClassDB::bind_method(D_METHOD("get_pole_node"), &SkeletonModification3DTwoBoneIK::get_pole_node);

	ClassDB::bind_method(D_METHOD("set_use_tip_node", "use_tip_node"), &SkeletonModification3DTwoBoneIK::set_use_tip_node);
	ClassDB::bind_method(D_METHOD("get_use_tip_node"), &SkeletonModification3DTwoBoneIK::get_use_tip_node);
	ClassDB::bind_method(D_METHOD("set_tip_node", "tip_nodepath"), &SkeletonModification3DTwoBoneIK::set_tip_node);
	ClassDB::bind_method(D_METHOD("get_tip_node"), &SkeletonModification3DTwoBoneIK::get_tip_node);

	ClassDB::bind_method(D_METHOD("set_auto_calculate_joint_length", "auto_calculate_joint_length"), &SkeletonModification3DTwoBoneIK::set_auto_calculate_joint_length);
	ClassDB::bind_method(D_METHOD("get_auto_calculate_joint_length"), &SkeletonModification3DTwoBoneIK::get_auto_calculate_joint_length);

	ClassDB::bind_method(D_METHOD("set_joint_one_bone_name", "bone_name"), &SkeletonModification3DTwoBoneIK::set_joint_one_bone_name);
	ClassDB::bind_method(D_METHOD("get_joint_one_bone_name"), &SkeletonModification3DTwoBoneIK::get_joint_one_bone_name);
	ClassDB::bind_method(D_METHOD("set_joint_one_bone_idx", "bone_idx"), &SkeletonModification3DTwoBoneIK::set_joint_one_bone_idx);
	ClassDB::bind_method(D_METHOD("get_joint_one_bone_idx"), &SkeletonModification3DTwoBoneIK::get_joint_one_bone_idx);
	ClassDB::bind_method(D_METHOD("set_joint_one_length", "bone_length"), &SkeletonModification3DTwoBoneIK::set_joint_one_length);
	ClassDB::bind_method(D_METHOD("get_joint_one_length"), &SkeletonModification3DTwoBoneIK::get_joint_one_length);

	ClassDB::bind_method(D_METHOD("set_joint_two_bone_name", "bone_name"), &SkeletonModification3DTwoBoneIK::set_joint_two_bone_name);
	ClassDB::bind_method(D_METHOD("get_joint_two_bone_name"), &SkeletonModification3DTwoBoneIK::get_joint_two_bone_name);
	ClassDB::bind_method(D_METHOD("set_joint_two_bone_idx", "bone_idx"), &SkeletonModification3DTwoBoneIK::set_joint_two_bone_idx);
	ClassDB::bind_method(D_METHOD("get_joint_two_bone_idx"), &SkeletonModification3DTwoBoneIK::get_joint_two_bone_idx);
	ClassDB::bind_method(D_METHOD("set_joint_two_length", "bone_length"), &SkeletonModification3DTwoBoneIK::set_joint_two_length);
	ClassDB::bind_method(D_METHOD("get_joint_two_length"), &SkeletonModification3DTwoBoneIK::get_joint_two_length);

	ClassDB::bind_method(D_METHOD("set_joint_one_roll", "roll"), &SkeletonModification3DTwoBoneIK::set_joint_one_roll);
	ClassDB::bind_method(D_METHOD("get_joint_one_roll"), &SkeletonModification3DTwoBoneIK::get_joint_one_roll);
	ClassDB::bind_method(D_METHOD("set_joint_two_roll", "roll"), &SkeletonModification3DTwoBoneIK::set_joint_two_roll);
	ClassDB::bind_method(D_METHOD("get_joint_two_roll"), &SkeletonModification3DTwoBoneIK::get_joint_two_roll);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_target_node", "get_target_node");
	ADD_GROUP("", "");
}

SkeletonModification3DTwoBoneIK::SkeletonModification3DTwoBoneIK() {
	stack = nullptr;
	is_setup = false;
}

SkeletonModification3DTwoBoneIK::~SkeletonModification3DTwoBoneIK() {
}
