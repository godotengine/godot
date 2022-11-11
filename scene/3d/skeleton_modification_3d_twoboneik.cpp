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

#include "scene/3d/skeleton_modification_3d_twoboneik.h"
#include "core/error/error_macros.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/skeleton_modification_3d.h"

void SkeletonModification3DTwoBoneIK::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	if (!p_target_node.is_empty()) {
		target_bone = String();
	}
	target_cache = Variant();
}

NodePath SkeletonModification3DTwoBoneIK::get_target_node() const {
	return target_node;
}

void SkeletonModification3DTwoBoneIK::set_target_bone(const String &p_target_bone) {
	target_bone = p_target_bone;
	if (!p_target_bone.is_empty()) {
		target_node = NodePath();
	}
	target_cache = Variant();
}

String SkeletonModification3DTwoBoneIK::get_target_bone() const {
	return target_bone;
}

void SkeletonModification3DTwoBoneIK::set_tip_node(const NodePath &p_tip_node) {
	tip_node = p_tip_node;
	if (!p_tip_node.is_empty()) {
		tip_bone = String();
	}
	tip_cache = Variant();
}

NodePath SkeletonModification3DTwoBoneIK::get_tip_node() const {
	return tip_node;
}

void SkeletonModification3DTwoBoneIK::set_tip_bone(const String &p_tip_bone) {
	tip_bone = p_tip_bone;
	if (!p_tip_bone.is_empty()) {
		tip_node = NodePath();
	}
	tip_cache = Variant();
}

String SkeletonModification3DTwoBoneIK::get_tip_bone() const {
	return tip_bone;
}

void SkeletonModification3DTwoBoneIK::set_pole_node(const NodePath &p_pole_node) {
	pole_node = p_pole_node;
	if (!p_pole_node.is_empty()) {
		pole_bone = String();
	}
	pole_cache = Variant();
}

NodePath SkeletonModification3DTwoBoneIK::get_pole_node() const {
	return pole_node;
}

void SkeletonModification3DTwoBoneIK::set_pole_bone(const String &p_pole_bone) {
	pole_bone = p_pole_bone;
	if (!p_pole_bone.is_empty()) {
		pole_node = NodePath();
	}
	pole_cache = Variant();
}

String SkeletonModification3DTwoBoneIK::get_pole_bone() const {
	return pole_bone;
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
	Skeleton3D *skeleton = get_skeleton();
	if (skeleton == nullptr || !_cache_bone(joint_one_bone_idx, joint_one_bone_name) || !_cache_bone(joint_two_bone_idx, joint_two_bone_name)) {
		WARN_PRINT_ONCE("Failed to cache joints during calculate_joint_lengths");
		return;
	}

	Transform3D bone_one_rest_trans = get_bone_transform(joint_one_bone_idx);
	Transform3D bone_two_rest_trans = get_bone_transform(joint_two_bone_idx);

	joint_one_length = bone_one_rest_trans.origin.distance_to(bone_two_rest_trans.origin);

	if (!tip_node.is_empty() || !tip_bone.is_empty()) {
		if (!_cache_target(tip_cache, tip_node, tip_bone)) {
			return;
		}
		Transform3D bone_tip_trans = get_target_transform(tip_cache);
		joint_two_length = bone_two_rest_trans.origin.distance_to(bone_tip_trans.origin);
	} else {
		// Attempt to use children bones to get the length
		Vector<int> bone_two_children = skeleton->get_bone_children(joint_two_bone_idx);
		if (bone_two_children.size() > 0) {
			joint_two_length = 0;
			for (int i = 0; i < bone_two_children.size(); i++) {
				joint_two_length += bone_two_rest_trans.origin.distance_to(
						get_bone_transform(bone_two_children[i]).origin);
			}
			joint_two_length = joint_two_length / bone_two_children.size();
		} else {
			WARN_PRINT("TwoBoneIK modification: Cannot auto calculate length for joint 2! Auto setting the length to 1...");
			joint_two_length = 1.0;
		}
	}
}

void SkeletonModification3DTwoBoneIK::set_joint_one_bone(String p_bone_name) {
	joint_one_bone_name = p_bone_name;
	joint_one_bone_idx = UNCACHED_BONE_IDX;
}

String SkeletonModification3DTwoBoneIK::get_joint_one_bone() const {
	return joint_one_bone_name;
}

void SkeletonModification3DTwoBoneIK::set_joint_one_length(real_t p_length) {
	joint_one_length = p_length;
}

real_t SkeletonModification3DTwoBoneIK::get_joint_one_length() const {
	return joint_one_length;
}

void SkeletonModification3DTwoBoneIK::set_joint_two_bone(String p_bone_name) {
	joint_two_bone_name = p_bone_name;
	joint_two_bone_idx = UNCACHED_BONE_IDX;
}

String SkeletonModification3DTwoBoneIK::get_joint_two_bone() const {
	return joint_two_bone_name;
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
	ClassDB::bind_method(D_METHOD("set_target_bone", "target_bone_name"), &SkeletonModification3DTwoBoneIK::set_target_bone);
	ClassDB::bind_method(D_METHOD("get_target_bone"), &SkeletonModification3DTwoBoneIK::get_target_bone);

	ClassDB::bind_method(D_METHOD("set_pole_node", "pole_nodepath"), &SkeletonModification3DTwoBoneIK::set_pole_node);
	ClassDB::bind_method(D_METHOD("get_pole_node"), &SkeletonModification3DTwoBoneIK::get_pole_node);
	ClassDB::bind_method(D_METHOD("set_pole_bone", "pole_bone_name"), &SkeletonModification3DTwoBoneIK::set_pole_bone);
	ClassDB::bind_method(D_METHOD("get_pole_bone"), &SkeletonModification3DTwoBoneIK::get_pole_bone);

	ClassDB::bind_method(D_METHOD("set_tip_node", "tip_nodepath"), &SkeletonModification3DTwoBoneIK::set_tip_node);
	ClassDB::bind_method(D_METHOD("get_tip_node"), &SkeletonModification3DTwoBoneIK::get_tip_node);
	ClassDB::bind_method(D_METHOD("set_tip_bone", "tip_bone_name"), &SkeletonModification3DTwoBoneIK::set_tip_bone);
	ClassDB::bind_method(D_METHOD("get_tip_bone"), &SkeletonModification3DTwoBoneIK::get_tip_bone);

	ClassDB::bind_method(D_METHOD("set_auto_calculate_joint_length", "auto_calculate_joint_length"), &SkeletonModification3DTwoBoneIK::set_auto_calculate_joint_length);
	ClassDB::bind_method(D_METHOD("get_auto_calculate_joint_length"), &SkeletonModification3DTwoBoneIK::get_auto_calculate_joint_length);

	ClassDB::bind_method(D_METHOD("set_joint_one_bone", "bone_name"), &SkeletonModification3DTwoBoneIK::set_joint_one_bone);
	ClassDB::bind_method(D_METHOD("get_joint_one_bone"), &SkeletonModification3DTwoBoneIK::get_joint_one_bone);
	ClassDB::bind_method(D_METHOD("set_joint_one_length", "bone_length"), &SkeletonModification3DTwoBoneIK::set_joint_one_length);
	ClassDB::bind_method(D_METHOD("get_joint_one_length"), &SkeletonModification3DTwoBoneIK::get_joint_one_length);

	ClassDB::bind_method(D_METHOD("set_joint_two_bone", "bone_name"), &SkeletonModification3DTwoBoneIK::set_joint_two_bone);
	ClassDB::bind_method(D_METHOD("get_joint_two_bone"), &SkeletonModification3DTwoBoneIK::get_joint_two_bone);
	ClassDB::bind_method(D_METHOD("set_joint_two_length", "bone_length"), &SkeletonModification3DTwoBoneIK::set_joint_two_length);
	ClassDB::bind_method(D_METHOD("get_joint_two_length"), &SkeletonModification3DTwoBoneIK::get_joint_two_length);

	ClassDB::bind_method(D_METHOD("set_joint_one_roll", "roll"), &SkeletonModification3DTwoBoneIK::set_joint_one_roll);
	ClassDB::bind_method(D_METHOD("get_joint_one_roll"), &SkeletonModification3DTwoBoneIK::get_joint_one_roll);
	ClassDB::bind_method(D_METHOD("set_joint_two_roll", "roll"), &SkeletonModification3DTwoBoneIK::set_joint_two_roll);
	ClassDB::bind_method(D_METHOD("get_joint_two_roll"), &SkeletonModification3DTwoBoneIK::get_joint_two_roll);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "target_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), "set_target_bone", "get_target_bone");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D", PROPERTY_USAGE_DEFAULT), "set_target_node", "get_target_node");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "tip_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), "set_tip_bone", "get_tip_bone");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "tip_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D", PROPERTY_USAGE_DEFAULT), "set_tip_node", "get_tip_node");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_calculate_joint_length", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), "set_auto_calculate_joint_length", "get_auto_calculate_joint_length");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "pole_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), "set_pole_bone", "get_pole_bone");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "pole_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D", PROPERTY_USAGE_DEFAULT), "set_pole_node", "get_pole_node");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "joint_one_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), "set_joint_one_bone", "get_joint_one_bone");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "joint_one_roll", PROPERTY_HINT_RANGE, "-360, 360, 0.01", PROPERTY_USAGE_DEFAULT), "set_joint_one_roll", "get_joint_one_roll");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "joint_one_length", PROPERTY_HINT_RANGE, "-1, 10000, 0.001", PROPERTY_USAGE_DEFAULT), "set_joint_one_length", "get_joint_one_length");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "joint_two_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), "set_joint_two_bone", "get_joint_two_bone");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "joint_two_roll", PROPERTY_HINT_RANGE, "-360, 360, 0.01", PROPERTY_USAGE_DEFAULT), "set_joint_two_roll", "get_joint_two_roll");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "joint_two_length", PROPERTY_HINT_RANGE, "-1, 10000, 0.001", PROPERTY_USAGE_DEFAULT), "set_joint_two_length", "get_joint_two_length");
}

SkeletonModification3DTwoBoneIK::SkeletonModification3DTwoBoneIK() {
}

SkeletonModification3DTwoBoneIK::~SkeletonModification3DTwoBoneIK() {
}

void SkeletonModification3DTwoBoneIK::execute(real_t delta) {
	SkeletonModification3D::execute(delta);

	// Update joint lengths (if needed)
	if (auto_calculate_joint_length && (joint_one_length < 0 || joint_two_length < 0)) {
		calculate_joint_lengths();
	}

	// Adopted from the links below:
	// http://theorangeduck.com/page/simple-two-joint
	// https://www.alanzucconi.com/2018/05/02/ik-2d-2/
	// With modifications by TwistedTwigleg
	Transform3D target_trans = get_target_transform(target_cache);
	Skeleton3D *skeleton = get_skeleton();
	if (skeleton == nullptr) {
		WARN_PRINT_ONCE("SkeletonModification3DTwoBoneIK skeleton not initialized in time to execute");
		return;
	}
	if (!_cache_bone(joint_one_bone_idx, joint_one_bone_name) || !_cache_bone(joint_one_bone_idx, joint_one_bone_name)) {
		WARN_PRINT_ONCE("SkeletonModification3DTwoBoneIK missing joint bones.");
		return;
	}

	Vector3 bone_one_forward = get_bone_rest_forward_vector(joint_one_bone_idx);
	Vector3 bone_two_forward = get_bone_rest_forward_vector(joint_two_bone_idx);

	Transform3D bone_one_trans = get_bone_transform(joint_one_bone_idx);
	//Transform3D bone_one_local_pos = skeleton->get_bone_pose(joint_one_bone_idx);
	//Transform3D bone_two_trans = get_bone_transform(joint_two_bone_idx);
	Transform3D bone_two_trans;
	Transform3D bone_two_local_pos = skeleton->get_bone_pose(joint_two_bone_idx);

	int bone_one_parent = skeleton->get_bone_parent(joint_one_bone_idx);
	int bone_two_parent = skeleton->get_bone_parent(joint_two_bone_idx);
	Transform3D bone_one_parent_trans = get_bone_transform(bone_one_parent);
	Transform3D bone_one_to_two_parent_trans;
	if (bone_two_parent != joint_one_bone_idx) {
		Transform3D bone_two_parent_trans = get_bone_transform(bone_one_parent);
		bone_one_to_two_parent_trans = bone_one_trans.affine_inverse() * bone_two_parent_trans;
	}

	// Make the first joint look at the pole, and the second look at the target. That way, the
	// TwoBoneIK solver has to really only handle extension/contraction, which should make it align with the pole.
	if (!pole_node.is_empty() || !pole_bone.is_empty()) {
		if (!_cache_target(pole_cache, pole_node, pole_bone)) {
			WARN_PRINT_ONCE("SkeletonModification3DTwoBoneIK pole not initialized in time to execute");
			return;
		}
		Transform3D pole_trans = get_target_transform(pole_cache);

		bone_one_trans = bone_one_trans.looking_at(pole_trans.origin, Vector3(0, 1, 0));
		bone_one_trans.basis = global_pose_z_forward_to_bone_forward(bone_one_forward, bone_one_trans.basis);
		bone_one_trans.basis.rotate_local(bone_one_forward, joint_one_roll);

		bone_two_trans = bone_one_to_two_parent_trans * bone_two_local_pos;
		bone_two_trans = bone_two_trans.looking_at(target_trans.origin, Vector3(0, 1, 0));
		bone_two_trans.basis = global_pose_z_forward_to_bone_forward(bone_two_forward, bone_two_trans.basis);
		bone_two_trans.basis.rotate_local(bone_two_forward, joint_two_roll);
	} else {
		bone_two_trans = get_bone_transform(joint_two_bone_idx);
	}

	Transform3D bone_two_tip_trans;
	if (!tip_node.is_empty() || !tip_bone.is_empty()) {
		if (!_cache_target(tip_cache, tip_node, tip_bone)) {
			WARN_PRINT_ONCE("SkeletonModification3DTwoBoneIK tip not initialized in time to execute");
			return;
		}
		bone_two_tip_trans = get_target_transform(tip_cache);
	} else {
		bone_two_tip_trans = bone_two_trans;
		bone_two_tip_trans.origin += bone_two_trans.basis.xform(bone_two_forward).normalized() * joint_two_length;
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

	bone_one_trans.basis.rotate_local(bone_one_forward, joint_one_roll);

	// Apply the rotation to the first joint
	skeleton->set_bone_pose_rotation(joint_one_bone_idx,
			(bone_one_parent_trans.basis.inverse() * bone_one_trans.basis).get_rotation_quaternion());
	Transform3D bone_two_parent_trans = bone_two_trans = bone_one_trans * bone_one_to_two_parent_trans;
	bone_two_trans = bone_two_parent_trans * bone_two_local_pos;

	if (!pole_node.is_empty() || !pole_bone.is_empty()) {
		// Update bone_two_trans so its at the latest position, with the rotation of bone_one_trans taken into account, then look at the target.
		bone_two_trans.basis.rotate_to_align(bone_two_forward, bone_two_trans.origin.direction_to(target_trans.origin));
	} else {
		// Calculate the angles for the second joint using the law of cosigns, make a quaternion with the delta rotation needed to rotate the joint into
		// alignment, and then apply it to the second joint.
		real_t ba_bc_0 = Math::acos(CLAMP(bone_two_trans.origin.direction_to(bone_one_trans.origin).dot(bone_two_trans.origin.direction_to(bone_two_tip_trans.origin)), -1, 1));
		real_t ba_bc_1 = Math::acos(CLAMP((sqr_three_length - sqr_one_length - sqr_two_length) / (-2.0 * joint_one_length * joint_two_length), -1, 1));
		Quaternion bone_two_quat = bone_two_trans.basis.get_rotation_quaternion();
		Quaternion rot_1 = Quaternion(bone_two_quat.inverse().xform(axis_0).normalized(), (ba_bc_1 - ba_bc_0));
		bone_two_trans.basis.set_quaternion(bone_two_quat * rot_1);
	}
	bone_two_trans.basis.rotate_local(bone_two_forward, joint_two_roll);

	skeleton->set_bone_pose_rotation(joint_two_bone_idx,
			(bone_two_parent_trans.basis.inverse() * bone_two_trans.basis).get_rotation_quaternion());
}

void SkeletonModification3DTwoBoneIK::skeleton_changed(Skeleton3D *skeleton) {
	target_cache = Variant();
	tip_cache = Variant();
	pole_cache = Variant();
	joint_one_bone_idx = UNCACHED_BONE_IDX;
	joint_two_bone_idx = UNCACHED_BONE_IDX;
	SkeletonModification3D::skeleton_changed(skeleton);
}

bool SkeletonModification3DTwoBoneIK::is_bone_property(String p_property_name) const {
	if (p_property_name == "target_bone" || p_property_name == "tip_bone" || p_property_name == "pole_bone" || p_property_name == "joint_one_bone" || p_property_name == "joint_two_bone") {
		return true;
	}
	return SkeletonModification3D::is_bone_property(p_property_name);
}

bool SkeletonModification3DTwoBoneIK::is_property_hidden(String p_property_name) const {
	if ((p_property_name == "target_bone" && !target_node.is_empty()) ||
			(p_property_name == "tip_bone" && !tip_node.is_empty()) ||
			(p_property_name == "pole_bone" && !pole_node.is_empty())) {
		return true;
	}
	return SkeletonModification3D::is_property_hidden(p_property_name);
}

PackedStringArray SkeletonModification3DTwoBoneIK::get_configuration_warnings() const {
	PackedStringArray ret = SkeletonModification3D::get_configuration_warnings();
	if (!get_skeleton()) {
		return ret;
	}
	if (!_cache_target(target_cache, target_node, target_bone)) {
		ret.append(vformat("Target %s %s was not found.", target_node.is_empty() ? "bone" : "node", target_node.is_empty() ? target_bone : (String)target_node));
	}
	if (!tip_node.is_empty() || !tip_bone.is_empty()) { // optional
		if (!_cache_target(tip_cache, tip_node, tip_bone)) {
			ret.append(vformat("Target %s %s was not found.", tip_node.is_empty() ? "bone" : "node", tip_node.is_empty() ? tip_bone : (String)tip_node));
		}
	}
	if (!pole_node.is_empty() || !pole_bone.is_empty()) { // optional
		if (!_cache_target(pole_cache, pole_node, pole_bone)) {
			ret.append(vformat("Target %s %s was not found.", pole_node.is_empty() ? "bone" : "node", pole_node.is_empty() ? pole_bone : (String)pole_node));
		}
	}
	if (!_cache_bone(joint_one_bone_idx, joint_one_bone_name)) {
		ret.append(vformat("Joint one bone %s was not found.", joint_one_bone_name));
	}
	if (!_cache_bone(joint_two_bone_idx, joint_two_bone_name)) {
		ret.append(vformat("Joint two bone %s was not found.", joint_two_bone_name));
	}
	return ret;
}
