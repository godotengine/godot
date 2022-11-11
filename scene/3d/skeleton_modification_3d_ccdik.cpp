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

#include "scene/3d/skeleton_modification_3d_ccdik.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/skeleton_modification_3d.h"

// Helper function. Needed for CCDIK.
static real_t clamp_angle(real_t p_angle, real_t p_min_bound, real_t p_max_bound, bool p_invert) {
	// Map to the 0 to 360 range (in radians though) instead of the -180 to 180 range.
	if (p_angle < 0) {
		p_angle = Math_TAU + p_angle;
	}

	// Make min and max in the range of 0 to 360 (in radians), and make sure they are in the right order
	if (p_min_bound < 0) {
		p_min_bound = Math_TAU + p_min_bound;
	}
	if (p_max_bound < 0) {
		p_max_bound = Math_TAU + p_max_bound;
	}
	if (p_min_bound > p_max_bound) {
		SWAP(p_min_bound, p_max_bound);
	}

	bool is_beyond_bounds = (p_angle < p_min_bound || p_angle > p_max_bound);
	bool is_within_bounds = (p_angle > p_min_bound && p_angle < p_max_bound);

	// Note: May not be the most optimal way to clamp, but it always constraints to the nearest angle.
	if ((!p_invert && is_beyond_bounds) || (p_invert && is_within_bounds)) {
		Vector2 min_bound_vec = Vector2(Math::cos(p_min_bound), Math::sin(p_min_bound));
		Vector2 max_bound_vec = Vector2(Math::cos(p_max_bound), Math::sin(p_max_bound));
		Vector2 angle_vec = Vector2(Math::cos(p_angle), Math::sin(p_angle));

		if (angle_vec.distance_squared_to(min_bound_vec) <= angle_vec.distance_squared_to(max_bound_vec)) {
			p_angle = p_min_bound;
		} else {
			p_angle = p_max_bound;
		}
	}

	return p_angle;
}

bool SkeletonModification3DCCDIK::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_")) {
		int ccdik_data_size = ccdik_data_chain.size();
		int which = path.get_slicec('/', 0).substr(6).to_int();
		String what = path.get_slicec('/', 1);
		ERR_FAIL_INDEX_V(which, ccdik_data_size, false);

		if (what == "bone") {
			set_joint_bone(which, p_value);
		} else if (what == "ccdik_axis") {
			set_joint_ccdik_axis(which, p_value);
		} else if (what == "enable_joint_constraint") {
			set_joint_enable_constraint(which, p_value);
		} else if (what == "joint_constraint_angle_min") {
			set_joint_constraint_angle_min(which, real_t(p_value));
		} else if (what == "joint_constraint_angle_max") {
			set_joint_constraint_angle_max(which, real_t(p_value));
		} else if (what == "joint_constraint_angles_invert") {
			set_joint_constraint_invert(which, p_value);
		}
		return true;
	}
	return true;
}

bool SkeletonModification3DCCDIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_")) {
		const int ccdik_data_size = ccdik_data_chain.size();
		int which = path.get_slicec('/', 0).substr(6).to_int();
		String what = path.get_slicec('/', 1);
		ERR_FAIL_INDEX_V(which, ccdik_data_size, false);

		if (what == "bone") {
			r_ret = get_joint_bone(which);
		} else if (what == "ccdik_axis") {
			r_ret = get_joint_ccdik_axis(which);
		} else if (what == "enable_joint_constraint") {
			r_ret = get_joint_enable_constraint(which);
		} else if (what == "joint_constraint_angle_min") {
			r_ret = get_joint_constraint_angle_min(which);
		} else if (what == "joint_constraint_angle_max") {
			r_ret = get_joint_constraint_angle_max(which);
		} else if (what == "joint_constraint_angles_invert") {
			r_ret = get_joint_constraint_invert(which);
		}
		return true;
	}
	return true;
}

void SkeletonModification3DCCDIK::_get_property_list(List<PropertyInfo> *p_list) const {
	for (uint32_t i = 0; i < ccdik_data_chain.size(); i++) {
		String base_string = "joint_" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::STRING, base_string + "bone", get_skeleton() ? PROPERTY_HINT_ENUM : PROPERTY_HINT_NONE, get_bone_name_list(), PROPERTY_USAGE_DEFAULT));

		p_list->push_back(PropertyInfo(Variant::INT, base_string + "ccdik_axis",
				PROPERTY_HINT_ENUM, "X Axis, Y Axis, Z Axis", PROPERTY_USAGE_DEFAULT));

		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "enable_joint_constraint", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		if (ccdik_data_chain[i].enable_constraint) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "joint_constraint_angle_min", PROPERTY_HINT_RANGE, "-360,360,0.01,radians", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "joint_constraint_angle_max", PROPERTY_HINT_RANGE, "-360,360,0.01,radians", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "joint_constraint_angles_invert", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
	}
}

void SkeletonModification3DCCDIK::_execute_ccdik_joint(int p_joint_idx, const Transform3D &target_trans, const Transform3D &tip_trans) {
	const CCDIK_Joint_Data &ccdik_data = ccdik_data_chain[p_joint_idx];
	Skeleton3D *skeleton = get_skeleton();

	if (skeleton == nullptr || !_cache_bone(ccdik_data.bone_idx, ccdik_data.bone_name)) {
		WARN_PRINT_ONCE(String("CCDIK joint: bone index for joint") + itos(p_joint_idx) + " not found. Cannot execute modification!");
		return;
	}

	Transform3D bone_trans;

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
	skeleton->set_bone_pose_rotation(ccdik_data.bone_idx, bone_trans.basis.get_rotation_quaternion());
}

void SkeletonModification3DCCDIK::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	if (target_node != NodePath()) {
		target_bone = String();
	}
	target_cache = Variant();
}

NodePath SkeletonModification3DCCDIK::get_target_node() const {
	return target_node;
}

void SkeletonModification3DCCDIK::set_target_bone(const String &p_target_bone) {
	target_bone = p_target_bone;
	if (target_bone != String()) {
		target_node = NodePath();
	}
	target_cache = Variant();
}

String SkeletonModification3DCCDIK::get_target_bone() const {
	return target_bone;
}

void SkeletonModification3DCCDIK::set_tip_node(const NodePath &p_tip_node) {
	tip_node = p_tip_node;
	if (tip_node != NodePath()) {
		tip_bone = String();
	}
	tip_cache = Variant();
}

NodePath SkeletonModification3DCCDIK::get_tip_node() const {
	return tip_node;
}

void SkeletonModification3DCCDIK::set_tip_bone(const String &p_tip_bone) {
	tip_bone = p_tip_bone;
	if (tip_bone != String()) {
		tip_node = NodePath();
	}
	tip_cache = Variant();
}

String SkeletonModification3DCCDIK::get_tip_bone() const {
	return tip_bone;
}

void SkeletonModification3DCCDIK::set_use_high_quality_solve(bool p_high_quality) {
	use_high_quality_solve = p_high_quality;
}

bool SkeletonModification3DCCDIK::get_use_high_quality_solve() const {
	return use_high_quality_solve;
}

// CCDIK joint data functions
String SkeletonModification3DCCDIK::get_joint_bone(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, String());
	return ccdik_data_chain[p_joint_idx].bone_name;
}

void SkeletonModification3DCCDIK::set_joint_bone(int p_joint_idx, String p_bone_name) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ccdik_data_chain[p_joint_idx].bone_name = p_bone_name;
	ccdik_data_chain[p_joint_idx].bone_idx = UNCACHED_BONE_IDX;
}

int SkeletonModification3DCCDIK::get_joint_ccdik_axis(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, -1);
	return ccdik_data_chain[p_joint_idx].ccdik_axis;
}

void SkeletonModification3DCCDIK::set_joint_ccdik_axis(int p_joint_idx, int p_axis) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ERR_FAIL_COND_MSG(p_axis < 0, "CCDIK axis is out of range: The axis mode is too low!");
	ccdik_data_chain[p_joint_idx].ccdik_axis = p_axis;
	notify_property_list_changed();
}

bool SkeletonModification3DCCDIK::get_joint_enable_constraint(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return ccdik_data_chain[p_joint_idx].enable_constraint;
}

void SkeletonModification3DCCDIK::set_joint_enable_constraint(int p_joint_idx, bool p_enable) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ccdik_data_chain[p_joint_idx].enable_constraint = p_enable;
	notify_property_list_changed();
}

real_t SkeletonModification3DCCDIK::get_joint_constraint_angle_min(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return ccdik_data_chain[p_joint_idx].constraint_angle_min;
}

void SkeletonModification3DCCDIK::set_joint_constraint_angle_min(int p_joint_idx, real_t p_angle_min) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ccdik_data_chain[p_joint_idx].constraint_angle_min = p_angle_min;
}

real_t SkeletonModification3DCCDIK::get_joint_constraint_angle_max(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return ccdik_data_chain[p_joint_idx].constraint_angle_max;
}

void SkeletonModification3DCCDIK::set_joint_constraint_angle_max(int p_joint_idx, real_t p_angle_max) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ccdik_data_chain[p_joint_idx].constraint_angle_max = p_angle_max;
}

bool SkeletonModification3DCCDIK::get_joint_constraint_invert(int p_joint_idx) const {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return ccdik_data_chain[p_joint_idx].constraint_angles_invert;
}

void SkeletonModification3DCCDIK::set_joint_constraint_invert(int p_joint_idx, bool p_invert) {
	const int bone_chain_size = ccdik_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	ccdik_data_chain[p_joint_idx].constraint_angles_invert = p_invert;
}

int SkeletonModification3DCCDIK::get_joint_count() {
	return ccdik_data_chain.size();
}
void SkeletonModification3DCCDIK::set_joint_count(int p_length) {
	ERR_FAIL_COND(p_length < 0);
	ccdik_data_chain.resize(p_length);
	notify_property_list_changed();
}

void SkeletonModification3DCCDIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification3DCCDIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification3DCCDIK::get_target_node);
	ClassDB::bind_method(D_METHOD("set_target_bone", "target_bone_name"), &SkeletonModification3DCCDIK::set_target_bone);
	ClassDB::bind_method(D_METHOD("get_target_bone"), &SkeletonModification3DCCDIK::get_target_bone);

	ClassDB::bind_method(D_METHOD("set_tip_node", "tip_nodepath"), &SkeletonModification3DCCDIK::set_tip_node);
	ClassDB::bind_method(D_METHOD("get_tip_node"), &SkeletonModification3DCCDIK::get_tip_node);
	ClassDB::bind_method(D_METHOD("set_tip_bone", "tip_bone_name"), &SkeletonModification3DCCDIK::set_tip_bone);
	ClassDB::bind_method(D_METHOD("get_tip_bone"), &SkeletonModification3DCCDIK::get_tip_bone);

	ClassDB::bind_method(D_METHOD("set_use_high_quality_solve", "high_quality_solve"), &SkeletonModification3DCCDIK::set_use_high_quality_solve);
	ClassDB::bind_method(D_METHOD("get_use_high_quality_solve"), &SkeletonModification3DCCDIK::get_use_high_quality_solve);

	// CCDIK joint data functions
	ClassDB::bind_method(D_METHOD("get_joint_bone", "joint_idx"), &SkeletonModification3DCCDIK::get_joint_bone);
	ClassDB::bind_method(D_METHOD("set_joint_bone", "joint_idx", "bone_name"), &SkeletonModification3DCCDIK::set_joint_bone);
	ClassDB::bind_method(D_METHOD("get_joint_ccdik_axis", "joint_idx"), &SkeletonModification3DCCDIK::get_joint_ccdik_axis);
	ClassDB::bind_method(D_METHOD("set_joint_ccdik_axis", "joint_idx", "axis"), &SkeletonModification3DCCDIK::set_joint_ccdik_axis);
	ClassDB::bind_method(D_METHOD("get_joint_enable_joint_constraint", "joint_idx"), &SkeletonModification3DCCDIK::get_joint_enable_constraint);
	ClassDB::bind_method(D_METHOD("set_joint_enable_joint_constraint", "joint_idx", "enable"), &SkeletonModification3DCCDIK::set_joint_enable_constraint);
	ClassDB::bind_method(D_METHOD("get_joint_constraint_angle_min", "joint_idx"), &SkeletonModification3DCCDIK::get_joint_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("set_joint_constraint_angle_min", "joint_idx", "min_angle"), &SkeletonModification3DCCDIK::set_joint_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("get_joint_constraint_angle_max", "joint_idx"), &SkeletonModification3DCCDIK::get_joint_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("set_joint_constraint_angle_max", "joint_idx", "max_angle"), &SkeletonModification3DCCDIK::set_joint_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("get_joint_constraint_invert", "joint_idx"), &SkeletonModification3DCCDIK::get_joint_constraint_invert);
	ClassDB::bind_method(D_METHOD("set_joint_constraint_invert", "joint_idx", "invert"), &SkeletonModification3DCCDIK::set_joint_constraint_invert);

	ClassDB::bind_method(D_METHOD("set_joint_count", "ccdik_chain_length"), &SkeletonModification3DCCDIK::set_joint_count);
	ClassDB::bind_method(D_METHOD("get_joint_count"), &SkeletonModification3DCCDIK::get_joint_count);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "target_bone", PROPERTY_HINT_NONE, ""), "set_target_bone", "get_target_bone");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "tip_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_tip_node", "get_tip_node");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "tip_bone", PROPERTY_HINT_NONE, ""), "set_tip_bone", "get_tip_bone");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "high_quality_solve", PROPERTY_HINT_NONE, ""), "set_use_high_quality_solve", "get_use_high_quality_solve");
	ADD_ARRAY_COUNT("CCDIK Joint Chain", "joint_count", "set_joint_count", "get_joint_count", "joint_");
}

SkeletonModification3DCCDIK::SkeletonModification3DCCDIK() {
}

SkeletonModification3DCCDIK::~SkeletonModification3DCCDIK() {
}

void SkeletonModification3DCCDIK::execute(real_t delta) {
	SkeletonModification3D::execute(delta);

	Transform3D target_transform = get_target_transform(target_cache);
	Transform3D tip_transform = get_target_transform(tip_cache);
	if (use_high_quality_solve) {
		for (uint32_t i = 0; i < ccdik_data_chain.size(); i++) {
			for (uint32_t j = i; j < ccdik_data_chain.size(); j++) {
				_execute_ccdik_joint(j, target_transform, tip_transform);
			}
		}
	} else {
		for (uint32_t i = 0; i < ccdik_data_chain.size(); i++) {
			_execute_ccdik_joint(i, target_transform, tip_transform);
		}
	}
}

void SkeletonModification3DCCDIK::skeleton_changed(Skeleton3D *skeleton) {
	target_cache = Variant();
	tip_cache = Variant();
	for (uint32_t i = 0; i < ccdik_data_chain.size(); i++) {
		ccdik_data_chain[i].bone_idx = UNCACHED_BONE_IDX;
	}
	SkeletonModification3D::skeleton_changed(skeleton);
}

bool SkeletonModification3DCCDIK::is_property_hidden(String p_property_name) const {
	if ((p_property_name == "target_bone" && !target_node.is_empty()) ||
			(p_property_name == "tip_bone" && !tip_node.is_empty())) {
		return true;
	}
	return SkeletonModification3D::is_property_hidden(p_property_name);
}

bool SkeletonModification3DCCDIK::is_bone_property(String p_property_name) const {
	if (p_property_name == "target_bone" || p_property_name == "tip_bone" || p_property_name.ends_with("/bone")) {
		return true;
	}
	return SkeletonModification3D::is_bone_property(p_property_name);
}

PackedStringArray SkeletonModification3DCCDIK::get_configuration_warnings() const {
	PackedStringArray ret = SkeletonModification3D::get_configuration_warnings();
	if (!get_skeleton()) {
		return ret;
	}
	if (!_cache_target(target_cache, target_node, target_bone)) {
		ret.append(vformat("Target %s %s was not found.", target_node.is_empty() ? "bone" : "node", target_node.is_empty() ? target_bone : (String)target_node));
	}
	if (!_cache_target(tip_cache, tip_node, tip_bone)) {
		ret.append(vformat("Tip %s %s was not found.", tip_node.is_empty() ? "bone" : "node", tip_node.is_empty() ? tip_bone : (String)tip_node));
	}
	for (uint32_t i = 0; i < ccdik_data_chain.size(); i++) {
		if (!_cache_bone(ccdik_data_chain[i].bone_idx, ccdik_data_chain[i].bone_name)) {
			ret.append(vformat("Joint %d bone %s was not found.", i, ccdik_data_chain[i].bone_name));
		}
	}
	return ret;
}
