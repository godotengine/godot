/*************************************************************************/
/*  skeleton_modification_2d_ccdik.cpp                                   */
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

#include "skeleton_modification_2d_ccdik.h"
#include "scene/2d/skeleton_2d.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif // TOOLS_ENABLED

bool SkeletonModification2DCCDIK::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_")) {
		int which = path.get_slicec('/', 0).substr(6).to_int();
		String what = path.get_slicec('/', 1);
		ERR_FAIL_INDEX_V(which, ccdik_data_chain.size(), false);

		if (what == "bone_node") {
			set_joint_bone_node(which, p_value);
		} else if (what == "rotate_from_joint") {
			set_joint_rotate_from_joint(which, p_value);
		} else if (what == "enable_constraint") {
			set_joint_enable_constraint(which, p_value);
		} else if (what == "constraint_angle_min") {
			set_joint_constraint_angle_min(which, float(p_value));
		} else if (what == "constraint_angle_max") {
			set_joint_constraint_angle_max(which, float(p_value));
		} else if (what == "constraint_angle_invert") {
			set_joint_constraint_angle_invert(which, p_value);
		} else if (what == "constraint_in_localspace") {
			set_joint_constraint_in_localspace(which, p_value);
		}

		return true;
	}

	return true;
}

bool SkeletonModification2DCCDIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_")) {
		int which = path.get_slicec('/', 0).substr(6).to_int();
		String what = path.get_slicec('/', 1);
		ERR_FAIL_INDEX_V(which, ccdik_data_chain.size(), false);

		if (what == "bone_node") {
			r_ret = get_joint_bone_node(which);
		} else if (what == "rotate_from_joint") {
			r_ret = get_joint_rotate_from_joint(which);
		} else if (what == "enable_constraint") {
			r_ret = get_joint_enable_constraint(which);
		} else if (what == "constraint_angle_min") {
			r_ret = get_joint_constraint_angle_min(which);
		} else if (what == "constraint_angle_max") {
			r_ret = get_joint_constraint_angle_max(which);
		} else if (what == "constraint_angle_invert") {
			r_ret = get_joint_constraint_angle_invert(which);
		} else if (what == "constraint_in_localspace") {
			r_ret = get_joint_constraint_in_localspace(which);
		}
		return true;
	}

	return true;
}

void SkeletonModification2DCCDIK::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < ccdik_data_chain.size(); i++) {
		String base_string = "joint_" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::NODE_PATH, base_string + "bone_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Bone2D", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "rotate_from_joint", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));

		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "enable_constraint", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		if (ccdik_data_chain[i].enable_constraint) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "constraint_angle_min", PROPERTY_HINT_RANGE, "-360,360,0.01,radians", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "constraint_angle_max", PROPERTY_HINT_RANGE, "-360,360,0.01,radians", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "constraint_angle_invert", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "constraint_in_localspace", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
	}
}

void SkeletonModification2DCCDIK::_execute_ccdik_joint(int p_joint_idx, Vector2 p_target_position, Vector2 p_tip_position) {
	const CCDIK_Joint_Data2D &ccdik_data = ccdik_data_chain[p_joint_idx];

	Bone2D *operation_bone = _cache_bone(ccdik_data.bone_node_cache, ccdik_data.bone_node);
	if (operation_bone == nullptr) {
		return;
	}
	Transform2D operation_transform = get_target_transform(operation_bone);

	if (ccdik_data.rotate_from_joint) {
		// To rotate from the joint, simply look at the target!
		operation_transform.set_rotation(
				operation_transform.looking_at(p_target_position).get_rotation() - operation_bone->get_bone_angle());
	} else {
		// How to rotate from the tip: get the difference of rotation needed from the tip to the target, from the perspective of the joint.
		// Because we are only using the offset, we do not need to account for the bone angle of the Bone2D node.
		float joint_to_tip = p_tip_position.angle_to_point(operation_transform.get_origin());
		float joint_to_target = p_target_position.angle_to_point(operation_transform.get_origin());
		operation_transform.set_rotation(
				operation_transform.get_rotation() + (joint_to_target - joint_to_tip));
	}

	// Reset scale
	operation_transform.set_scale(operation_bone->get_global_scale());

	// Apply constraints in globalspace:
	if (ccdik_data.enable_constraint && !ccdik_data.constraint_in_localspace) {
		operation_transform.set_rotation(clamp_angle(operation_transform.get_rotation(), ccdik_data.constraint_angle_min, ccdik_data.constraint_angle_max, ccdik_data.constraint_angle_invert));
	}

	// Convert from a global transform to a delta and then apply the delta to the local transform.
	float parent_rotation = 0.0f;
	CanvasItem *parent_node = cast_to<CanvasItem>(operation_bone->get_parent());
	if (parent_node != nullptr) {
		parent_rotation = get_target_rotation(parent_node);
	}
	float final_rotation = operation_transform.get_rotation() - parent_rotation;

	// Apply constraints in localspace:
	if (ccdik_data.enable_constraint && ccdik_data.constraint_in_localspace) {
		final_rotation = clamp_angle(final_rotation, ccdik_data.constraint_angle_min, ccdik_data.constraint_angle_max, ccdik_data.constraint_angle_invert);
	}

	// Set the local pose override, and to make sure child bones are also updated, set the transform of the bone.
	operation_bone->set_rotation(final_rotation);
}

void SkeletonModification2DCCDIK::draw_editor_gizmo() {
	for (int i = 0; i < ccdik_data_chain.size(); i++) {
		Bone2D *operation_bone = _cache_bone(ccdik_data_chain[i].bone_node_cache, ccdik_data_chain[i].bone_node);
		editor_draw_angle_constraints(operation_bone, ccdik_data_chain[i].constraint_angle_min, ccdik_data_chain[i].constraint_angle_max,
				ccdik_data_chain[i].enable_constraint, ccdik_data_chain[i].constraint_in_localspace, ccdik_data_chain[i].constraint_angle_invert);
	}
}

void SkeletonModification2DCCDIK::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	target_node_cache = Variant();
}

NodePath SkeletonModification2DCCDIK::get_target_node() const {
	return target_node;
}

void SkeletonModification2DCCDIK::set_tip_node(const NodePath &p_tip_node) {
	tip_node = p_tip_node;
	tip_node_cache = Variant();
}

NodePath SkeletonModification2DCCDIK::get_tip_node() const {
	return tip_node;
}

void SkeletonModification2DCCDIK::set_joint_count(int p_length) {
	ccdik_data_chain.resize(p_length);
	notify_property_list_changed();
}

int SkeletonModification2DCCDIK::get_joint_count() {
	return ccdik_data_chain.size();
}

void SkeletonModification2DCCDIK::set_joint_bone_node(int p_joint_idx, const NodePath &p_target_node) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].bone_node = p_target_node;
	ccdik_data_chain.write[p_joint_idx].bone_node_cache = Variant();

	notify_property_list_changed();
}

NodePath SkeletonModification2DCCDIK::get_joint_bone_node(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), NodePath(), "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].bone_node;
}

void SkeletonModification2DCCDIK::set_joint_rotate_from_joint(int p_joint_idx, bool p_rotate_from_joint) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].rotate_from_joint = p_rotate_from_joint;
}

bool SkeletonModification2DCCDIK::get_joint_rotate_from_joint(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].rotate_from_joint;
}

void SkeletonModification2DCCDIK::set_joint_enable_constraint(int p_joint_idx, bool p_constraint) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].enable_constraint = p_constraint;
	notify_property_list_changed();
}

bool SkeletonModification2DCCDIK::get_joint_enable_constraint(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].enable_constraint;
}

void SkeletonModification2DCCDIK::set_joint_constraint_angle_min(int p_joint_idx, float p_angle_min) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].constraint_angle_min = p_angle_min;
}

float SkeletonModification2DCCDIK::get_joint_constraint_angle_min(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), 0.0, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].constraint_angle_min;
}

void SkeletonModification2DCCDIK::set_joint_constraint_angle_max(int p_joint_idx, float p_angle_max) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].constraint_angle_max = p_angle_max;
}

float SkeletonModification2DCCDIK::get_joint_constraint_angle_max(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), 0.0, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].constraint_angle_max;
}

void SkeletonModification2DCCDIK::set_joint_constraint_angle_invert(int p_joint_idx, bool p_invert) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].constraint_angle_invert = p_invert;
}

bool SkeletonModification2DCCDIK::get_joint_constraint_angle_invert(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].constraint_angle_invert;
}

void SkeletonModification2DCCDIK::set_joint_constraint_in_localspace(int p_joint_idx, bool p_constraint_in_localspace) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].constraint_in_localspace = p_constraint_in_localspace;
}

bool SkeletonModification2DCCDIK::get_joint_constraint_in_localspace(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].constraint_in_localspace;
}

void SkeletonModification2DCCDIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification2DCCDIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification2DCCDIK::get_target_node);
	ClassDB::bind_method(D_METHOD("set_tip_node", "tip_nodepath"), &SkeletonModification2DCCDIK::set_tip_node);
	ClassDB::bind_method(D_METHOD("get_tip_node"), &SkeletonModification2DCCDIK::get_tip_node);

	ClassDB::bind_method(D_METHOD("set_joint_count", "length"), &SkeletonModification2DCCDIK::set_joint_count);
	ClassDB::bind_method(D_METHOD("get_joint_count"), &SkeletonModification2DCCDIK::get_joint_count);

	ClassDB::bind_method(D_METHOD("set_joint_bone_node", "joint_idx", "bone2d_nodepath"), &SkeletonModification2DCCDIK::set_joint_bone_node);
	ClassDB::bind_method(D_METHOD("get_joint_bone_node", "joint_idx"), &SkeletonModification2DCCDIK::get_joint_bone_node);
	ClassDB::bind_method(D_METHOD("set_joint_rotate_from_joint", "joint_idx", "rotate_from_joint"), &SkeletonModification2DCCDIK::set_joint_rotate_from_joint);
	ClassDB::bind_method(D_METHOD("get_joint_rotate_from_joint", "joint_idx"), &SkeletonModification2DCCDIK::get_joint_rotate_from_joint);
	ClassDB::bind_method(D_METHOD("set_joint_enable_constraint", "joint_idx", "enable_constraint"), &SkeletonModification2DCCDIK::set_joint_enable_constraint);
	ClassDB::bind_method(D_METHOD("get_joint_enable_constraint", "joint_idx"), &SkeletonModification2DCCDIK::get_joint_enable_constraint);
	ClassDB::bind_method(D_METHOD("set_joint_constraint_angle_min", "joint_idx", "angle_min"), &SkeletonModification2DCCDIK::set_joint_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("get_joint_constraint_angle_min", "joint_idx"), &SkeletonModification2DCCDIK::get_joint_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("set_joint_constraint_angle_max", "joint_idx", "angle_max"), &SkeletonModification2DCCDIK::set_joint_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("get_joint_constraint_angle_max", "joint_idx"), &SkeletonModification2DCCDIK::get_joint_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("set_joint_constraint_angle_invert", "joint_idx", "invert"), &SkeletonModification2DCCDIK::set_joint_constraint_angle_invert);
	ClassDB::bind_method(D_METHOD("get_joint_constraint_angle_invert", "joint_idx"), &SkeletonModification2DCCDIK::get_joint_constraint_angle_invert);
	ClassDB::bind_method(D_METHOD("set_joint_constraint_in_localspace", "joint_idx", "use_local_coords"), &SkeletonModification2DCCDIK::set_joint_constraint_in_localspace);
	ClassDB::bind_method(D_METHOD("get_joint_constraint_in_localspace", "joint_idx"), &SkeletonModification2DCCDIK::get_joint_constraint_in_localspace);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "CanvasItem"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "tip_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "CanvasItem"), "set_tip_node", "get_tip_node");
	ADD_ARRAY_COUNT("CCDIK Joint Chain", "joint_count", "set_joint_count", "get_joint_count", "joint_");
}

SkeletonModification2DCCDIK::SkeletonModification2DCCDIK() {
}

SkeletonModification2DCCDIK::~SkeletonModification2DCCDIK() {
}

void SkeletonModification2DCCDIK::execute(real_t p_delta) {
	SkeletonModification2D::execute(p_delta);

	if (!_cache_node(target_node_cache, target_node) || !_cache_node(tip_node_cache, tip_node)) {
		WARN_PRINT_ONCE("2DCCDIK: Failed to get target and tip nodes");
		return;
	}
	for (int i = 0; i < ccdik_data_chain.size(); i++) {
		if (!_cache_bone(ccdik_data_chain[i].bone_node_cache, ccdik_data_chain[i].bone_node)) {
			WARN_PRINT_ONCE("2DCCDIK: Failed to get chain bone node");
			return;
		}
	}
	Vector2 target_transform = get_target_position(target_node_cache);
	Vector2 tip_transform = get_target_position(tip_node_cache);
	for (int i = 0; i < ccdik_data_chain.size(); i++) {
		_execute_ccdik_joint(i, target_transform, tip_transform);
	}
}

PackedStringArray SkeletonModification2DCCDIK::get_configuration_warnings() const {
	PackedStringArray ret = SkeletonModification2D::get_configuration_warnings();
	if (!get_skeleton()) {
		return ret;
	}
	if (!_cache_node(target_node_cache, target_node)) {
		ret.append(vformat("Target node %s was not found.", (String)target_node));
	}
	if (!_cache_node(tip_node_cache, tip_node)) {
		ret.append(vformat("Tip node %s was not found.", (String)tip_node));
	}
	for (int i = 0; i < ccdik_data_chain.size(); i++) {
		if (!_cache_bone(ccdik_data_chain[i].bone_node_cache, ccdik_data_chain[i].bone_node)) {
			ret.append(vformat("Joint %d bone %s was not found.", i, ccdik_data_chain[i].bone_node));
		}
	}
	return ret;
}
