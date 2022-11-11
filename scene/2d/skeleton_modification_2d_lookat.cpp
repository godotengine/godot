/*************************************************************************/
/*  skeleton_modification_2d_lookat.cpp                                  */
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

#include "skeleton_modification_2d_lookat.h"
#include "scene/2d/skeleton_2d.h"

bool SkeletonModification2DLookAt::is_property_hidden(String p_property_name) const {
	return (!enable_constraint && p_property_name.begins_with("constraint_"));
}

void SkeletonModification2DLookAt::draw_editor_gizmo() {
	SkeletonModification2D::draw_editor_gizmo();

	Bone2D *operation_bone = _cache_bone(bone_node_cache, bone_node);
	if (!operation_bone) {
		return;
	}
	editor_draw_angle_constraints(operation_bone, constraint_angle_min, constraint_angle_max,
			enable_constraint, constraint_in_localspace, constraint_angle_invert);
}

void SkeletonModification2DLookAt::set_bone_node(const NodePath &p_bone_node) {
	bone_node = p_bone_node;
	bone_node_cache = Variant();
}

NodePath SkeletonModification2DLookAt::get_bone_node() const {
	return bone_node;
}

void SkeletonModification2DLookAt::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	target_node_cache = Variant();
}

NodePath SkeletonModification2DLookAt::get_target_node() const {
	return target_node;
}

float SkeletonModification2DLookAt::get_additional_rotation() const {
	return additional_rotation;
}

void SkeletonModification2DLookAt::set_additional_rotation(float p_rotation) {
	additional_rotation = p_rotation;
}

void SkeletonModification2DLookAt::set_enable_constraint(bool p_constraint) {
	enable_constraint = p_constraint;
	notify_property_list_changed();
}

bool SkeletonModification2DLookAt::get_enable_constraint() const {
	return enable_constraint;
}

void SkeletonModification2DLookAt::set_constraint_angle_min(float p_angle_min) {
	constraint_angle_min = p_angle_min;
}

float SkeletonModification2DLookAt::get_constraint_angle_min() const {
	return constraint_angle_min;
}

void SkeletonModification2DLookAt::set_constraint_angle_max(float p_angle_max) {
	constraint_angle_max = p_angle_max;
}

float SkeletonModification2DLookAt::get_constraint_angle_max() const {
	return constraint_angle_max;
}

void SkeletonModification2DLookAt::set_constraint_angle_invert(bool p_invert) {
	constraint_angle_invert = p_invert;
}

bool SkeletonModification2DLookAt::get_constraint_angle_invert() const {
	return constraint_angle_invert;
}

void SkeletonModification2DLookAt::set_constraint_in_localspace(bool p_constraint_in_localspace) {
	constraint_in_localspace = p_constraint_in_localspace;
}

bool SkeletonModification2DLookAt::get_constraint_in_localspace() const {
	return constraint_in_localspace;
}

void SkeletonModification2DLookAt::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bone_node", "bone_nodepath"), &SkeletonModification2DLookAt::set_bone_node);
	ClassDB::bind_method(D_METHOD("get_bone_node"), &SkeletonModification2DLookAt::get_bone_node);

	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification2DLookAt::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification2DLookAt::get_target_node);

	ClassDB::bind_method(D_METHOD("set_additional_rotation", "rotation"), &SkeletonModification2DLookAt::set_additional_rotation);
	ClassDB::bind_method(D_METHOD("get_additional_rotation"), &SkeletonModification2DLookAt::get_additional_rotation);

	ClassDB::bind_method(D_METHOD("set_enable_constraint", "enable_constraint"), &SkeletonModification2DLookAt::set_enable_constraint);
	ClassDB::bind_method(D_METHOD("get_enable_constraint"), &SkeletonModification2DLookAt::get_enable_constraint);
	ClassDB::bind_method(D_METHOD("set_constraint_angle_min", "angle_min"), &SkeletonModification2DLookAt::set_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("get_constraint_angle_min"), &SkeletonModification2DLookAt::get_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("set_constraint_angle_max", "angle_max"), &SkeletonModification2DLookAt::set_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("get_constraint_angle_max"), &SkeletonModification2DLookAt::get_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("set_constraint_angle_invert", "invert"), &SkeletonModification2DLookAt::set_constraint_angle_invert);
	ClassDB::bind_method(D_METHOD("get_constraint_angle_invert"), &SkeletonModification2DLookAt::get_constraint_angle_invert);
	ClassDB::bind_method(D_METHOD("set_constraint_in_localspace", "is_local_space"), &SkeletonModification2DLookAt::set_constraint_in_localspace);
	ClassDB::bind_method(D_METHOD("get_constraint_in_localspace"), &SkeletonModification2DLookAt::get_constraint_in_localspace);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "bone_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Bone2D"), "set_bone_node", "get_bone_node");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "CanvasItem"), "set_target_node", "get_target_node");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_constraint", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), "set_enable_constraint", "get_enable_constraint");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "constraint_angle_min", PROPERTY_HINT_RANGE, "-360,360,0.01,radians", PROPERTY_USAGE_DEFAULT), "set_constraint_angle_min", "get_constraint_angle_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "constraint_angle_max", PROPERTY_HINT_RANGE, "-360,360,0.01,radians", PROPERTY_USAGE_DEFAULT), "set_constraint_angle_max", "get_constraint_angle_max");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "constraint_angle_invert", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), "set_constraint_angle_invert", "get_constraint_angle_invert");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "constraint_in_localspace", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), "set_constraint_in_localspace", "get_constraint_in_localspace");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "additional_rotation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), "set_additional_rotation", "get_additional_rotation");
}

SkeletonModification2DLookAt::SkeletonModification2DLookAt() {
}

SkeletonModification2DLookAt::~SkeletonModification2DLookAt() {
}

void SkeletonModification2DLookAt::execute(real_t delta) {
	SkeletonModification2D::execute(delta);

	if (!_cache_node(bone_node_cache, bone_node) || !_cache_node(target_node_cache, target_node)) {
		WARN_PRINT_ONCE("SkeletonModification2DLookAt not initialized in time to execute");
		return;
	}
	Bone2D *operation_bone = cast_to<Bone2D>((Object *)bone_node_cache);
	if (operation_bone == nullptr) {
		ERR_PRINT_ONCE("bone_idx for modification does not point to a valid bone! Cannot execute modification");
		return;
	}

	Transform2D operation_transform = operation_bone->get_global_transform();
	Transform2D target_trans = get_target_transform(target_node_cache);

	// Look at the target!
	operation_transform = operation_transform.looking_at(target_trans.get_origin());
	// Apply whatever scale it had prior to looking_at
	operation_transform.set_scale(operation_bone->get_global_scale());

	// Account for the direction the bone faces in:
	operation_transform.set_rotation(operation_transform.get_rotation() - operation_bone->get_bone_angle());

	// Apply additional rotation
	operation_transform.set_rotation(operation_transform.get_rotation() + additional_rotation);

	// Apply constraints in globalspace:
	if (enable_constraint && !constraint_in_localspace) {
		operation_transform.set_rotation(clamp_angle(operation_transform.get_rotation(), constraint_angle_min, constraint_angle_max, constraint_angle_invert));
	}

	// Convert from a global transform to a local transform via the Bone2D node
	operation_bone->set_global_transform(operation_transform);
	operation_transform = operation_bone->get_transform();

	// Apply constraints in localspace:
	if (enable_constraint && constraint_in_localspace) {
		operation_transform.set_rotation(clamp_angle(operation_transform.get_rotation(), constraint_angle_min, constraint_angle_max, constraint_angle_invert));
	}

	operation_bone->set_transform(operation_transform);
}

PackedStringArray SkeletonModification2DLookAt::get_configuration_warnings() const {
	PackedStringArray ret = SkeletonModification2D::get_configuration_warnings();
	if (!get_skeleton()) {
		return ret;
	}
	if (!_cache_node(target_node_cache, target_node)) {
		ret.append(vformat("Target node %s was not found.", (String)target_node));
	}
	if (!_cache_node(bone_node_cache, bone_node)) {
		ret.append(vformat("Bone2d node %s was not found.", (String)bone_node));
	}
	return ret;
}
