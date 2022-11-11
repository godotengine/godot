/*************************************************************************/
/*  skeleton_modification_3d_lookat.cpp                                  */
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

#include "scene/3d/skeleton_modification_3d_lookat.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/skeleton_modification_3d.h"

void SkeletonModification3DLookAt::set_bone(const String &p_name) {
	bone_name = p_name;
	bone_idx = UNCACHED_BONE_IDX;
}

String SkeletonModification3DLookAt::get_bone() const {
	return bone_name;
}

void SkeletonModification3DLookAt::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	if (!p_target_node.is_empty()) {
		target_bone = StringName();
	}
	target_cache = Variant();
}

NodePath SkeletonModification3DLookAt::get_target_node() const {
	return target_node;
}

void SkeletonModification3DLookAt::set_target_bone(const String &p_target_bone) {
	target_bone = p_target_bone;
	if (!p_target_bone.is_empty()) {
		target_node = NodePath();
	}
	target_cache = Variant();
}

String SkeletonModification3DLookAt::get_target_bone() const {
	return target_bone;
}

Vector3 SkeletonModification3DLookAt::get_additional_rotation() const {
	return additional_rotation;
}

void SkeletonModification3DLookAt::set_additional_rotation(Vector3 p_offset) {
	additional_rotation = p_offset;
}

SkeletonModification3DLookAt::LockRotationPlane SkeletonModification3DLookAt::get_lock_rotation_plane() const {
	return lock_rotation_plane;
}

void SkeletonModification3DLookAt::set_lock_rotation_plane(LockRotationPlane p_plane) {
	lock_rotation_plane = p_plane;
}

void SkeletonModification3DLookAt::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bone", "bone_name"), &SkeletonModification3DLookAt::set_bone);
	ClassDB::bind_method(D_METHOD("get_bone"), &SkeletonModification3DLookAt::get_bone);

	ClassDB::bind_method(D_METHOD("set_target_node", "nodepath"), &SkeletonModification3DLookAt::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification3DLookAt::get_target_node);
	ClassDB::bind_method(D_METHOD("set_target_bone", "bone_name"), &SkeletonModification3DLookAt::set_target_bone);
	ClassDB::bind_method(D_METHOD("get_target_bone"), &SkeletonModification3DLookAt::get_target_bone);

	ClassDB::bind_method(D_METHOD("set_additional_rotation", "additional_rotation"), &SkeletonModification3DLookAt::set_additional_rotation);
	ClassDB::bind_method(D_METHOD("get_additional_rotation"), &SkeletonModification3DLookAt::get_additional_rotation);

	ClassDB::bind_method(D_METHOD("set_lock_rotation_plane", "plane"), &SkeletonModification3DLookAt::set_lock_rotation_plane);
	ClassDB::bind_method(D_METHOD("get_lock_rotation_plane"), &SkeletonModification3DLookAt::get_lock_rotation_plane);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "bone"), "set_bone", "get_bone");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "target_bone"), "set_target_bone", "get_target_bone");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "lock_rotation_plane", PROPERTY_HINT_ENUM, "Unlocked,X plane,Y plane,Z plane", PROPERTY_USAGE_DEFAULT), "set_lock_rotation_plane", "get_lock_rotation_plane");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "additional_rotation", PROPERTY_HINT_NONE, "radians", PROPERTY_USAGE_DEFAULT), "set_additional_rotation", "get_additional_rotation");

	BIND_ENUM_CONSTANT(ROTATION_UNLOCKED);
	BIND_ENUM_CONSTANT(ROTATION_PLANE_X);
	BIND_ENUM_CONSTANT(ROTATION_PLANE_Y);
	BIND_ENUM_CONSTANT(ROTATION_PLANE_Z);
}

void SkeletonModification3DLookAt::execute(real_t delta) {
	SkeletonModification3D::execute(delta);

	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton ||
			!_cache_target(target_cache, target_node, target_bone) ||
			!_cache_bone(bone_idx, bone_name)) {
		WARN_PRINT_ONCE("SkeletonModification3DLookAt not initialized in time to execute");
		return;
	}
	Transform3D new_bone_trans = skeleton->get_bone_pose(bone_idx);
	Vector3 target_pos = get_bone_transform(bone_idx).affine_inverse().xform(get_target_transform(target_cache).get_origin());

	// Lock the rotation to a plane relative to the bone by changing the target position
	if (lock_rotation_plane == LockRotationPlane::ROTATION_PLANE_X) {
		target_pos.x = new_bone_trans.origin.x;
	} else if (lock_rotation_plane == LockRotationPlane::ROTATION_PLANE_Y) {
		target_pos.y = new_bone_trans.origin.y;
	} else if (lock_rotation_plane == LockRotationPlane::ROTATION_PLANE_Z) {
		target_pos.z = new_bone_trans.origin.z;
	}

	// Look at the target!
	new_bone_trans = new_bone_trans.looking_at(target_pos, Vector3(0, 1, 0));
	// Convert from Z-forward to whatever direction the bone faces.
	;
	new_bone_trans.basis = global_pose_z_forward_to_bone_forward(get_bone_rest_forward_vector(bone_idx), new_bone_trans.basis);

	// Apply additional rotation
	new_bone_trans.basis.rotate_local(Vector3(1, 0, 0), additional_rotation.x);
	new_bone_trans.basis.rotate_local(Vector3(0, 1, 0), additional_rotation.y);
	new_bone_trans.basis.rotate_local(Vector3(0, 0, 1), additional_rotation.z);
	skeleton->set_bone_pose_rotation(bone_idx, new_bone_trans.basis.get_rotation_quaternion());
}

void SkeletonModification3DLookAt::skeleton_changed(Skeleton3D *skeleton) {
	target_cache = Variant();
	bone_idx = UNCACHED_BONE_IDX;
	SkeletonModification3D::skeleton_changed(skeleton);
}

bool SkeletonModification3DLookAt::is_bone_property(String p_property_name) const {
	if (p_property_name == "target_bone" || p_property_name == "bone") {
		return true;
	}
	return SkeletonModification3D::is_bone_property(p_property_name);
}

bool SkeletonModification3DLookAt::is_property_hidden(String p_property_name) const {
	if (p_property_name == "target_bone" && !target_node.is_empty()) {
		return true;
	}
	return SkeletonModification3D::is_property_hidden(p_property_name);
}

PackedStringArray SkeletonModification3DLookAt::get_configuration_warnings() const {
	PackedStringArray ret = SkeletonModification3D::get_configuration_warnings();
	if (!get_skeleton()) {
		return ret;
	}
	if (!_cache_target(target_cache, target_node, target_bone)) {
		ret.append(vformat("Target %s %s was not found.", target_node.is_empty() ? "bone" : "node", target_node.is_empty() ? target_bone : (String)target_node));
	}
	if (!_cache_bone(bone_idx, bone_name)) {
		ret.append(vformat("Bone %s was not found.", target_bone));
	}
	return ret;
}
