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

#include "scene/resources/skeleton_modification_3d_lookat.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/skeleton_modification_3d.h"

bool SkeletonModification3DLookAt::_set(const StringName &p_path, const Variant &p_value) {
	if (p_path == "lock_rotation_to_plane") {
		set_lock_rotation_to_plane(p_value);
	} else if (p_path == "lock_rotation_plane") {
		set_lock_rotation_plane(p_value);
	} else if (p_path == "additional_rotation") {
		Vector3 tmp = p_value;
		tmp.x = Math::deg2rad(tmp.x);
		tmp.y = Math::deg2rad(tmp.y);
		tmp.z = Math::deg2rad(tmp.z);
		set_additional_rotation(tmp);
	}

	return true;
}

bool SkeletonModification3DLookAt::_get(const StringName &p_path, Variant &r_ret) const {
	if (p_path == "lock_rotation_to_plane") {
		r_ret = get_lock_rotation_to_plane();
	} else if (p_path == "lock_rotation_plane") {
		r_ret = get_lock_rotation_plane();
	} else if (p_path == "additional_rotation") {
		Vector3 tmp = get_additional_rotation();
		tmp.x = Math::rad2deg(tmp.x);
		tmp.y = Math::rad2deg(tmp.y);
		tmp.z = Math::rad2deg(tmp.z);
		r_ret = tmp;
	}

	return true;
}

void SkeletonModification3DLookAt::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::BOOL, "lock_rotation_to_plane", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	if (lock_rotation_to_plane) {
		p_list->push_back(PropertyInfo(Variant::INT, "lock_rotation_plane", PROPERTY_HINT_ENUM, "X plane, Y plane, Z plane", PROPERTY_USAGE_DEFAULT));
	}
	p_list->push_back(PropertyInfo(Variant::VECTOR3, "additional_rotation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
}

void SkeletonModification3DLookAt::_execute(real_t p_delta) {
	ERR_FAIL_COND_MSG(!stack || !is_setup || stack->skeleton == nullptr,
			"Modification is not setup and therefore cannot execute!");
	if (!enabled) {
		return;
	}

	if (target_node_cache.is_null()) {
		_print_execution_error(true, "Target cache is out of date. Attempting to update...");
		update_cache();
		return;
	}

	if (bone_idx <= -2) {
		bone_idx = stack->skeleton->find_bone(bone_name);
	}

	Node3D *target = Object::cast_to<Node3D>(ObjectDB::get_instance(target_node_cache));
	if (_print_execution_error(!target || !target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!")) {
		return;
	}
	if (_print_execution_error(bone_idx <= -1, "Bone index is invalid. Cannot execute modification!")) {
		return;
	}
	Transform3D new_bone_trans = stack->skeleton->get_bone_local_pose_override(bone_idx);
	if (new_bone_trans == Transform3D()) {
		new_bone_trans = stack->skeleton->get_bone_pose(bone_idx);
	}
	Vector3 target_pos = stack->skeleton->global_pose_to_local_pose(bone_idx, stack->skeleton->world_transform_to_global_pose(target->get_global_transform())).origin;

	// Lock the rotation to a plane relative to the bone by changing the target position
	if (lock_rotation_to_plane) {
		if (lock_rotation_plane == ROTATION_PLANE::ROTATION_PLANE_X) {
			target_pos.x = new_bone_trans.origin.x;
		} else if (lock_rotation_plane == ROTATION_PLANE::ROTATION_PLANE_Y) {
			target_pos.y = new_bone_trans.origin.y;
		} else if (lock_rotation_plane == ROTATION_PLANE::ROTATION_PLANE_Z) {
			target_pos.z = new_bone_trans.origin.z;
		}
	}

	// Look at the target!
	new_bone_trans = new_bone_trans.looking_at(target_pos, Vector3(0, 1, 0));
	// Convert from Z-forward to whatever direction the bone faces.
	stack->skeleton->update_bone_rest_forward_vector(bone_idx);
	new_bone_trans.basis = stack->skeleton->global_pose_z_forward_to_bone_forward(bone_idx, new_bone_trans.basis);

	// Apply additional rotation
	new_bone_trans.basis.rotate_local(Vector3(1, 0, 0), additional_rotation.x);
	new_bone_trans.basis.rotate_local(Vector3(0, 1, 0), additional_rotation.y);
	new_bone_trans.basis.rotate_local(Vector3(0, 0, 1), additional_rotation.z);

	stack->skeleton->set_bone_local_pose_override(bone_idx, new_bone_trans, stack->strength, true);
	stack->skeleton->force_update_bone_children_transforms(bone_idx);

	// If we completed it successfully, then we can set execution_error_found to false
	execution_error_found = false;
}

void SkeletonModification3DLookAt::_setup_modification(SkeletonModificationStack3D *p_stack) {
	stack = p_stack;

	if (stack != nullptr) {
		is_setup = true;
		execution_error_found = false;
		update_cache();
	}
}

void SkeletonModification3DLookAt::set_bone_name(String p_name) {
	bone_name = p_name;
	if (stack) {
		if (stack->skeleton) {
			bone_idx = stack->skeleton->find_bone(bone_name);
		}
	}
	execution_error_found = false;
	notify_property_list_changed();
}

String SkeletonModification3DLookAt::get_bone_name() const {
	return bone_name;
}

int SkeletonModification3DLookAt::get_bone_index() const {
	return bone_idx;
}

void SkeletonModification3DLookAt::set_bone_index(int p_bone_idx) {
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");
	bone_idx = p_bone_idx;

	if (stack) {
		if (stack->skeleton) {
			bone_name = stack->skeleton->get_bone_name(p_bone_idx);
		}
	}
	execution_error_found = false;
	notify_property_list_changed();
}

void SkeletonModification3DLookAt::update_cache() {
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
						"Cannot update target cache: Node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: Node is not in the scene tree!");
				target_node_cache = node->get_instance_id();

				execution_error_found = false;
			}
		}
	}
}

void SkeletonModification3DLookAt::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_cache();
}

NodePath SkeletonModification3DLookAt::get_target_node() const {
	return target_node;
}

Vector3 SkeletonModification3DLookAt::get_additional_rotation() const {
	return additional_rotation;
}

void SkeletonModification3DLookAt::set_additional_rotation(Vector3 p_offset) {
	additional_rotation = p_offset;
}

bool SkeletonModification3DLookAt::get_lock_rotation_to_plane() const {
	return lock_rotation_plane;
}

void SkeletonModification3DLookAt::set_lock_rotation_to_plane(bool p_lock_rotation) {
	lock_rotation_to_plane = p_lock_rotation;
	notify_property_list_changed();
}

int SkeletonModification3DLookAt::get_lock_rotation_plane() const {
	return lock_rotation_plane;
}

void SkeletonModification3DLookAt::set_lock_rotation_plane(int p_plane) {
	lock_rotation_plane = p_plane;
}

void SkeletonModification3DLookAt::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bone_name", "name"), &SkeletonModification3DLookAt::set_bone_name);
	ClassDB::bind_method(D_METHOD("get_bone_name"), &SkeletonModification3DLookAt::get_bone_name);

	ClassDB::bind_method(D_METHOD("set_bone_index", "bone_idx"), &SkeletonModification3DLookAt::set_bone_index);
	ClassDB::bind_method(D_METHOD("get_bone_index"), &SkeletonModification3DLookAt::get_bone_index);

	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification3DLookAt::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification3DLookAt::get_target_node);

	ClassDB::bind_method(D_METHOD("set_additional_rotation", "additional_rotation"), &SkeletonModification3DLookAt::set_additional_rotation);
	ClassDB::bind_method(D_METHOD("get_additional_rotation"), &SkeletonModification3DLookAt::get_additional_rotation);

	ClassDB::bind_method(D_METHOD("set_lock_rotation_to_plane", "lock_to_plane"), &SkeletonModification3DLookAt::set_lock_rotation_to_plane);
	ClassDB::bind_method(D_METHOD("get_lock_rotation_to_plane"), &SkeletonModification3DLookAt::get_lock_rotation_to_plane);
	ClassDB::bind_method(D_METHOD("set_lock_rotation_plane", "plane"), &SkeletonModification3DLookAt::set_lock_rotation_plane);
	ClassDB::bind_method(D_METHOD("get_lock_rotation_plane"), &SkeletonModification3DLookAt::get_lock_rotation_plane);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "bone_name"), "set_bone_name", "get_bone_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone_index"), "set_bone_index", "get_bone_index");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_target_node", "get_target_node");
}

SkeletonModification3DLookAt::SkeletonModification3DLookAt() {
	stack = nullptr;
	is_setup = false;
	bone_name = "";
	bone_idx = -2;
	additional_rotation = Vector3();
	lock_rotation_to_plane = false;
	enabled = true;
}

SkeletonModification3DLookAt::~SkeletonModification3DLookAt() {
}
