/*************************************************************************/
/*  skeleton_modification_3d.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "skeleton_modification_3d.h"
#include "scene/3d/skeleton_3d.h"

///////////////////////////////////////
// ModificationStack3D
///////////////////////////////////////

void SkeletonModificationStack3D::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < modifications.size(); i++) {
		p_list->push_back(
				PropertyInfo(Variant::OBJECT, "modifications/" + itos(i),
						PROPERTY_HINT_RESOURCE_TYPE,
						"SkeletonModification3D",
						PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DEFERRED_SET_RESOURCE));
	}
}

bool SkeletonModificationStack3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("modifications/")) {
		int mod_idx = path.get_slicec('/', 1).to_int();
		set_modification(mod_idx, p_value);
		return true;
	}
	return true;
}

bool SkeletonModificationStack3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("modifications/")) {
		int mod_idx = path.get_slicec('/', 1).to_int();
		r_ret = get_modification(mod_idx);
		return true;
	}
	return true;
}

void SkeletonModificationStack3D::setup() {
	if (is_setup) {
		return;
	}

	if (skeleton != nullptr) {
		is_setup = true;
		for (int i = 0; i < modifications.size(); i++) {
			if (!modifications[i].is_valid()) {
				continue;
			}
			modifications.get(i)->setup_modification(this);
		}
	} else {
		WARN_PRINT("Cannot setup SkeletonModificationStack3D: no skeleton set!");
	}
}

void SkeletonModificationStack3D::execute(float delta) {
	ERR_FAIL_COND_MSG(!is_setup || skeleton == nullptr || is_queued_for_deletion(),
			"Modification stack is not properly setup and therefore cannot execute!");

	if (!skeleton->is_inside_tree()) {
		ERR_PRINT_ONCE("Skeleton is not inside SceneTree! Cannot execute modification!");
		return;
	}

	if (!enabled) {
		return;
	}

	// NOTE: is needed for CCDIK.
	skeleton->clear_bones_local_pose_override();

	for (int i = 0; i < modifications.size(); i++) {
		if (!modifications[i].is_valid()) {
			continue;
		}
		modifications.get(i)->execute(delta);
	}
}

void SkeletonModificationStack3D::enable_all_modifications(bool p_enabled) {
	for (int i = 0; i < modifications.size(); i++) {
		if (!modifications[i].is_valid()) {
			continue;
		}
		modifications.get(i)->set_enabled(p_enabled);
	}
}

Ref<SkeletonModification3D> SkeletonModificationStack3D::get_modification(int p_mod_idx) const {
	ERR_FAIL_INDEX_V(p_mod_idx, modifications.size(), nullptr);
	return modifications[p_mod_idx];
}

void SkeletonModificationStack3D::add_modification(Ref<SkeletonModification3D> p_mod) {
	p_mod->setup_modification(this);
	modifications.push_back(p_mod);
}

void SkeletonModificationStack3D::delete_modification(int p_mod_idx) {
	ERR_FAIL_INDEX(p_mod_idx, modifications.size());
	modifications.remove(p_mod_idx);
}

void SkeletonModificationStack3D::set_modification(int p_mod_idx, Ref<SkeletonModification3D> p_mod) {
	ERR_FAIL_INDEX(p_mod_idx, modifications.size());

	if (p_mod == nullptr) {
		modifications.set(p_mod_idx, nullptr);
	} else {
		p_mod->setup_modification(this);
		modifications.set(p_mod_idx, p_mod);
	}
}

void SkeletonModificationStack3D::set_modification_count(int p_count) {
	modifications.resize(p_count);
	_change_notify();
}

int SkeletonModificationStack3D::get_modification_count() const {
	return modifications.size();
}

void SkeletonModificationStack3D::set_skeleton(Skeleton3D *p_skeleton) {
	skeleton = p_skeleton;
}

Skeleton3D *SkeletonModificationStack3D::get_skeleton() const {
	return skeleton;
}

bool SkeletonModificationStack3D::get_is_setup() const {
	return is_setup;
}

void SkeletonModificationStack3D::set_enabled(bool p_enabled) {
	enabled = p_enabled;

	if (!enabled && is_setup && skeleton != nullptr) {
		skeleton->clear_bones_local_pose_override();
	}
}

bool SkeletonModificationStack3D::get_enabled() const {
	return enabled;
}

void SkeletonModificationStack3D::set_strength(float p_strength) {
	ERR_FAIL_COND_MSG(p_strength < 0, "Strength cannot be less than zero!");
	ERR_FAIL_COND_MSG(p_strength > 1, "Strength cannot be more than one!");
	strength = p_strength;
}

float SkeletonModificationStack3D::get_strength() const {
	return strength;
}

void SkeletonModificationStack3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("setup"), &SkeletonModificationStack3D::setup);
	ClassDB::bind_method(D_METHOD("execute", "delta"), &SkeletonModificationStack3D::execute);

	ClassDB::bind_method(D_METHOD("enable_all_modifications", "enabled"), &SkeletonModificationStack3D::enable_all_modifications);
	ClassDB::bind_method(D_METHOD("get_modification", "mod_idx"), &SkeletonModificationStack3D::get_modification);
	ClassDB::bind_method(D_METHOD("add_modification", "modification"), &SkeletonModificationStack3D::add_modification);
	ClassDB::bind_method(D_METHOD("delete_modification", "mod_idx"), &SkeletonModificationStack3D::delete_modification);
	ClassDB::bind_method(D_METHOD("set_modification", "mod_idx", "modification"), &SkeletonModificationStack3D::set_modification);

	ClassDB::bind_method(D_METHOD("set_modification_count"), &SkeletonModificationStack3D::set_modification_count);
	ClassDB::bind_method(D_METHOD("get_modification_count"), &SkeletonModificationStack3D::get_modification_count);

	ClassDB::bind_method(D_METHOD("get_is_setup"), &SkeletonModificationStack3D::get_is_setup);

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SkeletonModificationStack3D::set_enabled);
	ClassDB::bind_method(D_METHOD("get_enabled"), &SkeletonModificationStack3D::get_enabled);

	ClassDB::bind_method(D_METHOD("set_strength", "strength"), &SkeletonModificationStack3D::set_strength);
	ClassDB::bind_method(D_METHOD("get_strength"), &SkeletonModificationStack3D::get_strength);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "strength", PROPERTY_HINT_RANGE, "0, 1, 0.001"), "set_strength", "get_strength");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "modification_count", PROPERTY_HINT_RANGE, "0, 100, 1"), "set_modification_count", "get_modification_count");
}

SkeletonModificationStack3D::SkeletonModificationStack3D() {
	skeleton = nullptr;
	modifications = Vector<Ref<SkeletonModification3D>>();
	is_setup = false;
	enabled = false;
	modifications_count = 0;
	strength = 0;
}

///////////////////////////////////////
// Modification3D
///////////////////////////////////////

void SkeletonModification3D::execute(float delta) {
	if (!enabled)
		return;
}

void SkeletonModification3D::setup_modification(SkeletonModificationStack3D *p_stack) {
	stack = p_stack;
	if (stack) {
		is_setup = true;
	}
}

void SkeletonModification3D::set_enabled(bool p_enabled) {
	enabled = p_enabled;
}

bool SkeletonModification3D::get_enabled() {
	return enabled;
}

void SkeletonModification3D::_bind_methods() {
	BIND_VMETHOD(MethodInfo("execute"));
	BIND_VMETHOD(MethodInfo("setup_modification"));

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SkeletonModification3D::set_enabled);
	ClassDB::bind_method(D_METHOD("get_enabled"), &SkeletonModification3D::get_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");
}

SkeletonModification3D::SkeletonModification3D() {
	stack = nullptr;
	is_setup = false;
}

///////////////////////////////////////
// LookAt
///////////////////////////////////////

void SkeletonModification3DLookAt::execute(float delta) {
	ERR_FAIL_COND_MSG(!stack || !is_setup || stack->skeleton == nullptr,
			"Modification is not setup and therefore cannot execute!");
	if (!enabled) {
		return;
	}

	if (target_node_cache.is_null()) {
		update_cache();
		WARN_PRINT("Target cache is out of date. Updating...");
		return;
	}

	if (bone_idx <= -2) {
		bone_idx = stack->skeleton->find_bone(bone_name);
	}

	Node3D *target = Object::cast_to<Node3D>(ObjectDB::get_instance(target_node_cache));
	ERR_FAIL_COND_MSG(!target, "Target node is not a Node3D-based node. Cannot execute modification!");
	ERR_FAIL_COND_MSG(!target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!");
	ERR_FAIL_COND_MSG(bone_idx <= -1, "Bone index is invalid. Cannot execute modification!");

	Transform new_bone_trans = stack->skeleton->local_pose_to_global_pose(bone_idx, stack->skeleton->get_bone_local_pose_override(bone_idx));
	Vector3 target_pos = stack->skeleton->world_transform_to_global_pose(target->get_global_transform()).origin;

	// Limit rotation by changing the target's position.
	// NOTE: this does not work when two limits are applied. Will need to adjust.
	if (lock_rotation_x) {
		target_pos.x = new_bone_trans.origin.x;
	}
	if (lock_rotation_y) {
		target_pos.y = new_bone_trans.origin.y;
	}
	if (lock_rotation_z) {
		target_pos.z = new_bone_trans.origin.z;
	}

	// Look at the target!
	new_bone_trans = new_bone_trans.looking_at(target_pos, Vector3(0, 1, 0));
	// Convert from Z-forward to whatever direction the bone faces.
	stack->skeleton->update_bone_rest_forward_vector(bone_idx);
	new_bone_trans.basis = stack->skeleton->global_pose_z_forward_to_bone_forward(bone_idx, new_bone_trans.basis);

	// Apply additional rotation
	new_bone_trans.basis.rotate_local(Vector3(1, 0, 0), Math::deg2rad(additional_rotation.x));
	new_bone_trans.basis.rotate_local(Vector3(0, 1, 0), Math::deg2rad(additional_rotation.y));
	new_bone_trans.basis.rotate_local(Vector3(0, 0, 1), Math::deg2rad(additional_rotation.z));

	new_bone_trans = stack->skeleton->global_pose_to_local_pose(bone_idx, new_bone_trans);
	stack->skeleton->set_bone_local_pose_override(bone_idx, new_bone_trans, stack->strength, true);
	stack->skeleton->force_update_bone_children_transforms(bone_idx);
}

void SkeletonModification3DLookAt::setup_modification(SkeletonModificationStack3D *p_stack) {
	stack = p_stack;

	if (stack != nullptr) {
		is_setup = true;
		update_cache();
	}
}

void SkeletonModification3DLookAt::set_bone_name(String p_name) {
	bone_name = p_name;
	if (stack && stack->skeleton) {
		bone_idx = stack->skeleton->find_bone(bone_name);
	}
	_change_notify();
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
	_change_notify();
}

void SkeletonModification3DLookAt::update_cache() {
	if (!is_setup || !stack) {
		WARN_PRINT("Cannot update cache: modification is not properly setup!");
		return;
	}

	target_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(target_node)) {
				Node *node = stack->skeleton->get_node(target_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update cache: Target node is this modification's skeleton or cannot be found!");
				target_node_cache = node->get_instance_id();
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

Vector3 SkeletonModification3DLookAt::get_rotation_offset() const {
	return additional_rotation;
}

void SkeletonModification3DLookAt::set_rotation_offset(Vector3 p_offset) {
	additional_rotation = p_offset;
}

bool SkeletonModification3DLookAt::get_lock_rotation_x() const {
	return lock_rotation_x;
}

bool SkeletonModification3DLookAt::get_lock_rotation_y() const {
	return lock_rotation_y;
}

bool SkeletonModification3DLookAt::get_lock_rotation_z() const {
	return lock_rotation_z;
}

void SkeletonModification3DLookAt::set_lock_rotation_x(bool p_lock) {
	lock_rotation_x = p_lock;
}

void SkeletonModification3DLookAt::set_lock_rotation_y(bool p_lock) {
	lock_rotation_y = p_lock;
}

void SkeletonModification3DLookAt::set_lock_rotation_z(bool p_lock) {
	lock_rotation_z = p_lock;
}

void SkeletonModification3DLookAt::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bone_name", "name"), &SkeletonModification3DLookAt::set_bone_name);
	ClassDB::bind_method(D_METHOD("get_bone_name"), &SkeletonModification3DLookAt::get_bone_name);

	ClassDB::bind_method(D_METHOD("set_bone_index", "bone_idx"), &SkeletonModification3DLookAt::set_bone_index);
	ClassDB::bind_method(D_METHOD("get_bone_index"), &SkeletonModification3DLookAt::get_bone_index);

	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification3DLookAt::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification3DLookAt::get_target_node);

	ClassDB::bind_method(D_METHOD("set_rotation_offset", "offset"), &SkeletonModification3DLookAt::set_rotation_offset);
	ClassDB::bind_method(D_METHOD("get_rotation_offset"), &SkeletonModification3DLookAt::get_rotation_offset);

	ClassDB::bind_method(D_METHOD("set_lock_rotation_x", "lock"), &SkeletonModification3DLookAt::set_lock_rotation_x);
	ClassDB::bind_method(D_METHOD("get_lock_rotation_x"), &SkeletonModification3DLookAt::get_lock_rotation_x);
	ClassDB::bind_method(D_METHOD("set_lock_rotation_y", "lock"), &SkeletonModification3DLookAt::set_lock_rotation_y);
	ClassDB::bind_method(D_METHOD("get_lock_rotation_y"), &SkeletonModification3DLookAt::get_lock_rotation_y);
	ClassDB::bind_method(D_METHOD("set_lock_rotation_z", "lock"), &SkeletonModification3DLookAt::set_lock_rotation_z);
	ClassDB::bind_method(D_METHOD("get_lock_rotation_z"), &SkeletonModification3DLookAt::get_lock_rotation_z);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "bone_name"), "set_bone_name", "get_bone_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone_index"), "set_bone_index", "get_bone_index");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_target_node", "get_target_node");
	ADD_GROUP("Additional Settings", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "lock_rotation_x"), "set_lock_rotation_x", "get_lock_rotation_x");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "lock_rotation_y"), "set_lock_rotation_y", "get_lock_rotation_y");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "lock_rotation_z"), "set_lock_rotation_z", "get_lock_rotation_z");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "rotation_offset"), "set_rotation_offset", "get_rotation_offset");
	ADD_GROUP("", "");
}

SkeletonModification3DLookAt::SkeletonModification3DLookAt() {
	stack = nullptr;
	is_setup = false;
	bone_name = "";
	bone_idx = -2;
	additional_rotation = Vector3();
	lock_rotation_x = false;
	lock_rotation_y = false;
	lock_rotation_z = false;
	enabled = true;
}

SkeletonModification3DLookAt::~SkeletonModification3DLookAt() {
}
