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

///////////////////////////////////////
// CCDIK
///////////////////////////////////////

bool SkeletonModification3DCCDIK::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, ccdik_data_chain.size(), false);

		if (what == "bone_name") {
			ccdik_joint_set_bone_name(which, p_value);
		} else if (what == "bone_index") {
			ccdik_joint_set_bone_index(which, p_value);
		} else if (what == "ccdik_axis") {
			ccdik_joint_set_ccdik_axis(which, p_value);
		} else if (what == "ccdik_axis_custom") {
			ccdik_joint_set_ccdik_axis_vector(which, p_value);
		} else if (what == "rotate_mode") {
			ccdik_joint_set_rotate_mode(which, p_value);
		} else if (what == "enable_joint_constraint") {
			ccdik_joint_set_enable_constraint(which, p_value);
		} else if (what == "joint_constraint_angle_min") {
			ccdik_joint_set_constraint_angle_degrees_min(which, p_value);
		} else if (what == "joint_constraint_angle_max") {
			ccdik_joint_set_constraint_angle_degrees_max(which, p_value);
		} else if (what == "joint_constraint_angles_invert") {
			ccdik_joint_set_constraint_invert(which, p_value);
		}
		return true;
	}
	return true;
}

bool SkeletonModification3DCCDIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, ccdik_data_chain.size(), false);

		if (what == "bone_name") {
			r_ret = ccdik_joint_get_bone_name(which);
		} else if (what == "bone_index") {
			r_ret = ccdik_joint_get_bone_index(which);
		} else if (what == "ccdik_axis") {
			r_ret = ccdik_joint_get_ccdik_axis(which);
		} else if (what == "ccdik_axis_custom") {
			r_ret = ccdik_joint_get_ccdik_axis_vector(which);
		} else if (what == "rotate_mode") {
			r_ret = ccdik_joint_get_rotate_mode(which);
		} else if (what == "enable_joint_constraint") {
			r_ret = ccdik_joint_get_enable_constraint(which);
		} else if (what == "joint_constraint_angle_min") {
			r_ret = Math::rad2deg(ccdik_joint_get_constraint_angle_min(which));
		} else if (what == "joint_constraint_angle_max") {
			r_ret = Math::rad2deg(ccdik_joint_get_constraint_angle_max(which));
		} else if (what == "joint_constraint_angles_invert") {
			r_ret = ccdik_joint_get_constraint_invert(which);
		}
		return true;
	}
	return true;
}

void SkeletonModification3DCCDIK::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < ccdik_data_chain.size(); i++) {
		String base_string = "joint_data/" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::STRING, base_string + "bone_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::INT, base_string + "bone_index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));

		p_list->push_back(PropertyInfo(Variant::INT, base_string + "ccdik_axis",
				PROPERTY_HINT_ENUM, "X Axis, Y Axis, Z Axis, Custom Axis", PROPERTY_USAGE_DEFAULT));
		if (ccdik_data_chain[i].ccdik_axis >= AXIS_CUSTOM) {
			p_list->push_back(PropertyInfo(Variant::VECTOR3, base_string + "ccdik_axis_custom", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}

		p_list->push_back(PropertyInfo(Variant::INT, base_string + "rotate_mode",
				PROPERTY_HINT_ENUM, "From Tip, From Joint, Free", PROPERTY_USAGE_DEFAULT));

		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "enable_joint_constraint", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		if (ccdik_data_chain[i].enable_constraint == true) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "joint_constraint_angle_min", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "joint_constraint_angle_max", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "joint_constraint_angles_invert", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
	}
}

void SkeletonModification3DCCDIK::execute(float delta) {
	ERR_FAIL_COND_MSG(!stack || !is_setup || stack->skeleton == nullptr,
			"Modification is not setup and therefore cannot execute!");
	if (!enabled) {
		return;
	}

	if (target_node_cache.is_null()) {
		update_target_cache();
		WARN_PRINT("Target cache is out of date. Updating...");
		return;
	}
	if (tip_node_cache.is_null()) {
		update_tip_cache();
		WARN_PRINT("Tip cache is out of date. Updating...");
		return;
	}

	Node3D *node_target = Object::cast_to<Node3D>(ObjectDB::get_instance(target_node_cache));
	Node3D *node_tip = Object::cast_to<Node3D>(ObjectDB::get_instance(tip_node_cache));

	ERR_FAIL_COND_MSG(!node_target || !node_tip,
			"Either the target or tip node is not found. Cannot execute without both nodes!");
	ERR_FAIL_COND_MSG(!node_target->is_inside_tree() || !node_tip->is_inside_tree(),
			"Either the target or tip node is not in the scene. Cannot execute without both nodes in the scene!");

	for (int i = 0; i < ccdik_data_chain.size(); i++) {
		_execute_ccdik_joint(i, node_target, node_tip);
	}
}

void SkeletonModification3DCCDIK::_execute_ccdik_joint(int p_joint_idx, Node3D *target, Node3D *tip) {
	CCDIK_Joint_Data ccdik_data = ccdik_data_chain[p_joint_idx];
	ERR_FAIL_INDEX_MSG(ccdik_data.bone_idx, stack->skeleton->get_bone_count(), "CCDIK joint: bone index not found");
	ERR_FAIL_COND_MSG(ccdik_data.ccdik_axis_vector.length_squared() == 0, "CCDIK joint: axis vector not set!");

	Transform bone_trans;
	if (perform_in_local_pose) {
		bone_trans = stack->skeleton->get_bone_local_pose_override(ccdik_data.bone_idx);
	} else {
		bone_trans = stack->skeleton->local_pose_to_global_pose(ccdik_data.bone_idx, stack->skeleton->get_bone_local_pose_override(ccdik_data.bone_idx));
	}

	// Rotate the ccdik joint
	if (ccdik_data.rotate_mode == ROTATE_MODE_FROM_TIP) {
		Vector3 rotation_vector_from;
		Vector3 rotation_vector_to;

		// Get the two positions needed.
		if (perform_in_local_pose) {
			rotation_vector_from = stack->skeleton->global_pose_to_local_pose(ccdik_data.bone_idx, stack->skeleton->world_transform_to_global_pose(tip->get_global_transform())).origin;
			rotation_vector_to = stack->skeleton->global_pose_to_local_pose(ccdik_data.bone_idx, stack->skeleton->world_transform_to_global_pose(target->get_global_transform())).origin;
		} else {
			rotation_vector_from = bone_trans.origin.direction_to(stack->skeleton->world_transform_to_global_pose(tip->get_global_transform()).origin);
			rotation_vector_to = bone_trans.origin.direction_to(stack->skeleton->world_transform_to_global_pose(target->get_global_transform()).origin);
		}

		// Restrict them to the axis and normalize them.
		rotation_vector_from = (rotation_vector_from * ccdik_data.ccdik_axis_vector_inverse).normalized();
		rotation_vector_to = (rotation_vector_to * ccdik_data.ccdik_axis_vector_inverse).normalized();

		// Rotate the Basis from the first vector, to the second vector.
		bone_trans.basis.rotate_to_align(rotation_vector_from, rotation_vector_to);

	} else if (ccdik_data.rotate_mode == ROTATE_MODE_FROM_JOINT) {
		// Get the forward direction that the basis is facing in right now, with a fallback of using
		// the rest forward axis.
		Vector3 rotation_vector_from = Vector3(0, 0, 0);
		stack->skeleton->update_bone_rest_forward_vector(ccdik_data.bone_idx);
		int bone_forward_axis_enum = stack->skeleton->get_bone_axis_forward_enum(ccdik_data.bone_idx);
		if (bone_forward_axis_enum == stack->skeleton->BONE_AXIS_X_FORWARD) {
			rotation_vector_from = bone_trans.basis[0].normalized();
		} else if (bone_forward_axis_enum == stack->skeleton->BONE_AXIS_NEGATIVE_X_FORWARD) {
			rotation_vector_from = -bone_trans.basis[0].normalized();
		} else if (bone_forward_axis_enum == stack->skeleton->BONE_AXIS_Y_FORWARD) {
			rotation_vector_from = bone_trans.basis[1].normalized();
		} else if (bone_forward_axis_enum == stack->skeleton->BONE_AXIS_NEGATIVE_Y_FORWARD) {
			rotation_vector_from = -bone_trans.basis[1].normalized();
		} else if (bone_forward_axis_enum == stack->skeleton->BONE_AXIS_Z_FORWARD) {
			rotation_vector_from = bone_trans.basis[2].normalized();
		} else if (bone_forward_axis_enum == stack->skeleton->BONE_AXIS_NEGATIVE_Z_FORWARD) {
			rotation_vector_from = -bone_trans.basis[2].normalized();
		} else {
			// Assume Y+ when in doubt.
			rotation_vector_from = bone_trans.basis[1].normalized();
		}

		// The target's position
		Vector3 rotation_vector_to;
		if (perform_in_local_pose) {
			rotation_vector_to = stack->skeleton->global_pose_to_local_pose(ccdik_data.bone_idx, stack->skeleton->world_transform_to_global_pose(target->get_global_transform())).origin;
		} else {
			rotation_vector_to = bone_trans.origin.direction_to(stack->skeleton->world_transform_to_global_pose(target->get_global_transform()).origin);
		}

		// Restrict them to the axis and normalize them.
		rotation_vector_from = (rotation_vector_from * ccdik_data.ccdik_axis_vector_inverse).normalized();
		rotation_vector_to = (rotation_vector_to * ccdik_data.ccdik_axis_vector_inverse).normalized();

		// Rotate the Basis from the first vector, to the second vector
		bone_trans.basis.rotate_to_align(rotation_vector_from, rotation_vector_to);

	} else if (ccdik_data.rotate_mode == ROTATE_MODE_FREE) { // Free mode: allow rotation on any axis. Needs testing!
		Vector3 target_position = stack->skeleton->world_transform_to_global_pose(target->get_global_transform()).origin;
		if (perform_in_local_pose) {
			target_position = stack->skeleton->global_pose_to_local_pose(ccdik_data.bone_idx, stack->skeleton->world_transform_to_global_pose(target->get_global_transform())).origin;
		} else {
			target_position = stack->skeleton->world_transform_to_global_pose(target->get_global_transform()).origin;
		}

		bone_trans = bone_trans.looking_at(target_position, Vector3(0, 1, 0));
		bone_trans.basis = stack->skeleton->global_pose_z_forward_to_bone_forward(ccdik_data.bone_idx, bone_trans.basis);
	}

	// Apply constraints
	// (Todo: Still needs adjusting?)
	if (ccdik_data.enable_constraint) {
		float ccdik_rotation_angle;
		Vector3 ccdik_rotation_axis;
		bone_trans.basis.get_axis_angle(ccdik_rotation_axis, ccdik_rotation_angle);

		if (ccdik_data.constraint_angles_invert == false) { // Normal clamping:
			if (ccdik_rotation_angle < ccdik_data.constraint_angle_min) {
				ccdik_rotation_angle = ccdik_data.constraint_angle_min;
			} else if (ccdik_rotation_angle > ccdik_data.constraint_angle_max) {
				ccdik_rotation_angle = ccdik_data.constraint_angle_max;
			}
		} else { // Inverse clamping:
			if (ccdik_rotation_angle > ccdik_data.constraint_angle_min && ccdik_rotation_angle < ccdik_data.constraint_angle_max) {
				// Figure out which angle is closer by comparing their differences.
				if (ccdik_rotation_angle - ccdik_data.constraint_angle_min < ccdik_data.constraint_angle_max - ccdik_rotation_angle) {
					ccdik_rotation_angle = ccdik_data.constraint_angle_min;
				} else {
					ccdik_rotation_angle = ccdik_data.constraint_angle_max;
				}
			}
		}
		bone_trans.basis.set_axis_angle(ccdik_rotation_axis, ccdik_rotation_angle);
	}

	if (perform_in_local_pose) {
		stack->skeleton->set_bone_local_pose_override(ccdik_data.bone_idx, bone_trans, stack->strength, true);
	} else {
		bone_trans = stack->skeleton->global_pose_to_local_pose(ccdik_data.bone_idx, bone_trans);
		stack->skeleton->set_bone_local_pose_override(ccdik_data.bone_idx, bone_trans, stack->strength, true);
	}
	stack->skeleton->force_update_bone_children_transforms(ccdik_data.bone_idx);
}

void SkeletonModification3DCCDIK::setup_modification(SkeletonModificationStack3D *p_stack) {
	stack = p_stack;
	if (stack != nullptr) {
		is_setup = true;
		update_target_cache();
		update_tip_cache();
	}
}

void SkeletonModification3DCCDIK::update_target_cache() {
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

void SkeletonModification3DCCDIK::update_tip_cache() {
	if (!is_setup || !stack) {
		WARN_PRINT("Cannot update cache: modification is not properly setup!");
		return;
	}

	tip_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(tip_node)) {
				Node *node = stack->skeleton->get_node(tip_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update cache: Tip node is this modification's skeleton or cannot be found!");
				tip_node_cache = node->get_instance_id();
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

void SkeletonModification3DCCDIK::set_perform_in_local_pose(bool p_perform) {
	perform_in_local_pose = p_perform;
}

bool SkeletonModification3DCCDIK::get_perform_in_local_pose() const {
	return perform_in_local_pose;
}

// CCDIK joint data functions
String SkeletonModification3DCCDIK::ccdik_joint_get_bone_name(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, ccdik_data_chain.size(), String());
	return ccdik_data_chain[p_joint_idx].bone_name;
}

void SkeletonModification3DCCDIK::ccdik_joint_set_bone_name(int p_joint_idx, String p_bone_name) {
	ERR_FAIL_INDEX(p_joint_idx, ccdik_data_chain.size());
	ccdik_data_chain.write[p_joint_idx].bone_name = p_bone_name;

	if (stack) {
		if (stack->skeleton) {
			ccdik_data_chain.write[p_joint_idx].bone_idx = stack->skeleton->find_bone(p_bone_name);
		}
	}
	_change_notify();
}

int SkeletonModification3DCCDIK::ccdik_joint_get_bone_index(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, ccdik_data_chain.size(), -1);
	return ccdik_data_chain[p_joint_idx].bone_idx;
}

void SkeletonModification3DCCDIK::ccdik_joint_set_bone_index(int p_joint_idx, int p_bone_idx) {
	ERR_FAIL_INDEX(p_joint_idx, ccdik_data_chain.size());
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");
	ccdik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;

	if (stack) {
		if (stack->skeleton) {
			ccdik_data_chain.write[p_joint_idx].bone_name = stack->skeleton->get_bone_name(p_bone_idx);
		}
	}
	_change_notify();
}

int SkeletonModification3DCCDIK::ccdik_joint_get_ccdik_axis(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, ccdik_data_chain.size(), -1);
	return ccdik_data_chain[p_joint_idx].ccdik_axis;
}

void SkeletonModification3DCCDIK::ccdik_joint_set_ccdik_axis(int p_joint_idx, int p_axis) {
	ERR_FAIL_INDEX(p_joint_idx, ccdik_data_chain.size());
	ERR_FAIL_COND_MSG(p_axis < 0, "CCDIK axis is out of range: The axis mode is too low!");
	ERR_FAIL_COND_MSG(p_axis > AXIS_CUSTOM, "CCDIK axis is out of range: The axis mode is too high!");
	ccdik_data_chain.write[p_joint_idx].ccdik_axis = p_axis;

	if (p_axis == AXIS_X) {
		ccdik_joint_set_ccdik_axis_vector(p_joint_idx, Vector3(1, 0, 0));
	} else if (p_axis == AXIS_Y) {
		ccdik_joint_set_ccdik_axis_vector(p_joint_idx, Vector3(0, 1, 0));
	} else if (p_axis == AXIS_Z) {
		ccdik_joint_set_ccdik_axis_vector(p_joint_idx, Vector3(0, 0, 1));
	}
	_change_notify();
}

Vector3 SkeletonModification3DCCDIK::ccdik_joint_get_ccdik_axis_vector(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, ccdik_data_chain.size(), Vector3());
	return ccdik_data_chain[p_joint_idx].ccdik_axis_vector;
}

void SkeletonModification3DCCDIK::ccdik_joint_set_ccdik_axis_vector(int p_joint_idx, Vector3 p_axis) {
	ERR_FAIL_INDEX(p_joint_idx, ccdik_data_chain.size());
	ccdik_data_chain.write[p_joint_idx].ccdik_axis_vector = p_axis;
	ccdik_data_chain.write[p_joint_idx].ccdik_axis_vector_inverse = (Vector3(1, 1, 1) - p_axis).normalized();
}

int SkeletonModification3DCCDIK::ccdik_joint_get_rotate_mode(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, ccdik_data_chain.size(), -1);
	return ccdik_data_chain[p_joint_idx].rotate_mode;
}

void SkeletonModification3DCCDIK::ccdik_joint_set_rotate_mode(int p_joint_idx, int p_mode) {
	ERR_FAIL_INDEX(p_joint_idx, ccdik_data_chain.size());
	ERR_FAIL_COND_MSG(p_mode < 0 || p_mode > ROTATE_MODE_FREE, "Cannot assign unknown joint rotate mode!");
	ccdik_data_chain.write[p_joint_idx].rotate_mode = p_mode;
}

bool SkeletonModification3DCCDIK::ccdik_joint_get_enable_constraint(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, ccdik_data_chain.size(), false);
	return ccdik_data_chain[p_joint_idx].enable_constraint;
}

void SkeletonModification3DCCDIK::ccdik_joint_set_enable_constraint(int p_joint_idx, bool p_enable) {
	ERR_FAIL_INDEX(p_joint_idx, ccdik_data_chain.size());
	ccdik_data_chain.write[p_joint_idx].enable_constraint = p_enable;
	_change_notify();
}

float SkeletonModification3DCCDIK::ccdik_joint_get_constraint_angle_min(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, ccdik_data_chain.size(), false);
	return ccdik_data_chain[p_joint_idx].constraint_angle_min;
}

void SkeletonModification3DCCDIK::ccdik_joint_set_constraint_angle_min(int p_joint_idx, float p_angle_min) {
	ERR_FAIL_INDEX(p_joint_idx, ccdik_data_chain.size());
	ccdik_data_chain.write[p_joint_idx].constraint_angle_min = p_angle_min;
}

void SkeletonModification3DCCDIK::ccdik_joint_set_constraint_angle_degrees_min(int p_joint_idx, float p_angle_min) {
	ccdik_joint_set_constraint_angle_min(p_joint_idx, Math::deg2rad(p_angle_min));
}

float SkeletonModification3DCCDIK::ccdik_joint_get_constraint_angle_max(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, ccdik_data_chain.size(), false);
	return ccdik_data_chain[p_joint_idx].constraint_angle_max;
}

void SkeletonModification3DCCDIK::ccdik_joint_set_constraint_angle_max(int p_joint_idx, float p_angle_max) {
	ERR_FAIL_INDEX(p_joint_idx, ccdik_data_chain.size());
	ccdik_data_chain.write[p_joint_idx].constraint_angle_max = p_angle_max;
}

void SkeletonModification3DCCDIK::ccdik_joint_set_constraint_angle_degrees_max(int p_joint_idx, float p_angle_max) {
	ccdik_joint_set_constraint_angle_max(p_joint_idx, Math::deg2rad(p_angle_max));
}

bool SkeletonModification3DCCDIK::ccdik_joint_get_constraint_invert(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, ccdik_data_chain.size(), false);
	return ccdik_data_chain[p_joint_idx].constraint_angles_invert;
}

void SkeletonModification3DCCDIK::ccdik_joint_set_constraint_invert(int p_joint_idx, bool p_invert) {
	ERR_FAIL_INDEX(p_joint_idx, ccdik_data_chain.size());
	ccdik_data_chain.write[p_joint_idx].constraint_angles_invert = p_invert;
}

int SkeletonModification3DCCDIK::get_ccdik_data_chain_length() {
	return ccdik_data_chain.size();
}
void SkeletonModification3DCCDIK::set_ccdik_data_chain_length(int p_length) {
	ERR_FAIL_COND(p_length < 0);
	ccdik_data_chain.resize(p_length);
	_change_notify();
}

void SkeletonModification3DCCDIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification3DCCDIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification3DCCDIK::get_target_node);

	ClassDB::bind_method(D_METHOD("set_tip_node", "tip_nodepath"), &SkeletonModification3DCCDIK::set_tip_node);
	ClassDB::bind_method(D_METHOD("get_tip_node"), &SkeletonModification3DCCDIK::get_tip_node);

	ClassDB::bind_method(D_METHOD("set_perform_in_local_pose", "perform_in_local_pose"), &SkeletonModification3DCCDIK::set_perform_in_local_pose);
	ClassDB::bind_method(D_METHOD("get_perform_in_local_pose"), &SkeletonModification3DCCDIK::get_perform_in_local_pose);

	// CCDIK joint data functions
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_bone_name", "joint_idx"), &SkeletonModification3DCCDIK::ccdik_joint_get_bone_name);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_bone_name", "joint_idx", "bone_name"), &SkeletonModification3DCCDIK::ccdik_joint_set_bone_name);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_bone_index", "joint_idx"), &SkeletonModification3DCCDIK::ccdik_joint_get_bone_index);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_bone_index", "joint_idx", "bone_index"), &SkeletonModification3DCCDIK::ccdik_joint_set_bone_index);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_ccdik_axis", "joint_idx"), &SkeletonModification3DCCDIK::ccdik_joint_get_ccdik_axis);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_ccdik_axis", "joint_idx", "axis"), &SkeletonModification3DCCDIK::ccdik_joint_set_ccdik_axis);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_enable_joint_constraint", "joint_idx"), &SkeletonModification3DCCDIK::ccdik_joint_get_enable_constraint);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_enable_joint_constraint", "joint_idx", "enable"), &SkeletonModification3DCCDIK::ccdik_joint_set_enable_constraint);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_joint_constraint_angle_min", "joint_idx"), &SkeletonModification3DCCDIK::ccdik_joint_get_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_joint_constraint_angle_min", "joint_idx", "min_angle"), &SkeletonModification3DCCDIK::ccdik_joint_set_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_joint_constraint_angle_degrees_min", "joint_idx", "min_angle"), &SkeletonModification3DCCDIK::ccdik_joint_set_constraint_angle_degrees_min);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_joint_constraint_angle_max", "joint_idx"), &SkeletonModification3DCCDIK::ccdik_joint_get_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_joint_constraint_angle_max", "joint_idx", "max_angle"), &SkeletonModification3DCCDIK::ccdik_joint_set_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_joint_constraint_angle_degrees_max", "joint_idx", "max_angle"), &SkeletonModification3DCCDIK::ccdik_joint_set_constraint_angle_degrees_max);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_joint_constraint_invert", "joint_idx"), &SkeletonModification3DCCDIK::ccdik_joint_get_constraint_invert);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_joint_constraint_invert", "joint_idx", "invert"), &SkeletonModification3DCCDIK::ccdik_joint_set_constraint_invert);

	ClassDB::bind_method(D_METHOD("set_ccdik_data_chain_length", "length"), &SkeletonModification3DCCDIK::set_ccdik_data_chain_length);
	ClassDB::bind_method(D_METHOD("get_ccdik_data_chain_length"), &SkeletonModification3DCCDIK::get_ccdik_data_chain_length);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "tip_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_tip_node", "get_tip_node");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "perform_in_local_pose", PROPERTY_HINT_NONE, ""), "set_perform_in_local_pose", "get_perform_in_local_pose");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ccdik_data_chain_length", PROPERTY_HINT_RANGE, "0,100,1"), "set_ccdik_data_chain_length", "get_ccdik_data_chain_length");
}

SkeletonModification3DCCDIK::SkeletonModification3DCCDIK() {
	stack = nullptr;
	is_setup = false;
	enabled = true;
	perform_in_local_pose = true;
}

SkeletonModification3DCCDIK::~SkeletonModification3DCCDIK() {
}

///////////////////////////////////////
// FABRIK
///////////////////////////////////////

bool SkeletonModification3DFABRIK::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, fabrik_data_chain.size(), false);

		if (what == "bone_name") {
			fabrik_joint_set_bone_name(which, p_value);
		} else if (what == "bone_index") {
			fabrik_joint_set_bone_index(which, p_value);
		} else if (what == "length") {
			fabrik_joint_set_length(which, p_value);
		} else if (what == "magnet_position") {
			fabrik_joint_set_magnet(which, p_value);
		} else if (what == "auto_calculate_length") {
			fabrik_joint_set_auto_calculate_length(which, p_value);
		} else if (what == "use_tip_node") {
			fabrik_joint_set_use_tip_node(which, p_value);
		} else if (what == "tip_node") {
			fabrik_joint_set_tip_node(which, p_value);
		} else if (what == "use_target_basis") {
			fabrik_joint_set_use_target_basis(which, p_value);
		}
		return true;
	}
	return true;
}

bool SkeletonModification3DFABRIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, fabrik_data_chain.size(), false);

		if (what == "bone_name") {
			r_ret = fabrik_joint_get_bone_name(which);
		} else if (what == "bone_index") {
			r_ret = fabrik_joint_get_bone_index(which);
		} else if (what == "length") {
			r_ret = fabrik_joint_get_length(which);
		} else if (what == "magnet_position") {
			r_ret = fabrik_joint_get_magnet(which);
		} else if (what == "auto_calculate_length") {
			r_ret = fabrik_joint_get_auto_calculate_length(which);
		} else if (what == "use_tip_node") {
			r_ret = fabrik_joint_get_use_tip_node(which);
		} else if (what == "tip_node") {
			r_ret = fabrik_joint_get_tip_node(which);
		} else if (what == "use_target_basis") {
			r_ret = fabrik_joint_get_use_target_basis(which);
		}
		return true;
	}
	return true;
}

void SkeletonModification3DFABRIK::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		String base_string = "joint_data/" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::STRING, base_string + "bone_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::INT, base_string + "bone_index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "auto_calculate_length", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));

		if (fabrik_data_chain[i].auto_calculate_length == false) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "length", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		} else {
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "use_tip_node", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			if (fabrik_data_chain[i].use_tip_node == true) {
				p_list->push_back(PropertyInfo(Variant::NODE_PATH, base_string + "tip_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D", PROPERTY_USAGE_DEFAULT));
			}
		}

		// Cannot apply magnet to the origin of the chain, it will not do anything, so do not include this property for the origin.
		if (i > 0) {
			p_list->push_back(PropertyInfo(Variant::VECTOR3, base_string + "magnet_position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
		// Only give the override basis option on the last bone in the chain, so only include it for the last bone.
		if (i == fabrik_data_chain.size() - 1) {
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "use_target_basis", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
	}
}

void SkeletonModification3DFABRIK::execute(float delta) {
	ERR_FAIL_COND_MSG(!stack || !is_setup || stack->skeleton == nullptr,
			"Modification is not setup and therefore cannot execute!");
	if (!enabled) {
		return;
	}

	// TODO: support a single dummy tip/final bone? This will allow for setting the magnet position on a two bone FABRIK chain.

	if (target_node_cache.is_null()) {
		update_target_cache();
		WARN_PRINT("Target cache is out of date. Updating...");
		return;
	}

	Node3D *node_target = Object::cast_to<Node3D>(ObjectDB::get_instance(target_node_cache));
	ERR_FAIL_COND_MSG(!node_target, "The target node is not found. Cannot execute!");
	ERR_FAIL_COND_MSG(!node_target->is_inside_tree(), "The target node is not in the scene. Cannot execute!");

	// Verify that all joints have a valid bone ID, and that all bone lengths are zero or more
	// Also, while we are here, apply magnet positions.
	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		ERR_FAIL_COND_MSG(fabrik_data_chain[i].bone_idx < 0, "Joint " + itos(i) + " has an invalid bone ID! Cannot execute!");

		if (fabrik_data_chain[i].length < 0 && fabrik_data_chain[i].auto_calculate_length) {
			fabrik_joint_auto_calculate_length(i);
		}
		ERR_FAIL_COND_MSG(fabrik_data_chain[i].length < 0, "Joint " + itos(i) + " has an invalid joint length! Cannot execute!");

		// Apply magnet positions:
		Transform local_pose_override = stack->skeleton->get_bone_local_pose_override(fabrik_data_chain[i].bone_idx);
		if (stack->skeleton->get_bone_parent(fabrik_data_chain[i].bone_idx) >= 0) {
			int parent_bone_idx = stack->skeleton->get_bone_parent(fabrik_data_chain[i].bone_idx);
			Transform conversion_transform = (stack->skeleton->get_bone_global_pose(parent_bone_idx) * stack->skeleton->get_bone_rest(parent_bone_idx));
			local_pose_override.origin += conversion_transform.basis.xform_inv(fabrik_data_chain[i].magnet_position);
		} else {
			local_pose_override.origin += fabrik_data_chain[i].magnet_position;
		}

		stack->skeleton->set_bone_local_pose_override(fabrik_data_chain[i].bone_idx, local_pose_override, stack->strength, true);
	}

	target_global_pose = stack->skeleton->world_transform_to_global_pose(node_target->get_global_transform());
	origin_global_pose = stack->skeleton->local_pose_to_global_pose(
			fabrik_data_chain[0].bone_idx, stack->skeleton->get_bone_local_pose_override(fabrik_data_chain[0].bone_idx));

	final_joint_idx = fabrik_data_chain.size() - 1;
	float target_distance = stack->skeleton->global_pose_to_local_pose(fabrik_data_chain[final_joint_idx].bone_idx, target_global_pose).origin.length();
	chain_iterations = 0;

	while (target_distance > chain_tolerance) {
		chain_backwards();
		chain_forwards();
		chain_apply();

		// update the target distance
		target_distance = stack->skeleton->global_pose_to_local_pose(fabrik_data_chain[final_joint_idx].bone_idx, target_global_pose).origin.length();

		// update chain iterations
		chain_iterations += 1;
		if (chain_iterations >= chain_max_iterations) {
			break;
		}
	}
}

void SkeletonModification3DFABRIK::chain_backwards() {
	int final_bone_idx = fabrik_data_chain[final_joint_idx].bone_idx;
	Transform final_joint_trans = stack->skeleton->local_pose_to_global_pose(final_bone_idx, stack->skeleton->get_bone_local_pose_override(final_bone_idx));

	// Get the direction the final bone is facing in.
	stack->skeleton->update_bone_rest_forward_vector(final_bone_idx);
	Vector3 direction = final_joint_trans.xform(stack->skeleton->get_bone_axis_forward_vector(final_bone_idx)).normalized();

	// set the position of the final joint to the target position
	final_joint_trans.origin = target_global_pose.origin - (direction * fabrik_data_chain[final_joint_idx].length);
	final_joint_trans = stack->skeleton->global_pose_to_local_pose(final_bone_idx, final_joint_trans);
	stack->skeleton->set_bone_local_pose_override(final_bone_idx, final_joint_trans, stack->strength, true);

	// for all other joints, move them towards the target
	int i = final_joint_idx;
	while (i >= 1) {
		int next_bone_idx = fabrik_data_chain[i].bone_idx;
		Transform next_bone_trans = stack->skeleton->local_pose_to_global_pose(next_bone_idx, stack->skeleton->get_bone_local_pose_override(next_bone_idx));
		i -= 1;
		int current_bone_idx = fabrik_data_chain[i].bone_idx;
		Transform current_trans = stack->skeleton->local_pose_to_global_pose(current_bone_idx, stack->skeleton->get_bone_local_pose_override(current_bone_idx));

		float length = fabrik_data_chain[i].length / (next_bone_trans.origin - current_trans.origin).length();
		current_trans.origin = next_bone_trans.origin.lerp(current_trans.origin, length);

		// Apply it back to the skeleton
		stack->skeleton->set_bone_local_pose_override(current_bone_idx, stack->skeleton->global_pose_to_local_pose(current_bone_idx, current_trans), stack->strength, true);
	}
}

void SkeletonModification3DFABRIK::chain_forwards() {
	// Set root at the initial position.
	int origin_bone_idx = fabrik_data_chain[0].bone_idx;
	Transform root_transform = stack->skeleton->local_pose_to_global_pose(origin_bone_idx, stack->skeleton->get_bone_local_pose_override(origin_bone_idx));
	root_transform.origin = origin_global_pose.origin;
	stack->skeleton->set_bone_local_pose_override(origin_bone_idx, stack->skeleton->global_pose_to_local_pose(origin_bone_idx, root_transform), stack->strength, true);

	for (int i = 0; i < fabrik_data_chain.size() - 1; i++) {
		int current_bone_idx = fabrik_data_chain[i].bone_idx;
		Transform current_trans = stack->skeleton->local_pose_to_global_pose(current_bone_idx, stack->skeleton->get_bone_local_pose_override(current_bone_idx));
		int next_bone_idx = fabrik_data_chain[i + 1].bone_idx;
		Transform next_bone_trans = stack->skeleton->local_pose_to_global_pose(next_bone_idx, stack->skeleton->get_bone_local_pose_override(next_bone_idx));

		float length = fabrik_data_chain[i].length / (current_trans.origin - next_bone_trans.origin).length();
		next_bone_trans.origin = current_trans.origin.lerp(next_bone_trans.origin, length);

		// Apply it back to the skeleton
		stack->skeleton->set_bone_local_pose_override(next_bone_idx, stack->skeleton->global_pose_to_local_pose(next_bone_idx, next_bone_trans), stack->strength, true);
	}
}

void SkeletonModification3DFABRIK::chain_apply() {
	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		int current_bone_idx = fabrik_data_chain[i].bone_idx;
		Transform current_trans = stack->skeleton->get_bone_local_pose_override(current_bone_idx);
		current_trans = stack->skeleton->local_pose_to_global_pose(current_bone_idx, current_trans);

		// If this is the last bone in the chain...
		if (i == fabrik_data_chain.size() - 1) {
			if (fabrik_data_chain[i].use_target_basis == false) { // Point to target...
				// Get the forward direction that the basis is facing in right now.
				stack->skeleton->update_bone_rest_forward_vector(current_bone_idx);
				Vector3 forward_vector = stack->skeleton->get_bone_axis_forward_vector(current_bone_idx);
				// Rotate the bone towards the target:
				current_trans.basis.rotate_to_align(forward_vector, current_trans.origin.direction_to(target_global_pose.origin));

			} else { // Use the target's Basis...
				Vector3 tmp_scale = current_trans.basis.get_scale();
				current_trans.basis = target_global_pose.basis.orthonormalized();
				current_trans.basis.scale(tmp_scale);
			}
		} else { // every other bone in the chain...

			int next_bone_idx = fabrik_data_chain[i + 1].bone_idx;
			Transform next_trans = stack->skeleton->local_pose_to_global_pose(next_bone_idx, stack->skeleton->get_bone_local_pose_override(next_bone_idx));

			// Get the forward direction that the basis is facing in right now.
			stack->skeleton->update_bone_rest_forward_vector(current_bone_idx);
			Vector3 forward_vector = stack->skeleton->get_bone_axis_forward_vector(current_bone_idx);
			// Rotate the bone towards the next bone in the chain:
			current_trans.basis.rotate_to_align(forward_vector, current_trans.origin.direction_to(next_trans.origin));
		}
		current_trans = stack->skeleton->global_pose_to_local_pose(current_bone_idx, current_trans);
		current_trans.origin = Vector3(0, 0, 0);
		stack->skeleton->set_bone_local_pose_override(current_bone_idx, current_trans, stack->strength, true);
	}

	// Update all the bones so the next modification has up-to-date data.
	stack->skeleton->force_update_all_bone_transforms();
}

void SkeletonModification3DFABRIK::setup_modification(SkeletonModificationStack3D *p_stack) {
	stack = p_stack;
	if (stack != nullptr) {
		is_setup = true;
		update_target_cache();

		for (int i = 0; i < fabrik_data_chain.size(); i++) {
			update_joint_tip_cache(i);
		}
	}
}

void SkeletonModification3DFABRIK::update_target_cache() {
	if (!is_setup || !stack) {
		WARN_PRINT("Cannot update cache: modification is not properly setup!");
		return;
	}
	target_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree() && target_node.is_empty() == false) {
			if (stack->skeleton->has_node(target_node)) {
				Node *node = stack->skeleton->get_node(target_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update cache: Target node is this modification's skeleton or cannot be found!");
				target_node_cache = node->get_instance_id();
			}
		}
	}
}

void SkeletonModification3DFABRIK::update_joint_tip_cache(int p_joint_idx) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint not found");
	if (!is_setup || !stack) {
		WARN_PRINT("Cannot update cache: modification is not properly setup!");
		return;
	}
	fabrik_data_chain.write[p_joint_idx].tip_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree() && fabrik_data_chain[p_joint_idx].tip_node.is_empty() == false) {
			if (stack->skeleton->has_node(fabrik_data_chain[p_joint_idx].tip_node)) {
				Node *node = stack->skeleton->get_node(fabrik_data_chain[p_joint_idx].tip_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update tip cache for joint " + itos(p_joint_idx) + ":node is this modification's skeleton or cannot be found!");
				fabrik_data_chain.write[p_joint_idx].tip_node_cache = node->get_instance_id();
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
	_change_notify();
}

float SkeletonModification3DFABRIK::get_chain_tolerance() {
	return chain_tolerance;
}

void SkeletonModification3DFABRIK::set_chain_tolerance(float p_tolerance) {
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
String SkeletonModification3DFABRIK::fabrik_joint_get_bone_name(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, fabrik_data_chain.size(), String());
	return fabrik_data_chain[p_joint_idx].bone_name;
}

void SkeletonModification3DFABRIK::fabrik_joint_set_bone_name(int p_joint_idx, String p_bone_name) {
	ERR_FAIL_INDEX(p_joint_idx, fabrik_data_chain.size());
	fabrik_data_chain.write[p_joint_idx].bone_name = p_bone_name;

	if (stack) {
		if (stack->skeleton) {
			fabrik_data_chain.write[p_joint_idx].bone_idx = stack->skeleton->find_bone(p_bone_name);
		}
	}
	_change_notify();
}

int SkeletonModification3DFABRIK::fabrik_joint_get_bone_index(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, fabrik_data_chain.size(), -1);
	return fabrik_data_chain[p_joint_idx].bone_idx;
}

void SkeletonModification3DFABRIK::fabrik_joint_set_bone_index(int p_joint_idx, int p_bone_idx) {
	ERR_FAIL_INDEX(p_joint_idx, fabrik_data_chain.size());
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");
	fabrik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;

	if (stack) {
		if (stack->skeleton) {
			fabrik_data_chain.write[p_joint_idx].bone_name = stack->skeleton->get_bone_name(p_bone_idx);
		}
	}
	_change_notify();
}

float SkeletonModification3DFABRIK::fabrik_joint_get_length(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, fabrik_data_chain.size(), -1);
	return fabrik_data_chain[p_joint_idx].length;
}

void SkeletonModification3DFABRIK::fabrik_joint_set_length(int p_joint_idx, float p_bone_length) {
	ERR_FAIL_INDEX(p_joint_idx, fabrik_data_chain.size());
	ERR_FAIL_COND_MSG(p_bone_length < 0, "FABRIK joint length cannot be less than zero!");

	if (!is_setup) {
		fabrik_data_chain.write[p_joint_idx].length = p_bone_length;
		return;
	}

	if (fabrik_data_chain[p_joint_idx].auto_calculate_length) {
		WARN_PRINT("FABRIK Length not set: auto calculate length is enabled for this joint!");
		fabrik_joint_auto_calculate_length(p_joint_idx);
	} else {
		fabrik_data_chain.write[p_joint_idx].length = p_bone_length;
	}
}

Vector3 SkeletonModification3DFABRIK::fabrik_joint_get_magnet(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, fabrik_data_chain.size(), Vector3());
	return fabrik_data_chain[p_joint_idx].magnet_position;
}

void SkeletonModification3DFABRIK::fabrik_joint_set_magnet(int p_joint_idx, Vector3 p_magnet) {
	ERR_FAIL_INDEX(p_joint_idx, fabrik_data_chain.size());
	fabrik_data_chain.write[p_joint_idx].magnet_position = p_magnet;
}

bool SkeletonModification3DFABRIK::fabrik_joint_get_auto_calculate_length(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, fabrik_data_chain.size(), false);
	return fabrik_data_chain[p_joint_idx].auto_calculate_length;
}

void SkeletonModification3DFABRIK::fabrik_joint_set_auto_calculate_length(int p_joint_idx, bool p_auto_calculate) {
	ERR_FAIL_INDEX(p_joint_idx, fabrik_data_chain.size());
	fabrik_data_chain.write[p_joint_idx].auto_calculate_length = p_auto_calculate;
	fabrik_joint_auto_calculate_length(p_joint_idx);
	_change_notify();
}

void SkeletonModification3DFABRIK::fabrik_joint_auto_calculate_length(int p_joint_idx) {
	ERR_FAIL_INDEX(p_joint_idx, fabrik_data_chain.size());
	if (!fabrik_data_chain[p_joint_idx].auto_calculate_length) {
		return;
	}

	ERR_FAIL_COND_MSG(!stack || !stack->skeleton || !is_setup, "Cannot auto calculate joint length: modification is not setup!");
	ERR_FAIL_INDEX_MSG(fabrik_data_chain[p_joint_idx].bone_idx, stack->skeleton->get_bone_count(),
			"Bone for joint " + itos(p_joint_idx) + " is not set or points to an unknown bone!");

	if (fabrik_data_chain[p_joint_idx].use_tip_node) { // Use the tip node to update joint length.

		update_joint_tip_cache(p_joint_idx);

		Node3D *tip_node = Object::cast_to<Node3D>(ObjectDB::get_instance(fabrik_data_chain[p_joint_idx].tip_node_cache));
		ERR_FAIL_COND_MSG(!tip_node, "Tip node for joint " + itos(p_joint_idx) + "is not a Node3D-based node. Cannot calculate length...");
		ERR_FAIL_COND_MSG(!tip_node->is_inside_tree(), "Tip node for joint " + itos(p_joint_idx) + "is not in the scene tree. Cannot calculate length...");

		Transform node_trans = tip_node->get_global_transform();
		node_trans = stack->skeleton->world_transform_to_global_pose(node_trans);
		node_trans = stack->skeleton->global_pose_to_local_pose(fabrik_data_chain[p_joint_idx].bone_idx, node_trans);
		fabrik_data_chain.write[p_joint_idx].length = node_trans.origin.length();
	} else { // Use child bone(s) to update joint length, if possible
		Vector<int> bone_children = stack->skeleton->get_bone_children(fabrik_data_chain[p_joint_idx].bone_idx);
		if (bone_children.size() <= 0) {
			ERR_FAIL_MSG("Cannot calculate length for joint " + itos(p_joint_idx) + "joint uses leaf bone. \nPlease manually set the bone length or use a tip node!");
			return;
		}

		float final_length = 0;
		for (int i = 0; i < bone_children.size(); i++) {
			Transform child_transform = stack->skeleton->get_bone_global_pose(bone_children[i]);
			final_length += stack->skeleton->global_pose_to_local_pose(fabrik_data_chain[p_joint_idx].bone_idx, child_transform).origin.length();
		}
		fabrik_data_chain.write[p_joint_idx].length = final_length / bone_children.size();
	}
	_change_notify();
}

bool SkeletonModification3DFABRIK::fabrik_joint_get_use_tip_node(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, fabrik_data_chain.size(), false);
	return fabrik_data_chain[p_joint_idx].use_tip_node;
}

void SkeletonModification3DFABRIK::fabrik_joint_set_use_tip_node(int p_joint_idx, bool p_use_tip_node) {
	ERR_FAIL_INDEX(p_joint_idx, fabrik_data_chain.size());
	fabrik_data_chain.write[p_joint_idx].use_tip_node = p_use_tip_node;
	_change_notify();
}

NodePath SkeletonModification3DFABRIK::fabrik_joint_get_tip_node(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, fabrik_data_chain.size(), NodePath());
	return fabrik_data_chain[p_joint_idx].tip_node;
}

void SkeletonModification3DFABRIK::fabrik_joint_set_tip_node(int p_joint_idx, NodePath p_tip_node) {
	ERR_FAIL_INDEX(p_joint_idx, fabrik_data_chain.size());
	fabrik_data_chain.write[p_joint_idx].tip_node = p_tip_node;
	update_joint_tip_cache(p_joint_idx);
}

bool SkeletonModification3DFABRIK::fabrik_joint_get_use_target_basis(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, fabrik_data_chain.size(), false);
	return fabrik_data_chain[p_joint_idx].use_target_basis;
}

void SkeletonModification3DFABRIK::fabrik_joint_set_use_target_basis(int p_joint_idx, bool p_use_target_basis) {
	ERR_FAIL_INDEX(p_joint_idx, fabrik_data_chain.size());
	fabrik_data_chain.write[p_joint_idx].use_target_basis = p_use_target_basis;
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
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_bone_name", "joint_idx"), &SkeletonModification3DFABRIK::fabrik_joint_get_bone_name);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_bone_name", "joint_idx", "bone_name"), &SkeletonModification3DFABRIK::fabrik_joint_set_bone_name);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_bone_index", "joint_idx"), &SkeletonModification3DFABRIK::fabrik_joint_get_bone_index);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_bone_index", "joint_idx", "bone_index"), &SkeletonModification3DFABRIK::fabrik_joint_set_bone_index);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_length", "joint_idx"), &SkeletonModification3DFABRIK::fabrik_joint_get_length);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_length", "joint_idx", "length"), &SkeletonModification3DFABRIK::fabrik_joint_set_length);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_magnet", "joint_idx"), &SkeletonModification3DFABRIK::fabrik_joint_get_magnet);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_magnet", "joint_idx", "magnet_position"), &SkeletonModification3DFABRIK::fabrik_joint_set_magnet);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_auto_calculate_length", "joint_idx"), &SkeletonModification3DFABRIK::fabrik_joint_get_auto_calculate_length);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_auto_calculate_length", "joint_idx", "auto_calculate_length"), &SkeletonModification3DFABRIK::fabrik_joint_set_auto_calculate_length);
	ClassDB::bind_method(D_METHOD("fabrik_joint_auto_calculate_length", "joint_idx"), &SkeletonModification3DFABRIK::fabrik_joint_auto_calculate_length);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_use_tip_node", "joint_idx"), &SkeletonModification3DFABRIK::fabrik_joint_get_use_tip_node);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_use_tip_node", "joint_idx", "use_tip_node"), &SkeletonModification3DFABRIK::fabrik_joint_set_use_tip_node);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_tip_node", "joint_idx"), &SkeletonModification3DFABRIK::fabrik_joint_get_tip_node);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_tip_node", "joint_idx", "tip_node"), &SkeletonModification3DFABRIK::fabrik_joint_set_tip_node);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_use_target_basis", "joint_idx"), &SkeletonModification3DFABRIK::fabrik_joint_get_use_target_basis);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_use_target_basis", "joint_idx", "use_target_basis"), &SkeletonModification3DFABRIK::fabrik_joint_set_use_target_basis);

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
