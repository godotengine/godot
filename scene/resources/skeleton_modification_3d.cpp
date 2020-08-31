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
						PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DEFERRED_SET_RESOURCE | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE));
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

void SkeletonModificationStack3D::execute(float delta, int execution_mode) {
	ERR_FAIL_COND_MSG(!is_setup || skeleton == nullptr || is_queued_for_deletion(),
			"Modification stack is not properly setup and therefore cannot execute!");

	if (!skeleton->is_inside_tree()) {
		ERR_PRINT_ONCE("Skeleton is not inside SceneTree! Cannot execute modification!");
		return;
	}

	if (!enabled) {
		return;
	}

	for (int i = 0; i < modifications.size(); i++) {
		if (!modifications[i].is_valid()) {
			continue;
		}

		if (modifications[i]->get_execution_mode() == execution_mode) {
			modifications.get(i)->execute(delta);
		}
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
	ClassDB::bind_method(D_METHOD("execute", "delta", "execution_mode"), &SkeletonModificationStack3D::execute);

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

	ClassDB::bind_method(D_METHOD("get_skeleton"), &SkeletonModificationStack3D::get_skeleton);

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
	strength = 1.0;
}

///////////////////////////////////////
// Modification3D
///////////////////////////////////////

void SkeletonModification3D::execute(float delta) {
	if (get_script_instance()) {
		if (get_script_instance()->has_method("execute")) {
			get_script_instance()->call("execute", delta);
		}
	}

	if (!enabled)
		return;
}

void SkeletonModification3D::setup_modification(SkeletonModificationStack3D *p_stack) {
	stack = p_stack;
	if (stack) {
		is_setup = true;
	} else {
		WARN_PRINT("Could not setup modification with name " + this->get_name());
	}

	if (get_script_instance()) {
		if (get_script_instance()->has_method("setup_modification")) {
			get_script_instance()->call("setup_modification", p_stack);
		}
	}
}

void SkeletonModification3D::set_enabled(bool p_enabled) {
	enabled = p_enabled;
}

bool SkeletonModification3D::get_enabled() {
	return enabled;
}

// Helper function. Copied from the 2D IK PR. Needed for CCDIK.
float SkeletonModification3D::clamp_angle(float angle, float min_bound, float max_bound, bool invert) {
	// Map to the 0 to 360 range (in radians though) instead of the -180 to 180 range.
	if (angle < 0) {
		angle = (Math_PI * 2) + angle;
	}

	// Make min and max in the range of 0 to 360 (in radians), and make sure they are in the right order
	if (min_bound < 0) {
		min_bound = (Math_PI * 2) + min_bound;
	}
	if (max_bound < 0) {
		max_bound = (Math_PI * 2) + max_bound;
	}
	if (min_bound > max_bound) {
		float tmp = min_bound;
		min_bound = max_bound;
		max_bound = tmp;
	}

	// Note: May not be the most optimal way to clamp, but it always constraints to the nearest angle.
	if (invert == false) {
		if (angle < min_bound || angle > max_bound) {
			Vector2 min_bound_vec = Vector2(Math::cos(min_bound), Math::sin(min_bound));
			Vector2 max_bound_vec = Vector2(Math::cos(max_bound), Math::sin(max_bound));
			Vector2 angle_vec = Vector2(Math::cos(angle), Math::sin(angle));

			if (angle_vec.distance_squared_to(min_bound_vec) <= angle_vec.distance_squared_to(max_bound_vec)) {
				angle = min_bound;
			} else {
				angle = max_bound;
			}
		}
	} else {
		if (angle > min_bound && angle < max_bound) {
			Vector2 min_bound_vec = Vector2(Math::cos(min_bound), Math::sin(min_bound));
			Vector2 max_bound_vec = Vector2(Math::cos(max_bound), Math::sin(max_bound));
			Vector2 angle_vec = Vector2(Math::cos(angle), Math::sin(angle));

			if (angle_vec.distance_squared_to(min_bound_vec) <= angle_vec.distance_squared_to(max_bound_vec)) {
				angle = min_bound;
			} else {
				angle = max_bound;
			}
		}
	}
	return angle;
}

bool SkeletonModification3D::_print_execution_error(bool p_condition, String p_message) {
	if (p_condition && !execution_error_found) {
		ERR_PRINT(p_message);
		execution_error_found = true;
	}
	return p_condition;
}

SkeletonModificationStack3D *SkeletonModification3D::get_modification_stack() {
	return stack;
}

void SkeletonModification3D::set_is_setup(bool p_is_setup) {
	is_setup = p_is_setup;
}

bool SkeletonModification3D::get_is_setup() const {
	return is_setup;
}

void SkeletonModification3D::set_execution_mode(int p_mode) {
	execution_mode = p_mode;
}

int SkeletonModification3D::get_execution_mode() const {
	return execution_mode;
}

void SkeletonModification3D::_bind_methods() {
	BIND_VMETHOD(MethodInfo("execute", PropertyInfo(Variant::FLOAT, "delta")));
	BIND_VMETHOD(MethodInfo("setup_modification", PropertyInfo(Variant::OBJECT, "modification_stack", PROPERTY_HINT_RESOURCE_TYPE, "SkeletonModificationStack3D")));

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SkeletonModification3D::set_enabled);
	ClassDB::bind_method(D_METHOD("get_enabled"), &SkeletonModification3D::get_enabled);
	ClassDB::bind_method(D_METHOD("get_modification_stack"), &SkeletonModification3D::get_modification_stack);
	ClassDB::bind_method(D_METHOD("set_is_setup", "is_setup"), &SkeletonModification3D::set_is_setup);
	ClassDB::bind_method(D_METHOD("get_is_setup"), &SkeletonModification3D::get_is_setup);
	ClassDB::bind_method(D_METHOD("set_execution_mode", "execution_mode"), &SkeletonModification3D::set_execution_mode);
	ClassDB::bind_method(D_METHOD("get_execution_mode"), &SkeletonModification3D::get_execution_mode);
	ClassDB::bind_method(D_METHOD("clamp_angle", "angle", "min", "max", "invert"), &SkeletonModification3D::clamp_angle);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "execution_mode", PROPERTY_HINT_ENUM, "process, physics_process"), "set_execution_mode", "get_execution_mode");
}

SkeletonModification3D::SkeletonModification3D() {
	stack = nullptr;
	is_setup = false;
}

///////////////////////////////////////
// LookAt
///////////////////////////////////////

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

void SkeletonModification3DLookAt::execute(float delta) {
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

	Transform new_bone_trans = stack->skeleton->get_bone_local_pose_override(bone_idx);
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

void SkeletonModification3DLookAt::setup_modification(SkeletonModificationStack3D *p_stack) {
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
	execution_error_found = false;
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
	_change_notify();
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

void SkeletonModification3DCCDIK::execute(float delta) {
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
	for (int i = 0; i < ccdik_data_chain.size(); i++) {
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
		for (int i = 0; i < ccdik_data_chain.size(); i++) {
			for (int j = i; j < ccdik_data_chain.size(); j++) {
				_execute_ccdik_joint(j, node_target, node_tip);
			}
		}
	} else {
		for (int i = 0; i < ccdik_data_chain.size(); i++) {
			_execute_ccdik_joint(i, node_target, node_tip);
		}
	}

	execution_error_found = false;
}

void SkeletonModification3DCCDIK::_execute_ccdik_joint(int p_joint_idx, Node3D *target, Node3D *tip) {
	CCDIK_Joint_Data ccdik_data = ccdik_data_chain[p_joint_idx];

	if (_print_execution_error(ccdik_data.bone_idx < 0 || ccdik_data.bone_idx > stack->skeleton->get_bone_count(),
				"CCDIK joint: bone index for joint" + itos(p_joint_idx) + " not found. Cannot execute modification!")) {
		return;
	}

	Transform bone_trans = stack->skeleton->global_pose_to_local_pose(ccdik_data.bone_idx, stack->skeleton->get_bone_global_pose(ccdik_data.bone_idx));
	Transform tip_trans = stack->skeleton->global_pose_to_local_pose(ccdik_data.bone_idx, stack->skeleton->world_transform_to_global_pose(tip->get_global_transform()));
	Transform target_trans = stack->skeleton->global_pose_to_local_pose(ccdik_data.bone_idx, stack->skeleton->world_transform_to_global_pose(target->get_global_transform()));

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

	// TODO: probably needs to be redone so it works with the code above...
	if (ccdik_data.enable_constraint) {
		Vector3 rotation_axis;
		float rotation_angle;
		bone_trans.basis.get_axis_angle(rotation_axis, rotation_angle);

		rotation_angle = clamp_angle(rotation_angle, ccdik_data.constraint_angle_min, ccdik_data.constraint_angle_max, ccdik_data.constraint_angles_invert);
		bone_trans.basis.set_axis_angle(rotation_axis, rotation_angle);
	}

	stack->skeleton->set_bone_local_pose_override(ccdik_data.bone_idx, bone_trans, stack->strength, true);
	stack->skeleton->force_update_bone_children_transforms(ccdik_data.bone_idx);
}

void SkeletonModification3DCCDIK::setup_modification(SkeletonModificationStack3D *p_stack) {
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
		WARN_PRINT("Cannot update cache: modification is not properly setup!");
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
		WARN_PRINT("Cannot update cache: modification is not properly setup!");
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
	execution_error_found = false;
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
	execution_error_found = false;
	_change_notify();
}

int SkeletonModification3DCCDIK::ccdik_joint_get_ccdik_axis(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, ccdik_data_chain.size(), -1);
	return ccdik_data_chain[p_joint_idx].ccdik_axis;
}

void SkeletonModification3DCCDIK::ccdik_joint_set_ccdik_axis(int p_joint_idx, int p_axis) {
	ERR_FAIL_INDEX(p_joint_idx, ccdik_data_chain.size());
	ERR_FAIL_COND_MSG(p_axis < 0, "CCDIK axis is out of range: The axis mode is too low!");
	ccdik_data_chain.write[p_joint_idx].ccdik_axis = p_axis;
	_change_notify();
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
	execution_error_found = false;
	_change_notify();
}

void SkeletonModification3DCCDIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification3DCCDIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification3DCCDIK::get_target_node);

	ClassDB::bind_method(D_METHOD("set_tip_node", "tip_nodepath"), &SkeletonModification3DCCDIK::set_tip_node);
	ClassDB::bind_method(D_METHOD("get_tip_node"), &SkeletonModification3DCCDIK::get_tip_node);

	ClassDB::bind_method(D_METHOD("set_use_high_quality_solve", "high_quality_solve"), &SkeletonModification3DCCDIK::set_use_high_quality_solve);
	ClassDB::bind_method(D_METHOD("get_use_high_quality_solve"), &SkeletonModification3DCCDIK::get_use_high_quality_solve);

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
		} else if (what == "roll") {
			fabrik_joint_set_roll(which, Math::deg2rad(float(p_value)));
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
		} else if (what == "roll") {
			r_ret = Math::rad2deg(fabrik_joint_get_roll(which));
		}
		return true;
	}
	return true;
}

void SkeletonModification3DFABRIK::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		String base_string = "joint_data/" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::STRING_NAME, base_string + "bone_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::INT, base_string + "bone_index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "roll", PROPERTY_HINT_RANGE, "-360,360,0.01", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "auto_calculate_length", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));

		if (!fabrik_data_chain[i].auto_calculate_length) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "length", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		} else {
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "use_tip_node", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			if (fabrik_data_chain[i].use_tip_node) {
				p_list->push_back(PropertyInfo(Variant::NODE_PATH, base_string + "tip_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D", PROPERTY_USAGE_DEFAULT));
			}
		}

		// Cannot apply magnet to the origin of the chain, as it will not do anything.
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

	if (target_node_cache.is_null()) {
		_print_execution_error(true, "Target cache is out of date. Attempting to update...");
		update_target_cache();
		return;
	}

	if (_print_execution_error(fabrik_data_chain.size() <= 1, "FABRIK requires at least two joints to operate. Cannot execute modification!")) {
		return;
	}

	Node3D *node_target = Object::cast_to<Node3D>(ObjectDB::get_instance(target_node_cache));
	if (_print_execution_error(!node_target || !node_target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!")) {
		return;
	}

	// Verify that all joints have a valid bone ID, and that all bone lengths are zero or more
	// Also, while we are here, apply magnet positions.
	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		if (_print_execution_error(fabrik_data_chain[i].bone_idx < 0, "FABRIK Joint " + itos(i) + " has an invalid bone ID. Cannot execute!")) {
			return;
		}

		if (fabrik_data_chain[i].length < 0 && fabrik_data_chain[i].auto_calculate_length) {
			fabrik_joint_auto_calculate_length(i);
		}
		if (_print_execution_error(fabrik_data_chain[i].length < 0, "FABRIK Joint " + itos(i) + " has an invalid joint length. Cannot execute!")) {
			return;
		}

		Transform local_pose_override = stack->skeleton->get_bone_local_pose_override(fabrik_data_chain[i].bone_idx);

		// Apply magnet positions:
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
		chain_apply();

		// update the target distance
		target_distance = stack->skeleton->global_pose_to_local_pose(fabrik_data_chain[final_joint_idx].bone_idx, target_global_pose).origin.length();

		// update chain iterations
		chain_iterations += 1;
		if (chain_iterations >= chain_max_iterations) {
			break;
		}
	}
	execution_error_found = false;
}

void SkeletonModification3DFABRIK::chain_backwards() {
	int final_bone_idx = fabrik_data_chain[final_joint_idx].bone_idx;
	Transform final_joint_trans = stack->skeleton->local_pose_to_global_pose(final_bone_idx, stack->skeleton->get_bone_local_pose_override(final_bone_idx));

	// Get the direction the final bone is facing in.
	stack->skeleton->update_bone_rest_forward_vector(final_bone_idx);
	Transform final_bone_direction_trans = final_joint_trans.looking_at(target_global_pose.origin, Vector3(0, 1, 0));
	final_bone_direction_trans.basis = stack->skeleton->global_pose_z_forward_to_bone_forward(final_bone_idx, final_bone_direction_trans.basis);
	Vector3 direction = final_bone_direction_trans.basis.xform(stack->skeleton->get_bone_axis_forward_vector(final_bone_idx)).normalized();

	// If set to override, then use the target's Basis rather than the bone's
	if (fabrik_data_chain[final_joint_idx].use_target_basis) {
		direction = target_global_pose.basis.xform(stack->skeleton->get_bone_axis_forward_vector(final_bone_idx)).normalized();
	}

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

void SkeletonModification3DFABRIK::chain_apply() {
	// NOTE: We do not need a forward pass with this FABRIK, because we reset/undo the joint positions to origin
	// in this function, after we apply rotation.
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
				current_trans.basis.rotate_local(forward_vector, fabrik_data_chain[i].roll);
			} else { // Use the target's Basis...
				current_trans.basis = target_global_pose.basis.orthonormalized().scaled(current_trans.basis.get_scale());
			}
		} else { // every other bone in the chain...

			int next_bone_idx = fabrik_data_chain[i + 1].bone_idx;
			Transform next_trans = stack->skeleton->local_pose_to_global_pose(next_bone_idx, stack->skeleton->get_bone_local_pose_override(next_bone_idx));

			// Get the forward direction that the basis is facing in right now.
			stack->skeleton->update_bone_rest_forward_vector(current_bone_idx);
			Vector3 forward_vector = stack->skeleton->get_bone_axis_forward_vector(current_bone_idx);
			// Rotate the bone towards the next bone in the chain:
			current_trans.basis.rotate_to_align(forward_vector, current_trans.origin.direction_to(next_trans.origin));
			current_trans.basis.rotate_local(forward_vector, fabrik_data_chain[i].roll);
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
		execution_error_found = false;
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
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: node is not in the scene tree!");
				target_node_cache = node->get_instance_id();

				execution_error_found = false;
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
						"Cannot update tip cache for joint " + itos(p_joint_idx) + ": node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update tip cache for joint " + itos(p_joint_idx) + ": node is not in scene tree!");
				fabrik_data_chain.write[p_joint_idx].tip_node_cache = node->get_instance_id();

				execution_error_found = false;
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
	execution_error_found = false;
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
	execution_error_found = false;
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
	execution_error_found = false;
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

	execution_error_found = false;
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
	execution_error_found = false;
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

float SkeletonModification3DFABRIK::fabrik_joint_get_roll(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, fabrik_data_chain.size(), 0.0);
	return fabrik_data_chain[p_joint_idx].roll;
}

void SkeletonModification3DFABRIK::fabrik_joint_set_roll(int p_joint_idx, float p_roll) {
	ERR_FAIL_INDEX(p_joint_idx, fabrik_data_chain.size());
	fabrik_data_chain.write[p_joint_idx].roll = p_roll;
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

///////////////////////////////////////
// Jiggle
///////////////////////////////////////

bool SkeletonModification3DJiggle::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, jiggle_data_chain.size(), false);

		if (what == "bone_name") {
			jiggle_joint_set_bone_name(which, p_value);
		} else if (what == "bone_index") {
			jiggle_joint_set_bone_index(which, p_value);
		} else if (what == "override_defaults") {
			jiggle_joint_set_override(which, p_value);
		} else if (what == "stiffness") {
			jiggle_joint_set_stiffness(which, p_value);
		} else if (what == "mass") {
			jiggle_joint_set_mass(which, p_value);
		} else if (what == "damping") {
			jiggle_joint_set_damping(which, p_value);
		} else if (what == "use_gravity") {
			jiggle_joint_set_use_gravity(which, p_value);
		} else if (what == "gravity") {
			jiggle_joint_set_gravity(which, p_value);
		} else if (what == "roll") {
			jiggle_joint_set_roll(which, Math::deg2rad(float(p_value)));
		}
		return true;
	} else {
		if (path == "use_colliders") {
			set_use_colliders(p_value);
		} else if (path == "collision_mask") {
			set_collision_mask(p_value);
		}
		return true;
	}
	return true;
}

bool SkeletonModification3DJiggle::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, jiggle_data_chain.size(), false);

		if (what == "bone_name") {
			r_ret = jiggle_joint_get_bone_name(which);
		} else if (what == "bone_index") {
			r_ret = jiggle_joint_get_bone_index(which);
		} else if (what == "override_defaults") {
			r_ret = jiggle_joint_get_override(which);
		} else if (what == "stiffness") {
			r_ret = jiggle_joint_get_stiffness(which);
		} else if (what == "mass") {
			r_ret = jiggle_joint_get_mass(which);
		} else if (what == "damping") {
			r_ret = jiggle_joint_get_damping(which);
		} else if (what == "use_gravity") {
			r_ret = jiggle_joint_get_use_gravity(which);
		} else if (what == "gravity") {
			r_ret = jiggle_joint_get_gravity(which);
		} else if (what == "roll") {
			r_ret = Math::rad2deg(jiggle_joint_get_roll(which));
		}
		return true;
	} else {
		if (path == "use_colliders") {
			r_ret = get_use_colliders();
		} else if (path == "collision_mask") {
			r_ret = get_collision_mask();
		}
		return true;
	}
	return true;
}

void SkeletonModification3DJiggle::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::BOOL, "use_colliders", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	if (use_colliders) {
		p_list->push_back(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS, "", PROPERTY_USAGE_DEFAULT));
	}

	for (int i = 0; i < jiggle_data_chain.size(); i++) {
		String base_string = "joint_data/" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::STRING_NAME, base_string + "bone_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::INT, base_string + "bone_index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "roll", PROPERTY_HINT_RANGE, "-360,360,0.01", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "override_defaults", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));

		if (jiggle_data_chain[i].override_defaults) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "stiffness", PROPERTY_HINT_RANGE, "0, 1000, 0.01", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "mass", PROPERTY_HINT_RANGE, "0, 1000, 0.01", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "damping", PROPERTY_HINT_RANGE, "0, 1, 0.01", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "use_gravity", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			if (jiggle_data_chain[i].use_gravity) {
				p_list->push_back(PropertyInfo(Variant::VECTOR3, base_string + "gravity", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			}
		}
	}
}

void SkeletonModification3DJiggle::execute(float delta) {
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
	Node3D *target = Object::cast_to<Node3D>(ObjectDB::get_instance(target_node_cache));
	_print_execution_error(!target || !target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!");

	for (int i = 0; i < jiggle_data_chain.size(); i++) {
		_execute_jiggle_joint(i, target, delta);
	}

	execution_error_found = false;
}

void SkeletonModification3DJiggle::_execute_jiggle_joint(int p_joint_idx, Node3D *target, float delta) {
	// Adopted from: https://wiki.unity3d.com/index.php/JiggleBone
	// With modifications by TwistedTwigleg.

	if (jiggle_data_chain[p_joint_idx].bone_idx <= -2) {
		jiggle_data_chain.write[p_joint_idx].bone_idx = stack->skeleton->find_bone(jiggle_data_chain[p_joint_idx].bone_name);
	}
	if (_print_execution_error(
				jiggle_data_chain[p_joint_idx].bone_idx < 0 || jiggle_data_chain[p_joint_idx].bone_idx > stack->skeleton->get_bone_count(),
				"Jiggle joint " + itos(p_joint_idx) + " bone index is invald. Cannot execute modification!")) {
		return;
	}

	Transform new_bone_trans = stack->skeleton->local_pose_to_global_pose(jiggle_data_chain[p_joint_idx].bone_idx, stack->skeleton->get_bone_local_pose_override(jiggle_data_chain[p_joint_idx].bone_idx));
	Vector3 target_position = stack->skeleton->world_transform_to_global_pose(target->get_global_transform()).origin;

	jiggle_data_chain.write[p_joint_idx].force = (target_position - jiggle_data_chain[p_joint_idx].dynamic_position) * jiggle_data_chain[p_joint_idx].stiffness * delta;

	if (jiggle_data_chain[p_joint_idx].use_gravity) {
		Vector3 gravity_to_apply = new_bone_trans.basis.inverse().xform(jiggle_data_chain[p_joint_idx].gravity);
		jiggle_data_chain.write[p_joint_idx].force += gravity_to_apply * delta;
	}

	jiggle_data_chain.write[p_joint_idx].acceleration = jiggle_data_chain[p_joint_idx].force / jiggle_data_chain[p_joint_idx].mass;
	jiggle_data_chain.write[p_joint_idx].velocity += jiggle_data_chain[p_joint_idx].acceleration * (1 - jiggle_data_chain[p_joint_idx].damping);

	jiggle_data_chain.write[p_joint_idx].dynamic_position += jiggle_data_chain[p_joint_idx].velocity + jiggle_data_chain[p_joint_idx].force;
	jiggle_data_chain.write[p_joint_idx].dynamic_position += new_bone_trans.origin - jiggle_data_chain[p_joint_idx].last_position;
	jiggle_data_chain.write[p_joint_idx].last_position = new_bone_trans.origin;

	// Collision detection/response
	// (Does not run in the editor, unlike the 2D version. Not sure why though...)
	if (use_colliders) {
		if (execution_mode == SkeletonModificationStack3D::EXECUTION_MODE::execution_mode_physics_process) {
			Ref<World3D> world_3d = stack->skeleton->get_world_3d();
			ERR_FAIL_COND(world_3d.is_null());
			PhysicsDirectSpaceState3D *space_state = PhysicsServer3D::get_singleton()->space_get_direct_state(world_3d->get_space());
			PhysicsDirectSpaceState3D::RayResult ray_result;

			// Convert to world transforms, which is what the physics server needs
			Transform new_bone_trans_world = stack->skeleton->global_pose_to_world_transform(new_bone_trans);
			Transform dynamic_position_world = stack->skeleton->global_pose_to_world_transform(Transform(Basis(), jiggle_data_chain[p_joint_idx].dynamic_position));

			// Add exception support?
			bool ray_hit = space_state->intersect_ray(new_bone_trans_world.origin, dynamic_position_world.get_origin(),
					ray_result, Set<RID>(), collision_mask);

			if (ray_hit) {
				jiggle_data_chain.write[p_joint_idx].dynamic_position = jiggle_data_chain[p_joint_idx].last_noncollision_position;
				jiggle_data_chain.write[p_joint_idx].acceleration = Vector3(0, 0, 0);
				jiggle_data_chain.write[p_joint_idx].velocity = Vector3(0, 0, 0);
			} else {
				jiggle_data_chain.write[p_joint_idx].last_noncollision_position = jiggle_data_chain[p_joint_idx].dynamic_position;
			}

		} else {
			WARN_PRINT_ONCE("Jiggle modifier: You cannot detect colliders without the stack mode being set to _physics_process!");
		}
	}

	// Get the forward direction that the basis is facing in right now.
	stack->skeleton->update_bone_rest_forward_vector(jiggle_data_chain[p_joint_idx].bone_idx);
	Vector3 forward_vector = stack->skeleton->get_bone_axis_forward_vector(jiggle_data_chain[p_joint_idx].bone_idx);

	// Rotate the bone using the dynamic position!
	new_bone_trans.basis.rotate_to_align(forward_vector, new_bone_trans.origin.direction_to(jiggle_data_chain[p_joint_idx].dynamic_position));

	// Roll
	new_bone_trans.basis.rotate_local(forward_vector, jiggle_data_chain[p_joint_idx].roll);

	new_bone_trans = stack->skeleton->global_pose_to_local_pose(jiggle_data_chain[p_joint_idx].bone_idx, new_bone_trans);
	stack->skeleton->set_bone_local_pose_override(jiggle_data_chain[p_joint_idx].bone_idx, new_bone_trans, stack->strength, true);
	stack->skeleton->force_update_bone_children_transforms(jiggle_data_chain[p_joint_idx].bone_idx);
}

void SkeletonModification3DJiggle::_update_jiggle_joint_data() {
	for (int i = 0; i < jiggle_data_chain.size(); i++) {
		if (!jiggle_data_chain[i].override_defaults) {
			jiggle_joint_set_stiffness(i, stiffness);
			jiggle_joint_set_mass(i, mass);
			jiggle_joint_set_damping(i, damping);
			jiggle_joint_set_use_gravity(i, use_gravity);
			jiggle_joint_set_gravity(i, gravity);
		}
	}
}

void SkeletonModification3DJiggle::setup_modification(SkeletonModificationStack3D *p_stack) {
	stack = p_stack;

	if (stack) {
		is_setup = true;
		execution_error_found = false;

		if (stack->skeleton) {
			for (int i = 0; i < jiggle_data_chain.size(); i++) {
				int bone_idx = jiggle_data_chain[i].bone_idx;
				if (bone_idx > 0 && bone_idx < stack->skeleton->get_bone_count()) {
					jiggle_data_chain.write[i].dynamic_position = stack->skeleton->local_pose_to_global_pose(bone_idx, stack->skeleton->get_bone_local_pose_override(bone_idx)).origin;
				}
			}
		}

		update_cache();
	}
}

void SkeletonModification3DJiggle::update_cache() {
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
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: node is not in the scene tree!");
				target_node_cache = node->get_instance_id();

				execution_error_found = false;
			}
		}
	}
}

void SkeletonModification3DJiggle::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_cache();
}

NodePath SkeletonModification3DJiggle::get_target_node() const {
	return target_node;
}

void SkeletonModification3DJiggle::set_stiffness(float p_stiffness) {
	ERR_FAIL_COND_MSG(p_stiffness < 0, "Stiffness cannot be set to a negative value!");
	stiffness = p_stiffness;
	_update_jiggle_joint_data();
}

float SkeletonModification3DJiggle::get_stiffness() const {
	return stiffness;
}

void SkeletonModification3DJiggle::set_mass(float p_mass) {
	ERR_FAIL_COND_MSG(p_mass < 0, "Mass cannot be set to a negative value!");
	mass = p_mass;
	_update_jiggle_joint_data();
}

float SkeletonModification3DJiggle::get_mass() const {
	return mass;
}

void SkeletonModification3DJiggle::set_damping(float p_damping) {
	ERR_FAIL_COND_MSG(p_damping < 0, "Damping cannot be set to a negative value!");
	ERR_FAIL_COND_MSG(p_damping > 1, "Damping cannot be more than one!");
	damping = p_damping;
	_update_jiggle_joint_data();
}

float SkeletonModification3DJiggle::get_damping() const {
	return damping;
}

void SkeletonModification3DJiggle::set_use_gravity(bool p_use_gravity) {
	use_gravity = p_use_gravity;
	_update_jiggle_joint_data();
}

bool SkeletonModification3DJiggle::get_use_gravity() const {
	return use_gravity;
}

void SkeletonModification3DJiggle::set_gravity(Vector3 p_gravity) {
	gravity = p_gravity;
	_update_jiggle_joint_data();
}

Vector3 SkeletonModification3DJiggle::get_gravity() const {
	return gravity;
}

void SkeletonModification3DJiggle::set_use_colliders(bool p_use_collider) {
	use_colliders = p_use_collider;
	_change_notify();
}

bool SkeletonModification3DJiggle::get_use_colliders() const {
	return use_colliders;
}

void SkeletonModification3DJiggle::set_collision_mask(int p_mask) {
	collision_mask = p_mask;
}

int SkeletonModification3DJiggle::get_collision_mask() const {
	return collision_mask;
}

// Jiggle joint data functions
int SkeletonModification3DJiggle::get_jiggle_data_chain_length() {
	return jiggle_data_chain.size();
}

void SkeletonModification3DJiggle::set_jiggle_data_chain_length(int p_length) {
	ERR_FAIL_COND(p_length < 0);
	jiggle_data_chain.resize(p_length);
	execution_error_found = false;
	_change_notify();
}

void SkeletonModification3DJiggle::jiggle_joint_set_bone_name(int joint_idx, String p_name) {
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());

	jiggle_data_chain.write[joint_idx].bone_name = p_name;
	if (stack && stack->skeleton) {
		jiggle_data_chain.write[joint_idx].bone_idx = stack->skeleton->find_bone(p_name);
	}
	execution_error_found = false;
	_change_notify();
}

String SkeletonModification3DJiggle::jiggle_joint_get_bone_name(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), "");
	return jiggle_data_chain[joint_idx].bone_name;
}

int SkeletonModification3DJiggle::jiggle_joint_get_bone_index(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), -1);
	return jiggle_data_chain[joint_idx].bone_idx;
}

void SkeletonModification3DJiggle::jiggle_joint_set_bone_index(int joint_idx, int p_bone_idx) {
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");
	jiggle_data_chain.write[joint_idx].bone_idx = p_bone_idx;

	if (stack) {
		if (stack->skeleton) {
			jiggle_data_chain.write[joint_idx].bone_name = stack->skeleton->get_bone_name(p_bone_idx);
		}
	}
	execution_error_found = false;
	_change_notify();
}

void SkeletonModification3DJiggle::jiggle_joint_set_override(int joint_idx, bool p_override) {
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].override_defaults = p_override;
	_update_jiggle_joint_data();
	_change_notify();
}

bool SkeletonModification3DJiggle::jiggle_joint_get_override(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), false);
	return jiggle_data_chain[joint_idx].override_defaults;
}

void SkeletonModification3DJiggle::jiggle_joint_set_stiffness(int joint_idx, float p_stiffness) {
	ERR_FAIL_COND_MSG(p_stiffness < 0, "Stiffness cannot be set to a negative value!");
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].stiffness = p_stiffness;
}

float SkeletonModification3DJiggle::jiggle_joint_get_stiffness(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), -1);
	return jiggle_data_chain[joint_idx].stiffness;
}

void SkeletonModification3DJiggle::jiggle_joint_set_mass(int joint_idx, float p_mass) {
	ERR_FAIL_COND_MSG(p_mass < 0, "Mass cannot be set to a negative value!");
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].mass = p_mass;
}

float SkeletonModification3DJiggle::jiggle_joint_get_mass(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), -1);
	return jiggle_data_chain[joint_idx].mass;
}

void SkeletonModification3DJiggle::jiggle_joint_set_damping(int joint_idx, float p_damping) {
	ERR_FAIL_COND_MSG(p_damping < 0, "Damping cannot be set to a negative value!");
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].damping = p_damping;
}

float SkeletonModification3DJiggle::jiggle_joint_get_damping(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), -1);
	return jiggle_data_chain[joint_idx].damping;
}

void SkeletonModification3DJiggle::jiggle_joint_set_use_gravity(int joint_idx, bool p_use_gravity) {
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].use_gravity = p_use_gravity;
	_change_notify();
}

bool SkeletonModification3DJiggle::jiggle_joint_get_use_gravity(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), false);
	return jiggle_data_chain[joint_idx].use_gravity;
}

void SkeletonModification3DJiggle::jiggle_joint_set_gravity(int joint_idx, Vector3 p_gravity) {
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].gravity = p_gravity;
}

Vector3 SkeletonModification3DJiggle::jiggle_joint_get_gravity(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), Vector3(0, 0, 0));
	return jiggle_data_chain[joint_idx].gravity;
}

void SkeletonModification3DJiggle::jiggle_joint_set_roll(int joint_idx, float p_roll) {
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].roll = p_roll;
}

float SkeletonModification3DJiggle::jiggle_joint_get_roll(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), 0.0);
	return jiggle_data_chain[joint_idx].roll;
}

void SkeletonModification3DJiggle::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification3DJiggle::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification3DJiggle::get_target_node);

	ClassDB::bind_method(D_METHOD("set_jiggle_data_chain_length", "length"), &SkeletonModification3DJiggle::set_jiggle_data_chain_length);
	ClassDB::bind_method(D_METHOD("get_jiggle_data_chain_length"), &SkeletonModification3DJiggle::get_jiggle_data_chain_length);

	ClassDB::bind_method(D_METHOD("set_stiffness", "stiffness"), &SkeletonModification3DJiggle::set_stiffness);
	ClassDB::bind_method(D_METHOD("get_stiffness"), &SkeletonModification3DJiggle::get_stiffness);
	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &SkeletonModification3DJiggle::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &SkeletonModification3DJiggle::get_mass);
	ClassDB::bind_method(D_METHOD("set_damping", "damping"), &SkeletonModification3DJiggle::set_damping);
	ClassDB::bind_method(D_METHOD("get_damping"), &SkeletonModification3DJiggle::get_damping);
	ClassDB::bind_method(D_METHOD("set_use_gravity", "use_gravity"), &SkeletonModification3DJiggle::set_use_gravity);
	ClassDB::bind_method(D_METHOD("get_use_gravity"), &SkeletonModification3DJiggle::get_use_gravity);
	ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &SkeletonModification3DJiggle::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity"), &SkeletonModification3DJiggle::get_gravity);

	ClassDB::bind_method(D_METHOD("set_use_colliders", "use_colliders"), &SkeletonModification3DJiggle::set_use_colliders);
	ClassDB::bind_method(D_METHOD("get_use_colliders"), &SkeletonModification3DJiggle::get_use_colliders);
	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &SkeletonModification3DJiggle::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &SkeletonModification3DJiggle::get_collision_mask);

	// Jiggle joint data functions
	ClassDB::bind_method(D_METHOD("jiggle_joint_set_bone_name", "joint_idx", "name"), &SkeletonModification3DJiggle::jiggle_joint_set_bone_name);
	ClassDB::bind_method(D_METHOD("jiggle_joint_get_bone_name", "joint_idx"), &SkeletonModification3DJiggle::jiggle_joint_get_bone_name);
	ClassDB::bind_method(D_METHOD("jiggle_joint_set_bone_index", "joint_idx", "bone_idx"), &SkeletonModification3DJiggle::jiggle_joint_set_bone_index);
	ClassDB::bind_method(D_METHOD("jiggle_joint_get_bone_index", "joint_idx"), &SkeletonModification3DJiggle::jiggle_joint_get_bone_index);
	ClassDB::bind_method(D_METHOD("jiggle_joint_set_override", "joint_idx", "override"), &SkeletonModification3DJiggle::jiggle_joint_set_override);
	ClassDB::bind_method(D_METHOD("jiggle_joint_get_override", "joint_idx"), &SkeletonModification3DJiggle::jiggle_joint_get_override);
	ClassDB::bind_method(D_METHOD("jiggle_joint_set_stiffness", "joint_idx", "stiffness"), &SkeletonModification3DJiggle::jiggle_joint_set_stiffness);
	ClassDB::bind_method(D_METHOD("jiggle_joint_get_stiffness", "joint_idx"), &SkeletonModification3DJiggle::jiggle_joint_get_stiffness);
	ClassDB::bind_method(D_METHOD("jiggle_joint_set_mass", "joint_idx", "mass"), &SkeletonModification3DJiggle::jiggle_joint_set_mass);
	ClassDB::bind_method(D_METHOD("jiggle_joint_get_mass", "joint_idx"), &SkeletonModification3DJiggle::jiggle_joint_get_mass);
	ClassDB::bind_method(D_METHOD("jiggle_joint_set_damping", "joint_idx", "damping"), &SkeletonModification3DJiggle::jiggle_joint_set_damping);
	ClassDB::bind_method(D_METHOD("jiggle_joint_get_damping", "joint_idx"), &SkeletonModification3DJiggle::jiggle_joint_get_damping);
	ClassDB::bind_method(D_METHOD("jiggle_joint_set_use_gravity", "joint_idx", "use_gravity"), &SkeletonModification3DJiggle::jiggle_joint_set_use_gravity);
	ClassDB::bind_method(D_METHOD("jiggle_joint_get_use_gravity", "joint_idx"), &SkeletonModification3DJiggle::jiggle_joint_get_use_gravity);
	ClassDB::bind_method(D_METHOD("jiggle_joint_set_gravity", "joint_idx", "gravity"), &SkeletonModification3DJiggle::jiggle_joint_set_gravity);
	ClassDB::bind_method(D_METHOD("jiggle_joint_get_gravity", "joint_idx"), &SkeletonModification3DJiggle::jiggle_joint_get_gravity);
	ClassDB::bind_method(D_METHOD("jiggle_joint_set_roll", "joint_idx", "roll"), &SkeletonModification3DJiggle::jiggle_joint_set_roll);
	ClassDB::bind_method(D_METHOD("jiggle_joint_get_roll", "joint_idx"), &SkeletonModification3DJiggle::jiggle_joint_get_roll);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "jiggle_data_chain_length", PROPERTY_HINT_RANGE, "0,100,1"), "set_jiggle_data_chain_length", "get_jiggle_data_chain_length");
	ADD_GROUP("Default Joint Settings", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stiffness"), "set_stiffness", "get_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping", PROPERTY_HINT_RANGE, "0, 1, 0.01"), "set_damping", "get_damping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_gravity"), "set_use_gravity", "get_use_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
	ADD_GROUP("", "");
}

SkeletonModification3DJiggle::SkeletonModification3DJiggle() {
	stack = nullptr;
	is_setup = false;
	jiggle_data_chain = Vector<Jiggle_Joint_Data>();
	stiffness = 3;
	mass = 0.75;
	damping = 0.75;
	use_gravity = false;
	gravity = Vector3(0, -6.0, 0);
	enabled = true;
}

SkeletonModification3DJiggle::~SkeletonModification3DJiggle() {
}

///////////////////////////////////////
// TwoBoneIK
///////////////////////////////////////

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
		set_joint_one_roll(Math::deg2rad(float(p_value)));
	} else if (path == "joint_two/bone_name") {
		set_joint_two_bone_name(p_value);
	} else if (path == "joint_two/bone_idx") {
		set_joint_two_bone_idx(p_value);
	} else if (path == "joint_two/roll") {
		set_joint_two_roll(Math::deg2rad(float(p_value)));
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

void SkeletonModification3DTwoBoneIK::execute(float delta) {
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
	Transform target_trans = stack->skeleton->world_transform_to_global_pose(target->get_global_transform());

	Transform bone_one_trans;
	Transform bone_two_trans;

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
		Transform pole_trans = stack->skeleton->world_transform_to_global_pose(pole->get_global_transform());

		bone_one_trans = stack->skeleton->local_pose_to_global_pose(joint_one_bone_idx, stack->skeleton->get_bone_local_pose_override(joint_one_bone_idx));
		bone_one_trans = bone_one_trans.looking_at(pole_trans.origin, Vector3(0, 1, 0));
		bone_one_trans.basis = stack->skeleton->global_pose_z_forward_to_bone_forward(joint_one_bone_idx, bone_one_trans.basis);
		stack->skeleton->update_bone_rest_forward_vector(joint_one_bone_idx);
		bone_one_trans.basis.rotate_local(stack->skeleton->get_bone_axis_forward_vector(joint_one_bone_idx), joint_one_roll);
		stack->skeleton->set_bone_local_pose_override(joint_one_bone_idx, stack->skeleton->global_pose_to_local_pose(joint_one_bone_idx, bone_one_trans), stack->strength, true);
		stack->skeleton->force_update_bone_children_transforms(joint_one_bone_idx);

		bone_two_trans = stack->skeleton->local_pose_to_global_pose(joint_two_bone_idx, stack->skeleton->get_bone_local_pose_override(joint_two_bone_idx));
		bone_two_trans = bone_two_trans.looking_at(target_trans.origin, Vector3(0, 1, 0));
		bone_two_trans.basis = stack->skeleton->global_pose_z_forward_to_bone_forward(joint_two_bone_idx, bone_two_trans.basis);
		stack->skeleton->update_bone_rest_forward_vector(joint_two_bone_idx);
		bone_two_trans.basis.rotate_local(stack->skeleton->get_bone_axis_forward_vector(joint_two_bone_idx), joint_two_roll);
		stack->skeleton->set_bone_local_pose_override(joint_two_bone_idx, stack->skeleton->global_pose_to_local_pose(joint_two_bone_idx, bone_two_trans), stack->strength, true);
		stack->skeleton->force_update_bone_children_transforms(joint_two_bone_idx);
	} else {
		bone_one_trans = stack->skeleton->local_pose_to_global_pose(joint_one_bone_idx, stack->skeleton->get_bone_local_pose_override(joint_one_bone_idx));
		bone_two_trans = stack->skeleton->local_pose_to_global_pose(joint_two_bone_idx, stack->skeleton->get_bone_local_pose_override(joint_two_bone_idx));
	}

	Transform bone_two_tip_trans;
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

	float joint_one_to_target_length = bone_one_trans.origin.distance_to(target_trans.origin);
	if (joint_one_length + joint_two_length < joint_one_to_target_length) {
		// Set the target *just* out of reach to straighten the bones
		joint_one_to_target_length = joint_one_length + joint_two_length + 0.01;
	} else if (joint_one_to_target_length < joint_one_length) {
		// Place the target in reach so the solver doesn't do crazy things
		joint_one_to_target_length = joint_one_length;
	}

	// Get the square lengths for all three sides of the triangle we'll use to calculate the angles
	float sqr_one_length = joint_one_length * joint_one_length;
	float sqr_two_length = joint_two_length * joint_two_length;
	float sqr_three_length = joint_one_to_target_length * joint_one_to_target_length;

	// Calculate the angles for the first joint using the law of cosigns
	float ac_ab_0 = Math::acos(CLAMP(bone_two_tip_trans.origin.direction_to(bone_one_trans.origin).dot(bone_two_trans.origin.direction_to(bone_one_trans.origin)), -1, 1));
	float ac_at_0 = Math::acos(CLAMP(bone_one_trans.origin.direction_to(bone_two_tip_trans.origin).dot(bone_one_trans.origin.direction_to(target_trans.origin)), -1, 1));
	float ac_ab_1 = Math::acos(CLAMP((sqr_two_length - sqr_one_length - sqr_three_length) / (-2.0 * joint_one_length * joint_one_to_target_length), -1, 1));

	// Calculate the angles of rotation. Angle 0 is the extension/contraction axis, while angle 1 is the rotation axis to align the triangle to the target
	Vector3 axis_0 = bone_one_trans.origin.direction_to(bone_two_tip_trans.origin).cross(bone_one_trans.origin.direction_to(bone_two_trans.origin));
	Vector3 axis_1 = bone_one_trans.origin.direction_to(bone_two_tip_trans.origin).cross(bone_one_trans.origin.direction_to(target_trans.origin));

	// Make a quaternion with the delta rotation needed to rotate the first joint into alignment and apply it to the transform.
	Quat bone_one_quat = bone_one_trans.basis.get_rotation_quat();
	Quat rot_0 = Quat(bone_one_quat.inverse().xform(axis_0).normalized(), (ac_ab_1 - ac_ab_0));
	Quat rot_2 = Quat(bone_one_quat.inverse().xform(axis_1).normalized(), ac_at_0);
	bone_one_trans.basis.set_quat(bone_one_quat * (rot_0 * rot_2));

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
		float ba_bc_0 = Math::acos(CLAMP(bone_two_trans.origin.direction_to(bone_one_trans.origin).dot(bone_two_trans.origin.direction_to(bone_two_tip_trans.origin)), -1, 1));
		float ba_bc_1 = Math::acos(CLAMP((sqr_three_length - sqr_one_length - sqr_two_length) / (-2.0 * joint_one_length * joint_two_length), -1, 1));
		Quat bone_two_quat = bone_two_trans.basis.get_rotation_quat();
		Quat rot_1 = Quat(bone_two_quat.inverse().xform(axis_0).normalized(), (ba_bc_1 - ba_bc_0));
		bone_two_trans.basis.set_quat(bone_two_quat * rot_1);

		stack->skeleton->update_bone_rest_forward_vector(joint_two_bone_idx);
		bone_two_trans.basis.rotate_local(stack->skeleton->get_bone_axis_forward_vector(joint_two_bone_idx), joint_two_roll);

		bone_two_trans = stack->skeleton->global_pose_to_local_pose(joint_two_bone_idx, bone_two_trans);
		bone_two_trans.origin = Vector3(0, 0, 0);
		stack->skeleton->set_bone_local_pose_override(joint_two_bone_idx, bone_two_trans, stack->strength, true);
		stack->skeleton->force_update_bone_children_transforms(joint_two_bone_idx);
	}
}

void SkeletonModification3DTwoBoneIK::setup_modification(SkeletonModificationStack3D *p_stack) {
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
		WARN_PRINT("Cannot update target cache: modification is not properly setup!");
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
		WARN_PRINT("Cannot update tip cache: modification is not properly setup!");
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
		WARN_PRINT("Cannot update pole cache: modification is not properly setup!");
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
	_change_notify();
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
	_change_notify();
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
	_change_notify();
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

	Transform bone_one_rest_trans = stack->skeleton->get_bone_global_pose(joint_one_bone_idx);
	Transform bone_two_rest_trans = stack->skeleton->get_bone_global_pose(joint_two_bone_idx);

	joint_one_length = bone_one_rest_trans.origin.distance_to(bone_two_rest_trans.origin);

	if (use_tip_node) {
		if (tip_node_cache.is_null()) {
			update_cache_tip();
			WARN_PRINT("Tip cache is out of date. Updating...");
		}

		Node3D *tip = Object::cast_to<Node3D>(ObjectDB::get_instance(tip_node_cache));
		if (tip) {
			Transform bone_tip_trans = stack->skeleton->world_transform_to_global_pose(tip->get_global_transform());
			joint_two_length = bone_two_rest_trans.origin.distance_to(bone_tip_trans.origin);
		}
	} else {
		// Attempt to use children bones to get the length
		Vector<int> bone_two_children = stack->skeleton->get_bone_children(joint_two_bone_idx);
		if (bone_two_children.size() > 0) {
			joint_two_length = 0;
			for (int i = 0; i < bone_two_children.size(); i++) {
				joint_two_length += bone_two_rest_trans.origin.distance_to(
						stack->skeleton->local_pose_to_global_pose(bone_two_children[i], stack->skeleton->get_bone_rest(bone_two_children[i])).origin);
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
	_change_notify();
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
	_change_notify();
}

int SkeletonModification3DTwoBoneIK::get_joint_one_bone_idx() const {
	return joint_one_bone_idx;
}

void SkeletonModification3DTwoBoneIK::set_joint_one_length(float p_length) {
	joint_one_length = p_length;
}

float SkeletonModification3DTwoBoneIK::get_joint_one_length() const {
	return joint_one_length;
}

void SkeletonModification3DTwoBoneIK::set_joint_two_bone_name(String p_bone_name) {
	joint_two_bone_name = p_bone_name;
	if (stack && stack->skeleton) {
		joint_two_bone_idx = stack->skeleton->find_bone(p_bone_name);
	}
	execution_error_found = false;
	_change_notify();
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
	_change_notify();
}

int SkeletonModification3DTwoBoneIK::get_joint_two_bone_idx() const {
	return joint_two_bone_idx;
}

void SkeletonModification3DTwoBoneIK::set_joint_two_length(float p_length) {
	joint_two_length = p_length;
}

float SkeletonModification3DTwoBoneIK::get_joint_two_length() const {
	return joint_two_length;
}

void SkeletonModification3DTwoBoneIK::set_joint_one_roll(float p_roll) {
	joint_one_roll = p_roll;
}

float SkeletonModification3DTwoBoneIK::get_joint_one_roll() const {
	return joint_one_roll;
}

void SkeletonModification3DTwoBoneIK::set_joint_two_roll(float p_roll) {
	joint_two_roll = p_roll;
}

float SkeletonModification3DTwoBoneIK::get_joint_two_roll() const {
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

///////////////////////////////////////
// StackHolder
///////////////////////////////////////

bool SkeletonModification3DStackHolder::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path == "held_modification_stack") {
		set_held_modification_stack(p_value);
	}
	return true;
}

bool SkeletonModification3DStackHolder::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path == "held_modification_stack") {
		r_ret = get_held_modification_stack();
	}
	return true;
}

void SkeletonModification3DStackHolder::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::OBJECT, "held_modification_stack", PROPERTY_HINT_RESOURCE_TYPE, "SkeletonModificationStack3D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE));
}

void SkeletonModification3DStackHolder::execute(float delta) {
	ERR_FAIL_COND_MSG(!stack || !is_setup || stack->skeleton == nullptr,
			"Modification is not setup and therefore cannot execute!");

	if (held_modification_stack.is_valid()) {
		held_modification_stack->execute(delta, execution_mode);
	}
}

void SkeletonModification3DStackHolder::setup_modification(SkeletonModificationStack3D *p_stack) {
	stack = p_stack;

	if (stack != nullptr) {
		is_setup = true;

		if (held_modification_stack.is_valid()) {
			held_modification_stack->set_skeleton(stack->get_skeleton());
			held_modification_stack->setup();
		}
	}
}

void SkeletonModification3DStackHolder::set_held_modification_stack(Ref<SkeletonModificationStack3D> p_held_stack) {
	held_modification_stack = p_held_stack;

	if (is_setup && held_modification_stack.is_valid()) {
		held_modification_stack->set_skeleton(stack->get_skeleton());
		held_modification_stack->setup();
	}
}

Ref<SkeletonModificationStack3D> SkeletonModification3DStackHolder::get_held_modification_stack() const {
	return held_modification_stack;
}

void SkeletonModification3DStackHolder::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_held_modification_stack", "held_modification_stack"), &SkeletonModification3DStackHolder::set_held_modification_stack);
	ClassDB::bind_method(D_METHOD("get_held_modification_stack"), &SkeletonModification3DStackHolder::get_held_modification_stack);
}

SkeletonModification3DStackHolder::SkeletonModification3DStackHolder() {
	stack = nullptr;
	is_setup = false;
	enabled = true;
}

SkeletonModification3DStackHolder::~SkeletonModification3DStackHolder() {
}
