/*************************************************************************/
/*  skeleton_modification_2d.cpp                                         */
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

#include "skeleton_modification_2d.h"
#include "scene/2d/skeleton_2d.h"

///////////////////////////////////////
// ModificationStack2D
///////////////////////////////////////

void SkeletonModificationStack2D::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < modifications.size(); i++) {
		p_list->push_back(
				PropertyInfo(Variant::OBJECT, "modifications/" + itos(i),
						PROPERTY_HINT_RESOURCE_TYPE,
						"SkeletonModification2D",
						PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DEFERRED_SET_RESOURCE));
	}
}

bool SkeletonModificationStack2D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("modifications/")) {
		int mod_idx = path.get_slicec('/', 1).to_int();
		set_modification(mod_idx, p_value);
		return true;
	}
	return true;
}

bool SkeletonModificationStack2D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("modifications/")) {
		int mod_idx = path.get_slicec('/', 1).to_int();
		r_ret = get_modification(mod_idx);
		return true;
	}
	return true;
}

void SkeletonModificationStack2D::setup() {
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
		WARN_PRINT("Cannot setup SkeletonModificationStack2D: no Skeleton2D set!");
	}
}

void SkeletonModificationStack2D::execute(float delta) {
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
		modifications.get(i)->execute(delta);
	}
}

void SkeletonModificationStack2D::enable_all_modifications(bool p_enabled) {
	for (int i = 0; i < modifications.size(); i++) {
		if (!modifications[i].is_valid()) {
			continue;
		}
		modifications.get(i)->set_enabled(p_enabled);
	}
}

Ref<SkeletonModification2D> SkeletonModificationStack2D::get_modification(int p_mod_idx) const {
	ERR_FAIL_INDEX_V(p_mod_idx, modifications.size(), nullptr);
	return modifications[p_mod_idx];
}

void SkeletonModificationStack2D::add_modification(Ref<SkeletonModification2D> p_mod) {
	p_mod->setup_modification(this);
	modifications.push_back(p_mod);
}

void SkeletonModificationStack2D::delete_modification(int p_mod_idx) {
	ERR_FAIL_INDEX(p_mod_idx, modifications.size());
	modifications.remove(p_mod_idx);
}

void SkeletonModificationStack2D::set_modification(int p_mod_idx, Ref<SkeletonModification2D> p_mod) {
	ERR_FAIL_INDEX(p_mod_idx, modifications.size());

	if (p_mod == nullptr) {
		modifications.set(p_mod_idx, nullptr);
	} else {
		p_mod->setup_modification(this);
		modifications.set(p_mod_idx, p_mod);
	}
}

void SkeletonModificationStack2D::set_modification_count(int p_count) {
	modifications.resize(p_count);
	_change_notify();
}

int SkeletonModificationStack2D::get_modification_count() const {
	return modifications.size();
}

void SkeletonModificationStack2D::set_skeleton(Skeleton2D *p_skeleton) {
	skeleton = p_skeleton;
}

Skeleton2D *SkeletonModificationStack2D::get_skeleton() const {
	return skeleton;
}

bool SkeletonModificationStack2D::get_is_setup() const {
	return is_setup;
}

void SkeletonModificationStack2D::set_enabled(bool p_enabled) {
	enabled = p_enabled;
}

bool SkeletonModificationStack2D::get_enabled() const {
	return enabled;
}

void SkeletonModificationStack2D::set_strength(float p_strength) {
	ERR_FAIL_COND_MSG(p_strength < 0, "Strength cannot be less than zero!");
	ERR_FAIL_COND_MSG(p_strength > 1, "Strength cannot be more than one!");
	strength = p_strength;
}

float SkeletonModificationStack2D::get_strength() const {
	return strength;
}

void SkeletonModificationStack2D::set_execution_mode(int p_new_mode) {
	execution_mode = p_new_mode;
}

int SkeletonModificationStack2D::get_execution_mode() {
	return execution_mode;
}

void SkeletonModificationStack2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("setup"), &SkeletonModificationStack2D::setup);
	ClassDB::bind_method(D_METHOD("execute", "delta"), &SkeletonModificationStack2D::execute);

	ClassDB::bind_method(D_METHOD("enable_all_modifications", "enabled"), &SkeletonModificationStack2D::enable_all_modifications);
	ClassDB::bind_method(D_METHOD("get_modification", "mod_idx"), &SkeletonModificationStack2D::get_modification);
	ClassDB::bind_method(D_METHOD("add_modification", "modification"), &SkeletonModificationStack2D::add_modification);
	ClassDB::bind_method(D_METHOD("delete_modification", "mod_idx"), &SkeletonModificationStack2D::delete_modification);
	ClassDB::bind_method(D_METHOD("set_modification", "mod_idx", "modification"), &SkeletonModificationStack2D::set_modification);

	ClassDB::bind_method(D_METHOD("set_modification_count"), &SkeletonModificationStack2D::set_modification_count);
	ClassDB::bind_method(D_METHOD("get_modification_count"), &SkeletonModificationStack2D::get_modification_count);

	ClassDB::bind_method(D_METHOD("get_is_setup"), &SkeletonModificationStack2D::get_is_setup);

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SkeletonModificationStack2D::set_enabled);
	ClassDB::bind_method(D_METHOD("get_enabled"), &SkeletonModificationStack2D::get_enabled);

	ClassDB::bind_method(D_METHOD("set_strength", "strength"), &SkeletonModificationStack2D::set_strength);
	ClassDB::bind_method(D_METHOD("get_strength"), &SkeletonModificationStack2D::get_strength);

	ClassDB::bind_method(D_METHOD("set_execution_mode", "execution_mode"), &SkeletonModificationStack2D::set_execution_mode);
	ClassDB::bind_method(D_METHOD("get_execution_mode"), &SkeletonModificationStack2D::get_execution_mode);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "strength", PROPERTY_HINT_RANGE, "0, 1, 0.001"), "set_strength", "get_strength");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "execution_mode", PROPERTY_HINT_ENUM, "process, physics_process"), "set_execution_mode", "get_execution_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "modification_count", PROPERTY_HINT_RANGE, "0, 100, 1"), "set_modification_count", "get_modification_count");
}

SkeletonModificationStack2D::SkeletonModificationStack2D() {
	skeleton = nullptr;
	modifications = Vector<Ref<SkeletonModification2D>>();
	is_setup = false;
	enabled = false;
	modifications_count = 0;
	strength = 1;
}

///////////////////////////////////////
// Modification2D
///////////////////////////////////////

void SkeletonModification2D::execute(float delta) {
	if (!enabled)
		return;
}

void SkeletonModification2D::setup_modification(SkeletonModificationStack2D *p_stack) {
	stack = p_stack;
	if (stack) {
		is_setup = true;
	}
}

void SkeletonModification2D::set_enabled(bool p_enabled) {
	enabled = p_enabled;
}

bool SkeletonModification2D::get_enabled() {
	return enabled;
}

float SkeletonModification2D::clamp_angle(float angle, float min_bound, float max_bound, bool invert) {
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

void SkeletonModification2D::_bind_methods() {
	BIND_VMETHOD(MethodInfo("execute"));
	BIND_VMETHOD(MethodInfo("setup_modification"));

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SkeletonModification2D::set_enabled);
	ClassDB::bind_method(D_METHOD("get_enabled"), &SkeletonModification2D::get_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");
}

SkeletonModification2D::SkeletonModification2D() {
	stack = nullptr;
	is_setup = false;
}

///////////////////////////////////////
// LookAt
///////////////////////////////////////

bool SkeletonModification2DLookAt::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("enable_constraint")) {
		set_enable_constraint(p_value);
	} else if (path.begins_with("constraint_angle_min")) {
		set_constraint_angle_min(Math::deg2rad(float(p_value)));
	} else if (path.begins_with("constraint_angle_max")) {
		set_constraint_angle_max(Math::deg2rad(float(p_value)));
	} else if (path.begins_with("constraint_angle_invert")) {
		set_constraint_angle_invert(p_value);
	} else if (path.begins_with("constraint_in_localspace")) {
		set_constraint_in_localspace(p_value);
	} else if (path.begins_with("additional_rotation")) {
		set_additional_rotation(Math::deg2rad(float(p_value)));
	}
	return true;
}

bool SkeletonModification2DLookAt::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("enable_constraint")) {
		r_ret = get_enable_constraint();
	} else if (path.begins_with("constraint_angle_min")) {
		r_ret = Math::rad2deg(get_constraint_angle_min());
	} else if (path.begins_with("constraint_angle_max")) {
		r_ret = Math::rad2deg(get_constraint_angle_max());
	} else if (path.begins_with("constraint_angle_invert")) {
		r_ret = get_constraint_angle_invert();
	} else if (path.begins_with("constraint_in_localspace")) {
		r_ret = get_constraint_in_localspace();
	} else if (path.begins_with("additional_rotation")) {
		r_ret = Math::rad2deg(get_additional_rotation());
	}
	return true;
}

void SkeletonModification2DLookAt::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::BOOL, "enable_constraint", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	if (enable_constraint) {
		p_list->push_back(PropertyInfo(Variant::FLOAT, "constraint_angle_min", PROPERTY_HINT_RANGE, "-360, 360, 0.01", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "constraint_angle_max", PROPERTY_HINT_RANGE, "-360, 360, 0.01", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::BOOL, "constraint_angle_invert", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::BOOL, "constraint_in_localspace", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	}
	p_list->push_back(PropertyInfo(Variant::FLOAT, "additional_rotation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
}

void SkeletonModification2DLookAt::execute(float delta) {
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

	if (bone2d_node_cache.is_null() && !bone2d_node.is_empty()) {
		update_bone2d_cache();
		WARN_PRINT("Bone2D node cache is out of date. Updating...");
	}

	Node2D *target = Object::cast_to<Node2D>(ObjectDB::get_instance(target_node_cache));
	ERR_FAIL_COND_MSG(!target, "Target node is not a Node2D-based node. Cannot execute modification!");
	ERR_FAIL_COND_MSG(!target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!");
	ERR_FAIL_COND_MSG(bone_idx <= -1, "Bone index is invalid. Cannot execute modification!");

	Bone2D *operation_bone = stack->skeleton->get_bone(bone_idx);
	ERR_FAIL_COND_MSG(operation_bone == nullptr, "bone_idx for modification does not point to a valid bone! Cannot execute modification");
	Transform2D operation_transform = operation_bone->get_global_transform();
	Transform2D target_trans = target->get_global_transform();

	// Look at the target!
	operation_transform = operation_transform.looking_at(target_trans.get_origin());
	// Apply whatever scale it had prior to looking_at
	operation_transform.set_scale(operation_bone->get_global_transform().get_scale());

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

	// Set the local pose override, and to make sure child bones are also updated, set the transform of the bone.
	stack->skeleton->set_bone_local_pose_override(bone_idx, operation_transform, stack->strength, true);
	operation_bone->set_transform(operation_transform);
}

void SkeletonModification2DLookAt::setup_modification(SkeletonModificationStack2D *p_stack) {
	stack = p_stack;

	if (stack != nullptr) {
		is_setup = true;
		update_target_cache();
		update_bone2d_cache();
	}
}

void SkeletonModification2DLookAt::update_bone2d_cache() {
	if (!is_setup || !stack) {
		WARN_PRINT("Cannot update Bone2D cache: modification is not properly setup!");
		return;
	}

	bone2d_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(bone2d_node)) {
				Node *node = stack->skeleton->get_node(bone2d_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update Bone2D cache: node is this modification's skeleton or cannot be found!");
				bone2d_node_cache = node->get_instance_id();

				Bone2D *bone = Object::cast_to<Bone2D>(node);
				if (bone) {
					bone_idx = bone->get_index_in_skeleton();
				} else {
					ERR_FAIL_MSG("Error Bone2D cache: Nodepath to Bone2D is not a Bone2D node!");
				}
			}
		}
	}
}

void SkeletonModification2DLookAt::set_bone2d_node(const NodePath &p_target_node) {
	bone2d_node = p_target_node;
	update_bone2d_cache();
}

NodePath SkeletonModification2DLookAt::get_bone2d_node() const {
	return bone2d_node;
}

int SkeletonModification2DLookAt::get_bone_index() const {
	return bone_idx;
}

void SkeletonModification2DLookAt::set_bone_index(int p_bone_idx) {
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");

	if (is_setup) {
		if (stack->skeleton) {
			ERR_FAIL_INDEX_MSG(p_bone_idx, stack->skeleton->get_bone_count(), "Passed-in Bone index is out of range!");
			bone_idx = p_bone_idx;
			bone2d_node_cache = stack->skeleton->get_bone(p_bone_idx)->get_instance_id();
			bone2d_node = stack->skeleton->get_path_to(stack->skeleton->get_bone(p_bone_idx));
		} else {
			WARN_PRINT("Cannot verify the bone index for this modification...");
			bone_idx = p_bone_idx;
		}
	} else {
		WARN_PRINT("Cannot verify the bone index for this modification...");
		bone_idx = p_bone_idx;
	}

	_change_notify();
}

void SkeletonModification2DLookAt::update_target_cache() {
	if (!is_setup || !stack) {
		WARN_PRINT("Cannot update target cache: modification is not properly setup!");
		return;
	}

	target_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(target_node)) {
				Node *node = stack->skeleton->get_node(target_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				target_node_cache = node->get_instance_id();
			}
		}
	}
}

void SkeletonModification2DLookAt::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_target_cache();
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
	_change_notify();
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
	ClassDB::bind_method(D_METHOD("set_bone2d_node", "bone2d_nodepath"), &SkeletonModification2DLookAt::set_bone2d_node);
	ClassDB::bind_method(D_METHOD("get_bone2d_node"), &SkeletonModification2DLookAt::get_bone2d_node);
	ClassDB::bind_method(D_METHOD("set_bone_index", "bone_idx"), &SkeletonModification2DLookAt::set_bone_index);
	ClassDB::bind_method(D_METHOD("get_bone_index"), &SkeletonModification2DLookAt::get_bone_index);

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

	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone_index"), "set_bone_index", "get_bone_index");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "bone2d_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Bone2D"), "set_bone2d_node", "get_bone2d_node");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_target_node", "get_target_node");

	// TODO: make the additional_rotation, angle_min, and angle_max properties to use degrees in the editor!
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "additional_rotation"), "set_additional_rotation", "get_additional_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_constraint"), "set_enable_constraint", "get_enable_constraint");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "constraint_angle_min"), "set_constraint_angle_min", "get_constraint_angle_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "constraint_angle_max"), "set_constraint_angle_max", "get_constraint_angle_max");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "constraint_angle_invert"), "set_constraint_angle_invert", "get_constraint_angle_invert");
}

SkeletonModification2DLookAt::SkeletonModification2DLookAt() {
	stack = nullptr;
	is_setup = false;
	bone_idx = -1;
	additional_rotation = 0;
	enable_constraint = false;
	constraint_angle_min = 0;
	constraint_angle_max = Math_PI * 2;
	constraint_angle_invert = false;
	enabled = true;
}

SkeletonModification2DLookAt::~SkeletonModification2DLookAt() {
}

///////////////////////////////////////
// CCDIK
///////////////////////////////////////

bool SkeletonModification2DCCDIK::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, ccdik_data_chain.size(), false);

		if (what == "bone2d_node") {
			ccdik_joint_set_bone2d_node(which, p_value);
		} else if (what == "bone_index") {
			ccdik_joint_set_bone_index(which, p_value);
		} else if (what == "rotate_from_joint") {
			ccdik_joint_set_rotate_from_joint(which, p_value);
		} else if (what == "enable_constraint") {
			ccdik_joint_set_enable_constraint(which, p_value);
		} else if (what == "constraint_angle_min") {
			ccdik_joint_set_constraint_angle_min(which, Math::deg2rad(float(p_value)));
		} else if (what == "constraint_angle_max") {
			ccdik_joint_set_constraint_angle_max(which, Math::deg2rad(float(p_value)));
		} else if (what == "constraint_angle_invert") {
			ccdik_joint_set_constraint_angle_invert(which, p_value);
		} else if (what == "constraint_in_localspace") {
			ccdik_joint_set_constraint_in_localspace(which, p_value);
		}
		return true;
	}
	return true;
}

bool SkeletonModification2DCCDIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, ccdik_data_chain.size(), false);

		if (what == "bone2d_node") {
			r_ret = ccdik_joint_get_bone2d_node(which);
		} else if (what == "bone_index") {
			r_ret = ccdik_joint_get_bone_index(which);
		} else if (what == "rotate_from_joint") {
			r_ret = ccdik_joint_get_rotate_from_joint(which);
		} else if (what == "enable_constraint") {
			r_ret = ccdik_joint_get_enable_constraint(which);
		} else if (what == "constraint_angle_min") {
			r_ret = Math::rad2deg(ccdik_joint_get_constraint_angle_min(which));
		} else if (what == "constraint_angle_max") {
			r_ret = Math::rad2deg(ccdik_joint_get_constraint_angle_max(which));
		} else if (what == "constraint_angle_invert") {
			r_ret = ccdik_joint_get_constraint_angle_invert(which);
		} else if (what == "constraint_in_localspace") {
			r_ret = ccdik_joint_get_constraint_in_localspace(which);
		}
		return true;
	}
	return true;
}

void SkeletonModification2DCCDIK::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < ccdik_data_chain.size(); i++) {
		String base_string = "joint_data/" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::INT, base_string + "bone_index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, base_string + "bone2d_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Bone2D", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "rotate_from_joint", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));

		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "enable_constraint", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		if (ccdik_data_chain[i].enable_constraint) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "constraint_angle_min", PROPERTY_HINT_RANGE, "-360, 360, 0.01", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "constraint_angle_max", PROPERTY_HINT_RANGE, "-360, 360, 0.01", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "constraint_angle_invert", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "constraint_in_localspace", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
	}
}

void SkeletonModification2DCCDIK::execute(float delta) {
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

	Node2D *target = Object::cast_to<Node2D>(ObjectDB::get_instance(target_node_cache));
	ERR_FAIL_COND_MSG(!target, "Target node is not a Node2D-based node. Cannot execute modification!");
	ERR_FAIL_COND_MSG(!target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!");

	Node2D *tip = Object::cast_to<Node2D>(ObjectDB::get_instance(tip_node_cache));
	ERR_FAIL_COND_MSG(!tip, "Tip node is not a Node2D-based node. Cannot execute modification!");
	ERR_FAIL_COND_MSG(!tip->is_inside_tree(), "Tip node is not in the scene tree. Cannot execute modification!");

	for (int i = 0; i < ccdik_data_chain.size(); i++) {
		_execute_ccdik_joint(i, target, tip);
	}
}

void SkeletonModification2DCCDIK::_execute_ccdik_joint(int p_joint_idx, Node2D *target, Node2D *tip) {
	CCDIK_Joint_Data2D ccdik_data = ccdik_data_chain[p_joint_idx];
	ERR_FAIL_INDEX_MSG(ccdik_data.bone_idx, stack->skeleton->get_bone_count(), "2D CCDIK joint: bone index not found!");

	Bone2D *operation_bone = stack->skeleton->get_bone(ccdik_data.bone_idx);
	Transform2D operation_transform = operation_bone->get_global_transform();

	if (ccdik_data.rotate_from_joint) {
		// To rotate from the joint, simply look at the target!
		operation_transform.set_rotation(
				operation_transform.looking_at(target->get_global_transform().get_origin()).get_rotation() - operation_bone->get_bone_angle());
	} else {
		// How to rotate from the tip: get the difference of rotation needed from the tip to the target, from the perspective of the joint.
		// Because we are only using the offset, we do not need to account for the bone angle of the Bone2D node.
		float joint_to_tip = operation_transform.get_origin().angle_to_point(tip->get_global_transform().get_origin());
		float joint_to_target = operation_transform.get_origin().angle_to_point(target->get_global_transform().get_origin());
		operation_transform.set_rotation(
				operation_transform.get_rotation() + (joint_to_target - joint_to_tip));
	}

	// Reset scale
	operation_transform.set_scale(operation_bone->get_global_transform().get_scale());

	// Apply constraints in globalspace:
	if (ccdik_data.enable_constraint && !ccdik_data.constraint_in_localspace) {
		operation_transform.set_rotation(clamp_angle(operation_transform.get_rotation(), ccdik_data.constraint_angle_min, ccdik_data.constraint_angle_max, ccdik_data.constraint_angle_invert));
	}

	// Convert from a global transform to a delta and then apply the delta to the local transform.
	operation_bone->set_global_transform(operation_transform);
	operation_transform = operation_bone->get_transform();

	// Apply constraints in localspace:
	if (ccdik_data.enable_constraint && ccdik_data.constraint_in_localspace) {
		operation_transform.set_rotation(clamp_angle(operation_transform.get_rotation(), ccdik_data.constraint_angle_min, ccdik_data.constraint_angle_max, ccdik_data.constraint_angle_invert));
	}

	// Set the local pose override, and to make sure child bones are also updated, set the transform of the bone.
	stack->skeleton->set_bone_local_pose_override(ccdik_data.bone_idx, operation_transform, stack->strength, true);
	operation_bone->set_transform(operation_transform);
	operation_bone->notification(operation_bone->NOTIFICATION_TRANSFORM_CHANGED);
}

void SkeletonModification2DCCDIK::setup_modification(SkeletonModificationStack2D *p_stack) {
	stack = p_stack;

	if (stack != nullptr) {
		is_setup = true;
		update_target_cache();
		update_tip_cache();
	}
}

void SkeletonModification2DCCDIK::update_target_cache() {
	if (!is_setup || !stack) {
		WARN_PRINT("Cannot update target cache: modification is not properly setup!");
		return;
	}

	target_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(target_node)) {
				Node *node = stack->skeleton->get_node(target_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				target_node_cache = node->get_instance_id();
			}
		}
	}
}

void SkeletonModification2DCCDIK::update_tip_cache() {
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
						"Cannot update tip cache: node is this modification's skeleton or cannot be found!");
				tip_node_cache = node->get_instance_id();
			}
		}
	}
}

void SkeletonModification2DCCDIK::ccdik_joint_update_bone2d_cache(int p_joint_idx) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "Cannot update bone2d cache: joint index out of range!");
	if (!is_setup || !stack) {
		WARN_PRINT("Cannot update CCDIK Bone2D cache: modification is not properly setup!");
		return;
	}

	ccdik_data_chain.write[p_joint_idx].bone2d_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(ccdik_data_chain[p_joint_idx].bone2d_node)) {
				Node *node = stack->skeleton->get_node(ccdik_data_chain[p_joint_idx].bone2d_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update CCDIK Bone2D cache: node is this modification's skeleton or cannot be found!");
				ccdik_data_chain.write[p_joint_idx].bone2d_node_cache = node->get_instance_id();

				Bone2D *bone = Object::cast_to<Bone2D>(node);
				if (bone) {
					ccdik_data_chain.write[p_joint_idx].bone_idx = bone->get_index_in_skeleton();
				} else {
					ERR_FAIL_MSG("CCDIK Bone2D cache: Nodepath to Bone2D is not a Bone2D node!");
				}
			}
		}
	}
}

void SkeletonModification2DCCDIK::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_target_cache();
}

NodePath SkeletonModification2DCCDIK::get_target_node() const {
	return target_node;
}

void SkeletonModification2DCCDIK::set_tip_node(const NodePath &p_tip_node) {
	tip_node = p_tip_node;
	update_tip_cache();
}

NodePath SkeletonModification2DCCDIK::get_tip_node() const {
	return tip_node;
}

void SkeletonModification2DCCDIK::set_ccdik_data_chain_length(int p_length) {
	ccdik_data_chain.resize(p_length);
	_change_notify();
}

int SkeletonModification2DCCDIK::get_ccdik_data_chain_length() {
	return ccdik_data_chain.size();
}

void SkeletonModification2DCCDIK::ccdik_joint_set_bone2d_node(int p_joint_idx, const NodePath &p_target_node) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].bone2d_node = p_target_node;
	ccdik_joint_update_bone2d_cache(p_joint_idx);

	_change_notify();
}

NodePath SkeletonModification2DCCDIK::ccdik_joint_get_bone2d_node(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), NodePath(), "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].bone2d_node;
}

void SkeletonModification2DCCDIK::ccdik_joint_set_bone_index(int p_joint_idx, int p_bone_idx) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCCDIK joint out of range!");
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");

	if (is_setup) {
		if (stack->skeleton) {
			ERR_FAIL_INDEX_MSG(p_bone_idx, stack->skeleton->get_bone_count(), "Passed-in Bone index is out of range!");
			ccdik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
			ccdik_data_chain.write[p_joint_idx].bone2d_node_cache = stack->skeleton->get_bone(p_bone_idx)->get_instance_id();
			ccdik_data_chain.write[p_joint_idx].bone2d_node = stack->skeleton->get_path_to(stack->skeleton->get_bone(p_bone_idx));
		} else {
			WARN_PRINT("Cannot verify the CCDIK joint bone index for this modification...");
			ccdik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
		}
	} else {
		WARN_PRINT("Cannot verify the CCDIK joint bone index for this modification...");
		ccdik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
	}

	_change_notify();
}

int SkeletonModification2DCCDIK::ccdik_joint_get_bone_index(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), -1, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].bone_idx;
}

void SkeletonModification2DCCDIK::ccdik_joint_set_rotate_from_joint(int p_joint_idx, bool p_rotate_from_joint) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].rotate_from_joint = p_rotate_from_joint;
}

bool SkeletonModification2DCCDIK::ccdik_joint_get_rotate_from_joint(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].rotate_from_joint;
}

void SkeletonModification2DCCDIK::ccdik_joint_set_enable_constraint(int p_joint_idx, bool p_constraint) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].enable_constraint = p_constraint;
	_change_notify();
}

bool SkeletonModification2DCCDIK::ccdik_joint_get_enable_constraint(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].enable_constraint;
}

void SkeletonModification2DCCDIK::ccdik_joint_set_constraint_angle_min(int p_joint_idx, float p_angle_min) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].constraint_angle_min = p_angle_min;
}

float SkeletonModification2DCCDIK::ccdik_joint_get_constraint_angle_min(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), 0.0, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].constraint_angle_min;
}

void SkeletonModification2DCCDIK::ccdik_joint_set_constraint_angle_max(int p_joint_idx, float p_angle_max) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].constraint_angle_max = p_angle_max;
}

float SkeletonModification2DCCDIK::ccdik_joint_get_constraint_angle_max(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), 0.0, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].constraint_angle_max;
}

void SkeletonModification2DCCDIK::ccdik_joint_set_constraint_angle_invert(int p_joint_idx, bool p_invert) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].constraint_angle_invert = p_invert;
}

bool SkeletonModification2DCCDIK::ccdik_joint_get_constraint_angle_invert(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].constraint_angle_invert;
}

void SkeletonModification2DCCDIK::ccdik_joint_set_constraint_in_localspace(int p_joint_idx, bool p_constraint_in_localspace) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].constraint_in_localspace = p_constraint_in_localspace;
}

bool SkeletonModification2DCCDIK::ccdik_joint_get_constraint_in_localspace(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].constraint_in_localspace;
}

void SkeletonModification2DCCDIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification2DCCDIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification2DCCDIK::get_target_node);
	ClassDB::bind_method(D_METHOD("set_tip_node", "tip_nodepath"), &SkeletonModification2DCCDIK::set_tip_node);
	ClassDB::bind_method(D_METHOD("get_tip_node"), &SkeletonModification2DCCDIK::get_tip_node);

	ClassDB::bind_method(D_METHOD("set_ccdik_data_chain_length", "length"), &SkeletonModification2DCCDIK::set_ccdik_data_chain_length);
	ClassDB::bind_method(D_METHOD("get_ccdik_data_chain_length"), &SkeletonModification2DCCDIK::get_ccdik_data_chain_length);

	ClassDB::bind_method(D_METHOD("ccdik_joint_set_bone2d_node", "joint_idx", "bone2d_nodepath"), &SkeletonModification2DCCDIK::ccdik_joint_set_bone2d_node);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_bone2d_node", "joint_idx"), &SkeletonModification2DCCDIK::ccdik_joint_get_bone2d_node);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_bone_index", "joint_idx", "bone_idx"), &SkeletonModification2DCCDIK::ccdik_joint_set_bone_index);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_bone_index", "joint_idx"), &SkeletonModification2DCCDIK::ccdik_joint_get_bone_index);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_rotate_from_joint", "joint_idx", "rotate_from_joint"), &SkeletonModification2DCCDIK::ccdik_joint_set_rotate_from_joint);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_rotate_from_joint", "joint_idx"), &SkeletonModification2DCCDIK::ccdik_joint_get_rotate_from_joint);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_enable_constraint", "joint_idx", "enable_constraint"), &SkeletonModification2DCCDIK::ccdik_joint_set_enable_constraint);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_enable_constraint", "joint_idx"), &SkeletonModification2DCCDIK::ccdik_joint_get_enable_constraint);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_constraint_angle_min", "joint_idx", "angle_min"), &SkeletonModification2DCCDIK::ccdik_joint_set_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_constraint_angle_min", "joint_idx"), &SkeletonModification2DCCDIK::ccdik_joint_get_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_constraint_angle_max", "joint_idx", "angle_max"), &SkeletonModification2DCCDIK::ccdik_joint_set_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_constraint_angle_max", "joint_idx"), &SkeletonModification2DCCDIK::ccdik_joint_get_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("ccdik_joint_set_constraint_angle_invert", "joint_idx", "invert"), &SkeletonModification2DCCDIK::ccdik_joint_set_constraint_angle_invert);
	ClassDB::bind_method(D_METHOD("ccdik_joint_get_constraint_angle_invert", "joint_idx"), &SkeletonModification2DCCDIK::ccdik_joint_get_constraint_angle_invert);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "tip_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_tip_node", "get_tip_node");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ccdik_data_chain_length", PROPERTY_HINT_RANGE, "0, 100, 1"), "set_ccdik_data_chain_length", "get_ccdik_data_chain_length");
}

SkeletonModification2DCCDIK::SkeletonModification2DCCDIK() {
	stack = nullptr;
	is_setup = false;
	enabled = true;
}

SkeletonModification2DCCDIK::~SkeletonModification2DCCDIK() {
}

///////////////////////////////////////
// FABRIK
///////////////////////////////////////

bool SkeletonModification2DFABRIK::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, fabrik_data_chain.size(), false);

		if (what == "bone2d_node") {
			fabrik_joint_set_bone2d_node(which, p_value);
		} else if (what == "bone_index") {
			fabrik_joint_set_bone_index(which, p_value);
		} else if (what == "magnet_position") {
			fabrik_joint_set_magnet_position(which, p_value);
		} else if (what == "use_target_rotation") {
			fabrik_joint_set_use_target_rotation(which, p_value);
		} else if (what == "enable_constraint") {
			fabrik_joint_set_enable_constraint(which, p_value);
		} else if (what == "constraint_angle_min") {
			fabrik_joint_set_constraint_angle_min(which, Math::deg2rad(float(p_value)));
		} else if (what == "constraint_angle_max") {
			fabrik_joint_set_constraint_angle_max(which, Math::deg2rad(float(p_value)));
		} else if (what == "constraint_angle_invert") {
			fabrik_joint_set_constraint_angle_invert(which, p_value);
		} else if (what == "constraint_in_localspace") {
			fabrik_joint_set_constraint_in_localspace(which, p_value);
		}
		return true;
	}
	return true;
}

bool SkeletonModification2DFABRIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, fabrik_data_chain.size(), false);

		if (what == "bone2d_node") {
			r_ret = fabrik_joint_get_bone2d_node(which);
		} else if (what == "bone_index") {
			r_ret = fabrik_joint_get_bone_index(which);
		} else if (what == "magnet_position") {
			r_ret = fabrik_joint_get_magnet_position(which);
		} else if (what == "use_target_rotation") {
			r_ret = fabrik_joint_get_use_target_rotation(which);
		} else if (what == "enable_constraint") {
			r_ret = fabrik_joint_get_enable_constraint(which);
		} else if (what == "constraint_angle_min") {
			r_ret = Math::rad2deg(fabrik_joint_get_constraint_angle_min(which));
		} else if (what == "constraint_angle_max") {
			r_ret = Math::rad2deg(fabrik_joint_get_constraint_angle_max(which));
		} else if (what == "constraint_angle_invert") {
			r_ret = fabrik_joint_get_constraint_angle_invert(which);
		} else if (what == "constraint_in_localspace") {
			r_ret = fabrik_joint_get_constraint_in_localspace(which);
		}
		return true;
	}
	return true;
}

void SkeletonModification2DFABRIK::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		String base_string = "joint_data/" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::INT, base_string + "bone_index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, base_string + "bone2d_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Bone2D", PROPERTY_USAGE_DEFAULT));

		if (i > 0) {
			p_list->push_back(PropertyInfo(Variant::VECTOR2, base_string + "magnet_position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
		if (i == fabrik_data_chain.size() - 1) {
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "use_target_rotation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}

		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "enable_constraint", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		if (fabrik_data_chain[i].enable_constraint) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "constraint_angle_min", PROPERTY_HINT_RANGE, "-360, 360, 0.01", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "constraint_angle_max", PROPERTY_HINT_RANGE, "-360, 360, 0.01", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "constraint_angle_invert", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "constraint_in_localspace", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
	}
}

void SkeletonModification2DFABRIK::execute(float delta) {
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

	Node2D *target = Object::cast_to<Node2D>(ObjectDB::get_instance(target_node_cache));
	ERR_FAIL_COND_MSG(!target, "Target node is not a Node2D-based node. Cannot execute modification!");
	ERR_FAIL_COND_MSG(!target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!");
	ERR_FAIL_COND_MSG(fabrik_data_chain.size() <= 1, "FABRIK requires at least two nodes to opperate! Cannot execute modification!");
	target_global_pose = target->get_global_transform();

	if (fabrik_data_chain[0].bone2d_node_cache.is_null() && !fabrik_data_chain[0].bone2d_node.is_empty()) {
		fabrik_joint_update_bone2d_cache(0);
		WARN_PRINT("Bone2D cache for origin joint is out of date. Updating...");
	}

	Bone2D *origin_bone2d_node = Object::cast_to<Bone2D>(ObjectDB::get_instance(fabrik_data_chain[0].bone2d_node_cache));
	ERR_FAIL_COND_MSG(!origin_bone2d_node, "Origin joint's Bone2D node not found! Cannot execute modification!");
	origin_global_pose = origin_bone2d_node->get_global_transform();

	if (fabrik_transform_chain.size() != fabrik_data_chain.size()) {
		fabrik_transform_chain.resize(fabrik_data_chain.size());
	}

	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		// Update the transform chain
		if (fabrik_data_chain[i].bone2d_node_cache.is_null() && !fabrik_data_chain[i].bone2d_node.is_empty()) {
			fabrik_joint_update_bone2d_cache(i);
			WARN_PRINT("Bone2D cache for joint " + itos(i) + " is out of date. Updating...");
		}
		Bone2D *joint_bone2d_node = Object::cast_to<Bone2D>(ObjectDB::get_instance(fabrik_data_chain[i].bone2d_node_cache));
		ERR_FAIL_COND_MSG(!joint_bone2d_node, "Joint " + itos(i) + " does not have a Bone2D node set! Cannot execute modification!");
		fabrik_transform_chain.write[i] = joint_bone2d_node->get_global_transform();

		// Apply magnet positions
		if (i == 0) {
			continue; // The origin cannot use a magnet position!
		} else {
			Transform2D joint_trans = fabrik_transform_chain[i];
			joint_trans.set_origin(joint_trans.get_origin() + fabrik_data_chain[i].magnet_position);
			fabrik_transform_chain.write[i] = joint_trans;
		}
	}

	Bone2D *final_bone2d_node = Object::cast_to<Bone2D>(ObjectDB::get_instance(fabrik_data_chain[fabrik_data_chain.size() - 1].bone2d_node_cache));
	float final_bone2d_angle = final_bone2d_node->get_global_transform().get_rotation();
	if (fabrik_data_chain[fabrik_data_chain.size() - 1].use_target_rotation) {
		final_bone2d_angle = target_global_pose.get_rotation();
	}
	Vector2 final_bone2d_direction = Vector2(Math::cos(final_bone2d_angle), Math::sin(final_bone2d_angle));
	float final_bone2d_length = final_bone2d_node->get_length() * MIN(final_bone2d_node->get_global_scale().x, final_bone2d_node->get_global_scale().y);
	float target_distance = (final_bone2d_node->get_global_transform().get_origin() + (final_bone2d_direction * final_bone2d_length)).distance_to(target->get_global_transform().get_origin());
	chain_iterations = 0;

	while (target_distance > chain_tolarance) {
		chain_backwards();
		chain_forwards();

		final_bone2d_angle = final_bone2d_node->get_global_transform().get_rotation();
		if (fabrik_data_chain[fabrik_data_chain.size() - 1].use_target_rotation) {
			final_bone2d_angle = target_global_pose.get_rotation();
		}
		final_bone2d_direction = Vector2(Math::cos(final_bone2d_angle), Math::sin(final_bone2d_angle));
		target_distance = (final_bone2d_node->get_global_transform().get_origin() + (final_bone2d_direction * final_bone2d_length)).distance_to(target->get_global_transform().get_origin());

		chain_iterations += 1;
		if (chain_iterations >= chain_max_iterations) {
			break;
		}
	}

	// Apply all of the saved transforms to the Bone2D nodes
	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		Bone2D *joint_bone2d_node = Object::cast_to<Bone2D>(ObjectDB::get_instance(fabrik_data_chain[i].bone2d_node_cache));
		ERR_CONTINUE_MSG(!joint_bone2d_node, "Joint " + itos(i) + " does not have a Bone2D node set!");
		Transform2D chain_trans = fabrik_transform_chain[i];

		// Apply rotation
		if (i + 1 < fabrik_data_chain.size()) {
			chain_trans = chain_trans.looking_at(fabrik_transform_chain[i + 1].get_origin());
		} else {
			if (fabrik_data_chain[i].use_target_rotation) {
				chain_trans.set_rotation(target_global_pose.get_rotation());
			} else {
				chain_trans = chain_trans.looking_at(target_global_pose.get_origin());
			}

			// Account for constraints
			if (fabrik_data_chain[i].enable_constraint) {
				chain_trans.set_rotation(clamp_angle(chain_trans.get_rotation(), fabrik_data_chain[i].constraint_angle_min,
						fabrik_data_chain[i].constraint_angle_max, fabrik_data_chain[i].constraint_angle_invert));
			}
		}
		// Adjust for the bone angle
		chain_trans.set_rotation(chain_trans.get_rotation() - joint_bone2d_node->get_bone_angle());

		// Reset scale
		chain_trans.set_scale(joint_bone2d_node->get_global_transform().get_scale());

		// Apply to the bone, and to the override
		joint_bone2d_node->set_global_transform(chain_trans);
		stack->skeleton->set_bone_local_pose_override(fabrik_data_chain[i].bone_idx, joint_bone2d_node->get_transform(), stack->strength, true);
	}
}

void SkeletonModification2DFABRIK::chain_backwards() {
	int final_joint_index = fabrik_data_chain.size() - 1;
	Bone2D *final_bone2d_node = Object::cast_to<Bone2D>(ObjectDB::get_instance(fabrik_data_chain[final_joint_index].bone2d_node_cache));
	Transform2D final_bone2d_trans = fabrik_transform_chain[final_joint_index];

	// Set the rotation of the tip bone
	final_bone2d_trans = final_bone2d_trans.looking_at(target_global_pose.get_origin());

	// Set the position of the tip bone
	float final_bone2d_angle = final_bone2d_trans.get_rotation();
	if (fabrik_data_chain[final_joint_index].use_target_rotation) {
		final_bone2d_angle = target_global_pose.get_rotation();
	}

	if (fabrik_data_chain[final_joint_index].enable_constraint) {
		final_bone2d_angle = clamp_angle(final_bone2d_angle, fabrik_data_chain[final_joint_index].constraint_angle_min,
				fabrik_data_chain[final_joint_index].constraint_angle_max, fabrik_data_chain[final_joint_index].constraint_angle_invert);
	}
	Vector2 final_bone2d_direction = Vector2(Math::cos(final_bone2d_angle), Math::sin(final_bone2d_angle));
	float final_bone2d_length = final_bone2d_node->get_length() * MIN(final_bone2d_node->get_global_scale().x, final_bone2d_node->get_global_scale().y);
	final_bone2d_trans.set_origin(target_global_pose.get_origin() - (final_bone2d_direction * final_bone2d_length));

	// Save the transform
	fabrik_transform_chain.write[final_joint_index] = final_bone2d_trans;

	int i = final_joint_index;
	while (i >= 1) {
		Transform2D previous_pose = fabrik_transform_chain[i];
		i -= 1;
		Bone2D *current_bone2d_node = Object::cast_to<Bone2D>(ObjectDB::get_instance(fabrik_data_chain[i].bone2d_node_cache));
		Transform2D current_pose = fabrik_transform_chain[i];

		// Commenting out this chunk of code makes the solves look better, but then it doesn't always find a solution even when one is possible...
		if (fabrik_data_chain[i].enable_constraint) {
			float previous_to_current_angle = 0;
			float previous_to_current_length = previous_pose.get_origin().distance_to(current_pose.get_origin());

			// Calculate the current angle
			Vector2 previous_to_current_dir = previous_pose.get_origin().direction_to(current_pose.get_origin());
			previous_to_current_angle = Math::atan2(previous_to_current_dir.y, previous_to_current_dir.x);

			// Clamp the angle
			previous_to_current_angle = clamp_angle(previous_to_current_angle, fabrik_data_chain[i].constraint_angle_min, fabrik_data_chain[i].constraint_angle_max, fabrik_data_chain[i].constraint_angle_invert);

			current_pose.set_origin(previous_pose.get_origin() + (Vector2(Math::cos(previous_to_current_angle), Math::sin(previous_to_current_angle)) * previous_to_current_length));
		}

		float current_bone2d_node_length = current_bone2d_node->get_length() * MIN(current_bone2d_node->get_global_scale().x, current_bone2d_node->get_global_scale().y);
		float length = current_bone2d_node_length / (previous_pose.get_origin() - current_pose.get_origin()).length();
		Vector2 finish_position = previous_pose.get_origin().lerp(current_pose.get_origin(), length);
		current_pose.set_origin(finish_position);

		// Save the transform
		fabrik_transform_chain.write[i] = current_pose;
	}
}

void SkeletonModification2DFABRIK::chain_forwards() {
	Transform2D origin_bone2d_trans = fabrik_transform_chain[0];
	origin_bone2d_trans.set_origin(origin_global_pose.get_origin());
	// Save the position
	fabrik_transform_chain.write[0] = origin_bone2d_trans;

	for (int i = 0; i < fabrik_data_chain.size() - 1; i++) {
		Bone2D *current_bone2d_node = Object::cast_to<Bone2D>(ObjectDB::get_instance(fabrik_data_chain[i].bone2d_node_cache));
		Transform2D current_pose = fabrik_transform_chain[i];
		Transform2D next_pose = fabrik_transform_chain[i + 1];

		if (fabrik_data_chain[i].enable_constraint) {
			float next_to_current_angle = 0;
			float next_to_current_length = next_pose.get_origin().distance_to(current_pose.get_origin());

			// Calculate the current angle
			Vector2 next_to_current_dir = current_pose.get_origin().direction_to(next_pose.get_origin());
			next_to_current_angle = Math::atan2(next_to_current_dir.y, next_to_current_dir.x);

			// Clamp the angle
			next_to_current_angle = clamp_angle(next_to_current_angle, fabrik_data_chain[i].constraint_angle_min, fabrik_data_chain[i].constraint_angle_max, fabrik_data_chain[i].constraint_angle_invert);

			next_pose.set_origin(current_pose.get_origin() + (Vector2(Math::cos(next_to_current_angle), Math::sin(next_to_current_angle)) * next_to_current_length));
		}

		float current_bone2d_node_length = current_bone2d_node->get_length() * MIN(current_bone2d_node->get_global_scale().x, current_bone2d_node->get_global_scale().y);
		float length = current_bone2d_node_length / (current_pose.get_origin() - next_pose.get_origin()).length();
		Vector2 finish_position = current_pose.get_origin().lerp(next_pose.get_origin(), length);
		current_pose.set_origin(finish_position);

		// Apply to the bone
		fabrik_transform_chain.write[i + 1] = current_pose;
	}
}

void SkeletonModification2DFABRIK::setup_modification(SkeletonModificationStack2D *p_stack) {
	stack = p_stack;

	if (stack != nullptr) {
		is_setup = true;
		update_target_cache();
	}
}

void SkeletonModification2DFABRIK::update_target_cache() {
	if (!is_setup || !stack) {
		WARN_PRINT("Cannot update target cache: modification is not properly setup!");
		return;
	}

	target_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(target_node)) {
				Node *node = stack->skeleton->get_node(target_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				target_node_cache = node->get_instance_id();
			}
		}
	}
}

void SkeletonModification2DFABRIK::fabrik_joint_update_bone2d_cache(int p_joint_idx) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "Cannot update bone2d cache: joint index out of range!");
	if (!is_setup || !stack) {
		WARN_PRINT("Cannot update FABRIK Bone2D cache: modification is not properly setup!");
		return;
	}

	fabrik_data_chain.write[p_joint_idx].bone2d_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(fabrik_data_chain[p_joint_idx].bone2d_node)) {
				Node *node = stack->skeleton->get_node(fabrik_data_chain[p_joint_idx].bone2d_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update FABRIK Bone2D cache: node is this modification's skeleton or cannot be found!");
				fabrik_data_chain.write[p_joint_idx].bone2d_node_cache = node->get_instance_id();

				Bone2D *bone = Object::cast_to<Bone2D>(node);
				if (bone) {
					fabrik_data_chain.write[p_joint_idx].bone_idx = bone->get_index_in_skeleton();
				} else {
					ERR_FAIL_MSG("FABRIK Bone2D cache: Nodepath to Bone2D is not a Bone2D node!");
				}
			}
		}
	}
}

void SkeletonModification2DFABRIK::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_target_cache();
}

NodePath SkeletonModification2DFABRIK::get_target_node() const {
	return target_node;
}

void SkeletonModification2DFABRIK::set_fabrik_data_chain_length(int p_length) {
	fabrik_data_chain.resize(p_length);
	_change_notify();
}

int SkeletonModification2DFABRIK::get_fabrik_data_chain_length() {
	return fabrik_data_chain.size();
}

void SkeletonModification2DFABRIK::fabrik_joint_set_bone2d_node(int p_joint_idx, const NodePath &p_target_node) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].bone2d_node = p_target_node;
	fabrik_joint_update_bone2d_cache(p_joint_idx);

	_change_notify();
}

NodePath SkeletonModification2DFABRIK::fabrik_joint_get_bone2d_node(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), NodePath(), "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].bone2d_node;
}

void SkeletonModification2DFABRIK::fabrik_joint_set_bone_index(int p_joint_idx, int p_bone_idx) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");

	if (is_setup) {
		if (stack->skeleton) {
			ERR_FAIL_INDEX_MSG(p_bone_idx, stack->skeleton->get_bone_count(), "Passed-in Bone index is out of range!");
			fabrik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
			fabrik_data_chain.write[p_joint_idx].bone2d_node_cache = stack->skeleton->get_bone(p_bone_idx)->get_instance_id();
			fabrik_data_chain.write[p_joint_idx].bone2d_node = stack->skeleton->get_path_to(stack->skeleton->get_bone(p_bone_idx));
		} else {
			WARN_PRINT("Cannot verify the FABRIK joint bone index for this modification...");
			fabrik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
		}
	} else {
		WARN_PRINT("Cannot verify the FABRIK joint bone index for this modification...");
		fabrik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
	}

	_change_notify();
}

int SkeletonModification2DFABRIK::fabrik_joint_get_bone_index(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), -1, "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].bone_idx;
}

void SkeletonModification2DFABRIK::fabrik_joint_set_magnet_position(int p_joint_idx, Vector2 p_magnet_position) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].magnet_position = p_magnet_position;
}

Vector2 SkeletonModification2DFABRIK::fabrik_joint_get_magnet_position(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), Vector2(), "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].magnet_position;
}

void SkeletonModification2DFABRIK::fabrik_joint_set_use_target_rotation(int p_joint_idx, bool p_use_target_rotation) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].use_target_rotation = p_use_target_rotation;
}

bool SkeletonModification2DFABRIK::fabrik_joint_get_use_target_rotation(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), false, "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].use_target_rotation;
}

void SkeletonModification2DFABRIK::fabrik_joint_set_enable_constraint(int p_joint_idx, bool p_constraint) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].enable_constraint = p_constraint;
	_change_notify();
}

bool SkeletonModification2DFABRIK::fabrik_joint_get_enable_constraint(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), false, "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].enable_constraint;
}

void SkeletonModification2DFABRIK::fabrik_joint_set_constraint_angle_min(int p_joint_idx, float p_angle_min) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].constraint_angle_min = p_angle_min;
}

float SkeletonModification2DFABRIK::fabrik_joint_get_constraint_angle_min(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), 0.0, "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].constraint_angle_min;
}

void SkeletonModification2DFABRIK::fabrik_joint_set_constraint_angle_max(int p_joint_idx, float p_angle_max) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].constraint_angle_max = p_angle_max;
}

float SkeletonModification2DFABRIK::fabrik_joint_get_constraint_angle_max(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), 0.0, "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].constraint_angle_max;
}

void SkeletonModification2DFABRIK::fabrik_joint_set_constraint_angle_invert(int p_joint_idx, bool p_invert) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].constraint_angle_invert = p_invert;
}

bool SkeletonModification2DFABRIK::fabrik_joint_get_constraint_angle_invert(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), false, "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].constraint_angle_invert;
}

void SkeletonModification2DFABRIK::fabrik_joint_set_constraint_in_localspace(int p_joint_idx, bool p_constraint_in_localspace) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].constraint_in_localspace = p_constraint_in_localspace;
}

bool SkeletonModification2DFABRIK::fabrik_joint_get_constraint_in_localspace(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), false, "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].constraint_in_localspace;
}

void SkeletonModification2DFABRIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification2DFABRIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification2DFABRIK::get_target_node);

	ClassDB::bind_method(D_METHOD("set_fabrik_data_chain_length", "length"), &SkeletonModification2DFABRIK::set_fabrik_data_chain_length);
	ClassDB::bind_method(D_METHOD("get_fabrik_data_chain_length"), &SkeletonModification2DFABRIK::get_fabrik_data_chain_length);

	ClassDB::bind_method(D_METHOD("fabrik_joint_set_bone2d_node", "joint_idx", "bone2d_nodepath"), &SkeletonModification2DFABRIK::fabrik_joint_set_bone2d_node);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_bone2d_node", "joint_idx"), &SkeletonModification2DFABRIK::fabrik_joint_get_bone2d_node);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_bone_index", "joint_idx", "bone_idx"), &SkeletonModification2DFABRIK::fabrik_joint_set_bone_index);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_bone_index", "joint_idx"), &SkeletonModification2DFABRIK::fabrik_joint_get_bone_index);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_magnet_position", "joint_idx", "magnet_position"), &SkeletonModification2DFABRIK::fabrik_joint_set_magnet_position);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_magnet_position", "joint_idx"), &SkeletonModification2DFABRIK::fabrik_joint_get_magnet_position);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_use_target_rotation", "joint_idx", "use_target_rotation"), &SkeletonModification2DFABRIK::fabrik_joint_set_use_target_rotation);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_use_target_rotation", "joint_idx"), &SkeletonModification2DFABRIK::fabrik_joint_get_use_target_rotation);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_enable_constraint", "joint_idx", "enable_constraint"), &SkeletonModification2DFABRIK::fabrik_joint_set_enable_constraint);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_enable_constraint", "joint_idx"), &SkeletonModification2DFABRIK::fabrik_joint_get_enable_constraint);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_constraint_angle_min", "joint_idx", "angle_min"), &SkeletonModification2DFABRIK::fabrik_joint_set_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_constraint_angle_min", "joint_idx"), &SkeletonModification2DFABRIK::fabrik_joint_get_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_constraint_angle_max", "joint_idx", "angle_max"), &SkeletonModification2DFABRIK::fabrik_joint_set_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_constraint_angle_max", "joint_idx"), &SkeletonModification2DFABRIK::fabrik_joint_get_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("fabrik_joint_set_constraint_angle_invert", "joint_idx", "invert"), &SkeletonModification2DFABRIK::fabrik_joint_set_constraint_angle_invert);
	ClassDB::bind_method(D_METHOD("fabrik_joint_get_constraint_angle_invert", "joint_idx"), &SkeletonModification2DFABRIK::fabrik_joint_get_constraint_angle_invert);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fabrik_data_chain_length", PROPERTY_HINT_RANGE, "0, 100, 1"), "set_fabrik_data_chain_length", "get_fabrik_data_chain_length");
}

SkeletonModification2DFABRIK::SkeletonModification2DFABRIK() {
	stack = nullptr;
	is_setup = false;
	enabled = true;
}

SkeletonModification2DFABRIK::~SkeletonModification2DFABRIK() {
}
