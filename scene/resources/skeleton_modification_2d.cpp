/*************************************************************************/
/*  skeleton_modification_2d.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "scene/2d/collision_object_2d.h"
#include "scene/2d/collision_shape_2d.h"
#include "scene/2d/physical_bone_2d.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif // TOOLS_ENABLED

///////////////////////////////////////
// ModificationStack2D
///////////////////////////////////////

void SkeletonModificationStack2D::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < modifications.size(); i++) {
		p_list->push_back(
				PropertyInfo(Variant::OBJECT, "modifications/" + itos(i),
						PROPERTY_HINT_RESOURCE_TYPE,
						"SkeletonModification2D",
						PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DEFERRED_SET_RESOURCE | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE));
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
			modifications.get(i)->_setup_modification(this);
		}

#ifdef TOOLS_ENABLED
		set_editor_gizmos_dirty(true);
#endif // TOOLS_ENABLED

	} else {
		WARN_PRINT("Cannot setup SkeletonModificationStack2D: no Skeleton2D set!");
	}
}

void SkeletonModificationStack2D::execute(float delta, int p_execution_mode) {
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

		if (modifications[i]->get_execution_mode() == p_execution_mode) {
			modifications.get(i)->_execute(delta);
		}
	}
}

void SkeletonModificationStack2D::draw_editor_gizmos() {
	if (!is_setup) {
		return;
	}

	if (editor_gizmo_dirty) {
		for (int i = 0; i < modifications.size(); i++) {
			if (!modifications[i].is_valid()) {
				continue;
			}

			if (modifications[i]->editor_draw_gizmo) {
				modifications.get(i)->_draw_editor_gizmo();
			}
		}
		skeleton->draw_set_transform(Vector2(0, 0));
		editor_gizmo_dirty = false;
	}
}

void SkeletonModificationStack2D::set_editor_gizmos_dirty(bool p_dirty) {
	if (!is_setup) {
		return;
	}

	if (!editor_gizmo_dirty && p_dirty) {
		editor_gizmo_dirty = p_dirty;
		if (skeleton) {
			skeleton->update();
		}
	} else {
		editor_gizmo_dirty = p_dirty;
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
	p_mod->_setup_modification(this);
	modifications.push_back(p_mod);

#ifdef TOOLS_ENABLED
	set_editor_gizmos_dirty(true);
#endif // TOOLS_ENABLED
}

void SkeletonModificationStack2D::delete_modification(int p_mod_idx) {
	ERR_FAIL_INDEX(p_mod_idx, modifications.size());
	modifications.remove(p_mod_idx);

#ifdef TOOLS_ENABLED
	set_editor_gizmos_dirty(true);
#endif // TOOLS_ENABLED
}

void SkeletonModificationStack2D::set_modification(int p_mod_idx, Ref<SkeletonModification2D> p_mod) {
	ERR_FAIL_INDEX(p_mod_idx, modifications.size());

	if (p_mod == nullptr) {
		modifications.insert(p_mod_idx, nullptr);
	} else {
		p_mod->_setup_modification(this);
		modifications.insert(p_mod_idx, p_mod);
	}

#ifdef TOOLS_ENABLED
	set_editor_gizmos_dirty(true);
#endif // TOOLS_ENABLED
}

void SkeletonModificationStack2D::set_modification_count(int p_count) {
	modifications.resize(p_count);
	notify_property_list_changed();

#ifdef TOOLS_ENABLED
	set_editor_gizmos_dirty(true);
#endif // TOOLS_ENABLED
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

void SkeletonModificationStack2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("setup"), &SkeletonModificationStack2D::setup);
	ClassDB::bind_method(D_METHOD("execute", "delta", "execution_mode"), &SkeletonModificationStack2D::execute);

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

	ClassDB::bind_method(D_METHOD("get_skeleton"), &SkeletonModificationStack2D::get_skeleton);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "strength", PROPERTY_HINT_RANGE, "0, 1, 0.001"), "set_strength", "get_strength");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "modification_count", PROPERTY_HINT_RANGE, "0, 100, 1"), "set_modification_count", "get_modification_count");
}

SkeletonModificationStack2D::SkeletonModificationStack2D() {
}

///////////////////////////////////////
// Modification2D
///////////////////////////////////////

void SkeletonModification2D::_execute(float delta) {
	call("_execute", delta);

	if (!enabled)
		return;
}

void SkeletonModification2D::_setup_modification(SkeletonModificationStack2D *p_stack) {
	stack = p_stack;
	if (stack) {
		is_setup = true;
	} else {
		WARN_PRINT("Could not setup modification with name " + get_name());
	}

	call("_setup_modification", p_stack);
}

void SkeletonModification2D::_draw_editor_gizmo() {
	call("_draw_editor_gizmo");
}

bool SkeletonModification2D::_print_execution_error(bool p_condition, String p_message) {
	if (!is_setup) {
		return p_condition;
	}

	if (p_condition) {
		ERR_PRINT_ONCE(p_message);
	}
	return p_condition;
}

void SkeletonModification2D::set_enabled(bool p_enabled) {
	enabled = p_enabled;

#ifdef TOOLS_ENABLED
	if (editor_draw_gizmo) {
		if (stack) {
			stack->set_editor_gizmos_dirty(true);
		}
	}
#endif // TOOLS_ENABLED
}

bool SkeletonModification2D::get_enabled() {
	return enabled;
}

float SkeletonModification2D::clamp_angle(float angle, float min_bound, float max_bound, bool invert) {
	// Map to the 0 to 360 range (in radians though) instead of the -180 to 180 range.
	if (angle < 0) {
		angle = Math_TAU + angle;
	}

	// Make min and max in the range of 0 to 360 (in radians), and make sure they are in the right order
	if (min_bound < 0) {
		min_bound = Math_TAU + min_bound;
	}
	if (max_bound < 0) {
		max_bound = Math_TAU + max_bound;
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

void SkeletonModification2D::editor_draw_angle_constraints(Bone2D *operation_bone, float min_bound, float max_bound,
		bool constraint_enabled, bool constraint_in_localspace, bool constraint_inverted) {
	if (!operation_bone) {
		return;
	}

	Color bone_ik_color = Color(1.0, 0.65, 0.0, 0.4);
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		bone_ik_color = EditorSettings::get_singleton()->get("editors/2d/bone_ik_color");
	}
#endif // TOOLS_ENABLED

	float arc_angle_min = min_bound;
	float arc_angle_max = max_bound;
	if (arc_angle_min < 0) {
		arc_angle_min = (Math_PI * 2) + arc_angle_min;
	}
	if (arc_angle_max < 0) {
		arc_angle_max = (Math_PI * 2) + arc_angle_max;
	}
	if (arc_angle_min > arc_angle_max) {
		float tmp = arc_angle_min;
		arc_angle_min = arc_angle_max;
		arc_angle_max = tmp;
	}
	arc_angle_min += operation_bone->get_bone_angle();
	arc_angle_max += operation_bone->get_bone_angle();

	if (constraint_enabled) {
		if (constraint_in_localspace) {
			Node *operation_bone_parent = operation_bone->get_parent();
			Bone2D *operation_bone_parent_bone = Object::cast_to<Bone2D>(operation_bone_parent);

			if (operation_bone_parent_bone) {
				stack->skeleton->draw_set_transform(
						stack->skeleton->get_global_transform().affine_inverse().xform(operation_bone->get_global_position()),
						operation_bone_parent_bone->get_global_rotation() - stack->skeleton->get_global_rotation());
			} else {
				stack->skeleton->draw_set_transform(stack->skeleton->get_global_transform().affine_inverse().xform(operation_bone->get_global_position()));
			}
		} else {
			stack->skeleton->draw_set_transform(stack->skeleton->get_global_transform().affine_inverse().xform(operation_bone->get_global_position()));
		}

		if (constraint_inverted) {
			stack->skeleton->draw_arc(Vector2(0, 0), operation_bone->get_length(),
					arc_angle_min + (Math_PI * 2), arc_angle_max, 32, bone_ik_color, 1.0);
		} else {
			stack->skeleton->draw_arc(Vector2(0, 0), operation_bone->get_length(),
					arc_angle_min, arc_angle_max, 32, bone_ik_color, 1.0);
		}
		stack->skeleton->draw_line(Vector2(0, 0), Vector2(Math::cos(arc_angle_min), Math::sin(arc_angle_min)) * operation_bone->get_length(), bone_ik_color, 1.0);
		stack->skeleton->draw_line(Vector2(0, 0), Vector2(Math::cos(arc_angle_max), Math::sin(arc_angle_max)) * operation_bone->get_length(), bone_ik_color, 1.0);

	} else {
		stack->skeleton->draw_set_transform(stack->skeleton->get_global_transform().affine_inverse().xform(operation_bone->get_global_position()));
		stack->skeleton->draw_arc(Vector2(0, 0), operation_bone->get_length(), 0, Math_PI * 2, 32, bone_ik_color, 1.0);
		stack->skeleton->draw_line(Vector2(0, 0), Vector2(1, 0) * operation_bone->get_length(), bone_ik_color, 1.0);
	}
}

Ref<SkeletonModificationStack2D> SkeletonModification2D::get_modification_stack() {
	return stack;
}

void SkeletonModification2D::set_is_setup(bool p_setup) {
	is_setup = p_setup;
}

bool SkeletonModification2D::get_is_setup() const {
	return is_setup;
}

void SkeletonModification2D::set_execution_mode(int p_mode) {
	execution_mode = p_mode;
}

int SkeletonModification2D::get_execution_mode() const {
	return execution_mode;
}

void SkeletonModification2D::set_editor_draw_gizmo(bool p_draw_gizmo) {
	editor_draw_gizmo = p_draw_gizmo;
#ifdef TOOLS_ENABLED
	if (is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
}

bool SkeletonModification2D::get_editor_draw_gizmo() const {
	return editor_draw_gizmo;
}

void SkeletonModification2D::_bind_methods() {
	BIND_VMETHOD(MethodInfo("_execute", PropertyInfo(Variant::FLOAT, "delta")));
	BIND_VMETHOD(MethodInfo("_setup_modification", PropertyInfo(Variant::OBJECT, "modification_stack", PROPERTY_HINT_RESOURCE_TYPE, "SkeletonModificationStack2D")));
	BIND_VMETHOD(MethodInfo("_draw_editor_gizmo"));

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SkeletonModification2D::set_enabled);
	ClassDB::bind_method(D_METHOD("get_enabled"), &SkeletonModification2D::get_enabled);
	ClassDB::bind_method(D_METHOD("get_modification_stack"), &SkeletonModification2D::get_modification_stack);
	ClassDB::bind_method(D_METHOD("set_is_setup", "is_setup"), &SkeletonModification2D::set_is_setup);
	ClassDB::bind_method(D_METHOD("get_is_setup"), &SkeletonModification2D::get_is_setup);
	ClassDB::bind_method(D_METHOD("set_execution_mode", "execution_mode"), &SkeletonModification2D::set_execution_mode);
	ClassDB::bind_method(D_METHOD("get_execution_mode"), &SkeletonModification2D::get_execution_mode);
	ClassDB::bind_method(D_METHOD("clamp_angle", "angle", "min", "max", "invert"), &SkeletonModification2D::clamp_angle);
	ClassDB::bind_method(D_METHOD("set_editor_draw_gizmo", "draw_gizmo"), &SkeletonModification2D::set_editor_draw_gizmo);
	ClassDB::bind_method(D_METHOD("get_editor_draw_gizmo"), &SkeletonModification2D::get_editor_draw_gizmo);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "execution_mode", PROPERTY_HINT_ENUM, "process, physics_process"), "set_execution_mode", "get_execution_mode");
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

#ifdef TOOLS_ENABLED
	if (path.begins_with("editor/draw_gizmo")) {
		set_editor_draw_gizmo(p_value);
	}
#endif // TOOLS_ENABLED

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

#ifdef TOOLS_ENABLED
	if (path.begins_with("editor/draw_gizmo")) {
		r_ret = get_editor_draw_gizmo();
	}
#endif // TOOLS_ENABLED

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

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "editor/draw_gizmo", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	}
#endif // TOOLS_ENABLED
}

void SkeletonModification2DLookAt::_execute(float delta) {
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

	if (bone2d_node_cache.is_null() && !bone2d_node.is_empty()) {
		update_bone2d_cache();
		_print_execution_error(true, "Bone2D node cache is out of date. Attempting to update...");
		return;
	}

	if (target_node_reference == nullptr) {
		target_node_reference = Object::cast_to<Node2D>(ObjectDB::get_instance(target_node_cache));
	}
	if (_print_execution_error(!target_node_reference || !target_node_reference->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!")) {
		return;
	}
	if (_print_execution_error(bone_idx <= -1, "Bone index is invalid. Cannot execute modification!")) {
		return;
	}

	Bone2D *operation_bone = stack->skeleton->get_bone(bone_idx);
	if (_print_execution_error(operation_bone == nullptr, "bone_idx for modification does not point to a valid bone! Cannot execute modification")) {
		return;
	}

	Transform2D operation_transform = operation_bone->get_global_transform();
	Transform2D target_trans = target_node_reference->get_global_transform();

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

void SkeletonModification2DLookAt::_setup_modification(SkeletonModificationStack2D *p_stack) {
	stack = p_stack;

	if (stack != nullptr) {
		is_setup = true;
		update_target_cache();
		update_bone2d_cache();
	}
}

void SkeletonModification2DLookAt::_draw_editor_gizmo() {
	if (!enabled || !is_setup) {
		return;
	}

	Bone2D *operation_bone = stack->skeleton->get_bone(bone_idx);
	editor_draw_angle_constraints(operation_bone, constraint_angle_min, constraint_angle_max,
			enable_constraint, constraint_in_localspace, constraint_angle_invert);
}

void SkeletonModification2DLookAt::update_bone2d_cache() {
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update Bone2D cache: modification is not properly setup!");
		return;
	}

	bone2d_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(bone2d_node)) {
				Node *node = stack->skeleton->get_node(bone2d_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update Bone2D cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update Bone2D cache: node is not in the scene tree!");
				bone2d_node_cache = node->get_instance_id();

				Bone2D *bone = Object::cast_to<Bone2D>(node);
				if (bone) {
					bone_idx = bone->get_index_in_skeleton();
				} else {
					ERR_FAIL_MSG("Error Bone2D cache: Nodepath to Bone2D is not a Bone2D node!");
				}

				// Set this to null so we update it
				target_node_reference = nullptr;
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

	notify_property_list_changed();
}

void SkeletonModification2DLookAt::update_target_cache() {
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
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: node is not in the scene tree!");
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
	notify_property_list_changed();
#ifdef TOOLS_ENABLED
	if (stack && is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
}

bool SkeletonModification2DLookAt::get_enable_constraint() const {
	return enable_constraint;
}

void SkeletonModification2DLookAt::set_constraint_angle_min(float p_angle_min) {
	constraint_angle_min = p_angle_min;
#ifdef TOOLS_ENABLED
	if (stack && is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
}

float SkeletonModification2DLookAt::get_constraint_angle_min() const {
	return constraint_angle_min;
}

void SkeletonModification2DLookAt::set_constraint_angle_max(float p_angle_max) {
	constraint_angle_max = p_angle_max;
#ifdef TOOLS_ENABLED
	if (stack && is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
}

float SkeletonModification2DLookAt::get_constraint_angle_max() const {
	return constraint_angle_max;
}

void SkeletonModification2DLookAt::set_constraint_angle_invert(bool p_invert) {
	constraint_angle_invert = p_invert;
#ifdef TOOLS_ENABLED
	if (stack && is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
}

bool SkeletonModification2DLookAt::get_constraint_angle_invert() const {
	return constraint_angle_invert;
}

void SkeletonModification2DLookAt::set_constraint_in_localspace(bool p_constraint_in_localspace) {
	constraint_in_localspace = p_constraint_in_localspace;
#ifdef TOOLS_ENABLED
	if (stack && is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
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

	editor_draw_gizmo = true;
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
			set_ccdik_joint_bone2d_node(which, p_value);
		} else if (what == "bone_index") {
			set_ccdik_joint_bone_index(which, p_value);
		} else if (what == "rotate_from_joint") {
			set_ccdik_joint_rotate_from_joint(which, p_value);
		} else if (what == "enable_constraint") {
			set_ccdik_joint_enable_constraint(which, p_value);
		} else if (what == "constraint_angle_min") {
			set_ccdik_joint_constraint_angle_min(which, Math::deg2rad(float(p_value)));
		} else if (what == "constraint_angle_max") {
			set_ccdik_joint_constraint_angle_max(which, Math::deg2rad(float(p_value)));
		} else if (what == "constraint_angle_invert") {
			set_ccdik_joint_constraint_angle_invert(which, p_value);
		} else if (what == "constraint_in_localspace") {
			set_ccdik_joint_constraint_in_localspace(which, p_value);
		}

#ifdef TOOLS_ENABLED
		if (what.begins_with("editor_draw_gizmo")) {
			set_ccdik_joint_editor_draw_gizmo(which, p_value);
		}
#endif // TOOLS_ENABLED

		return true;
	}

#ifdef TOOLS_ENABLED
	if (path.begins_with("editor/draw_gizmo")) {
		set_editor_draw_gizmo(p_value);
	}
#endif // TOOLS_ENABLED

	return true;
}

bool SkeletonModification2DCCDIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, ccdik_data_chain.size(), false);

		if (what == "bone2d_node") {
			r_ret = get_ccdik_joint_bone2d_node(which);
		} else if (what == "bone_index") {
			r_ret = get_ccdik_joint_bone_index(which);
		} else if (what == "rotate_from_joint") {
			r_ret = get_ccdik_joint_rotate_from_joint(which);
		} else if (what == "enable_constraint") {
			r_ret = get_ccdik_joint_enable_constraint(which);
		} else if (what == "constraint_angle_min") {
			r_ret = Math::rad2deg(get_ccdik_joint_constraint_angle_min(which));
		} else if (what == "constraint_angle_max") {
			r_ret = Math::rad2deg(get_ccdik_joint_constraint_angle_max(which));
		} else if (what == "constraint_angle_invert") {
			r_ret = get_ccdik_joint_constraint_angle_invert(which);
		} else if (what == "constraint_in_localspace") {
			r_ret = get_ccdik_joint_constraint_in_localspace(which);
		}

#ifdef TOOLS_ENABLED
		if (what.begins_with("editor_draw_gizmo")) {
			r_ret = get_ccdik_joint_editor_draw_gizmo(which);
		}
#endif // TOOLS_ENABLED

		return true;
	}

#ifdef TOOLS_ENABLED
	if (path.begins_with("editor/draw_gizmo")) {
		r_ret = get_editor_draw_gizmo();
	}
#endif // TOOLS_ENABLED

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

#ifdef TOOLS_ENABLED
		if (Engine::get_singleton()->is_editor_hint()) {
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "editor_draw_gizmo", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
#endif // TOOLS_ENABLED
	}

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "editor/draw_gizmo", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	}
#endif // TOOLS_ENABLED
}

void SkeletonModification2DCCDIK::_execute(float delta) {
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
	if (tip_node_cache.is_null()) {
		_print_execution_error(true, "Tip cache is out of date. Attempting to update...");
		update_tip_cache();
		return;
	}

	Node2D *target = Object::cast_to<Node2D>(ObjectDB::get_instance(target_node_cache));
	if (_print_execution_error(!target || !target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!")) {
		return;
	}

	Node2D *tip = Object::cast_to<Node2D>(ObjectDB::get_instance(tip_node_cache));
	if (_print_execution_error(!tip || !tip->is_inside_tree(), "Tip node is not in the scene tree. Cannot execute modification!")) {
		return;
	}

	for (int i = 0; i < ccdik_data_chain.size(); i++) {
		_execute_ccdik_joint(i, target, tip);
	}
}

void SkeletonModification2DCCDIK::_execute_ccdik_joint(int p_joint_idx, Node2D *target, Node2D *tip) {
	CCDIK_Joint_Data2D ccdik_data = ccdik_data_chain[p_joint_idx];
	if (_print_execution_error(ccdik_data.bone_idx < 0 || ccdik_data.bone_idx > stack->skeleton->get_bone_count(), "2D CCDIK joint: bone index not found!")) {
		return;
	}

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

void SkeletonModification2DCCDIK::_setup_modification(SkeletonModificationStack2D *p_stack) {
	stack = p_stack;

	if (stack != nullptr) {
		is_setup = true;
		update_target_cache();
		update_tip_cache();
	}
}

void SkeletonModification2DCCDIK::_draw_editor_gizmo() {
	if (!enabled || !is_setup) {
		return;
	}

	for (int i = 0; i < ccdik_data_chain.size(); i++) {
		if (!ccdik_data_chain[i].editor_draw_gizmo) {
			continue;
		}

		Bone2D *operation_bone = stack->skeleton->get_bone(ccdik_data_chain[i].bone_idx);
		editor_draw_angle_constraints(operation_bone, ccdik_data_chain[i].constraint_angle_min, ccdik_data_chain[i].constraint_angle_max,
				ccdik_data_chain[i].enable_constraint, ccdik_data_chain[i].constraint_in_localspace, ccdik_data_chain[i].constraint_angle_invert);
	}
}

void SkeletonModification2DCCDIK::update_target_cache() {
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
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: node is not in the scene tree!");
				target_node_cache = node->get_instance_id();
			}
		}
	}
}

void SkeletonModification2DCCDIK::update_tip_cache() {
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update tip cache: modification is not properly setup!");
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
						"Cannot update tip cache: node is not in the scene tree!");
				tip_node_cache = node->get_instance_id();
			}
		}
	}
}

void SkeletonModification2DCCDIK::ccdik_joint_update_bone2d_cache(int p_joint_idx) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "Cannot update bone2d cache: joint index out of range!");
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update CCDIK Bone2D cache: modification is not properly setup!");
		return;
	}

	ccdik_data_chain.write[p_joint_idx].bone2d_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(ccdik_data_chain[p_joint_idx].bone2d_node)) {
				Node *node = stack->skeleton->get_node(ccdik_data_chain[p_joint_idx].bone2d_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update CCDIK joint " + itos(p_joint_idx) + " Bone2D cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update CCDIK joint " + itos(p_joint_idx) + " Bone2D cache: node is not in the scene tree!");
				ccdik_data_chain.write[p_joint_idx].bone2d_node_cache = node->get_instance_id();

				Bone2D *bone = Object::cast_to<Bone2D>(node);
				if (bone) {
					ccdik_data_chain.write[p_joint_idx].bone_idx = bone->get_index_in_skeleton();
				} else {
					ERR_FAIL_MSG("CCDIK joint " + itos(p_joint_idx) + " Bone2D cache: Nodepath to Bone2D is not a Bone2D node!");
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
	notify_property_list_changed();
}

int SkeletonModification2DCCDIK::get_ccdik_data_chain_length() {
	return ccdik_data_chain.size();
}

void SkeletonModification2DCCDIK::set_ccdik_joint_bone2d_node(int p_joint_idx, const NodePath &p_target_node) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].bone2d_node = p_target_node;
	ccdik_joint_update_bone2d_cache(p_joint_idx);

	notify_property_list_changed();
}

NodePath SkeletonModification2DCCDIK::get_ccdik_joint_bone2d_node(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), NodePath(), "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].bone2d_node;
}

void SkeletonModification2DCCDIK::set_ccdik_joint_bone_index(int p_joint_idx, int p_bone_idx) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCCDIK joint out of range!");
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");

	if (is_setup) {
		if (stack->skeleton) {
			ERR_FAIL_INDEX_MSG(p_bone_idx, stack->skeleton->get_bone_count(), "Passed-in Bone index is out of range!");
			ccdik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
			ccdik_data_chain.write[p_joint_idx].bone2d_node_cache = stack->skeleton->get_bone(p_bone_idx)->get_instance_id();
			ccdik_data_chain.write[p_joint_idx].bone2d_node = stack->skeleton->get_path_to(stack->skeleton->get_bone(p_bone_idx));
		} else {
			WARN_PRINT("Cannot verify the CCDIK joint " + itos(p_joint_idx) + " bone index for this modification...");
			ccdik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
		}
	} else {
		WARN_PRINT("Cannot verify the CCDIK joint " + itos(p_joint_idx) + " bone index for this modification...");
		ccdik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
	}

	notify_property_list_changed();
}

int SkeletonModification2DCCDIK::get_ccdik_joint_bone_index(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), -1, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].bone_idx;
}

void SkeletonModification2DCCDIK::set_ccdik_joint_rotate_from_joint(int p_joint_idx, bool p_rotate_from_joint) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].rotate_from_joint = p_rotate_from_joint;
}

bool SkeletonModification2DCCDIK::get_ccdik_joint_rotate_from_joint(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].rotate_from_joint;
}

void SkeletonModification2DCCDIK::set_ccdik_joint_enable_constraint(int p_joint_idx, bool p_constraint) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].enable_constraint = p_constraint;
	notify_property_list_changed();

#ifdef TOOLS_ENABLED
	if (stack && is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
}

bool SkeletonModification2DCCDIK::get_ccdik_joint_enable_constraint(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].enable_constraint;
}

void SkeletonModification2DCCDIK::set_ccdik_joint_constraint_angle_min(int p_joint_idx, float p_angle_min) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].constraint_angle_min = p_angle_min;

#ifdef TOOLS_ENABLED
	if (stack && is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
}

float SkeletonModification2DCCDIK::get_ccdik_joint_constraint_angle_min(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), 0.0, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].constraint_angle_min;
}

void SkeletonModification2DCCDIK::set_ccdik_joint_constraint_angle_max(int p_joint_idx, float p_angle_max) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].constraint_angle_max = p_angle_max;

#ifdef TOOLS_ENABLED
	if (stack && is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
}

float SkeletonModification2DCCDIK::get_ccdik_joint_constraint_angle_max(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), 0.0, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].constraint_angle_max;
}

void SkeletonModification2DCCDIK::set_ccdik_joint_constraint_angle_invert(int p_joint_idx, bool p_invert) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].constraint_angle_invert = p_invert;

#ifdef TOOLS_ENABLED
	if (stack && is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
}

bool SkeletonModification2DCCDIK::get_ccdik_joint_constraint_angle_invert(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].constraint_angle_invert;
}

void SkeletonModification2DCCDIK::set_ccdik_joint_constraint_in_localspace(int p_joint_idx, bool p_constraint_in_localspace) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].constraint_in_localspace = p_constraint_in_localspace;

#ifdef TOOLS_ENABLED
	if (stack && is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
}

bool SkeletonModification2DCCDIK::get_ccdik_joint_constraint_in_localspace(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].constraint_in_localspace;
}

void SkeletonModification2DCCDIK::set_ccdik_joint_editor_draw_gizmo(int p_joint_idx, bool p_draw_gizmo) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, ccdik_data_chain.size(), "CCDIK joint out of range!");
	ccdik_data_chain.write[p_joint_idx].editor_draw_gizmo = p_draw_gizmo;

#ifdef TOOLS_ENABLED
	if (stack && is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
}

bool SkeletonModification2DCCDIK::get_ccdik_joint_editor_draw_gizmo(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, ccdik_data_chain.size(), false, "CCDIK joint out of range!");
	return ccdik_data_chain[p_joint_idx].editor_draw_gizmo;
}

void SkeletonModification2DCCDIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification2DCCDIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification2DCCDIK::get_target_node);
	ClassDB::bind_method(D_METHOD("set_tip_node", "tip_nodepath"), &SkeletonModification2DCCDIK::set_tip_node);
	ClassDB::bind_method(D_METHOD("get_tip_node"), &SkeletonModification2DCCDIK::get_tip_node);

	ClassDB::bind_method(D_METHOD("set_ccdik_data_chain_length", "length"), &SkeletonModification2DCCDIK::set_ccdik_data_chain_length);
	ClassDB::bind_method(D_METHOD("get_ccdik_data_chain_length"), &SkeletonModification2DCCDIK::get_ccdik_data_chain_length);

	ClassDB::bind_method(D_METHOD("set_ccdik_joint_bone2d_node", "joint_idx", "bone2d_nodepath"), &SkeletonModification2DCCDIK::set_ccdik_joint_bone2d_node);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_bone2d_node", "joint_idx"), &SkeletonModification2DCCDIK::get_ccdik_joint_bone2d_node);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_bone_index", "joint_idx", "bone_idx"), &SkeletonModification2DCCDIK::set_ccdik_joint_bone_index);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_bone_index", "joint_idx"), &SkeletonModification2DCCDIK::get_ccdik_joint_bone_index);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_rotate_from_joint", "joint_idx", "rotate_from_joint"), &SkeletonModification2DCCDIK::set_ccdik_joint_rotate_from_joint);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_rotate_from_joint", "joint_idx"), &SkeletonModification2DCCDIK::get_ccdik_joint_rotate_from_joint);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_enable_constraint", "joint_idx", "enable_constraint"), &SkeletonModification2DCCDIK::set_ccdik_joint_enable_constraint);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_enable_constraint", "joint_idx"), &SkeletonModification2DCCDIK::get_ccdik_joint_enable_constraint);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_constraint_angle_min", "joint_idx", "angle_min"), &SkeletonModification2DCCDIK::set_ccdik_joint_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_constraint_angle_min", "joint_idx"), &SkeletonModification2DCCDIK::get_ccdik_joint_constraint_angle_min);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_constraint_angle_max", "joint_idx", "angle_max"), &SkeletonModification2DCCDIK::set_ccdik_joint_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_constraint_angle_max", "joint_idx"), &SkeletonModification2DCCDIK::get_ccdik_joint_constraint_angle_max);
	ClassDB::bind_method(D_METHOD("set_ccdik_joint_constraint_angle_invert", "joint_idx", "invert"), &SkeletonModification2DCCDIK::set_ccdik_joint_constraint_angle_invert);
	ClassDB::bind_method(D_METHOD("get_ccdik_joint_constraint_angle_invert", "joint_idx"), &SkeletonModification2DCCDIK::get_ccdik_joint_constraint_angle_invert);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "tip_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_tip_node", "get_tip_node");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ccdik_data_chain_length", PROPERTY_HINT_RANGE, "0, 100, 1"), "set_ccdik_data_chain_length", "get_ccdik_data_chain_length");
}

SkeletonModification2DCCDIK::SkeletonModification2DCCDIK() {
	stack = nullptr;
	is_setup = false;
	enabled = true;
	editor_draw_gizmo = true;
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
			set_fabrik_joint_bone2d_node(which, p_value);
		} else if (what == "bone_index") {
			set_fabrik_joint_bone_index(which, p_value);
		} else if (what == "magnet_position") {
			set_fabrik_joint_magnet_position(which, p_value);
		} else if (what == "use_target_rotation") {
			set_fabrik_joint_use_target_rotation(which, p_value);
		}
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
			r_ret = get_fabrik_joint_bone2d_node(which);
		} else if (what == "bone_index") {
			r_ret = get_fabrik_joint_bone_index(which);
		} else if (what == "magnet_position") {
			r_ret = get_fabrik_joint_magnet_position(which);
		} else if (what == "use_target_rotation") {
			r_ret = get_fabrik_joint_use_target_rotation(which);
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
	}
}

void SkeletonModification2DFABRIK::_execute(float delta) {
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

	if (_print_execution_error(fabrik_data_chain.size() <= 1, "FABRIK requires at least two joints to operate! Cannot execute modification!")) {
		return;
	}

	Node2D *target = Object::cast_to<Node2D>(ObjectDB::get_instance(target_node_cache));
	if (_print_execution_error(!target || !target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!")) {
		return;
	}
	target_global_pose = target->get_global_transform();

	if (fabrik_data_chain[0].bone2d_node_cache.is_null() && !fabrik_data_chain[0].bone2d_node.is_empty()) {
		fabrik_joint_update_bone2d_cache(0);
		WARN_PRINT("Bone2D cache for origin joint is out of date. Updating...");
	}

	Bone2D *origin_bone2d_node = Object::cast_to<Bone2D>(ObjectDB::get_instance(fabrik_data_chain[0].bone2d_node_cache));
	if (_print_execution_error(!origin_bone2d_node || !origin_bone2d_node->is_inside_tree(), "Origin joint's Bone2D node is not in the scene tree. Cannot execute modification!")) {
		return;
	}

	origin_global_pose = origin_bone2d_node->get_global_transform();

	if (fabrik_transform_chain.size() != fabrik_data_chain.size()) {
		fabrik_transform_chain.resize(fabrik_data_chain.size());
	}

	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		// Update the transform chain
		if (fabrik_data_chain[i].bone2d_node_cache.is_null() && !fabrik_data_chain[i].bone2d_node.is_empty()) {
			_print_execution_error(true, "Bone2D cache for joint " + itos(i) + " is out of date.. Attempting to update...");
			fabrik_joint_update_bone2d_cache(i);
		}
		Bone2D *joint_bone2d_node = Object::cast_to<Bone2D>(ObjectDB::get_instance(fabrik_data_chain[i].bone2d_node_cache));
		if (_print_execution_error(!joint_bone2d_node, "FABRIK Joint " + itos(i) + " does not have a Bone2D node set! Cannot execute modification!")) {
			return;
		}
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
		if (_print_execution_error(!joint_bone2d_node, "FABRIK Joint " + itos(i) + " does not have a Bone2D node set!")) {
			continue;
		}
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

		float current_bone2d_node_length = current_bone2d_node->get_length() * MIN(current_bone2d_node->get_global_scale().x, current_bone2d_node->get_global_scale().y);
		float length = current_bone2d_node_length / (current_pose.get_origin() - next_pose.get_origin()).length();
		Vector2 finish_position = current_pose.get_origin().lerp(next_pose.get_origin(), length);
		current_pose.set_origin(finish_position);

		// Apply to the bone
		fabrik_transform_chain.write[i + 1] = current_pose;
	}
}

void SkeletonModification2DFABRIK::_setup_modification(SkeletonModificationStack2D *p_stack) {
	stack = p_stack;

	if (stack != nullptr) {
		is_setup = true;
		update_target_cache();
	}
}

void SkeletonModification2DFABRIK::update_target_cache() {
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
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: node is not in scene tree!");
				target_node_cache = node->get_instance_id();
			}
		}
	}
}

void SkeletonModification2DFABRIK::fabrik_joint_update_bone2d_cache(int p_joint_idx) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "Cannot update bone2d cache: joint index out of range!");
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update FABRIK Bone2D cache: modification is not properly setup!");
		return;
	}

	fabrik_data_chain.write[p_joint_idx].bone2d_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(fabrik_data_chain[p_joint_idx].bone2d_node)) {
				Node *node = stack->skeleton->get_node(fabrik_data_chain[p_joint_idx].bone2d_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update FABRIK joint " + itos(p_joint_idx) + " Bone2D cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update FABRIK joint " + itos(p_joint_idx) + " Bone2D cache: node is not in scene tree!");
				fabrik_data_chain.write[p_joint_idx].bone2d_node_cache = node->get_instance_id();

				Bone2D *bone = Object::cast_to<Bone2D>(node);
				if (bone) {
					fabrik_data_chain.write[p_joint_idx].bone_idx = bone->get_index_in_skeleton();
				} else {
					ERR_FAIL_MSG("FABRIK joint " + itos(p_joint_idx) + " Bone2D cache: Nodepath to Bone2D is not a Bone2D node!");
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
	notify_property_list_changed();
}

int SkeletonModification2DFABRIK::get_fabrik_data_chain_length() {
	return fabrik_data_chain.size();
}

void SkeletonModification2DFABRIK::set_fabrik_joint_bone2d_node(int p_joint_idx, const NodePath &p_target_node) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].bone2d_node = p_target_node;
	fabrik_joint_update_bone2d_cache(p_joint_idx);

	notify_property_list_changed();
}

NodePath SkeletonModification2DFABRIK::get_fabrik_joint_bone2d_node(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), NodePath(), "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].bone2d_node;
}

void SkeletonModification2DFABRIK::set_fabrik_joint_bone_index(int p_joint_idx, int p_bone_idx) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");

	if (is_setup) {
		if (stack->skeleton) {
			ERR_FAIL_INDEX_MSG(p_bone_idx, stack->skeleton->get_bone_count(), "Passed-in Bone index is out of range!");
			fabrik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
			fabrik_data_chain.write[p_joint_idx].bone2d_node_cache = stack->skeleton->get_bone(p_bone_idx)->get_instance_id();
			fabrik_data_chain.write[p_joint_idx].bone2d_node = stack->skeleton->get_path_to(stack->skeleton->get_bone(p_bone_idx));
		} else {
			WARN_PRINT("Cannot verify the FABRIK joint " + itos(p_joint_idx) + " bone index for this modification...");
			fabrik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
		}
	} else {
		WARN_PRINT("Cannot verify the FABRIK joint " + itos(p_joint_idx) + " bone index for this modification...");
		fabrik_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
	}

	notify_property_list_changed();
}

int SkeletonModification2DFABRIK::get_fabrik_joint_bone_index(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), -1, "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].bone_idx;
}

void SkeletonModification2DFABRIK::set_fabrik_joint_magnet_position(int p_joint_idx, Vector2 p_magnet_position) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].magnet_position = p_magnet_position;
}

Vector2 SkeletonModification2DFABRIK::get_fabrik_joint_magnet_position(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), Vector2(), "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].magnet_position;
}

void SkeletonModification2DFABRIK::set_fabrik_joint_use_target_rotation(int p_joint_idx, bool p_use_target_rotation) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].use_target_rotation = p_use_target_rotation;
}

bool SkeletonModification2DFABRIK::get_fabrik_joint_use_target_rotation(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), false, "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].use_target_rotation;
}

void SkeletonModification2DFABRIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification2DFABRIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification2DFABRIK::get_target_node);

	ClassDB::bind_method(D_METHOD("set_fabrik_data_chain_length", "length"), &SkeletonModification2DFABRIK::set_fabrik_data_chain_length);
	ClassDB::bind_method(D_METHOD("get_fabrik_data_chain_length"), &SkeletonModification2DFABRIK::get_fabrik_data_chain_length);

	ClassDB::bind_method(D_METHOD("set_fabrik_joint_bone2d_node", "joint_idx", "bone2d_nodepath"), &SkeletonModification2DFABRIK::set_fabrik_joint_bone2d_node);
	ClassDB::bind_method(D_METHOD("get_fabrik_joint_bone2d_node", "joint_idx"), &SkeletonModification2DFABRIK::get_fabrik_joint_bone2d_node);
	ClassDB::bind_method(D_METHOD("set_fabrik_joint_bone_index", "joint_idx", "bone_idx"), &SkeletonModification2DFABRIK::set_fabrik_joint_bone_index);
	ClassDB::bind_method(D_METHOD("get_fabrik_joint_bone_index", "joint_idx"), &SkeletonModification2DFABRIK::get_fabrik_joint_bone_index);
	ClassDB::bind_method(D_METHOD("set_fabrik_joint_magnet_position", "joint_idx", "magnet_position"), &SkeletonModification2DFABRIK::set_fabrik_joint_magnet_position);
	ClassDB::bind_method(D_METHOD("get_fabrik_joint_magnet_position", "joint_idx"), &SkeletonModification2DFABRIK::get_fabrik_joint_magnet_position);
	ClassDB::bind_method(D_METHOD("set_fabrik_joint_use_target_rotation", "joint_idx", "use_target_rotation"), &SkeletonModification2DFABRIK::set_fabrik_joint_use_target_rotation);
	ClassDB::bind_method(D_METHOD("get_fabrik_joint_use_target_rotation", "joint_idx"), &SkeletonModification2DFABRIK::get_fabrik_joint_use_target_rotation);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fabrik_data_chain_length", PROPERTY_HINT_RANGE, "0, 100, 1"), "set_fabrik_data_chain_length", "get_fabrik_data_chain_length");
}

SkeletonModification2DFABRIK::SkeletonModification2DFABRIK() {
	stack = nullptr;
	is_setup = false;
	enabled = true;
	editor_draw_gizmo = false;
}

SkeletonModification2DFABRIK::~SkeletonModification2DFABRIK() {
}

///////////////////////////////////////
// Jiggle
///////////////////////////////////////

bool SkeletonModification2DJiggle::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, jiggle_data_chain.size(), false);

		if (what == "bone2d_node") {
			set_jiggle_joint_bone2d_node(which, p_value);
		} else if (what == "bone_index") {
			set_jiggle_joint_bone_index(which, p_value);
		} else if (what == "override_defaults") {
			set_jiggle_joint_override(which, p_value);
		} else if (what == "stiffness") {
			set_jiggle_joint_stiffness(which, p_value);
		} else if (what == "mass") {
			set_jiggle_joint_mass(which, p_value);
		} else if (what == "damping") {
			set_jiggle_joint_damping(which, p_value);
		} else if (what == "use_gravity") {
			set_jiggle_joint_use_gravity(which, p_value);
		} else if (what == "gravity") {
			set_jiggle_joint_gravity(which, p_value);
		}
		return true;
	} else {
		if (path == "use_colliders") {
			set_use_colliders(p_value);
		} else if (path == "collision_mask") {
			set_collision_mask(p_value);
		}
	}
	return true;
}

bool SkeletonModification2DJiggle::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_data/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, jiggle_data_chain.size(), false);

		if (what == "bone2d_node") {
			r_ret = get_jiggle_joint_bone2d_node(which);
		} else if (what == "bone_index") {
			r_ret = get_jiggle_joint_bone_index(which);
		} else if (what == "override_defaults") {
			r_ret = get_jiggle_joint_override(which);
		} else if (what == "stiffness") {
			r_ret = get_jiggle_joint_stiffness(which);
		} else if (what == "mass") {
			r_ret = get_jiggle_joint_mass(which);
		} else if (what == "damping") {
			r_ret = get_jiggle_joint_damping(which);
		} else if (what == "use_gravity") {
			r_ret = get_jiggle_joint_use_gravity(which);
		} else if (what == "gravity") {
			r_ret = get_jiggle_joint_gravity(which);
		}
		return true;
	} else {
		if (path == "use_colliders") {
			r_ret = get_use_colliders();
		} else if (path == "collision_mask") {
			r_ret = get_collision_mask();
		}
	}
	return true;
}

void SkeletonModification2DJiggle::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::BOOL, "use_colliders", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	if (use_colliders) {
		p_list->push_back(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_2D_PHYSICS, "", PROPERTY_USAGE_DEFAULT));
	}

	for (int i = 0; i < jiggle_data_chain.size(); i++) {
		String base_string = "joint_data/" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::INT, base_string + "bone_index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, base_string + "bone2d_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Bone2D", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "override_defaults", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));

		if (jiggle_data_chain[i].override_defaults) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "stiffness", PROPERTY_HINT_RANGE, "0, 1000, 0.01", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "mass", PROPERTY_HINT_RANGE, "0, 1000, 0.01", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::FLOAT, base_string + "damping", PROPERTY_HINT_RANGE, "0, 1, 0.01", PROPERTY_USAGE_DEFAULT));
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "use_gravity", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			if (jiggle_data_chain[i].use_gravity) {
				p_list->push_back(PropertyInfo(Variant::VECTOR2, base_string + "gravity", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
			}
		}
	}
}

void SkeletonModification2DJiggle::_execute(float delta) {
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
	Node2D *target = Object::cast_to<Node2D>(ObjectDB::get_instance(target_node_cache));
	if (_print_execution_error(!target || !target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!")) {
		return;
	}

	for (int i = 0; i < jiggle_data_chain.size(); i++) {
		_execute_jiggle_joint(i, target, delta);
	}
}

void SkeletonModification2DJiggle::_execute_jiggle_joint(int p_joint_idx, Node2D *target, float delta) {
	// Adopted from: https://wiki.unity3d.com/index.php/JiggleBone
	// With modifications by TwistedTwigleg.

	if (_print_execution_error(
				jiggle_data_chain[p_joint_idx].bone_idx <= -1 || jiggle_data_chain[p_joint_idx].bone_idx > stack->skeleton->get_bone_count(),
				"Jiggle joint " + itos(p_joint_idx) + " bone index is invalid. Cannot execute modification on joint...")) {
		return;
	}

	if (jiggle_data_chain[p_joint_idx].bone2d_node_cache.is_null() && !jiggle_data_chain[p_joint_idx].bone2d_node.is_empty()) {
		_print_execution_error(true, "Bone2D cache for joint " + itos(p_joint_idx) + " is out of date. Updating...");
		jiggle_joint_update_bone2d_cache(p_joint_idx);
	}

	Bone2D *operation_bone = stack->skeleton->get_bone(jiggle_data_chain[p_joint_idx].bone_idx);
	if (_print_execution_error(!operation_bone, "Jiggle joint " + itos(p_joint_idx) + " does not have a Bone2D node or it cannot be found!")) {
		return;
	}

	Transform2D operation_bone_trans = operation_bone->get_global_transform();
	Vector2 target_position = target->get_global_transform().get_origin();

	jiggle_data_chain.write[p_joint_idx].force = (target_position - jiggle_data_chain[p_joint_idx].dynamic_position) * jiggle_data_chain[p_joint_idx].stiffness * delta;

	if (jiggle_data_chain[p_joint_idx].use_gravity) {
		jiggle_data_chain.write[p_joint_idx].force += jiggle_data_chain[p_joint_idx].gravity * delta;
	}

	jiggle_data_chain.write[p_joint_idx].acceleration = jiggle_data_chain[p_joint_idx].force / jiggle_data_chain[p_joint_idx].mass;
	jiggle_data_chain.write[p_joint_idx].velocity += jiggle_data_chain[p_joint_idx].acceleration * (1 - jiggle_data_chain[p_joint_idx].damping);

	jiggle_data_chain.write[p_joint_idx].dynamic_position += jiggle_data_chain[p_joint_idx].velocity + jiggle_data_chain[p_joint_idx].force;
	jiggle_data_chain.write[p_joint_idx].dynamic_position += operation_bone_trans.get_origin() - jiggle_data_chain[p_joint_idx].last_position;
	jiggle_data_chain.write[p_joint_idx].last_position = operation_bone_trans.get_origin();

	// Collision detection/response
	if (use_colliders) {
		if (execution_mode == SkeletonModificationStack2D::EXECUTION_MODE::execution_mode_physics_process) {
			Ref<World2D> world_2d = stack->skeleton->get_world_2d();
			ERR_FAIL_COND(world_2d.is_null());
			PhysicsDirectSpaceState2D *space_state = PhysicsServer2D::get_singleton()->space_get_direct_state(world_2d->get_space());
			PhysicsDirectSpaceState2D::RayResult ray_result;

			// Add exception support?
			bool ray_hit = space_state->intersect_ray(operation_bone_trans.get_origin(), jiggle_data_chain[p_joint_idx].dynamic_position,
					ray_result, Set<RID>(), collision_mask);

			if (ray_hit) {
				jiggle_data_chain.write[p_joint_idx].dynamic_position = jiggle_data_chain[p_joint_idx].last_noncollision_position;
				jiggle_data_chain.write[p_joint_idx].acceleration = Vector2(0, 0);
				jiggle_data_chain.write[p_joint_idx].velocity = Vector2(0, 0);
			} else {
				jiggle_data_chain.write[p_joint_idx].last_noncollision_position = jiggle_data_chain[p_joint_idx].dynamic_position;
			}
		} else {
			WARN_PRINT_ONCE("Jiggle 2D modifier: You cannot detect colliders without the stack mode being set to _physics_process!");
		}
	}

	// Rotate the bone using the dynamic position!
	operation_bone_trans = operation_bone_trans.looking_at(jiggle_data_chain[p_joint_idx].dynamic_position);
	operation_bone_trans.set_rotation(operation_bone_trans.get_rotation() - operation_bone->get_bone_angle());

	// Reset scale
	operation_bone_trans.set_scale(operation_bone->get_global_transform().get_scale());

	operation_bone->set_global_transform(operation_bone_trans);
	stack->skeleton->set_bone_local_pose_override(jiggle_data_chain[p_joint_idx].bone_idx, operation_bone->get_transform(), stack->strength, true);
}

void SkeletonModification2DJiggle::_update_jiggle_joint_data() {
	for (int i = 0; i < jiggle_data_chain.size(); i++) {
		if (!jiggle_data_chain[i].override_defaults) {
			set_jiggle_joint_stiffness(i, stiffness);
			set_jiggle_joint_mass(i, mass);
			set_jiggle_joint_damping(i, damping);
			set_jiggle_joint_use_gravity(i, use_gravity);
			set_jiggle_joint_gravity(i, gravity);
		}
	}
}

void SkeletonModification2DJiggle::_setup_modification(SkeletonModificationStack2D *p_stack) {
	stack = p_stack;

	if (stack) {
		is_setup = true;

		if (stack->skeleton) {
			for (int i = 0; i < jiggle_data_chain.size(); i++) {
				int bone_idx = jiggle_data_chain[i].bone_idx;
				if (bone_idx > 0 && bone_idx < stack->skeleton->get_bone_count()) {
					Bone2D *bone2d_node = stack->skeleton->get_bone(bone_idx);
					jiggle_data_chain.write[i].dynamic_position = bone2d_node->get_global_transform().get_origin();
				}
			}
		}

		update_target_cache();
	}
}

void SkeletonModification2DJiggle::update_target_cache() {
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
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: node is not in scene tree!");
				target_node_cache = node->get_instance_id();
			}
		}
	}
}

void SkeletonModification2DJiggle::jiggle_joint_update_bone2d_cache(int p_joint_idx) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, jiggle_data_chain.size(), "Cannot update bone2d cache: joint index out of range!");
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update Jiggle " + itos(p_joint_idx) + " Bone2D cache: modification is not properly setup!");
		return;
	}

	jiggle_data_chain.write[p_joint_idx].bone2d_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(jiggle_data_chain[p_joint_idx].bone2d_node)) {
				Node *node = stack->skeleton->get_node(jiggle_data_chain[p_joint_idx].bone2d_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update Jiggle joint " + itos(p_joint_idx) + " Bone2D cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update Jiggle joint " + itos(p_joint_idx) + " Bone2D cache: node is not in scene tree!");
				jiggle_data_chain.write[p_joint_idx].bone2d_node_cache = node->get_instance_id();

				Bone2D *bone = Object::cast_to<Bone2D>(node);
				if (bone) {
					jiggle_data_chain.write[p_joint_idx].bone_idx = bone->get_index_in_skeleton();
				} else {
					ERR_FAIL_MSG("Jiggle joint " + itos(p_joint_idx) + " Bone2D cache: Nodepath to Bone2D is not a Bone2D node!");
				}
			}
		}
	}
}

void SkeletonModification2DJiggle::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_target_cache();
}

NodePath SkeletonModification2DJiggle::get_target_node() const {
	return target_node;
}

void SkeletonModification2DJiggle::set_stiffness(float p_stiffness) {
	ERR_FAIL_COND_MSG(p_stiffness < 0, "Stiffness cannot be set to a negative value!");
	stiffness = p_stiffness;
	_update_jiggle_joint_data();
}

float SkeletonModification2DJiggle::get_stiffness() const {
	return stiffness;
}

void SkeletonModification2DJiggle::set_mass(float p_mass) {
	ERR_FAIL_COND_MSG(p_mass < 0, "Mass cannot be set to a negative value!");
	mass = p_mass;
	_update_jiggle_joint_data();
}

float SkeletonModification2DJiggle::get_mass() const {
	return mass;
}

void SkeletonModification2DJiggle::set_damping(float p_damping) {
	ERR_FAIL_COND_MSG(p_damping < 0, "Damping cannot be set to a negative value!");
	ERR_FAIL_COND_MSG(p_damping > 1, "Damping cannot be more than one!");
	damping = p_damping;
	_update_jiggle_joint_data();
}

float SkeletonModification2DJiggle::get_damping() const {
	return damping;
}

void SkeletonModification2DJiggle::set_use_gravity(bool p_use_gravity) {
	use_gravity = p_use_gravity;
	_update_jiggle_joint_data();
}

bool SkeletonModification2DJiggle::get_use_gravity() const {
	return use_gravity;
}

void SkeletonModification2DJiggle::set_gravity(Vector2 p_gravity) {
	gravity = p_gravity;
	_update_jiggle_joint_data();
}

Vector2 SkeletonModification2DJiggle::get_gravity() const {
	return gravity;
}

void SkeletonModification2DJiggle::set_use_colliders(bool p_use_colliders) {
	use_colliders = p_use_colliders;
	notify_property_list_changed();
}

bool SkeletonModification2DJiggle::get_use_colliders() const {
	return use_colliders;
}

void SkeletonModification2DJiggle::set_collision_mask(int p_mask) {
	collision_mask = p_mask;
}

int SkeletonModification2DJiggle::get_collision_mask() const {
	return collision_mask;
}

// Jiggle joint data functions
int SkeletonModification2DJiggle::get_jiggle_data_chain_length() {
	return jiggle_data_chain.size();
}

void SkeletonModification2DJiggle::set_jiggle_data_chain_length(int p_length) {
	ERR_FAIL_COND(p_length < 0);
	jiggle_data_chain.resize(p_length);
	notify_property_list_changed();
}

void SkeletonModification2DJiggle::set_jiggle_joint_bone2d_node(int p_joint_idx, const NodePath &p_target_node) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, jiggle_data_chain.size(), "Jiggle joint out of range!");
	jiggle_data_chain.write[p_joint_idx].bone2d_node = p_target_node;
	jiggle_joint_update_bone2d_cache(p_joint_idx);

	notify_property_list_changed();
}

NodePath SkeletonModification2DJiggle::get_jiggle_joint_bone2d_node(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, jiggle_data_chain.size(), NodePath(), "Jiggle joint out of range!");
	return jiggle_data_chain[p_joint_idx].bone2d_node;
}

void SkeletonModification2DJiggle::set_jiggle_joint_bone_index(int p_joint_idx, int p_bone_idx) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, jiggle_data_chain.size(), "Jiggle joint out of range!");
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");

	if (is_setup) {
		if (stack->skeleton) {
			ERR_FAIL_INDEX_MSG(p_bone_idx, stack->skeleton->get_bone_count(), "Passed-in Bone index is out of range!");
			jiggle_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
			jiggle_data_chain.write[p_joint_idx].bone2d_node_cache = stack->skeleton->get_bone(p_bone_idx)->get_instance_id();
			jiggle_data_chain.write[p_joint_idx].bone2d_node = stack->skeleton->get_path_to(stack->skeleton->get_bone(p_bone_idx));
		} else {
			WARN_PRINT("Cannot verify the Jiggle joint " + itos(p_joint_idx) + " bone index for this modification...");
			jiggle_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
		}
	} else {
		WARN_PRINT("Cannot verify the Jiggle joint " + itos(p_joint_idx) + " bone index for this modification...");
		jiggle_data_chain.write[p_joint_idx].bone_idx = p_bone_idx;
	}

	notify_property_list_changed();
}

int SkeletonModification2DJiggle::get_jiggle_joint_bone_index(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, jiggle_data_chain.size(), -1, "Jiggle joint out of range!");
	return jiggle_data_chain[p_joint_idx].bone_idx;
}

void SkeletonModification2DJiggle::set_jiggle_joint_override(int joint_idx, bool p_override) {
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].override_defaults = p_override;
	_update_jiggle_joint_data();
	notify_property_list_changed();
}

bool SkeletonModification2DJiggle::get_jiggle_joint_override(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), false);
	return jiggle_data_chain[joint_idx].override_defaults;
}

void SkeletonModification2DJiggle::set_jiggle_joint_stiffness(int joint_idx, float p_stiffness) {
	ERR_FAIL_COND_MSG(p_stiffness < 0, "Stiffness cannot be set to a negative value!");
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].stiffness = p_stiffness;
}

float SkeletonModification2DJiggle::get_jiggle_joint_stiffness(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), -1);
	return jiggle_data_chain[joint_idx].stiffness;
}

void SkeletonModification2DJiggle::set_jiggle_joint_mass(int joint_idx, float p_mass) {
	ERR_FAIL_COND_MSG(p_mass < 0, "Mass cannot be set to a negative value!");
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].mass = p_mass;
}

float SkeletonModification2DJiggle::get_jiggle_joint_mass(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), -1);
	return jiggle_data_chain[joint_idx].mass;
}

void SkeletonModification2DJiggle::set_jiggle_joint_damping(int joint_idx, float p_damping) {
	ERR_FAIL_COND_MSG(p_damping < 0, "Damping cannot be set to a negative value!");
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].damping = p_damping;
}

float SkeletonModification2DJiggle::get_jiggle_joint_damping(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), -1);
	return jiggle_data_chain[joint_idx].damping;
}

void SkeletonModification2DJiggle::set_jiggle_joint_use_gravity(int joint_idx, bool p_use_gravity) {
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].use_gravity = p_use_gravity;
	notify_property_list_changed();
}

bool SkeletonModification2DJiggle::get_jiggle_joint_use_gravity(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), false);
	return jiggle_data_chain[joint_idx].use_gravity;
}

void SkeletonModification2DJiggle::set_jiggle_joint_gravity(int joint_idx, Vector2 p_gravity) {
	ERR_FAIL_INDEX(joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[joint_idx].gravity = p_gravity;
}

Vector2 SkeletonModification2DJiggle::get_jiggle_joint_gravity(int joint_idx) const {
	ERR_FAIL_INDEX_V(joint_idx, jiggle_data_chain.size(), Vector2(0, 0));
	return jiggle_data_chain[joint_idx].gravity;
}

void SkeletonModification2DJiggle::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification2DJiggle::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification2DJiggle::get_target_node);

	ClassDB::bind_method(D_METHOD("set_jiggle_data_chain_length", "length"), &SkeletonModification2DJiggle::set_jiggle_data_chain_length);
	ClassDB::bind_method(D_METHOD("get_jiggle_data_chain_length"), &SkeletonModification2DJiggle::get_jiggle_data_chain_length);

	ClassDB::bind_method(D_METHOD("set_stiffness", "stiffness"), &SkeletonModification2DJiggle::set_stiffness);
	ClassDB::bind_method(D_METHOD("get_stiffness"), &SkeletonModification2DJiggle::get_stiffness);
	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &SkeletonModification2DJiggle::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &SkeletonModification2DJiggle::get_mass);
	ClassDB::bind_method(D_METHOD("set_damping", "damping"), &SkeletonModification2DJiggle::set_damping);
	ClassDB::bind_method(D_METHOD("get_damping"), &SkeletonModification2DJiggle::get_damping);
	ClassDB::bind_method(D_METHOD("set_use_gravity", "use_gravity"), &SkeletonModification2DJiggle::set_use_gravity);
	ClassDB::bind_method(D_METHOD("get_use_gravity"), &SkeletonModification2DJiggle::get_use_gravity);
	ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &SkeletonModification2DJiggle::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity"), &SkeletonModification2DJiggle::get_gravity);

	ClassDB::bind_method(D_METHOD("set_use_colliders", "use_colliders"), &SkeletonModification2DJiggle::set_use_colliders);
	ClassDB::bind_method(D_METHOD("get_use_colliders"), &SkeletonModification2DJiggle::get_use_colliders);
	ClassDB::bind_method(D_METHOD("set_collision_mask", "collision_mask"), &SkeletonModification2DJiggle::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &SkeletonModification2DJiggle::get_collision_mask);

	// Jiggle joint data functions
	ClassDB::bind_method(D_METHOD("set_jiggle_joint_bone2d_node", "joint_idx", "bone2d_node"), &SkeletonModification2DJiggle::set_jiggle_joint_bone2d_node);
	ClassDB::bind_method(D_METHOD("get_jiggle_joint_bone2d_node", "joint_idx"), &SkeletonModification2DJiggle::get_jiggle_joint_bone2d_node);
	ClassDB::bind_method(D_METHOD("set_jiggle_joint_bone_index", "joint_idx", "bone_idx"), &SkeletonModification2DJiggle::set_jiggle_joint_bone_index);
	ClassDB::bind_method(D_METHOD("get_jiggle_joint_bone_index", "joint_idx"), &SkeletonModification2DJiggle::get_jiggle_joint_bone_index);
	ClassDB::bind_method(D_METHOD("set_jiggle_joint_override", "joint_idx", "override"), &SkeletonModification2DJiggle::set_jiggle_joint_override);
	ClassDB::bind_method(D_METHOD("get_jiggle_joint_override", "joint_idx"), &SkeletonModification2DJiggle::get_jiggle_joint_override);
	ClassDB::bind_method(D_METHOD("set_jiggle_joint_stiffness", "joint_idx", "stiffness"), &SkeletonModification2DJiggle::set_jiggle_joint_stiffness);
	ClassDB::bind_method(D_METHOD("get_jiggle_joint_stiffness", "joint_idx"), &SkeletonModification2DJiggle::get_jiggle_joint_stiffness);
	ClassDB::bind_method(D_METHOD("set_jiggle_joint_mass", "joint_idx", "mass"), &SkeletonModification2DJiggle::set_jiggle_joint_mass);
	ClassDB::bind_method(D_METHOD("get_jiggle_joint_mass", "joint_idx"), &SkeletonModification2DJiggle::get_jiggle_joint_mass);
	ClassDB::bind_method(D_METHOD("set_jiggle_joint_damping", "joint_idx", "damping"), &SkeletonModification2DJiggle::set_jiggle_joint_damping);
	ClassDB::bind_method(D_METHOD("get_jiggle_joint_damping", "joint_idx"), &SkeletonModification2DJiggle::get_jiggle_joint_damping);
	ClassDB::bind_method(D_METHOD("set_jiggle_joint_use_gravity", "joint_idx", "use_gravity"), &SkeletonModification2DJiggle::set_jiggle_joint_use_gravity);
	ClassDB::bind_method(D_METHOD("get_jiggle_joint_use_gravity", "joint_idx"), &SkeletonModification2DJiggle::get_jiggle_joint_use_gravity);
	ClassDB::bind_method(D_METHOD("set_jiggle_joint_gravity", "joint_idx", "gravity"), &SkeletonModification2DJiggle::set_jiggle_joint_gravity);
	ClassDB::bind_method(D_METHOD("get_jiggle_joint_gravity", "joint_idx"), &SkeletonModification2DJiggle::get_jiggle_joint_gravity);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "jiggle_data_chain_length", PROPERTY_HINT_RANGE, "0,100,1"), "set_jiggle_data_chain_length", "get_jiggle_data_chain_length");
	ADD_GROUP("Default Joint Settings", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stiffness"), "set_stiffness", "get_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping", PROPERTY_HINT_RANGE, "0, 1, 0.01"), "set_damping", "get_damping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_gravity"), "set_use_gravity", "get_use_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "gravity"), "set_gravity", "get_gravity");
	ADD_GROUP("", "");
}

SkeletonModification2DJiggle::SkeletonModification2DJiggle() {
	stack = nullptr;
	is_setup = false;
	jiggle_data_chain = Vector<Jiggle_Joint_Data2D>();
	stiffness = 3;
	mass = 0.75;
	damping = 0.75;
	use_gravity = false;
	gravity = Vector2(0, 6.0);
	enabled = true;
	editor_draw_gizmo = false; // Nothing to really show in a gizmo right now.
}

SkeletonModification2DJiggle::~SkeletonModification2DJiggle() {
}

///////////////////////////////////////
// TwoBoneIK
///////////////////////////////////////

bool SkeletonModification2DTwoBoneIK::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path == "joint_one_bone_idx") {
		set_joint_one_bone_idx(p_value);
	} else if (path == "joint_one_bone2d_node") {
		set_joint_one_bone2d_node(p_value);
	} else if (path == "joint_two_bone_idx") {
		set_joint_two_bone_idx(p_value);
	} else if (path == "joint_two_bone2d_node") {
		set_joint_two_bone2d_node(p_value);
	}

#ifdef TOOLS_ENABLED
	if (path.begins_with("editor/draw_gizmo")) {
		set_editor_draw_gizmo(p_value);
	} else if (path.begins_with("editor/draw_min_max")) {
		set_editor_draw_min_max(p_value);
	}
#endif // TOOLS_ENABLED

	return true;
}

bool SkeletonModification2DTwoBoneIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path == "joint_one_bone_idx") {
		r_ret = get_joint_one_bone_idx();
	} else if (path == "joint_one_bone2d_node") {
		r_ret = get_joint_one_bone2d_node();
	} else if (path == "joint_two_bone_idx") {
		r_ret = get_joint_two_bone_idx();
	} else if (path == "joint_two_bone2d_node") {
		r_ret = get_joint_two_bone2d_node();
	}

#ifdef TOOLS_ENABLED
	if (path.begins_with("editor/draw_gizmo")) {
		r_ret = get_editor_draw_gizmo();
	} else if (path.begins_with("editor/draw_min_max")) {
		r_ret = get_editor_draw_min_max();
	}
#endif // TOOLS_ENABLED

	return true;
}

void SkeletonModification2DTwoBoneIK::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::INT, "joint_one_bone_idx", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	p_list->push_back(PropertyInfo(Variant::NODE_PATH, "joint_one_bone2d_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Bone2D", PROPERTY_USAGE_DEFAULT));

	p_list->push_back(PropertyInfo(Variant::INT, "joint_two_bone_idx", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	p_list->push_back(PropertyInfo(Variant::NODE_PATH, "joint_two_bone2d_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Bone2D", PROPERTY_USAGE_DEFAULT));

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "editor/draw_gizmo", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::BOOL, "editor/draw_min_max", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	}
#endif // TOOLS_ENABLED
}

void SkeletonModification2DTwoBoneIK::_execute(float delta) {
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

	if (joint_one_bone2d_node_cache.is_null() && !joint_one_bone2d_node.is_empty()) {
		_print_execution_error(true, "Joint one Bone2D node cache is out of date. Attempting to update...");
		update_joint_one_bone2d_cache();
	}
	if (joint_two_bone2d_node_cache.is_null() && !joint_two_bone2d_node.is_empty()) {
		_print_execution_error(true, "Joint two Bone2D node cache is out of date. Attempting to update...");
		update_joint_two_bone2d_cache();
	}

	Node2D *target = Object::cast_to<Node2D>(ObjectDB::get_instance(target_node_cache));
	if (_print_execution_error(!target || !target->is_inside_tree(), "Target node is not in the scene tree. Cannot execute modification!")) {
		return;
	}

	Bone2D *joint_one_bone = stack->skeleton->get_bone(joint_one_bone_idx);
	if (_print_execution_error(joint_one_bone == nullptr, "Joint one bone_idx does not point to a valid bone! Cannot execute modification!")) {
		return;
	}

	Bone2D *joint_two_bone = stack->skeleton->get_bone(joint_two_bone_idx);
	if (_print_execution_error(joint_two_bone == nullptr, "Joint one bone_idx does not point to a valid bone! Cannot execute modification!")) {
		return;
	}

	// Adopted from the links below:
	// http://theorangeduck.com/page/simple-two-joint
	// https://www.alanzucconi.com/2018/05/02/ik-2d-2/
	// With modifications by TwistedTwigleg
	Vector2 target_difference = target->get_global_transform().get_origin() - joint_one_bone->get_global_transform().get_origin();
	float joint_one_to_target = target_difference.length();
	float angle_atan = Math::atan2(target_difference.y, target_difference.x);

	float bone_one_length = joint_one_bone->get_length() * MIN(joint_one_bone->get_global_scale().x, joint_one_bone->get_global_scale().y);
	float bone_two_length = joint_two_bone->get_length() * MIN(joint_two_bone->get_global_scale().x, joint_two_bone->get_global_scale().y);
	bool override_angles_due_to_out_of_range = false;

	if (joint_one_to_target < target_minimum_distance) {
		joint_one_to_target = target_minimum_distance;
	}
	if (joint_one_to_target > target_maximum_distance && target_maximum_distance > 0.0) {
		joint_one_to_target = target_maximum_distance;
	}

	if (bone_one_length + bone_two_length < joint_one_to_target) {
		override_angles_due_to_out_of_range = true;
	}

	if (!override_angles_due_to_out_of_range) {
		float angle_0 = Math::acos(((joint_one_to_target * joint_one_to_target) + (bone_one_length * bone_one_length) - (bone_two_length * bone_two_length)) / (2.0 * joint_one_to_target * bone_one_length));
		float angle_1 = Math::acos(((bone_two_length * bone_two_length) + (bone_one_length * bone_one_length) - (joint_one_to_target * joint_one_to_target)) / (2.0 * bone_two_length * bone_one_length));

		if (flip_bend_direction) {
			angle_0 = -angle_0;
			angle_1 = -angle_1;
		}

		if (isnan(angle_0) || isnan(angle_1)) {
			// We cannot solve for this angle! Do nothing to avoid setting the rotation (and scale) to NaN.
		} else {
			joint_one_bone->set_global_rotation(angle_atan - angle_0 - joint_one_bone->get_bone_angle());
			joint_two_bone->set_rotation(-Math_PI - angle_1 - joint_two_bone->get_bone_angle() + joint_one_bone->get_bone_angle());
		}
	} else {
		joint_one_bone->set_global_rotation(angle_atan - joint_one_bone->get_bone_angle());
		joint_two_bone->set_global_rotation(angle_atan - joint_two_bone->get_bone_angle());
	}

	stack->skeleton->set_bone_local_pose_override(joint_one_bone_idx, joint_one_bone->get_transform(), stack->strength, true);
	stack->skeleton->set_bone_local_pose_override(joint_two_bone_idx, joint_two_bone->get_transform(), stack->strength, true);
}

void SkeletonModification2DTwoBoneIK::_setup_modification(SkeletonModificationStack2D *p_stack) {
	stack = p_stack;

	if (stack) {
		is_setup = true;
		update_target_cache();
		update_joint_one_bone2d_cache();
		update_joint_two_bone2d_cache();
	}
}

void SkeletonModification2DTwoBoneIK::_draw_editor_gizmo() {
	if (!enabled || !is_setup) {
		return;
	}

	Bone2D *operation_bone_one = stack->skeleton->get_bone(joint_one_bone_idx);
	if (!operation_bone_one) {
		return;
	}
	stack->skeleton->draw_set_transform(
			stack->skeleton->get_global_transform().affine_inverse().xform(operation_bone_one->get_global_position()),
			operation_bone_one->get_global_rotation() - stack->skeleton->get_global_rotation());

	Color bone_ik_color = Color(1.0, 0.65, 0.0, 0.4);
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		bone_ik_color = EditorSettings::get_singleton()->get("editors/2d/bone_ik_color");
	}
#endif // TOOLS_ENABLED

	if (flip_bend_direction) {
		float angle = -(Math_PI * 0.5) + operation_bone_one->get_bone_angle();
		stack->skeleton->draw_line(Vector2(0, 0), Vector2(Math::cos(angle), sin(angle)) * (operation_bone_one->get_length() * 0.5), bone_ik_color, 2.0);
	} else {
		float angle = (Math_PI * 0.5) + operation_bone_one->get_bone_angle();
		stack->skeleton->draw_line(Vector2(0, 0), Vector2(Math::cos(angle), sin(angle)) * (operation_bone_one->get_length() * 0.5), bone_ik_color, 2.0);
	}

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		if (editor_draw_min_max) {
			if (target_maximum_distance != 0.0 || target_minimum_distance != 0.0) {
				Vector2 target_direction = Vector2(0, 1);
				if (target_node_cache.is_valid()) {
					stack->skeleton->draw_set_transform(Vector2(0, 0), 0.0);
					Node2D *target = Object::cast_to<Node2D>(ObjectDB::get_instance(target_node_cache));
					target_direction = operation_bone_one->get_global_position().direction_to(target->get_global_position());
				}

				stack->skeleton->draw_circle(target_direction * target_minimum_distance, 8, bone_ik_color);
				stack->skeleton->draw_circle(target_direction * target_maximum_distance, 8, bone_ik_color);
				stack->skeleton->draw_line(target_direction * target_minimum_distance, target_direction * target_maximum_distance, bone_ik_color, 2.0);
			}
		}
	}
#endif // TOOLS_ENABLED
}

void SkeletonModification2DTwoBoneIK::update_target_cache() {
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
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: node is not in the scene tree!");
				target_node_cache = node->get_instance_id();
			}
		}
	}
}

void SkeletonModification2DTwoBoneIK::update_joint_one_bone2d_cache() {
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update joint one Bone2D cache: modification is not properly setup!");
		return;
	}

	joint_one_bone2d_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(joint_one_bone2d_node)) {
				Node *node = stack->skeleton->get_node(joint_one_bone2d_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update update joint one Bone2D cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update update joint one Bone2D cache: node is not in the scene tree!");
				joint_one_bone2d_node_cache = node->get_instance_id();

				Bone2D *bone = Object::cast_to<Bone2D>(node);
				if (bone) {
					joint_one_bone_idx = bone->get_index_in_skeleton();
				} else {
					ERR_FAIL_MSG("update joint one Bone2D cache: Nodepath to Bone2D is not a Bone2D node!");
				}
			}
		}
	}
}

void SkeletonModification2DTwoBoneIK::update_joint_two_bone2d_cache() {
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update joint two Bone2D cache: modification is not properly setup!");
		return;
	}

	joint_two_bone2d_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(joint_two_bone2d_node)) {
				Node *node = stack->skeleton->get_node(joint_two_bone2d_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update update joint two Bone2D cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update update joint two Bone2D cache: node is not in scene tree!");
				joint_two_bone2d_node_cache = node->get_instance_id();

				Bone2D *bone = Object::cast_to<Bone2D>(node);
				if (bone) {
					joint_two_bone_idx = bone->get_index_in_skeleton();
				} else {
					ERR_FAIL_MSG("update joint two Bone2D cache: Nodepath to Bone2D is not a Bone2D node!");
				}
			}
		}
	}
}

void SkeletonModification2DTwoBoneIK::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	update_target_cache();
}

NodePath SkeletonModification2DTwoBoneIK::get_target_node() const {
	return target_node;
}

void SkeletonModification2DTwoBoneIK::set_joint_one_bone2d_node(const NodePath &p_target_node) {
	joint_one_bone2d_node = p_target_node;
	update_joint_one_bone2d_cache();
	notify_property_list_changed();
}

void SkeletonModification2DTwoBoneIK::set_target_minimum_distance(float p_distance) {
	ERR_FAIL_COND_MSG(p_distance < 0, "Target minimum distance cannot be less than zero!");
	target_minimum_distance = p_distance;
}

float SkeletonModification2DTwoBoneIK::get_target_minimum_distance() const {
	return target_minimum_distance;
}

void SkeletonModification2DTwoBoneIK::set_target_maximum_distance(float p_distance) {
	ERR_FAIL_COND_MSG(p_distance < 0, "Target maximum distance cannot be less than zero!");
	target_maximum_distance = p_distance;
}

float SkeletonModification2DTwoBoneIK::get_target_maximum_distance() const {
	return target_maximum_distance;
}

void SkeletonModification2DTwoBoneIK::set_flip_bend_direction(bool p_flip_direction) {
	flip_bend_direction = p_flip_direction;

#ifdef TOOLS_ENABLED
	if (stack && is_setup) {
		stack->set_editor_gizmos_dirty(true);
	}
#endif // TOOLS_ENABLED
}

bool SkeletonModification2DTwoBoneIK::get_flip_bend_direction() const {
	return flip_bend_direction;
}

NodePath SkeletonModification2DTwoBoneIK::get_joint_one_bone2d_node() const {
	return joint_one_bone2d_node;
}

void SkeletonModification2DTwoBoneIK::set_joint_two_bone2d_node(const NodePath &p_target_node) {
	joint_two_bone2d_node = p_target_node;
	update_joint_two_bone2d_cache();
	notify_property_list_changed();
}

NodePath SkeletonModification2DTwoBoneIK::get_joint_two_bone2d_node() const {
	return joint_two_bone2d_node;
}

void SkeletonModification2DTwoBoneIK::set_joint_one_bone_idx(int p_bone_idx) {
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");

	if (is_setup) {
		if (stack->skeleton) {
			ERR_FAIL_INDEX_MSG(p_bone_idx, stack->skeleton->get_bone_count(), "Passed-in Bone index is out of range!");
			joint_one_bone_idx = p_bone_idx;
			joint_one_bone2d_node_cache = stack->skeleton->get_bone(p_bone_idx)->get_instance_id();
			joint_one_bone2d_node = stack->skeleton->get_path_to(stack->skeleton->get_bone(p_bone_idx));
		} else {
			WARN_PRINT("TwoBoneIK: Cannot verify the joint bone index for joint one...");
			joint_one_bone_idx = p_bone_idx;
		}
	} else {
		WARN_PRINT("TwoBoneIK: Cannot verify the joint bone index for joint one...");
		joint_one_bone_idx = p_bone_idx;
	}

	notify_property_list_changed();
}

int SkeletonModification2DTwoBoneIK::get_joint_one_bone_idx() const {
	return joint_one_bone_idx;
}

void SkeletonModification2DTwoBoneIK::set_joint_two_bone_idx(int p_bone_idx) {
	ERR_FAIL_COND_MSG(p_bone_idx < 0, "Bone index is out of range: The index is too low!");

	if (is_setup) {
		if (stack->skeleton) {
			ERR_FAIL_INDEX_MSG(p_bone_idx, stack->skeleton->get_bone_count(), "Passed-in Bone index is out of range!");
			joint_two_bone_idx = p_bone_idx;
			joint_two_bone2d_node_cache = stack->skeleton->get_bone(p_bone_idx)->get_instance_id();
			joint_two_bone2d_node = stack->skeleton->get_path_to(stack->skeleton->get_bone(p_bone_idx));
		} else {
			WARN_PRINT("TwoBoneIK: Cannot verify the joint bone index for joint two...");
			joint_two_bone_idx = p_bone_idx;
		}
	} else {
		WARN_PRINT("TwoBoneIK: Cannot verify the joint bone index for joint two...");
		joint_two_bone_idx = p_bone_idx;
	}

	notify_property_list_changed();
}

int SkeletonModification2DTwoBoneIK::get_joint_two_bone_idx() const {
	return joint_two_bone_idx;
}

#ifdef TOOLS_ENABLED
void SkeletonModification2DTwoBoneIK::set_editor_draw_min_max(bool p_draw) {
	editor_draw_min_max = p_draw;
}

bool SkeletonModification2DTwoBoneIK::get_editor_draw_min_max() const {
	return editor_draw_min_max;
}
#endif // TOOLS_ENABLED

void SkeletonModification2DTwoBoneIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification2DTwoBoneIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification2DTwoBoneIK::get_target_node);

	ClassDB::bind_method(D_METHOD("set_target_minimum_distance", "minimum_distance"), &SkeletonModification2DTwoBoneIK::set_target_minimum_distance);
	ClassDB::bind_method(D_METHOD("get_target_minimum_distance"), &SkeletonModification2DTwoBoneIK::get_target_minimum_distance);
	ClassDB::bind_method(D_METHOD("set_target_maximum_distance", "maximum_distance"), &SkeletonModification2DTwoBoneIK::set_target_maximum_distance);
	ClassDB::bind_method(D_METHOD("get_target_maximum_distance"), &SkeletonModification2DTwoBoneIK::get_target_maximum_distance);
	ClassDB::bind_method(D_METHOD("set_flip_bend_direction", "flip_direction"), &SkeletonModification2DTwoBoneIK::set_flip_bend_direction);
	ClassDB::bind_method(D_METHOD("get_flip_bend_direction"), &SkeletonModification2DTwoBoneIK::get_flip_bend_direction);

	ClassDB::bind_method(D_METHOD("set_joint_one_bone2d_node", "bone2d_node"), &SkeletonModification2DTwoBoneIK::set_joint_one_bone2d_node);
	ClassDB::bind_method(D_METHOD("get_joint_one_bone2d_node"), &SkeletonModification2DTwoBoneIK::get_joint_one_bone2d_node);
	ClassDB::bind_method(D_METHOD("set_joint_one_bone_idx", "bone_idx"), &SkeletonModification2DTwoBoneIK::set_joint_one_bone_idx);
	ClassDB::bind_method(D_METHOD("get_joint_one_bone_idx"), &SkeletonModification2DTwoBoneIK::get_joint_one_bone_idx);

	ClassDB::bind_method(D_METHOD("set_joint_two_bone2d_node", "bone2d_node"), &SkeletonModification2DTwoBoneIK::set_joint_two_bone2d_node);
	ClassDB::bind_method(D_METHOD("get_joint_two_bone2d_node"), &SkeletonModification2DTwoBoneIK::get_joint_two_bone2d_node);
	ClassDB::bind_method(D_METHOD("set_joint_two_bone_idx", "bone_idx"), &SkeletonModification2DTwoBoneIK::set_joint_two_bone_idx);
	ClassDB::bind_method(D_METHOD("get_joint_two_bone_idx"), &SkeletonModification2DTwoBoneIK::get_joint_two_bone_idx);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "target_minimum_distance", PROPERTY_HINT_RANGE, "0, 100000000, 0.01"), "set_target_minimum_distance", "get_target_minimum_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "target_maximum_distance", PROPERTY_HINT_NONE, "0, 100000000, 0.01"), "set_target_maximum_distance", "get_target_maximum_distance");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_bend_direction", PROPERTY_HINT_NONE, ""), "set_flip_bend_direction", "get_flip_bend_direction");
	ADD_GROUP("", "");
}

SkeletonModification2DTwoBoneIK::SkeletonModification2DTwoBoneIK() {
	stack = nullptr;
	is_setup = false;
	enabled = true;
	editor_draw_gizmo = true;
}

SkeletonModification2DTwoBoneIK::~SkeletonModification2DTwoBoneIK() {
}

///////////////////////////////////////
// PhysicalBones
///////////////////////////////////////

bool SkeletonModification2DPhysicalBones::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

#ifdef TOOLS_ENABLED
	// Exposes a way to fetch the PhysicalBone2D nodes from the Godot editor.
	if (is_setup) {
		if (Engine::get_singleton()->is_editor_hint()) {
			if (path.begins_with("fetch_bones")) {
				fetch_physical_bones();
				notify_property_list_changed();
				return true;
			}
		}
	}
#endif //TOOLS_ENABLED

	if (path.begins_with("joint_")) {
		int which = path.get_slicec('_', 1).to_int();
		String what = path.get_slicec('_', 2);
		ERR_FAIL_INDEX_V(which, physical_bone_chain.size(), false);

		if (what == "nodepath") {
			set_physical_bone_node(which, p_value);
		}
		return true;
	}
	return true;
}

bool SkeletonModification2DPhysicalBones::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		if (path.begins_with("fetch_bones")) {
			return true; // Do nothing!
		}
	}
#endif //TOOLS_ENABLED

	if (path.begins_with("joint_")) {
		int which = path.get_slicec('_', 1).to_int();
		String what = path.get_slicec('_', 2);
		ERR_FAIL_INDEX_V(which, physical_bone_chain.size(), false);

		if (what == "nodepath") {
			r_ret = get_physical_bone_node(which);
		}
		return true;
	}
	return true;
}

void SkeletonModification2DPhysicalBones::_get_property_list(List<PropertyInfo> *p_list) const {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "fetch_bones", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	}
#endif //TOOLS_ENABLED

	for (int i = 0; i < physical_bone_chain.size(); i++) {
		String base_string = "joint_" + itos(i) + "_";

		p_list->push_back(PropertyInfo(Variant::NODE_PATH, base_string + "nodepath", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "PhysicalBone2D", PROPERTY_USAGE_DEFAULT));
	}
}

void SkeletonModification2DPhysicalBones::_execute(float delta) {
	ERR_FAIL_COND_MSG(!stack || !is_setup || stack->skeleton == nullptr,
			"Modification is not setup and therefore cannot execute!");
	if (!enabled) {
		return;
	}

	if (_simulation_state_dirty) {
		_update_simulation_state();
	}

	for (int i = 0; i < physical_bone_chain.size(); i++) {
		PhysicalBone_Data2D bone_data = physical_bone_chain[i];
		if (bone_data.physical_bone_node_cache.is_null()) {
			_print_execution_error(true, "PhysicalBone2D cache " + itos(i) + " is out of date. Attempting to update...");
			_physical_bone_update_cache(i);
			continue;
		}

		PhysicalBone2D *physical_bone = Object::cast_to<PhysicalBone2D>(ObjectDB::get_instance(bone_data.physical_bone_node_cache));
		if (_print_execution_error(!physical_bone, "PhysicalBone2D not found at index " + itos(i) + "!")) {
			return;
		}
		if (_print_execution_error(physical_bone->get_bone2d_index() < 0 || physical_bone->get_bone2d_index() > stack->skeleton->get_bone_count(),
					"PhysicalBone2D at index " + itos(i) + " has invalid Bone2D!")) {
			return;
		}
		Bone2D *bone_2d = stack->skeleton->get_bone(physical_bone->get_bone2d_index());

		if (physical_bone->get_simulate_physics() && !physical_bone->get_follow_bone_when_simulating()) {
			bone_2d->set_global_transform(physical_bone->get_global_transform());
			stack->skeleton->set_bone_local_pose_override(physical_bone->get_bone2d_index(), bone_2d->get_transform(), stack->strength, true);
		}
	}
}

void SkeletonModification2DPhysicalBones::_setup_modification(SkeletonModificationStack2D *p_stack) {
	stack = p_stack;

	if (stack) {
		is_setup = true;

		if (stack->skeleton) {
			for (int i = 0; i < physical_bone_chain.size(); i++) {
				_physical_bone_update_cache(i);
			}
		}
	}
}

void SkeletonModification2DPhysicalBones::_physical_bone_update_cache(int p_joint_idx) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, physical_bone_chain.size(), "Cannot update PhysicalBone2D cache: joint index out of range!");
	if (!is_setup || !stack) {
		_print_execution_error(true, "Cannot update PhysicalBone2D cache: modification is not properly setup!");
		return;
	}

	physical_bone_chain.write[p_joint_idx].physical_bone_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(physical_bone_chain[p_joint_idx].physical_bone_node)) {
				Node *node = stack->skeleton->get_node(physical_bone_chain[p_joint_idx].physical_bone_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update Physical Bone2D " + itos(p_joint_idx) + " cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update Physical Bone2D " + itos(p_joint_idx) + " cache: node is not in scene tree!");
				physical_bone_chain.write[p_joint_idx].physical_bone_node_cache = node->get_instance_id();
			}
		}
	}
}

int SkeletonModification2DPhysicalBones::get_physical_bone_chain_length() {
	return physical_bone_chain.size();
}

void SkeletonModification2DPhysicalBones::set_physical_bone_chain_length(int p_length) {
	ERR_FAIL_COND(p_length < 0);
	physical_bone_chain.resize(p_length);
	notify_property_list_changed();
}

void SkeletonModification2DPhysicalBones::fetch_physical_bones() {
	ERR_FAIL_COND_MSG(!stack->skeleton, "No skeleton found! Cannot fetch physical bones!");

	physical_bone_chain.clear();

	List<Node *> node_queue = List<Node *>();
	node_queue.push_back(stack->skeleton);

	while (node_queue.size() > 0) {
		Node *node_to_process = node_queue[0];
		node_queue.pop_front();

		PhysicalBone2D *potential_bone = Object::cast_to<PhysicalBone2D>(node_to_process);
		if (potential_bone) {
			PhysicalBone_Data2D new_data = PhysicalBone_Data2D();
			new_data.physical_bone_node = stack->skeleton->get_path_to(potential_bone);
			new_data.physical_bone_node_cache = potential_bone->get_instance_id();
			physical_bone_chain.push_back(new_data);
		}
		for (int i = 0; i < node_to_process->get_child_count(); i++) {
			node_queue.push_back(node_to_process->get_child(i));
		}
	}
}

void SkeletonModification2DPhysicalBones::start_simulation(const TypedArray<StringName> &p_bones) {
	_simulation_state_dirty = true;
	_simulation_state_dirty_names = p_bones;
	_simulation_state_dirty_process = true;

	if (is_setup) {
		_update_simulation_state();
	}
}

void SkeletonModification2DPhysicalBones::stop_simulation(const TypedArray<StringName> &p_bones) {
	_simulation_state_dirty = true;
	_simulation_state_dirty_names = p_bones;
	_simulation_state_dirty_process = false;

	if (is_setup) {
		_update_simulation_state();
	}
}

void SkeletonModification2DPhysicalBones::_update_simulation_state() {
	if (!_simulation_state_dirty) {
		return;
	}
	_simulation_state_dirty = false;

	if (_simulation_state_dirty_names.size() <= 0) {
		for (int i = 0; i < physical_bone_chain.size(); i++) {
			PhysicalBone2D *physical_bone = Object::cast_to<PhysicalBone2D>(stack->skeleton->get_node(physical_bone_chain[i].physical_bone_node));
			if (!physical_bone) {
				continue;
			}

			physical_bone->set_simulate_physics(_simulation_state_dirty_process);
		}
	} else {
		for (int i = 0; i < physical_bone_chain.size(); i++) {
			PhysicalBone2D *physical_bone = Object::cast_to<PhysicalBone2D>(ObjectDB::get_instance(physical_bone_chain[i].physical_bone_node_cache));
			if (!physical_bone) {
				continue;
			}
			if (_simulation_state_dirty_names.has(physical_bone->get_name())) {
				physical_bone->set_simulate_physics(_simulation_state_dirty_process);
			}
		}
	}
}

void SkeletonModification2DPhysicalBones::set_physical_bone_node(int p_joint_idx, const NodePath &p_nodepath) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, physical_bone_chain.size(), "Joint index out of range!");
	physical_bone_chain.write[p_joint_idx].physical_bone_node = p_nodepath;
	_physical_bone_update_cache(p_joint_idx);
}

NodePath SkeletonModification2DPhysicalBones::get_physical_bone_node(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, physical_bone_chain.size(), NodePath(), "Joint index out of range!");
	return physical_bone_chain[p_joint_idx].physical_bone_node;
}

void SkeletonModification2DPhysicalBones::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_physical_bone_chain_length", "length"), &SkeletonModification2DPhysicalBones::set_physical_bone_chain_length);
	ClassDB::bind_method(D_METHOD("get_physical_bone_chain_length"), &SkeletonModification2DPhysicalBones::get_physical_bone_chain_length);

	ClassDB::bind_method(D_METHOD("set_physical_bone_node", "joint_idx", "physicalbone2d_node"), &SkeletonModification2DPhysicalBones::set_physical_bone_node);
	ClassDB::bind_method(D_METHOD("get_physical_bone_node", "joint_idx"), &SkeletonModification2DPhysicalBones::get_physical_bone_node);

	ClassDB::bind_method(D_METHOD("fetch_physical_bones"), &SkeletonModification2DPhysicalBones::fetch_physical_bones);
	ClassDB::bind_method(D_METHOD("start_simulation", "bones"), &SkeletonModification2DPhysicalBones::start_simulation, DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("stop_simulation", "bones"), &SkeletonModification2DPhysicalBones::stop_simulation, DEFVAL(Array()));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "physical_bone_chain_length", PROPERTY_HINT_RANGE, "0,100,1"), "set_physical_bone_chain_length", "get_physical_bone_chain_length");
}

SkeletonModification2DPhysicalBones::SkeletonModification2DPhysicalBones() {
	stack = nullptr;
	is_setup = false;
	physical_bone_chain = Vector<PhysicalBone_Data2D>();
	enabled = true;
	editor_draw_gizmo = false; // Nothing to really show in a gizmo right now.
}

SkeletonModification2DPhysicalBones::~SkeletonModification2DPhysicalBones() {
}

///////////////////////////////////////
// StackHolder
///////////////////////////////////////

bool SkeletonModification2DStackHolder::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path == "held_modification_stack") {
		set_held_modification_stack(p_value);
	}

#ifdef TOOLS_ENABLED
	if (path == "editor/draw_gizmo") {
		set_editor_draw_gizmo(p_value);
	}
#endif // TOOLS_ENABLED

	return true;
}

bool SkeletonModification2DStackHolder::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path == "held_modification_stack") {
		r_ret = get_held_modification_stack();
	}

#ifdef TOOLS_ENABLED
	if (path == "editor/draw_gizmo") {
		r_ret = get_editor_draw_gizmo();
	}
#endif // TOOLS_ENABLED

	return true;
}

void SkeletonModification2DStackHolder::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::OBJECT, "held_modification_stack", PROPERTY_HINT_RESOURCE_TYPE, "SkeletonModificationStack2D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE));

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "editor/draw_gizmo", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
	}
#endif // TOOLS_ENABLED
}

void SkeletonModification2DStackHolder::_execute(float delta) {
	ERR_FAIL_COND_MSG(!stack || !is_setup || stack->skeleton == nullptr,
			"Modification is not setup and therefore cannot execute!");

	if (held_modification_stack.is_valid()) {
		held_modification_stack->execute(delta, execution_mode);
	}
}

void SkeletonModification2DStackHolder::_setup_modification(SkeletonModificationStack2D *p_stack) {
	stack = p_stack;

	if (stack != nullptr) {
		is_setup = true;

		if (held_modification_stack.is_valid()) {
			held_modification_stack->set_skeleton(stack->get_skeleton());
			held_modification_stack->setup();
		}
	}
}

void SkeletonModification2DStackHolder::_draw_editor_gizmo() {
	if (stack) {
		if (held_modification_stack.is_valid()) {
			held_modification_stack->draw_editor_gizmos();
		}
	}
}

void SkeletonModification2DStackHolder::set_held_modification_stack(Ref<SkeletonModificationStack2D> p_held_stack) {
	held_modification_stack = p_held_stack;

	if (is_setup && held_modification_stack.is_valid()) {
		held_modification_stack->set_skeleton(stack->get_skeleton());
		held_modification_stack->setup();
	}
}

Ref<SkeletonModificationStack2D> SkeletonModification2DStackHolder::get_held_modification_stack() const {
	return held_modification_stack;
}

void SkeletonModification2DStackHolder::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_held_modification_stack", "held_modification_stack"), &SkeletonModification2DStackHolder::set_held_modification_stack);
	ClassDB::bind_method(D_METHOD("get_held_modification_stack"), &SkeletonModification2DStackHolder::get_held_modification_stack);
}

SkeletonModification2DStackHolder::SkeletonModification2DStackHolder() {
	stack = nullptr;
	is_setup = false;
	enabled = true;
}

SkeletonModification2DStackHolder::~SkeletonModification2DStackHolder() {
}
