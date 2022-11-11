/*************************************************************************/
/*  ik_ewbik.cpp                                                         */
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

#include "ik_ewbik.h"
#include "core/core_string_names.h"
#include "core/error/error_macros.h"
#include "ik_bone_3d.h"
#include "ik_kusudama.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#endif

void SkeletonModification3DNBoneIK::set_pin_count(int32_t p_value) {
	int32_t old_count = pins.size();
	pin_count = p_value;
	pins.resize(p_value);
	for (int32_t pin_i = p_value; pin_i-- > old_count;) {
		pins.write[pin_i].instantiate();
	}
	set_dirty();
}

int32_t SkeletonModification3DNBoneIK::get_pin_count() const {
	return pin_count;
}

void SkeletonModification3DNBoneIK::add_pin(const StringName &p_name, const NodePath &p_target_node) {
	for (Ref<IKEffectorTemplate> pin : pins) {
		if (pin->get_name() == p_name) {
			return;
		}
	}
	int32_t count = get_pin_count();
	set_pin_count(count + 1);
	set_pin_bone(count, p_name);
	set_pin_target_nodepath(count, p_target_node);
	set_dirty();
}

void SkeletonModification3DNBoneIK::set_pin_bone(int32_t p_pin_index, const String &p_bone) {
	ERR_FAIL_INDEX(p_pin_index, pins.size());
	Ref<IKEffectorTemplate> effector_template = pins[p_pin_index];
	if (effector_template.is_null()) {
		effector_template.instantiate();
		pins.write[p_pin_index] = effector_template;
	}
	effector_template->set_name(p_bone);
	set_dirty();
}

void SkeletonModification3DNBoneIK::set_pin_target_nodepath(int32_t p_pin_index, const NodePath &p_target_node) {
	ERR_FAIL_INDEX(p_pin_index, pins.size());
	Ref<IKEffectorTemplate> effector_template = pins[p_pin_index];
	if (effector_template.is_null()) {
		effector_template.instantiate();
		pins.write[p_pin_index] = effector_template;
	}
	effector_template->set_target_node(p_target_node);
	set_dirty();
}

NodePath SkeletonModification3DNBoneIK::get_pin_target_nodepath(int32_t p_pin_index) {
	ERR_FAIL_INDEX_V(p_pin_index, pins.size(), NodePath());
	const Ref<IKEffectorTemplate> effector_template = pins[p_pin_index];
	return effector_template->get_target_node();
}

Vector<Ref<IKEffectorTemplate>> SkeletonModification3DNBoneIK::get_bone_effectors() const {
	return pins;
}

void SkeletonModification3DNBoneIK::remove_pin(int32_t p_index) {
	ERR_FAIL_INDEX(p_index, pins.size());
	pins.remove_at(p_index);
	pin_count--;
	pins.resize(pin_count);
	set_dirty();
}

void SkeletonModification3DNBoneIK::update_ik_bones_transform() {
	for (int32_t bone_i = bone_list.size(); bone_i-- > 0;) {
		Ref<IKBone3D> bone = bone_list[bone_i];
		if (bone.is_null()) {
			continue;
		}
		bone->set_initial_pose(get_skeleton());
		if (bone->is_pinned()) {
			bone->get_pin()->update_target_global_transform(get_skeleton(), this);
		}
	}
}

void SkeletonModification3DNBoneIK::update_skeleton_bones_transform() {
	for (int32_t bone_i = bone_list.size(); bone_i-- > 0;) {
		Ref<IKBone3D> bone = bone_list[bone_i];
		if (bone.is_null()) {
			continue;
		}
		if (bone->get_bone_id() == -1) {
			continue;
		}
		bone->set_skeleton_bone_pose(get_skeleton());
	}
}

void SkeletonModification3DNBoneIK::_validate_property(PropertyInfo &property) const {
	if (property.name == "root_bone") {
		if (get_skeleton()) {
			String names;
			for (int i = 0; i < get_skeleton()->get_bone_count(); i++) {
				String name = get_skeleton()->get_bone_name(i);
				name += ",";
				names += name;
			}
			property.hint = PROPERTY_HINT_ENUM_SUGGESTION;
			property.hint_string = names;
		} else {
			property.hint = PROPERTY_HINT_NONE;
			property.hint_string = "";
		}
	}
	if (property.name == "tip_bone") {
		if (get_skeleton()) {
			String names;
			BoneId root_bone_id = get_skeleton()->find_bone(root_bone);
			for (int i = 0; i < get_skeleton()->get_bone_count(); i++) {
				if (i <= root_bone_id || root_bone_id == -1) {
					continue;
				}
				String name = get_skeleton()->get_bone_name(i);
				name += ",";
				names += name;
			}
			property.hint = PROPERTY_HINT_ENUM_SUGGESTION;
			property.hint_string = names;
		} else {
			property.hint = PROPERTY_HINT_NONE;
			property.hint_string = "";
		}
	}
}

void SkeletonModification3DNBoneIK::_get_property_list(List<PropertyInfo> *p_list) const {
	RBSet<String> existing_pins;
	for (int32_t pin_i = 0; pin_i < get_pin_count(); pin_i++) {
		const String name = get_pin_bone_name(pin_i);
		existing_pins.insert(name);
	}
	p_list->push_back(
			PropertyInfo(Variant::INT, "pin_count",
					PROPERTY_HINT_RANGE, "0,1024,or_greater", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY,
					"Pins,pins/"));
	for (int pin_i = 0; pin_i < pin_count; pin_i++) {
		PropertyInfo effector_name;
		effector_name.type = Variant::STRING_NAME;
		effector_name.name = "pins/" + itos(pin_i) + "/name";
		if (get_skeleton()) {
			String names;
			for (int bone_i = 0; bone_i < get_skeleton()->get_bone_count(); bone_i++) {
				String name = get_skeleton()->get_bone_name(bone_i);
				if (existing_pins.has(name)) {
					continue;
				}
				name += ",";
				names += name;
			}
			effector_name.hint = PROPERTY_HINT_ENUM_SUGGESTION;
			effector_name.hint_string = names;
		} else {
			effector_name.hint = PROPERTY_HINT_NONE;
			effector_name.hint_string = "";
		}
		p_list->push_back(effector_name);
		p_list->push_back(
				PropertyInfo(Variant::NODE_PATH, "pins/" + itos(pin_i) + "/target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"));
		p_list->push_back(
				PropertyInfo(Variant::FLOAT, "pins/" + itos(pin_i) + "/depth_falloff", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater"));
		p_list->push_back(
				PropertyInfo(Variant::FLOAT, "pins/" + itos(pin_i) + "/weight", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater"));
		p_list->push_back(
				PropertyInfo(Variant::VECTOR3, "pins/" + itos(pin_i) + "/direction_priorities", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater"));
	}

	RBSet<String> existing_constraints;
	for (int32_t constraint_i = 0; constraint_i < get_constraint_count(); constraint_i++) {
		const String name = get_constraint_name(constraint_i);
		existing_constraints.insert(name);
	}
	p_list->push_back(
			PropertyInfo(Variant::INT, "constraint_count",
					PROPERTY_HINT_RANGE, "0,1024,or_greater", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY,
					"Constraints,constraints/"));
	for (int constraint_i = 0; constraint_i < get_constraint_count(); constraint_i++) {
		PropertyInfo bone_name;
		bone_name.type = Variant::STRING_NAME;
		bone_name.name = "constraints/" + itos(constraint_i) + "/name";
		if (get_skeleton()) {
			String names;
			for (int bone_i = 0; bone_i < get_skeleton()->get_bone_count(); bone_i++) {
				String name = get_skeleton()->get_bone_name(bone_i);
				if (existing_constraints.has(name)) {
					continue;
				}
				name += ",";
				names += name;
				existing_constraints.insert(name);
			}
			bone_name.hint = PROPERTY_HINT_ENUM_SUGGESTION;
			bone_name.hint_string = names;
		} else {
			bone_name.hint = PROPERTY_HINT_NONE;
			bone_name.hint_string = "";
		}
		p_list->push_back(bone_name);
		p_list->push_back(
				PropertyInfo(Variant::FLOAT, "constraints/" + itos(constraint_i) + "/kusudama_twist_from", PROPERTY_HINT_RANGE, "-360.0,360.0,0.1,radians"));
		p_list->push_back(
				PropertyInfo(Variant::FLOAT, "constraints/" + itos(constraint_i) + "/kusudama_twist_to", PROPERTY_HINT_RANGE, "-360.0,360.0,0.1,radians"));
		p_list->push_back(
				PropertyInfo(Variant::INT, "constraints/" + itos(constraint_i) + "/kusudama_limit_cone_count",
						PROPERTY_HINT_RANGE, "0,30,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY,
						"Limit Cones,constraints/" + itos(constraint_i) + "/kusudama_limit_cone/"));
		for (int cone_i = 0; cone_i < get_kusudama_limit_cone_count(constraint_i); cone_i++) {
			p_list->push_back(
					PropertyInfo(Variant::VECTOR3, "constraints/" + itos(constraint_i) + "/kusudama_limit_cone/" + itos(cone_i) + "/center", PROPERTY_HINT_RANGE, "0.0,1.0,0.01,or_greater"));
			p_list->push_back(
					PropertyInfo(Variant::FLOAT, "constraints/" + itos(constraint_i) + "/kusudama_limit_cone/" + itos(cone_i) + "/radius", PROPERTY_HINT_RANGE, "0,360,0.1,radians"));
		}
	}
}

bool SkeletonModification3DNBoneIK::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;
	if (name == "constraint_count") {
		r_ret = get_constraint_count();
		return true;
	} else if (name == "pin_count") {
		r_ret = get_pin_count();
		return true;
	} else if (name.begins_with("pins/")) {
		int index = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(index, pins.size(), false);
		Ref<IKEffectorTemplate> effector_template = pins[index];
		ERR_FAIL_NULL_V(effector_template, false);
		if (what == "name") {
			r_ret = effector_template->get_name();
			return true;
		} else if (what == "target_node") {
			r_ret = effector_template->get_target_node();
			return true;
		} else if (what == "depth_falloff") {
			r_ret = get_pin_depth_falloff(index);
			return true;
		} else if (what == "weight") {
			r_ret = get_pin_weight(index);
			return true;
		} else if (what == "direction_priorities") {
			r_ret = get_pin_direction_priorities(index);
			return true;
		}
	} else if (name.begins_with("constraints/")) {
		int index = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(index, constraint_count, false);
		String begins = "constraints/" + itos(index) + "/kusudama_limit_cone";
		if (what == "name") {
			ERR_FAIL_INDEX_V(index, constraint_names.size(), false);
			r_ret = constraint_names[index];
			return true;
		} else if (what == "kusudama_twist_from") {
			r_ret = get_kusudama_twist(index).x;
			return true;
		} else if (what == "kusudama_twist_to") {
			r_ret = get_kusudama_twist(index).y;
			return true;
		} else if (what == "kusudama_limit_cone_count") {
			r_ret = get_kusudama_limit_cone_count(index);
			return true;
		} else if (name.begins_with(begins)) {
			int32_t cone_index = name.get_slicec('/', 3).to_int();
			String cone_what = name.get_slicec('/', 4);
			if (cone_what == "center") {
				r_ret = get_kusudama_limit_cone_center(index, cone_index);
				return true;
			} else if (cone_what == "radius") {
				r_ret = get_kusudama_limit_cone_radius(index, cone_index);
				return true;
			}
		}
	}

	return false;
}

bool SkeletonModification3DNBoneIK::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;
	if (name == "constraint_count") {
		set_constraint_count(p_value);
		return true;
	} else if (name == "pin_count") {
		set_pin_count(p_value);
		return true;
	} else if (name.begins_with("pins/")) {
		int index = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(index, pin_count, true);
		if (what == "name") {
			set_pin_bone(index, p_value);
			return true;
		} else if (what == "target_node") {
			set_pin_target_nodepath(index, p_value);
			String existing_bone = get_pin_bone_name(index);
			if (existing_bone.is_empty()) {
				return false;
			}
			return true;
		} else if (what == "depth_falloff") {
			set_pin_depth_falloff(index, p_value);
			return true;
		} else if (what == "weight") {
			set_pin_weight(index, p_value);
			return true;
		} else if (what == "direction_priorities") {
			set_pin_direction_priorities(index, p_value);
			return true;
		}
	} else if (name.begins_with("constraints/")) {
		int index = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		String begins = "constraints/" + itos(index) + "/kusudama_limit_cone/";
		if (what == "name") {
			if (index >= constraint_names.size()) {
				set_constraint_count(constraint_count);
			}
			set_constraint_name(index, p_value);
			return true;
		} else if (what == "kusudama_twist_from") {
			Vector2 kusudama_twist_from = get_kusudama_twist(index);
			set_kusudama_twist(index, Vector2(p_value, kusudama_twist_from.y));
			return true;
		} else if (what == "kusudama_twist_to") {
			Vector2 kusudama_twist_to = get_kusudama_twist(index);
			set_kusudama_twist(index, Vector2(kusudama_twist_to.x, p_value));
			return true;
		} else if (what == "kusudama_limit_cone_count") {
			set_kusudama_limit_cone_count(index, p_value);
			return true;
		} else if (name.begins_with(begins)) {
			int cone_index = name.get_slicec('/', 3).to_int();
			String cone_what = name.get_slicec('/', 4);
			if (cone_what == "center") {
				Vector3 center = p_value;
				if (Math::is_zero_approx(center.length_squared())) {
					center = Vector3(0.0, 1.0, 0.0);
				}
				set_kusudama_limit_cone_center(index, cone_index, center);
				return true;
			} else if (cone_what == "radius") {
				set_kusudama_limit_cone_radius(index, cone_index, p_value);
				return true;
			}
		}
	}
	return false;
}

void SkeletonModification3DNBoneIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_dirty"), &SkeletonModification3DNBoneIK::set_dirty);
	ClassDB::bind_method(D_METHOD("set_root_bone", "root_bone"), &SkeletonModification3DNBoneIK::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone"), &SkeletonModification3DNBoneIK::get_root_bone);
	ClassDB::bind_method(D_METHOD("set_tip_bone", "tip_bone"), &SkeletonModification3DNBoneIK::set_tip_bone);
	ClassDB::bind_method(D_METHOD("get_tip_bone"), &SkeletonModification3DNBoneIK::get_tip_bone);
	ClassDB::bind_method(D_METHOD("set_kusudama_limit_cone_radius", "index", "cone_index", "radius"), &SkeletonModification3DNBoneIK::set_kusudama_limit_cone_radius);
	ClassDB::bind_method(D_METHOD("get_kusudama_limit_cone_radius", "index", "cone_index"), &SkeletonModification3DNBoneIK::get_kusudama_limit_cone_radius);
	ClassDB::bind_method(D_METHOD("set_kusudama_limit_cone_center", "index", "cone_index", "center"), &SkeletonModification3DNBoneIK::set_kusudama_limit_cone_center);
	ClassDB::bind_method(D_METHOD("get_kusudama_limit_cone_center", "index", "cone_index"), &SkeletonModification3DNBoneIK::get_kusudama_limit_cone_center);
	ClassDB::bind_method(D_METHOD("set_kusudama_limit_cone_count", "index", "count"), &SkeletonModification3DNBoneIK::set_kusudama_limit_cone_count);
	ClassDB::bind_method(D_METHOD("get_kusudama_limit_cone_count", "index"), &SkeletonModification3DNBoneIK::get_kusudama_limit_cone_count);
	ClassDB::bind_method(D_METHOD("set_kusudama_twist", "index", "limit"), &SkeletonModification3DNBoneIK::set_kusudama_twist);
	ClassDB::bind_method(D_METHOD("get_kusudama_twist", "index"), &SkeletonModification3DNBoneIK::get_kusudama_twist);
	ClassDB::bind_method(D_METHOD("set_pin_depth_falloff", "index", "falloff"), &SkeletonModification3DNBoneIK::set_pin_depth_falloff);
	ClassDB::bind_method(D_METHOD("get_pin_depth_falloff", "index"), &SkeletonModification3DNBoneIK::get_pin_depth_falloff);
	ClassDB::bind_method(D_METHOD("set_constraint_name", "index", "name"), &SkeletonModification3DNBoneIK::set_constraint_name);
	ClassDB::bind_method(D_METHOD("get_constraint_name", "index"), &SkeletonModification3DNBoneIK::get_constraint_name);
	ClassDB::bind_method(D_METHOD("get_segmented_skeleton"), &SkeletonModification3DNBoneIK::get_segmented_skeleton);
	ClassDB::bind_method(D_METHOD("get_max_ik_iterations"), &SkeletonModification3DNBoneIK::get_max_ik_iterations);
	ClassDB::bind_method(D_METHOD("set_max_ik_iterations", "count"), &SkeletonModification3DNBoneIK::set_max_ik_iterations);
	ClassDB::bind_method(D_METHOD("get_constraint_count"), &SkeletonModification3DNBoneIK::get_constraint_count);
	ClassDB::bind_method(D_METHOD("set_constraint_count", "count"),
			&SkeletonModification3DNBoneIK::set_constraint_count);
	ClassDB::bind_method(D_METHOD("get_pin_count"), &SkeletonModification3DNBoneIK::get_pin_count);
	ClassDB::bind_method(D_METHOD("set_pin_count", "count"),
			&SkeletonModification3DNBoneIK::set_pin_count);
	ClassDB::bind_method(D_METHOD("remove_pin", "index"),
			&SkeletonModification3DNBoneIK::remove_pin);
	ClassDB::bind_method(D_METHOD("get_pin_bone_name", "index"), &SkeletonModification3DNBoneIK::get_pin_bone_name);
	ClassDB::bind_method(D_METHOD("set_pin_bone_name", "index", "name"), &SkeletonModification3DNBoneIK::set_pin_bone_name);
	ClassDB::bind_method(D_METHOD("get_pin_direction_priorities", "index"), &SkeletonModification3DNBoneIK::get_pin_direction_priorities);
	ClassDB::bind_method(D_METHOD("set_pin_direction_priorities", "index", "priority"), &SkeletonModification3DNBoneIK::set_pin_direction_priorities);
	ClassDB::bind_method(D_METHOD("queue_print_skeleton", "enable"), &SkeletonModification3DNBoneIK::queue_print_skeleton);
	ClassDB::bind_method(D_METHOD("get_default_damp"), &SkeletonModification3DNBoneIK::get_default_damp);
	ClassDB::bind_method(D_METHOD("set_default_damp", "damp"), &SkeletonModification3DNBoneIK::set_default_damp);
	ClassDB::bind_method(D_METHOD("get_pin_nodepath", "index"), &SkeletonModification3DNBoneIK::get_pin_nodepath);
	ClassDB::bind_method(D_METHOD("set_pin_nodepath", "index", "nodepath"), &SkeletonModification3DNBoneIK::set_pin_nodepath);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "root_bone", PROPERTY_HINT_ENUM_SUGGESTION), "set_root_bone", "get_root_bone");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "tip_bone", PROPERTY_HINT_ENUM_SUGGESTION), "set_tip_bone", "get_tip_bone");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_ik_iterations", PROPERTY_HINT_RANGE, "1,150,1,or_greater"), "set_max_ik_iterations", "get_max_ik_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_damp", PROPERTY_HINT_RANGE, "0.01,180.0,0.01,radians,exp", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_default_damp", "get_default_damp");
}

SkeletonModification3DNBoneIK::SkeletonModification3DNBoneIK() {
}

SkeletonModification3DNBoneIK::~SkeletonModification3DNBoneIK() {
}

void SkeletonModification3DNBoneIK::queue_print_skeleton(bool p_skeleton_debug) {
	queue_debug_skeleton = p_skeleton_debug;
}

float SkeletonModification3DNBoneIK::get_pin_depth_falloff(int32_t p_effector_index) const {
	ERR_FAIL_INDEX_V(p_effector_index, pins.size(), 0.0f);
	const Ref<IKEffectorTemplate> effector_template = pins[p_effector_index];
	return effector_template->get_depth_falloff();
}

void SkeletonModification3DNBoneIK::set_pin_depth_falloff(int32_t p_effector_index, const float p_depth_falloff) {
	Ref<IKEffectorTemplate> effector_template = pins[p_effector_index];
	ERR_FAIL_NULL(effector_template);
	effector_template->set_depth_falloff(p_depth_falloff);
	set_dirty();
}

void SkeletonModification3DNBoneIK::set_constraint_count(int32_t p_count) {
	int32_t old_count = constraint_names.size();
	constraint_count = p_count;
	constraint_names.resize(p_count);
	kusudama_twist.resize(p_count);
	kusudama_limit_cone_count.resize(p_count);
	kusudama_limit_cones.resize(p_count);
	for (int32_t constraint_i = p_count; constraint_i-- > old_count;) {
		constraint_names.write[constraint_i] = String();
		kusudama_limit_cone_count.write[constraint_i] = 0;
		kusudama_limit_cones.write[constraint_i].resize(0);
		kusudama_twist.write[constraint_i] = Vector2();
	}
	set_dirty();
}

int32_t SkeletonModification3DNBoneIK::get_constraint_count() const {
	return constraint_count;
}

inline StringName SkeletonModification3DNBoneIK::get_constraint_name(int32_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, constraint_names.size(), StringName());
	return constraint_names[p_index];
}

void SkeletonModification3DNBoneIK::set_kusudama_twist(int32_t p_index, Vector2 p_to) {
	ERR_FAIL_INDEX(p_index, constraint_count);
	kusudama_twist.write[p_index] = p_to;
	set_dirty();
}

int32_t SkeletonModification3DNBoneIK::find_effector_id(StringName p_bone_name) {
	for (int32_t constraint_i = 0; constraint_i < constraint_count; constraint_i++) {
		if (constraint_names[constraint_i] == p_bone_name) {
			return constraint_i;
		}
	}
	return -1;
}

void SkeletonModification3DNBoneIK::set_kusudama_limit_cone(int32_t p_contraint_index, int32_t p_index,
		Vector3 p_center, float p_radius) {
	ERR_FAIL_INDEX(p_contraint_index, kusudama_limit_cones.size());
	Vector<Vector4> cones = kusudama_limit_cones.write[p_contraint_index];
	Vector3 center = p_center.normalized();
	Vector4 cone;
	cone.x = center.x;
	cone.y = center.y;
	cone.z = center.z;
	cone.w = p_radius;
	cones.write[p_index] = cone;
	kusudama_limit_cones.write[p_contraint_index] = cones;
	set_dirty();
}

Vector3 SkeletonModification3DNBoneIK::get_kusudama_limit_cone_center(int32_t p_contraint_index, int32_t p_index) const {
	ERR_FAIL_INDEX_V(p_contraint_index, kusudama_limit_cone_count.size(), Vector3(0.0, 1.0, 0.0));
	ERR_FAIL_INDEX_V(p_contraint_index, kusudama_limit_cones.size(), Vector3(0.0, 1.0, 0.0));
	ERR_FAIL_INDEX_V(p_index, kusudama_limit_cones[p_contraint_index].size(), Vector3(0.0, 1.0, 0.0));
	const Vector4 &cone = kusudama_limit_cones[p_contraint_index][p_index];
	Vector3 ret;
	ret.x = cone.x;
	ret.y = cone.y;
	ret.z = cone.z;
	return ret;
}

float SkeletonModification3DNBoneIK::get_kusudama_limit_cone_radius(int32_t p_contraint_index, int32_t p_index) const {
	ERR_FAIL_INDEX_V(p_contraint_index, kusudama_limit_cone_count.size(), Math_TAU);
	ERR_FAIL_INDEX_V(p_contraint_index, kusudama_limit_cones.size(), Math_TAU);
	ERR_FAIL_INDEX_V(p_index, kusudama_limit_cones[p_contraint_index].size(), Math_TAU);
	return kusudama_limit_cones[p_contraint_index][p_index].w;
}

int32_t SkeletonModification3DNBoneIK::get_kusudama_limit_cone_count(int32_t p_contraint_index) const {
	return kusudama_limit_cone_count[p_contraint_index];
}

void SkeletonModification3DNBoneIK::set_kusudama_limit_cone_count(int32_t p_contraint_index, int32_t p_count) {
	ERR_FAIL_INDEX(p_contraint_index, kusudama_limit_cone_count.size());
	ERR_FAIL_INDEX(p_contraint_index, kusudama_limit_cones.size());
	int32_t old_cone_count = kusudama_limit_cones[p_contraint_index].size();
	kusudama_limit_cone_count.write[p_contraint_index] = p_count;
	Vector<Vector4> &cones = kusudama_limit_cones.write[p_contraint_index];
	cones.resize(p_count);
	for (int32_t cone_i = p_count; cone_i-- > old_cone_count;) {
		Vector4 &cone = cones.write[cone_i];
		cone.x = 0.0f;
		cone.y = 1.0f;
		cone.z = 0.0f;
		cone.w = Math::deg_to_rad(10.0f);
	}
	set_dirty();
}

real_t SkeletonModification3DNBoneIK::get_default_damp() const {
	return default_damp;
}

void SkeletonModification3DNBoneIK::set_default_damp(float p_default_damp) {
	default_damp = p_default_damp;
	set_dirty();
}

StringName SkeletonModification3DNBoneIK::get_pin_bone_name(int32_t p_effector_index) const {
	ERR_FAIL_INDEX_V(p_effector_index, pins.size(), "");
	Ref<IKEffectorTemplate> effector_template = pins[p_effector_index];
	return effector_template->get_name();
}

void SkeletonModification3DNBoneIK::set_kusudama_limit_cone_radius(int32_t p_effector_index, int32_t p_index, float p_radius) {
	ERR_FAIL_INDEX(p_effector_index, kusudama_limit_cone_count.size());
	ERR_FAIL_INDEX(p_effector_index, kusudama_limit_cones.size());
	ERR_FAIL_INDEX(p_index, kusudama_limit_cone_count[p_effector_index]);
	ERR_FAIL_INDEX(p_index, kusudama_limit_cones[p_effector_index].size());
	Vector4 &cone = kusudama_limit_cones.write[p_effector_index].write[p_index];
	cone.w = p_radius;
	set_dirty();
}

void SkeletonModification3DNBoneIK::set_kusudama_limit_cone_center(int32_t p_effector_index, int32_t p_index, Vector3 p_center) {
	ERR_FAIL_INDEX(p_effector_index, kusudama_limit_cone_count.size());
	ERR_FAIL_INDEX(p_effector_index, kusudama_limit_cones.size());
	ERR_FAIL_INDEX(p_index, kusudama_limit_cones[p_effector_index].size());
	Vector4 &cone = kusudama_limit_cones.write[p_effector_index].write[p_index];
	cone.x = p_center.x;
	cone.y = p_center.y;
	cone.z = p_center.z;
	set_dirty();
}

Vector2 SkeletonModification3DNBoneIK::get_kusudama_twist(int32_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, kusudama_twist.size(), Vector2());
	return kusudama_twist[p_index];
}

void SkeletonModification3DNBoneIK::set_constraint_name(int32_t p_index, String p_name) {
	ERR_FAIL_INDEX(p_index, constraint_names.size());
	constraint_names.write[p_index] = p_name;
	set_dirty();
}

Ref<IKBoneSegment> SkeletonModification3DNBoneIK::get_segmented_skeleton() {
	return segmented_skeleton;
}
float SkeletonModification3DNBoneIK::get_max_ik_iterations() const {
	return max_ik_iterations;
}

void SkeletonModification3DNBoneIK::set_max_ik_iterations(const float &p_max_ik_iterations) {
	max_ik_iterations = p_max_ik_iterations;
}

void SkeletonModification3DNBoneIK::set_pin_bone_name(int32_t p_effector_index, StringName p_name) const {
	ERR_FAIL_INDEX(p_effector_index, pins.size());
	Ref<IKEffectorTemplate> effector_template = pins[p_effector_index];
	effector_template->set_name(p_name);
}

void SkeletonModification3DNBoneIK::set_pin_nodepath(int32_t p_effector_index, NodePath p_node_path) {
	ERR_FAIL_INDEX(p_effector_index, pins.size());
	Ref<IKEffectorTemplate> effector_template = pins[p_effector_index];
	effector_template->set_target_node(p_node_path);
}

NodePath SkeletonModification3DNBoneIK::get_pin_nodepath(int32_t p_effector_index) const {
	ERR_FAIL_INDEX_V(p_effector_index, pins.size(), NodePath());
	Ref<IKEffectorTemplate> effector_template = pins[p_effector_index];
	return effector_template->get_target_node();
}

void SkeletonModification3DNBoneIK::execute(real_t delta) {
	if (pin_count == 0) {
		return;
	}
	if (segmented_skeleton.is_null()) {
		set_dirty();
	}
	if (is_dirty) {
		skeleton_changed(get_skeleton());
		notify_property_list_changed();
		is_dirty = false;
	}
	if (bone_list.size()) {
		Ref<IKNode3D> root_ik_bone = bone_list.write[0]->get_ik_transform();
		if (root_ik_bone.is_null()) {
			return;
		}
		Ref<IKNode3D> root_ik_parent_transform = root_ik_bone->get_parent();
		if (root_ik_parent_transform.is_null()) {
			return;
		}
		root_ik_parent_transform->set_transform(get_skeleton()->get_transform());
	}
	update_ik_bones_transform();
	for (int32_t i = 0; i < get_max_ik_iterations(); i++) {
		if (segmented_skeleton.is_null()) {
			break;
		}
		segmented_skeleton->segment_solver(get_default_damp());
	}
	update_skeleton_bones_transform();
	get_skeleton()->update_gizmos();
}

void SkeletonModification3DNBoneIK::skeleton_changed(Skeleton3D *p_skeleton) {
	if (!p_skeleton) {
		return;
	}
	if (!root_bone) {
		Vector<int32_t> roots = p_skeleton->get_parentless_bones();
		if (roots.size()) {
			StringName parentless_bone = p_skeleton->get_bone_name(roots[0]);
			set_root_bone(parentless_bone);
		}
	}
	ERR_FAIL_COND(!root_bone);
	BoneId root_bone_index = p_skeleton->find_bone(root_bone);
	BoneId tip_bone_index = p_skeleton->find_bone(tip_bone);
	if (segmented_skeleton.is_valid()) {
		segmented_skeleton->unreference();
	}
	segmented_skeleton = Ref<IKBoneSegment>(memnew(IKBoneSegment(p_skeleton, root_bone, pins, nullptr, root_bone_index, tip_bone_index)));
	segmented_skeleton->get_root()->get_ik_transform()->set_parent(root_transform);
	segmented_skeleton->generate_default_segments_from_root(pins, root_bone_index, tip_bone_index);
	bone_list.clear();
	segmented_skeleton->create_bone_list(bone_list, true, queue_debug_skeleton);
	Vector<Vector<real_t>> weight_array;
	segmented_skeleton->update_pinned_list(weight_array);
	segmented_skeleton->recursive_create_headings_arrays_for(segmented_skeleton);
	update_ik_bones_transform();
	for (int constraint_i = 0; constraint_i < constraint_count; constraint_i++) {
		String bone = constraint_names[constraint_i];
		BoneId bone_id = p_skeleton->find_bone(bone);
		for (Ref<IKBone3D> ik_bone_3d : bone_list) {
			if (ik_bone_3d->get_bone_id() != bone_id) {
				continue;
			}
			Ref<IKNode3D> bone_direction_transform;
			bone_direction_transform.instantiate();
			bone_direction_transform->set_parent(ik_bone_3d->get_ik_transform());
			bone_direction_transform->set_transform(Transform3D(Basis(), ik_bone_3d->get_bone_direction_transform()->get_transform().origin));
			Ref<IKKusudama> constraint = Ref<IKKusudama>(memnew(IKKusudama(ik_bone_3d)));
			const Vector2 axial_limit = get_kusudama_twist(constraint_i);
			constraint->enable_orientational_limits();
			for (int32_t cone_i = 0; cone_i < kusudama_limit_cone_count[constraint_i]; cone_i++) {
				Ref<IKLimitCone> previous_cone;
				if (cone_i > 0) {
					previous_cone = constraint->get_limit_cones()[cone_i - 1];
				}
				Vector4 cone = kusudama_limit_cones[constraint_i][cone_i];
				constraint->add_limit_cone(Vector3(cone.x, cone.y, cone.z), cone.w);
			}
			constraint->_update_constraint();
			constraint->enable_axial_limits();
			constraint->set_axial_limits(axial_limit.x, axial_limit.y);
			ik_bone_3d->add_constraint(constraint);
			break;
		}
	}
	for (Ref<IKBone3D> ik_bone_3d : bone_list) {
		Ref<IKKusudama> constraint = ik_bone_3d->get_constraint();
		if (constraint.is_null()) {
			continue;
		}
		constraint->update_tangent_radii();
		constraint->update_rotational_freedom();
	}
	if (queue_debug_skeleton) {
		queue_debug_skeleton = false;
	}
}

StringName SkeletonModification3DNBoneIK::get_root_bone() const {
	return root_bone;
}

void SkeletonModification3DNBoneIK::set_root_bone(const StringName &p_root_bone) {
	root_bone = p_root_bone;
	set_dirty();
}

StringName SkeletonModification3DNBoneIK::get_tip_bone() const {
	return tip_bone;
}

void SkeletonModification3DNBoneIK::set_tip_bone(StringName p_bone) {
	tip_bone = p_bone;
	set_dirty();
}

real_t SkeletonModification3DNBoneIK::get_pin_weight(int32_t p_pin_index) const {
	ERR_FAIL_INDEX_V(p_pin_index, pins.size(), 0.0);
	const Ref<IKEffectorTemplate> effector_template = pins[p_pin_index];
	return effector_template->get_weight();
}

void SkeletonModification3DNBoneIK::set_pin_weight(int32_t p_pin_index, const real_t &p_weight) {
	ERR_FAIL_INDEX(p_pin_index, pins.size());
	Ref<IKEffectorTemplate> effector_template = pins[p_pin_index];
	if (effector_template.is_null()) {
		effector_template.instantiate();
		pins.write[p_pin_index] = effector_template;
	}
	effector_template->set_weight(p_weight);
	set_dirty();
}

Vector3 SkeletonModification3DNBoneIK::get_pin_direction_priorities(int32_t p_pin_index) const {
	ERR_FAIL_INDEX_V(p_pin_index, pins.size(), Vector3(0, 0, 0));
	const Ref<IKEffectorTemplate> effector_template = pins[p_pin_index];
	return effector_template->get_direction_priorities();
}

void SkeletonModification3DNBoneIK::set_pin_direction_priorities(int32_t p_pin_index, const Vector3 &p_priority_direction) {
	ERR_FAIL_INDEX(p_pin_index, pins.size());
	Ref<IKEffectorTemplate> effector_template = pins[p_pin_index];
	if (effector_template.is_null()) {
		effector_template.instantiate();
		pins.write[p_pin_index] = effector_template;
	}
	effector_template->set_direction_priorities(p_priority_direction);
	set_dirty();
}

void SkeletonModification3DNBoneIK::set_dirty() {
	is_dirty = true;
}
