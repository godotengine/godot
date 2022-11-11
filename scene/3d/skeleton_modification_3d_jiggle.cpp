/*************************************************************************/
/*  skeleton_modification_3d_jiggle.cpp                                  */
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

#include "scene/3d/skeleton_modification_3d_jiggle.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/skeleton_modification_3d.h"

bool SkeletonModification3DJiggle::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_")) {
		const int jiggle_size = jiggle_data_chain.size();
		int which = path.get_slicec('/', 0).substr(6).to_int();
		String what = path.get_slicec('/', 1);
		ERR_FAIL_INDEX_V(which, jiggle_size, false);

		if (what == "bone") {
			set_joint_bone(which, p_value);
		} else if (what == "override_defaults") {
			set_joint_override(which, p_value);
		} else if (what == "stiffness") {
			set_joint_stiffness(which, p_value);
		} else if (what == "mass") {
			set_joint_mass(which, p_value);
		} else if (what == "damping") {
			set_joint_damping(which, p_value);
		} else if (what == "use_gravity") {
			set_joint_use_gravity(which, p_value);
		} else if (what == "gravity") {
			set_joint_gravity(which, p_value);
		} else if (what == "roll") {
			set_joint_roll(which, Math::deg_to_rad(real_t(p_value)));
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

	if (path.begins_with("joint_")) {
		const int jiggle_size = jiggle_data_chain.size();
		int which = path.get_slicec('/', 0).substr(6).to_int();
		String what = path.get_slicec('/', 1);
		ERR_FAIL_INDEX_V(which, jiggle_size, false);

		if (what == "bone") {
			r_ret = get_joint_bone(which);
		} else if (what == "override_defaults") {
			r_ret = get_joint_override(which);
		} else if (what == "stiffness") {
			r_ret = get_joint_stiffness(which);
		} else if (what == "mass") {
			r_ret = get_joint_mass(which);
		} else if (what == "damping") {
			r_ret = get_joint_damping(which);
		} else if (what == "use_gravity") {
			r_ret = get_joint_use_gravity(which);
		} else if (what == "gravity") {
			r_ret = get_joint_gravity(which);
		} else if (what == "roll") {
			r_ret = Math::rad_to_deg(get_joint_roll(which));
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

	for (uint32_t i = 0; i < jiggle_data_chain.size(); i++) {
		String base_string = "joint_" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::STRING, base_string + "bone", get_skeleton() ? PROPERTY_HINT_ENUM : PROPERTY_HINT_NONE, get_bone_name_list(), PROPERTY_USAGE_DEFAULT));
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

void SkeletonModification3DJiggle::_execute_jiggle_joint(int p_joint_idx, Transform3D parent_transform, Vector3 target_position, real_t p_delta) {
	// Adopted from: https://wiki.unity3d.com/index.php/JiggleBone
	// With modifications by TwistedTwigleg.

	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton || !_cache_bone(jiggle_data_chain[p_joint_idx].bone_idx, jiggle_data_chain[p_joint_idx].bone_name)) {
		WARN_PRINT_ONCE(String("Jiggle joint ") + itos(p_joint_idx) + " bone index is invalid. Cannot execute modification!");
		return;
	}
	Transform3D new_bone_trans = parent_transform * skeleton->get_bone_pose(jiggle_data_chain[p_joint_idx].bone_idx);

	jiggle_data_chain[p_joint_idx].force = (target_position - jiggle_data_chain[p_joint_idx].dynamic_position) * jiggle_data_chain[p_joint_idx].stiffness * p_delta;

	if (jiggle_data_chain[p_joint_idx].use_gravity) {
		Vector3 gravity_to_apply = new_bone_trans.basis.inverse().xform(jiggle_data_chain[p_joint_idx].gravity);
		jiggle_data_chain[p_joint_idx].force += gravity_to_apply * p_delta;
	}

	jiggle_data_chain[p_joint_idx].acceleration = jiggle_data_chain[p_joint_idx].force / jiggle_data_chain[p_joint_idx].mass;
	jiggle_data_chain[p_joint_idx].velocity += jiggle_data_chain[p_joint_idx].acceleration * (1 - jiggle_data_chain[p_joint_idx].damping);

	jiggle_data_chain[p_joint_idx].dynamic_position += jiggle_data_chain[p_joint_idx].velocity + jiggle_data_chain[p_joint_idx].force;
	jiggle_data_chain[p_joint_idx].dynamic_position += new_bone_trans.origin - jiggle_data_chain[p_joint_idx].last_position;
	jiggle_data_chain[p_joint_idx].last_position = new_bone_trans.origin;

	// Get the forward direction that the basis is facing in right now.
	Vector3 forward_vector = get_bone_rest_forward_vector(jiggle_data_chain[p_joint_idx].bone_idx);

	// Rotate the bone using the dynamic position!
	new_bone_trans.basis.rotate_to_align(forward_vector, new_bone_trans.origin.direction_to(jiggle_data_chain[p_joint_idx].dynamic_position));

	// Roll
	new_bone_trans.basis.rotate_local(forward_vector, jiggle_data_chain[p_joint_idx].roll);

	new_bone_trans = parent_transform.affine_inverse() * new_bone_trans;
	skeleton->set_bone_pose_rotation(jiggle_data_chain[p_joint_idx].bone_idx, new_bone_trans.basis.get_rotation_quaternion());
}

void SkeletonModification3DJiggle::_update_jiggle_joint_data() {
	for (uint32_t i = 0; i < jiggle_data_chain.size(); i++) {
		if (!jiggle_data_chain[i].override_defaults) {
			set_joint_stiffness(i, stiffness);
			set_joint_mass(i, mass);
			set_joint_damping(i, damping);
			set_joint_use_gravity(i, use_gravity);
			set_joint_gravity(i, gravity);
		}
	}
}

void SkeletonModification3DJiggle::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	if (!target_node.is_empty()) {
		target_bone = String();
	}
	target_cache = Variant();
}

NodePath SkeletonModification3DJiggle::get_target_node() const {
	return target_node;
}

void SkeletonModification3DJiggle::set_target_bone(const String &p_target_bone) {
	target_bone = p_target_bone;
	if (!target_bone.is_empty()) {
		target_node = NodePath();
	}
	target_cache = Variant();
}

String SkeletonModification3DJiggle::get_target_bone() const {
	return target_bone;
}

void SkeletonModification3DJiggle::set_stiffness(real_t p_stiffness) {
	ERR_FAIL_COND_MSG(p_stiffness < 0, "Stiffness cannot be set to a negative value!");
	stiffness = p_stiffness;
	_update_jiggle_joint_data();
}

real_t SkeletonModification3DJiggle::get_stiffness() const {
	return stiffness;
}

void SkeletonModification3DJiggle::set_mass(real_t p_mass) {
	ERR_FAIL_COND_MSG(p_mass < 0, "Mass cannot be set to a negative value!");
	mass = p_mass;
	_update_jiggle_joint_data();
}

real_t SkeletonModification3DJiggle::get_mass() const {
	return mass;
}

void SkeletonModification3DJiggle::set_damping(real_t p_damping) {
	ERR_FAIL_COND_MSG(p_damping < 0, "Damping cannot be set to a negative value!");
	ERR_FAIL_COND_MSG(p_damping > 1, "Damping cannot be more than one!");
	damping = p_damping;
	_update_jiggle_joint_data();
}

real_t SkeletonModification3DJiggle::get_damping() const {
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
	if (is_inside_tree()) {
		set_physics_process_internal(use_colliders);
	}
	notify_property_list_changed();
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
int SkeletonModification3DJiggle::get_joint_count() {
	return jiggle_data_chain.size();
}

void SkeletonModification3DJiggle::set_joint_count(int p_length) {
	ERR_FAIL_COND(p_length < 0);
	jiggle_data_chain.resize(p_length);
	notify_property_list_changed();
}

void SkeletonModification3DJiggle::set_joint_bone(int p_joint_idx, String p_name) {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);

	jiggle_data_chain[p_joint_idx].bone_name = p_name;
	jiggle_data_chain[p_joint_idx].bone_idx = UNCACHED_BONE_IDX;
}

String SkeletonModification3DJiggle::get_joint_bone(int p_joint_idx) const {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, "");
	return jiggle_data_chain[p_joint_idx].bone_name;
}

void SkeletonModification3DJiggle::set_joint_override(int p_joint_idx, bool p_override) {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	jiggle_data_chain[p_joint_idx].override_defaults = p_override;
	_update_jiggle_joint_data();
	notify_property_list_changed();
}

bool SkeletonModification3DJiggle::get_joint_override(int p_joint_idx) const {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return jiggle_data_chain[p_joint_idx].override_defaults;
}

void SkeletonModification3DJiggle::set_joint_stiffness(int p_joint_idx, real_t p_stiffness) {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_COND_MSG(p_stiffness < 0, "Stiffness cannot be set to a negative value!");
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	jiggle_data_chain[p_joint_idx].stiffness = p_stiffness;
}

real_t SkeletonModification3DJiggle::get_joint_stiffness(int p_joint_idx) const {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, -1);
	return jiggle_data_chain[p_joint_idx].stiffness;
}

void SkeletonModification3DJiggle::set_joint_mass(int p_joint_idx, real_t p_mass) {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_COND_MSG(p_mass < 0, "Mass cannot be set to a negative value!");
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	jiggle_data_chain[p_joint_idx].mass = p_mass;
}

real_t SkeletonModification3DJiggle::get_joint_mass(int p_joint_idx) const {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, -1);
	return jiggle_data_chain[p_joint_idx].mass;
}

void SkeletonModification3DJiggle::set_joint_damping(int p_joint_idx, real_t p_damping) {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_COND_MSG(p_damping < 0, "Damping cannot be set to a negative value!");
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	jiggle_data_chain[p_joint_idx].damping = p_damping;
}

real_t SkeletonModification3DJiggle::get_joint_damping(int p_joint_idx) const {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, -1);
	return jiggle_data_chain[p_joint_idx].damping;
}

void SkeletonModification3DJiggle::set_joint_use_gravity(int p_joint_idx, bool p_use_gravity) {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	jiggle_data_chain[p_joint_idx].use_gravity = p_use_gravity;
	notify_property_list_changed();
}

bool SkeletonModification3DJiggle::get_joint_use_gravity(int p_joint_idx) const {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, false);
	return jiggle_data_chain[p_joint_idx].use_gravity;
}

void SkeletonModification3DJiggle::set_joint_gravity(int p_joint_idx, Vector3 p_gravity) {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	jiggle_data_chain[p_joint_idx].gravity = p_gravity;
}

Vector3 SkeletonModification3DJiggle::get_joint_gravity(int p_joint_idx) const {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, Vector3(0, 0, 0));
	return jiggle_data_chain[p_joint_idx].gravity;
}

void SkeletonModification3DJiggle::set_joint_roll(int p_joint_idx, real_t p_roll) {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX(p_joint_idx, bone_chain_size);
	jiggle_data_chain[p_joint_idx].roll = p_roll;
}

real_t SkeletonModification3DJiggle::get_joint_roll(int p_joint_idx) const {
	const int bone_chain_size = jiggle_data_chain.size();
	ERR_FAIL_INDEX_V(p_joint_idx, bone_chain_size, 0.0);
	return jiggle_data_chain[p_joint_idx].roll;
}

void SkeletonModification3DJiggle::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification3DJiggle::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification3DJiggle::get_target_node);
	ClassDB::bind_method(D_METHOD("set_target_bone", "target_bone_name"), &SkeletonModification3DJiggle::set_target_bone);
	ClassDB::bind_method(D_METHOD("get_target_bone"), &SkeletonModification3DJiggle::get_target_bone);

	ClassDB::bind_method(D_METHOD("set_joint_count", "length"), &SkeletonModification3DJiggle::set_joint_count);
	ClassDB::bind_method(D_METHOD("get_joint_count"), &SkeletonModification3DJiggle::get_joint_count);

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
	ClassDB::bind_method(D_METHOD("set_joint_bone", "joint_idx", "name"), &SkeletonModification3DJiggle::set_joint_bone);
	ClassDB::bind_method(D_METHOD("get_joint_bone", "joint_idx"), &SkeletonModification3DJiggle::get_joint_bone);
	ClassDB::bind_method(D_METHOD("set_joint_override", "joint_idx", "override"), &SkeletonModification3DJiggle::set_joint_override);
	ClassDB::bind_method(D_METHOD("get_joint_override", "joint_idx"), &SkeletonModification3DJiggle::get_joint_override);
	ClassDB::bind_method(D_METHOD("set_joint_stiffness", "joint_idx", "stiffness"), &SkeletonModification3DJiggle::set_joint_stiffness);
	ClassDB::bind_method(D_METHOD("get_joint_stiffness", "joint_idx"), &SkeletonModification3DJiggle::get_joint_stiffness);
	ClassDB::bind_method(D_METHOD("set_joint_mass", "joint_idx", "mass"), &SkeletonModification3DJiggle::set_joint_mass);
	ClassDB::bind_method(D_METHOD("get_joint_mass", "joint_idx"), &SkeletonModification3DJiggle::get_joint_mass);
	ClassDB::bind_method(D_METHOD("set_joint_damping", "joint_idx", "damping"), &SkeletonModification3DJiggle::set_joint_damping);
	ClassDB::bind_method(D_METHOD("get_joint_damping", "joint_idx"), &SkeletonModification3DJiggle::get_joint_damping);
	ClassDB::bind_method(D_METHOD("set_joint_use_gravity", "joint_idx", "use_gravity"), &SkeletonModification3DJiggle::set_joint_use_gravity);
	ClassDB::bind_method(D_METHOD("get_joint_use_gravity", "joint_idx"), &SkeletonModification3DJiggle::get_joint_use_gravity);
	ClassDB::bind_method(D_METHOD("set_joint_gravity", "joint_idx", "gravity"), &SkeletonModification3DJiggle::set_joint_gravity);
	ClassDB::bind_method(D_METHOD("get_joint_gravity", "joint_idx"), &SkeletonModification3DJiggle::get_joint_gravity);
	ClassDB::bind_method(D_METHOD("set_joint_roll", "joint_idx", "roll"), &SkeletonModification3DJiggle::set_joint_roll);
	ClassDB::bind_method(D_METHOD("get_joint_roll", "joint_idx"), &SkeletonModification3DJiggle::get_joint_roll);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "target_bone"), "set_target_bone", "get_target_bone");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_target_node", "get_target_node");
	ADD_GROUP("Default Joint Settings", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stiffness"), "set_stiffness", "get_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping", PROPERTY_HINT_RANGE, "0, 1, 0.01"), "set_damping", "get_damping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_gravity"), "set_use_gravity", "get_use_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
	ADD_GROUP("", "");
	ADD_ARRAY_COUNT("Jiggle Chain Joint Count", "joint_count", "set_joint_count", "get_joint_count", "joint_");
}

SkeletonModification3DJiggle::SkeletonModification3DJiggle() {
	jiggle_data_chain = Vector<Jiggle_Joint_Data>();
	stiffness = 3;
	mass = 0.75;
	damping = 0.75;
	use_gravity = false;
	gravity = Vector3(0, -6.0, 0);
}

SkeletonModification3DJiggle::~SkeletonModification3DJiggle() {
}

void SkeletonModification3DJiggle::execute(real_t delta) {
	SkeletonModification3D::execute(delta);

	Skeleton3D *skeleton = get_skeleton();
	if (skeleton == nullptr || !_cache_target(target_cache, target_node, target_bone)) {
		return;
	}

	if (!initialized_dynamic_position) {
		initialized_dynamic_position = true;
		for (uint32_t i = 0; i < jiggle_data_chain.size(); i++) {
			int bone_idx = jiggle_data_chain[i].bone_idx;
			if (bone_idx > 0 && bone_idx < skeleton->get_bone_count()) {
				jiggle_data_chain[i].dynamic_position = get_bone_transform(bone_idx).get_origin();
			}
		}
	}
	Vector3 target_position = get_target_transform(target_cache).origin;
	int prev_bone_idx = -1;
	Transform3D prev_transform;
	for (uint32_t i = 0; i < jiggle_data_chain.size(); i++) {
		if (!_cache_bone(jiggle_data_chain[i].bone_idx, jiggle_data_chain[i].bone_name)) {
			return;
		}
		int bone_idx = jiggle_data_chain[i].bone_idx;
		int par_idx = skeleton->get_bone_parent(bone_idx);
		Transform3D parent_bone_transform;
		bool first = true;
		while (par_idx >= 0 && par_idx != prev_bone_idx) {
			if (first) {
				parent_bone_transform = skeleton->get_bone_pose(par_idx);
				first = false;
			} else {
				parent_bone_transform = skeleton->get_bone_pose(par_idx) * parent_bone_transform;
			}
			par_idx = skeleton->get_bone_parent(par_idx);
		}
		if (i != 0 && par_idx < 0) {
			parent_bone_transform = prev_transform * parent_bone_transform;
		}
		_execute_jiggle_joint(i, parent_bone_transform, target_position, delta);
		prev_transform = parent_bone_transform * skeleton->get_bone_pose(bone_idx);
		prev_bone_idx = bone_idx;
	}
}

void SkeletonModification3DJiggle::_execute_joint_collision(Ref<World3D> world_3d, int p_joint_idx) {
	// Collision detection/response
	Skeleton3D *skeleton = get_skeleton();
	ERR_FAIL_COND(skeleton == nullptr);
	PhysicsDirectSpaceState3D *space_state = PhysicsServer3D::get_singleton()->space_get_direct_state(world_3d->get_space());
	PhysicsDirectSpaceState3D::RayResult ray_result;

	PhysicsDirectSpaceState3D::RayParameters ray_params;
	// Convert to world transforms, which is what the physics server needs
	ray_params.from = skeleton->get_global_transform().xform(jiggle_data_chain[p_joint_idx].last_noncollision_position);
	ray_params.to = skeleton->get_global_transform().xform(jiggle_data_chain[p_joint_idx].dynamic_position);
	ray_params.collision_mask = collision_mask;

	bool ray_hit = space_state->intersect_ray(ray_params, ray_result);

	if (ray_hit) {
		jiggle_data_chain[p_joint_idx].dynamic_position = jiggle_data_chain[p_joint_idx].last_noncollision_position;
		jiggle_data_chain[p_joint_idx].acceleration = Vector3(0, 0, 0);
		jiggle_data_chain[p_joint_idx].velocity = Vector3(0, 0, 0);
	} else {
		jiggle_data_chain[p_joint_idx].last_noncollision_position = jiggle_data_chain[p_joint_idx].dynamic_position;
	}
}

void SkeletonModification3DJiggle::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
			set_physics_process_internal(use_colliders);
			break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS:
			if (use_colliders) {
				Skeleton3D *skeleton = get_skeleton();
				ERR_FAIL_COND(skeleton == nullptr);
				Ref<World3D> world_3d = skeleton->get_world_3d();
				ERR_FAIL_COND(world_3d.is_null());
				for (uint32_t i = 0; i < jiggle_data_chain.size(); i++) {
					_execute_joint_collision(world_3d, i);
				}
			}
			break;
	}
}

void SkeletonModification3DJiggle::skeleton_changed(Skeleton3D *skeleton) {
	target_cache = Variant();
	for (uint32_t i = 0; i < jiggle_data_chain.size(); i++) {
		jiggle_data_chain[i].bone_idx = UNCACHED_BONE_IDX;
	}
	SkeletonModification3D::skeleton_changed(skeleton);
}

bool SkeletonModification3DJiggle::is_bone_property(String property_name) const {
	if (property_name == "target_bone" || property_name.ends_with("/bone")) {
		return true;
	}
	return SkeletonModification3D::is_bone_property(property_name);
}

bool SkeletonModification3DJiggle::is_property_hidden(String p_property_name) const {
	if (p_property_name == "target_bone" && !target_node.is_empty()) {
		return true;
	}
	return SkeletonModification3D::is_property_hidden(p_property_name);
}

PackedStringArray SkeletonModification3DJiggle::get_configuration_warnings() const {
	PackedStringArray ret = SkeletonModification3D::get_configuration_warnings();
	if (!get_skeleton()) {
		return ret;
	}
	if (!_cache_target(target_cache, target_node, target_bone)) {
		ret.append(vformat("Target %s %s was not found.", target_node.is_empty() ? "bone" : "node", target_node.is_empty() ? target_bone : (String)target_node));
	}
	for (uint32_t i = 0; i < jiggle_data_chain.size(); i++) {
		if (!_cache_bone(jiggle_data_chain[i].bone_idx, jiggle_data_chain[i].bone_name)) {
			ret.append(vformat("Joint %d bone %s was not found.", i, jiggle_data_chain[i].bone_name));
		}
	}
	return ret;
}
