/*************************************************************************/
/*  skeleton_modification_2d_jiggle.cpp                                  */
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

#include "skeleton_modification_2d_jiggle.h"

#include "scene/2d/skeleton_2d.h"
#include "scene/resources/world_2d.h"

bool SkeletonModification2DJiggle::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_")) {
		int which = path.get_slicec('/', 0).substr(6).to_int();
		String what = path.get_slicec('/', 1);
		ERR_FAIL_INDEX_V(which, jiggle_data_chain.size(), false);

		if (what == "bone_node") {
			set_joint_bone_node(which, p_value);
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

	if (path.begins_with("joint_")) {
		int which = path.get_slicec('/', 0).substr(6).to_int();
		String what = path.get_slicec('/', 1);
		ERR_FAIL_INDEX_V(which, jiggle_data_chain.size(), false);

		if (what == "bone_node") {
			r_ret = get_joint_bone_node(which);
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
		String base_string = "joint_" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::INT, base_string + "bone_index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, base_string + "bone_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Bone2D", PROPERTY_USAGE_DEFAULT));
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

void SkeletonModification2DJiggle::_execute_jiggle_joint(int p_joint_idx, Vector2 target_position, float p_delta) {
	// Adopted from: https://wiki.unity3d.com/index.php/JiggleBone
	// With modifications by TwistedTwigleg.

	Bone2D *operation_bone = _cache_bone(jiggle_data_chain[p_joint_idx].bone_node_cache, jiggle_data_chain[p_joint_idx].bone_node);
	ERR_FAIL_COND(operation_bone == nullptr);
	Transform2D operation_bone_trans = operation_bone->get_global_transform();

	jiggle_data_chain.write[p_joint_idx].force = (target_position - jiggle_data_chain[p_joint_idx].dynamic_position) * jiggle_data_chain[p_joint_idx].stiffness * p_delta;

	if (jiggle_data_chain[p_joint_idx].use_gravity) {
		jiggle_data_chain.write[p_joint_idx].force += jiggle_data_chain[p_joint_idx].gravity * p_delta;
	}

	jiggle_data_chain.write[p_joint_idx].acceleration = jiggle_data_chain[p_joint_idx].force / jiggle_data_chain[p_joint_idx].mass;
	jiggle_data_chain.write[p_joint_idx].velocity += jiggle_data_chain[p_joint_idx].acceleration * (1 - jiggle_data_chain[p_joint_idx].damping);

	jiggle_data_chain.write[p_joint_idx].dynamic_position += jiggle_data_chain[p_joint_idx].velocity + jiggle_data_chain[p_joint_idx].force;
	jiggle_data_chain.write[p_joint_idx].dynamic_position += operation_bone_trans.get_origin() - jiggle_data_chain[p_joint_idx].last_position;
	jiggle_data_chain.write[p_joint_idx].last_position = operation_bone_trans.get_origin();

	// Rotate the bone using the dynamic position!
	operation_bone_trans = operation_bone_trans.looking_at(jiggle_data_chain[p_joint_idx].dynamic_position);

	operation_bone->set_global_rotation(operation_bone_trans.get_rotation() - operation_bone->get_bone_angle());
}

void SkeletonModification2DJiggle::_update_jiggle_joint_data() {
	for (int i = 0; i < jiggle_data_chain.size(); i++) {
		if (!jiggle_data_chain[i].override_defaults) {
			set_joint_stiffness(i, stiffness);
			set_joint_mass(i, mass);
			set_joint_damping(i, damping);
			set_joint_use_gravity(i, use_gravity);
			set_joint_gravity(i, gravity);
		}
	}
}

void SkeletonModification2DJiggle::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	target_node_cache = Variant();
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
	if (is_inside_tree()) {
		set_physics_process_internal(use_colliders);
	}
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
int SkeletonModification2DJiggle::get_joint_count() {
	return jiggle_data_chain.size();
}

void SkeletonModification2DJiggle::set_joint_count(int p_length) {
	ERR_FAIL_COND(p_length < 0);
	jiggle_data_chain.resize(p_length);
	notify_property_list_changed();
}

void SkeletonModification2DJiggle::set_joint_bone_node(int p_joint_idx, const NodePath &p_target_node) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, jiggle_data_chain.size(), "Jiggle joint out of range!");
	jiggle_data_chain.write[p_joint_idx].bone_node = p_target_node;
	jiggle_data_chain.write[p_joint_idx].bone_node_cache = Variant();
}

NodePath SkeletonModification2DJiggle::get_joint_bone_node(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, jiggle_data_chain.size(), NodePath(), "Jiggle joint out of range!");
	return jiggle_data_chain[p_joint_idx].bone_node;
}

void SkeletonModification2DJiggle::set_joint_override(int p_joint_idx, bool p_override) {
	ERR_FAIL_INDEX(p_joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[p_joint_idx].override_defaults = p_override;
	_update_jiggle_joint_data();
	notify_property_list_changed();
}

bool SkeletonModification2DJiggle::get_joint_override(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, jiggle_data_chain.size(), false);
	return jiggle_data_chain[p_joint_idx].override_defaults;
}

void SkeletonModification2DJiggle::set_joint_stiffness(int p_joint_idx, float p_stiffness) {
	ERR_FAIL_COND_MSG(p_stiffness < 0, "Stiffness cannot be set to a negative value!");
	ERR_FAIL_INDEX(p_joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[p_joint_idx].stiffness = p_stiffness;
}

float SkeletonModification2DJiggle::get_joint_stiffness(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, jiggle_data_chain.size(), -1);
	return jiggle_data_chain[p_joint_idx].stiffness;
}

void SkeletonModification2DJiggle::set_joint_mass(int p_joint_idx, float p_mass) {
	ERR_FAIL_COND_MSG(p_mass < 0, "Mass cannot be set to a negative value!");
	ERR_FAIL_INDEX(p_joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[p_joint_idx].mass = p_mass;
}

float SkeletonModification2DJiggle::get_joint_mass(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, jiggle_data_chain.size(), -1);
	return jiggle_data_chain[p_joint_idx].mass;
}

void SkeletonModification2DJiggle::set_joint_damping(int p_joint_idx, float p_damping) {
	ERR_FAIL_COND_MSG(p_damping < 0, "Damping cannot be set to a negative value!");
	ERR_FAIL_INDEX(p_joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[p_joint_idx].damping = p_damping;
}

float SkeletonModification2DJiggle::get_joint_damping(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, jiggle_data_chain.size(), -1);
	return jiggle_data_chain[p_joint_idx].damping;
}

void SkeletonModification2DJiggle::set_joint_use_gravity(int p_joint_idx, bool p_use_gravity) {
	ERR_FAIL_INDEX(p_joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[p_joint_idx].use_gravity = p_use_gravity;
	notify_property_list_changed();
}

bool SkeletonModification2DJiggle::get_joint_use_gravity(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, jiggle_data_chain.size(), false);
	return jiggle_data_chain[p_joint_idx].use_gravity;
}

void SkeletonModification2DJiggle::set_joint_gravity(int p_joint_idx, Vector2 p_gravity) {
	ERR_FAIL_INDEX(p_joint_idx, jiggle_data_chain.size());
	jiggle_data_chain.write[p_joint_idx].gravity = p_gravity;
}

Vector2 SkeletonModification2DJiggle::get_joint_gravity(int p_joint_idx) const {
	ERR_FAIL_INDEX_V(p_joint_idx, jiggle_data_chain.size(), Vector2(0, 0));
	return jiggle_data_chain[p_joint_idx].gravity;
}

void SkeletonModification2DJiggle::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_node"), &SkeletonModification2DJiggle::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification2DJiggle::get_target_node);

	ClassDB::bind_method(D_METHOD("set_joint_count", "length"), &SkeletonModification2DJiggle::set_joint_count);
	ClassDB::bind_method(D_METHOD("get_joint_count"), &SkeletonModification2DJiggle::get_joint_count);

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
	ClassDB::bind_method(D_METHOD("set_joint_bone_node", "joint_idx", "bone2d_node"), &SkeletonModification2DJiggle::set_joint_bone_node);
	ClassDB::bind_method(D_METHOD("get_joint_bone_node", "joint_idx"), &SkeletonModification2DJiggle::get_joint_bone_node);
	ClassDB::bind_method(D_METHOD("set_joint_override", "joint_idx", "override"), &SkeletonModification2DJiggle::set_joint_override);
	ClassDB::bind_method(D_METHOD("get_joint_override", "joint_idx"), &SkeletonModification2DJiggle::get_joint_override);
	ClassDB::bind_method(D_METHOD("set_joint_stiffness", "joint_idx", "stiffness"), &SkeletonModification2DJiggle::set_joint_stiffness);
	ClassDB::bind_method(D_METHOD("get_joint_stiffness", "joint_idx"), &SkeletonModification2DJiggle::get_joint_stiffness);
	ClassDB::bind_method(D_METHOD("set_joint_mass", "joint_idx", "mass"), &SkeletonModification2DJiggle::set_joint_mass);
	ClassDB::bind_method(D_METHOD("get_joint_mass", "joint_idx"), &SkeletonModification2DJiggle::get_joint_mass);
	ClassDB::bind_method(D_METHOD("set_joint_damping", "joint_idx", "damping"), &SkeletonModification2DJiggle::set_joint_damping);
	ClassDB::bind_method(D_METHOD("get_joint_damping", "joint_idx"), &SkeletonModification2DJiggle::get_joint_damping);
	ClassDB::bind_method(D_METHOD("set_joint_use_gravity", "joint_idx", "use_gravity"), &SkeletonModification2DJiggle::set_joint_use_gravity);
	ClassDB::bind_method(D_METHOD("get_joint_use_gravity", "joint_idx"), &SkeletonModification2DJiggle::get_joint_use_gravity);
	ClassDB::bind_method(D_METHOD("set_joint_gravity", "joint_idx", "gravity"), &SkeletonModification2DJiggle::set_joint_gravity);
	ClassDB::bind_method(D_METHOD("get_joint_gravity", "joint_idx"), &SkeletonModification2DJiggle::get_joint_gravity);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "CanvasItem"), "set_target_node", "get_target_node");
	ADD_ARRAY_COUNT("Jiggle joint chain length", "joint_count", "set_joint_count", "get_joint_count", "joint_");
	ADD_GROUP("Default Joint Settings", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stiffness"), "set_stiffness", "get_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping", PROPERTY_HINT_RANGE, "0, 1, 0.01"), "set_damping", "get_damping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_gravity"), "set_use_gravity", "get_use_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "gravity"), "set_gravity", "get_gravity");
	ADD_GROUP("", "");
}

SkeletonModification2DJiggle::SkeletonModification2DJiggle() {
	jiggle_data_chain = Vector<Jiggle_Joint_Data2D>();
	stiffness = 3;
	mass = 0.75;
	damping = 0.75;
	use_gravity = false;
	gravity = Vector2(0, 6.0);
}

SkeletonModification2DJiggle::~SkeletonModification2DJiggle() {
}

void SkeletonModification2DJiggle::execute(real_t p_delta) {
	SkeletonModification2D::execute(p_delta);

	for (int i = 0; i < jiggle_data_chain.size(); i++) {
		if (!_cache_bone(jiggle_data_chain[i].bone_node_cache, jiggle_data_chain[i].bone_node)) {
			WARN_PRINT_ONCE("2DJiggle: unable to lookup joint");
			return;
		}
		jiggle_data_chain.write[i].dynamic_position = get_target_position(jiggle_data_chain[i].bone_node_cache);
	}
	if (!_cache_node(target_node_cache, target_node)) {
		WARN_PRINT_ONCE("2DJiggle: unable to lookup target");
		return;
	}
	Vector2 target_position = get_target_position(target_node_cache);
	for (int i = 0; i < jiggle_data_chain.size(); i++) {
		_execute_jiggle_joint(i, target_position, p_delta);
	}
}

void SkeletonModification2DJiggle::_execute_joint_collision(Ref<World2D> world_2d, int p_joint_idx) {
	// Collision detection/response
	PhysicsDirectSpaceState2D *space_state = PhysicsServer2D::get_singleton()->space_get_direct_state(world_2d->get_space());
	PhysicsDirectSpaceState2D::RayResult ray_result;

	PhysicsDirectSpaceState2D::RayParameters ray_params;
	ray_params.from = jiggle_data_chain[p_joint_idx].last_noncollision_position;
	ray_params.to = jiggle_data_chain[p_joint_idx].dynamic_position;
	ray_params.collision_mask = collision_mask;

	// Add exception support?
	bool ray_hit = space_state->intersect_ray(ray_params, ray_result);

	if (ray_hit) {
		jiggle_data_chain.write[p_joint_idx].dynamic_position = jiggle_data_chain[p_joint_idx].last_noncollision_position;
		jiggle_data_chain.write[p_joint_idx].acceleration = Vector2(0, 0);
		jiggle_data_chain.write[p_joint_idx].velocity = Vector2(0, 0);
	} else {
		jiggle_data_chain.write[p_joint_idx].last_noncollision_position = jiggle_data_chain[p_joint_idx].dynamic_position;
	}
}

void SkeletonModification2DJiggle::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
			set_physics_process_internal(use_colliders);
			break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS:
			if (use_colliders) {
				Skeleton2D *skeleton = get_skeleton();
				ERR_FAIL_COND(skeleton == nullptr);
				Ref<World2D> world_2d = skeleton->get_world_2d();
				ERR_FAIL_COND(world_2d.is_null());
				for (int i = 0; i < jiggle_data_chain.size(); i++) {
					_execute_joint_collision(world_2d, i);
				}
			}
			break;
	}
}

PackedStringArray SkeletonModification2DJiggle::get_configuration_warnings() const {
	PackedStringArray ret = SkeletonModification2D::get_configuration_warnings();
	if (!get_skeleton()) {
		return ret;
	}
	if (!_cache_node(target_node_cache, target_node)) {
		ret.append(vformat("Target node %s was not found.", (String)target_node));
	}
	for (int i = 0; i < jiggle_data_chain.size(); i++) {
		if (!_cache_bone(jiggle_data_chain[i].bone_node_cache, jiggle_data_chain[i].bone_node)) {
			ret.append(vformat("Joint %d bone %s was not found.", i, jiggle_data_chain[i].bone_node));
		}
	}
	return ret;
}
