/**************************************************************************/
/*  jiggle_modifier_2d.cpp                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "jiggle_modifier_2d.h"

#include "core/object/class_db.h"
#include "scene/2d/skeleton_2d.h"

static void _build_jiggle_chain(Skeleton2D *p_skeleton, int p_root_bone, int p_tip_bone, LocalVector<int> &r_chain) {
	r_chain.clear();
	for (int bone = p_tip_bone; bone >= 0; bone = p_skeleton->get_bone_parent(bone)) {
		r_chain.push_back(bone);
		if (bone == p_root_bone) {
			break;
		}
	}
	r_chain.reverse();
}

JiggleModifier2D::JiggleState *JiggleModifier2D::_get_state(int p_bone) {
	for (uint32_t i = 0; i < states.size(); i++) {
		if (states[i].bone == p_bone) {
			return &states[i];
		}
	}
	JiggleState state;
	state.bone = p_bone;
	states.push_back(state);
	return &states[states.size() - 1];
}

void JiggleModifier2D::_reset_jiggle_state() {
	states.clear();
}

PackedStringArray JiggleModifier2D::get_configuration_warnings() const {
	PackedStringArray warnings = SkeletonModifier2D::get_configuration_warnings();
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton) {
		return warnings;
	}
	if (tip_bone < 0 || tip_bone >= skeleton->get_bone_count()) {
		warnings.push_back(RTR("Tip bone index is out of range."));
	}
	if (root_bone >= skeleton->get_bone_count()) {
		warnings.push_back(RTR("Root bone index is out of range."));
	}
	return warnings;
}

void JiggleModifier2D::_skeleton_changed(Skeleton2D *p_old, Skeleton2D *p_new) {
	SkeletonModifier2D::_skeleton_changed(p_old, p_new);
	_reset_jiggle_state();
}

void JiggleModifier2D::_process_modification(double p_delta) {
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton || tip_bone < 0 || tip_bone >= skeleton->get_bone_count()) {
		return;
	}
	LocalVector<int> chain;
	_build_jiggle_chain(skeleton, root_bone, tip_bone, chain);
	if (chain.is_empty() || (root_bone >= 0 && chain[0] != root_bone)) {
		return;
	}

	const real_t delta = CLAMP((real_t)p_delta, (real_t)0.0, (real_t)0.1);
	const real_t inv_mass = 1.0 / MAX(mass, (real_t)CMP_EPSILON);
	const real_t damping_factor = MAX((real_t)0.0, (real_t)1.0 - damping * delta);
	const Vector2 gravity_force = use_gravity ? gravity : Vector2();

	for (uint32_t i = 0; i < chain.size(); i++) {
		const int bone = chain[i];
		Bone2D *bone_node = skeleton->get_bone(bone);
		if (!bone_node) {
			continue;
		}
		const Transform2D bone_global_pose = skeleton->get_bone_global_pose(bone);
		const Vector2 base_position = bone_global_pose.get_origin();
		Vector2 rest_end_position;
		if (i + 1 < chain.size()) {
			rest_end_position = skeleton->get_bone_global_pose(chain[i + 1]).get_origin();
		} else {
			rest_end_position = base_position + bone_global_pose.basis_xform(Vector2(bone_node->get_length(), 0));
		}
		const real_t length = base_position.distance_to(rest_end_position);
		if (Math::is_zero_approx(length)) {
			continue;
		}
		JiggleState *state = _get_state(bone);
		if (!state->initialized) {
			state->position = rest_end_position;
			state->velocity = Vector2();
			state->initialized = true;
		}
		const Vector2 spring_force = (rest_end_position - state->position) * stiffness;
		state->velocity += (spring_force * inv_mass + gravity_force) * delta;
		state->velocity *= damping_factor;
		state->position += state->velocity * delta;
		Vector2 bone_to_state = state->position - base_position;
		if (bone_to_state.is_zero_approx()) {
			bone_to_state = rest_end_position - base_position;
		}
		bone_to_state.normalize();
		state->position = base_position + bone_to_state * length;
		Vector2 current_vector = rest_end_position - base_position;
		if (current_vector.is_zero_approx()) {
			continue;
		}
		Transform2D pose = skeleton->get_bone_pose(bone);
		pose.set_rotation(pose.get_rotation() + current_vector.angle_to(bone_to_state));
		skeleton->set_bone_pose(bone, pose);
	}
}

void JiggleModifier2D::set_root_bone(int p_bone) {
	root_bone = p_bone;
	_reset_jiggle_state();
	update_configuration_warnings();
}
int JiggleModifier2D::get_root_bone() const {
	return root_bone;
}
void JiggleModifier2D::set_tip_bone(int p_bone) {
	tip_bone = p_bone;
	_reset_jiggle_state();
	update_configuration_warnings();
}
int JiggleModifier2D::get_tip_bone() const {
	return tip_bone;
}
void JiggleModifier2D::set_stiffness(real_t p_stiffness) {
	stiffness = MAX((real_t)0.0, p_stiffness);
}
real_t JiggleModifier2D::get_stiffness() const {
	return stiffness;
}
void JiggleModifier2D::set_damping(real_t p_damping) {
	damping = MAX((real_t)0.0, p_damping);
}
real_t JiggleModifier2D::get_damping() const {
	return damping;
}
void JiggleModifier2D::set_mass(real_t p_mass) {
	mass = MAX((real_t)CMP_EPSILON, p_mass);
}
real_t JiggleModifier2D::get_mass() const {
	return mass;
}
void JiggleModifier2D::set_use_gravity(bool p_use_gravity) {
	use_gravity = p_use_gravity;
}
bool JiggleModifier2D::is_using_gravity() const {
	return use_gravity;
}
void JiggleModifier2D::set_gravity(const Vector2 &p_gravity) {
	gravity = p_gravity;
}
Vector2 JiggleModifier2D::get_gravity() const {
	return gravity;
}
void JiggleModifier2D::reset() {
	_reset_jiggle_state();
}

void JiggleModifier2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_bone", "root_bone"), &JiggleModifier2D::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone"), &JiggleModifier2D::get_root_bone);
	ClassDB::bind_method(D_METHOD("set_tip_bone", "tip_bone"), &JiggleModifier2D::set_tip_bone);
	ClassDB::bind_method(D_METHOD("get_tip_bone"), &JiggleModifier2D::get_tip_bone);
	ClassDB::bind_method(D_METHOD("set_stiffness", "stiffness"), &JiggleModifier2D::set_stiffness);
	ClassDB::bind_method(D_METHOD("get_stiffness"), &JiggleModifier2D::get_stiffness);
	ClassDB::bind_method(D_METHOD("set_damping", "damping"), &JiggleModifier2D::set_damping);
	ClassDB::bind_method(D_METHOD("get_damping"), &JiggleModifier2D::get_damping);
	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &JiggleModifier2D::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &JiggleModifier2D::get_mass);
	ClassDB::bind_method(D_METHOD("set_use_gravity", "use_gravity"), &JiggleModifier2D::set_use_gravity);
	ClassDB::bind_method(D_METHOD("is_using_gravity"), &JiggleModifier2D::is_using_gravity);
	ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &JiggleModifier2D::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity"), &JiggleModifier2D::get_gravity);
	ClassDB::bind_method(D_METHOD("reset"), &JiggleModifier2D::reset);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "root_bone", PROPERTY_HINT_RANGE, "-1,1024,1,or_greater"), "set_root_bone", "get_root_bone");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tip_bone", PROPERTY_HINT_RANGE, "-1,1024,1,or_greater"), "set_tip_bone", "get_tip_bone");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stiffness", PROPERTY_HINT_RANGE, "0,128,0.01,or_greater"), "set_stiffness", "get_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping", PROPERTY_HINT_RANGE, "0,64,0.01,or_greater"), "set_damping", "get_damping");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_gravity"), "set_use_gravity", "is_using_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "gravity", PROPERTY_HINT_NONE, "suffix:px/s^2"), "set_gravity", "get_gravity");
}
