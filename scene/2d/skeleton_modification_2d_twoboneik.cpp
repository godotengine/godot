/*************************************************************************/
/*  skeleton_modification_2d_twoboneik.cpp                               */
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

#include "skeleton_modification_2d_twoboneik.h"
#include "scene/2d/skeleton_2d.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif // TOOLS_ENABLED

void SkeletonModification2DTwoBoneIK::draw_editor_gizmo() {
	Skeleton2D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	Bone2D *operation_bone_one = _cache_bone(joint_one_bone_node_cache, joint_one_bone_node);
	if (!operation_bone_one) {
		return;
	}
	skeleton->draw_set_transform(
			skeleton->to_local(operation_bone_one->get_global_position()),
			operation_bone_one->get_global_rotation() - skeleton->get_global_rotation());

	Color bone_ik_color = Color(1.0, 0.65, 0.0, 0.4);
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		bone_ik_color = EditorSettings::get_singleton()->get("editors/2d/bone_ik_color");
	}
#endif // TOOLS_ENABLED

	if (flip_bend_direction) {
		float angle = -(Math_PI * 0.5) + operation_bone_one->get_bone_angle();
		skeleton->draw_line(Vector2(0, 0), Vector2(Math::cos(angle), sin(angle)) * (operation_bone_one->get_length() * 0.5), bone_ik_color, 2.0);
	} else {
		float angle = (Math_PI * 0.5) + operation_bone_one->get_bone_angle();
		skeleton->draw_line(Vector2(0, 0), Vector2(Math::cos(angle), sin(angle)) * (operation_bone_one->get_length() * 0.5), bone_ik_color, 2.0);
	}

	if (target_maximum_distance != 0.0 || target_minimum_distance != 0.0) {
		Vector2 target_direction = Vector2(0, 1);
		_cache_node(target_node_cache, target_node);
		Node2D *target = Object::cast_to<Node2D>((Object *)target_node_cache);
		if (target) {
			skeleton->draw_set_transform(Vector2(0, 0), 0.0);
			target_direction = operation_bone_one->get_global_position().direction_to(target->get_global_position());
		}

		skeleton->draw_circle(target_direction * target_minimum_distance, 8, bone_ik_color);
		skeleton->draw_circle(target_direction * target_maximum_distance, 8, bone_ik_color);
		skeleton->draw_line(target_direction * target_minimum_distance, target_direction * target_maximum_distance, bone_ik_color, 2.0);
	}
}

void SkeletonModification2DTwoBoneIK::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	target_node_cache = Variant();
}

NodePath SkeletonModification2DTwoBoneIK::get_target_node() const {
	return target_node;
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
}

bool SkeletonModification2DTwoBoneIK::get_flip_bend_direction() const {
	return flip_bend_direction;
}

void SkeletonModification2DTwoBoneIK::set_joint_one_bone_node(const NodePath &p_target_node) {
	joint_one_bone_node = p_target_node;
	joint_one_bone_node_cache = Variant();
}

NodePath SkeletonModification2DTwoBoneIK::get_joint_one_bone_node() const {
	return joint_one_bone_node;
}

void SkeletonModification2DTwoBoneIK::set_joint_two_bone_node(const NodePath &p_target_node) {
	joint_two_bone_node = p_target_node;
	joint_two_bone_node_cache = Variant();
}

NodePath SkeletonModification2DTwoBoneIK::get_joint_two_bone_node() const {
	return joint_two_bone_node;
}

void SkeletonModification2DTwoBoneIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification2DTwoBoneIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification2DTwoBoneIK::get_target_node);

	ClassDB::bind_method(D_METHOD("set_target_minimum_distance", "minimum_distance"), &SkeletonModification2DTwoBoneIK::set_target_minimum_distance);
	ClassDB::bind_method(D_METHOD("get_target_minimum_distance"), &SkeletonModification2DTwoBoneIK::get_target_minimum_distance);
	ClassDB::bind_method(D_METHOD("set_target_maximum_distance", "maximum_distance"), &SkeletonModification2DTwoBoneIK::set_target_maximum_distance);
	ClassDB::bind_method(D_METHOD("get_target_maximum_distance"), &SkeletonModification2DTwoBoneIK::get_target_maximum_distance);
	ClassDB::bind_method(D_METHOD("set_flip_bend_direction", "flip_direction"), &SkeletonModification2DTwoBoneIK::set_flip_bend_direction);
	ClassDB::bind_method(D_METHOD("get_flip_bend_direction"), &SkeletonModification2DTwoBoneIK::get_flip_bend_direction);

	ClassDB::bind_method(D_METHOD("set_joint_one_bone_node", "bone_node"), &SkeletonModification2DTwoBoneIK::set_joint_one_bone_node);
	ClassDB::bind_method(D_METHOD("get_joint_one_bone_node"), &SkeletonModification2DTwoBoneIK::get_joint_one_bone_node);
	ClassDB::bind_method(D_METHOD("set_joint_two_bone_node", "bone_node"), &SkeletonModification2DTwoBoneIK::set_joint_two_bone_node);
	ClassDB::bind_method(D_METHOD("get_joint_two_bone_node"), &SkeletonModification2DTwoBoneIK::get_joint_two_bone_node);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "CanvasItem"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "target_minimum_distance", PROPERTY_HINT_RANGE, "0,100000000,0.01,suffix:m"), "set_target_minimum_distance", "get_target_minimum_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "target_maximum_distance", PROPERTY_HINT_NONE, "0,100000000,0.01,suffix:m"), "set_target_maximum_distance", "get_target_maximum_distance");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_bend_direction", PROPERTY_HINT_NONE, ""), "set_flip_bend_direction", "get_flip_bend_direction");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "joint_one_bone_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Bone2D", PROPERTY_USAGE_DEFAULT), "set_joint_one_bone_node", "get_joint_one_bone_node");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "joint_two_bone_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Bone2D", PROPERTY_USAGE_DEFAULT), "set_joint_two_bone_node", "get_joint_two_bone_node");
}

SkeletonModification2DTwoBoneIK::SkeletonModification2DTwoBoneIK() {
}

SkeletonModification2DTwoBoneIK::~SkeletonModification2DTwoBoneIK() {
}

void SkeletonModification2DTwoBoneIK::execute(real_t p_delta) {
	SkeletonModification2D::execute(p_delta);

	Bone2D *joint_one_bone = _cache_bone(joint_one_bone_node_cache, joint_one_bone_node);
	Bone2D *joint_two_bone = _cache_bone(joint_two_bone_node_cache, joint_two_bone_node);
	if (!_cache_node(target_node_cache, target_node) ||
			!joint_one_bone || !joint_two_bone) {
		WARN_PRINT_ONCE("2DTwoBoneIK unable to get nodes");
		return;
	}
	Vector2 target_position = get_target_position(target_node_cache);

	// Adopted from the links below:
	// http://theorangeduck.com/page/simple-two-joint
	// https://www.alanzucconi.com/2018/05/02/ik-2d-2/
	// With modifications by TwistedTwigleg
	Vector2 target_difference = target_position - joint_one_bone->get_global_position();
	float joint_one_to_target = target_difference.length();
	float angle_atan = target_difference.angle();

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

		if (isfinite(angle_0) && isfinite(angle_1)) {
			joint_one_bone->set_global_rotation(angle_atan - angle_0 - joint_one_bone->get_bone_angle());
			joint_two_bone->set_rotation(-Math_PI - angle_1 - joint_two_bone->get_bone_angle() + joint_one_bone->get_bone_angle());
		}
	} else {
		joint_one_bone->set_global_rotation(angle_atan - joint_one_bone->get_bone_angle());
		joint_two_bone->set_global_rotation(angle_atan - joint_two_bone->get_bone_angle());
	}
}

PackedStringArray SkeletonModification2DTwoBoneIK::get_configuration_warnings() const {
	PackedStringArray ret = SkeletonModification2D::get_configuration_warnings();
	if (!get_skeleton()) {
		return ret;
	}
	if (!_cache_node(target_node_cache, target_node)) {
		ret.append(vformat("Target node %s was not found.", (String)target_node));
	}
	if (!_cache_bone(joint_one_bone_node_cache, joint_one_bone_node)) {
		ret.append(vformat("Joint one bone %s was not found.", joint_one_bone_node));
	}
	if (!_cache_bone(joint_two_bone_node_cache, joint_two_bone_node)) {
		ret.append(vformat("Joint two bone %s was not found.", joint_two_bone_node));
	}
	return ret;
}
