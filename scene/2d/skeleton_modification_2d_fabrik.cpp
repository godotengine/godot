/*************************************************************************/
/*  skeleton_modification_2d_fabrik.cpp                                  */
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

#include "skeleton_modification_2d_fabrik.h"
#include "scene/2d/skeleton_2d.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif // TOOLS_ENABLED

bool SkeletonModification2DFABRIK::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("joint_")) {
		int which = path.get_slicec('/', 0).substr(6).to_int();
		String what = path.get_slicec('/', 1);
		ERR_FAIL_INDEX_V(which, fabrik_data_chain.size(), false);

		if (what == "bone_node") {
			set_joint_bone_node(which, p_value);
		} else if (what == "magnet_position") {
			set_joint_magnet_position(which, p_value);
		} else if (what == "use_target_rotation") {
			set_joint_use_target_rotation(which, p_value);
		}
	}

	return true;
}

bool SkeletonModification2DFABRIK::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("joint_")) {
		int which = path.get_slicec('/', 0).substr(6).to_int();
		String what = path.get_slicec('/', 1);
		ERR_FAIL_INDEX_V(which, fabrik_data_chain.size(), false);

		if (what == "bone_node") {
			r_ret = get_joint_bone_node(which);
		} else if (what == "magnet_position") {
			r_ret = get_joint_magnet_position(which);
		} else if (what == "use_target_rotation") {
			r_ret = get_joint_use_target_rotation(which);
		}
		return true;
	}
	return true;
}

void SkeletonModification2DFABRIK::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		String base_string = "joint_" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::NODE_PATH, base_string + "bone_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "CanvasItem", PROPERTY_USAGE_DEFAULT));

		if (i > 0) {
			p_list->push_back(PropertyInfo(Variant::VECTOR2, base_string + "magnet_position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
		if (i == fabrik_data_chain.size() - 1) {
			p_list->push_back(PropertyInfo(Variant::BOOL, base_string + "use_target_rotation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT));
		}
	}
}

void SkeletonModification2DFABRIK::chain_backwards() {
	int final_joint_index = fabrik_data_chain.size() - 1;
	Bone2D *final_bone2d_node = Object::cast_to<Bone2D>(ObjectDB::get_instance(fabrik_data_chain[final_joint_index].bone_node_cache));
	Transform2D final_bone2d_trans = fabrik_data_chain[final_joint_index].global_transform;

	// Apply magnet position
	if (final_joint_index != 0) {
		final_bone2d_trans.set_origin(final_bone2d_trans.get_origin() + fabrik_data_chain[final_joint_index].magnet_position);
	}

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
	fabrik_data_chain.write[final_joint_index].global_transform = final_bone2d_trans;

	int i = final_joint_index;
	while (i >= 1) {
		Transform2D previous_pose = fabrik_data_chain[i].global_transform;
		i -= 1;
		Bone2D *current_bone2d_node = Object::cast_to<Bone2D>((Object *)fabrik_data_chain[i].bone_node_cache);
		ERR_FAIL_COND(current_bone2d_node == nullptr);
		Transform2D current_pose = fabrik_data_chain[i].global_transform;

		// Apply magnet position
		if (i != 0) {
			current_pose.set_origin(current_pose.get_origin() + fabrik_data_chain[i].magnet_position);
		}

		float current_bone2d_node_length = current_bone2d_node->get_length() * MIN(current_bone2d_node->get_global_scale().x, current_bone2d_node->get_global_scale().y);
		float length = current_bone2d_node_length / (current_pose.get_origin().distance_to(previous_pose.get_origin()));
		Vector2 finish_position = previous_pose.get_origin().lerp(current_pose.get_origin(), length);
		current_pose.set_origin(finish_position);

		// Save the transform
		fabrik_data_chain.write[i].global_transform = current_pose;
	}
}

void SkeletonModification2DFABRIK::chain_forwards() {
	Transform2D origin_bone2d_trans = fabrik_data_chain[0].global_transform;
	origin_bone2d_trans.set_origin(origin_global_pose.get_origin());
	// Save the position
	fabrik_data_chain.write[0].global_transform = origin_bone2d_trans;

	for (int i = 0; i < fabrik_data_chain.size() - 1; i++) {
		Bone2D *current_bone2d_node = Object::cast_to<Bone2D>(ObjectDB::get_instance(fabrik_data_chain[i].bone_node_cache));
		Transform2D current_pose = fabrik_data_chain[i].global_transform;
		Transform2D next_pose = fabrik_data_chain[i + 1].global_transform;

		float current_bone2d_node_length = current_bone2d_node->get_length() * MIN(current_bone2d_node->get_global_scale().x, current_bone2d_node->get_global_scale().y);
		float length = current_bone2d_node_length / (next_pose.get_origin().distance_to(current_pose.get_origin()));
		Vector2 finish_position = current_pose.get_origin().lerp(next_pose.get_origin(), length);
		current_pose.set_origin(finish_position);

		// Apply to the bone
		fabrik_data_chain.write[i + 1].global_transform = current_pose;
	}
}

void SkeletonModification2DFABRIK::set_target_node(const NodePath &p_target_node) {
	target_node = p_target_node;
	target_node_cache = Variant();
}

NodePath SkeletonModification2DFABRIK::get_target_node() const {
	return target_node;
}

void SkeletonModification2DFABRIK::set_joint_count(int p_fabrik_chain_length) {
	fabrik_data_chain.resize(p_fabrik_chain_length);
	notify_property_list_changed();
}

int SkeletonModification2DFABRIK::get_joint_count() {
	return fabrik_data_chain.size();
}

void SkeletonModification2DFABRIK::set_joint_bone_node(int p_joint_idx, const NodePath &p_target_node) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].bone_node = p_target_node;
	fabrik_data_chain.write[p_joint_idx].bone_node_cache = Variant();
}

NodePath SkeletonModification2DFABRIK::get_joint_bone_node(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), NodePath(), "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].bone_node;
}

void SkeletonModification2DFABRIK::set_joint_magnet_position(int p_joint_idx, Vector2 p_magnet_position) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].magnet_position = p_magnet_position;
}

Vector2 SkeletonModification2DFABRIK::get_joint_magnet_position(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), Vector2(), "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].magnet_position;
}

void SkeletonModification2DFABRIK::set_joint_use_target_rotation(int p_joint_idx, bool p_use_target_rotation) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, fabrik_data_chain.size(), "FABRIK joint out of range!");
	fabrik_data_chain.write[p_joint_idx].use_target_rotation = p_use_target_rotation;
}

bool SkeletonModification2DFABRIK::get_joint_use_target_rotation(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, fabrik_data_chain.size(), false, "FABRIK joint out of range!");
	return fabrik_data_chain[p_joint_idx].use_target_rotation;
}

void SkeletonModification2DFABRIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_node", "target_nodepath"), &SkeletonModification2DFABRIK::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonModification2DFABRIK::get_target_node);

	ClassDB::bind_method(D_METHOD("set_joint_count", "fabrik_chain_length"), &SkeletonModification2DFABRIK::set_joint_count);
	ClassDB::bind_method(D_METHOD("get_joint_count"), &SkeletonModification2DFABRIK::get_joint_count);

	ClassDB::bind_method(D_METHOD("set_joint_bone_node", "joint_idx", "bone2d_nodepath"), &SkeletonModification2DFABRIK::set_joint_bone_node);
	ClassDB::bind_method(D_METHOD("get_joint_bone_node", "joint_idx"), &SkeletonModification2DFABRIK::get_joint_bone_node);
	ClassDB::bind_method(D_METHOD("set_joint_magnet_position", "joint_idx", "magnet_position"), &SkeletonModification2DFABRIK::set_joint_magnet_position);
	ClassDB::bind_method(D_METHOD("get_joint_magnet_position", "joint_idx"), &SkeletonModification2DFABRIK::get_joint_magnet_position);
	ClassDB::bind_method(D_METHOD("set_joint_use_target_rotation", "joint_idx", "use_target_rotation"), &SkeletonModification2DFABRIK::set_joint_use_target_rotation);
	ClassDB::bind_method(D_METHOD("get_joint_use_target_rotation", "joint_idx"), &SkeletonModification2DFABRIK::get_joint_use_target_rotation);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "CanvasItem"), "set_target_node", "get_target_node");
	ADD_ARRAY_COUNT("FABRIK Joint Chain", "joint_count", "set_joint_count", "get_joint_count", "joint_");
}

SkeletonModification2DFABRIK::SkeletonModification2DFABRIK() {
}

SkeletonModification2DFABRIK::~SkeletonModification2DFABRIK() {
}

void SkeletonModification2DFABRIK::execute(real_t delta) {
	SkeletonModification2D::execute(delta);

	if (!_cache_node(target_node_cache, target_node)) {
		WARN_PRINT_ONCE("2DFABRIK was unable get target node");
		return;
	}
	ERR_FAIL_COND(fabrik_data_chain.size() == 0);
	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		if (!_cache_node(fabrik_data_chain[i].bone_node_cache, fabrik_data_chain[i].bone_node)) {
			WARN_PRINT_ONCE("2DFABRIK was unable get chain node");
			return;
		}
		fabrik_data_chain.write[i].global_transform = get_target_transform(fabrik_data_chain[i].bone_node_cache);
	}
	target_global_pose = get_target_transform(target_node_cache);

	origin_global_pose = get_target_transform(fabrik_data_chain[0].bone_node_cache);

	Bone2D *final_bone2d_node = Object::cast_to<Bone2D>((Object *)(fabrik_data_chain[fabrik_data_chain.size() - 1].bone_node_cache));
	float final_bone2d_angle = final_bone2d_node->get_global_rotation();
	if (fabrik_data_chain[fabrik_data_chain.size() - 1].use_target_rotation) {
		final_bone2d_angle = target_global_pose.get_rotation();
	}
	Vector2 final_bone2d_direction = Vector2(Math::cos(final_bone2d_angle), Math::sin(final_bone2d_angle));
	float final_bone2d_length = final_bone2d_node->get_length() * MIN(final_bone2d_node->get_global_scale().x, final_bone2d_node->get_global_scale().y);
	float target_distance = (final_bone2d_node->get_global_position() + (final_bone2d_direction * final_bone2d_length)).distance_to(target_global_pose.get_origin());
	chain_iterations = 0;

	while (target_distance > chain_tolarance) {
		chain_backwards();
		chain_forwards();

		final_bone2d_angle = final_bone2d_node->get_global_rotation();
		if (fabrik_data_chain[fabrik_data_chain.size() - 1].use_target_rotation) {
			final_bone2d_angle = target_global_pose.get_rotation();
		}
		final_bone2d_direction = Vector2(Math::cos(final_bone2d_angle), Math::sin(final_bone2d_angle));
		target_distance = (final_bone2d_node->get_global_position() + (final_bone2d_direction * final_bone2d_length)).distance_to(target_global_pose.get_origin());

		chain_iterations += 1;
		if (chain_iterations >= chain_max_iterations) {
			break;
		}
	}

	// Apply all of the saved transforms to the Bone2D nodes
	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		Bone2D *joint_bone2d_node = Object::cast_to<Bone2D>((Object *)(fabrik_data_chain[i].bone_node_cache));
		if (!joint_bone2d_node) {
			ERR_PRINT_ONCE("FABRIK Joint " + itos(i) + " does not have a Bone2D node set!");
			continue;
		}
		Transform2D chain_trans = fabrik_data_chain[i].global_transform;

		// Apply rotation
		if (i + 1 < fabrik_data_chain.size()) {
			chain_trans = chain_trans.looking_at(fabrik_data_chain[i + 1].global_transform.get_origin());
		} else {
			if (fabrik_data_chain[i].use_target_rotation) {
				chain_trans.set_rotation(target_global_pose.get_rotation());
			} else {
				chain_trans = chain_trans.looking_at(target_global_pose.get_origin());
			}
		}
		// Adjust for the bone angle and apply to the bone.
		joint_bone2d_node->set_global_rotation(chain_trans.get_rotation() - joint_bone2d_node->get_bone_angle());
	}
}

PackedStringArray SkeletonModification2DFABRIK::get_configuration_warnings() const {
	PackedStringArray ret = SkeletonModification2D::get_configuration_warnings();
	if (!get_skeleton()) {
		return ret;
	}
	if (!_cache_node(target_node_cache, target_node)) {
		ret.append(vformat("Target node %s was not found.", (String)target_node));
	}
	for (int i = 0; i < fabrik_data_chain.size(); i++) {
		if (!_cache_bone(fabrik_data_chain[i].bone_node_cache, fabrik_data_chain[i].bone_node)) {
			ret.append(vformat("Joint %d bone %s was not found.", i, fabrik_data_chain[i].bone_node));
		}
	}
	return ret;
}
