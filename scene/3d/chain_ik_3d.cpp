/**************************************************************************/
/*  chain_ik_3d.cpp                                                       */
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

#include "chain_ik_3d.h"

bool ChainIK3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "root_bone_name") {
			set_root_bone_name(which, p_value);
		} else if (what == "root_bone") {
			set_root_bone(which, p_value);
		} else if (what == "end_bone_name") {
			set_end_bone_name(which, p_value);
		} else if (what == "end_bone") {
			String opt = path.get_slicec('/', 3);
			if (opt.is_empty()) {
				set_end_bone(which, p_value);
			} else if (opt == "direction") {
				set_end_bone_direction(which, static_cast<BoneDirection>((int)p_value));
			} else if (opt == "length") {
				set_end_bone_length(which, p_value);
			} else {
				return false;
			}
		} else if (what == "extend_end_bone") {
			set_extend_end_bone(which, p_value);
		} else if (what == "joint_count") {
			set_joint_count(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool ChainIK3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "root_bone_name") {
			r_ret = get_root_bone_name(which);
		} else if (what == "root_bone") {
			r_ret = get_root_bone(which);
		} else if (what == "end_bone_name") {
			r_ret = get_end_bone_name(which);
		} else if (what == "end_bone") {
			String opt = path.get_slicec('/', 3);
			if (opt.is_empty()) {
				r_ret = get_end_bone(which);
			} else if (opt == "direction") {
				r_ret = (int)get_end_bone_direction(which);
			} else if (opt == "length") {
				r_ret = get_end_bone_length(which);
			} else {
				return false;
			}
		} else if (what == "extend_end_bone") {
			r_ret = is_end_bone_extended(which);
		} else if (what == "joint_count") {
			r_ret = get_joint_count(which);
		} else if (what == "joints") {
			int idx = path.get_slicec('/', 3).to_int();
			String prop = path.get_slicec('/', 4);
			if (prop == "bone_name") {
				r_ret = get_joint_bone_name(which, idx);
			} else if (prop == "bone") {
				r_ret = get_joint_bone(which, idx);
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	return true;
}

void ChainIK3D::get_property_list(List<PropertyInfo> *p_list) const {
	String enum_hint;
	Skeleton3D *skeleton = get_skeleton();
	if (skeleton) {
		enum_hint = skeleton->get_concatenated_bone_names();
	}

	LocalVector<PropertyInfo> props;

	for (uint32_t i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		props.push_back(PropertyInfo(Variant::STRING, path + "root_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		props.push_back(PropertyInfo(Variant::INT, path + "root_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		props.push_back(PropertyInfo(Variant::STRING, path + "end_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		props.push_back(PropertyInfo(Variant::INT, path + "end_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		props.push_back(PropertyInfo(Variant::BOOL, path + "extend_end_bone"));
		props.push_back(PropertyInfo(Variant::INT, path + "end_bone/direction", PROPERTY_HINT_ENUM, SkeletonModifier3D::get_hint_bone_direction()));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "end_bone/length", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"));
		props.push_back(PropertyInfo(Variant::INT, path + "joint_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Joints," + path + "joints/,static,const"));
		for (uint32_t j = 0; j < chain_settings[i]->joints.size(); j++) {
			String joint_path = path + "joints/" + itos(j) + "/";
			props.push_back(PropertyInfo(Variant::STRING, joint_path + "bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint, PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
			props.push_back(PropertyInfo(Variant::INT, joint_path + "bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY));
		}
	}

	for (PropertyInfo &p : props) {
		_validate_dynamic_prop(p);
		p_list->push_back(p);
	}
}

void ChainIK3D::_validate_dynamic_prop(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() > 2 && split[0] == "settings") {
		int which = split[1].to_int();

		// Extended end bone option.
		bool force_hide = false;
		if (split[2] == "extend_end_bone" && get_end_bone(which) == -1) {
			p_property.usage = PROPERTY_USAGE_NONE;
			force_hide = true;
		}
		if (force_hide || (split[2] == "end_bone" && !is_end_bone_extended(which) && split.size() > 3)) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
	}
}

// Setting.

void ChainIK3D::set_root_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	chain_settings[p_index]->root_bone.name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_root_bone(p_index, sk->find_bone(chain_settings[p_index]->root_bone.name));
	}
}

String ChainIK3D::get_root_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	return chain_settings[p_index]->root_bone.name;
}

void ChainIK3D::set_root_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	bool changed = chain_settings[p_index]->root_bone.bone != p_bone;
	chain_settings[p_index]->root_bone.bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (chain_settings[p_index]->root_bone.bone <= -1 || chain_settings[p_index]->root_bone.bone >= sk->get_bone_count()) {
			WARN_PRINT("Root bone index out of range!");
			chain_settings[p_index]->root_bone.bone = -1;
		} else {
			chain_settings[p_index]->root_bone.name = sk->get_bone_name(chain_settings[p_index]->root_bone.bone);
		}
	}
	if (changed) {
		_update_joints(p_index);
	}
}

int ChainIK3D::get_root_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return chain_settings[p_index]->root_bone.bone;
}

void ChainIK3D::set_end_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	chain_settings[p_index]->end_bone.name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_end_bone(p_index, sk->find_bone(chain_settings[p_index]->end_bone.name));
	}
}

String ChainIK3D::get_end_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	return chain_settings[p_index]->end_bone.name;
}

void ChainIK3D::set_end_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	bool changed = chain_settings[p_index]->end_bone.bone != p_bone;
	chain_settings[p_index]->end_bone.bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (chain_settings[p_index]->end_bone.bone <= -1 || chain_settings[p_index]->end_bone.bone >= sk->get_bone_count()) {
			WARN_PRINT("End bone index out of range!");
			chain_settings[p_index]->end_bone.bone = -1;
		} else {
			chain_settings[p_index]->end_bone.name = sk->get_bone_name(chain_settings[p_index]->end_bone.bone);
		}
	}
	if (changed) {
		_update_joints(p_index);
	}
	notify_property_list_changed();
}

int ChainIK3D::get_end_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return chain_settings[p_index]->end_bone.bone;
}

void ChainIK3D::set_extend_end_bone(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	chain_settings[p_index]->extend_end_bone = p_enabled;
	_make_simulation_dirty(p_index);
	Skeleton3D *sk = get_skeleton();
	if (sk && !chain_settings[p_index]->joints.is_empty()) {
		_validate_axis(sk, p_index, chain_settings[p_index]->joints.size() - 1);
	}
	notify_property_list_changed();
#ifdef TOOLS_ENABLED
	_make_gizmo_dirty();
#endif // TOOLS_ENABLED
}

bool ChainIK3D::is_end_bone_extended(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), false);
	return chain_settings[p_index]->extend_end_bone;
}

void ChainIK3D::set_end_bone_direction(int p_index, BoneDirection p_bone_direction) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	chain_settings[p_index]->end_bone_direction = p_bone_direction;
	Skeleton3D *sk = get_skeleton();
	if (sk && !chain_settings[p_index]->joints.is_empty()) {
		_validate_axis(sk, p_index, chain_settings[p_index]->joints.size() - 1);
	}
#ifdef TOOLS_ENABLED
	_make_gizmo_dirty();
#endif // TOOLS_ENABLED
	if (mutable_bone_axes) {
		return; // Chain dir will be recaluclated in _update_bone_axis().
	}
	_make_simulation_dirty(p_index);
}

SkeletonModifier3D::BoneDirection ChainIK3D::get_end_bone_direction(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), BONE_DIRECTION_FROM_PARENT);
	return chain_settings[p_index]->end_bone_direction;
}

void ChainIK3D::set_end_bone_length(int p_index, float p_length) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	float old = chain_settings[p_index]->end_bone_length;
	chain_settings[p_index]->end_bone_length = p_length;
#ifdef TOOLS_ENABLED
	_make_gizmo_dirty();
#endif // TOOLS_ENABLED
	if (mutable_bone_axes && Math::is_zero_approx(old) == Math::is_zero_approx(p_length)) {
		return; // If chain size is not changed, length will be recaluclated in _update_bone_axis().
	}
	_make_simulation_dirty(p_index);
}

float ChainIK3D::get_end_bone_length(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	return chain_settings[p_index]->end_bone_length;
}

// Individual joints.

String ChainIK3D::get_joint_bone_name(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	const LocalVector<BoneJoint> &joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), String());
	return joints[p_joint].name;
}

void ChainIK3D::_set_joint_bone(int p_index, int p_joint, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	LocalVector<BoneJoint> &joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, (int)joints.size());
	joints[p_joint].bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (joints[p_joint].bone <= -1 || joints[p_joint].bone >= sk->get_bone_count()) {
			WARN_PRINT("Joint bone index out of range!");
			joints[p_joint].bone = -1;
		} else {
			joints[p_joint].name = sk->get_bone_name(joints[p_joint].bone);
		}
	}
}

int ChainIK3D::get_joint_bone(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	const LocalVector<BoneJoint> &joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), -1);
	return joints[p_joint].bone;
}

void ChainIK3D::set_joint_count(int p_index, int p_count) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ERR_FAIL_COND(p_count < 0);
	LocalVector<BoneJoint> &joints = chain_settings[p_index]->joints;
	joints.resize(p_count);
	_set_joint_count(p_index, p_count);
	notify_property_list_changed();
}

void ChainIK3D::_set_joint_count(int p_index, int p_count) {
	//
}

int ChainIK3D::get_joint_count(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	const LocalVector<BoneJoint> &joints = chain_settings[p_index]->joints;
	return joints.size();
}

void ChainIK3D::_bind_methods() {
	// Setting.
	ClassDB::bind_method(D_METHOD("set_root_bone_name", "index", "bone_name"), &ChainIK3D::set_root_bone_name);
	ClassDB::bind_method(D_METHOD("get_root_bone_name", "index"), &ChainIK3D::get_root_bone_name);
	ClassDB::bind_method(D_METHOD("set_root_bone", "index", "bone"), &ChainIK3D::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone", "index"), &ChainIK3D::get_root_bone);

	ClassDB::bind_method(D_METHOD("set_end_bone_name", "index", "bone_name"), &ChainIK3D::set_end_bone_name);
	ClassDB::bind_method(D_METHOD("get_end_bone_name", "index"), &ChainIK3D::get_end_bone_name);
	ClassDB::bind_method(D_METHOD("set_end_bone", "index", "bone"), &ChainIK3D::set_end_bone);
	ClassDB::bind_method(D_METHOD("get_end_bone", "index"), &ChainIK3D::get_end_bone);

	ClassDB::bind_method(D_METHOD("set_extend_end_bone", "index", "enabled"), &ChainIK3D::set_extend_end_bone);
	ClassDB::bind_method(D_METHOD("is_end_bone_extended", "index"), &ChainIK3D::is_end_bone_extended);
	ClassDB::bind_method(D_METHOD("set_end_bone_direction", "index", "bone_direction"), &ChainIK3D::set_end_bone_direction);
	ClassDB::bind_method(D_METHOD("get_end_bone_direction", "index"), &ChainIK3D::get_end_bone_direction);
	ClassDB::bind_method(D_METHOD("set_end_bone_length", "index", "length"), &ChainIK3D::set_end_bone_length);
	ClassDB::bind_method(D_METHOD("get_end_bone_length", "index"), &ChainIK3D::get_end_bone_length);

	// Individual joints.
	ClassDB::bind_method(D_METHOD("get_joint_bone_name", "index", "joint"), &ChainIK3D::get_joint_bone_name);
	ClassDB::bind_method(D_METHOD("get_joint_bone", "index", "joint"), &ChainIK3D::get_joint_bone);

	ClassDB::bind_method(D_METHOD("get_joint_count", "index"), &ChainIK3D::get_joint_count);
}

void ChainIK3D::_validate_bone_names() {
	for (uint32_t i = 0; i < settings.size(); i++) {
		// Prior bone name.
		if (!chain_settings[i]->root_bone.name.is_empty()) {
			set_root_bone_name(i, chain_settings[i]->root_bone.name);
		} else if (chain_settings[i]->root_bone.bone != -1) {
			set_root_bone(i, chain_settings[i]->root_bone.bone);
		}
		// Prior bone name.
		if (!chain_settings[i]->end_bone.name.is_empty()) {
			set_end_bone_name(i, chain_settings[i]->end_bone.name);
		} else if (chain_settings[i]->end_bone.bone != -1) {
			set_end_bone(i, chain_settings[i]->end_bone.bone);
		}
	}
}

void ChainIK3D::_validate_axes(Skeleton3D *p_skeleton) const {
	for (uint32_t i = 0; i < settings.size(); i++) {
		for (uint32_t j = 0; j < chain_settings[i]->joints.size(); j++) {
			_validate_axis(p_skeleton, i, j);
		}
	}
}

void ChainIK3D::_validate_axis(Skeleton3D *p_skeleton, int p_index, int p_joint) const {
	//
}

void ChainIK3D::_make_all_joints_dirty() {
	for (uint32_t i = 0; i < settings.size(); i++) {
		_update_joints(i);
	}
}

void ChainIK3D::_update_joints(int p_index) {
	_make_simulation_dirty(p_index);

#ifdef TOOLS_ENABLED
	update_gizmos(); // To clear invalid setting.
#endif // TOOLS_ENABLED

	Skeleton3D *sk = get_skeleton();
	int current_bone = chain_settings[p_index]->end_bone.bone;
	int root_bone = chain_settings[p_index]->root_bone.bone;
	if (!sk || current_bone < 0 || root_bone < 0) {
		set_joint_count(p_index, 0);
		return;
	}

	// Validation.
	bool valid = false;
	while (current_bone >= 0) {
		if (current_bone == root_bone) {
			valid = true;
			break;
		}
		current_bone = sk->get_bone_parent(current_bone);
	}

	if (!valid) {
		set_joint_count(p_index, 0);
		ERR_FAIL_EDMSG("End bone must be the same as or a child of the root bone.");
	}

	Vector<int> new_joints;
	current_bone = chain_settings[p_index]->end_bone.bone;
	while (current_bone != root_bone) {
		new_joints.push_back(current_bone);
		current_bone = sk->get_bone_parent(current_bone);
	}
	new_joints.push_back(current_bone);
	new_joints.reverse();

	set_joint_count(p_index, new_joints.size());
	for (uint32_t i = 0; i < new_joints.size(); i++) {
		_set_joint_bone(p_index, i, new_joints[i]);
	}

	if (sk) {
		_validate_axes(sk);
	}

#ifdef TOOLS_ENABLED
	_make_gizmo_dirty();
#endif // TOOLS_ENABLED
}

void ChainIK3D::_process_ik(Skeleton3D *p_skeleton, double p_delta) {
	//
}

#ifdef TOOLS_ENABLED
void ChainIK3D::_update_mutable_info() {
	if (!is_inside_tree()) {
		return;
	}
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		for (uint32_t i = 0; i < settings.size(); i++) {
			chain_settings[i]->root_global_rest = Transform3D();
		}
		return;
	}
	bool changed = false;
	for (uint32_t i = 0; i < settings.size(); i++) {
		int root_bone = chain_settings[i]->root_bone.bone;
		if (root_bone < 0) {
			continue;
		}
		Transform3D new_tr = get_bone_global_rest_mutable(skeleton, root_bone);
		changed = changed || !chain_settings[i]->root_global_rest.is_equal_approx(new_tr);
		chain_settings[i]->root_global_rest = new_tr;
	}
	if (changed) {
		_make_gizmo_dirty();
	}
}

Transform3D ChainIK3D::get_bone_global_rest_mutable(Skeleton3D *p_skeleton, int p_bone) {
	int current = p_bone;
	Transform3D accum;
	int parent = p_skeleton->get_bone_parent(current);
	if (parent >= 0) {
		accum = p_skeleton->get_bone_global_rest(parent);
	}
	Transform3D tr = p_skeleton->get_bone_rest(current);
	// Note:
	// Chain IK gizmo might not be able to retrieve this pose in SkeletonModifier update process.
	// So the gizmo uses bone_vector insteads but parent of root bone doesn't have bone_vector.
	// Then, we needs to cache this pose in IK node.
	tr.origin = p_skeleton->get_bone_pose_position(current);
	accum *= tr;
	return accum;
}

Transform3D ChainIK3D::get_chain_root_global_rest(int p_index) {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Transform3D());
	return chain_settings[p_index]->root_global_rest;
}

Vector3 ChainIK3D::get_bone_vector(int p_index, int p_joint) const {
	return Vector3();
}
#endif // TOOLS_ENABLED

ChainIK3D::~ChainIK3D() {
	clear_settings();
}
