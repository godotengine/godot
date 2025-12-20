/**************************************************************************/
/*  limit_angular_velocity_modifier_3d.cpp                                */
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

#include "limit_angular_velocity_modifier_3d.h"

bool LimitAngularVelocityModifier3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("chains/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)chains.size(), false);

		if (what == "root_bone_name") {
			set_root_bone_name(which, p_value);
		} else if (what == "root_bone") {
			set_root_bone(which, p_value);
		} else if (what == "end_bone_name") {
			set_end_bone_name(which, p_value);
		} else if (what == "end_bone") {
			set_end_bone(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool LimitAngularVelocityModifier3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("chains/")) {
		int which = path.get_slicec('/', 1).to_int();
		ERR_FAIL_INDEX_V(which, (int)chains.size(), false);
		String what = path.get_slicec('/', 2);
		if (what == "root_bone_name") {
			r_ret = get_root_bone_name(which);
		} else if (what == "root_bone") {
			r_ret = get_root_bone(which);
		} else if (what == "end_bone_name") {
			r_ret = get_end_bone_name(which);
		} else if (what == "end_bone") {
			r_ret = get_end_bone(which);
		} else {
			return false;
		}
	}
	if (path.begins_with("joints/")) {
		int which = path.get_slicec('/', 1).to_int();
		ERR_FAIL_INDEX_V(which, (int)joints.size(), false);
		String what = path.get_slicec('/', 2);
		if (what == "bone_name") {
			r_ret = get_joint_bone_name(which);
		} else if (what == "bone") {
			r_ret = get_joint_bone(which);
		} else {
			return false;
		}
	}
	return true;
}

void LimitAngularVelocityModifier3D::_get_property_list(List<PropertyInfo> *p_list) const {
	String enum_hint;
	Skeleton3D *skeleton = get_skeleton();
	if (skeleton) {
		enum_hint = skeleton->get_concatenated_bone_names();
	}

	for (uint32_t i = 0; i < chains.size(); i++) {
		String path = "chains/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::STRING, path + "root_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		p_list->push_back(PropertyInfo(Variant::INT, path + "root_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::STRING, path + "end_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		p_list->push_back(PropertyInfo(Variant::INT, path + "end_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	}

	for (uint32_t i = 0; i < joints.size(); i++) {
		String path = "joints/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::STRING, path + "bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint, PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
		p_list->push_back(PropertyInfo(Variant::INT, path + "bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY));
	}
}

void LimitAngularVelocityModifier3D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "joint_count") {
		p_property.usage = PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY | PROPERTY_USAGE_READ_ONLY;
		p_property.class_name = "Joints,joints/,static,const";
	}
}

void LimitAngularVelocityModifier3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_make_joints_dirty();
		} break;
	}
}

// Setting.

void LimitAngularVelocityModifier3D::set_root_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)chains.size());
	chains[p_index].root_bone.name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_root_bone(p_index, sk->find_bone(chains[p_index].root_bone.name));
	}
}

String LimitAngularVelocityModifier3D::get_root_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)chains.size(), String());
	return chains[p_index].root_bone.name;
}

void LimitAngularVelocityModifier3D::set_root_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)chains.size());
	bool changed = chains[p_index].root_bone.bone != p_bone;
	chains[p_index].root_bone.bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (chains[p_index].root_bone.bone <= -1 || chains[p_index].root_bone.bone >= sk->get_bone_count()) {
			WARN_PRINT("Root bone index out of range!");
			chains[p_index].root_bone.bone = -1;
		} else {
			chains[p_index].root_bone.name = sk->get_bone_name(chains[p_index].root_bone.bone);
		}
	}
	if (changed) {
		_make_joints_dirty();
	}
}

int LimitAngularVelocityModifier3D::get_root_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)chains.size(), -1);
	return chains[p_index].root_bone.bone;
}

void LimitAngularVelocityModifier3D::set_end_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)chains.size());
	chains[p_index].end_bone.name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_end_bone(p_index, sk->find_bone(chains[p_index].end_bone.name));
	}
}

String LimitAngularVelocityModifier3D::get_end_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)chains.size(), String());
	return chains[p_index].end_bone.name;
}

void LimitAngularVelocityModifier3D::set_end_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)chains.size());
	bool changed = chains[p_index].end_bone.bone != p_bone;
	chains[p_index].end_bone.bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (chains[p_index].end_bone.bone <= -1 || chains[p_index].end_bone.bone >= sk->get_bone_count()) {
			WARN_PRINT("End bone index out of range!");
			chains[p_index].end_bone.bone = -1;
		} else {
			chains[p_index].end_bone.name = sk->get_bone_name(chains[p_index].end_bone.bone);
		}
	}
	if (changed) {
		_make_joints_dirty();
	}
}

int LimitAngularVelocityModifier3D::get_end_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)chains.size(), -1);
	return chains[p_index].end_bone.bone;
}

void LimitAngularVelocityModifier3D::set_chain_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);
	chains.resize(p_count);
	_make_joints_dirty();
}

int LimitAngularVelocityModifier3D::get_chain_count() const {
	return chains.size();
}

void LimitAngularVelocityModifier3D::clear_chains() {
	set_chain_count(0);
}

String LimitAngularVelocityModifier3D::get_joint_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)joints.size(), String());
	return joints[p_index].name;
}

int LimitAngularVelocityModifier3D::get_joint_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)joints.size(), -1);
	return joints[p_index].bone;
}

int LimitAngularVelocityModifier3D::_get_joint_count() const {
	return joints.size();
}

void LimitAngularVelocityModifier3D::set_max_angular_velocity(double p_angular_velocity) {
	max_angular_velocity = p_angular_velocity;
}

double LimitAngularVelocityModifier3D::get_max_angular_velocity() const {
	return max_angular_velocity;
}

void LimitAngularVelocityModifier3D::set_exclude(bool p_exclude) {
	exclude = p_exclude;
}

bool LimitAngularVelocityModifier3D::is_exclude() const {
	return exclude;
}

void LimitAngularVelocityModifier3D::_bind_methods() {
	// Setting.
	ClassDB::bind_method(D_METHOD("set_root_bone_name", "index", "bone_name"), &LimitAngularVelocityModifier3D::set_root_bone_name);
	ClassDB::bind_method(D_METHOD("get_root_bone_name", "index"), &LimitAngularVelocityModifier3D::get_root_bone_name);
	ClassDB::bind_method(D_METHOD("set_root_bone", "index", "bone"), &LimitAngularVelocityModifier3D::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone", "index"), &LimitAngularVelocityModifier3D::get_root_bone);

	ClassDB::bind_method(D_METHOD("set_end_bone_name", "index", "bone_name"), &LimitAngularVelocityModifier3D::set_end_bone_name);
	ClassDB::bind_method(D_METHOD("get_end_bone_name", "index"), &LimitAngularVelocityModifier3D::get_end_bone_name);
	ClassDB::bind_method(D_METHOD("set_end_bone", "index", "bone"), &LimitAngularVelocityModifier3D::set_end_bone);
	ClassDB::bind_method(D_METHOD("get_end_bone", "index"), &LimitAngularVelocityModifier3D::get_end_bone);

	ClassDB::bind_method(D_METHOD("set_chain_count", "count"), &LimitAngularVelocityModifier3D::set_chain_count);
	ClassDB::bind_method(D_METHOD("get_chain_count"), &LimitAngularVelocityModifier3D::get_chain_count);
	ClassDB::bind_method(D_METHOD("clear_chains"), &LimitAngularVelocityModifier3D::clear_chains);

	ClassDB::bind_method(D_METHOD("set_max_angular_velocity", "angular_velocity"), &LimitAngularVelocityModifier3D::set_max_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_max_angular_velocity"), &LimitAngularVelocityModifier3D::get_max_angular_velocity);
	ClassDB::bind_method(D_METHOD("set_exclude", "exclude"), &LimitAngularVelocityModifier3D::set_exclude);
	ClassDB::bind_method(D_METHOD("is_exclude"), &LimitAngularVelocityModifier3D::is_exclude);

	ClassDB::bind_method(D_METHOD("reset"), &LimitAngularVelocityModifier3D::reset);

	ClassDB::bind_method(D_METHOD("_get_joint_count"), &LimitAngularVelocityModifier3D::_get_joint_count);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_angular_velocity", PROPERTY_HINT_RANGE, "0,720,or_greater,radians_as_degrees,suffix:" + String(U"°") + "/s"), "set_max_angular_velocity", "get_max_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "exclude"), "set_exclude", "is_exclude");
	ADD_ARRAY_COUNT("Chains", "chain_count", "set_chain_count", "get_chain_count", "chains/");
	ADD_ARRAY_COUNT("Joints", "joint_count", "", "_get_joint_count", "joints/");
}

void LimitAngularVelocityModifier3D::_set_active(bool p_active) {
	if (p_active) {
		reset();
	}
}

void LimitAngularVelocityModifier3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
	_make_joints_dirty();
}

void LimitAngularVelocityModifier3D::_validate_bone_names() {
	for (uint32_t i = 0; i < chains.size(); i++) {
		// Prior bone name.
		if (!chains[i].root_bone.name.is_empty()) {
			set_root_bone_name(i, chains[i].root_bone.name);
		} else if (chains[i].root_bone.bone != -1) {
			set_root_bone(i, chains[i].root_bone.bone);
		}
		// Prior bone name.
		if (!chains[i].end_bone.name.is_empty()) {
			set_end_bone_name(i, chains[i].end_bone.name);
		} else if (chains[i].end_bone.bone != -1) {
			set_end_bone(i, chains[i].end_bone.bone);
		}
	}
}

void LimitAngularVelocityModifier3D::_make_joints_dirty() {
	if (joints_dirty) {
		return;
	}
	joints_dirty = true;
	callable_mp(this, &LimitAngularVelocityModifier3D::_update_joints).call_deferred();
}

bool LimitAngularVelocityModifier3D::_is_joint_contained(int p_bone) {
	bool ret = false;
	for (const BoneJoint &joint : joints) {
		if (joint.bone == p_bone) {
			ret = true;
			break;
		}
	}
	return ret;
}

void LimitAngularVelocityModifier3D::_update_joints() {
	joints.clear();
	bones.clear();

	Skeleton3D *sk = get_skeleton();
	if (!sk) {
		joints_dirty = false;
		notify_property_list_changed();
		return;
	}

	LocalVector<int> tmp_joints;
	for (uint32_t i = 0; i < chains.size(); i++) {
		tmp_joints.clear();
		Chain cn = chains[i];
		int current_bone = cn.end_bone.bone;
		int root_bone = cn.root_bone.bone;
		if (current_bone < 0 || root_bone < 0) {
			continue;
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
			ERR_PRINT_ED("Chains[" + itos(i) + "]: End bone must be the same as or a child of root bone.");
			continue;
		}
		current_bone = cn.end_bone.bone;
		while (current_bone != root_bone) {
			tmp_joints.push_back(current_bone);
			current_bone = sk->get_bone_parent(current_bone);
		}
		tmp_joints.push_back(current_bone);
		tmp_joints.reverse();
		for (uint32_t j = 0; j < tmp_joints.size(); j++) {
			int bn = tmp_joints[j];
			if (!_is_joint_contained(bn)) {
				BoneJoint bj;
				bj.bone = bn;
				bj.name = sk->get_bone_name(bn);
				joints.push_back(bj);
			}
		}
	}

	if (exclude) {
		for (int b = 0; b < sk->get_bone_count(); b++) {
			if (_is_joint_contained(b)) {
				continue;
			}
			BoneRot br;
			br.first = b;
			br.second = sk->get_bone_pose_rotation(b);
			bones.push_back(br);
		}
	} else {
		for (const BoneJoint &E : joints) {
			BoneRot br;
			br.first = E.bone;
			br.second = sk->get_bone_pose_rotation(E.bone);
			bones.push_back(br);
		}
	}

	joints_dirty = false;
	reset();

	notify_property_list_changed();
}

void LimitAngularVelocityModifier3D::_process_modification(double p_delta) {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	if (init_needed) {
		// Note:
		// The pose retrieval within `_update_joints()` is done outside the skeleton's update process,
		// so it ignores the pose resulting from the previous modifier's modification.
		// This causes unintended initialization when `active` is set to true, so it must be initialized here.
		for (uint32_t i = 0; i < bones.size(); i++) {
			bones[i].second = skeleton->get_bone_pose_rotation(bones[i].first);
		}
		init_needed = false;
	}

	double limit_in_frame = max_angular_velocity * p_delta;
	for (uint32_t i = 0; i < bones.size(); i++) {
		int bn = bones[i].first;
		Quaternion dest = skeleton->get_bone_pose_rotation(bn);
		double diff = bones[i].second.angle_to(dest);
		if (!Math::is_zero_approx(diff)) {
			bones[i].second = bones[i].second.slerp(dest, MIN(1.0, limit_in_frame / diff));
		}
		skeleton->set_bone_pose_rotation(bn, bones[i].second);
	}
}

void LimitAngularVelocityModifier3D::reset() {
	init_needed = true;
}

LimitAngularVelocityModifier3D::~LimitAngularVelocityModifier3D() {
	clear_chains();
}
