/**************************************************************************/
/*  bone_twist_disperser_3d.cpp                                           */
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

#include "bone_twist_disperser_3d.h"

bool BoneTwistDisperser3D::_set(const StringName &p_path, const Variant &p_value) {
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
			set_end_bone(which, p_value);
		} else if (what == "end_bone_direction") {
			set_end_bone_direction(which, static_cast<BoneDirection>((int)p_value));
		} else if (what == "extend_end_bone") {
			set_extend_end_bone(which, p_value);
		} else if (what == "twist_from_rest") {
			set_twist_from_rest(which, p_value);
		} else if (what == "twist_from") {
			set_twist_from(which, p_value);
		} else if (what == "disperse_mode") {
			set_disperse_mode(which, static_cast<DisperseMode>((int)p_value));
		} else if (what == "weight_position") {
			set_weight_position(which, p_value);
		} else if (what == "damping_curve") {
			set_damping_curve(which, p_value);
		} else if (what == "joint_count") {
			set_joint_count(which, p_value);
		} else if (what == "joints") {
			int idx = path.get_slicec('/', 3).to_int();
			String prop = path.get_slicec('/', 4);
			if (prop == "twist_amount") {
				set_joint_twist_amount(which, idx, p_value);
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	return true;
}

bool BoneTwistDisperser3D::_get(const StringName &p_path, Variant &r_ret) const {
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
			r_ret = get_end_bone(which);
		} else if (what == "end_bone_direction") {
			r_ret = (int)get_end_bone_direction(which);
		} else if (what == "reference_bone_name") {
			r_ret = get_reference_bone_name(which);
		} else if (what == "extend_end_bone") {
			r_ret = is_end_bone_extended(which);
		} else if (what == "twist_from_rest") {
			r_ret = is_twist_from_rest(which);
		} else if (what == "twist_from") {
			r_ret = get_twist_from(which);
		} else if (what == "disperse_mode") {
			r_ret = (int)get_disperse_mode(which);
		} else if (what == "weight_position") {
			r_ret = get_weight_position(which);
		} else if (what == "damping_curve") {
			r_ret = get_damping_curve(which);
		} else if (what == "joint_count") {
			r_ret = get_joint_count(which);
		} else if (what == "joints") {
			int idx = path.get_slicec('/', 3).to_int();
			String prop = path.get_slicec('/', 4);
			if (prop == "bone_name") {
				r_ret = get_joint_bone_name(which, idx);
			} else if (prop == "bone") {
				r_ret = get_joint_bone(which, idx);
			} else if (prop == "twist_amount") {
				r_ret = get_joint_twist_amount(which, idx);
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	return true;
}

void BoneTwistDisperser3D::_get_property_list(List<PropertyInfo> *p_list) const {
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
		props.push_back(PropertyInfo(Variant::INT, path + "end_bone_direction", PROPERTY_HINT_ENUM, SkeletonModifier3D::get_hint_bone_direction()));

		props.push_back(PropertyInfo(Variant::STRING, path + "reference_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint, PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
		props.push_back(PropertyInfo(Variant::BOOL, path + "twist_from_rest"));
		props.push_back(PropertyInfo(Variant::QUATERNION, path + "twist_from"));
		props.push_back(PropertyInfo(Variant::INT, path + "disperse_mode", PROPERTY_HINT_ENUM, "Even,Weighted,Custom"));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "weight_position", PROPERTY_HINT_RANGE, "0,1,0.001"));
		props.push_back(PropertyInfo(Variant::OBJECT, path + "damping_curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"));

		props.push_back(PropertyInfo(Variant::INT, path + "joint_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Joints," + path + "joints/,static,const"));
		for (uint32_t j = 0; j < settings[i]->joints.size(); j++) {
			String joint_path = path + "joints/" + itos(j) + "/";
			props.push_back(PropertyInfo(Variant::STRING, joint_path + "bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint, PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
			props.push_back(PropertyInfo(Variant::INT, joint_path + "bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY));
			props.push_back(PropertyInfo(Variant::FLOAT, joint_path + "twist_amount", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,or_less"));
		}
	}

	for (PropertyInfo &p : props) {
		_validate_dynamic_prop(p);
		p_list->push_back(p);
	}
}

void BoneTwistDisperser3D::_validate_dynamic_prop(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() > 2 && split[0] == "settings") {
		int which = split[1].to_int();

		// Extended end bone option.
		bool force_hide = false;
		if (split[2] == "extend_end_bone" && get_end_bone(which) == -1) {
			p_property.usage = PROPERTY_USAGE_NONE;
			force_hide = true;
		}
		if (force_hide || (split[2] == "end_bone_direction" && !is_end_bone_extended(which))) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}

		if (split[2] == "twist_from" && is_twist_from_rest(which)) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}

		if (split[2] == "weight_position" && get_disperse_mode(which) != DISPERSE_MODE_WEIGHTED) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}

		if (split[2] == "damping_curve" && get_disperse_mode(which) != DISPERSE_MODE_CUSTOM) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}

		if (split[2] == "joints" && split[4] == "twist_amount") {
			bool mutable_amount = true;
			if (get_disperse_mode(which) != DISPERSE_MODE_CUSTOM) {
				mutable_amount = false;
			} else if (!is_end_bone_extended(which)) {
				int joint = split[3].to_int();
				mutable_amount = joint < get_joint_count(which) - 1; // Hide child of reference bone.
			}
			if (get_damping_curve(which).is_valid()) {
				p_property.usage |= PROPERTY_USAGE_READ_ONLY;
			}
			if (!mutable_amount) {
				p_property.usage = PROPERTY_USAGE_NONE;
			}
		}
	}
}

void BoneTwistDisperser3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_make_all_joints_dirty();
		} break;
	}
}

void BoneTwistDisperser3D::_set_active(bool p_active) {
	if (p_active) {
		_make_all_joints_dirty();
	}
}

void BoneTwistDisperser3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
	_make_all_joints_dirty();
}

// Setting.

void BoneTwistDisperser3D::set_mutable_bone_axes(bool p_enabled) {
	mutable_bone_axes = p_enabled;
}

bool BoneTwistDisperser3D::are_bone_axes_mutable() const {
	return mutable_bone_axes;
}

void BoneTwistDisperser3D::set_root_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->root_bone.name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_root_bone(p_index, sk->find_bone(settings[p_index]->root_bone.name));
	}
}

String BoneTwistDisperser3D::get_root_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	return settings[p_index]->root_bone.name;
}

void BoneTwistDisperser3D::set_root_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	bool changed = settings[p_index]->root_bone.bone != p_bone;
	settings[p_index]->root_bone.bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index]->root_bone.bone <= -1 || settings[p_index]->root_bone.bone >= sk->get_bone_count()) {
			WARN_PRINT("Root bone index out of range!");
			settings[p_index]->root_bone.bone = -1;
		} else {
			settings[p_index]->root_bone.name = sk->get_bone_name(settings[p_index]->root_bone.bone);
		}
	}
	if (changed) {
		_make_joints_dirty(p_index);
	}
}

int BoneTwistDisperser3D::get_root_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return settings[p_index]->root_bone.bone;
}

void BoneTwistDisperser3D::set_end_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->end_bone.name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_end_bone(p_index, sk->find_bone(settings[p_index]->end_bone.name));
	}
}

String BoneTwistDisperser3D::get_end_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	return settings[p_index]->end_bone.name;
}

void BoneTwistDisperser3D::set_end_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	bool changed = settings[p_index]->end_bone.bone != p_bone;
	settings[p_index]->end_bone.bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index]->end_bone.bone <= -1 || settings[p_index]->end_bone.bone >= sk->get_bone_count()) {
			WARN_PRINT("End bone index out of range!");
			settings[p_index]->end_bone.bone = -1;
		} else {
			settings[p_index]->end_bone.name = sk->get_bone_name(settings[p_index]->end_bone.bone);
		}
	}
	if (changed) {
		_make_joints_dirty(p_index);
	}
	notify_property_list_changed();
}

int BoneTwistDisperser3D::get_end_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return settings[p_index]->end_bone.bone;
}

void BoneTwistDisperser3D::set_extend_end_bone(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->extend_end_bone = p_enabled;
	_update_reference_bone(p_index);
	notify_property_list_changed();
}

bool BoneTwistDisperser3D::is_end_bone_extended(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), false);
	return settings[p_index]->extend_end_bone;
}

void BoneTwistDisperser3D::set_end_bone_direction(int p_index, BoneDirection p_bone_direction) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->end_bone_direction = p_bone_direction;
}

SkeletonModifier3D::BoneDirection BoneTwistDisperser3D::get_end_bone_direction(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), BONE_DIRECTION_FROM_PARENT);
	return settings[p_index]->end_bone_direction;
}

void BoneTwistDisperser3D::set_twist_from_rest(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->twist_from_rest = p_enabled;
	notify_property_list_changed();
}

bool BoneTwistDisperser3D::is_twist_from_rest(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), true);
	return settings[p_index]->twist_from_rest;
}

void BoneTwistDisperser3D::set_twist_from(int p_index, const Quaternion &p_from) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->twist_from = p_from;
}

Quaternion BoneTwistDisperser3D::get_twist_from(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Quaternion());
	return settings[p_index]->twist_from;
}

void BoneTwistDisperser3D::_update_reference_bone(int p_index) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	LocalVector<DisperseJointSetting> &joints = settings[p_index]->joints;
	if (joints.size() >= 2) {
		if (settings[p_index]->extend_end_bone) {
			settings[p_index]->reference_bone = settings[p_index]->end_bone;
			_update_curve(p_index);
			return;
		} else {
			Skeleton3D *sk = get_skeleton();
			if (sk) {
				int parent = sk->get_bone_parent(settings[p_index]->end_bone.bone);
				if (parent >= 0) {
					settings[p_index]->reference_bone.bone = parent;
					settings[p_index]->reference_bone.name = sk->get_bone_name(parent);
					_update_curve(p_index);
					return;
				}
			}
		}
	}
	settings[p_index]->reference_bone.bone = -1;
	settings[p_index]->reference_bone.name = String();
}

void BoneTwistDisperser3D::_update_curve(int p_index) {
	Ref<Curve> curve = settings[p_index]->damping_curve;
	if (curve.is_null()) {
		return;
	}
	LocalVector<DisperseJointSetting> &joints = settings[p_index]->joints;
	float unit = (int)joints.size() > 0 ? (1.0 / float((int)joints.size() - 1)) : 0.0;
	for (uint32_t i = 0; i < joints.size(); i++) {
		joints[i].custom_amount = curve->sample_baked(i * unit);
	}
}

String BoneTwistDisperser3D::get_reference_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	return settings[p_index]->reference_bone.name;
}

int BoneTwistDisperser3D::get_reference_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return settings[p_index]->reference_bone.bone;
}

void BoneTwistDisperser3D::set_disperse_mode(int p_index, DisperseMode p_disperse_mode) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->disperse_mode = p_disperse_mode;
	notify_property_list_changed();
}

BoneTwistDisperser3D::DisperseMode BoneTwistDisperser3D::get_disperse_mode(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), DISPERSE_MODE_EVEN);
	return settings[p_index]->disperse_mode;
}

void BoneTwistDisperser3D::set_weight_position(int p_index, float p_position) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->weight_position = p_position;
}

float BoneTwistDisperser3D::get_weight_position(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0.0);
	return settings[p_index]->weight_position;
}

void BoneTwistDisperser3D::set_damping_curve(int p_index, const Ref<Curve> &p_damping_curve) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	bool changed = settings[p_index]->damping_curve != p_damping_curve;
	if (settings[p_index]->damping_curve.is_valid()) {
		settings[p_index]->damping_curve->disconnect_changed(callable_mp(this, &BoneTwistDisperser3D::_update_curve));
	}
	settings[p_index]->damping_curve = p_damping_curve;
	if (settings[p_index]->damping_curve.is_valid()) {
		settings[p_index]->damping_curve->connect_changed(callable_mp(this, &BoneTwistDisperser3D::_update_curve).bind(p_index));
	}
	if (changed) {
		_make_joints_dirty(p_index);
	}
	notify_property_list_changed();
}

Ref<Curve> BoneTwistDisperser3D::get_damping_curve(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Ref<Curve>());
	return settings[p_index]->damping_curve;
}

// Individual joints.

String BoneTwistDisperser3D::get_joint_bone_name(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	const LocalVector<DisperseJointSetting> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), String());
	return joints[p_joint].joint.name;
}

void BoneTwistDisperser3D::_set_joint_bone(int p_index, int p_joint, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	LocalVector<DisperseJointSetting> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, (int)joints.size());
	joints[p_joint].joint.bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (joints[p_joint].joint.bone <= -1 || joints[p_joint].joint.bone >= sk->get_bone_count()) {
			WARN_PRINT("Joint bone index out of range!");
			joints[p_joint].joint.bone = -1;
		} else {
			joints[p_joint].joint.name = sk->get_bone_name(joints[p_joint].joint.bone);
		}
	}
}

int BoneTwistDisperser3D::get_joint_bone(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	const LocalVector<DisperseJointSetting> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), -1);
	return joints[p_joint].joint.bone;
}

void BoneTwistDisperser3D::set_joint_count(int p_index, int p_count) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ERR_FAIL_COND(p_count < 0);
	LocalVector<DisperseJointSetting> &joints = settings[p_index]->joints;
	joints.resize(p_count);
	notify_property_list_changed();
}

int BoneTwistDisperser3D::get_joint_count(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	const LocalVector<DisperseJointSetting> &joints = settings[p_index]->joints;
	return joints.size();
}

void BoneTwistDisperser3D::set_joint_twist_amount(int p_index, int p_joint, float p_amount) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	LocalVector<DisperseJointSetting> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, (int)joints.size());
	joints[p_joint].custom_amount = p_amount;
}

float BoneTwistDisperser3D::get_joint_twist_amount(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	const LocalVector<DisperseJointSetting> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), 0);
	return joints[p_joint].custom_amount;
}

void BoneTwistDisperser3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_setting_count", "count"), &BoneTwistDisperser3D::set_setting_count);
	ClassDB::bind_method(D_METHOD("get_setting_count"), &BoneTwistDisperser3D::get_setting_count);
	ClassDB::bind_method(D_METHOD("clear_settings"), &BoneTwistDisperser3D::clear_settings);

	ClassDB::bind_method(D_METHOD("set_mutable_bone_axes", "enabled"), &BoneTwistDisperser3D::set_mutable_bone_axes);
	ClassDB::bind_method(D_METHOD("are_bone_axes_mutable"), &BoneTwistDisperser3D::are_bone_axes_mutable);

	// Setting.
	ClassDB::bind_method(D_METHOD("set_root_bone_name", "index", "bone_name"), &BoneTwistDisperser3D::set_root_bone_name);
	ClassDB::bind_method(D_METHOD("get_root_bone_name", "index"), &BoneTwistDisperser3D::get_root_bone_name);
	ClassDB::bind_method(D_METHOD("set_root_bone", "index", "bone"), &BoneTwistDisperser3D::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone", "index"), &BoneTwistDisperser3D::get_root_bone);

	ClassDB::bind_method(D_METHOD("set_end_bone_name", "index", "bone_name"), &BoneTwistDisperser3D::set_end_bone_name);
	ClassDB::bind_method(D_METHOD("get_end_bone_name", "index"), &BoneTwistDisperser3D::get_end_bone_name);
	ClassDB::bind_method(D_METHOD("set_end_bone", "index", "bone"), &BoneTwistDisperser3D::set_end_bone);
	ClassDB::bind_method(D_METHOD("get_end_bone", "index"), &BoneTwistDisperser3D::get_end_bone);

	ClassDB::bind_method(D_METHOD("get_reference_bone_name", "index"), &BoneTwistDisperser3D::get_reference_bone_name);
	ClassDB::bind_method(D_METHOD("get_reference_bone", "index"), &BoneTwistDisperser3D::get_reference_bone);

	ClassDB::bind_method(D_METHOD("set_extend_end_bone", "index", "enabled"), &BoneTwistDisperser3D::set_extend_end_bone);
	ClassDB::bind_method(D_METHOD("is_end_bone_extended", "index"), &BoneTwistDisperser3D::is_end_bone_extended);
	ClassDB::bind_method(D_METHOD("set_end_bone_direction", "index", "bone_direction"), &BoneTwistDisperser3D::set_end_bone_direction);
	ClassDB::bind_method(D_METHOD("get_end_bone_direction", "index"), &BoneTwistDisperser3D::get_end_bone_direction);

	ClassDB::bind_method(D_METHOD("set_twist_from_rest", "index", "enabled"), &BoneTwistDisperser3D::set_twist_from_rest);
	ClassDB::bind_method(D_METHOD("is_twist_from_rest", "index"), &BoneTwistDisperser3D::is_twist_from_rest);
	ClassDB::bind_method(D_METHOD("set_twist_from", "index", "from"), &BoneTwistDisperser3D::set_twist_from);
	ClassDB::bind_method(D_METHOD("get_twist_from", "index"), &BoneTwistDisperser3D::get_twist_from);

	ClassDB::bind_method(D_METHOD("set_disperse_mode", "index", "disperse_mode"), &BoneTwistDisperser3D::set_disperse_mode);
	ClassDB::bind_method(D_METHOD("get_disperse_mode", "index"), &BoneTwistDisperser3D::get_disperse_mode);
	ClassDB::bind_method(D_METHOD("set_weight_position", "index", "weight_position"), &BoneTwistDisperser3D::set_weight_position);
	ClassDB::bind_method(D_METHOD("get_weight_position", "index"), &BoneTwistDisperser3D::get_weight_position);
	ClassDB::bind_method(D_METHOD("set_damping_curve", "index", "curve"), &BoneTwistDisperser3D::set_damping_curve);
	ClassDB::bind_method(D_METHOD("get_damping_curve", "index"), &BoneTwistDisperser3D::get_damping_curve);

	// Individual joints.
	ClassDB::bind_method(D_METHOD("get_joint_bone_name", "index", "joint"), &BoneTwistDisperser3D::get_joint_bone_name);
	ClassDB::bind_method(D_METHOD("get_joint_bone", "index", "joint"), &BoneTwistDisperser3D::get_joint_bone);
	ClassDB::bind_method(D_METHOD("get_joint_twist_amount", "index", "joint"), &BoneTwistDisperser3D::get_joint_twist_amount);
	ClassDB::bind_method(D_METHOD("set_joint_twist_amount", "index", "joint", "twist_amount"), &BoneTwistDisperser3D::set_joint_twist_amount);

	ClassDB::bind_method(D_METHOD("get_joint_count", "index"), &BoneTwistDisperser3D::get_joint_count);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mutable_bone_axes"), "set_mutable_bone_axes", "are_bone_axes_mutable");
	ADD_ARRAY_COUNT("Settings", "setting_count", "set_setting_count", "get_setting_count", "settings/");

	BIND_ENUM_CONSTANT(DISPERSE_MODE_EVEN);
	BIND_ENUM_CONSTANT(DISPERSE_MODE_WEIGHTED);
	BIND_ENUM_CONSTANT(DISPERSE_MODE_CUSTOM);
}

void BoneTwistDisperser3D::_validate_bone_names() {
	for (uint32_t i = 0; i < settings.size(); i++) {
		// Prior bone name.
		if (!settings[i]->root_bone.name.is_empty()) {
			set_root_bone_name(i, settings[i]->root_bone.name);
		} else if (settings[i]->root_bone.bone != -1) {
			set_root_bone(i, settings[i]->root_bone.bone);
		}
		// Prior bone name.
		if (!settings[i]->end_bone.name.is_empty()) {
			set_end_bone_name(i, settings[i]->end_bone.name);
		} else if (settings[i]->end_bone.bone != -1) {
			set_end_bone(i, settings[i]->end_bone.bone);
		}
	}
}

void BoneTwistDisperser3D::_make_all_joints_dirty() {
	for (uint32_t i = 0; i < settings.size(); i++) {
		_make_joints_dirty(i);
	}
}

void BoneTwistDisperser3D::_make_joints_dirty(int p_index) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	if (settings[p_index]->joints_dirty) {
		return;
	}
	settings[p_index]->joints_dirty = true;
	callable_mp(this, &BoneTwistDisperser3D::_update_joints).call_deferred(p_index);
}

void BoneTwistDisperser3D::_update_joints(int p_index) {
	Skeleton3D *sk = get_skeleton();
	int current_bone = settings[p_index]->end_bone.bone;
	int root_bone = settings[p_index]->root_bone.bone;
	if (!sk || current_bone < 0 || root_bone < 0) {
		set_joint_count(p_index, 0);
		settings[p_index]->joints_dirty = false;
		return;
	}

	// Validation.
	bool valid = false;
	while (current_bone >= 0) {
		current_bone = sk->get_bone_parent(current_bone);
		if (current_bone == root_bone) {
			valid = true;
			break;
		}
	}

	if (!valid) {
		set_joint_count(p_index, 0);
		_update_reference_bone(p_index);
		settings[p_index]->joints_dirty = false;
		ERR_FAIL_EDMSG("End bone must be a child of the root bone.");
	}

	Vector<int> new_joints;
	current_bone = settings[p_index]->end_bone.bone;
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

	_update_reference_bone(p_index);
	settings[p_index]->joints_dirty = false;
}

int BoneTwistDisperser3D::get_setting_count() const {
	return (int)settings.size();
}

void BoneTwistDisperser3D::set_setting_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);
	int delta = p_count - settings.size();
	if (delta < 0) {
		for (int i = delta; i < 0; i++) {
			memdelete(settings[settings.size() + i]);
			settings[settings.size() + i] = nullptr;
		}
	}
	settings.resize(p_count);
	delta++;
	if (delta > 1) {
		for (int i = 1; i < delta; i++) {
			settings[p_count - i] = memnew(BoneTwistDisperser3DSetting);
		}
	}
	notify_property_list_changed();
}

void BoneTwistDisperser3D::clear_settings() {
	set_setting_count(0);
}

void BoneTwistDisperser3D::_process_modification(double p_delta) {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	for (BoneTwistDisperser3DSetting *setting : settings) {
		if (!setting || setting->reference_bone.bone < 0) {
			continue;
		}
		LocalVector<DisperseJointSetting> &joints = setting->joints;
		// Calc amount.
		int actual_joint_size = setting->extend_end_bone ? (int)joints.size() : (int)joints.size() - 1;
		if (actual_joint_size <= 1) {
			continue;
		}
		if (setting->disperse_mode == DISPERSE_MODE_EVEN) {
			double div = 1.0 / actual_joint_size;
			for (int i = 0; i < actual_joint_size; i++) {
				joints[i].amount = ((double)i + 1.0) * div;
			}
		} else if (setting->disperse_mode == DISPERSE_MODE_WEIGHTED) {
			// Assign length for each bone.
			double total_length = 0.0;
			double weight_sub = 1.0 - setting->weight_position;
			if (mutable_bone_axes) {
				for (int i = 0; i < actual_joint_size; i++) {
					double length = 0.0;
					if (i == 0) {
						length = skeleton->get_bone_pose_position(joints[i + 1].joint.bone).length() * setting->weight_position;
					} else if (i == actual_joint_size - 1) {
						length = skeleton->get_bone_pose_position(joints[i].joint.bone).length() * weight_sub;
					} else {
						length = skeleton->get_bone_pose_position(joints[i].joint.bone).length() * setting->weight_position + skeleton->get_bone_pose_position(joints[i + 1].joint.bone).length() * weight_sub;
					}
					total_length += length;
					joints[i].amount = total_length;
				}
			} else {
				for (int i = 0; i < actual_joint_size; i++) {
					double length = 0.0;
					if (i == 0) {
						length = skeleton->get_bone_rest(joints[i + 1].joint.bone).origin.length() * setting->weight_position;
					} else if (i == actual_joint_size - 1) {
						length = skeleton->get_bone_rest(joints[i].joint.bone).origin.length() * weight_sub;
					} else {
						length = skeleton->get_bone_rest(joints[i].joint.bone).origin.length() * setting->weight_position + skeleton->get_bone_rest(joints[i + 1].joint.bone).origin.length() * weight_sub;
					}
					total_length += length;
					joints[i].amount = total_length;
				}
			}
			if (Math::is_zero_approx(total_length)) {
				continue;
			}
			// Normalize.
			double div = 1.0 / total_length;
			for (int i = 0; i < actual_joint_size; i++) {
				joints[i].amount *= div;
			}
		} else {
			for (int i = 0; i < actual_joint_size; i++) {
				joints[i].amount = joints[i].custom_amount;
			}
		}
		int end = actual_joint_size - 1;
		joints[end].amount -= 1.0; // Remove twist from current pose.

		// Retrieve axes.
		if (mutable_bone_axes) {
			for (int i = 0; i < end; i++) {
				joints[i].axis = skeleton->get_bone_pose_position(joints[i + 1].joint.bone).normalized();
				if (joints[i].axis.is_zero_approx() && i > 0) {
					joints[i].axis = joints[i - 1].axis;
				}
			}
		} else {
			for (int i = 0; i < end; i++) {
				joints[i].axis = skeleton->get_bone_rest(joints[i + 1].joint.bone).origin.normalized();
				if (joints[i].axis.is_zero_approx() && i > 0) {
					joints[i].axis = joints[i - 1].axis;
				}
			}
		}

		if (!setting->extend_end_bone) {
			joints[end].axis = mutable_bone_axes ? skeleton->get_bone_pose_position(setting->end_bone.bone) : skeleton->get_bone_rest(setting->end_bone.bone).origin;
			joints[end].axis.normalize();
		} else if (setting->end_bone_direction == BONE_DIRECTION_FROM_PARENT) {
			joints[end].axis = skeleton->get_bone_rest(setting->end_bone.bone).basis.xform_inv(mutable_bone_axes ? skeleton->get_bone_pose_position(setting->end_bone.bone) : skeleton->get_bone_rest(setting->end_bone.bone).origin);
			joints[end].axis.normalize();
		} else {
			joints[end].axis = get_vector_from_bone_axis(static_cast<BoneAxis>((int)setting->end_bone_direction));
		}
		if (joints[end].axis.is_zero_approx() && end > 0) {
			joints[end].axis = joints[end - 1].axis;
		}

		// Extract twist.
		Quaternion twist_rest = setting->twist_from_rest ? skeleton->get_bone_rest(setting->reference_bone.bone).basis.get_rotation_quaternion() : setting->twist_from.normalized();
		Quaternion ref_rot = twist_rest.inverse() * skeleton->get_bone_pose_rotation(setting->reference_bone.bone);
		ref_rot.normalize();
		double twist = get_roll_angle(ref_rot, joints[end].axis);

		// Apply twist for each bone by their amount.
		// Twist parent, then cancel all twists caused by this modifier in child, and re-apply accumulated twist.
		Quaternion prev_rot;
		if (mutable_bone_axes) {
			for (int i = 0; i < actual_joint_size; i++) {
				int bn = joints[i].joint.bone;
				Quaternion cur_rot = Quaternion(joints[i].axis, twist * joints[i].amount);
				skeleton->set_bone_pose_rotation(bn, prev_rot.inverse() * skeleton->get_bone_pose_rotation(bn) * cur_rot);
				prev_rot = cur_rot;
			}
		} else {
			for (int i = 0; i < actual_joint_size; i++) {
				int bn = joints[i].joint.bone;
				Quaternion cur_rot = Quaternion(joints[i].axis, twist * joints[i].amount);
				skeleton->set_bone_pose_rotation(bn, prev_rot.inverse() * skeleton->get_bone_pose_rotation(bn) * cur_rot);
				prev_rot = cur_rot;
			}
		}
	}
}

BoneTwistDisperser3D::~BoneTwistDisperser3D() {
	clear_settings();
}
