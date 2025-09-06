/**************************************************************************/
/*  bone_constraint_3d.cpp                                                */
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

#include "bone_constraint_3d.h"

bool BoneConstraint3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, settings.size(), false);

		if (what == "amount") {
			set_amount(which, p_value);
		} else if (what == "apply_bone_name") {
			set_apply_bone_name(which, p_value);
		} else if (what == "reference_bone_name") {
			set_reference_bone_name(which, p_value);
		} else if (what == "apply_bone") {
			set_apply_bone(which, p_value);
		} else if (what == "reference_bone") {
			set_reference_bone(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool BoneConstraint3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, settings.size(), false);

		if (what == "amount") {
			r_ret = get_amount(which);
		} else if (what == "apply_bone_name") {
			r_ret = get_apply_bone_name(which);
		} else if (what == "reference_bone_name") {
			r_ret = get_reference_bone_name(which);
		} else if (what == "apply_bone") {
			r_ret = get_apply_bone(which);
		} else if (what == "reference_bone") {
			r_ret = get_reference_bone(which);
		} else {
			return false;
		}
	}
	return true;
}

void BoneConstraint3D::get_property_list(List<PropertyInfo> *p_list) const {
	String enum_hint;
	Skeleton3D *skeleton = get_skeleton();
	if (skeleton) {
		enum_hint = skeleton->get_concatenated_bone_names();
	}

	for (int i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::FLOAT, path + "amount", PROPERTY_HINT_RANGE, "0,1,0.001"));
		p_list->push_back(PropertyInfo(Variant::STRING, path + "apply_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		p_list->push_back(PropertyInfo(Variant::INT, path + "apply_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::STRING, path + "reference_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		p_list->push_back(PropertyInfo(Variant::INT, path + "reference_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	}
}

void BoneConstraint3D::set_setting_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);

	int delta = p_count - settings.size();
	if (delta < 0) {
		for (int i = delta; i < 0; i++) {
			memdelete(settings[settings.size() + i]);
		}
	}
	settings.resize(p_count);
	delta++;

	if (delta > 1) {
		for (int i = 1; i < delta; i++) {
			_validate_setting(p_count - i);
		}
	}

	notify_property_list_changed();
}

int BoneConstraint3D::get_setting_count() const {
	return settings.size();
}

void BoneConstraint3D::_validate_setting(int p_index) {
	settings.write[p_index] = memnew(BoneConstraint3DSetting);
}

void BoneConstraint3D::clear_settings() {
	set_setting_count(0);
}

void BoneConstraint3D::set_amount(int p_index, float p_amount) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->amount = p_amount;
}

float BoneConstraint3D::get_amount(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0.0);
	return settings[p_index]->amount;
}

void BoneConstraint3D::set_apply_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->apply_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_apply_bone(p_index, sk->find_bone(settings[p_index]->apply_bone_name));
	}
}

String BoneConstraint3D::get_apply_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	return settings[p_index]->apply_bone_name;
}

void BoneConstraint3D::set_apply_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->apply_bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index]->apply_bone <= -1 || settings[p_index]->apply_bone >= sk->get_bone_count()) {
			WARN_PRINT("apply bone index out of range!");
			settings[p_index]->apply_bone = -1;
		} else {
			settings[p_index]->apply_bone_name = sk->get_bone_name(settings[p_index]->apply_bone);
		}
	}
}

int BoneConstraint3D::get_apply_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	return settings[p_index]->apply_bone;
}

void BoneConstraint3D::set_reference_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->reference_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_reference_bone(p_index, sk->find_bone(settings[p_index]->reference_bone_name));
	}
}

String BoneConstraint3D::get_reference_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	return settings[p_index]->reference_bone_name;
}

void BoneConstraint3D::set_reference_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->reference_bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index]->reference_bone <= -1 || settings[p_index]->reference_bone >= sk->get_bone_count()) {
			WARN_PRINT("reference bone index out of range!");
			settings[p_index]->reference_bone = -1;
		} else {
			settings[p_index]->reference_bone_name = sk->get_bone_name(settings[p_index]->reference_bone);
		}
	}
}

int BoneConstraint3D::get_reference_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	return settings[p_index]->reference_bone;
}

void BoneConstraint3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_amount", "index", "amount"), &BoneConstraint3D::set_amount);
	ClassDB::bind_method(D_METHOD("get_amount", "index"), &BoneConstraint3D::get_amount);
	ClassDB::bind_method(D_METHOD("set_apply_bone_name", "index", "bone_name"), &BoneConstraint3D::set_apply_bone_name);
	ClassDB::bind_method(D_METHOD("get_apply_bone_name", "index"), &BoneConstraint3D::get_apply_bone_name);
	ClassDB::bind_method(D_METHOD("set_apply_bone", "index", "bone"), &BoneConstraint3D::set_apply_bone);
	ClassDB::bind_method(D_METHOD("get_apply_bone", "index"), &BoneConstraint3D::get_apply_bone);
	ClassDB::bind_method(D_METHOD("set_reference_bone_name", "index", "bone_name"), &BoneConstraint3D::set_reference_bone_name);
	ClassDB::bind_method(D_METHOD("get_reference_bone_name", "index"), &BoneConstraint3D::get_reference_bone_name);
	ClassDB::bind_method(D_METHOD("set_reference_bone", "index", "bone"), &BoneConstraint3D::set_reference_bone);
	ClassDB::bind_method(D_METHOD("get_reference_bone", "index"), &BoneConstraint3D::get_reference_bone);

	ClassDB::bind_method(D_METHOD("set_setting_count", "count"), &BoneConstraint3D::set_setting_count);
	ClassDB::bind_method(D_METHOD("get_setting_count"), &BoneConstraint3D::get_setting_count);
	ClassDB::bind_method(D_METHOD("clear_setting"), &BoneConstraint3D::clear_settings);
}

void BoneConstraint3D::_validate_bone_names() {
	for (int i = 0; i < settings.size(); i++) {
		// Prior bone name.
		if (!settings[i]->apply_bone_name.is_empty()) {
			set_apply_bone_name(i, settings[i]->apply_bone_name);
		} else if (settings[i]->apply_bone != -1) {
			set_apply_bone(i, settings[i]->apply_bone);
		}
		// Prior bone name.
		if (!settings[i]->reference_bone_name.is_empty()) {
			set_reference_bone_name(i, settings[i]->reference_bone_name);
		} else if (settings[i]->reference_bone != -1) {
			set_reference_bone(i, settings[i]->reference_bone);
		}
	}
}

void BoneConstraint3D::_process_modification(double p_delta) {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	for (int i = 0; i < settings.size(); i++) {
		int apply_bone = settings[i]->apply_bone;
		if (apply_bone < 0) {
			continue;
		}

		int reference_bone = settings[i]->reference_bone;
		if (reference_bone < 0) {
			continue;
		}

		float amount = settings[i]->amount;
		if (amount <= 0) {
			continue;
		}

		_process_constraint(i, skeleton, apply_bone, reference_bone, amount);
	}
}

void BoneConstraint3D::_process_constraint(int p_index, Skeleton3D *p_skeleton, int p_apply_bone, int p_reference_bone, float p_amount) {
	//
}

double BoneConstraint3D::symmetrize_angle(double p_angle) {
	double angle = Math::fposmod(p_angle, Math::TAU);
	return angle > Math::PI ? angle - Math::TAU : angle;
}

double BoneConstraint3D::get_roll_angle(const Quaternion &p_rotation, const Vector3 &p_roll_axis) {
	// Ensure roll axis is normalized.
	Vector3 roll_axis = p_roll_axis.normalized();

	// Project the quaternion rotation onto the roll axis.
	// This gives us the component of rotation around that axis.
	double dot = p_rotation.x * roll_axis.x +
			p_rotation.y * roll_axis.y +
			p_rotation.z * roll_axis.z;

	// Create a quaternion representing just the roll component.
	Quaternion roll_component;
	roll_component.x = roll_axis.x * dot;
	roll_component.y = roll_axis.y * dot;
	roll_component.z = roll_axis.z * dot;
	roll_component.w = p_rotation.w;

	// Normalize this component.
	double length = roll_component.length();
	if (length > CMP_EPSILON) {
		roll_component = roll_component / length;
	} else {
		return 0.0;
	}

	// Extract the angle.
	double angle = 2.0 * Math::acos(CLAMP(roll_component.w, -1.0, 1.0));

	// Determine the sign.
	double direction = (roll_component.x * roll_axis.x + roll_component.y * roll_axis.y + roll_component.z * roll_axis.z > 0) ? 1.0 : -1.0;

	return symmetrize_angle(angle * direction);
}

BoneConstraint3D::~BoneConstraint3D() {
	clear_settings();
}
