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
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "amount") {
			set_amount(which, p_value);
		} else if (what == "reference_type") {
			set_reference_type(which, static_cast<ReferenceType>((int)p_value));
		} else if (what == "apply_bone_name") {
			set_apply_bone_name(which, p_value);
		} else if (what == "reference_bone_name") {
			set_reference_bone_name(which, p_value);
		} else if (what == "apply_bone") {
			set_apply_bone(which, p_value);
		} else if (what == "reference_bone") {
			set_reference_bone(which, p_value);
		} else if (what == "reference_node") {
			set_reference_node(which, p_value);
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
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "amount") {
			r_ret = get_amount(which);
		} else if (what == "reference_type") {
			r_ret = (int)get_reference_type(which);
		} else if (what == "apply_bone_name") {
			r_ret = get_apply_bone_name(which);
		} else if (what == "reference_bone_name") {
			r_ret = get_reference_bone_name(which);
		} else if (what == "apply_bone") {
			r_ret = get_apply_bone(which);
		} else if (what == "reference_bone") {
			r_ret = get_reference_bone(which);
		} else if (what == "reference_node") {
			r_ret = get_reference_node(which);
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

	LocalVector<PropertyInfo> props;

	for (uint32_t i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		props.push_back(PropertyInfo(Variant::FLOAT, path + "amount", PROPERTY_HINT_RANGE, "0,1,0.001"));
		props.push_back(PropertyInfo(Variant::STRING, path + "apply_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		props.push_back(PropertyInfo(Variant::INT, path + "apply_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		props.push_back(PropertyInfo(Variant::INT, path + "reference_type", PROPERTY_HINT_ENUM, "Bone,Node"));
		props.push_back(PropertyInfo(Variant::STRING, path + "reference_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		props.push_back(PropertyInfo(Variant::INT, path + "reference_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		props.push_back(PropertyInfo(Variant::NODE_PATH, path + "reference_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"));
	}

	for (PropertyInfo &p : props) {
		_validate_dynamic_prop(p);
		p_list->push_back(p);
	}
}

void BoneConstraint3D::_validate_dynamic_prop(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() > 2 && split[0] == "settings") {
		int which = split[1].to_int();
		if (split[2].begins_with("reference_bone") && get_reference_type(which) != REFERENCE_TYPE_BONE) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
		if (split[2].begins_with("reference_node") && get_reference_type(which) != REFERENCE_TYPE_NODE) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
	}
}

void BoneConstraint3D::set_setting_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);

	int delta = p_count - (int)settings.size();
	if (delta < 0) {
		for (int i = delta; i < 0; i++) {
			memdelete(settings[(int)settings.size() + i]);
			settings[(int)settings.size() + i] = nullptr;
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
	return (int)settings.size();
}

void BoneConstraint3D::_validate_setting(int p_index) {
	settings[p_index] = memnew(BoneConstraint3DSetting);
}

void BoneConstraint3D::clear_settings() {
	set_setting_count(0);
}

void BoneConstraint3D::set_amount(int p_index, float p_amount) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->amount = p_amount;
}

float BoneConstraint3D::get_amount(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0.0);
	return settings[p_index]->amount;
}

void BoneConstraint3D::set_apply_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->apply_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_apply_bone(p_index, sk->find_bone(settings[p_index]->apply_bone_name));
	}
}

String BoneConstraint3D::get_apply_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	return settings[p_index]->apply_bone_name;
}

void BoneConstraint3D::set_apply_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->apply_bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index]->apply_bone <= -1 || settings[p_index]->apply_bone >= sk->get_bone_count()) {
			WARN_PRINT("Apply bone index out of range!");
			settings[p_index]->apply_bone = -1;
		} else {
			settings[p_index]->apply_bone_name = sk->get_bone_name(settings[p_index]->apply_bone);
		}
	}
}

int BoneConstraint3D::get_apply_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return settings[p_index]->apply_bone;
}

void BoneConstraint3D::set_reference_type(int p_index, ReferenceType p_type) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->reference_type = p_type;
	notify_property_list_changed();
}

BoneConstraint3D::ReferenceType BoneConstraint3D::get_reference_type(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), REFERENCE_TYPE_BONE);
	return settings[p_index]->reference_type;
}

void BoneConstraint3D::set_reference_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->reference_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_reference_bone(p_index, sk->find_bone(settings[p_index]->reference_bone_name));
	}
}

String BoneConstraint3D::get_reference_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), String());
	return settings[p_index]->reference_bone_name;
}

void BoneConstraint3D::set_reference_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
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
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return settings[p_index]->reference_bone;
}

void BoneConstraint3D::set_reference_node(int p_index, const NodePath &p_node) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index]->reference_node = p_node;
}

NodePath BoneConstraint3D::get_reference_node(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), NodePath());
	return settings[p_index]->reference_node;
}

void BoneConstraint3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_amount", "index", "amount"), &BoneConstraint3D::set_amount);
	ClassDB::bind_method(D_METHOD("get_amount", "index"), &BoneConstraint3D::get_amount);
	ClassDB::bind_method(D_METHOD("set_apply_bone_name", "index", "bone_name"), &BoneConstraint3D::set_apply_bone_name);
	ClassDB::bind_method(D_METHOD("get_apply_bone_name", "index"), &BoneConstraint3D::get_apply_bone_name);
	ClassDB::bind_method(D_METHOD("set_apply_bone", "index", "bone"), &BoneConstraint3D::set_apply_bone);
	ClassDB::bind_method(D_METHOD("get_apply_bone", "index"), &BoneConstraint3D::get_apply_bone);
	ClassDB::bind_method(D_METHOD("set_reference_type", "index", "type"), &BoneConstraint3D::set_reference_type);
	ClassDB::bind_method(D_METHOD("get_reference_type", "index"), &BoneConstraint3D::get_reference_type);
	ClassDB::bind_method(D_METHOD("set_reference_bone_name", "index", "bone_name"), &BoneConstraint3D::set_reference_bone_name);
	ClassDB::bind_method(D_METHOD("get_reference_bone_name", "index"), &BoneConstraint3D::get_reference_bone_name);
	ClassDB::bind_method(D_METHOD("set_reference_bone", "index", "bone"), &BoneConstraint3D::set_reference_bone);
	ClassDB::bind_method(D_METHOD("get_reference_bone", "index"), &BoneConstraint3D::get_reference_bone);
	ClassDB::bind_method(D_METHOD("set_reference_node", "index", "node"), &BoneConstraint3D::set_reference_node);
	ClassDB::bind_method(D_METHOD("get_reference_node", "index"), &BoneConstraint3D::get_reference_node);

	ClassDB::bind_method(D_METHOD("set_setting_count", "count"), &BoneConstraint3D::set_setting_count);
	ClassDB::bind_method(D_METHOD("get_setting_count"), &BoneConstraint3D::get_setting_count);
	ClassDB::bind_method(D_METHOD("clear_setting"), &BoneConstraint3D::clear_settings);

	BIND_ENUM_CONSTANT(REFERENCE_TYPE_BONE);
	BIND_ENUM_CONSTANT(REFERENCE_TYPE_NODE);
}

void BoneConstraint3D::_validate_bone_names() {
	for (int i = 0; i < (int)settings.size(); i++) {
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

	for (int i = 0; i < (int)settings.size(); i++) {
		float amount = settings[i]->amount;
		if (amount <= 0) {
			continue;
		}

		int apply_bone = settings[i]->apply_bone;
		if (apply_bone < 0) {
			continue;
		}

		if (settings[i]->reference_type == REFERENCE_TYPE_BONE) {
			int reference_bone = settings[i]->reference_bone;
			if (reference_bone < 0) {
				continue;
			}
			_process_constraint_by_bone(i, skeleton, apply_bone, reference_bone, amount);
		} else {
			NodePath pt = settings[i]->reference_node;
			if (pt.is_empty()) {
				continue;
			}
			_process_constraint_by_node(i, skeleton, apply_bone, pt, amount);
		}
	}
}

void BoneConstraint3D::_process_constraint_by_bone(int p_index, Skeleton3D *p_skeleton, int p_apply_bone, int p_reference_bone, float p_amount) {
	//
}

void BoneConstraint3D::_process_constraint_by_node(int p_index, Skeleton3D *p_skeleton, int p_apply_bone, const NodePath &p_reference_node, float p_amount) {
	//
}

BoneConstraint3D::~BoneConstraint3D() {
	clear_settings();
}
