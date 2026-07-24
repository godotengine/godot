/**************************************************************************/
/*  bone_spreader_3d.cpp                                                  */
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

#include "bone_spreader_3d.h"

#include "core/object/class_db.h"

bool BoneSpreader3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "bone_name") {
			set_bone_name(which, p_value);
		} else if (what == "bone") {
			set_bone(which, p_value);
		} else if (what == "use_bone_skin_scale") {
			set_use_bone_skin_scale(which, p_value);
		} else if (what == "bone_scale") {
			set_bone_scale(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool BoneSpreader3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "bone_name") {
			r_ret = get_bone_name(which);
		} else if (what == "bone") {
			r_ret = get_bone(which);
		} else if (what == "use_bone_skin_scale") {
			r_ret = is_using_bone_skin_scale(which);
		} else if (what == "bone_scale") {
			r_ret = get_bone_scale(which);
		} else {
			return false;
		}
	}
	return true;
}

void BoneSpreader3D::_get_property_list(List<PropertyInfo> *p_list) const {
	String enum_hint;
	Skeleton3D *skeleton = get_skeleton();
	if (skeleton) {
		enum_hint = skeleton->get_concatenated_bone_names();
	}

	LocalVector<PropertyInfo> props;

	for (uint32_t i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		props.push_back(PropertyInfo(Variant::STRING_NAME, path + "bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		props.push_back(PropertyInfo(Variant::INT, path + "bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		props.push_back(PropertyInfo(Variant::BOOL, path + "use_bone_skin_scale"));
		props.push_back(PropertyInfo(Variant::VECTOR3, path + "bone_scale"));
	}

	for (PropertyInfo &p : props) {
		_validate_dynamic_prop(p);
		p_list->push_back(p);
	}
}

void BoneSpreader3D::_validate_dynamic_prop(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() > 2 && split[0] == "settings") {
		int which = split[1].to_int();
		if (split[2] == "bone_scale" && is_using_bone_skin_scale(which)) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
	}
}

int BoneSpreader3D::get_setting_size() {
	return settings.size();
}

void BoneSpreader3D::set_setting_size(int p_size) {
	ERR_FAIL_COND(p_size < 0);
	settings.resize(p_size);
	notify_property_list_changed();
}

void BoneSpreader3D::clear_settings() {
	settings.clear();
}

void BoneSpreader3D::set_bone_name(int p_index, const StringName &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index].bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_bone(p_index, sk->find_bone(settings[p_index].bone_name));
	}
}

StringName BoneSpreader3D::get_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), StringName());
	return settings[p_index].bone_name;
}

void BoneSpreader3D::set_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index].bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index].bone <= -1 || settings[p_index].bone >= sk->get_bone_count()) {
			WARN_PRINT("Apply bone index out of range!");
			settings[p_index].bone = -1;
		} else {
			settings[p_index].bone_name = sk->get_bone_name(settings[p_index].bone);
		}
	}
}

int BoneSpreader3D::get_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return settings[p_index].bone;
}

void BoneSpreader3D::set_use_bone_skin_scale(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index].use_bone_skin_scale = p_enabled;
	notify_property_list_changed();
}

bool BoneSpreader3D::is_using_bone_skin_scale(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), false);
	return settings[p_index].use_bone_skin_scale;
}

void BoneSpreader3D::set_bone_scale(int p_index, const Vector3 &p_scale) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ERR_FAIL_COND_MSG(Math::is_zero_approx(p_scale.x * p_scale.y * p_scale.z), "Scale must not be zero.");
	settings[p_index].scale = p_scale;
}

Vector3 BoneSpreader3D::get_bone_scale(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3(1, 1, 1));
	return settings[p_index].scale;
}

void BoneSpreader3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bone_name", "index", "bone_name"), &BoneSpreader3D::set_bone_name);
	ClassDB::bind_method(D_METHOD("get_bone_name", "index"), &BoneSpreader3D::get_bone_name);
	ClassDB::bind_method(D_METHOD("set_bone", "index", "bone"), &BoneSpreader3D::set_bone);
	ClassDB::bind_method(D_METHOD("get_bone", "index"), &BoneSpreader3D::get_bone);
	ClassDB::bind_method(D_METHOD("set_use_bone_skin_scale", "index", "enabled"), &BoneSpreader3D::set_use_bone_skin_scale);
	ClassDB::bind_method(D_METHOD("is_using_bone_skin_scale", "index"), &BoneSpreader3D::is_using_bone_skin_scale);
	ClassDB::bind_method(D_METHOD("set_bone_scale", "index", "scale"), &BoneSpreader3D::set_bone_scale);
	ClassDB::bind_method(D_METHOD("get_bone_scale", "index"), &BoneSpreader3D::get_bone_scale);

	ClassDB::bind_method(D_METHOD("set_setting_size", "size"), &BoneSpreader3D::set_setting_size);
	ClassDB::bind_method(D_METHOD("get_setting_size"), &BoneSpreader3D::get_setting_size);
	ClassDB::bind_method(D_METHOD("clear_setting"), &BoneSpreader3D::clear_settings);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "setting_size", PROPERTY_HINT_RANGE, "0,1000,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Settings,settings/"), "set_setting_size", "get_setting_size");
}

void BoneSpreader3D::_process_modification(double p_delta) {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	for (const BoneSpreader3DSetting &setting : settings) {
		int bone = setting.bone;
		if (bone < 0) {
			continue;
		}

		Vector3 scl = setting.use_bone_skin_scale ? skeleton->get_bone_skin_scale(bone) : setting.scale;
		Vector<int> children = skeleton->get_bone_children(bone);
		for (int c : children) {
			skeleton->set_bone_pose_position(c, skeleton->get_bone_pose_position(c) * scl);
		}
	}
}
