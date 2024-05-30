/*************************************************************************/
/*  retarget_profile.cpp                                                 */
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

#include "retarget_profile.h"

bool RetargetProfile::_set(const StringName &p_path, const Variant &p_value) {
	ERR_FAIL_COND_V(is_read_only, false);
	String path = p_path;

	if (path.begins_with("global_transform_targets/")) {
		int which = path.get_slicec('/', 1).to_int();
		if (which >= global_transform_targets.size()) {
			add_global_transform_target(p_value);
		} else {
			set_global_transform_target(which, p_value);
		}
	}

	if (path.begins_with("local_transform_targets/")) {
		int which = path.get_slicec('/', 1).to_int();
		if (which >= local_transform_targets.size()) {
			add_local_transform_target(p_value);
		} else {
			set_local_transform_target(which, p_value);
		}
	}

	if (path.begins_with("absolute_transform_targets/")) {
		int which = path.get_slicec('/', 1).to_int();
		if (which >= absolute_transform_targets.size()) {
			add_absolute_transform_target(p_value);
		} else {
			set_absolute_transform_target(which, p_value);
		}
	}

	return true;
}

bool RetargetProfile::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("global_transform_targets/")) {
		int which = path.get_slicec('/', 1).to_int();
		ERR_FAIL_INDEX_V(which, global_transform_targets.size(), false);
		r_ret = get_global_transform_target(which);
	}

	if (path.begins_with("local_transform_targets/")) {
		int which = path.get_slicec('/', 1).to_int();
		ERR_FAIL_INDEX_V(which, local_transform_targets.size(), false);
		r_ret = get_local_transform_target(which);
	}

	if (path.begins_with("absolute_transform_targets/")) {
		int which = path.get_slicec('/', 1).to_int();
		ERR_FAIL_INDEX_V(which, absolute_transform_targets.size(), false);
		r_ret = get_absolute_transform_target(which);
	}

	return true;
}

void RetargetProfile::set_label_for_animation_name(const String p_label_for_animation_name) {
	label_for_animation_name = p_label_for_animation_name;
}

String RetargetProfile::get_label_for_animation_name() const {
	return label_for_animation_name;
}

void RetargetProfile::set_global_transform_target_size(int p_size) {
	if (is_read_only) {
		return;
	}
	ERR_FAIL_COND(p_size < 0);
	global_transform_targets.resize(p_size);
	notify_property_list_changed();
}

int RetargetProfile::get_global_transform_target_size() const {
	return global_transform_targets.size();
}

void RetargetProfile::add_global_transform_target(const StringName p_bone_name) {
	if (is_read_only) {
		return;
	}
	global_transform_targets.push_back(p_bone_name);
}

void RetargetProfile::set_global_transform_target(int p_idx, const StringName p_bone_name) {
	if (is_read_only) {
		return;
	}
	ERR_FAIL_INDEX(p_idx, global_transform_targets.size());
	global_transform_targets.write[p_idx] = p_bone_name;
}

StringName RetargetProfile::get_global_transform_target(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, global_transform_targets.size(), StringName());
	return global_transform_targets[p_idx];
}

bool RetargetProfile::has_global_transform_target(const StringName p_bone_name) {
	return global_transform_targets.has(p_bone_name);
}

void RetargetProfile::set_local_transform_target_size(int p_size) {
	if (is_read_only) {
		return;
	}
	ERR_FAIL_COND(p_size < 0);
	local_transform_targets.resize(p_size);
	notify_property_list_changed();
}

int RetargetProfile::get_local_transform_target_size() const {
	return local_transform_targets.size();
}

void RetargetProfile::add_local_transform_target(const StringName p_bone_name) {
	if (is_read_only) {
		return;
	}
	local_transform_targets.push_back(p_bone_name);
}

void RetargetProfile::set_local_transform_target(int p_idx, const StringName p_bone_name) {
	if (is_read_only) {
		return;
	}
	ERR_FAIL_INDEX(p_idx, local_transform_targets.size());
	local_transform_targets.write[p_idx] = p_bone_name;
}

StringName RetargetProfile::get_local_transform_target(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, local_transform_targets.size(), StringName());
	return local_transform_targets[p_idx];
}

bool RetargetProfile::has_local_transform_target(const StringName p_bone_name) {
	return local_transform_targets.has(p_bone_name);
}

void RetargetProfile::set_absolute_transform_target_size(int p_size) {
	if (is_read_only) {
		return;
	}
	ERR_FAIL_COND(p_size < 0);
	absolute_transform_targets.resize(p_size);
	notify_property_list_changed();
}

int RetargetProfile::get_absolute_transform_target_size() const {
	return absolute_transform_targets.size();
}

void RetargetProfile::add_absolute_transform_target(const StringName p_bone_name) {
	if (is_read_only) {
		return;
	}
	absolute_transform_targets.push_back(p_bone_name);
}

void RetargetProfile::set_absolute_transform_target(int p_idx, const StringName p_bone_name) {
	if (is_read_only) {
		return;
	}
	ERR_FAIL_INDEX(p_idx, absolute_transform_targets.size());
	absolute_transform_targets.write[p_idx] = p_bone_name;
}

StringName RetargetProfile::get_absolute_transform_target(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, absolute_transform_targets.size(), StringName());
	return absolute_transform_targets[p_idx];
}

bool RetargetProfile::has_absolute_transform_target(const StringName p_bone_name) {
	return absolute_transform_targets.has(p_bone_name);
}

void RetargetProfile::_validate_property(PropertyInfo &p_property) const {
	if (is_read_only) {
		if (p_property.name == ("label_for_animation_name") || p_property.name == ("global_transform_target_size") || p_property.name == ("local_transform_target_size") || p_property.name == ("absolute_transform_target_size")) {
			p_property.usage = PROPERTY_USAGE_NO_EDITOR;
			return;
		}
	}
}

void RetargetProfile::_get_property_list(List<PropertyInfo> *p_list) const {
	if (is_read_only) {
		return;
	}

	for (int i = 0; i < global_transform_targets.size(); i++) {
		String path = "global_transform_targets/" + itos(i);
		p_list->push_back(PropertyInfo(Variant::STRING_NAME, path));
	}

	for (int i = 0; i < local_transform_targets.size(); i++) {
		String path = "local_transform_targets/" + itos(i);
		p_list->push_back(PropertyInfo(Variant::STRING_NAME, path));
	}

	for (int i = 0; i < absolute_transform_targets.size(); i++) {
		String path = "absolute_transform_targets/" + itos(i);
		p_list->push_back(PropertyInfo(Variant::STRING_NAME, path));
	}

	for (PropertyInfo &E : *p_list) {
		_validate_property(E);
	}
}

void RetargetProfile::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_label_for_animation_name", "label_for_animation_name"), &RetargetProfile::set_label_for_animation_name);
	ClassDB::bind_method(D_METHOD("get_label_for_animation_name"), &RetargetProfile::get_label_for_animation_name);

	ClassDB::bind_method(D_METHOD("set_global_transform_target_size", "size"), &RetargetProfile::set_global_transform_target_size);
	ClassDB::bind_method(D_METHOD("get_global_transform_target_size"), &RetargetProfile::get_global_transform_target_size);
	ClassDB::bind_method(D_METHOD("add_global_transform_target", "bone_name"), &RetargetProfile::add_global_transform_target);
	ClassDB::bind_method(D_METHOD("set_global_transform_target", "idx", "bone_name"), &RetargetProfile::set_global_transform_target);
	ClassDB::bind_method(D_METHOD("get_global_transform_target", "idx"), &RetargetProfile::get_global_transform_target);
	ClassDB::bind_method(D_METHOD("has_global_transform_target", "bone_name"), &RetargetProfile::has_global_transform_target);

	ClassDB::bind_method(D_METHOD("set_local_transform_target_size", "size"), &RetargetProfile::set_local_transform_target_size);
	ClassDB::bind_method(D_METHOD("get_local_transform_target_size"), &RetargetProfile::get_local_transform_target_size);
	ClassDB::bind_method(D_METHOD("add_local_transform_target", "bone_name"), &RetargetProfile::add_local_transform_target);
	ClassDB::bind_method(D_METHOD("set_local_transform_target", "idx", "bone_name"), &RetargetProfile::set_local_transform_target);
	ClassDB::bind_method(D_METHOD("get_local_transform_target", "idx"), &RetargetProfile::get_local_transform_target);
	ClassDB::bind_method(D_METHOD("has_local_transform_target", "bone_name"), &RetargetProfile::has_local_transform_target);

	ClassDB::bind_method(D_METHOD("set_absolute_transform_target_size", "size"), &RetargetProfile::set_absolute_transform_target_size);
	ClassDB::bind_method(D_METHOD("get_absolute_transform_target_size"), &RetargetProfile::get_absolute_transform_target_size);
	ClassDB::bind_method(D_METHOD("add_absolute_transform_target", "bone_name"), &RetargetProfile::add_absolute_transform_target);
	ClassDB::bind_method(D_METHOD("set_absolute_transform_target", "idx", "bone_name"), &RetargetProfile::set_absolute_transform_target);
	ClassDB::bind_method(D_METHOD("get_absolute_transform_target", "idx"), &RetargetProfile::get_absolute_transform_target);
	ClassDB::bind_method(D_METHOD("has_absolute_transform_target", "bone_name"), &RetargetProfile::has_absolute_transform_target);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "label_for_animation_name"), "set_label_for_animation_name", "get_label_for_animation_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "global_transform_target_size", PROPERTY_HINT_RANGE, "0,100,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Global Transform Targets,global_transform_targets/"), "set_global_transform_target_size", "get_global_transform_target_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "local_transform_target_size", PROPERTY_HINT_RANGE, "0,100,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Local Transform Targets,local_transform_targets/"), "set_local_transform_target_size", "get_local_transform_target_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "absolute_transform_target_size", PROPERTY_HINT_RANGE, "0,100,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Absolute Transform Targets,absolute_transform_targets/"), "set_absolute_transform_target_size", "get_absolute_transform_target_size");
}

RetargetProfile::RetargetProfile() {
}

RetargetProfile::~RetargetProfile() {
}

// Presets.

RetargetProfileGlobalAll::RetargetProfileGlobalAll() {
	is_read_only = true;

	label_for_animation_name = "Glb-All";

	global_transform_targets.resize(56);
	global_transform_targets.write[0] = "Root";
	global_transform_targets.write[1] = "Hips";
	global_transform_targets.write[2] = "Spine";
	global_transform_targets.write[3] = "Chest";
	global_transform_targets.write[4] = "UpperChest";
	global_transform_targets.write[5] = "Neck";
	global_transform_targets.write[6] = "Head";
	global_transform_targets.write[7] = "LeftEye";
	global_transform_targets.write[8] = "RightEye";
	global_transform_targets.write[9] = "Jaw";
	global_transform_targets.write[10] = "LeftShoulder";
	global_transform_targets.write[11] = "LeftUpperArm";
	global_transform_targets.write[12] = "LeftLowerArm";
	global_transform_targets.write[13] = "LeftHand";
	global_transform_targets.write[14] = "LeftThumbMetacarpal";
	global_transform_targets.write[15] = "LeftThumbProximal";
	global_transform_targets.write[16] = "LeftThumbDistal";
	global_transform_targets.write[17] = "LeftIndexProximal";
	global_transform_targets.write[18] = "LeftIndexIntermediate";
	global_transform_targets.write[19] = "LeftIndexDistal";
	global_transform_targets.write[20] = "LeftMiddleProximal";
	global_transform_targets.write[21] = "LeftMiddleIntermediate";
	global_transform_targets.write[22] = "LeftMiddleDistal";
	global_transform_targets.write[23] = "LeftRingProximal";
	global_transform_targets.write[24] = "LeftRingIntermediate";
	global_transform_targets.write[25] = "LeftRingDistal";
	global_transform_targets.write[26] = "LeftLittleProximal";
	global_transform_targets.write[27] = "LeftLittleIntermediate";
	global_transform_targets.write[28] = "LeftLittleDistal";
	global_transform_targets.write[29] = "RightShoulder";
	global_transform_targets.write[30] = "RightUpperArm";
	global_transform_targets.write[31] = "RightLowerArm";
	global_transform_targets.write[32] = "RightHand";
	global_transform_targets.write[33] = "RightThumbMetacarpal";
	global_transform_targets.write[34] = "RightThumbProximal";
	global_transform_targets.write[35] = "RightThumbDistal";
	global_transform_targets.write[36] = "RightIndexProximal";
	global_transform_targets.write[37] = "RightIndexIntermediate";
	global_transform_targets.write[38] = "RightIndexDistal";
	global_transform_targets.write[39] = "RightMiddleProximal";
	global_transform_targets.write[40] = "RightMiddleIntermediate";
	global_transform_targets.write[41] = "RightMiddleDistal";
	global_transform_targets.write[42] = "RightRingProximal";
	global_transform_targets.write[43] = "RightRingIntermediate";
	global_transform_targets.write[44] = "RightRingDistal";
	global_transform_targets.write[45] = "RightLittleProximal";
	global_transform_targets.write[46] = "RightLittleIntermediate";
	global_transform_targets.write[47] = "RightLittleDistal";
	global_transform_targets.write[48] = "LeftUpperLeg";
	global_transform_targets.write[49] = "LeftLowerLeg";
	global_transform_targets.write[50] = "LeftFoot";
	global_transform_targets.write[51] = "LeftToes";
	global_transform_targets.write[52] = "RightUpperLeg";
	global_transform_targets.write[53] = "RightLowerLeg";
	global_transform_targets.write[54] = "RightFoot";
	global_transform_targets.write[55] = "RightToes";
}

RetargetProfileGlobalAll::~RetargetProfileGlobalAll() {
}

RetargetProfileLocalAll::RetargetProfileLocalAll() {
	is_read_only = true;

	label_for_animation_name = "Loc-All";

	local_transform_targets.resize(56);
	local_transform_targets.write[0] = "Root";
	local_transform_targets.write[1] = "Hips";
	local_transform_targets.write[2] = "Spine";
	local_transform_targets.write[3] = "Chest";
	local_transform_targets.write[4] = "UpperChest";
	local_transform_targets.write[5] = "Neck";
	local_transform_targets.write[6] = "Head";
	local_transform_targets.write[7] = "LeftEye";
	local_transform_targets.write[8] = "RightEye";
	local_transform_targets.write[9] = "Jaw";
	local_transform_targets.write[10] = "LeftShoulder";
	local_transform_targets.write[11] = "LeftUpperArm";
	local_transform_targets.write[12] = "LeftLowerArm";
	local_transform_targets.write[13] = "LeftHand";
	local_transform_targets.write[14] = "LeftThumbMetacarpal";
	local_transform_targets.write[15] = "LeftThumbProximal";
	local_transform_targets.write[16] = "LeftThumbDistal";
	local_transform_targets.write[17] = "LeftIndexProximal";
	local_transform_targets.write[18] = "LeftIndexIntermediate";
	local_transform_targets.write[19] = "LeftIndexDistal";
	local_transform_targets.write[20] = "LeftMiddleProximal";
	local_transform_targets.write[21] = "LeftMiddleIntermediate";
	local_transform_targets.write[22] = "LeftMiddleDistal";
	local_transform_targets.write[23] = "LeftRingProximal";
	local_transform_targets.write[24] = "LeftRingIntermediate";
	local_transform_targets.write[25] = "LeftRingDistal";
	local_transform_targets.write[26] = "LeftLittleProximal";
	local_transform_targets.write[27] = "LeftLittleIntermediate";
	local_transform_targets.write[28] = "LeftLittleDistal";
	local_transform_targets.write[29] = "RightShoulder";
	local_transform_targets.write[30] = "RightUpperArm";
	local_transform_targets.write[31] = "RightLowerArm";
	local_transform_targets.write[32] = "RightHand";
	local_transform_targets.write[33] = "RightThumbMetacarpal";
	local_transform_targets.write[34] = "RightThumbProximal";
	local_transform_targets.write[35] = "RightThumbDistal";
	local_transform_targets.write[36] = "RightIndexProximal";
	local_transform_targets.write[37] = "RightIndexIntermediate";
	local_transform_targets.write[38] = "RightIndexDistal";
	local_transform_targets.write[39] = "RightMiddleProximal";
	local_transform_targets.write[40] = "RightMiddleIntermediate";
	local_transform_targets.write[41] = "RightMiddleDistal";
	local_transform_targets.write[42] = "RightRingProximal";
	local_transform_targets.write[43] = "RightRingIntermediate";
	local_transform_targets.write[44] = "RightRingDistal";
	local_transform_targets.write[45] = "RightLittleProximal";
	local_transform_targets.write[46] = "RightLittleIntermediate";
	local_transform_targets.write[47] = "RightLittleDistal";
	local_transform_targets.write[48] = "LeftUpperLeg";
	local_transform_targets.write[49] = "LeftLowerLeg";
	local_transform_targets.write[50] = "LeftFoot";
	local_transform_targets.write[51] = "LeftToes";
	local_transform_targets.write[52] = "RightUpperLeg";
	local_transform_targets.write[53] = "RightLowerLeg";
	local_transform_targets.write[54] = "RightFoot";
	local_transform_targets.write[55] = "RightToes";
}

RetargetProfileLocalAll::~RetargetProfileLocalAll() {
}

RetargetProfileAbsoluteAll::RetargetProfileAbsoluteAll() {
	is_read_only = true;

	label_for_animation_name = "Abs-All";

	absolute_transform_targets.resize(56);
	absolute_transform_targets.write[0] = "Root";
	absolute_transform_targets.write[1] = "Hips";
	absolute_transform_targets.write[2] = "Spine";
	absolute_transform_targets.write[3] = "Chest";
	absolute_transform_targets.write[4] = "UpperChest";
	absolute_transform_targets.write[5] = "Neck";
	absolute_transform_targets.write[6] = "Head";
	absolute_transform_targets.write[7] = "LeftEye";
	absolute_transform_targets.write[8] = "RightEye";
	absolute_transform_targets.write[9] = "Jaw";
	absolute_transform_targets.write[10] = "LeftShoulder";
	absolute_transform_targets.write[11] = "LeftUpperArm";
	absolute_transform_targets.write[12] = "LeftLowerArm";
	absolute_transform_targets.write[13] = "LeftHand";
	absolute_transform_targets.write[14] = "LeftThumbMetacarpal";
	absolute_transform_targets.write[15] = "LeftThumbProximal";
	absolute_transform_targets.write[16] = "LeftThumbDistal";
	absolute_transform_targets.write[17] = "LeftIndexProximal";
	absolute_transform_targets.write[18] = "LeftIndexIntermediate";
	absolute_transform_targets.write[19] = "LeftIndexDistal";
	absolute_transform_targets.write[20] = "LeftMiddleProximal";
	absolute_transform_targets.write[21] = "LeftMiddleIntermediate";
	absolute_transform_targets.write[22] = "LeftMiddleDistal";
	absolute_transform_targets.write[23] = "LeftRingProximal";
	absolute_transform_targets.write[24] = "LeftRingIntermediate";
	absolute_transform_targets.write[25] = "LeftRingDistal";
	absolute_transform_targets.write[26] = "LeftLittleProximal";
	absolute_transform_targets.write[27] = "LeftLittleIntermediate";
	absolute_transform_targets.write[28] = "LeftLittleDistal";
	absolute_transform_targets.write[29] = "RightShoulder";
	absolute_transform_targets.write[30] = "RightUpperArm";
	absolute_transform_targets.write[31] = "RightLowerArm";
	absolute_transform_targets.write[32] = "RightHand";
	absolute_transform_targets.write[33] = "RightThumbMetacarpal";
	absolute_transform_targets.write[34] = "RightThumbProximal";
	absolute_transform_targets.write[35] = "RightThumbDistal";
	absolute_transform_targets.write[36] = "RightIndexProximal";
	absolute_transform_targets.write[37] = "RightIndexIntermediate";
	absolute_transform_targets.write[38] = "RightIndexDistal";
	absolute_transform_targets.write[39] = "RightMiddleProximal";
	absolute_transform_targets.write[40] = "RightMiddleIntermediate";
	absolute_transform_targets.write[41] = "RightMiddleDistal";
	absolute_transform_targets.write[42] = "RightRingProximal";
	absolute_transform_targets.write[43] = "RightRingIntermediate";
	absolute_transform_targets.write[44] = "RightRingDistal";
	absolute_transform_targets.write[45] = "RightLittleProximal";
	absolute_transform_targets.write[46] = "RightLittleIntermediate";
	absolute_transform_targets.write[47] = "RightLittleDistal";
	absolute_transform_targets.write[48] = "LeftUpperLeg";
	absolute_transform_targets.write[49] = "LeftLowerLeg";
	absolute_transform_targets.write[50] = "LeftFoot";
	absolute_transform_targets.write[51] = "LeftToes";
	absolute_transform_targets.write[52] = "RightUpperLeg";
	absolute_transform_targets.write[53] = "RightLowerLeg";
	absolute_transform_targets.write[54] = "RightFoot";
	absolute_transform_targets.write[55] = "RightToes";
}

RetargetProfileAbsoluteAll::~RetargetProfileAbsoluteAll() {
}

RetargetProfileLocalFingersGlobalOthers::RetargetProfileLocalFingersGlobalOthers() {
	is_read_only = true;

	label_for_animation_name = "Loc-Fingers Glb-Others";

	local_transform_targets.resize(30);
	local_transform_targets.write[0] = "LeftThumbMetacarpal";
	local_transform_targets.write[1] = "LeftThumbProximal";
	local_transform_targets.write[2] = "LeftThumbDistal";
	local_transform_targets.write[3] = "LeftIndexProximal";
	local_transform_targets.write[4] = "LeftIndexIntermediate";
	local_transform_targets.write[5] = "LeftIndexDistal";
	local_transform_targets.write[6] = "LeftMiddleProximal";
	local_transform_targets.write[7] = "LeftMiddleIntermediate";
	local_transform_targets.write[8] = "LeftMiddleDistal";
	local_transform_targets.write[9] = "LeftRingProximal";
	local_transform_targets.write[10] = "LeftRingIntermediate";
	local_transform_targets.write[11] = "LeftRingDistal";
	local_transform_targets.write[12] = "LeftLittleProximal";
	local_transform_targets.write[13] = "LeftLittleIntermediate";
	local_transform_targets.write[14] = "LeftLittleDistal";
	local_transform_targets.write[15] = "RightThumbMetacarpal";
	local_transform_targets.write[16] = "RightThumbProximal";
	local_transform_targets.write[17] = "RightThumbDistal";
	local_transform_targets.write[18] = "RightIndexProximal";
	local_transform_targets.write[19] = "RightIndexIntermediate";
	local_transform_targets.write[20] = "RightIndexDistal";
	local_transform_targets.write[21] = "RightMiddleProximal";
	local_transform_targets.write[22] = "RightMiddleIntermediate";
	local_transform_targets.write[23] = "RightMiddleDistal";
	local_transform_targets.write[24] = "RightRingProximal";
	local_transform_targets.write[25] = "RightRingIntermediate";
	local_transform_targets.write[26] = "RightRingDistal";
	local_transform_targets.write[27] = "RightLittleProximal";
	local_transform_targets.write[28] = "RightLittleIntermediate";
	local_transform_targets.write[29] = "RightLittleDistal";

	global_transform_targets.resize(26);
	global_transform_targets.write[0] = "Root";
	global_transform_targets.write[1] = "Hips";
	global_transform_targets.write[2] = "Spine";
	global_transform_targets.write[3] = "Chest";
	global_transform_targets.write[4] = "UpperChest";
	global_transform_targets.write[5] = "Neck";
	global_transform_targets.write[6] = "Head";
	global_transform_targets.write[7] = "LeftEye";
	global_transform_targets.write[8] = "RightEye";
	global_transform_targets.write[9] = "Jaw";
	global_transform_targets.write[10] = "LeftShoulder";
	global_transform_targets.write[11] = "LeftUpperArm";
	global_transform_targets.write[12] = "LeftLowerArm";
	global_transform_targets.write[13] = "LeftHand";
	global_transform_targets.write[14] = "RightShoulder";
	global_transform_targets.write[15] = "RightUpperArm";
	global_transform_targets.write[16] = "RightLowerArm";
	global_transform_targets.write[17] = "RightHand";
	global_transform_targets.write[18] = "LeftUpperLeg";
	global_transform_targets.write[19] = "LeftLowerLeg";
	global_transform_targets.write[20] = "LeftFoot";
	global_transform_targets.write[21] = "LeftToes";
	global_transform_targets.write[22] = "RightUpperLeg";
	global_transform_targets.write[23] = "RightLowerLeg";
	global_transform_targets.write[24] = "RightFoot";
	global_transform_targets.write[25] = "RightToes";
}

RetargetProfileLocalFingersGlobalOthers::~RetargetProfileLocalFingersGlobalOthers() {
}

RetargetProfileLocalLimbsGlobalOthers::RetargetProfileLocalLimbsGlobalOthers() {
	is_read_only = true;

	label_for_animation_name = "Loc-Limbs Glb-Others";

	local_transform_targets.resize(44);
	local_transform_targets.write[0] = "LeftUpperArm";
	local_transform_targets.write[1] = "LeftLowerArm";
	local_transform_targets.write[2] = "LeftHand";
	local_transform_targets.write[3] = "LeftThumbMetacarpal";
	local_transform_targets.write[4] = "LeftThumbProximal";
	local_transform_targets.write[5] = "LeftThumbDistal";
	local_transform_targets.write[6] = "LeftIndexProximal";
	local_transform_targets.write[7] = "LeftIndexIntermediate";
	local_transform_targets.write[8] = "LeftIndexDistal";
	local_transform_targets.write[9] = "LeftMiddleProximal";
	local_transform_targets.write[10] = "LeftMiddleIntermediate";
	local_transform_targets.write[11] = "LeftMiddleDistal";
	local_transform_targets.write[12] = "LeftRingProximal";
	local_transform_targets.write[13] = "LeftRingIntermediate";
	local_transform_targets.write[14] = "LeftRingDistal";
	local_transform_targets.write[15] = "LeftLittleProximal";
	local_transform_targets.write[16] = "LeftLittleIntermediate";
	local_transform_targets.write[17] = "LeftLittleDistal";
	local_transform_targets.write[18] = "RightUpperArm";
	local_transform_targets.write[19] = "RightLowerArm";
	local_transform_targets.write[20] = "RightHand";
	local_transform_targets.write[21] = "RightThumbMetacarpal";
	local_transform_targets.write[22] = "RightThumbProximal";
	local_transform_targets.write[23] = "RightThumbDistal";
	local_transform_targets.write[24] = "RightIndexProximal";
	local_transform_targets.write[25] = "RightIndexIntermediate";
	local_transform_targets.write[26] = "RightIndexDistal";
	local_transform_targets.write[27] = "RightMiddleProximal";
	local_transform_targets.write[28] = "RightMiddleIntermediate";
	local_transform_targets.write[29] = "RightMiddleDistal";
	local_transform_targets.write[30] = "RightRingProximal";
	local_transform_targets.write[31] = "RightRingIntermediate";
	local_transform_targets.write[32] = "RightRingDistal";
	local_transform_targets.write[33] = "RightLittleProximal";
	local_transform_targets.write[34] = "RightLittleIntermediate";
	local_transform_targets.write[35] = "RightLittleDistal";
	local_transform_targets.write[36] = "LeftUpperLeg";
	local_transform_targets.write[37] = "LeftLowerLeg";
	local_transform_targets.write[38] = "LeftFoot";
	local_transform_targets.write[39] = "LeftToes";
	local_transform_targets.write[40] = "RightUpperLeg";
	local_transform_targets.write[41] = "RightLowerLeg";
	local_transform_targets.write[42] = "RightFoot";
	local_transform_targets.write[43] = "RightToes";

	global_transform_targets.resize(12);
	global_transform_targets.write[0] = "Root";
	global_transform_targets.write[1] = "Hips";
	global_transform_targets.write[2] = "Spine";
	global_transform_targets.write[3] = "Chest";
	global_transform_targets.write[4] = "UpperChest";
	global_transform_targets.write[5] = "Neck";
	global_transform_targets.write[6] = "Head";
	global_transform_targets.write[7] = "LeftEye";
	global_transform_targets.write[8] = "RightEye";
	global_transform_targets.write[9] = "Jaw";
	global_transform_targets.write[10] = "LeftShoulder";
	global_transform_targets.write[11] = "RightShoulder";
}

RetargetProfileLocalLimbsGlobalOthers::~RetargetProfileLocalLimbsGlobalOthers() {
}

RetargetProfileAbsoluteFingersGlobalOthers::RetargetProfileAbsoluteFingersGlobalOthers() {
	is_read_only = true;

	label_for_animation_name = "Abs-Fingers Glb-Others";

	absolute_transform_targets.resize(30);
	absolute_transform_targets.write[0] = "LeftThumbMetacarpal";
	absolute_transform_targets.write[1] = "LeftThumbProximal";
	absolute_transform_targets.write[2] = "LeftThumbDistal";
	absolute_transform_targets.write[3] = "LeftIndexProximal";
	absolute_transform_targets.write[4] = "LeftIndexIntermediate";
	absolute_transform_targets.write[5] = "LeftIndexDistal";
	absolute_transform_targets.write[6] = "LeftMiddleProximal";
	absolute_transform_targets.write[7] = "LeftMiddleIntermediate";
	absolute_transform_targets.write[8] = "LeftMiddleDistal";
	absolute_transform_targets.write[9] = "LeftRingProximal";
	absolute_transform_targets.write[10] = "LeftRingIntermediate";
	absolute_transform_targets.write[11] = "LeftRingDistal";
	absolute_transform_targets.write[12] = "LeftLittleProximal";
	absolute_transform_targets.write[13] = "LeftLittleIntermediate";
	absolute_transform_targets.write[14] = "LeftLittleDistal";
	absolute_transform_targets.write[15] = "RightThumbMetacarpal";
	absolute_transform_targets.write[16] = "RightThumbProximal";
	absolute_transform_targets.write[17] = "RightThumbDistal";
	absolute_transform_targets.write[18] = "RightIndexProximal";
	absolute_transform_targets.write[19] = "RightIndexIntermediate";
	absolute_transform_targets.write[20] = "RightIndexDistal";
	absolute_transform_targets.write[21] = "RightMiddleProximal";
	absolute_transform_targets.write[22] = "RightMiddleIntermediate";
	absolute_transform_targets.write[23] = "RightMiddleDistal";
	absolute_transform_targets.write[24] = "RightRingProximal";
	absolute_transform_targets.write[25] = "RightRingIntermediate";
	absolute_transform_targets.write[26] = "RightRingDistal";
	absolute_transform_targets.write[27] = "RightLittleProximal";
	absolute_transform_targets.write[28] = "RightLittleIntermediate";
	absolute_transform_targets.write[29] = "RightLittleDistal";

	global_transform_targets.resize(26);
	global_transform_targets.write[0] = "Root";
	global_transform_targets.write[1] = "Hips";
	global_transform_targets.write[2] = "Spine";
	global_transform_targets.write[3] = "Chest";
	global_transform_targets.write[4] = "UpperChest";
	global_transform_targets.write[5] = "Neck";
	global_transform_targets.write[6] = "Head";
	global_transform_targets.write[7] = "LeftEye";
	global_transform_targets.write[8] = "RightEye";
	global_transform_targets.write[9] = "Jaw";
	global_transform_targets.write[10] = "LeftShoulder";
	global_transform_targets.write[11] = "LeftUpperArm";
	global_transform_targets.write[12] = "LeftLowerArm";
	global_transform_targets.write[13] = "LeftHand";
	global_transform_targets.write[14] = "RightShoulder";
	global_transform_targets.write[15] = "RightUpperArm";
	global_transform_targets.write[16] = "RightLowerArm";
	global_transform_targets.write[17] = "RightHand";
	global_transform_targets.write[18] = "LeftUpperLeg";
	global_transform_targets.write[19] = "LeftLowerLeg";
	global_transform_targets.write[20] = "LeftFoot";
	global_transform_targets.write[21] = "LeftToes";
	global_transform_targets.write[22] = "RightUpperLeg";
	global_transform_targets.write[23] = "RightLowerLeg";
	global_transform_targets.write[24] = "RightFoot";
	global_transform_targets.write[25] = "RightToes";
}

RetargetProfileAbsoluteFingersGlobalOthers::~RetargetProfileAbsoluteFingersGlobalOthers() {
}

RetargetProfileAbsoluteLimbsGlobalOthers::RetargetProfileAbsoluteLimbsGlobalOthers() {
	is_read_only = true;

	label_for_animation_name = "Abs-Limbs Glb-Others";

	absolute_transform_targets.resize(44);
	absolute_transform_targets.write[0] = "LeftUpperArm";
	absolute_transform_targets.write[1] = "LeftLowerArm";
	absolute_transform_targets.write[2] = "LeftHand";
	absolute_transform_targets.write[3] = "LeftThumbMetacarpal";
	absolute_transform_targets.write[4] = "LeftThumbProximal";
	absolute_transform_targets.write[5] = "LeftThumbDistal";
	absolute_transform_targets.write[6] = "LeftIndexProximal";
	absolute_transform_targets.write[7] = "LeftIndexIntermediate";
	absolute_transform_targets.write[8] = "LeftIndexDistal";
	absolute_transform_targets.write[9] = "LeftMiddleProximal";
	absolute_transform_targets.write[10] = "LeftMiddleIntermediate";
	absolute_transform_targets.write[11] = "LeftMiddleDistal";
	absolute_transform_targets.write[12] = "LeftRingProximal";
	absolute_transform_targets.write[13] = "LeftRingIntermediate";
	absolute_transform_targets.write[14] = "LeftRingDistal";
	absolute_transform_targets.write[15] = "LeftLittleProximal";
	absolute_transform_targets.write[16] = "LeftLittleIntermediate";
	absolute_transform_targets.write[17] = "LeftLittleDistal";
	absolute_transform_targets.write[18] = "RightUpperArm";
	absolute_transform_targets.write[19] = "RightLowerArm";
	absolute_transform_targets.write[20] = "RightHand";
	absolute_transform_targets.write[21] = "RightThumbMetacarpal";
	absolute_transform_targets.write[22] = "RightThumbProximal";
	absolute_transform_targets.write[23] = "RightThumbDistal";
	absolute_transform_targets.write[24] = "RightIndexProximal";
	absolute_transform_targets.write[25] = "RightIndexIntermediate";
	absolute_transform_targets.write[26] = "RightIndexDistal";
	absolute_transform_targets.write[27] = "RightMiddleProximal";
	absolute_transform_targets.write[28] = "RightMiddleIntermediate";
	absolute_transform_targets.write[29] = "RightMiddleDistal";
	absolute_transform_targets.write[30] = "RightRingProximal";
	absolute_transform_targets.write[31] = "RightRingIntermediate";
	absolute_transform_targets.write[32] = "RightRingDistal";
	absolute_transform_targets.write[33] = "RightLittleProximal";
	absolute_transform_targets.write[34] = "RightLittleIntermediate";
	absolute_transform_targets.write[35] = "RightLittleDistal";
	absolute_transform_targets.write[36] = "LeftUpperLeg";
	absolute_transform_targets.write[37] = "LeftLowerLeg";
	absolute_transform_targets.write[38] = "LeftFoot";
	absolute_transform_targets.write[39] = "LeftToes";
	absolute_transform_targets.write[40] = "RightUpperLeg";
	absolute_transform_targets.write[41] = "RightLowerLeg";
	absolute_transform_targets.write[42] = "RightFoot";
	absolute_transform_targets.write[43] = "RightToes";

	global_transform_targets.resize(12);
	global_transform_targets.write[0] = "Root";
	global_transform_targets.write[1] = "Hips";
	global_transform_targets.write[2] = "Spine";
	global_transform_targets.write[3] = "Chest";
	global_transform_targets.write[4] = "UpperChest";
	global_transform_targets.write[5] = "Neck";
	global_transform_targets.write[6] = "Head";
	global_transform_targets.write[7] = "LeftEye";
	global_transform_targets.write[8] = "RightEye";
	global_transform_targets.write[9] = "Jaw";
	global_transform_targets.write[10] = "LeftShoulder";
	global_transform_targets.write[11] = "RightShoulder";
}

RetargetProfileAbsoluteLimbsGlobalOthers::~RetargetProfileAbsoluteLimbsGlobalOthers() {
}

RetargetProfileAbsoluteFingersLocalLimbsGlobalOthers::RetargetProfileAbsoluteFingersLocalLimbsGlobalOthers() {
	is_read_only = true;

	label_for_animation_name = "Abs-Fingers Loc-Limbs Glb-Others";

	absolute_transform_targets.resize(30);
	absolute_transform_targets.write[0] = "LeftThumbMetacarpal";
	absolute_transform_targets.write[1] = "LeftThumbProximal";
	absolute_transform_targets.write[2] = "LeftThumbDistal";
	absolute_transform_targets.write[3] = "LeftIndexProximal";
	absolute_transform_targets.write[4] = "LeftIndexIntermediate";
	absolute_transform_targets.write[5] = "LeftIndexDistal";
	absolute_transform_targets.write[6] = "LeftMiddleProximal";
	absolute_transform_targets.write[7] = "LeftMiddleIntermediate";
	absolute_transform_targets.write[8] = "LeftMiddleDistal";
	absolute_transform_targets.write[9] = "LeftRingProximal";
	absolute_transform_targets.write[10] = "LeftRingIntermediate";
	absolute_transform_targets.write[11] = "LeftRingDistal";
	absolute_transform_targets.write[12] = "LeftLittleProximal";
	absolute_transform_targets.write[13] = "LeftLittleIntermediate";
	absolute_transform_targets.write[14] = "LeftLittleDistal";
	absolute_transform_targets.write[15] = "RightThumbMetacarpal";
	absolute_transform_targets.write[16] = "RightThumbProximal";
	absolute_transform_targets.write[17] = "RightThumbDistal";
	absolute_transform_targets.write[18] = "RightIndexProximal";
	absolute_transform_targets.write[19] = "RightIndexIntermediate";
	absolute_transform_targets.write[20] = "RightIndexDistal";
	absolute_transform_targets.write[21] = "RightMiddleProximal";
	absolute_transform_targets.write[22] = "RightMiddleIntermediate";
	absolute_transform_targets.write[23] = "RightMiddleDistal";
	absolute_transform_targets.write[24] = "RightRingProximal";
	absolute_transform_targets.write[25] = "RightRingIntermediate";
	absolute_transform_targets.write[26] = "RightRingDistal";
	absolute_transform_targets.write[27] = "RightLittleProximal";
	absolute_transform_targets.write[28] = "RightLittleIntermediate";
	absolute_transform_targets.write[29] = "RightLittleDistal";

	local_transform_targets.resize(14);
	local_transform_targets.write[0] = "LeftUpperArm";
	local_transform_targets.write[1] = "LeftLowerArm";
	local_transform_targets.write[2] = "LeftHand";
	local_transform_targets.write[3] = "RightUpperArm";
	local_transform_targets.write[4] = "RightLowerArm";
	local_transform_targets.write[5] = "RightHand";
	local_transform_targets.write[6] = "LeftUpperLeg";
	local_transform_targets.write[7] = "LeftLowerLeg";
	local_transform_targets.write[8] = "LeftFoot";
	local_transform_targets.write[9] = "LeftToes";
	local_transform_targets.write[10] = "RightUpperLeg";
	local_transform_targets.write[11] = "RightLowerLeg";
	local_transform_targets.write[12] = "RightFoot";
	local_transform_targets.write[13] = "RightToes";

	global_transform_targets.resize(12);
	global_transform_targets.write[0] = "Root";
	global_transform_targets.write[1] = "Hips";
	global_transform_targets.write[2] = "Spine";
	global_transform_targets.write[3] = "Chest";
	global_transform_targets.write[4] = "UpperChest";
	global_transform_targets.write[5] = "Neck";
	global_transform_targets.write[6] = "Head";
	global_transform_targets.write[7] = "LeftEye";
	global_transform_targets.write[8] = "RightEye";
	global_transform_targets.write[9] = "Jaw";
	global_transform_targets.write[10] = "LeftShoulder";
	global_transform_targets.write[11] = "RightShoulder";
}

RetargetProfileAbsoluteFingersLocalLimbsGlobalOthers::~RetargetProfileAbsoluteFingersLocalLimbsGlobalOthers() {
}
