/**************************************************************************/
/*  modifier_bone_target_3d.cpp                                           */
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

#include "modifier_bone_target_3d.h"

void ModifierBoneTarget3D::_validate_bone_names() {
	// Prior bone name.
	if (!bone_name.is_empty()) {
		set_bone_name(bone_name);
	} else if (bone != -1) {
		set_bone(bone);
	}
}

void ModifierBoneTarget3D::set_bone_name(const String &p_bone_name) {
	bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_bone(sk->find_bone(bone_name));
	}
}

String ModifierBoneTarget3D::get_bone_name() const {
	return bone_name;
}

void ModifierBoneTarget3D::set_bone(int p_bone) {
	bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (bone <= -1 || bone >= sk->get_bone_count()) {
			WARN_PRINT("Bone index out of range!");
			bone = -1;
		} else {
			bone_name = sk->get_bone_name(bone);
		}
	}
}

int ModifierBoneTarget3D::get_bone() const {
	return bone;
}

void ModifierBoneTarget3D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "influence") {
		p_property.usage = PROPERTY_USAGE_READ_ONLY;
	}
	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}
	if (p_property.name == "bone_name") {
		Skeleton3D *skeleton = get_skeleton();
		if (skeleton) {
			p_property.hint = PROPERTY_HINT_ENUM_SUGGESTION;
			p_property.hint_string = skeleton->get_concatenated_bone_names();
		} else {
			p_property.hint = PROPERTY_HINT_NONE;
			p_property.hint_string = "";
		}
	}
}

void ModifierBoneTarget3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bone_name", "bone_name"), &ModifierBoneTarget3D::set_bone_name);
	ClassDB::bind_method(D_METHOD("get_bone_name"), &ModifierBoneTarget3D::get_bone_name);
	ClassDB::bind_method(D_METHOD("set_bone", "bone"), &ModifierBoneTarget3D::set_bone);
	ClassDB::bind_method(D_METHOD("get_bone"), &ModifierBoneTarget3D::get_bone);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "bone_name", PROPERTY_HINT_ENUM_SUGGESTION, ""), "set_bone_name", "get_bone_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_bone", "get_bone");
}

void ModifierBoneTarget3D::_process_modification(double p_delta) {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton || bone < 0 || bone >= skeleton->get_bone_count()) {
		return;
	}

	set_transform(skeleton->get_bone_global_pose(bone));
}
