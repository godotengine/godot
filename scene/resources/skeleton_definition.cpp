/*************************************************************************/
/*  skeleton_definition.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "skeleton_definition.h"

#include "scene/3d/skeleton.h"

bool SkeletonDefinition::_get(const StringName &p_path, Variant &r_ret) const {
	const String path = p_path;

	if (!path.begins_with("bones/"))
		return false;

	const BoneId bone = path.get_slicec('/', 1).to_int();
	const String what = path.get_slicec('/', 2);

	if (what == "name") {
		r_ret = get_bone_name(bone);
		return true;
	} else if (what == "parent") {
		r_ret = get_bone_parent(bone);
		return true;
	} else if (what == "rest") {
		r_ret = get_bone_rest(bone);
		return true;
	}

	return false;
}

bool SkeletonDefinition::_set(const StringName &p_path, const Variant &p_value) {
	const String path = p_path;

	if (!path.begins_with("bones/"))
		return false;

	const BoneId bone = path.get_slicec('/', 1).to_int();
	const String what = path.get_slicec('/', 2);

	if (what == "name") {
		add_bone(p_value);
		return true;
	} else if (what == "parent") {
		set_bone_parent(bone, p_value);
		return true;
	} else if (what == "rest") {
		set_bone_rest(bone, p_value);
		return true;
	}

	return false;
}

void SkeletonDefinition::_get_property_list(List<PropertyInfo> *p_list) const {
	for (BoneId i = 0; i < bones.size(); ++i) {
		String prep = "bones/" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::STRING, prep + "name"));
		p_list->push_back(PropertyInfo(Variant::INT, prep + "parent", PROPERTY_HINT_RANGE, "-1," + itos(bones.size() - 1) + ",1"));
		p_list->push_back(PropertyInfo(Variant::TRANSFORM, prep + "rest"));
	}
}

void SkeletonDefinition::add_bone(const String &p_name) {
	ERR_FAIL_COND(p_name == "" || p_name.find(":") != -1 || p_name.find("/") != -1);

	for (BoneId i = 0; i < bones.size(); ++i) {
		ERR_FAIL_COND(bones[i].name == p_name);
	}

	Bone b;
	b.name = p_name;
	bones.push_back(b);
}

BoneId SkeletonDefinition::find_bone(const String &p_name) const {
	for (BoneId i = 0; i < bones.size(); ++i) {
		if (bones[i].name == p_name)
			return i;
	}

	return -1;
}

String SkeletonDefinition::get_bone_name(const BoneId p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), "");

	return bones[p_bone].name;
}

bool SkeletonDefinition::is_bone_parent_of(const BoneId p_bone, const BoneId p_parent_bone) const {
	BoneId parent_of_bone = get_bone_parent(p_bone);

	if (parent_of_bone == -1)
		return false;

	if (parent_of_bone == p_parent_bone)
		return true;

	return is_bone_parent_of(parent_of_bone, p_parent_bone);
}

void SkeletonDefinition::set_bone_parent(const BoneId p_bone, const BoneId p_parent) {
	ERR_FAIL_INDEX(p_bone, bones.size());
	ERR_FAIL_COND(p_parent != -1 && (p_parent < 0));

	bones.write[p_bone].parent = p_parent;
}

BoneId SkeletonDefinition::get_bone_parent(const BoneId p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), -1);

	return bones[p_bone].parent;
}

int SkeletonDefinition::get_bone_count() const {
	return bones.size();
}

void SkeletonDefinition::set_bone_rest(const BoneId p_bone, const Transform &p_rest) {
	ERR_FAIL_INDEX(p_bone, bones.size());

	bones.write[p_bone].rest = p_rest;
}

Transform SkeletonDefinition::get_bone_rest(const BoneId p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), Transform());

	return bones[p_bone].rest;
}

void SkeletonDefinition::clear_bones() {
	bones.clear();
}

Ref<SkeletonDefinition> SkeletonDefinition::create_from_skeleton(const Skeleton *skeleton) {
	Ref<SkeletonDefinition> def;
	def.instance();

	const Vector<Skeleton::Bone> &bones = skeleton->bones;

	for (BoneId i = 0; i < bones.size(); ++i) {
		def->add_bone(bones[i].name);
		def->set_bone_parent(i, bones[i].parent);
		def->set_bone_rest(i, bones[i].rest);
	}

	return def;
}
