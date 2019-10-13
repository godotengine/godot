/*************************************************************************/
/*  skeleton_definition.h                                                */
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

#ifndef SKELETON_DEFINITION_H
#define SKELETON_DEFINITION_H

#include "core/resource.h"

/**
	@author Marios Staikopoulos <marios@staik.net>
*/

#ifndef BONE_ID_DEF
#define BONE_ID_DEF
typedef int BoneId;
#endif // BONE_ID_DEF

class Skeleton;

class SkeletonDefinition : public Resource {
	GDCLASS(SkeletonDefinition, Resource);
	RES_BASE_EXTENSION("skel");

public:
private:
	struct Bone {
		String name;
		BoneId parent;
		Transform rest;

		Bone() :
				parent(-1) {
		}
	};

	Vector<Bone> bones;

protected:
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void add_bone(const String &p_name);
	BoneId find_bone(const String &p_name) const;
	String get_bone_name(const BoneId p_bone) const;

	bool is_bone_parent_of(const BoneId p_bone_id, const BoneId p_parent_bone_id) const;

	void set_bone_parent(const BoneId p_bone, const BoneId p_parent);
	BoneId get_bone_parent(const BoneId p_bone) const;

	int get_bone_count() const;

	void set_bone_rest(const BoneId p_bone, const Transform &p_rest);
	Transform get_bone_rest(const BoneId p_bone) const;

	void clear_bones();

	static Ref<SkeletonDefinition> create_from_skeleton(const Skeleton *skeleton);
};

#endif
