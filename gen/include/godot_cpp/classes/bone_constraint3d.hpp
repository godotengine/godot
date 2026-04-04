/**************************************************************************/
/*  bone_constraint3d.hpp                                                 */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/skeleton_modifier3d.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class BoneConstraint3D : public SkeletonModifier3D {
	GDEXTENSION_CLASS(BoneConstraint3D, SkeletonModifier3D)

public:
	enum ReferenceType {
		REFERENCE_TYPE_BONE = 0,
		REFERENCE_TYPE_NODE = 1,
	};

	void set_amount(int32_t p_index, float p_amount);
	float get_amount(int32_t p_index) const;
	void set_apply_bone_name(int32_t p_index, const String &p_bone_name);
	String get_apply_bone_name(int32_t p_index) const;
	void set_apply_bone(int32_t p_index, int32_t p_bone);
	int32_t get_apply_bone(int32_t p_index) const;
	void set_reference_type(int32_t p_index, BoneConstraint3D::ReferenceType p_type);
	BoneConstraint3D::ReferenceType get_reference_type(int32_t p_index) const;
	void set_reference_bone_name(int32_t p_index, const String &p_bone_name);
	String get_reference_bone_name(int32_t p_index) const;
	void set_reference_bone(int32_t p_index, int32_t p_bone);
	int32_t get_reference_bone(int32_t p_index) const;
	void set_reference_node(int32_t p_index, const NodePath &p_node);
	NodePath get_reference_node(int32_t p_index) const;
	void set_setting_count(int32_t p_count);
	int32_t get_setting_count() const;
	void clear_setting();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		SkeletonModifier3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(BoneConstraint3D::ReferenceType);

