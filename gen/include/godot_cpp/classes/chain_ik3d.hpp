/**************************************************************************/
/*  chain_ik3d.hpp                                                        */
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

#include <godot_cpp/classes/ik_modifier3d.hpp>
#include <godot_cpp/classes/skeleton_modifier3d.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class ChainIK3D : public IKModifier3D {
	GDEXTENSION_CLASS(ChainIK3D, IKModifier3D)

public:
	void set_root_bone_name(int32_t p_index, const String &p_bone_name);
	String get_root_bone_name(int32_t p_index) const;
	void set_root_bone(int32_t p_index, int32_t p_bone);
	int32_t get_root_bone(int32_t p_index) const;
	void set_end_bone_name(int32_t p_index, const String &p_bone_name);
	String get_end_bone_name(int32_t p_index) const;
	void set_end_bone(int32_t p_index, int32_t p_bone);
	int32_t get_end_bone(int32_t p_index) const;
	void set_extend_end_bone(int32_t p_index, bool p_enabled);
	bool is_end_bone_extended(int32_t p_index) const;
	void set_end_bone_direction(int32_t p_index, SkeletonModifier3D::BoneDirection p_bone_direction);
	SkeletonModifier3D::BoneDirection get_end_bone_direction(int32_t p_index) const;
	void set_end_bone_length(int32_t p_index, float p_length);
	float get_end_bone_length(int32_t p_index) const;
	String get_joint_bone_name(int32_t p_index, int32_t p_joint) const;
	int32_t get_joint_bone(int32_t p_index, int32_t p_joint) const;
	int32_t get_joint_count(int32_t p_index) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		IKModifier3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

