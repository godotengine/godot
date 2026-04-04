/**************************************************************************/
/*  skeleton_profile.hpp                                                  */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class SkeletonProfile : public Resource {
	GDEXTENSION_CLASS(SkeletonProfile, Resource)

public:
	enum TailDirection {
		TAIL_DIRECTION_AVERAGE_CHILDREN = 0,
		TAIL_DIRECTION_SPECIFIC_CHILD = 1,
		TAIL_DIRECTION_END = 2,
	};

	void set_root_bone(const StringName &p_bone_name);
	StringName get_root_bone();
	void set_scale_base_bone(const StringName &p_bone_name);
	StringName get_scale_base_bone();
	void set_group_size(int32_t p_size);
	int32_t get_group_size();
	StringName get_group_name(int32_t p_group_idx) const;
	void set_group_name(int32_t p_group_idx, const StringName &p_group_name);
	Ref<Texture2D> get_texture(int32_t p_group_idx) const;
	void set_texture(int32_t p_group_idx, const Ref<Texture2D> &p_texture);
	void set_bone_size(int32_t p_size);
	int32_t get_bone_size();
	int32_t find_bone(const StringName &p_bone_name) const;
	StringName get_bone_name(int32_t p_bone_idx) const;
	void set_bone_name(int32_t p_bone_idx, const StringName &p_bone_name);
	StringName get_bone_parent(int32_t p_bone_idx) const;
	void set_bone_parent(int32_t p_bone_idx, const StringName &p_bone_parent);
	SkeletonProfile::TailDirection get_tail_direction(int32_t p_bone_idx) const;
	void set_tail_direction(int32_t p_bone_idx, SkeletonProfile::TailDirection p_tail_direction);
	StringName get_bone_tail(int32_t p_bone_idx) const;
	void set_bone_tail(int32_t p_bone_idx, const StringName &p_bone_tail);
	Transform3D get_reference_pose(int32_t p_bone_idx) const;
	void set_reference_pose(int32_t p_bone_idx, const Transform3D &p_bone_name);
	Vector2 get_handle_offset(int32_t p_bone_idx) const;
	void set_handle_offset(int32_t p_bone_idx, const Vector2 &p_handle_offset);
	StringName get_group(int32_t p_bone_idx) const;
	void set_group(int32_t p_bone_idx, const StringName &p_group);
	bool is_required(int32_t p_bone_idx) const;
	void set_required(int32_t p_bone_idx, bool p_required);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(SkeletonProfile::TailDirection);

