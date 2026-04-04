/**************************************************************************/
/*  copy_transform_modifier3d.hpp                                         */
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

#include <godot_cpp/classes/bone_constraint3d.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class CopyTransformModifier3D : public BoneConstraint3D {
	GDEXTENSION_CLASS(CopyTransformModifier3D, BoneConstraint3D)

public:
	enum TransformFlag : uint64_t {
		TRANSFORM_FLAG_POSITION = 1,
		TRANSFORM_FLAG_ROTATION = 2,
		TRANSFORM_FLAG_SCALE = 4,
		TRANSFORM_FLAG_ALL = 7,
	};

	enum AxisFlag : uint64_t {
		AXIS_FLAG_X = 1,
		AXIS_FLAG_Y = 2,
		AXIS_FLAG_Z = 4,
		AXIS_FLAG_ALL = 7,
	};

	void set_copy_flags(int32_t p_index, BitField<CopyTransformModifier3D::TransformFlag> p_copy_flags);
	BitField<CopyTransformModifier3D::TransformFlag> get_copy_flags(int32_t p_index) const;
	void set_axis_flags(int32_t p_index, BitField<CopyTransformModifier3D::AxisFlag> p_axis_flags);
	BitField<CopyTransformModifier3D::AxisFlag> get_axis_flags(int32_t p_index) const;
	void set_invert_flags(int32_t p_index, BitField<CopyTransformModifier3D::AxisFlag> p_axis_flags);
	BitField<CopyTransformModifier3D::AxisFlag> get_invert_flags(int32_t p_index) const;
	void set_copy_position(int32_t p_index, bool p_enabled);
	bool is_position_copying(int32_t p_index) const;
	void set_copy_rotation(int32_t p_index, bool p_enabled);
	bool is_rotation_copying(int32_t p_index) const;
	void set_copy_scale(int32_t p_index, bool p_enabled);
	bool is_scale_copying(int32_t p_index) const;
	void set_axis_x_enabled(int32_t p_index, bool p_enabled);
	bool is_axis_x_enabled(int32_t p_index) const;
	void set_axis_y_enabled(int32_t p_index, bool p_enabled);
	bool is_axis_y_enabled(int32_t p_index) const;
	void set_axis_z_enabled(int32_t p_index, bool p_enabled);
	bool is_axis_z_enabled(int32_t p_index) const;
	void set_axis_x_inverted(int32_t p_index, bool p_enabled);
	bool is_axis_x_inverted(int32_t p_index) const;
	void set_axis_y_inverted(int32_t p_index, bool p_enabled);
	bool is_axis_y_inverted(int32_t p_index) const;
	void set_axis_z_inverted(int32_t p_index, bool p_enabled);
	bool is_axis_z_inverted(int32_t p_index) const;
	void set_relative(int32_t p_index, bool p_enabled);
	bool is_relative(int32_t p_index) const;
	void set_additive(int32_t p_index, bool p_enabled);
	bool is_additive(int32_t p_index) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		BoneConstraint3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_BITFIELD_CAST(CopyTransformModifier3D::TransformFlag);
VARIANT_BITFIELD_CAST(CopyTransformModifier3D::AxisFlag);

