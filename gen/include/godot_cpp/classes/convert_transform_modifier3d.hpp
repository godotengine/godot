/**************************************************************************/
/*  convert_transform_modifier3d.hpp                                      */
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
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class ConvertTransformModifier3D : public BoneConstraint3D {
	GDEXTENSION_CLASS(ConvertTransformModifier3D, BoneConstraint3D)

public:
	enum TransformMode {
		TRANSFORM_MODE_POSITION = 0,
		TRANSFORM_MODE_ROTATION = 1,
		TRANSFORM_MODE_SCALE = 2,
	};

	void set_apply_transform_mode(int32_t p_index, ConvertTransformModifier3D::TransformMode p_transform_mode);
	ConvertTransformModifier3D::TransformMode get_apply_transform_mode(int32_t p_index) const;
	void set_apply_axis(int32_t p_index, Vector3::Axis p_axis);
	Vector3::Axis get_apply_axis(int32_t p_index) const;
	void set_apply_range_min(int32_t p_index, float p_range_min);
	float get_apply_range_min(int32_t p_index) const;
	void set_apply_range_max(int32_t p_index, float p_range_max);
	float get_apply_range_max(int32_t p_index) const;
	void set_reference_transform_mode(int32_t p_index, ConvertTransformModifier3D::TransformMode p_transform_mode);
	ConvertTransformModifier3D::TransformMode get_reference_transform_mode(int32_t p_index) const;
	void set_reference_axis(int32_t p_index, Vector3::Axis p_axis);
	Vector3::Axis get_reference_axis(int32_t p_index) const;
	void set_reference_range_min(int32_t p_index, float p_range_min);
	float get_reference_range_min(int32_t p_index) const;
	void set_reference_range_max(int32_t p_index, float p_range_max);
	float get_reference_range_max(int32_t p_index) const;
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

VARIANT_ENUM_CAST(ConvertTransformModifier3D::TransformMode);

