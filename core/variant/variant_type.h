/**************************************************************************/
/*  variant_type.h                                                        */
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

#pragma once

class String;

namespace VariantType {

// If this changes the table in variant_op must be updated
enum Type {
	NIL,

	// atomic types
	BOOL,
	INT,
	FLOAT,
	STRING,

	// math types
	VECTOR2,
	VECTOR2I,
	RECT2,
	RECT2I,
	VECTOR3,
	VECTOR3I,
	TRANSFORM2D,
	VECTOR4,
	VECTOR4I,
	PLANE,
	QUATERNION,
	AABB,
	BASIS,
	TRANSFORM3D,
	PROJECTION,

	// misc types
	COLOR,
	STRING_NAME,
	NODE_PATH,
	RID,
	OBJECT,
	CALLABLE,
	SIGNAL,
	DICTIONARY,
	ARRAY,

	// typed arrays
	PACKED_BYTE_ARRAY,
	PACKED_INT32_ARRAY,
	PACKED_INT64_ARRAY,
	PACKED_FLOAT32_ARRAY,
	PACKED_FLOAT64_ARRAY,
	PACKED_STRING_ARRAY,
	PACKED_VECTOR2_ARRAY,
	PACKED_VECTOR3_ARRAY,
	PACKED_COLOR_ARRAY,
	PACKED_VECTOR4_ARRAY,

	VARIANT_MAX
};

static constexpr bool needs_deinit[VariantType::VARIANT_MAX] = {
	false, //NIL,
	false, //BOOL,
	false, //INT,
	false, //FLOAT,
	true, //STRING,
	false, //VECTOR2,
	false, //VECTOR2I,
	false, //RECT2,
	false, //RECT2I,
	false, //VECTOR3,
	false, //VECTOR3I,
	true, //TRANSFORM2D,
	false, //VECTOR4,
	false, //VECTOR4I,
	false, //PLANE,
	false, //QUATERNION,
	true, //AABB,
	true, //BASIS,
	true, //TRANSFORM,
	true, //PROJECTION,

	// misc types
	false, //COLOR,
	true, //STRING_NAME,
	true, //NODE_PATH,
	false, //RID,
	true, //OBJECT,
	true, //CALLABLE,
	true, //SIGNAL,
	true, //DICTIONARY,
	true, //ARRAY,

	// typed arrays
	true, //PACKED_BYTE_ARRAY,
	true, //PACKED_INT32_ARRAY,
	true, //PACKED_INT64_ARRAY,
	true, //PACKED_FLOAT32_ARRAY,
	true, //PACKED_FLOAT64_ARRAY,
	true, //PACKED_STRING_ARRAY,
	true, //PACKED_VECTOR2_ARRAY,
	true, //PACKED_VECTOR3_ARRAY,
	true, //PACKED_COLOR_ARRAY,
	true, //PACKED_VECTOR4_ARRAY,
};

String get_type_name(Type p_type);
Type get_type_by_name(const String &p_type_name);
bool can_convert(Type p_type_from, Type p_type_to);
bool can_convert_strict(Type p_type_from, Type p_type_to);
bool is_type_shared(Type p_type);

} // namespace VariantType
