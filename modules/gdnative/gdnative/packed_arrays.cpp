/*************************************************************************/
/*  packed_arrays.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gdnative/packed_arrays.h"

#include "core/variant/variant.h"

#include "core/math/vector2.h"
#include "core/math/vector3i.h"

static_assert(sizeof(godot_packed_byte_array) == sizeof(PackedByteArray), "PackedByteArray size mismatch");
static_assert(sizeof(godot_packed_int32_array) == sizeof(PackedInt32Array), "PackedInt32Array size mismatch");
static_assert(sizeof(godot_packed_int64_array) == sizeof(PackedInt64Array), "PackedInt64Array size mismatch");
static_assert(sizeof(godot_packed_float32_array) == sizeof(PackedFloat32Array), "PackedFloat32Array size mismatch");
static_assert(sizeof(godot_packed_float64_array) == sizeof(PackedFloat64Array), "PackedFloat64Array size mismatch");
static_assert(sizeof(godot_packed_string_array) == sizeof(PackedStringArray), "PackedStringArray size mismatch");
static_assert(sizeof(godot_packed_vector2_array) == sizeof(PackedVector2Array), "PackedVector2Array size mismatch");
static_assert(sizeof(godot_packed_vector2i_array) == sizeof(Vector<Vector2i>), "Vector<Vector2i> size mismatch");
static_assert(sizeof(godot_packed_vector3_array) == sizeof(PackedVector3Array), "PackedVector3Array size mismatch");
static_assert(sizeof(godot_packed_vector3i_array) == sizeof(Vector<Vector3i>), "Vector<Vector3i> size mismatch");
static_assert(sizeof(godot_packed_color_array) == sizeof(PackedColorArray), "PackedColorArray size mismatch");

#ifdef __cplusplus
extern "C" {
#endif

#define memnew_placement_custom(m_placement, m_class, m_constr) _post_initialize(new (m_placement, sizeof(m_class), "") m_constr)

// byte

void GDAPI godot_packed_byte_array_new(godot_packed_byte_array *p_self) {
	memnew_placement(p_self, PackedByteArray);
}

void GDAPI godot_packed_byte_array_destroy(godot_packed_byte_array *p_self) {
	((PackedByteArray *)p_self)->~PackedByteArray();
}

// int32

void GDAPI godot_packed_int32_array_new(godot_packed_int32_array *p_self) {
	memnew_placement(p_self, PackedInt32Array);
}

void GDAPI godot_packed_int32_array_destroy(godot_packed_int32_array *p_self) {
	((PackedInt32Array *)p_self)->~PackedInt32Array();
}

// int64

void GDAPI godot_packed_int64_array_new(godot_packed_int64_array *p_self) {
	memnew_placement(p_self, PackedInt64Array);
}

void GDAPI godot_packed_int64_array_destroy(godot_packed_int64_array *p_self) {
	((PackedInt64Array *)p_self)->~PackedInt64Array();
}

// float32

void GDAPI godot_packed_float32_array_new(godot_packed_float32_array *p_self) {
	memnew_placement(p_self, PackedFloat32Array);
}

void GDAPI godot_packed_float32_array_destroy(godot_packed_float32_array *p_self) {
	((PackedFloat32Array *)p_self)->~PackedFloat32Array();
}

// float64

void GDAPI godot_packed_float64_array_new(godot_packed_float64_array *p_self) {
	memnew_placement(p_self, PackedFloat64Array);
}

void GDAPI godot_packed_float64_array_destroy(godot_packed_float64_array *p_self) {
	((PackedFloat64Array *)p_self)->~PackedFloat64Array();
}

// string

void GDAPI godot_packed_string_array_new(godot_packed_string_array *p_self) {
	memnew_placement(p_self, PackedStringArray);
}

void GDAPI godot_packed_string_array_destroy(godot_packed_string_array *p_self) {
	((PackedStringArray *)p_self)->~PackedStringArray();
}

// vector2

void GDAPI godot_packed_vector2_array_new(godot_packed_vector2_array *p_self) {
	memnew_placement(p_self, PackedVector2Array);
}

void GDAPI godot_packed_vector2_array_destroy(godot_packed_vector2_array *p_self) {
	((PackedVector2Array *)p_self)->~PackedVector2Array();
}

// vector2i

void GDAPI godot_packed_vector2i_array_new(godot_packed_vector2i_array *p_self) {
	memnew_placement(p_self, Vector<Vector2i>);
}

void GDAPI godot_packed_vector2i_array_destroy(godot_packed_vector2i_array *p_self) {
	((Vector<Vector2i> *)p_self)->~Vector();
}

// vector3

void GDAPI godot_packed_vector3_array_new(godot_packed_vector3_array *p_self) {
	memnew_placement(p_self, PackedVector3Array);
}

void GDAPI godot_packed_vector3_array_destroy(godot_packed_vector3_array *p_self) {
	((PackedVector3Array *)p_self)->~PackedVector3Array();
}

// vector3i

void GDAPI godot_packed_vector3i_array_new(godot_packed_vector3i_array *p_self) {
	memnew_placement(p_self, Vector<Vector3i>);
}

void GDAPI godot_packed_vector3i_array_destroy(godot_packed_vector3i_array *p_self) {
	((Vector<Vector3i> *)p_self)->~Vector();
}

// color

void GDAPI godot_packed_color_array_new(godot_packed_color_array *p_self) {
	memnew_placement(p_self, PackedColorArray);
}

void GDAPI godot_packed_color_array_destroy(godot_packed_color_array *p_self) {
	((PackedColorArray *)p_self)->~PackedColorArray();
}

#ifdef __cplusplus
}
#endif
