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

void GDAPI godot_packed_byte_array_new_copy(godot_packed_byte_array *r_dest, const godot_packed_byte_array *p_src) {
	memnew_placement(r_dest, PackedByteArray(*(PackedByteArray *)p_src));
}

void GDAPI godot_packed_byte_array_destroy(godot_packed_byte_array *p_self) {
	((PackedByteArray *)p_self)->~PackedByteArray();
}

uint8_t GDAPI *godot_packed_byte_array_operator_index(godot_packed_byte_array *p_self, godot_int p_index) {
	PackedByteArray *self = (PackedByteArray *)p_self;
	return (uint8_t *)&self->operator[](p_index);
}

const uint8_t GDAPI *godot_packed_byte_array_operator_index_const(const godot_packed_byte_array *p_self, godot_int p_index) {
	const PackedByteArray *self = (const PackedByteArray *)p_self;
	return (const uint8_t *)&self->operator[](p_index);
}

// int32

void GDAPI godot_packed_int32_array_new(godot_packed_int32_array *p_self) {
	memnew_placement(p_self, PackedInt32Array);
}

void GDAPI godot_packed_int32_array_new_copy(godot_packed_int32_array *r_dest, const godot_packed_int32_array *p_src) {
	memnew_placement(r_dest, PackedInt32Array(*(PackedInt32Array *)p_src));
}

void GDAPI godot_packed_int32_array_destroy(godot_packed_int32_array *p_self) {
	((PackedInt32Array *)p_self)->~PackedInt32Array();
}

int32_t GDAPI *godot_packed_int32_array_operator_index(godot_packed_int32_array *p_self, godot_int p_index) {
	PackedInt32Array *self = (PackedInt32Array *)p_self;
	return (int32_t *)&self->operator[](p_index);
}

const int32_t GDAPI *godot_packed_int32_array_operator_index_const(const godot_packed_int32_array *p_self, godot_int p_index) {
	const PackedInt32Array *self = (const PackedInt32Array *)p_self;
	return (const int32_t *)&self->operator[](p_index);
}

// int64

void GDAPI godot_packed_int64_array_new(godot_packed_int64_array *p_self) {
	memnew_placement(p_self, PackedInt64Array);
}

void GDAPI godot_packed_int64_array_new_copy(godot_packed_int64_array *r_dest, const godot_packed_int64_array *p_src) {
	memnew_placement(r_dest, PackedInt64Array(*(PackedInt64Array *)p_src));
}

void GDAPI godot_packed_int64_array_destroy(godot_packed_int64_array *p_self) {
	((PackedInt64Array *)p_self)->~PackedInt64Array();
}

int64_t GDAPI *godot_packed_int64_array_operator_index(godot_packed_int64_array *p_self, godot_int p_index) {
	PackedInt64Array *self = (PackedInt64Array *)p_self;
	return (int64_t *)&self->operator[](p_index);
}

const int64_t GDAPI *godot_packed_int64_array_operator_index_const(const godot_packed_int64_array *p_self, godot_int p_index) {
	const PackedInt64Array *self = (const PackedInt64Array *)p_self;
	return (const int64_t *)&self->operator[](p_index);
}

// float32

void GDAPI godot_packed_float32_array_new(godot_packed_float32_array *p_self) {
	memnew_placement(p_self, PackedFloat32Array);
}

void GDAPI godot_packed_float32_array_new_copy(godot_packed_float32_array *r_dest, const godot_packed_float32_array *p_src) {
	memnew_placement(r_dest, PackedFloat32Array(*(PackedFloat32Array *)p_src));
}

void GDAPI godot_packed_float32_array_destroy(godot_packed_float32_array *p_self) {
	((PackedFloat32Array *)p_self)->~PackedFloat32Array();
}

float GDAPI *godot_packed_float32_array_operator_index(godot_packed_float32_array *p_self, godot_int p_index) {
	PackedFloat32Array *self = (PackedFloat32Array *)p_self;
	return (float *)&self->operator[](p_index);
}

const float GDAPI *godot_packed_float32_array_operator_index_const(const godot_packed_float32_array *p_self, godot_int p_index) {
	const PackedFloat32Array *self = (const PackedFloat32Array *)p_self;
	return (const float *)&self->operator[](p_index);
}

// float64

void GDAPI godot_packed_float64_array_new(godot_packed_float64_array *p_self) {
	memnew_placement(p_self, PackedFloat64Array);
}

void GDAPI godot_packed_float64_array_new_copy(godot_packed_float64_array *r_dest, const godot_packed_float64_array *p_src) {
	memnew_placement(r_dest, PackedFloat64Array(*(PackedFloat64Array *)p_src));
}

void GDAPI godot_packed_float64_array_destroy(godot_packed_float64_array *p_self) {
	((PackedFloat64Array *)p_self)->~PackedFloat64Array();
}

double GDAPI *godot_packed_float64_array_operator_index(godot_packed_float64_array *p_self, godot_int p_index) {
	PackedFloat64Array *self = (PackedFloat64Array *)p_self;
	return (double *)&self->operator[](p_index);
}

const double GDAPI *godot_packed_float64_array_operator_index_const(const godot_packed_float64_array *p_self, godot_int p_index) {
	const PackedFloat64Array *self = (const PackedFloat64Array *)p_self;
	return (const double *)&self->operator[](p_index);
}

// string

void GDAPI godot_packed_string_array_new(godot_packed_string_array *p_self) {
	memnew_placement(p_self, PackedStringArray);
}

void GDAPI godot_packed_string_array_new_copy(godot_packed_string_array *r_dest, const godot_packed_string_array *p_src) {
	memnew_placement(r_dest, PackedStringArray(*(PackedStringArray *)p_src));
}

void GDAPI godot_packed_string_array_destroy(godot_packed_string_array *p_self) {
	((PackedStringArray *)p_self)->~PackedStringArray();
}

godot_string GDAPI *godot_packed_string_array_operator_index(godot_packed_string_array *p_self, godot_int p_index) {
	PackedStringArray *self = (PackedStringArray *)p_self;
	return (godot_string *)&self->operator[](p_index);
}

const godot_string GDAPI *godot_packed_string_array_operator_index_const(const godot_packed_string_array *p_self, godot_int p_index) {
	const PackedStringArray *self = (const PackedStringArray *)p_self;
	return (const godot_string *)&self->operator[](p_index);
}

// vector2

void GDAPI godot_packed_vector2_array_new(godot_packed_vector2_array *p_self) {
	memnew_placement(p_self, PackedVector2Array);
}

void GDAPI godot_packed_vector2_array_new_copy(godot_packed_vector2_array *r_dest, const godot_packed_vector2_array *p_src) {
	memnew_placement(r_dest, PackedVector2Array(*(PackedVector2Array *)p_src));
}

void GDAPI godot_packed_vector2_array_destroy(godot_packed_vector2_array *p_self) {
	((PackedVector2Array *)p_self)->~PackedVector2Array();
}

godot_vector2 GDAPI *godot_packed_vector2_array_operator_index(godot_packed_vector2_array *p_self, godot_int p_index) {
	PackedVector2Array *self = (PackedVector2Array *)p_self;
	return (godot_vector2 *)&self->operator[](p_index);
}

const godot_vector2 GDAPI *godot_packed_vector2_array_operator_index_const(const godot_packed_vector2_array *p_self, godot_int p_index) {
	const PackedVector2Array *self = (const PackedVector2Array *)p_self;
	return (const godot_vector2 *)&self->operator[](p_index);
}

// vector2i

void GDAPI godot_packed_vector2i_array_new(godot_packed_vector2i_array *p_self) {
	memnew_placement(p_self, Vector<Vector2i>);
}

void GDAPI godot_packed_vector2i_array_new_copy(godot_packed_vector2i_array *r_dest, const godot_packed_vector2i_array *p_src) {
	memnew_placement(r_dest, Vector<Vector2i>(*(Vector<Vector2i> *)p_src));
}

void GDAPI godot_packed_vector2i_array_destroy(godot_packed_vector2i_array *p_self) {
	((Vector<Vector2i> *)p_self)->~Vector();
}

godot_vector2i GDAPI *godot_packed_vector2i_array_operator_index(godot_packed_vector2i_array *p_self, godot_int p_index) {
	Vector<Vector2i> *self = (Vector<Vector2i> *)p_self;
	return (godot_vector2i *)&self->operator[](p_index);
}

const godot_vector2i GDAPI *godot_packed_vector2i_array_operator_index_const(const godot_packed_vector2i_array *p_self, godot_int p_index) {
	const Vector<Vector2i> *self = (const Vector<Vector2i> *)p_self;
	return (const godot_vector2i *)&self->operator[](p_index);
}

// vector3

void GDAPI godot_packed_vector3_array_new(godot_packed_vector3_array *p_self) {
	memnew_placement(p_self, PackedVector3Array);
}

void GDAPI godot_packed_vector3_array_new_copy(godot_packed_vector3_array *r_dest, const godot_packed_vector3_array *p_src) {
	memnew_placement(r_dest, PackedVector3Array(*(PackedVector3Array *)p_src));
}

void GDAPI godot_packed_vector3_array_destroy(godot_packed_vector3_array *p_self) {
	((PackedVector3Array *)p_self)->~PackedVector3Array();
}

godot_vector3 GDAPI *godot_packed_vector3_array_operator_index(godot_packed_vector3_array *p_self, godot_int p_index) {
	PackedVector3Array *self = (PackedVector3Array *)p_self;
	return (godot_vector3 *)&self->operator[](p_index);
}

const godot_vector3 GDAPI *godot_packed_vector3_array_operator_index_const(const godot_packed_vector3_array *p_self, godot_int p_index) {
	const PackedVector3Array *self = (const PackedVector3Array *)p_self;
	return (const godot_vector3 *)&self->operator[](p_index);
}

// vector3i

void GDAPI godot_packed_vector3i_array_new(godot_packed_vector3i_array *p_self) {
	memnew_placement(p_self, Vector<Vector3i>);
}

void GDAPI godot_packed_vector3i_array_new_copy(godot_packed_vector3i_array *r_dest, const godot_packed_vector3i_array *p_src) {
	memnew_placement(r_dest, Vector<Vector3i>(*(Vector<Vector3i> *)p_src));
}

void GDAPI godot_packed_vector3i_array_destroy(godot_packed_vector3i_array *p_self) {
	((Vector<Vector3i> *)p_self)->~Vector();
}

godot_vector3i GDAPI *godot_packed_vector3i_array_operator_index(godot_packed_vector3i_array *p_self, godot_int p_index) {
	Vector<Vector3i> *self = (Vector<Vector3i> *)p_self;
	return (godot_vector3i *)&self->operator[](p_index);
}

const godot_vector3i GDAPI *godot_packed_vector3i_array_operator_index_const(const godot_packed_vector3i_array *p_self, godot_int p_index) {
	const Vector<Vector3i> *self = (const Vector<Vector3i> *)p_self;
	return (const godot_vector3i *)&self->operator[](p_index);
}

// color

void GDAPI godot_packed_color_array_new(godot_packed_color_array *p_self) {
	memnew_placement(p_self, PackedColorArray);
}

void GDAPI godot_packed_color_array_new_copy(godot_packed_color_array *r_dest, const godot_packed_color_array *p_src) {
	memnew_placement(r_dest, PackedColorArray(*(PackedColorArray *)p_src));
}

void GDAPI godot_packed_color_array_destroy(godot_packed_color_array *p_self) {
	((PackedColorArray *)p_self)->~PackedColorArray();
}

godot_color GDAPI *godot_packed_color_array_operator_index(godot_packed_color_array *p_self, godot_int p_index) {
	PackedColorArray *self = (PackedColorArray *)p_self;
	return (godot_color *)&self->operator[](p_index);
}

const godot_color GDAPI *godot_packed_color_array_operator_index_const(const godot_packed_color_array *p_self, godot_int p_index) {
	const PackedColorArray *self = (const PackedColorArray *)p_self;
	return (const godot_color *)&self->operator[](p_index);
}

#ifdef __cplusplus
}
#endif
