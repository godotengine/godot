/*************************************************************************/
/*  packed_arrays.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/array.h"

#include "core/variant.h"

#include "core/color.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"

#ifdef __cplusplus
extern "C" {
#endif

static_assert(sizeof(godot_packed_byte_array) == sizeof(Vector<uint8_t>), "Vector<uint8_t> size mismatch");
static_assert(sizeof(godot_packed_int32_array) == sizeof(Vector<int32_t>), "Vector<int32_t> size mismatch");
static_assert(sizeof(godot_packed_int64_array) == sizeof(Vector<int64_t>), "Vector<int64_t> size mismatch");
static_assert(sizeof(godot_packed_float32_array) == sizeof(Vector<float>), "Vector<float> size mismatch");
static_assert(sizeof(godot_packed_float64_array) == sizeof(Vector<double>), "Vector<double> size mismatch");
static_assert(sizeof(godot_packed_string_array) == sizeof(Vector<String>), "Vector<String> size mismatch");
static_assert(sizeof(godot_packed_vector2_array) == sizeof(Vector<Vector2>), "Vector<Vector2> size mismatch");
static_assert(sizeof(godot_packed_vector3_array) == sizeof(Vector<Vector3>), "Vector<Vector3> size mismatch");
static_assert(sizeof(godot_packed_color_array) == sizeof(Vector<Color>), "Vector<Color> size mismatch");

#define memnew_placement_custom(m_placement, m_class, m_constr) _post_initialize(new (m_placement, sizeof(m_class), "") m_constr)

// byte

void GDAPI godot_packed_byte_array_new(godot_packed_byte_array *r_dest) {
	Vector<uint8_t> *dest = (Vector<uint8_t> *)r_dest;
	memnew_placement(dest, Vector<uint8_t>);
}

void GDAPI godot_packed_byte_array_new_copy(godot_packed_byte_array *r_dest, const godot_packed_byte_array *p_src) {
	Vector<uint8_t> *dest = (Vector<uint8_t> *)r_dest;
	const Vector<uint8_t> *src = (const Vector<uint8_t> *)p_src;
	memnew_placement(dest, Vector<uint8_t>(*src));
}

void GDAPI godot_packed_byte_array_new_with_array(godot_packed_byte_array *r_dest, const godot_array *p_a) {
	Vector<uint8_t> *dest = (Vector<uint8_t> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, Vector<uint8_t>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

const uint8_t GDAPI *godot_packed_byte_array_ptr(const godot_packed_byte_array *p_self) {
	const Vector<uint8_t> *self = (const Vector<uint8_t> *)p_self;
	return self->ptr();
}

uint8_t GDAPI *godot_packed_byte_array_ptrw(godot_packed_byte_array *p_self) {
	Vector<uint8_t> *self = (Vector<uint8_t> *)p_self;
	return self->ptrw();
}

void GDAPI godot_packed_byte_array_append(godot_packed_byte_array *p_self, const uint8_t p_data) {
	Vector<uint8_t> *self = (Vector<uint8_t> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_packed_byte_array_append_array(godot_packed_byte_array *p_self, const godot_packed_byte_array *p_array) {
	Vector<uint8_t> *self = (Vector<uint8_t> *)p_self;
	Vector<uint8_t> *array = (Vector<uint8_t> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_packed_byte_array_insert(godot_packed_byte_array *p_self, const godot_int p_idx, const uint8_t p_data) {
	Vector<uint8_t> *self = (Vector<uint8_t> *)p_self;
	return (godot_error)self->insert(p_idx, p_data);
}

godot_bool GDAPI godot_packed_byte_array_has(godot_packed_byte_array *p_self, const uint8_t p_value) {
	Vector<uint8_t> *self = (Vector<uint8_t> *)p_self;
	return (godot_bool)self->has(p_value);
}

void GDAPI godot_packed_byte_array_sort(godot_packed_byte_array *p_self) {
	Vector<uint8_t> *self = (Vector<uint8_t> *)p_self;
	self->sort();
}

void GDAPI godot_packed_byte_array_invert(godot_packed_byte_array *p_self) {
	Vector<uint8_t> *self = (Vector<uint8_t> *)p_self;
	self->invert();
}

void GDAPI godot_packed_byte_array_push_back(godot_packed_byte_array *p_self, const uint8_t p_data) {
	Vector<uint8_t> *self = (Vector<uint8_t> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_packed_byte_array_remove(godot_packed_byte_array *p_self, const godot_int p_idx) {
	Vector<uint8_t> *self = (Vector<uint8_t> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_packed_byte_array_resize(godot_packed_byte_array *p_self, const godot_int p_size) {
	Vector<uint8_t> *self = (Vector<uint8_t> *)p_self;
	self->resize(p_size);
}

void GDAPI godot_packed_byte_array_set(godot_packed_byte_array *p_self, const godot_int p_idx, const uint8_t p_data) {
	Vector<uint8_t> *self = (Vector<uint8_t> *)p_self;
	self->set(p_idx, p_data);
}

uint8_t GDAPI godot_packed_byte_array_get(const godot_packed_byte_array *p_self, const godot_int p_idx) {
	const Vector<uint8_t> *self = (const Vector<uint8_t> *)p_self;
	return self->get(p_idx);
}

godot_int GDAPI godot_packed_byte_array_size(const godot_packed_byte_array *p_self) {
	const Vector<uint8_t> *self = (const Vector<uint8_t> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_packed_byte_array_empty(const godot_packed_byte_array *p_self) {
	const Vector<uint8_t> *self = (const Vector<uint8_t> *)p_self;
	return self->empty();
}

void GDAPI godot_packed_byte_array_destroy(godot_packed_byte_array *p_self) {
	((Vector<uint8_t> *)p_self)->~Vector();
}

// int32

void GDAPI godot_packed_int32_array_new(godot_packed_int32_array *r_dest) {
	Vector<int32_t> *dest = (Vector<int32_t> *)r_dest;
	memnew_placement(dest, Vector<int32_t>);
}

void GDAPI godot_packed_int32_array_new_copy(godot_packed_int32_array *r_dest, const godot_packed_int32_array *p_src) {
	Vector<int32_t> *dest = (Vector<int32_t> *)r_dest;
	const Vector<int32_t> *src = (const Vector<int32_t> *)p_src;
	memnew_placement(dest, Vector<int32_t>(*src));
}

void GDAPI godot_packed_int32_array_new_with_array(godot_packed_int32_array *r_dest, const godot_array *p_a) {
	Vector<int32_t> *dest = (Vector<int32_t> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, Vector<int32_t>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

const int32_t GDAPI *godot_packed_int32_array_ptr(const godot_packed_int32_array *p_self) {
	const Vector<int32_t> *self = (const Vector<int32_t> *)p_self;
	return self->ptr();
}

int32_t GDAPI *godot_packed_int32_array_ptrw(godot_packed_int32_array *p_self) {
	Vector<int32_t> *self = (Vector<int32_t> *)p_self;
	return self->ptrw();
}

void GDAPI godot_packed_int32_array_append(godot_packed_int32_array *p_self, const int32_t p_data) {
	Vector<int32_t> *self = (Vector<int32_t> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_packed_int32_array_append_array(godot_packed_int32_array *p_self, const godot_packed_int32_array *p_array) {
	Vector<int32_t> *self = (Vector<int32_t> *)p_self;
	Vector<int32_t> *array = (Vector<int32_t> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_packed_int32_array_insert(godot_packed_int32_array *p_self, const godot_int p_idx, const int32_t p_data) {
	Vector<int32_t> *self = (Vector<int32_t> *)p_self;
	return (godot_error)self->insert(p_idx, p_data);
}

godot_bool GDAPI godot_packed_int32_array_has(godot_packed_int32_array *p_self, const int32_t p_value) {
	Vector<int32_t> *self = (Vector<int32_t> *)p_self;
	return (godot_bool)self->has(p_value);
}

void GDAPI godot_packed_int32_array_sort(godot_packed_int32_array *p_self) {
	Vector<int32_t> *self = (Vector<int32_t> *)p_self;
	self->sort();
}

void GDAPI godot_packed_int32_array_invert(godot_packed_int32_array *p_self) {
	Vector<int32_t> *self = (Vector<int32_t> *)p_self;
	self->invert();
}

void GDAPI godot_packed_int32_array_push_back(godot_packed_int32_array *p_self, const int32_t p_data) {
	Vector<int32_t> *self = (Vector<int32_t> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_packed_int32_array_remove(godot_packed_int32_array *p_self, const godot_int p_idx) {
	Vector<int32_t> *self = (Vector<int32_t> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_packed_int32_array_resize(godot_packed_int32_array *p_self, const godot_int p_size) {
	Vector<int32_t> *self = (Vector<int32_t> *)p_self;
	self->resize(p_size);
}

void GDAPI godot_packed_int32_array_set(godot_packed_int32_array *p_self, const godot_int p_idx, const int32_t p_data) {
	Vector<int32_t> *self = (Vector<int32_t> *)p_self;
	self->set(p_idx, p_data);
}

int32_t GDAPI godot_packed_int32_array_get(const godot_packed_int32_array *p_self, const godot_int p_idx) {
	const Vector<int32_t> *self = (const Vector<int32_t> *)p_self;
	return self->get(p_idx);
}

godot_int GDAPI godot_packed_int32_array_size(const godot_packed_int32_array *p_self) {
	const Vector<int32_t> *self = (const Vector<int32_t> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_packed_int32_array_empty(const godot_packed_int32_array *p_self) {
	const Vector<int32_t> *self = (const Vector<int32_t> *)p_self;
	return self->empty();
}

void GDAPI godot_packed_int32_array_destroy(godot_packed_int32_array *p_self) {
	((Vector<int32_t> *)p_self)->~Vector();
}

// int64

void GDAPI godot_packed_int64_array_new(godot_packed_int64_array *r_dest) {
	Vector<int64_t> *dest = (Vector<int64_t> *)r_dest;
	memnew_placement(dest, Vector<int64_t>);
}

void GDAPI godot_packed_int64_array_new_copy(godot_packed_int64_array *r_dest, const godot_packed_int64_array *p_src) {
	Vector<int64_t> *dest = (Vector<int64_t> *)r_dest;
	const Vector<int64_t> *src = (const Vector<int64_t> *)p_src;
	memnew_placement(dest, Vector<int64_t>(*src));
}

void GDAPI godot_packed_int64_array_new_with_array(godot_packed_int64_array *r_dest, const godot_array *p_a) {
	Vector<int64_t> *dest = (Vector<int64_t> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, Vector<int64_t>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

const int64_t GDAPI *godot_packed_int64_array_ptr(const godot_packed_int64_array *p_self) {
	const Vector<int64_t> *self = (const Vector<int64_t> *)p_self;
	return self->ptr();
}

int64_t GDAPI *godot_packed_int64_array_ptrw(godot_packed_int64_array *p_self) {
	Vector<int64_t> *self = (Vector<int64_t> *)p_self;
	return self->ptrw();
}

void GDAPI godot_packed_int64_array_append(godot_packed_int64_array *p_self, const int64_t p_data) {
	Vector<int64_t> *self = (Vector<int64_t> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_packed_int64_array_append_array(godot_packed_int64_array *p_self, const godot_packed_int64_array *p_array) {
	Vector<int64_t> *self = (Vector<int64_t> *)p_self;
	Vector<int64_t> *array = (Vector<int64_t> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_packed_int64_array_insert(godot_packed_int64_array *p_self, const godot_int p_idx, const int64_t p_data) {
	Vector<int64_t> *self = (Vector<int64_t> *)p_self;
	return (godot_error)self->insert(p_idx, p_data);
}

godot_bool GDAPI godot_packed_int64_array_has(godot_packed_int64_array *p_self, const int64_t p_value) {
	Vector<int64_t> *self = (Vector<int64_t> *)p_self;
	return (godot_bool)self->has(p_value);
}

void GDAPI godot_packed_int64_array_sort(godot_packed_int64_array *p_self) {
	Vector<int64_t> *self = (Vector<int64_t> *)p_self;
	self->sort();
}

void GDAPI godot_packed_int64_array_invert(godot_packed_int64_array *p_self) {
	Vector<int64_t> *self = (Vector<int64_t> *)p_self;
	self->invert();
}

void GDAPI godot_packed_int64_array_push_back(godot_packed_int64_array *p_self, const int64_t p_data) {
	Vector<int64_t> *self = (Vector<int64_t> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_packed_int64_array_remove(godot_packed_int64_array *p_self, const godot_int p_idx) {
	Vector<int64_t> *self = (Vector<int64_t> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_packed_int64_array_resize(godot_packed_int64_array *p_self, const godot_int p_size) {
	Vector<int64_t> *self = (Vector<int64_t> *)p_self;
	self->resize(p_size);
}

void GDAPI godot_packed_int64_array_set(godot_packed_int64_array *p_self, const godot_int p_idx, const int64_t p_data) {
	Vector<int64_t> *self = (Vector<int64_t> *)p_self;
	self->set(p_idx, p_data);
}

int64_t GDAPI godot_packed_int64_array_get(const godot_packed_int64_array *p_self, const godot_int p_idx) {
	const Vector<int64_t> *self = (const Vector<int64_t> *)p_self;
	return self->get(p_idx);
}

godot_int GDAPI godot_packed_int64_array_size(const godot_packed_int64_array *p_self) {
	const Vector<int64_t> *self = (const Vector<int64_t> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_packed_int64_array_empty(const godot_packed_int64_array *p_self) {
	const Vector<int64_t> *self = (const Vector<int64_t> *)p_self;
	return self->empty();
}

void GDAPI godot_packed_int64_array_destroy(godot_packed_int64_array *p_self) {
	((Vector<int64_t> *)p_self)->~Vector();
}

// float32

void GDAPI godot_packed_float32_array_new(godot_packed_float32_array *r_dest) {
	Vector<float> *dest = (Vector<float> *)r_dest;
	memnew_placement(dest, Vector<float>);
}

void GDAPI godot_packed_float32_array_new_copy(godot_packed_float32_array *r_dest, const godot_packed_float32_array *p_src) {
	Vector<float> *dest = (Vector<float> *)r_dest;
	const Vector<float> *src = (const Vector<float> *)p_src;
	memnew_placement(dest, Vector<float>(*src));
}

void GDAPI godot_packed_float32_array_new_with_array(godot_packed_float32_array *r_dest, const godot_array *p_a) {
	Vector<float> *dest = (Vector<float> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, Vector<float>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

const float GDAPI *godot_packed_float32_array_ptr(const godot_packed_float32_array *p_self) {
	const Vector<float> *self = (const Vector<float> *)p_self;
	return self->ptr();
}

float GDAPI *godot_packed_float32_array_ptrw(godot_packed_float32_array *p_self) {
	Vector<float> *self = (Vector<float> *)p_self;
	return self->ptrw();
}

void GDAPI godot_packed_float32_array_append(godot_packed_float32_array *p_self, const float p_data) {
	Vector<float> *self = (Vector<float> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_packed_float32_array_append_array(godot_packed_float32_array *p_self, const godot_packed_float32_array *p_array) {
	Vector<float> *self = (Vector<float> *)p_self;
	Vector<float> *array = (Vector<float> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_packed_float32_array_insert(godot_packed_float32_array *p_self, const godot_int p_idx, const float p_data) {
	Vector<float> *self = (Vector<float> *)p_self;
	return (godot_error)self->insert(p_idx, p_data);
}

godot_bool GDAPI godot_packed_float32_array_has(godot_packed_float32_array *p_self, const float p_value) {
	Vector<float> *self = (Vector<float> *)p_self;
	return (godot_bool)self->has(p_value);
}

void GDAPI godot_packed_float32_array_sort(godot_packed_float32_array *p_self) {
	Vector<float> *self = (Vector<float> *)p_self;
	self->sort();
}

void GDAPI godot_packed_float32_array_invert(godot_packed_float32_array *p_self) {
	Vector<float> *self = (Vector<float> *)p_self;
	self->invert();
}

void GDAPI godot_packed_float32_array_push_back(godot_packed_float32_array *p_self, const float p_data) {
	Vector<float> *self = (Vector<float> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_packed_float32_array_remove(godot_packed_float32_array *p_self, const godot_int p_idx) {
	Vector<float> *self = (Vector<float> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_packed_float32_array_resize(godot_packed_float32_array *p_self, const godot_int p_size) {
	Vector<float> *self = (Vector<float> *)p_self;
	self->resize(p_size);
}

void GDAPI godot_packed_float32_array_set(godot_packed_float32_array *p_self, const godot_int p_idx, const float p_data) {
	Vector<float> *self = (Vector<float> *)p_self;
	self->set(p_idx, p_data);
}

float GDAPI godot_packed_float32_array_get(const godot_packed_float32_array *p_self, const godot_int p_idx) {
	const Vector<float> *self = (const Vector<float> *)p_self;
	return self->get(p_idx);
}

godot_int GDAPI godot_packed_float32_array_size(const godot_packed_float32_array *p_self) {
	const Vector<float> *self = (const Vector<float> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_packed_float32_array_empty(const godot_packed_float32_array *p_self) {
	const Vector<float> *self = (const Vector<float> *)p_self;
	return self->empty();
}

void GDAPI godot_packed_float32_array_destroy(godot_packed_float32_array *p_self) {
	((Vector<float> *)p_self)->~Vector();
}

// float64

void GDAPI godot_packed_float64_array_new(godot_packed_float64_array *r_dest) {
	Vector<double> *dest = (Vector<double> *)r_dest;
	memnew_placement(dest, Vector<double>);
}

void GDAPI godot_packed_float64_array_new_copy(godot_packed_float64_array *r_dest, const godot_packed_float64_array *p_src) {
	Vector<double> *dest = (Vector<double> *)r_dest;
	const Vector<double> *src = (const Vector<double> *)p_src;
	memnew_placement(dest, Vector<double>(*src));
}

void GDAPI godot_packed_float64_array_new_with_array(godot_packed_float64_array *r_dest, const godot_array *p_a) {
	Vector<double> *dest = (Vector<double> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, Vector<double>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

const double GDAPI *godot_packed_float64_array_ptr(const godot_packed_float64_array *p_self) {
	const Vector<double> *self = (const Vector<double> *)p_self;
	return self->ptr();
}

double GDAPI *godot_packed_float64_array_ptrw(godot_packed_float64_array *p_self) {
	Vector<double> *self = (Vector<double> *)p_self;
	return self->ptrw();
}

void GDAPI godot_packed_float64_array_append(godot_packed_float64_array *p_self, const double p_data) {
	Vector<double> *self = (Vector<double> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_packed_float64_array_append_array(godot_packed_float64_array *p_self, const godot_packed_float64_array *p_array) {
	Vector<double> *self = (Vector<double> *)p_self;
	Vector<double> *array = (Vector<double> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_packed_float64_array_insert(godot_packed_float64_array *p_self, const godot_int p_idx, const double p_data) {
	Vector<double> *self = (Vector<double> *)p_self;
	return (godot_error)self->insert(p_idx, p_data);
}

godot_bool GDAPI godot_packed_float64_array_has(godot_packed_float64_array *p_self, const double p_value) {
	Vector<double> *self = (Vector<double> *)p_self;
	return (godot_bool)self->has(p_value);
}

void GDAPI godot_packed_float64_array_sort(godot_packed_float64_array *p_self) {
	Vector<double> *self = (Vector<double> *)p_self;
	self->sort();
}

void GDAPI godot_packed_float64_array_invert(godot_packed_float64_array *p_self) {
	Vector<double> *self = (Vector<double> *)p_self;
	self->invert();
}

void GDAPI godot_packed_float64_array_push_back(godot_packed_float64_array *p_self, const double p_data) {
	Vector<double> *self = (Vector<double> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_packed_float64_array_remove(godot_packed_float64_array *p_self, const godot_int p_idx) {
	Vector<double> *self = (Vector<double> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_packed_float64_array_resize(godot_packed_float64_array *p_self, const godot_int p_size) {
	Vector<double> *self = (Vector<double> *)p_self;
	self->resize(p_size);
}

void GDAPI godot_packed_float64_array_set(godot_packed_float64_array *p_self, const godot_int p_idx, const double p_data) {
	Vector<double> *self = (Vector<double> *)p_self;
	self->set(p_idx, p_data);
}

double GDAPI godot_packed_float64_array_get(const godot_packed_float64_array *p_self, const godot_int p_idx) {
	const Vector<double> *self = (const Vector<double> *)p_self;
	return self->get(p_idx);
}

godot_int GDAPI godot_packed_float64_array_size(const godot_packed_float64_array *p_self) {
	const Vector<double> *self = (const Vector<double> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_packed_float64_array_empty(const godot_packed_float64_array *p_self) {
	const Vector<double> *self = (const Vector<double> *)p_self;
	return self->empty();
}

void GDAPI godot_packed_float64_array_destroy(godot_packed_float64_array *p_self) {
	((Vector<double> *)p_self)->~Vector();
}

// string

void GDAPI godot_packed_string_array_new(godot_packed_string_array *r_dest) {
	Vector<String> *dest = (Vector<String> *)r_dest;
	memnew_placement(dest, Vector<String>);
}

void GDAPI godot_packed_string_array_new_copy(godot_packed_string_array *r_dest, const godot_packed_string_array *p_src) {
	Vector<String> *dest = (Vector<String> *)r_dest;
	const Vector<String> *src = (const Vector<String> *)p_src;
	memnew_placement(dest, Vector<String>(*src));
}

void GDAPI godot_packed_string_array_new_with_array(godot_packed_string_array *r_dest, const godot_array *p_a) {
	Vector<String> *dest = (Vector<String> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, Vector<String>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

const godot_string GDAPI *godot_packed_string_array_ptr(const godot_packed_string_array *p_self) {
	const Vector<String> *self = (const Vector<String> *)p_self;
	return (const godot_string *)self->ptr();
}

godot_string GDAPI *godot_packed_string_array_ptrw(godot_packed_string_array *p_self) {
	Vector<String> *self = (Vector<String> *)p_self;
	return (godot_string *)self->ptrw();
}

void GDAPI godot_packed_string_array_append(godot_packed_string_array *p_self, const godot_string *p_data) {
	Vector<String> *self = (Vector<String> *)p_self;
	String &s = *(String *)p_data;
	self->push_back(s);
}

void GDAPI godot_packed_string_array_append_array(godot_packed_string_array *p_self, const godot_packed_string_array *p_array) {
	Vector<String> *self = (Vector<String> *)p_self;
	Vector<String> *array = (Vector<String> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_packed_string_array_insert(godot_packed_string_array *p_self, const godot_int p_idx, const godot_string *p_data) {
	Vector<String> *self = (Vector<String> *)p_self;
	String &s = *(String *)p_data;
	return (godot_error)self->insert(p_idx, s);
}

godot_bool GDAPI godot_packed_string_array_has(godot_packed_string_array *p_self, const godot_string *p_value) {
	Vector<String> *self = (Vector<String> *)p_self;
	String &s = *(String *)p_value;
	return (godot_bool)self->has(s);
}

void GDAPI godot_packed_string_array_sort(godot_packed_string_array *p_self) {
	Vector<String> *self = (Vector<String> *)p_self;
	self->sort();
}

void GDAPI godot_packed_string_array_invert(godot_packed_string_array *p_self) {
	Vector<String> *self = (Vector<String> *)p_self;
	self->invert();
}

void GDAPI godot_packed_string_array_push_back(godot_packed_string_array *p_self, const godot_string *p_data) {
	Vector<String> *self = (Vector<String> *)p_self;
	String &s = *(String *)p_data;
	self->push_back(s);
}

void GDAPI godot_packed_string_array_remove(godot_packed_string_array *p_self, const godot_int p_idx) {
	Vector<String> *self = (Vector<String> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_packed_string_array_resize(godot_packed_string_array *p_self, const godot_int p_size) {
	Vector<String> *self = (Vector<String> *)p_self;
	self->resize(p_size);
}

void GDAPI godot_packed_string_array_set(godot_packed_string_array *p_self, const godot_int p_idx, const godot_string *p_data) {
	Vector<String> *self = (Vector<String> *)p_self;
	String &s = *(String *)p_data;
	self->set(p_idx, s);
}

godot_string GDAPI godot_packed_string_array_get(const godot_packed_string_array *p_self, const godot_int p_idx) {
	const Vector<String> *self = (const Vector<String> *)p_self;
	godot_string str;
	String *s = (String *)&str;
	memnew_placement(s, String);
	*s = self->get(p_idx);
	return str;
}

godot_int GDAPI godot_packed_string_array_size(const godot_packed_string_array *p_self) {
	const Vector<String> *self = (const Vector<String> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_packed_string_array_empty(const godot_packed_string_array *p_self) {
	const Vector<String> *self = (const Vector<String> *)p_self;
	return self->empty();
}

void GDAPI godot_packed_string_array_destroy(godot_packed_string_array *p_self) {
	((Vector<String> *)p_self)->~Vector();
}

// vector2

void GDAPI godot_packed_vector2_array_new(godot_packed_vector2_array *r_dest) {
	Vector<Vector2> *dest = (Vector<Vector2> *)r_dest;
	memnew_placement(dest, Vector<Vector2>);
}

void GDAPI godot_packed_vector2_array_new_copy(godot_packed_vector2_array *r_dest, const godot_packed_vector2_array *p_src) {
	Vector<Vector2> *dest = (Vector<Vector2> *)r_dest;
	const Vector<Vector2> *src = (const Vector<Vector2> *)p_src;
	memnew_placement(dest, Vector<Vector2>(*src));
}

void GDAPI godot_packed_vector2_array_new_with_array(godot_packed_vector2_array *r_dest, const godot_array *p_a) {
	Vector<Vector2> *dest = (Vector<Vector2> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, Vector<Vector2>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

const godot_vector2 GDAPI *godot_packed_vector2_array_ptr(const godot_packed_vector2_array *p_self) {
	const Vector<Vector2> *self = (const Vector<Vector2> *)p_self;
	return (const godot_vector2 *)self->ptr();
}

godot_vector2 GDAPI *godot_packed_vector2_array_ptrw(godot_packed_vector2_array *p_self) {
	Vector<Vector2> *self = (Vector<Vector2> *)p_self;
	return (godot_vector2 *)self->ptrw();
}

void GDAPI godot_packed_vector2_array_append(godot_packed_vector2_array *p_self, const godot_vector2 *p_data) {
	Vector<Vector2> *self = (Vector<Vector2> *)p_self;
	Vector2 &s = *(Vector2 *)p_data;
	self->push_back(s);
}

void GDAPI godot_packed_vector2_array_append_array(godot_packed_vector2_array *p_self, const godot_packed_vector2_array *p_array) {
	Vector<Vector2> *self = (Vector<Vector2> *)p_self;
	Vector<Vector2> *array = (Vector<Vector2> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_packed_vector2_array_insert(godot_packed_vector2_array *p_self, const godot_int p_idx, const godot_vector2 *p_data) {
	Vector<Vector2> *self = (Vector<Vector2> *)p_self;
	Vector2 &s = *(Vector2 *)p_data;
	return (godot_error)self->insert(p_idx, s);
}

godot_bool GDAPI godot_packed_vector2_array_has(godot_packed_vector2_array *p_self, const godot_vector2 *p_value) {
	Vector<Vector2> *self = (Vector<Vector2> *)p_self;
	Vector2 &v = *(Vector2 *)p_value;
	return (godot_bool)self->has(v);
}

void GDAPI godot_packed_vector2_array_sort(godot_packed_vector2_array *p_self) {
	Vector<Vector2> *self = (Vector<Vector2> *)p_self;
	self->sort();
}

void GDAPI godot_packed_vector2_array_invert(godot_packed_vector2_array *p_self) {
	Vector<Vector2> *self = (Vector<Vector2> *)p_self;
	self->invert();
}

void GDAPI godot_packed_vector2_array_push_back(godot_packed_vector2_array *p_self, const godot_vector2 *p_data) {
	Vector<Vector2> *self = (Vector<Vector2> *)p_self;
	Vector2 &s = *(Vector2 *)p_data;
	self->push_back(s);
}

void GDAPI godot_packed_vector2_array_remove(godot_packed_vector2_array *p_self, const godot_int p_idx) {
	Vector<Vector2> *self = (Vector<Vector2> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_packed_vector2_array_resize(godot_packed_vector2_array *p_self, const godot_int p_size) {
	Vector<Vector2> *self = (Vector<Vector2> *)p_self;
	self->resize(p_size);
}

void GDAPI godot_packed_vector2_array_set(godot_packed_vector2_array *p_self, const godot_int p_idx, const godot_vector2 *p_data) {
	Vector<Vector2> *self = (Vector<Vector2> *)p_self;
	Vector2 &s = *(Vector2 *)p_data;
	self->set(p_idx, s);
}

godot_vector2 GDAPI godot_packed_vector2_array_get(const godot_packed_vector2_array *p_self, const godot_int p_idx) {
	const Vector<Vector2> *self = (const Vector<Vector2> *)p_self;
	godot_vector2 v;
	Vector2 *s = (Vector2 *)&v;
	*s = self->get(p_idx);
	return v;
}

godot_int GDAPI godot_packed_vector2_array_size(const godot_packed_vector2_array *p_self) {
	const Vector<Vector2> *self = (const Vector<Vector2> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_packed_vector2_array_empty(const godot_packed_vector2_array *p_self) {
	const Vector<Vector2> *self = (const Vector<Vector2> *)p_self;
	return self->empty();
}

void GDAPI godot_packed_vector2_array_destroy(godot_packed_vector2_array *p_self) {
	((Vector<Vector2> *)p_self)->~Vector();
}

// vector3

void GDAPI godot_packed_vector3_array_new(godot_packed_vector3_array *r_dest) {
	Vector<Vector3> *dest = (Vector<Vector3> *)r_dest;
	memnew_placement(dest, Vector<Vector3>);
}

void GDAPI godot_packed_vector3_array_new_copy(godot_packed_vector3_array *r_dest, const godot_packed_vector3_array *p_src) {
	Vector<Vector3> *dest = (Vector<Vector3> *)r_dest;
	const Vector<Vector3> *src = (const Vector<Vector3> *)p_src;
	memnew_placement(dest, Vector<Vector3>(*src));
}

void GDAPI godot_packed_vector3_array_new_with_array(godot_packed_vector3_array *r_dest, const godot_array *p_a) {
	Vector<Vector3> *dest = (Vector<Vector3> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, Vector<Vector3>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

const godot_vector3 GDAPI *godot_packed_vector3_array_ptr(const godot_packed_vector3_array *p_self) {
	const Vector<Vector3> *self = (const Vector<Vector3> *)p_self;
	return (const godot_vector3 *)self->ptr();
}

godot_vector3 GDAPI *godot_packed_vector3_array_ptrw(godot_packed_vector3_array *p_self) {
	Vector<Vector3> *self = (Vector<Vector3> *)p_self;
	return (godot_vector3 *)self->ptrw();
}

void GDAPI godot_packed_vector3_array_append(godot_packed_vector3_array *p_self, const godot_vector3 *p_data) {
	Vector<Vector3> *self = (Vector<Vector3> *)p_self;
	Vector3 &s = *(Vector3 *)p_data;
	self->push_back(s);
}

void GDAPI godot_packed_vector3_array_append_array(godot_packed_vector3_array *p_self, const godot_packed_vector3_array *p_array) {
	Vector<Vector3> *self = (Vector<Vector3> *)p_self;
	Vector<Vector3> *array = (Vector<Vector3> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_packed_vector3_array_insert(godot_packed_vector3_array *p_self, const godot_int p_idx, const godot_vector3 *p_data) {
	Vector<Vector3> *self = (Vector<Vector3> *)p_self;
	Vector3 &s = *(Vector3 *)p_data;
	return (godot_error)self->insert(p_idx, s);
}

godot_bool GDAPI godot_packed_vector3_array_has(godot_packed_vector3_array *p_self, const godot_vector3 *p_value) {
	Vector<Vector3> *self = (Vector<Vector3> *)p_self;
	Vector3 &v = *(Vector3 *)p_value;
	return (godot_bool)self->has(v);
}

void GDAPI godot_packed_vector3_array_sort(godot_packed_vector3_array *p_self) {
	Vector<Vector3> *self = (Vector<Vector3> *)p_self;
	self->sort();
}

void GDAPI godot_packed_vector3_array_invert(godot_packed_vector3_array *p_self) {
	Vector<Vector3> *self = (Vector<Vector3> *)p_self;
	self->invert();
}

void GDAPI godot_packed_vector3_array_push_back(godot_packed_vector3_array *p_self, const godot_vector3 *p_data) {
	Vector<Vector3> *self = (Vector<Vector3> *)p_self;
	Vector3 &s = *(Vector3 *)p_data;
	self->push_back(s);
}

void GDAPI godot_packed_vector3_array_remove(godot_packed_vector3_array *p_self, const godot_int p_idx) {
	Vector<Vector3> *self = (Vector<Vector3> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_packed_vector3_array_resize(godot_packed_vector3_array *p_self, const godot_int p_size) {
	Vector<Vector3> *self = (Vector<Vector3> *)p_self;
	self->resize(p_size);
}

void GDAPI godot_packed_vector3_array_set(godot_packed_vector3_array *p_self, const godot_int p_idx, const godot_vector3 *p_data) {
	Vector<Vector3> *self = (Vector<Vector3> *)p_self;
	Vector3 &s = *(Vector3 *)p_data;
	self->set(p_idx, s);
}

godot_vector3 GDAPI godot_packed_vector3_array_get(const godot_packed_vector3_array *p_self, const godot_int p_idx) {
	const Vector<Vector3> *self = (const Vector<Vector3> *)p_self;
	godot_vector3 v;
	Vector3 *s = (Vector3 *)&v;
	*s = self->get(p_idx);
	return v;
}

godot_int GDAPI godot_packed_vector3_array_size(const godot_packed_vector3_array *p_self) {
	const Vector<Vector3> *self = (const Vector<Vector3> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_packed_vector3_array_empty(const godot_packed_vector3_array *p_self) {
	const Vector<Vector3> *self = (const Vector<Vector3> *)p_self;
	return self->empty();
}

void GDAPI godot_packed_vector3_array_destroy(godot_packed_vector3_array *p_self) {
	((Vector<Vector3> *)p_self)->~Vector();
}

// color

void GDAPI godot_packed_color_array_new(godot_packed_color_array *r_dest) {
	Vector<Color> *dest = (Vector<Color> *)r_dest;
	memnew_placement(dest, Vector<Color>);
}

void GDAPI godot_packed_color_array_new_copy(godot_packed_color_array *r_dest, const godot_packed_color_array *p_src) {
	Vector<Color> *dest = (Vector<Color> *)r_dest;
	const Vector<Color> *src = (const Vector<Color> *)p_src;
	memnew_placement(dest, Vector<Color>(*src));
}

void GDAPI godot_packed_color_array_new_with_array(godot_packed_color_array *r_dest, const godot_array *p_a) {
	Vector<Color> *dest = (Vector<Color> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, Vector<Color>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

const godot_color GDAPI *godot_packed_color_array_ptr(const godot_packed_color_array *p_self) {
	const Vector<Color> *self = (const Vector<Color> *)p_self;
	return (const godot_color *)self->ptr();
}

godot_color GDAPI *godot_packed_color_array_ptrw(godot_packed_color_array *p_self) {
	Vector<Color> *self = (Vector<Color> *)p_self;
	return (godot_color *)self->ptrw();
}

void GDAPI godot_packed_color_array_append(godot_packed_color_array *p_self, const godot_color *p_data) {
	Vector<Color> *self = (Vector<Color> *)p_self;
	Color &s = *(Color *)p_data;
	self->push_back(s);
}

void GDAPI godot_packed_color_array_append_array(godot_packed_color_array *p_self, const godot_packed_color_array *p_array) {
	Vector<Color> *self = (Vector<Color> *)p_self;
	Vector<Color> *array = (Vector<Color> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_packed_color_array_insert(godot_packed_color_array *p_self, const godot_int p_idx, const godot_color *p_data) {
	Vector<Color> *self = (Vector<Color> *)p_self;
	Color &s = *(Color *)p_data;
	return (godot_error)self->insert(p_idx, s);
}

godot_bool GDAPI godot_packed_color_array_has(godot_packed_color_array *p_self, const godot_color *p_value) {
	Vector<Color> *self = (Vector<Color> *)p_self;
	Color &c = *(Color *)p_value;
	return (godot_bool)self->has(c);
}

void GDAPI godot_packed_color_array_sort(godot_packed_color_array *p_self) {
	Vector<Color> *self = (Vector<Color> *)p_self;
	self->sort();
}

void GDAPI godot_packed_color_array_invert(godot_packed_color_array *p_self) {
	Vector<Color> *self = (Vector<Color> *)p_self;
	self->invert();
}

void GDAPI godot_packed_color_array_push_back(godot_packed_color_array *p_self, const godot_color *p_data) {
	Vector<Color> *self = (Vector<Color> *)p_self;
	Color &s = *(Color *)p_data;
	self->push_back(s);
}

void GDAPI godot_packed_color_array_remove(godot_packed_color_array *p_self, const godot_int p_idx) {
	Vector<Color> *self = (Vector<Color> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_packed_color_array_resize(godot_packed_color_array *p_self, const godot_int p_size) {
	Vector<Color> *self = (Vector<Color> *)p_self;
	self->resize(p_size);
}

void GDAPI godot_packed_color_array_set(godot_packed_color_array *p_self, const godot_int p_idx, const godot_color *p_data) {
	Vector<Color> *self = (Vector<Color> *)p_self;
	Color &s = *(Color *)p_data;
	self->set(p_idx, s);
}

godot_color GDAPI godot_packed_color_array_get(const godot_packed_color_array *p_self, const godot_int p_idx) {
	const Vector<Color> *self = (const Vector<Color> *)p_self;
	godot_color v;
	Color *s = (Color *)&v;
	*s = self->get(p_idx);
	return v;
}

godot_int GDAPI godot_packed_color_array_size(const godot_packed_color_array *p_self) {
	const Vector<Color> *self = (const Vector<Color> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_packed_color_array_empty(const godot_packed_color_array *p_self) {
	const Vector<Color> *self = (const Vector<Color> *)p_self;
	return self->empty();
}

void GDAPI godot_packed_color_array_destroy(godot_packed_color_array *p_self) {
	((Vector<Color> *)p_self)->~Vector();
}

#ifdef __cplusplus
}
#endif
