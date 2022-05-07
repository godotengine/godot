/*************************************************************************/
/*  pool_arrays.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gdnative/pool_arrays.h"

#include "core/array.h"
#include "core/pool_vector.h"
#include "core/variant.h"

#include "core/color.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"

#ifdef __cplusplus
extern "C" {
#endif

static_assert(sizeof(godot_pool_byte_array) == sizeof(PoolVector<uint8_t>), "PoolVector<uint8_t> size mismatch");
static_assert(sizeof(godot_pool_int_array) == sizeof(PoolVector<godot_int>), "PoolVector<godot_int> size mismatch");
static_assert(sizeof(godot_pool_real_array) == sizeof(PoolVector<godot_real>), "PoolVector<godot_real> size mismatch");
static_assert(sizeof(godot_pool_string_array) == sizeof(PoolVector<String>), "PoolVector<String> size mismatch");
static_assert(sizeof(godot_pool_vector2_array) == sizeof(PoolVector<Vector2>), "PoolVector<Vector2> size mismatch");
static_assert(sizeof(godot_pool_vector3_array) == sizeof(PoolVector<Vector3>), "PoolVector<Vector3> size mismatch");
static_assert(sizeof(godot_pool_color_array) == sizeof(PoolVector<Color>), "PoolVector<Color> size mismatch");

#define memnew_placement_custom(m_placement, m_class, m_constr) _post_initialize(new (m_placement, sizeof(m_class), "") m_constr)

// byte

void GDAPI godot_pool_byte_array_new(godot_pool_byte_array *r_dest) {
	PoolVector<uint8_t> *dest = (PoolVector<uint8_t> *)r_dest;
	memnew_placement(dest, PoolVector<uint8_t>);
}

void GDAPI godot_pool_byte_array_new_copy(godot_pool_byte_array *r_dest, const godot_pool_byte_array *p_src) {
	PoolVector<uint8_t> *dest = (PoolVector<uint8_t> *)r_dest;
	const PoolVector<uint8_t> *src = (const PoolVector<uint8_t> *)p_src;
	memnew_placement(dest, PoolVector<uint8_t>(*src));
}

void GDAPI godot_pool_byte_array_new_with_array(godot_pool_byte_array *r_dest, const godot_array *p_a) {
	PoolVector<uint8_t> *dest = (PoolVector<uint8_t> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, PoolVector<uint8_t>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_byte_array_append(godot_pool_byte_array *p_self, const uint8_t p_data) {
	PoolVector<uint8_t> *self = (PoolVector<uint8_t> *)p_self;
	self->append(p_data);
}

void GDAPI godot_pool_byte_array_append_array(godot_pool_byte_array *p_self, const godot_pool_byte_array *p_array) {
	PoolVector<uint8_t> *self = (PoolVector<uint8_t> *)p_self;
	PoolVector<uint8_t> *array = (PoolVector<uint8_t> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_pool_byte_array_insert(godot_pool_byte_array *p_self, const godot_int p_idx, const uint8_t p_data) {
	PoolVector<uint8_t> *self = (PoolVector<uint8_t> *)p_self;
	return (godot_error)self->insert(p_idx, p_data);
}

void GDAPI godot_pool_byte_array_invert(godot_pool_byte_array *p_self) {
	PoolVector<uint8_t> *self = (PoolVector<uint8_t> *)p_self;
	self->invert();
}

void GDAPI godot_pool_byte_array_push_back(godot_pool_byte_array *p_self, const uint8_t p_data) {
	PoolVector<uint8_t> *self = (PoolVector<uint8_t> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_pool_byte_array_remove(godot_pool_byte_array *p_self, const godot_int p_idx) {
	PoolVector<uint8_t> *self = (PoolVector<uint8_t> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_pool_byte_array_resize(godot_pool_byte_array *p_self, const godot_int p_size) {
	PoolVector<uint8_t> *self = (PoolVector<uint8_t> *)p_self;
	self->resize(p_size);
}

godot_pool_byte_array_read_access GDAPI *godot_pool_byte_array_read(const godot_pool_byte_array *p_self) {
	const PoolVector<uint8_t> *self = (const PoolVector<uint8_t> *)p_self;
	return (godot_pool_byte_array_read_access *)memnew(PoolVector<uint8_t>::Read(self->read()));
}

godot_pool_byte_array_write_access GDAPI *godot_pool_byte_array_write(godot_pool_byte_array *p_self) {
	PoolVector<uint8_t> *self = (PoolVector<uint8_t> *)p_self;
	return (godot_pool_byte_array_write_access *)memnew(PoolVector<uint8_t>::Write(self->write()));
}

void GDAPI godot_pool_byte_array_set(godot_pool_byte_array *p_self, const godot_int p_idx, const uint8_t p_data) {
	PoolVector<uint8_t> *self = (PoolVector<uint8_t> *)p_self;
	self->set(p_idx, p_data);
}

uint8_t GDAPI godot_pool_byte_array_get(const godot_pool_byte_array *p_self, const godot_int p_idx) {
	const PoolVector<uint8_t> *self = (const PoolVector<uint8_t> *)p_self;
	return self->get(p_idx);
}

godot_int GDAPI godot_pool_byte_array_size(const godot_pool_byte_array *p_self) {
	const PoolVector<uint8_t> *self = (const PoolVector<uint8_t> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_pool_byte_array_empty(const godot_pool_byte_array *p_self) {
	const PoolVector<uint8_t> *self = (const PoolVector<uint8_t> *)p_self;
	return self->empty();
}

void GDAPI godot_pool_byte_array_destroy(godot_pool_byte_array *p_self) {
	((PoolVector<uint8_t> *)p_self)->~PoolVector();
}

// int

void GDAPI godot_pool_int_array_new(godot_pool_int_array *r_dest) {
	PoolVector<godot_int> *dest = (PoolVector<godot_int> *)r_dest;
	memnew_placement(dest, PoolVector<godot_int>);
}

void GDAPI godot_pool_int_array_new_copy(godot_pool_int_array *r_dest, const godot_pool_int_array *p_src) {
	PoolVector<godot_int> *dest = (PoolVector<godot_int> *)r_dest;
	const PoolVector<godot_int> *src = (const PoolVector<godot_int> *)p_src;
	memnew_placement(dest, PoolVector<godot_int>(*src));
}

void GDAPI godot_pool_int_array_new_with_array(godot_pool_int_array *r_dest, const godot_array *p_a) {
	PoolVector<godot_int> *dest = (PoolVector<godot_int> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, PoolVector<godot_int>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_int_array_append(godot_pool_int_array *p_self, const godot_int p_data) {
	PoolVector<godot_int> *self = (PoolVector<godot_int> *)p_self;
	self->append(p_data);
}

void GDAPI godot_pool_int_array_append_array(godot_pool_int_array *p_self, const godot_pool_int_array *p_array) {
	PoolVector<godot_int> *self = (PoolVector<godot_int> *)p_self;
	PoolVector<godot_int> *array = (PoolVector<godot_int> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_pool_int_array_insert(godot_pool_int_array *p_self, const godot_int p_idx, const godot_int p_data) {
	PoolVector<godot_int> *self = (PoolVector<godot_int> *)p_self;
	return (godot_error)self->insert(p_idx, p_data);
}

void GDAPI godot_pool_int_array_invert(godot_pool_int_array *p_self) {
	PoolVector<godot_int> *self = (PoolVector<godot_int> *)p_self;
	self->invert();
}

void GDAPI godot_pool_int_array_push_back(godot_pool_int_array *p_self, const godot_int p_data) {
	PoolVector<godot_int> *self = (PoolVector<godot_int> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_pool_int_array_remove(godot_pool_int_array *p_self, const godot_int p_idx) {
	PoolVector<godot_int> *self = (PoolVector<godot_int> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_pool_int_array_resize(godot_pool_int_array *p_self, const godot_int p_size) {
	PoolVector<godot_int> *self = (PoolVector<godot_int> *)p_self;
	self->resize(p_size);
}

godot_pool_int_array_read_access GDAPI *godot_pool_int_array_read(const godot_pool_int_array *p_self) {
	const PoolVector<godot_int> *self = (const PoolVector<godot_int> *)p_self;
	return (godot_pool_int_array_read_access *)memnew(PoolVector<godot_int>::Read(self->read()));
}

godot_pool_int_array_write_access GDAPI *godot_pool_int_array_write(godot_pool_int_array *p_self) {
	PoolVector<godot_int> *self = (PoolVector<godot_int> *)p_self;
	return (godot_pool_int_array_write_access *)memnew(PoolVector<godot_int>::Write(self->write()));
}

void GDAPI godot_pool_int_array_set(godot_pool_int_array *p_self, const godot_int p_idx, const godot_int p_data) {
	PoolVector<godot_int> *self = (PoolVector<godot_int> *)p_self;
	self->set(p_idx, p_data);
}

godot_int GDAPI godot_pool_int_array_get(const godot_pool_int_array *p_self, const godot_int p_idx) {
	const PoolVector<godot_int> *self = (const PoolVector<godot_int> *)p_self;
	return self->get(p_idx);
}

godot_int GDAPI godot_pool_int_array_size(const godot_pool_int_array *p_self) {
	const PoolVector<godot_int> *self = (const PoolVector<godot_int> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_pool_int_array_empty(const godot_pool_int_array *p_self) {
	const PoolVector<godot_int> *self = (const PoolVector<godot_int> *)p_self;
	return self->empty();
}

void GDAPI godot_pool_int_array_destroy(godot_pool_int_array *p_self) {
	((PoolVector<godot_int> *)p_self)->~PoolVector();
}

// real

void GDAPI godot_pool_real_array_new(godot_pool_real_array *r_dest) {
	PoolVector<godot_real> *dest = (PoolVector<godot_real> *)r_dest;
	memnew_placement(dest, PoolVector<godot_real>);
}

void GDAPI godot_pool_real_array_new_copy(godot_pool_real_array *r_dest, const godot_pool_real_array *p_src) {
	PoolVector<godot_real> *dest = (PoolVector<godot_real> *)r_dest;
	const PoolVector<godot_real> *src = (const PoolVector<godot_real> *)p_src;
	memnew_placement(dest, PoolVector<godot_real>(*src));
}

void GDAPI godot_pool_real_array_new_with_array(godot_pool_real_array *r_dest, const godot_array *p_a) {
	PoolVector<godot_real> *dest = (PoolVector<godot_real> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, PoolVector<godot_real>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_real_array_append(godot_pool_real_array *p_self, const godot_real p_data) {
	PoolVector<godot_real> *self = (PoolVector<godot_real> *)p_self;
	self->append(p_data);
}

void GDAPI godot_pool_real_array_append_array(godot_pool_real_array *p_self, const godot_pool_real_array *p_array) {
	PoolVector<godot_real> *self = (PoolVector<godot_real> *)p_self;
	PoolVector<godot_real> *array = (PoolVector<godot_real> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_pool_real_array_insert(godot_pool_real_array *p_self, const godot_int p_idx, const godot_real p_data) {
	PoolVector<godot_real> *self = (PoolVector<godot_real> *)p_self;
	return (godot_error)self->insert(p_idx, p_data);
}

void GDAPI godot_pool_real_array_invert(godot_pool_real_array *p_self) {
	PoolVector<godot_real> *self = (PoolVector<godot_real> *)p_self;
	self->invert();
}

void GDAPI godot_pool_real_array_push_back(godot_pool_real_array *p_self, const godot_real p_data) {
	PoolVector<godot_real> *self = (PoolVector<godot_real> *)p_self;
	self->push_back(p_data);
}

void GDAPI godot_pool_real_array_remove(godot_pool_real_array *p_self, const godot_int p_idx) {
	PoolVector<godot_real> *self = (PoolVector<godot_real> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_pool_real_array_resize(godot_pool_real_array *p_self, const godot_int p_size) {
	PoolVector<godot_real> *self = (PoolVector<godot_real> *)p_self;
	self->resize(p_size);
}

godot_pool_real_array_read_access GDAPI *godot_pool_real_array_read(const godot_pool_real_array *p_self) {
	const PoolVector<godot_real> *self = (const PoolVector<godot_real> *)p_self;
	return (godot_pool_real_array_read_access *)memnew(PoolVector<godot_real>::Read(self->read()));
}

godot_pool_int_array_write_access GDAPI *godot_pool_real_array_write(godot_pool_real_array *p_self) {
	PoolVector<godot_real> *self = (PoolVector<godot_real> *)p_self;
	return (godot_pool_real_array_write_access *)memnew(PoolVector<godot_real>::Write(self->write()));
}

void GDAPI godot_pool_real_array_set(godot_pool_real_array *p_self, const godot_int p_idx, const godot_real p_data) {
	PoolVector<godot_real> *self = (PoolVector<godot_real> *)p_self;
	self->set(p_idx, p_data);
}

godot_real GDAPI godot_pool_real_array_get(const godot_pool_real_array *p_self, const godot_int p_idx) {
	const PoolVector<godot_real> *self = (const PoolVector<godot_real> *)p_self;
	return self->get(p_idx);
}

godot_int GDAPI godot_pool_real_array_size(const godot_pool_real_array *p_self) {
	const PoolVector<godot_real> *self = (const PoolVector<godot_real> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_pool_real_array_empty(const godot_pool_real_array *p_self) {
	const PoolVector<godot_real> *self = (const PoolVector<godot_real> *)p_self;
	return self->empty();
}

void GDAPI godot_pool_real_array_destroy(godot_pool_real_array *p_self) {
	((PoolVector<godot_real> *)p_self)->~PoolVector();
}

// string

void GDAPI godot_pool_string_array_new(godot_pool_string_array *r_dest) {
	PoolVector<String> *dest = (PoolVector<String> *)r_dest;
	memnew_placement(dest, PoolVector<String>);
}

void GDAPI godot_pool_string_array_new_copy(godot_pool_string_array *r_dest, const godot_pool_string_array *p_src) {
	PoolVector<String> *dest = (PoolVector<String> *)r_dest;
	const PoolVector<String> *src = (const PoolVector<String> *)p_src;
	memnew_placement(dest, PoolVector<String>(*src));
}

void GDAPI godot_pool_string_array_new_with_array(godot_pool_string_array *r_dest, const godot_array *p_a) {
	PoolVector<String> *dest = (PoolVector<String> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, PoolVector<String>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_string_array_append(godot_pool_string_array *p_self, const godot_string *p_data) {
	PoolVector<String> *self = (PoolVector<String> *)p_self;
	String &s = *(String *)p_data;
	self->append(s);
}

void GDAPI godot_pool_string_array_append_array(godot_pool_string_array *p_self, const godot_pool_string_array *p_array) {
	PoolVector<String> *self = (PoolVector<String> *)p_self;
	PoolVector<String> *array = (PoolVector<String> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_pool_string_array_insert(godot_pool_string_array *p_self, const godot_int p_idx, const godot_string *p_data) {
	PoolVector<String> *self = (PoolVector<String> *)p_self;
	String &s = *(String *)p_data;
	return (godot_error)self->insert(p_idx, s);
}

void GDAPI godot_pool_string_array_invert(godot_pool_string_array *p_self) {
	PoolVector<String> *self = (PoolVector<String> *)p_self;
	self->invert();
}

void GDAPI godot_pool_string_array_push_back(godot_pool_string_array *p_self, const godot_string *p_data) {
	PoolVector<String> *self = (PoolVector<String> *)p_self;
	String &s = *(String *)p_data;
	self->push_back(s);
}

void GDAPI godot_pool_string_array_remove(godot_pool_string_array *p_self, const godot_int p_idx) {
	PoolVector<String> *self = (PoolVector<String> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_pool_string_array_resize(godot_pool_string_array *p_self, const godot_int p_size) {
	PoolVector<String> *self = (PoolVector<String> *)p_self;
	self->resize(p_size);
}

godot_pool_string_array_read_access GDAPI *godot_pool_string_array_read(const godot_pool_string_array *p_self) {
	const PoolVector<String> *self = (const PoolVector<String> *)p_self;
	return (godot_pool_string_array_read_access *)memnew(PoolVector<String>::Read(self->read()));
}

godot_pool_string_array_write_access GDAPI *godot_pool_string_array_write(godot_pool_string_array *p_self) {
	PoolVector<String> *self = (PoolVector<String> *)p_self;
	return (godot_pool_string_array_write_access *)memnew(PoolVector<String>::Write(self->write()));
}

void GDAPI godot_pool_string_array_set(godot_pool_string_array *p_self, const godot_int p_idx, const godot_string *p_data) {
	PoolVector<String> *self = (PoolVector<String> *)p_self;
	String &s = *(String *)p_data;
	self->set(p_idx, s);
}

godot_string GDAPI godot_pool_string_array_get(const godot_pool_string_array *p_self, const godot_int p_idx) {
	const PoolVector<String> *self = (const PoolVector<String> *)p_self;
	godot_string str;
	String *s = (String *)&str;
	memnew_placement(s, String);
	*s = self->get(p_idx);
	return str;
}

godot_int GDAPI godot_pool_string_array_size(const godot_pool_string_array *p_self) {
	const PoolVector<String> *self = (const PoolVector<String> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_pool_string_array_empty(const godot_pool_string_array *p_self) {
	const PoolVector<String> *self = (const PoolVector<String> *)p_self;
	return self->empty();
}

void GDAPI godot_pool_string_array_destroy(godot_pool_string_array *p_self) {
	((PoolVector<String> *)p_self)->~PoolVector();
}

// vector2

void GDAPI godot_pool_vector2_array_new(godot_pool_vector2_array *r_dest) {
	PoolVector<Vector2> *dest = (PoolVector<Vector2> *)r_dest;
	memnew_placement(dest, PoolVector<Vector2>);
}

void GDAPI godot_pool_vector2_array_new_copy(godot_pool_vector2_array *r_dest, const godot_pool_vector2_array *p_src) {
	PoolVector<Vector2> *dest = (PoolVector<Vector2> *)r_dest;
	const PoolVector<Vector2> *src = (const PoolVector<Vector2> *)p_src;
	memnew_placement(dest, PoolVector<Vector2>(*src));
}

void GDAPI godot_pool_vector2_array_new_with_array(godot_pool_vector2_array *r_dest, const godot_array *p_a) {
	PoolVector<Vector2> *dest = (PoolVector<Vector2> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, PoolVector<Vector2>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_vector2_array_append(godot_pool_vector2_array *p_self, const godot_vector2 *p_data) {
	PoolVector<Vector2> *self = (PoolVector<Vector2> *)p_self;
	Vector2 &s = *(Vector2 *)p_data;
	self->append(s);
}

void GDAPI godot_pool_vector2_array_append_array(godot_pool_vector2_array *p_self, const godot_pool_vector2_array *p_array) {
	PoolVector<Vector2> *self = (PoolVector<Vector2> *)p_self;
	PoolVector<Vector2> *array = (PoolVector<Vector2> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_pool_vector2_array_insert(godot_pool_vector2_array *p_self, const godot_int p_idx, const godot_vector2 *p_data) {
	PoolVector<Vector2> *self = (PoolVector<Vector2> *)p_self;
	Vector2 &s = *(Vector2 *)p_data;
	return (godot_error)self->insert(p_idx, s);
}

void GDAPI godot_pool_vector2_array_invert(godot_pool_vector2_array *p_self) {
	PoolVector<Vector2> *self = (PoolVector<Vector2> *)p_self;
	self->invert();
}

void GDAPI godot_pool_vector2_array_push_back(godot_pool_vector2_array *p_self, const godot_vector2 *p_data) {
	PoolVector<Vector2> *self = (PoolVector<Vector2> *)p_self;
	Vector2 &s = *(Vector2 *)p_data;
	self->push_back(s);
}

void GDAPI godot_pool_vector2_array_remove(godot_pool_vector2_array *p_self, const godot_int p_idx) {
	PoolVector<Vector2> *self = (PoolVector<Vector2> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_pool_vector2_array_resize(godot_pool_vector2_array *p_self, const godot_int p_size) {
	PoolVector<Vector2> *self = (PoolVector<Vector2> *)p_self;
	self->resize(p_size);
}

godot_pool_vector2_array_read_access GDAPI *godot_pool_vector2_array_read(const godot_pool_vector2_array *p_self) {
	const PoolVector<Vector2> *self = (const PoolVector<Vector2> *)p_self;
	return (godot_pool_vector2_array_read_access *)memnew(PoolVector<Vector2>::Read(self->read()));
}

godot_pool_vector2_array_write_access GDAPI *godot_pool_vector2_array_write(godot_pool_vector2_array *p_self) {
	PoolVector<Vector2> *self = (PoolVector<Vector2> *)p_self;
	return (godot_pool_vector2_array_write_access *)memnew(PoolVector<Vector2>::Write(self->write()));
}

void GDAPI godot_pool_vector2_array_set(godot_pool_vector2_array *p_self, const godot_int p_idx, const godot_vector2 *p_data) {
	PoolVector<Vector2> *self = (PoolVector<Vector2> *)p_self;
	Vector2 &s = *(Vector2 *)p_data;
	self->set(p_idx, s);
}

godot_vector2 GDAPI godot_pool_vector2_array_get(const godot_pool_vector2_array *p_self, const godot_int p_idx) {
	const PoolVector<Vector2> *self = (const PoolVector<Vector2> *)p_self;
	godot_vector2 v;
	Vector2 *s = (Vector2 *)&v;
	*s = self->get(p_idx);
	return v;
}

godot_int GDAPI godot_pool_vector2_array_size(const godot_pool_vector2_array *p_self) {
	const PoolVector<Vector2> *self = (const PoolVector<Vector2> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_pool_vector2_array_empty(const godot_pool_vector2_array *p_self) {
	const PoolVector<Vector2> *self = (const PoolVector<Vector2> *)p_self;
	return self->empty();
}

void GDAPI godot_pool_vector2_array_destroy(godot_pool_vector2_array *p_self) {
	((PoolVector<Vector2> *)p_self)->~PoolVector();
}

// vector3

void GDAPI godot_pool_vector3_array_new(godot_pool_vector3_array *r_dest) {
	PoolVector<Vector3> *dest = (PoolVector<Vector3> *)r_dest;
	memnew_placement(dest, PoolVector<Vector3>);
}

void GDAPI godot_pool_vector3_array_new_copy(godot_pool_vector3_array *r_dest, const godot_pool_vector3_array *p_src) {
	PoolVector<Vector3> *dest = (PoolVector<Vector3> *)r_dest;
	const PoolVector<Vector3> *src = (const PoolVector<Vector3> *)p_src;
	memnew_placement(dest, PoolVector<Vector3>(*src));
}

void GDAPI godot_pool_vector3_array_new_with_array(godot_pool_vector3_array *r_dest, const godot_array *p_a) {
	PoolVector<Vector3> *dest = (PoolVector<Vector3> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, PoolVector<Vector3>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_vector3_array_append(godot_pool_vector3_array *p_self, const godot_vector3 *p_data) {
	PoolVector<Vector3> *self = (PoolVector<Vector3> *)p_self;
	Vector3 &s = *(Vector3 *)p_data;
	self->append(s);
}

void GDAPI godot_pool_vector3_array_append_array(godot_pool_vector3_array *p_self, const godot_pool_vector3_array *p_array) {
	PoolVector<Vector3> *self = (PoolVector<Vector3> *)p_self;
	PoolVector<Vector3> *array = (PoolVector<Vector3> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_pool_vector3_array_insert(godot_pool_vector3_array *p_self, const godot_int p_idx, const godot_vector3 *p_data) {
	PoolVector<Vector3> *self = (PoolVector<Vector3> *)p_self;
	Vector3 &s = *(Vector3 *)p_data;
	return (godot_error)self->insert(p_idx, s);
}

void GDAPI godot_pool_vector3_array_invert(godot_pool_vector3_array *p_self) {
	PoolVector<Vector3> *self = (PoolVector<Vector3> *)p_self;
	self->invert();
}

void GDAPI godot_pool_vector3_array_push_back(godot_pool_vector3_array *p_self, const godot_vector3 *p_data) {
	PoolVector<Vector3> *self = (PoolVector<Vector3> *)p_self;
	Vector3 &s = *(Vector3 *)p_data;
	self->push_back(s);
}

void GDAPI godot_pool_vector3_array_remove(godot_pool_vector3_array *p_self, const godot_int p_idx) {
	PoolVector<Vector3> *self = (PoolVector<Vector3> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_pool_vector3_array_resize(godot_pool_vector3_array *p_self, const godot_int p_size) {
	PoolVector<Vector3> *self = (PoolVector<Vector3> *)p_self;
	self->resize(p_size);
}

godot_pool_vector3_array_read_access GDAPI *godot_pool_vector3_array_read(const godot_pool_vector3_array *p_self) {
	const PoolVector<Vector3> *self = (const PoolVector<Vector3> *)p_self;
	return (godot_pool_vector3_array_read_access *)memnew(PoolVector<Vector3>::Read(self->read()));
}

godot_pool_vector3_array_write_access GDAPI *godot_pool_vector3_array_write(godot_pool_vector3_array *p_self) {
	PoolVector<Vector3> *self = (PoolVector<Vector3> *)p_self;
	return (godot_pool_vector3_array_write_access *)memnew(PoolVector<Vector3>::Write(self->write()));
}

void GDAPI godot_pool_vector3_array_set(godot_pool_vector3_array *p_self, const godot_int p_idx, const godot_vector3 *p_data) {
	PoolVector<Vector3> *self = (PoolVector<Vector3> *)p_self;
	Vector3 &s = *(Vector3 *)p_data;
	self->set(p_idx, s);
}

godot_vector3 GDAPI godot_pool_vector3_array_get(const godot_pool_vector3_array *p_self, const godot_int p_idx) {
	const PoolVector<Vector3> *self = (const PoolVector<Vector3> *)p_self;
	godot_vector3 v;
	Vector3 *s = (Vector3 *)&v;
	*s = self->get(p_idx);
	return v;
}

godot_int GDAPI godot_pool_vector3_array_size(const godot_pool_vector3_array *p_self) {
	const PoolVector<Vector3> *self = (const PoolVector<Vector3> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_pool_vector3_array_empty(const godot_pool_vector3_array *p_self) {
	const PoolVector<Vector3> *self = (const PoolVector<Vector3> *)p_self;
	return self->empty();
}

void GDAPI godot_pool_vector3_array_destroy(godot_pool_vector3_array *p_self) {
	((PoolVector<Vector3> *)p_self)->~PoolVector();
}

// color

void GDAPI godot_pool_color_array_new(godot_pool_color_array *r_dest) {
	PoolVector<Color> *dest = (PoolVector<Color> *)r_dest;
	memnew_placement(dest, PoolVector<Color>);
}

void GDAPI godot_pool_color_array_new_copy(godot_pool_color_array *r_dest, const godot_pool_color_array *p_src) {
	PoolVector<Color> *dest = (PoolVector<Color> *)r_dest;
	const PoolVector<Color> *src = (const PoolVector<Color> *)p_src;
	memnew_placement(dest, PoolVector<Color>(*src));
}

void GDAPI godot_pool_color_array_new_with_array(godot_pool_color_array *r_dest, const godot_array *p_a) {
	PoolVector<Color> *dest = (PoolVector<Color> *)r_dest;
	Array *a = (Array *)p_a;
	memnew_placement(dest, PoolVector<Color>);

	dest->resize(a->size());
	for (int i = 0; i < a->size(); i++) {
		dest->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_color_array_append(godot_pool_color_array *p_self, const godot_color *p_data) {
	PoolVector<Color> *self = (PoolVector<Color> *)p_self;
	Color &s = *(Color *)p_data;
	self->append(s);
}

void GDAPI godot_pool_color_array_append_array(godot_pool_color_array *p_self, const godot_pool_color_array *p_array) {
	PoolVector<Color> *self = (PoolVector<Color> *)p_self;
	PoolVector<Color> *array = (PoolVector<Color> *)p_array;
	self->append_array(*array);
}

godot_error GDAPI godot_pool_color_array_insert(godot_pool_color_array *p_self, const godot_int p_idx, const godot_color *p_data) {
	PoolVector<Color> *self = (PoolVector<Color> *)p_self;
	Color &s = *(Color *)p_data;
	return (godot_error)self->insert(p_idx, s);
}

void GDAPI godot_pool_color_array_invert(godot_pool_color_array *p_self) {
	PoolVector<Color> *self = (PoolVector<Color> *)p_self;
	self->invert();
}

void GDAPI godot_pool_color_array_push_back(godot_pool_color_array *p_self, const godot_color *p_data) {
	PoolVector<Color> *self = (PoolVector<Color> *)p_self;
	Color &s = *(Color *)p_data;
	self->push_back(s);
}

void GDAPI godot_pool_color_array_remove(godot_pool_color_array *p_self, const godot_int p_idx) {
	PoolVector<Color> *self = (PoolVector<Color> *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_pool_color_array_resize(godot_pool_color_array *p_self, const godot_int p_size) {
	PoolVector<Color> *self = (PoolVector<Color> *)p_self;
	self->resize(p_size);
}

godot_pool_color_array_read_access GDAPI *godot_pool_color_array_read(const godot_pool_color_array *p_self) {
	const PoolVector<Color> *self = (const PoolVector<Color> *)p_self;
	return (godot_pool_color_array_read_access *)memnew(PoolVector<Color>::Read(self->read()));
}

godot_pool_color_array_write_access GDAPI *godot_pool_color_array_write(godot_pool_color_array *p_self) {
	PoolVector<Color> *self = (PoolVector<Color> *)p_self;
	return (godot_pool_color_array_write_access *)memnew(PoolVector<Color>::Write(self->write()));
}

void GDAPI godot_pool_color_array_set(godot_pool_color_array *p_self, const godot_int p_idx, const godot_color *p_data) {
	PoolVector<Color> *self = (PoolVector<Color> *)p_self;
	Color &s = *(Color *)p_data;
	self->set(p_idx, s);
}

godot_color GDAPI godot_pool_color_array_get(const godot_pool_color_array *p_self, const godot_int p_idx) {
	const PoolVector<Color> *self = (const PoolVector<Color> *)p_self;
	godot_color v;
	Color *s = (Color *)&v;
	*s = self->get(p_idx);
	return v;
}

godot_int GDAPI godot_pool_color_array_size(const godot_pool_color_array *p_self) {
	const PoolVector<Color> *self = (const PoolVector<Color> *)p_self;
	return self->size();
}

godot_bool GDAPI godot_pool_color_array_empty(const godot_pool_color_array *p_self) {
	const PoolVector<Color> *self = (const PoolVector<Color> *)p_self;
	return self->empty();
}

void GDAPI godot_pool_color_array_destroy(godot_pool_color_array *p_self) {
	((PoolVector<Color> *)p_self)->~PoolVector();
}

//
// read accessor functions
//

godot_pool_byte_array_read_access GDAPI *godot_pool_byte_array_read_access_copy(const godot_pool_byte_array_read_access *p_other) {
	PoolVector<uint8_t>::Read *other = (PoolVector<uint8_t>::Read *)p_other;
	return (godot_pool_byte_array_read_access *)memnew(PoolVector<uint8_t>::Read(*other));
}
const uint8_t GDAPI *godot_pool_byte_array_read_access_ptr(const godot_pool_byte_array_read_access *p_read) {
	const PoolVector<uint8_t>::Read *read = (const PoolVector<uint8_t>::Read *)p_read;
	return read->ptr();
}
void GDAPI godot_pool_byte_array_read_access_operator_assign(godot_pool_byte_array_read_access *p_read, godot_pool_byte_array_read_access *p_other) {
	PoolVector<uint8_t>::Read *read = (PoolVector<uint8_t>::Read *)p_read;
	PoolVector<uint8_t>::Read *other = (PoolVector<uint8_t>::Read *)p_other;
	read->operator=(*other);
}
void GDAPI godot_pool_byte_array_read_access_destroy(godot_pool_byte_array_read_access *p_read) {
	memdelete((PoolVector<uint8_t>::Read *)p_read);
}

godot_pool_int_array_read_access GDAPI *godot_pool_int_array_read_access_copy(const godot_pool_int_array_read_access *p_other) {
	PoolVector<godot_int>::Read *other = (PoolVector<godot_int>::Read *)p_other;
	return (godot_pool_int_array_read_access *)memnew(PoolVector<godot_int>::Read(*other));
}
const godot_int GDAPI *godot_pool_int_array_read_access_ptr(const godot_pool_int_array_read_access *p_read) {
	const PoolVector<godot_int>::Read *read = (const PoolVector<godot_int>::Read *)p_read;
	return read->ptr();
}
void GDAPI godot_pool_int_array_read_access_operator_assign(godot_pool_int_array_read_access *p_read, godot_pool_int_array_read_access *p_other) {
	PoolVector<godot_int>::Read *read = (PoolVector<godot_int>::Read *)p_read;
	PoolVector<godot_int>::Read *other = (PoolVector<godot_int>::Read *)p_other;
	read->operator=(*other);
}
void GDAPI godot_pool_int_array_read_access_destroy(godot_pool_int_array_read_access *p_read) {
	memdelete((PoolVector<godot_int>::Read *)p_read);
}

godot_pool_real_array_read_access GDAPI *godot_pool_real_array_read_access_copy(const godot_pool_real_array_read_access *p_other) {
	PoolVector<godot_real>::Read *other = (PoolVector<godot_real>::Read *)p_other;
	return (godot_pool_real_array_read_access *)memnew(PoolVector<godot_real>::Read(*other));
}
const godot_real GDAPI *godot_pool_real_array_read_access_ptr(const godot_pool_real_array_read_access *p_read) {
	const PoolVector<godot_real>::Read *read = (const PoolVector<godot_real>::Read *)p_read;
	return read->ptr();
}
void GDAPI godot_pool_real_array_read_access_operator_assign(godot_pool_real_array_read_access *p_read, godot_pool_real_array_read_access *p_other) {
	PoolVector<godot_real>::Read *read = (PoolVector<godot_real>::Read *)p_read;
	PoolVector<godot_real>::Read *other = (PoolVector<godot_real>::Read *)p_other;
	read->operator=(*other);
}
void GDAPI godot_pool_real_array_read_access_destroy(godot_pool_real_array_read_access *p_read) {
	memdelete((PoolVector<godot_real>::Read *)p_read);
}

godot_pool_string_array_read_access GDAPI *godot_pool_string_array_read_access_copy(const godot_pool_string_array_read_access *p_other) {
	PoolVector<String>::Read *other = (PoolVector<String>::Read *)p_other;
	return (godot_pool_string_array_read_access *)memnew(PoolVector<String>::Read(*other));
}
const godot_string GDAPI *godot_pool_string_array_read_access_ptr(const godot_pool_string_array_read_access *p_read) {
	const PoolVector<String>::Read *read = (const PoolVector<String>::Read *)p_read;
	return (const godot_string *)read->ptr();
}
void GDAPI godot_pool_string_array_read_access_operator_assign(godot_pool_string_array_read_access *p_read, godot_pool_string_array_read_access *p_other) {
	PoolVector<String>::Read *read = (PoolVector<String>::Read *)p_read;
	PoolVector<String>::Read *other = (PoolVector<String>::Read *)p_other;
	read->operator=(*other);
}
void GDAPI godot_pool_string_array_read_access_destroy(godot_pool_string_array_read_access *p_read) {
	memdelete((PoolVector<String>::Read *)p_read);
}

godot_pool_vector2_array_read_access GDAPI *godot_pool_vector2_array_read_access_copy(const godot_pool_vector2_array_read_access *p_other) {
	PoolVector<Vector2>::Read *other = (PoolVector<Vector2>::Read *)p_other;
	return (godot_pool_vector2_array_read_access *)memnew(PoolVector<Vector2>::Read(*other));
}
const godot_vector2 GDAPI *godot_pool_vector2_array_read_access_ptr(const godot_pool_vector2_array_read_access *p_read) {
	const PoolVector<Vector2>::Read *read = (const PoolVector<Vector2>::Read *)p_read;
	return (const godot_vector2 *)read->ptr();
}
void GDAPI godot_pool_vector2_array_read_access_operator_assign(godot_pool_vector2_array_read_access *p_read, godot_pool_vector2_array_read_access *p_other) {
	PoolVector<Vector2>::Read *read = (PoolVector<Vector2>::Read *)p_read;
	PoolVector<Vector2>::Read *other = (PoolVector<Vector2>::Read *)p_other;
	read->operator=(*other);
}
void GDAPI godot_pool_vector2_array_read_access_destroy(godot_pool_vector2_array_read_access *p_read) {
	memdelete((PoolVector<Vector2>::Read *)p_read);
}

godot_pool_vector3_array_read_access GDAPI *godot_pool_vector3_array_read_access_copy(const godot_pool_vector3_array_read_access *p_other) {
	PoolVector<Vector3>::Read *other = (PoolVector<Vector3>::Read *)p_other;
	return (godot_pool_vector3_array_read_access *)memnew(PoolVector<Vector3>::Read(*other));
}
const godot_vector3 GDAPI *godot_pool_vector3_array_read_access_ptr(const godot_pool_vector3_array_read_access *p_read) {
	const PoolVector<Vector3>::Read *read = (const PoolVector<Vector3>::Read *)p_read;
	return (const godot_vector3 *)read->ptr();
}
void GDAPI godot_pool_vector3_array_read_access_operator_assign(godot_pool_vector3_array_read_access *p_read, godot_pool_vector3_array_read_access *p_other) {
	PoolVector<Vector3>::Read *read = (PoolVector<Vector3>::Read *)p_read;
	PoolVector<Vector3>::Read *other = (PoolVector<Vector3>::Read *)p_other;
	read->operator=(*other);
}
void GDAPI godot_pool_vector3_array_read_access_destroy(godot_pool_vector3_array_read_access *p_read) {
	memdelete((PoolVector<Vector2>::Read *)p_read);
}

godot_pool_color_array_read_access GDAPI *godot_pool_color_array_read_access_copy(const godot_pool_color_array_read_access *p_other) {
	PoolVector<Color>::Read *other = (PoolVector<Color>::Read *)p_other;
	return (godot_pool_color_array_read_access *)memnew(PoolVector<Color>::Read(*other));
}
const godot_color GDAPI *godot_pool_color_array_read_access_ptr(const godot_pool_color_array_read_access *p_read) {
	const PoolVector<Color>::Read *read = (const PoolVector<Color>::Read *)p_read;
	return (const godot_color *)read->ptr();
}
void GDAPI godot_pool_color_array_read_access_operator_assign(godot_pool_color_array_read_access *p_read, godot_pool_color_array_read_access *p_other) {
	PoolVector<Color>::Read *read = (PoolVector<Color>::Read *)p_read;
	PoolVector<Color>::Read *other = (PoolVector<Color>::Read *)p_other;
	read->operator=(*other);
}
void GDAPI godot_pool_color_array_read_access_destroy(godot_pool_color_array_read_access *p_read) {
	memdelete((PoolVector<Color>::Read *)p_read);
}

//
// write accessor functions
//

godot_pool_byte_array_write_access GDAPI *godot_pool_byte_array_write_access_copy(const godot_pool_byte_array_write_access *p_other) {
	PoolVector<uint8_t>::Write *other = (PoolVector<uint8_t>::Write *)p_other;
	return (godot_pool_byte_array_write_access *)memnew(PoolVector<uint8_t>::Write(*other));
}
uint8_t GDAPI *godot_pool_byte_array_write_access_ptr(const godot_pool_byte_array_write_access *p_write) {
	PoolVector<uint8_t>::Write *write = (PoolVector<uint8_t>::Write *)p_write;
	return write->ptr();
}
void GDAPI godot_pool_byte_array_write_access_operator_assign(godot_pool_byte_array_write_access *p_write, godot_pool_byte_array_write_access *p_other) {
	PoolVector<uint8_t>::Write *write = (PoolVector<uint8_t>::Write *)p_write;
	PoolVector<uint8_t>::Write *other = (PoolVector<uint8_t>::Write *)p_other;
	write->operator=(*other);
}
void GDAPI godot_pool_byte_array_write_access_destroy(godot_pool_byte_array_write_access *p_write) {
	memdelete((PoolVector<uint8_t>::Write *)p_write);
}

godot_pool_int_array_write_access GDAPI *godot_pool_int_array_write_access_copy(const godot_pool_int_array_write_access *p_other) {
	PoolVector<godot_int>::Write *other = (PoolVector<godot_int>::Write *)p_other;
	return (godot_pool_int_array_write_access *)memnew(PoolVector<godot_int>::Write(*other));
}
godot_int GDAPI *godot_pool_int_array_write_access_ptr(const godot_pool_int_array_write_access *p_write) {
	PoolVector<godot_int>::Write *write = (PoolVector<godot_int>::Write *)p_write;
	return write->ptr();
}
void GDAPI godot_pool_int_array_write_access_operator_assign(godot_pool_int_array_write_access *p_write, godot_pool_int_array_write_access *p_other) {
	PoolVector<godot_int>::Write *write = (PoolVector<godot_int>::Write *)p_write;
	PoolVector<godot_int>::Write *other = (PoolVector<godot_int>::Write *)p_other;
	write->operator=(*other);
}
void GDAPI godot_pool_int_array_write_access_destroy(godot_pool_int_array_write_access *p_write) {
	memdelete((PoolVector<godot_int>::Write *)p_write);
}

godot_pool_real_array_write_access GDAPI *godot_pool_real_array_write_access_copy(const godot_pool_real_array_write_access *p_other) {
	PoolVector<godot_real>::Write *other = (PoolVector<godot_real>::Write *)p_other;
	return (godot_pool_real_array_write_access *)memnew(PoolVector<godot_real>::Write(*other));
}
godot_real GDAPI *godot_pool_real_array_write_access_ptr(const godot_pool_real_array_write_access *p_write) {
	PoolVector<godot_real>::Write *write = (PoolVector<godot_real>::Write *)p_write;
	return write->ptr();
}
void GDAPI godot_pool_real_array_write_access_operator_assign(godot_pool_real_array_write_access *p_write, godot_pool_real_array_write_access *p_other) {
	PoolVector<godot_real>::Write *write = (PoolVector<godot_real>::Write *)p_write;
	PoolVector<godot_real>::Write *other = (PoolVector<godot_real>::Write *)p_other;
	write->operator=(*other);
}
void GDAPI godot_pool_real_array_write_access_destroy(godot_pool_real_array_write_access *p_write) {
	memdelete((PoolVector<godot_real>::Write *)p_write);
}

godot_pool_string_array_write_access GDAPI *godot_pool_string_array_write_access_copy(const godot_pool_string_array_write_access *p_other) {
	PoolVector<String>::Write *other = (PoolVector<String>::Write *)p_other;
	return (godot_pool_string_array_write_access *)memnew(PoolVector<String>::Write(*other));
}
godot_string GDAPI *godot_pool_string_array_write_access_ptr(const godot_pool_string_array_write_access *p_write) {
	PoolVector<String>::Write *write = (PoolVector<String>::Write *)p_write;
	return (godot_string *)write->ptr();
}
void GDAPI godot_pool_string_array_write_access_operator_assign(godot_pool_string_array_write_access *p_write, godot_pool_string_array_write_access *p_other) {
	PoolVector<String>::Write *write = (PoolVector<String>::Write *)p_write;
	PoolVector<String>::Write *other = (PoolVector<String>::Write *)p_other;
	write->operator=(*other);
}
void GDAPI godot_pool_string_array_write_access_destroy(godot_pool_string_array_write_access *p_write) {
	memdelete((PoolVector<String>::Write *)p_write);
}

godot_pool_vector2_array_write_access GDAPI *godot_pool_vector2_array_write_access_copy(const godot_pool_vector2_array_write_access *p_other) {
	PoolVector<Vector2>::Write *other = (PoolVector<Vector2>::Write *)p_other;
	return (godot_pool_vector2_array_write_access *)memnew(PoolVector<Vector2>::Write(*other));
}
godot_vector2 GDAPI *godot_pool_vector2_array_write_access_ptr(const godot_pool_vector2_array_write_access *p_write) {
	PoolVector<Vector2>::Write *write = (PoolVector<Vector2>::Write *)p_write;
	return (godot_vector2 *)write->ptr();
}
void GDAPI godot_pool_vector2_array_write_access_operator_assign(godot_pool_vector2_array_write_access *p_write, godot_pool_vector2_array_write_access *p_other) {
	PoolVector<Vector2>::Write *write = (PoolVector<Vector2>::Write *)p_write;
	PoolVector<Vector2>::Write *other = (PoolVector<Vector2>::Write *)p_other;
	write->operator=(*other);
}
void GDAPI godot_pool_vector2_array_write_access_destroy(godot_pool_vector2_array_write_access *p_write) {
	memdelete((PoolVector<Vector2>::Write *)p_write);
}

godot_pool_vector3_array_write_access GDAPI *godot_pool_vector3_array_write_access_copy(const godot_pool_vector3_array_write_access *p_other) {
	PoolVector<Vector3>::Write *other = (PoolVector<Vector3>::Write *)p_other;
	return (godot_pool_vector3_array_write_access *)memnew(PoolVector<Vector3>::Write(*other));
}
godot_vector3 GDAPI *godot_pool_vector3_array_write_access_ptr(const godot_pool_vector3_array_write_access *p_write) {
	PoolVector<Vector3>::Write *write = (PoolVector<Vector3>::Write *)p_write;
	return (godot_vector3 *)write->ptr();
}
void GDAPI godot_pool_vector3_array_write_access_operator_assign(godot_pool_vector3_array_write_access *p_write, godot_pool_vector3_array_write_access *p_other) {
	PoolVector<Vector3>::Write *write = (PoolVector<Vector3>::Write *)p_write;
	PoolVector<Vector3>::Write *other = (PoolVector<Vector3>::Write *)p_other;
	write->operator=(*other);
}
void GDAPI godot_pool_vector3_array_write_access_destroy(godot_pool_vector3_array_write_access *p_write) {
	memdelete((PoolVector<Vector3>::Write *)p_write);
}

godot_pool_color_array_write_access GDAPI *godot_pool_color_array_write_access_copy(const godot_pool_color_array_write_access *p_other) {
	PoolVector<Color>::Write *other = (PoolVector<Color>::Write *)p_other;
	return (godot_pool_color_array_write_access *)memnew(PoolVector<Color>::Write(*other));
}
godot_color GDAPI *godot_pool_color_array_write_access_ptr(const godot_pool_color_array_write_access *p_write) {
	PoolVector<Color>::Write *write = (PoolVector<Color>::Write *)p_write;
	return (godot_color *)write->ptr();
}
void GDAPI godot_pool_color_array_write_access_operator_assign(godot_pool_color_array_write_access *p_write, godot_pool_color_array_write_access *p_other) {
	PoolVector<Color>::Write *write = (PoolVector<Color>::Write *)p_write;
	PoolVector<Color>::Write *other = (PoolVector<Color>::Write *)p_other;
	write->operator=(*other);
}
void GDAPI godot_pool_color_array_write_access_destroy(godot_pool_color_array_write_access *p_write) {
	memdelete((PoolVector<Color>::Write *)p_write);
}

#ifdef __cplusplus
}
#endif
