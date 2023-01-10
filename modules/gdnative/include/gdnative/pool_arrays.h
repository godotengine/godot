/**************************************************************************/
/*  pool_arrays.h                                                         */
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

#ifndef GDNATIVE_POOL_ARRAYS_H
#define GDNATIVE_POOL_ARRAYS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/////// Read Access

#define GODOT_POOL_ARRAY_READ_ACCESS_SIZE 1

typedef struct {
	uint8_t _dont_touch_that[GODOT_POOL_ARRAY_READ_ACCESS_SIZE];
} godot_pool_array_read_access;

typedef godot_pool_array_read_access godot_pool_byte_array_read_access;
typedef godot_pool_array_read_access godot_pool_int_array_read_access;
typedef godot_pool_array_read_access godot_pool_real_array_read_access;
typedef godot_pool_array_read_access godot_pool_string_array_read_access;
typedef godot_pool_array_read_access godot_pool_vector2_array_read_access;
typedef godot_pool_array_read_access godot_pool_vector3_array_read_access;
typedef godot_pool_array_read_access godot_pool_color_array_read_access;

/////// Write Access

#define GODOT_POOL_ARRAY_WRITE_ACCESS_SIZE 1

typedef struct {
	uint8_t _dont_touch_that[GODOT_POOL_ARRAY_WRITE_ACCESS_SIZE];
} godot_pool_array_write_access;

typedef godot_pool_array_write_access godot_pool_byte_array_write_access;
typedef godot_pool_array_write_access godot_pool_int_array_write_access;
typedef godot_pool_array_write_access godot_pool_real_array_write_access;
typedef godot_pool_array_write_access godot_pool_string_array_write_access;
typedef godot_pool_array_write_access godot_pool_vector2_array_write_access;
typedef godot_pool_array_write_access godot_pool_vector3_array_write_access;
typedef godot_pool_array_write_access godot_pool_color_array_write_access;

/////// PoolByteArray

#define GODOT_POOL_BYTE_ARRAY_SIZE sizeof(void *)

#ifndef GODOT_CORE_API_GODOT_POOL_BYTE_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_POOL_BYTE_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_POOL_BYTE_ARRAY_SIZE];
} godot_pool_byte_array;
#endif

/////// PoolIntArray

#define GODOT_POOL_INT_ARRAY_SIZE sizeof(void *)

#ifndef GODOT_CORE_API_GODOT_POOL_INT_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_POOL_INT_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_POOL_INT_ARRAY_SIZE];
} godot_pool_int_array;
#endif

/////// PoolRealArray

#define GODOT_POOL_REAL_ARRAY_SIZE sizeof(void *)

#ifndef GODOT_CORE_API_GODOT_POOL_REAL_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_POOL_REAL_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_POOL_REAL_ARRAY_SIZE];
} godot_pool_real_array;
#endif

/////// PoolStringArray

#define GODOT_POOL_STRING_ARRAY_SIZE sizeof(void *)

#ifndef GODOT_CORE_API_GODOT_POOL_STRING_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_POOL_STRING_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_POOL_STRING_ARRAY_SIZE];
} godot_pool_string_array;
#endif

/////// PoolVector2Array

#define GODOT_POOL_VECTOR2_ARRAY_SIZE sizeof(void *)

#ifndef GODOT_CORE_API_GODOT_POOL_VECTOR2_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_POOL_VECTOR2_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_POOL_VECTOR2_ARRAY_SIZE];
} godot_pool_vector2_array;
#endif

/////// PoolVector3Array

#define GODOT_POOL_VECTOR3_ARRAY_SIZE sizeof(void *)

#ifndef GODOT_CORE_API_GODOT_POOL_VECTOR3_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_POOL_VECTOR3_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_POOL_VECTOR3_ARRAY_SIZE];
} godot_pool_vector3_array;
#endif

/////// PoolColorArray

#define GODOT_POOL_COLOR_ARRAY_SIZE sizeof(void *)

#ifndef GODOT_CORE_API_GODOT_POOL_COLOR_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_POOL_COLOR_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_POOL_COLOR_ARRAY_SIZE];
} godot_pool_color_array;
#endif

// reduce extern "C" nesting for VS2013
#ifdef __cplusplus
}
#endif

#include <gdnative/array.h>
#include <gdnative/color.h>
#include <gdnative/vector2.h>
#include <gdnative/vector3.h>

#include <gdnative/gdnative.h>

#ifdef __cplusplus
extern "C" {
#endif

// byte

void GDAPI godot_pool_byte_array_new(godot_pool_byte_array *r_dest);
void GDAPI godot_pool_byte_array_new_copy(godot_pool_byte_array *r_dest, const godot_pool_byte_array *p_src);
void GDAPI godot_pool_byte_array_new_with_array(godot_pool_byte_array *r_dest, const godot_array *p_a);

void GDAPI godot_pool_byte_array_append(godot_pool_byte_array *p_self, const uint8_t p_data);

void GDAPI godot_pool_byte_array_append_array(godot_pool_byte_array *p_self, const godot_pool_byte_array *p_array);

godot_error GDAPI godot_pool_byte_array_insert(godot_pool_byte_array *p_self, const godot_int p_idx, const uint8_t p_data);

void GDAPI godot_pool_byte_array_invert(godot_pool_byte_array *p_self);

void GDAPI godot_pool_byte_array_push_back(godot_pool_byte_array *p_self, const uint8_t p_data);

void GDAPI godot_pool_byte_array_remove(godot_pool_byte_array *p_self, const godot_int p_idx);

void GDAPI godot_pool_byte_array_resize(godot_pool_byte_array *p_self, const godot_int p_size);

void GDAPI godot_pool_byte_array_sort(godot_pool_byte_array *p_self);

godot_pool_byte_array_read_access GDAPI *godot_pool_byte_array_read(const godot_pool_byte_array *p_self);

godot_pool_byte_array_write_access GDAPI *godot_pool_byte_array_write(godot_pool_byte_array *p_self);

void GDAPI godot_pool_byte_array_set(godot_pool_byte_array *p_self, const godot_int p_idx, const uint8_t p_data);
uint8_t GDAPI godot_pool_byte_array_get(const godot_pool_byte_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_pool_byte_array_size(const godot_pool_byte_array *p_self);

godot_bool GDAPI godot_pool_byte_array_empty(const godot_pool_byte_array *p_self);

godot_bool GDAPI godot_pool_byte_array_has(const godot_pool_byte_array *p_self, const uint8_t p_data);

void GDAPI godot_pool_byte_array_destroy(godot_pool_byte_array *p_self);

// int

void GDAPI godot_pool_int_array_new(godot_pool_int_array *r_dest);
void GDAPI godot_pool_int_array_new_copy(godot_pool_int_array *r_dest, const godot_pool_int_array *p_src);
void GDAPI godot_pool_int_array_new_with_array(godot_pool_int_array *r_dest, const godot_array *p_a);

void GDAPI godot_pool_int_array_append(godot_pool_int_array *p_self, const godot_int p_data);

void GDAPI godot_pool_int_array_append_array(godot_pool_int_array *p_self, const godot_pool_int_array *p_array);

godot_error GDAPI godot_pool_int_array_insert(godot_pool_int_array *p_self, const godot_int p_idx, const godot_int p_data);

void GDAPI godot_pool_int_array_invert(godot_pool_int_array *p_self);

void GDAPI godot_pool_int_array_push_back(godot_pool_int_array *p_self, const godot_int p_data);

void GDAPI godot_pool_int_array_remove(godot_pool_int_array *p_self, const godot_int p_idx);

void GDAPI godot_pool_int_array_resize(godot_pool_int_array *p_self, const godot_int p_size);

void GDAPI godot_pool_int_array_sort(godot_pool_int_array *p_self);

godot_pool_int_array_read_access GDAPI *godot_pool_int_array_read(const godot_pool_int_array *p_self);

godot_pool_int_array_write_access GDAPI *godot_pool_int_array_write(godot_pool_int_array *p_self);

void GDAPI godot_pool_int_array_set(godot_pool_int_array *p_self, const godot_int p_idx, const godot_int p_data);
godot_int GDAPI godot_pool_int_array_get(const godot_pool_int_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_pool_int_array_size(const godot_pool_int_array *p_self);

godot_bool GDAPI godot_pool_int_array_empty(const godot_pool_int_array *p_self);

godot_bool GDAPI godot_pool_int_array_has(const godot_pool_int_array *p_self, const godot_int p_data);

void GDAPI godot_pool_int_array_destroy(godot_pool_int_array *p_self);

// real

void GDAPI godot_pool_real_array_new(godot_pool_real_array *r_dest);
void GDAPI godot_pool_real_array_new_copy(godot_pool_real_array *r_dest, const godot_pool_real_array *p_src);
void GDAPI godot_pool_real_array_new_with_array(godot_pool_real_array *r_dest, const godot_array *p_a);

void GDAPI godot_pool_real_array_append(godot_pool_real_array *p_self, const godot_real p_data);

void GDAPI godot_pool_real_array_append_array(godot_pool_real_array *p_self, const godot_pool_real_array *p_array);

godot_error GDAPI godot_pool_real_array_insert(godot_pool_real_array *p_self, const godot_int p_idx, const godot_real p_data);

void GDAPI godot_pool_real_array_invert(godot_pool_real_array *p_self);

void GDAPI godot_pool_real_array_push_back(godot_pool_real_array *p_self, const godot_real p_data);

void GDAPI godot_pool_real_array_remove(godot_pool_real_array *p_self, const godot_int p_idx);

void GDAPI godot_pool_real_array_resize(godot_pool_real_array *p_self, const godot_int p_size);

void GDAPI godot_pool_real_array_sort(godot_pool_real_array *p_self);

godot_pool_real_array_read_access GDAPI *godot_pool_real_array_read(const godot_pool_real_array *p_self);

godot_pool_real_array_write_access GDAPI *godot_pool_real_array_write(godot_pool_real_array *p_self);

void GDAPI godot_pool_real_array_set(godot_pool_real_array *p_self, const godot_int p_idx, const godot_real p_data);
godot_real GDAPI godot_pool_real_array_get(const godot_pool_real_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_pool_real_array_size(const godot_pool_real_array *p_self);

godot_bool GDAPI godot_pool_real_array_empty(const godot_pool_real_array *p_self);

godot_bool GDAPI godot_pool_real_array_has(const godot_pool_real_array *p_self, const godot_real p_data);

void GDAPI godot_pool_real_array_destroy(godot_pool_real_array *p_self);

// string

void GDAPI godot_pool_string_array_new(godot_pool_string_array *r_dest);
void GDAPI godot_pool_string_array_new_copy(godot_pool_string_array *r_dest, const godot_pool_string_array *p_src);
void GDAPI godot_pool_string_array_new_with_array(godot_pool_string_array *r_dest, const godot_array *p_a);

void GDAPI godot_pool_string_array_append(godot_pool_string_array *p_self, const godot_string *p_data);

void GDAPI godot_pool_string_array_append_array(godot_pool_string_array *p_self, const godot_pool_string_array *p_array);

godot_error GDAPI godot_pool_string_array_insert(godot_pool_string_array *p_self, const godot_int p_idx, const godot_string *p_data);

void GDAPI godot_pool_string_array_invert(godot_pool_string_array *p_self);

godot_string GDAPI godot_pool_string_array_join(const godot_pool_string_array *p_self, const godot_string *p_delimiter);

void GDAPI godot_pool_string_array_push_back(godot_pool_string_array *p_self, const godot_string *p_data);

void GDAPI godot_pool_string_array_remove(godot_pool_string_array *p_self, const godot_int p_idx);

void GDAPI godot_pool_string_array_resize(godot_pool_string_array *p_self, const godot_int p_size);

void GDAPI godot_pool_string_array_sort(godot_pool_string_array *p_self);

godot_pool_string_array_read_access GDAPI *godot_pool_string_array_read(const godot_pool_string_array *p_self);

godot_pool_string_array_write_access GDAPI *godot_pool_string_array_write(godot_pool_string_array *p_self);

void GDAPI godot_pool_string_array_set(godot_pool_string_array *p_self, const godot_int p_idx, const godot_string *p_data);
godot_string GDAPI godot_pool_string_array_get(const godot_pool_string_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_pool_string_array_size(const godot_pool_string_array *p_self);

godot_bool GDAPI godot_pool_string_array_empty(const godot_pool_string_array *p_self);

godot_bool GDAPI godot_pool_string_array_has(const godot_pool_string_array *p_self, const godot_string *p_data);

void GDAPI godot_pool_string_array_destroy(godot_pool_string_array *p_self);

// vector2

void GDAPI godot_pool_vector2_array_new(godot_pool_vector2_array *r_dest);
void GDAPI godot_pool_vector2_array_new_copy(godot_pool_vector2_array *r_dest, const godot_pool_vector2_array *p_src);
void GDAPI godot_pool_vector2_array_new_with_array(godot_pool_vector2_array *r_dest, const godot_array *p_a);

void GDAPI godot_pool_vector2_array_append(godot_pool_vector2_array *p_self, const godot_vector2 *p_data);

void GDAPI godot_pool_vector2_array_append_array(godot_pool_vector2_array *p_self, const godot_pool_vector2_array *p_array);

godot_error GDAPI godot_pool_vector2_array_insert(godot_pool_vector2_array *p_self, const godot_int p_idx, const godot_vector2 *p_data);

void GDAPI godot_pool_vector2_array_invert(godot_pool_vector2_array *p_self);

void GDAPI godot_pool_vector2_array_push_back(godot_pool_vector2_array *p_self, const godot_vector2 *p_data);

void GDAPI godot_pool_vector2_array_remove(godot_pool_vector2_array *p_self, const godot_int p_idx);

void GDAPI godot_pool_vector2_array_resize(godot_pool_vector2_array *p_self, const godot_int p_size);

void GDAPI godot_pool_vector2_array_sort(godot_pool_vector2_array *p_self);

godot_pool_vector2_array_read_access GDAPI *godot_pool_vector2_array_read(const godot_pool_vector2_array *p_self);

godot_pool_vector2_array_write_access GDAPI *godot_pool_vector2_array_write(godot_pool_vector2_array *p_self);

void GDAPI godot_pool_vector2_array_set(godot_pool_vector2_array *p_self, const godot_int p_idx, const godot_vector2 *p_data);
godot_vector2 GDAPI godot_pool_vector2_array_get(const godot_pool_vector2_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_pool_vector2_array_size(const godot_pool_vector2_array *p_self);

godot_bool GDAPI godot_pool_vector2_array_empty(const godot_pool_vector2_array *p_self);

godot_bool GDAPI godot_pool_vector2_array_has(const godot_pool_vector2_array *p_self, const godot_vector2 *p_data);

void GDAPI godot_pool_vector2_array_destroy(godot_pool_vector2_array *p_self);

// vector3

void GDAPI godot_pool_vector3_array_new(godot_pool_vector3_array *r_dest);
void GDAPI godot_pool_vector3_array_new_copy(godot_pool_vector3_array *r_dest, const godot_pool_vector3_array *p_src);
void GDAPI godot_pool_vector3_array_new_with_array(godot_pool_vector3_array *r_dest, const godot_array *p_a);

void GDAPI godot_pool_vector3_array_append(godot_pool_vector3_array *p_self, const godot_vector3 *p_data);

void GDAPI godot_pool_vector3_array_append_array(godot_pool_vector3_array *p_self, const godot_pool_vector3_array *p_array);

godot_error GDAPI godot_pool_vector3_array_insert(godot_pool_vector3_array *p_self, const godot_int p_idx, const godot_vector3 *p_data);

void GDAPI godot_pool_vector3_array_invert(godot_pool_vector3_array *p_self);

void GDAPI godot_pool_vector3_array_push_back(godot_pool_vector3_array *p_self, const godot_vector3 *p_data);

void GDAPI godot_pool_vector3_array_remove(godot_pool_vector3_array *p_self, const godot_int p_idx);

void GDAPI godot_pool_vector3_array_resize(godot_pool_vector3_array *p_self, const godot_int p_size);

void GDAPI godot_pool_vector3_array_sort(godot_pool_vector3_array *p_self);

godot_pool_vector3_array_read_access GDAPI *godot_pool_vector3_array_read(const godot_pool_vector3_array *p_self);

godot_pool_vector3_array_write_access GDAPI *godot_pool_vector3_array_write(godot_pool_vector3_array *p_self);

void GDAPI godot_pool_vector3_array_set(godot_pool_vector3_array *p_self, const godot_int p_idx, const godot_vector3 *p_data);
godot_vector3 GDAPI godot_pool_vector3_array_get(const godot_pool_vector3_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_pool_vector3_array_size(const godot_pool_vector3_array *p_self);

godot_bool GDAPI godot_pool_vector3_array_empty(const godot_pool_vector3_array *p_self);

godot_bool GDAPI godot_pool_vector3_array_has(const godot_pool_vector3_array *p_self, const godot_vector3 *p_data);

void GDAPI godot_pool_vector3_array_destroy(godot_pool_vector3_array *p_self);

// color

void GDAPI godot_pool_color_array_new(godot_pool_color_array *r_dest);
void GDAPI godot_pool_color_array_new_copy(godot_pool_color_array *r_dest, const godot_pool_color_array *p_src);
void GDAPI godot_pool_color_array_new_with_array(godot_pool_color_array *r_dest, const godot_array *p_a);

void GDAPI godot_pool_color_array_append(godot_pool_color_array *p_self, const godot_color *p_data);

void GDAPI godot_pool_color_array_append_array(godot_pool_color_array *p_self, const godot_pool_color_array *p_array);

godot_error GDAPI godot_pool_color_array_insert(godot_pool_color_array *p_self, const godot_int p_idx, const godot_color *p_data);

void GDAPI godot_pool_color_array_invert(godot_pool_color_array *p_self);

void GDAPI godot_pool_color_array_push_back(godot_pool_color_array *p_self, const godot_color *p_data);

void GDAPI godot_pool_color_array_remove(godot_pool_color_array *p_self, const godot_int p_idx);

void GDAPI godot_pool_color_array_resize(godot_pool_color_array *p_self, const godot_int p_size);

void GDAPI godot_pool_color_array_sort(godot_pool_color_array *p_self);

godot_pool_color_array_read_access GDAPI *godot_pool_color_array_read(const godot_pool_color_array *p_self);

godot_pool_color_array_write_access GDAPI *godot_pool_color_array_write(godot_pool_color_array *p_self);

void GDAPI godot_pool_color_array_set(godot_pool_color_array *p_self, const godot_int p_idx, const godot_color *p_data);
godot_color GDAPI godot_pool_color_array_get(const godot_pool_color_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_pool_color_array_size(const godot_pool_color_array *p_self);

godot_bool GDAPI godot_pool_color_array_empty(const godot_pool_color_array *p_self);

godot_bool GDAPI godot_pool_color_array_has(const godot_pool_color_array *p_self, const godot_color *p_data);

void GDAPI godot_pool_color_array_destroy(godot_pool_color_array *p_self);

//
// read accessor functions
//

godot_pool_byte_array_read_access GDAPI *godot_pool_byte_array_read_access_copy(const godot_pool_byte_array_read_access *p_other);
const uint8_t GDAPI *godot_pool_byte_array_read_access_ptr(const godot_pool_byte_array_read_access *p_read);
void GDAPI godot_pool_byte_array_read_access_operator_assign(godot_pool_byte_array_read_access *p_read, godot_pool_byte_array_read_access *p_other);
void GDAPI godot_pool_byte_array_read_access_destroy(godot_pool_byte_array_read_access *p_read);

godot_pool_int_array_read_access GDAPI *godot_pool_int_array_read_access_copy(const godot_pool_int_array_read_access *p_other);
const godot_int GDAPI *godot_pool_int_array_read_access_ptr(const godot_pool_int_array_read_access *p_read);
void GDAPI godot_pool_int_array_read_access_operator_assign(godot_pool_int_array_read_access *p_read, godot_pool_int_array_read_access *p_other);
void GDAPI godot_pool_int_array_read_access_destroy(godot_pool_int_array_read_access *p_read);

godot_pool_real_array_read_access GDAPI *godot_pool_real_array_read_access_copy(const godot_pool_real_array_read_access *p_other);
const godot_real GDAPI *godot_pool_real_array_read_access_ptr(const godot_pool_real_array_read_access *p_read);
void GDAPI godot_pool_real_array_read_access_operator_assign(godot_pool_real_array_read_access *p_read, godot_pool_real_array_read_access *p_other);
void GDAPI godot_pool_real_array_read_access_destroy(godot_pool_real_array_read_access *p_read);

godot_pool_string_array_read_access GDAPI *godot_pool_string_array_read_access_copy(const godot_pool_string_array_read_access *p_other);
const godot_string GDAPI *godot_pool_string_array_read_access_ptr(const godot_pool_string_array_read_access *p_read);
void GDAPI godot_pool_string_array_read_access_operator_assign(godot_pool_string_array_read_access *p_read, godot_pool_string_array_read_access *p_other);
void GDAPI godot_pool_string_array_read_access_destroy(godot_pool_string_array_read_access *p_read);

godot_pool_vector2_array_read_access GDAPI *godot_pool_vector2_array_read_access_copy(const godot_pool_vector2_array_read_access *p_other);
const godot_vector2 GDAPI *godot_pool_vector2_array_read_access_ptr(const godot_pool_vector2_array_read_access *p_read);
void GDAPI godot_pool_vector2_array_read_access_operator_assign(godot_pool_vector2_array_read_access *p_read, godot_pool_vector2_array_read_access *p_other);
void GDAPI godot_pool_vector2_array_read_access_destroy(godot_pool_vector2_array_read_access *p_read);

godot_pool_vector3_array_read_access GDAPI *godot_pool_vector3_array_read_access_copy(const godot_pool_vector3_array_read_access *p_other);
const godot_vector3 GDAPI *godot_pool_vector3_array_read_access_ptr(const godot_pool_vector3_array_read_access *p_read);
void GDAPI godot_pool_vector3_array_read_access_operator_assign(godot_pool_vector3_array_read_access *p_read, godot_pool_vector3_array_read_access *p_other);
void GDAPI godot_pool_vector3_array_read_access_destroy(godot_pool_vector3_array_read_access *p_read);

godot_pool_color_array_read_access GDAPI *godot_pool_color_array_read_access_copy(const godot_pool_color_array_read_access *p_other);
const godot_color GDAPI *godot_pool_color_array_read_access_ptr(const godot_pool_color_array_read_access *p_read);
void GDAPI godot_pool_color_array_read_access_operator_assign(godot_pool_color_array_read_access *p_read, godot_pool_color_array_read_access *p_other);
void GDAPI godot_pool_color_array_read_access_destroy(godot_pool_color_array_read_access *p_read);

//
// write accessor functions
//

godot_pool_byte_array_write_access GDAPI *godot_pool_byte_array_write_access_copy(const godot_pool_byte_array_write_access *p_other);
uint8_t GDAPI *godot_pool_byte_array_write_access_ptr(const godot_pool_byte_array_write_access *p_write);
void GDAPI godot_pool_byte_array_write_access_operator_assign(godot_pool_byte_array_write_access *p_write, godot_pool_byte_array_write_access *p_other);
void GDAPI godot_pool_byte_array_write_access_destroy(godot_pool_byte_array_write_access *p_write);

godot_pool_int_array_write_access GDAPI *godot_pool_int_array_write_access_copy(const godot_pool_int_array_write_access *p_other);
godot_int GDAPI *godot_pool_int_array_write_access_ptr(const godot_pool_int_array_write_access *p_write);
void GDAPI godot_pool_int_array_write_access_operator_assign(godot_pool_int_array_write_access *p_write, godot_pool_int_array_write_access *p_other);
void GDAPI godot_pool_int_array_write_access_destroy(godot_pool_int_array_write_access *p_write);

godot_pool_real_array_write_access GDAPI *godot_pool_real_array_write_access_copy(const godot_pool_real_array_write_access *p_other);
godot_real GDAPI *godot_pool_real_array_write_access_ptr(const godot_pool_real_array_write_access *p_write);
void GDAPI godot_pool_real_array_write_access_operator_assign(godot_pool_real_array_write_access *p_write, godot_pool_real_array_write_access *p_other);
void GDAPI godot_pool_real_array_write_access_destroy(godot_pool_real_array_write_access *p_write);

godot_pool_string_array_write_access GDAPI *godot_pool_string_array_write_access_copy(const godot_pool_string_array_write_access *p_other);
godot_string GDAPI *godot_pool_string_array_write_access_ptr(const godot_pool_string_array_write_access *p_write);
void GDAPI godot_pool_string_array_write_access_operator_assign(godot_pool_string_array_write_access *p_write, godot_pool_string_array_write_access *p_other);
void GDAPI godot_pool_string_array_write_access_destroy(godot_pool_string_array_write_access *p_write);

godot_pool_vector2_array_write_access GDAPI *godot_pool_vector2_array_write_access_copy(const godot_pool_vector2_array_write_access *p_other);
godot_vector2 GDAPI *godot_pool_vector2_array_write_access_ptr(const godot_pool_vector2_array_write_access *p_write);
void GDAPI godot_pool_vector2_array_write_access_operator_assign(godot_pool_vector2_array_write_access *p_write, godot_pool_vector2_array_write_access *p_other);
void GDAPI godot_pool_vector2_array_write_access_destroy(godot_pool_vector2_array_write_access *p_write);

godot_pool_vector3_array_write_access GDAPI *godot_pool_vector3_array_write_access_copy(const godot_pool_vector3_array_write_access *p_other);
godot_vector3 GDAPI *godot_pool_vector3_array_write_access_ptr(const godot_pool_vector3_array_write_access *p_write);
void GDAPI godot_pool_vector3_array_write_access_operator_assign(godot_pool_vector3_array_write_access *p_write, godot_pool_vector3_array_write_access *p_other);
void GDAPI godot_pool_vector3_array_write_access_destroy(godot_pool_vector3_array_write_access *p_write);

godot_pool_color_array_write_access GDAPI *godot_pool_color_array_write_access_copy(const godot_pool_color_array_write_access *p_other);
godot_color GDAPI *godot_pool_color_array_write_access_ptr(const godot_pool_color_array_write_access *p_write);
void GDAPI godot_pool_color_array_write_access_operator_assign(godot_pool_color_array_write_access *p_write, godot_pool_color_array_write_access *p_other);
void GDAPI godot_pool_color_array_write_access_destroy(godot_pool_color_array_write_access *p_write);

#ifdef __cplusplus
}
#endif

#endif // GDNATIVE_POOL_ARRAYS_H
