/*************************************************************************/
/*  packed_arrays.h                                                      */
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

#ifndef GODOT_PACKED_ARRAYS_H
#define GODOT_PACKED_ARRAYS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/////// PackedByteArray

#define GODOT_PACKED_BYTE_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_BYTE_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_BYTE_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_BYTE_ARRAY_SIZE];
} godot_packed_byte_array;
#endif

/////// PackedInt32Array

#define GODOT_PACKED_INT32_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_INT32_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_INT32_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_INT32_ARRAY_SIZE];
} godot_packed_int32_array;
#endif

/////// PackedInt64Array

#define GODOT_PACKED_INT64_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_INT64_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_INT64_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_INT64_ARRAY_SIZE];
} godot_packed_int64_array;
#endif

/////// PackedFloat32Array

#define GODOT_PACKED_FLOAT32_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_FLOAT32_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_FLOAT32_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_FLOAT32_ARRAY_SIZE];
} godot_packed_float32_array;
#endif

/////// PackedFloat64Array

#define GODOT_PACKED_FLOAT64_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_FLOAT64_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_FLOAT64_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_FLOAT64_ARRAY_SIZE];
} godot_packed_float64_array;
#endif

/////// PackedStringArray

#define GODOT_PACKED_STRING_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_STRING_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_STRING_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_STRING_ARRAY_SIZE];
} godot_packed_string_array;
#endif

/////// PackedVector2Array

#define GODOT_PACKED_VECTOR2_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_VECTOR2_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_VECTOR2_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_VECTOR2_ARRAY_SIZE];
} godot_packed_vector2_array;
#endif

/////// PackedVector3Array

#define GODOT_PACKED_VECTOR3_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_VECTOR3_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_VECTOR3_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_VECTOR3_ARRAY_SIZE];
} godot_packed_vector3_array;
#endif

/////// PackedColorArray

#define GODOT_PACKED_COLOR_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_COLOR_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_COLOR_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_COLOR_ARRAY_SIZE];
} godot_packed_color_array;
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

void GDAPI godot_packed_byte_array_new(godot_packed_byte_array *r_dest);
void GDAPI godot_packed_byte_array_new_copy(godot_packed_byte_array *r_dest, const godot_packed_byte_array *p_src);
void GDAPI godot_packed_byte_array_new_with_array(godot_packed_byte_array *r_dest, const godot_array *p_a);

const uint8_t GDAPI *godot_packed_byte_array_ptr(const godot_packed_byte_array *p_self);
uint8_t GDAPI *godot_packed_byte_array_ptrw(godot_packed_byte_array *p_self);

void GDAPI godot_packed_byte_array_append(godot_packed_byte_array *p_self, const uint8_t p_data);

void GDAPI godot_packed_byte_array_append_array(godot_packed_byte_array *p_self, const godot_packed_byte_array *p_array);

godot_error GDAPI godot_packed_byte_array_insert(godot_packed_byte_array *p_self, const godot_int p_idx, const uint8_t p_data);

godot_bool GDAPI godot_packed_byte_array_has(godot_packed_byte_array *p_self, const uint8_t p_value);

void GDAPI godot_packed_byte_array_sort(godot_packed_byte_array *p_self);

void GDAPI godot_packed_byte_array_invert(godot_packed_byte_array *p_self);

void GDAPI godot_packed_byte_array_push_back(godot_packed_byte_array *p_self, const uint8_t p_data);

void GDAPI godot_packed_byte_array_remove(godot_packed_byte_array *p_self, const godot_int p_idx);

void GDAPI godot_packed_byte_array_resize(godot_packed_byte_array *p_self, const godot_int p_size);

void GDAPI godot_packed_byte_array_set(godot_packed_byte_array *p_self, const godot_int p_idx, const uint8_t p_data);
uint8_t GDAPI godot_packed_byte_array_get(const godot_packed_byte_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_packed_byte_array_size(const godot_packed_byte_array *p_self);

godot_bool GDAPI godot_packed_byte_array_empty(const godot_packed_byte_array *p_self);

void GDAPI godot_packed_byte_array_destroy(godot_packed_byte_array *p_self);

// int32

void GDAPI godot_packed_int32_array_new(godot_packed_int32_array *r_dest);
void GDAPI godot_packed_int32_array_new_copy(godot_packed_int32_array *r_dest, const godot_packed_int32_array *p_src);
void GDAPI godot_packed_int32_array_new_with_array(godot_packed_int32_array *r_dest, const godot_array *p_a);

const int32_t GDAPI *godot_packed_int32_array_ptr(const godot_packed_int32_array *p_self);
int32_t GDAPI *godot_packed_int32_array_ptrw(godot_packed_int32_array *p_self);

void GDAPI godot_packed_int32_array_append(godot_packed_int32_array *p_self, const int32_t p_data);

void GDAPI godot_packed_int32_array_append_array(godot_packed_int32_array *p_self, const godot_packed_int32_array *p_array);

godot_error GDAPI godot_packed_int32_array_insert(godot_packed_int32_array *p_self, const godot_int p_idx, const int32_t p_data);

godot_bool GDAPI godot_packed_int32_array_has(godot_packed_int32_array *p_self, const int32_t p_value);

void GDAPI godot_packed_int32_array_sort(godot_packed_int32_array *p_self);

void GDAPI godot_packed_int32_array_invert(godot_packed_int32_array *p_self);

void GDAPI godot_packed_int32_array_push_back(godot_packed_int32_array *p_self, const int32_t p_data);

void GDAPI godot_packed_int32_array_remove(godot_packed_int32_array *p_self, const godot_int p_idx);

void GDAPI godot_packed_int32_array_resize(godot_packed_int32_array *p_self, const godot_int p_size);

void GDAPI godot_packed_int32_array_set(godot_packed_int32_array *p_self, const godot_int p_idx, const int32_t p_data);
int32_t GDAPI godot_packed_int32_array_get(const godot_packed_int32_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_packed_int32_array_size(const godot_packed_int32_array *p_self);

godot_bool GDAPI godot_packed_int32_array_empty(const godot_packed_int32_array *p_self);

void GDAPI godot_packed_int32_array_destroy(godot_packed_int32_array *p_self);

// int64

void GDAPI godot_packed_int64_array_new(godot_packed_int64_array *r_dest);
void GDAPI godot_packed_int64_array_new_copy(godot_packed_int64_array *r_dest, const godot_packed_int64_array *p_src);
void GDAPI godot_packed_int64_array_new_with_array(godot_packed_int64_array *r_dest, const godot_array *p_a);

const int64_t GDAPI *godot_packed_int64_array_ptr(const godot_packed_int64_array *p_self);
int64_t GDAPI *godot_packed_int64_array_ptrw(godot_packed_int64_array *p_self);

void GDAPI godot_packed_int64_array_append(godot_packed_int64_array *p_self, const int64_t p_data);

void GDAPI godot_packed_int64_array_append_array(godot_packed_int64_array *p_self, const godot_packed_int64_array *p_array);

godot_error GDAPI godot_packed_int64_array_insert(godot_packed_int64_array *p_self, const godot_int p_idx, const int64_t p_data);

godot_bool GDAPI godot_packed_int64_array_has(godot_packed_int64_array *p_self, const int64_t p_value);

void GDAPI godot_packed_int64_array_sort(godot_packed_int64_array *p_self);

void GDAPI godot_packed_int64_array_invert(godot_packed_int64_array *p_self);

void GDAPI godot_packed_int64_array_push_back(godot_packed_int64_array *p_self, const int64_t p_data);

void GDAPI godot_packed_int64_array_remove(godot_packed_int64_array *p_self, const godot_int p_idx);

void GDAPI godot_packed_int64_array_resize(godot_packed_int64_array *p_self, const godot_int p_size);

void GDAPI godot_packed_int64_array_set(godot_packed_int64_array *p_self, const godot_int p_idx, const int64_t p_data);
int64_t GDAPI godot_packed_int64_array_get(const godot_packed_int64_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_packed_int64_array_size(const godot_packed_int64_array *p_self);

godot_bool GDAPI godot_packed_int64_array_empty(const godot_packed_int64_array *p_self);

void GDAPI godot_packed_int64_array_destroy(godot_packed_int64_array *p_self);

// float32

void GDAPI godot_packed_float32_array_new(godot_packed_float32_array *r_dest);
void GDAPI godot_packed_float32_array_new_copy(godot_packed_float32_array *r_dest, const godot_packed_float32_array *p_src);
void GDAPI godot_packed_float32_array_new_with_array(godot_packed_float32_array *r_dest, const godot_array *p_a);

const float GDAPI *godot_packed_float32_array_ptr(const godot_packed_float32_array *p_self);
float GDAPI *godot_packed_float32_array_ptrw(godot_packed_float32_array *p_self);

void GDAPI godot_packed_float32_array_append(godot_packed_float32_array *p_self, const float p_data);

void GDAPI godot_packed_float32_array_append_array(godot_packed_float32_array *p_self, const godot_packed_float32_array *p_array);

godot_error GDAPI godot_packed_float32_array_insert(godot_packed_float32_array *p_self, const godot_int p_idx, const float p_data);

godot_bool GDAPI godot_packed_float32_array_has(godot_packed_float32_array *p_self, const float p_value);

void GDAPI godot_packed_float32_array_sort(godot_packed_float32_array *p_self);

void GDAPI godot_packed_float32_array_invert(godot_packed_float32_array *p_self);

void GDAPI godot_packed_float32_array_push_back(godot_packed_float32_array *p_self, const float p_data);

void GDAPI godot_packed_float32_array_remove(godot_packed_float32_array *p_self, const godot_int p_idx);

void GDAPI godot_packed_float32_array_resize(godot_packed_float32_array *p_self, const godot_int p_size);

void GDAPI godot_packed_float32_array_set(godot_packed_float32_array *p_self, const godot_int p_idx, const float p_data);
float GDAPI godot_packed_float32_array_get(const godot_packed_float32_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_packed_float32_array_size(const godot_packed_float32_array *p_self);

godot_bool GDAPI godot_packed_float32_array_empty(const godot_packed_float32_array *p_self);

void GDAPI godot_packed_float32_array_destroy(godot_packed_float32_array *p_self);

// float64

void GDAPI godot_packed_float64_array_new(godot_packed_float64_array *r_dest);
void GDAPI godot_packed_float64_array_new_copy(godot_packed_float64_array *r_dest, const godot_packed_float64_array *p_src);
void GDAPI godot_packed_float64_array_new_with_array(godot_packed_float64_array *r_dest, const godot_array *p_a);

const double GDAPI *godot_packed_float64_array_ptr(const godot_packed_float64_array *p_self);
double GDAPI *godot_packed_float64_array_ptrw(godot_packed_float64_array *p_self);

void GDAPI godot_packed_float64_array_append(godot_packed_float64_array *p_self, const double p_data);

void GDAPI godot_packed_float64_array_append_array(godot_packed_float64_array *p_self, const godot_packed_float64_array *p_array);

godot_error GDAPI godot_packed_float64_array_insert(godot_packed_float64_array *p_self, const godot_int p_idx, const double p_data);

godot_bool GDAPI godot_packed_float64_array_has(godot_packed_float64_array *p_self, const double p_value);

void GDAPI godot_packed_float64_array_sort(godot_packed_float64_array *p_self);

void GDAPI godot_packed_float64_array_invert(godot_packed_float64_array *p_self);

void GDAPI godot_packed_float64_array_push_back(godot_packed_float64_array *p_self, const double p_data);

void GDAPI godot_packed_float64_array_remove(godot_packed_float64_array *p_self, const godot_int p_idx);

void GDAPI godot_packed_float64_array_resize(godot_packed_float64_array *p_self, const godot_int p_size);

void GDAPI godot_packed_float64_array_set(godot_packed_float64_array *p_self, const godot_int p_idx, const double p_data);
double GDAPI godot_packed_float64_array_get(const godot_packed_float64_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_packed_float64_array_size(const godot_packed_float64_array *p_self);

godot_bool GDAPI godot_packed_float64_array_empty(const godot_packed_float64_array *p_self);

void GDAPI godot_packed_float64_array_destroy(godot_packed_float64_array *p_self);

// string

void GDAPI godot_packed_string_array_new(godot_packed_string_array *r_dest);
void GDAPI godot_packed_string_array_new_copy(godot_packed_string_array *r_dest, const godot_packed_string_array *p_src);
void GDAPI godot_packed_string_array_new_with_array(godot_packed_string_array *r_dest, const godot_array *p_a);

const godot_string GDAPI *godot_packed_string_array_ptr(const godot_packed_string_array *p_self);
godot_string GDAPI *godot_packed_string_array_ptrw(godot_packed_string_array *p_self);

void GDAPI godot_packed_string_array_append(godot_packed_string_array *p_self, const godot_string *p_data);

void GDAPI godot_packed_string_array_append_array(godot_packed_string_array *p_self, const godot_packed_string_array *p_array);

godot_error GDAPI godot_packed_string_array_insert(godot_packed_string_array *p_self, const godot_int p_idx, const godot_string *p_data);

godot_bool GDAPI godot_packed_string_array_has(godot_packed_string_array *p_self, const godot_string *p_value);

void GDAPI godot_packed_string_array_sort(godot_packed_string_array *p_self);

void GDAPI godot_packed_string_array_invert(godot_packed_string_array *p_self);

void GDAPI godot_packed_string_array_push_back(godot_packed_string_array *p_self, const godot_string *p_data);

void GDAPI godot_packed_string_array_remove(godot_packed_string_array *p_self, const godot_int p_idx);

void GDAPI godot_packed_string_array_resize(godot_packed_string_array *p_self, const godot_int p_size);

void GDAPI godot_packed_string_array_set(godot_packed_string_array *p_self, const godot_int p_idx, const godot_string *p_data);
godot_string GDAPI godot_packed_string_array_get(const godot_packed_string_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_packed_string_array_size(const godot_packed_string_array *p_self);

godot_bool GDAPI godot_packed_string_array_empty(const godot_packed_string_array *p_self);

void GDAPI godot_packed_string_array_destroy(godot_packed_string_array *p_self);

// vector2

void GDAPI godot_packed_vector2_array_new(godot_packed_vector2_array *r_dest);
void GDAPI godot_packed_vector2_array_new_copy(godot_packed_vector2_array *r_dest, const godot_packed_vector2_array *p_src);
void GDAPI godot_packed_vector2_array_new_with_array(godot_packed_vector2_array *r_dest, const godot_array *p_a);

const godot_vector2 GDAPI *godot_packed_vector2_array_ptr(const godot_packed_vector2_array *p_self);
godot_vector2 GDAPI *godot_packed_vector2_array_ptrw(godot_packed_vector2_array *p_self);

void GDAPI godot_packed_vector2_array_append(godot_packed_vector2_array *p_self, const godot_vector2 *p_data);

void GDAPI godot_packed_vector2_array_append_array(godot_packed_vector2_array *p_self, const godot_packed_vector2_array *p_array);

godot_error GDAPI godot_packed_vector2_array_insert(godot_packed_vector2_array *p_self, const godot_int p_idx, const godot_vector2 *p_data);

godot_bool GDAPI godot_packed_vector2_array_has(godot_packed_vector2_array *p_self, const godot_vector2 *p_value);

void GDAPI godot_packed_vector2_array_sort(godot_packed_vector2_array *p_self);

void GDAPI godot_packed_vector2_array_invert(godot_packed_vector2_array *p_self);

void GDAPI godot_packed_vector2_array_push_back(godot_packed_vector2_array *p_self, const godot_vector2 *p_data);

void GDAPI godot_packed_vector2_array_remove(godot_packed_vector2_array *p_self, const godot_int p_idx);

void GDAPI godot_packed_vector2_array_resize(godot_packed_vector2_array *p_self, const godot_int p_size);

void GDAPI godot_packed_vector2_array_set(godot_packed_vector2_array *p_self, const godot_int p_idx, const godot_vector2 *p_data);
godot_vector2 GDAPI godot_packed_vector2_array_get(const godot_packed_vector2_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_packed_vector2_array_size(const godot_packed_vector2_array *p_self);

godot_bool GDAPI godot_packed_vector2_array_empty(const godot_packed_vector2_array *p_self);

void GDAPI godot_packed_vector2_array_destroy(godot_packed_vector2_array *p_self);

// vector3

void GDAPI godot_packed_vector3_array_new(godot_packed_vector3_array *r_dest);
void GDAPI godot_packed_vector3_array_new_copy(godot_packed_vector3_array *r_dest, const godot_packed_vector3_array *p_src);
void GDAPI godot_packed_vector3_array_new_with_array(godot_packed_vector3_array *r_dest, const godot_array *p_a);

const godot_vector3 GDAPI *godot_packed_vector3_array_ptr(const godot_packed_vector3_array *p_self);
godot_vector3 GDAPI *godot_packed_vector3_array_ptrw(godot_packed_vector3_array *p_self);

void GDAPI godot_packed_vector3_array_append(godot_packed_vector3_array *p_self, const godot_vector3 *p_data);

void GDAPI godot_packed_vector3_array_append_array(godot_packed_vector3_array *p_self, const godot_packed_vector3_array *p_array);

godot_error GDAPI godot_packed_vector3_array_insert(godot_packed_vector3_array *p_self, const godot_int p_idx, const godot_vector3 *p_data);

godot_bool GDAPI godot_packed_vector3_array_has(godot_packed_vector3_array *p_self, const godot_vector3 *p_value);

void GDAPI godot_packed_vector3_array_sort(godot_packed_vector3_array *p_self);

void GDAPI godot_packed_vector3_array_invert(godot_packed_vector3_array *p_self);

void GDAPI godot_packed_vector3_array_push_back(godot_packed_vector3_array *p_self, const godot_vector3 *p_data);

void GDAPI godot_packed_vector3_array_remove(godot_packed_vector3_array *p_self, const godot_int p_idx);

void GDAPI godot_packed_vector3_array_resize(godot_packed_vector3_array *p_self, const godot_int p_size);

void GDAPI godot_packed_vector3_array_set(godot_packed_vector3_array *p_self, const godot_int p_idx, const godot_vector3 *p_data);
godot_vector3 GDAPI godot_packed_vector3_array_get(const godot_packed_vector3_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_packed_vector3_array_size(const godot_packed_vector3_array *p_self);

godot_bool GDAPI godot_packed_vector3_array_empty(const godot_packed_vector3_array *p_self);

void GDAPI godot_packed_vector3_array_destroy(godot_packed_vector3_array *p_self);

// color

void GDAPI godot_packed_color_array_new(godot_packed_color_array *r_dest);
void GDAPI godot_packed_color_array_new_copy(godot_packed_color_array *r_dest, const godot_packed_color_array *p_src);
void GDAPI godot_packed_color_array_new_with_array(godot_packed_color_array *r_dest, const godot_array *p_a);

const godot_color GDAPI *godot_packed_color_array_ptr(const godot_packed_color_array *p_self);
godot_color GDAPI *godot_packed_color_array_ptrw(godot_packed_color_array *p_self);

void GDAPI godot_packed_color_array_append(godot_packed_color_array *p_self, const godot_color *p_data);

void GDAPI godot_packed_color_array_append_array(godot_packed_color_array *p_self, const godot_packed_color_array *p_array);

godot_error GDAPI godot_packed_color_array_insert(godot_packed_color_array *p_self, const godot_int p_idx, const godot_color *p_data);

godot_bool GDAPI godot_packed_color_array_has(godot_packed_color_array *p_self, const godot_color *p_value);

void GDAPI godot_packed_color_array_sort(godot_packed_color_array *p_self);

void GDAPI godot_packed_color_array_invert(godot_packed_color_array *p_self);

void GDAPI godot_packed_color_array_push_back(godot_packed_color_array *p_self, const godot_color *p_data);

void GDAPI godot_packed_color_array_remove(godot_packed_color_array *p_self, const godot_int p_idx);

void GDAPI godot_packed_color_array_resize(godot_packed_color_array *p_self, const godot_int p_size);

void GDAPI godot_packed_color_array_set(godot_packed_color_array *p_self, const godot_int p_idx, const godot_color *p_data);
godot_color GDAPI godot_packed_color_array_get(const godot_packed_color_array *p_self, const godot_int p_idx);

godot_int GDAPI godot_packed_color_array_size(const godot_packed_color_array *p_self);

godot_bool GDAPI godot_packed_color_array_empty(const godot_packed_color_array *p_self);

void GDAPI godot_packed_color_array_destroy(godot_packed_color_array *p_self);

#ifdef __cplusplus
}
#endif

#endif // GODOT_POOL_ARRAYS_H
