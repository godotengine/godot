/*************************************************************************/
/*  packed_arrays.h                                                      */
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

/////// PackedVector2iArray

#define GODOT_PACKED_VECTOR2I_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_VECTOR2I_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_VECTOR2I_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_VECTOR2I_ARRAY_SIZE];
} godot_packed_vector2i_array;
#endif

/////// PackedVector3Array

#define GODOT_PACKED_VECTOR3_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_VECTOR3_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_VECTOR3_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_VECTOR3_ARRAY_SIZE];
} godot_packed_vector3_array;
#endif

/////// PackedVector3iArray

#define GODOT_PACKED_VECTOR3I_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_VECTOR3I_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_VECTOR3I_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_VECTOR3I_ARRAY_SIZE];
} godot_packed_vector3i_array;
#endif

/////// PackedColorArray

#define GODOT_PACKED_COLOR_ARRAY_SIZE (2 * sizeof(void *))

#ifndef GODOT_CORE_API_GODOT_PACKED_COLOR_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_PACKED_COLOR_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_PACKED_COLOR_ARRAY_SIZE];
} godot_packed_color_array;
#endif

#include <gdnative/gdnative.h>

// Byte.

void GDAPI godot_packed_byte_array_new(godot_packed_byte_array *p_self);
void GDAPI godot_packed_byte_array_destroy(godot_packed_byte_array *p_self);

// Int32.

void GDAPI godot_packed_int32_array_new(godot_packed_int32_array *p_self);
void GDAPI godot_packed_int32_array_destroy(godot_packed_int32_array *p_self);

// Int64.

void GDAPI godot_packed_int64_array_new(godot_packed_int64_array *p_self);
void GDAPI godot_packed_int64_array_destroy(godot_packed_int64_array *p_self);

// Float32.

void GDAPI godot_packed_float32_array_new(godot_packed_float32_array *p_self);
void GDAPI godot_packed_float32_array_destroy(godot_packed_float32_array *p_self);

// Float64.

void GDAPI godot_packed_float64_array_new(godot_packed_float64_array *p_self);
void GDAPI godot_packed_float64_array_destroy(godot_packed_float64_array *p_self);

// String.

void GDAPI godot_packed_string_array_new(godot_packed_string_array *p_self);
void GDAPI godot_packed_string_array_destroy(godot_packed_string_array *p_self);

// Vector2.

void GDAPI godot_packed_vector2_array_new(godot_packed_vector2_array *p_self);
void GDAPI godot_packed_vector2_array_destroy(godot_packed_vector2_array *p_self);

// Vector2i.

void GDAPI godot_packed_vector2i_array_new(godot_packed_vector2i_array *p_self);
void GDAPI godot_packed_vector2i_array_destroy(godot_packed_vector2i_array *p_self);

// Vector3.

void GDAPI godot_packed_vector3_array_new(godot_packed_vector3_array *p_self);
void GDAPI godot_packed_vector3_array_destroy(godot_packed_vector3_array *p_self);

// Color.

void GDAPI godot_packed_color_array_new(godot_packed_color_array *p_self);
void GDAPI godot_packed_color_array_destroy(godot_packed_color_array *p_self);

#ifdef __cplusplus
}
#endif

#endif // GODOT_POOL_ARRAYS_H
