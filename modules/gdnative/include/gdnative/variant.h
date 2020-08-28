/*************************************************************************/
/*  variant.h                                                            */
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

#ifndef GODOT_VARIANT_H
#define GODOT_VARIANT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define GODOT_VARIANT_SIZE (16 + sizeof(int64_t))

#ifndef GODOT_CORE_API_GODOT_VARIANT_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_VARIANT_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_VARIANT_SIZE];
} godot_variant;
#endif

typedef enum godot_variant_type {
	GODOT_VARIANT_TYPE_NIL,

	// atomic types
	GODOT_VARIANT_TYPE_BOOL,
	GODOT_VARIANT_TYPE_INT,
	GODOT_VARIANT_TYPE_REAL,
	GODOT_VARIANT_TYPE_STRING,

	// math types
	GODOT_VARIANT_TYPE_VECTOR2,
	GODOT_VARIANT_TYPE_VECTOR2I,
	GODOT_VARIANT_TYPE_RECT2,
	GODOT_VARIANT_TYPE_RECT2I,
	GODOT_VARIANT_TYPE_VECTOR3,
	GODOT_VARIANT_TYPE_VECTOR3I,
	GODOT_VARIANT_TYPE_TRANSFORM2D,
	GODOT_VARIANT_TYPE_PLANE,
	GODOT_VARIANT_TYPE_QUAT,
	GODOT_VARIANT_TYPE_AABB,
	GODOT_VARIANT_TYPE_BASIS,
	GODOT_VARIANT_TYPE_TRANSFORM,

	// misc types
	GODOT_VARIANT_TYPE_COLOR,
	GODOT_VARIANT_TYPE_STRING_NAME,
	GODOT_VARIANT_TYPE_NODE_PATH,
	GODOT_VARIANT_TYPE_RID,
	GODOT_VARIANT_TYPE_OBJECT,
	GODOT_VARIANT_TYPE_CALLABLE,
	GODOT_VARIANT_TYPE_SIGNAL,
	GODOT_VARIANT_TYPE_DICTIONARY,
	GODOT_VARIANT_TYPE_ARRAY,

	// arrays
	GODOT_VARIANT_TYPE_PACKED_BYTE_ARRAY,
	GODOT_VARIANT_TYPE_PACKED_INT32_ARRAY,
	GODOT_VARIANT_TYPE_PACKED_INT64_ARRAY,
	GODOT_VARIANT_TYPE_PACKED_FLOAT32_ARRAY,
	GODOT_VARIANT_TYPE_PACKED_FLOAT64_ARRAY,
	GODOT_VARIANT_TYPE_PACKED_STRING_ARRAY,
	GODOT_VARIANT_TYPE_PACKED_VECTOR2_ARRAY,
	GODOT_VARIANT_TYPE_PACKED_VECTOR3_ARRAY,
	GODOT_VARIANT_TYPE_PACKED_COLOR_ARRAY,
} godot_variant_type;

typedef enum godot_variant_call_error_error {
	GODOT_CALL_ERROR_CALL_OK,
	GODOT_CALL_ERROR_CALL_ERROR_INVALID_METHOD,
	GODOT_CALL_ERROR_CALL_ERROR_INVALID_ARGUMENT,
	GODOT_CALL_ERROR_CALL_ERROR_TOO_MANY_ARGUMENTS,
	GODOT_CALL_ERROR_CALL_ERROR_TOO_FEW_ARGUMENTS,
	GODOT_CALL_ERROR_CALL_ERROR_INSTANCE_IS_NULL,
} godot_variant_call_error_error;

typedef struct godot_variant_call_error {
	godot_variant_call_error_error error;
	int argument;
	godot_variant_type expected;
} godot_variant_call_error;

typedef enum godot_variant_operator {
	// comparison
	GODOT_VARIANT_OP_EQUAL,
	GODOT_VARIANT_OP_NOT_EQUAL,
	GODOT_VARIANT_OP_LESS,
	GODOT_VARIANT_OP_LESS_EQUAL,
	GODOT_VARIANT_OP_GREATER,
	GODOT_VARIANT_OP_GREATER_EQUAL,

	// mathematic
	GODOT_VARIANT_OP_ADD,
	GODOT_VARIANT_OP_SUBTRACT,
	GODOT_VARIANT_OP_MULTIPLY,
	GODOT_VARIANT_OP_DIVIDE,
	GODOT_VARIANT_OP_NEGATE,
	GODOT_VARIANT_OP_POSITIVE,
	GODOT_VARIANT_OP_MODULE,
	GODOT_VARIANT_OP_STRING_CONCAT,

	// bitwise
	GODOT_VARIANT_OP_SHIFT_LEFT,
	GODOT_VARIANT_OP_SHIFT_RIGHT,
	GODOT_VARIANT_OP_BIT_AND,
	GODOT_VARIANT_OP_BIT_OR,
	GODOT_VARIANT_OP_BIT_XOR,
	GODOT_VARIANT_OP_BIT_NEGATE,

	// logic
	GODOT_VARIANT_OP_AND,
	GODOT_VARIANT_OP_OR,
	GODOT_VARIANT_OP_XOR,
	GODOT_VARIANT_OP_NOT,

	// containment
	GODOT_VARIANT_OP_IN,

	GODOT_VARIANT_OP_MAX,
} godot_variant_operator;

// reduce extern "C" nesting for VS2013
#ifdef __cplusplus
}
#endif

#include <gdnative/aabb.h>
#include <gdnative/array.h>
#include <gdnative/basis.h>
#include <gdnative/callable.h>
#include <gdnative/color.h>
#include <gdnative/dictionary.h>
#include <gdnative/node_path.h>
#include <gdnative/packed_arrays.h>
#include <gdnative/plane.h>
#include <gdnative/quat.h>
#include <gdnative/rect2.h>
#include <gdnative/rid.h>
#include <gdnative/string.h>
#include <gdnative/string_name.h>
#include <gdnative/transform.h>
#include <gdnative/transform2d.h>
#include <gdnative/variant.h>
#include <gdnative/vector2.h>
#include <gdnative/vector3.h>

#include <gdnative/gdnative.h>

#ifdef __cplusplus
extern "C" {
#endif

godot_variant_type GDAPI godot_variant_get_type(const godot_variant *p_v);

void GDAPI godot_variant_new_copy(godot_variant *r_dest, const godot_variant *p_src);

void GDAPI godot_variant_new_nil(godot_variant *r_dest);

void GDAPI godot_variant_new_bool(godot_variant *r_dest, const godot_bool p_b);
void GDAPI godot_variant_new_uint(godot_variant *r_dest, const uint64_t p_i);
void GDAPI godot_variant_new_int(godot_variant *r_dest, const int64_t p_i);
void GDAPI godot_variant_new_real(godot_variant *r_dest, const double p_r);
void GDAPI godot_variant_new_string(godot_variant *r_dest, const godot_string *p_s);
void GDAPI godot_variant_new_string_name(godot_variant *r_dest, const godot_string_name *p_s);
void GDAPI godot_variant_new_vector2(godot_variant *r_dest, const godot_vector2 *p_v2);
void GDAPI godot_variant_new_vector2i(godot_variant *r_dest, const godot_vector2i *p_v2);
void GDAPI godot_variant_new_rect2(godot_variant *r_dest, const godot_rect2 *p_rect2);
void GDAPI godot_variant_new_rect2i(godot_variant *r_dest, const godot_rect2i *p_rect2);
void GDAPI godot_variant_new_vector3(godot_variant *r_dest, const godot_vector3 *p_v3);
void GDAPI godot_variant_new_vector3i(godot_variant *r_dest, const godot_vector3i *p_v3);
void GDAPI godot_variant_new_transform2d(godot_variant *r_dest, const godot_transform2d *p_t2d);
void GDAPI godot_variant_new_plane(godot_variant *r_dest, const godot_plane *p_plane);
void GDAPI godot_variant_new_quat(godot_variant *r_dest, const godot_quat *p_quat);
void GDAPI godot_variant_new_aabb(godot_variant *r_dest, const godot_aabb *p_aabb);
void GDAPI godot_variant_new_basis(godot_variant *r_dest, const godot_basis *p_basis);
void GDAPI godot_variant_new_transform(godot_variant *r_dest, const godot_transform *p_trans);
void GDAPI godot_variant_new_color(godot_variant *r_dest, const godot_color *p_color);
void GDAPI godot_variant_new_node_path(godot_variant *r_dest, const godot_node_path *p_np);
void GDAPI godot_variant_new_rid(godot_variant *r_dest, const godot_rid *p_rid);
void GDAPI godot_variant_new_callable(godot_variant *r_dest, const godot_callable *p_callable);
void GDAPI godot_variant_new_signal(godot_variant *r_dest, const godot_signal *p_signal);
void GDAPI godot_variant_new_object(godot_variant *r_dest, const godot_object *p_obj);
void GDAPI godot_variant_new_dictionary(godot_variant *r_dest, const godot_dictionary *p_dict);
void GDAPI godot_variant_new_array(godot_variant *r_dest, const godot_array *p_arr);
void GDAPI godot_variant_new_packed_byte_array(godot_variant *r_dest, const godot_packed_byte_array *p_pba);
void GDAPI godot_variant_new_packed_int32_array(godot_variant *r_dest, const godot_packed_int32_array *p_pia);
void GDAPI godot_variant_new_packed_int64_array(godot_variant *r_dest, const godot_packed_int64_array *p_pia);
void GDAPI godot_variant_new_packed_float32_array(godot_variant *r_dest, const godot_packed_float32_array *p_pra);
void GDAPI godot_variant_new_packed_float64_array(godot_variant *r_dest, const godot_packed_float64_array *p_pra);
void GDAPI godot_variant_new_packed_string_array(godot_variant *r_dest, const godot_packed_string_array *p_psa);
void GDAPI godot_variant_new_packed_vector2_array(godot_variant *r_dest, const godot_packed_vector2_array *p_pv2a);
void GDAPI godot_variant_new_packed_vector3_array(godot_variant *r_dest, const godot_packed_vector3_array *p_pv3a);
void GDAPI godot_variant_new_packed_color_array(godot_variant *r_dest, const godot_packed_color_array *p_pca);

godot_bool GDAPI godot_variant_as_bool(const godot_variant *p_self);
uint64_t GDAPI godot_variant_as_uint(const godot_variant *p_self);
int64_t GDAPI godot_variant_as_int(const godot_variant *p_self);
double GDAPI godot_variant_as_real(const godot_variant *p_self);
godot_string GDAPI godot_variant_as_string(const godot_variant *p_self);
godot_string_name GDAPI godot_variant_as_string_name(const godot_variant *p_self);
godot_vector2 GDAPI godot_variant_as_vector2(const godot_variant *p_self);
godot_vector2i GDAPI godot_variant_as_vector2i(const godot_variant *p_self);
godot_rect2 GDAPI godot_variant_as_rect2(const godot_variant *p_self);
godot_rect2i GDAPI godot_variant_as_rect2i(const godot_variant *p_self);
godot_vector3 GDAPI godot_variant_as_vector3(const godot_variant *p_self);
godot_vector3i GDAPI godot_variant_as_vector3i(const godot_variant *p_self);
godot_transform2d GDAPI godot_variant_as_transform2d(const godot_variant *p_self);
godot_plane GDAPI godot_variant_as_plane(const godot_variant *p_self);
godot_quat GDAPI godot_variant_as_quat(const godot_variant *p_self);
godot_aabb GDAPI godot_variant_as_aabb(const godot_variant *p_self);
godot_basis GDAPI godot_variant_as_basis(const godot_variant *p_self);
godot_transform GDAPI godot_variant_as_transform(const godot_variant *p_self);
godot_color GDAPI godot_variant_as_color(const godot_variant *p_self);
godot_node_path GDAPI godot_variant_as_node_path(const godot_variant *p_self);
godot_rid GDAPI godot_variant_as_rid(const godot_variant *p_self);
godot_callable GDAPI godot_variant_as_callable(const godot_variant *p_self);
godot_signal GDAPI godot_variant_as_signal(const godot_variant *p_self);
godot_object GDAPI *godot_variant_as_object(const godot_variant *p_self);
godot_dictionary GDAPI godot_variant_as_dictionary(const godot_variant *p_self);
godot_array GDAPI godot_variant_as_array(const godot_variant *p_self);
godot_packed_byte_array GDAPI godot_variant_as_packed_byte_array(const godot_variant *p_self);
godot_packed_int32_array GDAPI godot_variant_as_packed_int32_array(const godot_variant *p_self);
godot_packed_int64_array GDAPI godot_variant_as_packed_int64_array(const godot_variant *p_self);
godot_packed_float32_array GDAPI godot_variant_as_packed_float32_array(const godot_variant *p_self);
godot_packed_float64_array GDAPI godot_variant_as_packed_float64_array(const godot_variant *p_self);
godot_packed_string_array GDAPI godot_variant_as_packed_string_array(const godot_variant *p_self);
godot_packed_vector2_array GDAPI godot_variant_as_packed_vector2_array(const godot_variant *p_self);
godot_packed_vector3_array GDAPI godot_variant_as_packed_vector3_array(const godot_variant *p_self);
godot_packed_color_array GDAPI godot_variant_as_packed_color_array(const godot_variant *p_self);

godot_variant GDAPI godot_variant_call(godot_variant *p_self, const godot_string *p_method, const godot_variant **p_args, const godot_int p_argcount, godot_variant_call_error *r_error);

godot_bool GDAPI godot_variant_has_method(const godot_variant *p_self, const godot_string *p_method);

godot_bool GDAPI godot_variant_operator_equal(const godot_variant *p_self, const godot_variant *p_other);
godot_bool GDAPI godot_variant_operator_less(const godot_variant *p_self, const godot_variant *p_other);

uint32_t GDAPI godot_variant_hash(const godot_variant *p_self);
godot_bool GDAPI godot_variant_hash_compare(const godot_variant *p_self, const godot_variant *p_other);

godot_bool GDAPI godot_variant_booleanize(const godot_variant *p_self);

void GDAPI godot_variant_destroy(godot_variant *p_self);

// GDNative core 1.1

godot_string GDAPI godot_variant_get_operator_name(godot_variant_operator p_op);
void GDAPI godot_variant_evaluate(godot_variant_operator p_op, const godot_variant *p_a, const godot_variant *p_b, godot_variant *r_ret, godot_bool *r_valid);

#ifdef __cplusplus
}
#endif

#endif
