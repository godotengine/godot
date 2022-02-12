/*************************************************************************/
/*  variant.h                                                            */
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

#ifndef GODOT_VARIANT_H
#define GODOT_VARIANT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <gdnative/math_defs.h>
#include <gdnative/variant_struct.h>

typedef enum godot_variant_type {
	GODOT_VARIANT_TYPE_NIL,

	// atomic types
	GODOT_VARIANT_TYPE_BOOL,
	GODOT_VARIANT_TYPE_INT,
	GODOT_VARIANT_TYPE_FLOAT,
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
	GODOT_VARIANT_TYPE_QUATERNION,
	GODOT_VARIANT_TYPE_AABB,
	GODOT_VARIANT_TYPE_BASIS,
	GODOT_VARIANT_TYPE_TRANSFORM3D,

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

typedef enum godot_variant_utility_function_type {
	GODOT_UTILITY_FUNC_TYPE_MATH,
	GODOT_UTILITY_FUNC_TYPE_RANDOM,
	GODOT_UTILITY_FUNC_TYPE_GENERAL,
} godot_variant_utility_function_type;

// Types for function pointers.
typedef void (*godot_validated_operator_evaluator)(const godot_variant *p_left, const godot_variant *p_right, godot_variant *r_result);
typedef void (*godot_ptr_operator_evaluator)(const void *p_left, const void *p_right, void *r_result);
typedef void (*godot_validated_builtin_method)(godot_variant *p_base, const godot_variant **p_args, int p_argument_count, godot_variant *r_return);
typedef void (*godot_ptr_builtin_method)(void *p_base, const void **p_args, void *r_return, int p_argument_count);
typedef void (*godot_validated_constructor)(godot_variant *p_base, const godot_variant **p_args);
typedef void (*godot_ptr_constructor)(void *p_base, const void **p_args);
typedef void (*godot_validated_setter)(godot_variant *p_base, const godot_variant *p_value);
typedef void (*godot_validated_getter)(const godot_variant *p_base, godot_variant *r_value);
typedef void (*godot_ptr_setter)(void *p_base, const void *p_value);
typedef void (*godot_ptr_getter)(const void *p_base, void *r_value);
typedef void (*godot_validated_indexed_setter)(godot_variant *p_base, godot_int p_index, const godot_variant *p_value, bool *r_oob);
typedef void (*godot_validated_indexed_getter)(const godot_variant *p_base, godot_int p_index, godot_variant *r_value, bool *r_oob);
typedef void (*godot_ptr_indexed_setter)(void *p_base, godot_int p_index, const void *p_value);
typedef void (*godot_ptr_indexed_getter)(const void *p_base, godot_int p_index, void *r_value);
typedef void (*godot_validated_keyed_setter)(godot_variant *p_base, const godot_variant *p_key, const godot_variant *p_value, bool *r_valid);
typedef void (*godot_validated_keyed_getter)(const godot_variant *p_base, const godot_variant *p_key, godot_variant *r_value, bool *r_valid);
typedef bool (*godot_validated_keyed_checker)(const godot_variant *p_base, const godot_variant *p_key, bool *r_valid);
typedef void (*godot_ptr_keyed_setter)(void *p_base, const void *p_key, const void *p_value);
typedef void (*godot_ptr_keyed_getter)(const void *p_base, const void *p_key, void *r_value);
typedef uint32_t (*godot_ptr_keyed_checker)(const godot_variant *p_base, const godot_variant *p_key);
typedef void (*godot_validated_utility_function)(godot_variant *r_return, const godot_variant **p_arguments, int p_argument_count);
typedef void (*godot_ptr_utility_function)(void *r_return, const void **p_arguments, int p_argument_count);

#include <gdnative/aabb.h>
#include <gdnative/array.h>
#include <gdnative/basis.h>
#include <gdnative/callable.h>
#include <gdnative/color.h>
#include <gdnative/dictionary.h>
#include <gdnative/node_path.h>
#include <gdnative/packed_arrays.h>
#include <gdnative/plane.h>
#include <gdnative/quaternion.h>
#include <gdnative/rect2.h>
#include <gdnative/rid.h>
#include <gdnative/signal.h>
#include <gdnative/string.h>
#include <gdnative/string_name.h>
#include <gdnative/transform2d.h>
#include <gdnative/transform_3d.h>
#include <gdnative/variant.h>
#include <gdnative/vector2.h>
#include <gdnative/vector3.h>

#include <gdnative/gdnative.h>

// Memory.

void GDAPI godot_variant_new_copy(godot_variant *r_dest, const godot_variant *p_src);

void GDAPI godot_variant_new_nil(godot_variant *r_dest);
void GDAPI godot_variant_new_bool(godot_variant *r_dest, const godot_bool p_b);
void GDAPI godot_variant_new_int(godot_variant *r_dest, const godot_int p_i);
void GDAPI godot_variant_new_float(godot_variant *r_dest, const godot_float p_f);
void GDAPI godot_variant_new_string(godot_variant *r_dest, const godot_string *p_s);
void GDAPI godot_variant_new_vector2(godot_variant *r_dest, const godot_vector2 *p_v2);
void GDAPI godot_variant_new_vector2i(godot_variant *r_dest, const godot_vector2i *p_v2);
void GDAPI godot_variant_new_rect2(godot_variant *r_dest, const godot_rect2 *p_rect2);
void GDAPI godot_variant_new_rect2i(godot_variant *r_dest, const godot_rect2i *p_rect2);
void GDAPI godot_variant_new_vector3(godot_variant *r_dest, const godot_vector3 *p_v3);
void GDAPI godot_variant_new_vector3i(godot_variant *r_dest, const godot_vector3i *p_v3);
void GDAPI godot_variant_new_transform2d(godot_variant *r_dest, const godot_transform2d *p_t2d);
void GDAPI godot_variant_new_plane(godot_variant *r_dest, const godot_plane *p_plane);
void GDAPI godot_variant_new_quaternion(godot_variant *r_dest, const godot_quaternion *p_quaternion);
void GDAPI godot_variant_new_aabb(godot_variant *r_dest, const godot_aabb *p_aabb);
void GDAPI godot_variant_new_basis(godot_variant *r_dest, const godot_basis *p_basis);
void GDAPI godot_variant_new_transform3d(godot_variant *r_dest, const godot_transform3d *p_trans);
void GDAPI godot_variant_new_color(godot_variant *r_dest, const godot_color *p_color);
void GDAPI godot_variant_new_string_name(godot_variant *r_dest, const godot_string_name *p_s);
void GDAPI godot_variant_new_node_path(godot_variant *r_dest, const godot_node_path *p_np);
void GDAPI godot_variant_new_rid(godot_variant *r_dest, const godot_rid *p_rid);
void GDAPI godot_variant_new_object(godot_variant *r_dest, const godot_object *p_obj);
void GDAPI godot_variant_new_callable(godot_variant *r_dest, const godot_callable *p_callable);
void GDAPI godot_variant_new_signal(godot_variant *r_dest, const godot_signal *p_signal);
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
godot_int GDAPI godot_variant_as_int(const godot_variant *p_self);
godot_float GDAPI godot_variant_as_float(const godot_variant *p_self);
godot_string GDAPI godot_variant_as_string(const godot_variant *p_self);
godot_vector2 GDAPI godot_variant_as_vector2(const godot_variant *p_self);
godot_vector2i GDAPI godot_variant_as_vector2i(const godot_variant *p_self);
godot_rect2 GDAPI godot_variant_as_rect2(const godot_variant *p_self);
godot_rect2i GDAPI godot_variant_as_rect2i(const godot_variant *p_self);
godot_vector3 GDAPI godot_variant_as_vector3(const godot_variant *p_self);
godot_vector3i GDAPI godot_variant_as_vector3i(const godot_variant *p_self);
godot_transform2d GDAPI godot_variant_as_transform2d(const godot_variant *p_self);
godot_plane GDAPI godot_variant_as_plane(const godot_variant *p_self);
godot_quaternion GDAPI godot_variant_as_quaternion(const godot_variant *p_self);
godot_aabb GDAPI godot_variant_as_aabb(const godot_variant *p_self);
godot_basis GDAPI godot_variant_as_basis(const godot_variant *p_self);
godot_transform3d GDAPI godot_variant_as_transform3d(const godot_variant *p_self);
godot_color GDAPI godot_variant_as_color(const godot_variant *p_self);
godot_string_name GDAPI godot_variant_as_string_name(const godot_variant *p_self);
godot_node_path GDAPI godot_variant_as_node_path(const godot_variant *p_self);
godot_rid GDAPI godot_variant_as_rid(const godot_variant *p_self);
godot_object GDAPI *godot_variant_as_object(const godot_variant *p_self);
godot_callable GDAPI godot_variant_as_callable(const godot_variant *p_self);
godot_signal GDAPI godot_variant_as_signal(const godot_variant *p_self);
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

void GDAPI godot_variant_destroy(godot_variant *p_self);

// Dynamic interaction.

void GDAPI godot_variant_call(godot_variant *p_self, const godot_string_name *p_method, const godot_variant **p_args, const godot_int p_argument_count, godot_variant *r_return, godot_variant_call_error *r_error);
void GDAPI godot_variant_call_with_cstring(godot_variant *p_self, const char *p_method, const godot_variant **p_args, const godot_int p_argument_count, godot_variant *r_return, godot_variant_call_error *r_error);
void GDAPI godot_variant_call_static(godot_variant_type p_type, const godot_string_name *p_method, const godot_variant **p_args, const godot_int p_argument_count, godot_variant *r_return, godot_variant_call_error *r_error);
void GDAPI godot_variant_call_static_with_cstring(godot_variant_type p_type, const char *p_method, const godot_variant **p_args, const godot_int p_argument_count, godot_variant *r_return, godot_variant_call_error *r_error);
void GDAPI godot_variant_evaluate(godot_variant_operator p_op, const godot_variant *p_a, const godot_variant *p_b, godot_variant *r_return, bool *r_valid);
void GDAPI godot_variant_set(godot_variant *p_self, const godot_variant *p_key, const godot_variant *p_value, bool *r_valid);
void GDAPI godot_variant_set_named(godot_variant *p_self, const godot_string_name *p_name, const godot_variant *p_value, bool *r_valid);
void GDAPI godot_variant_set_named_with_cstring(godot_variant *p_self, const char *p_name, const godot_variant *p_value, bool *r_valid);
void GDAPI godot_variant_set_keyed(godot_variant *p_self, const godot_variant *p_key, const godot_variant *p_value, bool *r_valid);
void GDAPI godot_variant_set_indexed(godot_variant *p_self, godot_int p_index, const godot_variant *p_value, bool *r_valid, bool *r_oob);
godot_variant GDAPI godot_variant_get(const godot_variant *p_self, const godot_variant *p_key, bool *r_valid);
godot_variant GDAPI godot_variant_get_named(const godot_variant *p_self, const godot_string_name *p_key, bool *r_valid);
godot_variant GDAPI godot_variant_get_named_with_cstring(const godot_variant *p_self, const char *p_key, bool *r_valid);
godot_variant GDAPI godot_variant_get_keyed(const godot_variant *p_self, const godot_variant *p_key, bool *r_valid);
godot_variant GDAPI godot_variant_get_indexed(const godot_variant *p_self, godot_int p_index, bool *r_valid, bool *r_oob);
/// Iteration.
bool GDAPI godot_variant_iter_init(const godot_variant *p_self, godot_variant *r_iter, bool *r_valid);
bool GDAPI godot_variant_iter_next(const godot_variant *p_self, godot_variant *r_iter, bool *r_valid);
godot_variant GDAPI godot_variant_iter_get(const godot_variant *p_self, godot_variant *r_iter, bool *r_valid);

/// Variant functions.
godot_bool GDAPI godot_variant_hash_compare(const godot_variant *p_self, const godot_variant *p_other);
godot_bool GDAPI godot_variant_booleanize(const godot_variant *p_self);
void GDAPI godot_variant_blend(const godot_variant *p_a, const godot_variant *p_b, float p_c, godot_variant *r_dst);
void GDAPI godot_variant_interpolate(const godot_variant *p_a, const godot_variant *p_b, float p_c, godot_variant *r_dst);
godot_variant GDAPI godot_variant_duplicate(const godot_variant *p_self, godot_bool p_deep);
godot_string GDAPI godot_variant_stringify(const godot_variant *p_self);

// Discovery API.

/// Operators.
godot_validated_operator_evaluator GDAPI godot_variant_get_validated_operator_evaluator(godot_variant_operator p_operator, godot_variant_type p_type_a, godot_variant_type p_type_b);
godot_ptr_operator_evaluator GDAPI godot_variant_get_ptr_operator_evaluator(godot_variant_operator p_operator, godot_variant_type p_type_a, godot_variant_type p_type_b);
godot_variant_type GDAPI godot_variant_get_operator_return_type(godot_variant_operator p_operator, godot_variant_type p_type_a, godot_variant_type p_type_b);
godot_string GDAPI godot_variant_get_operator_name(godot_variant_operator p_operator);

/// Built-in methods.
bool GDAPI godot_variant_has_builtin_method(godot_variant_type p_type, const godot_string_name *p_method);
bool GDAPI godot_variant_has_builtin_method_with_cstring(godot_variant_type p_type, const char *p_method);
godot_validated_builtin_method GDAPI godot_variant_get_validated_builtin_method(godot_variant_type p_type, const godot_string_name *p_method);
godot_validated_builtin_method GDAPI godot_variant_get_validated_builtin_method_with_cstring(godot_variant_type p_type, const char *p_method);
godot_ptr_builtin_method GDAPI godot_variant_get_ptr_builtin_method(godot_variant_type p_type, const godot_string_name *p_method);
godot_ptr_builtin_method GDAPI godot_variant_get_ptr_builtin_method_with_cstring(godot_variant_type p_type, const char *p_method);
int GDAPI godot_variant_get_builtin_method_argument_count(godot_variant_type p_type, const godot_string_name *p_method);
int GDAPI godot_variant_get_builtin_method_argument_count_with_cstring(godot_variant_type p_type, const char *p_method);
godot_variant_type GDAPI godot_variant_get_builtin_method_argument_type(godot_variant_type p_type, const godot_string_name *p_method, int p_argument);
godot_variant_type GDAPI godot_variant_get_builtin_method_argument_type_with_cstring(godot_variant_type p_type, const char *p_method, int p_argument);
godot_string GDAPI godot_variant_get_builtin_method_argument_name(godot_variant_type p_type, const godot_string_name *p_method, int p_argument);
godot_string GDAPI godot_variant_get_builtin_method_argument_name_with_cstring(godot_variant_type p_type, const char *p_method, int p_argument);
bool GDAPI godot_variant_has_builtin_method_return_value(godot_variant_type p_type, const godot_string_name *p_method);
bool GDAPI godot_variant_has_builtin_method_return_value_with_cstring(godot_variant_type p_type, const char *p_method);
godot_variant_type GDAPI godot_variant_get_builtin_method_return_type(godot_variant_type p_type, const godot_string_name *p_method);
godot_variant_type GDAPI godot_variant_get_builtin_method_return_type_with_cstring(godot_variant_type p_type, const char *p_method);
bool GDAPI godot_variant_is_builtin_method_const(godot_variant_type p_type, const godot_string_name *p_method);
bool GDAPI godot_variant_is_builtin_method_const_with_cstring(godot_variant_type p_type, const char *p_method);
bool GDAPI godot_variant_is_builtin_method_static(godot_variant_type p_type, const godot_string_name *p_method);
bool GDAPI godot_variant_is_builtin_method_static_with_cstring(godot_variant_type p_type, const char *p_method);
bool GDAPI godot_variant_is_builtin_method_vararg(godot_variant_type p_type, const godot_string_name *p_method);
bool GDAPI godot_variant_is_builtin_method_vararg_with_cstring(godot_variant_type p_type, const char *p_method);
int GDAPI godot_variant_get_builtin_method_count(godot_variant_type p_type);
void GDAPI godot_variant_get_builtin_method_list(godot_variant_type p_type, godot_string_name *r_list);

/// Constructors.
int GDAPI godot_variant_get_constructor_count(godot_variant_type p_type);
godot_validated_constructor GDAPI godot_variant_get_validated_constructor(godot_variant_type p_type, int p_constructor);
godot_ptr_constructor GDAPI godot_variant_get_ptr_constructor(godot_variant_type p_type, int p_constructor);
int GDAPI godot_variant_get_constructor_argument_count(godot_variant_type p_type, int p_constructor);
godot_variant_type GDAPI godot_variant_get_constructor_argument_type(godot_variant_type p_type, int p_constructor, int p_argument);
godot_string GDAPI godot_variant_get_constructor_argument_name(godot_variant_type p_type, int p_constructor, int p_argument);
void GDAPI godot_variant_construct(godot_variant_type p_type, godot_variant *p_base, const godot_variant **p_args, int p_argument_count, godot_variant_call_error *r_error);

/// Properties.
godot_variant_type GDAPI godot_variant_get_member_type(godot_variant_type p_type, const godot_string_name *p_member);
godot_variant_type GDAPI godot_variant_get_member_type_with_cstring(godot_variant_type p_type, const char *p_member);
int GDAPI godot_variant_get_member_count(godot_variant_type p_type);
void GDAPI godot_variant_get_member_list(godot_variant_type p_type, godot_string_name *r_list);
godot_validated_setter GDAPI godot_variant_get_validated_setter(godot_variant_type p_type, const godot_string_name *p_member);
godot_validated_setter GDAPI godot_variant_get_validated_setter_with_cstring(godot_variant_type p_type, const char *p_member);
godot_validated_getter GDAPI godot_variant_get_validated_getter(godot_variant_type p_type, const godot_string_name *p_member);
godot_validated_getter GDAPI godot_variant_get_validated_getter_with_cstring(godot_variant_type p_type, const char *p_member);
godot_ptr_setter GDAPI godot_variant_get_ptr_setter(godot_variant_type p_type, const godot_string_name *p_member);
godot_ptr_setter GDAPI godot_variant_get_ptr_setter_with_cstring(godot_variant_type p_type, const char *p_member);
godot_ptr_getter GDAPI godot_variant_get_ptr_getter(godot_variant_type p_type, const godot_string_name *p_member);
godot_ptr_getter GDAPI godot_variant_get_ptr_getter_with_cstring(godot_variant_type p_type, const char *p_member);

/// Indexing.
bool GDAPI godot_variant_has_indexing(godot_variant_type p_type);
godot_variant_type GDAPI godot_variant_get_indexed_element_type(godot_variant_type p_type);
godot_validated_indexed_setter GDAPI godot_variant_get_validated_indexed_setter(godot_variant_type p_type);
godot_validated_indexed_getter GDAPI godot_variant_get_validated_indexed_getter(godot_variant_type p_type);
godot_ptr_indexed_setter GDAPI godot_variant_get_ptr_indexed_setter(godot_variant_type p_type);
godot_ptr_indexed_getter GDAPI godot_variant_get_ptr_indexed_getter(godot_variant_type p_type);
uint64_t GDAPI godot_variant_get_indexed_size(const godot_variant *p_self);

/// Keying.
bool GDAPI godot_variant_is_keyed(godot_variant_type p_type);
godot_validated_keyed_setter GDAPI godot_variant_get_validated_keyed_setter(godot_variant_type p_type);
godot_validated_keyed_getter GDAPI godot_variant_get_validated_keyed_getter(godot_variant_type p_type);
godot_validated_keyed_checker GDAPI godot_variant_get_validated_keyed_checker(godot_variant_type p_type);
godot_ptr_keyed_setter GDAPI godot_variant_get_ptr_keyed_setter(godot_variant_type p_type);
godot_ptr_keyed_getter GDAPI godot_variant_get_ptr_keyed_getter(godot_variant_type p_type);
godot_ptr_keyed_checker GDAPI godot_variant_get_ptr_keyed_checker(godot_variant_type p_type);

/// Constants.
int GDAPI godot_variant_get_constants_count(godot_variant_type p_type);
void GDAPI godot_variant_get_constants_list(godot_variant_type p_type, godot_string_name *r_list);
bool GDAPI godot_variant_has_constant(godot_variant_type p_type, const godot_string_name *p_constant);
bool GDAPI godot_variant_has_constant_with_cstring(godot_variant_type p_type, const char *p_constant);
godot_variant GDAPI godot_variant_get_constant_value(godot_variant_type p_type, const godot_string_name *p_constant);
godot_variant GDAPI godot_variant_get_constant_value_with_cstring(godot_variant_type p_type, const char *p_constant);

/// Utilities.
bool GDAPI godot_variant_has_utility_function(const godot_string_name *p_function);
bool GDAPI godot_variant_has_utility_function_with_cstring(const char *p_function);
void GDAPI godot_variant_call_utility_function(const godot_string_name *p_function, godot_variant *r_ret, const godot_variant **p_args, int p_argument_count, godot_variant_call_error *r_error);
void GDAPI godot_variant_call_utility_function_with_cstring(const char *p_function, godot_variant *r_ret, const godot_variant **p_args, int p_argument_count, godot_variant_call_error *r_error);
godot_ptr_utility_function GDAPI godot_variant_get_ptr_utility_function(const godot_string_name *p_function);
godot_ptr_utility_function GDAPI godot_variant_get_ptr_utility_function_with_cstring(const char *p_function);
godot_validated_utility_function GDAPI godot_variant_get_validated_utility_function(const godot_string_name *p_function);
godot_validated_utility_function GDAPI godot_variant_get_validated_utility_function_with_cstring(const char *p_function);
godot_variant_utility_function_type GDAPI godot_variant_get_utility_function_type(const godot_string_name *p_function);
godot_variant_utility_function_type GDAPI godot_variant_get_utility_function_type_with_cstring(const char *p_function);
int GDAPI godot_variant_get_utility_function_argument_count(const godot_string_name *p_function);
int GDAPI godot_variant_get_utility_function_argument_count_with_cstring(const char *p_function);
godot_variant_type GDAPI godot_variant_get_utility_function_argument_type(const godot_string_name *p_function, int p_argument);
godot_variant_type GDAPI godot_variant_get_utility_function_argument_type_with_cstring(const char *p_function, int p_argument);
godot_string GDAPI godot_variant_get_utility_function_argument_name(const godot_string_name *p_function, int p_argument);
godot_string GDAPI godot_variant_get_utility_function_argument_name_with_cstring(const char *p_function, int p_argument);
bool GDAPI godot_variant_has_utility_function_return_value(const godot_string_name *p_function);
bool GDAPI godot_variant_has_utility_function_return_value_with_cstring(const char *p_function);
godot_variant_type GDAPI godot_variant_get_utility_function_return_type(const godot_string_name *p_function);
godot_variant_type GDAPI godot_variant_get_utility_function_return_type_with_cstring(const char *p_function);
bool GDAPI godot_variant_is_utility_function_vararg(const godot_string_name *p_function);
bool GDAPI godot_variant_is_utility_function_vararg_with_cstring(const char *p_function);
int GDAPI godot_variant_get_utility_function_count();
void GDAPI godot_variant_get_utility_function_list(godot_string_name *r_functions);

// Introspection.

godot_variant_type GDAPI godot_variant_get_type(const godot_variant *p_self);
bool GDAPI godot_variant_has_method(const godot_variant *p_self, const godot_string_name *p_method);
bool GDAPI godot_variant_has_member(godot_variant_type p_type, const godot_string_name *p_member);
bool GDAPI godot_variant_has_key(const godot_variant *p_self, const godot_variant *p_key, bool *r_valid);

godot_string GDAPI godot_variant_get_type_name(godot_variant_type p_type);
bool GDAPI godot_variant_can_convert(godot_variant_type p_from, godot_variant_type p_to);
bool GDAPI godot_variant_can_convert_strict(godot_variant_type p_from, godot_variant_type p_to);

#ifdef __cplusplus
}
#endif

#endif
