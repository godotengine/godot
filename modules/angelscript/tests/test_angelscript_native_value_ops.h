/**************************************************************************/
/*  test_angelscript_native_value_ops.h                                   */
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

#include "tests/test_macros.h"

// 用例注册头：不得 include 任何 AngelScript 头；实现见 test_angelscript_native_value_ops.cpp。

void as_native_vo_scaffold_fallback_unchanged();
void as_native_vo_vector2_opadd();
void as_native_vo_vector2_arith_assign();
void as_native_vo_vector3_arith_assign();
void as_native_vo_integer_vectors();
void as_native_vo_op_equals();
void as_native_vo_method_core();
void as_native_vo_cross();
void as_native_vo_geometry();
void as_native_vo_vector2_methods_all();
void as_native_vo_vector3_methods_all();
void as_native_vo_integer_methods_all();
void as_native_vo_storage_slot_helpers();
void as_native_vo_native_storage_round_trip();
void as_native_vo_native_fast_path_active();
void as_native_vo_boundary_conversions();
void as_native_vo_packed_index_accessors();
void as_native_vo_packed_size();
void as_native_vo_packed_index_oob();
void as_native_vo_packed_push_back();
void as_native_vo_string_ops();
void as_native_vo_array_index();
void as_native_vo_array_refcount_payload();
void as_native_vo_dictionary_keyed_access();
void as_native_vo_container_methods();
void as_native_vo_copy_ctor_unambiguous();
void as_native_vo_extended_native_storage();
void as_native_vo_extended_types_script();
void as_native_vo_extended_types_ops_parity();
void as_native_vo_platform_gate();
void as_native_vo_generic_fallback_path();
void as_native_vo_variant_lifecycle_conversions();
void as_native_vo_packed_storage_parity();
void as_native_vo_container_constraints();
void as_native_vo_literal_direct_paths();
void as_native_vo_geometry_lifetime_probe();
void as_native_vo_array_mutator_constraints();

#ifndef ANGELSCRIPT_NATIVE_VALUE_OPS_TESTS_IMPL

TEST_CASE("[AngelScript] native value ops geometry lifetime probe") {
	as_native_vo_geometry_lifetime_probe();
}
TEST_CASE("[AngelScript] native value ops Array mutator constraints") {
	as_native_vo_array_mutator_constraints();
}

TEST_CASE("[AngelScript] native value ops literal direct paths") {
	as_native_vo_literal_direct_paths();
}

TEST_CASE("[AngelScript] native value ops Variant lifecycle conversions") {
	as_native_vo_variant_lifecycle_conversions();
}

TEST_CASE("[AngelScript] native value ops Packed storage parity") {
	as_native_vo_packed_storage_parity();
}

TEST_CASE("[AngelScript] native value ops container constraints") {
	as_native_vo_container_constraints();
}

TEST_CASE("[AngelScript] native value ops scaffold fallback") {
	as_native_vo_scaffold_fallback_unchanged();
}

TEST_CASE("[AngelScript] native value ops vector2 opAdd") {
	as_native_vo_vector2_opadd();
}

TEST_CASE("[AngelScript] native value ops vector2 arith assign") {
	as_native_vo_vector2_arith_assign();
}

TEST_CASE("[AngelScript] native value ops vector3 arith assign") {
	as_native_vo_vector3_arith_assign();
}

TEST_CASE("[AngelScript] native value ops integer vectors") {
	as_native_vo_integer_vectors();
}

TEST_CASE("[AngelScript] native value ops op equals") {
	as_native_vo_op_equals();
}

TEST_CASE("[AngelScript] native value ops method core") {
	as_native_vo_method_core();
}

TEST_CASE("[AngelScript] native value ops cross") {
	as_native_vo_cross();
}

TEST_CASE("[AngelScript] native value ops geometry") {
	as_native_vo_geometry();
}

TEST_CASE("[AngelScript] native value ops vector2 methods all") {
	as_native_vo_vector2_methods_all();
}

TEST_CASE("[AngelScript] native value ops vector3 methods all") {
	as_native_vo_vector3_methods_all();
}

TEST_CASE("[AngelScript] native value ops integer methods all") {
	as_native_vo_integer_methods_all();
}

TEST_CASE("[AngelScript] native value ops native storage slot helpers") {
	as_native_vo_storage_slot_helpers();
}

TEST_CASE("[AngelScript] native value ops native storage round trip") {
	as_native_vo_native_storage_round_trip();
}

TEST_CASE("[AngelScript] native value ops native fast path active") {
	as_native_vo_native_fast_path_active();
}

TEST_CASE("[AngelScript] native value ops boundary conversions") {
	as_native_vo_boundary_conversions();
}

TEST_CASE("[AngelScript] native value ops packed index accessors") {
	as_native_vo_packed_index_accessors();
}

TEST_CASE("[AngelScript] native value ops packed size") {
	as_native_vo_packed_size();
}

TEST_CASE("[AngelScript] native value ops packed index oob") {
	as_native_vo_packed_index_oob();
}

TEST_CASE("[AngelScript] native value ops packed push back") {
	as_native_vo_packed_push_back();
}

TEST_CASE("[AngelScript] native value ops string ops") {
	as_native_vo_string_ops();
}

TEST_CASE("[AngelScript] native value ops array index") {
	as_native_vo_array_index();
}

TEST_CASE("[AngelScript] native value ops array refcount payload") {
	as_native_vo_array_refcount_payload();
}

TEST_CASE("[AngelScript] native value ops dictionary keyed access") {
	as_native_vo_dictionary_keyed_access();
}

TEST_CASE("[AngelScript] native value ops container methods") {
	as_native_vo_container_methods();
}

TEST_CASE("[AngelScript] native value ops copy ctor unambiguous") {
	as_native_vo_copy_ctor_unambiguous();
}

TEST_CASE("[AngelScript] native value ops extended native storage") {
	as_native_vo_extended_native_storage();
}

TEST_CASE("[AngelScript] native value ops extended types script") {
	as_native_vo_extended_types_script();
}

TEST_CASE("[AngelScript] native value ops extended types ops parity") {
	as_native_vo_extended_types_ops_parity();
}

TEST_CASE("[AngelScript] native value ops platform gate") {
	as_native_vo_platform_gate();
}

TEST_CASE("[AngelScript] native value ops generic fallback path") {
	as_native_vo_generic_fallback_path();
}

#endif // ANGELSCRIPT_NATIVE_VALUE_OPS_TESTS_IMPL
