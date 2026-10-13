/**************************************************************************/
/*  test_angelscript_binding_value_types.h                                */
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

// 用例注册头：不得 include 任何 AngelScript 头；实现见 test_angelscript_binding_value_types.cpp。

void as_binding_value_types_register_all();
void as_binding_metadata_auxiliary();
void as_binding_value_types_vector2_roundtrip();
void as_binding_value_types_members_and_ops();
void as_binding_value_types_indexing();
void as_binding_value_types_variant_array();
void as_binding_value_types_variant_dictionary();
void as_binding_value_types_string_interop();
void as_binding_value_types_string_variable_forms();
void as_binding_value_types_builtin_variant_return();
void as_binding_value_types_static_builtin_methods();

#ifndef ANGELSCRIPT_BINDING_VALUE_TYPES_TESTS_IMPL

TEST_CASE("[AngelScript] binding metadata uses lock-free auxiliary") {
	as_binding_metadata_auxiliary();
}

TEST_CASE("[AngelScript] binding value types register all builtin types") {
	as_binding_value_types_register_all();
}
TEST_CASE("[AngelScript] binding value types Vector2 roundtrip") {
	as_binding_value_types_vector2_roundtrip();
}
TEST_CASE("[AngelScript] binding value types members and operators") {
	as_binding_value_types_members_and_ops();
}
TEST_CASE("[AngelScript] binding value types indexing") {
	as_binding_value_types_indexing();
}
TEST_CASE("[AngelScript] binding value types Variant in Array") {
	as_binding_value_types_variant_array();
}
TEST_CASE("[AngelScript] binding value types Variant in Dictionary") {
	as_binding_value_types_variant_dictionary();
}
TEST_CASE("[AngelScript] binding value types String and string interop") {
	as_binding_value_types_string_interop();
}
TEST_CASE("[AngelScript] binding value types string copy and assignment forms") {
	as_binding_value_types_string_variable_forms();
}
TEST_CASE("[AngelScript] binding value types builtin method returning Variant") {
	as_binding_value_types_builtin_variant_return();
}
TEST_CASE("[AngelScript] binding value types static builtin methods") {
	as_binding_value_types_static_builtin_methods();
}

#endif // ANGELSCRIPT_BINDING_VALUE_TYPES_TESTS_IMPL
