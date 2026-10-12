/**************************************************************************/
/*  test_angelscript_binding_decl.h                                       */
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

// 本文件只负责注册用例（经 modules/modules_tests.gen.h 由 tests/test_main.cpp 展开），
// 因此不能 include 任何 AngelScript 头。断言放在同目录的 test_angelscript_binding_decl.cpp。

void as_binding_decl_scalar_mapping();
void as_binding_decl_object_and_enum();
void as_binding_decl_method_forms();
void as_binding_decl_static_method_forms();
void as_binding_decl_variant_abi();
void as_binding_decl_rejects_vararg_and_unknown();

#ifndef ANGELSCRIPT_BINDING_DECL_TESTS_IMPL

TEST_CASE("[AngelScript] binding decl maps Variant scalar types") {
	as_binding_decl_scalar_mapping();
}
TEST_CASE("[AngelScript] binding decl resolves object types and enums") {
	as_binding_decl_object_and_enum();
}
TEST_CASE("[AngelScript] binding decl renders method declarations") {
	as_binding_decl_method_forms();
}
TEST_CASE("[AngelScript] binding decl renders static methods as global functions") {
	as_binding_decl_static_method_forms();
}
TEST_CASE("[AngelScript] binding decl marshals Variant parameters and returns") {
	as_binding_decl_variant_abi();
}
TEST_CASE("[AngelScript] binding decl rejects vararg") {
	as_binding_decl_rejects_vararg_and_unknown();
}

#endif // ANGELSCRIPT_BINDING_DECL_TESTS_IMPL
