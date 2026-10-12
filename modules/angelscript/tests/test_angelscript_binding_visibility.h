/**************************************************************************/
/*  test_angelscript_binding_visibility.h                                 */
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

// 用例注册头：不得 include 任何 AngelScript 头；实现见 test_angelscript_binding_visibility.cpp。

void as_binding_visibility_project_setting_applies();
void as_binding_visibility_utility_function_clampf();
void as_binding_visibility_utility_variant_function();
void as_binding_visibility_static_method_global();
void as_binding_visibility_variant_abi();

#ifndef ANGELSCRIPT_BINDING_VISIBILITY_TESTS_IMPL

TEST_CASE("[AngelScript] binding visibility project setting applies") {
	as_binding_visibility_project_setting_applies();
}
TEST_CASE("[AngelScript] binding visibility utility function clampf") {
	as_binding_visibility_utility_function_clampf();
}
TEST_CASE("[AngelScript] binding visibility Variant utility function") {
	as_binding_visibility_utility_variant_function();
}
TEST_CASE("[AngelScript] binding visibility static method as global function") {
	as_binding_visibility_static_method_global();
}
TEST_CASE("[AngelScript] binding visibility marshals Variant parameters and returns") {
	as_binding_visibility_variant_abi();
}

#endif // ANGELSCRIPT_BINDING_VISIBILITY_TESTS_IMPL
