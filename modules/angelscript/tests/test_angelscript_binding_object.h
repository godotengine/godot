/**************************************************************************/
/*  test_angelscript_binding_object.h                                     */
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

// 用例注册头：不得 include 任何 AngelScript 头；实现见 test_angelscript_binding_object.cpp。

void as_binding_object_register_all();
void as_binding_object_method_call();
void as_binding_object_property_access();
void as_binding_object_upcast();
void as_binding_object_refcount_lifetime();
void as_binding_object_nonowning_assign();
void as_binding_object_discarded_return();
void as_binding_object_enums_and_constants();
void as_binding_object_native_hot_method();

#ifndef ANGELSCRIPT_BINDING_OBJECT_TESTS_IMPL

TEST_CASE("[AngelScript] binding object register class and enum") {
	as_binding_object_register_all();
}
TEST_CASE("[AngelScript] binding object method call") {
	as_binding_object_method_call();
}
TEST_CASE("[AngelScript] binding object property access") {
	as_binding_object_property_access();
}
TEST_CASE("[AngelScript] binding object upcast") {
	as_binding_object_upcast();
}
TEST_CASE("[AngelScript] binding object refcount lifetime") {
	as_binding_object_refcount_lifetime();
}
TEST_CASE("[AngelScript] binding object nonowning assign") {
	as_binding_object_nonowning_assign();
}
TEST_CASE("[AngelScript] binding object discarded return") {
	as_binding_object_discarded_return();
}
TEST_CASE("[AngelScript] binding object enums and constants") {
	as_binding_object_enums_and_constants();
}
TEST_CASE("[AngelScript] binding object native hot method") {
	as_binding_object_native_hot_method();
}

#endif // ANGELSCRIPT_BINDING_OBJECT_TESTS_IMPL
