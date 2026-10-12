/**************************************************************************/
/*  test_angelscript_binding_dump.h                                       */
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

// 用例注册头：不得 include 任何 AngelScript 头；实现见 test_angelscript_binding_dump.cpp。

void as_binding_dump_files_and_determinism();
void as_binding_dump_content_matches_runtime();
void as_binding_dump_accessor_name_taken();
void as_binding_dump_static_methods_are_global();

#ifndef ANGELSCRIPT_BINDING_DUMP_TESTS_IMPL

TEST_CASE("[AngelScript] binding dump files and determinism") {
	as_binding_dump_files_and_determinism();
}
TEST_CASE("[AngelScript] binding dump content matches runtime") {
	as_binding_dump_content_matches_runtime();
}
TEST_CASE("[AngelScript] binding dump reports name-taken property accessors") {
	as_binding_dump_accessor_name_taken();
}
TEST_CASE("[AngelScript] binding dump renders static methods as global functions") {
	as_binding_dump_static_methods_are_global();
}

#endif // ANGELSCRIPT_BINDING_DUMP_TESTS_IMPL
