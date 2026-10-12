/**************************************************************************/
/*  test_angelscript_binding_plan.h                                       */
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

// 用例注册头：不得 include 任何 AngelScript 头；实现见 test_angelscript_binding_plan.cpp。

void as_binding_plan_default_scope_is_ordered();
void as_binding_plan_collects_unbound_vararg();
void as_binding_plan_binds_static_methods();
void as_binding_plan_binds_variant_abi();
void as_binding_plan_whitelist_keeps_ancestors();
void as_binding_plan_blacklist_wins();
void as_binding_plan_accessor_name_taken();

#ifndef ANGELSCRIPT_BINDING_PLAN_TESTS_IMPL

TEST_CASE("[AngelScript] binding plan default scope is ordered") {
	as_binding_plan_default_scope_is_ordered();
}
TEST_CASE("[AngelScript] binding plan collects unbound vararg methods") {
	as_binding_plan_collects_unbound_vararg();
}
TEST_CASE("[AngelScript] binding plan binds static methods as global functions") {
	as_binding_plan_binds_static_methods();
}
TEST_CASE("[AngelScript] binding plan marshals Variant parameters and returns") {
	as_binding_plan_binds_variant_abi();
}
TEST_CASE("[AngelScript] binding plan whitelist keeps ancestors") {
	as_binding_plan_whitelist_keeps_ancestors();
}
TEST_CASE("[AngelScript] binding plan blacklist wins over whitelist") {
	as_binding_plan_blacklist_wins();
}
TEST_CASE("[AngelScript] binding plan records name-taken property accessors") {
	as_binding_plan_accessor_name_taken();
}

#endif // ANGELSCRIPT_BINDING_PLAN_TESTS_IMPL
