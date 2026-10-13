/**************************************************************************/
/*  test_angelscript_instance.h                                           */
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

// 与 test_angelscript_script.h 同一模式：本头只注册用例（由 tests/test_main.cpp 经
// modules/modules_tests.gen.h 展开，根 env 没有 AngelScript / 场景树的 include 路径），
// 断言与辅助实现全部放在 test_angelscript_instance.cpp。

void as_instance_ready_is_dispatched_by_name();
void as_instance_notification_reaches_notification();
void as_instance_process_receives_delta();
void as_instance_signature_mismatch_is_not_exposed();
void as_instance_requires_default_constructor();
void as_instance_scalar_properties_round_trip();
void as_instance_repeated_dispatch_is_cached();
void as_instance_scalar_call_plan_is_cached();
void as_instance_call_plan_safety();
void as_instance_retained_context_probe();

#ifndef ANGELSCRIPT_INSTANCE_TESTS_IMPL

TEST_CASE("[AngelScript] instance is created and _ready is dispatched by name") {
	as_instance_ready_is_dispatched_by_name();
}

TEST_CASE("[AngelScript] notification(int) reaches _notification") {
	as_instance_notification_reaches_notification();
}

TEST_CASE("[AngelScript] _process receives delta as double") {
	as_instance_process_receives_delta();
}

TEST_CASE("[AngelScript] callback with mismatched signature is not exposed") {
	as_instance_signature_mismatch_is_not_exposed();
}

TEST_CASE("[AngelScript] class without a default constructor cannot be instantiated") {
	as_instance_requires_default_constructor();
}

TEST_CASE("[AngelScript] scalar properties round-trip through the instance") {
	as_instance_scalar_properties_round_trip();
}

TEST_CASE("[AngelScript] repeated dispatch resolves the method name only once") {
	as_instance_repeated_dispatch_is_cached();
}
TEST_CASE("[AngelScript] scalar dispatch analyzes its call plan only once") {
	as_instance_scalar_call_plan_is_cached();
}

TEST_CASE("[AngelScript] call plan rejects refs and survives reentry") {
	as_instance_call_plan_safety();
}
TEST_CASE("[AngelScript] retained context isolated lifetime probe") {
	as_instance_retained_context_probe();
}

#endif // ANGELSCRIPT_INSTANCE_TESTS_IMPL
