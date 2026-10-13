/**************************************************************************/
/*  test_angelscript_debugger.h                                           */
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

namespace TestAngelScriptDebugger {

void test_line_stepping();
void test_callback_activation();
void test_builds_stack_info();
void test_skips_caught_exception();
void test_reads_locals();
void test_decodes_bound_value_type();
void test_decodes_native_string();
void test_expands_script_class();
void test_decodes_object_handle();
void test_handles_invalid_level();
void test_reads_this_members();
void test_exception_flag_is_per_context();
void test_attaches_engine_internal_contexts();

#ifndef ANGELSCRIPT_DEBUGGER_TESTS_IMPL

TEST_CASE("[AngelScript] debugger callbacks follow execution activity") {
	test_callback_activation();
}

TEST_CASE("[AngelScript] debugger line breakpoints step next out and resume") {
	test_line_stepping();
}

TEST_CASE("[AngelScript] 调试器能在异常回调内构建调用栈") {
	test_builds_stack_info();
}

TEST_CASE("[AngelScript] 调试器不拦截脚本自己 catch 掉的异常") {
	test_skips_caught_exception();
}

TEST_CASE("[AngelScript] 调试器能读出当前帧的局部标量") {
	test_reads_locals();
}

TEST_CASE("[AngelScript] 调试器能解码绑定值类型局部变量") {
	test_decodes_bound_value_type();
}

TEST_CASE("[AngelScript] 调试器能解码 AS 原生 string 局部变量") {
	test_decodes_native_string();
}

TEST_CASE("[AngelScript] 调试器能递归展开脚本类并挡住环与深度") {
	test_expands_script_class();
}

TEST_CASE("[AngelScript] 调试器解码 Godot 对象句柄且容忍对象已销毁") {
	test_decodes_object_handle();
}

TEST_CASE("[AngelScript] 调试器对失效输入返回安全默认值") {
	test_handles_invalid_level();
}

TEST_CASE("[AngelScript] 调试器能读出类方法帧的 this 成员") {
	test_reads_this_members();
}

TEST_CASE("[AngelScript] 异常上报标志只对产生它的上下文生效") {
	test_exception_flag_is_per_context();
}

TEST_CASE("[AngelScript] AS 内部请求的上下文同样装上异常回调") {
	test_attaches_engine_internal_contexts();
}

#endif

} // namespace TestAngelScriptDebugger
