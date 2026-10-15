/**************************************************************************/
/*  test_angelscript_engine.h                                             */
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

// 本文件只负责注册用例：它会经 modules/modules_tests.gen.h 被 tests/test_main.cpp 展开，
// 而 test_main.cpp 用的是根 env（没有 AngelScript 的 include 路径），所以这里不能 include
// 任何 AS 头。真正的断言放在同目录的 test_angelscript_engine.cpp：它由模块 SCsub 编进
// module_angelscript.lib，再被链接器按需拉入（与 modules/mbedtls/tests 的做法一致）。

void as_engine_lazy_init_is_idempotent();
void as_engine_compiles_and_executes(const String &p_source, String *r_error, int *r_result);
bool as_engine_compile_module(const String &p_name, const String &p_source, String *r_error);

// 本头会被两处 include：tests/test_main.cpp（用于注册用例）与同目录的
// test_angelscript_engine.cpp（只为拿到上面的辅助函数声明）。实现方必须先定义
// ANGELSCRIPT_ENGINE_TESTS_IMPL，否则 TEST_CASE 会在两个 TU 里各注册一次、
// 用例被跑两遍（上游 modules/mbedtls 目前就是这样重复注册的）。
#ifndef ANGELSCRIPT_ENGINE_TESTS_IMPL

TEST_CASE("[AngelScript] engine lazily initializes without ScriptServer::init_languages()") {
	as_engine_lazy_init_is_idempotent();
}

TEST_CASE("[AngelScript] engine compiles and executes a script") {
	String error;
	int result = 0;
	as_engine_compiles_and_executes("int main() { return 42; }", &error, &result);
	CHECK(result == 42);
}

TEST_CASE("[AngelScript] compile_module reports diagnostics on syntax error") {
	String error;
	CHECK_FALSE(as_engine_compile_module("as_engine_bad", "int main( { return 42; }", &error));
	CHECK_FALSE(error.is_empty());
	// 编译失败后必须仍能编译有效脚本；此断言同时暴露二进制布局对 AS 计时的影响（见三语言报告）。
	CHECK(as_engine_compile_module("as_engine_good_after_bad", "int main() { return 42; }", &error));
}

#endif // ANGELSCRIPT_ENGINE_TESTS_IMPL
