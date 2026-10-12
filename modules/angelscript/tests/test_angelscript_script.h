/**************************************************************************/
/*  test_angelscript_script.h                                             */
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
// 任何 AS 头，用例体也不能直接触碰 ASScript。断言全部放在同目录的
// test_angelscript_script.cpp 里：它由模块 SCsub 编进 module_angelscript.lib，
// 再被链接器按需拉入（与 test_angelscript_engine.cpp 同一模式）。

void as_script_parses_base_type_and_class_name();
void as_script_derives_class_name_from_path();
void as_script_tolerates_utf8_bom_and_crlf();
void as_script_rejects_missing_base_directive();
void as_script_rejects_unknown_base_type();
void as_script_ignores_base_directive_inside_block_comment();
void as_script_rejects_directive_only_inside_block_comment();
void as_script_rejects_class_name_that_does_not_match_file_name();
void as_script_rejects_syntax_errors();
void as_script_rejects_empty_and_whitespace_source();
void as_script_rejects_directive_after_line_10();
void as_script_clear_resets_source_code();
void as_script_loads_through_resource_loader_from_user_path();
void as_script_loader_ignores_cache_without_cyclic_conflict();
void as_script_reload_recompiles_from_updated_source();
void as_script_hot_reload_reattaches_and_refreshes();

// 本头会被两处 include：tests/test_main.cpp（用于注册用例）与同目录的
// test_angelscript_script.cpp（只为拿到上面的辅助函数声明）。实现方必须先定义
// ANGELSCRIPT_SCRIPT_TESTS_IMPL，否则 TEST_CASE 会在两个 TU 里各注册一次、
// 用例被跑两遍（上游 modules/mbedtls 目前就是这样重复注册的）。
#ifndef ANGELSCRIPT_SCRIPT_TESTS_IMPL

TEST_CASE("[AngelScript] ASScript parses base type and class name") {
	as_script_parses_base_type_and_class_name();
}

TEST_CASE("[AngelScript] ASScript derives class name from path") {
	as_script_derives_class_name_from_path();
}

TEST_CASE("[AngelScript] ASScript tolerates UTF-8 BOM and CRLF") {
	as_script_tolerates_utf8_bom_and_crlf();
}

TEST_CASE("[AngelScript] ASScript rejects missing base directive") {
	as_script_rejects_missing_base_directive();
}

TEST_CASE("[AngelScript] ASScript rejects unknown base type") {
	as_script_rejects_unknown_base_type();
}

TEST_CASE("[AngelScript] ASScript ignores base directive inside block comment") {
	as_script_ignores_base_directive_inside_block_comment();
}

TEST_CASE("[AngelScript] ASScript rejects base directive only inside block comment") {
	as_script_rejects_directive_only_inside_block_comment();
}

TEST_CASE("[AngelScript] ASScript rejects class name that does not match file name") {
	as_script_rejects_class_name_that_does_not_match_file_name();
}

TEST_CASE("[AngelScript] ASScript rejects syntax errors and stays invalid") {
	as_script_rejects_syntax_errors();
}

TEST_CASE("[AngelScript] ASScript rejects empty and whitespace-only source") {
	as_script_rejects_empty_and_whitespace_source();
}

TEST_CASE("[AngelScript] ASScript only scans the first 10 lines for the base directive") {
	as_script_rejects_directive_after_line_10();
}

TEST_CASE("[AngelScript] ASScript clear() resets source code") {
	as_script_clear_resets_source_code();
}

TEST_CASE("[AngelScript] ASScript loads through ResourceLoader from user://") {
	as_script_loads_through_resource_loader_from_user_path();
}

TEST_CASE("[AngelScript] ASScript CACHE_MODE_IGNORE reload ignores existing cache entry") {
	as_script_loader_ignores_cache_without_cyclic_conflict();
}

TEST_CASE("[AngelScript] ASScript reload() recompiles from updated source") {
	as_script_reload_recompiles_from_updated_source();
}

TEST_CASE("[AngelScript] ASScript hot reload reattaches instances and restores state") {
	as_script_hot_reload_reattaches_and_refreshes();
}

#endif // ANGELSCRIPT_SCRIPT_TESTS_IMPL
