/**************************************************************************/
/*  test_macros.h                                                         */
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

#include "core/core_globals.h" // IWYU pragma: Used in macro.
#include "core/variant/variant.h"

#if defined(_MSC_VER) && !defined(DOCTEST_THREAD_LOCAL)
// NOTE: We must disable the THREAD_LOCAL entirely in doctest to prevent crashes on debugging.
//  Since we link with /MT, thread_local is always expired when the header is used, so the
//  debugger crashes the engine and it causes weird errors.
// See: https://github.com/onqtam/doctest/issues/401
#define DOCTEST_THREAD_LOCAL
#endif

#if !__cpp_exceptions && !__EXCEPTIONS && !defined(DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS)
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#endif

// See documentation for doctest at:
// https://github.com/onqtam/doctest/blob/master/doc/markdown/readme.md#reference
#include "thirdparty/doctest/doctest.h"

// Forces a test file to be linked.
#define TEST_FORCE_LINK(m_name) \
	namespace ForceLink { \
	void force_link_##m_name() {} \
	}

// The test is skipped with this, run pending tests with `--test --no-skip`.
#define TEST_CASE_PENDING(name) TEST_CASE(name *doctest::skip())

// The test case is marked as failed, but does not fail the entire test run.
#define TEST_CASE_MAY_FAIL(name) TEST_CASE(name *doctest::may_fail())

// Provide aliases to conform with Godot naming conventions (see error macros).
#define TEST_COND(cond, ...) DOCTEST_CHECK_FALSE_MESSAGE(cond, __VA_ARGS__)
#define TEST_FAIL(cond, ...) DOCTEST_FAIL(cond, __VA_ARGS__)
#define TEST_FAIL_COND(cond, ...) DOCTEST_REQUIRE_FALSE_MESSAGE(cond, __VA_ARGS__)
#define TEST_FAIL_COND_WARN(cond, ...) DOCTEST_WARN_FALSE_MESSAGE(cond, __VA_ARGS__)

// Temporarily disable error prints to test failure paths.
// This allows to avoid polluting the test summary with error messages.
// The `print_error_enabled` boolean is defined in `core/core_globals.cpp` and
// works at global scope. It's used by various loggers in `should_log()` method,
// which are used by error macros which call into `OS::print_error`, effectively
// disabling any error messages to be printed from the engine side (not tests).
#define ERR_PRINT_OFF CoreGlobals::print_error_enabled = false;
#define ERR_PRINT_ON CoreGlobals::print_error_enabled = true;

// Stringify all `Variant` compatible types for doctest output by default.
// https://github.com/onqtam/doctest/blob/master/doc/markdown/stringification.md

#define DOCTEST_STRINGIFY_VARIANT(m_type) \
	template <> \
	struct doctest::StringMaker<m_type> { \
		static doctest::String convert(const m_type &p_val) { \
			const Variant val = p_val; \
			return val.operator ::String().utf8().get_data(); \
		} \
	};

#define DOCTEST_STRINGIFY_VARIANT_POINTER(m_type) \
	template <> \
	struct doctest::StringMaker<m_type> { \
		static doctest::String convert(const m_type *p_val) { \
			const Variant val = p_val; \
			return val.operator ::String().utf8().get_data(); \
		} \
	};

DOCTEST_STRINGIFY_VARIANT(Variant);
DOCTEST_STRINGIFY_VARIANT(::String); // Disambiguate from `doctest::String`.

DOCTEST_STRINGIFY_VARIANT(Vector2);
DOCTEST_STRINGIFY_VARIANT(Vector2i);
DOCTEST_STRINGIFY_VARIANT(Rect2);
DOCTEST_STRINGIFY_VARIANT(Rect2i);
DOCTEST_STRINGIFY_VARIANT(Vector3);
DOCTEST_STRINGIFY_VARIANT(Vector3i);
DOCTEST_STRINGIFY_VARIANT(Vector4);
DOCTEST_STRINGIFY_VARIANT(Vector4i);
DOCTEST_STRINGIFY_VARIANT(Transform2D);
DOCTEST_STRINGIFY_VARIANT(Plane);
DOCTEST_STRINGIFY_VARIANT(Projection);
DOCTEST_STRINGIFY_VARIANT(Quaternion);
DOCTEST_STRINGIFY_VARIANT(AABB);
DOCTEST_STRINGIFY_VARIANT(Basis);
DOCTEST_STRINGIFY_VARIANT(Transform3D);

DOCTEST_STRINGIFY_VARIANT(::Color); // Disambiguate from `doctest::Color`.
DOCTEST_STRINGIFY_VARIANT(StringName);
DOCTEST_STRINGIFY_VARIANT(NodePath);
DOCTEST_STRINGIFY_VARIANT(RID);
DOCTEST_STRINGIFY_VARIANT_POINTER(Object);
DOCTEST_STRINGIFY_VARIANT(Callable);
DOCTEST_STRINGIFY_VARIANT(Signal);
DOCTEST_STRINGIFY_VARIANT(Dictionary);
DOCTEST_STRINGIFY_VARIANT(Array);

DOCTEST_STRINGIFY_VARIANT(PackedByteArray);
DOCTEST_STRINGIFY_VARIANT(PackedInt32Array);
DOCTEST_STRINGIFY_VARIANT(PackedInt64Array);
DOCTEST_STRINGIFY_VARIANT(PackedFloat32Array);
DOCTEST_STRINGIFY_VARIANT(PackedFloat64Array);
DOCTEST_STRINGIFY_VARIANT(PackedStringArray);
DOCTEST_STRINGIFY_VARIANT(PackedVector2Array);
DOCTEST_STRINGIFY_VARIANT(PackedVector3Array);
DOCTEST_STRINGIFY_VARIANT(PackedColorArray);
DOCTEST_STRINGIFY_VARIANT(PackedVector4Array);

// Register test commands to be launched from the command-line.
// For instance: REGISTER_TEST_COMMAND("gdscript-parser" &test_parser_func).
// Example usage: `godot --test gdscript-parser`.

typedef void (*TestFunc)();
extern HashMap<String, TestFunc> *test_commands;
int register_test_command(String p_command, TestFunc p_function);

#define REGISTER_TEST_COMMAND(m_command, m_function) \
	DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_ANONYMOUS(DOCTEST_ANON_VAR_), \
			register_test_command(m_command, m_function))

#define MULTICHECK_STRING_EQ(m_obj, m_func, m_param1, m_eq) \
	CHECK(m_obj.m_func(m_param1) == m_eq); \
	CHECK(m_obj.m_func(U##m_param1) == m_eq); \
	CHECK(m_obj.m_func(L##m_param1) == m_eq); \
	CHECK(m_obj.m_func(String(m_param1)) == m_eq);

#define MULTICHECK_STRING_INT_EQ(m_obj, m_func, m_param1, m_param2, m_eq) \
	CHECK(m_obj.m_func(m_param1, m_param2) == m_eq); \
	CHECK(m_obj.m_func(U##m_param1, m_param2) == m_eq); \
	CHECK(m_obj.m_func(L##m_param1, m_param2) == m_eq); \
	CHECK(m_obj.m_func(String(m_param1), m_param2) == m_eq);

#define MULTICHECK_STRING_INT_INT_EQ(m_obj, m_func, m_param1, m_param2, m_param3, m_eq) \
	CHECK(m_obj.m_func(m_param1, m_param2, m_param3) == m_eq); \
	CHECK(m_obj.m_func(U##m_param1, m_param2, m_param3) == m_eq); \
	CHECK(m_obj.m_func(L##m_param1, m_param2, m_param3) == m_eq); \
	CHECK(m_obj.m_func(String(m_param1), m_param2, m_param3) == m_eq);

#define MULTICHECK_STRING_STRING_EQ(m_obj, m_func, m_param1, m_param2, m_eq) \
	CHECK(m_obj.m_func(m_param1, m_param2) == m_eq); \
	CHECK(m_obj.m_func(U##m_param1, U##m_param2) == m_eq); \
	CHECK(m_obj.m_func(L##m_param1, L##m_param2) == m_eq); \
	CHECK(m_obj.m_func(String(m_param1), String(m_param2)) == m_eq);

#define MULTICHECK_GET_SLICE(m_obj, m_param1, m_slices) \
	for (int i = 0; i < m_obj.get_slice_count(m_param1); ++i) { \
		CHECK(m_obj.get_slice(m_param1, i) == m_slices[i]); \
	} \
	for (int i = 0; i < m_obj.get_slice_count(U##m_param1); ++i) { \
		CHECK(m_obj.get_slice(U##m_param1, i) == m_slices[i]); \
	} \
	for (int i = 0; i < m_obj.get_slice_count(L##m_param1); ++i) { \
		CHECK(m_obj.get_slice(L##m_param1, i) == m_slices[i]); \
	} \
	for (int i = 0; i < m_obj.get_slice_count(String(m_param1)); ++i) { \
		CHECK(m_obj.get_slice(String(m_param1), i) == m_slices[i]); \
	}

#define MULTICHECK_SPLIT(m_obj, m_func, m_param1, m_param2, m_param3, m_slices, m_expected_size) \
	do { \
		Vector<String> string_list; \
\
		string_list = m_obj.m_func(m_param1, m_param2, m_param3); \
		CHECK(m_expected_size == string_list.size()); \
		for (int i = 0; i < string_list.size(); ++i) { \
			CHECK(string_list[i] == m_slices[i]); \
		} \
\
		string_list = m_obj.m_func(U##m_param1, m_param2, m_param3); \
		CHECK(m_expected_size == string_list.size()); \
		for (int i = 0; i < string_list.size(); ++i) { \
			CHECK(string_list[i] == m_slices[i]); \
		} \
\
		string_list = m_obj.m_func(L##m_param1, m_param2, m_param3); \
		CHECK(m_expected_size == string_list.size()); \
		for (int i = 0; i < string_list.size(); ++i) { \
			CHECK(string_list[i] == m_slices[i]); \
		} \
\
		string_list = m_obj.m_func(String(m_param1), m_param2, m_param3); \
		CHECK(m_expected_size == string_list.size()); \
		for (int i = 0; i < string_list.size(); ++i) { \
			CHECK(string_list[i] == m_slices[i]); \
		} \
	} while (false)
