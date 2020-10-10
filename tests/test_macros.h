/*************************************************************************/
/*  test_macros.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef TEST_MACROS_H
#define TEST_MACROS_H

#include "core/map.h"
#include "core/variant.h"

// See documentation for doctest at:
// https://github.com/onqtam/doctest/blob/master/doc/markdown/readme.md#reference
#include "thirdparty/doctest/doctest.h"

// The test is skipped with this, run pending tests with `--test --no-skip`.
#define TEST_CASE_PENDING(name) TEST_CASE(name *doctest::skip())

// Temporarily disable error prints to test failure paths.
// This allows to avoid polluting the test summary with error messages.
// The `_print_error_enabled` boolean is defined in `core/print_string.cpp` and
// works at global scope. It's used by various loggers in `should_log()` method,
// which are used by error macros which call into `OS::print_error`, effectively
// disabling any error messages to be printed from the engine side (not tests).
#define ERR_PRINT_OFF _print_error_enabled = false;
#define ERR_PRINT_ON _print_error_enabled = true;

// Stringify all `Variant` compatible types for doctest output by default.
// https://github.com/onqtam/doctest/blob/master/doc/markdown/stringification.md

#define DOCTEST_STRINGIFY_VARIANT(m_type)                        \
	template <>                                                  \
	struct doctest::StringMaker<m_type> {                        \
		static doctest::String convert(const m_type &p_val) {    \
			const Variant val = p_val;                           \
			return val.get_construct_string().utf8().get_data(); \
		}                                                        \
	};

#define DOCTEST_STRINGIFY_VARIANT_POINTER(m_type)                \
	template <>                                                  \
	struct doctest::StringMaker<m_type> {                        \
		static doctest::String convert(const m_type *p_val) {    \
			const Variant val = p_val;                           \
			return val.get_construct_string().utf8().get_data(); \
		}                                                        \
	};

DOCTEST_STRINGIFY_VARIANT(Variant);
DOCTEST_STRINGIFY_VARIANT(::String); // Disambiguate from `doctest::String`.

DOCTEST_STRINGIFY_VARIANT(Vector2);
DOCTEST_STRINGIFY_VARIANT(Vector2i);
DOCTEST_STRINGIFY_VARIANT(Rect2);
DOCTEST_STRINGIFY_VARIANT(Rect2i);
DOCTEST_STRINGIFY_VARIANT(Vector3);
DOCTEST_STRINGIFY_VARIANT(Vector3i);
DOCTEST_STRINGIFY_VARIANT(Transform2D);
DOCTEST_STRINGIFY_VARIANT(Plane);
DOCTEST_STRINGIFY_VARIANT(Quat);
DOCTEST_STRINGIFY_VARIANT(AABB);
DOCTEST_STRINGIFY_VARIANT(Basis);
DOCTEST_STRINGIFY_VARIANT(Transform);

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

// Register test commands to be launched from the command-line.
// For instance: REGISTER_TEST_COMMAND("gdscript-parser" &test_parser_func).
// Example usage: `godot --test gdscript-parser`.

typedef void (*TestFunc)();
extern Map<String, TestFunc> *test_commands;
int register_test_command(String p_command, TestFunc p_function);

#define REGISTER_TEST_COMMAND(m_command, m_function)                    \
	DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_ANONYMOUS(_DOCTEST_ANON_VAR_)) = \
			register_test_command(m_command, m_function);               \
	DOCTEST_GLOBAL_NO_WARNINGS_END()

#endif // TEST_MACROS_H
