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
//==============================================================================
// See documentation for doctest at:
// https://github.com/onqtam/doctest/blob/master/doc/markdown/readme.md#reference
#include "thirdparty/doctest/doctest.h"
//==============================================================================

// Use `CHECK(condition)` to for simple assertions,
// and `CHECK_MSG(condition, message)` for assertions with descriptions.
#define CHECK_MSG DOCTEST_CHECK_MESSAGE
#define TEST_CHECK DOCTEST_CHECK
#define TEST_CHECK_MSG DOCTEST_CHECK_MESSAGE

// Use `REQUIRE(condition)` to quit a test immediately if condition is not met,
// and `REQUIRE_MSG(condition, message)` to quit a test by providing rationale.
#define REQUIRE_MSG DOCTEST_REQUIRE_MESSAGE
#define TEST_REQUIRE DOCTEST_REQUIRE
#define TEST_REQUIRE_MSG DOCTEST_REQUIRE_MESSAGE

// The following are like `ERR_FAIL_*` macros defined in `core/error_macros.h`.
// Quit the test case if any assert fails and mark the test case as failed.
// NOTE: these are the opposite of `REQUIRE`.
#define TEST_FAIL() DOCTEST_FAIL();
#define TEST_FAIL_MSG(m_msg) DOCTEST_FAIL(m_msg)
#define TEST_FAIL_COND(m_cond) DOCTEST_REQUIRE_FALSE(m_cond);
#define TEST_FAIL_COND_MSG(m_cond, m_msg) DOCTEST_REQUIRE_FALSE_MESSAGE(m_cond, m_msg);

//==============================================================================

// Use this if you need to test imprecise floating point values.
// Always prefer using `Math::is_equal_approx` defined for most `Variant` types
// unless you need more tolerant comparisons of floating point values.
#define CHECK_EQUAL_APPROX(m_got, m_expected, m_eps) \
	DOCTEST_CHECK(m_got == doctest::Approx(m_expected).epsilon(m_eps));

//==============================================================================

// The test case is skipped, run pending tests with `--test --no-skip`.
#define TEST_CASE_PENDING(m_name) DOCTEST_TEST_CASE(m_name *doctest::skip())

// The test case is allowed to fail without causing the entire test run to fail.
#define TEST_CASE_MAY_FAIL(m_name) DOCTEST_TEST_CASE(m_name *doctest::may_fail())

// The test is skipped and allowed to fail (a work-in-progress test).
#define TEST_CASE_WIP(m_name) \
	DOCTEST_TEST_CASE(m_name *doctest::skip() * doctest::may_fail())

//==============================================================================

// Temporarily disable error prints to test failure paths.
// This allows to avoid polluting the test summary with error messages.
// The `_print_error_enabled` boolean is defined in `core/print_string.cpp` and
// works at global scope. It's used by various loggers in `should_log()` method,
// which are used by error macros which call into `OS::print_error`, effectively
// disabling any error messages to be printed from the engine side (not tests).
#define ERR_PRINT_OFF _print_error_enabled = false;
#define ERR_PRINT_ON _print_error_enabled = true;

#endif // TEST_MACROS_H
