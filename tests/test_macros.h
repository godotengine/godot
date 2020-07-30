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

#endif // TEST_MACROS_H
