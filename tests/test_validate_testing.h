/*************************************************************************/
/*  test_validate_testing.h                                              */
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

#ifndef TEST_VALIDATE_TESTING_H
#define TEST_VALIDATE_TESTING_H

#include "core/os/os.h"

#include "tests/test_macros.h"

TEST_SUITE("Validate tests") {
	TEST_CASE("Always pass") {
		CHECK(true);
	}
	TEST_CASE_PENDING("Pending tests are skipped") {
		if (!doctest::getContextOptions()->no_skip) { // Normal run.
			FAIL("This should be skipped if `--no-skip` is NOT set (missing `doctest::skip()` decorator?)");
		} else {
			CHECK_MESSAGE(true, "Pending test is run with `--no-skip`");
		}
	}
	TEST_CASE("Muting Godot error messages") {
		ERR_PRINT_OFF;
		CHECK_MESSAGE(!_print_error_enabled, "Error printing should be disabled.");
		ERR_PRINT("Still waiting for Godot!"); // This should never get printed!
		ERR_PRINT_ON;
		CHECK_MESSAGE(_print_error_enabled, "Error printing should be re-enabled.");
	}
}

#endif // TEST_VALIDATE_TESTING_H
