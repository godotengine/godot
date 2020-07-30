/*************************************************************************/
/*  test_reporters.h                                                     */
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

#ifndef TEST_REPORTERS_H
#define TEST_REPORTERS_H

#include "test_macros.h"

// NOTE: we need make sure to re-enable Godot error prints with `ERR_PRINT_ON`
// automatically in case of a human error or other unexpected failures.

struct GodotConsoleReporter : public doctest::ConsoleReporter {
	GodotConsoleReporter(const doctest::ContextOptions &co) :
			doctest::ConsoleReporter(co) {}

	void test_run_start() override {
		ERR_PRINT_ON;
		ConsoleReporter::test_run_start();
	}
	void test_case_start(const doctest::TestCaseData &p_in) override {
		ERR_PRINT_ON;
		ConsoleReporter::test_case_start(p_in);
	}
	void test_case_reenter(const doctest::TestCaseData &p_in) override {
		ERR_PRINT_ON;
		ConsoleReporter::test_case_reenter(p_in);
	}
	void test_case_exception(const doctest::TestCaseException &p_in) override {
		ERR_PRINT_ON;
		ConsoleReporter::test_case_exception(p_in);
	}
	void subcase_start(const doctest::SubcaseSignature &p_in) override {
		ERR_PRINT_ON;
		ConsoleReporter::subcase_start(p_in);
	}
};

REGISTER_REPORTER("godot_console", 1, GodotConsoleReporter);

#endif // TEST_REPORTERS_H
