/*************************************************************************/
/*  gdscript_test_runner.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GDSCRIPT_TEST_H
#define GDSCRIPT_TEST_H

#include "../gdscript.h"
#include "core/error/error_macros.h"
#include "core/string/print_string.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"

namespace GDScriptTests {

void init_autoloads();
void init_language(const String &p_base_path);
void finish_language();

// Single test instance in a suite.
class GDScriptTest {
public:
	enum TestStatus {
		GDTEST_OK,
		GDTEST_LOAD_ERROR,
		GDTEST_PARSER_ERROR,
		GDTEST_ANALYZER_ERROR,
		GDTEST_COMPILER_ERROR,
		GDTEST_RUNTIME_ERROR,
	};

	struct TestResult {
		TestStatus status;
		String output;
		bool passed;
	};

private:
	struct ErrorHandlerData {
		TestResult *result;
		GDScriptTest *self;
		ErrorHandlerData(TestResult *p_result, GDScriptTest *p_this) {
			result = p_result;
			self = p_this;
		}
	};

	String source_file;
	String output_file;
	String base_dir;

	PrintHandlerList _print_handler;
	ErrorHandlerList _error_handler;

	void enable_stdout();
	void disable_stdout();
	bool check_output(const String &p_output) const;
	String get_text_for_status(TestStatus p_status) const;

	TestResult execute_test_code(bool p_is_generating);

public:
	static void print_handler(void *p_this, const String &p_message, bool p_error);
	static void error_handler(void *p_this, const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_explanation, bool p_editor_notify, ErrorHandlerType p_type);
	TestResult run_test();
	bool generate_output();

	const String &get_source_file() const { return source_file; }
	const String &get_output_file() const { return output_file; }

	GDScriptTest(const String &p_source_path, const String &p_output_path, const String &p_base_dir);
	GDScriptTest() :
			GDScriptTest(String(), String(), String()) {} // Needed to use in Vector.
};

class GDScriptTestRunner {
	String source_dir;
	Vector<GDScriptTest> tests;

	bool is_generating = false;
	bool do_init_languages = false;

	bool make_tests();
	bool make_tests_for_dir(const String &p_dir);
	bool generate_class_index();

public:
	static StringName test_function_name;

	static void handle_cmdline();
	int run_tests();
	bool generate_outputs();

	GDScriptTestRunner(const String &p_source_dir, bool p_init_language);
	~GDScriptTestRunner();
};

} // namespace GDScriptTests

#endif // GDSCRIPT_TEST_H
