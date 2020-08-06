/*************************************************************************/
/*  gdscript_test.cpp                                                    */
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

#include "gdscript_test.h"

#include "../gdscript.h"
#include "../gdscript_compiler.h"
#include "../gdscript_parser.h"

#include "core/core_string_names.h"
#include "core/os/dir_access.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "core/string_builder.h"

GDScriptTestRunner::GDScriptTestRunner(const String &p_source_dir) {
	source_dir = p_source_dir;
	if (!source_dir.ends_with("/")) {
		source_dir += "/";
	}

	// Enable all warnings for GDScript, so we can test them.
	ProjectSettings::get_singleton()->set_setting("debug/gdscript/warnings/enable", true);
	for (int i = 0; i < (int)GDScriptWarning::WARNING_MAX; i++) {
		String warning = GDScriptWarning::get_name_from_code((GDScriptWarning::Code)i).to_lower();
		ProjectSettings::get_singleton()->set_setting("debug/gdscript/warnings/" + warning, true);
	}

	// Enable printing to show results
	_print_line_enabled = true;
	_print_error_enabled = true;
}

int GDScriptTestRunner::run_tests() {
	if (!make_tests()) {
		print_line("An error occured while making the tests.");
		return -1;
	}

	if (!generate_class_index()) {
		return -2;
	}

	int failed = 0;
	for (int i = 0; i < tests.size(); i++) {
		OS::get_singleton()->print(".");
		GDScriptTest test = tests[i];
		GDScriptTest::TestResult result = test.run_test();

		if (!result.passed) {
			failed++;
			print_line("\nTest failed: " + test.get_source_file());
		}
	}
	OS::get_singleton()->print("\n");

	if (failed == 0) {
		print_line("All " + itos(tests.size()) + " tests passed.");
	} else {
		print_line(itos(failed) + " test(s) from " + itos(tests.size()) + " failed.");
	}

	return failed;
}

bool GDScriptTestRunner::generate_outputs() {
	is_generating = true;

	if (!make_tests()) {
		print_line("Failed to generate a test output.");
		return false;
	}

	if (!generate_class_index()) {
		return false;
	}

	for (int i = 0; i < tests.size(); i++) {
		OS::get_singleton()->print(".");
		GDScriptTest test = tests[i];
		bool result = test.generate_output();

		if (!result) {
			print_line("\nCould not generate output for " + test.get_source_file());
			return false;
		}
	}
	print_line("\nGenerated output files for " + itos(tests.size()) + " tests successfully.");

	return true;
}

bool GDScriptTestRunner::make_tests_for_dir(const String &p_dir) {
	Error err = OK;
	DirAccessRef dir(DirAccess::open(p_dir, &err));

	if (err != OK) {
		return false;
	}

	String current_dir = dir->get_current_dir();

	dir->list_dir_begin();
	String next = dir->get_next();

	while (!next.empty()) {
		if (dir->current_is_dir()) {
			if (next == "." || next == "..") {
				next = dir->get_next();
				continue;
			}
			if (!make_tests_for_dir(current_dir.plus_file(next))) {
				return false;
			}
		} else {
			if (next.get_extension().to_lower() == "gd") {
				String out_file = next.get_basename() + ".out";
				if (!is_generating && !dir->file_exists(out_file)) {
					ERR_FAIL_V_MSG(false, "Could not find output file for " + next);
				}
				GDScriptTest test(current_dir.plus_file(next), current_dir.plus_file(out_file), source_dir);
				tests.push_back(test);
			}
		}

		next = dir->get_next();
	}

	dir->list_dir_end();

	return true;
}

bool GDScriptTestRunner::make_tests() {
	Error err = OK;
	DirAccessRef dir(DirAccess::open(source_dir, &err));

	ERR_FAIL_COND_V_MSG(err != OK, false, "Could not open specified test directory.");

	return make_tests_for_dir(dir->get_current_dir());
}

bool GDScriptTestRunner::generate_class_index() {
	StringName gdscript_name = GDScriptLanguage::get_singleton()->get_name();
	for (int i = 0; i < tests.size(); i++) {
		GDScriptTest test = tests[i];
		String base_type;

		String class_name = GDScriptLanguage::get_singleton()->get_global_class_name(test.get_source_file(), &base_type);
		if (class_name == String()) {
			continue;
		}
		ERR_FAIL_COND_V_MSG(ScriptServer::is_global_class(class_name), false,
				"Class name '" + class_name + "' from " + test.get_source_file() + " is already used in " + ScriptServer::get_global_class_path(class_name));

		ScriptServer::add_global_class(class_name, base_type, gdscript_name, test.get_source_file());
	}
	return true;
}

GDScriptTest::GDScriptTest(const String &p_source_path, const String &p_output_path, const String &p_base_dir) {
	source_file = p_source_path;
	output_file = p_output_path;
	base_dir = p_base_dir;
	_print_handler.printfunc = print_handler;
	_error_handler.errfunc = error_handler;

	if (test_function_name == StringName()) {
		test_function_name = StaticCString::create("test");
	}
}

void GDScriptTestRunner::handle_cmdline() {
	List<String> cmdline_args = OS::get_singleton()->get_cmdline_args();
	String test_cmd = "--test-gdscript";
	String gen_cmd = "--generate-gdscript-tests";

	for (List<String>::Element *E = cmdline_args.front(); E != nullptr; E = E->next()) {
		String &cmd = E->get();
		if (cmd == test_cmd || cmd == gen_cmd) {
			if (E->next() == nullptr) {
				ERR_PRINT("Needed a path for the test files.");
				exit(-1);
			}

			const String &path = E->next()->get();

			GDScriptTestRunner runner(path);
			int failed = 0;
			if (cmd == test_cmd) {
				failed = runner.run_tests();
			} else {
				bool completed = runner.generate_outputs();
				failed = completed ? 0 : -1;
			}
			exit(failed);
		}
	}
}

StringName GDScriptTest::test_function_name = StringName();

void GDScriptTest::enable_stdout() {
	// _stdout_enabled = true;
	// _stderr_enabled = true;
}

void GDScriptTest::disable_stdout() {
	// _stdout_enabled = false;
	// _stderr_enabled = false;
}

void GDScriptTest::print_handler(void *p_this, const String &p_message, bool p_error) {
	TestResult *result = (TestResult *)p_this;
	result->output += p_message + "\n";
}

void GDScriptTest::error_handler(void *p_this, const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_explanation, ErrorHandlerType p_type) {
	ErrorHandlerData *data = (ErrorHandlerData *)p_this;
	GDScriptTest *self = data->self;
	TestResult *result = data->result;

	result->status = GDTEST_RUNTIME_ERROR;

	StringBuilder builder;
	builder.append(">> ");
	switch (p_type) {
		case ERR_HANDLER_ERROR:
			builder.append("ERROR");
			break;
		case ERR_HANDLER_WARNING:
			builder.append("WARNING");
			break;
		case ERR_HANDLER_SCRIPT:
			builder.append("SCRIPT ERROR");
			break;
		case ERR_HANDLER_SHADER:
			builder.append("SHADER ERROR");
			break;
		default:
			builder.append("Unknown error type");
			break;
	}

	builder.append("\n>> ");
	builder.append(p_function);
	builder.append("\n>> ");
	builder.append(p_function);
	builder.append("\n>> ");
	builder.append(String(p_file).trim_prefix(self->base_dir));
	builder.append("\n>> ");
	builder.append(itos(p_line));
	builder.append("\n>> ");
	builder.append(p_error);
	if (strlen(p_explanation) > 0) {
		builder.append("\n>> ");
		builder.append(p_explanation);
	}
	builder.append("\n");

	result->output = builder.as_string();
}

bool GDScriptTest::check_output(const String &p_output) const {
	Error err = OK;
	String expected = FileAccess::get_file_as_string(output_file, &err);

	ERR_FAIL_COND_V_MSG(err != OK, false, "Error when opening the output file.");

	return expected == p_output;
}

String GDScriptTest::get_text_for_status(GDScriptTest::TestStatus p_status) const {
	switch (p_status) {
		case GDTEST_OK:
			return "GDTEST_OK";
		case GDTEST_LOAD_ERROR:
			return "GDTEST_LOAD_ERROR";
		case GDTEST_PARSER_ERROR:
			return "GDTEST_PARSER_ERROR";
		case GDTEST_COMPILER_ERROR:
			return "GDTEST_COMPILER_ERROR";
		case GDTEST_RUNTIME_ERROR:
			return "GDTEST_RUNTIME_ERROR";
	}
	return "";
}

GDScriptTest::TestResult GDScriptTest::execute_test_code(bool p_is_generating) {
	disable_stdout();

	TestResult result;
	result.status = GDTEST_OK;
	result.output = String();

	Error err = OK;

	// Create script.
	Ref<GDScript> script;
	script.instance();
	script->set_path(source_file);
	script->set_script_path(source_file);
	err = script->load_source_code(source_file);
	if (err != OK) {
		enable_stdout();
		result.status = GDTEST_LOAD_ERROR;
		result.passed = false;
		ERR_FAIL_V_MSG(result, "\nCould not load source code for: '" + source_file + "'");
	}

	// Test parsing.
	GDScriptParser parser;
	err = parser.parse(script->get_source_code(), source_file, false);
	if (err != OK) {
		enable_stdout();
		result.status = GDTEST_PARSER_ERROR;
		result.output = get_text_for_status(result.status) + "\n";

		const List<GDScriptParser::ParserError> &errors = parser.get_errors();
		for (auto *E = errors.front(); E; E = E->next()) {
			result.output += E->get().message + "\n"; // TODO: line, column?
		}
		if (!p_is_generating) {
			result.passed = check_output(result.output);
		}
		return result;
	}

	StringBuilder warning_string;
	for (const List<GDScriptWarning>::Element *E = parser.get_warnings().front(); E != nullptr; E = E->next()) {
		const GDScriptWarning warning = E->get();
		warning_string.append(">> WARNING");
		warning_string.append("\n>> Line: ");
		warning_string.append(itos(warning.start_line));
		warning_string.append("\n>> ");
		warning_string.append(warning.get_name());
		warning_string.append("\n>> ");
		warning_string.append(warning.get_message());
		warning_string.append("\n");
	}
	result.output += warning_string.as_string();

	// Test compiling.
	GDScriptCompiler compiler;
	err = compiler.compile(&parser, script.ptr(), false);
	if (err != OK) {
		enable_stdout();
		result.status = GDTEST_COMPILER_ERROR;
		result.output = get_text_for_status(result.status) + "\n";
		result.output = compiler.get_error();
		if (!p_is_generating) {
			result.passed = check_output(result.output);
		}
		return result;
	}

	// Test running.
	const Map<StringName, GDScriptFunction *>::Element *test_function_element = script->get_member_functions().find(test_function_name);
	if (test_function_element == nullptr) {
		enable_stdout();
		result.status = GDTEST_LOAD_ERROR;
		result.output = "";
		result.passed = false;
		ERR_FAIL_V_MSG(result, "\nCould not find test function on: '" + source_file + "'");
	}

	script->reload();

	// Create object instance for test.
	Object *obj = ClassDB::instance(script->get_native()->get_name());
	obj->set_script(script);
	GDScriptInstance *instance = static_cast<GDScriptInstance *>(obj->get_script_instance());

	// Setup output handlers.
	ErrorHandlerData error_data(&result, this);

	_print_handler.userdata = &result;
	_error_handler.userdata = &error_data;
	add_print_handler(&_print_handler);
	add_error_handler(&_error_handler);

	// Call test function.
	Callable::CallError call_err;
	instance->call(test_function_name, nullptr, 0, call_err);

	// Tear down output handlers.
	remove_print_handler(&_print_handler);
	remove_error_handler(&_error_handler);

	// Check results.
	if (call_err.error != Callable::CallError::CALL_OK) {
		enable_stdout();
		result.status = GDTEST_LOAD_ERROR;
		result.passed = false;
		ERR_FAIL_V_MSG(result, "\nCould not call test function on: '" + source_file + "'");
	}

	result.output = get_text_for_status(result.status) + "\n" + result.output;
	if (!p_is_generating) {
		result.passed = check_output(result.output);
	}

	enable_stdout();
	return result;
}

GDScriptTest::TestResult GDScriptTest::run_test() {
	return execute_test_code(false);
}

bool GDScriptTest::generate_output() {
	TestResult result = execute_test_code(true);
	if (result.status == GDTEST_LOAD_ERROR) {
		return false;
	}

	Error err = OK;
	FileAccessRef out_file = FileAccess::open(output_file, FileAccess::WRITE, &err);
	if (err != OK) {
		return false;
	}

	out_file->store_string(result.output);
	out_file->close();

	return true;
}
