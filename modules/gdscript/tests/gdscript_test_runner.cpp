/*************************************************************************/
/*  gdscript_test_runner.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gdscript_test_runner.h"

#include "../gdscript.h"
#include "../gdscript_analyzer.h"
#include "../gdscript_compiler.h"
#include "../gdscript_parser.h"

#include "core/config/project_settings.h"
#include "core/core_string_names.h"
#include "core/io/dir_access.h"
#include "core/io/file_access_pack.h"
#include "core/os/os.h"
#include "core/string/string_builder.h"
#include "scene/resources/packed_scene.h"

#include "tests/test_macros.h"

namespace GDScriptTests {

void init_autoloads() {
	OrderedHashMap<StringName, ProjectSettings::AutoloadInfo> autoloads = ProjectSettings::get_singleton()->get_autoload_list();

	// First pass, add the constants so they exist before any script is loaded.
	for (OrderedHashMap<StringName, ProjectSettings::AutoloadInfo>::Element E = autoloads.front(); E; E = E.next()) {
		const ProjectSettings::AutoloadInfo &info = E.get();

		if (info.is_singleton) {
			for (int i = 0; i < ScriptServer::get_language_count(); i++) {
				ScriptServer::get_language(i)->add_global_constant(info.name, Variant());
			}
		}
	}

	// Second pass, load into global constants.
	for (OrderedHashMap<StringName, ProjectSettings::AutoloadInfo>::Element E = autoloads.front(); E; E = E.next()) {
		const ProjectSettings::AutoloadInfo &info = E.get();

		if (!info.is_singleton) {
			// Skip non-singletons since we don't have a scene tree here anyway.
			continue;
		}

		RES res = ResourceLoader::load(info.path);
		ERR_CONTINUE_MSG(res.is_null(), "Can't autoload: " + info.path);
		Node *n = nullptr;
		if (res->is_class("PackedScene")) {
			Ref<PackedScene> ps = res;
			n = ps->instantiate();
		} else if (res->is_class("Script")) {
			Ref<Script> script_res = res;
			StringName ibt = script_res->get_instance_base_type();
			bool valid_type = ClassDB::is_parent_class(ibt, "Node");
			ERR_CONTINUE_MSG(!valid_type, "Script does not inherit a Node: " + info.path);

			Object *obj = ClassDB::instantiate(ibt);

			ERR_CONTINUE_MSG(obj == nullptr,
					"Cannot instance script for autoload, expected 'Node' inheritance, got: " +
							String(ibt));

			n = Object::cast_to<Node>(obj);
			n->set_script(script_res);
		}

		ERR_CONTINUE_MSG(!n, "Path in autoload not a node or script: " + info.path);
		n->set_name(info.name);

		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptServer::get_language(i)->add_global_constant(info.name, n);
		}
	}
}

void init_language(const String &p_base_path) {
	// Setup project settings since it's needed by the languages to get the global scripts.
	// This also sets up the base resource path.
	Error err = ProjectSettings::get_singleton()->setup(p_base_path, String(), true);
	if (err) {
		print_line("Could not load project settings.");
		// Keep going since some scripts still work without this.
	}

	// Initialize the language for the test routine.
	GDScriptLanguage::get_singleton()->init();
	init_autoloads();
}

void finish_language() {
	GDScriptLanguage::get_singleton()->finish();
	ScriptServer::global_classes_clear();
}

StringName GDScriptTestRunner::test_function_name;

GDScriptTestRunner::GDScriptTestRunner(const String &p_source_dir, bool p_init_language) {
	test_function_name = StaticCString::create("test");
	do_init_languages = p_init_language;

	source_dir = p_source_dir;
	if (!source_dir.ends_with("/")) {
		source_dir += "/";
	}

	if (do_init_languages) {
		init_language(p_source_dir);
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

GDScriptTestRunner::~GDScriptTestRunner() {
	test_function_name = StringName();
	if (do_init_languages) {
		finish_language();
	}
}

int GDScriptTestRunner::run_tests() {
	if (!make_tests()) {
		FAIL("An error occurred while making the tests.");
		return -1;
	}

	if (!generate_class_index()) {
		FAIL("An error occurred while generating class index.");
		return -1;
	}

	int failed = 0;
	for (int i = 0; i < tests.size(); i++) {
		GDScriptTest test = tests[i];
		GDScriptTest::TestResult result = test.run_test();

		String expected = FileAccess::get_file_as_string(test.get_output_file());
		INFO(test.get_source_file());
		if (!result.passed) {
			INFO(expected);
			failed++;
		}

		CHECK_MESSAGE(result.passed, (result.passed ? String() : result.output));
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

	while (!next.is_empty()) {
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

	source_dir = dir->get_current_dir() + "/"; // Make it absolute path.
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
}

void GDScriptTestRunner::handle_cmdline() {
	List<String> cmdline_args = OS::get_singleton()->get_cmdline_args();
	// TODO: this could likely be ported to use test commands:
	// https://github.com/godotengine/godot/pull/41355
	// Currently requires to startup the whole engine, which is slow.
	String test_cmd = "--gdscript-test";
	String gen_cmd = "--gdscript-generate-tests";

	for (List<String>::Element *E = cmdline_args.front(); E; E = E->next()) {
		String &cmd = E->get();
		if (cmd == test_cmd || cmd == gen_cmd) {
			if (E->next() == nullptr) {
				ERR_PRINT("Needed a path for the test files.");
				exit(-1);
			}

			const String &path = E->next()->get();

			GDScriptTestRunner runner(path, false);
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

void GDScriptTest::enable_stdout() {
	// TODO: this could likely be handled by doctest or `tests/test_macros.h`.
	OS::get_singleton()->set_stdout_enabled(true);
	OS::get_singleton()->set_stderr_enabled(true);
}

void GDScriptTest::disable_stdout() {
	// TODO: this could likely be handled by doctest or `tests/test_macros.h`.
	OS::get_singleton()->set_stdout_enabled(false);
	OS::get_singleton()->set_stderr_enabled(false);
}

void GDScriptTest::print_handler(void *p_this, const String &p_message, bool p_error) {
	TestResult *result = (TestResult *)p_this;
	result->output += p_message + "\n";
}

void GDScriptTest::error_handler(void *p_this, const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_explanation, bool p_editor_notify, ErrorHandlerType p_type) {
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

	builder.append("\n>> on function: ");
	builder.append(p_function);
	builder.append("()\n>> ");
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

	String got = p_output.strip_edges(); // TODO: may be hacky.
	got += "\n"; // Make sure to insert newline for CI static checks.

	return got == expected;
}

String GDScriptTest::get_text_for_status(GDScriptTest::TestStatus p_status) const {
	switch (p_status) {
		case GDTEST_OK:
			return "GDTEST_OK";
		case GDTEST_LOAD_ERROR:
			return "GDTEST_LOAD_ERROR";
		case GDTEST_PARSER_ERROR:
			return "GDTEST_PARSER_ERROR";
		case GDTEST_ANALYZER_ERROR:
			return "GDTEST_ANALYZER_ERROR";
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
	result.passed = false;

	Error err = OK;

	// Create script.
	Ref<GDScript> script;
	script.instantiate();
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
		for (const GDScriptParser::ParserError &E : errors) {
			result.output += E.message + "\n"; // TODO: line, column?
			break; // Only the first error since the following might be cascading.
		}
		if (!p_is_generating) {
			result.passed = check_output(result.output);
		}
		return result;
	}

	// Test type-checking.
	GDScriptAnalyzer analyzer(&parser);
	err = analyzer.analyze();
	if (err != OK) {
		enable_stdout();
		result.status = GDTEST_ANALYZER_ERROR;
		result.output = get_text_for_status(result.status) + "\n";

		const List<GDScriptParser::ParserError> &errors = parser.get_errors();
		for (const GDScriptParser::ParserError &E : errors) {
			result.output += E.message + "\n"; // TODO: line, column?
			break; // Only the first error since the following might be cascading.
		}
		if (!p_is_generating) {
			result.passed = check_output(result.output);
		}
		return result;
	}

	StringBuilder warning_string;
	for (const GDScriptWarning &E : parser.get_warnings()) {
		const GDScriptWarning warning = E;
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
	// Script files matching this pattern are allowed to not contain a test() function.
	if (source_file.match("*.notest.gd")) {
		enable_stdout();
		result.passed = check_output(result.output);
		return result;
	}
	// Test running.
	const Map<StringName, GDScriptFunction *>::Element *test_function_element = script->get_member_functions().find(GDScriptTestRunner::test_function_name);
	if (test_function_element == nullptr) {
		enable_stdout();
		result.status = GDTEST_LOAD_ERROR;
		result.output = "";
		result.passed = false;
		ERR_FAIL_V_MSG(result, "\nCould not find test function on: '" + source_file + "'");
	}

	script->reload();

	// Create object instance for test.
	Object *obj = ClassDB::instantiate(script->get_native()->get_name());
	Ref<RefCounted> obj_ref;
	if (obj->is_ref_counted()) {
		obj_ref = Ref<RefCounted>(Object::cast_to<RefCounted>(obj));
	}
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
	instance->call(GDScriptTestRunner::test_function_name, nullptr, 0, call_err);

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

	if (obj_ref.is_null()) {
		memdelete(obj);
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

	String output = result.output.strip_edges(); // TODO: may be hacky.
	output += "\n"; // Make sure to insert newline for CI static checks.

	out_file->store_string(output);
	out_file->close();

	return true;
}

} // namespace GDScriptTests
