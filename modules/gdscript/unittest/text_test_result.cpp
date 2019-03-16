/*************************************************************************/
/*  text_test_result.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "text_test_result.h"
#include "test_config.h"

#include "core/io/json.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/script_language.h"

TextTestResult::TextTestResult() {
	m_log.instance();
	m_total_assert_count = 0;
}

void TextTestResult::start_test(TestState *test_state) {
	TestResult::start_test(test_state);
	test_state->log()->info(test_state->test_name(), test_state->method_name(), "setup");
}

void TextTestResult::add_error(TestState *test_state, TestError *error) {
	TestResult::add_error(test_state, error);
	test_state->log()->fatal(test_state->test_name(), test_state->method_name(), error->m_message);
}

void TextTestResult::add_failure(TestState *test_state, TestError *error) {
	TestResult::add_failure(test_state, error);
	test_state->log()->fatal(test_state->test_name(), test_state->method_name(), error->m_message);
}

void TextTestResult::add_success(TestState *test_state) {
	TestResult::add_success(test_state);
}

void TextTestResult::stop_test(TestState *test_state) {
	TestResult::stop_test(test_state);
	TestConfig *singleton = TestConfig::get_singleton();
	if (!test_state->is_valid() || singleton->log_on_success()) {
		m_log->append(test_state->log());
		m_log->info(test_state->test_name(), test_state->method_name(), "teardown");
		Array args;
		args.resize(1);
		args[0] = test_state->assert_count();
		m_log->info(test_state->test_name(), test_state->method_name(), String("Ran {0} asserts.").format(args));
	}
}

void TextTestResult::finish() {
	DirAccessRef dir = DirAccess::open("res://");
	dir->make_dir("Testing");
	FileAccessRef file = FileAccess::open("Testing/results.json", FileAccess::WRITE);
	file->store_string(JSON::print(m_log->to_array(), "\t"));
}

void TextTestResult::summary(const String &msg) {
	Ref<Script> script = get_script_instance()->get_script();
	String name;
	if (script.is_valid()) {
		name = script->get_path();
	} else {
		name = get_class_name();
	}
	m_log->info(name, "summary", msg);
}

void TextTestResult::_bind_methods() {
}
