/*************************************************************************/
/*  test_result.cpp                                                      */
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

#include "test_result.h"

#include "core/script_language.h"

TestResult::TestResult() :
		m_should_stop(false),
		m_tests_run(0) {
}

TestResult::~TestResult() {
	for (int i = 0; i < m_failures.size(); i++) {
		memfree(m_failures[i]);
	}
	for (int i = 0; i < m_errors.size(); i++) {
		memfree(m_errors[i]);
	}
}

void TestResult::start_test(TestState *p_test_state) {
	m_tests_run++;
	if (get_script_instance() && get_script_instance()->has_method("start_test")) {
		get_script_instance()->call("start_test", p_test_state);
	}
}

void TestResult::stop_test(TestState *p_test_state) {
	if (get_script_instance() && get_script_instance()->has_method("stop_test")) {
		get_script_instance()->call("stop_test", p_test_state);
	}
}

void TestResult::add_error(TestState *p_test_state, TestError *p_error) {
	m_errors.push_back(p_error);
	if (get_script_instance() && get_script_instance()->has_method("add_error")) {
		get_script_instance()->call("add_error", p_test_state, p_error);
	}
}

void TestResult::add_failure(TestState *p_test_state, TestError *p_error) {
	m_failures.push_back(p_error);
	if (get_script_instance() && get_script_instance()->has_method("add_failure")) {
		get_script_instance()->call("add_failure", p_test_state, p_error);
	}
}

void TestResult::add_success(TestState *p_test_state) {
	if (get_script_instance() && get_script_instance()->has_method("add_success")) {
		get_script_instance()->call("add_success", p_test_state);
	}
}

void TestResult::finish() {
}

bool TestResult::was_successful() const {
	return m_failures.size() == 0 && m_errors.size() == 0;
}

void TestResult::stop() {
	m_should_stop = true;
}

bool TestResult::should_stop() const {
	return m_should_stop;
}

void TestResult::_bind_methods() {
	ClassDB::bind_method(D_METHOD("start_test", "test_state"), &TestResult::_start_test);
	ClassDB::bind_method(D_METHOD("stop_test", "test_state"), &TestResult::_stop_test);
	ClassDB::bind_method(D_METHOD("add_error", "test_state", "error"), &TestResult::_add_error);
	ClassDB::bind_method(D_METHOD("add_failure", "test_state", "error"), &TestResult::_add_failure);
	ClassDB::bind_method(D_METHOD("add_success", "test_state"), &TestResult::_add_success);
	ClassDB::bind_method(D_METHOD("was_successful"), &TestResult::was_successful);
	ClassDB::bind_method(D_METHOD("stop"), &TestResult::stop);
	ClassDB::bind_method(D_METHOD("should_stop"), &TestResult::should_stop);
}
