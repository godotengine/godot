/*************************************************************************/
/*  test_suite.cpp                                                       */
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

#include "test_suite.h"

TestSuite::TestSuite() {
	m_case_index = 0;
}

TestSuite::~TestSuite() {
	int count = count_test_cases();
	for (int i = 0; i < count; i++) {
		memfree(m_test_cases[i]);
	}
}

int TestSuite::count_test_cases() const {
	return m_test_cases.size();
}

void TestSuite::add_test(TestCase *p_test_case) {
	ERR_FAIL_COND(NULL == p_test_case);
	m_test_cases.push_back(p_test_case);
}

void TestSuite::add_tests(Array p_test_cases) {
	int count = p_test_cases.size();
	for (int i = 0; i < count; i++) {
		add_test(cast_to<TestCase>(p_test_cases[i]));
	}
}

void TestSuite::init(Viewport *root, Ref<TestResult> p_test_result) {
	m_root = root;
	m_case_index = 0;
	if (m_case_index < count_test_cases()) {
		m_root->add_child(m_test_cases[m_case_index]);
		m_test_cases[m_case_index]->init(p_test_result);
	}
}

bool TestSuite::iteration(Ref<TestResult> p_test_result) {
	if (m_case_index < count_test_cases() && !p_test_result->should_stop()) {
		TestCase *test_case = m_test_cases[m_case_index];
		bool finished = test_case->iteration(p_test_result);
		if (finished) {
			m_root->remove_child(m_test_cases[m_case_index]);
			m_case_index++;
			if (m_case_index < count_test_cases()) {
				m_root->add_child(m_test_cases[m_case_index]);
				m_test_cases[m_case_index]->init(p_test_result);
			}
		}
		return p_test_result->should_stop();
	}
	return true;
}

void TestSuite::_bind_methods() {
	ADD_SIGNAL(MethodInfo("timeout"));
}
