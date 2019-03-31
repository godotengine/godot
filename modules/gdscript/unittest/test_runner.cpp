/*************************************************************************/
/*  test_runner.cpp                                                      */
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

#include "test_runner.h"
#include "test_config.h"
#include "test_loader.h"

void TestRunner::init() {
	SceneTree::init();
	m_test_result = Ref<TestResult>(TestConfig::get_singleton()->make_result());
	Ref<TestLoader> loader(memnew(TestLoader));
	Ref<TestSuite> test_suite(memnew(TestSuite));
	if (loader->from_path(test_suite, TestConfig::get_singleton()->test_directory())) {
		m_test_suite = test_suite;
		m_test_suite->init(get_root(), m_test_result);
	}
}

bool TestRunner::iteration(float p_time) {
	bool finished = SceneTree::iteration(p_time);
	if (m_test_suite.is_valid()) {
		return m_test_suite->iteration(*m_test_result);
	}
	return true;
}

void TestRunner::finish() {
	if (m_test_result.is_valid()) {
		m_test_result->finish();
	}
	SceneTree::finish();
}

void TestRunner::_bind_methods() {
}
