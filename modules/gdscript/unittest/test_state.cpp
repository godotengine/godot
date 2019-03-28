/*************************************************************************/
/*  test_state.cpp                                                       */
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

#include "test_state.h"
#include "test_config.h"

#include "gdscript.h"

#include "core/script_language.h"

bool StageIter::init() {
	m_stage = SETUP;
	return true;
}

bool StageIter::next() {
	switch (m_stage) {
		case SETUP: {
			m_stage = TEST;
			return true;
		}
		case TEST: {
			m_stage = TEARDOWN;
			return true;
		}
		case TEARDOWN: {
			m_stage = DONE;
			return true;
		}
		case DONE: {
			return false;
		}
	}
	return false;
}

void StageIter::skip_test() {
	switch (m_stage) {
		case SETUP:
		case TEST:
			m_stage = TEARDOWN;
			break;
		case TEARDOWN:
			m_stage = DONE;
			break;
	}
}

StageIter::Stage StageIter::get() const {
	return m_stage;
}

bool MethodIter::next_test() {
	while (m_method_info) {
		const String &name = m_method_info->get().name;
		if (name.match(TestConfig::get_singleton()->test_func_match())) {
			return true;
		}
		m_method_info = m_method_info->next();
	}
	return false;
}

bool MethodIter::init(const Object *p_object) {
	p_object->get_method_list(&m_methods);
	m_method_info = m_methods.front();
	return next_test();
}

const String &MethodIter::get() const {
	return m_method_info->get().name;
}

bool MethodIter::next() {
	m_method_info = m_method_info->next();
	return next_test();
}

TestState::TestState() {
	m_log.instance();
	m_log->set_filter(TestLog::LogLevel::TRACE);
}

bool TestState::init(const Object *p_object) {
	m_test_name = p_object->get_script_instance()->get_script()->get_path();
	m_assert_count = 0;
	m_test_count = 0;
	m_log->clear();
	if (m_method_iter.init(p_object)) {
		m_test_count++;
		return m_stage_iter.init();
	}
	return false;
}

const String &TestState::method_name() const {
	return m_method_iter.get();
}

StageIter::Stage TestState::stage() const {
	return m_stage_iter.get();
}

Ref<TestLog> TestState::log() const {
	return m_log;
}

bool TestState::next() {
	if (!m_stage_iter.next()) {
		m_assert_count = 0;
		m_log->clear();
		if (m_method_iter.next()) {
			m_test_count++;
			m_stage_iter.init();
			return true;
		}
		return false;
	}
	return true;
}

void TestState::skip_test() {
	m_stage_iter.skip_test();
}

const String &TestState::test_name() const {
	return m_test_name;
}

int TestState::test_count() const {
	return m_test_count;
}

int TestState::assert_count() const {
	return m_assert_count;
}

bool TestState::is_valid() const {
	return m_log->get_max_level() >= TestConfig::get_singleton()->log_fail_greater_equal();
}

void TestState::assert() {
	m_assert_count++;
}

void TestState::_bind_methods() {
}
