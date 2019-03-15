/*************************************************************************/
/*  test_case.cpp                                                        */
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

#include "test_case.h"
#include "test_config.h"

#include "core/math/math_funcs.h"
#include "core/script_language.h"

class AssertGuard {
public:
	AssertGuard(bool *can_assert) :
			m_can_assert(can_assert) {
		*m_can_assert = true;
	}

	virtual ~AssertGuard() {
		*m_can_assert = false;
	}

private:
	bool *m_can_assert;
};

void assert(Dictionary &input, const String &custom_msg, const String &default_msg) {
	String msg = default_msg;
	if (!custom_msg.empty()) {
		msg = custom_msg;
	} else {
		input["msg"] = "Assertion : ";
	}
	throw Failure{ msg.format(input) };
}

void assert(const Variant &x, const String &custom_msg, const String &default_msg) {
	Dictionary input;
	input["x"] = x;
	assert(input, custom_msg, default_msg);
}

void assert(const Variant &a, const Variant &b, const String &custom_msg, const String &default_msg) {
	Dictionary input;
	input["a"] = a;
	input["b"] = b;
	assert(input, custom_msg, default_msg);
}

TestCase::TestCase() {
	m_state = memnew(TestState);
}

TestCase::~TestCase() {
	memfree(m_state);
}

void TestCase::setup() {
	if (get_script_instance() && get_script_instance()->has_method("setup")) {
		m_yield = get_script_instance()->call("setup");
	}
}

void TestCase::teardown() {
	if (get_script_instance() && get_script_instance()->has_method("teardown")) {
		m_yield = get_script_instance()->call("teardown");
	}
}

void make_result(Ref<TestResult> &test_result) {
	if (test_result.is_null()) {
		test_result = Ref<TestResult>(TestConfig::get_singleton()->make_result());
	}
}

void TestCase::init(Ref<TestResult> test_result) {
	make_result(test_result);
	m_has_next = m_state->init(this);
}

bool TestCase::iteration(Ref<TestResult> test_result) {
	make_result(test_result);
	if (m_yield.is_valid()) {
		if (m_yield_handled) {
			m_yield = m_yield->resume();
		}
		return false;
	}
	if (m_has_next) {
		switch (m_state->stage()) {
			case StageIter::SETUP: {
				test_result->start_test(m_state);
				setup();
				break;
			}
			case StageIter::TEST: {
				try {
					AssertGuard guard(&m_can_assert);
					m_yield = get_script_instance()->call(m_state->method_name());
					test_result->add_success(m_state);
				} catch (const Failure &failure) {
					TestError *error = memnew(TestError);
					test_result->add_failure(m_state, error);
				}
				break;
			}
			case StageIter::TEARDOWN: {
				teardown();
				break;
			}
		}
		m_has_next = m_state->next();
	}
	if (!m_has_next) {
		test_result->stop_test(m_state);
	}
	return !m_has_next;
}

bool TestCase::is_done() const {
	return StageIter::DONE == m_state->stage();
}

void TestCase::assert_equal(const Variant &a, const Variant &b, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
	if (a != b) {
		assert(a, b, "{msg}: {a} != {b}", msg);
	}
}

void TestCase::assert_not_equal(const Variant &a, const Variant &b, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
	if (a != b) {
		assert(a, b, "{msg}: {a} == {b}", msg);
	}
}

void TestCase::assert_true(const Variant &x, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
	if (!x) {
	}
}

void TestCase::assert_false(const Variant &x, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
	if (x) {
	}
}

void TestCase::assert_is(const Variant &a, const Ref<GDScriptNativeClass> b, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
	Object *obj = cast_to<Object>(a);
	const String &klass = obj->get_class();
	const String &inherits = obj->get_class();
	if (!ClassDB::is_parent_class(klass, inherits)) {
		assert(klass, inherits, msg, "{a} is not {b}");
	}
}

void TestCase::assert_is_not(const Variant &a, const Ref<GDScriptNativeClass> b, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
	Object *obj = cast_to<Object>(a);
	const String &klass = obj->get_class();
	const String &inherits = obj->get_class();
	if (!ClassDB::is_parent_class(klass, inherits)) {
		assert(klass, inherits, msg, "{a} is {b}");
	}
}

void TestCase::assert_is_nil(const Variant &x, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
	if (Variant::NIL != x.get_type()) {
	}
}

void TestCase::assert_is_not_nil(const Variant &x, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
	if (Variant::NIL == x.get_type()) {
	}
}

void TestCase::assert_in(const Variant &a, const Variant &b, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
}

void TestCase::assert_not_in(const Variant &a, const Variant &b, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
}

void TestCase::assert_is_type(const Variant &a, const Variant::Type type, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
	if (a.get_type() != type) {
	}
}

void TestCase::assert_is_not_type(const Variant &a, const Variant::Type type, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
	if (a.get_type() == type) {
	}
}

void TestCase::assert_almost_equal(const Variant &a, const Variant &b, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
	if (!Math::is_equal_approx(a, b)) {
	}
}

void TestCase::assert_not_almost_equal(const Variant &a, const Variant &b, const String &msg) {
	ERR_FAIL_COND(!m_can_assert);
	if (Math::is_equal_approx(a, b)) {
	}
}

void TestCase::_clear_connections() {
	List<Connection> connections;
	this->get_signals_connected_to_this(&connections);
	int size = connections.size();
	for (int i = 0; i < size; i++) {
		const Connection &connection = connections[i];
		connection.source->disconnect(connection.signal, connection.target, connection.method);
	}
}

void TestCase::_handle_yield() {
	_clear_connections();
	m_yield_handled = true;
}

void TestCase::_yield_timer(float delay_sec) {
	Vector<Variant> binds;
	SceneTree::get_singleton()->create_timer(delay_sec)->connect("timeout", this, "_handle_yield", binds, CONNECT_ONESHOT);
}

TestCase *TestCase::yield_on(Object *object, const String &signal_name, real_t max_time) {
	_clear_connections();
	m_yield_handled = false;
	Vector<Variant> binds;
	object->connect(signal_name, this, "_handle_yield", binds, CONNECT_ONESHOT);
	if (0 < max_time) {
		_yield_timer(max_time);
	}
	return this;
}

TestCase *TestCase::yield_for(real_t time_in_seconds) {
	_clear_connections();
	m_yield_handled = false;
	_yield_timer(time_in_seconds);
	return this;
}

void TestCase::trace(const String &msg) {
	m_state->log()->trace(msg);
}

void TestCase::debug(const String &msg) {
	m_state->log()->debug(msg);
}

void TestCase::info(const String &msg) {
	m_state->log()->info(msg);
}

void TestCase::warn(const String &msg) {
	m_state->log()->warn(msg);
}

void TestCase::error(const String &msg) {
	m_state->log()->error(msg);
}

void TestCase::fatal(const String &msg) {
	m_state->log()->fatal(msg);
}

void TestCase::_bind_methods() {
	ClassDB::bind_method(D_METHOD("assert_equal", "a", "b", "msg"), &TestCase::assert_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_not_equal", "a", "b", "msg"), &TestCase::assert_not_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_true", "a", "msg"), &TestCase::assert_true, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_false", "a", "msg"), &TestCase::assert_false, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_is", "a", "b", "msg"), &TestCase::assert_is, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_is_not", "a", "b", "msg"), &TestCase::assert_is_not, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_is_null", "a", "msg"), &TestCase::assert_is_nil, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_is_not_null", "a", "msg"), &TestCase::assert_is_not_nil, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_in", "a", "b", "msg"), &TestCase::assert_in, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_not_in", "a", "b", "msg"), &TestCase::assert_not_in, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_is_type", "a", "b", "msg"), &TestCase::assert_is_type, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_is_not_type", "a", "b", "msg"), &TestCase::assert_is_not_type, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_almost_equal", "a", "b", "msg"), &TestCase::assert_almost_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_not_almost_equal", "a", "b", "msg"), &TestCase::assert_not_almost_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("yield_on", "object", "signal_name", "max_time"), &TestCase::yield_on, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("yield_for", "time_in_seconds"), &TestCase::yield_for);

	ClassDB::bind_method(D_METHOD("trace", "msg"), &TestCase::trace);
	ClassDB::bind_method(D_METHOD("debug", "msg"), &TestCase::debug);
	ClassDB::bind_method(D_METHOD("info", "msg"), &TestCase::info);
	ClassDB::bind_method(D_METHOD("warn", "msg"), &TestCase::warn);
	ClassDB::bind_method(D_METHOD("error", "msg"), &TestCase::error);
	ClassDB::bind_method(D_METHOD("fatal", "msg"), &TestCase::fatal);

	ClassDB::bind_method(D_METHOD("_handle_yield"), &TestCase::_handle_yield);
}
