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

void assert(Dictionary &input, const String &default_msg, const String &custom_msg) {
	String msg = default_msg;
	if (!custom_msg.empty()) {
		msg = custom_msg;
	} else {
		input["msg"] = "Assertion : ";
	}
	TestError *test_error = memnew(TestError);
	test_error->m_message = msg.format(input);
	throw test_error;
}

void assert(const Variant &x, const String &default_msg, const String &custom_msg) {
	Dictionary input;
	input["x"] = x;
	assert(input, default_msg, custom_msg);
}

void assert(const Variant &a, const Variant &b, const String &default_msg, const String &custom_msg) {
	Dictionary input;
	input["a"] = a;
	input["b"] = b;
	assert(input, default_msg, custom_msg);
}

TestCase::TestCase() {
	m_state = memnew(TestState);
	m_signal_watcher.instance();
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
	try {
		REF guard = m_state->allow_assert();
		if (m_yield.is_valid() && m_yield->is_valid()) {
			if (m_yield_handled) {
				_clear_connections();
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
					m_yield = get_script_instance()->call(m_state->method_name());
					test_result->add_success(m_state);
					break;
				}
				case StageIter::TEARDOWN: {
					teardown();
					break;
				}
				case StageIter::DONE: {
					test_result->stop_test(m_state);
					m_signal_watcher->reset();
					break;
				}
			}
			m_has_next = m_state->next();
		}
	} catch (TestError *test_error) {
		m_yield.unref();
		test_result->add_failure(m_state, test_error);
		m_state->skip_test();
	}
	return !m_has_next || test_result->should_stop();
}

bool TestCase::can_assert() const {
	return m_state->can_assert();
}

void TestCase::assert_equal(const Variant &a, const Variant &b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (a != b) {
		assert(a, b, "{msg}{a} != {b}", msg);
	}
}

void TestCase::assert_not_equal(const Variant &a, const Variant &b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (a != b) {
		assert(a, b, "{msg}{a} == {b}", msg);
	}
}

void TestCase::assert_true(const Variant &x, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (!x) {
		assert(x, "{msg}{x} is false", msg);
	}
}

void TestCase::assert_false(const Variant &x, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (x) {
		assert(x, "{msg}{x} is true", msg);
	}
}

void TestCase::assert_is(const Variant &a, const Ref<GDScriptNativeClass> b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	Object *obj = cast_to<Object>(a);
	const String &klass = obj->get_class();
	const String &inherits = obj->get_class();
	if (!ClassDB::is_parent_class(klass, inherits)) {
		assert(klass, inherits, msg, "{msg}{a} is not {b}");
	}
}

void TestCase::assert_is_not(const Variant &a, const Ref<GDScriptNativeClass> b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	Object *obj = cast_to<Object>(a);
	const String &klass = obj->get_class();
	const String &inherits = obj->get_class();
	if (!ClassDB::is_parent_class(klass, inherits)) {
		assert(klass, inherits, msg, "{msg}{a} is {b}");
	}
}

void TestCase::assert_is_nil(const Variant &x, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (Variant::NIL != x.get_type()) {
		assert(x, "{msg}{x} is not null", msg);
	}
}

void TestCase::assert_is_not_nil(const Variant &x, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (Variant::NIL == x.get_type()) {
		assert(x, "{msg}{x} is null", msg);
	}
}

void TestCase::assert_in(const Variant &a, const Variant &b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	bool is_in = false;
	if (b.is_array()) {
		const Array &collection = b;
		is_in = collection.has(a);
	} else if (b.get_type() == Variant::DICTIONARY) {
		const Dictionary &collection = b;
		is_in = collection.has(a);
	}
	if (!is_in) {
		assert(a, b, "{msg}{a} is not in {b}", msg);
	}
}

void TestCase::assert_not_in(const Variant &a, const Variant &b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	bool is_in = false;
	if (b.is_array()) {
		const Array &collection = b;
		is_in = collection.has(a);
	} else if (b.get_type() == Variant::DICTIONARY) {
		const Dictionary &collection = b;
		is_in = collection.has(a);
	}
	if (is_in) {
		assert(a, b, "{msg}{a} is in {b}", msg);
	}
}

void TestCase::assert_is_type(const Variant &a, const Variant::Type type, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (a.get_type() != type) {
		assert(a, Variant::get_type_name(type), "{msg}{a} is not {b}", msg);
	}
}

void TestCase::assert_is_not_type(const Variant &a, const Variant::Type type, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (a.get_type() == type) {
		assert(a, Variant::get_type_name(type), "{msg}{a} is {b}", msg);
	}
}

void TestCase::assert_aprox_equal(const Variant &a, const Variant &b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (!Math::is_equal_approx(a, b)) {
		assert(a, b, "{msg}{a} is not close to {b}", msg);
	}
}

void TestCase::assert_aprox_not_equal(const Variant &a, const Variant &b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (Math::is_equal_approx(a, b)) {
		assert(a, b, "{msg}{a} is close to {b}", msg);
	}
}
void TestCase::assert_greater(const Variant &a, const Variant &b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (a == b || a < b) {
		assert(a, b, "{msg}{a} <= {b}", msg);
	}
}

void TestCase::assert_greater_equal(const Variant &a, const Variant &b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (a < b) {
		assert(a, b, "{msg}{a} < {b}", msg);
	}
}

void TestCase::assert_less(const Variant &a, const Variant &b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (a == b || b < a) {
		assert(a, b, "{msg}{a} >= {b}", msg);
	}
}

void TestCase::assert_less_equal(const Variant &a, const Variant &b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (b < a) {
		assert(a, b, "{msg}{a} > {b}", msg);
	}
}

void TestCase::assert_match(const String &a, const String &b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (!a.match(b)) {
		assert(a, b, "{msg}{a} does not match {b}", msg);
	}
}

void TestCase::assert_not_match(const String &a, const String &b, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (a.match(b)) {
		assert(a, b, "{msg}{a} matches {b}", msg);
	}
}

void TestCase::_clear_connections() {
	List<Connection> connections;
	this->get_signals_connected_to_this(&connections);
	int size = connections.size();
	for (int i = 0; i < size; i++) {
		const Connection &connection = connections[i];
		if (connection.method == "_handle_yield") {
			connection.source->disconnect(connection.signal, connection.target, connection.method);
		}
	}
}

Variant TestCase::_handle_yield(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	r_error.error = Variant::CallError::CALL_OK;
	m_yield_handled = true;
	return NULL;
}

void TestCase::_yield_timer(float delay_sec) {
	Vector<Variant> binds;
	SceneTree::get_singleton()->create_timer(delay_sec)->connect("timeout", this, "_handle_yield", binds, CONNECT_ONESHOT);
}

TestCase *TestCase::yield_on(Object *object, const String &signal_name, real_t max_time) {
	m_yield_handled = false;
	Vector<Variant> binds;
	ERR_FAIL_COND_V(object->connect(signal_name, this, "_handle_yield", binds, CONNECT_ONESHOT), NULL);
	if (0 < max_time) {
		_yield_timer(max_time);
	}
	return this;
}

TestCase *TestCase::yield_for(real_t time_in_seconds) {
	m_yield_handled = false;
	_yield_timer(time_in_seconds);
	return this;
}

void TestCase::log(TestLog::LogLevel level, const String &msg) {
	m_state->log()->log(level, m_state->test_name(), m_state->method_name(), msg);
}

void TestCase::trace(const String &msg) {
	log(TestLog::TRACE, msg);
}

void TestCase::debug(const String &msg) {
	log(TestLog::DEBUG, msg);
}

void TestCase::info(const String &msg) {
	log(TestLog::INFO, msg);
}

void TestCase::warn(const String &msg) {
	log(TestLog::WARN, msg);
}

void TestCase::error(const String &msg) {
	log(TestLog::ERROR, msg);
}

void TestCase::fatal(const String &msg) {
	log(TestLog::FATAL, msg);
}

void TestCase::watch_signals(Object *object, const String &signal) {
	m_signal_watcher->watch(object, signal);
}

void TestCase::assert_called(const Object *object, const String &signal, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (!m_signal_watcher->called(object, signal)) {
		assert(object, signal, msg, "{msg}{a}.{b} was not called");
	}
}

void TestCase::assert_called_once(const Object *object, const String &signal, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (!m_signal_watcher->called_once(object, signal)) {
		assert(object, signal, msg, "{msg}{a}.{b} was not called exactly once");
	}
}

Variant TestCase::_assert_called_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	ERR_FAIL_COND_V(!can_assert(), Variant());
	SignalWatcher::Params params;
	if (SignalWatcher::parse_params(p_args, p_argcount, r_error, params)) {
		if (!m_signal_watcher->called_with(params.object, params.signal, params.arguments)) {
			assert("invalid", "error", "{msg}{a}.{b} was not called with");
		}
	}
	return Variant();
}

Variant TestCase::_assert_called_once_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	ERR_FAIL_COND_V(!can_assert(), Variant());
	SignalWatcher::Params params;
	if (SignalWatcher::parse_params(p_args, p_argcount, r_error, params)) {
		if (!m_signal_watcher->called_once_with(params.object, params.signal, params.arguments)) {
			assert(params.object, params.signal, "{msg}{a}.{b} was not called exactly once", "");
		}
	}
	return Variant();
}

Variant TestCase::_assert_any_call(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	ERR_FAIL_COND_V(!can_assert(), Variant());
	SignalWatcher::Params params;
	if (SignalWatcher::parse_params(p_args, p_argcount, r_error, params)) {
		if (!m_signal_watcher->any_call(params.object, params.signal, params.arguments)) {
			assert("invalid", "error", "{msg}{a}.{b} was not called ");
		}
	}
	return Variant();
}

void TestCase::assert_has_calls(const Object *object, const String &signal, const Array &arguments, bool any_order, const String &msg) const {
	int error = m_signal_watcher->has_calls(object, signal, arguments, any_order);
	if (error < arguments.size()) {
		if (error == -1) {
		} else {
		}
	}
}

void TestCase::assert_not_called(const Object *object, const String &signal, const String &msg) const {
	ERR_FAIL_COND(!can_assert());
	if (!m_signal_watcher->not_called(object, signal)) {
		assert(object, signal, msg, "{msg}{a}.{b} was not called");
	}
}

int TestCase::get_signal_call_count(const Object *object, const String &signal) const {
	return m_signal_watcher->call_count(object, signal);
}

Array TestCase::get_signal_calls(const Object *object, const String &signal) const {
	return m_signal_watcher->calls(object, signal);
}

void TestCase::_bind_methods() {
	ClassDB::bind_method(D_METHOD("assert_equals", "a", "b", "msg"), &TestCase::assert_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_equal", "a", "b", "msg"), &TestCase::assert_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_eq", "a", "b", "msg"), &TestCase::assert_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_not_equals", "a", "b", "msg"), &TestCase::assert_not_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_not_equal", "a", "b", "msg"), &TestCase::assert_not_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_ne", "a", "b", "msg"), &TestCase::assert_not_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_true", "a", "msg"), &TestCase::assert_true, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_t", "a", "msg"), &TestCase::assert_true, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_false", "a", "msg"), &TestCase::assert_false, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_f", "a", "msg"), &TestCase::assert_false, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_is", "a", "b", "msg"), &TestCase::assert_is, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_is_not", "a", "b", "msg"), &TestCase::assert_is_not, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_is_null", "a", "msg"), &TestCase::assert_is_nil, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_is_not_null", "a", "msg"), &TestCase::assert_is_not_nil, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_in", "a", "b", "msg"), &TestCase::assert_in, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_not_in", "a", "b", "msg"), &TestCase::assert_not_in, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_is_type", "a", "b", "msg"), &TestCase::assert_is_type, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_is_not_type", "a", "b", "msg"), &TestCase::assert_is_not_type, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_aprox_equals", "a", "b", "msg"), &TestCase::assert_aprox_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_aprox_equal", "a", "b", "msg"), &TestCase::assert_aprox_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_aprox_eq", "a", "b", "msg"), &TestCase::assert_aprox_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_aprox_not_equals", "a", "b", "msg"), &TestCase::assert_aprox_not_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_aprox_not_equal", "a", "b", "msg"), &TestCase::assert_aprox_not_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_aprox_not_eq", "a", "b", "msg"), &TestCase::assert_aprox_not_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_aprox_ne", "a", "b", "msg"), &TestCase::assert_aprox_not_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_greater_than", "a", "b", "msg"), &TestCase::assert_greater, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_greater", "a", "b", "msg"), &TestCase::assert_greater, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_gt", "a", "b", "msg"), &TestCase::assert_greater, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_greater_equals", "a", "b", "msg"), &TestCase::assert_greater_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_greater_equal", "a", "b", "msg"), &TestCase::assert_greater_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_greater_eq", "a", "b", "msg"), &TestCase::assert_greater_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_ge", "a", "b", "msg"), &TestCase::assert_greater_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_less_than", "a", "b", "msg"), &TestCase::assert_less, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_less", "a", "b", "msg"), &TestCase::assert_less, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_lt", "a", "b", "msg"), &TestCase::assert_less, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_less_equals", "a", "b", "msg"), &TestCase::assert_less_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_less_equal", "a", "b", "msg"), &TestCase::assert_less_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_less_eq", "a", "b", "msg"), &TestCase::assert_less_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_le", "a", "b", "msg"), &TestCase::assert_less_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_match", "a", "b", "msg"), &TestCase::assert_match, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_not_match", "a", "b", "msg"), &TestCase::assert_not_match, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("yield_on", "object", "signal_name", "max_time"), &TestCase::yield_on, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("yield_for", "time_in_seconds"), &TestCase::yield_for);

	ClassDB::bind_method(D_METHOD("trace", "msg"), &TestCase::trace);
	ClassDB::bind_method(D_METHOD("debug", "msg"), &TestCase::debug);
	ClassDB::bind_method(D_METHOD("info", "msg"), &TestCase::info);
	ClassDB::bind_method(D_METHOD("warn", "msg"), &TestCase::warn);
	ClassDB::bind_method(D_METHOD("error", "msg"), &TestCase::error);
	ClassDB::bind_method(D_METHOD("fatal", "msg"), &TestCase::fatal);

	ClassDB::bind_method(D_METHOD("watch_signals", "object", "signal"), &TestCase::watch_signals);
	ClassDB::bind_method(D_METHOD("assert_called", "object", "signal", "msg"), &TestCase::assert_called, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_called_once", "object", "signal", "msg"), &TestCase::assert_called_once, DEFVAL(""));
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "assert_called_with", &TestCase::_assert_called_with);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "assert_called_once_with", &TestCase::_assert_called_once_with);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "assert_any_call", &TestCase::_assert_any_call);
	ClassDB::bind_method(D_METHOD("assert_has_calls", "object", "signal", "arguments", "any_order", "msg"), &TestCase::assert_has_calls, DEFVAL(false), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_not_called", "object", "signal", "msg"), &TestCase::assert_not_called, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_signal_call_count", "object", "signal"), &TestCase::get_signal_call_count);
	ClassDB::bind_method(D_METHOD("get_signal_calls", "object", "signal"), &TestCase::get_signal_calls);

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "_handle_yield", &TestCase::_handle_yield, MethodInfo("_handle_yield"));
}
