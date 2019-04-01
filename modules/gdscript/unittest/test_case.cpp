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
#include "proxy_script.h"
#include "test_compare.h"
#include "test_config.h"

#include "core/math/math_funcs.h"
#include "core/script_language.h"

void TestCase::assert(Dictionary &input, const String &default_msg, const String &custom_msg) {
	String p_message = default_msg;
	if (!custom_msg.empty()) {
		p_message = custom_msg;
	} else {
		input["message"] = "Assertion : ";
	}
	TestError *test_error = memnew(TestError);
	test_error->m_message = p_message.format(input);
	m_pending_errors.push_back(test_error);
}

void TestCase::assert(const Variant &p_value, const String &default_msg, const String &custom_msg) {
	Dictionary input;
	input["value"] = p_value;
	assert(input, default_msg, custom_msg);
}

void TestCase::assert(const Variant &p_left, const Variant &p_right, const String &default_msg, const String &custom_msg) {
	Dictionary input;
	input["left"] = p_left;
	input["right"] = p_right;
	assert(input, default_msg, custom_msg);
}

TestCase::TestCase() {
	m_state = memnew(TestState);
	m_signal_watcher.instance();
}

TestCase::~TestCase() {
	memfree(m_state);
	m_state = NULL;
}

bool TestCase::setup() {
	if (get_script_instance() && get_script_instance()->has_method("setup")) {
		Variant::CallError ce;
		m_yield = get_script_instance()->call("setup", NULL, 0, ce);
		return ce.error != Variant::CallError::CALL_OK;
	}
	return false;
}

bool TestCase::test() {
	Variant::CallError ce;
	m_yield = get_script_instance()->call(m_state->method_name(), NULL, 0, ce);
	return ce.error != Variant::CallError::CALL_OK;
}

bool TestCase::teardown() {
	if (get_script_instance() && get_script_instance()->has_method("teardown")) {
		Variant::CallError ce;
		m_yield = get_script_instance()->call("teardown", NULL, 0, ce);
		return ce.error != Variant::CallError::CALL_OK;
	}
	return false;
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
	if (m_yield.is_valid() && m_yield->is_valid()) {
		if (m_yield_handled) {
			_clear_connections();
			m_yield = m_yield->resume();
		}
	} else if (m_has_next) {
		switch (m_state->stage()) {
			case StageIter::SETUP: {
				test_result->start_test(m_state);
				setup();
				break;
			}
			case StageIter::TEST: {
				if (!test() && m_pending_errors.empty()) {
					test_result->add_success(m_state);
				}
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
		if (!m_pending_errors.empty()) {
			m_state->skip_test();
		} else {
			m_has_next = m_state->next();
		}
	}
	if (!m_pending_errors.empty()) {
		do {
			test_result->add_failure(m_state, m_pending_errors.front()->get());
			m_pending_errors.pop_front();
		} while (!m_pending_errors.empty());
		m_state->skip_test();
	}
	return !m_has_next || test_result->should_stop();
}

Variant TestCase::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	Variant result = Node::call(p_method, p_args, p_argcount, r_error);
	if (String(p_method).begins_with("assert_")) {
		m_state->assert();
	}
	// un-wind stack
	if (!m_pending_errors.empty()) {
		r_error.error = (Variant::CallError::Error)10;
	}
	return result;
}

void TestCase::assert_equal(const Variant &p_left, const Variant &p_right, const String &p_message) {
	if (!TestCompare::deep_equal(p_left, p_right)) {
		assert(p_left, p_right, "{message}{left} != {right}", p_message);
	}
}

void TestCase::assert_not_equal(const Variant &p_left, const Variant &p_right, const String &p_message) {
	if (p_left != p_right) {
		assert(p_left, p_right, "{message}{left} == {right}", p_message);
	}
}

void TestCase::assert_true(const Variant &p_value, const String &p_message) {
	if (!p_value) {
		assert(p_value, "{message}{value} is false", p_message);
	}
}

void TestCase::assert_false(const Variant &p_value, const String &p_message) {
	if (p_value) {
		assert(p_value, "{message}{value} is true", p_message);
	}
}

void TestCase::assert_is(const Variant &p_left, const Ref<GDScriptNativeClass> p_right, const String &p_message) {
	Object *obj = cast_to<Object>(p_left);
	const String &klass = obj->get_class();
	const String &inherits = p_right->get_class();
	if (!ClassDB::is_parent_class(klass, inherits)) {
		assert(klass, inherits, p_message, "{message}{left} is not {right}");
	}
}

void TestCase::assert_is_not(const Variant &p_left, const Ref<GDScriptNativeClass> p_right, const String &p_message) {
	Object *obj = cast_to<Object>(p_left);
	const String &klass = obj->get_class();
	const String &inherits = p_right->get_class();
	if (!ClassDB::is_parent_class(klass, inherits)) {
		assert(klass, inherits, p_message, "{message}{left} is {right}");
	}
}

void TestCase::assert_is_nil(const Variant &p_value, const String &p_message) {
	if (Variant::NIL != p_value.get_type()) {
		assert(p_value, "{message}{value} is not null", p_message);
	}
}

void TestCase::assert_is_not_nil(const Variant &p_value, const String &p_message) {
	if (Variant::NIL == p_value.get_type()) {
		assert(p_value, "{message}{value} is null", p_message);
	}
}

void TestCase::assert_in(const Variant &p_left, const Variant &p_right, const String &p_message) {
	bool is_in = false;
	if (p_right.is_array()) {
		const Array &collection = p_right;
		is_in = collection.has(p_left);
	} else if (p_right.get_type() == Variant::DICTIONARY) {
		const Dictionary &collection = p_right;
		is_in = collection.has(p_left);
	}
	if (!is_in) {
		assert(p_left, p_right, "{message}{left} is not in {right}", p_message);
	}
}

void TestCase::assert_not_in(const Variant &p_left, const Variant &p_right, const String &p_message) {
	bool is_in = false;
	if (p_right.is_array()) {
		const Array &collection = p_right;
		is_in = collection.has(p_left);
	} else if (p_right.get_type() == Variant::DICTIONARY) {
		const Dictionary &collection = p_right;
		is_in = collection.has(p_left);
	}
	if (is_in) {
		assert(p_left, p_right, "{message}{left} is in {right}", p_message);
	}
}

void TestCase::assert_is_type(const Variant &p_left, const Variant::Type type, const String &p_message) {
	if (p_left.get_type() != type) {
		assert(p_left, Variant::get_type_name(type), "{message}{left} is not {right}", p_message);
	}
}

void TestCase::assert_is_not_type(const Variant &p_left, const Variant::Type type, const String &p_message) {
	if (p_left.get_type() == type) {
		assert(p_left, Variant::get_type_name(type), "{message}{left} is {right}", p_message);
	}
}

void TestCase::assert_aprox_equal(const Variant &p_left, const Variant &p_right, const String &p_message) {
	if (!Math::is_equal_approx(p_left, p_right)) {
		assert(p_left, p_right, "{message}{left} is not close to {right}", p_message);
	}
}

void TestCase::assert_aprox_not_equal(const Variant &p_left, const Variant &p_right, const String &p_message) {
	if (Math::is_equal_approx(p_left, p_right)) {
		assert(p_left, p_right, "{message}{left} is close to {right}", p_message);
	}
}
void TestCase::assert_greater(const Variant &p_left, const Variant &p_right, const String &p_message) {
	if (p_left == p_right || p_left < p_right) {
		assert(p_left, p_right, "{message}{left} <= {right}", p_message);
	}
}

void TestCase::assert_greater_equal(const Variant &p_left, const Variant &p_right, const String &p_message) {
	if (p_left < p_right) {
		assert(p_left, p_right, "{message}{left} < {right}", p_message);
	}
}

void TestCase::assert_less(const Variant &p_left, const Variant &p_right, const String &p_message) {
	if (p_left == p_right || p_right < p_left) {
		assert(p_left, p_right, "{message}{left} >= {right}", p_message);
	}
}

void TestCase::assert_less_equal(const Variant &p_left, const Variant &p_right, const String &p_message) {
	if (p_right < p_left) {
		assert(p_left, p_right, "{message}{left} > {right}", p_message);
	}
}

void TestCase::assert_match(const String &p_left, const String &p_right, const String &p_message) {
	if (!p_left.match(p_right)) {
		assert(p_left, p_right, "{message}{left} does not match {right}", p_message);
	}
}

void TestCase::assert_not_match(const String &p_left, const String &p_right, const String &p_message) {
	if (p_left.match(p_right)) {
		assert(p_left, p_right, "{message}{left} matches {right}", p_message);
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
	return Variant();
}

void TestCase::_yield_timer(real_t delay_sec) {
	Vector<Variant> binds;
	SceneTree::get_singleton()->create_timer(delay_sec)->connect("timeout", this, "_handle_yield", binds, CONNECT_ONESHOT);
}

TestCase *TestCase::yield_on(Object *p_object, const String &p_signal_name, real_t p_max_time) {
	m_yield_handled = false;
	Vector<Variant> binds;
	ERR_FAIL_COND_V(p_object->connect(p_signal_name, this, "_handle_yield", binds, CONNECT_ONESHOT), NULL);
	if (0 < p_max_time) {
		_yield_timer(p_max_time);
	}
	return this;
}

TestCase *TestCase::yield_for(real_t time_in_seconds) {
	m_yield_handled = false;
	_yield_timer(time_in_seconds);
	return this;
}

void TestCase::log(TestLog::LogLevel level, const String &p_message) {
	m_state->log()->log(level, m_state->test_name(), m_state->method_name(), p_message);
}

void TestCase::trace(const String &p_message) {
	log(TestLog::TRACE, p_message);
}

void TestCase::debug(const String &p_message) {
	log(TestLog::DEBUG, p_message);
}

void TestCase::info(const String &p_message) {
	log(TestLog::INFO, p_message);
}

void TestCase::warn(const String &p_message) {
	log(TestLog::WARN, p_message);
}

void TestCase::error(const String &p_message) {
	log(TestLog::ERROR, p_message);
}

void TestCase::fatal(const String &p_message) {
	log(TestLog::FATAL, p_message);
}

void TestCase::watch_signal(Object *p_object, const String &p_signal) {
	m_signal_watcher->watch(p_object, p_signal);
}

void TestCase::watch_all_signals(Object *p_object) {
	m_signal_watcher->watch_all(p_object);
}

void TestCase::assert_signal_called(const Object *p_object, const String &p_signal, const String &p_message) {
	if (!m_signal_watcher->called(p_object, p_signal)) {
		assert(p_object, p_signal, p_message, "{message}{left}.{right} was not called");
	}
}

void TestCase::assert_signal_called_once(const Object *p_object, const String &p_signal, const String &p_message) {
	if (!m_signal_watcher->called_once(p_object, p_signal)) {
		assert(p_object, p_signal, p_message, "{message}{left}.{right} was not called exactly once");
	}
}

Variant TestCase::_assert_signal_called_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	SignalWatcher::Params params;
	if (SignalWatcher::parse_params(p_args, p_argcount, r_error, params)) {
		if (!m_signal_watcher->called_with(params.m_object, params.m_signal, params.m_arguments)) {
			assert("invalid", "error", "{message}{left}.{right} was not called with");
		}
	}
	return Variant();
}

Variant TestCase::_assert_signal_called_once_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	SignalWatcher::Params params;
	if (SignalWatcher::parse_params(p_args, p_argcount, r_error, params)) {
		if (!m_signal_watcher->called_once_with(params.m_object, params.m_signal, params.m_arguments)) {
			assert(params.m_object, params.m_signal, "{message}{left}.{right} was not called exactly once", "");
		}
	}
	return Variant();
}

Variant TestCase::_assert_signal_any_call(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	SignalWatcher::Params params;
	if (SignalWatcher::parse_params(p_args, p_argcount, r_error, params)) {
		if (!m_signal_watcher->any_call(params.m_object, params.m_signal, params.m_arguments)) {
			assert("invalid", "error", "{message}{left}.{right} was not called ");
		}
	}
	return Variant();
}

void TestCase::assert_signal_has_calls(const Object *p_object, const String &p_signal, const Array &p_arguments, bool p_any_order, const String &p_message) {
	int error = m_signal_watcher->has_calls(p_object, p_signal, p_arguments, p_any_order);
	if (error < p_arguments.size()) {
		if (error == -1) {
		} else {
		}
	}
}

void TestCase::assert_signal_not_called(const Object *p_object, const String &p_signal, const String &p_message) {
	if (!m_signal_watcher->not_called(p_object, p_signal)) {
		assert(p_object, p_signal, p_message, "{message}{left}.{right} was not called");
	}
}

int TestCase::get_signal_call_count(const Object *p_object, const String &p_signal) const {
	return m_signal_watcher->call_count(p_object, p_signal);
}

Array TestCase::get_signal_calls(const Object *p_object, const String &p_signal) const {
	return m_signal_watcher->calls(p_object, p_signal);
}

Object *TestCase::mock(Object *p_object) {
	Ref<GDScript> script(cast_to<GDScript>(p_object));
	if (script.is_valid()) {
		Variant::CallError error;
		p_object = script->_new(NULL, 0, error);
	}
	Ref<GDScriptNativeClass> native(cast_to<GDScriptNativeClass>(p_object));
	if (native.is_valid()) {
		p_object = native->instance();
	}
	if (!p_object) {
		p_object = memnew(Object);
	}
	Ref<Script> proxy_script(memnew(ProxyScript(p_object->get_script())));
	p_object->set_script(proxy_script.get_ref_ptr());

	return p_object;
}

int TestCase::get_mock_call_count(const Object *p_object, const String &p_method) const {
	ProxyScriptInstance *proxy_script_instance = dynamic_cast<ProxyScriptInstance *>(p_object->get_script_instance());
	ERR_FAIL_COND_V(!proxy_script_instance, -1);
	return proxy_script_instance->get_calls(p_method).size();
}

Array TestCase::get_mock_calls(const Object *p_object, const String &p_method) const {
	Array result;
	ProxyScriptInstance *proxy_script_instance = dynamic_cast<ProxyScriptInstance *>(p_object->get_script_instance());
	ERR_FAIL_COND_V(!proxy_script_instance, result);
	const Vector<MethodWatcher::Args> &calls = proxy_script_instance->get_calls(p_method);
	int size = calls.size();
	result.resize(size);
	for (int i = 0; i < size; i++) {
		result[i] = calls[i];
	}
	return result;
}

Ref<FuncRef> TestCase::fr(Object *p_object, const String &p_function) {
	Ref<FuncRef> func(memnew(FuncRef));
	func->set_instance(p_object);
	func->set_function(p_function);
	return func;
}

void TestCase::_bind_methods() {
	ClassDB::bind_method(D_METHOD("assert_equals", "left", "right", "message"), &TestCase::assert_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_equal", "left", "right", "message"), &TestCase::assert_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_eq", "left", "right", "message"), &TestCase::assert_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_not_equals", "left", "right", "message"), &TestCase::assert_not_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_not_equal", "left", "right", "message"), &TestCase::assert_not_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_ne", "left", "right", "message"), &TestCase::assert_not_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_true", "left", "message"), &TestCase::assert_true, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_t", "left", "message"), &TestCase::assert_true, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_false", "left", "message"), &TestCase::assert_false, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_f", "left", "message"), &TestCase::assert_false, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_is", "left", "right", "message"), &TestCase::assert_is, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_is_not", "left", "right", "message"), &TestCase::assert_is_not, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_is_null", "left", "message"), &TestCase::assert_is_nil, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_is_not_null", "left", "message"), &TestCase::assert_is_not_nil, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_in", "left", "right", "message"), &TestCase::assert_in, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_not_in", "left", "right", "message"), &TestCase::assert_not_in, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_is_type", "left", "right", "message"), &TestCase::assert_is_type, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_is_not_type", "left", "right", "message"), &TestCase::assert_is_not_type, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_aprox_equals", "left", "right", "message"), &TestCase::assert_aprox_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_aprox_equal", "left", "right", "message"), &TestCase::assert_aprox_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_aprox_eq", "left", "right", "message"), &TestCase::assert_aprox_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_aprox_not_equals", "left", "right", "message"), &TestCase::assert_aprox_not_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_aprox_not_equal", "left", "right", "message"), &TestCase::assert_aprox_not_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_aprox_not_eq", "left", "right", "message"), &TestCase::assert_aprox_not_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_aprox_ne", "left", "right", "message"), &TestCase::assert_aprox_not_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_greater_than", "left", "right", "message"), &TestCase::assert_greater, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_greater", "left", "right", "message"), &TestCase::assert_greater, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_gt", "left", "right", "message"), &TestCase::assert_greater, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_greater_equals", "left", "right", "message"), &TestCase::assert_greater_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_greater_equal", "left", "right", "message"), &TestCase::assert_greater_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_greater_eq", "left", "right", "message"), &TestCase::assert_greater_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_ge", "left", "right", "message"), &TestCase::assert_greater_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_less_than", "left", "right", "message"), &TestCase::assert_less, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_less", "left", "right", "message"), &TestCase::assert_less, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_lt", "left", "right", "message"), &TestCase::assert_less, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_less_equals", "left", "right", "message"), &TestCase::assert_less_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_less_equal", "left", "right", "message"), &TestCase::assert_less_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_less_eq", "left", "right", "message"), &TestCase::assert_less_equal, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_le", "left", "right", "message"), &TestCase::assert_less_equal, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("assert_match", "left", "right", "message"), &TestCase::assert_match, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_not_match", "left", "right", "message"), &TestCase::assert_not_match, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("yield_on", "object", "p_signal_name", "max_time"), &TestCase::yield_on, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("yield_for", "time_in_seconds"), &TestCase::yield_for);

	ClassDB::bind_method(D_METHOD("trace", "message"), &TestCase::trace);
	ClassDB::bind_method(D_METHOD("debug", "message"), &TestCase::debug);
	ClassDB::bind_method(D_METHOD("info", "message"), &TestCase::info);
	ClassDB::bind_method(D_METHOD("warn", "message"), &TestCase::warn);
	ClassDB::bind_method(D_METHOD("error", "message"), &TestCase::error);
	ClassDB::bind_method(D_METHOD("fatal", "message"), &TestCase::fatal);

	ClassDB::bind_method(D_METHOD("watch_signal", "object", "signal"), &TestCase::watch_signal);
	ClassDB::bind_method(D_METHOD("watch", "object", "signal"), &TestCase::watch_signal);
	ClassDB::bind_method(D_METHOD("watch_all_signals", "object"), &TestCase::watch_all_signals);
	ClassDB::bind_method(D_METHOD("watch_all", "object"), &TestCase::watch_all_signals);

	ClassDB::bind_method(D_METHOD("assert_signal_called", "object", "signal", "message"), &TestCase::assert_signal_called, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_signal_called_once", "object", "signal", "message"), &TestCase::assert_signal_called_once, DEFVAL(""));
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "assert_signal_called_with", &TestCase::_assert_signal_called_with);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "assert_signal_called_once_with", &TestCase::_assert_signal_called_once_with);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "assert_signal_any_call", &TestCase::_assert_signal_any_call);
	ClassDB::bind_method(D_METHOD("assert_signal_has_calls", "object", "signal", "arguments", "any_order", "message"), &TestCase::assert_signal_has_calls, DEFVAL(false), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("assert_signal_not_called", "object", "signal", "message"), &TestCase::assert_signal_not_called, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_signal_call_count", "object", "signal"), &TestCase::get_signal_call_count);
	ClassDB::bind_method(D_METHOD("get_signal_calls", "object", "signal"), &TestCase::get_signal_calls);

	ClassDB::bind_method(D_METHOD("mock", "base"), &TestCase::mock, DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("get_mock_calls", "object", "method"), &TestCase::get_mock_calls);
	ClassDB::bind_method(D_METHOD("fr", "object", "function"), &TestCase::fr);

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "_handle_yield", &TestCase::_handle_yield, MethodInfo("_handle_yield"));
}
