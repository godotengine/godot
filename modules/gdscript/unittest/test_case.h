/*************************************************************************/
/*  test_case.h                                                          */
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

#ifndef TEST_CASE_H
#define TEST_CASE_H

#include "signal_watcher.h"
#include "test_result.h"
#include "test_state.h"

#include "gdscript.h"

#include "core/func_ref.h"
#include "core/list.h"
#include "scene/main/node.h"

class TestCase : public Node {
	GDCLASS(TestCase, Node);

public:
	TestCase();
	virtual ~TestCase();
	void init(Ref<TestResult> test_result);
	bool iteration(Ref<TestResult> test_result = NULL);

	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error);

protected:
	bool setup();
	bool test();
	bool teardown();

	void assert_equal(const Variant &p_left, const Variant &p_right, const String &p_message = "");
	void assert_not_equal(const Variant &p_left, const Variant &p_right, const String &p_message = "");
	void assert_true(const Variant &p_value, const String &p_message = "");
	void assert_false(const Variant &p_value, const String &p_message = "");
	void assert_is_type(const Variant &p_left, const Variant::Type type, const String &p_message = "");
	void assert_is_not_type(const Variant &p_left, const Variant::Type type, const String &p_message = "");
	void assert_is_nil(const Variant &p_value, const String &p_message = "");
	void assert_is_not_nil(const Variant &p_value, const String &p_message = "");
	void assert_in(const Variant &p_left, const Variant &p_right, const String &p_message = "");
	void assert_not_in(const Variant &p_left, const Variant &p_right, const String &p_message = "");
	void assert_is(const Variant &p_left, const Ref<GDScriptNativeClass> p_right, const String &p_message = "");
	void assert_is_not(const Variant &p_left, const Ref<GDScriptNativeClass> p_right, const String &p_message = "");

	void assert_aprox_equal(const Variant &p_left, const Variant &p_right, const String &p_message = "");
	void assert_aprox_not_equal(const Variant &p_left, const Variant &p_right, const String &p_message = "");
	void assert_greater(const Variant &p_left, const Variant &p_right, const String &p_message = "");
	void assert_greater_equal(const Variant &p_left, const Variant &p_right, const String &p_message = "");
	void assert_less(const Variant &p_left, const Variant &p_right, const String &p_message = "");
	void assert_less_equal(const Variant &p_left, const Variant &p_right, const String &p_message = "");
	void assert_match(const String &p_left, const String &p_right, const String &p_message = "");
	void assert_not_match(const String &p_left, const String &p_right, const String &p_message = "");

	TestCase *yield_on(Object *p_object, const String &p_signal_name, real_t p_max_time = -1);
	TestCase *yield_for(real_t time_in_seconds);

	void log(TestLog::LogLevel p_level, const String &p_message);
	void trace(const String &p_message);
	void debug(const String &p_message);
	void info(const String &p_message);
	void warn(const String &p_message);
	void error(const String &p_message);
	void fatal(const String &p_message);

	void watch_signal(Object *p_object, const String &p_signal);
	void watch_all_signals(Object *p_object);
	void assert_signal_called(const Object *p_object, const String &p_signal, const String &p_message = "");
	void assert_signal_called_once(const Object *p_object, const String &p_signal, const String &p_message = "");
	Variant _assert_signal_called_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	Variant _assert_signal_called_once_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	Variant _assert_signal_any_call(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	void assert_signal_has_calls(const Object *p_object, const String &p_signal, const Array &arguments, bool any_order = false, const String &p_message = "");
	void assert_signal_not_called(const Object *p_object, const String &p_signal, const String &p_message = "");
	int get_signal_call_count(const Object *p_object, const String &p_signal) const;
	Array get_signal_calls(const Object *p_object, const String &p_signal) const;

	Object *mock(Object *p_object = NULL);
	Ref<FuncRef> fr(Object *p_object, const String &p_function);
	int get_mock_call_count(const Object *p_object, const String &p_method) const;
	Array get_mock_calls(const Object *p_object, const String &p_method) const;

	static void _bind_methods();

private:
	TestState *m_state;
	bool m_has_next;
	List<TestError *> m_pending_errors;

	Ref<GDScriptFunctionState> m_yield;
	bool m_yield_handled;

	Ref<SignalWatcher> m_signal_watcher;

	void _clear_connections();
	Variant _handle_yield(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	void _yield_timer(real_t timeout);

	void assert(Dictionary &input, const String &default_msg, const String &custom_msg);
	void assert(const Variant &p_value, const String &default_msg, const String &custom_msg);
	void assert(const Variant &p_left, const Variant &p_right, const String &default_msg, const String &custom_msg);
};

#endif // TEST_CASE_H
