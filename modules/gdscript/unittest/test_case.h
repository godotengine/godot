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

#include "test_result.h"
#include "test_state.h"
#include "signal_watcher.h"

#include "gdscript.h"

#include "scene/main/node.h"

class TestCase : public Node {
	GDCLASS(TestCase, Node);

public:
	TestCase();
	virtual ~TestCase();

	void setup();
	void teardown();
	void init(Ref<TestResult> test_result);
	bool iteration(Ref<TestResult> test_result = NULL);
	bool can_assert() const;

	void assert_equal(const Variant &a, const Variant &b, const String &msg = "");
	void assert_not_equal(const Variant &a, const Variant &b, const String &msg = "");
	void assert_true(const Variant &x, const String &msg = "");
	void assert_false(const Variant &x, const String &msg = "");
	void assert_is_type(const Variant &a, const Variant::Type type, const String &msg = "");
	void assert_is_not_type(const Variant &a, const Variant::Type type, const String &msg = "");
	void assert_is_nil(const Variant &x, const String &msg = "");
	void assert_is_not_nil(const Variant &x, const String &msg = "");
	void assert_in(const Variant &a, const Variant &b, const String &msg = "");
	void assert_not_in(const Variant &a, const Variant &b, const String &msg = "");
	void assert_is(const Variant &a, const Ref<GDScriptNativeClass> b, const String &msg = "");
	void assert_is_not(const Variant &a, const Ref<GDScriptNativeClass> b, const String &msg = "");

	void assert_aprox_equal(const Variant &a, const Variant &b, const String &msg = "");
	void assert_aprox_not_equal(const Variant &a, const Variant &b, const String &msg = "");

	TestCase* yield_on(Object *object, const String &signal_name, real_t max_time=-1);
	TestCase* yield_for(real_t time_in_seconds);

	void log(TestLog::LogLevel level, const String &msg);
	void trace(const String &msg);
	void debug(const String &msg);
	void info(const String &msg);
	void warn(const String &msg);
	void error(const String &msg);
	void fatal(const String &msg);

	/*
    assert_greater(a, b)	a > b	2.7
    assert_greater_equal(a, b)	a >= b	2.7
    assert_less(a, b)	a < b	2.7
    assert_less_equal(a, b)	a <= b	2.7
    assert_match(s, r)	r.search(s)	2.7
    assert_not_match(s, r)	not r.search(s)	2.7
    */
protected:
	static void _bind_methods();

private:
	TestState* m_state;
	bool m_has_next;

	Ref<GDScriptFunctionState> m_yield; 
	bool m_yield_handled;

	Ref<SignalWatcher> signal_watcher;

	void _clear_connections();
	Variant _handle_yield(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	void _yield_timer(real_t timeout);
};

#endif // TEST_CASE_H
