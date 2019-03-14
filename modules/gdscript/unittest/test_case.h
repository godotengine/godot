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

#include "core/os/main_loop.h"
#include "gdscript.h"

struct Failure {
	String msg;
};

class TestYield : public Object {
	GDCLASS(TestYield, Object);

protected:
	static void _bind_methods();
};

class TestCase : public MainLoop {
	GDCLASS(TestCase, MainLoop);

public:

	virtual void init();
	virtual bool iteration(float p_time);
	virtual void finish();

	virtual void setup();
	virtual void teardown();
	void run(Ref<TestResult> test_result = NULL);

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

	void assert_almost_equal(const Variant &a, const Variant &b, const String &msg = "");
	void assert_not_almost_equal(const Variant &a, const Variant &b, const String &msg = "");

	void yield_on(const Object *object, const String &signal_name, real_t max_time=-1);
	void yield_for(real_t time_in_seconds);

	/*
    assertGreater(a, b)	a > b	2.7
    assertGreaterEqual(a, b)	a >= b	2.7
    assertLess(a, b)	a < b	2.7
    assertLessEqual(a, b)	a <= b	2.7
    assertRegexpMatches(s, r)	r.search(s)	2.7
    assertNotRegexpMatches(s, r)	not r.search(s)	2.7
    assertItemsEqual(a, b)	sorted(a) == sorted(b) and works with unhashable objs	2.7
    assertDictContainsSubset(a, b)	all the key/value pairs in a exist in b	2.7
    */

	/*
    assertMultiLineEqual(a, b)	strings	2.7
    assertSequenceEqual(a, b)	sequences	2.7
    assertListEqual(a, b)	lists	2.7
    assertTupleEqual(a, b)	tuples	2.7
    assertSetEqual(a, b)	sets or frozensets	2.7
    assertDictEqual(a, b)	dicts	2.7
    */
protected:
	static void _bind_methods();

private:
	TestState m_state;
	bool m_can_assert;
};

#endif // TEST_CASE_H
