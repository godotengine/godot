/*************************************************************************/
/*  test_state.h                                                         */
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

#ifndef TEST_STATE_H
#define TEST_STATE_H

#include "test_log.h"

#include "core/object.h"

class MethodIter {
public:
	bool init(const Object *p_object);
	bool next();
	const String &get() const;

private:
	bool next_test();

	List<MethodInfo> m_methods;
	List<MethodInfo>::Element *m_method_info;
};

class StageIter {
public:
	enum Stage {
		SETUP,
		TEST,
		TEARDOWN,
		DONE
	};

	bool init();
	bool next();
	void skip_test();
	Stage get() const;

private:
	Stage m_stage;
};

class TestState : public Object {
public:
	TestState();
	bool init(const Object *p_object);
	const String &method_name() const;
	StageIter::Stage stage() const;
	Ref<TestLog> log() const;
	bool next();
	void skip_test();
	const String &test_name() const;
	int test_count() const;
	int assert_count() const;

	bool is_valid() const;

	void assert();

protected:
	static void _bind_methods();

private:
	MethodIter m_method_iter;
	StageIter m_stage_iter;
	Ref<TestLog> m_log;
	String m_test_name;
	int m_test_count;
	int m_assert_count;
};

#endif // TEST_STATE_H
