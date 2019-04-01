/*************************************************************************/
/*  test_result.h                                                        */
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

#ifndef TEST_RESULT_H
#define TEST_RESULT_H

#include "test_error.h"
#include "test_state.h"

#include "core/reference.h"
#include "core/vector.h"

class TestResult : public Reference {
	GDCLASS(TestResult, Reference);

public:
	TestResult();
	virtual ~TestResult();
	virtual void start_test(TestState *p_test_state);
	virtual void stop_test(TestState *p_test_state);
	virtual void add_error(TestState *p_test_state, TestError *p_error);
	virtual void add_failure(TestState *p_test_state, TestError *p_error);
	virtual void add_success(TestState *p_test_state);
	virtual void finish();
	bool was_successful() const;
	void stop();
	bool should_stop() const;

protected:
	static void _bind_methods();

private:
	void _start_test(Object *p_test_state) { start_test(cast_to<TestState>(p_test_state)); }
	void _stop_test(Object *p_test_state) { stop_test(cast_to<TestState>(p_test_state)); }
	void _add_error(Object *p_test_state, Object *p_error) { add_error(cast_to<TestState>(p_test_state), cast_to<TestError>(p_error)); }
	void _add_failure(Object *p_test_state, Object *p_error) { add_failure(cast_to<TestState>(p_test_state), cast_to<TestError>(p_error)); }
	void _add_success(Object *p_test_state) { add_success(cast_to<TestState>(p_test_state)); }

	bool m_should_stop;
	int m_tests_run;
	Vector<TestError *> m_errors;
	Vector<TestError *> m_failures;
};

#endif // TEST_RESULT_H
