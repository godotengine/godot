/*************************************************************************/
/*  text_test_result.h                                                   */
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

#ifndef TEXT_TEST_RESULT_H
#define TEXT_TEST_RESULT_H

#include "test_log.h"
#include "test_result.h"

class TextTestResult : public TestResult {
	GDCLASS(TextTestResult, TestResult);

public:
	TextTestResult();
	virtual void start_test(TestState *p_test_state);
	virtual void add_error(TestState *p_test_state, TestError *p_error);
	virtual void add_failure(TestState *p_test_state, TestError *p_error);
	virtual void add_success(TestState *p_test_state);
	virtual void stop_test(TestState *p_test_state);
	virtual void finish();
	void summary(const String &p_message);

protected:
	static void _bind_methods();

private:
	Ref<TestLog> m_log;
	int m_total_assert_count;
};

#endif // TEXT_TEST_RESULT_H
