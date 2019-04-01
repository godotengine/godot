/*************************************************************************/
/*  test_suite.h                                                         */
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

#ifndef TEST_SUITE_H
#define TEST_SUITE_H

#include "test_case.h"
#include "test_result.h"

#include "core/reference.h"
#include "core/vector.h"

#include "scene/main/viewport.h"

class TestSuite : public Reference {
	GDCLASS(TestSuite, Reference);

public:
	TestSuite();
	virtual ~TestSuite();
	int count_test_cases() const;
	void add_test(TestCase *p_test_case);
	void add_tests(Array p_test_cases);
	void init(Viewport *root, Ref<TestResult> p_test_result);
	bool iteration(Ref<TestResult> p_test_result);

protected:
	static void _bind_methods();

private:
	Viewport *m_root;
	Vector<TestCase *> m_test_cases;
	int m_case_index;
};

#endif // TEST_SUITE_H
