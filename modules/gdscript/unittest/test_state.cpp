/*************************************************************************/
/*  test_state.cpp                                                       */
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

#include "test_state.h"

bool TestState::next_test() {
	while (m_method_info) {
		if (m_method_info->get().name.begins_with("test_")) {
			return true;
		}
		m_method_info = m_method_info->next();
	}
	return false;
}

bool TestState::init(const Object *object) {
	object->get_method_list(&m_methods);
	m_method_info = m_methods.front();
	return next_test();
}

const String &TestState::get() {
	return m_method_info->get().name;
}

bool TestState::next() {
	m_method_info = m_method_info->next();
	return next_test();
}

void TestState::_bind_methods() {
    ClassDB::bind_method(D_METHOD("init", "object"), &TestState::init);
    ClassDB::bind_method(D_METHOD("get"), &TestState::get);
    ClassDB::bind_method(D_METHOD("next"), &TestState::next);
}
