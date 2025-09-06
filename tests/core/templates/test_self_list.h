/**************************************************************************/
/*  test_self_list.h                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/templates/self_list.h"

#include "tests/test_macros.h"

namespace TestSelfList {

TEST_CASE("[SelfList] Sort") {
	const int SIZE = 5;
	int numbers[SIZE]{ 3, 2, 5, 1, 4 };
	SelfList<int> elements[SIZE]{
		SelfList<int>(&numbers[0]),
		SelfList<int>(&numbers[1]),
		SelfList<int>(&numbers[2]),
		SelfList<int>(&numbers[3]),
		SelfList<int>(&numbers[4]),
	};

	SelfList<int>::List list;
	for (int i = 0; i < SIZE; i++) {
		list.add_last(&elements[i]);
	}

	SelfList<int> *it = list.first();
	for (int i = 0; i < SIZE; i++) {
		CHECK_EQ(numbers[i], *it->self());
		it = it->next();
	}

	list.sort();
	it = list.first();
	for (int i = 1; i <= SIZE; i++) {
		CHECK_EQ(i, *it->self());
		it = it->next();
	}

	for (SelfList<int> &element : elements) {
		element.remove_from_list();
	}
}
} // namespace TestSelfList
