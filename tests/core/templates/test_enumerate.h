/**************************************************************************/
/*  test_enumerate.h                                                      */
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

#include "core/templates/enumerate.h"
#include "core/templates/vector.h"

#include "tests/test_macros.h"

namespace TestEnumerate {

TEST_CASE("[Enumerate] Enumerate by value") {
	SUBCASE("Vector") {
		Vector<int32_t> vector = { 0, 1, 2, 3, 4 };
		int32_t index = 0;
		for (int32_t item : Enumerate(vector, index)) {
			CHECK_EQ(item, index);
			item = -1;
			CHECK_NE(item, vector[index]);
		}
		CHECK_NE(index, 0);
	}

	SUBCASE("C Array") {
		int32_t carray[] = { 0, 1, 2, 3, 4 };
		int32_t index = 0;
		for (int32_t item : Enumerate(carray, index)) {
			CHECK_EQ(item, index);
			item = -1;
			CHECK_NE(item, carray[index]);
		}
		CHECK_NE(index, 0);
	}
}

TEST_CASE("[Enumerate] Enumerate by reference") {
	SUBCASE("Vector") {
		Vector<int32_t> vector = { 0, 1, 2, 3, 4 };
		Vector<int32_t> vector_zeroed = { 0, 0, 0, 0, 0 };
		int32_t index = 0;
		for (int32_t &item : Enumerate(vector_zeroed, index)) {
			item = index;
		}
		CHECK_NE(index, 0);
		for (int32_t i = 0; i < index; i++) {
			CHECK_EQ(vector[i], vector_zeroed[i]);
		}
	}

	SUBCASE("C Array") {
		int32_t carray[] = { 0, 1, 2, 3, 4 };
		int32_t carray_zeroed[] = { 0, 0, 0, 0, 0 };
		int32_t index = 0;
		for (int32_t &item : Enumerate(carray_zeroed, index)) {
			item = index;
		}
		CHECK_NE(index, 0);
		for (int32_t i = 0; i < index; i++) {
			CHECK_EQ(carray[i], carray_zeroed[i]);
		}
	}
}

} // namespace TestEnumerate
