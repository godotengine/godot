/*************************************************************************/
/*  test_array.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_ARRAY_H
#define TEST_ARRAY_H

#include "core/variant/array.h"
#include "core/string/print_string.h"
#include "tests/test_macros.h"

#include "thirdparty/doctest/doctest.h"

namespace TestArray {
TEST_CASE("Array < Single Element") {
	const Array a1 = [0.8]
	const Array a2 = [9.5]

	CHECK_MESSAGE(
			a1 < a2,
			"Array with a smaller item at index 0 should be smaller");
}

TEST_CASE("Array < More Elements") {
	const Array a1 = [1, 1]
	const Array a2 = [1, 2]

	CHECK_MESSAGE(
			a1 < a2,
			"Array with a smaller item at index 1 should be smaller");
}

TEST_CASE("Array < More Elements 2") {
	const Array a1 = [1, 2, 3]
	const Array a2 = [2, 3, 4]

	CHECK_MESSAGE(
			a1 < a2,
			"Array with a smaller item at index 1 should be smaller");
}

TEST_CASE("Array < More Elements 3") {
	const Array a1 = ["b", 2, 3]
	const Array a2 = ["a", 2, 3]

	CHECK_MESSAGE(
			a2 < a1,
			"Array with a smaller item at index 1 should be smaller");
}

TEST_CASE("Array < Length") {
	const Array a1 = [1]
	const Array a2 = [1, 0]

	CHECK_MESSAGE(
			a1 < a2,
			"Array with a shorter length should be smaller");
}
} // namespace TestArray

#endif // TEST_Array_H
