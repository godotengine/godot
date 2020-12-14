/*************************************************************************/
/*  test_array.h                                                         */
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

#include "core/object/class_db.h"
#include "core/object/script_language.h"
#include "core/templates/hashfuncs.h"
#include "core/templates/vector.h"
#include "core/variant/array.h"
#include "core/variant/container_type_validate.h"
#include "core/variant/variant.h"
#include "tests/test_macros.h"

namespace TestArray {

TEST_CASE("[Array] Size, Clear, and Empty") {
	Array arr;
	CHECK(arr.size() == 0);
	CHECK(arr.empty());
	arr.push_back(1);
	CHECK(arr.size() == 1);
	arr.clear();
	CHECK(arr.empty());
	CHECK(arr.size() == 0);
}

TEST_CASE("[Array] Assignment and Compare Operators") {
	Array arr1;
	Array arr2;
	arr1.push_back(1);
	CHECK(arr1 != arr2);
	CHECK(arr1 > arr2);
	CHECK(arr1 >= arr2);
	arr2.push_back(2);
	CHECK(arr1 != arr2);
	CHECK(arr1 < arr2);
	CHECK(arr1 <= arr2);
	CHECK(arr2 > arr1);
	CHECK(arr2 >= arr1);
	Array arr3 = arr2;
	CHECK(arr3 == arr2);
}

TEST_CASE("[Array] append_array") {
	Array arr1;
	Array arr2;
	arr1.push_back(1);
	arr1.append_array(arr2);
	CHECK(arr1.size() == 1);
	arr2.push_back(2);
	arr1.append_array(arr2);
	CHECK(arr1.size() == 2);
	CHECK(int(arr1[0]) == 1);
	CHECK(int(arr1[1]) == 2);
}

TEST_CASE("[Array] Resize, Insert, and Erase") {
	Array arr;
	arr.resize(2);
	CHECK(arr.size() == 2);
	arr.insert(0, 1);
	CHECK(int(arr[0]) == 1);
	arr.insert(0, 2);
	CHECK(int(arr[0]) == 2);
	arr.erase(2);
	CHECK(int(arr[0]) == 1);
}

TEST_CASE("[Array] Front and Back") {
	Array arr;
	arr.push_back(1);
	CHECK(int(arr.front()) == 1);
	CHECK(int(arr.back()) == 1);
	arr.push_back(3);
	CHECK(int(arr.front()) == 1);
	CHECK(int(arr.back()) == 3);
}

TEST_CASE("[Array] Has and Count") {
	Array arr;
	arr.push_back(1);
	arr.push_back(1);
	CHECK(arr.has(1));
	CHECK(arr.has(2) == false);
	CHECK(arr.count(1) == 2);
	CHECK(arr.count(2) == 0);
}

TEST_CASE("[Array] Remove") {
	Array arr;
	arr.push_back(1);
	arr.push_back(2);
	arr.remove(0);
	CHECK(arr.size() == 1);
	CHECK(int(arr[0]) == 2);
	arr.remove(0);
	arr.remove(0);
	CHECK(arr.size() == 0);
}

// test get
TEST_CASE("[Array] Get") {
	Array arr;
	arr.push_back(1);
	CHECK(int(arr.get(0)) == 1);
}

TEST_CASE("[Array] Sort") {
	Array arr;

	arr.push_back(1);
	arr.push_back(2);
	arr.push_back(3);
	arr.push_back(4);
	arr.sort();
	int val = 1;
	for (int i = 0; i < arr.size(); i++) {
		CHECK(int(arr[i]) == val);
		val++;
	}
}

TEST_CASE("[Array] Push_front, Pop_front, Pop_back") {
	Array arr;
	arr.push_front(1);
	arr.push_front(2);
	CHECK(int(arr[0]) == 2);
	arr.pop_front();
	CHECK(int(arr[0]) == 1);
	CHECK(arr.size() == 1);
	arr.push_front(2);
	arr.push_front(3);
	arr.pop_back();
	CHECK(int(arr[1]) == 2);
	CHECK(arr.size() == 2);
}

TEST_CASE("[Array] Max and Min") {
	Array arr;
	arr.push_back(3);
	arr.push_front(4);
	arr.push_back(5);
	arr.push_back(2);
	int max = int(arr.max());
	int min = int(arr.min());
	CHECK(max == 5);
	CHECK(min == 2);
}

} // namespace TestArray
#endif // TEST_ARRAY_H
