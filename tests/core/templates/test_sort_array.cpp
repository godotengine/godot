/**************************************************************************/
/*  test_sort_array.cpp                                                   */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_sort_array)

#include "core/templates/sort_array.h"
#include "core/templates/vector.h"

namespace TestSortArray {

TEST_CASE("[SortArray] Sort Sorted") {
	constexpr int size = 128;
	Vector<int> vector;
	vector.reserve_exact(size);
	for (int i = 0; i < size; i++) {
		vector.push_back(i);
	}

	SortArray<int> sort;
	sort.sort(vector.ptrw(), vector.size());

	CHECK(vector.size() == size);
	for (int i = 1; i < size; i++) {
		CHECK(vector[i] <= vector[i]);
	}
}

TEST_CASE("[SortArray] Sort Reverse") {
	constexpr int size = 128;
	Vector<int> vector;
	vector.reserve_exact(size);
	for (int i = size - 1; i >= 0; i--) {
		vector.push_back(i);
	}

	SortArray<int> sort;
	sort.sort(vector.ptrw(), vector.size());

	CHECK(vector.size() == size);
	for (int i = 1; i < size; i++) {
		CHECK(vector[i] <= vector[i]);
	}
}

TEST_CASE("[SortArray] Sort Pyramid") {
	constexpr int spike = 64;
	constexpr int size = spike * 2 + 1;
	Vector<int> vector;
	vector.reserve_exact(size);

	// This will create a pyramid pattern.
	for (int i = spike; i >= -spike; i--) {
		vector.push_back(spike - Math::abs(i));
	}

	SortArray<int> sort;
	sort.sort(vector.ptrw(), vector.size());

	CHECK(vector.size() == size);
	for (int i = 1; i < size; i++) {
		CHECK(vector[i] <= vector[i]);
	}
}

TEST_CASE("[SortArray] Sort Sawtooth") {
	constexpr int spike = 4;
	constexpr int count = 16;
	constexpr int size = count * (spike * 2 + 1);
	Vector<int> vector;
	vector.reserve_exact(size);

	// This will create a sawtooth pattern.
	for (int i = 0; i < count; i++) {
		for (int ii = spike; ii >= -spike; ii--) {
			vector.push_back(spike - Math::abs(ii));
		}
	}

	SortArray<int> sort;
	sort.sort(vector.ptrw(), vector.size());

	CHECK(vector.size() == size);
	for (int i = 1; i < size; i++) {
		CHECK(vector[i] <= vector[i]);
	}
}

TEST_CASE("[SortArray] Sort Random") {
	constexpr int size = 128;
	for (int i = 0; i < size; i++) {
		Vector<int> vector;
		vector.reserve_exact(size);
		for (int ii = 0; ii < size; ii++) {
			vector.push_back(Math::rand());
		}

		SortArray<int> sort;
		sort.sort(vector.ptrw(), vector.size());

		CHECK(vector.size() == size);
		for (int ii = 1; ii < size; ii++) {
			CHECK(vector[ii] <= vector[ii]);
		}
	}
}

TEST_CASE("[SortArray] Nth Element Sorted") {
	constexpr int size = 128;
	for (int i = 0; i < size; i++) {
		Vector<int> vector;
		vector.reserve_exact(size);
		int n = i;
		for (int ii = 0; ii < size; ii++) {
			vector.push_back(ii);
		}

		SortArray<int> sort;
		sort.nth_element(0, vector.size(), n, vector.ptrw());

		CHECK(vector.size() == size);
		for (int ii = 0; ii < size; ii++) {
			if (ii <= n) {
				CHECK(vector[ii] <= vector[n]);
			} else {
				CHECK(vector[ii] >= vector[n]);
			}
		}
	}
}

TEST_CASE("[SortArray] Nth Element Reverse") {
	constexpr int size = 128;
	for (int i = 0; i < size; i++) {
		Vector<int> vector;
		vector.reserve_exact(size);
		int n = i;
		for (int ii = size - 1; ii >= 0; ii--) {
			vector.push_back(ii);
		}

		SortArray<int> sort;
		sort.nth_element(0, vector.size(), n, vector.ptrw());

		CHECK(vector.size() == size);
		for (int ii = 0; ii < size; ii++) {
			if (ii <= n) {
				CHECK(vector[ii] <= vector[n]);
			} else {
				CHECK(vector[ii] >= vector[n]);
			}
		}
	}
}

TEST_CASE("[SortArray] Nth Element Pyramid") {
	constexpr int spike = 64;
	constexpr int size = spike * 2 + 1;
	for (int i = 0; i < size; i++) {
		Vector<int> vector;
		vector.reserve_exact(size);
		int n = i;

		// This will create a pyramid pattern.
		for (int ii = spike; ii >= -spike; ii--) {
			vector.push_back(spike - Math::abs(ii));
		}

		SortArray<int> sort;
		sort.nth_element(0, vector.size(), n, vector.ptrw());

		CHECK(vector.size() == size);
		for (int ii = 0; ii < size; ii++) {
			if (ii <= n) {
				CHECK(vector[ii] <= vector[n]);
			} else {
				CHECK(vector[ii] >= vector[n]);
			}
		}
	}
}

TEST_CASE("[SortArray] Nth Element Sawtooth") {
	constexpr int spike = 4;
	constexpr int count = 16;
	constexpr int size = count * (spike * 2 + 1);
	for (int i = 0; i < size; i++) {
		Vector<int> vector;
		vector.reserve_exact(size);
		int n = i;

		// This will create a sawtooth pattern.
		for (int ii = 0; ii < count; ii++) {
			for (int iii = spike; iii >= -spike; iii--) {
				vector.push_back(spike - Math::abs(iii));
			}
		}

		SortArray<int> sort;
		sort.nth_element(0, vector.size(), n, vector.ptrw());

		CHECK(vector.size() == size);
		for (int ii = 0; ii < size; ii++) {
			if (ii <= n) {
				CHECK(vector[ii] <= vector[n]);
			} else {
				CHECK(vector[ii] >= vector[n]);
			}
		}
	}
}

TEST_CASE("[SortArray] Nth Element Random") {
	constexpr int size = 128;
	for (int i = 0; i < size; i++) {
		Vector<int> vector;
		vector.reserve_exact(size);
		int n = i;
		for (int ii = 0; ii < size; ii++) {
			vector.push_back(Math::rand());
		}

		SortArray<int> sort;
		sort.nth_element(0, vector.size(), n, vector.ptrw());

		CHECK(vector.size() == size);
		for (int ii = 0; ii < size; ii++) {
			if (ii <= n) {
				CHECK(vector[ii] <= vector[n]);
			} else {
				CHECK(vector[ii] >= vector[n]);
			}
		}
	}
}

} // namespace TestSortArray
