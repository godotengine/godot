/**************************************************************************/
/*  test_sort_array.h                                                     */
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

#ifndef TEST_SORT_ARRAY_H
#define TEST_SORT_ARRAY_H

#include "core/templates/sort_array.h"
#include "core/templates/vector.h"

#include "tests/test_macros.h"

namespace TestSortArray {

template <class T>
struct GreaterThan {
	bool operator()(const T &p_a, const T &p_b) const { return p_a > p_b; }
};

TEST_CASE("[SortArray] Array sorting") {
	SUBCASE("Sort in increasing order") {
		Vector<int> vector = { 5, 6, 8, 6, 3, 1 };
		SortArray<int> sorter;

		SUBCASE("Sort the whole array with sort()") {
			sorter.sort(vector.ptrw(), vector.size());
			CHECK(vector == Vector{ 1, 3, 5, 6, 6, 8 });
		}

		SUBCASE("Sort the whole array with sort_range()") {
			sorter.sort_range(0, vector.size(), vector.ptrw());
			CHECK(vector == Vector{ 1, 3, 5, 6, 6, 8 });
		}

		SUBCASE("Sort a subset of the array") {
			sorter.sort_range(1, 5, vector.ptrw());
			CHECK(vector == Vector{ 5, 3, 6, 6, 8, 1 });
		}
	}

	SUBCASE("Sort in decreasing order with a custom comparison") {
		Vector<int> vector = { 5, 6, 8, 6, 3, 1 };
		SortArray<int, GreaterThan<int>> sorter;

		SUBCASE("Sort the whole array with sort()") {
			sorter.sort(vector.ptrw(), vector.size());
			CHECK(vector == Vector{ 8, 6, 6, 5, 3, 1 });
		}

		SUBCASE("Sort the whole array with sort_range()") {
			sorter.sort_range(0, vector.size(), vector.ptrw());
			CHECK(vector == Vector{ 8, 6, 6, 5, 3, 1 });
		}

		SUBCASE("Sort a subset of the array") {
			sorter.sort_range(1, 5, vector.ptrw());
			CHECK(vector == Vector{ 5, 8, 6, 6, 3, 1 });
		}
	}

	SUBCASE("Do nothing on an empty array") {
		Vector<int> vector = { 5, 6 };
		SortArray<int> sorter;

		sorter.sort_range(0, 0, vector.ptrw());
		CHECK(vector == Vector{ 5, 6 });
	}
}

TEST_CASE("[SortArray] Sort the nth element") {
	SUBCASE("Sort in increasing order") {
		Vector<int> vector = { 5, 6, 7, 8, 3, 1 };
		SortArray<int> sorter;

		SUBCASE("Sort the first element") {
			sorter.nth_element(0, vector.size(), 0, vector.ptrw());
			CHECK(vector[0] == 1);
		}

		SUBCASE("Sort the third element") {
			sorter.nth_element(0, vector.size(), 2, vector.ptrw());
			CHECK(vector[2] == 5);
		}

		SUBCASE("Sort the last element") {
			sorter.nth_element(0, vector.size(), vector.size() - 1, vector.ptrw());
			CHECK(vector[vector.size() - 1] == 8);
		}

		SUBCASE("Sort in a subset of the array") {
			SUBCASE("Sort the first element") {
				sorter.nth_element(1, 5, 0, vector.ptrw());
				CHECK(vector[1] == 3);
			}

			SUBCASE("Sort the third element") {
				sorter.nth_element(1, 5, 2, vector.ptrw());
				CHECK(vector[3] == 7);
			}

			SUBCASE("Sort the last element") {
				sorter.nth_element(1, 5, 3, vector.ptrw());
				CHECK(vector[4] == 8);
			}
		}
	}

	SUBCASE("Sort in decreasing order with a custom comparison") {
		Vector<int> vector = { 5, 6, 7, 8, 3, 1 };
		SortArray<int, GreaterThan<int>> sorter;

		SUBCASE("Sort the first element") {
			sorter.nth_element(0, vector.size(), 0, vector.ptrw());
			CHECK(vector[0] == 8);
		}

		SUBCASE("Sort the third element") {
			sorter.nth_element(0, vector.size(), 2, vector.ptrw());
			CHECK(vector[2] == 6);
		}

		SUBCASE("Sort the last element") {
			sorter.nth_element(0, vector.size(), vector.size() - 1, vector.ptrw());
			CHECK(vector[vector.size() - 1] == 1);
		}

		SUBCASE("Sort in a subset of the array") {
			SUBCASE("Sort the first element") {
				sorter.nth_element(1, 5, 0, vector.ptrw());
				CHECK(vector[1] == 8);
			}

			SUBCASE("Sort the third element") {
				sorter.nth_element(1, 5, 2, vector.ptrw());
				CHECK(vector[3] == 6);
			}

			SUBCASE("Sort the last element") {
				sorter.nth_element(1, 5, 3, vector.ptrw());
				CHECK(vector[4] == 3);
			}
		}
	}

	SUBCASE("Do nothing on an empty array") {
		Vector<int> vector = { 5, 6 };
		SortArray<int> sorter;

		sorter.nth_element(0, 0, 0, vector.ptrw());
		CHECK(vector == Vector{ 5, 6 });
	}
}

template <class T, class Comp = _DefaultComparator<T>>
static Vector<T> push_pop_vector(const Vector<T> &p_elements) {
	SortArray<T, Comp> sorter;

	Vector<T> vector;
	for (const T &n : p_elements) {
		vector.push_back(n);
		sorter.push_heap(0, vector.size(), vector.ptrw());
	}

	Vector<T> sorted_vector;
	while (!vector.is_empty()) {
		sorter.pop_heap(0, vector.size(), vector.ptrw());
		sorted_vector.push_back(vector[vector.size() - 1]);
		vector.remove_at(vector.size() - 1);
	}

	return sorted_vector;
}

TEST_CASE("[SortArray] Max-heap push/pop") {
	Vector<int> sorted_vector = push_pop_vector<int>(Vector{ 4, 2, 3, 7, 6, 2 });

	CHECK(sorted_vector == Vector{ 7, 6, 4, 3, 2, 2 });
}

TEST_CASE("[SortArray] Min-heap push/pop") {
	Vector<int> sorted_vector =
			push_pop_vector<int, GreaterThan<int>>(Vector{ 4, 2, 3, 7, 6, 2 });

	CHECK(sorted_vector == Vector{ 2, 2, 3, 4, 6, 7 });
}

TEST_CASE("[SortArray] Sort heap element up after value update") {
	Vector<int> vector{ 1, 5, 4, 2, 3 };
	Vector<int> heap;
	SortArray<int> sorter;

	for (int number : vector) {
		heap.push_back(number);
		sorter.push_heap(0, heap.size(), heap.ptrw());
	}

	int idx = heap.find(3);
	heap.set(idx, 7);
	sorter.push_heap(0, idx + 1, heap.ptrw());

	Vector<int> sorted_vector;
	while (!heap.is_empty()) {
		sorter.pop_heap(0, heap.size(), heap.ptrw());
		sorted_vector.push_back(heap[heap.size() - 1]);
		heap.remove_at(heap.size() - 1);
	}

	CHECK(sorted_vector == Vector{ 7, 5, 4, 2, 1 });
}

} // namespace TestSortArray

#endif // TEST_SORT_ARRAY_H
