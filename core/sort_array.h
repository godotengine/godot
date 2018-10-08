/*************************************************************************/
/*  sort_array.h                                                         */
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

#ifndef SORT_ARRAY_H
#define SORT_ARRAY_H

#include "core/typedefs.h"

#define ERR_BAD_COMPARE(cond)                                         \
	if (unlikely(cond)) {                                             \
		ERR_PRINT("bad comparison function; sorting will be broken"); \
		break;                                                        \
	}

template <class T>
struct _DefaultComparator {
	_FORCE_INLINE_ bool operator()(const T &a, const T &b) const { return (a < b); }
};

#ifdef DEBUG_ENABLED
#define SORT_ARRAY_VALIDATE_ENABLED true
#else
#define SORT_ARRAY_VALIDATE_ENABLED false
#endif

template <class T, class Comparator = _DefaultComparator<T>, bool Validate = SORT_ARRAY_VALIDATE_ENABLED>
class SortArray {

	enum {
		INTROSORT_THRESHOLD = 16
	};

	inline const T &_median_of_3(const T &a, const T &b, const T &c) const {
		if (compare(a, b))
			if (compare(b, c))
				return b;
			else if (compare(a, c))
				return c;
			else
				return a;
		else if (compare(a, c))
			return a;
		else if (compare(b, c))
			return c;
		else
			return b;
	}

	inline int _bitlog(int n) const {
		int k;
		for (k = 0; n != 1; n >>= 1)
			++k;
		return k;
	}

	/* Heapsort */

	inline void _make_heap(int p_first, int p_last, T *p_array) const {
		if (p_last <= p_first)
			return;

		int relative_last = p_last - p_first; // Find p_last's index relative to p_first
		int parent = (relative_last - 1) / 2;
		while (parent >= 0) {
			_heapify(p_first, relative_last, parent, p_array);
			parent--;
		}
	}

	inline void _heapify(int p_first, int p_relative_last, int p_hole_index, T *p_array) const {
		T val = p_array[p_first + p_hole_index];
		int top_index = p_hole_index;
		int parent = p_hole_index;
		int child = 2 * parent + 2;

		while (child <= p_relative_last) {
			if (compare(p_array[p_first + child], p_array[p_first + (child - 1)]))
				child--;

			p_array[p_first + parent] = p_array[p_first + child];
			parent = child;
			child = 2 * parent + 2;
		}

		if (child == p_relative_last + 1) {
			p_array[p_first + parent] = p_array[p_first + p_relative_last];
			p_hole_index = p_relative_last;
		} else {
			p_hole_index = parent;
			parent = (p_hole_index - 1) / 2;
		}

		while (p_hole_index > top_index && compare(p_array[p_first + parent], val)) {
			p_array[p_first + p_hole_index] = p_array[p_first + parent];
			p_hole_index = parent;
			parent = (p_hole_index - 1) / 2;
		}

		p_array[p_first + p_hole_index] = val;
	}

	inline void _pop_heap(int p_first, int p_last, int p_swap, T *p_array) const {
		SWAP(p_array[p_swap], p_array[p_first]);
		_heapify(p_first, p_last - p_first, 0, p_array);
	}

	inline void _heap_sort(int p_first, int p_last, T *p_array) const {
		_make_heap(p_first, p_last, p_array);
		while (p_last > p_first) {
			_pop_heap(p_first, p_last - 1, p_last, p_array);
			p_last--;
		}
	}

	// Find and sort the first nth elements of the array
	inline void _partial_heap_sort(int p_first, int p_last, int p_nth, T *p_array) const {
		_make_heap(p_first, p_nth, p_array);
		for (int i = p_nth + 1; i <= p_last; i++)
			if (compare(p_array[i], p_array[p_first]))
				_pop_heap(p_first, p_nth, i, p_array);

		while (p_nth > p_first) {
			_pop_heap(p_first, p_nth - 1, p_nth, p_array);
			p_nth--;
		}
	}

	// Find the nth element, if the array were sorted, and place it at the nth position.
	// Also partially sorts the array.
	inline void _partial_heap_select(int p_first, int p_last, int p_nth, T *p_array) const {
		_make_heap(p_first, p_nth, p_array);
		for (int i = p_nth + 1; i <= p_last; i++)
			if (compare(p_array[i], p_array[p_first]))
				_pop_heap(p_first, p_nth, i, p_array);

		SWAP(p_first, p_nth);
	}

	/* Quicksort */

	inline int _partition(int p_first, int p_last, T p_pivot, T *p_array) const {
		const int unmodified_first = p_first;
		const int unmodified_last = p_last;

		while (true) {
			while (compare(p_array[p_first], p_pivot)) {
				if (Validate) {
					ERR_BAD_COMPARE(p_first == unmodified_last)
				}
				p_first++;
			}

			while (compare(p_pivot, p_array[p_last])) {
				if (Validate) {
					ERR_BAD_COMPARE(p_last == unmodified_first)
				}
				p_last--;
			}

			if (p_first >= p_last)
				return p_first;

			SWAP(p_array[p_first], p_array[p_last]);
			p_first++;
			p_last--;
		}
	}

	/* Introsort and Introselect */

	inline void _introsort(int p_first, int p_last, T *p_array, int p_max_depth) const {
		while (p_last - p_first > INTROSORT_THRESHOLD) { // (p_last - p_first) is equal to (len - 1)
			if (p_max_depth == 0) {
				_heap_sort(p_first, p_last, p_array);
				return;
			}

			p_max_depth--;

			int cut = _partition(
					p_first,
					p_last,
					_median_of_3(
							p_array[p_first],
							p_array[(p_first + p_last) / 2],
							p_array[p_last]),
					p_array);

			_introsort(cut, p_last, p_array, p_max_depth);
			p_last = cut - 1;
		}
	}

	inline void _introselect(int p_first, int p_last, int p_nth, T *p_array, int p_max_depth) const {
		while (p_last - p_first > INTROSORT_THRESHOLD) { // (p_last - p_first) is equal to (len - 1)
			if (p_max_depth == 0) {
				_partial_heap_select(p_first, p_nth + 1, p_last, p_array);
				return;
			}

			p_max_depth--;

			int cut = _partition(
					p_first,
					p_last,
					_median_of_3(
							p_array[p_first],
							p_array[(p_first + p_last) / 2],
							p_array[p_last]),
					p_array);

			if (cut <= p_nth)
				p_first = cut;
			else
				p_last = cut - 1;
		}

		_insertion_sort(p_first, p_last, p_array);
	}

	/* Insertion Sort */

	inline void _insertion_sort(int p_first, int p_last, T *p_array) const {
		if (p_first >= p_last)
			return;

		int index = p_first + 1;
		int sorted_count = 1;
		int compare_index;
		while (index <= p_last) { // Sort array from p_first to p_last inclusive
			T val = p_array[index];
			compare_index = index - 1;

			for (int i = 0; i < sorted_count; i++) {
				if (compare(val, p_array[compare_index])) {
					p_array[compare_index + 1] = p_array[compare_index];
					compare_index--;
				} else {
					break;
				}
			}

			if (compare_index != index - 1)
				p_array[compare_index + 1] = val;

			index++;
			sorted_count++;
		}
	}

public:
	Comparator compare;

	inline void sort(T *p_array, int p_len) const {
		if (p_len < 2)
			return;

		_introsort(0, p_len - 1, p_array, _bitlog(p_len) * 2);
		_insertion_sort(0, p_len - 1, p_array);
	}

	inline void nth_element(T *p_array, int p_len, int p_nth) const {
		if (p_len < 2 || p_nth >= p_len)
			return;

		_introselect(0, p_len - 1, p_nth, p_array, _bitlog(p_len) * 2);
	}
};

#endif // SORT_ARRAY_H
