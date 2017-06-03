/*************************************************************************/
/*  sort.h                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef SORT_H
#define SORT_H

#include "typedefs.h"
/**
	@author ,,, <red@lunatea>
*/

template <class T>
struct _DefaultComparator {

	inline bool operator()(const T &a, const T &b) const { return (a < b); }
};

template <class T, class Comparator = _DefaultComparator<T> >
class SortArray {

	enum {

		INTROSORT_TRESHOLD = 16
	};

public:
	Comparator compare;

	inline const T &median_of_3(const T &a, const T &b, const T &c) const {

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

	inline int bitlog(int n) const {
		int k;
		for (k = 0; n != 1; n >>= 1)
			++k;
		return k;
	}

	/* Heap / Heapsort functions */

	inline void push_heap(int p_first, int p_hole_idx, int p_top_index, T p_value, T *p_array) const {

		int parent = (p_hole_idx - 1) / 2;
		while (p_hole_idx > p_top_index && compare(p_array[p_first + parent], p_value)) {

			p_array[p_first + p_hole_idx] = p_array[p_first + parent];
			p_hole_idx = parent;
			parent = (p_hole_idx - 1) / 2;
		}
		p_array[p_first + p_hole_idx] = p_value;
	}

	inline void pop_heap(int p_first, int p_last, int p_result, T p_value, T *p_array) const {

		p_array[p_result] = p_array[p_first];
		adjust_heap(p_first, 0, p_last - p_first, p_value, p_array);
	}
	inline void pop_heap(int p_first, int p_last, T *p_array) const {

		pop_heap(p_first, p_last - 1, p_last - 1, p_array[p_last - 1], p_array);
	}

	inline void adjust_heap(int p_first, int p_hole_idx, int p_len, T p_value, T *p_array) const {

		int top_index = p_hole_idx;
		int second_child = 2 * p_hole_idx + 2;

		while (second_child < p_len) {

			if (compare(p_array[p_first + second_child], p_array[p_first + (second_child - 1)]))
				second_child--;

			p_array[p_first + p_hole_idx] = p_array[p_first + second_child];
			p_hole_idx = second_child;
			second_child = 2 * (second_child + 1);
		}

		if (second_child == p_len) {
			p_array[p_first + p_hole_idx] = p_array[p_first + (second_child - 1)];
			p_hole_idx = second_child - 1;
		}
		push_heap(p_first, p_hole_idx, top_index, p_value, p_array);
	}

	inline void sort_heap(int p_first, int p_last, T *p_array) const {

		while (p_last - p_first > 1) {

			pop_heap(p_first, p_last--, p_array);
		}
	}

	inline void make_heap(int p_first, int p_last, T *p_array) const {
		if (p_last - p_first < 2)
			return;
		int len = p_last - p_first;
		int parent = (len - 2) / 2;

		while (true) {
			adjust_heap(p_first, parent, len, p_array[p_first + parent], p_array);
			if (parent == 0)
				return;
			parent--;
		}
	}

	inline void partial_sort(int p_first, int p_last, int p_middle, T *p_array) const {

		make_heap(p_first, p_middle, p_array);
		for (int i = p_middle; i < p_last; i++)
			if (compare(p_array[i], p_array[p_first]))
				pop_heap(p_first, p_middle, i, p_array[i], p_array);
		sort_heap(p_first, p_middle, p_array);
	}

	inline void partial_select(int p_first, int p_last, int p_middle, T *p_array) const {

		make_heap(p_first, p_middle, p_array);
		for (int i = p_middle; i < p_last; i++)
			if (compare(p_array[i], p_array[p_first]))
				pop_heap(p_first, p_middle, i, p_array[i], p_array);
	}

	inline int partitioner(int p_first, int p_last, T p_pivot, T *p_array) const {

		while (true) {
			while (compare(p_array[p_first], p_pivot))
				p_first++;
			p_last--;
			while (compare(p_pivot, p_array[p_last]))
				p_last--;

			if (!(p_first < p_last))
				return p_first;

			SWAP(p_array[p_first], p_array[p_last]);
			p_first++;
		}
	}

	inline void introsort(int p_first, int p_last, T *p_array, int p_max_depth) const {

		while (p_last - p_first > INTROSORT_TRESHOLD) {

			if (p_max_depth == 0) {
				partial_sort(p_first, p_last, p_last, p_array);
				return;
			}

			p_max_depth--;

			int cut = partitioner(
					p_first,
					p_last,
					median_of_3(
							p_array[p_first],
							p_array[p_first + (p_last - p_first) / 2],
							p_array[p_last - 1]),
					p_array);

			introsort(cut, p_last, p_array, p_max_depth);
			p_last = cut;
		}
	}

	inline void introselect(int p_first, int p_nth, int p_last, T *p_array, int p_max_depth) const {

		while (p_last - p_first > 3) {

			if (p_max_depth == 0) {
				partial_select(p_first, p_nth + 1, p_last, p_array);
				SWAP(p_first, p_nth);
				return;
			}

			p_max_depth--;

			int cut = partitioner(
					p_first,
					p_last,
					median_of_3(
							p_array[p_first],
							p_array[p_first + (p_last - p_first) / 2],
							p_array[p_last - 1]),
					p_array);

			if (cut <= p_nth)
				p_first = cut;
			else
				p_last = cut;
		}

		insertion_sort(p_first, p_last, p_array);
	}

	inline void unguarded_linear_insert(int p_last, T p_value, T *p_array) const {

		int next = p_last - 1;
		while (compare(p_value, p_array[next])) {
			p_array[p_last] = p_array[next];
			p_last = next;
			next--;
		}
		p_array[p_last] = p_value;
	}

	inline void linear_insert(int p_first, int p_last, T *p_array) const {

		T val = p_array[p_last];
		if (compare(val, p_array[p_first])) {

			for (int i = p_last; i > p_first; i--)
				p_array[i] = p_array[i - 1];

			p_array[p_first] = val;
		} else
			unguarded_linear_insert(p_last, val, p_array);
	}

	inline void insertion_sort(int p_first, int p_last, T *p_array) const {

		if (p_first == p_last)
			return;
		for (int i = p_first + 1; i != p_last; i++)
			linear_insert(p_first, i, p_array);
	}

	inline void unguarded_insertion_sort(int p_first, int p_last, T *p_array) const {

		for (int i = p_first; i != p_last; i++)
			unguarded_linear_insert(i, p_array[i], p_array);
	}

	inline void final_insertion_sort(int p_first, int p_last, T *p_array) const {

		if (p_last - p_first > INTROSORT_TRESHOLD) {
			insertion_sort(p_first, p_first + INTROSORT_TRESHOLD, p_array);
			unguarded_insertion_sort(p_first + INTROSORT_TRESHOLD, p_last, p_array);
		} else {

			insertion_sort(p_first, p_last, p_array);
		}
	}

	inline void sort_range(int p_first, int p_last, T *p_array) const {

		if (p_first != p_last) {
			introsort(p_first, p_last, p_array, bitlog(p_last - p_first) * 2);
			final_insertion_sort(p_first, p_last, p_array);
		}
	}

	inline void sort(T *p_array, int p_len) const {

		sort_range(0, p_len, p_array);
	}

	inline void nth_element(int p_first, int p_last, int p_nth, T *p_array) const {

		if (p_first == p_last || p_nth == p_last)
			return;
		introselect(p_first, p_nth, p_last, p_array, bitlog(p_last - p_first) * 2);
	}
};

#endif
