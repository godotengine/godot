/**************************************************************************/
/*  sort_list.h                                                           */
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

#include "core/typedefs.h"

template <typename Element, typename T, T Element::*value, Element *Element::*prev, Element *Element::*next, typename Comparator = Comparator<T>>
class SortList {
public:
	Comparator compare;

	void sort(Element *&r_head, Element *&r_tail) {
		Element *sorted_until;
		if (_is_sorted(r_head, r_tail, sorted_until)) {
			return;
		}

		// In case we're sorting only part of a larger list.
		Element *head_prev = r_head->*prev;
		r_head->*prev = nullptr;
		Element *tail_next = r_tail->*next;
		r_tail->*next = nullptr;

		// Sort unsorted section and merge.
		Element *head2 = sorted_until->*next;
		_split(sorted_until, head2);
		_merge_sort(head2, r_tail);
		_merge(r_head, sorted_until, head2, r_tail, r_head, r_tail);

		// Reconnect to larger list if needed.
		if (head_prev) {
			_connect(head_prev, r_head);
		}
		if (tail_next) {
			_connect(r_tail, tail_next);
		}
	}

private:
	bool _is_sorted(Element *p_head, Element *p_tail, Element *&r_sorted_until) {
		r_sorted_until = p_head;
		while (r_sorted_until != p_tail) {
			if (compare(r_sorted_until->*next->*value, r_sorted_until->*value)) {
				return false;
			}

			r_sorted_until = r_sorted_until->*next;
		}

		return true;
	}

	void _merge_sort(Element *&r_head, Element *&r_tail) {
		if (r_head == r_tail) {
			return;
		}

		Element *tail1 = _get_mid(r_head);
		Element *head2 = tail1->*next;
		_split(tail1, head2);

		_merge_sort(r_head, tail1);
		_merge_sort(head2, r_tail);
		_merge(r_head, tail1, head2, r_tail, r_head, r_tail);
	}

	void _merge(
			Element *p_head1, Element *p_tail1,
			Element *p_head2, Element *p_tail2,
			Element *&r_head, Element *&r_tail) {
		if (compare(p_head2->*value, p_head1->*value)) {
			r_head = p_head2;
			p_head2 = p_head2->*next;
		} else {
			r_head = p_head1;
			p_head1 = p_head1->*next;
		}

		Element *curr = r_head;
		while (p_head1 && p_head2) {
			if (compare(p_head2->*value, p_head1->*value)) {
				_connect(curr, p_head2);
				p_head2 = p_head2->*next;
			} else {
				_connect(curr, p_head1);
				p_head1 = p_head1->*next;
			}
			curr = curr->*next;
		}

		if (p_head1) {
			_connect(curr, p_head1);
			r_tail = p_tail1;
		} else {
			_connect(curr, p_head2);
			r_tail = p_tail2;
		}
	}

	Element *_get_mid(Element *p_head) {
		Element *end = p_head;
		Element *mid = p_head;
		while (end->*next && end->*next->*next) {
			end = end->*next->*next;
			mid = mid->*next;
		}

		return mid;
	}

	_FORCE_INLINE_ void _connect(Element *p_a, Element *p_b) {
		p_a->*next = p_b;
		p_b->*prev = p_a;
	}

	_FORCE_INLINE_ void _split(Element *p_a, Element *p_b) {
		p_a->*next = nullptr;
		p_b->*prev = nullptr;
	}
};
