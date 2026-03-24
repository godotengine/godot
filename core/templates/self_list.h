/**************************************************************************/
/*  self_list.h                                                           */
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

#include "core/error/error_macros.h"
#include "core/templates/local_vector.h"
#include "core/templates/sort_list.h"
#include "core/typedefs.h"

template <typename T>
class SelfList {
public:
	class List {
		SelfList<T> *_first = nullptr;
		SelfList<T> *_last = nullptr;

	public:
		void add(SelfList<T> *p_elem) {
			ERR_FAIL_COND(p_elem->_root);

			p_elem->_root = this;
			p_elem->_next = _first;
			p_elem->_prev = nullptr;

			if (_first) {
				_first->_prev = p_elem;

			} else {
				_last = p_elem;
			}

			_first = p_elem;
		}

		void add_last(SelfList<T> *p_elem) {
			ERR_FAIL_COND(p_elem->_root);

			p_elem->_root = this;
			p_elem->_next = nullptr;
			p_elem->_prev = _last;

			if (_last) {
				_last->_next = p_elem;

			} else {
				_first = p_elem;
			}

			_last = p_elem;
		}

		void remove(SelfList<T> *p_elem) {
			ERR_FAIL_COND(p_elem->_root != this);
			if (p_elem->_next) {
				p_elem->_next->_prev = p_elem->_prev;
			}

			if (p_elem->_prev) {
				p_elem->_prev->_next = p_elem->_next;
			}

			if (_first == p_elem) {
				_first = p_elem->_next;
			}

			if (_last == p_elem) {
				_last = p_elem->_prev;
			}

			p_elem->_next = nullptr;
			p_elem->_prev = nullptr;
			p_elem->_root = nullptr;
		}

		void clear() {
			while (_first) {
				remove(_first);
			}
		}

		void sort() {
			sort_custom<Comparator<T>>();
		}

		template <typename C>
		void sort_custom() {
			if (_first == _last) {
				return;
			}

			// Collect into a flat array, sort, then relink.
			// This avoids SortList member-pointer template args which break on MSVC.
			int count = 0;
			for (SelfList<T> *e = _first; e; e = e->_next) {
				count++;
			}
			if (count < 2) {
				return;
			}
			LocalVector<SelfList<T> *> elems;
			elems.resize(count);
			int idx = 0;
			for (SelfList<T> *e = _first; e; e = e->_next) {
				elems[idx++] = e;
			}

			// Sort using a simple insertion sort to avoid nested-struct issues on MSVC.
			C compare;
			for (int i = 1; i < count; i++) {
				SelfList<T> *key = elems[i];
				int j = i - 1;
				while (j >= 0 && compare(*key->_self, *elems[j]->_self)) {
					elems[j + 1] = elems[j];
					j--;
				}
				elems[j + 1] = key;
			}

			// Relink the list.
			_first = elems[0];
			_last = elems[count - 1];
			for (int i = 0; i < count; i++) {
				elems[i]->_prev = (i > 0) ? elems[i - 1] : nullptr;
				elems[i]->_next = (i < count - 1) ? elems[i + 1] : nullptr;
			}
		}

		_FORCE_INLINE_ SelfList<T> *first() { return _first; }
		_FORCE_INLINE_ const SelfList<T> *first() const { return _first; }

		// Forbid copying, which has broken behavior.
		void operator=(const List &) = delete;

		_FORCE_INLINE_ ~List() {
			// A self list must be empty on destruction.
			DEV_ASSERT(_first == nullptr);
		}
	};

public:
	// These are public to allow SortList template member pointer access on MSVC.
	List *_root = nullptr;
	T *_self = nullptr;
	SelfList<T> *_next = nullptr;
	SelfList<T> *_prev = nullptr;

	_FORCE_INLINE_ bool in_list() const { return _root; }
	_FORCE_INLINE_ void remove_from_list() {
		if (_root) {
			_root->remove(this);
		}
	}
	_FORCE_INLINE_ SelfList<T> *next() { return _next; }
	_FORCE_INLINE_ SelfList<T> *prev() { return _prev; }
	_FORCE_INLINE_ const SelfList<T> *next() const { return _next; }
	_FORCE_INLINE_ const SelfList<T> *prev() const { return _prev; }
	_FORCE_INLINE_ T *self() const { return _self; }

	// Forbid copying, which has broken behavior.
	void operator=(const SelfList<T> &) = delete;

	_FORCE_INLINE_ SelfList(T *p_self) {
		_self = p_self;
	}

	_FORCE_INLINE_ ~SelfList() {
		if (_root) {
			_root->remove(this);
		}
	}
};
