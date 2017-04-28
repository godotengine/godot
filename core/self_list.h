/*************************************************************************/
/*  self_list.h                                                          */
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
#ifndef SELF_LIST_H
#define SELF_LIST_H

#include "typedefs.h"

template <class T>
class SelfList {
public:
	class List {

		SelfList<T> *_first;

	public:
		void add(SelfList<T> *p_elem) {

			ERR_FAIL_COND(p_elem->_root);

			p_elem->_root = this;
			p_elem->_next = _first;
			p_elem->_prev = NULL;
			if (_first)
				_first->_prev = p_elem;
			_first = p_elem;
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

			p_elem->_next = NULL;
			p_elem->_prev = NULL;
			p_elem->_root = NULL;
		}

		_FORCE_INLINE_ SelfList<T> *first() { return _first; }
		_FORCE_INLINE_ const SelfList<T> *first() const { return _first; }
		_FORCE_INLINE_ List() { _first = NULL; }
		_FORCE_INLINE_ ~List() { ERR_FAIL_COND(_first != NULL); }
	};

private:
	List *_root;
	T *_self;
	SelfList<T> *_next;
	SelfList<T> *_prev;

public:
	_FORCE_INLINE_ bool in_list() const { return _root; }
	_FORCE_INLINE_ SelfList<T> *next() { return _next; }
	_FORCE_INLINE_ SelfList<T> *prev() { return _prev; }
	_FORCE_INLINE_ const SelfList<T> *next() const { return _next; }
	_FORCE_INLINE_ const SelfList<T> *prev() const { return _prev; }
	_FORCE_INLINE_ T *self() const { return _self; }

	_FORCE_INLINE_ SelfList(T *p_self) {

		_self = p_self;
		_next = NULL;
		_prev = NULL;
		_root = NULL;
	}

	_FORCE_INLINE_ ~SelfList() {

		if (_root)
			_root->remove(this);
	}
};

#endif // SELF_LIST_H
