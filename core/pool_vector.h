/**************************************************************************/
/*  pool_vector.h                                                         */
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

#ifndef POOL_VECTOR_H
#define POOL_VECTOR_H

#include "core/ustring.h"
#include "core/vector.h"

template <class T>
class PoolVector {
	Vector<T> _vec;

public:
	class Read {
		const Vector<T> *_vec = nullptr;

	public:
		const T &operator[](int p_index) const { return (*this->_vec)[p_index]; }
		const T *ptr() const { return _vec ? this->_vec->ptr() : nullptr; }
		void release() {
			_vec = nullptr;
		}

		Read() {}
		Read(const Vector<T> *p_vec) { _vec = p_vec; }
	};

	class Write {
		Vector<T> *_vec = nullptr;

	public:
		T &operator[](int p_index) {
			CRASH_COND(p_index >= _vec->size());
			T &var = ptr()[p_index];
			return var;
		}

		T *ptr() { return _vec ? this->_vec->ptrw() : nullptr; }
		void release() {
			_vec = nullptr;
		}

		Write() {}
		Write(Vector<T> *p_vec) { _vec = p_vec; }
	};

	Read read() const {
		Read r(&_vec);
		return r;
	}
	Write write() {
		Write w(&_vec);
		return w;
	}

	template <class MC>
	void fill_with(const MC &p_mc) {
		int c = p_mc.size();
		resize(c);
		Write w = write();
		int idx = 0;
		for (const typename MC::Element *E = p_mc.front(); E; E = E->next()) {
			w[idx++] = E->get();
		}
	}

	void remove(int p_index) {
		_vec.remove(p_index);
	}

	int find(const T &p_val, int p_from = 0) const {
		return _vec.find(p_val, p_from);
	}

	int rfind(const T &p_val, int p_from = -1) const {
		const int s = size();
		const Read r = read();

		if (p_from < 0) {
			p_from = s + p_from;
		}
		if (p_from < 0 || p_from >= s) {
			p_from = s - 1;
		}

		for (int i = p_from; i >= 0; i--) {
			if (r[i] == p_val) {
				return i;
			}
		}
		return -1;
	}

	int count(const T &p_val) const {
		const int s = size();
		const Read r = read();
		int amount = 0;
		for (int i = 0; i < s; i++) {
			if (r[i] == p_val) {
				amount++;
			}
		}
		return amount;
	}

	bool has(const T &p_val) const {
		return find(p_val) != -1;
	}

	int size() const { return _vec.size(); }
	bool empty() const { return _vec.empty(); }

	T get(int p_index) const { return _vec[p_index]; }
	void set(int p_index, const T &p_val) {
		_vec.set(p_index, p_val);
	}

	void fill(const T &p_val) {
		_vec.fill(p_val);
	}
	void push_back(const T &p_val) {
		_vec.push_back(p_val);
	}
	void append(const T &p_val) { push_back(p_val); }
	void append_array(const PoolVector<T> &p_arr) {
		int ds = p_arr.size();
		if (ds == 0) {
			return;
		}
		int bs = size();
		resize(bs + ds);
		Write w = write();
		Read r = p_arr.read();
		for (int i = 0; i < ds; i++) {
			w[bs + i] = r[i];
		}
	}

	PoolVector<T> subarray(int p_from, int p_to) {
		if (p_from < 0) {
			p_from = size() + p_from;
		}
		if (p_to < 0) {
			p_to = size() + p_to;
		}

		ERR_FAIL_INDEX_V(p_from, size(), PoolVector<T>());
		ERR_FAIL_INDEX_V(p_to, size(), PoolVector<T>());

		PoolVector<T> slice;
		int span = 1 + p_to - p_from;
		slice.resize(span);
		Read r = read();
		Write w = slice.write();
		for (int i = 0; i < span; ++i) {
			w[i] = r[p_from + i];
		}

		return slice;
	}

	Error insert(int p_pos, const T &p_val) {
		int s = size();
		ERR_FAIL_INDEX_V(p_pos, s + 1, ERR_INVALID_PARAMETER);
		resize(s + 1);
		{
			Write w = write();
			for (int i = s; i > p_pos; i--) {
				w[i] = w[i - 1];
			}
			w[p_pos] = p_val;
		}

		return OK;
	}

	String join(String delimiter) const {
		String rs = "";
		int s = size();
		Read r = read();
		for (int i = 0; i < s; i++) {
			rs += r[i] + delimiter;
		}
		rs.erase(rs.length() - delimiter.length(), delimiter.length());
		return rs;
	}

	const T &operator[](int p_index) const { return _vec[p_index]; }

	Error resize(int p_size) {
		return _vec.resize(p_size);
	}
	Error clear() { return resize(0); }

	void invert() { _vec.invert(); }
	void sort();
};

template <class T>
void PoolVector<T>::sort() {
	int len = size();
	if (len == 0) {
		return;
	}

	Write w = write();
	SortArray<T> sorter;
	sorter.sort(w.ptr(), len);
}

#endif // POOL_VECTOR_H
