/*************************************************************************/
/*  vector.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef VECTOR_H
#define VECTOR_H

/**
 * @class Vector
 * @author Juan Linietsky
 * Vector container. Regular Vector Container. Use with care and for smaller arrays when possible. Use Vector for large arrays.
 */

#include "core/error/error_macros.h"
#include "core/os/memory.h"
#include "core/templates/cowdata.h"
#include "core/templates/search_array.h"
#include "core/templates/sort_array.h"

template <class T>
class VectorWriteProxy {
public:
	_FORCE_INLINE_ T &operator[](int p_index) {
		CRASH_BAD_INDEX(p_index, ((Vector<T> *)(this))->_cowdata.size());

		return ((Vector<T> *)(this))->_cowdata.ptrw()[p_index];
	}
};

template <class T>
class Vector {
	friend class VectorWriteProxy<T>;

public:
	VectorWriteProxy<T> write;

private:
	CowData<T> _cowdata;

public:
	bool push_back(T p_elem);
	_FORCE_INLINE_ bool append(const T &p_elem) { return push_back(p_elem); } //alias
	void fill(T p_elem);

	void remove_at(int p_index) { _cowdata.remove_at(p_index); }
	void erase(const T &p_val) {
		int idx = find(p_val);
		if (idx >= 0) {
			remove_at(idx);
		}
	}
	void reverse();

	_FORCE_INLINE_ T *ptrw() { return _cowdata.ptrw(); }
	_FORCE_INLINE_ const T *ptr() const { return _cowdata.ptr(); }
	_FORCE_INLINE_ void clear() { resize(0); }
	_FORCE_INLINE_ bool is_empty() const { return _cowdata.is_empty(); }

	_FORCE_INLINE_ T get(int p_index) { return _cowdata.get(p_index); }
	_FORCE_INLINE_ const T &get(int p_index) const { return _cowdata.get(p_index); }
	_FORCE_INLINE_ void set(int p_index, const T &p_elem) { _cowdata.set(p_index, p_elem); }
	_FORCE_INLINE_ int size() const { return _cowdata.size(); }
	Error resize(int p_size) { return _cowdata.resize(p_size); }
	_FORCE_INLINE_ const T &operator[](int p_index) const { return _cowdata.get(p_index); }
	Error insert(int p_pos, T p_val) { return _cowdata.insert(p_pos, p_val); }
	int find(const T &p_val, int p_from = 0) const { return _cowdata.find(p_val, p_from); }

	void append_array(Vector<T> p_other);

	bool has(const T &p_val) const {
		return find(p_val, 0) != -1;
	}

	template <class C>
	void sort_custom() {
		int len = _cowdata.size();
		if (len == 0) {
			return;
		}

		T *data = ptrw();
		SortArray<T, C> sorter;
		sorter.sort(data, len);
	}

	void sort() {
		sort_custom<_DefaultComparator<T>>();
	}

	int bsearch(const T &p_value, bool p_before) {
		SearchArray<T> search;
		return search.bisect(ptrw(), size(), p_value, p_before);
	}

	Vector<T> duplicate() {
		return *this;
	}

	void ordered_insert(const T &p_val) {
		int i;
		for (i = 0; i < _cowdata.size(); i++) {
			if (p_val < operator[](i)) {
				break;
			}
		}
		insert(i, p_val);
	}

	inline void operator=(const Vector &p_from) {
		_cowdata._ref(p_from._cowdata);
	}

	Vector<uint8_t> to_byte_array() const {
		Vector<uint8_t> ret;
		ret.resize(size() * sizeof(T));
		memcpy(ret.ptrw(), ptr(), sizeof(T) * size());
		return ret;
	}

	Vector<T> subarray(int p_from, int p_to) const {
		if (p_from < 0) {
			p_from = size() + p_from;
		}
		if (p_to < 0) {
			p_to = size() + p_to;
		}

		ERR_FAIL_INDEX_V(p_from, size(), Vector<T>());
		ERR_FAIL_INDEX_V(p_to, size(), Vector<T>());

		Vector<T> slice;
		int span = 1 + p_to - p_from;
		slice.resize(span);
		const T *r = ptr();
		T *w = slice.ptrw();
		for (int i = 0; i < span; ++i) {
			w[i] = r[p_from + i];
		}

		return slice;
	}

	bool operator==(const Vector<T> &p_arr) const {
		int s = size();
		if (s != p_arr.size()) {
			return false;
		}
		for (int i = 0; i < s; i++) {
			if (operator[](i) != p_arr[i]) {
				return false;
			}
		}
		return true;
	}

	bool operator!=(const Vector<T> &p_arr) const {
		int s = size();
		if (s != p_arr.size()) {
			return true;
		}
		for (int i = 0; i < s; i++) {
			if (operator[](i) != p_arr[i]) {
				return true;
			}
		}
		return false;
	}

	struct Iterator {
		_FORCE_INLINE_ T &operator*() const {
			return *elem_ptr;
		}
		_FORCE_INLINE_ T *operator->() const { return elem_ptr; }
		_FORCE_INLINE_ Iterator &operator++() {
			elem_ptr++;
			return *this;
		}
		_FORCE_INLINE_ Iterator &operator--() {
			elem_ptr--;
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const Iterator &b) const { return elem_ptr == b.elem_ptr; }
		_FORCE_INLINE_ bool operator!=(const Iterator &b) const { return elem_ptr != b.elem_ptr; }

		Iterator(T *p_ptr) { elem_ptr = p_ptr; }
		Iterator() {}
		Iterator(const Iterator &p_it) { elem_ptr = p_it.elem_ptr; }

	private:
		T *elem_ptr = nullptr;
	};

	struct ConstIterator {
		_FORCE_INLINE_ const T &operator*() const {
			return *elem_ptr;
		}
		_FORCE_INLINE_ const T *operator->() const { return elem_ptr; }
		_FORCE_INLINE_ ConstIterator &operator++() {
			elem_ptr++;
			return *this;
		}
		_FORCE_INLINE_ ConstIterator &operator--() {
			elem_ptr--;
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const ConstIterator &b) const { return elem_ptr == b.elem_ptr; }
		_FORCE_INLINE_ bool operator!=(const ConstIterator &b) const { return elem_ptr != b.elem_ptr; }

		ConstIterator(const T *p_ptr) { elem_ptr = p_ptr; }
		ConstIterator() {}
		ConstIterator(const ConstIterator &p_it) { elem_ptr = p_it.elem_ptr; }

	private:
		const T *elem_ptr = nullptr;
	};

	_FORCE_INLINE_ Iterator begin() {
		return Iterator(ptrw());
	}
	_FORCE_INLINE_ Iterator end() {
		return Iterator(ptrw() + size());
	}

	_FORCE_INLINE_ ConstIterator begin() const {
		return ConstIterator(ptr());
	}
	_FORCE_INLINE_ ConstIterator end() const {
		return ConstIterator(ptr() + size());
	}

	_FORCE_INLINE_ Vector() {}
	_FORCE_INLINE_ Vector(const Vector &p_from) { _cowdata._ref(p_from._cowdata); }

	_FORCE_INLINE_ ~Vector() {}
};

template <class T>
void Vector<T>::reverse() {
	for (int i = 0; i < size() / 2; i++) {
		T *p = ptrw();
		SWAP(p[i], p[size() - i - 1]);
	}
}

template <class T>
void Vector<T>::append_array(Vector<T> p_other) {
	const int ds = p_other.size();
	if (ds == 0) {
		return;
	}
	const int bs = size();
	resize(bs + ds);
	for (int i = 0; i < ds; ++i) {
		ptrw()[bs + i] = p_other[i];
	}
}

template <class T>
bool Vector<T>::push_back(T p_elem) {
	Error err = resize(size() + 1);
	ERR_FAIL_COND_V(err, true);
	set(size() - 1, p_elem);

	return false;
}

template <class T>
void Vector<T>::fill(T p_elem) {
	T *p = ptrw();
	for (int i = 0; i < size(); i++) {
		p[i] = p_elem;
	}
}

#endif // VECTOR_H
