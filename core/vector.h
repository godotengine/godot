/*************************************************************************/
/*  vector.h                                                             */
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

#ifndef VECTOR_H
#define VECTOR_H

/**
 * @class Vector
 * @author Juan Linietsky
 * Vector container. Regular Vector Container. Use with care and for smaller arrays when possible. Use PoolVector for large arrays.
*/

#include "core/cowdata.h"
#include "core/error_macros.h"
#include "core/os/memory.h"
#include "core/sort_array.h"

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
	bool push_back(const T &p_elem);

	void remove(int p_index) { _cowdata.remove(p_index); }
	void erase(const T &p_val) {
		int idx = find(p_val);
		if (idx >= 0) remove(idx);
	};
	void invert();

	_FORCE_INLINE_ T *ptrw() { return _cowdata.ptrw(); }
	_FORCE_INLINE_ const T *ptr() const { return _cowdata.ptr(); }
	_FORCE_INLINE_ void clear() { resize(0); }
	_FORCE_INLINE_ bool empty() const { return _cowdata.empty(); }

	_FORCE_INLINE_ T get(int p_index) { return _cowdata.get(p_index); }
	_FORCE_INLINE_ const T get(int p_index) const { return _cowdata.get(p_index); }
	_FORCE_INLINE_ void set(int p_index, const T &p_elem) { _cowdata.set(p_index, p_elem); }
	_FORCE_INLINE_ int size() const { return _cowdata.size(); }
	Error resize(int p_size) { return _cowdata.resize(p_size); }
	_FORCE_INLINE_ const T &operator[](int p_index) const { return _cowdata.get(p_index); }
	Error insert(int p_pos, const T &p_val) { return _cowdata.insert(p_pos, p_val); }
	int find(const T &p_val, int p_from = 0) const { return _cowdata.find(p_val, p_from); }

	void append_array(const Vector<T> &p_other);

	template <class C>
	void sort_custom() {
		int len = _cowdata.size();
		if (len == 0)
			return;

		T *data = _cowdata.ptrw();
		SortArray<T, C> sorter;
		sorter.sort(data, len);
	}

	void sort() {
		sort_custom<_DefaultComparator<T> >();
	}

	void ordered_insert(const T &p_val) {
		int len = _cowdata.size();
		for (int i = 0; i < len; i++) {
			if (p_val < operator[](i)) {
				insert(i, p_val);
				return;
			};
		};

		insert(len, p_val);
	}

	_FORCE_INLINE_ Vector() {}
	_FORCE_INLINE_ Vector(const Vector &p_from) { _cowdata._ref(p_from._cowdata); }
	inline Vector &operator=(const Vector &p_from) {
		_cowdata._ref(p_from._cowdata);
		return *this;
	}

	_FORCE_INLINE_ ~Vector() {}
};

template <class T>
void Vector<T>::invert() {
	int len = _cowdata.size();
	int half_len = len / 2;
	T *p = _cowdata.ptrw();

	for (int i = 0; i < half_len; i++) {
		SWAP(p[i], p[len - i - 1]);
	}
}

template <class T>
void Vector<T>::append_array(const Vector<T> &p_other) {
	const int other_len = p_other._cowdata.size();
	if (other_len == 0)
		return;

	const int len = _cowdata.size();
	resize(len + other_len);
	for (int i = 0; i < other_len; ++i)
		_cowdata.ptrw()[len + i] = p_other[i];
}

template <class T>
bool Vector<T>::push_back(const T &p_elem) {
	Error err = resize(_cowdata.size() + 1);
	ERR_FAIL_COND_V(err, true)
	set(_cowdata.size() - 1, p_elem);

	return false;
}

#endif /* VECTOR_H */
