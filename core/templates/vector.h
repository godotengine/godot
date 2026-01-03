/**************************************************************************/
/*  vector.h                                                              */
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

/**
 * @class Vector
 * Vector container. Simple copy-on-write container.
 *
 * LocalVector is an alternative available for internal use when COW is not
 * required.
 */

#include "core/error/error_macros.h"
#include "core/templates/cowdata.h"
#include "core/templates/sort_array.h"

#include <initializer_list>

template <typename T>
class Vector;

template <typename T>
class VectorWriteProxy {
public:
	_FORCE_INLINE_ T &operator[](typename CowData<T>::Size p_index) {
		CRASH_BAD_INDEX(p_index, ((Vector<T> *)(this))->_cowdata.size());

		return ((Vector<T> *)(this))->_cowdata.ptrw()[p_index];
	}
};

template <typename T>
class Vector {
	friend class VectorWriteProxy<T>;

public:
	VectorWriteProxy<T> write;
	typedef typename CowData<T>::Size Size;
	typedef typename CowData<T>::USize USize;

private:
	CowData<T> _cowdata;

public:
	// Must take a copy instead of a reference (see GH-31736).
	_FORCE_INLINE_ bool push_back(T p_elem) { return _cowdata.push_back(std::move(p_elem)); }
	_FORCE_INLINE_ bool append(T p_elem) { return _cowdata.push_back(std::move(p_elem)); } //alias
	void fill(T p_elem);

	void remove_at(Size p_index) { _cowdata.remove_at(p_index); }
	_FORCE_INLINE_ bool erase(const T &p_val) {
		Size idx = find(p_val);
		if (idx >= 0) {
			remove_at(idx);
			return true;
		}
		return false;
	}

	void reverse();

	_FORCE_INLINE_ T *ptrw() { return _cowdata.ptrw(); }
	_FORCE_INLINE_ const T *ptr() const { return _cowdata.ptr(); }
	_FORCE_INLINE_ Size size() const { return _cowdata.size(); }
	_FORCE_INLINE_ USize capacity() const { return _cowdata.capacity(); }

	_FORCE_INLINE_ operator Span<T>() const { return _cowdata.span(); }
	_FORCE_INLINE_ Span<T> span() const { return _cowdata.span(); }

	_FORCE_INLINE_ void clear() { _cowdata.clear(); }
	_FORCE_INLINE_ bool is_empty() const { return _cowdata.is_empty(); }

	_FORCE_INLINE_ T get(Size p_index) { return _cowdata.get(p_index); }
	_FORCE_INLINE_ const T &get(Size p_index) const { return _cowdata.get(p_index); }
	_FORCE_INLINE_ void set(Size p_index, const T &p_elem) { _cowdata.set(p_index, p_elem); }

	/// Resize the vector.
	/// Elements are initialized (or not) depending on what the default C++ behavior for this type is.
	_FORCE_INLINE_ Error resize(Size p_size) {
		return _cowdata.template resize<!std::is_trivially_constructible_v<T>>(p_size);
	}

	/// Resize and set all values to 0 / false / nullptr.
	/// This is only available for zero constructible types.
	_FORCE_INLINE_ Error resize_initialized(Size p_size) {
		return _cowdata.template resize<true>(p_size);
	}

	/// Resize and set all values to 0 / false / nullptr.
	/// This is only available for trivially destructible types (otherwise, trivial resize might be UB).
	_FORCE_INLINE_ Error resize_uninitialized(Size p_size) {
		// resize() statically asserts that T is compatible, no need to do it ourselves.
		return _cowdata.template resize<false>(p_size);
	}

	Error reserve(Size p_size) {
		ERR_FAIL_COND_V(p_size < 0, ERR_INVALID_PARAMETER);
		return _cowdata.reserve(p_size);
	}

	Error reserve_exact(Size p_size) {
		ERR_FAIL_COND_V(p_size < 0, ERR_INVALID_PARAMETER);
		return _cowdata.reserve_exact(p_size);
	}

	_FORCE_INLINE_ const T &operator[](Size p_index) const { return _cowdata.get(p_index); }
	// Must take a copy instead of a reference (see GH-31736).
	Error insert(Size p_pos, T p_val) { return _cowdata.insert(p_pos, std::move(p_val)); }
	Size find(const T &p_val, Size p_from = 0) const {
		if (p_from < 0) {
			p_from = size() + p_from;
		}
		if (p_from < 0 || p_from >= size()) {
			return -1;
		}
		return span().find(p_val, p_from);
	}
	Size rfind(const T &p_val, Size p_from = -1) const {
		if (p_from < 0) {
			p_from = size() + p_from;
		}
		if (p_from < 0 || p_from >= size()) {
			return -1;
		}
		return span().rfind(p_val, p_from);
	}
	Size count(const T &p_val) const { return span().count(p_val); }

	// Must take a copy instead of a reference (see GH-31736).
	void append_array(Vector<T> p_other);

	_FORCE_INLINE_ bool has(const T &p_val) const { return find(p_val) != -1; }

	void sort() {
		sort_custom<Comparator<T>>();
	}

	template <typename Comparator, bool Validate = SORT_ARRAY_VALIDATE_ENABLED, typename... Args>
	void sort_custom(Args &&...args) {
		Size len = _cowdata.size();
		if (len == 0) {
			return;
		}

		T *data = ptrw();
		SortArray<T, Comparator, Validate> sorter{ args... };
		sorter.sort(data, len);
	}

	Size bsearch(const T &p_value, bool p_before) const {
		return bsearch_custom<Comparator<T>>(p_value, p_before);
	}

	template <typename Comparator, typename Value, typename... Args>
	Size bsearch_custom(const Value &p_value, bool p_before, Args &&...args) const {
		return span().bisect(p_value, p_before, Comparator{ args... });
	}

	Vector<T> duplicate() const {
		return *this;
	}

	void ordered_insert(const T &p_val) {
		Size i;
		for (i = 0; i < _cowdata.size(); i++) {
			if (p_val < operator[](i)) {
				break;
			}
		}
		insert(i, p_val);
	}

	void operator=(const Vector &p_from) { _cowdata = p_from._cowdata; }
	void operator=(Vector &&p_from) { _cowdata = std::move(p_from._cowdata); }

	Vector<uint8_t> to_byte_array() const {
		Vector<uint8_t> ret;
		if (is_empty()) {
			return ret;
		}
		size_t alloc_size = size() * sizeof(T);
		ret.resize(alloc_size);
		if (alloc_size) {
			memcpy(ret.ptrw(), ptr(), alloc_size);
		}
		return ret;
	}

	Vector<T> slice(Size p_begin, Size p_end = CowData<T>::MAX_INT) const {
		Vector<T> result;

		const Size s = size();

		Size begin = CLAMP(p_begin, -s, s);
		if (begin < 0) {
			begin += s;
		}
		Size end = CLAMP(p_end, -s, s);
		if (end < 0) {
			end += s;
		}

		ERR_FAIL_COND_V(begin > end, result);

		Size result_size = end - begin;
		result.resize(result_size);

		const T *const r = ptr();
		T *const w = result.ptrw();
		for (Size i = 0; i < result_size; ++i) {
			w[i] = r[begin + i];
		}

		return result;
	}

	bool operator==(const Vector<T> &p_arr) const { return span() == p_arr.span(); }
	bool operator!=(const Vector<T> &p_arr) const { return span() != p_arr.span(); }

	using Iterator = T *;
	using ConstIterator = const T *;

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
	_FORCE_INLINE_ Vector(std::initializer_list<T> p_init) :
			_cowdata(p_init) {}
	_FORCE_INLINE_ Vector(const Vector &p_from) = default;
	_FORCE_INLINE_ Vector(Vector &&p_from) = default;
};

template <typename T>
void Vector<T>::reverse() {
	T *p = ptrw();
	for (Size i = 0; i < size() / 2; i++) {
		SWAP(p[i], p[size() - i - 1]);
	}
}

template <typename T>
void Vector<T>::append_array(Vector<T> p_other) {
	const Size ds = p_other.size();
	if (ds == 0) {
		return;
	}
	const Size bs = size();
	resize(bs + ds);
	T *p = ptrw();
	for (Size i = 0; i < ds; ++i) {
		p[bs + i] = p_other[i];
	}
}

template <typename T>
void Vector<T>::fill(T p_elem) {
	T *p = ptrw();
	for (Size i = 0; i < size(); i++) {
		p[i] = p_elem;
	}
}

// Zero-constructing Vector initializes CowData.ptr() to nullptr and thus empty.
template <typename T>
struct is_zero_constructible<Vector<T>> : std::true_type {};
