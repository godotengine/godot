/**************************************************************************/
/*  local_vector.h                                                        */
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

#ifndef LOCAL_VECTOR_H
#define LOCAL_VECTOR_H

#include "core/error_macros.h"
#include "core/os/memory.h"
#include "core/sort_array.h"
#include "core/span.h"
#include "core/vector.h"

#include <type_traits>
#include <utility>

template <class T, class U = uint32_t, bool force_trivial = false>
class LocalVector {
protected:
	U count = 0;
	U capacity = 0;
	T *data = nullptr;

	static constexpr bool RUN_CONSTRUCTORS = !std::is_trivially_constructible<T>::value && !force_trivial;
	static constexpr bool RUN_DESTRUCTORS = !std::is_trivially_destructible<T>::value && !force_trivial;

	void _realloc(U p_capacity) {
		if (p_capacity == 0) {
			if (data) {
				memfree(data);
				data = nullptr;
			}
			capacity = 0;
			return;
		}

		// Guard against memrealloc for no reason.
		// The logic should prevent this.
		DEV_ASSERT(p_capacity != capacity);

		// Strictly speaking, for modern c++, we should use std::move and destruct
		// elements after moving to a new memalloced array.
		// HOWEVER, Godot makes heavy use of structs that are trivially copyable,
		// and don't contain move semantics, which would result in double deletes if
		// we did this.
		// So we maintain the backward compatible method here.
		data = (T *)memrealloc(data, p_capacity * sizeof(T));
		CRASH_COND_MSG(!data, "Out of memory");
		capacity = p_capacity;
	}

	template <bool p_init>
	void _resize(U p_size) {
		if (p_size < count) {
			if constexpr (RUN_DESTRUCTORS) {
				for (U i = p_size; i < count; i++) {
					data[i].~T();
				}
			}

			// Compiler warning solve for signed U.
			// This should be compiled out for unsigned.
			if constexpr (std::is_signed<U>::value) {
				CRASH_COND(p_size < 0);
			}
			count = p_size;
		} else if (p_size > count) {
			if (unlikely(p_size > capacity)) {
				reserve(p_size);
			}
			if constexpr (p_init) {
				for (U i = count; i < p_size; i++) {
					memnew_placement(&data[i], T);
				}
			} else {
				static_assert(std::is_trivially_destructible<T>::value, "T must be trivially destructible to resize uninitialized");
			}
			count = p_size;
		}
	}

public:
	T *ptr() _LIFETIME_BOUND_ {
		return data;
	}

	const T *ptr() const _LIFETIME_BOUND_ {
		return data;
	}

	_FORCE_INLINE_ void push_back(T p_elem) {
		if (unlikely(count == capacity)) {
			U new_capacity = capacity == 0 ? 1 : capacity << 1;
			_realloc(new_capacity);
		}

		memnew_placement(&data[count++], T(std::move(p_elem)));
	}

	void remove(U p_index) {
		ERR_FAIL_UNSIGNED_INDEX(p_index, count);
		count--;
		for (U i = p_index; i < count; i++) {
			data[i] = std::move(data[i + 1]);
		}
		if constexpr (RUN_DESTRUCTORS) {
			data[count].~T();
		}
	}

	// Removes the item copying the last value into the position of the one to
	// remove. It's generally faster than `remove`.
	void remove_unordered(U p_index) {
		ERR_FAIL_UNSIGNED_INDEX(p_index, count);
		count--;
		if (count > p_index) {
			data[p_index] = std::move(data[count]);
		}
		if constexpr (RUN_DESTRUCTORS) {
			data[count].~T();
		}
	}

	void erase(const T &p_val) {
		int64_t idx = find(p_val);
		if (idx >= 0) {
			remove(idx);
		}
	}

	bool erase_unordered(const T &p_val) {
		int64_t idx = find(p_val);
		if (idx >= 0) {
			remove_unordered(idx);
			return true;
		}
		return false;
	}

	U erase_multiple_unordered(const T &p_val) {
		U from = 0;
		U removed = 0;
		while (true) {
			int64_t idx = find(p_val, from);

			if (idx == -1) {
				break;
			}
			remove_unordered(idx);
			from = idx;
			removed++;
		}
		return removed;
	}

	void invert() {
		for (U i = 0; i < count / 2; i++) {
			SWAP(data[i], data[count - i - 1]);
		}
	}

	_FORCE_INLINE_ void clear() { resize(0); }
	void reset() {
		clear();
		if (data) {
			memfree(data);
			data = nullptr;
			capacity = 0;
		}
	}
	_FORCE_INLINE_ bool empty() const { return count == 0; }
	_FORCE_INLINE_ U get_capacity() const { return capacity; }

	void reserve(U p_size, bool p_allow_shrink = false) {
		if (p_size == 0) {
			if (p_allow_shrink && empty() && capacity) {
				reset();
			}
			return;
		}
		p_size = nearest_power_of_2_templated(p_size);
		if (!p_allow_shrink ? p_size > capacity : ((p_size >= count) && (p_size != capacity))) {
			_realloc(p_size);
		}
	}

	_FORCE_INLINE_ U size() const { return count; }

	// Resize the vector.
	// Elements are initialized (or not) depending on what the default C++ behavior for T is.
	// Note: If force_trivial is set, this will behave like resize_uninitialized instead.
	_FORCE_INLINE_ void resize(U p_size) {
		_resize<RUN_CONSTRUCTORS>(p_size);
	}

	// Resize and set new values to 0 / false / nullptr.
	_FORCE_INLINE_ void resize_initialized(U p_size) { _resize<true>(p_size); }

	// Resize and keep memory uninitialized.
	// This means that any newly added elements have an unknown value, and are expected to be set after the `resize_uninitialized` call.
	// This is only available for trivially destructible types (otherwise, trivial resize might be UB).
	_FORCE_INLINE_ void resize_uninitialized(U p_size) { _resize<false>(p_size); }

	_FORCE_INLINE_ const T &operator[](U p_index) const {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return data[p_index];
	}
	_FORCE_INLINE_ T &operator[](U p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return data[p_index];
	}

	_FORCE_INLINE_ const T &get_unchecked(U p_index) const {
		DEV_ASSERT(p_index < count);
		return data[p_index];
	}
	_FORCE_INLINE_ T &get_unchecked(U p_index) {
		DEV_ASSERT(p_index < count);
		return data[p_index];
	}

	void fill(T p_val) {
		for (U i = 0; i < count; i++) {
			data[i] = p_val;
		}
	}

	void insert(U p_pos, T p_val) {
		ERR_FAIL_UNSIGNED_INDEX(p_pos, count + 1);
		if (p_pos == count) {
			push_back(std::move(p_val));
		} else {
			resize(count + 1);

			if constexpr (std::is_trivially_copyable<T>::value || force_trivial) {
				U num_elements = count - 1 - p_pos;
				if (num_elements) {
					memmove(&data[p_pos + 1], &data[p_pos], num_elements * sizeof(T));
				}
			} else {
				for (U i = count - 1; i > p_pos; i--) {
					data[i] = std::move(data[i - 1]);
				}
			}

			data[p_pos] = std::move(p_val);
		}
	}

	int64_t find(const T &p_val, U p_from = 0) const {
		for (U i = p_from; i < count; i++) {
			if (data[i] == p_val) {
				return int64_t(i);
			}
		}
		return -1;
	}

	template <class C>
	void sort_custom() {
		U len = count;
		if (len == 0) {
			return;
		}

		SortArray<T, C> sorter;
		sorter.sort(data, len);
	}

	void sort() {
		sort_custom<_DefaultComparator<T>>();
	}

	void ordered_insert(T p_val) {
		U idx = span().bisect(p_val, false);
		insert(idx, std::move(p_val));
	}

	Vector<uint8_t> to_byte_array() const { //useful to pass stuff to gpu or variant
		Vector<uint8_t> ret;
		if (count > 0) {
			ret.resize(count * sizeof(T));
			uint8_t *w = ret.ptrw();
			memcpy(w, data, sizeof(T) * count);
		}
		return ret;
	}

	_FORCE_INLINE_ Span<T> span() const _LIFETIME_BOUND_ {
		// Ensure span is unsigned.
		// NOOP for default LocalVector, but converts any LocalVectors with signed U.
		using UnsignedType = std::make_unsigned_t<U>;
		return Span<T>(data, static_cast<UnsignedType>(count));
	}
	_FORCE_INLINE_ operator Span<T>() const _LIFETIME_BOUND_ { return span(); }

	LocalVector() {}
	LocalVector(const LocalVector &p_from) {
		resize(p_from.size());
		copy_arr(ptr(), p_from.ptr(), count);
	}

	explicit LocalVector(const Span<T> &p_from) {
		resize(p_from.size());
		copy_arr(ptr(), p_from.ptr(), count);
	}

	LocalVector(const Vector<T> &p_from) {
		resize(p_from.size());
		copy_arr(ptr(), p_from.ptr(), count);
	}

	LocalVector(LocalVector &&p_from) {
		data = p_from.data;
		count = p_from.count;
		capacity = p_from.capacity;

		p_from.data = nullptr;
		p_from.count = 0;
		p_from.capacity = 0;
	}

	LocalVector &operator=(const LocalVector &p_from) {
		if (unlikely(this == &p_from)) {
			return *this;
		}

		resize(p_from.size());
		copy_arr(ptr(), p_from.ptr(), count);
		return *this;
	}

	LocalVector &operator=(LocalVector &&p_from) {
		if (unlikely(this == &p_from)) {
			return *this;
		}
		reset();

		data = p_from.data;
		count = p_from.count;
		capacity = p_from.capacity;

		p_from.data = nullptr;
		p_from.count = 0;
		p_from.capacity = 0;
		return *this;
	}

	LocalVector &operator=(Vector<T> &&p_from) {
		resize(p_from.size());
		for (U i = 0; i < count; i++) {
			data[i] = std::move(p_from[i]);
		}
		return *this;
	}

	LocalVector &operator=(const Vector<T> &p_from) {
		resize(p_from.size());
		copy_arr(ptr(), p_from.ptr(), count);
		return *this;
	}

	LocalVector &operator=(const Span<T> &p_from) {
		if (data && p_from.ptr() >= data && p_from.ptr() < data + capacity) {
			LocalVector temp(p_from);
			*this = std::move(temp);
		} else {
			resize(p_from.size());
			copy_arr(ptr(), p_from.ptr(), count);
		}
		return *this;
	}

	~LocalVector() {
		if (data) {
			reset();
		}
	}
};

// Integer default version
template <class T, class I = int32_t, bool force_trivial = false>
using LocalVectori = LocalVector<T, I, force_trivial>;

#endif // LOCAL_VECTOR_H
