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

#pragma once

#include "core/error/error_macros.h"
#include "core/os/memory.h"
#include "core/string/print_string.h"
#include "core/templates/sort_array.h"
#include "core/templates/vector.h"

#include <initializer_list>
#include <type_traits>

// If tight, it grows strictly as much as needed.
// Otherwise, it grows exponentially (the default and what you want in most cases).
template <typename T, typename U = uint32_t, bool force_trivial = false, bool tight = false>
class LocalVector {
	static_assert(!force_trivial, "force_trivial is no longer supported. Use resize_uninitialized instead.");

private:
	U count = 0;
	U capacity = 0;
	T *data = nullptr;

	template <bool p_init>
	void _resize(U p_size) {
		if (p_size < count) {
			if constexpr (!std::is_trivially_destructible_v<T>) {
				for (U i = p_size; i < count; i++) {
					data[i].~T();
				}
			}
			count = p_size;
		} else if (p_size > count) {
			reserve(p_size);
			if constexpr (p_init) {
				memnew_arr_placement(data + count, p_size - count);
			} else {
				static_assert(std::is_trivially_destructible_v<T>, "T must be trivially destructible to resize uninitialized");
			}
			count = p_size;
		}
	}

public:
	_FORCE_INLINE_ T *ptr() { return data; }
	_FORCE_INLINE_ const T *ptr() const { return data; }
	_FORCE_INLINE_ U size() const { return count; }

	_FORCE_INLINE_ Span<T> span() const { return Span(data, count); }
	_FORCE_INLINE_ operator Span<T>() const { return span(); }

	// Must take a copy instead of a reference (see GH-31736).
	_FORCE_INLINE_ void push_back(T p_elem) {
		if (unlikely(count == capacity)) {
			reserve(count + 1);
		}

		memnew_placement(&data[count++], T(std::move(p_elem)));
	}

	void remove_at(U p_index) {
		ERR_FAIL_UNSIGNED_INDEX(p_index, count);
		count--;
		for (U i = p_index; i < count; i++) {
			data[i] = std::move(data[i + 1]);
		}
		data[count].~T();
	}

	/// Removes the item copying the last value into the position of the one to
	/// remove. It's generally faster than `remove_at`.
	void remove_at_unordered(U p_index) {
		ERR_FAIL_INDEX(p_index, count);
		count--;
		if (count > p_index) {
			data[p_index] = std::move(data[count]);
		}
		data[count].~T();
	}

	_FORCE_INLINE_ bool erase(const T &p_val) {
		int64_t idx = find(p_val);
		if (idx >= 0) {
			remove_at(idx);
			return true;
		}
		return false;
	}

	bool erase_unordered(const T &p_val) {
		int64_t idx = find(p_val);
		if (idx >= 0) {
			remove_at_unordered(idx);
			return true;
		}
		return false;
	}

	U erase_multiple_unordered(const T &p_val) {
		U from = 0;
		U occurrences = 0;
		while (true) {
			int64_t idx = find(p_val, from);

			if (idx == -1) {
				break;
			}
			remove_at_unordered(idx);
			from = idx;
			occurrences++;
		}
		return occurrences;
	}

	void reverse() {
		for (U i = 0; i < count / 2; i++) {
			SWAP(data[i], data[count - i - 1]);
		}
	}
#ifndef DISABLE_DEPRECATED
	[[deprecated("Use reverse() instead")]] void invert() { reverse(); }
#endif

	_FORCE_INLINE_ void clear() { resize(0); }
	_FORCE_INLINE_ void reset() {
		clear();
		if (data) {
			memfree(data);
			data = nullptr;
			capacity = 0;
		}
	}
	_FORCE_INLINE_ bool is_empty() const { return count == 0; }
	_FORCE_INLINE_ U get_capacity() const { return capacity; }
	void reserve(U p_size) {
		if (p_size > capacity) {
			if (tight) {
				capacity = p_size;
			} else {
				// Try 1.5x the current capacity.
				// This ratio was chosen because it is close to the ideal growth rate of the golden ratio.
				// See https://archive.ph/Z2R8w for details.
				capacity = MAX((U)2, capacity + ((1 + capacity) >> 1));
				// If 1.5x growth isn't enough, just use the needed size exactly.
				if (p_size > capacity) {
					capacity = p_size;
				}
			}
			data = (T *)memrealloc(data, capacity * sizeof(T));
			CRASH_COND_MSG(!data, "Out of memory");
		} else if (p_size < count) {
			WARN_VERBOSE("reserve() called with a capacity smaller than the current size. This is likely a mistake.");
		}
	}

	/// Resize the vector.
	/// Elements are initialized (or not) depending on what the default C++ behavior for T is.
	/// Note: If force_trivial is set, this will behave like resize_uninitialized instead.
	void resize(U p_size) {
		// Don't init when trivially constructible.
		_resize<!std::is_trivially_constructible_v<T>>(p_size);
	}

	/// Resize and set all values to 0 / false / nullptr.
	_FORCE_INLINE_ void resize_initialized(U p_size) { _resize<true>(p_size); }

	/// Resize and set all values to 0 / false / nullptr.
	/// This is only available for trivially destructible types (otherwise, trivial resize might be UB).
	_FORCE_INLINE_ void resize_uninitialized(U p_size) { _resize<false>(p_size); }

	_FORCE_INLINE_ const T &operator[](U p_index) const {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return data[p_index];
	}
	_FORCE_INLINE_ T &operator[](U p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return data[p_index];
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
		return Iterator(data);
	}
	_FORCE_INLINE_ Iterator end() {
		return Iterator(data + size());
	}

	_FORCE_INLINE_ ConstIterator begin() const {
		return ConstIterator(ptr());
	}
	_FORCE_INLINE_ ConstIterator end() const {
		return ConstIterator(ptr() + size());
	}

	void insert(U p_pos, T p_val) {
		ERR_FAIL_UNSIGNED_INDEX(p_pos, count + 1);
		if (p_pos == count) {
			push_back(std::move(p_val));
		} else {
			resize(count + 1);
			for (U i = count - 1; i > p_pos; i--) {
				data[i] = std::move(data[i - 1]);
			}
			data[p_pos] = std::move(p_val);
		}
	}

	int64_t find(const T &p_val, int64_t p_from = 0) const {
		if (p_from < 0) {
			p_from = size() + p_from;
		}
		if (p_from < 0 || p_from >= size()) {
			return -1;
		}
		return span().find(p_val, p_from);
	}

	bool has(const T &p_val) const {
		return find(p_val) != -1;
	}

	template <typename C>
	void sort_custom() {
		U len = count;
		if (len == 0) {
			return;
		}

		SortArray<T, C> sorter;
		sorter.sort(data, len);
	}

	void sort() {
		sort_custom<Comparator<T>>();
	}

	void ordered_insert(T p_val) {
		U i;
		for (i = 0; i < count; i++) {
			if (p_val < data[i]) {
				break;
			}
		}
		insert(i, p_val);
	}

	explicit operator Vector<T>() const {
		Vector<T> ret;
		ret.resize(count);
		T *w = ret.ptrw();
		if (w) {
			copy_arr_placement(w, data, count);
		}
		return ret;
	}

	Vector<uint8_t> to_byte_array() const { //useful to pass stuff to gpu or variant
		Vector<uint8_t> ret;
		ret.resize(count * sizeof(T));
		uint8_t *w = ret.ptrw();
		if (w) {
			memcpy(w, data, sizeof(T) * count);
		}
		return ret;
	}

	_FORCE_INLINE_ LocalVector() {}
	_FORCE_INLINE_ LocalVector(std::initializer_list<T> p_init) {
		reserve(p_init.size());
		for (const T &element : p_init) {
			push_back(element);
		}
	}
	_FORCE_INLINE_ LocalVector(const LocalVector &p_from) {
		resize(p_from.size());
		for (U i = 0; i < p_from.count; i++) {
			data[i] = p_from.data[i];
		}
	}
	_FORCE_INLINE_ LocalVector(LocalVector &&p_from) {
		data = p_from.data;
		count = p_from.count;
		capacity = p_from.capacity;

		p_from.data = nullptr;
		p_from.count = 0;
		p_from.capacity = 0;
	}

	inline void operator=(const LocalVector &p_from) {
		resize(p_from.size());
		for (U i = 0; i < p_from.count; i++) {
			data[i] = p_from.data[i];
		}
	}
	inline void operator=(const Vector<T> &p_from) {
		resize(p_from.size());
		for (U i = 0; i < count; i++) {
			data[i] = p_from[i];
		}
	}
	inline void operator=(LocalVector &&p_from) {
		if (unlikely(this == &p_from)) {
			return;
		}
		reset();

		data = p_from.data;
		count = p_from.count;
		capacity = p_from.capacity;

		p_from.data = nullptr;
		p_from.count = 0;
		p_from.capacity = 0;
	}
	inline void operator=(Vector<T> &&p_from) {
		resize(p_from.size());
		for (U i = 0; i < count; i++) {
			data[i] = std::move(p_from[i]);
		}
	}

	_FORCE_INLINE_ ~LocalVector() {
		if (data) {
			reset();
		}
	}
};

template <typename T, typename U = uint32_t>
using TightLocalVector = LocalVector<T, U, false, true>;

// Zero-constructing LocalVector initializes count, capacity and data to 0 and thus empty.
template <typename T, typename U, bool force_trivial, bool tight>
struct is_zero_constructible<LocalVector<T, U, force_trivial, tight>> : std::true_type {};
