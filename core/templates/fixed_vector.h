/**************************************************************************/
/*  fixed_vector.h                                                        */
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

#include "core/templates/relocate_init_list.h"
#include "core/templates/span.h"

GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Warray-bounds")

/**
 * A high performance Vector of fixed capacity.
 * Especially useful if you need to create an array on the stack, to
 *  prevent dynamic allocations (especially in bottleneck code).
 *
 * Choose CAPACITY such that it is enough for all elements that could be added through all branches.
 *
 */
template <class T, uint32_t CAPACITY>
class FixedVector {
	// This declaration allows us to access other FixedVector's private members.
	template <class T_, uint32_t CAPACITY_>
	friend class FixedVector;

	uint32_t _size = 0;
	alignas(T) uint8_t _data[CAPACITY * sizeof(T)];

	constexpr static uint32_t DATA_PADDING = MAX(alignof(T), alignof(uint32_t)) - alignof(uint32_t);

public:
	_FORCE_INLINE_ constexpr FixedVector() = default;

	constexpr FixedVector(std::initializer_list<T> p_init) = delete;
	constexpr explicit FixedVector(RelocateInitList<T> p_init) {
		memcpy((void *)_data, p_init.ptr, p_init.size * sizeof(T));
		_size = p_init.size;
	}

	constexpr FixedVector(const FixedVector &p_from) {
		if constexpr (std::is_trivially_copyable_v<T>) {
			// Copy size and all provided elements at once.
			memcpy((void *)&_size, (void *)&p_from._size, sizeof(_size) + DATA_PADDING + p_from.size() * sizeof(T));
		} else {
			for (const T &element : p_from) {
				memnew_placement(ptr() + _size++, T(element));
			}
		}
	}

	constexpr FixedVector(FixedVector &&p_from) {
		// Copy size and all provided elements at once.
		// Note: Assumes trivial relocatability.
		memcpy((void *)&_size, (void *)&p_from._size, sizeof(_size) + DATA_PADDING + p_from.size() * sizeof(T));
		p_from._size = 0;
	}

	template <typename... Args>
	static FixedVector make(Args &&...args) {
		static_assert(sizeof...(Args) <= CAPACITY);
		RelocateInitData<T, sizeof...(Args)> data(std::forward<Args>(args)...);
		return FixedVector(data);
	}

	constexpr FixedVector &operator=(const FixedVector &p_from) {
		if constexpr (std::is_trivially_copyable_v<T>) {
			// Copy size and all provided elements at once.
			memcpy((void *)&_size, (void *)&p_from._size, sizeof(_size) + DATA_PADDING + p_from.size() * sizeof(T));
		} else {
			// Destruct extraneous elements.
			if constexpr (!std::is_trivially_destructible_v<T>) {
				for (uint32_t i = p_from.size(); i < _size; i++) {
					ptr()[i].~T();
				}
			}

			_size = 0; // Loop-assign the rest.
			for (const T &element : p_from) {
				ptr()[_size++] = element;
			}
		}
		return *this;
	}

	constexpr FixedVector &operator=(FixedVector &&p_from) {
		// Destruct extraneous elements.
		if constexpr (!std::is_trivially_destructible_v<T>) {
			for (uint32_t i = p_from.size(); i < _size; i++) {
				ptr()[i].~T();
			}
		}

		// Relocate elements (and size) into our buffer.
		memcpy((void *)&_size, (void *)&p_from._size, sizeof(_size) + DATA_PADDING + p_from.size() * sizeof(T));
		p_from._size = 0;

		return *this;
	}

	~FixedVector() {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			for (uint32_t i = 0; i < _size; i++) {
				ptr()[i].~T();
			}
		}
	}

	_FORCE_INLINE_ constexpr T *ptr() { return (T *)(_data); }
	_FORCE_INLINE_ constexpr const T *ptr() const { return (const T *)(_data); }

	_FORCE_INLINE_ constexpr operator Span<T>() const { return Span<T>(ptr(), size()); }
	_FORCE_INLINE_ constexpr Span<T> span() const { return operator Span<T>(); }

	_FORCE_INLINE_ constexpr uint32_t size() const { return _size; }
	_FORCE_INLINE_ constexpr bool is_empty() const { return !_size; }
	_FORCE_INLINE_ constexpr bool is_full() const { return _size == CAPACITY; }
	_FORCE_INLINE_ constexpr uint32_t capacity() const { return CAPACITY; }

	_FORCE_INLINE_ constexpr void clear() { resize_initialized(0); }

	/// Changes the size of the vector.
	/// If p_size > size(), constructs new elements.
	/// If p_size < size(), destructs new elements.
	constexpr Error resize_initialized(uint32_t p_size) {
		if (p_size > _size) {
			ERR_FAIL_COND_V(p_size > CAPACITY, ERR_OUT_OF_MEMORY);
			memnew_arr_placement(ptr() + _size, p_size - _size);
		} else if (p_size < _size) {
			if constexpr (!std::is_trivially_destructible_v<T>) {
				for (uint32_t i = p_size; i < _size; i++) {
					ptr()[i].~T();
				}
			}
		}

		_size = p_size;
		return OK;
	}

	/// Changes the size of the vector.
	/// The initializer of new elements is skipped, making this function faster than resize_initialized.
	/// The caller is required to initialize the new values.
	constexpr Error resize_uninitialized(uint32_t p_size) {
		static_assert(std::is_trivially_destructible_v<T>, "resize_uninitialized is unsafe to call if T is not trivially destructible.");
		ERR_FAIL_COND_V(p_size > CAPACITY, ERR_OUT_OF_MEMORY);
		_size = p_size;
		return OK;
	}

	constexpr void push_back(const T &p_val) {
		ERR_FAIL_COND(_size >= CAPACITY);
		memnew_placement(ptr() + _size, T(p_val));
		_size++;
	}

	constexpr void push_back(T &&p_val) {
		ERR_FAIL_COND(_size >= CAPACITY);
		memnew_placement(ptr() + _size, T(std::move(p_val)));
		_size++;
	}

	constexpr void pop_back() {
		ERR_FAIL_COND(_size == 0);
		_size--;
		ptr()[_size].~T();
	}

	// NOTE: Subscripts sanity check the bounds to avoid undefined behavior.
	//       This is slower than direct buffer access and can prevent autovectorization.
	//       If the bounds are known, use ptr() subscript instead.
	constexpr const T &operator[](uint32_t p_index) const {
		CRASH_COND(p_index >= _size);
		return ptr()[p_index];
	}

	constexpr T &operator[](uint32_t p_index) {
		CRASH_COND(p_index >= _size);
		return ptr()[p_index];
	}

	_FORCE_INLINE_ constexpr T *begin() { return ptr(); }
	_FORCE_INLINE_ constexpr T *end() { return ptr() + _size; }

	_FORCE_INLINE_ constexpr const T *begin() const { return ptr(); }
	_FORCE_INLINE_ constexpr const T *end() const { return ptr() + _size; }
};

GODOT_GCC_WARNING_POP
