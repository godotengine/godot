/**************************************************************************/
/*  fixed_array.h                                                         */
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

#include "core/templates/span.h"

/**
 * A high performance array of fixed size, analogous to std::array.
 * Especially useful if you need to create arrays statically,
 * or return a known number of elements from a function.
 *
 */
template <class T, uint32_t SIZE>
class FixedArray {
	alignas(T) uint8_t _data[SIZE * sizeof(T)];

public:
	constexpr FixedArray() {
		for (uint32_t i = 0; i < SIZE; i++) {
			memnew_placement(ptr() + i, T);
		}
	}

	constexpr FixedArray(std::initializer_list<T> p_init) {
		CRASH_COND(p_init.size() != SIZE);

		if constexpr (std::is_trivially_copyable_v<T>) {
			memcpy((void *)_data, (void *)p_init.begin(), SIZE * sizeof(T));
		} else {
			for (uint32_t i = 0; i < SIZE; i++) {
				memnew_placement(ptr() + i, T(p_init.begin()[i]));
			}
		}
	}

	constexpr FixedArray(const FixedArray &p_from) {
		if constexpr (std::is_trivially_copyable_v<T>) {
			memcpy((void *)_data, (void *)p_from._data, SIZE * sizeof(T));
		} else {
			for (uint32_t i = 0; i < SIZE; i++) {
				memnew_placement(ptr() + i, T(p_from.ptr()[i]));
			}
		}
	}

	constexpr FixedArray(FixedArray &&p_from) {
		if constexpr (std::is_trivially_copyable_v<T>) {
			memcpy((void *)_data, (void *)p_from._data, SIZE * sizeof(T));
		} else {
			for (uint32_t i = 0; i < SIZE; i++) {
				memnew_placement(ptr() + i, T(std::move(p_from.ptr()[i])));
			}
		}
	}

	constexpr FixedArray &operator=(const FixedArray &p_from) {
		if constexpr (std::is_trivially_copyable_v<T>) {
			memcpy((void *)_data, (void *)p_from._data, SIZE * sizeof(T));
		} else {
			for (uint32_t i = 0; i < SIZE; i++) {
				ptr()[i] = p_from.ptr()[i];
			}
		}
		return *this;
	}

	constexpr FixedArray &operator=(FixedArray &&p_from) {
		if constexpr (std::is_trivially_copyable_v<T>) {
			memcpy((void *)_data, (void *)p_from._data, SIZE * sizeof(T));
		} else {
			for (uint32_t i = 0; i < SIZE; i++) {
				ptr()[i] = std::move(p_from.ptr()[i]);
			}
		}
		return *this;
	}

	~FixedArray() {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			for (uint32_t i = 0; i < SIZE; i++) {
				ptr()[i].~T();
			}
		}
	}

	_FORCE_INLINE_ constexpr T *ptr() { return (T *)(_data); }
	_FORCE_INLINE_ constexpr const T *ptr() const { return (const T *)(_data); }

	_FORCE_INLINE_ constexpr operator Span<T>() const { return Span<T>(ptr(), SIZE); }
	_FORCE_INLINE_ constexpr Span<T> span() const { return operator Span<T>(); }

	_FORCE_INLINE_ static constexpr uint32_t size() { return SIZE; }

	// NOTE: Subscripts sanity check the bounds to avoid undefined behavior.
	//       This is slower than direct buffer access and can prevent autovectorization.
	//       If the bounds are known, use ptr() subscript instead.
	constexpr const T &operator[](uint32_t p_index) const {
		CRASH_COND(p_index >= SIZE);
		return ptr()[p_index];
	}

	constexpr T &operator[](uint32_t p_index) {
		CRASH_COND(p_index >= SIZE);
		return ptr()[p_index];
	}

	_FORCE_INLINE_ constexpr T *begin() { return ptr(); }
	_FORCE_INLINE_ constexpr T *end() { return ptr() + SIZE; }

	_FORCE_INLINE_ constexpr const T *begin() const { return ptr(); }
	_FORCE_INLINE_ constexpr const T *end() const { return ptr() + SIZE; }
};
