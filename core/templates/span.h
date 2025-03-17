/**************************************************************************/
/*  span.h                                                                */
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

#include "core/typedefs.h"

// Equivalent of std::span.
// Represents a view into a contiguous memory space.
// DISCLAIMER: This data type does not own the underlying buffer. DO NOT STORE IT.
//  Additionally, for the lifetime of the Span, do not resize the buffer, and do not insert or remove elements from it.
//  Failure to respect this may lead to crashes or undefined behavior.
template <typename T>
class Span {
	const T *_ptr = nullptr;
	uint64_t _len = 0;

public:
	_FORCE_INLINE_ constexpr Span() = default;
	_FORCE_INLINE_ constexpr Span(const T *p_ptr, uint64_t p_len) :
			_ptr(p_ptr), _len(p_len) {}

	_FORCE_INLINE_ constexpr uint64_t size() const { return _len; }
	_FORCE_INLINE_ constexpr bool is_empty() const { return _len == 0; }

	_FORCE_INLINE_ constexpr const T *ptr() const { return _ptr; }

	// NOTE: Span subscripts sanity check the bounds to avoid undefined behavior.
	//       This is slower than direct buffer access and can prevent autovectorization.
	//       If the bounds are known, use ptr() subscript instead.
	_FORCE_INLINE_ constexpr const T &operator[](uint64_t p_idx) const {
		CRASH_COND(p_idx >= _len);
		return _ptr[p_idx];
	}

	_FORCE_INLINE_ constexpr const T *begin() const { return _ptr; }
	_FORCE_INLINE_ constexpr const T *end() const { return _ptr + _len; }

	// Algorithms.
	constexpr int64_t find(const T &p_val, int64_t p_from = 0) const;
	constexpr int64_t rfind(const T &p_val, int64_t p_from = 0) const;
	constexpr uint64_t count(const T &p_val) const;
};

template <typename T>
constexpr int64_t Span<T>::find(const T &p_val, int64_t p_from) const {
	if (p_from < 0 || size() == 0) {
		return -1;
	}

	for (uint64_t i = p_from; i < size(); i++) {
		if (ptr()[i] == p_val) {
			return i;
		}
	}

	return -1;
}

template <typename T>
constexpr int64_t Span<T>::rfind(const T &p_val, int64_t p_from) const {
	const int64_t s = size();

	if (p_from < 0) {
		p_from = s + p_from;
	}
	if (p_from < 0 || p_from >= s) {
		p_from = s - 1;
	}

	for (int64_t i = p_from; i >= 0; i--) {
		if (ptr()[i] == p_val) {
			return i;
		}
	}
	return -1;
}

template <typename T>
constexpr uint64_t Span<T>::count(const T &p_val) const {
	uint64_t amount = 0;
	for (uint64_t i = 0; i < size(); i++) {
		if (ptr()[i] == p_val) {
			amount++;
		}
	}
	return amount;
}

template <typename LHS, typename RHS>
bool operator==(const Span<LHS> &p_lhs, const Span<RHS> &p_rhs) {
	if (p_lhs.size() != p_rhs.size()) {
		return false;
	}

	if constexpr (std::is_same_v<LHS, RHS> && std::is_fundamental_v<LHS>) {
		// Optimize trivial type comparison.
		// is_trivially_equality_comparable would help, but it doesn't exist.
		return memcmp(p_lhs.ptr(), p_rhs.ptr(), p_lhs.size() * sizeof(LHS)) == 0;
	} else if constexpr (std::is_fundamental_v<LHS> && std::is_fundamental_v<RHS> && (std::is_signed_v<LHS> != std::is_signed_v<RHS>)) {
		// Special case: Comparing a signed and an unsigned type.
		// This is undefined behavior, so we need to cast to a common type before comparison.
		using CommonType = std::common_type_t<LHS, RHS>;

		// Normal case: Need to iterate the array manually.
		for (size_t j = 0; j < p_lhs.size(); j++) {
			if (static_cast<CommonType>(p_lhs.ptr()[j]) != static_cast<CommonType>(p_rhs.ptr()[j])) {
				return false;
			}
		}

		return true;
	} else {
		// Normal case: Need to iterate the array manually.
		for (size_t j = 0; j < p_lhs.size(); j++) {
			if (p_lhs.ptr()[j] != p_rhs.ptr()[j]) {
				return false;
			}
		}

		return true;
	}
}

template <typename LHS, typename RHS>
_FORCE_INLINE_ bool operator!=(const Span<LHS> &p_lhs, const Span<RHS> &p_rhs) {
	return !(p_lhs == p_rhs);
}
