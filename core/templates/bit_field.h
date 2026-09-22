/**************************************************************************/
/*  bit_field.h                                                           */
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

#include <type_traits>

// TODO: Replace `typename` with enum concept once C++20 concepts/constraints are allowed.

template <typename T>
class BitField {
	static_assert(std::is_enum_v<T>);
	uint64_t value;

public:
	_ALWAYS_INLINE_ constexpr void set_flag(BitField p_flag) { value |= p_flag.value; }
	_ALWAYS_INLINE_ constexpr bool has_flag(BitField p_flag) const { return value & p_flag.value; }
	_ALWAYS_INLINE_ constexpr bool is_empty() const { return value == 0; }
	_ALWAYS_INLINE_ constexpr void clear_flag(BitField p_flag) { value &= ~p_flag.value; }
	_ALWAYS_INLINE_ constexpr void clear() { value = 0; }

	[[nodiscard]] _ALWAYS_INLINE_ constexpr BitField get_combined(BitField p_other) const { return BitField(value | p_other.value); }
	[[nodiscard]] _ALWAYS_INLINE_ constexpr BitField get_shared(BitField p_other) const { return BitField(value & p_other.value); }
	[[nodiscard]] _ALWAYS_INLINE_ constexpr BitField get_different(BitField p_other) const { return BitField(value ^ p_other.value); }

	_ALWAYS_INLINE_ constexpr BitField() = default;
	_ALWAYS_INLINE_ constexpr BitField(T p_value) :
			value(static_cast<uint64_t>(p_value)) {}
	_ALWAYS_INLINE_ constexpr operator T() const { return static_cast<T>(value); }

	// TODO: Unify as single constructor once C++20 `explicit` conditionals are allowed.

	template <typename V, std::enable_if_t<std::is_arithmetic_v<V> && std::is_convertible_v<T, int>, int> = 0>
	_ALWAYS_INLINE_ constexpr BitField(V p_value) :
			value(static_cast<uint64_t>(p_value)) {}
	template <typename V, std::enable_if_t<std::is_arithmetic_v<V> && !std::is_convertible_v<T, int>, int> = 0>
	_ALWAYS_INLINE_ constexpr explicit BitField(V p_value) :
			value(static_cast<uint64_t>(p_value)) {}
	template <typename V, std::enable_if_t<std::is_arithmetic_v<V>, int> = 0>
	_ALWAYS_INLINE_ constexpr explicit operator V() const { return static_cast<V>(value); }
};

// Implicitly zero-constructible as a trivially-constructible type.
static_assert(is_zero_constructible_v<BitField<Error>>);
