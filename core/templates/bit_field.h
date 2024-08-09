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

#ifndef BIT_FIELD_H
#define BIT_FIELD_H

#include "core/typedefs.h"

#include <type_traits>

template <typename T, typename = void>
class BitField {};

// Legacy enums

template <typename T>
class BitField<T, std::enable_if_t<std::is_enum_v<T> && std::is_convertible_v<T, std::underlying_type_t<T>>>> {
	using enum_t = std::underlying_type_t<T>;
	enum_t value = 0;

public:
	constexpr void set_flag(T p_flag) { value |= static_cast<enum_t>(p_flag); }
	constexpr bool has_flag(T p_flag) const { return value & static_cast<enum_t>(p_flag); }
	constexpr bool is_empty() const { return value == 0; }
	constexpr void clear_flag(T p_flag) { value &= ~static_cast<enum_t>(p_flag); }
	constexpr void clear() { value = 0; }

	constexpr BitField() = default;
	constexpr BitField(T p_value) { value = static_cast<enum_t>(p_value); }
	constexpr explicit operator T() const { return static_cast<T>(value); }

	constexpr BitField(int64_t p_value) { value = static_cast<enum_t>(p_value); }
	constexpr operator int64_t() const { return static_cast<int64_t>(value); }
};

// Enum classes

template <typename T>
class BitField<T, std::enable_if_t<std::is_enum_v<T> && !std::is_convertible_v<T, std::underlying_type_t<T>>>> {
	using enum_t = std::underlying_type_t<T>;
	enum_t value = 0;

public:
	constexpr void set_flag(T p_flag) { value |= static_cast<enum_t>(p_flag); }
	constexpr bool has_flag(T p_flag) const { return value & static_cast<enum_t>(p_flag); }
	constexpr bool is_empty() const { return value == 0; }
	constexpr void clear_flag(T p_flag) { value &= ~static_cast<enum_t>(p_flag); }
	constexpr void clear() { value = 0; }

	constexpr BitField() = default;
	constexpr BitField(T p_value) { value = static_cast<enum_t>(p_value); }
	constexpr operator T() const { return static_cast<T>(value); }

	template <typename V, std::enable_if_t<std::is_arithmetic_v<V>, int> = 0>
	constexpr explicit BitField(V p_value) { value = static_cast<enum_t>(p_value); }
	template <typename V, std::enable_if_t<std::is_arithmetic_v<V>, int> = 0>
	constexpr explicit operator V() const { return static_cast<V>(value); }
};

#endif // BIT_FIELD_H
