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
	constexpr BitField(int64_t p_value) { value = p_value; }
	constexpr BitField(T p_value) { value = static_cast<enum_t>(p_value); }
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

	constexpr explicit BitField(int64_t p_value) { value = static_cast<enum_t>(p_value); }
	constexpr explicit BitField(int32_t p_value) { value = static_cast<enum_t>(p_value); }
	constexpr explicit BitField(int16_t p_value) { value = static_cast<enum_t>(p_value); }
	constexpr explicit BitField(int8_t p_value) { value = static_cast<enum_t>(p_value); }
	constexpr explicit BitField(uint64_t p_value) { value = static_cast<enum_t>(p_value); }
	constexpr explicit BitField(uint32_t p_value) { value = static_cast<enum_t>(p_value); }
	constexpr explicit BitField(uint16_t p_value) { value = static_cast<enum_t>(p_value); }
	constexpr explicit BitField(uint8_t p_value) { value = static_cast<enum_t>(p_value); }
	constexpr explicit BitField(wchar_t p_value) { value = static_cast<enum_t>(p_value); }
	constexpr explicit BitField(char16_t p_value) { value = static_cast<enum_t>(p_value); }
	constexpr explicit BitField(char32_t p_value) { value = static_cast<enum_t>(p_value); }
	constexpr explicit operator int64_t() const { return static_cast<int64_t>(value); }
	constexpr explicit operator int32_t() const { return static_cast<int32_t>(value); }
	constexpr explicit operator int16_t() const { return static_cast<int16_t>(value); }
	constexpr explicit operator int8_t() const { return static_cast<int8_t>(value); }
	constexpr explicit operator uint64_t() const { return static_cast<uint64_t>(value); }
	constexpr explicit operator uint32_t() const { return static_cast<uint32_t>(value); }
	constexpr explicit operator uint16_t() const { return static_cast<uint16_t>(value); }
	constexpr explicit operator uint8_t() const { return static_cast<uint8_t>(value); }
	constexpr explicit operator wchar_t() const { return static_cast<wchar_t>(value); }
	constexpr explicit operator char16_t() const { return static_cast<char16_t>(value); }
	constexpr explicit operator char32_t() const { return static_cast<char32_t>(value); }
};

#endif // BIT_FIELD_H
