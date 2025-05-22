/**************************************************************************/
/*  required_obj.h                                                        */
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

#include "core/object/object.h"
#include "core/templates/simple_type.h"

#include <compare>
#include <source_location>

namespace Internal {

template <typename T, typename = void>
struct GetElementType;

template <typename T>
struct GetElementType<T, std::enable_if_t<!std::is_same_v<T, GetSimpleTypeT<T>>>> : GetElementType<GetSimpleTypeT<T>> {};

template <typename T>
struct GetElementType<T *> {
	using element_type = GetSimpleTypeT<T>;
};

template <typename T>
struct GetElementType<Ref<T>> {
	using element_type = T;
};

} // namespace Internal

template <typename T>
class RequiredObj {
	static_assert(!std::is_scalar_v<T>, "T may not be a scalar type");
	static_assert(!std::is_reference_v<T>, "T may not be a reference type");
	static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>, "T may not be cv-qualified");

	T *_value = nullptr;

	_ALWAYS_INLINE_ void _null_validate(const std::source_location p_source) {
		if (_value == nullptr) [[unlikely]] {
			_err_print_error(p_source.function_name(), p_source.file_name(), p_source.line(), "Null value was passed to a required object. This is an issue in the engine, and should be reported.");
		}
	}

	constexpr RequiredObj() = default;

public:
	using element_type = T;

	// RequiredObj variables assumed to already have validated contents; no check required.
	constexpr RequiredObj(const RequiredObj &p_other) = default;
	constexpr RequiredObj(RequiredObj &&p_other) = default;
	constexpr RequiredObj &operator=(const RequiredObj &p_other) = default;
	constexpr RequiredObj &operator=(RequiredObj &&p_other) = default;

	// RequiredObj of different type also validated, albeit with a derived-type constraint.
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	constexpr RequiredObj(const RequiredObj<T_Other> &p_other) :
			_value(const_cast<T_Other *>(p_other.ptr())) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	constexpr RequiredObj &operator=(const RequiredObj<T_Other> &p_other) {
		_value = const_cast<T_Other *>(p_other.ptr());
		return *this;
	}

	// Non-validated types pass along source location metadata.
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_ALWAYS_INLINE_ RequiredObj(const T_Other *p_ptr, const std::source_location p_source = std::source_location::current()) :
			_value(const_cast<std::remove_const_t<T_Other> *>(p_ptr)) {
		_null_validate(p_source);
	}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_ALWAYS_INLINE_ RequiredObj(const Ref<T_Other> &p_ref, const std::source_location p_source = std::source_location::current()) :
			_value(p_ref.ptr()) {
		_null_validate(p_source);
	}

	// Assignment operators excluded for non-validated types, as they cannot be passed location
	//  metadata. This has the benefit of requiring assignments to be explicit.
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	RequiredObj &operator=(const T_Other *) = delete;
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	RequiredObj &operator=(const Ref<T_Other> &) = delete;

	// Prevent erroneously assigning null values by explicitly removing nullptr constructor/assignment.
	RequiredObj(std::nullptr_t) = delete;
	RequiredObj &operator=(std::nullptr_t) = delete;

	// As an alternative to non-validated constructors, RequiredObj supports construction directly.
	//  This doesn't need a validation step, as there's guaranteed to be *something* after `memnew`.
	template <typename... VarArgs>
	static constexpr RequiredObj construct(VarArgs... p_params) {
		RequiredObj tmp = RequiredObj();
		tmp._value = memnew(T(p_params...));
		return tmp;
	}

	// We will occasionally need to return an empty RequiredObj when the function itself encounters
	//  an error. However, we don't want to use the default construction methods, as that would note
	//  that this intended fallback is an engine bug. Instead, this function silently returns null,
	//  but within a deprecated wrapper, so that it cannot be used directly within the codebase.
	//  Deprecation-suppression macros surround the return value of error macros, meaning this should
	//  only be possible to call in that specific context.
	[[nodiscard, deprecated("Should not be called directly; must be used as return argument in error macro.")]]
	static constexpr RequiredObj silent_null() { return RequiredObj<T>(); }

	[[nodiscard]] constexpr T *operator->() { return _value; }
	[[nodiscard]] constexpr const T *operator->() const { return _value; }
	[[nodiscard]] constexpr T *ptr() { return _value; }
	[[nodiscard]] constexpr const T *ptr() const { return _value; }

	[[nodiscard]] constexpr bool is_null() const { return _value == nullptr; }
	[[nodiscard]] constexpr bool is_valid() const { return _value != nullptr; }

	// Comparison operators.
	[[nodiscard]] constexpr std::strong_ordering operator<=>(const RequiredObj &p_other) const = default;
	[[nodiscard]] constexpr bool operator==(const RequiredObj &p_other) const = default;

	template <typename T_Other>
	[[nodiscard]] constexpr std::strong_ordering operator<=>(const RequiredObj<T_Other> &p_other) const { return ptr() <=> p_other.ptr(); }
	template <typename T_Other>
	[[nodiscard]] constexpr bool operator==(const RequiredObj<T_Other> &p_other) const { return operator<=>(p_other) == 0; }

	template <typename T_Other>
	[[nodiscard]] constexpr std::strong_ordering operator<=>(const T_Other *p_ptr) const { return ptr() <=> p_ptr; }
	template <typename T_Other>
	[[nodiscard]] constexpr bool operator==(const T_Other *p_ptr) const { return operator<=>(p_ptr) == 0; }

	template <typename T_Other>
	[[nodiscard]] constexpr std::strong_ordering operator<=>(const Ref<T_Other> &p_ref) const { return ptr() <=> p_ref.ptr(); }
	template <typename T_Other>
	[[nodiscard]] constexpr bool operator==(const Ref<T_Other> &p_ref) const { return operator<=>(p_ref) == 0; }

	// Operators we never want.
	bool operator==(std::nullptr_t) const = delete;
	RequiredObj &operator++() = delete;
	RequiredObj &operator--() = delete;
	RequiredObj operator++(int) = delete;
	RequiredObj operator--(int) = delete;
	RequiredObj operator+(size_t) const = delete;
	RequiredObj &operator+=(size_t) = delete;
	RequiredObj operator-(size_t) const = delete;
	RequiredObj &operator-=(size_t) = delete;
	RequiredObj operator+(std::ptrdiff_t) const = delete;
	RequiredObj &operator+=(std::ptrdiff_t) = delete;
	RequiredObj operator-(std::ptrdiff_t) const = delete;
	RequiredObj &operator-=(std::ptrdiff_t) = delete;
	void operator[](std::ptrdiff_t) const = delete;
	template <typename T_Other>
	std::ptrdiff_t operator+(const RequiredObj<T_Other> &) const = delete;
	template <typename T_Other>
	std::ptrdiff_t operator-(const RequiredObj<T_Other> &) const = delete;
};

template <typename T>
RequiredObj(const T *) -> RequiredObj<T>;
template <typename T>
RequiredObj(const Ref<T> &) -> RequiredObj<T>;

// Technically zero-constructible, but not recommended.
template <typename T>
struct is_zero_constructible<RequiredObj<T>> : std::true_type {};
