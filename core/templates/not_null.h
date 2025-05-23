/**************************************************************************/
/*  not_null.h                                                            */
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

// Implementation derived heavily from gsl-lite:
// https://github.com/gsl-lite/gsl-lite

template <typename T, bool Fatal = false>
class NotNull;

template <typename T>
using NeverNull = NotNull<T, true>;

namespace Internal {

// Helper to figure out whether a pointer has an element type.
// Avoid SFINAE for unary `operator*` (doesn't work for `std::unique_ptr<>` and the like) if an `element_type` member exists.
template <typename T, typename E = void>
struct has_element_type_ : std::false_type {};
template <typename T>
struct has_element_type_<T, std::void_t<decltype(*std::declval<T>())>> : std::true_type {};
template <typename T, typename E = void>
struct has_element_type : has_element_type_<T> {};
template <typename T>
struct has_element_type<T, std::void_t<typename T::element_type>> : std::true_type {};

template <typename T, typename E = void>
inline constexpr bool has_element_type_v = has_element_type<T, E>::value;

// Helper to figure out the pointed-to type of a pointer.
template <typename T, typename E = void>
struct element_type_helper {
	// For types without a member element_type (this could handle typed raw pointers but not `void*`)
	using type = std::remove_reference_t<decltype(*std::declval<T>())>;
};
template <typename T>
struct element_type_helper<T, std::void_t<typename T::element_type>> {
	// For types with a member element_type
	using type = typename T::element_type;
};
template <typename T>
struct element_type_helper<T *> {
	using type = T;
};

template <typename T>
using element_type_helper_t = typename element_type_helper<T>::type;

template <typename T>
struct is_not_null_or_bool_oracle : std::false_type {};
template <typename T, bool Fatal>
struct is_not_null_or_bool_oracle<NotNull<T, Fatal>> : std::true_type {};
template <>
struct is_not_null_or_bool_oracle<bool> : std::true_type {};

template <typename T>
inline constexpr bool is_not_null_or_bool_oracle_v = is_not_null_or_bool_oracle<T>::value;

template <typename T, bool Fatal, bool IsCopyable>
struct not_null_data;

template <typename T, bool Fatal>
struct not_null_data<T, Fatal, false> {
	T _ptr;

	constexpr not_null_data(T &&p_ptr) :
			_ptr(std::move(p_ptr)) {}

	constexpr not_null_data(not_null_data &&p_other) :
			_ptr(std::move(p_other._ptr)) {
		if constexpr (Fatal) {
			CRASH_COND(_ptr == nullptr);
		} else {
			ERR_FAIL_NULL(_ptr);
		}
	}
	constexpr not_null_data &operator=(not_null_data &&p_other) {
		if (unlikely(&p_other == this)) {
			return *this;
		}
		if constexpr (Fatal) {
			CRASH_COND(p_other._ptr == nullptr);
		} else {
			ERR_FAIL_NULL_V(p_other._ptr, *this);
		}
		_ptr = std::move(p_other._ptr);
		return *this;
	}

	not_null_data(const not_null_data &) = delete;
	not_null_data &operator=(const not_null_data &) = delete;
};

template <typename T, bool Fatal>
struct not_null_data<T, Fatal, true> {
	T _ptr;

	constexpr not_null_data(const T &p_ptr) :
			_ptr(p_ptr) {}

	constexpr not_null_data(T &&p_ptr) :
			_ptr(std::move(p_ptr)) {}

	constexpr not_null_data(not_null_data &&p_other) :
			_ptr(std::move(p_other._ptr)) {
		if constexpr (Fatal) {
			CRASH_COND(_ptr == nullptr);
		} else {
			ERR_FAIL_NULL(_ptr);
		}
	}
	constexpr not_null_data &operator=(not_null_data &&p_other) {
		if (unlikely(&p_other == this)) {
			return *this;
		}
		if constexpr (Fatal) {
			CRASH_COND(p_other._ptr == nullptr);
		} else {
			ERR_FAIL_NULL_V(p_other._ptr, *this);
		}
		_ptr = std::move(p_other._ptr);
		return *this;
	}

	constexpr not_null_data(const not_null_data &p_other) :
			_ptr(p_other._ptr) {
		if constexpr (Fatal) {
			CRASH_COND(_ptr == nullptr);
		} else {
			ERR_FAIL_NULL(_ptr);
		}
	}
	constexpr not_null_data &operator=(const not_null_data &p_other) {
		if constexpr (Fatal) {
			CRASH_COND(p_other._ptr == nullptr);
		} else {
			ERR_FAIL_NULL_V(p_other._ptr, *this);
		}
		_ptr = p_other._ptr;
		return *this;
	}
};

template <typename T, bool Fatal>
struct not_null_data<T *, Fatal, true> {
	T *_ptr;

	constexpr not_null_data(T *p_ptr) :
			_ptr(p_ptr) {}
};

template <typename T>
struct is_copyable : std::bool_constant<std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T>> {};

template <typename T>
inline constexpr bool is_copyable_v = is_copyable<T>::value;

template <typename T>
struct not_null_accessor;

template <typename Derived, typename T, bool Fatal, bool HasElementType>
struct not_null_elem {
	using element_type = element_type_helper_t<T>;

	[[nodiscard]] constexpr element_type *get() const {
		return not_null_accessor<T>::get_checked<Fatal>(static_cast<const Derived &>(*this)).get();
	}
};
template <typename Derived, typename T, bool Fatal>
struct not_null_elem<Derived, T, Fatal, false> {};

template <typename Derived, typename T, bool Fatal, bool IsDereferencable>
struct not_null_deref : not_null_elem<Derived, T, Fatal, has_element_type_v<T>> {
	using element_type = element_type_helper_t<T>;

	[[nodiscard]] constexpr element_type &operator*() const {
		return *not_null_accessor<T>::get_checked<Fatal>(static_cast<const Derived &>(*this));
	}
};

template <typename Derived, typename T, bool Fatal>
struct not_null_deref<Derived, T, Fatal, false> : not_null_elem<Derived, T, Fatal, has_element_type_v<T>> {};

template <typename T>
struct is_void_ptr : std::is_void<element_type_helper_t<T>> {};

template <typename T>
struct is_dereferencable : std::conjunction<has_element_type<T>, std::negation<is_void_ptr<T>>> {};

template <typename T>
inline constexpr bool is_dereferencable_v = is_dereferencable<T>::value;

} // namespace Internal

template <typename T>
struct is_nullable : std::is_assignable<std::remove_cv_t<T> &, std::nullptr_t> {};

template <typename T>
inline constexpr bool is_nullable_v = is_nullable<T>::value;

template <typename T, bool Fatal>
class NotNull : public Internal::not_null_deref<NotNull<T, Fatal>, T, Fatal, Internal::is_dereferencable_v<T>> {
	static_assert(!std::is_reference_v<T>, "T may not be a reference type");
	static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>, "T may not be cv-qualified");
	static_assert(is_nullable_v<T>, "T must be a nullable type");

	Internal::not_null_data<T, Fatal, Internal::is_copyable_v<T>> _data;

	// need to access `NotNull<U>::_data`
	template <typename U>
	friend struct Internal::not_null_accessor;

	using accessor = Internal::not_null_accessor<T>;

public:
	// Cannot be constructed as empty, nor via `nullptr`.
	NotNull() = delete;
	NotNull(std::nullptr_t) = delete;

	// `NotNull` of same type.
	constexpr NotNull(const NotNull &p_other) = default;
	constexpr NotNull(NotNull &&p_other) = default;
	constexpr NotNull &operator=(const NotNull &p_other) = default;
	constexpr NotNull &operator=(NotNull &&p_other) = default;

	// `NotNull` of different type.
	template <typename U, bool UFatal, std::enable_if_t<std::is_constructible_v<T, U> && !std::is_convertible_v<U, T>, int> = 0>
	constexpr explicit NotNull(NotNull<U, UFatal> p_other) :
			_data(T(Internal::not_null_accessor<U>::get_checked<Fatal || UFatal>(std::move(p_other)))) {}

	template <typename U, bool UFatal, std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
	constexpr NotNull(NotNull<U, UFatal> p_other) :
			_data(T(Internal::not_null_accessor<U>::get_checked<Fatal || UFatal>(std::move(p_other)))) {}

	// Fatal constructors (explicit).
	template <typename U, std::enable_if_t<Fatal && std::is_constructible_v<T, U> && is_nullable_v<U>, int> = 0>
	constexpr explicit NotNull(U p_other) :
			_data(T(std::move(p_other))) {
		CRASH_COND(_data._ptr == nullptr);
	}
	template <typename U, std::enable_if_t<Fatal && std::is_constructible_v<T, U> && std::is_function_v<U>, int> = 0>
	constexpr NotNull(const U &p_other) :
			_data(T(p_other)) {}

	template <typename U, std::enable_if_t<Fatal && std::is_constructible_v<T, U> && !std::is_function_v<U> && !is_nullable_v<U>, int> = 0>
	constexpr NotNull(U p_other) :
			_data(T(std::move(p_other))) {
		CRASH_COND(_data._ptr == nullptr);
	}

	// Nonfatal constructors (implicit).
	template <typename U, std::enable_if_t<!Fatal && std::is_constructible_v<T, U> && !std::is_convertible_v<U, T>, int> = 0>
	constexpr explicit NotNull(U other) :
			_data(T(std::move(other))) {
		ERR_FAIL_NULL(_data._ptr);
	}
	template <typename U, std::enable_if_t<!Fatal && std::is_convertible_v<U, T>, int> = 0>
	constexpr NotNull(U other) :
			_data(std::move(other)) {
		ERR_FAIL_NULL(_data._ptr);
	}

	// Conversion operators from explicitly convertible types.
	template <typename U, std::enable_if_t<std::is_constructible_v<U, const T &> && !std::is_convertible_v<T, U> && !Internal::is_not_null_or_bool_oracle_v<U>, int> = 0>
	[[nodiscard]] constexpr explicit operator U() const & {
		return U(accessor::get_checked<Fatal>(*this));
	}
	template <typename U, std::enable_if_t<std::is_constructible_v<U, T> && !std::is_convertible_v<T, U> && !Internal::is_not_null_or_bool_oracle_v<U>, int> = 0>
	[[nodiscard]] constexpr explicit operator U() && {
		return U(accessor::get_checked<Fatal>(std::move(*this)));
	}

	// Conversion operators from implicitly convertible types.
	template <typename U, std::enable_if_t<std::is_constructible_v<U, const T &> && std::is_convertible_v<T, U> && !Internal::is_not_null_or_bool_oracle_v<U>, int> = 0>
	[[nodiscard]] constexpr operator U() const & {
		return accessor::get_checked<Fatal>(*this);
	}
	template <typename U, std::enable_if_t<std::is_convertible_v<T, U> && !Internal::is_not_null_or_bool_oracle_v<U>, int> = 0>
	[[nodiscard]] constexpr operator U() && {
		return accessor::get_checked<Fatal>(std::move(*this));
	}

	[[nodiscard]] constexpr const T &operator->() const {
		return accessor::get_checked<Fatal>(*this);
	}

	template <typename... Ts>
	constexpr auto operator()(Ts &&...args) const
			-> decltype(_data._ptr(std::forward<Ts>(args)...)) {
		return accessor::get_checked<Fatal>(*this)(std::forward<Ts>(args)...);
	}

	// Comparison operators.
	template <typename U, bool UFatal>
	[[nodiscard]] inline constexpr bool operator==(const NotNull<U, UFatal> &p_right) { return (*this).operator->() == p_right.operator->(); }
	template <typename U>
	[[nodiscard]] inline constexpr bool operator==(const U &p_right) { return (*this).operator->() == p_right; }

	template <typename U, bool UFatal>
	[[nodiscard]] inline constexpr bool operator!=(const NotNull<U, UFatal> &p_right) { return !(*this == p_right); }
	template <typename U>
	[[nodiscard]] inline constexpr bool operator!=(const U &p_right) { return !(*this == p_right); }

	template <typename U, bool UFatal>
	[[nodiscard]] inline constexpr bool operator<(const NotNull<U, UFatal> &p_right) { return (*this).operator->() < p_right.operator->(); }
	template <typename U>
	[[nodiscard]] inline constexpr bool operator<(const U &p_right) { return (*this).operator->() < p_right; }

	template <typename U, bool UFatal>
	[[nodiscard]] inline constexpr bool operator<=(const NotNull<U, UFatal> &p_right) { return !(p_right < *this); }
	template <typename U>
	[[nodiscard]] inline constexpr bool operator<=(const U &p_right) { return !(p_right < *this); }

	template <typename U, bool UFatal>
	[[nodiscard]] inline constexpr bool operator>(const NotNull<U, UFatal> &p_right) { return p_right < *this; }
	template <typename U>
	[[nodiscard]] inline constexpr bool operator>(const U &p_right) { return p_right < *this; }

	template <typename U, bool UFatal>
	[[nodiscard]] inline constexpr bool operator>=(const NotNull<U, UFatal> &p_right) { return !(*this < p_right); }
	template <typename U>
	[[nodiscard]] inline constexpr bool operator>=(const U &p_right) { return !(*this < p_right); }

	// Operators we never want.
	NotNull &operator=(std::nullptr_t) = delete;
	NotNull &operator++() = delete;
	NotNull &operator--() = delete;
	NotNull operator++(int) = delete;
	NotNull operator--(int) = delete;
	NotNull operator+(size_t) = delete;
	NotNull &operator+=(size_t) = delete;
	NotNull operator-(size_t) = delete;
	NotNull &operator-=(size_t) = delete;
	NotNull operator+(std::ptrdiff_t) = delete;
	NotNull &operator+=(std::ptrdiff_t) = delete;
	NotNull operator=(std::ptrdiff_t) = delete;
	NotNull &operator-=(std::ptrdiff_t) = delete;
	void operator[](std::ptrdiff_t) const = delete;
	template <typename U, bool UFatal>
	std::ptrdiff_t operator+(const NotNull<U, UFatal> &) = delete;
	template <typename U, bool UFatal>
	std::ptrdiff_t operator-(const NotNull<U, UFatal> &) = delete;
};

// template <typename U>
// NotNull(U) -> NotNull<U, false>;
// template <typename U>
// NotNull(NotNull<U, false>) -> NotNull<U, false>;

void make_not_null(std::nullptr_t) = delete;

template <typename U, bool Fatal>
[[nodiscard]] constexpr NotNull<U, Fatal> make_not_null(U u) {
	return NotNull<U, Fatal>(std::move(u));
}
template <typename U, bool Fatal>
[[nodiscard]] constexpr NotNull<U, Fatal> make_not_null(NotNull<U, Fatal> u) {
	return std::move(u);
}

namespace Internal {

template <typename T>
struct as_nullable_helper {
	using type = std::remove_reference_t<std::remove_cv_t<T>>;
};
template <typename T, bool Fatal>
struct as_nullable_helper<NotNull<T, Fatal>> {};

template <typename T>
using as_nullable_helper_t = typename as_nullable_helper<T>::type;

template <typename T>
struct not_null_accessor {
	template <bool Fatal>
	static T get(NotNull<T, Fatal> &&p_value) {
		return std::move(p_value._data._ptr);
	}
	template <bool Fatal>
	static T get_checked(NotNull<T, Fatal> &&p_value) {
		if constexpr (Fatal) {
			CRASH_COND(p_value._data._ptr == nullptr);
		} else {
			ERR_FAIL_NULL_V(p_value._data._ptr, std::move(p_value._data._ptr));
		}
		return std::move(p_value._data._ptr);
	}
	template <bool Fatal>
	static const T &get(const NotNull<T, Fatal> &p_value) {
		return p_value._data._ptr;
	}
	template <bool Fatal>
	static bool is_valid(const NotNull<T, Fatal> &p_value) {
		return p_value._data._ptr != nullptr;
	}
	template <bool Fatal>
	static void check(const NotNull<T, Fatal> &p_value) {
		if constexpr (Fatal) {
			CRASH_COND(p_value._data._ptr == nullptr);
		} else {
			ERR_FAIL_NULL(p_value._data._ptr);
		}
	}
	template <bool Fatal>
	static const T &get_checked(const NotNull<T, Fatal> &p_value) {
		if constexpr (Fatal) {
			CRASH_COND(p_value._data._ptr == nullptr);
		} else {
			ERR_FAIL_NULL_V(p_value._data._ptr, p_value._data._ptr);
		}
		return p_value._data._ptr;
	}
};
template <typename T>
struct not_null_accessor<T *> {
	template <bool Fatal>
	static T *const &get(const NotNull<T *, Fatal> &p_value) { return p_value._data._ptr; }
	template <bool Fatal>
	static bool is_valid(const NotNull<T *, Fatal> & /*p_value*/) { return true; }
	template <bool Fatal>
	static void checkconst(NotNull<T *, Fatal> & /*p_value*/) {}
	template <bool Fatal>
	static T *const &get_checked(const NotNull<T *, Fatal> &p_value) { return p_value._data._ptr; }
};

namespace NoADL {

template <typename T>
[[nodiscard]] constexpr as_nullable_helper_t<T> as_nullable(T &&p_value) {
	return std::move(p_value);
}

template <typename T, bool Fatal>
[[nodiscard]] constexpr T as_nullable(NotNull<T, Fatal> &&p_value) {
	return not_null_accessor<T>::get_checked<Fatal>(std::move(p_value));
}

template <typename T, bool Fatal>
[[nodiscard]] constexpr const T &as_nullable(const NotNull<T, Fatal> &p_value) {
	return not_null_accessor<T>::get_checked<Fatal>(p_value);
}

template <typename T, bool Fatal>
[[nodiscard]] constexpr bool is_valid(const NotNull<T, Fatal> &p_value) {
	return not_null_accessor<T>::is_valid<Fatal>(p_value);
}

} // namespace NoADL
} // namespace Internal

using namespace Internal::NoADL;

// Unwanted operators.

template <typename T, bool Fatal>
NotNull<T, Fatal> operator+(std::ptrdiff_t, const NotNull<T, Fatal> &) = delete;
template <typename T, bool Fatal>
NotNull<T, Fatal> operator-(std::ptrdiff_t, const NotNull<T, Fatal> &) = delete;
template <typename T /* Fatal == true */>
constexpr bool operator==(const NotNull<T, true> &, std::nullptr_t) = delete;
template <typename T /* Fatal == true */>
constexpr bool operator==(std::nullptr_t, const NotNull<T, true> &) = delete;
template <typename T /* Fatal == true */>
constexpr bool operator!=(const NotNull<T, true> &, std::nullptr_t) = delete;
template <typename T /* Fatal == true */>
constexpr bool operator!=(std::nullptr_t, const NotNull<T, true> &) = delete;

// Right-hand operators. Convert to class methods in C++20.

template <typename T, typename U, bool Fatal>
[[nodiscard]] inline constexpr bool operator==(const T &p_left, const NotNull<U, Fatal> &p_right) {
	return p_left == p_right.operator->();
}
template <typename T, typename U, bool Fatal>
[[nodiscard]] inline constexpr bool operator!=(const T &p_left, const NotNull<U, Fatal> &p_right) {
	return p_left != p_right.operator->();
}
template <typename T, typename U, bool Fatal>
[[nodiscard]] inline constexpr bool operator<(const T &p_left, const NotNull<U, Fatal> &p_right) {
	return p_left < p_right.operator->();
}
template <typename T, typename U, bool Fatal>
[[nodiscard]] inline constexpr bool operator<=(const T &p_left, const NotNull<U, Fatal> &p_right) {
	return p_left <= p_right.operator->();
}
template <typename T, typename U, bool Fatal>
[[nodiscard]] inline constexpr bool operator>(const T &p_left, const NotNull<U, Fatal> &p_right) {
	return p_left > p_right.operator->();
}
template <typename T, typename U, bool Fatal>
[[nodiscard]] inline constexpr bool operator>=(const T &p_left, const NotNull<U, Fatal> &p_right) {
	return p_left >= p_right.operator->();
}
