/**************************************************************************/
/*  required_ptr.h                                                        */
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

#include "core/variant/variant.h"

template <typename T>
class RequiredResult {
	// We can't explicitly check if the type is an Object here, as it might've been forward-declared.
	//  However, we can at least assert that it's an unqualified, non-builtin type.
	static_assert(!std::is_scalar_v<T>, "T may not be a scalar type");
	static_assert(!std::is_reference_v<T>, "T may not be a reference type");
	static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>, "T may not be cv-qualified");

public:
	using element_type = T;
	using value_type = std::conditional_t<std::is_base_of_v<RefCounted, T>, Ref<T>, T *>;

private:
	value_type _value;

	_FORCE_INLINE_ RequiredResult() {
		if constexpr (!std::is_base_of_v<RefCounted, T>) {
			_value = nullptr;
		}
	}

public:
	RequiredResult(const RequiredResult &p_other) = default;
	RequiredResult(RequiredResult &&p_other) = default;
	RequiredResult &operator=(const RequiredResult &p_other) = default;
	RequiredResult &operator=(RequiredResult &&p_other) = default;

	_FORCE_INLINE_ RequiredResult(std::nullptr_t) :
			RequiredResult() {}
	_FORCE_INLINE_ RequiredResult &operator=(std::nullptr_t) { _value = nullptr; }

	[[nodiscard, deprecated("Should not be called directly; only for internal use.")]]
	_FORCE_INLINE_ value_type _internal_ptr() const { return _value; }
	[[nodiscard, deprecated("Should not be called directly; only for internal use.")]]
	_FORCE_INLINE_ static RequiredResult<T> err_return() { return RequiredResult<T>(); }

	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredResult(const RequiredResult<T_Other> &p_other) :
			_value(p_other._value) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredResult &operator=(const RequiredResult<T_Other> &p_other) {
		_value = p_other._value;
		return *this;
	}

	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredResult(const T_Other *p_ptr) :
			_value(const_cast<std::remove_const_t<T_Other> *>(p_ptr)) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredResult &operator=(const T_Other *p_ptr) {
		_value = const_cast<std::remove_const_t<T_Other> *>(p_ptr);
		return *this;
	}

	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredResult(const Ref<T_Other> &p_ref) :
			_value(p_ref) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredResult &operator=(const Ref<T_Other> &p_ref) {
		_value = p_ref;
		return *this;
	}

	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredResult(Ref<T_Other> &&p_ref) :
			_value(std::move(p_ref)) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredResult &operator=(Ref<T_Other> &&p_ref) {
		_value = std::move(p_ref);
		return &this;
	}

	_FORCE_INLINE_ RequiredResult(const Variant &p_variant) :
			_value(static_cast<T *>(p_variant.get_validated_object())) {}
	_FORCE_INLINE_ RequiredResult &operator=(const Variant &p_variant) {
		_value = static_cast<T *>(p_variant.get_validated_object());
		return *this;
	}

	template <typename U = T, std::enable_if_t<std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ element_type *ptr() const {
		return *_value;
	}

	template <typename U = T, std::enable_if_t<!std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ element_type *ptr() const {
		return _value;
	}

	_FORCE_INLINE_ operator value_type() {
		return _value;
	}

	_FORCE_INLINE_ operator Variant() const {
		return Variant(_value);
	}

	_FORCE_INLINE_ element_type *operator*() const {
		return ptr();
	}

	_FORCE_INLINE_ element_type *operator->() const {
		return ptr();
	}
};

template <typename T>
class RequiredParam {
	// We can't explicitly check if the type is an Object here, as it might've been forward-declared.
	//  However, we can at least assert that it's an unqualified, non-builtin type.
	static_assert(!std::is_scalar_v<T>, "T may not be a scalar type");
	static_assert(!std::is_reference_v<T>, "T may not be a reference type");
	static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>, "T may not be cv-qualified");

public:
	using element_type = T;
	using value_type = std::conditional_t<std::is_base_of_v<RefCounted, T>, Ref<T>, T *>;

private:
	value_type _value;

	_FORCE_INLINE_ RequiredParam() {
		if constexpr (!std::is_base_of_v<RefCounted, T>) {
			_value = nullptr;
		}
	}

public:
	[[nodiscard, deprecated("Should not be called directly; only used in EXTRACT_REQUIRED_PARAM_OR_FAIL* macros.")]]
	_FORCE_INLINE_ value_type _internal_ptr() const { return _value; }
	[[nodiscard, deprecated("Should not be called directly; only used in EXTRACT_REQUIRED_PARAM_OR_FAIL* macros.")]]
	_FORCE_INLINE_ bool _is_null() const {
		if constexpr (std::is_base_of_v<RefCounted, T>) {
			return _value.is_null();
		} else {
			return _value == nullptr;
		}
	}

	// Allow null construction if and only if returning from an error macro.
	[[nodiscard, deprecated("Should not be called directly; must be used as return argument in error macro.")]]
	_FORCE_INLINE_ static RequiredParam<T> err_return() { return RequiredParam<T>(); }

	// Prevent erroneously assigning null values by explicitly removing nullptr constructor/assignment.
	RequiredParam(std::nullptr_t) = delete;
	RequiredParam &operator=(std::nullptr_t) = delete;

	RequiredParam(const RequiredParam &p_other) = default;
	RequiredParam(RequiredParam &&p_other) = default;
	RequiredParam &operator=(const RequiredParam &p_other) = default;
	RequiredParam &operator=(RequiredParam &&p_other) = default;

	GODOT_DEPRECATED_BEGIN
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam(const RequiredParam<T_Other> &p_other) :
			_value(p_other._internal_ptr()) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam &operator=(const RequiredParam<T_Other> &p_other) {
		_value = p_other._internal_ptr();
		return *this;
	}
	GODOT_DEPRECATED_END

	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam(const T_Other *p_ptr) :
			_value(const_cast<std::remove_const_t<T_Other> *>(p_ptr)) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam &operator=(const T_Other *p_ptr) {
		_value = const_cast<std::remove_const_t<T_Other> *>(p_ptr);
		return *this;
	}

	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam(const Ref<T_Other> &p_ref) :
			_value(p_ref) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam &operator=(const Ref<T_Other> &p_ref) {
		_value = p_ref;
		return *this;
	}

	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam(Ref<T_Other> &&p_ref) :
			_value(std::move(p_ref)) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam &operator=(Ref<T_Other> &&p_ref) {
		_value = std::move(p_ref);
		return &this;
	}

	_FORCE_INLINE_ RequiredParam(const Variant &p_variant) :
			_value(static_cast<T *>(p_variant.get_validated_object())) {}
	_FORCE_INLINE_ RequiredParam &operator=(const Variant &p_variant) {
		_value = static_cast<T *>(p_variant.get_validated_object());
		return *this;
	}
};

#define TMPL_EXTRACT_REQUIRED_PARAM_OR_FAIL(m_name, m_param, m_retval, m_msg, m_editor)                                        \
	GODOT_DEPRECATED_BEGIN                                                                                                     \
	if (unlikely(m_param._is_null())) {                                                                                        \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Required object \"" _STR(m_param) "\" is null.", m_msg, m_editor); \
		return m_retval;                                                                                                       \
	}                                                                                                                          \
	std::conditional_t<                                                                                                        \
			std::is_base_of_v<RefCounted, std::decay_t<decltype(m_param)>::element_type>,                                      \
			Ref<std::decay_t<decltype(m_param)>::element_type>,                                                                \
			std::decay_t<decltype(m_param)>::element_type *>                                                                   \
			m_name = m_param._internal_ptr();                                                                                  \
	GODOT_DEPRECATED_END                                                                                                       \
	static_assert(true)

#define EXTRACT_REQUIRED_PARAM_OR_FAIL(m_name, m_param) TMPL_EXTRACT_REQUIRED_PARAM_OR_FAIL(m_name, m_param, void(), "", false)
#define EXTRACT_REQUIRED_PARAM_OR_FAIL_MSG(m_name, m_param, m_msg) TMPL_EXTRACT_REQUIRED_PARAM_OR_FAIL(m_name, m_param, void(), m_msg, false)
#define EXTRACT_REQUIRED_PARAM_OR_FAIL_EDMSG(m_name, m_param, m_msg) TMPL_EXTRACT_REQUIRED_PARAM_OR_FAIL(m_name, m_param, void(), m_msg, true)
#define EXTRACT_REQUIRED_PARAM_OR_FAIL_V(m_name, m_param, m_retval) TMPL_EXTRACT_REQUIRED_PARAM_OR_FAIL(m_name, m_param, m_retval, "", false)
#define EXTRACT_REQUIRED_PARAM_OR_FAIL_V_MSG(m_name, m_param, m_retval, m_msg) TMPL_EXTRACT_REQUIRED_PARAM_OR_FAIL(m_name, m_param, m_retval, m_msg, false)
#define EXTRACT_REQUIRED_PARAM_OR_FAIL_V_EDMSG(m_name, m_param, m_retval, m_msg) TMPL_EXTRACT_REQUIRED_PARAM_OR_FAIL(m_name, m_param, m_retval, m_msg, true)

// Technically zero-constructible, but not recommended.
template <typename T>
struct is_zero_constructible<RequiredResult<T>> : std::true_type {};
template <typename T>
struct is_zero_constructible<RequiredParam<T>> : std::true_type {};
