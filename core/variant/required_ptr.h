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
	static_assert(!is_fully_defined_v<T> || std::is_base_of_v<Object, T>, "T must be an Object subtype");

public:
	using element_type = T;
	using ptr_type = std::conditional_t<std::is_base_of_v<RefCounted, T>, Ref<T>, T *>;

private:
	ptr_type _value = ptr_type();

	_FORCE_INLINE_ RequiredResult() = default;

public:
	RequiredResult(const RequiredResult &p_other) = default;
	RequiredResult(RequiredResult &&p_other) = default;
	RequiredResult &operator=(const RequiredResult &p_other) = default;
	RequiredResult &operator=(RequiredResult &&p_other) = default;

	_FORCE_INLINE_ RequiredResult(std::nullptr_t) :
			RequiredResult() {}
	_FORCE_INLINE_ RequiredResult &operator=(std::nullptr_t) { _value = nullptr; }

	// These functions should not be called directly, they are only for internal use.
	_FORCE_INLINE_ ptr_type _internal_ptr_dont_use() const { return _value; }
	_FORCE_INLINE_ static RequiredResult<T> _err_return_dont_use() { return RequiredResult<T>(); }

	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredResult(const RequiredResult<T_Other> &p_other) :
			_value(p_other._value) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredResult &operator=(const RequiredResult<T_Other> &p_other) {
		_value = p_other._value;
		return *this;
	}

	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredResult(T_Other *p_ptr) :
			_value(p_ptr) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredResult &operator=(T_Other *p_ptr) {
		_value = p_ptr;
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

	template <typename U = T, std::enable_if_t<std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredResult(const Variant &p_variant) :
			_value(static_cast<T *>(p_variant.get_validated_object())) {}
	template <typename U = T, std::enable_if_t<std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredResult &operator=(const Variant &p_variant) {
		_value = static_cast<T *>(p_variant.get_validated_object());
		return *this;
	}

	template <typename U = T, std::enable_if_t<!std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredResult(const Variant &p_variant) :
			_value(static_cast<T *>(p_variant.operator Object *())) {}
	template <typename U = T, std::enable_if_t<!std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredResult &operator=(const Variant &p_variant) {
		_value = static_cast<T *>(p_variant.operator Object *());
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

	_FORCE_INLINE_ operator ptr_type() {
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
	static_assert(!is_fully_defined_v<T> || std::is_base_of_v<Object, T>, "T must be an Object subtype");

public:
	using element_type = T;
	using ptr_type = std::conditional_t<std::is_base_of_v<RefCounted, T>, Ref<T>, T *>;

private:
	ptr_type _value = ptr_type();

	_FORCE_INLINE_ RequiredParam() = default;

public:
	// These functions should not be called directly, they are only for internal use.
	_FORCE_INLINE_ ptr_type _internal_ptr_dont_use() const { return _value; }
	_FORCE_INLINE_ bool _is_null_dont_use() const {
		if constexpr (std::is_base_of_v<RefCounted, T>) {
			return _value.is_null();
		} else {
			return _value == nullptr;
		}
	}
	_FORCE_INLINE_ static RequiredParam<T> _err_return_dont_use() { return RequiredParam<T>(); }

	// Prevent erroneously assigning null values by explicitly removing nullptr constructor/assignment.
	RequiredParam(std::nullptr_t) = delete;
	RequiredParam &operator=(std::nullptr_t) = delete;

	RequiredParam(const RequiredParam &p_other) = default;
	RequiredParam(RequiredParam &&p_other) = default;
	RequiredParam &operator=(const RequiredParam &p_other) = default;
	RequiredParam &operator=(RequiredParam &&p_other) = default;

	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam(const RequiredParam<T_Other> &p_other) :
			_value(p_other._internal_ptr_dont_use()) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam &operator=(const RequiredParam<T_Other> &p_other) {
		_value = p_other._internal_ptr_dont_use();
		return *this;
	}

	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam(T_Other *p_ptr) :
			_value(p_ptr) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam &operator=(T_Other *p_ptr) {
		_value = p_ptr;
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

	template <typename U = T, std::enable_if_t<std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredParam(const Variant &p_variant) :
			_value(static_cast<T *>(p_variant.get_validated_object())) {}
	template <typename U = T, std::enable_if_t<std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredParam &operator=(const Variant &p_variant) {
		_value = static_cast<T *>(p_variant.get_validated_object());
		return *this;
	}

	template <typename U = T, std::enable_if_t<!std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredParam(const Variant &p_variant) :
			_value(static_cast<T *>(p_variant.operator Object *())) {}
	template <typename U = T, std::enable_if_t<!std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredParam &operator=(const Variant &p_variant) {
		_value = static_cast<T *>(p_variant.operator Object *());
		return *this;
	}
};

#define TMPL_EXTRACT_PARAM_OR_FAIL(m_name, m_param, m_retval, m_msg, m_editor)                                                 \
	if (unlikely(m_param._is_null_dont_use())) {                                                                               \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Required object \"" _STR(m_param) "\" is null.", m_msg, m_editor); \
		return m_retval;                                                                                                       \
	}                                                                                                                          \
	typename std::decay_t<decltype(m_param)>::ptr_type m_name = m_param._internal_ptr_dont_use();                              \
	static_assert(true)

// These macros are equivalent to the ERR_FAIL_NULL*() family of macros, only for RequiredParam<T> instead of raw pointers.
#define EXTRACT_PARAM_OR_FAIL(m_name, m_param) TMPL_EXTRACT_PARAM_OR_FAIL(m_name, m_param, void(), "", false)
#define EXTRACT_PARAM_OR_FAIL_MSG(m_name, m_param, m_msg) TMPL_EXTRACT_PARAM_OR_FAIL(m_name, m_param, void(), m_msg, false)
#define EXTRACT_PARAM_OR_FAIL_EDMSG(m_name, m_param, m_msg) TMPL_EXTRACT_PARAM_OR_FAIL(m_name, m_param, void(), m_msg, true)
#define EXTRACT_PARAM_OR_FAIL_V(m_name, m_param, m_retval) TMPL_EXTRACT_PARAM_OR_FAIL(m_name, m_param, m_retval, "", false)
#define EXTRACT_PARAM_OR_FAIL_V_MSG(m_name, m_param, m_retval, m_msg) TMPL_EXTRACT_PARAM_OR_FAIL(m_name, m_param, m_retval, m_msg, false)
#define EXTRACT_PARAM_OR_FAIL_V_EDMSG(m_name, m_param, m_retval, m_msg) TMPL_EXTRACT_PARAM_OR_FAIL(m_name, m_param, m_retval, m_msg, true)
