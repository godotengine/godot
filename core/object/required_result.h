/**************************************************************************/
/*  required_result.h                                                     */
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
#include "core/variant/variant.h"

// Using `RequiredResult<T>` as the return type indicates that null will only be returned in the case of an error.
// This allows GDExtension language bindings to use the appropriate error handling mechanism for that language
// when null is returned (for example, throwing an exception), rather than simply returning the value.
template <typename T>
class RequiredResult {
	static_assert(!is_fully_defined_v<T> || std::is_base_of_v<Object, T>, "T must be an Object subtype");

public:
	using element_type = T;
	using ptr_type = std::conditional_t<std::is_base_of_v<RefCounted, T>, Ref<T>, T *>;

private:
	ptr_type _value = ptr_type();

public:
	_FORCE_INLINE_ RequiredResult() = default;

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
			_value(Object::cast_to<T>(p_variant.get_validated_object())) {}
	template <typename U = T, std::enable_if_t<std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredResult &operator=(const Variant &p_variant) {
		_value = Object::cast_to<T>(p_variant.get_validated_object());
		return *this;
	}

	template <typename U = T, std::enable_if_t<!std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredResult(const Variant &p_variant) :
			_value(Object::cast_to<T>(p_variant.operator Object *())) {}
	template <typename U = T, std::enable_if_t<!std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredResult &operator=(const Variant &p_variant) {
		_value = Object::cast_to<T>(p_variant.operator Object *());
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

	_FORCE_INLINE_ operator ptr_type() const {
		return _value;
	}

	template <typename U = T, typename T_Other, std::enable_if_t<std::is_base_of_v<RefCounted, U> && std::is_base_of_v<U, T_Other>, int> = 0>
	_FORCE_INLINE_ operator Ref<T_Other>() const {
		return Ref<T_Other>(_value);
	}

	_FORCE_INLINE_ element_type *operator*() const {
		return ptr();
	}

	_FORCE_INLINE_ element_type *operator->() const {
		return ptr();
	}
};
