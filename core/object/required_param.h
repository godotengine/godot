/**************************************************************************/
/*  required_param.h                                                      */
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

// Using `RequiredParam<T>` as an argument type indicates that passing null as that parameter is an error,
// that will prevent the method from doing its intended function.
// This allows GDExtension bindings to use language-specific mechanisms to prevent users from passing null,
// because it is never valid to do so.
template <typename T>
class RequiredParam {
	static_assert(!is_fully_defined_v<T> || std::is_base_of_v<Object, T>, "T must be an Object subtype");

public:
	static constexpr bool is_ref = std::is_base_of_v<RefCounted, T>;

	using element_type = T;
	using extracted_type = std::conditional_t<is_ref, const Ref<T> &, T *>;
	using persistent_type = std::conditional_t<is_ref, Ref<T>, T *>;

private:
	T *_value = nullptr;

	_FORCE_INLINE_ RequiredParam() = default;

public:
	// These functions should not be called directly, they are only for internal use.
	_FORCE_INLINE_ extracted_type _internal_ptr_dont_use() const {
		if constexpr (is_ref) {
			// Pretend _value is a Ref, for ease of use with existing `const Ref &` accepting APIs.
			// This only works as long as Ref is internally T *.
			// The double indirection should be optimized away by the compiler.
			static_assert(sizeof(Ref<T>) == sizeof(T *));
			return *((const Ref<T> *)&_value);
		} else {
			return _value;
		}
	}
	_FORCE_INLINE_ bool _is_null_dont_use() const { return _value == nullptr; }
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
			_value(*p_ref) {}
	template <typename T_Other, std::enable_if_t<std::is_base_of_v<T, T_Other>, int> = 0>
	_FORCE_INLINE_ RequiredParam &operator=(const Ref<T_Other> &p_ref) {
		_value = *p_ref;
		return *this;
	}

	template <typename U = T, std::enable_if_t<std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredParam(const Variant &p_variant) :
			_value(Object::cast_to<T>(p_variant.get_validated_object())) {}
	template <typename U = T, std::enable_if_t<std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredParam &operator=(const Variant &p_variant) {
		_value = Object::cast_to<T>(p_variant.get_validated_object());
		return *this;
	}

	template <typename U = T, std::enable_if_t<!std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredParam(const Variant &p_variant) :
			_value(Object::cast_to<T>(p_variant.operator Object *())) {}
	template <typename U = T, std::enable_if_t<!std::is_base_of_v<RefCounted, U>, int> = 0>
	_FORCE_INLINE_ RequiredParam &operator=(const Variant &p_variant) {
		_value = Object::cast_to<T>(p_variant.operator Object *());
		return *this;
	}
};
