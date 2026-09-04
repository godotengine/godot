/**************************************************************************/
/*  value_ptr.h                                                           */
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

#include "core/os/memory.h"

#include <cstddef>
#include <utility>

/**
 * A smart pointer with unique ownership, similar to std::indirect.
 * As opposed to unique_ptr, it supports copy assignment.
 */
template <typename T>
class ValuePtr {
	T *_ptr = nullptr;

	explicit ValuePtr(T *p_data) :
			_ptr(p_data) {}

public:
	// Allows access to the pointee type statically.
	// Matches the same declaration from std smart pointers (std::shared_ptr etc.).
	using element_type = T;

	_FORCE_INLINE_ T &operator*() { return *_ptr; }
	_FORCE_INLINE_ const T &operator*() const { return *_ptr; }

	_FORCE_INLINE_ T *operator->() { return _ptr; }
	_FORCE_INLINE_ const T *operator->() const { return _ptr; }

	_FORCE_INLINE_ operator bool() const { return _ptr; }

	_FORCE_INLINE_ T *ptr() { return _ptr; }
	_FORCE_INLINE_ const T *ptr() const { return _ptr; }

	_FORCE_INLINE_ void reset() {
		if (_ptr) {
			memdelete(_ptr);
			_ptr = nullptr;
		}
	}

	_FORCE_INLINE_ ValuePtr &operator=(const ValuePtr &p_ptr) {
		if (this == &p_ptr) {
			return *this;
		}
		reset();
		if (p_ptr._ptr) {
			_ptr = memnew(T(*p_ptr._ptr));
		} else {
			_ptr = nullptr;
		}
		return *this;
	}
	_FORCE_INLINE_ ValuePtr &operator=(ValuePtr &&p_ptr) {
		if (_ptr == p_ptr._ptr) {
			return *this;
		}
		reset();
		_ptr = p_ptr._ptr;
		p_ptr._ptr = nullptr;
		return *this;
	}

	template <typename... Args>
	static ValuePtr make(Args &&...p_args) {
		return ValuePtr(memnew(T(std::forward<Args>(p_args)...)));
	}

	ValuePtr() = default;
	ValuePtr(std::nullptr_t) :
			_ptr(nullptr) {}
	ValuePtr(const ValuePtr &p_ptr) {
		if (!p_ptr._ptr) {
			return;
		}
		_ptr = memnew(T(*p_ptr._ptr));
	}
	ValuePtr(ValuePtr &&p_ptr) {
		_ptr = p_ptr._ptr;
		p_ptr._ptr = nullptr;
	}
	~ValuePtr() {
		reset();
	}
};

template <typename TL, typename TR>
bool operator==(const ValuePtr<TL> &p_lhs, const ValuePtr<TR> &p_rhs) {
	return p_lhs.ptr() == p_rhs.ptr() || (p_lhs.ptr() && p_rhs.ptr() && *p_lhs == *p_rhs);
}

template <typename TL, typename TR>
bool operator!=(const ValuePtr<TL> &p_lhs, const ValuePtr<TR> &p_rhs) {
	return !(p_lhs == p_rhs);
}

template <typename TL, typename TR>
bool operator==(const TL &p_lhs, const ValuePtr<TR> &p_rhs) {
	return p_rhs.ptr() && p_lhs == *p_rhs;
}

template <typename TL, typename TR>
bool operator!=(const TL &p_lhs, const ValuePtr<TR> &p_rhs) {
	return !(p_lhs == p_rhs);
}

template <typename TL, typename TR>
bool operator==(const ValuePtr<TL> &p_lhs, const TR &p_rhs) {
	return p_lhs.ptr() && *p_lhs == p_rhs;
}

template <typename TL, typename TR>
bool operator!=(const ValuePtr<TL> &p_lhs, const TR &p_rhs) {
	return !(p_lhs == p_rhs);
}

template <typename TL>
bool operator==(const ValuePtr<TL> &p_lhs, std::nullptr_t p_rhs) {
	return p_lhs.ptr() == nullptr;
}

template <typename TL>
bool operator!=(const ValuePtr<TL> &p_lhs, std::nullptr_t p_rhs) {
	return p_lhs.ptr() != nullptr;
}

template <typename TR>
bool operator==(std::nullptr_t p_lhs, const ValuePtr<TR> &p_rhs) {
	return nullptr == p_rhs.ptr();
}

template <typename TR>
bool operator!=(std::nullptr_t p_lhs, const ValuePtr<TR> &p_rhs) {
	return nullptr != p_rhs.ptr();
}
