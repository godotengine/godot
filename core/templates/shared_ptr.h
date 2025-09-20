/**************************************************************************/
/*  shared_ptr.h                                                          */
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
#include "core/templates/safe_refcount.h"

#include <memory>

/**
 * A smart pointer with shared ownership, similar to std::shared_ptr.
 * Uses SafeRefCount to handle reference counters.
 * When the last reference to the pointee destructs, the pointee is destructed.
 */
template <typename T>
class SharedPtr {
	struct Data {
		SafeRefCount refcount;
		T t;
	};

	Data *_ptr = nullptr;

	explicit SharedPtr(Data *p_data) :
			_ptr(p_data) {}

	void _ref(const SharedPtr &p_ptr) {
		if (_ptr == p_ptr._ptr) {
			return;
		}
		if (p_ptr._ptr && !p_ptr._ptr->refcount.ref()) {
			return; // Failed to reference.
		}

		_unref();
		_ptr = p_ptr._ptr;
	}

	void _unref() {
		if (_ptr && _ptr->refcount.unref()) {
			_ptr->~Data();
			std::allocator<Data> allocator;
			allocator.deallocate(_ptr, 1);
		}
		_ptr = nullptr;
	}

public:
	// Allows access to the pointee type statically.
	// Matches the same declaration from std smart pointers (std::shared_ptr etc.).
	using element_type = T;

	_FORCE_INLINE_ SafeRefCount &refcount() { return _ptr->refcount; }
	_FORCE_INLINE_ const SafeRefCount &refcount() const { return _ptr->refcount; }

	_FORCE_INLINE_ void *ptr() { return _ptr; }
	_FORCE_INLINE_ const void *ptr() const { return _ptr; }

	_FORCE_INLINE_ operator T *() { return _ptr ? &_ptr->t : nullptr; }
	_FORCE_INLINE_ operator const T *() const { return _ptr ? &_ptr->t : nullptr; }

	_FORCE_INLINE_ T &operator*() { return _ptr->t; }
	_FORCE_INLINE_ const T &operator*() const { return _ptr->t; }

	_FORCE_INLINE_ T *operator->() { return _ptr ? &_ptr->t : nullptr; }
	_FORCE_INLINE_ const T *operator->() const { return _ptr ? &_ptr->t : nullptr; }

	_FORCE_INLINE_ operator bool() const { return _ptr; }

	_FORCE_INLINE_ SharedPtr &operator=(const SharedPtr &p_ptr) {
		_ref(p_ptr);
		return *this;
	}
	_FORCE_INLINE_ SharedPtr &operator=(SharedPtr &&p_ptr) {
		if (_ptr == p_ptr._ptr) {
			return *this;
		}
		_unref();
		_ptr = p_ptr._ptr;
		p_ptr._ptr = nullptr;
		return *this;
	}

	_FORCE_INLINE_ void reset() { _unref(); }

	template <typename... Args>
	static SharedPtr make(Args... args) {
		std::allocator<Data> allocator;
		Data *data = allocator.allocate(1);
		memnew_placement(&data->refcount, SafeRefCount);
		data->refcount.init();
		memnew_placement(&data->t, T(std::forward<Args>(args)...));
		return SharedPtr(data);
	}

	SharedPtr() = default;
	SharedPtr(std::nullptr_t) :
			_ptr(nullptr) {}
	SharedPtr(const SharedPtr &p_ptr) {
		if (!p_ptr._ptr || !p_ptr._ptr->refcount.ref()) {
			return;
		}
		_ptr = p_ptr._ptr;
	}
	SharedPtr(SharedPtr &&p_ptr) {
		_ptr = p_ptr._ptr;
		p_ptr._ptr = nullptr;
	}
	~SharedPtr() { _unref(); }
};

template <typename TL, typename TR>
bool operator==(SharedPtr<TL> p_lhs, SharedPtr<TR> p_rhs) {
	return p_lhs.ptr() == p_rhs.ptr();
}

template <typename TL, typename TR>
bool operator==(SharedPtr<TL> p_lhs, const TR *p_rhs) {
	return p_lhs.ptr() == p_rhs;
}

template <typename TL, typename TR>
bool operator==(const TL *p_lhs, const SharedPtr<TR> p_rhs) {
	return p_lhs == p_rhs.ptr();
}

template <typename TL>
bool operator==(SharedPtr<TL> p_lhs, std::nullptr_t p_rhs) {
	return p_lhs.ptr() == p_rhs;
}

template <typename TR>
bool operator==(std::nullptr_t p_lhs, SharedPtr<TR> p_rhs) {
	return p_lhs == p_rhs.ptr();
}
