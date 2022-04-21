/*************************************************************************/
/*  optional.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef OPTIONAL_H
#define OPTIONAL_H

/**
 * @class Optional
 * Optional value container. Represents optional objects which may or may not contain a valid value.
 */

#include <initializer_list>

struct NullOpt {
	struct init {};
	constexpr NullOpt(init){};
};
constexpr NullOpt nullopt{ NullOpt::init{} };

constexpr struct TrivialInit {
} trivial_init{};

template <typename T>
union OptionalStorage {
	unsigned char dummy;
	T value;

	constexpr OptionalStorage(TrivialInit) :
			dummy(){};

	template <class... Args>
	_FORCE_INLINE_ OptionalStorage(const Args &&...p_args) :
			value(T(p_args...)) {}

	_FORCE_INLINE_ OptionalStorage() :
			value() {}

	_FORCE_INLINE_ ~OptionalStorage() {}
};

template <typename T>
class Optional {
public:
	typedef T ValueType;

	_FORCE_INLINE_ Optional() :
			_data(trivial_init), _ptr(nullptr) {}
	_FORCE_INLINE_ Optional(NullOpt) :
			_data(trivial_init), _ptr(nullptr) {}
	_FORCE_INLINE_ Optional(const Optional &other) :
			_data(trivial_init), _ptr(other ? new (&_data) ValueType(*other) : nullptr) {}
	_FORCE_INLINE_ explicit Optional(const ValueType &value) :
			_data(value), _ptr(&_data) {}

	_FORCE_INLINE_ ~Optional() {
		if (_ptr) {
			_ptr->ValueType::~ValueType();
		}
	}

	_FORCE_INLINE_ ValueType const &operator*() const {
		return *_ptr;
	}

	_FORCE_INLINE_ ValueType &operator*() {
		return *_ptr;
	}

	_FORCE_INLINE_ ValueType const *operator->() const {
		return _ptr;
	}

	_FORCE_INLINE_ ValueType *operator->() {
		return _ptr;
	}

	_FORCE_INLINE_ Optional &operator=(NullOpt) {
		clear();
		return *this;
	}

	_FORCE_INLINE_ Optional &operator=(Optional const &other) {
		if (_ptr && !other._ptr) {
			clear();
		} else if (!_ptr && other._ptr) {
			_ptr = new (&_data) ValueType(*other);
		} else if (_ptr && other._ptr) {
			*_ptr = *other;
		}
		return *this;
	}

	_FORCE_INLINE_ Optional &operator=(ValueType const &value) {
		if (_ptr) {
			*_ptr = value;
		} else {
			_ptr = new (&_data) ValueType(value);
		}
		return *this;
	}

	_FORCE_INLINE_ operator bool() const { return _ptr; }

private:
	OptionalStorage<ValueType> _data;
	ValueType *_ptr;

	_FORCE_INLINE_ void clear() {
		if (_ptr) {
			_ptr->ValueType::~ValueType();
			_ptr = nullptr;
		}
	}
};

#endif // OPTIONAL_H
