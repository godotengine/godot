/**************************************************************************/
/*  either.h                                                              */
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
#include "core/os/memory.h"
#include "core/typedefs.h"

template <typename T1, typename T2>
class Either {
private:
	typename std::aligned_storage<
			(sizeof(T1) > sizeof(T2) ? sizeof(T1) : sizeof(T2)),
			(alignof(T1) > alignof(T2) ? alignof(T1) : alignof(T2))>::type storage;
	bool is_first;

	void destroy() {
		if (is_first) {
			reinterpret_cast<T1 *>(&storage)->~T1();
		} else {
			reinterpret_cast<T2 *>(&storage)->~T2();
		}
	}

public:
	Either(const T1 &value) :
			is_first(true) { memnew_placement(&storage, T1(value)); }
	Either(const T2 &value) :
			is_first(false) { memnew_placement(&storage, T2(value)); }

	Either(T1 &&value) :
			is_first(true) { memnew_placement(&storage, T1(std::move(value))); }
	Either(T2 &&value) :
			is_first(false) { memnew_placement(&storage, T2(std::move(value))); }

	~Either() { destroy(); }

	Either(const Either &other) :
			is_first(other.is_first) {
		if (is_first) {
			memnew_placement(&storage, T1(*reinterpret_cast<const T1 *>(&other.storage)));
		} else {
			memnew_placement(&storage, T2(*reinterpret_cast<const T2 *>(&other.storage)));
		}
	}

	Either &operator=(const Either &other) {
		if (this != &other) {
			destroy();
			is_first = other.is_first;
			if (is_first) {
				memnew_placement(&storage, T1(*reinterpret_cast<const T1 *>(&other.storage)));
			} else {
				memnew_placement(&storage, T2(*reinterpret_cast<const T2 *>(&other.storage)));
			}
		}
		return *this;
	}

	Either(Either &&other) :
			is_first(other.is_first) {
		if (is_first) {
			memnew_placement(&storage, T1(std::move(*reinterpret_cast<T1 *>(&other.storage))));
		} else {
			memnew_placement(&storage, T2(std::move(*reinterpret_cast<T2 *>(&other.storage))));
		}
	}

	Either &operator=(Either &&other) {
		if (this != &other) {
			destroy();
			is_first = other.is_first;
			if (is_first) {
				memnew_placement(&storage, T1(std::move(*reinterpret_cast<T1 *>(&other.storage))));
			} else {
				memnew_placement(&storage, T2(std::move(*reinterpret_cast<T2 *>(&other.storage))));
			}
		}
		return *this;
	}

	template <typename T>
	bool is() const {
		if constexpr (std::is_same_v<T, T1>) {
			return is_first;
		} else if constexpr (std::is_same_v<T, T2>) {
			return !is_first;
		}
	}

	template <typename T>
	T &get() {
		CRASH_COND(!is<T>());
		return *reinterpret_cast<T *>(&storage);
	}

	template <typename T>
	const T &get() const {
		CRASH_COND(!is<T>());
		return *reinterpret_cast<const T *>(&storage);
	}

	template <typename T>
	T &get_unchecked() {
		return *reinterpret_cast<T *>(&storage);
	}

	template <typename T>
	const T &get_unchecked() const {
		return *reinterpret_cast<const T *>(&storage);
	}
};

// Either is zero-constructible if and only if both constrained types are zero-constructible.
template <typename T1, typename T2>
struct is_zero_constructible<Either<T1, T2>> : std::conjunction<is_zero_constructible<T1>, is_zero_constructible<T2>> {};
