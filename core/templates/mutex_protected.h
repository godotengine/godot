/**************************************************************************/
/*  mutex_protected.h                                                     */
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

#include "core/os/mutex.h"

#include <utility>

// Holds a value that can only be reached through lock(), so there is no way
// to touch it without holding the mutex. Single recursive lock, no read/write split.
template <typename T>
class MutexProtected {
	mutable Mutex mutex;
	T value;

public:
	class [[nodiscard]] Guard {
		friend class MutexProtected;
		MutexProtected &owner;

		explicit Guard(MutexProtected &p_owner) :
				owner(p_owner) {
			owner.mutex.lock();
		}

	public:
		Guard(const Guard &) = delete;
		Guard &operator=(const Guard &) = delete;
		Guard(Guard &&) = delete;
		Guard &operator=(Guard &&) = delete;
		~Guard() { owner.mutex.unlock(); }

		// SAFETY: don't let the returned pointer/reference outlive this Guard.
		T *operator->() const { return &owner.value; }
		T &operator*() const { return owner.value; }
	};

	class [[nodiscard]] ConstGuard {
		friend class MutexProtected;
		const MutexProtected &owner;

		explicit ConstGuard(const MutexProtected &p_owner) :
				owner(p_owner) {
			owner.mutex.lock();
		}

	public:
		ConstGuard(const ConstGuard &) = delete;
		ConstGuard &operator=(const ConstGuard &) = delete;
		ConstGuard(ConstGuard &&) = delete;
		ConstGuard &operator=(ConstGuard &&) = delete;
		~ConstGuard() { owner.mutex.unlock(); }

		// SAFETY: don't let the returned pointer/reference outlive this ConstGuard.
		const T *operator->() const { return &owner.value; }
		const T &operator*() const { return owner.value; }
	};

	Guard lock() & { return Guard(*this); }
	ConstGuard lock() const & { return ConstGuard(*this); }
	Guard lock() && = delete;
	ConstGuard lock() const && = delete;

	MutexProtected() {}
	explicit MutexProtected(T p_value) :
			value(std::move(p_value)) {}

	MutexProtected(const MutexProtected &) = delete;
	MutexProtected &operator=(const MutexProtected &) = delete;
	MutexProtected(MutexProtected &&) = delete;
	MutexProtected &operator=(MutexProtected &&) = delete;
};
