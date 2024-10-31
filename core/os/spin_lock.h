/**************************************************************************/
/*  spin_lock.h                                                           */
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

#ifndef SPIN_LOCK_H
#define SPIN_LOCK_H

#include "core/typedefs.h"

#if defined(__APPLE__)

#include <os/lock.h>

#if defined(__aarch64__) && defined(MACOS_ENABLED)
static constexpr uint32_t OS_UNFAIR_LOCK_DATA_SYNCHRONIZATION = 0x00010000;
static constexpr uint32_t OS_UNFAIR_LOCK_ADAPTIVE_SPIN = 0x00040000;

extern "C" {
typedef uint32_t os_unfair_lock_options_t;
OS_UNFAIR_LOCK_AVAILABILITY OS_EXPORT OS_NOTHROW OS_NONNULL_ALL void os_unfair_lock_lock_with_options(os_unfair_lock_t lock, os_unfair_lock_options_t options);
}
#endif

class SpinLock {
	mutable os_unfair_lock _lock = OS_UNFAIR_LOCK_INIT;

public:
	_ALWAYS_INLINE_ void lock() const {
#if defined(__aarch64__) && defined(MACOS_ENABLED)
		os_unfair_lock_lock_with_options(&_lock, OS_UNFAIR_LOCK_DATA_SYNCHRONIZATION | OS_UNFAIR_LOCK_ADAPTIVE_SPIN);
#else
		os_unfair_lock_lock(&_lock);
#endif
	}

	_ALWAYS_INLINE_ void unlock() const {
		os_unfair_lock_unlock(&_lock);
	}
};

#else

#include <atomic>

class SpinLock {
	mutable std::atomic_flag locked = ATOMIC_FLAG_INIT;

public:
	_ALWAYS_INLINE_ void lock() const {
		while (locked.test_and_set(std::memory_order_acquire)) {
			// Continue.
		}
	}
	_ALWAYS_INLINE_ void unlock() const {
		locked.clear(std::memory_order_release);
	}
};

#endif // __APPLE__

#endif // SPIN_LOCK_H
