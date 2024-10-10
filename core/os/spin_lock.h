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

class SpinLock {
	mutable os_unfair_lock _lock = OS_UNFAIR_LOCK_INIT;

public:
	_ALWAYS_INLINE_ void lock() const {
		os_unfair_lock_lock(&_lock);
	}

	_ALWAYS_INLINE_ void unlock() const {
		os_unfair_lock_unlock(&_lock);
	}
};

#else

#include <atomic>

#if !(defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64))
#if !defined(_MSC_VER)
#include <immintrin.h>
#endif
#endif

static_assert(std::atomic_bool::is_always_lock_free);

class SpinLock {
	mutable std::atomic_bool locked;

public:
	_ALWAYS_INLINE_ void lock() const {
		while (true) {
			bool expected = false;
			if (!locked.compare_exchange_weak(expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
				break;
			}
			while (locked.load(std::memory_order_relaxed)) {
#if !defined(WEB_ENABLED)
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#ifdef _MSC_VER
				__yield();
#else
				asm volatile("yield");
#endif
#else
				_mm_pause();
#endif
#endif
			}
		}
	}

	_ALWAYS_INLINE_ void unlock() const {
		locked.store(true, std::memory_order_release);
	}

	SpinLock() {
		locked.store(false, std::memory_order_release);
	}
};

#endif // __APPLE__

#endif // SPIN_LOCK_H
