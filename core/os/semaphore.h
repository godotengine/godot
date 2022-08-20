/*************************************************************************/
/*  semaphore.h                                                          */
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

#ifndef SEMAPHORE_H
#define SEMAPHORE_H

#include "core/error/error_list.h"
#include "core/typedefs.h"

#if !defined(NO_THREADS)

#include <condition_variable>
#include <mutex>

class Semaphore {
private:
	mutable std::mutex mutex_;
	mutable std::condition_variable condition_;
	mutable unsigned long count_ = 0; // Initialized as locked.

public:
	_ALWAYS_INLINE_ void post() const {
		std::lock_guard<decltype(mutex_)> lock(mutex_);
		++count_;
		condition_.notify_one();
	}

	_ALWAYS_INLINE_ void wait() const {
		std::unique_lock<decltype(mutex_)> lock(mutex_);
		while (!count_) { // Handle spurious wake-ups.
			condition_.wait(lock);
		}
		--count_;
	}

	_ALWAYS_INLINE_ bool try_wait() const {
		std::lock_guard<decltype(mutex_)> lock(mutex_);
		if (count_) {
			--count_;
			return true;
		}
		return false;
	}
};

#else

class Semaphore {
public:
	_ALWAYS_INLINE_ void post() const {}
	_ALWAYS_INLINE_ void wait() const {}
	_ALWAYS_INLINE_ bool try_wait() const { return true; }
};

#endif

#endif // SEMAPHORE_H
