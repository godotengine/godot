/**************************************************************************/
/*  semaphore.h                                                           */
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

#ifndef SEMAPHORE_H
#define SEMAPHORE_H

#include "core/error/error_list.h"
#include "core/typedefs.h"

#include <condition_variable>
#include <mutex>

class Semaphore {
private:
	mutable std::mutex mutex;
	mutable std::condition_variable condition;
	mutable uint32_t count = 0; // Initialized as locked.

public:
	_ALWAYS_INLINE_ void post() const {
		std::lock_guard lock(mutex);
		count++;
		condition.notify_one();
	}

	_ALWAYS_INLINE_ void wait() const {
		std::unique_lock lock(mutex);
		while (!count) { // Handle spurious wake-ups.
			condition.wait(lock);
		}
		count--;
	}

	_ALWAYS_INLINE_ bool try_wait() const {
		std::lock_guard lock(mutex);
		if (count) {
			count--;
			return true;
		} else {
			return false;
		}
	}
};

#endif // SEMAPHORE_H
