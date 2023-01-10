/**************************************************************************/
/*  mutex.h                                                               */
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

#ifndef MUTEX_H
#define MUTEX_H

#include "core/error_list.h"
#include "core/typedefs.h"

#if !defined(NO_THREADS)

#include <mutex>

template <class StdMutexT>
class MutexImpl {
	mutable StdMutexT mutex;
	friend class MutexLock;

public:
	_ALWAYS_INLINE_ void lock() const {
		mutex.lock();
	}

	_ALWAYS_INLINE_ void unlock() const {
		mutex.unlock();
	}

	_ALWAYS_INLINE_ Error try_lock() const {
		return mutex.try_lock() ? OK : ERR_BUSY;
	}
};

// This is written this way instead of being a template to overcome a limitation of C++ pre-17
// that would require MutexLock to be used like this: MutexLock<Mutex> lock;
class MutexLock {
	union {
		std::recursive_mutex *recursive_mutex;
		std::mutex *mutex;
	};
	bool recursive;

public:
	_ALWAYS_INLINE_ explicit MutexLock(const MutexImpl<std::recursive_mutex> &p_mutex) :
			recursive_mutex(&p_mutex.mutex),
			recursive(true) {
		recursive_mutex->lock();
	}
	_ALWAYS_INLINE_ explicit MutexLock(const MutexImpl<std::mutex> &p_mutex) :
			mutex(&p_mutex.mutex),
			recursive(false) {
		mutex->lock();
	}

	_ALWAYS_INLINE_ ~MutexLock() {
		if (recursive) {
			recursive_mutex->unlock();
		} else {
			mutex->unlock();
		}
	}
};

using Mutex = MutexImpl<std::recursive_mutex>; // Recursive, for general use
using BinaryMutex = MutexImpl<std::mutex>; // Non-recursive, handle with care

extern template class MutexImpl<std::recursive_mutex>;
extern template class MutexImpl<std::mutex>;

#else

class FakeMutex {
	FakeMutex() {}
};

template <class MutexT>
class MutexImpl {
public:
	_ALWAYS_INLINE_ void lock() const {}
	_ALWAYS_INLINE_ void unlock() const {}
	_ALWAYS_INLINE_ Error try_lock() const { return OK; }
};

class MutexLock {
public:
	explicit MutexLock(const MutexImpl<FakeMutex> &p_mutex) {}
};

using Mutex = MutexImpl<FakeMutex>;
using BinaryMutex = MutexImpl<FakeMutex>; // Non-recursive, handle with care

#endif // !NO_THREADS

#endif // MUTEX_H
