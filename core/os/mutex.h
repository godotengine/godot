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

#include "core/error/error_macros.h"
#include "core/typedefs.h"

#ifdef MINGW_ENABLED
#define MINGW_STDTHREAD_REDUNDANCY_WARNING
#include "thirdparty/mingw-std-threads/mingw.mutex.h"
#define THREADING_NAMESPACE mingw_stdthread
#else
#include <mutex>
#define THREADING_NAMESPACE std
#endif

template <class MutexT>
class MutexLock;

template <class StdMutexT>
class MutexImpl {
	friend class MutexLock<MutexImpl<StdMutexT>>;

	using StdMutexType = StdMutexT;

	mutable StdMutexT mutex;

public:
	_ALWAYS_INLINE_ void lock() const {
		mutex.lock();
	}

	_ALWAYS_INLINE_ void unlock() const {
		mutex.unlock();
	}

	_ALWAYS_INLINE_ bool try_lock() const {
		return mutex.try_lock();
	}
};

// A very special kind of mutex, used in scenarios where these
// requirements hold at the same time:
// - Must be used with a condition variable (only binary mutexes are suitable).
// - Must have recursive semnantics (or simulate, as this one does).
// The implementation keeps the lock count in TS. Therefore, only
// one object of each version of the template can exists; hence the Tag argument.
// Tags must be unique across the Godot codebase.
// Also, don't forget to declare the thread_local variable on each use.
template <int Tag>
class SafeBinaryMutex {
	friend class MutexLock<SafeBinaryMutex>;

	using StdMutexType = THREADING_NAMESPACE::mutex;

	mutable THREADING_NAMESPACE::mutex mutex;
	static thread_local uint32_t count;

public:
	_ALWAYS_INLINE_ void lock() const {
		if (++count == 1) {
			mutex.lock();
		}
	}

	_ALWAYS_INLINE_ void unlock() const {
		DEV_ASSERT(count);
		if (--count == 0) {
			mutex.unlock();
		}
	}

	_ALWAYS_INLINE_ bool try_lock() const {
		if (count) {
			count++;
			return true;
		} else {
			if (mutex.try_lock()) {
				count++;
				return true;
			} else {
				return false;
			}
		}
	}

	~SafeBinaryMutex() {
		DEV_ASSERT(!count);
	}
};

template <class MutexT>
class MutexLock {
	friend class ConditionVariable;

	THREADING_NAMESPACE::unique_lock<typename MutexT::StdMutexType> lock;

public:
	_ALWAYS_INLINE_ explicit MutexLock(const MutexT &p_mutex) :
			lock(p_mutex.mutex){};
};

// This specialization is needed so manual locking and MutexLock can be used
// at the same time on a SafeBinaryMutex.
template <int Tag>
class MutexLock<SafeBinaryMutex<Tag>> {
	friend class ConditionVariable;

	THREADING_NAMESPACE::unique_lock<THREADING_NAMESPACE::mutex> lock;

public:
	_ALWAYS_INLINE_ explicit MutexLock(const SafeBinaryMutex<Tag> &p_mutex) :
			lock(p_mutex.mutex) {
		SafeBinaryMutex<Tag>::count++;
	};
	_ALWAYS_INLINE_ ~MutexLock() {
		SafeBinaryMutex<Tag>::count--;
	};
};

using Mutex = MutexImpl<THREADING_NAMESPACE::recursive_mutex>; // Recursive, for general use
using BinaryMutex = MutexImpl<THREADING_NAMESPACE::mutex>; // Non-recursive, handle with care

extern template class MutexImpl<THREADING_NAMESPACE::recursive_mutex>;
extern template class MutexImpl<THREADING_NAMESPACE::mutex>;
extern template class MutexLock<MutexImpl<THREADING_NAMESPACE::recursive_mutex>>;
extern template class MutexLock<MutexImpl<THREADING_NAMESPACE::mutex>>;

#endif // MUTEX_H
