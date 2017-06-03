/*************************************************************************/
/*  thread_safe.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef THREAD_SAFE_H
#define THREAD_SAFE_H

#include "os/mutex.h"

class ThreadSafe {

	Mutex *mutex;

public:
	inline void lock() const {
		if (mutex) mutex->lock();
	}
	inline void unlock() const {
		if (mutex) mutex->unlock();
	}

	ThreadSafe();
	~ThreadSafe();
};

class ThreadSafeMethod {

	const ThreadSafe *_ts;

public:
	ThreadSafeMethod(const ThreadSafe *p_ts) {

		_ts = p_ts;
		_ts->lock();
	}

	~ThreadSafeMethod() { _ts->unlock(); }
};

#ifndef NO_THREADS

#define _THREAD_SAFE_CLASS_ ThreadSafe __thread__safe__;
#define _THREAD_SAFE_METHOD_ ThreadSafeMethod __thread_safe_method__(&__thread__safe__);
#define _THREAD_SAFE_LOCK_ __thread__safe__.lock();
#define _THREAD_SAFE_UNLOCK_ __thread__safe__.unlock();

#else

#define _THREAD_SAFE_CLASS_
#define _THREAD_SAFE_METHOD_
#define _THREAD_SAFE_LOCK_
#define _THREAD_SAFE_UNLOCK_

#endif

#endif
