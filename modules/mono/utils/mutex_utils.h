/*************************************************************************/
/*  mutex_utils.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef MUTEX_UTILS_H
#define MUTEX_UTILS_H

#include "core/error_macros.h"
#include "core/os/mutex.h"

#include "macros.h"

class ScopedMutexLock {
	Mutex *mutex;

public:
	ScopedMutexLock(Mutex *mutex) {
		this->mutex = mutex;
#ifndef NO_THREADS
#ifdef DEBUG_ENABLED
		CRASH_COND(!mutex);
#endif
		this->mutex->lock();
#endif
	}

	~ScopedMutexLock() {
#ifndef NO_THREADS
#ifdef DEBUG_ENABLED
		CRASH_COND(!mutex);
#endif
		mutex->unlock();
#endif
	}
};

#define SCOPED_MUTEX_LOCK(m_mutex) ScopedMutexLock GD_UNIQUE_NAME(__scoped_mutex_lock__)(m_mutex);

// TODO: Add version that receives a lambda instead, once C++11 is allowed

#endif // MUTEX_UTILS_H
