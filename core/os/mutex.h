/*************************************************************************/
/*  mutex.h                                                              */
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
#ifndef MUTEX_H
#define MUTEX_H

#include "error_list.h"

/**
 * @class Mutex
 * @author Juan Linietsky
 * Portable Mutex (thread-safe locking) implementation.
 * Mutexes are always recursive ( they don't self-lock in a single thread ).
 * Mutexes can be used with a Lockp object like this, to avoid having to worry about unlocking:
 * Lockp( mutex );
 */

class Mutex {
protected:
	static Mutex *(*create_func)(bool);

public:
	virtual void lock() = 0; ///< Lock the mutex, block if locked by someone else
	virtual void unlock() = 0; ///< Unlock the mutex, let other threads continue
	virtual Error try_lock() = 0; ///< Attempt to lock the mutex, OK on success, ERROR means it can't lock.

	static Mutex *create(bool p_recursive = true); ///< Create a mutex

	virtual ~Mutex();
};

class MutexLock {

	Mutex *mutex;

public:
	MutexLock(Mutex *p_mutex) {
		mutex = p_mutex;
		if (mutex) mutex->lock();
	}
	~MutexLock() {
		if (mutex) mutex->unlock();
	}
};

#endif
