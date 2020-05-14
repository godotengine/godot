/*************************************************************************/
/*  rw_lock.h                                                            */
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

#ifndef RW_LOCK_H
#define RW_LOCK_H

#include "core/error_list.h"

class RWLock {
protected:
	static RWLock *(*create_func)();

public:
	virtual void read_lock() = 0; ///< Lock the rwlock, block if locked by someone else
	virtual void read_unlock() = 0; ///< Unlock the rwlock, let other threads continue
	virtual Error read_try_lock() = 0; ///< Attempt to lock the rwlock, OK on success, ERROR means it can't lock.

	virtual void write_lock() = 0; ///< Lock the rwlock, block if locked by someone else
	virtual void write_unlock() = 0; ///< Unlock the rwlock, let other thwrites continue
	virtual Error write_try_lock() = 0; ///< Attempt to lock the rwlock, OK on success, ERROR means it can't lock.

	static RWLock *create(); ///< Create a rwlock

	virtual ~RWLock() {}
};

class RWLockRead {
	RWLock *lock;

public:
	RWLockRead(const RWLock *p_lock) {
		lock = const_cast<RWLock *>(p_lock);
		if (lock) {
			lock->read_lock();
		}
	}
	~RWLockRead() {
		if (lock) {
			lock->read_unlock();
		}
	}
};

class RWLockWrite {
	RWLock *lock;

public:
	RWLockWrite(RWLock *p_lock) {
		lock = p_lock;
		if (lock) {
			lock->write_lock();
		}
	}
	~RWLockWrite() {
		if (lock) {
			lock->write_unlock();
		}
	}
};

#endif // RW_LOCK_H
