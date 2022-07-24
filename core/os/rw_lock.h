/*************************************************************************/
/*  rw_lock.h                                                            */
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

#ifndef RW_LOCK_H
#define RW_LOCK_H

#include "core/error/error_list.h"
#include "core/os/mutex.h"

#if !defined(NO_THREADS)

#include <shared_mutex>

class RWLock {
	mutable std::shared_timed_mutex mutex;

public:
	// Lock the rwlock, block if locked by someone else
	void read_lock() const {
		mutex.lock_shared();
	}

	// Unlock the rwlock, let other threads continue
	void read_unlock() const {
		mutex.unlock_shared();
	}

	// Attempt to lock the rwlock, OK on success, ERR_BUSY means it can't lock.
	Error read_try_lock() const {
		return mutex.try_lock_shared() ? OK : ERR_BUSY;
	}

	// Lock the rwlock, block if locked by someone else
	void write_lock() {
		mutex.lock();
	}

	// Unlock the rwlock, let other thwrites continue
	void write_unlock() {
		mutex.unlock();
	}

	// Attempt to lock the rwlock, OK on success, ERR_BUSY means it can't lock.
	Error write_try_lock() {
		return mutex.try_lock() ? OK : ERR_BUSY;
	}
};

#else

class RWLock {
public:
	void read_lock() const {}
	void read_unlock() const {}
	Error read_try_lock() const { return OK; }

	void write_lock() {}
	void write_unlock() {}
	Error write_try_lock() { return OK; }
};

#endif

class RWLockRead {
	const RWLock &lock;
	bool has_ownership;

public:
	RWLockRead(const RWLock &p_lock, LockMode p_lock_mode = LOCK_MODE_LOCK) :
			lock(p_lock) {
		switch (p_lock_mode) {
			case LOCK_MODE_LOCK:
				read_lock();
				break;

			case LOCK_MODE_TRY_LOCK:
				read_try_lock();
				break;

			case LOCK_MODE_DEFER:
				has_ownership = false;
				break;
		}
	}
	~RWLockRead() {
		if (has_ownership) {
			read_unlock();
		}
	}

	void read_lock() {
		lock.read_lock();
		has_ownership = true;
	}

	Error read_try_lock() {
		Error err = lock.read_try_lock();
		if (err == OK) {
			has_ownership = true;
		}
		return err;
	}

	void read_unlock() {
		has_ownership = false;
		lock.read_unlock();
	}
};

class RWLockWrite {
	RWLock &lock;
	bool has_ownership;

public:
	RWLockWrite(RWLock &p_lock, LockMode p_lock_mode = LOCK_MODE_LOCK) :
			lock(p_lock) {
		switch (p_lock_mode) {
			case LOCK_MODE_LOCK:
				write_lock();
				break;

			case LOCK_MODE_TRY_LOCK:
				write_try_lock();
				break;

			case LOCK_MODE_DEFER:
				has_ownership = false;
				break;
		}
	}
	~RWLockWrite() {
		if (has_ownership) {
			write_unlock();
		}
	}

	void write_lock() {
		lock.write_lock();
		has_ownership = true;
	}

	Error write_try_lock() {
		Error err = lock.write_try_lock();
		if (err == OK) {
			has_ownership = true;
		}
		return err;
	}

	void write_unlock() {
		has_ownership = false;
		lock.write_unlock();
	}
};

#endif // RW_LOCK_H
