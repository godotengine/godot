/*************************************************************************/
/*  thread_dummy.h                                                       */
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

#ifndef THREAD_DUMMY_H
#define THREAD_DUMMY_H

#include "core/os/rw_lock.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"

class ThreadDummy : public Thread {
	static Thread *create(ThreadCreateCallback p_callback, void *p_user, const Settings &p_settings = Settings());

public:
	virtual ID get_id() const { return 0; };

	static void make_default();
};

class RWLockDummy : public RWLock {
	static RWLock *create();

public:
	virtual void read_lock() {}
	virtual void read_unlock() {}
	virtual Error read_try_lock() { return OK; }

	virtual void write_lock() {}
	virtual void write_unlock() {}
	virtual Error write_try_lock() { return OK; }

	static void make_default();
};

#endif // THREAD_DUMMY_H
