/*************************************************************************/
/*  thread_dummy.h                                                       */
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
#ifndef THREAD_DUMMY_H
#define THREAD_DUMMY_H

#include "mutex.h"
#include "semaphore.h"
#include "thread.h"

class ThreadDummy : public Thread {

	static Thread *create(ThreadCreateCallback p_callback, void *p_user, const Settings &p_settings = Settings());

public:
	virtual ID get_ID() const { return 0; };

	static void make_default();
};

class MutexDummy : public Mutex {

	static Mutex *create(bool p_recursive);

public:
	virtual void lock(){};
	virtual void unlock(){};
	virtual Error try_lock() { return OK; };

	static void make_default();
};

class SemaphoreDummy : public Semaphore {

	static Semaphore *create();

public:
	virtual Error wait() { return OK; };
	virtual Error post() { return OK; };
	virtual int get() const { return 0; }; ///< get semaphore value

	static void make_default();
};

#endif
