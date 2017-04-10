/*************************************************************************/
/*  thread.cpp                                                           */
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
#include "thread.h"

Thread *(*Thread::create_func)(ThreadCreateCallback, void *, const Settings &) = NULL;
Thread::ID (*Thread::get_thread_ID_func)() = NULL;
void (*Thread::wait_to_finish_func)(Thread *) = NULL;
Error (*Thread::set_name_func)(const String &) = NULL;

Thread::ID Thread::_main_thread_id = 0;

Thread::ID Thread::get_caller_ID() {

	if (get_thread_ID_func)
		return get_thread_ID_func();
	return 0;
}

Thread *Thread::create(ThreadCreateCallback p_callback, void *p_user, const Settings &p_settings) {

	if (create_func) {

		return create_func(p_callback, p_user, p_settings);
	}
	return NULL;
}

void Thread::wait_to_finish(Thread *p_thread) {

	if (wait_to_finish_func)
		wait_to_finish_func(p_thread);
}

Error Thread::set_name(const String &p_name) {

	if (set_name_func)
		return set_name_func(p_name);

	return ERR_UNAVAILABLE;
};

Thread::Thread() {
}

Thread::~Thread() {
}
