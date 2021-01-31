/*************************************************************************/
/*  thread.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/script_language.h"

#if !defined(NO_THREADS)

#include "core/safe_refcount.h"

Error (*Thread::set_name_func)(const String &) = nullptr;
void (*Thread::set_priority_func)(Thread::Priority) = nullptr;
void (*Thread::init_func)() = nullptr;
void (*Thread::term_func)() = nullptr;

Thread::ID Thread::main_thread_id = 1;
SafeNumeric<Thread::ID> Thread::last_thread_id{ 1 };
thread_local Thread::ID Thread::caller_id = 1;

void Thread::_set_platform_funcs(
		Error (*p_set_name_func)(const String &),
		void (*p_set_priority_func)(Thread::Priority),
		void (*p_init_func)(),
		void (*p_term_func)()) {
	Thread::set_name_func = p_set_name_func;
	Thread::set_priority_func = p_set_priority_func;
	Thread::init_func = p_init_func;
	Thread::term_func = p_term_func;
}

void Thread::callback(Thread *p_self, const Settings &p_settings, Callback p_callback, void *p_userdata) {
	Thread::caller_id = p_self->id;
	if (set_priority_func) {
		set_priority_func(p_settings.priority);
	}
	if (init_func) {
		init_func();
	}
	ScriptServer::thread_enter(); //scripts may need to attach a stack
	p_callback(p_userdata);
	ScriptServer::thread_exit();
	if (term_func) {
		term_func();
	}
}

void Thread::start(Thread::Callback p_callback, void *p_user, const Settings &p_settings) {
	if (id != 0) {
#ifdef DEBUG_ENABLED
		WARN_PRINT("A Thread object has been re-started without wait_to_finish() having been called on it. Please do so to ensure correct cleanup of the thread.");
#endif
		thread.detach();
		std::thread empty_thread;
		thread.swap(empty_thread);
	}
	id = last_thread_id.increment();
	std::thread new_thread(&Thread::callback, this, p_settings, p_callback, p_user);
	thread.swap(new_thread);
}

bool Thread::is_started() const {
	return id != 0;
}

void Thread::wait_to_finish() {
	if (id != 0) {
		thread.join();
		std::thread empty_thread;
		thread.swap(empty_thread);
		id = 0;
	}
}

Error Thread::set_name(const String &p_name) {
	if (set_name_func) {
		return set_name_func(p_name);
	}

	return ERR_UNAVAILABLE;
}

Thread::Thread() :
		id(0) {}

Thread::~Thread() {
	if (id != 0) {
#ifdef DEBUG_ENABLED
		WARN_PRINT("A Thread object has been destroyed without wait_to_finish() having been called on it. Please do so to ensure correct cleanup of the thread.");
#endif
		thread.detach();
	}
}

#endif
