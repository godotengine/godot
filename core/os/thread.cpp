/**************************************************************************/
/*  thread.cpp                                                            */
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

#include "platform_config.h"
#ifndef PLATFORM_THREAD_OVERRIDE // See details in thread.h

#include "thread.h"

#include "core/object/script_language.h"
#include "core/templates/safe_refcount.h"

Thread::PlatformFunctions Thread::platform_functions;

uint64_t Thread::_thread_id_hash(const std::thread::id &p_t) {
	static std::hash<std::thread::id> hasher;
	return hasher(p_t);
}

Thread::ID Thread::main_thread_id = _thread_id_hash(std::this_thread::get_id());
thread_local Thread::ID Thread::caller_id = 0;

void Thread::_set_platform_functions(const PlatformFunctions &p_functions) {
	platform_functions = p_functions;
}

void Thread::callback(Thread *p_self, const Settings &p_settings, Callback p_callback, void *p_userdata) {
	Thread::caller_id = _thread_id_hash(p_self->thread.get_id());
	if (platform_functions.set_priority) {
		platform_functions.set_priority(p_settings.priority);
	}
	if (platform_functions.init) {
		platform_functions.init();
	}
	ScriptServer::thread_enter(); // Scripts may need to attach a stack.
	if (platform_functions.wrapper) {
		platform_functions.wrapper(p_callback, p_userdata);
	} else {
		p_callback(p_userdata);
	}
	ScriptServer::thread_exit();
	if (platform_functions.term) {
		platform_functions.term();
	}
}

void Thread::start(Thread::Callback p_callback, void *p_user, const Settings &p_settings) {
	if (id != _thread_id_hash(std::thread::id())) {
#ifdef DEBUG_ENABLED
		WARN_PRINT("A Thread object has been re-started without wait_to_finish() having been called on it. Please do so to ensure correct cleanup of the thread.");
#endif
		thread.detach();
		std::thread empty_thread;
		thread.swap(empty_thread);
	}
	std::thread new_thread(&Thread::callback, this, p_settings, p_callback, p_user);
	thread.swap(new_thread);
	id = _thread_id_hash(thread.get_id());
}

bool Thread::is_started() const {
	return id != _thread_id_hash(std::thread::id());
}

void Thread::wait_to_finish() {
	if (id != _thread_id_hash(std::thread::id())) {
		ERR_FAIL_COND_MSG(id == get_caller_id(), "A Thread can't wait for itself to finish.");
		thread.join();
		std::thread empty_thread;
		thread.swap(empty_thread);
		id = _thread_id_hash(std::thread::id());
	}
}

Error Thread::set_name(const String &p_name) {
	if (platform_functions.set_name) {
		return platform_functions.set_name(p_name);
	}

	return ERR_UNAVAILABLE;
}

Thread::Thread() {
	caller_id = _thread_id_hash(std::this_thread::get_id());
}

Thread::~Thread() {
	if (id != _thread_id_hash(std::thread::id())) {
#ifdef DEBUG_ENABLED
		WARN_PRINT("A Thread object has been destroyed without wait_to_finish() having been called on it. Please do so to ensure correct cleanup of the thread.");
#endif
		thread.detach();
	}
}

#endif // PLATFORM_THREAD_OVERRIDE
