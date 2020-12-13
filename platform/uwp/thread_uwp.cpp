/*************************************************************************/
/*  thread_uwp.cpp                                                       */
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

#include "thread_uwp.h"

#include "core/os/memory.h"

Thread *ThreadUWP::create_func_uwp(ThreadCreateCallback p_callback, void *p_user, const Settings &) {
	ThreadUWP *thread = memnew(ThreadUWP);

	std::thread new_thread(p_callback, p_user);
	std::swap(thread->thread, new_thread);

	return thread;
};

Thread::ID ThreadUWP::get_thread_id_func_uwp() {
	return std::hash<std::thread::id>()(std::this_thread::get_id());
};

void ThreadUWP::wait_to_finish_func_uwp(Thread *p_thread) {
	ThreadUWP *tp = static_cast<ThreadUWP *>(p_thread);
	tp->thread.join();
};

Thread::ID ThreadUWP::get_id() const {
	return std::hash<std::thread::id>()(thread.get_id());
};

void ThreadUWP::make_default() {
	create_func = create_func_uwp;
	get_thread_id_func = get_thread_id_func_uwp;
	wait_to_finish_func = wait_to_finish_func_uwp;
};
