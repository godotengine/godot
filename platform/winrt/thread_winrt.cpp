/*************************************************************************/
/*  thread_winrt.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "thread_winrt.h"

#include "os/memory.h"

Thread* ThreadWinrt::create_func_winrt(ThreadCreateCallback p_callback,void *p_user,const Settings&) {

	ThreadWinrt* thread = memnew(ThreadWinrt);


	std::thread new_thread(p_callback, p_user);
	std::swap(thread->thread, new_thread);

	return thread;
};

Thread::ID ThreadWinrt::get_thread_ID_func_winrt() {

	return std::hash<std::thread::id>()(std::this_thread::get_id());
};

void ThreadWinrt::wait_to_finish_func_winrt(Thread* p_thread) {

	ThreadWinrt *tp=static_cast<ThreadWinrt*>(p_thread);
	tp->thread.join();
};


Thread::ID ThreadWinrt::get_ID() const {

	return std::hash<std::thread::id>()(thread.get_id());
};

void ThreadWinrt::make_default() {
	create_func = create_func_winrt;
	get_thread_ID_func = get_thread_ID_func_winrt;
	wait_to_finish_func = wait_to_finish_func_winrt;
};

ThreadWinrt::ThreadWinrt() {

};

ThreadWinrt::~ThreadWinrt() {

};

