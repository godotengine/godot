/*************************************************************************/
/*  thread_windows.cpp                                                   */
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

#include "thread_windows.h"

#if defined(WINDOWS_ENABLED) && !defined(UWP_ENABLED)

#include "core/os/memory.h"

Thread::ID ThreadWindows::get_id() const {
	return id;
}

Thread *ThreadWindows::create_thread_windows() {
	return memnew(ThreadWindows);
}

DWORD ThreadWindows::thread_callback(LPVOID userdata) {
	ThreadWindows *t = reinterpret_cast<ThreadWindows *>(userdata);

	ScriptServer::thread_enter(); //scripts may need to attach a stack

	t->id = (ID)GetCurrentThreadId(); // must implement
	t->callback(t->user);
	SetEvent(t->handle);

	ScriptServer::thread_exit();

	return 0;
}

Thread *ThreadWindows::create_func_windows(ThreadCreateCallback p_callback, void *p_user, const Settings &) {
	ThreadWindows *tr = memnew(ThreadWindows);
	tr->callback = p_callback;
	tr->user = p_user;
	tr->handle = CreateEvent(nullptr, TRUE, FALSE, nullptr);

	QueueUserWorkItem(thread_callback, tr, WT_EXECUTELONGFUNCTION);

	return tr;
}

Thread::ID ThreadWindows::get_thread_id_func_windows() {
	return (ID)GetCurrentThreadId(); //must implement
}

void ThreadWindows::wait_to_finish_func_windows(Thread *p_thread) {
	ThreadWindows *tp = static_cast<ThreadWindows *>(p_thread);
	ERR_FAIL_COND(!tp);
	WaitForSingleObject(tp->handle, INFINITE);
	CloseHandle(tp->handle);
	//`memdelete(tp);
}

void ThreadWindows::make_default() {
	create_func = create_func_windows;
	get_thread_id_func = get_thread_id_func_windows;
	wait_to_finish_func = wait_to_finish_func_windows;
}

#endif
