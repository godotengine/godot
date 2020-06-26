/*************************************************************************/
/*  thread_windows.h                                                     */
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

#ifndef THREAD_WINDOWS_H
#define THREAD_WINDOWS_H

#ifdef WINDOWS_ENABLED

#include "core/os/thread.h"
#include "core/script_language.h"

#include <windows.h>

class ThreadWindows : public Thread {
	ThreadCreateCallback callback;
	void *user;
	ID id;
	HANDLE handle = nullptr;

	static Thread *create_thread_windows();

	static DWORD WINAPI thread_callback(LPVOID userdata);

	static Thread *create_func_windows(ThreadCreateCallback p_callback, void *, const Settings &);
	static ID get_thread_id_func_windows();
	static void wait_to_finish_func_windows(Thread *p_thread);

	ThreadWindows() {}

public:
	virtual ID get_id() const;

	static void make_default();

	~ThreadWindows() {}
};

#endif

#endif
