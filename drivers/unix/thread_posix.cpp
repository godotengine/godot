/*************************************************************************/
/*  thread_posix.cpp                                                     */
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
#include "thread_posix.h"
#include "script_language.h"

#if defined(UNIX_ENABLED) || defined(PTHREAD_ENABLED)

#ifdef PTHREAD_BSD_SET_NAME
#include <pthread_np.h>
#endif

#include "os/memory.h"

Thread::ID ThreadPosix::get_ID() const {

	return id;
}

Thread *ThreadPosix::create_thread_posix() {

	return memnew(ThreadPosix);
}

void *ThreadPosix::thread_callback(void *userdata) {

	ThreadPosix *t = reinterpret_cast<ThreadPosix *>(userdata);
	t->id = (ID)pthread_self();

	ScriptServer::thread_enter(); //scripts may need to attach a stack

	t->callback(t->user);

	ScriptServer::thread_exit();

	return NULL;
}

Thread *ThreadPosix::create_func_posix(ThreadCreateCallback p_callback, void *p_user, const Settings &) {

	ThreadPosix *tr = memnew(ThreadPosix);
	tr->callback = p_callback;
	tr->user = p_user;
	pthread_attr_init(&tr->pthread_attr);
	pthread_attr_setdetachstate(&tr->pthread_attr, PTHREAD_CREATE_JOINABLE);
	pthread_attr_setstacksize(&tr->pthread_attr, 256 * 1024);

	pthread_create(&tr->pthread, &tr->pthread_attr, thread_callback, tr);

	return tr;
}
Thread::ID ThreadPosix::get_thread_ID_func_posix() {

	return (ID)pthread_self();
}
void ThreadPosix::wait_to_finish_func_posix(Thread *p_thread) {

	ThreadPosix *tp = static_cast<ThreadPosix *>(p_thread);
	ERR_FAIL_COND(!tp);
	ERR_FAIL_COND(tp->pthread == 0);

	pthread_join(tp->pthread, NULL);
	tp->pthread = 0;
}

Error ThreadPosix::set_name_func_posix(const String &p_name) {

	pthread_t running_thread = pthread_self();

#ifdef PTHREAD_NO_RENAME
	return ERR_UNAVAILABLE;

#else

#ifdef PTHREAD_RENAME_SELF

	// check if thread is the same as caller
	int err = pthread_setname_np(p_name.utf8().get_data());

#else

#ifdef PTHREAD_BSD_SET_NAME
	pthread_set_name_np(running_thread, p_name.utf8().get_data());
	int err = 0; // Open/FreeBSD ignore errors in this function
#else
	int err = pthread_setname_np(running_thread, p_name.utf8().get_data());
#endif // PTHREAD_BSD_SET_NAME

#endif // PTHREAD_RENAME_SELF

	return err == 0 ? OK : ERR_INVALID_PARAMETER;

#endif // PTHREAD_NO_RENAME
};

void ThreadPosix::make_default() {

	create_func = create_func_posix;
	get_thread_ID_func = get_thread_ID_func_posix;
	wait_to_finish_func = wait_to_finish_func_posix;
	set_name_func = set_name_func_posix;
}

ThreadPosix::ThreadPosix() {

	pthread = 0;
}

ThreadPosix::~ThreadPosix() {
}

#endif
