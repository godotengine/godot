/*************************************************************************/
/*  thread_posix.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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

#if defined(UNIX_ENABLED) || defined(PTHREAD_ENABLED)

#include "os/memory.h"

Thread::ID ThreadPosix::get_ID() const {

	return id;	
}

Thread* ThreadPosix::create_thread_posix() {

	return memnew( ThreadPosix );
}

void *ThreadPosix::thread_callback(void *userdata) {

	ThreadPosix *t=reinterpret_cast<ThreadPosix*>(userdata);
	t->id=(ID)pthread_self();
	t->callback(t->user);
	return NULL;
}

Thread* ThreadPosix::create_func_posix(ThreadCreateCallback p_callback,void *p_user,const Settings&) {

	ThreadPosix *tr= memnew(ThreadPosix);
	tr->callback=p_callback;
	tr->user=p_user;
	pthread_attr_init(&tr->pthread_attr);
	pthread_attr_setdetachstate(&tr->pthread_attr, PTHREAD_CREATE_JOINABLE);
	pthread_attr_setstacksize(&tr->pthread_attr, 256 * 1024);
	
	pthread_create(&tr->pthread, &tr->pthread_attr, thread_callback, tr);
	
	return tr;
}
Thread::ID ThreadPosix::get_thread_ID_func_posix() {

	return (ID)pthread_self();
}
void ThreadPosix::wait_to_finish_func_posix(Thread* p_thread) {

	ThreadPosix *tp=static_cast<ThreadPosix*>(p_thread);
	ERR_FAIL_COND(!tp);
	ERR_FAIL_COND(tp->pthread==0);
	
	pthread_join(tp->pthread,NULL);
	tp->pthread=0;		
}

Error ThreadPosix::set_name(const String& p_name) {

	ERR_FAIL_COND_V(pthread == 0, ERR_UNCONFIGURED);

	#ifdef PTHREAD_RENAME_SELF

	// check if thread is the same as caller
	int caller = Thread::get_caller_ID();
	int self = get_ID();
	if (caller != self) {
		ERR_EXPLAIN("On this platform, thread can only be renamed with calls from the threads to be renamed.");
		ERR_FAIL_V(ERR_UNAVAILABLE);
		return ERR_UNAVAILABLE;
	};
	int err = pthread_setname_np(p_name.utf8().get_data());
	
	#else

	int err = pthread_setname_np(pthread, p_name.utf8().get_data());

	#endif

	return err == 0 ? OK : ERR_INVALID_PARAMETER;
};

void ThreadPosix::make_default() {

	create_func=create_func_posix;
	get_thread_ID_func=get_thread_ID_func_posix;
	wait_to_finish_func=wait_to_finish_func_posix;
	
}

ThreadPosix::ThreadPosix() {

	pthread=0;
}


ThreadPosix::~ThreadPosix() {

}


#endif
