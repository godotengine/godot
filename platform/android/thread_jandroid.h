/*************************************************************************/
/*  thread_jandroid.h                                                    */
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

#ifndef THREAD_POSIX_H
#define THREAD_POSIX_H

#include "core/os/thread.h"
#include <jni.h>
#include <pthread.h>
#include <sys/types.h>

class ThreadAndroid : public Thread {
	static pthread_key_t thread_id_key;
	static ID next_thread_id;

	pthread_t pthread;
	pthread_attr_t pthread_attr;
	ThreadCreateCallback callback;
	void *user;
	ID id;

	static Thread *create_thread_jandroid();

	static void *thread_callback(void *userdata);

	static Thread *create_func_jandroid(ThreadCreateCallback p_callback, void *, const Settings &);
	static ID get_thread_id_func_jandroid();
	static void wait_to_finish_func_jandroid(Thread *p_thread);

	static void _thread_destroyed(void *value);
	ThreadAndroid();

	static pthread_key_t jvm_key;
	static JavaVM *java_vm;

public:
	virtual ID get_id() const;

	static void make_default(JavaVM *p_java_vm);
	static void setup_thread();
	static JNIEnv *get_env();

	~ThreadAndroid();
};

#endif
