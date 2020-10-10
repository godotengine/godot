/*************************************************************************/
/*  thread_jandroid.cpp                                                  */
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

#include "thread_jandroid.h"

#include "core/os/memory.h"
#include "core/safe_refcount.h"
#include "core/script_language.h"

static void _thread_id_key_destr_callback(void *p_value) {
	memdelete(static_cast<Thread::ID *>(p_value));
}

static pthread_key_t _create_thread_id_key() {
	pthread_key_t key;
	pthread_key_create(&key, &_thread_id_key_destr_callback);
	return key;
}

pthread_key_t ThreadAndroid::thread_id_key = _create_thread_id_key();
Thread::ID ThreadAndroid::next_thread_id = 0;

Thread::ID ThreadAndroid::get_id() const {
	return id;
}

Thread *ThreadAndroid::create_thread_jandroid() {
	return memnew(ThreadAndroid);
}

void *ThreadAndroid::thread_callback(void *userdata) {
	ThreadAndroid *t = reinterpret_cast<ThreadAndroid *>(userdata);
	setup_thread();
	ScriptServer::thread_enter(); //scripts may need to attach a stack
	t->id = atomic_increment(&next_thread_id);
	pthread_setspecific(thread_id_key, (void *)memnew(ID(t->id)));
	t->callback(t->user);
	ScriptServer::thread_exit();
	return nullptr;
}

Thread *ThreadAndroid::create_func_jandroid(ThreadCreateCallback p_callback, void *p_user, const Settings &) {
	ThreadAndroid *tr = memnew(ThreadAndroid);
	tr->callback = p_callback;
	tr->user = p_user;
	pthread_attr_init(&tr->pthread_attr);
	pthread_attr_setdetachstate(&tr->pthread_attr, PTHREAD_CREATE_JOINABLE);

	pthread_create(&tr->pthread, &tr->pthread_attr, thread_callback, tr);

	return tr;
}

Thread::ID ThreadAndroid::get_thread_id_func_jandroid() {
	void *value = pthread_getspecific(thread_id_key);

	if (value)
		return *static_cast<ID *>(value);

	ID new_id = atomic_increment(&next_thread_id);
	pthread_setspecific(thread_id_key, (void *)memnew(ID(new_id)));
	return new_id;
}

void ThreadAndroid::wait_to_finish_func_jandroid(Thread *p_thread) {
	ThreadAndroid *tp = static_cast<ThreadAndroid *>(p_thread);
	ERR_FAIL_COND(!tp);
	ERR_FAIL_COND(tp->pthread == 0);

	pthread_join(tp->pthread, nullptr);
	tp->pthread = 0;
}

void ThreadAndroid::_thread_destroyed(void *value) {
	/* The thread is being destroyed, detach it from the Java VM and set the mThreadKey value to NULL as required */
	JNIEnv *env = (JNIEnv *)value;
	if (env != nullptr) {
		java_vm->DetachCurrentThread();
		pthread_setspecific(jvm_key, nullptr);
	}
}

pthread_key_t ThreadAndroid::jvm_key;
JavaVM *ThreadAndroid::java_vm = nullptr;

void ThreadAndroid::setup_thread() {
	if (pthread_getspecific(jvm_key))
		return; //already setup
	JNIEnv *env;
	java_vm->AttachCurrentThread(&env, nullptr);
	pthread_setspecific(jvm_key, (void *)env);
}

void ThreadAndroid::make_default(JavaVM *p_java_vm) {
	java_vm = p_java_vm;
	create_func = create_func_jandroid;
	get_thread_id_func = get_thread_id_func_jandroid;
	wait_to_finish_func = wait_to_finish_func_jandroid;
	pthread_key_create(&jvm_key, _thread_destroyed);
	setup_thread();
}

JNIEnv *ThreadAndroid::get_env() {
	if (!pthread_getspecific(jvm_key)) {
		setup_thread();
	}

	JNIEnv *env = nullptr;
	java_vm->AttachCurrentThread(&env, nullptr);
	return env;
}

ThreadAndroid::ThreadAndroid() {
	pthread = 0;
}

ThreadAndroid::~ThreadAndroid() {
}
