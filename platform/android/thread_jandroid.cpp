/**************************************************************************/
/*  thread_jandroid.cpp                                                   */
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

#include "thread_jandroid.h"

#include "core/os/thread.h"

#include <android/log.h>

static JavaVM *java_vm = nullptr;
static thread_local JNIEnv *env = nullptr;

// The logic here need to improve, init_thread/term_tread are designed to work with Thread::callback
// Calling init_thread from setup_android_thread and get_jni_env to setup an env we're keeping and not detaching
// could cause issues on app termination.
//
// We should be making sure that any thread started calls a nice cleanup function when it's done,
// especially now that we use many more threads.

static void init_thread() {
	if (env) {
		// thread never detached! just keep using...
		return;
	}

	java_vm->AttachCurrentThread(&env, nullptr);
}

static void term_thread() {
	java_vm->DetachCurrentThread();

	// this is no longer valid, must called init_thread to re-establish
	env = nullptr;
}

void init_thread_jandroid(JavaVM *p_jvm, JNIEnv *p_env) {
	java_vm = p_jvm;
	env = p_env;
	Thread::_set_platform_functions({ .init = init_thread, .term = &term_thread });
}

void setup_android_thread() {
	if (!env) {
		// !BAS! see remarks above
		init_thread();
	}
}

JNIEnv *get_jni_env() {
	if (!env) {
		// !BAS! see remarks above
		init_thread();
	}

	return env;
}
