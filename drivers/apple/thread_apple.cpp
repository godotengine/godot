/**************************************************************************/
/*  thread_apple.cpp                                                      */
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

#include "thread_apple.h"

#include "core/error/error_macros.h"
#include "core/object/script_language.h"
#include "core/string/ustring.h"

SafeNumeric<uint64_t> Thread::id_counter(1); // The first value after .increment() is 2, hence by default the main thread ID should be 1.
thread_local Thread::ID Thread::caller_id = Thread::id_counter.increment();

struct ThreadData {
	Thread::Callback callback;
	void *userdata;
	Thread::ID caller_id;
};

void *Thread::thread_callback(void *p_data) {
	ThreadData *thread_data = static_cast<ThreadData *>(p_data);

	// Set the caller ID for this thread
	caller_id = thread_data->caller_id;

	ScriptServer::thread_enter(); // Scripts may need to attach a stack.

	// Call the actual callback
	thread_data->callback(thread_data->userdata);

	ScriptServer::thread_exit();

	// Clean up
	memdelete(thread_data);

	return nullptr;
}

Error Thread::set_name(const String &p_name) {
	int err = pthread_setname_np(p_name.utf8().get_data());
	return err == 0 ? OK : ERR_INVALID_PARAMETER;
}

Thread::ID Thread::start(Thread::Callback p_callback, void *p_user, const Settings &p_settings) {
	ERR_FAIL_COND_V_MSG(id != UNASSIGNED_ID, UNASSIGNED_ID, "A Thread object has been re-started without wait_to_finish() having been called on it.");
	id = id_counter.increment();

	ThreadData *thread_data = memnew(ThreadData);
	thread_data->callback = p_callback;
	thread_data->userdata = p_user;
	thread_data->caller_id = id;

	// Create the thread
	pthread_attr_t attr;
	pthread_attr_init(&attr);

	switch (p_settings.priority) {
		case PRIORITY_LOW:
			pthread_attr_set_qos_class_np(&attr, QOS_CLASS_UTILITY, 0);
			break;
		case PRIORITY_NORMAL:
			pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INITIATED, 0);
			break;
		case PRIORITY_HIGH:
			pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
			break;
	}

	// The default stack size for secondary threads on Apple platforms is 512KiB.
	// This is insufficient when using a library like SPIRV-Cross, which can generate deep stacks and result in a stack overflow.
	// It also creates a problematic discrepancy with other platforms, where secondary threads are often at least 1 MiB.
	pthread_attr_setstacksize(&attr,
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
			// ASan (and to some degree TSan) needs a lot of extra stack size.
			4 * 1024 * 1024 // 4 MiB
#elif !defined(__OPTIMIZE__)
			// Unoptimized builds also need a larger stack size.
			2 * 1024 * 1024 // 2 MiB
#else
			1 * 1024 * 1024 // 1 MiB
#endif
	);

	// Create the thread
	pthread_create(&pthread, &attr, thread_callback, thread_data);

	// Clean up attributes
	pthread_attr_destroy(&attr);

	return id;
}

void Thread::wait_to_finish() {
	ERR_FAIL_COND_MSG(id == UNASSIGNED_ID, "Attempt of waiting to finish on a thread that was never started.");
	ERR_FAIL_COND_MSG(id == get_caller_id(), "Threads can't wait to finish on themselves, another thread must wait.");

	int err = pthread_join(pthread, nullptr);
	if (err != 0) {
		ERR_FAIL_MSG("Thread::wait_to_finish() failed to join thread.");
	}
	pthread = pthread_t();
	id = UNASSIGNED_ID;
}

Thread::~Thread() {
	if (id != UNASSIGNED_ID) {
#ifdef DEBUG_ENABLED
		WARN_PRINT(
				"A Thread object is being destroyed without its completion having been realized.\n"
				"Please call wait_to_finish() on it to ensure correct cleanup.");
#endif
		pthread_detach(pthread);
	}
}
