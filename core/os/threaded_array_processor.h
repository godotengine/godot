/*************************************************************************/
/*  threaded_array_processor.h                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef THREADED_ARRAY_PROCESSOR_H
#define THREADED_ARRAY_PROCESSOR_H

#include "core/os/mutex.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "core/os/thread_safe.h"
#include "core/templates/safe_refcount.h"

template <class C, class U>
struct ThreadArrayProcessData {
	uint32_t elements;
	SafeNumeric<uint32_t> index;
	C *instance;
	U userdata;
	void (C::*method)(uint32_t, U);

	void process(uint32_t p_index) {
		(instance->*method)(p_index, userdata);
	}
};

#ifndef NO_THREADS

template <class T>
void process_array_thread(void *ud) {
	T &data = *(T *)ud;
	while (true) {
		uint32_t index = data.index.increment();
		if (index >= data.elements) {
			break;
		}
		data.process(index);
	}
}

template <class C, class M, class U>
void thread_process_array(uint32_t p_elements, C *p_instance, M p_method, U p_userdata) {
	ThreadArrayProcessData<C, U> data;
	data.method = p_method;
	data.instance = p_instance;
	data.userdata = p_userdata;
	data.index.set(0);
	data.elements = p_elements;
	data.process(0); //process first, let threads increment for next

	int thread_count = OS::get_singleton()->get_processor_count();
	Thread *threads = memnew_arr(Thread, thread_count);

	for (int i = 0; i < thread_count; i++) {
		threads[i].start(process_array_thread<ThreadArrayProcessData<C, U>>, &data);
	}

	for (int i = 0; i < thread_count; i++) {
		threads[i].wait_to_finish();
	}
	memdelete_arr(threads);
}

#else

template <class C, class M, class U>
void thread_process_array(uint32_t p_elements, C *p_instance, M p_method, U p_userdata) {
	ThreadArrayProcessData<C, U> data;
	data.method = p_method;
	data.instance = p_instance;
	data.userdata = p_userdata;
	data.index.set(0);
	data.elements = p_elements;
	for (uint32_t i = 0; i < p_elements; i++) {
		data.process(i);
	}
}

#endif

#endif // THREADED_ARRAY_PROCESSOR_H
