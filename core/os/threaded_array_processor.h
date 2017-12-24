#ifndef THREADED_ARRAY_PROCESSOR_H
#define THREADED_ARRAY_PROCESSOR_H

#include "os/mutex.h"
#include "os/os.h"
#include "os/thread.h"
#include "safe_refcount.h"
#include "thread_safe.h"

template <class C, class U>
struct ThreadArrayProcessData {
	uint32_t elements;
	uint32_t index;
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
		uint32_t index = atomic_increment(&data.index);
		if (index >= data.elements)
			break;
		data.process(index);
	}
}

template <class C, class M, class U>
void thread_process_array(uint32_t p_elements, C *p_instance, M p_method, U p_userdata) {

	ThreadArrayProcessData<C, U> data;
	data.method = p_method;
	data.instance = p_instance;
	data.userdata = p_userdata;
	data.index = 0;
	data.elements = p_elements;
	data.process(data.index); //process first, let threads increment for next

	Vector<Thread *> threads;

	threads.resize(OS::get_singleton()->get_processor_count());

	for (int i = 0; i < threads.size(); i++) {
		threads[i] = Thread::create(process_array_thread<ThreadArrayProcessData<C, U> >, &data);
	}

	for (int i = 0; i < threads.size(); i++) {
		Thread::wait_to_finish(threads[i]);
		memdelete(threads[i]);
	}
}

#else

template <class C, class M, class U>
void thread_process_array(uint32_t p_elements, C *p_instance, M p_method, U p_userdata) {

	ThreadArrayProcessData<C, U> data;
	data.method = p_method;
	data.instance = p_instance;
	data.userdata = p_userdata;
	data.index = 0;
	data.elements = p_elements;
	for (uint32_t i = 0; i < p_elements; i++) {
		data.process(i);
	}
}

#endif

#endif // THREADED_ARRAY_PROCESSOR_H
