#include "thread_work_pool.h"
#include "core/os/os.h"

void ThreadWorkPool::_thread_function(ThreadData *p_thread) {

	while (true) {
		p_thread->start.wait();
		if (p_thread->exit.load()) {
			break;
		}
		p_thread->work->work();
		p_thread->completed.post();
	}
}

void ThreadWorkPool::init(int p_thread_count) {
	ERR_FAIL_COND(threads != nullptr);
	if (p_thread_count < 0) {
		p_thread_count = OS::get_singleton()->get_processor_count();
	}

	thread_count = p_thread_count;
	threads = memnew_arr(ThreadData, thread_count);

	for (uint32_t i = 0; i < thread_count; i++) {
		threads[i].exit.store(false);
		threads[i].thread = memnew(std::thread(ThreadWorkPool::_thread_function, &threads[i]));
	}
}

void ThreadWorkPool::finish() {

	if (threads == nullptr) {
		return;
	}

	for (uint32_t i = 0; i < thread_count; i++) {
		threads[i].exit.store(true);
		threads[i].start.post();
	}
	for (uint32_t i = 0; i < thread_count; i++) {
		threads[i].thread->join();
		memdelete(threads[i].thread);
	}

	memdelete_arr(threads);
	threads = nullptr;
}

ThreadWorkPool::~ThreadWorkPool() {

	finish();
}
