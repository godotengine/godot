/**************************************************************************/
/*  test_worker_thread_pool.h                                             */
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

#ifndef TEST_WORKER_THREAD_POOL_H
#define TEST_WORKER_THREAD_POOL_H

#include "core/object/worker_thread_pool.h"

#include "tests/test_macros.h"

namespace TestWorkerThreadPool {

static LocalVector<SafeNumeric<int>> counter;

static void static_test(void *p_arg) {
	counter[(uint64_t)p_arg].increment();
	counter[0].add(2);
}
static void static_callable_test() {
	counter[0].sub(2);
}
TEST_CASE("[WorkerThreadPool] Process threads using individual tasks") {
	for (int iterations = 0; iterations < 500; iterations++) {
		const int count = Math::pow(2.0f, Math::random(0.0f, 5.0f));
		const bool low_priority = Math::rand() % 2;

		LocalVector<WorkerThreadPool::TaskID> tasks1;
		LocalVector<WorkerThreadPool::TaskID> tasks2;
		tasks1.resize(count);
		tasks2.resize(count);

		counter.clear();
		counter.resize(count);
		for (int i = 0; i < count; i++) {
			tasks1[i] = WorkerThreadPool::get_singleton()->add_native_task(static_test, (void *)(uintptr_t)i, low_priority);
			tasks2[i] = WorkerThreadPool::get_singleton()->add_task(callable_mp_static(static_callable_test), !low_priority);
		}
		for (int i = 0; i < count; i++) {
			WorkerThreadPool::get_singleton()->wait_for_task_completion(tasks1[i]);
			WorkerThreadPool::get_singleton()->wait_for_task_completion(tasks2[i]);
		}

		bool all_run_once = true;
		for (int i = 0; i < count; i++) {
			//Reduce number of check messages
			all_run_once &= counter[i].get() == 1;
		}
		CHECK(all_run_once);
	}
}

static void static_group_test(void *p_arg, uint32_t p_index) {
	counter[p_index].increment();
	counter[0].add((uintptr_t)p_arg);
}
static void static_callable_group_test(uint32_t p_index) {
	counter[p_index].increment();
	counter[0].sub(2);
}
TEST_CASE("[WorkerThreadPool] Process elements using group tasks") {
	for (int iterations = 0; iterations < 500; iterations++) {
		const int count = Math::pow(2.0f, Math::random(0.0f, 5.0f));
		const int tasks = Math::pow(2.0f, Math::random(0.0f, 5.0f));
		const bool low_priority = Math::rand() % 2;

		counter.clear();
		counter.resize(count);
		WorkerThreadPool::GroupID group1 = WorkerThreadPool::get_singleton()->add_native_group_task(static_group_test, (void *)2, count, tasks, !low_priority);
		WorkerThreadPool::GroupID group2 = WorkerThreadPool::get_singleton()->add_group_task(callable_mp_static(static_callable_group_test), count, tasks, low_priority);
		WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group1);
		WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group2);

		bool all_run_once = true;
		for (int i = 0; i < count; i++) {
			//Reduce number of check messages
			all_run_once &= counter[i].get() == 2;
		}
		CHECK(all_run_once);
	}
}

} // namespace TestWorkerThreadPool

#endif // TEST_WORKER_THREAD_POOL_H
