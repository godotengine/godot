/*************************************************************************/
/*  test_worker_thread_pool.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_WORKER_THREAD_POOL_H
#define TEST_WORKER_THREAD_POOL_H

#include "core/object/worker_thread_pool.h"

#include "tests/test_macros.h"

namespace TestWorkerThreadPool {

int u32scmp(const char32_t *l, const char32_t *r) {
	for (; *l == *r && *l && *r; l++, r++) {
		// Continue.
	}
	return *l - *r;
}

static void static_test(void *p_arg) {
	SafeNumeric<uint32_t> *counter = (SafeNumeric<uint32_t> *)p_arg;
	counter->increment();
}

static SafeNumeric<uint32_t> callable_counter;

static void static_callable_test() {
	callable_counter.increment();
}

TEST_CASE("[WorkerThreadPool] Process 256 threads using native task") {
	const int count = 256;
	SafeNumeric<uint32_t> counter;
	WorkerThreadPool::TaskID tasks[count];
	for (int i = 0; i < count; i++) {
		tasks[i] = WorkerThreadPool::get_singleton()->add_native_task(static_test, &counter, true);
	}
	for (int i = 0; i < count; i++) {
		WorkerThreadPool::get_singleton()->wait_for_task_completion(tasks[i]);
	}

	CHECK(counter.get() == count);
}

TEST_CASE("[WorkerThreadPool] Process 256 threads using native low priority") {
	const int count = 256;
	SafeNumeric<uint32_t> counter = SafeNumeric<uint32_t>(0);
	WorkerThreadPool::TaskID tasks[count];
	for (int i = 0; i < count; i++) {
		tasks[i] = WorkerThreadPool::get_singleton()->add_native_task(static_test, &counter, false);
	}
	for (int i = 0; i < count; i++) {
		WorkerThreadPool::get_singleton()->wait_for_task_completion(tasks[i]);
	}

	CHECK(counter.get() == count);
}

TEST_CASE("[WorkerThreadPool] Process 256 threads using callable") {
	const int count = 256;
	WorkerThreadPool::TaskID tasks[count];
	callable_counter.set(0);
	for (int i = 0; i < count; i++) {
		tasks[i] = WorkerThreadPool::get_singleton()->add_task(callable_mp_static(static_callable_test), true);
	}
	for (int i = 0; i < count; i++) {
		WorkerThreadPool::get_singleton()->wait_for_task_completion(tasks[i]);
	}

	CHECK(callable_counter.get() == count);
}

TEST_CASE("[WorkerThreadPool] Process 256 threads using callable low priority") {
	const int count = 256;
	WorkerThreadPool::TaskID tasks[count];
	callable_counter.set(0);
	for (int i = 0; i < count; i++) {
		tasks[i] = WorkerThreadPool::get_singleton()->add_task(callable_mp_static(static_callable_test), false);
	}
	for (int i = 0; i < count; i++) {
		WorkerThreadPool::get_singleton()->wait_for_task_completion(tasks[i]);
	}

	CHECK(callable_counter.get() == count);
}

static void static_group_test(void *p_arg, uint32_t p_index) {
	SafeNumeric<uint32_t> *counter = (SafeNumeric<uint32_t> *)p_arg;
	counter->exchange_if_greater(p_index);
}

TEST_CASE("[WorkerThreadPool] Process 256 elements on native task group") {
	const int count = 256;
	SafeNumeric<uint32_t> counter;
	WorkerThreadPool::GroupID group = WorkerThreadPool::get_singleton()->add_native_group_task(static_group_test, &counter, count, -1, true);
	WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group);
	CHECK(counter.get() == count - 1);
}

TEST_CASE("[WorkerThreadPool] Process 256 elements on native task group low priority") {
	const int count = 256;
	SafeNumeric<uint32_t> counter;
	WorkerThreadPool::GroupID group = WorkerThreadPool::get_singleton()->add_native_group_task(static_group_test, &counter, count, -1, false);
	WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group);
	CHECK(counter.get() == count - 1);
}

static SafeNumeric<uint32_t> callable_group_counter;

static void static_callable_group_test(uint32_t p_index) {
	callable_group_counter.exchange_if_greater(p_index);
}

TEST_CASE("[WorkerThreadPool] Process 256 elements on native task group") {
	const int count = 256;
	WorkerThreadPool::GroupID group = WorkerThreadPool::get_singleton()->add_group_task(callable_mp_static(static_callable_group_test), count, -1, true);
	WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group);
	CHECK(callable_group_counter.get() == count - 1);
}

TEST_CASE("[WorkerThreadPool] Process 256 elements on native task group low priority") {
	const int count = 256;
	callable_group_counter.set(0);
	WorkerThreadPool::GroupID group = WorkerThreadPool::get_singleton()->add_group_task(callable_mp_static(static_callable_group_test), count, -1, false);
	WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group);
	CHECK(callable_group_counter.get() == count - 1);
}

} // namespace TestWorkerThreadPool

#endif // TEST_WORKER_THREAD_POOL_H
