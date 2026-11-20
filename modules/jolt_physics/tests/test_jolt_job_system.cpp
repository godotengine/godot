/**************************************************************************/
/*  test_jolt_job_system.cpp                                              */
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

#include "../spaces/jolt_job_system.h"

#include "core/object/ref_counted.h"
#include "core/object/worker_thread_pool.h"
#include "core/os/os.h"
#include "core/os/semaphore.h"
#include "tests/test_macros.h"

namespace TestJoltJobSystem {

void test_completed_job_reclamation(bool p_post_step) {
	Ref<RefCounted> captured;
	captured.instantiate();
	CHECK(captured->get_reference_count() == 1);

	{
		JoltJobSystem system;
		JPH::JobSystem &api = system;
		WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
		WorkerThreadPool::TaskID task_id = WorkerThreadPool::INVALID_TASK_ID;
		Semaphore started;
		bool ran = false;
		JPH::JobHandle handle = api.CreateJob("Jolt job reclamation test", JPH::Color::sBlue, [captured, pool, &task_id, &started, &ran]() {
			ran = captured.is_valid();
			task_id = pool->get_caller_task_id();
			started.post();
		});
		started.wait();

		// Check completion without consuming the task ID. Job's destructor waits on it.
		if (pool->get_thread_count() > 0) {
			while (!pool->is_task_completed(task_id)) {
				OS::get_singleton()->delay_usec(100);
			}
		}
		CHECK(ran);
		CHECK(handle.IsDone());

		// Keep the last handle through post_step(), then release it after workers finish.
		system.post_step();
		handle = JPH::JobHandle();
		CHECK(captured->get_reference_count() == 2);

		if (p_post_step) {
			system.post_step();
			CHECK(captured->get_reference_count() == 1);
		}
	}

	CHECK(captured->get_reference_count() == 1);
}

} // namespace TestJoltJobSystem
