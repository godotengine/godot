/*************************************************************************/
/*  test_command_queue.h                                                 */
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

#ifndef TEST_COMMAND_QUEUE_H
#define TEST_COMMAND_QUEUE_H

#include "test_command_queue.h"

#include "core/command_queue_mt.h"
#include "core/os/mutex.h"
#include "core/os/os.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "core/project_settings.h"

#if !defined(NO_THREADS)

namespace TestCommandQueue {

class ThreadWork {
	Semaphore thread_sem;
	Semaphore main_sem;
	Mutex mut;
	int threading_errors = 0;
	enum State {
		MAIN_START,
		MAIN_DONE,
		THREAD_START,
		THREAD_DONE,
	} state;

public:
	ThreadWork() {
		mut.lock();
		state = MAIN_START;
	}
	~ThreadWork() {
		CHECK_MESSAGE(threading_errors == 0, "threads did not lock/unlock correctly");
	}
	void thread_wait_for_work() {
		thread_sem.wait();
		mut.lock();
		if (state != MAIN_DONE) {
			threading_errors++;
		}
		state = THREAD_START;
	}
	void thread_done_work() {
		if (state != THREAD_START) {
			threading_errors++;
		}
		state = THREAD_DONE;
		mut.unlock();
		main_sem.post();
	}

	void main_wait_for_done() {
		main_sem.wait();
		mut.lock();
		if (state != THREAD_DONE) {
			threading_errors++;
		}
		state = MAIN_START;
	}
	void main_start_work() {
		if (state != MAIN_START) {
			threading_errors++;
		}
		state = MAIN_DONE;
		mut.unlock();
		thread_sem.post();
	}
};

class SharedThreadState {
public:
	ThreadWork reader_threadwork;
	ThreadWork writer_threadwork;

	CommandQueueMT command_queue = CommandQueueMT(true);

	enum TestMsgType {
		TEST_MSG_FUNC1_TRANSFORM,
		TEST_MSG_FUNC2_TRANSFORM_FLOAT,
		TEST_MSG_FUNC3_TRANSFORMx6,
		TEST_MSGSYNC_FUNC1_TRANSFORM,
		TEST_MSGSYNC_FUNC2_TRANSFORM_FLOAT,
		TEST_MSGRET_FUNC1_TRANSFORM,
		TEST_MSGRET_FUNC2_TRANSFORM_FLOAT,
		TEST_MSG_MAX
	};

	Vector<TestMsgType> message_types_to_write;
	bool during_writing = false;
	int message_count_to_read = 0;
	bool exit_threads = false;

	Thread *reader_thread = nullptr;
	Thread *writer_thread = nullptr;

	int func1_count = 0;

	void func1(Transform t) {
		func1_count++;
	}
	void func2(Transform t, float f) {
		func1_count++;
	}
	void func3(Transform t1, Transform t2, Transform t3, Transform t4, Transform t5, Transform t6) {
		func1_count++;
	}
	Transform func1r(Transform t) {
		func1_count++;
		return t;
	}
	Transform func2r(Transform t, float f) {
		func1_count++;
		return t;
	}

	void add_msg_to_write(TestMsgType type) {
		message_types_to_write.push_back(type);
	}

	void reader_thread_loop() {
		reader_threadwork.thread_wait_for_work();
		while (!exit_threads) {
			if (message_count_to_read < 0) {
				command_queue.flush_all();
			}
			for (int i = 0; i < message_count_to_read; i++) {
				command_queue.wait_and_flush_one();
			}
			message_count_to_read = 0;

			reader_threadwork.thread_done_work();
			reader_threadwork.thread_wait_for_work();
		}
		command_queue.flush_all();
		reader_threadwork.thread_done_work();
	}
	static void static_reader_thread_loop(void *stsvoid) {
		SharedThreadState *sts = static_cast<SharedThreadState *>(stsvoid);
		sts->reader_thread_loop();
	}

	void writer_thread_loop() {
		during_writing = false;
		writer_threadwork.thread_wait_for_work();
		while (!exit_threads) {
			Transform tr;
			Transform otr;
			float f = 1;
			during_writing = true;
			for (int i = 0; i < message_types_to_write.size(); i++) {
				TestMsgType msg_type = message_types_to_write[i];
				switch (msg_type) {
					case TEST_MSG_FUNC1_TRANSFORM:
						command_queue.push(this, &SharedThreadState::func1, tr);
						break;
					case TEST_MSG_FUNC2_TRANSFORM_FLOAT:
						command_queue.push(this, &SharedThreadState::func2, tr, f);
						break;
					case TEST_MSG_FUNC3_TRANSFORMx6:
						command_queue.push(this, &SharedThreadState::func3, tr, tr, tr, tr, tr, tr);
						break;
					case TEST_MSGSYNC_FUNC1_TRANSFORM:
						command_queue.push_and_sync(this, &SharedThreadState::func1, tr);
						break;
					case TEST_MSGSYNC_FUNC2_TRANSFORM_FLOAT:
						command_queue.push_and_sync(this, &SharedThreadState::func2, tr, f);
						break;
					case TEST_MSGRET_FUNC1_TRANSFORM:
						command_queue.push_and_ret(this, &SharedThreadState::func1r, tr, &otr);
						break;
					case TEST_MSGRET_FUNC2_TRANSFORM_FLOAT:
						command_queue.push_and_ret(this, &SharedThreadState::func2r, tr, f, &otr);
						break;
					default:
						break;
				}
			}
			message_types_to_write.clear();
			during_writing = false;

			writer_threadwork.thread_done_work();
			writer_threadwork.thread_wait_for_work();
		}
		writer_threadwork.thread_done_work();
	}
	static void static_writer_thread_loop(void *stsvoid) {
		SharedThreadState *sts = static_cast<SharedThreadState *>(stsvoid);
		sts->writer_thread_loop();
	}

	void init_threads() {
		reader_thread = Thread::create(&SharedThreadState::static_reader_thread_loop, this);
		writer_thread = Thread::create(&SharedThreadState::static_writer_thread_loop, this);
	}
	void destroy_threads() {
		exit_threads = true;
		reader_threadwork.main_start_work();
		writer_threadwork.main_start_work();

		Thread::wait_to_finish(reader_thread);
		memdelete(reader_thread);
		reader_thread = nullptr;
		Thread::wait_to_finish(writer_thread);
		memdelete(writer_thread);
		writer_thread = nullptr;
	}
};

TEST_CASE("[CommandQueue] Test Queue Basics") {
	const char *COMMAND_QUEUE_SETTING = "memory/limits/command_queue/multithreading_queue_size_kb";
	ProjectSettings::get_singleton()->set_setting(COMMAND_QUEUE_SETTING, 1);
	SharedThreadState sts;
	sts.init_threads();

	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC1_TRANSFORM);
	sts.writer_threadwork.main_start_work();
	sts.writer_threadwork.main_wait_for_done();
	CHECK_MESSAGE(sts.func1_count == 0,
			"Control: no messages read before reader has run.");

	sts.message_count_to_read = 1;
	sts.reader_threadwork.main_start_work();
	sts.reader_threadwork.main_wait_for_done();
	CHECK_MESSAGE(sts.func1_count == 1,
			"Reader should have read one message");

	sts.message_count_to_read = -1;
	sts.reader_threadwork.main_start_work();
	sts.reader_threadwork.main_wait_for_done();
	CHECK_MESSAGE(sts.func1_count == 1,
			"Reader should have read no additional messages from flush_all");

	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC1_TRANSFORM);
	sts.writer_threadwork.main_start_work();
	sts.writer_threadwork.main_wait_for_done();

	sts.message_count_to_read = -1;
	sts.reader_threadwork.main_start_work();
	sts.reader_threadwork.main_wait_for_done();
	CHECK_MESSAGE(sts.func1_count == 2,
			"Reader should have read one additional message from flush_all");

	sts.destroy_threads();

	CHECK_MESSAGE(sts.func1_count == 2,
			"Reader should have read no additional messages after join");
	ProjectSettings::get_singleton()->set_setting(COMMAND_QUEUE_SETTING,
			ProjectSettings::get_singleton()->property_get_revert(COMMAND_QUEUE_SETTING));
}

TEST_CASE("[CommandQueue] Test Waiting at Queue Full") {
	const char *COMMAND_QUEUE_SETTING = "memory/limits/command_queue/multithreading_queue_size_kb";
	ProjectSettings::get_singleton()->set_setting(COMMAND_QUEUE_SETTING, 1);
	SharedThreadState sts;
	sts.init_threads();

	int msgs_to_add = 24; // a queue of size 1kB fundamentally cannot fit 24 matrices.
	for (int i = 0; i < msgs_to_add; i++) {
		sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC1_TRANSFORM);
	}
	sts.writer_threadwork.main_start_work();
	// If we call main_wait_for_done, we will deadlock. So instead...
	sts.message_count_to_read = 1;
	sts.reader_threadwork.main_start_work();
	sts.reader_threadwork.main_wait_for_done();
	CHECK_MESSAGE(sts.func1_count == 1,
			"Reader should have read one message");
	CHECK_MESSAGE(sts.during_writing,
			"Writer thread should still be blocked on writing.");
	sts.message_count_to_read = msgs_to_add - 3;
	sts.reader_threadwork.main_start_work();
	sts.reader_threadwork.main_wait_for_done();
	CHECK_MESSAGE(sts.func1_count >= msgs_to_add - 3,
			"Reader should have read most messages");
	sts.writer_threadwork.main_wait_for_done();
	CHECK_MESSAGE(sts.during_writing == false,
			"Writer thread should no longer be blocked on writing.");
	sts.message_count_to_read = 2;
	sts.reader_threadwork.main_start_work();
	sts.reader_threadwork.main_wait_for_done();
	sts.message_count_to_read = -1;
	sts.reader_threadwork.main_start_work();
	sts.reader_threadwork.main_wait_for_done();
	CHECK_MESSAGE(sts.func1_count == msgs_to_add,
			"Reader should have read all messages");

	sts.destroy_threads();

	CHECK_MESSAGE(sts.func1_count == msgs_to_add,
			"Reader should have read no additional messages after join");
	ProjectSettings::get_singleton()->set_setting(COMMAND_QUEUE_SETTING,
			ProjectSettings::get_singleton()->property_get_revert(COMMAND_QUEUE_SETTING));
}

TEST_CASE("[CommandQueue] Test Queue Wrapping to same spot.") {
	const char *COMMAND_QUEUE_SETTING = "memory/limits/command_queue/multithreading_queue_size_kb";
	ProjectSettings::get_singleton()->set_setting(COMMAND_QUEUE_SETTING, 1);
	SharedThreadState sts;
	sts.init_threads();

	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC3_TRANSFORMx6);
	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC3_TRANSFORMx6);
	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC1_TRANSFORM);
	sts.writer_threadwork.main_start_work();
	sts.writer_threadwork.main_wait_for_done();

	sts.message_count_to_read = -1;
	sts.reader_threadwork.main_start_work();
	sts.reader_threadwork.main_wait_for_done();
	CHECK_MESSAGE(sts.func1_count == 3,
			"Reader should have read at least three messages");

	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC3_TRANSFORMx6);
	sts.writer_threadwork.main_start_work();
	sts.writer_threadwork.main_wait_for_done();
	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC1_TRANSFORM);
	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC3_TRANSFORMx6);
	sts.writer_threadwork.main_start_work();
	OS::get_singleton()->delay_usec(1000);

	sts.message_count_to_read = -1;
	sts.reader_threadwork.main_start_work();
	OS::get_singleton()->delay_usec(1000);

	sts.writer_threadwork.main_wait_for_done();
	sts.reader_threadwork.main_wait_for_done();
	CHECK_MESSAGE(sts.func1_count >= 3,
			"Reader should have read at least three messages");

	sts.message_count_to_read = 6 - sts.func1_count;
	sts.reader_threadwork.main_start_work();

	// The following will fail immediately.
	// The reason it hangs indefinitely in engine, is all subsequent calls to
	// CommandQueue.wait_and_flush_one will also fail.
	sts.reader_threadwork.main_wait_for_done();

	// Because looping around uses an extra message, easiest to consume all.
	sts.message_count_to_read = -1;
	sts.reader_threadwork.main_start_work();
	sts.reader_threadwork.main_wait_for_done();
	CHECK_MESSAGE(sts.func1_count == 6,
			"Reader should have read both message sets");

	sts.destroy_threads();

	CHECK_MESSAGE(sts.func1_count == 6,
			"Reader should have read no additional messages after join");
	ProjectSettings::get_singleton()->set_setting(COMMAND_QUEUE_SETTING,
			ProjectSettings::get_singleton()->property_get_revert(COMMAND_QUEUE_SETTING));
}

TEST_CASE("[CommandQueue] Test Queue Lapping") {
	const char *COMMAND_QUEUE_SETTING = "memory/limits/command_queue/multithreading_queue_size_kb";
	ProjectSettings::get_singleton()->set_setting(COMMAND_QUEUE_SETTING, 1);
	SharedThreadState sts;
	sts.init_threads();

	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC1_TRANSFORM);
	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC3_TRANSFORMx6);
	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC3_TRANSFORMx6);
	sts.writer_threadwork.main_start_work();
	sts.writer_threadwork.main_wait_for_done();

	// We need to read an extra message so that it triggers the dealloc logic once.
	// Otherwise, the queue will be considered full.
	sts.message_count_to_read = 3;
	sts.reader_threadwork.main_start_work();
	sts.reader_threadwork.main_wait_for_done();
	CHECK_MESSAGE(sts.func1_count == 3,
			"Reader should have read first set of messages");

	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC3_TRANSFORMx6);
	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC3_TRANSFORMx6);
	sts.writer_threadwork.main_start_work();
	// Don't wait for these, because the queue isn't big enough.
	sts.writer_threadwork.main_wait_for_done();

	sts.add_msg_to_write(SharedThreadState::TEST_MSG_FUNC2_TRANSFORM_FLOAT);
	sts.writer_threadwork.main_start_work();
	OS::get_singleton()->delay_usec(1000);

	sts.message_count_to_read = 3;
	sts.reader_threadwork.main_start_work();
	sts.reader_threadwork.main_wait_for_done();

	sts.writer_threadwork.main_wait_for_done();

	sts.message_count_to_read = -1;
	sts.reader_threadwork.main_start_work();
	sts.reader_threadwork.main_wait_for_done();

	CHECK_MESSAGE(sts.func1_count == 6,
			"Reader should have read rest of the messages after lapping writers.");

	sts.destroy_threads();

	CHECK_MESSAGE(sts.func1_count == 6,
			"Reader should have read no additional messages after join");
	ProjectSettings::get_singleton()->set_setting(COMMAND_QUEUE_SETTING,
			ProjectSettings::get_singleton()->property_get_revert(COMMAND_QUEUE_SETTING));
}

TEST_CASE("[Stress][CommandQueue] Stress test command queue") {
	const char *COMMAND_QUEUE_SETTING = "memory/limits/command_queue/multithreading_queue_size_kb";
	ProjectSettings::get_singleton()->set_setting(COMMAND_QUEUE_SETTING, 1);
	SharedThreadState sts;
	sts.init_threads();

	RandomNumberGenerator rng;

	rng.set_seed(1837267);

	int msgs_to_add = 2048;

	for (int i = 0; i < msgs_to_add; i++) {
		// randi_range is inclusive, so allow any enum value except MAX.
		sts.add_msg_to_write((SharedThreadState::TestMsgType)rng.randi_range(0, SharedThreadState::TEST_MSG_MAX - 1));
	}
	sts.writer_threadwork.main_start_work();

	int max_loop_iters = msgs_to_add * 2;
	int loop_iters = 0;
	while (sts.func1_count < msgs_to_add && loop_iters < max_loop_iters) {
		int remaining = (msgs_to_add - sts.func1_count);
		sts.message_count_to_read = rng.randi_range(1, remaining < 128 ? remaining : 128);
		if (loop_iters % 3 == 0) {
			sts.message_count_to_read = -1;
		}
		sts.reader_threadwork.main_start_work();
		sts.reader_threadwork.main_wait_for_done();
		loop_iters++;
	}
	CHECK_MESSAGE(loop_iters < max_loop_iters,
			"Reader needed too many iterations to read messages!");
	sts.writer_threadwork.main_wait_for_done();

	sts.destroy_threads();

	CHECK_MESSAGE(sts.func1_count == msgs_to_add,
			"Reader should have read no additional messages after join");
	ProjectSettings::get_singleton()->set_setting(COMMAND_QUEUE_SETTING,
			ProjectSettings::get_singleton()->property_get_revert(COMMAND_QUEUE_SETTING));
}

} // namespace TestCommandQueue

#endif // !defined(NO_THREADS)

#endif // TEST_COMMAND_QUEUE_H
