/**
 * test_time_limit.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_TIME_LIMIT_H
#define TEST_TIME_LIMIT_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_time_limit.h"

namespace TestTimeLimit {

TEST_CASE("[Modules][LimboAI] BTTimeLimit") {
	Ref<BTTimeLimit> lim = memnew(BTTimeLimit);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		CHECK(lim->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	Ref<BTTestAction> task = memnew(BTTestAction);
	lim->add_child(task);
	lim->set_time_limit(1.0);

	SUBCASE("With a long-running task") {
		task->ret_status = BTTask::RUNNING;

		CHECK(lim->execute(0.0) == BTTask::RUNNING); // * elapsed 0.0
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 1, 1, 0); // * running

		CHECK(lim->execute(0.4) == BTTask::RUNNING); // * elapsed 0.4
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 1, 2, 0); // * running

		CHECK(lim->execute(0.4) == BTTask::RUNNING); // * elapsed 0.8
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 1, 3, 0); // * running

		SUBCASE("When exceeding the time limit") {
			CHECK(lim->execute(0.4) == BTTask::FAILURE); // * elapsed 1.2
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FRESH, 1, 4, 1); // * cancelled & exited
		}
		SUBCASE("When finishing on time") {
			task->ret_status = BTTask::SUCCESS;
			CHECK(lim->execute(0.1) == BTTask::SUCCESS); // * elapsed 0.9
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 1, 4, 1); // * succeeded & exited
		}
	}

	SUBCASE("With a quick task") {
		task->ret_status = BTTask::SUCCESS;
		CHECK(lim->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 1, 1, 1); // * succeeded

		task->ret_status = BTTask::FAILURE;
		CHECK(lim->execute(0.01666) == BTTask::FAILURE);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FAILURE, 2, 2, 2); // * failed
	}

	SUBCASE("If time limit is reset") {
		task->ret_status = BTTask::RUNNING;

		CHECK(lim->execute(0.0) == BTTask::RUNNING); // * elapsed 0.0
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 1, 1, 0); // * running
		CHECK(lim->execute(1.1) == BTTask::FAILURE); // * elapsed 1.1
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FRESH, 1, 2, 1); // * cancelled due to time limit exceeded

		CHECK(lim->execute(0.0) == BTTask::RUNNING); // * elapsed 0.0
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 2, 3, 1); // * running
		CHECK(lim->execute(0.8) == BTTask::RUNNING); // * elapsed 0.8
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 2, 4, 1); // * running

		task->ret_status = BTTask::SUCCESS;
		CHECK(lim->execute(0.8) == BTTask::SUCCESS); // * elapsed 1.6
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 2, 5, 2); // * succeeded, despite time limit exceeded
	}
}

} //namespace TestTimeLimit

#endif // TEST_TIME_LIMIT_H
