/**
 * test_repeat.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_REPEAT_H
#define TEST_REPEAT_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_repeat.h"

namespace TestRepeat {

TEST_CASE("[Modules][LimboAI] BTRepeat") {
	Ref<BTRepeat> rep = memnew(BTRepeat);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		CHECK(rep->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	Ref<BTTestAction> task = memnew(BTTestAction);
	rep->add_child(task);

	SUBCASE("When repeating forever") {
		rep->set_times(3);
		rep->set_forever(true);
		task->ret_status = BTTask::SUCCESS;

		for (int i = 1; i <= 100; i++) {
			CHECK(rep->execute(0.01666) == BTTask::RUNNING);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, i, i, i);
		}
	}

	SUBCASE("When repeated x3 times") {
		rep->set_times(3);
		rep->set_forever(false);
		task->ret_status = BTTask::SUCCESS;

		CHECK(rep->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 1, 1, 1);
		CHECK(rep->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 2, 2, 2);
		CHECK(rep->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 3, 3, 3);
	}

	SUBCASE("When the child task takes more than one tick to finish") {
		rep->set_times(2);
		rep->set_forever(false);

		task->ret_status = BTTask::RUNNING;
		CHECK(rep->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 1, 1, 0);

		task->ret_status = BTTask::SUCCESS;
		CHECK(rep->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 1, 2, 1);

		task->ret_status = BTTask::RUNNING;
		CHECK(rep->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 2, 3, 1);

		task->ret_status = BTTask::SUCCESS;
		CHECK(rep->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 2, 4, 2);
	}

	SUBCASE("When the child task fails") {
		rep->set_times(2);
		rep->set_forever(false);

		task->ret_status = BTTask::FAILURE;

		SUBCASE("When set to abort on failure") {
			rep->set_abort_on_failure(true);
			CHECK(rep->execute(0.01666) == BTTask::FAILURE);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FAILURE, 1, 1, 1);
			CHECK(rep->execute(0.01666) == BTTask::FAILURE);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FAILURE, 2, 2, 2);
		}

		SUBCASE("When not set to abort on failure") {
			rep->set_abort_on_failure(false);
			CHECK(rep->execute(0.01666) == BTTask::RUNNING);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FAILURE, 1, 1, 1);
			CHECK(rep->execute(0.01666) == BTTask::SUCCESS);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FAILURE, 2, 2, 2);
		}
	}
}

} //namespace TestRepeat

#endif // TEST_REPEAT_H
