/**
 * test_run_limit.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_RUN_LIMIT_H
#define TEST_RUN_LIMIT_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_run_limit.h"

namespace TestRunLimit {

TEST_CASE("[Modules][LimboAI] BTRunLimit") {
	Ref<BTRunLimit> lim = memnew(BTRunLimit);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		CHECK(lim->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	Ref<BTTestAction> task = memnew(BTTestAction);
	lim->add_child(task);

	SUBCASE("With run limit set to 2") {
		lim->set_run_limit(2);

		task->ret_status = BTTask::SUCCESS;
		CHECK(lim->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 1, 1, 1); // * task executed

		SUBCASE("When the child task succeeds") {
			task->ret_status = BTTask::SUCCESS;

			lim->set_count_policy(BTRunLimit::COUNT_FAILED);
			CHECK(lim->execute(0.01666) == BTTask::SUCCESS);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 2, 2, 2); // * task executed

			lim->set_count_policy(BTRunLimit::COUNT_SUCCESSFUL);
			CHECK(lim->execute(0.01666) == BTTask::SUCCESS);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 3, 3, 3); // * task executed
		}
		SUBCASE("When the child task fails") {
			task->ret_status = BTTask::FAILURE;

			lim->set_count_policy(BTRunLimit::COUNT_SUCCESSFUL);
			CHECK(lim->execute(0.01666) == BTTask::FAILURE);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FAILURE, 2, 2, 2); // * task executed

			lim->set_count_policy(BTRunLimit::COUNT_FAILED);
			CHECK(lim->execute(0.01666) == BTTask::FAILURE);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FAILURE, 3, 3, 3); // * task executed
		}

		task->ret_status = BTTask::SUCCESS;
		lim->set_count_policy(BTRunLimit::COUNT_SUCCESSFUL);

		CHECK(lim->execute(0.01666) == BTTask::FAILURE);
		CHECK_ENTRIES_TICKS_EXITS(task, 3, 3, 3); // * task not executed

		CHECK(lim->execute(0.01666) == BTTask::FAILURE);
		CHECK_ENTRIES_TICKS_EXITS(task, 3, 3, 3); // * task not executed
	}

	SUBCASE("When the child task takes more than one tick to finish") {
		lim->set_run_limit(2);
		lim->set_count_policy(BTRunLimit::COUNT_ALL);

		task->ret_status = BTTask::RUNNING;
		CHECK(lim->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 1, 1, 0);

		task->ret_status = BTTask::SUCCESS;
		CHECK(lim->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 1, 2, 1);

		task->ret_status = BTTask::RUNNING;
		CHECK(lim->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 2, 3, 1);

		task->ret_status = BTTask::SUCCESS;
		CHECK(lim->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 2, 4, 2);

		CHECK(lim->execute(0.01666) == BTTask::FAILURE);
		CHECK_ENTRIES_TICKS_EXITS(task, 2, 4, 2);
	}
}

} //namespace TestRunLimit

#endif // TEST_RUN_LIMIT_H
