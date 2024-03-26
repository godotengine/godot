/**
 * test_delay.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_DELAY_H
#define TEST_DELAY_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_delay.h"

namespace TestDelay {

TEST_CASE("[Modules][LimboAI] BTDelay") {
	Ref<BTDelay> del = memnew(BTDelay);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		CHECK(del->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	Ref<BTTestAction> task = memnew(BTTestAction(BTTask::SUCCESS));
	del->add_child(task);

	SUBCASE("Check if delay is observed correctly") {
		del->set_seconds(1.0);
		CHECK(del->execute(0.35) == BTTask::RUNNING); // * first execution: elapsed 0.0
		CHECK_ENTRIES_TICKS_EXITS(task, 0, 0, 0);
		CHECK(del->execute(0.35) == BTTask::RUNNING); // * second execution: elapsed 0.35
		CHECK_ENTRIES_TICKS_EXITS(task, 0, 0, 0);
		CHECK(del->execute(0.35) == BTTask::RUNNING); // * third execution: elapsed 0.7
		CHECK_ENTRIES_TICKS_EXITS(task, 0, 0, 0);

		SUBCASE("When child task returns SUCCESS") {
			task->ret_status = BTTask::SUCCESS;
			CHECK(del->execute(0.35) == BTTask::SUCCESS); // * fourth execution: elapsed 1.05
			CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
		}
		SUBCASE("When child task returns FAILURE") {
			task->ret_status = BTTask::FAILURE;
			CHECK(del->execute(0.35) == BTTask::FAILURE); // * fourth execution: elapsed 1.05
			CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
		}
		SUBCASE("When child task returns RUNNING") {
			task->ret_status = BTTask::RUNNING;
			CHECK(del->execute(0.35) == BTTask::RUNNING); // * fourth execution: elapsed 1.05
			CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 0);

			task->ret_status = BTTask::SUCCESS;
			CHECK(del->execute(0.35) == BTTask::SUCCESS); // * fifth execution: elapsed 1.4
			CHECK_ENTRIES_TICKS_EXITS(task, 1, 2, 1);
		}
	}
}

} //namespace TestDelay

#endif // TEST_DELAY_H
