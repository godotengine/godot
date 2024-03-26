/**
 * test_repeat_until_success.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_REPEAT_UNTIL_SUCCESS_H
#define TEST_REPEAT_UNTIL_SUCCESS_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_repeat_until_success.h"

namespace TestRepeatUntilSuccess {

TEST_CASE("[Modules][LimboAI] BTRepeatUntilSuccess") {
	Ref<BTRepeatUntilSuccess> rep = memnew(BTRepeatUntilSuccess);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		CHECK(rep->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	Ref<BTTestAction> task = memnew(BTTestAction);
	rep->add_child(task);

	SUBCASE("With various return statuses") {
		task->ret_status = BTTask::FAILURE;
		CHECK(rep->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FAILURE, 1, 1, 1);

		task->ret_status = BTTask::RUNNING;
		CHECK(rep->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 2, 2, 1);

		task->ret_status = BTTask::SUCCESS;
		CHECK(rep->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 2, 3, 2);
	}
}

} //namespace TestRepeatUntilSuccess

#endif // TEST_REPEAT_UNTIL_SUCCESS_H
