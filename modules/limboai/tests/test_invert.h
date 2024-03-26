/**
 * test_invert.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_INVERT_H
#define TEST_INVERT_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_invert.h"

namespace TestInvert {

TEST_CASE("[Modules][LimboAI] BTInvert") {
	Ref<BTInvert> inv = memnew(BTInvert);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		CHECK(inv->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	Ref<BTTestAction> task = memnew(BTTestAction);
	inv->add_child(task);

	SUBCASE("With SUCCESS") {
		task->ret_status = BTTask::SUCCESS;
		CHECK(inv->execute(0.01666) == BTTask::FAILURE);
		CHECK(task->get_status() == BTTask::SUCCESS);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
	}

	SUBCASE("With FAILURE") {
		task->ret_status = BTTask::FAILURE;
		CHECK(inv->execute(0.01666) == BTTask::SUCCESS);
		CHECK(task->get_status() == BTTask::FAILURE);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
	}

	SUBCASE("With RUNNING followed by SUCCESS") {
		task->ret_status = BTTask::RUNNING;
		CHECK(inv->execute(0.01666) == BTTask::RUNNING);
		CHECK(task->get_status() == BTTask::RUNNING);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 0);

		task->ret_status = BTTask::SUCCESS;
		CHECK(inv->execute(0.01666) == BTTask::FAILURE);
		CHECK(task->get_status() == BTTask::SUCCESS);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 2, 1);
	}
}

} //namespace TestInvert

#endif // TEST_INVERT_H
