/**
 * test_always_succeed.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_ALWAYS_SUCCEED_H
#define TEST_ALWAYS_SUCCEED_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_always_succeed.h"

namespace TestAlwaysSucceed {

TEST_CASE("[Modules][LimboAI] BTAlwaysSucceed") {
	Ref<BTAlwaysSucceed> as = memnew(BTAlwaysSucceed);

	SUBCASE("When empty") {
		CHECK(as->execute(0.01666) == BTTask::SUCCESS);
	}

	Ref<BTTestAction> task = memnew(BTTestAction);

	as->add_child(task);

	SUBCASE("When child returns FAILURE") {
		task->ret_status = BTTask::FAILURE;

		CHECK(as->execute(0.01666) == BTTask::SUCCESS);

		CHECK(task->get_status() == BTTask::FAILURE);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
	}

	SUBCASE("When child returns SUCCESS") {
		task->ret_status = BTTask::SUCCESS;

		CHECK(as->execute(0.01666) == BTTask::SUCCESS);

		CHECK(task->get_status() == BTTask::SUCCESS);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
	}

	SUBCASE("When child returns RUNNING") {
		task->ret_status = BTTask::RUNNING;

		CHECK(as->execute(0.01666) == BTTask::RUNNING);

		CHECK(task->get_status() == BTTask::RUNNING);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 0);
	}
}

} //namespace TestAlwaysSucceed

#endif // TEST_ALWAYS_SUCCEED_H
