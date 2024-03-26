/**
 * test_always_fail.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_ALWAYS_FAIL_H
#define TEST_ALWAYS_FAIL_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_always_fail.h"

namespace TestAlwaysFail {

TEST_CASE("[Modules][LimboAI] BTAlwaysFail") {
	Ref<BTAlwaysFail> af = memnew(BTAlwaysFail);

	SUBCASE("When empty") {
		CHECK(af->execute(0.01666) == BTTask::FAILURE);
	}

	Ref<BTTestAction> task = memnew(BTTestAction);

	af->add_child(task);

	SUBCASE("When child returns FAILURE") {
		task->ret_status = BTTask::FAILURE;

		CHECK(af->execute(0.01666) == BTTask::FAILURE);

		CHECK(task->get_status() == BTTask::FAILURE);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
	}

	SUBCASE("When child returns SUCCESS") {
		task->ret_status = BTTask::SUCCESS;

		CHECK(af->execute(0.01666) == BTTask::FAILURE);

		CHECK(task->get_status() == BTTask::SUCCESS);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
	}

	SUBCASE("When child returns RUNNING") {
		task->ret_status = BTTask::RUNNING;

		CHECK(af->execute(0.01666) == BTTask::RUNNING);

		CHECK(task->get_status() == BTTask::RUNNING);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 0);
	}
}

} //namespace TestAlwaysFail

#endif // TEST_ALWAYS_FAIL_H
