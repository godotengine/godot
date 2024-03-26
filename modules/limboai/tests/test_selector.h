/**
 * test_selector.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_SELECTOR_H
#define TEST_SELECTOR_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/composites/bt_selector.h"

namespace TestSelector {

TEST_CASE("[Modules][LimboAI] BTSelector when all return FAILURE") {
	Ref<BTSelector> sel = memnew(BTSelector);
	Ref<BTTestAction> task1 = memnew(BTTestAction(BTTask::FAILURE));
	Ref<BTTestAction> task2 = memnew(BTTestAction(BTTask::FAILURE));
	Ref<BTTestAction> task3 = memnew(BTTestAction(BTTask::FAILURE));

	sel->add_child(task1);
	sel->add_child(task2);
	sel->add_child(task3);

	REQUIRE(sel->get_child_count() == 3);

	// * First execution.
	CHECK(sel->execute(0.01666) == BTTask::FAILURE);

	CHECK(task1->get_status() == BTTask::FAILURE);
	CHECK(task2->get_status() == BTTask::FAILURE);
	CHECK(task3->get_status() == BTTask::FAILURE);

	CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 1);
	CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 1);
	CHECK_ENTRIES_TICKS_EXITS(task3, 1, 1, 1);

	// * Second execution.
	CHECK(sel->execute(0.01666) == BTTask::FAILURE);

	CHECK(task1->get_status() == BTTask::FAILURE);
	CHECK(task2->get_status() == BTTask::FAILURE);
	CHECK(task3->get_status() == BTTask::FAILURE);

	CHECK_ENTRIES_TICKS_EXITS(task1, 2, 2, 2);
	CHECK_ENTRIES_TICKS_EXITS(task2, 2, 2, 2);
	CHECK_ENTRIES_TICKS_EXITS(task3, 2, 2, 2);
}

TEST_CASE("[Modules][LimboAI] BTSelector when second returns SUCCESS") {
	Ref<BTSelector> sel = memnew(BTSelector);
	Ref<BTTestAction> task1 = memnew(BTTestAction(BTTask::FAILURE));
	Ref<BTTestAction> task2 = memnew(BTTestAction(BTTask::SUCCESS));
	Ref<BTTestAction> task3 = memnew(BTTestAction(BTTask::FAILURE));

	sel->add_child(task1);
	sel->add_child(task2);
	sel->add_child(task3);

	REQUIRE(sel->get_child_count() == 3);

	// * First execution.
	CHECK(sel->execute(0.01666) == BTTask::SUCCESS);

	CHECK(task1->get_status() == BTTask::FAILURE);
	CHECK(task2->get_status() == BTTask::SUCCESS);
	CHECK(task3->get_status() == BTTask::FRESH);

	CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 1);
	CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 1);
	CHECK_ENTRIES_TICKS_EXITS(task3, 0, 0, 0);

	// * Second execution.
	CHECK(sel->execute(0.01666) == BTTask::SUCCESS);

	CHECK(task1->get_status() == BTTask::FAILURE);
	CHECK(task2->get_status() == BTTask::SUCCESS);
	CHECK(task3->get_status() == BTTask::FRESH);

	CHECK_ENTRIES_TICKS_EXITS(task1, 2, 2, 2);
	CHECK_ENTRIES_TICKS_EXITS(task2, 2, 2, 2);
	CHECK_ENTRIES_TICKS_EXITS(task3, 0, 0, 0);
}

TEST_CASE("[Modules][LimboAI] BTSelector when second returns RUNNING") {
	Ref<BTSelector> sel = memnew(BTSelector);
	Ref<BTTestAction> task1 = memnew(BTTestAction(BTTask::FAILURE));
	Ref<BTTestAction> task2 = memnew(BTTestAction(BTTask::RUNNING));
	Ref<BTTestAction> task3 = memnew(BTTestAction(BTTask::FAILURE));

	sel->add_child(task1);
	sel->add_child(task2);
	sel->add_child(task3);

	REQUIRE(sel->get_child_count() == 3);

	// * First execution.
	CHECK(sel->execute(0.01666) == BTTask::RUNNING);

	CHECK(task1->get_status() == BTTask::FAILURE);
	CHECK(task2->get_status() == BTTask::RUNNING);
	CHECK(task3->get_status() == BTTask::FRESH);

	CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 1);
	CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 0);
	CHECK_ENTRIES_TICKS_EXITS(task3, 0, 0, 0);

	// * Second execution.
	CHECK(sel->execute(0.01666) == BTTask::RUNNING);

	CHECK(task1->get_status() == BTTask::FAILURE);
	CHECK(task2->get_status() == BTTask::RUNNING);
	CHECK(task3->get_status() == BTTask::FRESH);

	CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 1);
	CHECK_ENTRIES_TICKS_EXITS(task2, 1, 2, 0);
	CHECK_ENTRIES_TICKS_EXITS(task3, 0, 0, 0);

	// * Third execution with second task returning FAILURE.
	task2->ret_status = BTTask::FAILURE;
	CHECK(sel->execute(0.01666) == BTTask::FAILURE);

	CHECK(task1->get_status() == BTTask::FAILURE);
	CHECK(task2->get_status() == BTTask::FAILURE);
	CHECK(task3->get_status() == BTTask::FAILURE);

	CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 1);
	CHECK_ENTRIES_TICKS_EXITS(task2, 1, 3, 1);
	CHECK_ENTRIES_TICKS_EXITS(task3, 1, 1, 1);
}

} //namespace TestSelector

#endif // TEST_SELECTOR_H
