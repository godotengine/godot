/**
 * test_for_each.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_FOR_EACH_H
#define TEST_FOR_EACH_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_for_each.h"

namespace TestForEach {

TEST_CASE("[Modules][LimboAI] BTForEach") {
	Ref<BTForEach> fe = memnew(BTForEach);
	Node *dummy = memnew(Node);
	Ref<Blackboard> blackboard = memnew(Blackboard);
	fe->initialize(dummy, blackboard, dummy);

	Array arr;
	arr.append("apple");
	arr.append("raspberry");
	arr.append("mushroom");
	blackboard->set_var("array", arr);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		CHECK(fe->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	Ref<BTTestAction> task = memnew(BTTestAction(BTTask::SUCCESS));
	fe->add_child(task);
	fe->set_array_var("array");
	fe->set_save_var("element");

	SUBCASE("When child returns SUCCESS") {
		CHECK(fe->execute(0.01666) == BTTask::RUNNING);
		CHECK(task->get_status() == BTTask::SUCCESS);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
		CHECK(blackboard->get_var("element", "wetgoop") == "apple");

		CHECK(fe->execute(0.01666) == BTTask::RUNNING);
		CHECK(task->get_status() == BTTask::SUCCESS);
		CHECK_ENTRIES_TICKS_EXITS(task, 2, 2, 2);
		CHECK(blackboard->get_var("element", "wetgoop") == "raspberry");

		CHECK(fe->execute(0.01666) == BTTask::SUCCESS); // * finished iterating - returning SUCCESS
		CHECK(task->get_status() == BTTask::SUCCESS);
		CHECK_ENTRIES_TICKS_EXITS(task, 3, 3, 3);
		CHECK(blackboard->get_var("element", "wetgoop") == "mushroom");
	}

	SUBCASE("When child task takes more than one tick to finish") {
		task->ret_status = BTTask::RUNNING;
		CHECK(fe->execute(0.01666) == BTTask::RUNNING);
		CHECK(task->get_status() == BTTask::RUNNING);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 0);
		CHECK(blackboard->get_var("element", "wetgoop") == "apple");

		task->ret_status = BTTask::SUCCESS;
		CHECK(fe->execute(0.01666) == BTTask::RUNNING);
		CHECK(task->get_status() == BTTask::SUCCESS);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 2, 1);
		CHECK(blackboard->get_var("element", "wetgoop") == "apple");

		task->ret_status = BTTask::RUNNING;
		CHECK(fe->execute(0.01666) == BTTask::RUNNING);
		CHECK(task->get_status() == BTTask::RUNNING);
		CHECK_ENTRIES_TICKS_EXITS(task, 2, 3, 1);
		CHECK(blackboard->get_var("element", "wetgoop") == "raspberry");

		task->ret_status = BTTask::SUCCESS;
		CHECK(fe->execute(0.01666) == BTTask::RUNNING);
		CHECK(task->get_status() == BTTask::SUCCESS);
		CHECK_ENTRIES_TICKS_EXITS(task, 2, 4, 2);
		CHECK(blackboard->get_var("element", "wetgoop") == "raspberry");

		task->ret_status = BTTask::RUNNING;
		CHECK(fe->execute(0.01666) == BTTask::RUNNING);
		CHECK(task->get_status() == BTTask::RUNNING);
		CHECK_ENTRIES_TICKS_EXITS(task, 3, 5, 2);
		CHECK(blackboard->get_var("element", "wetgoop") == "mushroom");

		task->ret_status = BTTask::SUCCESS;
		CHECK(fe->execute(0.01666) == BTTask::SUCCESS);
		CHECK(task->get_status() == BTTask::SUCCESS);
		CHECK_ENTRIES_TICKS_EXITS(task, 3, 6, 3);
		CHECK(blackboard->get_var("element", "wetgoop") == "mushroom");
	}
}

} //namespace TestForEach

#endif // TEST_FOR_EACH_H
