/**
 * test_subtree.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_SUBTREE_H
#define TEST_SUBTREE_H

#include "limbo_test.h"

#include "modules/limboai/bt/behavior_tree.h"
#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_subtree.h"

namespace TestSubtree {

TEST_CASE("[Modules][LimboAI] BTSubtree") {
	ClassDB::register_class<BTTestAction>();

	Ref<BTSubtree> st = memnew(BTSubtree);
	Ref<Blackboard> bb = memnew(Blackboard);
	Node *dummy = memnew(Node);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		st->initialize(dummy, bb, dummy);
		CHECK(st->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	SUBCASE("With a subtree assigned") {
		Ref<BehaviorTree> bt = memnew(BehaviorTree);
		Ref<BTTestAction> task = memnew(BTTestAction(BTTask::SUCCESS));
		bt->set_root_task(task);
		st->set_subtree(bt);

		CHECK(st->get_child_count() == 0);
		st->initialize(dummy, bb, dummy);
		CHECK(st->get_child_count() == 1);
		CHECK(st->get_child(0) != task);

		Ref<BTTestAction> ta = st->get_child(0);
		REQUIRE(ta.is_valid());

		SUBCASE("When child succeeds") {
			ta->ret_status = BTTask::SUCCESS;
			CHECK(st->execute(0.01666) == BTTask::SUCCESS);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(ta, BTTask::SUCCESS, 1, 1, 1);
		}
		SUBCASE("When child fails") {
			ta->ret_status = BTTask::FAILURE;
			CHECK(st->execute(0.01666) == BTTask::FAILURE);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(ta, BTTask::FAILURE, 1, 1, 1);
		}
		SUBCASE("When child is running") {
			ta->ret_status = BTTask::RUNNING;
			CHECK(st->execute(0.01666) == BTTask::RUNNING);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(ta, BTTask::RUNNING, 1, 1, 0);
		}
	}

	memdelete(dummy);
}

} //namespace TestSubtree

#endif // TEST_SUBTREE_H
