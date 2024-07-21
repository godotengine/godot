/**
 * test_task.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_TASK_H
#define TEST_TASK_H

#include "limbo_test.h"

#include "modules/limboai/blackboard/blackboard.h"
#include "modules/limboai/bt/tasks/bt_task.h"
#include "tests/test_macros.h"

namespace TestTask {

TEST_CASE("[Modules][LimboAI] BTTask") {
	SUBCASE("Test with hierarchy") {
		Ref<BTTask> task = memnew(BTTask);
		Ref<BTTask> child1 = memnew(BTTask);
		Ref<BTTask> child3 = memnew(BTTask);

		// * add_child, get_child_count & get_child
		REQUIRE(task->get_child_count() == 0);
		task->add_child(child1);
		task->add_child(child3);
		REQUIRE(task->get_child_count() == 2);
		REQUIRE(task->get_child(0) == child1);
		REQUIRE(task->get_child(1) == child3);
		CHECK(child1->get_index() == 0);
		CHECK(child3->get_index() == 1);

		// * add_child_at_index
		Ref<BTTask> child2 = memnew(BTTask);
		task->add_child_at_index(child2, 1);
		REQUIRE(task->get_child_count() == 3);
		REQUIRE(task->get_child(0) == child1);
		REQUIRE(task->get_child(1) == child2);
		REQUIRE(task->get_child(2) == child3);

		/** Hierarchy:
		 *      task->
		 *          -> child1 (0)
		 *          -> child2 (1)
		 *          -> child3 (2)
		 */

		SUBCASE("Test has_child()") {
			CHECK(task->has_child(child1));
			CHECK(task->has_child(child2));
			CHECK(task->has_child(child3));

			Ref<BTTask> other = memnew(BTTask);
			CHECK_FALSE(task->has_child(other));
		}
		SUBCASE("Test get_index()") {
			CHECK(child1->get_index() == 0);
			CHECK(child2->get_index() == 1);
			CHECK(child3->get_index() == 2);
		}
		SUBCASE("Test get_index() with an out-of-hierarchy task") {
			Ref<BTTask> other = memnew(BTTask);
			CHECK(other->get_index() == -1);
		}
		SUBCASE("Test is_descendant_of()") {
			Ref<BTTask> grandchild = memnew(BTTask);
			child1->add_child(grandchild);
			CHECK(child1->has_child(grandchild));
			CHECK(child1->get_child_count() == 1);
			CHECK(grandchild->is_descendant_of(task));
		}
		SUBCASE("Test next_sibling()") {
			CHECK(child1->next_sibling() == child2);
			CHECK(child2->next_sibling() == child3);
			CHECK(child3->next_sibling() == nullptr);
		}
		SUBCASE("Test remove_child()") {
			task->remove_child(child2);
			REQUIRE(task->get_child_count() == 2);
			CHECK(task->get_child(0) == child1);
			CHECK(task->get_child(1) == child3);
			CHECK(child1->get_index() == 0);
			CHECK(child2->get_index() == -1);
			CHECK(child3->get_index() == 1);

			task->remove_child(child3);
			REQUIRE(task->get_child_count() == 1);
			CHECK(task->get_child(0) == child1);
			CHECK(child1->get_index() == 0);
			CHECK(child2->get_index() == -1);
			CHECK(child3->get_index() == -1);

			task->remove_child(child1);
			REQUIRE(task->get_child_count() == 0);
			CHECK(child1->get_index() == -1);
			CHECK(child2->get_index() == -1);
			CHECK(child3->get_index() == -1);
		}
		SUBCASE("Test remove_child() with an out-of-hierarchy task") {
			Ref<BTTask> other = memnew(BTTask);
			// * Must not crash.
			ERR_PRINT_OFF;
			task->remove_child(other);
			ERR_PRINT_ON;
		}
		SUBCASE("Test remove_child_at_index()") {
			task->remove_child_at_index(1);
			REQUIRE(task->get_child_count() == 2);
			CHECK(task->get_child(0) == child1);
			CHECK(task->get_child(1) == child3);
			CHECK(child1->get_index() == 0);
			CHECK(child2->get_index() == -1);
			CHECK(child3->get_index() == 1);

			task->remove_child_at_index(1);
			REQUIRE(task->get_child_count() == 1);
			CHECK(task->get_child(0) == child1);
			CHECK(child1->get_index() == 0);
			CHECK(child2->get_index() == -1);
			CHECK(child3->get_index() == -1);

			task->remove_child_at_index(0);
			REQUIRE(task->get_child_count() == 0);
			CHECK(child1->get_index() == -1);
			CHECK(child2->get_index() == -1);
			CHECK(child3->get_index() == -1);
		}
		SUBCASE("Test remove_child_at_index() with an out-of-bounds index") {
			// * Must not crash.
			ERR_PRINT_OFF;
			task->remove_child_at_index(-1);
			CHECK(task->get_child_count() == 3);
			task->remove_child_at_index(task->get_child_count());
			CHECK(task->get_child_count() == 3);
			ERR_PRINT_ON;
		}
		SUBCASE("Test is_root()") {
			CHECK(task->is_root());
			CHECK_FALSE(child1->is_root());
			CHECK_FALSE(child2->is_root());
			CHECK_FALSE(child3->is_root());
		}
		SUBCASE("Test get_root()") {
			CHECK(task->get_root() == task);
			CHECK(child1->get_root() == task);
			CHECK(child2->get_root() == task);
			CHECK(child3->get_root() == task);
		}
		SUBCASE("Test get_parent()") {
			CHECK(task->get_parent() == nullptr);
			CHECK(child1->get_parent() == task);
			CHECK(child2->get_parent() == task);
			CHECK(child2->get_parent() == task);
		}
		SUBCASE("Test initialize()") {
			Node *dummy = memnew(Node);
			Ref<Blackboard> bb = memnew(Blackboard);
			SUBCASE("With valid parameters") {
				task->initialize(dummy, bb, dummy);
				CHECK(task->get_agent() == dummy);
				CHECK(task->get_blackboard() == bb);
				CHECK(child1->get_agent() == dummy);
				CHECK(child1->get_blackboard() == bb);
				CHECK(child2->get_agent() == dummy);
				CHECK(child2->get_blackboard() == bb);
				CHECK(child3->get_agent() == dummy);
				CHECK(child3->get_blackboard() == bb);
			}
			SUBCASE("Test if not crashes when agent is null") {
				ERR_PRINT_OFF;
				task->initialize(nullptr, bb, dummy);
				ERR_PRINT_ON;
			}
			SUBCASE("Test if not crashes when scene_owner is null") {
				ERR_PRINT_OFF;
				task->initialize(dummy, bb, nullptr);
				ERR_PRINT_ON;
			}
			SUBCASE("Test if not crashes when BB is null") {
				ERR_PRINT_OFF;
				task->initialize(dummy, nullptr, dummy);
				ERR_PRINT_ON;
			}
			memdelete(dummy);
		}
	}

	SUBCASE("Test get_elapsed_time()") {
		Ref<BTTestAction> task = memnew(BTTestAction);
		task->ret_status = BTTask::RUNNING;

		CHECK(task->get_elapsed_time() == 0.0);

		task->execute(888.0);
		CHECK(task->get_elapsed_time() == 0.0); // * delta_time shouldn't contribute to the elapsed_time on the first tick.
		task->execute(10.0);
		CHECK(task->get_elapsed_time() == 10.0);
		task->execute(10.0);
		CHECK(task->get_elapsed_time() == 20.0);

		SUBCASE("When finishing with SUCCESS or FAILURE") {
			task->ret_status = BTTask::SUCCESS;
			task->execute(10.0);
			CHECK(task->get_elapsed_time() == 0.0);
		}
		SUBCASE("When cancelled") {
			task->abort();
			CHECK(task->get_elapsed_time() == 0.0);
		}
	}

	SUBCASE("Test clone()") {
		// * Note: BTTask cannot be duplicated, thus using BTTestAction.
		Ref<BTTestAction> task = memnew(BTTestAction);
		Ref<BTTestAction> child1 = memnew(BTTestAction);
		Ref<BTTestAction> child2 = memnew(BTTestAction);

		task->add_child(child1);
		task->add_child(child2);
		REQUIRE(task->get_child_count() == 2);
		REQUIRE(task->get_child(0) == child1);
		REQUIRE(task->get_child(1) == child2);

		Ref<BTTestAction> cloned = task->clone();
		CHECK_FALSE(cloned == task);
		REQUIRE(cloned->get_child_count() == 2);
		CHECK_FALSE(cloned->get_child(0) == child1);
		CHECK_FALSE(cloned->get_child(1) == child2);
	}
}

} //namespace TestTask

#endif // TEST_TASK_H
