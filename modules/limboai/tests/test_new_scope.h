/**
 * test_new_scope.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_NEW_SCOPE_H
#define TEST_NEW_SCOPE_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_new_scope.h"

namespace TestNewScope {

TEST_CASE("[Modules][LimboAI] BTNewScope") {
	Ref<BTNewScope> ns = memnew(BTNewScope);
	Node *dummy = memnew(Node);
	Ref<Blackboard> parent_bb = memnew(Blackboard);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		ns->initialize(dummy, parent_bb, dummy);
		CHECK(ns->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	SUBCASE("When not empty") {
		Ref<BTTask> parent = memnew(BTTask);
		parent->add_child(ns);

		Ref<BTTestAction> child = memnew(BTTestAction);
		ns->add_child(child);

		parent_bb->set_var("fruit", "apple");
		parent_bb->set_var("vegetable", "carrot");
		REQUIRE(parent_bb->has_var("fruit"));
		REQUIRE(parent_bb->get_var("fruit", "wetgoop") == "apple");
		REQUIRE(parent_bb->has_var("vegetable"));
		REQUIRE(parent_bb->get_var("vegetable", "wetgoop") == "carrot");

		parent->initialize(dummy, parent_bb, dummy);

		CHECK(ns->get_blackboard() != parent->get_blackboard());
		CHECK(ns->get_blackboard() == child->get_blackboard());
		CHECK(parent->get_blackboard() == parent_bb);
		CHECK(ns->get_blackboard()->get_parent() == parent_bb);

		ns->get_blackboard()->set_var("fruit", "pear"); // * override "fruit"

		CHECK(ns->get_blackboard()->get_var("fruit", "wetgoop") == "pear");
		CHECK(child->get_blackboard()->get_var("fruit", "wetgoop") == "pear");
		CHECK(parent->get_blackboard()->get_var("fruit", "wetgoop") == "apple");

		// * Check if new scope inherits "vegetable"
		CHECK(ns->get_blackboard()->has_var("vegetable"));
		CHECK(ns->get_blackboard()->get_var("vegetable", "wetgoop") == "carrot");
		CHECK(child->get_blackboard()->get_var("vegetable", "wetgoop") == "carrot");

		// * Check if "vegetable" from the parent scope is accessible
		CHECK(ns->get_blackboard()->has_var("vegetable"));
		CHECK(child->get_blackboard()->has_var("vegetable"));
		CHECK(ns->get_blackboard()->get_var("vegetable", "wetgoop") == "carrot");
		CHECK(child->get_blackboard()->get_var("vegetable", "wetgoop") == "carrot");

		// * Check if setting a variable doesn't propagate it up the scope
		ns->get_blackboard()->set_var("berry", "raspberry");
		CHECK(ns->get_blackboard()->get_var("berry", "wetgoop") == "raspberry");
		CHECK(child->get_blackboard()->get_var("berry", "wetgoop") == "raspberry");
		CHECK(parent->get_blackboard()->get_var("berry", "wetgoop", false) == "wetgoop");
		CHECK_FALSE(parent->get_blackboard()->has_var("berry"));

		// * Check if setting a variable doesn't propagate it up the scope (now with the child task)
		child->get_blackboard()->set_var("seed", "sunflower");
		CHECK(child->get_blackboard()->get_var("seed", "wetgoop") == "sunflower");
		CHECK(ns->get_blackboard()->get_var("seed", "wetgoop") == "sunflower");
		CHECK(parent->get_blackboard()->get_var("seed", "wetgoop", false) == "wetgoop");
		CHECK_FALSE(parent->get_blackboard()->has_var("seed"));

		// * Check return status
		child->ret_status = BTTask::SUCCESS;
		CHECK(ns->execute(0.01666) == BTTask::SUCCESS);
		child->ret_status = BTTask::FAILURE;
		CHECK(ns->execute(0.01666) == BTTask::FAILURE);
		child->ret_status = BTTask::RUNNING;
		CHECK(ns->execute(0.01666) == BTTask::RUNNING);
	}

	memdelete(dummy);
}

} //namespace TestNewScope

#endif // TEST_NEW_SCOPE_H
