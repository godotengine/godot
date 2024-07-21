/**
 * test_check_trigger.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_CHECK_TRIGGER_H
#define TEST_CHECK_TRIGGER_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/blackboard/bt_check_trigger.h"
#include "modules/limboai/bt/tasks/bt_task.h"

namespace TestCheckTrigger {

TEST_CASE("[Modules][LimboAI] BTCheckTrigger") {
	Ref<BTCheckTrigger> ct = memnew(BTCheckTrigger);
	Node *dummy = memnew(Node);
	Ref<Blackboard> bb = memnew(Blackboard);

	ct->initialize(dummy, bb, dummy);

	SUBCASE("Empty") {
		ERR_PRINT_OFF;
		ct->set_variable("");
		CHECK(ct->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	ct->set_variable("trigger");

	SUBCASE("When variable is not found") {
		ERR_PRINT_OFF;
		CHECK(ct->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}
	SUBCASE("When variable set to false") {
		bb->set_var("trigger", false);
		CHECK(ct->execute(0.01666) == BTTask::FAILURE);
		CHECK(bb->get_var("trigger", false) == Variant(false));
	}
	SUBCASE("When variable set to true") {
		bb->set_var("trigger", true);
		CHECK(bb->get_var("trigger", false) == Variant(true));
		CHECK(ct->execute(0.01666) == BTTask::SUCCESS);
		CHECK(bb->get_var("trigger", false) == Variant(false));
	}
	SUBCASE("When variable set to non-bool") {
		bb->set_var("trigger", "Some text");
		CHECK(ct->execute(0.01666) == BTTask::FAILURE);
		CHECK(bb->get_var("trigger", Variant()) == "Some text");
	}

	memdelete(dummy);
}

} //namespace TestCheckTrigger

#endif // TEST_CHECK_TRIGGER_H
