/**
 * test_probability_selector.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_PROBABILITY_SELECTOR_H
#define TEST_PROBABILITY_SELECTOR_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/composites/bt_probability_selector.h"

namespace TestProbabilitySelector {

TEST_CASE("[Modules][LimboAI] BTProbabilitySelector") {
	Ref<BTProbabilitySelector> sel = memnew(BTProbabilitySelector);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		CHECK(sel->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	Ref<BTTestAction> task1 = memnew(BTTestAction);
	Ref<BTTestAction> task2 = memnew(BTTestAction);
	Ref<BTTestAction> task3 = memnew(BTTestAction);
	sel->add_child(task1);
	sel->add_child(task2);
	sel->add_child(task3);

	Math::randomize();

	SUBCASE("With zero weight") {
		sel->set_weight(0, 0.0);
		sel->set_weight(1, 0.0);
		sel->set_weight(2, 0.0);

		CHECK(sel->execute(0.01666) == BTTask::FAILURE);

		for (int i = 0; i < 100; i++) {
			sel->execute(0.01666);
		}

		CHECK_STATUS_ENTRIES_TICKS_EXITS(task1, BTTask::FRESH, 0, 0, 0);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task2, BTTask::FRESH, 0, 0, 0);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task3, BTTask::FRESH, 0, 0, 0);
	}
	SUBCASE("When a child task returns SUCCESS") {
		sel->set_weight(0, 1.0);
		sel->set_weight(1, 0.0);
		sel->set_weight(2, 0.0);
		task1->ret_status = BTTask::SUCCESS;

		CHECK(sel->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task1, BTTask::SUCCESS, 1, 1, 1);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task2, BTTask::FRESH, 0, 0, 0);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task3, BTTask::FRESH, 0, 0, 0);
		CHECK(sel->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task1, BTTask::SUCCESS, 2, 2, 2);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task2, BTTask::FRESH, 0, 0, 0);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task3, BTTask::FRESH, 0, 0, 0);
	}
	SUBCASE("With a RUNNING status and a low-weight remaining child") {
		sel->set_weight(0, 0.0);
		sel->set_weight(1, 1.0);
		sel->set_weight(2, 0.0);
		task1->ret_status = BTTask::FAILURE;
		task2->ret_status = BTTask::RUNNING;
		task3->ret_status = BTTask::FAILURE;

		CHECK(sel->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task1, BTTask::FRESH, 0, 0, 0); // * ignored
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task2, BTTask::RUNNING, 1, 1, 0); // * running
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task3, BTTask::FRESH, 0, 0, 0); // * ignored

		CHECK(sel->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task1, BTTask::FRESH, 0, 0, 0);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task2, BTTask::RUNNING, 1, 2, 0); // * continued
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task3, BTTask::FRESH, 0, 0, 0);

		task2->ret_status = BTTask::FAILURE;
		task1->ret_status = BTTask::SUCCESS;
		sel->set_weight(0, 0.000000000001); // * extremely low weight, however, when it is the only child to evaluate, it should have 100% probability of being chosen.
		CHECK(sel->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task1, BTTask::SUCCESS, 1, 1, 1); // * started & succeeded (2)
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task2, BTTask::FAILURE, 1, 3, 1); // * continued & failed (1)
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task3, BTTask::FRESH, 0, 0, 0); // * ignored
	}
	SUBCASE("When all return SUCCESS status") {
		task1->ret_status = BTTask::SUCCESS;
		task2->ret_status = BTTask::SUCCESS;
		task3->ret_status = BTTask::SUCCESS;

		CHECK(sel->execute(0.01666) == BTTask::SUCCESS);
		CHECK(sel->execute(0.01666) == BTTask::SUCCESS);
		CHECK(sel->execute(0.01666) == BTTask::SUCCESS);

		int num_ticks = task1->num_ticks + task2->num_ticks + task3->num_ticks;
		CHECK(num_ticks == 3);

		int num_entries = task1->num_entries + task2->num_entries + task3->num_entries;
		CHECK(num_entries == 3);

		int num_exits = task1->num_exits + task2->num_exits + task3->num_exits;
		CHECK(num_exits == 3);

		CHECK(task1->is_status_either(BTTask::SUCCESS, BTTask::FRESH));
		CHECK(task2->is_status_either(BTTask::SUCCESS, BTTask::FRESH));
		CHECK(task3->is_status_either(BTTask::SUCCESS, BTTask::FRESH));
	}
	SUBCASE("With balanced weights") {
		task1->ret_status = BTTask::SUCCESS;
		task2->ret_status = BTTask::SUCCESS;
		task3->ret_status = BTTask::SUCCESS;

		int sample_size = 1000;
		sel->set_weight(0, 1.0);
		sel->set_weight(1, 1.0);
		sel->set_weight(2, 1.0);

		for (int i = 0; i < sample_size; i++) {
			sel->execute(0.01666);
		}

		CHECK(task1->num_ticks > 300);
		CHECK(task1->num_ticks < 366);
		CHECK(task2->num_ticks > 300);
		CHECK(task2->num_ticks < 366);
		CHECK(task3->num_ticks > 300);
		CHECK(task3->num_ticks < 366);
	}
	SUBCASE("With imbalanced weights") {
		task1->ret_status = BTTask::SUCCESS;
		task2->ret_status = BTTask::SUCCESS;
		task3->ret_status = BTTask::SUCCESS;

		int sample_size = 10000;
		sel->set_weight(0, 1.0); // * ~1250
		sel->set_weight(1, 2.0); // * ~2500
		sel->set_weight(2, 5.0); // * ~6250

		for (int i = 0; i < sample_size; i++) {
			sel->execute(0.01666);
		}

		CHECK(task1->num_ticks > 1150);
		CHECK(task1->num_ticks < 1350);
		CHECK(task2->num_ticks > 2250);
		CHECK(task2->num_ticks < 2750);
		CHECK(task3->num_ticks > 5750);
		CHECK(task3->num_ticks < 6750);
	}
	SUBCASE("Test abort_on_failure") {
		task1->ret_status = BTTask::FAILURE;
		task2->ret_status = BTTask::FAILURE;
		task3->ret_status = BTTask::FAILURE;

		int expected_child_executions = 0;

		SUBCASE("When abort_on_failure == false") {
			sel->set_abort_on_failure(false);
			expected_child_executions = 3;
		}
		SUBCASE("When abort_on_failure == true") {
			sel->set_abort_on_failure(true);
			expected_child_executions = 1;
		}

		sel->execute(0.01666);
		int num_ticks = task1->num_ticks + task2->num_ticks + task3->num_ticks;
		CHECK(num_ticks == expected_child_executions);
		int num_entries = task1->num_entries + task2->num_entries + task3->num_entries;
		CHECK(num_entries == expected_child_executions);
		int num_exits = task1->num_exits + task2->num_exits + task3->num_exits;
		CHECK(num_exits == expected_child_executions);
	}
}

} //namespace TestProbabilitySelector

#endif // TEST_PROBABILITY_SELECTOR_H
