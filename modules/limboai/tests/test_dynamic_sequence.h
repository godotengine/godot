/**
 * test_dynamic_sequence.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_DYNAMIC_SEQUENCE_H
#define TEST_DYNAMIC_SEQUENCE_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/composites/bt_dynamic_sequence.h"

namespace TestDynamicSequence {

TEST_CASE("[Modules][LimboAI] BTDynamicSequence") {
	Ref<BTDynamicSequence> seq = memnew(BTDynamicSequence);
	Ref<BTTestAction> task1 = memnew(BTTestAction());
	Ref<BTTestAction> task2 = memnew(BTTestAction());
	Ref<BTTestAction> task3 = memnew(BTTestAction());

	seq->add_child(task1);
	seq->add_child(task2);
	seq->add_child(task3);

	REQUIRE(seq->get_child_count() == 3);

	SUBCASE("Subcase #1: Dynamic sequence processes tasks sequentially, while re-evaluating its child tasks in every execution tick.") {
		task1->ret_status = BTTask::SUCCESS;
		task2->ret_status = BTTask::RUNNING;
		task3->ret_status = BTTask::SUCCESS;

		CHECK(seq->execute(0.01666) == BTTask::RUNNING);

		CHECK(task1->get_status() == BTTask::SUCCESS);
		CHECK(task2->get_status() == BTTask::RUNNING);
		CHECK(task3->get_status() == BTTask::FRESH);

		CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 1); // * finished
		CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 0); // * running
		CHECK_ENTRIES_TICKS_EXITS(task3, 0, 0, 0); // * still fresh

		SUBCASE("Subcase 1A: With no changes, first task is re-evaluated.") {
			CHECK(seq->execute(0.01666) == BTTask::RUNNING);

			CHECK(task1->get_status() == BTTask::SUCCESS);
			CHECK(task2->get_status() == BTTask::RUNNING);
			CHECK(task3->get_status() == BTTask::FRESH);

			CHECK_ENTRIES_TICKS_EXITS(task1, 2, 2, 2); // * re-evaluated with SUCCESS
			CHECK_ENTRIES_TICKS_EXITS(task2, 1, 2, 0); // * continued
			CHECK_ENTRIES_TICKS_EXITS(task3, 0, 0, 0); // * still fresh
		}

		SUBCASE("Subcase 1B: When first task re-evaluates to FAILURE, second task should be cancelled and exited.") {
			task1->ret_status = BTTask::FAILURE;
			CHECK(seq->execute(0.01666) == BTTask::FAILURE);

			CHECK(task1->get_status() == BTTask::FAILURE);
			CHECK(task2->get_status() == BTTask::FRESH); // * cancelled - status changed to FRESH
			CHECK(task3->get_status() == BTTask::FRESH);

			CHECK_ENTRIES_TICKS_EXITS(task1, 2, 2, 2); // * re-evaluated with FAILURE
			CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 1); // * cancelled - not ticked and exited
			CHECK_ENTRIES_TICKS_EXITS(task3, 0, 0, 0); // * still fresh
		}

		SUBCASE("Subcase 1C: When second task finished, third one is executed.") {
			task1->ret_status = BTTask::SUCCESS;
			task2->ret_status = BTTask::SUCCESS;
			task3->ret_status = BTTask::RUNNING;
			CHECK(seq->execute(0.01666) == BTTask::RUNNING);

			CHECK(task1->get_status() == BTTask::SUCCESS);
			CHECK(task2->get_status() == BTTask::SUCCESS);
			CHECK(task3->get_status() == BTTask::RUNNING);

			CHECK_ENTRIES_TICKS_EXITS(task1, 2, 2, 2); // * re-evaluated with SUCCESS
			CHECK_ENTRIES_TICKS_EXITS(task2, 1, 2, 1); // * ticked and exited with SUCCESS
			CHECK_ENTRIES_TICKS_EXITS(task3, 1, 1, 0); // * entered and running

			SUBCASE("Subcase 1C1: First two are re-evaluated, and when all finish with SUCCESS, we expect SUCCESS.") {
				task1->ret_status = BTTask::SUCCESS;
				task2->ret_status = BTTask::SUCCESS;
				task3->ret_status = BTTask::SUCCESS;
				CHECK(seq->execute(0.01666) == BTTask::SUCCESS);

				CHECK(task1->get_status() == BTTask::SUCCESS);
				CHECK(task2->get_status() == BTTask::SUCCESS);
				CHECK(task3->get_status() == BTTask::SUCCESS);

				CHECK_ENTRIES_TICKS_EXITS(task1, 3, 3, 3); // * re-evaluated with SUCCESS
				CHECK_ENTRIES_TICKS_EXITS(task2, 2, 3, 2); // * re-evaluated with SUCCESS
				CHECK_ENTRIES_TICKS_EXITS(task3, 1, 2, 1); // * ticked and exited with SUCCESS
			}
		}
	}
}

} //namespace TestDynamicSequence

#endif // TEST_DYNAMIC_SEQUENCE_H
