/**
 * test_random_sequence.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_RANDOM_SEQUENCE_H
#define TEST_RANDOM_SEQUENCE_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/composites/bt_random_sequence.h"

namespace TestRandomSequence {

TEST_CASE("[Modules][LimboAI] BTRandomSequence") {
	Ref<BTRandomSequence> seq = memnew(BTRandomSequence);
	Ref<BTTestAction> task1 = memnew(BTTestAction());
	Ref<BTTestAction> task2 = memnew(BTTestAction());
	Ref<BTTestAction> task3 = memnew(BTTestAction());

	seq->add_child(task1);
	seq->add_child(task2);
	seq->add_child(task3);

	REQUIRE(seq->get_child_count() == 3);

	SUBCASE("Expecting RUNNING status when a child task returns RUNNING") {
		task1->ret_status = BTTask::SUCCESS;
		task2->ret_status = BTTask::RUNNING;
		task3->ret_status = BTTask::SUCCESS;

		CHECK(seq->execute(0.01666) == BTTask::RUNNING);

		CHECK(task1->is_status_either(BTTask::SUCCESS, BTTask::FRESH));
		CHECK(task2->get_status() == BTTask::RUNNING);
		CHECK(task3->is_status_either(BTTask::SUCCESS, BTTask::FRESH));

		CHECK_ENTRIES_TICKS_EXITS_UP_TO(task1, 1, 1, 1); // * ran no more than once
		CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 0); // * running - enters and ticks
		CHECK_ENTRIES_TICKS_EXITS_UP_TO(task3, 1, 1, 1); // * ran no more than once

		SUBCASE("Resuming and succeeding when all tasks succeed") {
			task2->ret_status = BTTask::SUCCESS;

			CHECK(seq->execute(0.01666) == BTTask::SUCCESS);

			CHECK(task1->get_status() == BTTask::SUCCESS);
			CHECK(task2->get_status() == BTTask::SUCCESS);
			CHECK(task3->get_status() == BTTask::SUCCESS);

			CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 1); // * ran once
			CHECK_ENTRIES_TICKS_EXITS(task2, 1, 2, 1); // * finishes - ticks and exits with SUCCESS
			CHECK_ENTRIES_TICKS_EXITS(task3, 1, 1, 1); // * ran once
		}

		SUBCASE("Resuming and failing when a child task fails") {
			task2->ret_status = BTTask::FAILURE;

			CHECK(seq->execute(0.01666) == BTTask::FAILURE);

			CHECK(task1->is_status_either(BTTask::SUCCESS, BTTask::FRESH));
			CHECK(task2->get_status() == BTTask::FAILURE);
			CHECK(task3->is_status_either(BTTask::SUCCESS, BTTask::FRESH));

			CHECK_ENTRIES_TICKS_EXITS_UP_TO(task1, 1, 1, 1); // * ran no more than once
			CHECK_ENTRIES_TICKS_EXITS(task2, 1, 2, 1); // * finishes - ticks and exits with FAILURE
			CHECK_ENTRIES_TICKS_EXITS_UP_TO(task3, 1, 1, 1); // * ran no more than once
		}
	}

	SUBCASE("Verify that tasks are executed in random order") {
		task1->ret_status = BTTask::SUCCESS;
		task2->ret_status = BTTask::SUCCESS;
		task3->ret_status = BTTask::RUNNING;

		int num_tries = 10;
		bool is_confirmed = false;
		while (!is_confirmed && num_tries--) {
			CHECK(seq->execute(0.01666) == BTTask::RUNNING);
			int checksum = 0;
			if (task1->get_status() == BTTask::SUCCESS) {
				checksum += 1;
			}
			if (task2->get_status() == BTTask::SUCCESS) {
				checksum += 2;
			}
			if (task3->get_status() == BTTask::RUNNING) {
				checksum += 4;
			}
			is_confirmed = (checksum != 7);
		}
		CHECK(is_confirmed);
	}
}

TEST_CASE("[Modules][LimboAI] Empty BTRandomSequence returns SUCCESS") {
	Ref<BTRandomSequence> seq = memnew(BTRandomSequence);
	CHECK(seq->execute(0.01666) == BTTask::SUCCESS);
}

} //namespace TestRandomSequence

#endif // TEST_RANDOM_SEQUENCE_H
