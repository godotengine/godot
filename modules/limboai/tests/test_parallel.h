/**
 * test_parallel.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_PARALLEL_H
#define TEST_PARALLEL_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/composites/bt_parallel.h"

namespace TestParallel {

TEST_CASE("[Modules][LimboAI] BTParallel with num_required_successes: 1 and num_required_failures: 1") {
	Ref<BTParallel> par = memnew(BTParallel);
	Ref<BTTestAction> task1 = memnew(BTTestAction());
	Ref<BTTestAction> task2 = memnew(BTTestAction());
	Ref<BTTestAction> task3 = memnew(BTTestAction());

	par->add_child(task1);
	par->add_child(task2);
	par->add_child(task3);

	REQUIRE(par->get_child_count() == 3);

	SUBCASE("BTParallel composition {RUNNING, SUCCESS, FAILURE} and successes/failures required 1/1") {
		// * Case #1: When reached both success and failure required, we expect one that triggered sooner (SUCCESS in this case).
		task1->ret_status = BTTask::RUNNING;
		task2->ret_status = BTTask::SUCCESS;
		task3->ret_status = BTTask::FAILURE;
		par->set_num_successes_required(1);
		par->set_num_failures_required(1);
		par->set_repeat(false);

		CHECK(par->execute(0.01666) == BTTask::SUCCESS); // When reached both conditions.

		CHECK(task1->get_status() == BTTask::RUNNING);
		CHECK(task2->get_status() == BTTask::SUCCESS);
		CHECK(task3->get_status() == BTTask::FAILURE);

		CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 0); // * running
		CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 1); // * finished
		CHECK_ENTRIES_TICKS_EXITS(task3, 1, 1, 1); // * finished
	}

	SUBCASE("BTParallel composition {RUNNING, SUCCESS, RUNNING} and successes/failures required 1/1") {
		// * Case #1b: When reached required number of successes, we expect SUCCESS.
		task1->ret_status = BTTask::RUNNING;
		task2->ret_status = BTTask::SUCCESS;
		task3->ret_status = BTTask::RUNNING;
		par->set_num_successes_required(1);
		par->set_num_failures_required(1);
		par->set_repeat(false);

		CHECK(par->execute(0.01666) == BTTask::SUCCESS);

		CHECK(task1->get_status() == BTTask::RUNNING);
		CHECK(task2->get_status() == BTTask::SUCCESS);
		CHECK(task3->get_status() == BTTask::RUNNING);

		CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 0); // * running
		CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 1); // * finished
		CHECK_ENTRIES_TICKS_EXITS(task3, 1, 1, 0); // * running
	}

	SUBCASE("BTParallel composition {RUNNING, FAILURE, RUNNING} and successes/failures required 1/1") {
		// * Case #1c: When reached required number of failures, we expect FAILURE.
		task1->ret_status = BTTask::RUNNING;
		task2->ret_status = BTTask::FAILURE;
		task3->ret_status = BTTask::RUNNING;
		par->set_num_successes_required(1);
		par->set_num_failures_required(1);
		par->set_repeat(false);

		CHECK(par->execute(0.01666) == BTTask::FAILURE);

		CHECK(task1->get_status() == BTTask::RUNNING);
		CHECK(task2->get_status() == BTTask::FAILURE);
		CHECK(task3->get_status() == BTTask::RUNNING);

		CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 0); // * running
		CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 1); // * finished
		CHECK_ENTRIES_TICKS_EXITS(task3, 1, 1, 0); // * running
	}

	SUBCASE("BTParallel composition {SUCCESS, RUNNING, FAILURE} with successes/failures required 3/3 (not repeating)") {
		// * Case #2: When failed to reach required number of successes or failures,
		// * and not all children finished executing while not repeating, we expect RUNNING.
		task1->ret_status = BTTask::SUCCESS;
		task2->ret_status = BTTask::RUNNING;
		task3->ret_status = BTTask::FAILURE;
		par->set_num_successes_required(3);
		par->set_num_failures_required(3);
		par->set_repeat(false);

		CHECK(par->execute(0.01666) == BTTask::RUNNING);

		CHECK(task1->get_status() == BTTask::SUCCESS);
		CHECK(task2->get_status() == BTTask::RUNNING);
		CHECK(task3->get_status() == BTTask::FAILURE);

		CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 1); // * finished
		CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 0); // * running
		CHECK_ENTRIES_TICKS_EXITS(task3, 1, 1, 1); // * finished
	}

	SUBCASE("BTParallel composition {SUCCESS, FAILURE, SUCCESS} with successes/failures required 3/3 (not repeating)") {
		// * Case #3: When failed to reach required number of successes or failures,
		// * and all children finished executing while not repeating, we expect FAILURE.
		task1->ret_status = BTTask::SUCCESS;
		task2->ret_status = BTTask::FAILURE;
		task3->ret_status = BTTask::SUCCESS;
		par->set_num_successes_required(3);
		par->set_num_failures_required(3);
		par->set_repeat(false);
		CHECK(par->execute(0.01666) == BTTask::FAILURE);

		CHECK(task1->get_status() == BTTask::SUCCESS);
		CHECK(task2->get_status() == BTTask::FAILURE);
		CHECK(task3->get_status() == BTTask::SUCCESS);

		CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 1);
		CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 1);
		CHECK_ENTRIES_TICKS_EXITS(task3, 1, 1, 1);
	}

	SUBCASE("BTParallel composition {SUCCESS, FAILURE, SUCCESS} with successes/failures required 3/3 (repeating)") {
		// * Case #4: When failed to reach required number of successes or failures,
		// * and all children finished executing while repeating, we expect RUNNING.
		task1->ret_status = BTTask::SUCCESS;
		task2->ret_status = BTTask::FAILURE;
		task3->ret_status = BTTask::SUCCESS;
		par->set_num_successes_required(3);
		par->set_num_failures_required(3);
		par->set_repeat(true);
		CHECK(par->execute(0.01666) == BTTask::RUNNING);

		CHECK(task1->get_status() == BTTask::SUCCESS);
		CHECK(task2->get_status() == BTTask::FAILURE);
		CHECK(task3->get_status() == BTTask::SUCCESS);

		CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 1);
		CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 1);
		CHECK_ENTRIES_TICKS_EXITS(task3, 1, 1, 1);

		// * Execution #2: Check if tasks are repeated, when set so (there is no RUNNING task).
		CHECK(par->execute(0.01666) == BTTask::RUNNING);

		CHECK(task1->get_status() == BTTask::SUCCESS);
		CHECK(task2->get_status() == BTTask::FAILURE);
		CHECK(task3->get_status() == BTTask::SUCCESS);

		CHECK_ENTRIES_TICKS_EXITS(task1, 2, 2, 2); // * repeated
		CHECK_ENTRIES_TICKS_EXITS(task2, 2, 2, 2); // * repeated
		CHECK_ENTRIES_TICKS_EXITS(task3, 2, 2, 2); // * repeated
	}

	SUBCASE("BTParallel composition {SUCCESS, RUNNING, FAILURE} with successes/failures required 2/2 (not repeating)") {
		// * Case #5: When failed to reach required number of successes or failures,
		// * but not all children finished executing (not repeating), we expect RUNNING.
		task1->ret_status = BTTask::SUCCESS;
		task2->ret_status = BTTask::RUNNING;
		task3->ret_status = BTTask::FAILURE;
		par->set_num_successes_required(2);
		par->set_num_failures_required(2);
		par->set_repeat(false);
		CHECK(par->execute(0.01666) == BTTask::RUNNING);

		CHECK(task1->get_status() == BTTask::SUCCESS);
		CHECK(task2->get_status() == BTTask::RUNNING);
		CHECK(task3->get_status() == BTTask::FAILURE);

		CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 1);
		CHECK_ENTRIES_TICKS_EXITS(task2, 1, 1, 0);
		CHECK_ENTRIES_TICKS_EXITS(task3, 1, 1, 1);

		// * Execution #2: Check if tasks are not repeated, when set so.
		CHECK(par->execute(0.01666) == BTTask::RUNNING);

		CHECK(task1->get_status() == BTTask::SUCCESS);
		CHECK(task2->get_status() == BTTask::RUNNING);
		CHECK(task3->get_status() == BTTask::FAILURE);

		CHECK_ENTRIES_TICKS_EXITS(task1, 1, 1, 1); // * not repeated
		CHECK_ENTRIES_TICKS_EXITS(task2, 1, 2, 0); // * continued
		CHECK_ENTRIES_TICKS_EXITS(task3, 1, 1, 1); // * not repeated

		// * Execution #3: Check if tasks are repeated, when set so.
		par->set_repeat(true);
		CHECK(par->execute(0.01666) == BTTask::RUNNING);

		CHECK(task1->get_status() == BTTask::SUCCESS);
		CHECK(task2->get_status() == BTTask::RUNNING);
		CHECK(task3->get_status() == BTTask::FAILURE);

		CHECK_ENTRIES_TICKS_EXITS(task1, 2, 2, 2); // * repeated
		CHECK_ENTRIES_TICKS_EXITS(task2, 1, 3, 0); // * continued
		CHECK_ENTRIES_TICKS_EXITS(task3, 2, 2, 2); // * repeated
	}
}

} //namespace TestParallel

#endif // TEST_PARALLEL_H
