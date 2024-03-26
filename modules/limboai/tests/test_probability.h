/**
 * test_probability.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_PROBABILITY_H
#define TEST_PROBABILITY_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_probability.h"

#include "core/math/math_funcs.h"

namespace TestProbability {

TEST_CASE("[Modules][LimboAI] BTProbability") {
	Ref<BTProbability> prob = memnew(BTProbability);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		CHECK(prob->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	Ref<BTTestAction> task = memnew(BTTestAction);
	prob->add_child(task);

	Math::randomize();

	SUBCASE("Check if probability meets expectation") {
		task->ret_status = BTTask::SUCCESS;
		prob->set_run_chance(0.5);

		for (int i = 0; i < 1000; i++) {
			prob->execute(0.01666);
		}

		CHECK(task->num_ticks > 450);
		CHECK(task->num_ticks < 550);
	}

	SUBCASE("When probability is 0") {
		task->ret_status = BTTask::SUCCESS;
		prob->set_run_chance(0.0);

		for (int i = 0; i < 1000; i++) {
			prob->execute(0.01666);
		}

		CHECK(task->num_ticks == 0);
	}

	SUBCASE("When probability is 1") {
		task->ret_status = BTTask::SUCCESS;
		prob->set_run_chance(1.0);

		for (int i = 0; i < 1000; i++) {
			prob->execute(0.01666);
		}

		CHECK(task->num_ticks == 1000);
	}

	SUBCASE("Test return status") {
		prob->set_run_chance(1.0);

		task->ret_status = BTTask::SUCCESS;
		CHECK(prob->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 1, 1, 1);

		task->ret_status = BTTask::RUNNING;
		CHECK(prob->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 2, 2, 1);

		task->ret_status = BTTask::FAILURE;
		CHECK(prob->execute(0.01666) == BTTask::FAILURE);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FAILURE, 2, 3, 2);
	}
}

} //namespace TestProbability

#endif // TEST_PROBABILITY_H
