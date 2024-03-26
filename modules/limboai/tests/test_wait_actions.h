/**
 * test_wait_actions.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_WAIT_ACTIONS_H
#define TEST_WAIT_ACTIONS_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/utility/bt_random_wait.h"
#include "modules/limboai/bt/tasks/utility/bt_wait.h"
#include "modules/limboai/bt/tasks/utility/bt_wait_ticks.h"

#include "core/math/math_funcs.h"

namespace TestWaitActions {

TEST_CASE("[Modules][LimboAI] BTWait") {
	Ref<BTWait> wait = memnew(BTWait);

	SUBCASE("With zero duration") {
		wait->set_duration(0.0);
		CHECK(wait->execute(0.0) == BTWait::SUCCESS);
	}
	SUBCASE("With one second duration") {
		wait->set_duration(1.0);
		CHECK(wait->execute(0.0) == BTWait::RUNNING);
		CHECK(wait->execute(0.5) == BTWait::RUNNING); // * elapsed 0.5
		CHECK(wait->execute(0.5) == BTWait::SUCCESS); // * elapsed 1.0
	}
}

TEST_CASE("[Modules][LimboAI] BTWaitTicks") {
	Ref<BTWaitTicks> wait = memnew(BTWaitTicks);

	SUBCASE("With zero ticks") {
		wait->set_num_ticks(0);
		CHECK(wait->execute(0.01666) == BTWait::SUCCESS); // * elapsed 0 ticks
	}
	SUBCASE("With 1 tick") {
		wait->set_num_ticks(1);
		CHECK(wait->execute(0.01666) == BTWait::RUNNING); // * elapsed 0 ticks
		CHECK(wait->execute(0.01666) == BTWait::SUCCESS); // * elapsed 1 tick
	}
	SUBCASE("With 2 ticks") {
		wait->set_num_ticks(2);
		CHECK(wait->execute(0.01666) == BTWait::RUNNING); // * elapsed 0 ticks
		CHECK(wait->execute(0.01666) == BTWait::RUNNING); // * elapsed 1 tick
		CHECK(wait->execute(0.01666) == BTWait::SUCCESS); // * elapsed 2 ticks
	}
}

TEST_CASE("[Modules][LimboAI] BTRandomWait") {
	Ref<BTRandomWait> wait = memnew(BTRandomWait);

	Math::randomize();

	SUBCASE("With duration range [0, 0]") {
		wait->set_min_duration(0.0);
		wait->set_max_duration(0.0);
		CHECK(wait->execute(0.01666) == BTWait::SUCCESS);
	}
	SUBCASE("With certain SUCCESS") {
		wait->set_min_duration(0.5);
		wait->set_max_duration(1.0);

		CHECK(wait->execute(0.00) == BTWait::RUNNING); // * elapsed 0.00
		CHECK(wait->execute(0.25) == BTWait::RUNNING); // * elapsed 0.25
		CHECK(wait->execute(0.76) == BTWait::SUCCESS); // * elapsed 1.01
	}
	SUBCASE("With duration range [0.5, 1.0]") {
		wait->set_min_duration(0.5);
		wait->set_max_duration(1.0);

		int num_successes = 0;
		int num_running = 0;
		int num_failures = 0;
		int num_undefined = 0;

		for (int i = 0; i < 1000; i++) {
			wait->execute(0.00); // * elapsed 0.00
			wait->execute(0.75); // * elapsed 0.75
			switch (wait->get_status()) {
				case BTTask::RUNNING: {
					num_running += 1;
				} break;
				case BTTask::SUCCESS: {
					num_successes += 1;
				} break;
				case BTTask::FAILURE: {
					num_failures += 1;
				} break;
				default: {
					num_undefined += 1;
				} break;
			}
			wait->abort();
		}

		// * Expected ~500/500 SUCCESS/RUNNING.
		CHECK(num_successes > 450);
		CHECK(num_successes < 550);
		CHECK(num_running > 450);
		CHECK(num_running < 550);
		CHECK(num_failures == 0);
		CHECK(num_undefined == 0);
	}
}

} //namespace TestWaitActions

#endif // TEST_WAIT_ACTIONS_H
