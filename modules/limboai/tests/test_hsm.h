/**
 * test_hsm.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_HSM_H
#define TEST_HSM_H

#include "limbo_test.h"

#include "modules/limboai/hsm/limbo_hsm.h"
#include "modules/limboai/hsm/limbo_state.h"

#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/os/memory.h"
#include "core/variant/variant.h"

namespace TestHSM {

class TestGuard : public RefCounted {
	GDCLASS(TestGuard, RefCounted);

public:
	bool permitted_to_enter = false;
	bool can_enter() { return permitted_to_enter; }
};

TEST_CASE("[Modules][LimboAI] HSM") {
	Node *agent = memnew(Node);
	LimboHSM *hsm = memnew(LimboHSM);

	Ref<CallbackCounter> alpha_entries = memnew(CallbackCounter);
	Ref<CallbackCounter> alpha_exits = memnew(CallbackCounter);
	Ref<CallbackCounter> alpha_updates = memnew(CallbackCounter);
	Ref<CallbackCounter> beta_entries = memnew(CallbackCounter);
	Ref<CallbackCounter> beta_exits = memnew(CallbackCounter);
	Ref<CallbackCounter> beta_updates = memnew(CallbackCounter);

	LimboState *state_alpha = memnew(LimboState);
	state_alpha->call_on_enter(callable_mp(alpha_entries.ptr(), &CallbackCounter::callback));
	state_alpha->call_on_update(callable_mp(alpha_updates.ptr(), &CallbackCounter::callback_delta));
	state_alpha->call_on_exit(callable_mp(alpha_exits.ptr(), &CallbackCounter::callback));

	LimboState *state_beta = memnew(LimboState);
	state_beta->call_on_enter(callable_mp(beta_entries.ptr(), &CallbackCounter::callback));
	state_beta->call_on_update(callable_mp(beta_updates.ptr(), &CallbackCounter::callback_delta));
	state_beta->call_on_exit(callable_mp(beta_exits.ptr(), &CallbackCounter::callback));

	hsm->add_child(state_alpha);
	hsm->add_child(state_beta);

	hsm->add_transition(state_alpha, state_beta, "event_one");
	hsm->add_transition(state_beta, state_alpha, "event_two");

	hsm->set_initial_state(state_alpha);
	Ref<Blackboard> parent_scope = memnew(Blackboard);
	hsm->initialize(agent, parent_scope);
	hsm->set_active(true);

	SUBCASE("Test get_root()") {
		CHECK(state_alpha->get_root() == hsm);
		CHECK(state_beta->get_root() == hsm);
		CHECK(hsm->get_root() == hsm);
	}
	SUBCASE("Test with basic workflow and transitions") {
		REQUIRE(hsm->is_active());
		REQUIRE(hsm->get_active_state() == state_alpha);
		CHECK(alpha_entries->num_callbacks == 1); // * entered
		CHECK(alpha_updates->num_callbacks == 0);
		CHECK(alpha_exits->num_callbacks == 0);
		CHECK(beta_entries->num_callbacks == 0);
		CHECK(beta_updates->num_callbacks == 0);
		CHECK(beta_exits->num_callbacks == 0);

		hsm->update(0.01666);
		CHECK(alpha_entries->num_callbacks == 1);
		CHECK(alpha_updates->num_callbacks == 1); // * updated
		CHECK(alpha_exits->num_callbacks == 0);
		CHECK(beta_entries->num_callbacks == 0);
		CHECK(beta_updates->num_callbacks == 0);
		CHECK(beta_exits->num_callbacks == 0);

		hsm->update(0.01666);
		CHECK(alpha_entries->num_callbacks == 1);
		CHECK(alpha_updates->num_callbacks == 2); // * updated x2
		CHECK(alpha_exits->num_callbacks == 0);
		CHECK(beta_entries->num_callbacks == 0);
		CHECK(beta_updates->num_callbacks == 0);
		CHECK(beta_exits->num_callbacks == 0);

		hsm->dispatch("event_one");
		REQUIRE(hsm->get_active_state() == state_beta);
		CHECK(alpha_entries->num_callbacks == 1);
		CHECK(alpha_updates->num_callbacks == 2);
		CHECK(alpha_exits->num_callbacks == 1); // * (1) exited
		CHECK(beta_entries->num_callbacks == 1); // * (2) entered
		CHECK(beta_updates->num_callbacks == 0);
		CHECK(beta_exits->num_callbacks == 0);

		hsm->update(0.01666);
		CHECK(alpha_entries->num_callbacks == 1);
		CHECK(alpha_updates->num_callbacks == 2);
		CHECK(alpha_exits->num_callbacks == 1);
		CHECK(beta_entries->num_callbacks == 1);
		CHECK(beta_updates->num_callbacks == 1); // * updated
		CHECK(beta_exits->num_callbacks == 0);

		hsm->update(0.01666);
		CHECK(alpha_entries->num_callbacks == 1);
		CHECK(alpha_updates->num_callbacks == 2);
		CHECK(alpha_exits->num_callbacks == 1);
		CHECK(beta_entries->num_callbacks == 1);
		CHECK(beta_updates->num_callbacks == 2); // * updated x2
		CHECK(beta_exits->num_callbacks == 0);

		hsm->dispatch("event_two");
		REQUIRE(hsm->get_active_state() == state_alpha);
		CHECK(alpha_entries->num_callbacks == 2); // * (2) entered
		CHECK(alpha_updates->num_callbacks == 2);
		CHECK(alpha_exits->num_callbacks == 1);
		CHECK(beta_entries->num_callbacks == 1);
		CHECK(beta_updates->num_callbacks == 2);
		CHECK(beta_exits->num_callbacks == 1); // * (1) exited

		hsm->update(0.01666);
		CHECK(alpha_entries->num_callbacks == 2);
		CHECK(alpha_updates->num_callbacks == 3); // * updated
		CHECK(alpha_exits->num_callbacks == 1);
		CHECK(beta_entries->num_callbacks == 1);
		CHECK(beta_updates->num_callbacks == 2);
		CHECK(beta_exits->num_callbacks == 1);

		hsm->dispatch(hsm->event_finished());
		CHECK(alpha_entries->num_callbacks == 2);
		CHECK(alpha_updates->num_callbacks == 3);
		CHECK(alpha_exits->num_callbacks == 2); // * exited
		CHECK(beta_entries->num_callbacks == 1);
		CHECK(beta_updates->num_callbacks == 2);
		CHECK(beta_exits->num_callbacks == 1);
		CHECK_FALSE(hsm->is_active()); // * not active
		CHECK(hsm->get_active_state() == nullptr);
	}
	SUBCASE("Test transition with guard") {
		Ref<TestGuard> guard = memnew(TestGuard);
		state_beta->set_guard(callable_mp(guard.ptr(), &TestGuard::can_enter));

		SUBCASE("When entry is permitted") {
			guard->permitted_to_enter = true;
			hsm->dispatch("event_one");
			CHECK(hsm->get_active_state() == state_beta);
			CHECK(alpha_exits->num_callbacks == 1);
			CHECK(beta_entries->num_callbacks == 1);
		}
		SUBCASE("When entry is not permitted") {
			guard->permitted_to_enter = false;
			hsm->dispatch("event_one");
			CHECK(hsm->get_active_state() == state_alpha);
			CHECK(alpha_exits->num_callbacks == 0);
			CHECK(beta_entries->num_callbacks == 0);
		}
	}
	SUBCASE("When there is no transition for given event") {
		hsm->dispatch("not_found");
		CHECK(alpha_exits->num_callbacks == 0);
		CHECK(beta_entries->num_callbacks == 0);
		CHECK(hsm->is_active());
		CHECK(hsm->get_active_state() == state_alpha);
	}
	SUBCASE("Check if parent scope is accessible") {
		parent_scope->set_var("parent_var", 100);
		CHECK(state_alpha->get_blackboard()->get_parent() == parent_scope);
		CHECK(state_beta->get_blackboard()->get_parent() == parent_scope);
		CHECK(state_alpha->get_blackboard()->get_var("parent_var", Variant()) == Variant(100));
	}

	memdelete(agent);
	memdelete(hsm);
}

} //namespace TestHSM

#endif // TEST_HSM_H
