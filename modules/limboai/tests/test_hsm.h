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

inline void wire_callbacks(LimboState *p_state, Ref<CallbackCounter> p_entries_counter, Ref<CallbackCounter> p_updates_counter, Ref<CallbackCounter> p_exits_counter) {
	p_state->call_on_enter(callable_mp(p_entries_counter.ptr(), &CallbackCounter::callback));
	p_state->call_on_update(callable_mp(p_updates_counter.ptr(), &CallbackCounter::callback_delta));
	p_state->call_on_exit(callable_mp(p_exits_counter.ptr(), &CallbackCounter::callback));
}

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
	Ref<CallbackCounter> nested_entries = memnew(CallbackCounter);
	Ref<CallbackCounter> nested_exits = memnew(CallbackCounter);
	Ref<CallbackCounter> nested_updates = memnew(CallbackCounter);
	Ref<CallbackCounter> gamma_entries = memnew(CallbackCounter);
	Ref<CallbackCounter> gamma_exits = memnew(CallbackCounter);
	Ref<CallbackCounter> gamma_updates = memnew(CallbackCounter);
	Ref<CallbackCounter> delta_entries = memnew(CallbackCounter);
	Ref<CallbackCounter> delta_exits = memnew(CallbackCounter);
	Ref<CallbackCounter> delta_updates = memnew(CallbackCounter);

	LimboState *state_alpha = memnew(LimboState);
	wire_callbacks(state_alpha, alpha_entries, alpha_updates, alpha_exits);
	LimboState *state_beta = memnew(LimboState);
	wire_callbacks(state_beta, beta_entries, beta_updates, beta_exits);
	LimboHSM *nested_hsm = memnew(LimboHSM);
	wire_callbacks(nested_hsm, nested_entries, nested_updates, nested_exits);
	LimboState *state_gamma = memnew(LimboState);
	wire_callbacks(state_gamma, gamma_entries, gamma_updates, gamma_exits);
	LimboState *state_delta = memnew(LimboState);
	wire_callbacks(state_delta, delta_entries, delta_updates, delta_exits);

	hsm->add_child(state_alpha);
	hsm->add_child(state_beta);
	hsm->add_child(nested_hsm);
	nested_hsm->add_child(state_gamma);
	nested_hsm->add_child(state_delta);

	hsm->add_transition(state_alpha, state_beta, "event_one");
	hsm->add_transition(state_beta, state_alpha, "event_two");
	hsm->add_transition(hsm->anystate(), nested_hsm, "goto_nested");
	nested_hsm->add_transition(state_gamma, state_delta, "goto_delta");
	nested_hsm->add_transition(state_delta, state_gamma, "goto_gamma");

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
	SUBCASE("Test flow with a nested HSM, and test dispatch() from nested states") {
		state_gamma->dispatch("goto_nested");
		CHECK(hsm->get_leaf_state() == state_gamma);
		CHECK(nested_entries->num_callbacks == 1);
		CHECK(nested_updates->num_callbacks == 0);
		CHECK(nested_exits->num_callbacks == 0);
		CHECK(gamma_entries->num_callbacks == 1);
		CHECK(gamma_updates->num_callbacks == 0);
		CHECK(gamma_exits->num_callbacks == 0);

		hsm->update(0.01666);
		CHECK(nested_entries->num_callbacks == 1);
		CHECK(nested_updates->num_callbacks == 1);
		CHECK(nested_exits->num_callbacks == 0);
		CHECK(gamma_entries->num_callbacks == 1);
		CHECK(gamma_updates->num_callbacks == 1);
		CHECK(gamma_exits->num_callbacks == 0);

		state_gamma->dispatch("goto_delta");
		CHECK(hsm->get_leaf_state() == state_delta);
		CHECK(nested_entries->num_callbacks == 1);
		CHECK(nested_updates->num_callbacks == 1);
		CHECK(nested_exits->num_callbacks == 0);
		CHECK(gamma_entries->num_callbacks == 1);
		CHECK(gamma_updates->num_callbacks == 1);
		CHECK(gamma_exits->num_callbacks == 1);
		CHECK(delta_entries->num_callbacks == 1);
		CHECK(delta_updates->num_callbacks == 0);
		CHECK(delta_exits->num_callbacks == 0);

		state_delta->dispatch(hsm->event_finished());
		CHECK(nested_entries->num_callbacks == 1);
		CHECK(nested_updates->num_callbacks == 1);
		CHECK(nested_exits->num_callbacks == 1);
		CHECK(gamma_entries->num_callbacks == 1);
		CHECK(gamma_updates->num_callbacks == 1);
		CHECK(gamma_exits->num_callbacks == 1);
		CHECK(delta_entries->num_callbacks == 1);
		CHECK(delta_updates->num_callbacks == 0);
		CHECK(delta_exits->num_callbacks == 1);
		CHECK(hsm->is_active() == false);
		CHECK(hsm->get_leaf_state() == hsm);
	}
	SUBCASE("Test get_root()") {
		CHECK(hsm->get_root() == hsm);
		CHECK(state_alpha->get_root() == hsm);
		CHECK(state_beta->get_root() == hsm);
		CHECK(nested_hsm->get_root() == hsm);
		CHECK(state_delta->get_root() == hsm);
		CHECK(state_gamma->get_root() == hsm);
	}

	memdelete(agent);
	memdelete(hsm);
}

} //namespace TestHSM

#endif // TEST_HSM_H
