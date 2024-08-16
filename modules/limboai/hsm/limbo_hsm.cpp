/**
 * limbo_hsm.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "limbo_hsm.h"

VARIANT_ENUM_CAST(LimboHSM::UpdateMode);

void LimboHSM::set_active(bool p_active) {
	ERR_FAIL_COND_MSG(agent == nullptr, "LimboHSM is not initialized.");
	ERR_FAIL_COND_MSG(p_active && initial_state == nullptr, "LimboHSM has no initial substate candidate.");

	if (active == p_active) {
		return;
	}

	active = p_active;
	switch (update_mode) {
		case UpdateMode::IDLE: {
			set_process(p_active);
			set_physics_process(false);
		} break;
		case UpdateMode::PHYSICS: {
			set_process(false);
			set_physics_process(p_active);
		} break;
		case UpdateMode::MANUAL: {
			set_process(false);
			set_physics_process(false);
		} break;
	}
	set_process_input(p_active);

	if (active) {
		_enter();
	} else {
		_exit();
	}
}

void LimboHSM::change_active_state(LimboState *p_state) {
	ERR_FAIL_NULL(p_state);
	ERR_FAIL_COND_MSG(!is_active(), "LimboHSM: Unable to change active state when HSM is not active.");
	ERR_FAIL_COND_MSG(p_state->get_parent() != this, "LimboHSM: Unable to perform transition to a state that is not a child of this HSM.");

	if (active_state) {
		active_state->_exit();
		active_state->set_process_input(false);
		previous_active = active_state;
	}

	active_state = p_state;
	active_state->_enter();
	active_state->set_process_input(true);

	emit_signal(LimboStringNames::get_singleton()->active_state_changed, active_state, previous_active);
}

void LimboHSM::_enter() {
	ERR_FAIL_COND_MSG(get_child_count() == 0, "LimboHSM has no candidate for initial substate.");
	ERR_FAIL_COND(active_state != nullptr);
	ERR_FAIL_COND_MSG(initial_state == nullptr, "LimboHSM: Initial state is not set.");

	LimboState::_enter();
	change_active_state(initial_state);
}

void LimboHSM::_exit() {
	ERR_FAIL_COND(active_state == nullptr);
	active_state->_exit();
	active_state = nullptr;
	LimboState::_exit();
}

void LimboHSM::_update(double p_delta) {
	if (active) {
		ERR_FAIL_NULL(active_state);
		LimboState *last_active_state = active_state;
		LimboState::_update(p_delta);
		if (last_active_state == active_state) {
			active_state->_update(p_delta);
		}
	}
}

void LimboHSM::update(double p_delta) {
	updating = true;
	_update(p_delta);
	updating = false;
	if (next_active) {
		change_active_state(next_active);
		next_active = nullptr;
	}
}

void LimboHSM::add_transition(LimboState *p_from_state, LimboState *p_to_state, const StringName &p_event) {
	ERR_FAIL_COND_MSG(p_from_state != nullptr && p_from_state->get_parent() != this, "LimboHSM: Unable to add a transition from a state that is not an immediate child of mine.");
	ERR_FAIL_COND_MSG(p_to_state == nullptr, "LimboHSM: Unable to add a transition to a null state.");
	ERR_FAIL_COND_MSG(p_to_state->get_parent() != this, "LimboHSM: Unable to add a transition to a state that is not an immediate child of mine.");
	ERR_FAIL_COND_MSG(p_event == StringName(), "LimboHSM: Failed to add transition due to empty event string.");

	TransitionKey key = Transition::make_key(p_from_state, p_event);
	ERR_FAIL_COND_MSG(transitions.has(key), "LimboHSM: Unable to add another transition with the same event and origin.");
	// Note: Explicit casting needed for GDExtension.
	transitions[key] = { p_from_state != nullptr ? ObjectID(p_from_state->get_instance_id()) : ObjectID(), ObjectID(p_to_state->get_instance_id()), p_event };
}

void LimboHSM::remove_transition(LimboState *p_from_state, const StringName &p_event) {
	ERR_FAIL_COND_MSG(p_from_state != nullptr && p_from_state->get_parent() != this, "LimboHSM: Unable to remove a transition from a state that is not an immediate child of mine.");
	ERR_FAIL_COND_MSG(p_event == StringName(), "LimboHSM: Unable to remove a transition due to empty event string.");

	TransitionKey key = Transition::make_key(p_from_state, p_event);
	ERR_FAIL_COND_MSG(!transitions.has(key), "LimboHSM: Unable to remove a transition that does not exist.");
	transitions.erase(key);
}

void LimboHSM::_get_transition(LimboState *p_from_state, const StringName &p_event, Transition &r_transition) const {
	ERR_FAIL_COND_MSG(p_from_state != nullptr && p_from_state->get_parent() != this, "LimboHSM: Unable to get a transition from a state that is not an immediate child of this HSM.");
	ERR_FAIL_COND_MSG(p_event == StringName(), "LimboHSM: Unable to get a transition with an empty event string.");

	TransitionKey key = Transition::make_key(p_from_state, p_event);
	if (transitions.has(key)) {
		r_transition = transitions[key];
	}
}

LimboState *LimboHSM::get_leaf_state() const {
	LimboHSM *hsm = const_cast<LimboHSM *>(this);
	while (hsm->active_state != nullptr && hsm->active_state->is_class("LimboHSM")) {
		hsm = Object::cast_to<LimboHSM>(hsm->active_state);
	}
	if (hsm->active_state) {
		return hsm->active_state;
	} else {
		return hsm;
	}
}

void LimboHSM::set_initial_state(LimboState *p_state) {
	ERR_FAIL_COND(p_state == nullptr || !p_state->is_class("LimboState"));
	initial_state = Object::cast_to<LimboState>(p_state);
}

bool LimboHSM::_dispatch(const StringName &p_event, const Variant &p_cargo) {
	ERR_FAIL_COND_V(p_event == StringName(), false);

	bool event_consumed = false;

	if (active_state) {
		event_consumed = active_state->_dispatch(p_event, p_cargo);
	}

	if (!event_consumed) {
		event_consumed = LimboState::_dispatch(p_event, p_cargo);
	}

	if (!event_consumed && active_state) {
		LimboState *to_state = nullptr;

		Transition transition;
		_get_transition(active_state, p_event, transition);
		if (transition.is_valid()) {
			to_state = Object::cast_to<LimboState>(ObjectDB::get_instance(transition.to_state));
		}
		if (to_state == nullptr) {
			// Get ANYSTATE transition.
			_get_transition(nullptr, p_event, transition);
			if (transition.is_valid()) {
				to_state = Object::cast_to<LimboState>(ObjectDB::get_instance(transition.to_state));
				if (to_state == active_state) {
					// Transitions to self are not allowed with ANYSTATE.
					to_state = nullptr;
				}
			}
		}
		if (to_state != nullptr) {
			bool permitted = true;
			if (to_state->guard_callable.is_valid()) {
				Variant ret;

#ifdef LIMBOAI_MODULE
				Callable::CallError ce;
				to_state->guard_callable.callp(nullptr, 0, ret, ce);
				if (unlikely(ce.error != Callable::CallError::CALL_OK)) {
					ERR_PRINT_ONCE("LimboHSM: Error calling substate's guard callable: " + Variant::get_callable_error_text(to_state->guard_callable, nullptr, 0, ce));
				}
#elif LIMBOAI_GDEXTENSION
				ret = to_state->guard_callable.call();
#endif

				if (unlikely(ret.get_type() != Variant::BOOL)) {
					ERR_PRINT_ONCE(vformat("State guard callable %s returned non-boolean value (%s).", to_state->guard_callable, to_state));
				} else {
					permitted = bool(ret);
				}
			}
			if (permitted) {
				if (!updating) {
					change_active_state(to_state);
				} else if (!next_active) {
					// Only set next_active if we are not already in the process of changing states.
					next_active = to_state;
				}
				event_consumed = true;
			}
		}
	}

	if (!event_consumed && p_event == LW_NAME(EVENT_FINISHED) && !(get_parent() && get_parent()->is_class("LimboState"))) {
		_exit();
	}

	return event_consumed;
}

void LimboHSM::initialize(Node *p_agent, const Ref<Blackboard> &p_parent_scope) {
	ERR_FAIL_COND(p_agent == nullptr);
	ERR_FAIL_COND_MSG(!is_root(), "LimboHSM: initialize() must be called on the root HSM.");

	_initialize(p_agent, p_parent_scope);

	if (initial_state == nullptr) {
		initial_state = Object::cast_to<LimboState>(get_child(0));
	}
}

void LimboHSM::_initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard) {
	ERR_FAIL_COND(p_agent == nullptr);
	ERR_FAIL_COND_MSG(agent != nullptr, "LimboAI: HSM already initialized.");
	ERR_FAIL_COND_MSG(get_child_count() == 0, "Cannot initialize LimboHSM: no candidate for initial substate.");

	if (initial_state == nullptr) {
		initial_state = Object::cast_to<LimboState>(get_child(0));
		ERR_FAIL_COND_MSG(initial_state == nullptr, "LimboHSM: Child at index 0 is not a LimboState.");
	}

	LimboState::_initialize(p_agent, p_blackboard);

	for (int i = 0; i < get_child_count(); i++) {
		LimboState *c = Object::cast_to<LimboState>(get_child(i));
		if (unlikely(c == nullptr)) {
			ERR_PRINT(vformat("LimboHSM: Child at index %d is not a LimboState.", i));
		} else {
			c->_initialize(agent, blackboard);
		}
	}
}

void LimboHSM::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == LW_NAME(update_mode) && !is_root()) {
		// Hide update_mode for non-root HSMs.
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void LimboHSM::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POST_ENTER_TREE: {
		} break;
		case NOTIFICATION_PROCESS: {
			_update(get_process_delta_time());
		} break;
		case NOTIFICATION_PHYSICS_PROCESS: {
			_update(get_physics_process_delta_time());
		} break;
	}
}

void LimboHSM::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_update_mode", "mode"), &LimboHSM::set_update_mode);
	ClassDB::bind_method(D_METHOD("get_update_mode"), &LimboHSM::get_update_mode);

	ClassDB::bind_method(D_METHOD("set_initial_state", "state"), &LimboHSM::set_initial_state);
	ClassDB::bind_method(D_METHOD("get_initial_state"), &LimboHSM::get_initial_state);

	ClassDB::bind_method(D_METHOD("get_active_state"), &LimboHSM::get_active_state);
	ClassDB::bind_method(D_METHOD("get_previous_active_state"), &LimboHSM::get_previous_active_state);
	ClassDB::bind_method(D_METHOD("get_leaf_state"), &LimboHSM::get_leaf_state);
	ClassDB::bind_method(D_METHOD("set_active", "active"), &LimboHSM::set_active);
	ClassDB::bind_method(D_METHOD("update", "delta"), &LimboHSM::update);
	ClassDB::bind_method(D_METHOD("add_transition", "from_state", "to_state", "event"), &LimboHSM::add_transition);
	ClassDB::bind_method(D_METHOD("remove_transition", "from_state", "event"), &LimboHSM::remove_transition);
	ClassDB::bind_method(D_METHOD("has_transition", "from_state", "event"), &LimboHSM::has_transition);
	ClassDB::bind_method(D_METHOD("anystate"), &LimboHSM::anystate);
	ClassDB::bind_method(D_METHOD("initialize", "agent", "parent_scope"), &LimboHSM::initialize, Variant());
	ClassDB::bind_method(D_METHOD("change_active_state", "state"), &LimboHSM::change_active_state);

	BIND_ENUM_CONSTANT(IDLE);
	BIND_ENUM_CONSTANT(PHYSICS);
	BIND_ENUM_CONSTANT(MANUAL);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "update_mode", PROPERTY_HINT_ENUM, "Idle, Physics, Manual"), "set_update_mode", "get_update_mode");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "ANYSTATE", PROPERTY_HINT_RESOURCE_TYPE, "LimboState", 0), "", "anystate");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "initial_state", PROPERTY_HINT_RESOURCE_TYPE, "LimboState", 0), "set_initial_state", "get_initial_state");

	ADD_SIGNAL(MethodInfo("active_state_changed",
			PropertyInfo(Variant::OBJECT, "current", PROPERTY_HINT_RESOURCE_TYPE, "LimboState"),
			PropertyInfo(Variant::OBJECT, "previous", PROPERTY_HINT_RESOURCE_TYPE, "LimboState")));
}

LimboHSM::LimboHSM() {
	update_mode = UpdateMode::PHYSICS;
	active_state = nullptr;
	previous_active = nullptr;
	next_active = nullptr;
	initial_state = nullptr;
}
