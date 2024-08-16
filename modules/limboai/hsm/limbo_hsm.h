/**
 * limbo_hsm.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef LIMBO_HSM_H
#define LIMBO_HSM_H

#include "limbo_state.h"

#define TransitionKey Pair<uint64_t, StringName>

class LimboHSM : public LimboState {
	GDCLASS(LimboHSM, LimboState);

public:
	enum UpdateMode : unsigned int {
		IDLE, // automatically call update() during NOTIFICATION_PROCESS
		PHYSICS, // automatically call update() during NOTIFICATION_PHYSICS
		MANUAL, // manually update state machine: user must call update(delta)
	};

private:
	struct TransitionKeyHasher {
		static uint32_t hash(const TransitionKey &P) {
			uint64_t h1 = HashMapHasherDefault::hash(P.first);
			uint64_t h2 = HashMapHasherDefault::hash(P.second);
			return hash_one_uint64((h1 << 32) | h2);
		}
	};

	struct Transition {
		ObjectID from_state;
		ObjectID to_state;
		StringName event;

		inline bool is_valid() const { return to_state != ObjectID(); }

		static _FORCE_INLINE_ TransitionKey make_key(LimboState *p_from_state, const StringName &p_event) {
			return TransitionKey(
					p_from_state != nullptr ? uint64_t(p_from_state->get_instance_id()) : 0,
					p_event);
		}
	};

	UpdateMode update_mode;
	LimboState *initial_state;
	LimboState *active_state;
	LimboState *previous_active;
	LimboState *next_active;
	bool updating = false;

	HashMap<TransitionKey, Transition, TransitionKeyHasher> transitions;

	void _get_transition(LimboState *p_from_state, const StringName &p_event, Transition &r_transition) const;

protected:
	static void _bind_methods();

	void _notification(int p_what);
	void _validate_property(PropertyInfo &p_property) const;

	virtual void _initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard) override;
	virtual bool _dispatch(const StringName &p_event, const Variant &p_cargo = Variant()) override;

	virtual void _enter() override;
	virtual void _exit() override;
	virtual void _update(double p_delta) override;

public:
	void set_update_mode(UpdateMode p_mode) { update_mode = p_mode; }
	UpdateMode get_update_mode() const { return update_mode; }

	void set_active(bool p_active);

	void change_active_state(LimboState *p_state);

	LimboState *get_active_state() const { return active_state; }
	LimboState *get_previous_active_state() const { return previous_active; }
	LimboState *get_leaf_state() const;

	void set_initial_state(LimboState *p_state);
	LimboState *get_initial_state() const { return initial_state; }

	virtual void initialize(Node *p_agent, const Ref<Blackboard> &p_parent_scope = nullptr);

	void update(double p_delta);

	void add_transition(LimboState *p_from_state, LimboState *p_to_state, const StringName &p_event);
	void remove_transition(LimboState *p_from_state, const StringName &p_event);
	bool has_transition(LimboState *p_from_state, const StringName &p_event) const { return transitions.has(Transition::make_key(p_from_state, p_event)); }

	LimboState *anystate() const { return nullptr; }

	LimboHSM();
};

#endif // LIMBO_HSM_H
