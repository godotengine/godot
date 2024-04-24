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

class LimboHSM : public LimboState {
	GDCLASS(LimboHSM, LimboState);

public:
	enum UpdateMode : unsigned int {
		IDLE, // automatically call update() during NOTIFICATION_PROCESS
		PHYSICS, // automatically call update() during NOTIFICATION_PHYSICS
		MANUAL, // manually update state machine: user must call update(delta)
	};

private:
	UpdateMode update_mode;
	LimboState *initial_state;
	LimboState *active_state;
	LimboState *previous_active;
	LimboState *next_active;
	HashMap<uint64_t, LimboState *> transitions;
	bool updating = false;

	_FORCE_INLINE_ uint64_t _get_transition_key(LimboState *p_from_state, const StringName &p_event) {
		uint64_t key = hash_djb2_one_64(Variant::OBJECT);
		if (p_from_state != nullptr) {
			key = hash_djb2_one_64(hash_one_uint64(hash_make_uint64_t(p_from_state)), key);
		}
		key = hash_djb2_one_64(p_event.hash(), key);
		return key;
	}

protected:
	static void _bind_methods();

	void _notification(int p_what);

	virtual void _initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard) override;
	virtual bool _dispatch(const StringName &p_event, const Variant &p_cargo = Variant()) override;

	virtual void _enter() override;
	virtual void _exit() override;
	virtual void _update(double p_delta) override;

	void _change_state(LimboState *p_state);

public:
	void set_update_mode(UpdateMode p_mode) { update_mode = p_mode; }
	UpdateMode get_update_mode() const { return update_mode; }

	LimboState *get_active_state() const { return active_state; }
	LimboState *get_previous_active_state() const { return previous_active; }
	LimboState *get_leaf_state() const;
	void set_active(bool p_active);

	void set_initial_state(LimboState *p_state);
	LimboState *get_initial_state() const { return initial_state; }

	virtual void initialize(Node *p_agent, const Ref<Blackboard> &p_parent_scope = nullptr);

	void update(double p_delta);
	void add_transition(LimboState *p_from_state, LimboState *p_to_state, const StringName &p_event);
	LimboState *anystate() const { return nullptr; }

	LimboHSM();
};

#endif // LIMBO_HSM_H
