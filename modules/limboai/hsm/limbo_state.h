/**
 * limbo_state.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef LIMBO_STATE_H
#define LIMBO_STATE_H

#include "../blackboard/blackboard.h"
#include "../blackboard/blackboard_plan.h"

#include "../util/limbo_compat.h"
#include "../util/limbo_string_names.h"

#ifdef LIMBOAI_MODULE
#include "scene/main/node.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/core/gdvirtual.gen.inc>
#include <godot_cpp/templates/hash_map.hpp>
#endif // LIMBOAI_GDEXTENSION

class LimboHSM;

class LimboState : public Node {
	GDCLASS(LimboState, Node);

private:
	bool active;
	Ref<BlackboardPlan> blackboard_plan;
	Node *agent;
	Ref<Blackboard> blackboard;
	HashMap<StringName, Callable> handlers;
	Callable guard_callable;

	Ref<BlackboardPlan> _get_parent_scope_plan() const;

protected:
	friend LimboHSM;

	static void _bind_methods();

	void _notification(int p_what);

	virtual void _initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard);
	virtual bool _dispatch(const StringName &p_event, const Variant &p_cargo = Variant());

	virtual bool _should_use_new_scope() const { return blackboard_plan.is_valid() || is_root(); }
	virtual void _update_blackboard_plan();

	virtual void _setup();
	virtual void _enter();
	virtual void _exit();
	virtual void _update(double p_delta);

	GDVIRTUAL0(_setup);
	GDVIRTUAL0(_enter);
	GDVIRTUAL0(_exit);
	GDVIRTUAL1(_update, double);

public:
	void set_blackboard_plan(const Ref<BlackboardPlan> &p_plan);
	_FORCE_INLINE_ Ref<BlackboardPlan> get_blackboard_plan() const { return blackboard_plan; }

	Ref<Blackboard> get_blackboard() const { return blackboard; }

	Node *get_agent() const { return agent; }
	void set_agent(Node *p_agent) { agent = p_agent; }

	LimboState *named(const String &p_name);
	LimboState *call_on_enter(const Callable &p_callable);
	LimboState *call_on_exit(const Callable &p_callable);
	LimboState *call_on_update(const Callable &p_callable);

	void add_event_handler(const StringName &p_event, const Callable &p_handler);
	bool dispatch(const StringName &p_event, const Variant &p_cargo = Variant());

	_FORCE_INLINE_ StringName event_finished() const { return LW_NAME(EVENT_FINISHED); }
	LimboState *get_root() const;
	_FORCE_INLINE_ bool is_root() const { return !(get_parent() && IS_CLASS(get_parent(), LimboState)); }
	_FORCE_INLINE_ bool is_active() const { return active; }

	void set_guard(const Callable &p_guard_callable);
	void clear_guard();

	LimboState();
};

#endif // LIMBO_STATE_H
