/**
 * limbo_state.h
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
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

#include "../util/limbo_string_names.h"

#ifdef LIMBOAI_MODULE
#include "core/object/gdvirtual.gen.inc"
#include "core/object/object.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/variant/callable.h"
#include "core/variant/variant.h"
#include "scene/main/node.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/templates/hash_map.hpp>
#endif // LIMBOAI_GDEXTENSION

class LimboHSM;

class LimboState : public Node {
	GDCLASS(LimboState, Node);

private:
	Ref<BlackboardPlan> blackboard_plan;
	Node *agent;
	Ref<Blackboard> blackboard;
	HashMap<String, Callable> handlers;
	Callable guard_callable;

protected:
	friend LimboHSM;

	bool active;

	static void _bind_methods();

	void _notification(int p_what);

	virtual void _initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard);
	virtual bool _dispatch(const String &p_event, const Variant &p_cargo = Variant());

	virtual void _setup();
	virtual void _enter();
	virtual void _exit();
	virtual void _update(double p_delta);

#ifdef LIMBOAI_MODULE
	GDVIRTUAL0(_setup);
	GDVIRTUAL0(_enter);
	GDVIRTUAL0(_exit);
	GDVIRTUAL1(_update, double);
#endif // LIMBOAI_MODULE

public:
	void set_blackboard_plan(const Ref<BlackboardPlan> p_plan) { blackboard_plan = p_plan; }
	Ref<BlackboardPlan> get_blackboard_plan() const { return blackboard_plan; }

	Ref<Blackboard> get_blackboard() const { return blackboard; }

	Node *get_agent() const { return agent; }
	void set_agent(Node *p_agent) { agent = p_agent; }

	LimboState *named(String p_name);
	LimboState *call_on_enter(const Callable &p_callable);
	LimboState *call_on_exit(const Callable &p_callable);
	LimboState *call_on_update(const Callable &p_callable);

	void add_event_handler(const String &p_event, const Callable &p_handler);
	bool dispatch(const String &p_event, const Variant &p_cargo = Variant());

	_FORCE_INLINE_ String event_finished() const { return LW_NAME(EVENT_FINISHED); }
	LimboState *get_root() const;
	bool is_active() const { return active; }

	void set_guard(const Callable &p_guard_callable);
	void clear_guard();

	LimboState();
};

#endif // LIMBO_STATE_H
