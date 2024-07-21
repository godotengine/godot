/**
 * bt_state.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_state.h"

#include "../editor/debugger/limbo_debugger.h"
#include "../util/limbo_compat.h"
#include "../util/limbo_string_names.h"

#ifdef LIMBOAI_MODULE
#include "core/debugger/engine_debugger.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/engine_debugger.hpp>
#endif // LIMBOAI_GDEXTENSION

void BTState::set_behavior_tree(const Ref<BehaviorTree> &p_tree) {
	if (Engine::get_singleton()->is_editor_hint()) {
		if (behavior_tree.is_valid() && behavior_tree->is_connected(LW_NAME(plan_changed), callable_mp(this, &BTState::_update_blackboard_plan))) {
			behavior_tree->disconnect(LW_NAME(plan_changed), callable_mp(this, &BTState::_update_blackboard_plan));
		}
		if (p_tree.is_valid()) {
			p_tree->connect(LW_NAME(plan_changed), callable_mp(this, &BTState::_update_blackboard_plan));
		}
		behavior_tree = p_tree;
	} else {
		behavior_tree = p_tree;
	}
	_update_blackboard_plan();
}

void BTState::_update_blackboard_plan() {
	if (get_blackboard_plan().is_null()) {
		set_blackboard_plan(memnew(BlackboardPlan));
	} else if (!RESOURCE_IS_BUILT_IN(get_blackboard_plan())) {
		WARN_PRINT_ED("BTState: Using external resource for derived blackboard plan is not supported. Converted to built-in resource.");
		set_blackboard_plan(get_blackboard_plan()->duplicate());
	} else {
		get_blackboard_plan()->set_base_plan(behavior_tree.is_valid() ? behavior_tree->get_blackboard_plan() : nullptr);
	}
}

void BTState::_setup() {
	LimboState::_setup();
	ERR_FAIL_COND_MSG(behavior_tree.is_null(), "BTState: BehaviorTree is not assigned.");
	Node *scene_root = get_owner();
	ERR_FAIL_NULL_MSG(scene_root, "BTState: Initialization failed - can't get scene root (make sure the BTState's owner property is set).");
	tree_instance = behavior_tree->instantiate(get_agent(), get_blackboard(), scene_root);

#ifdef DEBUG_ENABLED
	if (tree_instance.is_valid() && IS_DEBUGGER_ACTIVE()) {
		LimboDebugger::get_singleton()->register_bt_instance(tree_instance, get_path());
	}
#endif
}

void BTState::_exit() {
	if (tree_instance.is_valid()) {
		tree_instance->abort();
	} else {
		ERR_PRINT_ONCE("BTState: BehaviorTree is not assigned.");
	}
	LimboState::_exit();
}

void BTState::_update(double p_delta) {
	GDVIRTUAL_CALL(_update, p_delta);
	if (!is_active()) {
		// Bail out if a transition happened in the meantime.
		return;
	}
	ERR_FAIL_NULL(tree_instance);
	int status = tree_instance->execute(p_delta);
	if (status == BTTask::SUCCESS) {
		get_root()->dispatch(success_event, Variant());
	} else if (status == BTTask::FAILURE) {
		get_root()->dispatch(failure_event, Variant());
	}
	emit_signal(LW_NAME(updated), p_delta);
}

void BTState::_notification(int p_notification) {
	switch (p_notification) {
#ifdef DEBUG_ENABLED
		case NOTIFICATION_ENTER_TREE: {
			if (tree_instance.is_valid() && IS_DEBUGGER_ACTIVE()) {
				LimboDebugger::get_singleton()->register_bt_instance(tree_instance, get_path());
			}
		} break;
#endif // DEBUG_ENABLED
		case NOTIFICATION_EXIT_TREE: {
#ifdef DEBUG_ENABLED
			if (tree_instance.is_valid() && IS_DEBUGGER_ACTIVE()) {
				LimboDebugger::get_singleton()->unregister_bt_instance(tree_instance, get_path());
			}
#endif // DEBUG_ENABLED

			if (Engine::get_singleton()->is_editor_hint()) {
				if (behavior_tree.is_valid() && behavior_tree->is_connected(LW_NAME(plan_changed), callable_mp(this, &BTState::_update_blackboard_plan))) {
					behavior_tree->disconnect(LW_NAME(plan_changed), callable_mp(this, &BTState::_update_blackboard_plan));
				}
			}
		} break;
	}
}

void BTState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_behavior_tree", "behavior_tree"), &BTState::set_behavior_tree);
	ClassDB::bind_method(D_METHOD("get_behavior_tree"), &BTState::get_behavior_tree);

	ClassDB::bind_method(D_METHOD("get_tree_instance"), &BTState::get_tree_instance);

	ClassDB::bind_method(D_METHOD("set_success_event", "event"), &BTState::set_success_event);
	ClassDB::bind_method(D_METHOD("get_success_event"), &BTState::get_success_event);

	ClassDB::bind_method(D_METHOD("set_failure_event", "event"), &BTState::set_failure_event);
	ClassDB::bind_method(D_METHOD("get_failure_event"), &BTState::get_failure_event);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "behavior_tree", PROPERTY_HINT_RESOURCE_TYPE, "BehaviorTree"), "set_behavior_tree", "get_behavior_tree");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "success_event"), "set_success_event", "get_success_event");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "failure_event"), "set_failure_event", "get_failure_event");
}

BTState::BTState() {
	success_event = LW_NAME(EVENT_SUCCESS);
	failure_event = LW_NAME(EVENT_FAILURE);
}
