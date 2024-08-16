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

void BTState::set_scene_root_hint(Node *p_scene_root) {
	ERR_FAIL_NULL_MSG(p_scene_root, "BTState: Failed to set scene root hint - scene root is null.");
	ERR_FAIL_COND_MSG(bt_instance.is_valid(), "BTState: Scene root hint shouldn't be set after initialization. This change will not affect the current behavior tree instance.");

	scene_root_hint = p_scene_root;
}

void BTState::set_monitor_performance(bool p_monitor) {
	monitor_performance = p_monitor;

#ifdef DEBUG_ENABLED
	if (bt_instance.is_valid()) {
		bt_instance->set_monitor_performance(monitor_performance);
	}
#endif
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
	Node *scene_root = scene_root_hint ? scene_root_hint : get_owner();
	ERR_FAIL_NULL_MSG(scene_root, "BTState: Initialization failed - unable to establish scene root. This is likely due to BTState not being owned by a scene node. Check BTState.set_scene_root_hint().");
	bt_instance = behavior_tree->instantiate(get_agent(), get_blackboard(), this);
	ERR_FAIL_COND_MSG(bt_instance.is_null(), "BTState: Initialization failed - failed to instantiate behavior tree.");

#ifdef DEBUG_ENABLED
	bt_instance->register_with_debugger();
	bt_instance->set_monitor_performance(monitor_performance);
#endif
}

void BTState::_exit() {
	if (bt_instance.is_valid()) {
		bt_instance->get_root_task()->abort();
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
	ERR_FAIL_NULL(bt_instance);
	BT::Status status = bt_instance->update(p_delta);
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
			if (bt_instance.is_valid()) {
				bt_instance->register_with_debugger();
				bt_instance->set_monitor_performance(monitor_performance);
			}
		} break;
#endif // DEBUG_ENABLED
		case NOTIFICATION_EXIT_TREE: {
#ifdef DEBUG_ENABLED
			if (bt_instance.is_valid()) {
				bt_instance->unregister_with_debugger();
				bt_instance->set_monitor_performance(false);
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

	ClassDB::bind_method(D_METHOD("get_bt_instance"), &BTState::get_bt_instance);

	ClassDB::bind_method(D_METHOD("set_success_event", "event"), &BTState::set_success_event);
	ClassDB::bind_method(D_METHOD("get_success_event"), &BTState::get_success_event);

	ClassDB::bind_method(D_METHOD("set_failure_event", "event"), &BTState::set_failure_event);
	ClassDB::bind_method(D_METHOD("get_failure_event"), &BTState::get_failure_event);

	ClassDB::bind_method(D_METHOD("set_monitor_performance", "enable"), &BTState::set_monitor_performance);
	ClassDB::bind_method(D_METHOD("get_monitor_performance"), &BTState::get_monitor_performance);

	ClassDB::bind_method(D_METHOD("set_scene_root_hint", "scene_root"), &BTState::set_scene_root_hint);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "behavior_tree", PROPERTY_HINT_RESOURCE_TYPE, "BehaviorTree"), "set_behavior_tree", "get_behavior_tree");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "success_event"), "set_success_event", "get_success_event");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "failure_event"), "set_failure_event", "get_failure_event");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "monitor_performance"), "set_monitor_performance", "get_monitor_performance");
}

BTState::BTState() {
	success_event = LW_NAME(EVENT_SUCCESS);
	failure_event = LW_NAME(EVENT_FAILURE);
}
