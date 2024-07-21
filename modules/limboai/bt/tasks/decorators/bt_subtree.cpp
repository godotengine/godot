/**
 * bt_subtree.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_subtree.h"

void BTSubtree::set_subtree(const Ref<BehaviorTree> &p_subtree) {
	if (Engine::get_singleton()->is_editor_hint()) {
		if (subtree.is_valid() && subtree->is_connected(LW_NAME(changed), callable_mp(this, &BTSubtree::_update_blackboard_plan))) {
			subtree->disconnect(LW_NAME(changed), callable_mp(this, &BTSubtree::_update_blackboard_plan));
		}
		if (p_subtree.is_valid()) {
			p_subtree->connect(LW_NAME(changed), callable_mp(this, &BTSubtree::_update_blackboard_plan));
		}
	}
	subtree = p_subtree;
	_update_blackboard_plan();
	emit_changed();
}

void BTSubtree::_update_blackboard_plan() {
	if (get_blackboard_plan().is_null()) {
		set_blackboard_plan(Ref<BlackboardPlan>(memnew(BlackboardPlan)));
	}
	get_blackboard_plan()->set_base_plan(subtree.is_valid() ? subtree->get_blackboard_plan() : nullptr);
}

String BTSubtree::_generate_name() {
	String s;
	if (subtree.is_null()) {
		s = "(unassigned)";
	} else if (subtree->get_path().is_empty()) {
		s = "(unsaved)";
	} else {
		s = vformat("\"%s\"", subtree->get_path());
	}
	return vformat("Subtree %s", s);
}

void BTSubtree::initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard, Node *p_scene_root) {
	ERR_FAIL_COND_MSG(!subtree.is_valid(), "Subtree is not assigned.");
	ERR_FAIL_COND_MSG(!subtree->get_root_task().is_valid(), "Subtree root task is not valid.");
	ERR_FAIL_COND_MSG(get_child_count() != 0, "Subtree task shouldn't have children during initialization.");

	add_child(subtree->get_root_task()->clone());

	BTNewScope::initialize(p_agent, p_blackboard, p_scene_root);
}

BT::Status BTSubtree::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "BT decorator doesn't have a child.");
	return get_child(0)->execute(p_delta);
}

PackedStringArray BTSubtree::get_configuration_warnings() {
	PackedStringArray warnings = BTTask::get_configuration_warnings(); // ! BTDecorator skipped intentionally
	if (subtree.is_null()) {
		warnings.append("Subtree needs to be assigned.");
	}
	return warnings;
}

void BTSubtree::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_subtree", "behavior_tree"), &BTSubtree::set_subtree);
	ClassDB::bind_method(D_METHOD("get_subtree"), &BTSubtree::get_subtree);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "subtree", PROPERTY_HINT_RESOURCE_TYPE, "BehaviorTree"), "set_subtree", "get_subtree");
}

BTSubtree::~BTSubtree() {
	if (Engine::get_singleton()->is_editor_hint()) {
		if (subtree.is_valid() && subtree->is_connected(LW_NAME(changed), callable_mp(this, &BTSubtree::_update_blackboard_plan))) {
			subtree->disconnect(LW_NAME(changed), callable_mp(this, &BTSubtree::_update_blackboard_plan));
		}
	}
}
