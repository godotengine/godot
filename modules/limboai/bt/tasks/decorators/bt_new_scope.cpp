/**
 * bt_new_scope.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_new_scope.h"

#include "../../behavior_tree.h"

void BTNewScope::set_blackboard_plan(const Ref<BlackboardPlan> &p_plan) {
	blackboard_plan = p_plan;
	if (blackboard_plan.is_null()) {
		blackboard_plan.instantiate();
	}

	_update_blackboard_plan();

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		callable_mp(this, &BTNewScope::_set_parent_scope_plan_from_bt).call_deferred();
	}
#endif // TOOLS_ENABLED

	emit_changed();
}

#ifdef TOOLS_ENABLED
void BTNewScope::_set_parent_scope_plan_from_bt() {
	ERR_FAIL_NULL(get_blackboard_plan());
	Ref<BehaviorTree> bt = get_root()->editor_get_behavior_tree();
	ERR_FAIL_NULL(bt);
	get_blackboard_plan()->set_parent_scope_plan_provider(callable_mp(bt.ptr(), &BehaviorTree::get_blackboard_plan));
}
#endif // TOOLS_ENABLED

void BTNewScope::initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard, Node *p_scene_root) {
	ERR_FAIL_COND(p_agent == nullptr);
	ERR_FAIL_COND(p_blackboard == nullptr);

	Ref<Blackboard> bb;
	if (blackboard_plan.is_valid()) {
		bb = blackboard_plan->create_blackboard(p_agent, p_blackboard);
	} else {
		bb = Ref<Blackboard>(memnew(Blackboard));
		bb->set_parent(p_blackboard);
	}
	BTDecorator::initialize(p_agent, bb, p_scene_root);
}

BT::Status BTNewScope::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "BT decorator has no child.");
	return get_child(0)->execute(p_delta);
}

void BTNewScope::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_blackboard_plan", "plan"), &BTNewScope::set_blackboard_plan);
	ClassDB::bind_method(D_METHOD("get_blackboard_plan"), &BTNewScope::get_blackboard_plan);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan", PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT), "set_blackboard_plan", "get_blackboard_plan");
}
