/**
 * bt_new_scope.cpp
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_new_scope.h"

void BTNewScope::initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard) {
	ERR_FAIL_COND(p_agent == nullptr);
	ERR_FAIL_COND(p_blackboard == nullptr);

	Ref<Blackboard> bb = memnew(Blackboard);

	bb->set_data(blackboard_data.duplicate());
	bb->set_parent_scope(p_blackboard);

	BTDecorator::initialize(p_agent, bb);
}

BT::Status BTNewScope::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "BT decorator has no child.");
	return get_child(0)->execute(p_delta);
}

void BTNewScope::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_set_blackboard_data", "p_data"), &BTNewScope::_set_blackboard_data);
	ClassDB::bind_method(D_METHOD("_get_blackboard_data"), &BTNewScope::_get_blackboard_data);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "_blackboard_data"), "_set_blackboard_data", "_get_blackboard_data");
}
