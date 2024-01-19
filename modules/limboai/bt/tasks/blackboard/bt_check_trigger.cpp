/**
 * bt_check_trigger.cpp
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_check_trigger.h"

#include "../../../util/limbo_utility.h"

void BTCheckTrigger::set_variable(String p_variable) {
	variable = p_variable;
	emit_changed();
}

PackedStringArray BTCheckTrigger::get_configuration_warnings() {
	PackedStringArray warnings = BTCondition::get_configuration_warnings();
	if (variable.is_empty()) {
		warnings.append("Variable is not set.");
	}
	return warnings;
}

String BTCheckTrigger::_generate_name() {
	if (variable.is_empty()) {
		return "CheckTrigger ???";
	}
	return "CheckTrigger " + LimboUtility::get_singleton()->decorate_var(variable);
}

BT::Status BTCheckTrigger::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(variable.is_empty(), FAILURE, "BBCheckVar: `variable` is not set.");
	Variant trigger_value = get_blackboard()->get_var(variable, false);
	if (trigger_value == Variant(true)) {
		get_blackboard()->set_var(variable, false);
		return SUCCESS;
	}
	return FAILURE;
}

void BTCheckTrigger::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_variable", "p_variable"), &BTCheckTrigger::set_variable);
	ClassDB::bind_method(D_METHOD("get_variable"), &BTCheckTrigger::get_variable);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "variable"), "set_variable", "get_variable");
}
