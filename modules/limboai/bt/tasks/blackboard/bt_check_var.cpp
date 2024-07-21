/**
 * bt_check_var.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_check_var.h"

void BTCheckVar::set_variable(const StringName &p_variable) {
	variable = p_variable;
	emit_changed();
}

void BTCheckVar::set_check_type(LimboUtility::CheckType p_check_type) {
	check_type = p_check_type;
	emit_changed();
}

void BTCheckVar::set_value(const Ref<BBVariant> &p_value) {
	value = p_value;
	emit_changed();
	if (Engine::get_singleton()->is_editor_hint() && value.is_valid()) {
		value->connect(LW_NAME(changed), Callable(this, LW_NAME(emit_changed)));
	}
}

PackedStringArray BTCheckVar::get_configuration_warnings() {
	PackedStringArray warnings = BTCondition::get_configuration_warnings();
	if (variable == StringName()) {
		warnings.append("`variable` should be assigned.");
	}
	if (!value.is_valid()) {
		warnings.append("`value` should be assigned.");
	}
	return warnings;
}

String BTCheckVar::_generate_name() {
	if (variable == StringName()) {
		return "CheckVar ???";
	}

	return vformat("Check if: %s %s %s", LimboUtility::get_singleton()->decorate_var(variable),
			LimboUtility::get_singleton()->get_check_operator_string(check_type),
			value.is_valid() ? Variant(value) : Variant("???"));
}

BT::Status BTCheckVar::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(variable == StringName(), FAILURE, "BTCheckVar: `variable` is not set.");
	ERR_FAIL_COND_V_MSG(!value.is_valid(), FAILURE, "BTCheckVar: `value` is not set.");

	ERR_FAIL_COND_V_MSG(!get_blackboard()->has_var(variable), FAILURE, vformat("BTCheckVar: Blackboard variable doesn't exist: \"%s\". Returning FAILURE.", variable));

	Variant left_value = get_blackboard()->get_var(variable, Variant());
	Variant right_value = value->get_value(get_scene_root(), get_blackboard());

	return LimboUtility::get_singleton()->perform_check(check_type, left_value, right_value) ? SUCCESS : FAILURE;
}

void BTCheckVar::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_variable", "variable"), &BTCheckVar::set_variable);
	ClassDB::bind_method(D_METHOD("get_variable"), &BTCheckVar::get_variable);
	ClassDB::bind_method(D_METHOD("set_check_type", "check_type"), &BTCheckVar::set_check_type);
	ClassDB::bind_method(D_METHOD("get_check_type"), &BTCheckVar::get_check_type);
	ClassDB::bind_method(D_METHOD("set_value", "value"), &BTCheckVar::set_value);
	ClassDB::bind_method(D_METHOD("get_value"), &BTCheckVar::get_value);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "variable"), "set_variable", "get_variable");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "check_type", PROPERTY_HINT_ENUM, "Equal,Less Than,Less Than Or Equal,Greater Than,Greater Than Or Equal,Not Equal"), "set_check_type", "get_check_type");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "value", PROPERTY_HINT_RESOURCE_TYPE, "BBVariant"), "set_value", "get_value");
}
