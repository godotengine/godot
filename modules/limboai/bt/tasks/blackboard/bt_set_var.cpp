/**
 * bt_set_var.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_set_var.h"

String BTSetVar::_generate_name() {
	if (variable == StringName()) {
		return "SetVar ???";
	}
	return vformat("Set %s %s= %s",
			LimboUtility::get_singleton()->decorate_var(variable),
			LimboUtility::get_singleton()->get_operation_string(operation),
			value.is_valid() ? Variant(value) : Variant("???"));
}

BT::Status BTSetVar::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(variable == StringName(), FAILURE, "BTSetVar: `variable` is not set.");
	ERR_FAIL_COND_V_MSG(!value.is_valid(), FAILURE, "BTSetVar: `value` is not set.");
	Variant result;
	Variant error_result = LW_NAME(error_value);
	Variant right_value = value->get_value(get_scene_root(), get_blackboard(), error_result);
	ERR_FAIL_COND_V_MSG(right_value == error_result, FAILURE, "BTSetVar: Failed to get parameter value. Returning FAILURE.");
	if (operation == LimboUtility::OPERATION_NONE) {
		result = right_value;
	} else if (operation != LimboUtility::OPERATION_NONE) {
		Variant left_value = get_blackboard()->get_var(variable, error_result);
		ERR_FAIL_COND_V_MSG(left_value == error_result, FAILURE, vformat("BTSetVar: Failed to get \"%s\" blackboard variable. Returning FAILURE.", variable));
		result = LimboUtility::get_singleton()->perform_operation(operation, left_value, right_value);
		ERR_FAIL_COND_V_MSG(result == Variant(), FAILURE, "BTSetVar: Operation not valid. Returning FAILURE.");
	}
	get_blackboard()->set_var(variable, result);
	return SUCCESS;
};

void BTSetVar::set_variable(const StringName &p_variable) {
	variable = p_variable;
	emit_changed();
}

void BTSetVar::set_value(const Ref<BBVariant> &p_value) {
	value = p_value;
	emit_changed();
	if (Engine::get_singleton()->is_editor_hint() && value.is_valid()) {
		value->connect(LW_NAME(changed), Callable(this, LW_NAME(emit_changed)));
	}
}

void BTSetVar::set_operation(LimboUtility::Operation p_operation) {
	operation = p_operation;
	emit_changed();
}

PackedStringArray BTSetVar::get_configuration_warnings() {
	PackedStringArray warnings = BTAction::get_configuration_warnings();
	if (variable == StringName()) {
		warnings.append("`variable` should be assigned.");
	}
	if (!value.is_valid()) {
		warnings.append("`value` should be assigned.");
	}
	return warnings;
}

void BTSetVar::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_variable", "variable"), &BTSetVar::set_variable);
	ClassDB::bind_method(D_METHOD("get_variable"), &BTSetVar::get_variable);
	ClassDB::bind_method(D_METHOD("set_value", "value"), &BTSetVar::set_value);
	ClassDB::bind_method(D_METHOD("get_value"), &BTSetVar::get_value);
	ClassDB::bind_method(D_METHOD("get_operation"), &BTSetVar::get_operation);
	ClassDB::bind_method(D_METHOD("set_operation", "operation"), &BTSetVar::set_operation);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "variable"), "set_variable", "get_variable");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "value", PROPERTY_HINT_RESOURCE_TYPE, "BBVariant"), "set_value", "get_value");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "operation", PROPERTY_HINT_ENUM, "None,Addition,Subtraction,Multiplication,Division,Modulo,Power,Bitwise Shift Left,Bitwise Shift Right,Bitwise AND,Bitwise OR,Bitwise XOR"), "set_operation", "get_operation");
}
