/**
 * bt_set_agent_property.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_set_agent_property.h"

void BTSetAgentProperty::set_property(StringName p_prop) {
	property = p_prop;
	emit_changed();
}

void BTSetAgentProperty::set_value(Ref<BBVariant> p_value) {
	value = p_value;
	emit_changed();
	if (Engine::get_singleton()->is_editor_hint() && value.is_valid()) {
		value->connect(LW_NAME(changed), Callable(this, LW_NAME(emit_changed)));
	}
}

void BTSetAgentProperty::set_operation(LimboUtility::Operation p_operation) {
	operation = p_operation;
	emit_changed();
}

PackedStringArray BTSetAgentProperty::get_configuration_warnings() {
	PackedStringArray warnings = BTAction::get_configuration_warnings();
	if (property == StringName()) {
		warnings.append("`property` should be assigned.");
	}
	if (!value.is_valid()) {
		warnings.append("`value` should be assigned.");
	}
	return warnings;
}

String BTSetAgentProperty::_generate_name() {
	if (property == StringName()) {
		return "SetAgentProperty ???";
	}

	return vformat("Set agent.%s = %s", property,
			value.is_valid() ? Variant(value) : Variant("???"));
}

BT::Status BTSetAgentProperty::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(property == StringName(), FAILURE, "BTSetAgentProperty: `property` is not set.");
	ERR_FAIL_COND_V_MSG(!value.is_valid(), FAILURE, "BTSetAgentProperty: `value` is not set.");

	Variant result;
	StringName error_value = LW_NAME(error_value);
	Variant right_value = value->get_value(get_scene_root(), get_blackboard(), error_value);
	ERR_FAIL_COND_V_MSG(right_value == Variant(error_value), FAILURE, "BTSetAgentProperty: Couldn't get value of value-parameter.");
	bool r_valid;
	if (operation == LimboUtility::OPERATION_NONE) {
		result = right_value;
	} else {
#ifdef LIMBOAI_MODULE
		Variant left_value = get_agent()->get(property, &r_valid);
		ERR_FAIL_COND_V_MSG(!r_valid, FAILURE, vformat("BTSetAgentProperty: Failed to get agent's \"%s\" property. Returning FAILURE.", property));
#elif LIMBOAI_GDEXTENSION
		Variant left_value = get_agent()->get(property);
#endif
		result = LimboUtility::get_singleton()->perform_operation(operation, left_value, right_value);
		ERR_FAIL_COND_V_MSG(result == Variant(), FAILURE, "BTSetAgentProperty: Operation not valid. Returning FAILURE.");
	}

#ifdef LIMBOAI_MODULE
	get_agent()->set(property, result, &r_valid);
	ERR_FAIL_COND_V_MSG(!r_valid, FAILURE, vformat("BTSetAgentProperty: Couldn't set property \"%s\" with value \"%s\"", property, result));
#elif LIMBOAI_GDEXTENSION
	get_agent()->set(property, result);
#endif
	return SUCCESS;
}

void BTSetAgentProperty::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_property", "property"), &BTSetAgentProperty::set_property);
	ClassDB::bind_method(D_METHOD("get_property"), &BTSetAgentProperty::get_property);
	ClassDB::bind_method(D_METHOD("set_value", "value"), &BTSetAgentProperty::set_value);
	ClassDB::bind_method(D_METHOD("get_value"), &BTSetAgentProperty::get_value);
	ClassDB::bind_method(D_METHOD("set_operation", "operation"), &BTSetAgentProperty::set_operation);
	ClassDB::bind_method(D_METHOD("get_operation"), &BTSetAgentProperty::get_operation);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "property"), "set_property", "get_property");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "value", PROPERTY_HINT_RESOURCE_TYPE, "BBVariant"), "set_value", "get_value");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "operation", PROPERTY_HINT_ENUM, "None,Addition,Subtraction,Multiplication,Division,Modulo,Power,Bitwise Shift Left,Bitwise Shift Right,Bitwise AND,Bitwise OR,Bitwise XOR"), "set_operation", "get_operation");
}
