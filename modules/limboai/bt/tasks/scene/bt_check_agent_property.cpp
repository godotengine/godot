/**
 * bt_check_agent_property.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_check_agent_property.h"

void BTCheckAgentProperty::set_property(StringName p_prop) {
	property = p_prop;
	emit_changed();
}

void BTCheckAgentProperty::set_check_type(LimboUtility::CheckType p_check_type) {
	check_type = p_check_type;
	emit_changed();
}

void BTCheckAgentProperty::set_value(Ref<BBVariant> p_value) {
	value = p_value;
	emit_changed();
	if (Engine::get_singleton()->is_editor_hint() && value.is_valid()) {
		value->connect(LW_NAME(changed), Callable(this, LW_NAME(emit_changed)));
	}
}

PackedStringArray BTCheckAgentProperty::get_configuration_warnings() {
	PackedStringArray warnings = BTCondition::get_configuration_warnings();
	if (property == StringName()) {
		warnings.append("`property` should be assigned.");
	}
	if (!value.is_valid()) {
		warnings.append("`value` should be assigned.");
	}
	return warnings;
}

String BTCheckAgentProperty::_generate_name() {
	if (property == StringName()) {
		return "CheckAgentProperty ???";
	}

	return vformat("Check if: agent.%s %s %s", property,
			LimboUtility::get_singleton()->get_check_operator_string(check_type),
			value.is_valid() ? Variant(value) : Variant("???"));
}

BT::Status BTCheckAgentProperty::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(property == StringName(), FAILURE, "BTCheckAgentProperty: `property` is not set.");
	ERR_FAIL_COND_V_MSG(!value.is_valid(), FAILURE, "BTCheckAgentProperty: `value` is not set.");

#ifdef LIMBOAI_MODULE
	bool r_valid;
	Variant left_value = get_agent()->get(property, &r_valid);
	ERR_FAIL_COND_V_MSG(r_valid == false, FAILURE, vformat("BTCheckAgentProperty: Agent has no property named \"%s\"", property));
#elif LIMBOAI_GDEXTENSION
	Variant left_value = get_agent()->get(property);
#endif

	Variant right_value = value->get_value(get_scene_root(), get_blackboard());

	return LimboUtility::get_singleton()->perform_check(check_type, left_value, right_value) ? SUCCESS : FAILURE;
}

void BTCheckAgentProperty::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_property", "property"), &BTCheckAgentProperty::set_property);
	ClassDB::bind_method(D_METHOD("get_property"), &BTCheckAgentProperty::get_property);
	ClassDB::bind_method(D_METHOD("set_check_type", "check_type"), &BTCheckAgentProperty::set_check_type);
	ClassDB::bind_method(D_METHOD("get_check_type"), &BTCheckAgentProperty::get_check_type);
	ClassDB::bind_method(D_METHOD("set_value", "value"), &BTCheckAgentProperty::set_value);
	ClassDB::bind_method(D_METHOD("get_value"), &BTCheckAgentProperty::get_value);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "property"), "set_property", "get_property");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "check_type", PROPERTY_HINT_ENUM, "Equal,Less Than,Less Than Or Equal,Greater Than,Greater Than Or Equal,Not Equal"), "set_check_type", "get_check_type");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "value", PROPERTY_HINT_RESOURCE_TYPE, "BBVariant"), "set_value", "get_value");
}
