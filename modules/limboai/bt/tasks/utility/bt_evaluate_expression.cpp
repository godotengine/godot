/**
 * bt_evaluate_expression.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 * Copyright 2024 Wilson E. Alvarez
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_evaluate_expression.h"

#include "../../../util/limbo_compat.h"
#include "../../../util/limbo_utility.h"

#ifdef LIMBOAI_GDEXTENSION
#include "godot_cpp/classes/global_constants.hpp"
#endif // LIMBOAI_GDEXTENSION

//**** Setters / Getters

void BTEvaluateExpression::set_expression_string(const String &p_expression_string) {
	expression_string = p_expression_string;
	emit_changed();
}

void BTEvaluateExpression::set_node_param(Ref<BBNode> p_object) {
	node_param = p_object;
	emit_changed();
	if (Engine::get_singleton()->is_editor_hint() && node_param.is_valid()) {
		node_param->connect(LW_NAME(changed), Callable(this, LW_NAME(emit_changed)));
	}
}

void BTEvaluateExpression::set_input_include_delta(bool p_input_include_delta) {
	if (input_include_delta != p_input_include_delta) {
		processed_input_values.resize(input_values.size() + int(p_input_include_delta));
	}
	input_include_delta = p_input_include_delta;
	emit_changed();
}

void BTEvaluateExpression::set_input_names(const PackedStringArray &p_input_names) {
	input_names = p_input_names;
	emit_changed();
}

void BTEvaluateExpression::set_input_values(const TypedArray<BBVariant> &p_input_values) {
	if (input_values.size() != p_input_values.size()) {
		processed_input_values.resize(p_input_values.size() + int(input_include_delta));
	}
	input_values = p_input_values;
	emit_changed();
}

void BTEvaluateExpression::set_result_var(const StringName &p_result_var) {
	result_var = p_result_var;
	emit_changed();
}

//**** Task Implementation

PackedStringArray BTEvaluateExpression::get_configuration_warnings() {
	PackedStringArray warnings = BTAction::get_configuration_warnings();
	if (expression_string.is_empty()) {
		warnings.append("Expression string is not set.");
	}
	if (node_param.is_null()) {
		warnings.append("Node parameter is not set.");
	} else if (node_param->get_value_source() == BBParam::SAVED_VALUE && node_param->get_saved_value() == Variant()) {
		warnings.append("Path to node is not set.");
	} else if (node_param->get_value_source() == BBParam::BLACKBOARD_VAR && node_param->get_variable() == String()) {
		warnings.append("Node blackboard variable is not set.");
	}
	return warnings;
}

void BTEvaluateExpression::_setup() {
	parse();
	ERR_FAIL_COND_MSG(is_parsed != Error::OK, "BTEvaluateExpression: Failed to parse expression: " + expression->get_error_text());
}

Error BTEvaluateExpression::parse() {
	PackedStringArray processed_input_names;
	processed_input_names.resize(input_names.size() + int(input_include_delta));
	String *processed_input_names_ptr = processed_input_names.ptrw();
	if (input_include_delta) {
		processed_input_names_ptr[0] = "delta";
	}
	for (int i = 0; i < input_names.size(); ++i) {
		processed_input_names_ptr[i + int(input_include_delta)] = input_names[i];
	}

	is_parsed = expression->parse(expression_string, processed_input_names);
	return is_parsed;
}

String BTEvaluateExpression::_generate_name() {
	return vformat("EvaluateExpression %s  node: %s  %s",
			!expression_string.is_empty() ? expression_string : "???",
			node_param.is_valid() && !node_param->to_string().is_empty() ? node_param->to_string() : "???",
			result_var == StringName() ? "" : LimboUtility::get_singleton()->decorate_output_var(result_var));
}

BT::Status BTEvaluateExpression::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(expression_string.is_empty(), FAILURE, "BTEvaluateExpression: Expression String is not set.");
	ERR_FAIL_COND_V_MSG(node_param.is_null(), FAILURE, "BTEvaluateExpression: Node parameter is not set.");
	Object *obj = node_param->get_value(get_scene_root(), get_blackboard());
	ERR_FAIL_COND_V_MSG(obj == nullptr, FAILURE, "BTEvaluateExpression: Failed to get object: " + node_param->to_string());
	ERR_FAIL_COND_V_MSG(is_parsed != Error::OK, FAILURE, "BTEvaluateExpression: Failed to parse expression: " + expression->get_error_text());

	if (input_include_delta) {
		processed_input_values[0] = p_delta;
	}
	for (int i = 0; i < input_values.size(); ++i) {
		const Ref<BBVariant> &bb_variant = input_values[i];
		processed_input_values[i + int(input_include_delta)] = bb_variant->get_value(get_scene_root(), get_blackboard());
	}

	Variant result = expression->execute(processed_input_values, obj, false);
	ERR_FAIL_COND_V_MSG(expression->has_execute_failed(), FAILURE, "BTEvaluateExpression: Failed to execute: " + expression->get_error_text());

	if (result_var != StringName()) {
		get_blackboard()->set_var(result_var, result);
	}

	return SUCCESS;
}

//**** Godot

void BTEvaluateExpression::_bind_methods() {
	ClassDB::bind_method(D_METHOD("parse"), &BTEvaluateExpression::parse);
	ClassDB::bind_method(D_METHOD("set_expression_string", "expression_string"), &BTEvaluateExpression::set_expression_string);
	ClassDB::bind_method(D_METHOD("get_expression_string"), &BTEvaluateExpression::get_expression_string);
	ClassDB::bind_method(D_METHOD("set_node_param", "param"), &BTEvaluateExpression::set_node_param);
	ClassDB::bind_method(D_METHOD("get_node_param"), &BTEvaluateExpression::get_node_param);
	ClassDB::bind_method(D_METHOD("set_input_names", "input_names"), &BTEvaluateExpression::set_input_names);
	ClassDB::bind_method(D_METHOD("get_input_names"), &BTEvaluateExpression::get_input_names);
	ClassDB::bind_method(D_METHOD("set_input_values", "input_values"), &BTEvaluateExpression::set_input_values);
	ClassDB::bind_method(D_METHOD("get_input_values"), &BTEvaluateExpression::get_input_values);
	ClassDB::bind_method(D_METHOD("set_input_include_delta", "input_include_delta"), &BTEvaluateExpression::set_input_include_delta);
	ClassDB::bind_method(D_METHOD("is_input_delta_included"), &BTEvaluateExpression::is_input_delta_included);
	ClassDB::bind_method(D_METHOD("set_result_var", "variable"), &BTEvaluateExpression::set_result_var);
	ClassDB::bind_method(D_METHOD("get_result_var"), &BTEvaluateExpression::get_result_var);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "BBNode"), "set_node_param", "get_node_param");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "expression_string"), "set_expression_string", "get_expression_string");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "result_var"), "set_result_var", "get_result_var");
	ADD_GROUP("Inputs", "input_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "input_include_delta"), "set_input_include_delta", "is_input_delta_included");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "input_names", PROPERTY_HINT_ARRAY_TYPE, "String"), "set_input_names", "get_input_names");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "input_values", PROPERTY_HINT_ARRAY_TYPE, RESOURCE_TYPE_HINT("BBVariant")), "set_input_values", "get_input_values");
}

BTEvaluateExpression::BTEvaluateExpression() {
	expression.instantiate();
}
