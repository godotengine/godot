/**
 * bt_call_method.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_call_method.h"

#include "../../../util/limbo_compat.h"
#include "../../../util/limbo_utility.h"

#ifdef LIMBOAI_GDEXTENSION
#include "godot_cpp/classes/global_constants.hpp"
#endif // LIMBOAI_GDEXTENSION

//**** Setters / Getters

void BTCallMethod::set_method(const StringName &p_method_name) {
	method = p_method_name;
	emit_changed();
}

void BTCallMethod::set_node_param(const Ref<BBNode> &p_object) {
	node_param = p_object;
	emit_changed();
	if (Engine::get_singleton()->is_editor_hint() && node_param.is_valid()) {
		node_param->connect(LW_NAME(changed), Callable(this, LW_NAME(emit_changed)));
	}
}

void BTCallMethod::set_include_delta(bool p_include_delta) {
	include_delta = p_include_delta;
	emit_changed();
}

void BTCallMethod::set_args(TypedArray<BBVariant> p_args) {
	args = p_args;
	emit_changed();
}

void BTCallMethod::set_result_var(const StringName &p_result_var) {
	result_var = p_result_var;
	emit_changed();
}

//**** Task Implementation

PackedStringArray BTCallMethod::get_configuration_warnings() {
	PackedStringArray warnings = BTAction::get_configuration_warnings();
	if (method == StringName()) {
		warnings.append("Method Name is not set.");
	}
	if (node_param.is_null()) {
		warnings.append("Node parameter is not set.");
	} else if (node_param->get_value_source() == BBParam::SAVED_VALUE && node_param->get_saved_value() == Variant()) {
		warnings.append("Path to node is not set.");
	} else if (node_param->get_value_source() == BBParam::BLACKBOARD_VAR && node_param->get_variable() == StringName()) {
		warnings.append("Node blackboard variable is not set.");
	}
	return warnings;
}

String BTCallMethod::_generate_name() {
	String args_str = include_delta ? "delta" : "";
	if (args.size() > 0) {
		if (!args_str.is_empty()) {
			args_str += ", ";
		}
		args_str += vformat("%s", args).trim_prefix("[").trim_suffix("]");
	}
	return vformat("CallMethod %s(%s)  node: %s  %s",
			method != StringName() ? method : "???",
			args_str,
			node_param.is_valid() && !node_param->to_string().is_empty() ? node_param->to_string() : "???",
			result_var == StringName() ? "" : LimboUtility::get_singleton()->decorate_output_var(result_var));
}

BT::Status BTCallMethod::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(method == StringName(), FAILURE, "BTCallMethod: Method Name is not set.");
	ERR_FAIL_COND_V_MSG(node_param.is_null(), FAILURE, "BTCallMethod: Node parameter is not set.");
	Object *obj = node_param->get_value(get_scene_root(), get_blackboard());
	ERR_FAIL_COND_V_MSG(obj == nullptr, FAILURE, "BTCallMethod: Failed to get object: " + node_param->to_string());

	Variant result;
	Array call_args;

#ifdef LIMBOAI_MODULE
	const Variant delta = include_delta ? Variant(p_delta) : Variant();
	const Variant **argptrs = nullptr;

	int argument_count = include_delta ? args.size() + 1 : args.size();
	if (argument_count > 0) {
		argptrs = (const Variant **)alloca(sizeof(Variant *) * argument_count);
		if (include_delta) {
			argptrs[0] = &delta;
		}
		for (int i = 0; i < args.size(); i++) {
			Ref<BBVariant> param = args[i];
			call_args.push_back(param->get_value(get_scene_root(), get_blackboard()));
			argptrs[i + int(include_delta)] = &call_args[i];
		}
	}

	Callable::CallError ce;
	result = obj->callp(method, argptrs, argument_count, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		ERR_FAIL_V_MSG(FAILURE, "BTCallMethod: Error calling method: " + Variant::get_call_error_text(obj, method, argptrs, argument_count, ce) + ".");
	}
#elif LIMBOAI_GDEXTENSION
	if (include_delta) {
		call_args.push_back(Variant(p_delta));
	}
	for (int i = 0; i < args.size(); i++) {
		Ref<BBVariant> param = args[i];
		call_args.push_back(param->get_value(get_scene_root(), get_blackboard()));
	}

	// TODO: Unsure how to detect call error, so we return SUCCESS for now...
	result = obj->callv(method, call_args);
#endif // LIMBOAI_MODULE & LIMBOAI_GDEXTENSION

	if (result_var != StringName()) {
		get_blackboard()->set_var(result_var, result);
	}

	return SUCCESS;
}

//**** Godot

void BTCallMethod::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_method", "method_name"), &BTCallMethod::set_method);
	ClassDB::bind_method(D_METHOD("get_method"), &BTCallMethod::get_method);
	ClassDB::bind_method(D_METHOD("set_node_param", "param"), &BTCallMethod::set_node_param);
	ClassDB::bind_method(D_METHOD("get_node_param"), &BTCallMethod::get_node_param);
	ClassDB::bind_method(D_METHOD("set_args", "args"), &BTCallMethod::set_args);
	ClassDB::bind_method(D_METHOD("get_args"), &BTCallMethod::get_args);
	ClassDB::bind_method(D_METHOD("set_include_delta", "include_delta"), &BTCallMethod::set_include_delta);
	ClassDB::bind_method(D_METHOD("is_delta_included"), &BTCallMethod::is_delta_included);
	ClassDB::bind_method(D_METHOD("set_result_var", "variable"), &BTCallMethod::set_result_var);
	ClassDB::bind_method(D_METHOD("get_result_var"), &BTCallMethod::get_result_var);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "BBNode"), "set_node_param", "get_node_param");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "method"), "set_method", "get_method");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "result_var"), "set_result_var", "get_result_var");
	ADD_GROUP("Arguments", "args_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "args_include_delta"), "set_include_delta", "is_delta_included");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "args", PROPERTY_HINT_ARRAY_TYPE, RESOURCE_TYPE_HINT("BBVariant")), "set_args", "get_args");
}

BTCallMethod::BTCallMethod() {
}
