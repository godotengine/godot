/*************************************************************************/
/*  gdscript_byte_codegen.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "gdscript_byte_codegen.h"

#include "core/debugger/engine_debugger.h"
#include "gdscript.h"

uint32_t GDScriptByteCodeGenerator::add_parameter(const StringName &p_name, bool p_is_optional, const GDScriptDataType &p_type) {
#ifdef TOOLS_ENABLED
	function->arg_names.push_back(p_name);
#endif
	function->_argument_count++;
	function->argument_types.push_back(p_type);
	if (p_is_optional) {
		function->_default_arg_count++;
	}

	return add_local(p_name, p_type);
}

uint32_t GDScriptByteCodeGenerator::add_local(const StringName &p_name, const GDScriptDataType &p_type) {
	int stack_pos = locals.size() + RESERVED_STACK;
	locals.push_back(StackSlot(p_type.builtin_type));
	add_stack_identifier(p_name, stack_pos);
	return stack_pos;
}

uint32_t GDScriptByteCodeGenerator::add_local_constant(const StringName &p_name, const Variant &p_constant) {
	int index = add_or_get_constant(p_constant);
	local_constants[p_name] = index;
	return index;
}

uint32_t GDScriptByteCodeGenerator::add_or_get_constant(const Variant &p_constant) {
	return get_constant_pos(p_constant);
}

uint32_t GDScriptByteCodeGenerator::add_or_get_name(const StringName &p_name) {
	return get_name_map_pos(p_name);
}

uint32_t GDScriptByteCodeGenerator::add_temporary(const GDScriptDataType &p_type) {
	Variant::Type temp_type = Variant::NIL;
	if (p_type.has_type) {
		if (p_type.kind == GDScriptDataType::BUILTIN) {
			switch (p_type.builtin_type) {
				case Variant::NIL:
				case Variant::BOOL:
				case Variant::INT:
				case Variant::FLOAT:
				case Variant::STRING:
				case Variant::VECTOR2:
				case Variant::VECTOR2I:
				case Variant::RECT2:
				case Variant::RECT2I:
				case Variant::VECTOR3:
				case Variant::VECTOR3I:
				case Variant::TRANSFORM2D:
				case Variant::PLANE:
				case Variant::QUATERNION:
				case Variant::AABB:
				case Variant::BASIS:
				case Variant::TRANSFORM3D:
				case Variant::COLOR:
				case Variant::STRING_NAME:
				case Variant::NODE_PATH:
				case Variant::RID:
				case Variant::OBJECT:
				case Variant::CALLABLE:
				case Variant::SIGNAL:
				case Variant::DICTIONARY:
				case Variant::ARRAY:
					temp_type = p_type.builtin_type;
					break;
				case Variant::PACKED_BYTE_ARRAY:
				case Variant::PACKED_INT32_ARRAY:
				case Variant::PACKED_INT64_ARRAY:
				case Variant::PACKED_FLOAT32_ARRAY:
				case Variant::PACKED_FLOAT64_ARRAY:
				case Variant::PACKED_STRING_ARRAY:
				case Variant::PACKED_VECTOR2_ARRAY:
				case Variant::PACKED_VECTOR3_ARRAY:
				case Variant::PACKED_COLOR_ARRAY:
				case Variant::VARIANT_MAX:
					// Packed arrays are reference counted, so we don't use the pool for them.
					temp_type = Variant::NIL;
					break;
			}
		} else {
			temp_type = Variant::OBJECT;
		}
	}

	if (!temporaries_pool.has(temp_type)) {
		temporaries_pool[temp_type] = List<int>();
	}

	List<int> &pool = temporaries_pool[temp_type];
	if (pool.is_empty()) {
		StackSlot new_temp(temp_type);
		int idx = temporaries.size();
		pool.push_back(idx);
		temporaries.push_back(new_temp);
	}
	int slot = pool.front()->get();
	pool.pop_front();
	used_temporaries.push_back(slot);
	return slot;
}

void GDScriptByteCodeGenerator::pop_temporary() {
	ERR_FAIL_COND(used_temporaries.is_empty());
	int slot_idx = used_temporaries.back()->get();
	const StackSlot &slot = temporaries[slot_idx];
	temporaries_pool[slot.type].push_back(slot_idx);
	used_temporaries.pop_back();
}

void GDScriptByteCodeGenerator::start_parameters() {
	if (function->_default_arg_count > 0) {
		append(GDScriptFunction::OPCODE_JUMP_TO_DEF_ARGUMENT);
		function->default_arguments.push_back(opcodes.size());
	}
}

void GDScriptByteCodeGenerator::end_parameters() {
	function->default_arguments.reverse();
}

void GDScriptByteCodeGenerator::write_start(GDScript *p_script, const StringName &p_function_name, bool p_static, Multiplayer::RPCConfig p_rpc_config, const GDScriptDataType &p_return_type) {
	function = memnew(GDScriptFunction);
	debug_stack = EngineDebugger::is_active();

	function->name = p_function_name;
	function->_script = p_script;
	function->source = p_script->get_path();

#ifdef DEBUG_ENABLED
	function->func_cname = (String(function->source) + " - " + String(p_function_name)).utf8();
	function->_func_cname = function->func_cname.get_data();
#endif

	function->_static = p_static;
	function->return_type = p_return_type;
	function->rpc_config = p_rpc_config;
	function->_argument_count = 0;
}

GDScriptFunction *GDScriptByteCodeGenerator::write_end() {
#ifdef DEBUG_ENABLED
	if (!used_temporaries.is_empty()) {
		ERR_PRINT("Non-zero temporary variables at end of function: " + itos(used_temporaries.size()));
	}
#endif
	append(GDScriptFunction::OPCODE_END, 0);

	for (int i = 0; i < temporaries.size(); i++) {
		int stack_index = i + max_locals + RESERVED_STACK;
		for (int j = 0; j < temporaries[i].bytecode_indices.size(); j++) {
			opcodes.write[temporaries[i].bytecode_indices[j]] = stack_index | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
		}
		if (temporaries[i].type != Variant::NIL) {
			function->temporary_slots[stack_index] = temporaries[i].type;
		}
	}

	if (constant_map.size()) {
		function->_constant_count = constant_map.size();
		function->constants.resize(constant_map.size());
		function->_constants_ptr = function->constants.ptrw();
		const Variant *K = nullptr;
		while ((K = constant_map.next(K))) {
			int idx = constant_map[*K];
			function->constants.write[idx] = *K;
		}
	} else {
		function->_constants_ptr = nullptr;
		function->_constant_count = 0;
	}

	if (name_map.size()) {
		function->global_names.resize(name_map.size());
		function->_global_names_ptr = &function->global_names[0];
		for (const KeyValue<StringName, int> &E : name_map) {
			function->global_names.write[E.value] = E.key;
		}
		function->_global_names_count = function->global_names.size();

	} else {
		function->_global_names_ptr = nullptr;
		function->_global_names_count = 0;
	}

	if (opcodes.size()) {
		function->code = opcodes;
		function->_code_ptr = &function->code[0];
		function->_code_size = opcodes.size();

	} else {
		function->_code_ptr = nullptr;
		function->_code_size = 0;
	}

	if (function->default_arguments.size()) {
		function->_default_arg_count = function->default_arguments.size() - 1;
		function->_default_arg_ptr = &function->default_arguments[0];
	} else {
		function->_default_arg_count = 0;
		function->_default_arg_ptr = nullptr;
	}

	if (operator_func_map.size()) {
		function->operator_funcs.resize(operator_func_map.size());
		function->_operator_funcs_count = function->operator_funcs.size();
		function->_operator_funcs_ptr = function->operator_funcs.ptr();
		for (const KeyValue<Variant::ValidatedOperatorEvaluator, int> &E : operator_func_map) {
			function->operator_funcs.write[E.value] = E.key;
		}
	} else {
		function->_operator_funcs_count = 0;
		function->_operator_funcs_ptr = nullptr;
	}

	if (setters_map.size()) {
		function->setters.resize(setters_map.size());
		function->_setters_count = function->setters.size();
		function->_setters_ptr = function->setters.ptr();
		for (const KeyValue<Variant::ValidatedSetter, int> &E : setters_map) {
			function->setters.write[E.value] = E.key;
		}
	} else {
		function->_setters_count = 0;
		function->_setters_ptr = nullptr;
	}

	if (getters_map.size()) {
		function->getters.resize(getters_map.size());
		function->_getters_count = function->getters.size();
		function->_getters_ptr = function->getters.ptr();
		for (const KeyValue<Variant::ValidatedGetter, int> &E : getters_map) {
			function->getters.write[E.value] = E.key;
		}
	} else {
		function->_getters_count = 0;
		function->_getters_ptr = nullptr;
	}

	if (keyed_setters_map.size()) {
		function->keyed_setters.resize(keyed_setters_map.size());
		function->_keyed_setters_count = function->keyed_setters.size();
		function->_keyed_setters_ptr = function->keyed_setters.ptr();
		for (const KeyValue<Variant::ValidatedKeyedSetter, int> &E : keyed_setters_map) {
			function->keyed_setters.write[E.value] = E.key;
		}
	} else {
		function->_keyed_setters_count = 0;
		function->_keyed_setters_ptr = nullptr;
	}

	if (keyed_getters_map.size()) {
		function->keyed_getters.resize(keyed_getters_map.size());
		function->_keyed_getters_count = function->keyed_getters.size();
		function->_keyed_getters_ptr = function->keyed_getters.ptr();
		for (const KeyValue<Variant::ValidatedKeyedGetter, int> &E : keyed_getters_map) {
			function->keyed_getters.write[E.value] = E.key;
		}
	} else {
		function->_keyed_getters_count = 0;
		function->_keyed_getters_ptr = nullptr;
	}

	if (indexed_setters_map.size()) {
		function->indexed_setters.resize(indexed_setters_map.size());
		function->_indexed_setters_count = function->indexed_setters.size();
		function->_indexed_setters_ptr = function->indexed_setters.ptr();
		for (const KeyValue<Variant::ValidatedIndexedSetter, int> &E : indexed_setters_map) {
			function->indexed_setters.write[E.value] = E.key;
		}
	} else {
		function->_indexed_setters_count = 0;
		function->_indexed_setters_ptr = nullptr;
	}

	if (indexed_getters_map.size()) {
		function->indexed_getters.resize(indexed_getters_map.size());
		function->_indexed_getters_count = function->indexed_getters.size();
		function->_indexed_getters_ptr = function->indexed_getters.ptr();
		for (const KeyValue<Variant::ValidatedIndexedGetter, int> &E : indexed_getters_map) {
			function->indexed_getters.write[E.value] = E.key;
		}
	} else {
		function->_indexed_getters_count = 0;
		function->_indexed_getters_ptr = nullptr;
	}

	if (builtin_method_map.size()) {
		function->builtin_methods.resize(builtin_method_map.size());
		function->_builtin_methods_ptr = function->builtin_methods.ptr();
		function->_builtin_methods_count = builtin_method_map.size();
		for (const KeyValue<Variant::ValidatedBuiltInMethod, int> &E : builtin_method_map) {
			function->builtin_methods.write[E.value] = E.key;
		}
	} else {
		function->_builtin_methods_ptr = nullptr;
		function->_builtin_methods_count = 0;
	}

	if (constructors_map.size()) {
		function->constructors.resize(constructors_map.size());
		function->_constructors_ptr = function->constructors.ptr();
		function->_constructors_count = constructors_map.size();
		for (const KeyValue<Variant::ValidatedConstructor, int> &E : constructors_map) {
			function->constructors.write[E.value] = E.key;
		}
	} else {
		function->_constructors_ptr = nullptr;
		function->_constructors_count = 0;
	}

	if (utilities_map.size()) {
		function->utilities.resize(utilities_map.size());
		function->_utilities_ptr = function->utilities.ptr();
		function->_utilities_count = utilities_map.size();
		for (const KeyValue<Variant::ValidatedUtilityFunction, int> &E : utilities_map) {
			function->utilities.write[E.value] = E.key;
		}
	} else {
		function->_utilities_ptr = nullptr;
		function->_utilities_count = 0;
	}

	if (gds_utilities_map.size()) {
		function->gds_utilities.resize(gds_utilities_map.size());
		function->_gds_utilities_ptr = function->gds_utilities.ptr();
		function->_gds_utilities_count = gds_utilities_map.size();
		for (const KeyValue<GDScriptUtilityFunctions::FunctionPtr, int> &E : gds_utilities_map) {
			function->gds_utilities.write[E.value] = E.key;
		}
	} else {
		function->_gds_utilities_ptr = nullptr;
		function->_gds_utilities_count = 0;
	}

	if (method_bind_map.size()) {
		function->methods.resize(method_bind_map.size());
		function->_methods_ptr = function->methods.ptrw();
		function->_methods_count = method_bind_map.size();
		for (const KeyValue<MethodBind *, int> &E : method_bind_map) {
			function->methods.write[E.value] = E.key;
		}
	} else {
		function->_methods_ptr = nullptr;
		function->_methods_count = 0;
	}

	if (lambdas_map.size()) {
		function->lambdas.resize(lambdas_map.size());
		function->_lambdas_ptr = function->lambdas.ptrw();
		function->_lambdas_count = lambdas_map.size();
		for (const KeyValue<GDScriptFunction *, int> &E : lambdas_map) {
			function->lambdas.write[E.value] = E.key;
		}
	} else {
		function->_lambdas_ptr = nullptr;
		function->_lambdas_count = 0;
	}

	if (debug_stack) {
		function->stack_debug = stack_debug;
	}
	function->_stack_size = RESERVED_STACK + max_locals + temporaries.size();
	function->_instruction_args_size = instr_args_max;
	function->_ptrcall_args_size = ptrcall_max;

	ended = true;
	return function;
}

#ifdef DEBUG_ENABLED
void GDScriptByteCodeGenerator::set_signature(const String &p_signature) {
	function->profile.signature = p_signature;
}
#endif

void GDScriptByteCodeGenerator::set_initial_line(int p_line) {
	function->_initial_line = p_line;
}

#define HAS_BUILTIN_TYPE(m_var) \
	(m_var.type.has_type && m_var.type.kind == GDScriptDataType::BUILTIN)

#define IS_BUILTIN_TYPE(m_var, m_type) \
	(m_var.type.has_type && m_var.type.kind == GDScriptDataType::BUILTIN && m_var.type.builtin_type == m_type)

void GDScriptByteCodeGenerator::write_type_adjust(const Address &p_target, Variant::Type p_new_type) {
	switch (p_new_type) {
		case Variant::BOOL:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_BOOL, 1);
			break;
		case Variant::INT:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_INT, 1);
			break;
		case Variant::FLOAT:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_FLOAT, 1);
			break;
		case Variant::STRING:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_STRING, 1);
			break;
		case Variant::VECTOR2:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_VECTOR2, 1);
			break;
		case Variant::VECTOR2I:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_VECTOR2I, 1);
			break;
		case Variant::RECT2:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_RECT2, 1);
			break;
		case Variant::RECT2I:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_RECT2I, 1);
			break;
		case Variant::VECTOR3:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_VECTOR3, 1);
			break;
		case Variant::VECTOR3I:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_VECTOR3I, 1);
			break;
		case Variant::TRANSFORM2D:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_TRANSFORM2D, 1);
			break;
		case Variant::PLANE:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_PLANE, 1);
			break;
		case Variant::QUATERNION:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_QUATERNION, 1);
			break;
		case Variant::AABB:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_AABB, 1);
			break;
		case Variant::BASIS:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_BASIS, 1);
			break;
		case Variant::TRANSFORM3D:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_TRANSFORM, 1);
			break;
		case Variant::COLOR:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_COLOR, 1);
			break;
		case Variant::STRING_NAME:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_STRING_NAME, 1);
			break;
		case Variant::NODE_PATH:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_NODE_PATH, 1);
			break;
		case Variant::RID:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_RID, 1);
			break;
		case Variant::OBJECT:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_OBJECT, 1);
			break;
		case Variant::CALLABLE:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_CALLABLE, 1);
			break;
		case Variant::SIGNAL:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_SIGNAL, 1);
			break;
		case Variant::DICTIONARY:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_DICTIONARY, 1);
			break;
		case Variant::ARRAY:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_ARRAY, 1);
			break;
		case Variant::PACKED_BYTE_ARRAY:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_BYTE_ARRAY, 1);
			break;
		case Variant::PACKED_INT32_ARRAY:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_INT32_ARRAY, 1);
			break;
		case Variant::PACKED_INT64_ARRAY:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_INT64_ARRAY, 1);
			break;
		case Variant::PACKED_FLOAT32_ARRAY:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_FLOAT32_ARRAY, 1);
			break;
		case Variant::PACKED_FLOAT64_ARRAY:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_FLOAT64_ARRAY, 1);
			break;
		case Variant::PACKED_STRING_ARRAY:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_STRING_ARRAY, 1);
			break;
		case Variant::PACKED_VECTOR2_ARRAY:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_VECTOR2_ARRAY, 1);
			break;
		case Variant::PACKED_VECTOR3_ARRAY:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_VECTOR3_ARRAY, 1);
			break;
		case Variant::PACKED_COLOR_ARRAY:
			append(GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_COLOR_ARRAY, 1);
			break;
		case Variant::NIL:
		case Variant::VARIANT_MAX:
			return;
	}
	append(p_target);
}

void GDScriptByteCodeGenerator::write_unary_operator(const Address &p_target, Variant::Operator p_operator, const Address &p_left_operand) {
	if (HAS_BUILTIN_TYPE(p_left_operand)) {
		// Gather specific operator.
		Variant::ValidatedOperatorEvaluator op_func = Variant::get_validated_operator_evaluator(p_operator, p_left_operand.type.builtin_type, Variant::NIL);

		append(GDScriptFunction::OPCODE_OPERATOR_VALIDATED, 3);
		append(p_left_operand);
		append(Address());
		append(p_target);
		append(op_func);
		return;
	}

	// No specific types, perform variant evaluation.
	append(GDScriptFunction::OPCODE_OPERATOR, 3);
	append(p_left_operand);
	append(Address());
	append(p_target);
	append(p_operator);
}

void GDScriptByteCodeGenerator::write_binary_operator(const Address &p_target, Variant::Operator p_operator, const Address &p_left_operand, const Address &p_right_operand) {
	if (HAS_BUILTIN_TYPE(p_left_operand) && HAS_BUILTIN_TYPE(p_right_operand)) {
		if (p_target.mode == Address::TEMPORARY) {
			Variant::Type result_type = Variant::get_operator_return_type(p_operator, p_left_operand.type.builtin_type, p_right_operand.type.builtin_type);
			Variant::Type temp_type = temporaries[p_target.address].type;
			if (result_type != temp_type) {
				write_type_adjust(p_target, result_type);
			}
		}

		// Gather specific operator.
		Variant::ValidatedOperatorEvaluator op_func = Variant::get_validated_operator_evaluator(p_operator, p_left_operand.type.builtin_type, p_right_operand.type.builtin_type);

		append(GDScriptFunction::OPCODE_OPERATOR_VALIDATED, 3);
		append(p_left_operand);
		append(p_right_operand);
		append(p_target);
		append(op_func);
		return;
	}

	// No specific types, perform variant evaluation.
	append(GDScriptFunction::OPCODE_OPERATOR, 3);
	append(p_left_operand);
	append(p_right_operand);
	append(p_target);
	append(p_operator);
}

void GDScriptByteCodeGenerator::write_type_test(const Address &p_target, const Address &p_source, const Address &p_type) {
	append(GDScriptFunction::OPCODE_EXTENDS_TEST, 3);
	append(p_source);
	append(p_type);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_type_test_builtin(const Address &p_target, const Address &p_source, Variant::Type p_type) {
	append(GDScriptFunction::OPCODE_IS_BUILTIN, 2);
	append(p_source);
	append(p_target);
	append(p_type);
}

void GDScriptByteCodeGenerator::write_and_left_operand(const Address &p_left_operand) {
	append(GDScriptFunction::OPCODE_JUMP_IF_NOT, 1);
	append(p_left_operand);
	logic_op_jump_pos1.push_back(opcodes.size());
	append(0); // Jump target, will be patched.
}

void GDScriptByteCodeGenerator::write_and_right_operand(const Address &p_right_operand) {
	append(GDScriptFunction::OPCODE_JUMP_IF_NOT, 1);
	append(p_right_operand);
	logic_op_jump_pos2.push_back(opcodes.size());
	append(0); // Jump target, will be patched.
}

void GDScriptByteCodeGenerator::write_end_and(const Address &p_target) {
	// If here means both operands are true.
	append(GDScriptFunction::OPCODE_ASSIGN_TRUE, 1);
	append(p_target);
	// Jump away from the fail condition.
	append(GDScriptFunction::OPCODE_JUMP, 0);
	append(opcodes.size() + 3);
	// Here it means one of operands is false.
	patch_jump(logic_op_jump_pos1.back()->get());
	patch_jump(logic_op_jump_pos2.back()->get());
	logic_op_jump_pos1.pop_back();
	logic_op_jump_pos2.pop_back();
	append(GDScriptFunction::OPCODE_ASSIGN_FALSE, 1);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_or_left_operand(const Address &p_left_operand) {
	append(GDScriptFunction::OPCODE_JUMP_IF, 1);
	append(p_left_operand);
	logic_op_jump_pos1.push_back(opcodes.size());
	append(0); // Jump target, will be patched.
}

void GDScriptByteCodeGenerator::write_or_right_operand(const Address &p_right_operand) {
	append(GDScriptFunction::OPCODE_JUMP_IF, 1);
	append(p_right_operand);
	logic_op_jump_pos2.push_back(opcodes.size());
	append(0); // Jump target, will be patched.
}

void GDScriptByteCodeGenerator::write_end_or(const Address &p_target) {
	// If here means both operands are false.
	append(GDScriptFunction::OPCODE_ASSIGN_FALSE, 1);
	append(p_target);
	// Jump away from the success condition.
	append(GDScriptFunction::OPCODE_JUMP, 0);
	append(opcodes.size() + 3);
	// Here it means one of operands is true.
	patch_jump(logic_op_jump_pos1.back()->get());
	patch_jump(logic_op_jump_pos2.back()->get());
	logic_op_jump_pos1.pop_back();
	logic_op_jump_pos2.pop_back();
	append(GDScriptFunction::OPCODE_ASSIGN_TRUE, 1);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_start_ternary(const Address &p_target) {
	ternary_result.push_back(p_target);
}

void GDScriptByteCodeGenerator::write_ternary_condition(const Address &p_condition) {
	append(GDScriptFunction::OPCODE_JUMP_IF_NOT, 1);
	append(p_condition);
	ternary_jump_fail_pos.push_back(opcodes.size());
	append(0); // Jump target, will be patched.
}

void GDScriptByteCodeGenerator::write_ternary_true_expr(const Address &p_expr) {
	append(GDScriptFunction::OPCODE_ASSIGN, 2);
	append(ternary_result.back()->get());
	append(p_expr);
	// Jump away from the false path.
	append(GDScriptFunction::OPCODE_JUMP, 0);
	ternary_jump_skip_pos.push_back(opcodes.size());
	append(0);
	// Fail must jump here.
	patch_jump(ternary_jump_fail_pos.back()->get());
	ternary_jump_fail_pos.pop_back();
}

void GDScriptByteCodeGenerator::write_ternary_false_expr(const Address &p_expr) {
	append(GDScriptFunction::OPCODE_ASSIGN, 2);
	append(ternary_result.back()->get());
	append(p_expr);
}

void GDScriptByteCodeGenerator::write_end_ternary() {
	patch_jump(ternary_jump_skip_pos.back()->get());
	ternary_jump_skip_pos.pop_back();
}

void GDScriptByteCodeGenerator::write_set(const Address &p_target, const Address &p_index, const Address &p_source) {
	if (HAS_BUILTIN_TYPE(p_target)) {
		if (IS_BUILTIN_TYPE(p_index, Variant::INT) && Variant::get_member_validated_indexed_setter(p_target.type.builtin_type) &&
				IS_BUILTIN_TYPE(p_source, Variant::get_indexed_element_type(p_target.type.builtin_type))) {
			// Use indexed setter instead.
			Variant::ValidatedIndexedSetter setter = Variant::get_member_validated_indexed_setter(p_target.type.builtin_type);
			append(GDScriptFunction::OPCODE_SET_INDEXED_VALIDATED, 3);
			append(p_target);
			append(p_index);
			append(p_source);
			append(setter);
			return;
		} else if (Variant::get_member_validated_keyed_setter(p_target.type.builtin_type)) {
			Variant::ValidatedKeyedSetter setter = Variant::get_member_validated_keyed_setter(p_target.type.builtin_type);
			append(GDScriptFunction::OPCODE_SET_KEYED_VALIDATED, 3);
			append(p_target);
			append(p_index);
			append(p_source);
			append(setter);
			return;
		}
	}

	append(GDScriptFunction::OPCODE_SET_KEYED, 3);
	append(p_target);
	append(p_index);
	append(p_source);
}

void GDScriptByteCodeGenerator::write_get(const Address &p_target, const Address &p_index, const Address &p_source) {
	if (HAS_BUILTIN_TYPE(p_source)) {
		if (IS_BUILTIN_TYPE(p_index, Variant::INT) && Variant::get_member_validated_indexed_getter(p_source.type.builtin_type)) {
			// Use indexed getter instead.
			Variant::ValidatedIndexedGetter getter = Variant::get_member_validated_indexed_getter(p_source.type.builtin_type);
			append(GDScriptFunction::OPCODE_GET_INDEXED_VALIDATED, 3);
			append(p_source);
			append(p_index);
			append(p_target);
			append(getter);
			return;
		} else if (Variant::get_member_validated_keyed_getter(p_source.type.builtin_type)) {
			Variant::ValidatedKeyedGetter getter = Variant::get_member_validated_keyed_getter(p_source.type.builtin_type);
			append(GDScriptFunction::OPCODE_GET_KEYED_VALIDATED, 3);
			append(p_source);
			append(p_index);
			append(p_target);
			append(getter);
			return;
		}
	}
	append(GDScriptFunction::OPCODE_GET_KEYED, 3);
	append(p_source);
	append(p_index);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_set_named(const Address &p_target, const StringName &p_name, const Address &p_source) {
	if (HAS_BUILTIN_TYPE(p_target) && Variant::get_member_validated_setter(p_target.type.builtin_type, p_name) &&
			IS_BUILTIN_TYPE(p_source, Variant::get_member_type(p_target.type.builtin_type, p_name))) {
		Variant::ValidatedSetter setter = Variant::get_member_validated_setter(p_target.type.builtin_type, p_name);
		append(GDScriptFunction::OPCODE_SET_NAMED_VALIDATED, 2);
		append(p_target);
		append(p_source);
		append(setter);
		return;
	}
	append(GDScriptFunction::OPCODE_SET_NAMED, 2);
	append(p_target);
	append(p_source);
	append(p_name);
}

void GDScriptByteCodeGenerator::write_get_named(const Address &p_target, const StringName &p_name, const Address &p_source) {
	if (HAS_BUILTIN_TYPE(p_source) && Variant::get_member_validated_getter(p_source.type.builtin_type, p_name)) {
		Variant::ValidatedGetter getter = Variant::get_member_validated_getter(p_source.type.builtin_type, p_name);
		append(GDScriptFunction::OPCODE_GET_NAMED_VALIDATED, 2);
		append(p_source);
		append(p_target);
		append(getter);
		return;
	}
	append(GDScriptFunction::OPCODE_GET_NAMED, 2);
	append(p_source);
	append(p_target);
	append(p_name);
}

void GDScriptByteCodeGenerator::write_set_member(const Address &p_value, const StringName &p_name) {
	append(GDScriptFunction::OPCODE_SET_MEMBER, 1);
	append(p_value);
	append(p_name);
}

void GDScriptByteCodeGenerator::write_get_member(const Address &p_target, const StringName &p_name) {
	append(GDScriptFunction::OPCODE_GET_MEMBER, 1);
	append(p_target);
	append(p_name);
}

void GDScriptByteCodeGenerator::write_assign_with_conversion(const Address &p_target, const Address &p_source) {
	switch (p_target.type.kind) {
		case GDScriptDataType::BUILTIN: {
			if (p_target.type.builtin_type == Variant::ARRAY && p_target.type.has_container_element_type()) {
				append(GDScriptFunction::OPCODE_ASSIGN_TYPED_ARRAY, 2);
				append(p_target);
				append(p_source);
			} else {
				append(GDScriptFunction::OPCODE_ASSIGN_TYPED_BUILTIN, 2);
				append(p_target);
				append(p_source);
				append(p_target.type.builtin_type);
			}
		} break;
		case GDScriptDataType::NATIVE: {
			int class_idx = GDScriptLanguage::get_singleton()->get_global_map()[p_target.type.native_type];
			Variant nc = GDScriptLanguage::get_singleton()->get_global_array()[class_idx];
			class_idx = get_constant_pos(nc) | (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS);
			append(GDScriptFunction::OPCODE_ASSIGN_TYPED_NATIVE, 3);
			append(p_target);
			append(p_source);
			append(class_idx);
		} break;
		case GDScriptDataType::SCRIPT:
		case GDScriptDataType::GDSCRIPT: {
			Variant script = p_target.type.script_type;
			int idx = get_constant_pos(script) | (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS);

			append(GDScriptFunction::OPCODE_ASSIGN_TYPED_SCRIPT, 3);
			append(p_target);
			append(p_source);
			append(idx);
		} break;
		default: {
			ERR_PRINT("Compiler bug: unresolved assign.");

			// Shouldn't get here, but fail-safe to a regular assignment
			append(GDScriptFunction::OPCODE_ASSIGN, 2);
			append(p_target);
			append(p_source);
		}
	}
}

void GDScriptByteCodeGenerator::write_assign(const Address &p_target, const Address &p_source) {
	if (p_target.type.kind == GDScriptDataType::BUILTIN && p_target.type.builtin_type == Variant::ARRAY && p_target.type.has_container_element_type()) {
		append(GDScriptFunction::OPCODE_ASSIGN_TYPED_ARRAY, 2);
		append(p_target);
		append(p_source);
	} else if (p_target.type.kind == GDScriptDataType::BUILTIN && p_source.type.kind == GDScriptDataType::BUILTIN && p_target.type.builtin_type != p_source.type.builtin_type) {
		// Need conversion.
		append(GDScriptFunction::OPCODE_ASSIGN_TYPED_BUILTIN, 2);
		append(p_target);
		append(p_source);
		append(p_target.type.builtin_type);
	} else {
		append(GDScriptFunction::OPCODE_ASSIGN, 2);
		append(p_target);
		append(p_source);
	}
}

void GDScriptByteCodeGenerator::write_assign_true(const Address &p_target) {
	append(GDScriptFunction::OPCODE_ASSIGN_TRUE, 1);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_assign_false(const Address &p_target) {
	append(GDScriptFunction::OPCODE_ASSIGN_FALSE, 1);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_assign_default_parameter(const Address &p_dst, const Address &p_src) {
	write_assign(p_dst, p_src);
	function->default_arguments.push_back(opcodes.size());
}

void GDScriptByteCodeGenerator::write_store_global(const Address &p_dst, int p_global_index) {
	append(GDScriptFunction::OPCODE_STORE_GLOBAL, 1);
	append(p_dst);
	append(p_global_index);
}

void GDScriptByteCodeGenerator::write_store_named_global(const Address &p_dst, const StringName &p_global) {
	append(GDScriptFunction::OPCODE_STORE_NAMED_GLOBAL, 1);
	append(p_dst);
	append(p_global);
}

void GDScriptByteCodeGenerator::write_cast(const Address &p_target, const Address &p_source, const GDScriptDataType &p_type) {
	int index = 0;

	switch (p_type.kind) {
		case GDScriptDataType::BUILTIN: {
			append(GDScriptFunction::OPCODE_CAST_TO_BUILTIN, 2);
			index = p_type.builtin_type;
		} break;
		case GDScriptDataType::NATIVE: {
			int class_idx = GDScriptLanguage::get_singleton()->get_global_map()[p_type.native_type];
			Variant nc = GDScriptLanguage::get_singleton()->get_global_array()[class_idx];
			append(GDScriptFunction::OPCODE_CAST_TO_NATIVE, 3);
			index = get_constant_pos(nc) | (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS);
		} break;
		case GDScriptDataType::SCRIPT:
		case GDScriptDataType::GDSCRIPT: {
			Variant script = p_type.script_type;
			int idx = get_constant_pos(script) | (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS);
			append(GDScriptFunction::OPCODE_CAST_TO_SCRIPT, 3);
			index = idx;
		} break;
		default: {
			return;
		}
	}

	append(p_source);
	append(p_target);
	append(index);
}

void GDScriptByteCodeGenerator::write_call(const Address &p_target, const Address &p_base, const StringName &p_function_name, const Vector<Address> &p_arguments) {
	append(p_target.mode == Address::NIL ? GDScriptFunction::OPCODE_CALL : GDScriptFunction::OPCODE_CALL_RETURN, 2 + p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_base);
	append(p_target);
	append(p_arguments.size());
	append(p_function_name);
}

void GDScriptByteCodeGenerator::write_super_call(const Address &p_target, const StringName &p_function_name, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CALL_SELF_BASE, 1 + p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	append(p_arguments.size());
	append(p_function_name);
}

void GDScriptByteCodeGenerator::write_call_async(const Address &p_target, const Address &p_base, const StringName &p_function_name, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CALL_ASYNC, 2 + p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_base);
	append(p_target);
	append(p_arguments.size());
	append(p_function_name);
}

void GDScriptByteCodeGenerator::write_call_gdscript_utility(const Address &p_target, GDScriptUtilityFunctions::FunctionPtr p_function, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CALL_GDSCRIPT_UTILITY, 1 + p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	append(p_arguments.size());
	append(p_function);
}

void GDScriptByteCodeGenerator::write_call_utility(const Address &p_target, const StringName &p_function, const Vector<Address> &p_arguments) {
	bool is_validated = true;
	if (Variant::is_utility_function_vararg(p_function)) {
		is_validated = true; // Vararg works fine with any argument, since they can be any type.
	} else if (p_arguments.size() == Variant::get_utility_function_argument_count(p_function)) {
		bool all_types_exact = true;
		for (int i = 0; i < p_arguments.size(); i++) {
			if (!IS_BUILTIN_TYPE(p_arguments[i], Variant::get_utility_function_argument_type(p_function, i))) {
				all_types_exact = false;
				break;
			}
		}

		is_validated = all_types_exact;
	}

	if (is_validated) {
		append(GDScriptFunction::OPCODE_CALL_UTILITY_VALIDATED, 1 + p_arguments.size());
		for (int i = 0; i < p_arguments.size(); i++) {
			append(p_arguments[i]);
		}
		append(p_target);
		append(p_arguments.size());
		append(Variant::get_validated_utility_function(p_function));
	} else {
		append(GDScriptFunction::OPCODE_CALL_UTILITY, 1 + p_arguments.size());
		for (int i = 0; i < p_arguments.size(); i++) {
			append(p_arguments[i]);
		}
		append(p_target);
		append(p_arguments.size());
		append(p_function);
	}
}

void GDScriptByteCodeGenerator::write_call_builtin_type(const Address &p_target, const Address &p_base, Variant::Type p_type, const StringName &p_method, const Vector<Address> &p_arguments) {
	bool is_validated = false;

	// Check if all types are correct.
	if (Variant::is_builtin_method_vararg(p_type, p_method)) {
		is_validated = true; // Vararg works fine with any argument, since they can be any type.
	} else if (p_arguments.size() == Variant::get_builtin_method_argument_count(p_type, p_method)) {
		bool all_types_exact = true;
		for (int i = 0; i < p_arguments.size(); i++) {
			if (!IS_BUILTIN_TYPE(p_arguments[i], Variant::get_builtin_method_argument_type(p_type, p_method, i))) {
				all_types_exact = false;
				break;
			}
		}

		is_validated = all_types_exact;
	}

	if (!is_validated) {
		// Perform regular call.
		write_call(p_target, p_base, p_method, p_arguments);
		return;
	}

	if (p_target.mode == Address::TEMPORARY) {
		Variant::Type result_type = Variant::get_builtin_method_return_type(p_type, p_method);
		Variant::Type temp_type = temporaries[p_target.address].type;
		if (result_type != temp_type) {
			write_type_adjust(p_target, result_type);
		}
	}

	append(GDScriptFunction::OPCODE_CALL_BUILTIN_TYPE_VALIDATED, 2 + p_arguments.size());

	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_base);
	append(p_target);
	append(p_arguments.size());
	append(Variant::get_validated_builtin_method(p_type, p_method));
}

void GDScriptByteCodeGenerator::write_call_builtin_type_static(const Address &p_target, Variant::Type p_type, const StringName &p_method, const Vector<Address> &p_arguments) {
	bool is_validated = false;

	// Check if all types are correct.
	if (Variant::is_builtin_method_vararg(p_type, p_method)) {
		is_validated = true; // Vararg works fine with any argument, since they can be any type.
	} else if (p_arguments.size() == Variant::get_builtin_method_argument_count(p_type, p_method)) {
		bool all_types_exact = true;
		for (int i = 0; i < p_arguments.size(); i++) {
			if (!IS_BUILTIN_TYPE(p_arguments[i], Variant::get_builtin_method_argument_type(p_type, p_method, i))) {
				all_types_exact = false;
				break;
			}
		}

		is_validated = all_types_exact;
	}

	if (!is_validated) {
		// Perform regular call.
		append(GDScriptFunction::OPCODE_CALL_BUILTIN_STATIC, p_arguments.size() + 1);
		for (int i = 0; i < p_arguments.size(); i++) {
			append(p_arguments[i]);
		}
		append(p_target);
		append(p_type);
		append(p_method);
		append(p_arguments.size());
		return;
	}

	if (p_target.mode == Address::TEMPORARY) {
		Variant::Type result_type = Variant::get_builtin_method_return_type(p_type, p_method);
		Variant::Type temp_type = temporaries[p_target.address].type;
		if (result_type != temp_type) {
			write_type_adjust(p_target, result_type);
		}
	}

	append(GDScriptFunction::OPCODE_CALL_BUILTIN_TYPE_VALIDATED, 2 + p_arguments.size());

	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(Address()); // No base since it's static.
	append(p_target);
	append(p_arguments.size());
	append(Variant::get_validated_builtin_method(p_type, p_method));
}

void GDScriptByteCodeGenerator::write_call_method_bind(const Address &p_target, const Address &p_base, MethodBind *p_method, const Vector<Address> &p_arguments) {
	append(p_target.mode == Address::NIL ? GDScriptFunction::OPCODE_CALL_METHOD_BIND : GDScriptFunction::OPCODE_CALL_METHOD_BIND_RET, 2 + p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_base);
	append(p_target);
	append(p_arguments.size());
	append(p_method);
}

void GDScriptByteCodeGenerator::write_call_ptrcall(const Address &p_target, const Address &p_base, MethodBind *p_method, const Vector<Address> &p_arguments) {
#define CASE_TYPE(m_type)                                                               \
	case Variant::m_type:                                                               \
		append(GDScriptFunction::OPCODE_CALL_PTRCALL_##m_type, 2 + p_arguments.size()); \
		break

	bool is_ptrcall = true;

	if (p_method->has_return()) {
		MethodInfo info;
		ClassDB::get_method_info(p_method->get_instance_class(), p_method->get_name(), &info);
		switch (info.return_val.type) {
			CASE_TYPE(BOOL);
			CASE_TYPE(INT);
			CASE_TYPE(FLOAT);
			CASE_TYPE(STRING);
			CASE_TYPE(VECTOR2);
			CASE_TYPE(VECTOR2I);
			CASE_TYPE(RECT2);
			CASE_TYPE(RECT2I);
			CASE_TYPE(VECTOR3);
			CASE_TYPE(VECTOR3I);
			CASE_TYPE(TRANSFORM2D);
			CASE_TYPE(PLANE);
			CASE_TYPE(AABB);
			CASE_TYPE(BASIS);
			CASE_TYPE(TRANSFORM3D);
			CASE_TYPE(COLOR);
			CASE_TYPE(STRING_NAME);
			CASE_TYPE(NODE_PATH);
			CASE_TYPE(RID);
			CASE_TYPE(QUATERNION);
			CASE_TYPE(OBJECT);
			CASE_TYPE(CALLABLE);
			CASE_TYPE(SIGNAL);
			CASE_TYPE(DICTIONARY);
			CASE_TYPE(ARRAY);
			CASE_TYPE(PACKED_BYTE_ARRAY);
			CASE_TYPE(PACKED_INT32_ARRAY);
			CASE_TYPE(PACKED_INT64_ARRAY);
			CASE_TYPE(PACKED_FLOAT32_ARRAY);
			CASE_TYPE(PACKED_FLOAT64_ARRAY);
			CASE_TYPE(PACKED_STRING_ARRAY);
			CASE_TYPE(PACKED_VECTOR2_ARRAY);
			CASE_TYPE(PACKED_VECTOR3_ARRAY);
			CASE_TYPE(PACKED_COLOR_ARRAY);
			default:
				append(p_target.mode == Address::NIL ? GDScriptFunction::OPCODE_CALL_METHOD_BIND : GDScriptFunction::OPCODE_CALL_METHOD_BIND_RET, 2 + p_arguments.size());
				is_ptrcall = false;
				break;
		}
	} else {
		append(GDScriptFunction::OPCODE_CALL_PTRCALL_NO_RETURN, 2 + p_arguments.size());
	}

	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_base);
	append(p_target);
	append(p_arguments.size());
	append(p_method);
	if (is_ptrcall) {
		alloc_ptrcall(p_arguments.size());
	}

#undef CASE_TYPE
}

void GDScriptByteCodeGenerator::write_call_self(const Address &p_target, const StringName &p_function_name, const Vector<Address> &p_arguments) {
	append(p_target.mode == Address::NIL ? GDScriptFunction::OPCODE_CALL : GDScriptFunction::OPCODE_CALL_RETURN, 2 + p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
	append(p_target);
	append(p_arguments.size());
	append(p_function_name);
}

void GDScriptByteCodeGenerator::write_call_self_async(const Address &p_target, const StringName &p_function_name, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CALL_ASYNC, 2 + p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(GDScriptFunction::ADDR_SELF);
	append(p_target);
	append(p_arguments.size());
	append(p_function_name);
}

void GDScriptByteCodeGenerator::write_call_script_function(const Address &p_target, const Address &p_base, const StringName &p_function_name, const Vector<Address> &p_arguments) {
	append(p_target.mode == Address::NIL ? GDScriptFunction::OPCODE_CALL : GDScriptFunction::OPCODE_CALL_RETURN, 2 + p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_base);
	append(p_target);
	append(p_arguments.size());
	append(p_function_name);
}

void GDScriptByteCodeGenerator::write_lambda(const Address &p_target, GDScriptFunction *p_function, const Vector<Address> &p_captures) {
	append(GDScriptFunction::OPCODE_CREATE_LAMBDA, 1 + p_captures.size());
	for (int i = 0; i < p_captures.size(); i++) {
		append(p_captures[i]);
	}

	append(p_target);
	append(p_captures.size());
	append(p_function);
}

void GDScriptByteCodeGenerator::write_construct(const Address &p_target, Variant::Type p_type, const Vector<Address> &p_arguments) {
	// Try to find an appropriate constructor.
	bool all_have_type = true;
	Vector<Variant::Type> arg_types;
	for (int i = 0; i < p_arguments.size(); i++) {
		if (!HAS_BUILTIN_TYPE(p_arguments[i])) {
			all_have_type = false;
			break;
		}
		arg_types.push_back(p_arguments[i].type.builtin_type);
	}
	if (all_have_type) {
		int valid_constructor = -1;
		for (int i = 0; i < Variant::get_constructor_count(p_type); i++) {
			if (Variant::get_constructor_argument_count(p_type, i) != p_arguments.size()) {
				continue;
			}
			int types_correct = true;
			for (int j = 0; j < arg_types.size(); j++) {
				if (arg_types[j] != Variant::get_constructor_argument_type(p_type, i, j)) {
					types_correct = false;
					break;
				}
			}
			if (types_correct) {
				valid_constructor = i;
				break;
			}
		}
		if (valid_constructor >= 0) {
			append(GDScriptFunction::OPCODE_CONSTRUCT_VALIDATED, 1 + p_arguments.size());
			for (int i = 0; i < p_arguments.size(); i++) {
				append(p_arguments[i]);
			}
			append(p_target);
			append(p_arguments.size());
			append(Variant::get_validated_constructor(p_type, valid_constructor));
			return;
		}
	}

	append(GDScriptFunction::OPCODE_CONSTRUCT, 1 + p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	append(p_arguments.size());
	append(p_type);
}

void GDScriptByteCodeGenerator::write_construct_array(const Address &p_target, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CONSTRUCT_ARRAY, 1 + p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	append(p_arguments.size());
}

void GDScriptByteCodeGenerator::write_construct_typed_array(const Address &p_target, const GDScriptDataType &p_element_type, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CONSTRUCT_TYPED_ARRAY, 2 + p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	if (p_element_type.script_type) {
		Variant script_type = Ref<Script>(p_element_type.script_type);
		int addr = get_constant_pos(script_type);
		addr |= GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS;
		append(addr);
	} else {
		append(Address()); // null.
	}
	append(p_arguments.size());
	append(p_element_type.builtin_type);
	append(p_element_type.native_type);
}

void GDScriptByteCodeGenerator::write_construct_dictionary(const Address &p_target, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CONSTRUCT_DICTIONARY, 1 + p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	append(p_arguments.size() / 2); // This is number of key-value pairs, so only half of actual arguments.
}

void GDScriptByteCodeGenerator::write_await(const Address &p_target, const Address &p_operand) {
	append(GDScriptFunction::OPCODE_AWAIT, 1);
	append(p_operand);
	append(GDScriptFunction::OPCODE_AWAIT_RESUME, 1);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_if(const Address &p_condition) {
	append(GDScriptFunction::OPCODE_JUMP_IF_NOT, 1);
	append(p_condition);
	if_jmp_addrs.push_back(opcodes.size());
	append(0); // Jump destination, will be patched.
}

void GDScriptByteCodeGenerator::write_else() {
	append(GDScriptFunction::OPCODE_JUMP, 0); // Jump from true if block;
	int else_jmp_addr = opcodes.size();
	append(0); // Jump destination, will be patched.

	patch_jump(if_jmp_addrs.back()->get());
	if_jmp_addrs.pop_back();
	if_jmp_addrs.push_back(else_jmp_addr);
}

void GDScriptByteCodeGenerator::write_endif() {
	patch_jump(if_jmp_addrs.back()->get());
	if_jmp_addrs.pop_back();
}

void GDScriptByteCodeGenerator::start_for(const GDScriptDataType &p_iterator_type, const GDScriptDataType &p_list_type) {
	Address counter(Address::LOCAL_VARIABLE, add_local("@counter_pos", p_iterator_type), p_iterator_type);
	Address container(Address::LOCAL_VARIABLE, add_local("@container_pos", p_list_type), p_list_type);

	// Store state.
	for_counter_variables.push_back(counter);
	for_container_variables.push_back(container);
}

void GDScriptByteCodeGenerator::write_for_assignment(const Address &p_variable, const Address &p_list) {
	const Address &container = for_container_variables.back()->get();

	// Assign container.
	append(GDScriptFunction::OPCODE_ASSIGN, 2);
	append(container);
	append(p_list);

	for_iterator_variables.push_back(p_variable);
}

void GDScriptByteCodeGenerator::write_for() {
	const Address &iterator = for_iterator_variables.back()->get();
	const Address &counter = for_counter_variables.back()->get();
	const Address &container = for_container_variables.back()->get();

	current_breaks_to_patch.push_back(List<int>());

	GDScriptFunction::Opcode begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN;
	GDScriptFunction::Opcode iterate_opcode = GDScriptFunction::OPCODE_ITERATE;

	if (container.type.has_type) {
		if (container.type.kind == GDScriptDataType::BUILTIN) {
			switch (container.type.builtin_type) {
				case Variant::INT:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_INT;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_INT;
					break;
				case Variant::FLOAT:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_FLOAT;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_FLOAT;
					break;
				case Variant::VECTOR2:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_VECTOR2;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_VECTOR2;
					break;
				case Variant::VECTOR2I:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_VECTOR2I;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_VECTOR2I;
					break;
				case Variant::VECTOR3:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_VECTOR3;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_VECTOR3;
					break;
				case Variant::VECTOR3I:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_VECTOR3I;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_VECTOR3I;
					break;
				case Variant::STRING:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_STRING;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_STRING;
					break;
				case Variant::DICTIONARY:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_DICTIONARY;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_DICTIONARY;
					break;
				case Variant::ARRAY:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_ARRAY;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_ARRAY;
					break;
				case Variant::PACKED_BYTE_ARRAY:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_BYTE_ARRAY;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_PACKED_BYTE_ARRAY;
					break;
				case Variant::PACKED_INT32_ARRAY:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_INT32_ARRAY;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_PACKED_INT32_ARRAY;
					break;
				case Variant::PACKED_INT64_ARRAY:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_INT64_ARRAY;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_PACKED_INT64_ARRAY;
					break;
				case Variant::PACKED_FLOAT32_ARRAY:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_FLOAT32_ARRAY;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_PACKED_FLOAT32_ARRAY;
					break;
				case Variant::PACKED_FLOAT64_ARRAY:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_FLOAT64_ARRAY;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_PACKED_FLOAT64_ARRAY;
					break;
				case Variant::PACKED_STRING_ARRAY:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_STRING_ARRAY;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_PACKED_STRING_ARRAY;
					break;
				case Variant::PACKED_VECTOR2_ARRAY:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_VECTOR2_ARRAY;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_PACKED_VECTOR2_ARRAY;
					break;
				case Variant::PACKED_VECTOR3_ARRAY:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_VECTOR3_ARRAY;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_PACKED_VECTOR3_ARRAY;
					break;
				case Variant::PACKED_COLOR_ARRAY:
					begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_COLOR_ARRAY;
					iterate_opcode = GDScriptFunction::OPCODE_ITERATE_PACKED_COLOR_ARRAY;
					break;
				default:
					break;
			}
		} else {
			begin_opcode = GDScriptFunction::OPCODE_ITERATE_BEGIN_OBJECT;
			iterate_opcode = GDScriptFunction::OPCODE_ITERATE_OBJECT;
		}
	}

	// Begin loop.
	append(begin_opcode, 3);
	append(counter);
	append(container);
	append(iterator);
	for_jmp_addrs.push_back(opcodes.size());
	append(0); // End of loop address, will be patched.
	append(GDScriptFunction::OPCODE_JUMP, 0);
	append(opcodes.size() + 6); // Skip over 'continue' code.

	// Next iteration.
	int continue_addr = opcodes.size();
	continue_addrs.push_back(continue_addr);
	append(iterate_opcode, 3);
	append(counter);
	append(container);
	append(iterator);
	for_jmp_addrs.push_back(opcodes.size());
	append(0); // Jump destination, will be patched.
}

void GDScriptByteCodeGenerator::write_endfor() {
	// Jump back to loop check.
	append(GDScriptFunction::OPCODE_JUMP, 0);
	append(continue_addrs.back()->get());
	continue_addrs.pop_back();

	// Patch end jumps (two of them).
	for (int i = 0; i < 2; i++) {
		patch_jump(for_jmp_addrs.back()->get());
		for_jmp_addrs.pop_back();
	}

	// Patch break statements.
	for (const int &E : current_breaks_to_patch.back()->get()) {
		patch_jump(E);
	}
	current_breaks_to_patch.pop_back();

	// Pop state.
	for_iterator_variables.pop_back();
	for_counter_variables.pop_back();
	for_container_variables.pop_back();
}

void GDScriptByteCodeGenerator::start_while_condition() {
	current_breaks_to_patch.push_back(List<int>());
	continue_addrs.push_back(opcodes.size());
}

void GDScriptByteCodeGenerator::write_while(const Address &p_condition) {
	// Condition check.
	append(GDScriptFunction::OPCODE_JUMP_IF_NOT, 1);
	append(p_condition);
	while_jmp_addrs.push_back(opcodes.size());
	append(0); // End of loop address, will be patched.
}

void GDScriptByteCodeGenerator::write_endwhile() {
	// Jump back to loop check.
	append(GDScriptFunction::OPCODE_JUMP, 0);
	append(continue_addrs.back()->get());
	continue_addrs.pop_back();

	// Patch end jump.
	patch_jump(while_jmp_addrs.back()->get());
	while_jmp_addrs.pop_back();

	// Patch break statements.
	for (const int &E : current_breaks_to_patch.back()->get()) {
		patch_jump(E);
	}
	current_breaks_to_patch.pop_back();
}

void GDScriptByteCodeGenerator::start_match() {
	match_continues_to_patch.push_back(List<int>());
}

void GDScriptByteCodeGenerator::start_match_branch() {
	// Patch continue statements.
	for (const int &E : match_continues_to_patch.back()->get()) {
		patch_jump(E);
	}
	match_continues_to_patch.pop_back();
	// Start a new list for next branch.
	match_continues_to_patch.push_back(List<int>());
}

void GDScriptByteCodeGenerator::end_match() {
	// Patch continue statements.
	for (const int &E : match_continues_to_patch.back()->get()) {
		patch_jump(E);
	}
	match_continues_to_patch.pop_back();
}

void GDScriptByteCodeGenerator::write_break() {
	append(GDScriptFunction::OPCODE_JUMP, 0);
	current_breaks_to_patch.back()->get().push_back(opcodes.size());
	append(0);
}

void GDScriptByteCodeGenerator::write_continue() {
	append(GDScriptFunction::OPCODE_JUMP, 0);
	append(continue_addrs.back()->get());
}

void GDScriptByteCodeGenerator::write_continue_match() {
	append(GDScriptFunction::OPCODE_JUMP, 0);
	match_continues_to_patch.back()->get().push_back(opcodes.size());
	append(0);
}

void GDScriptByteCodeGenerator::write_breakpoint() {
	append(GDScriptFunction::OPCODE_BREAKPOINT, 0);
}

void GDScriptByteCodeGenerator::write_newline(int p_line) {
	append(GDScriptFunction::OPCODE_LINE, 0);
	append(p_line);
	current_line = p_line;
}

void GDScriptByteCodeGenerator::write_return(const Address &p_return_value) {
	if (!function->return_type.has_type || p_return_value.type.has_type) {
		// Either the function is untyped or the return value is also typed.

		// If this is a typed function, then we need to check for potential conversions.
		if (function->return_type.has_type) {
			if (function->return_type.kind == GDScriptDataType::BUILTIN && function->return_type.builtin_type == Variant::ARRAY && function->return_type.has_container_element_type()) {
				// Typed array.
				const GDScriptDataType &element_type = function->return_type.get_container_element_type();

				Variant script = function->return_type.script_type;
				int script_idx = get_constant_pos(script) | (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS);

				append(GDScriptFunction::OPCODE_RETURN_TYPED_ARRAY, 2);
				append(p_return_value);
				append(script_idx);
				append(element_type.kind == GDScriptDataType::BUILTIN ? element_type.builtin_type : Variant::OBJECT);
				append(element_type.native_type);
			} else if (function->return_type.kind == GDScriptDataType::BUILTIN && p_return_value.type.kind == GDScriptDataType::BUILTIN && function->return_type.builtin_type != p_return_value.type.builtin_type) {
				// Add conversion.
				append(GDScriptFunction::OPCODE_RETURN_TYPED_BUILTIN, 1);
				append(p_return_value);
				append(function->return_type.builtin_type);
			} else {
				// Just assign.
				append(GDScriptFunction::OPCODE_RETURN, 1);
				append(p_return_value);
			}
		} else {
			append(GDScriptFunction::OPCODE_RETURN, 1);
			append(p_return_value);
		}
	} else {
		switch (function->return_type.kind) {
			case GDScriptDataType::BUILTIN: {
				if (function->return_type.builtin_type == Variant::ARRAY && function->return_type.has_container_element_type()) {
					const GDScriptDataType &element_type = function->return_type.get_container_element_type();

					Variant script = function->return_type.script_type;
					int script_idx = get_constant_pos(script);
					script_idx |= (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS);

					append(GDScriptFunction::OPCODE_RETURN_TYPED_ARRAY, 2);
					append(p_return_value);
					append(script_idx);
					append(element_type.kind == GDScriptDataType::BUILTIN ? element_type.builtin_type : Variant::OBJECT);
					append(element_type.native_type);
				} else {
					append(GDScriptFunction::OPCODE_RETURN_TYPED_BUILTIN, 1);
					append(p_return_value);
					append(function->return_type.builtin_type);
				}
			} break;
			case GDScriptDataType::NATIVE: {
				append(GDScriptFunction::OPCODE_RETURN_TYPED_NATIVE, 2);
				append(p_return_value);
				int class_idx = GDScriptLanguage::get_singleton()->get_global_map()[function->return_type.native_type];
				Variant nc = GDScriptLanguage::get_singleton()->get_global_array()[class_idx];
				class_idx = get_constant_pos(nc) | (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS);
				append(class_idx);
			} break;
			case GDScriptDataType::GDSCRIPT:
			case GDScriptDataType::SCRIPT: {
				Variant script = function->return_type.script_type;
				int script_idx = get_constant_pos(script) | (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS);

				append(GDScriptFunction::OPCODE_RETURN_TYPED_SCRIPT, 2);
				append(p_return_value);
				append(script_idx);
			} break;
			default: {
				ERR_PRINT("Compiler bug: unresolved return.");

				// Shouldn't get here, but fail-safe to a regular return;
				append(GDScriptFunction::OPCODE_RETURN, 1);
				append(p_return_value);
			} break;
		}
	}
}

void GDScriptByteCodeGenerator::write_assert(const Address &p_test, const Address &p_message) {
	append(GDScriptFunction::OPCODE_ASSERT, 2);
	append(p_test);
	append(p_message);
}

void GDScriptByteCodeGenerator::start_block() {
	push_stack_identifiers();
}

void GDScriptByteCodeGenerator::end_block() {
	pop_stack_identifiers();
}

GDScriptByteCodeGenerator::~GDScriptByteCodeGenerator() {
	if (!ended && function != nullptr) {
		memdelete(function);
	}
}
