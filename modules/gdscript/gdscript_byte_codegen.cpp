/*************************************************************************/
/*  gdscript_byte_codegen.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
		if (function->_default_arg_count == 0) {
			append(GDScriptFunction::OPCODE_JUMP_TO_DEF_ARGUMENT);
		}
		function->default_arguments.push_back(opcodes.size());
		function->_default_arg_count++;
	}

	return add_local(p_name, p_type);
}

uint32_t GDScriptByteCodeGenerator::add_local(const StringName &p_name, const GDScriptDataType &p_type) {
	int stack_pos = increase_stack();
	add_stack_identifier(p_name, stack_pos);
	return stack_pos;
}

uint32_t GDScriptByteCodeGenerator::add_local_constant(const StringName &p_name, const Variant &p_constant) {
	int index = add_or_get_constant(p_constant);
	local_constants[p_name] = index;
	return index;
}

uint32_t GDScriptByteCodeGenerator::add_or_get_constant(const Variant &p_constant) {
	if (constant_map.has(p_constant)) {
		return constant_map[p_constant];
	}
	int index = constant_map.size();
	constant_map[p_constant] = index;
	return index;
}

uint32_t GDScriptByteCodeGenerator::add_or_get_name(const StringName &p_name) {
	return get_name_map_pos(p_name);
}

uint32_t GDScriptByteCodeGenerator::add_temporary() {
	current_temporaries++;
	return increase_stack();
}

void GDScriptByteCodeGenerator::pop_temporary() {
	current_stack_size--;
	current_temporaries--;
}

void GDScriptByteCodeGenerator::start_parameters() {}

void GDScriptByteCodeGenerator::end_parameters() {
	function->default_arguments.invert();
}

void GDScriptByteCodeGenerator::write_start(GDScript *p_script, const StringName &p_function_name, bool p_static, MultiplayerAPI::RPCMode p_rpc_mode, const GDScriptDataType &p_return_type) {
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
	function->rpc_mode = p_rpc_mode;
	function->_argument_count = 0;
}

GDScriptFunction *GDScriptByteCodeGenerator::write_end() {
	append(GDScriptFunction::OPCODE_END);

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
		for (Map<StringName, int>::Element *E = name_map.front(); E; E = E->next()) {
			function->global_names.write[E->get()] = E->key();
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
		function->_default_arg_count = function->default_arguments.size();
		function->_default_arg_ptr = &function->default_arguments[0];
	} else {
		function->_default_arg_count = 0;
		function->_default_arg_ptr = nullptr;
	}

	if (debug_stack) {
		function->stack_debug = stack_debug;
	}
	function->_stack_size = stack_max;
	function->_call_size = call_max;

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

void GDScriptByteCodeGenerator::write_operator(const Address &p_target, Variant::Operator p_operator, const Address &p_left_operand, const Address &p_right_operand) {
	append(GDScriptFunction::OPCODE_OPERATOR);
	append(p_operator);
	append(p_left_operand);
	append(p_right_operand);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_type_test(const Address &p_target, const Address &p_source, const Address &p_type) {
	append(GDScriptFunction::OPCODE_EXTENDS_TEST);
	append(p_source);
	append(p_type);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_type_test_builtin(const Address &p_target, const Address &p_source, Variant::Type p_type) {
	append(GDScriptFunction::OPCODE_IS_BUILTIN);
	append(p_source);
	append(p_type);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_and_left_operand(const Address &p_left_operand) {
	append(GDScriptFunction::OPCODE_JUMP_IF_NOT);
	append(p_left_operand);
	logic_op_jump_pos1.push_back(opcodes.size());
	append(0); // Jump target, will be patched.
}

void GDScriptByteCodeGenerator::write_and_right_operand(const Address &p_right_operand) {
	append(GDScriptFunction::OPCODE_JUMP_IF_NOT);
	append(p_right_operand);
	logic_op_jump_pos2.push_back(opcodes.size());
	append(0); // Jump target, will be patched.
}

void GDScriptByteCodeGenerator::write_end_and(const Address &p_target) {
	// If here means both operands are true.
	append(GDScriptFunction::OPCODE_ASSIGN_TRUE);
	append(p_target);
	// Jump away from the fail condition.
	append(GDScriptFunction::OPCODE_JUMP);
	append(opcodes.size() + 3);
	// Here it means one of operands is false.
	patch_jump(logic_op_jump_pos1.back()->get());
	patch_jump(logic_op_jump_pos2.back()->get());
	logic_op_jump_pos1.pop_back();
	logic_op_jump_pos2.pop_back();
	append(GDScriptFunction::OPCODE_ASSIGN_FALSE);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_or_left_operand(const Address &p_left_operand) {
	append(GDScriptFunction::OPCODE_JUMP_IF);
	append(p_left_operand);
	logic_op_jump_pos1.push_back(opcodes.size());
	append(0); // Jump target, will be patched.
}

void GDScriptByteCodeGenerator::write_or_right_operand(const Address &p_right_operand) {
	append(GDScriptFunction::OPCODE_JUMP_IF);
	append(p_right_operand);
	logic_op_jump_pos2.push_back(opcodes.size());
	append(0); // Jump target, will be patched.
}

void GDScriptByteCodeGenerator::write_end_or(const Address &p_target) {
	// If here means both operands are false.
	append(GDScriptFunction::OPCODE_ASSIGN_FALSE);
	append(p_target);
	// Jump away from the success condition.
	append(GDScriptFunction::OPCODE_JUMP);
	append(opcodes.size() + 3);
	// Here it means one of operands is false.
	patch_jump(logic_op_jump_pos1.back()->get());
	patch_jump(logic_op_jump_pos2.back()->get());
	logic_op_jump_pos1.pop_back();
	logic_op_jump_pos2.pop_back();
	append(GDScriptFunction::OPCODE_ASSIGN_TRUE);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_start_ternary(const Address &p_target) {
	ternary_result.push_back(p_target);
}

void GDScriptByteCodeGenerator::write_ternary_condition(const Address &p_condition) {
	append(GDScriptFunction::OPCODE_JUMP_IF_NOT);
	append(p_condition);
	ternary_jump_fail_pos.push_back(opcodes.size());
	append(0); // Jump target, will be patched.
}

void GDScriptByteCodeGenerator::write_ternary_true_expr(const Address &p_expr) {
	append(GDScriptFunction::OPCODE_ASSIGN);
	append(ternary_result.back()->get());
	append(p_expr);
	// Jump away from the false path.
	append(GDScriptFunction::OPCODE_JUMP);
	ternary_jump_skip_pos.push_back(opcodes.size());
	append(0);
	// Fail must jump here.
	patch_jump(ternary_jump_fail_pos.back()->get());
	ternary_jump_fail_pos.pop_back();
}

void GDScriptByteCodeGenerator::write_ternary_false_expr(const Address &p_expr) {
	append(GDScriptFunction::OPCODE_ASSIGN);
	append(ternary_result.back()->get());
	append(p_expr);
}

void GDScriptByteCodeGenerator::write_end_ternary() {
	patch_jump(ternary_jump_skip_pos.back()->get());
	ternary_jump_skip_pos.pop_back();
}

void GDScriptByteCodeGenerator::write_set(const Address &p_target, const Address &p_index, const Address &p_source) {
	append(GDScriptFunction::OPCODE_SET);
	append(p_target);
	append(p_index);
	append(p_source);
}

void GDScriptByteCodeGenerator::write_get(const Address &p_target, const Address &p_index, const Address &p_source) {
	append(GDScriptFunction::OPCODE_GET);
	append(p_source);
	append(p_index);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_set_named(const Address &p_target, const StringName &p_name, const Address &p_source) {
	append(GDScriptFunction::OPCODE_SET_NAMED);
	append(p_target);
	append(p_name);
	append(p_source);
}

void GDScriptByteCodeGenerator::write_get_named(const Address &p_target, const StringName &p_name, const Address &p_source) {
	append(GDScriptFunction::OPCODE_GET_NAMED);
	append(p_source);
	append(p_name);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_set_member(const Address &p_value, const StringName &p_name) {
	append(GDScriptFunction::OPCODE_SET_MEMBER);
	append(p_name);
	append(p_value);
}

void GDScriptByteCodeGenerator::write_get_member(const Address &p_target, const StringName &p_name) {
	append(GDScriptFunction::OPCODE_GET_MEMBER);
	append(p_name);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_assign(const Address &p_target, const Address &p_source) {
	if (p_target.type.has_type && !p_source.type.has_type) {
		// Typed assignment.
		switch (p_target.type.kind) {
			case GDScriptDataType::BUILTIN: {
				append(GDScriptFunction::OPCODE_ASSIGN_TYPED_BUILTIN);
				append(p_target.type.builtin_type);
				append(p_target);
				append(p_source);
			} break;
			case GDScriptDataType::NATIVE: {
				int class_idx = GDScriptLanguage::get_singleton()->get_global_map()[p_target.type.native_type];
				class_idx |= (GDScriptFunction::ADDR_TYPE_GLOBAL << GDScriptFunction::ADDR_BITS);
				append(GDScriptFunction::OPCODE_ASSIGN_TYPED_NATIVE);
				append(class_idx);
				append(p_target);
				append(p_source);
			} break;
			case GDScriptDataType::SCRIPT:
			case GDScriptDataType::GDSCRIPT: {
				Variant script = p_target.type.script_type;
				int idx = get_constant_pos(script);

				append(GDScriptFunction::OPCODE_ASSIGN_TYPED_SCRIPT);
				append(idx);
				append(p_target);
				append(p_source);
			} break;
			default: {
				ERR_PRINT("Compiler bug: unresolved assign.");

				// Shouldn't get here, but fail-safe to a regular assignment
				append(GDScriptFunction::OPCODE_ASSIGN);
				append(p_target);
				append(p_source);
			}
		}
	} else {
		if (p_target.type.kind == GDScriptDataType::BUILTIN && p_source.type.kind == GDScriptDataType::BUILTIN && p_target.type.builtin_type != p_source.type.builtin_type) {
			// Need conversion..
			append(GDScriptFunction::OPCODE_ASSIGN_TYPED_BUILTIN);
			append(p_target.type.builtin_type);
			append(p_target);
			append(p_source);
		} else {
			// Either untyped assignment or already type-checked by the parser
			append(GDScriptFunction::OPCODE_ASSIGN);
			append(p_target);
			append(p_source);
		}
	}
}

void GDScriptByteCodeGenerator::write_assign_true(const Address &p_target) {
	append(GDScriptFunction::OPCODE_ASSIGN_TRUE);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_assign_false(const Address &p_target) {
	append(GDScriptFunction::OPCODE_ASSIGN_FALSE);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_cast(const Address &p_target, const Address &p_source, const GDScriptDataType &p_type) {
	switch (p_type.kind) {
		case GDScriptDataType::BUILTIN: {
			append(GDScriptFunction::OPCODE_CAST_TO_BUILTIN);
			append(p_type.builtin_type);
		} break;
		case GDScriptDataType::NATIVE: {
			int class_idx = GDScriptLanguage::get_singleton()->get_global_map()[p_type.native_type];
			class_idx |= (GDScriptFunction::ADDR_TYPE_GLOBAL << GDScriptFunction::ADDR_BITS);
			append(GDScriptFunction::OPCODE_CAST_TO_NATIVE);
			append(class_idx);
		} break;
		case GDScriptDataType::SCRIPT:
		case GDScriptDataType::GDSCRIPT: {
			Variant script = p_type.script_type;
			int idx = get_constant_pos(script);

			append(GDScriptFunction::OPCODE_CAST_TO_SCRIPT);
			append(idx);
		} break;
		default: {
			return;
		}
	}

	append(p_source);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_call(const Address &p_target, const Address &p_base, const StringName &p_function_name, const Vector<Address> &p_arguments) {
	append(p_target.mode == Address::NIL ? GDScriptFunction::OPCODE_CALL : GDScriptFunction::OPCODE_CALL_RETURN);
	append(p_arguments.size());
	append(p_base);
	append(p_function_name);
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	alloc_call(p_arguments.size());
}

void GDScriptByteCodeGenerator::write_super_call(const Address &p_target, const StringName &p_function_name, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CALL_SELF_BASE);
	append(p_function_name);
	append(p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	alloc_call(p_arguments.size());
}

void GDScriptByteCodeGenerator::write_call_async(const Address &p_target, const Address &p_base, const StringName &p_function_name, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CALL_ASYNC);
	append(p_arguments.size());
	append(p_base);
	append(p_function_name);
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	alloc_call(p_arguments.size());
}

void GDScriptByteCodeGenerator::write_call_builtin(const Address &p_target, GDScriptFunctions::Function p_function, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CALL_BUILT_IN);
	append(p_function);
	append(p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	alloc_call(p_arguments.size());
}

void GDScriptByteCodeGenerator::write_call_method_bind(const Address &p_target, const Address &p_base, const MethodBind *p_method, const Vector<Address> &p_arguments) {
	append(p_target.mode == Address::NIL ? GDScriptFunction::OPCODE_CALL : GDScriptFunction::OPCODE_CALL_RETURN);
	append(p_arguments.size());
	append(p_base);
	append(p_method->get_name());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	alloc_call(p_arguments.size());
}

void GDScriptByteCodeGenerator::write_call_ptrcall(const Address &p_target, const Address &p_base, const MethodBind *p_method, const Vector<Address> &p_arguments) {
	append(p_target.mode == Address::NIL ? GDScriptFunction::OPCODE_CALL : GDScriptFunction::OPCODE_CALL_RETURN);
	append(p_arguments.size());
	append(p_base);
	append(p_method->get_name());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	alloc_call(p_arguments.size());
}

void GDScriptByteCodeGenerator::write_call_self(const Address &p_target, const StringName &p_function_name, const Vector<Address> &p_arguments) {
	append(p_target.mode == Address::NIL ? GDScriptFunction::OPCODE_CALL : GDScriptFunction::OPCODE_CALL_RETURN);
	append(p_arguments.size());
	append(GDScriptFunction::ADDR_TYPE_SELF << GDScriptFunction::ADDR_BITS);
	append(p_function_name);
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	alloc_call(p_arguments.size());
}

void GDScriptByteCodeGenerator::write_call_script_function(const Address &p_target, const Address &p_base, const StringName &p_function_name, const Vector<Address> &p_arguments) {
	append(p_target.mode == Address::NIL ? GDScriptFunction::OPCODE_CALL : GDScriptFunction::OPCODE_CALL_RETURN);
	append(p_arguments.size());
	append(p_base);
	append(p_function_name);
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	alloc_call(p_arguments.size());
}

void GDScriptByteCodeGenerator::write_construct(const Address &p_target, Variant::Type p_type, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CONSTRUCT);
	append(p_type);
	append(p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
	alloc_call(p_arguments.size());
}

void GDScriptByteCodeGenerator::write_construct_array(const Address &p_target, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CONSTRUCT_ARRAY);
	append(p_arguments.size());
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
}

void GDScriptByteCodeGenerator::write_construct_dictionary(const Address &p_target, const Vector<Address> &p_arguments) {
	append(GDScriptFunction::OPCODE_CONSTRUCT_DICTIONARY);
	append(p_arguments.size() / 2); // This is number of key-value pairs, so only half of actual arguments.
	for (int i = 0; i < p_arguments.size(); i++) {
		append(p_arguments[i]);
	}
	append(p_target);
}

void GDScriptByteCodeGenerator::write_await(const Address &p_target, const Address &p_operand) {
	append(GDScriptFunction::OPCODE_AWAIT);
	append(p_operand);
	append(GDScriptFunction::OPCODE_AWAIT_RESUME);
	append(p_target);
}

void GDScriptByteCodeGenerator::write_if(const Address &p_condition) {
	append(GDScriptFunction::OPCODE_JUMP_IF_NOT);
	append(p_condition);
	if_jmp_addrs.push_back(opcodes.size());
	append(0); // Jump destination, will be patched.
}

void GDScriptByteCodeGenerator::write_else() {
	append(GDScriptFunction::OPCODE_JUMP); // Jump from true if block;
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

void GDScriptByteCodeGenerator::write_for(const Address &p_variable, const Address &p_list) {
	int counter_pos = add_temporary() | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
	int container_pos = add_temporary() | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);

	current_breaks_to_patch.push_back(List<int>());

	// Assign container.
	append(GDScriptFunction::OPCODE_ASSIGN);
	append(container_pos);
	append(p_list);

	// Begin loop.
	append(GDScriptFunction::OPCODE_ITERATE_BEGIN);
	append(counter_pos);
	append(container_pos);
	for_jmp_addrs.push_back(opcodes.size());
	append(0); // End of loop address, will be patched.
	append(p_variable);
	append(GDScriptFunction::OPCODE_JUMP);
	append(opcodes.size() + 6); // Skip over 'continue' code.

	// Next iteration.
	int continue_addr = opcodes.size();
	continue_addrs.push_back(continue_addr);
	append(GDScriptFunction::OPCODE_ITERATE);
	append(counter_pos);
	append(container_pos);
	for_jmp_addrs.push_back(opcodes.size());
	append(0); // Jump destination, will be patched.
	append(p_variable);
}

void GDScriptByteCodeGenerator::write_endfor() {
	// Jump back to loop check.
	append(GDScriptFunction::OPCODE_JUMP);
	append(continue_addrs.back()->get());
	continue_addrs.pop_back();

	// Patch end jumps (two of them).
	for (int i = 0; i < 2; i++) {
		patch_jump(for_jmp_addrs.back()->get());
		for_jmp_addrs.pop_back();
	}

	// Patch break statements.
	for (const List<int>::Element *E = current_breaks_to_patch.back()->get().front(); E; E = E->next()) {
		patch_jump(E->get());
	}
	current_breaks_to_patch.pop_back();

	// Remove loop temporaries.
	pop_temporary();
	pop_temporary();
}

void GDScriptByteCodeGenerator::start_while_condition() {
	current_breaks_to_patch.push_back(List<int>());
	continue_addrs.push_back(opcodes.size());
}

void GDScriptByteCodeGenerator::write_while(const Address &p_condition) {
	// Condition check.
	append(GDScriptFunction::OPCODE_JUMP_IF_NOT);
	append(p_condition);
	while_jmp_addrs.push_back(opcodes.size());
	append(0); // End of loop address, will be patched.
}

void GDScriptByteCodeGenerator::write_endwhile() {
	// Jump back to loop check.
	append(GDScriptFunction::OPCODE_JUMP);
	append(continue_addrs.back()->get());
	continue_addrs.pop_back();

	// Patch end jump.
	patch_jump(while_jmp_addrs.back()->get());
	while_jmp_addrs.pop_back();

	// Patch break statements.
	for (const List<int>::Element *E = current_breaks_to_patch.back()->get().front(); E; E = E->next()) {
		patch_jump(E->get());
	}
	current_breaks_to_patch.pop_back();
}

void GDScriptByteCodeGenerator::start_match() {
	match_continues_to_patch.push_back(List<int>());
}

void GDScriptByteCodeGenerator::start_match_branch() {
	// Patch continue statements.
	for (const List<int>::Element *E = match_continues_to_patch.back()->get().front(); E; E = E->next()) {
		patch_jump(E->get());
	}
	match_continues_to_patch.pop_back();
	// Start a new list for next branch.
	match_continues_to_patch.push_back(List<int>());
}

void GDScriptByteCodeGenerator::end_match() {
	// Patch continue statements.
	for (const List<int>::Element *E = match_continues_to_patch.back()->get().front(); E; E = E->next()) {
		patch_jump(E->get());
	}
	match_continues_to_patch.pop_back();
}

void GDScriptByteCodeGenerator::write_break() {
	append(GDScriptFunction::OPCODE_JUMP);
	current_breaks_to_patch.back()->get().push_back(opcodes.size());
	append(0);
}

void GDScriptByteCodeGenerator::write_continue() {
	append(GDScriptFunction::OPCODE_JUMP);
	append(continue_addrs.back()->get());
}

void GDScriptByteCodeGenerator::write_continue_match() {
	append(GDScriptFunction::OPCODE_JUMP);
	match_continues_to_patch.back()->get().push_back(opcodes.size());
	append(0);
}

void GDScriptByteCodeGenerator::write_breakpoint() {
	append(GDScriptFunction::OPCODE_BREAKPOINT);
}

void GDScriptByteCodeGenerator::write_newline(int p_line) {
	append(GDScriptFunction::OPCODE_LINE);
	append(p_line);
	current_line = p_line;
}

void GDScriptByteCodeGenerator::write_return(const Address &p_return_value) {
	append(GDScriptFunction::OPCODE_RETURN);
	append(p_return_value);
}

void GDScriptByteCodeGenerator::write_assert(const Address &p_test, const Address &p_message) {
	append(GDScriptFunction::OPCODE_ASSERT);
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
