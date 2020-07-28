/*************************************************************************/
/*  gdscript_codegen.h                                                   */
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

#ifndef GDSCRIPT_CODEGEN
#define GDSCRIPT_CODEGEN

#include "core/io/multiplayer_api.h"
#include "core/string_name.h"
#include "core/variant.h"
#include "gdscript_function.h"
#include "gdscript_functions.h"

class GDScriptCodeGenerator {
public:
	struct Address {
		enum AddressMode {
			SELF,
			CLASS,
			MEMBER,
			CONSTANT,
			CLASS_CONSTANT,
			LOCAL_CONSTANT,
			LOCAL_VARIABLE,
			FUNCTION_PARAMETER,
			TEMPORARY,
			GLOBAL,
			NAMED_GLOBAL,
			NIL,
		};
		AddressMode mode = NIL;
		uint32_t address = 0;
		GDScriptDataType type;

		Address() {}
		Address(AddressMode p_mode, const GDScriptDataType &p_type = GDScriptDataType()) {
			mode = p_mode;
			type = p_type;
		}
		Address(AddressMode p_mode, uint32_t p_address, const GDScriptDataType &p_type = GDScriptDataType()) {
			mode = p_mode,
			address = p_address;
			type = p_type;
		}
	};

	virtual uint32_t add_parameter(const StringName &p_name, bool p_is_optional, const GDScriptDataType &p_type) = 0;
	virtual uint32_t add_local(const StringName &p_name, const GDScriptDataType &p_type) = 0;
	virtual uint32_t add_local_constant(const StringName &p_name, const Variant &p_constant) = 0;
	virtual uint32_t add_or_get_constant(const Variant &p_constant) = 0;
	virtual uint32_t add_or_get_name(const StringName &p_name) = 0;
	virtual uint32_t add_temporary() = 0;
	virtual void pop_temporary() = 0;

	virtual void start_parameters() = 0;
	virtual void end_parameters() = 0;

	virtual void start_block() = 0;
	virtual void end_block() = 0;

	// virtual int get_max_stack_level() = 0;
	// virtual int get_max_function_arguments() = 0;

	virtual void write_start(GDScript *p_script, const StringName &p_function_name, bool p_static, MultiplayerAPI::RPCMode p_rpc_mode, const GDScriptDataType &p_return_type) = 0;
	virtual GDScriptFunction *write_end() = 0;

#ifdef DEBUG_ENABLED
	virtual void set_signature(const String &p_signature) = 0;
#endif
	virtual void set_initial_line(int p_line) = 0;

	// virtual void alloc_stack(int p_level) = 0; // Is this needed?
	// virtual void alloc_call(int p_arg_count) = 0; // This might be automatic from other functions.

	virtual void write_operator(const Address &p_target, Variant::Operator p_operator, const Address &p_left_operand, const Address &p_right_operand) = 0;
	virtual void write_type_test(const Address &p_target, const Address &p_source, const Address &p_type) = 0;
	virtual void write_type_test_builtin(const Address &p_target, const Address &p_source, Variant::Type p_type) = 0;
	virtual void write_and_left_operand(const Address &p_left_operand) = 0;
	virtual void write_and_right_operand(const Address &p_right_operand) = 0;
	virtual void write_end_and(const Address &p_target) = 0;
	virtual void write_or_left_operand(const Address &p_left_operand) = 0;
	virtual void write_or_right_operand(const Address &p_right_operand) = 0;
	virtual void write_end_or(const Address &p_target) = 0;
	virtual void write_start_ternary(const Address &p_target) = 0;
	virtual void write_ternary_condition(const Address &p_condition) = 0;
	virtual void write_ternary_true_expr(const Address &p_expr) = 0;
	virtual void write_ternary_false_expr(const Address &p_expr) = 0;
	virtual void write_end_ternary() = 0;
	virtual void write_set(const Address &p_target, const Address &p_index, const Address &p_source) = 0;
	virtual void write_get(const Address &p_target, const Address &p_index, const Address &p_source) = 0;
	virtual void write_set_named(const Address &p_target, const StringName &p_name, const Address &p_source) = 0;
	virtual void write_get_named(const Address &p_target, const StringName &p_name, const Address &p_source) = 0;
	virtual void write_set_member(const Address &p_value, const StringName &p_name) = 0;
	virtual void write_get_member(const Address &p_target, const StringName &p_name) = 0;
	virtual void write_assign(const Address &p_target, const Address &p_source) = 0;
	virtual void write_assign_true(const Address &p_target) = 0;
	virtual void write_assign_false(const Address &p_target) = 0;
	virtual void write_cast(const Address &p_target, const Address &p_source, const GDScriptDataType &p_type) = 0;
	virtual void write_call(const Address &p_target, const Address &p_base, const StringName &p_function_name, const Vector<Address> &p_arguments) = 0;
	virtual void write_super_call(const Address &p_target, const StringName &p_function_name, const Vector<Address> &p_arguments) = 0;
	virtual void write_call_async(const Address &p_target, const Address &p_base, const StringName &p_function_name, const Vector<Address> &p_arguments) = 0;
	virtual void write_call_builtin(const Address &p_target, GDScriptFunctions::Function p_function, const Vector<Address> &p_arguments) = 0;
	virtual void write_call_method_bind(const Address &p_target, const Address &p_base, const MethodBind *p_method, const Vector<Address> &p_arguments) = 0;
	virtual void write_call_ptrcall(const Address &p_target, const Address &p_base, const MethodBind *p_method, const Vector<Address> &p_arguments) = 0;
	virtual void write_call_self(const Address &p_target, const StringName &p_function_name, const Vector<Address> &p_arguments) = 0;
	virtual void write_call_script_function(const Address &p_target, const Address &p_base, const StringName &p_function_name, const Vector<Address> &p_arguments) = 0;
	virtual void write_construct(const Address &p_target, Variant::Type p_type, const Vector<Address> &p_arguments) = 0;
	virtual void write_construct_array(const Address &p_target, const Vector<Address> &p_arguments) = 0;
	virtual void write_construct_dictionary(const Address &p_target, const Vector<Address> &p_arguments) = 0;
	virtual void write_await(const Address &p_target, const Address &p_operand) = 0;
	virtual void write_if(const Address &p_condition) = 0;
	// virtual void write_elseif(const Address &p_condition) = 0; This kind of makes things more difficult for no real benefit.
	virtual void write_else() = 0;
	virtual void write_endif() = 0;
	virtual void write_for(const Address &p_variable, const Address &p_list) = 0;
	virtual void write_endfor() = 0;
	virtual void start_while_condition() = 0; // Used to allow a jump to the expression evaluation.
	virtual void write_while(const Address &p_condition) = 0;
	virtual void write_endwhile() = 0;
	virtual void start_match() = 0;
	virtual void start_match_branch() = 0;
	virtual void end_match() = 0;
	virtual void write_break() = 0;
	virtual void write_continue() = 0;
	virtual void write_continue_match() = 0;
	virtual void write_breakpoint() = 0;
	virtual void write_newline(int p_line) = 0;
	virtual void write_return(const Address &p_return_value) = 0;
	virtual void write_assert(const Address &p_test, const Address &p_message) = 0;

	virtual ~GDScriptCodeGenerator() {}
};

#endif // GDSCRIPT_CODEGEN
