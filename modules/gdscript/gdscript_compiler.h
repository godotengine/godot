/*************************************************************************/
/*  gdscript_compiler.h                                                  */
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

#ifndef GDSCRIPT_COMPILER_H
#define GDSCRIPT_COMPILER_H

#include "core/set.h"
#include "gdscript.h"
#include "gdscript_codegen.h"
#include "gdscript_function.h"
#include "gdscript_parser.h"

class GDScriptCompiler {
	const GDScriptParser *parser = nullptr;
	Set<GDScript *> parsed_classes;
	Set<GDScript *> parsing_classes;
	GDScript *main_script = nullptr;

	struct CodeGen {
		GDScript *script = nullptr;
		const GDScriptParser::ClassNode *class_node = nullptr;
		const GDScriptParser::FunctionNode *function_node = nullptr;
		StringName function_name;
		GDScriptCodeGenerator *generator = nullptr;
		Map<StringName, GDScriptCodeGenerator::Address> parameters;
		Map<StringName, GDScriptCodeGenerator::Address> locals;
		List<Map<StringName, GDScriptCodeGenerator::Address>> locals_stack;

		GDScriptCodeGenerator::Address add_local(const StringName &p_name, const GDScriptDataType &p_type) {
			uint32_t addr = generator->add_local(p_name, p_type);
			locals[p_name] = GDScriptCodeGenerator::Address(GDScriptCodeGenerator::Address::LOCAL_VARIABLE, addr, p_type);
			return locals[p_name];
		}

		GDScriptCodeGenerator::Address add_local_constant(const StringName &p_name, const Variant &p_value) {
			uint32_t addr = generator->add_local_constant(p_name, p_value);
			locals[p_name] = GDScriptCodeGenerator::Address(GDScriptCodeGenerator::Address::LOCAL_CONSTANT, addr);
			return locals[p_name];
		}

		GDScriptCodeGenerator::Address add_temporary(const GDScriptDataType &p_type = GDScriptDataType()) {
			uint32_t addr = generator->add_temporary();
			return GDScriptCodeGenerator::Address(GDScriptCodeGenerator::Address::TEMPORARY, addr, p_type);
		}

		GDScriptCodeGenerator::Address add_constant(const Variant &p_constant) {
			GDScriptDataType type;
			type.has_type = true;
			type.kind = GDScriptDataType::BUILTIN;
			type.builtin_type = p_constant.get_type();
			if (type.builtin_type == Variant::OBJECT) {
				Object *obj = p_constant;
				if (obj) {
					type.kind = GDScriptDataType::NATIVE;
					type.native_type = obj->get_class_name();

					Ref<Script> script = obj->get_script();
					if (script.is_valid()) {
						type.script_type = script.ptr();
						Ref<GDScript> gdscript = script;
						if (gdscript.is_valid()) {
							type.kind = GDScriptDataType::GDSCRIPT;
						} else {
							type.kind = GDScriptDataType::SCRIPT;
						}
					}
				} else {
					type.builtin_type = Variant::NIL;
				}
			}

			uint32_t addr = generator->add_or_get_constant(p_constant);
			return GDScriptCodeGenerator::Address(GDScriptCodeGenerator::Address::CONSTANT, addr, type);
		}

		void start_block() {
			Map<StringName, GDScriptCodeGenerator::Address> old_locals = locals;
			locals_stack.push_back(old_locals);
			generator->start_block();
		}

		void end_block() {
			locals = locals_stack.back()->get();
			locals_stack.pop_back();
			generator->end_block();
		}
	};

	bool _is_class_member_property(CodeGen &codegen, const StringName &p_name);
	bool _is_class_member_property(GDScript *owner, const StringName &p_name);

	void _set_error(const String &p_error, const GDScriptParser::Node *p_node);

	Error _create_binary_operator(CodeGen &codegen, const GDScriptParser::BinaryOpNode *on, Variant::Operator op, bool p_initializer = false, const GDScriptCodeGenerator::Address &p_index_addr = GDScriptCodeGenerator::Address());
	Error _create_binary_operator(CodeGen &codegen, const GDScriptParser::ExpressionNode *p_left_operand, const GDScriptParser::ExpressionNode *p_right_operand, Variant::Operator op, bool p_initializer = false, const GDScriptCodeGenerator::Address &p_index_addr = GDScriptCodeGenerator::Address());

	GDScriptDataType _gdtype_from_datatype(const GDScriptParser::DataType &p_datatype, GDScript *p_owner = nullptr) const;

	GDScriptCodeGenerator::Address _parse_assign_right_expression(CodeGen &codegen, Error &r_error, const GDScriptParser::AssignmentNode *p_assignmentint, const GDScriptCodeGenerator::Address &p_index_addr = GDScriptCodeGenerator::Address());
	GDScriptCodeGenerator::Address _parse_expression(CodeGen &codegen, Error &r_error, const GDScriptParser::ExpressionNode *p_expression, bool p_root = false, bool p_initializer = false, const GDScriptCodeGenerator::Address &p_index_addr = GDScriptCodeGenerator::Address());
	GDScriptCodeGenerator::Address _parse_match_pattern(CodeGen &codegen, Error &r_error, const GDScriptParser::PatternNode *p_pattern, const GDScriptCodeGenerator::Address &p_value_addr, const GDScriptCodeGenerator::Address &p_type_addr, const GDScriptCodeGenerator::Address &p_previous_test, bool p_is_first, bool p_is_nested);
	void _add_locals_in_block(CodeGen &codegen, const GDScriptParser::SuiteNode *p_block);
	Error _parse_block(CodeGen &codegen, const GDScriptParser::SuiteNode *p_block, bool p_add_locals = true);
	Error _parse_function(GDScript *p_script, const GDScriptParser::ClassNode *p_class, const GDScriptParser::FunctionNode *p_func, bool p_for_ready = false);
	Error _parse_setter_getter(GDScript *p_script, const GDScriptParser::ClassNode *p_class, const GDScriptParser::VariableNode *p_variable, bool p_is_setter);
	Error _parse_class_level(GDScript *p_script, const GDScriptParser::ClassNode *p_class, bool p_keep_state);
	Error _parse_class_blocks(GDScript *p_script, const GDScriptParser::ClassNode *p_class, bool p_keep_state);
	void _make_scripts(GDScript *p_script, const GDScriptParser::ClassNode *p_class, bool p_keep_state);
	int err_line;
	int err_column;
	StringName source;
	String error;
	bool within_await = false;

public:
	Error compile(const GDScriptParser *p_parser, GDScript *p_script, bool p_keep_state = false);

	String get_error() const;
	int get_error_line() const;
	int get_error_column() const;

	GDScriptCompiler();
};

#endif // GDSCRIPT_COMPILER_H
