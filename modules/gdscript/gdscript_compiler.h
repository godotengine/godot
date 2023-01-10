/**************************************************************************/
/*  gdscript_compiler.h                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef GDSCRIPT_COMPILER_H
#define GDSCRIPT_COMPILER_H

#include "core/set.h"
#include "gdscript.h"
#include "gdscript_parser.h"

class GDScriptCompiler {
	const GDScriptParser *parser;
	Set<GDScript *> parsed_classes;
	Set<GDScript *> parsing_classes;
	GDScript *main_script;
	struct CodeGen {
		GDScript *script;
		const GDScriptParser::ClassNode *class_node;
		const GDScriptParser::FunctionNode *function_node;
		bool debug_stack;

		List<Map<StringName, int>> stack_id_stack;
		Map<StringName, int> stack_identifiers;

		List<GDScriptFunction::StackDebug> stack_debug;
		List<Map<StringName, int>> block_identifier_stack;
		Map<StringName, int> block_identifiers;

		void add_stack_identifier(const StringName &p_id, int p_stackpos) {
			stack_identifiers[p_id] = p_stackpos;
			if (debug_stack) {
				block_identifiers[p_id] = p_stackpos;
				GDScriptFunction::StackDebug sd;
				sd.added = true;
				sd.line = current_line;
				sd.identifier = p_id;
				sd.pos = p_stackpos;
				stack_debug.push_back(sd);
			}
		}

		void push_stack_identifiers() {
			stack_id_stack.push_back(stack_identifiers);
			if (debug_stack) {
				block_identifier_stack.push_back(block_identifiers);
				block_identifiers.clear();
			}
		}

		void pop_stack_identifiers() {
			stack_identifiers = stack_id_stack.back()->get();
			stack_id_stack.pop_back();

			if (debug_stack) {
				for (Map<StringName, int>::Element *E = block_identifiers.front(); E; E = E->next()) {
					GDScriptFunction::StackDebug sd;
					sd.added = false;
					sd.identifier = E->key();
					sd.line = current_line;
					sd.pos = E->get();
					stack_debug.push_back(sd);
				}
				block_identifiers = block_identifier_stack.back()->get();
				block_identifier_stack.pop_back();
			}
		}

		HashMap<Variant, int, VariantHasher, VariantComparator> constant_map;
		Map<StringName, int> name_map;
#ifdef TOOLS_ENABLED
		Vector<StringName> named_globals;
#endif

		int get_name_map_pos(const StringName &p_identifier) {
			int ret;
			if (!name_map.has(p_identifier)) {
				ret = name_map.size();
				name_map[p_identifier] = ret;
			} else {
				ret = name_map[p_identifier];
			}
			return ret;
		}

		int get_constant_pos(const Variant &p_constant) {
			if (constant_map.has(p_constant)) {
				return constant_map[p_constant];
			}
			int pos = constant_map.size();
			constant_map[p_constant] = pos;
			return pos;
		}

		Vector<int> opcodes;
		void alloc_stack(int p_level) {
			if (p_level >= stack_max) {
				stack_max = p_level + 1;
			}
		}
		void alloc_call(int p_params) {
			if (p_params >= call_max) {
				call_max = p_params;
			}
		}

		int current_line;
		int stack_max;
		int call_max;
	};

	bool _is_class_member_property(CodeGen &codegen, const StringName &p_name);
	bool _is_class_member_property(GDScript *owner, const StringName &p_name);

	void _set_error(const String &p_error, const GDScriptParser::Node *p_node);

	bool _create_unary_operator(CodeGen &codegen, const GDScriptParser::OperatorNode *on, Variant::Operator op, int p_stack_level);
	bool _create_binary_operator(CodeGen &codegen, const GDScriptParser::OperatorNode *on, Variant::Operator op, int p_stack_level, bool p_initializer = false, int p_index_addr = 0);

	GDScriptDataType _gdtype_from_datatype(const GDScriptParser::DataType &p_datatype, GDScript *p_owner = nullptr) const;

	int _parse_assign_right_expression(CodeGen &codegen, const GDScriptParser::OperatorNode *p_expression, int p_stack_level, int p_index_addr = 0);
	int _parse_expression(CodeGen &codegen, const GDScriptParser::Node *p_expression, int p_stack_level, bool p_root = false, bool p_initializer = false, int p_index_addr = 0);
	Error _parse_block(CodeGen &codegen, const GDScriptParser::BlockNode *p_block, int p_stack_level = 0, int p_break_addr = -1, int p_continue_addr = -1);
	Error _parse_function(GDScript *p_script, const GDScriptParser::ClassNode *p_class, const GDScriptParser::FunctionNode *p_func, bool p_for_ready = false);
	Error _parse_class_level(GDScript *p_script, const GDScriptParser::ClassNode *p_class, bool p_keep_state);
	Error _parse_class_blocks(GDScript *p_script, const GDScriptParser::ClassNode *p_class, bool p_keep_state);
	void _make_scripts(GDScript *p_script, const GDScriptParser::ClassNode *p_class, bool p_keep_state);
	int err_line;
	int err_column;
	StringName source;
	String error;

public:
	Error compile(const GDScriptParser *p_parser, GDScript *p_script, bool p_keep_state = false);

	String get_error() const;
	int get_error_line() const;
	int get_error_column() const;

	GDScriptCompiler();
};

#endif // GDSCRIPT_COMPILER_H
