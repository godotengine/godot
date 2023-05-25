/**************************************************************************/
/*  gdscript_format.h                                                     */
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

#ifndef GDSCRIPT_FORMAT_H
#define GDSCRIPT_FORMAT_H

#include "core/string/ustring.h"
#include "gdscript_parser.h"

using GDP = GDScriptParser;

class GDScriptFormat {
private:
	enum BreakType {
		NONE,
		WRAP,
		SPLIT,
		FORCE_SPLIT,
	};

	typedef String (GDScriptFormat::*ParserFunc)(GDP::Node *, int, BreakType);
	HashSet<int> new_lines;
	List<GDScriptParser::ParserError> parser_errors;

private:
	static bool node_has_comments(const GDP::Node *p_node);
	static bool children_have_comments(const GDP::Node *p_parent, bool p_recursive = true);

	static int get_operation_priority(const GDP::BinaryOpNode::OpType p_op_type) {
		switch (p_op_type) {
			case GDP::BinaryOpNode::OpType::OP_MULTIPLICATION:
			case GDP::BinaryOpNode::OpType::OP_DIVISION:
			case GDP::BinaryOpNode::OpType::OP_MODULO:
			case GDP::BinaryOpNode::OpType::OP_POWER:
				return 0;
			case GDP::BinaryOpNode::OpType::OP_ADDITION:
			case GDP::BinaryOpNode::OpType::OP_SUBTRACTION:
				return 1;
			case GDP::BinaryOpNode::OpType::OP_BIT_LEFT_SHIFT:
			case GDP::BinaryOpNode::OpType::OP_BIT_RIGHT_SHIFT:
				return 2;
			case GDP::BinaryOpNode::OpType::OP_COMP_LESS:
			case GDP::BinaryOpNode::OpType::OP_COMP_LESS_EQUAL:
			case GDP::BinaryOpNode::OpType::OP_COMP_GREATER:
			case GDP::BinaryOpNode::OpType::OP_COMP_GREATER_EQUAL:
				return 3;
			case GDP::BinaryOpNode::OpType::OP_CONTENT_TEST:
			case GDP::BinaryOpNode::OpType::OP_COMP_EQUAL:
			case GDP::BinaryOpNode::OpType::OP_COMP_NOT_EQUAL:
				return 4;
			case GDP::BinaryOpNode::OpType::OP_BIT_AND:
				return 5;
			case GDP::BinaryOpNode::OpType::OP_BIT_XOR:
				return 6;
			case GDP::BinaryOpNode::OpType::OP_BIT_OR:
				return 7;
			case GDP::BinaryOpNode::OpType::OP_LOGIC_AND:
				return 8;
			case GDP::BinaryOpNode::OpType::OP_LOGIC_OR:
				return 9;
		}

		return 10;
	}

	static bool is_nestable_statement(const GDP::Node *p_node) {
		bool nestable_statement = true;
		if (p_node != nullptr) {
			switch (p_node->type) {
				case GDP::Node::Type::TYPE:
				case GDP::Node::Type::CAST:
				case GDP::Node::Type::LITERAL:
				case GDP::Node::Type::ASSIGNMENT:
				case GDP::Node::Type::IDENTIFIER:
				case GDP::Node::Type::GET_NODE:
				case GDP::Node::Type::SELF:
					nestable_statement = false;
					break;
				default:
					break;
			}
		}

		return nestable_statement;
	}

	static bool should_not_hold_comments(const GDP::Node *p_node) {
		return p_node->type == GDP::Node::Type::BINARY_OPERATOR || p_node->type == GDP::Node::Type::TERNARY_OPERATOR || p_node->type == GDP::Node::Type::SUITE;
	}

	static bool has_special_line_wrapping(const GDP::Node *p_node) {
		return p_node != nullptr && p_node->type == GDP::Node::Type::ASSERT;
	}

	static int get_length_without_comments(const String &p_string);

	static String parse_literal(const GDP::LiteralNode *p_node);
	static String parse_get_node(const GDP::GetNodeNode *p_node, int p_indent_level = 0);

	String indent(int p_count, bool p_newline = true);
	String print_comment(const GDP::Node *p_node, bool p_headers, int p_indent_level = 0);
	void find_custom_newlines(const String &p_code);
	String make_disabled_lines_from_headers(const String &p_input) const;

	String parse_type(const GDP::TypeNode *p_node);

	String parse_class(const GDP::ClassNode *p_node, int p_indent_level = 0);
	String parse_class_variable(const GDP::ClassNode *p_node, int p_indent_level, int p_member_index);

	String parse_expression(const GDP::ExpressionNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_suite(const GDP::SuiteNode *p_node, int p_indent_level = 0);

	String parse_var_value(const GDP::ExpressionNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_constant(const GDP::ConstantNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_signal(const GDP::SignalNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);

	String parse_annotations(const List<GDP::AnnotationNode *> &p_annotations, int p_indent_level = 0, BreakType p_break_type = NONE);

	String parse_property(const GDP::VariableNode *p_node, int p_indent_level = 0);
	String parse_variable(const GDP::VariableNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_variable_with_comments(const GDP::VariableNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);

	String parse_binary_operator_element(const GDP::ExpressionNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_binary_operator(const GDP::BinaryOpNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);

	String parse_array_element(const GDP::ExpressionNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_array_elements(const GDP::ArrayNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_array(const GDP::ArrayNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);

	String parse_dictionary_element(const GDP::ExpressionNode *p_key, const GDP::ExpressionNode *p_value, bool p_is_lua, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_dictionary_elements(const GDP::DictionaryNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_dictionary(const GDP::DictionaryNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);

	String parse_assignment(const GDP::AssignmentNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);

	String parse_call_arguments(const Vector<GDP::ExpressionNode *> &p_nodes, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_call(const GDP::CallNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);

	String parse_cast(const GDP::CastNode *p_node, int p_indent_level = 0);
	String parse_await(const GDP::AwaitNode *p_node, int p_indent_level = 0);
	String parse_subscript(const GDP::SubscriptNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_preload(const GDP::PreloadNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_ternary_op(const GDP::TernaryOpNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_type_test(const GDP::TypeTestNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_unary_op(const GDP::UnaryOpNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_lambda(const GDP::LambdaNode *p_node, int p_indent_level = 0);

	String parse_assert(const GDP::AssertNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_return(const GDP::ReturnNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_if(const GDP::IfNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_while(const GDP::WhileNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_for(const GDP::ForNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);

	String parse_enum_elements(const GDP::EnumNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_enum(const GDP::EnumNode *p_node, int p_indent_level = 0);

	String parse_match_pattern(const GDP::PatternNode *p_node, int p_indent_level = 0);
	String parse_match_branch(const GDP::MatchBranchNode *p_node, int p_indent_level = 0);
	String parse_match(const GDP::MatchNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);

	String parse_function(const GDP::FunctionNode *p_node, int p_indent_level = 0);
	String parse_function_signature(const GDP::FunctionNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_parameters(const Vector<GDP::ParameterNode *> &p_nodes, int p_indent_level = 0, BreakType p_break_type = NONE);
	String parse_parameter(const GDP::ParameterNode *p_node, int p_indent_level = 0, BreakType p_break_type = NONE);

public:
	int line_length_maximum;
	int tab_size;
	int tab_type;
	int lines_between_functions;
	int indent_in_multiline_block;

	GDScriptFormat();

	Error format(const String &p_code, String &r_formatted_code);
	List<GDScriptParser::ParserError> get_parser_errors() { return parser_errors; };
};

#endif // GDSCRIPT_FORMAT_H
