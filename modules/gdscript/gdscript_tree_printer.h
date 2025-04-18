/**************************************************************************/
/*  gdscript_tree_printer.h                                               */
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

#pragma once

#ifdef DEBUG_ENABLED

#include "core/string/string_builder.h"
#include "core/string/ustring.h"
#include "gdscript_parser.h"

class GDScriptTreePrinter {
	int indent_level = 0;
	String indent;
	StringBuilder printed;
	bool pending_indent = false;

	void increase_indent();
	void decrease_indent();
	void push_line(const String &p_line = String());
	void push_text(const String &p_text);

	void print_annotation(const GDScriptParser::AnnotationNode *p_annotation);
	void print_array(GDScriptParser::ArrayNode *p_array);
	void print_assert(GDScriptParser::AssertNode *p_assert);
	void print_assignment(GDScriptParser::AssignmentNode *p_assignment);
	void print_await(GDScriptParser::AwaitNode *p_await);
	void print_binary_op(GDScriptParser::BinaryOpNode *p_binary_op);
	void print_call(GDScriptParser::CallNode *p_call);
	void print_cast(GDScriptParser::CastNode *p_cast);
	void print_class(GDScriptParser::ClassNode *p_class);
	void print_constant(GDScriptParser::ConstantNode *p_constant);
	void print_dictionary(GDScriptParser::DictionaryNode *p_dictionary);
	void print_expression(GDScriptParser::ExpressionNode *p_expression);
	void print_enum(GDScriptParser::EnumNode *p_enum);
	void print_for(GDScriptParser::ForNode *p_for);
	void print_function(GDScriptParser::FunctionNode *p_function, const String &p_context = "Function");
	void print_get_node(GDScriptParser::GetNodeNode *p_get_node);
	void print_if(GDScriptParser::IfNode *p_if, bool p_is_elif = false);
	void print_identifier(GDScriptParser::IdentifierNode *p_identifier);
	void print_lambda(GDScriptParser::LambdaNode *p_lambda);
	void print_literal(GDScriptParser::LiteralNode *p_literal);
	void print_match(GDScriptParser::MatchNode *p_match);
	void print_match_branch(GDScriptParser::MatchBranchNode *p_match_branch);
	void print_match_pattern(GDScriptParser::PatternNode *p_match_pattern);
	void print_parameter(GDScriptParser::ParameterNode *p_parameter);
	void print_preload(GDScriptParser::PreloadNode *p_preload);
	void print_return(GDScriptParser::ReturnNode *p_return);
	void print_self(GDScriptParser::SelfNode *p_self);
	void print_signal(GDScriptParser::SignalNode *p_signal);
	void print_statement(GDScriptParser::Node *p_statement);
	void print_subscript(GDScriptParser::SubscriptNode *p_subscript);
	void print_suite(GDScriptParser::SuiteNode *p_suite);
	void print_ternary_op(GDScriptParser::TernaryOpNode *p_ternary_op);
	void print_type(GDScriptParser::TypeNode *p_type);
	void print_type_test(GDScriptParser::TypeTestNode *p_type_test);
	void print_unary_op(GDScriptParser::UnaryOpNode *p_unary_op);
	void print_variable(GDScriptParser::VariableNode *p_variable);
	void print_while(GDScriptParser::WhileNode *p_while);

public:
	void print_tree(const GDScriptParser &p_parser);
};
#endif // DEBUG_ENABLED
