/**************************************************************************/
/*  gdscript_inline_info_generator.h                                      */
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

#include "gdscript_parser.h"

class GDScriptInlineInfoGenerator {
	GDScriptParser *parser = nullptr;
	HashMap<int, TypedArray<Dictionary>> *inline_info = nullptr;

	void add_color_info(GDScriptParser::Node *p_node);
	void add_parameters_info(GDScriptParser::Node *p_node, const PropertyInfo &p_info);

	// Visitor functions.
	void visit_class(GDScriptParser::ClassNode *p_class);
	void visit_variable(GDScriptParser::VariableNode *p_variable);
	void visit_constant(GDScriptParser::ConstantNode *p_constant);
	void visit_function(GDScriptParser::FunctionNode *p_function);
	void visit_suite(GDScriptParser::SuiteNode *p_suite);
	void visit_statement(GDScriptParser::Node *p_statement);
	void visit_if(GDScriptParser::IfNode *p_if);
	void visit_for(GDScriptParser::ForNode *p_for);
	void visit_while(GDScriptParser::WhileNode *p_while);
	void visit_match(GDScriptParser::MatchNode *p_match);
	void visit_return(GDScriptParser::ReturnNode *p_return);
	void visit_assert(GDScriptParser::AssertNode *p_assert);
	void visit_expression(GDScriptParser::ExpressionNode *p_expression);
	void visit_call(GDScriptParser::CallNode *p_call);

	// Utility functions.
	MethodInfo get_method_info_on_base(GDScriptParser::Node *p_base, const StringName &p_function);

public:
	Error generate(HashMap<int, TypedArray<Dictionary>> *p_info);
	GDScriptInlineInfoGenerator(GDScriptParser *p_parser);
};
