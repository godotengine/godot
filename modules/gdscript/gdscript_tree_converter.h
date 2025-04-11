/**************************************************************************/
/*  gdscript_tree_converter.h                                             */
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

#ifdef TOOLS_ENABLED

#include "core/variant/dictionary.h"
#include "gdscript_parser.h"

class GDScriptTreeConverter {
private:
	Dictionary to_dictionary(const PropertyInfo &self);
	Dictionary to_dictionary(const MethodInfo &self);
	Dictionary to_dictionary(const GDScriptParser::DataType &self);
	Dictionary to_dictionary(const GDScriptParser::ClassDocData &self);
	Dictionary to_dictionary(const GDScriptParser::MemberDocData &self);
	Dictionary to_dictionary(const GDScriptParser::Node &self);
	Dictionary to_dictionary_without_elevation(const GDScriptParser::Node &self);
	Dictionary to_dictionary(const GDScriptParser::ExpressionNode &self);
	Dictionary to_dictionary_without_elevation(const GDScriptParser::ExpressionNode &self);
	Dictionary to_dictionary(const GDScriptParser::AnnotationNode &self);
	Dictionary to_dictionary(const GDScriptParser::ArrayNode &self);
	Dictionary to_dictionary(const GDScriptParser::AssertNode &self);
	Dictionary to_dictionary(const GDScriptParser::AssignableNode &self);
	Dictionary to_dictionary_without_elevation(const GDScriptParser::AssignableNode &self);
	Dictionary to_dictionary(const GDScriptParser::AssignmentNode &self);
	Dictionary to_dictionary(const GDScriptParser::AwaitNode &self);
	Dictionary to_dictionary(const GDScriptParser::BinaryOpNode &self);
	Dictionary to_dictionary(const GDScriptParser::BreakNode &self);
	Dictionary to_dictionary(const GDScriptParser::BreakpointNode &self);
	Dictionary to_dictionary(const GDScriptParser::CallNode &self);
	Dictionary to_dictionary(const GDScriptParser::CastNode &self);
	Dictionary to_dictionary(const GDScriptParser::EnumNode::Value &self);
	Dictionary to_dictionary(const GDScriptParser::EnumNode &self);
	Dictionary to_dictionary(const GDScriptParser::ClassNode::Member &self);

public:
	Dictionary to_dictionary(const GDScriptParser::ClassNode &self);

private:
	Dictionary to_dictionary(const GDScriptParser::ConstantNode &self);
	Dictionary to_dictionary(const GDScriptParser::ContinueNode &self);
	Dictionary to_dictionary(const GDScriptParser::DictionaryNode::Pair &self);
	Dictionary to_dictionary(const GDScriptParser::DictionaryNode &self);
	Dictionary to_dictionary(const GDScriptParser::ForNode &self);
	Dictionary to_dictionary(const GDScriptParser::FunctionNode &self);
	Dictionary to_dictionary(const GDScriptParser::GetNodeNode &self);
	Dictionary to_dictionary(const GDScriptParser::IdentifierNode &self);
	Dictionary to_dictionary(const GDScriptParser::IfNode &self);
	Dictionary to_dictionary(const GDScriptParser::LambdaNode &self);
	Dictionary to_dictionary(const GDScriptParser::LiteralNode &self);
	Dictionary to_dictionary(const GDScriptParser::MatchNode &self);
	Dictionary to_dictionary(const GDScriptParser::MatchBranchNode &self);
	Dictionary to_dictionary(const GDScriptParser::ParameterNode &self);
	Dictionary to_dictionary(const GDScriptParser::PassNode &self);
	Dictionary to_dictionary(const GDScriptParser::PatternNode::Pair &self);
	Dictionary to_dictionary(const GDScriptParser::PatternNode &self);
	Dictionary to_dictionary(const GDScriptParser::PreloadNode &self);
	Dictionary to_dictionary(const GDScriptParser::ReturnNode &self);
	Dictionary to_dictionary(const GDScriptParser::SelfNode &self);
	Dictionary to_dictionary(const GDScriptParser::SignalNode &self);
	Dictionary to_dictionary(const GDScriptParser::SubscriptNode &self);
	Dictionary to_dictionary(const GDScriptParser::SuiteNode::Local &self);
	Dictionary to_dictionary(const GDScriptParser::SuiteNode &self);
	Dictionary to_dictionary(const GDScriptParser::TernaryOpNode &self);
	Dictionary to_dictionary(const GDScriptParser::TypeNode &self);
	Dictionary to_dictionary(const GDScriptParser::TypeTestNode &self);
	Dictionary to_dictionary(const GDScriptParser::UnaryOpNode &self);
	Dictionary to_dictionary(const GDScriptParser::VariableNode &self);
	Dictionary to_dictionary(const GDScriptParser::WhileNode &self);
	//Dictionary to_dictionary(const GDScriptParser::AnnotationInfo &self);
};
#endif // TOOLS_ENABLED
