/*************************************************************************/
/*  gdscript_analyzer.h                                                  */
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

#ifndef GDSCRIPT_ANALYZER_H
#define GDSCRIPT_ANALYZER_H

#include "core/object.h"
#include "core/reference.h"
#include "core/set.h"
#include "gdscript_cache.h"
#include "gdscript_parser.h"

class GDScriptAnalyzer {
	GDScriptParser *parser = nullptr;
	HashMap<String, Ref<GDScriptParserRef>> depended_parsers;

	const GDScriptParser::EnumNode *current_enum = nullptr;

	Error resolve_inheritance(GDScriptParser::ClassNode *p_class, bool p_recursive = true);
	GDScriptParser::DataType resolve_datatype(GDScriptParser::TypeNode *p_type);

	void decide_suite_type(GDScriptParser::Node *p_suite, GDScriptParser::Node *p_statement);

	// This traverses the tree to resolve all TypeNodes.
	Error resolve_program();

	void resolve_annotation(GDScriptParser::AnnotationNode *p_annotation);
	void resolve_class_interface(GDScriptParser::ClassNode *p_class);
	void resolve_class_body(GDScriptParser::ClassNode *p_class);
	void resolve_function_signature(GDScriptParser::FunctionNode *p_function);
	void resolve_function_body(GDScriptParser::FunctionNode *p_function);
	void resolve_node(GDScriptParser::Node *p_node);
	void resolve_suite(GDScriptParser::SuiteNode *p_suite);
	void resolve_if(GDScriptParser::IfNode *p_if);
	void resolve_for(GDScriptParser::ForNode *p_for);
	void resolve_while(GDScriptParser::WhileNode *p_while);
	void resolve_variable(GDScriptParser::VariableNode *p_variable);
	void resolve_constant(GDScriptParser::ConstantNode *p_constant);
	void resolve_assert(GDScriptParser::AssertNode *p_assert);
	void resolve_match(GDScriptParser::MatchNode *p_match);
	void resolve_match_branch(GDScriptParser::MatchBranchNode *p_match_branch, GDScriptParser::ExpressionNode *p_match_test);
	void resolve_match_pattern(GDScriptParser::PatternNode *p_match_pattern, GDScriptParser::ExpressionNode *p_match_test);
	void resolve_parameter(GDScriptParser::ParameterNode *p_parameter);
	void resolve_return(GDScriptParser::ReturnNode *p_return);

	// Reduction functions.
	void reduce_expression(GDScriptParser::ExpressionNode *p_expression);
	void reduce_array(GDScriptParser::ArrayNode *p_array);
	void reduce_assignment(GDScriptParser::AssignmentNode *p_assignment);
	void reduce_await(GDScriptParser::AwaitNode *p_await);
	void reduce_binary_op(GDScriptParser::BinaryOpNode *p_binary_op);
	void reduce_call(GDScriptParser::CallNode *p_call, bool is_await = false);
	void reduce_cast(GDScriptParser::CastNode *p_cast);
	void reduce_dictionary(GDScriptParser::DictionaryNode *p_dictionary);
	void reduce_get_node(GDScriptParser::GetNodeNode *p_get_node);
	void reduce_identifier(GDScriptParser::IdentifierNode *p_identifier, bool can_be_builtin = false);
	void reduce_identifier_from_base(GDScriptParser::IdentifierNode *p_identifier, GDScriptParser::DataType *p_base = nullptr);
	void reduce_literal(GDScriptParser::LiteralNode *p_literal);
	void reduce_preload(GDScriptParser::PreloadNode *p_preload);
	void reduce_self(GDScriptParser::SelfNode *p_self);
	void reduce_subscript(GDScriptParser::SubscriptNode *p_subscript);
	void reduce_ternary_op(GDScriptParser::TernaryOpNode *p_ternary_op);
	void reduce_unary_op(GDScriptParser::UnaryOpNode *p_unary_op);

	void const_fold_array(GDScriptParser::ArrayNode *p_array);
	void const_fold_dictionary(GDScriptParser::DictionaryNode *p_dictionary);

	// Helpers.
	GDScriptParser::DataType type_from_variant(const Variant &p_value, const GDScriptParser::Node *p_source);
	GDScriptParser::DataType type_from_metatype(const GDScriptParser::DataType &p_meta_type) const;
	GDScriptParser::DataType type_from_property(const PropertyInfo &p_property) const;
	GDScriptParser::DataType make_global_class_meta_type(const StringName &p_class_name);
	bool get_function_signature(GDScriptParser::Node *p_source, GDScriptParser::DataType base_type, const StringName &p_function, GDScriptParser::DataType &r_return_type, List<GDScriptParser::DataType> &r_par_types, int &r_default_arg_count, bool &r_static, bool &r_vararg);
	bool function_signature_from_info(const MethodInfo &p_info, GDScriptParser::DataType &r_return_type, List<GDScriptParser::DataType> &r_par_types, int &r_default_arg_count, bool &r_static, bool &r_vararg);
	bool validate_call_arg(const List<GDScriptParser::DataType> &p_par_types, int p_default_args_count, bool p_is_vararg, const GDScriptParser::CallNode *p_call);
	bool validate_call_arg(const MethodInfo &p_method, const GDScriptParser::CallNode *p_call);
	GDScriptParser::DataType get_operation_type(Variant::Operator p_operation, const GDScriptParser::DataType &p_a, const GDScriptParser::DataType &p_b, bool &r_valid, const GDScriptParser::Node *p_source);
	bool is_type_compatible(const GDScriptParser::DataType &p_target, const GDScriptParser::DataType &p_source, bool p_allow_implicit_conversion = false) const;
	void push_error(const String &p_message, const GDScriptParser::Node *p_origin);
	void mark_node_unsafe(const GDScriptParser::Node *p_node);
	bool class_exists(const StringName &p_class);
	Ref<GDScriptParserRef> get_parser_for(const String &p_path);
#ifdef DEBUG_ENABLED
	bool is_shadowing(GDScriptParser::IdentifierNode *p_local, const String &p_context);
#endif

public:
	Error resolve_inheritance();
	Error resolve_interface();
	Error resolve_body();
	Error analyze();

	GDScriptAnalyzer(GDScriptParser *p_parser);

	static void cleanup();
};

#endif // GDSCRIPT_ANALYZER_H
