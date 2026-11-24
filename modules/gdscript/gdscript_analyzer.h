/**************************************************************************/
/*  gdscript_analyzer.h                                                   */
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

#include "gdscript_cache.h"
#include "gdscript_parser.h"
#include "toolchain/gdscript_ast.h"

#include "core/object/ref_counted.h"

class GDScriptAnalyzer {
	GDScriptParser *parser = nullptr;

	template <typename Fn>
	class Finally {
		Fn fn;

	public:
		Finally(Fn p_fn) :
				fn(p_fn) {}
		~Finally() {
			fn();
		}
	};

	const GDScriptAST::EnumNode *current_enum = nullptr;
	GDScriptAST::LambdaNode *current_lambda = nullptr;
	List<GDScriptAST::LambdaNode *> pending_body_resolution_lambdas;
	HashMap<const GDScriptAST::ClassNode *, Ref<GDScriptParserRef>> external_class_parser_cache;
	bool static_context = false;

	// Tests for detecting invalid overloading of script members
	static _FORCE_INLINE_ bool has_member_name_conflict_in_script_class(const StringName &p_name, const GDScriptAST::ClassNode *p_current_class_node, const GDScriptAST::Node *p_member);
	static _FORCE_INLINE_ bool has_member_name_conflict_in_native_type(const StringName &p_name, const StringName &p_native_type_string);
	Error check_native_member_name_conflict(const StringName &p_member_name, const GDScriptAST::Node *p_member_node, const StringName &p_native_type_string);
	Error check_class_member_name_conflict(const GDScriptAST::ClassNode *p_class_node, const StringName &p_member_name, const GDScriptAST::Node *p_member_node);

	void get_class_node_current_scope_classes(GDScriptAST::ClassNode *p_node, List<GDScriptAST::ClassNode *> *p_list, GDScriptAST::Node *p_source);

	Error resolve_class_inheritance(GDScriptAST::ClassNode *p_class, const GDScriptAST::Node *p_source = nullptr);
	Error resolve_class_inheritance(GDScriptAST::ClassNode *p_class, bool p_recursive);
	GDScriptAST::DataType resolve_datatype(GDScriptAST::TypeNode *p_type);

	void resolve_annotation(GDScriptAST::AnnotationNode *p_annotation);
	void resolve_class_member(GDScriptAST::ClassNode *p_class, const StringName &p_name, const GDScriptAST::Node *p_source = nullptr);
	void resolve_class_member(GDScriptAST::ClassNode *p_class, uint32_t p_index, const GDScriptAST::Node *p_source = nullptr);
	void resolve_class_interface(GDScriptAST::ClassNode *p_class, const GDScriptAST::Node *p_source = nullptr);
	void resolve_class_interface(GDScriptAST::ClassNode *p_class, bool p_recursive);
	void resolve_class_body(GDScriptAST::ClassNode *p_class, const GDScriptAST::Node *p_source = nullptr);
	void resolve_class_body(GDScriptAST::ClassNode *p_class, bool p_recursive);
	void resolve_function_signature(GDScriptAST::FunctionNode *p_function, const GDScriptAST::Node *p_source = nullptr, bool p_is_lambda = false);
	void resolve_function_body(GDScriptAST::FunctionNode *p_function, bool p_is_lambda = false);
	void resolve_node(GDScriptAST::Node *p_node, bool p_is_root = true);
	void resolve_suite(GDScriptAST::SuiteNode *p_suite, bool p_is_root = true);
	void resolve_assignable(GDScriptAST::AssignableNode *p_assignable, const char *p_kind);
	void resolve_variable(GDScriptAST::VariableNode *p_variable, bool p_is_local);
	void resolve_constant(GDScriptAST::ConstantNode *p_constant, bool p_is_local);
	void resolve_parameter(GDScriptAST::ParameterNode *p_parameter);
	void resolve_if(GDScriptAST::IfNode *p_if);
	void resolve_for(GDScriptAST::ForNode *p_for);
	void resolve_while(GDScriptAST::WhileNode *p_while);
	void resolve_assert(GDScriptAST::AssertNode *p_assert);
	void resolve_match(GDScriptAST::MatchNode *p_match);
	void resolve_match_branch(GDScriptAST::MatchBranchNode *p_match_branch, GDScriptAST::ExpressionNode *p_match_test);
	void resolve_match_pattern(GDScriptAST::PatternNode *p_match_pattern, GDScriptAST::ExpressionNode *p_match_test);
	void resolve_return(GDScriptAST::ReturnNode *p_return);

	// Reduction functions.
	void reduce_expression(GDScriptAST::ExpressionNode *p_expression, bool p_is_root = false);
	void reduce_array(GDScriptAST::ArrayNode *p_array);
	void reduce_assignment(GDScriptAST::AssignmentNode *p_assignment);
	void reduce_await(GDScriptAST::AwaitNode *p_await);
	void reduce_binary_op(GDScriptAST::BinaryOpNode *p_binary_op);
	void reduce_call(GDScriptAST::CallNode *p_call, bool p_is_await = false, bool p_is_root = false);
	void reduce_cast(GDScriptAST::CastNode *p_cast);
	void reduce_dictionary(GDScriptAST::DictionaryNode *p_dictionary);
	void reduce_get_node(GDScriptAST::GetNodeNode *p_get_node);
	void reduce_identifier(GDScriptAST::IdentifierNode *p_identifier, bool can_be_builtin = false);
	void reduce_identifier_from_base(GDScriptAST::IdentifierNode *p_identifier, GDScriptAST::DataType *p_base = nullptr);
	void reduce_lambda(GDScriptAST::LambdaNode *p_lambda);
	void reduce_literal(GDScriptAST::LiteralNode *p_literal);
	void reduce_preload(GDScriptAST::PreloadNode *p_preload);
	void reduce_self(GDScriptAST::SelfNode *p_self);
	void reduce_subscript(GDScriptAST::SubscriptNode *p_subscript, bool p_can_be_pseudo_type = false);
	void reduce_ternary_op(GDScriptAST::TernaryOpNode *p_ternary_op, bool p_is_root = false);
	void reduce_type_test(GDScriptAST::TypeTestNode *p_type_test);
	void reduce_unary_op(GDScriptAST::UnaryOpNode *p_unary_op);

	// These methods provide a fallback mechanism for constant folding in constant contexts (constant initializers,
	// annotation arguments) and allow constant folding even when the normal constant folding mechanism does not consider
	// an expression to be constant (usually due to the presence of `[...]` and `{...}` constructs in it).
	// These methods assume that the normal expression reduction mechanism has already been performed.
	Variant make_expression_reduced_value(GDScriptAST::ExpressionNode *p_expression, bool &r_is_reduced);
	Variant make_array_reduced_value(GDScriptAST::ArrayNode *p_array, bool &r_is_reduced);
	Variant make_dictionary_reduced_value(GDScriptAST::DictionaryNode *p_dictionary, bool &r_is_reduced);
	Variant make_subscript_reduced_value(GDScriptAST::SubscriptNode *p_subscript, bool &r_is_reduced);
	Variant make_call_reduced_value(GDScriptAST::CallNode *p_call, bool &r_is_reduced);
	Variant make_binary_op_reduced_value(GDScriptAST::BinaryOpNode *p_binary_op, bool &r_is_reduced);
	Variant make_ternary_op_reduced_value(GDScriptAST::TernaryOpNode *p_ternary_op, bool &r_is_reduced);
	Variant make_cast_reduced_value(GDScriptAST::CastNode *p_cast, bool &r_is_reduced);
	Variant make_type_test_reduced_value(GDScriptAST::TypeTestNode *p_type_test, bool &r_is_reduced);

	// Helpers.
	Array make_array_from_element_datatype(const GDScriptAST::DataType &p_element_datatype, const GDScriptAST::Node *p_source_node = nullptr);
	Dictionary make_dictionary_from_element_datatype(const GDScriptAST::DataType &p_key_element_datatype, const GDScriptAST::DataType &p_value_element_datatype, const GDScriptAST::Node *p_source_node = nullptr);
	GDScriptAST::DataType meta_type_from_container_type(const ContainerType &p_type, const GDScriptAST::Node *p_source);
	GDScriptAST::DataType type_from_script(const Ref<Script> &p_script, const GDScriptAST::Node *p_source, bool p_is_meta_type);
	GDScriptAST::DataType type_from_variant(const Variant &p_value, const GDScriptAST::Node *p_source);
	GDScriptAST::DataType type_from_property_hint_string(const String &p_type_name) const;
	GDScriptAST::DataType type_from_property(const PropertyInfo &p_property, bool p_is_arg, const GDScriptAST::Node *p_source) const;
	GDScriptAST::DataType make_global_class_meta_type(const StringName &p_class_name, const GDScriptAST::Node *p_source) const;
	bool get_function_signature(GDScriptAST::Node *p_source, bool p_is_constructor, GDScriptAST::DataType base_type, const StringName &p_function, GDScriptAST::DataType &r_return_type, List<GDScriptAST::DataType> &r_par_types, int &r_default_arg_count, BitField<MethodFlags> &r_method_flags, StringName *r_native_class = nullptr);
	bool function_signature_from_info(const MethodInfo &p_info, GDScriptAST::DataType &r_return_type, List<GDScriptAST::DataType> &r_par_types, int &r_default_arg_count, BitField<MethodFlags> &r_method_flags, const GDScriptAST::Node *p_source);
	void validate_call_arg(const List<GDScriptAST::DataType> &p_par_types, int p_default_args_count, bool p_is_vararg, const GDScriptAST::CallNode *p_call);
	void validate_call_arg(const MethodInfo &p_method, const GDScriptAST::CallNode *p_call);
	GDScriptAST::DataType get_operation_type(Variant::Operator p_operation, const GDScriptAST::DataType &p_a, const GDScriptAST::DataType &p_b, bool &r_valid, const GDScriptAST::Node *p_source);
	GDScriptAST::DataType get_operation_type(Variant::Operator p_operation, const GDScriptAST::DataType &p_a, bool &r_valid, const GDScriptAST::Node *p_source);
	void update_const_expression_builtin_type(GDScriptAST::ExpressionNode *p_expression, const GDScriptAST::DataType &p_type, const char *p_usage, bool p_is_cast = false);
	void update_array_literal_element_type(GDScriptAST::ArrayNode *p_array, const GDScriptAST::DataType &p_element_type);
	void update_dictionary_literal_element_type(GDScriptAST::DictionaryNode *p_dictionary, const GDScriptAST::DataType &p_key_element_type, const GDScriptAST::DataType &p_value_element_type);
	bool is_type_compatible(const GDScriptAST::DataType &p_target, const GDScriptAST::DataType &p_source, bool p_allow_implicit_conversion = false, const GDScriptAST::Node *p_source_node = nullptr);
	bool is_type_compatible_strict_collections(const GDScriptAST::DataType &p_target, const GDScriptAST::DataType &p_source);
	void push_error(const String &p_message, const GDScriptAST::Node *p_origin = nullptr) const;
	void push_error(const String &p_message, const GDScriptAST::Node *p_origin, int p_start_line, int p_start_column, int p_end_line, int p_end_column) const;
	/// Marks all lines of `p_node` for being displayed as unsafe in the editor. This has no side effects in the analyzer and is a no-op in template builds.
	void mark_node_unsafe(const GDScriptAST::Node *p_node) const;
	void downgrade_node_type_source(GDScriptAST::ExpressionNode *p_node);
	void mark_lambda_use_self();
	void resolve_pending_lambda_bodies();
	void reduce_identifier_from_base_set_class(GDScriptAST::IdentifierNode *p_identifier, GDScriptAST::DataType p_identifier_datatype);
	Ref<GDScriptParserRef> ensure_cached_external_parser_for_class(const GDScriptAST::ClassNode *p_class, const GDScriptAST::ClassNode *p_from_class, const char *p_context, const GDScriptAST::Node *p_source);
	Ref<GDScriptParserRef> find_cached_external_parser_for_class(const GDScriptAST::ClassNode *p_class, const Ref<GDScriptParserRef> &p_dependant_parser);
	Ref<GDScriptParserRef> find_cached_external_parser_for_class(const GDScriptAST::ClassNode *p_class, GDScriptParser *p_dependant_parser);
	Ref<GDScript> get_depended_shallow_script(const String &p_path, Error &r_error);

#ifdef DEBUG_ENABLED
	void is_shadowing(GDScriptAST::IdentifierNode *p_identifier, const String &p_context, const bool p_in_local_scope);
	void warn_confusable_temporary_modification(GDScriptAST::ExpressionNode *p_expression);
#endif // DEBUG_ENABLED

public:
	Error resolve_inheritance();
	Error resolve_interface();
	Error resolve_body();
	Error resolve_dependencies();
	Error analyze();

	Variant make_variable_default_value(GDScriptAST::VariableNode *p_variable);

	static bool check_type_compatibility(const GDScriptAST::DataType &p_target, const GDScriptAST::DataType &p_source, bool p_allow_implicit_conversion = false, const GDScriptAST::Node *p_source_node = nullptr);
	static GDScriptAST::DataType type_from_metatype(const GDScriptAST::DataType &p_meta_type);
	static bool class_exists(const StringName &p_class);

	GDScriptAnalyzer(GDScriptParser *p_parser);
};
