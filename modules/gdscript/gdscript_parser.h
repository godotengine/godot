/**************************************************************************/
/*  gdscript_parser.h                                                     */
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
#include "gdscript_tokenizer.h"
#include "toolchain/gdscript_ast.h"

#ifdef DEBUG_ENABLED
#include "gdscript_warning.h"

#include "core/string/string_builder.h" // For TreePrinter.
#endif

#include "core/object/ref_counted.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/list.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

class GDScriptParser {
public:
	using DataType = GDScriptAST::DataType;
	using AnnotationInfo = GDScriptAST::AnnotationInfo;

#ifdef TOOLS_ENABLED
	using MemberDocData = GDScriptAST::MemberDocData;
	using ClassDocData = GDScriptAST::ClassDocData;
#endif // TOOLS_ENABLED

	using AnnotationNode = GDScriptAST::AnnotationNode;
	using ArrayNode = GDScriptAST::ArrayNode;
	using AssertNode = GDScriptAST::AssertNode;
	using AssignableNode = GDScriptAST::AssignableNode;
	using AssignmentNode = GDScriptAST::AssignmentNode;
	using AwaitNode = GDScriptAST::AwaitNode;
	using BinaryOpNode = GDScriptAST::BinaryOpNode;
	using BreakNode = GDScriptAST::BreakNode;
	using BreakpointNode = GDScriptAST::BreakpointNode;
	using CallNode = GDScriptAST::CallNode;
	using CastNode = GDScriptAST::CastNode;
	using ClassNode = GDScriptAST::ClassNode;
	using ConstantNode = GDScriptAST::ConstantNode;
	using ContinueNode = GDScriptAST::ContinueNode;
	using DictionaryNode = GDScriptAST::DictionaryNode;
	using EnumNode = GDScriptAST::EnumNode;
	using ExpressionNode = GDScriptAST::ExpressionNode;
	using ForNode = GDScriptAST::ForNode;
	using FunctionNode = GDScriptAST::FunctionNode;
	using GetNodeNode = GDScriptAST::GetNodeNode;
	using IdentifierNode = GDScriptAST::IdentifierNode;
	using IfNode = GDScriptAST::IfNode;
	using LambdaNode = GDScriptAST::LambdaNode;
	using LiteralNode = GDScriptAST::LiteralNode;
	using MatchNode = GDScriptAST::MatchNode;
	using MatchBranchNode = GDScriptAST::MatchBranchNode;
	using Node = GDScriptAST::Node;
	using ParameterNode = GDScriptAST::ParameterNode;
	using PassNode = GDScriptAST::PassNode;
	using PatternNode = GDScriptAST::PatternNode;
	using PreloadNode = GDScriptAST::PreloadNode;
	using ReturnNode = GDScriptAST::ReturnNode;
	using SelfNode = GDScriptAST::SelfNode;
	using SignalNode = GDScriptAST::SignalNode;
	using SubscriptNode = GDScriptAST::SubscriptNode;
	using SuiteNode = GDScriptAST::SuiteNode;
	using TernaryOpNode = GDScriptAST::TernaryOpNode;
	using TypeNode = GDScriptAST::TypeNode;
	using TypeTestNode = GDScriptAST::TypeTestNode;
	using UnaryOpNode = GDScriptAST::UnaryOpNode;
	using VariableNode = GDScriptAST::VariableNode;
	using WhileNode = GDScriptAST::WhileNode;
	typedef bool (GDScriptParser::*AnnotationAction)(AnnotationNode *p_annotation, Node *p_target, ClassNode *p_class);

	struct ParserError {
		// TODO: Do I really need a "type"?
		// enum Type {
		//     NO_ERROR,
		//     EMPTY_FILE,
		//     CLASS_NAME_USED_TWICE,
		//     EXTENDS_USED_TWICE,
		//     EXPECTED_END_STATEMENT,
		// };
		// Type type = NO_ERROR;
		String message;
		int start_line;
		int start_column;
		int end_line;
		int end_column;
	};

	enum CompletionType {
		COMPLETION_NONE,
		COMPLETION_ANNOTATION, // Annotation (following @).
		COMPLETION_ANNOTATION_ARGUMENTS, // Annotation arguments hint.
		COMPLETION_ASSIGN, // Assignment based on type (e.g. enum values).
		COMPLETION_ATTRIBUTE, // After id.| to look for members.
		COMPLETION_ATTRIBUTE_METHOD, // After id.| to look for methods.
		COMPLETION_BUILT_IN_TYPE_CONSTANT_OR_STATIC_METHOD, // Constants inside a built-in type (e.g. Color.BLUE) or static methods (e.g. Color.html).
		COMPLETION_CALL_ARGUMENTS, // Complete with nodes, input actions, enum values (or usual expressions).
		COMPLETION_DECLARATION, // Potential declaration (var, const, class, etc.).
		COMPLETION_GET_NODE, // Get node with $ notation.
		COMPLETION_IDENTIFIER, // List available identifiers in scope.
		COMPLETION_INHERIT_TYPE, // Type after extends. Exclude non-viable types (built-ins, enums, void). Includes subtypes using the argument index.
		COMPLETION_METHOD, // List available methods in scope.
		COMPLETION_OVERRIDE_METHOD, // Override implementation, also for native virtuals.
		COMPLETION_PROPERTY_DECLARATION, // Property declaration (get, set).
		COMPLETION_PROPERTY_DECLARATION_OR_TYPE, // Property declaration (get, set) or a type hint.
		COMPLETION_PROPERTY_METHOD, // Property setter or getter (list available methods).
		COMPLETION_RESOURCE_PATH, // For load/preload.
		COMPLETION_SUBSCRIPT, // Inside id[|].
		COMPLETION_SUPER, // super(), used for lookup.
		COMPLETION_SUPER_METHOD, // After super.
		COMPLETION_TYPE_ATTRIBUTE, // Attribute in type name (Type.|).
		COMPLETION_TYPE_NAME, // Name of type (after :).
		COMPLETION_TYPE_NAME_OR_VOID, // Same as TYPE_NAME, but allows void (in function return type).
	};

	struct CompletionCall {
		GDScriptAST::Node *call = nullptr;
		int argument = -1;
	};

	struct CompletionContext {
		CompletionType type = COMPLETION_NONE;
		GDScriptAST::ClassNode *current_class = nullptr;
		GDScriptAST::FunctionNode *current_function = nullptr;
		GDScriptAST::SuiteNode *current_suite = nullptr;
		int current_line = -1;
		union {
			int current_argument = -1;
			int type_chain_index;
		};
		Variant::Type builtin_type = Variant::VARIANT_MAX;
		GDScriptAST::Node *node = nullptr;
		Object *base = nullptr;
		GDScriptParser *parser = nullptr;
		CompletionCall call;
	};

private:
	friend class GDScriptAnalyzer;
	friend class GDScriptParserRef;
	friend class GDScriptLinter;

	bool _is_tool = false;
	String script_path;
	bool for_completion = false;
	bool parse_body = true;
	bool panic_mode = false;
	bool can_break = false;
	bool can_continue = false;
	List<bool> multiline_stack;
	HashMap<String, Ref<GDScriptParserRef>> depended_parsers;

	GDScriptAST::ClassNode *head = nullptr;
	GDScriptAST::Node *list = nullptr;
	List<ParserError> errors;

#ifdef DEBUG_ENABLED
public:
	struct WarningDirectoryRule {
		enum Decision {
			DECISION_EXCLUDE,
			DECISION_INCLUDE,
			DECISION_MAX,
		};

		String directory_path; // With a trailing slash.
		Decision decision = DECISION_EXCLUDE;
	};

private:
	struct PendingWarning {
		int start_line = 0;
		int start_column = 0;
		int end_line = 0;
		int end_column = 0;
		GDScriptWarning::Code code = GDScriptWarning::WARNING_MAX;
		bool treated_as_error = false;
		Vector<String> symbols;
	};

	static bool is_project_ignoring_warnings;
	static GDScriptWarning::WarnLevel warning_levels[GDScriptWarning::WARNING_MAX];
	static LocalVector<WarningDirectoryRule> warning_directory_rules;

	List<GDScriptWarning> warnings;
	List<PendingWarning> pending_warnings;
	bool is_script_ignoring_warnings = false;
	HashSet<int> warning_ignored_lines[GDScriptWarning::WARNING_MAX];
	int warning_ignore_start_lines[GDScriptWarning::WARNING_MAX];
#endif // DEBUG_ENABLED

#ifdef TOOLS_ENABLED
	HashSet<int> unsafe_lines;
#endif

	GDScriptTokenizer *tokenizer = nullptr;
	GDScriptTokenizer::Token previous;
	GDScriptTokenizer::Token current;

	GDScriptAST::ClassNode *current_class = nullptr;
	GDScriptAST::FunctionNode *current_function = nullptr;
	GDScriptAST::LambdaNode *current_lambda = nullptr;
	GDScriptAST::SuiteNode *current_suite = nullptr;

	CompletionContext completion_context;
	List<CompletionCall> completion_call_stack;
	bool in_lambda = false;
	bool lambda_ended = false; // Marker for when a lambda ends, to apply an end of statement if needed.

	static HashMap<StringName, GDScriptAST::AnnotationInfo> valid_annotations;
	List<GDScriptAST::AnnotationNode *> annotation_stack;

	typedef GDScriptAST::ExpressionNode *(GDScriptParser::*ParseFunction)(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	// Higher value means higher precedence (i.e. is evaluated first).
	enum Precedence {
		PREC_NONE,
		PREC_ASSIGNMENT,
		PREC_CAST,
		PREC_TERNARY,
		PREC_LOGIC_OR,
		PREC_LOGIC_AND,
		PREC_LOGIC_NOT,
		PREC_CONTENT_TEST,
		PREC_COMPARISON,
		PREC_BIT_OR,
		PREC_BIT_XOR,
		PREC_BIT_AND,
		PREC_BIT_SHIFT,
		PREC_ADDITION_SUBTRACTION,
		PREC_FACTOR,
		PREC_SIGN,
		PREC_BIT_NOT,
		PREC_POWER,
		PREC_TYPE_TEST,
		PREC_AWAIT,
		PREC_CALL,
		PREC_ATTRIBUTE,
		PREC_SUBSCRIPT,
		PREC_PRIMARY,
	};
	struct ParseRule {
		ParseFunction prefix = nullptr;
		ParseFunction infix = nullptr;
		Precedence precedence = PREC_NONE;
	};
	static ParseRule *get_rule(GDScriptTokenizer::Token::Type p_token_type);

	List<GDScriptAST::Node *> nodes_in_progress;
	void complete_extents(GDScriptAST::Node *p_node);
	void update_extents(GDScriptAST::Node *p_node);
	void reset_extents(GDScriptAST::Node *p_node, GDScriptTokenizer::Token p_token);
	void reset_extents(GDScriptAST::Node *p_node, GDScriptAST::Node *p_from);
	void set_synthetic_extents(GDScriptAST::Node *p_node) {
		p_node->start_line = p_node->end_line = p_node->start_column = p_node->end_column = -1;
	}

	template <typename T>
	T *alloc_node() {
		T *node = memnew(T);

		node->next = list;
		list = node;

		reset_extents(node, previous);
		nodes_in_progress.push_back(node);

		return node;
	}

	// Allocates a node for patching up the parse tree when an error occurred.
	// Such nodes don't track their extents as they don't relate to actual tokens.
	template <typename T>
	T *alloc_recovery_node() {
		T *node = memnew(T);
		node->next = list;
		list = node;

		return node;
	}

	GDScriptAST::SuiteNode *alloc_recovery_suite() {
		GDScriptAST::SuiteNode *suite = alloc_recovery_node<GDScriptAST::SuiteNode>();
		suite->parent_block = current_suite;
		suite->parent_function = current_function;
		suite->is_in_loop = current_suite->is_in_loop;
		return suite;
	}

	void clear();

	void push_error(const String &p_message, const GDScriptAST::Node *p_origin = nullptr);
	void push_error(const String &p_message, int p_start_line, int p_start_column, int p_end_line, int p_end_column);
	void push_error(const String &p_message, const GDScriptTokenizer::Token &p_origin);

#ifdef DEBUG_ENABLED
public:
	void push_warning(const GDScriptAST::Node *p_source, GDScriptWarning::Code p_code, const Vector<String> &p_symbols);
	template <typename... Symbols>
	void push_warning(const GDScriptAST::Node *p_source, GDScriptWarning::Code p_code, const Symbols &...p_symbols) {
		push_warning(p_source, p_code, Vector<String>{ p_symbols... });
	}
	void push_warning(int p_start_line, int p_start_column, int p_end_line, int p_end_column, GDScriptWarning::Code p_code, const Vector<String> &p_symbols);
	template <typename... Symbols>
	void push_warning(int p_start_line, int p_start_column, int p_end_line, int p_end_column, GDScriptWarning::Code p_code, const Symbols &...p_symbols) {
		push_warning(p_start_line, p_start_column, p_end_line, p_end_column, p_code, Vector<String>{ p_symbols... });
	}

private:
	void apply_pending_warnings();
	void evaluate_warning_directory_rules_for_script_path();
#endif // DEBUG_ENABLED

	// Setting p_force to false will prevent the completion context from being update if a context was already set before.
	// This should only be done when we push context before we consumed any tokens for the corresponding structure.
	// See parse_precedence for an example.
	void make_completion_context(CompletionType p_type, GDScriptAST::Node *p_node, int p_argument = -1, bool p_force = true);
	void make_completion_context(CompletionType p_type, Variant::Type p_builtin_type, bool p_force = true);
	// In some cases it might become necessary to alter the completion context after parsing a subexpression.
	// For example to not override COMPLETE_CALL_ARGUMENTS with COMPLETION_NONE from string literals.
	void override_completion_context(const GDScriptAST::Node *p_for_node, CompletionType p_type, GDScriptAST::Node *p_node, int p_argument = -1);
	void push_completion_call(GDScriptAST::Node *p_call);
	void pop_completion_call();
	void set_last_completion_call_arg(int p_argument);

	GDScriptTokenizer::Token advance();
	bool match(GDScriptTokenizer::Token::Type p_token_type);
	bool check(GDScriptTokenizer::Token::Type p_token_type) const;
	bool consume(GDScriptTokenizer::Token::Type p_token_type, const String &p_error_message);
	bool is_at_end() const;
	bool is_statement_end_token() const;
	bool is_statement_end() const;
	void end_statement(const String &p_context);
	void synchronize();
	void push_multiline(bool p_state);
	void pop_multiline();

	// Main blocks.
	void parse_program();
	GDScriptAST::ClassNode *parse_class(bool p_is_static);
	void parse_class_name();
	void parse_extends();
	void parse_class_body(bool p_is_multiline);
	template <typename T>
	void parse_class_member(T *(GDScriptParser::*p_parse_function)(bool), GDScriptAST::AnnotationInfo::TargetKind p_target, const String &p_member_kind, bool p_is_static = false);
	GDScriptAST::SignalNode *parse_signal(bool p_is_static);
	GDScriptAST::EnumNode *parse_enum(bool p_is_static);
	GDScriptAST::ParameterNode *parse_parameter();
	GDScriptAST::FunctionNode *parse_function(bool p_is_static);
	bool parse_function_signature(GDScriptAST::FunctionNode *p_function, GDScriptAST::SuiteNode *p_body, const String &p_type, int p_signature_start);
	GDScriptAST::SuiteNode *parse_suite(const String &p_context, GDScriptAST::SuiteNode *p_suite = nullptr, bool p_for_lambda = false);
	// Annotations
	GDScriptAST::AnnotationNode *parse_annotation(uint32_t p_valid_targets);
	static bool register_annotation(const MethodInfo &p_info, uint32_t p_target_kinds, GDScriptAST::AnnotationAction p_apply, const Vector<Variant> &p_default_arguments = Vector<Variant>(), bool p_is_vararg = false);
	bool validate_annotation_arguments(GDScriptAST::AnnotationNode *p_annotation);
	void clear_unused_annotations();
	bool tool_annotation(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	bool icon_annotation(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	bool static_unload_annotation(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	bool abstract_annotation(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	bool onready_annotation(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	template <PropertyHint t_hint, Variant::Type t_type>
	bool export_annotations(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	bool _export_annotations(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class, PropertyHint p_hint, Variant::Type p_type);
	bool export_storage_annotation(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	bool export_custom_annotation(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	bool export_tool_button_annotation(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	template <PropertyUsageFlags t_usage>
	bool export_group_annotations(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	bool warning_ignore_annotation(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	bool warning_ignore_region_annotations(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	bool rpc_annotation(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);
	// Statements.
	GDScriptAST::Node *parse_statement();
	GDScriptAST::VariableNode *parse_variable(bool p_is_static);
	GDScriptAST::VariableNode *parse_variable(bool p_is_static, bool p_allow_property);
	GDScriptAST::VariableNode *parse_property(GDScriptAST::VariableNode *p_variable, bool p_need_indent);
	void parse_property_getter(GDScriptAST::VariableNode *p_variable);
	void parse_property_setter(GDScriptAST::VariableNode *p_variable);
	GDScriptAST::ConstantNode *parse_constant(bool p_is_static);
	GDScriptAST::AssertNode *parse_assert();
	GDScriptAST::BreakNode *parse_break();
	GDScriptAST::ContinueNode *parse_continue();
	GDScriptAST::ForNode *parse_for();
	GDScriptAST::IfNode *parse_if(const String &p_token = "if");
	GDScriptAST::MatchNode *parse_match();
	GDScriptAST::MatchBranchNode *parse_match_branch();
	GDScriptAST::PatternNode *parse_match_pattern(GDScriptAST::PatternNode *p_root_pattern = nullptr);
	GDScriptAST::WhileNode *parse_while();
	// Expressions.
	GDScriptAST::ExpressionNode *parse_expression(bool p_can_assign, bool p_stop_on_assign = false);
	GDScriptAST::ExpressionNode *parse_precedence(Precedence p_precedence, bool p_can_assign, bool p_stop_on_assign = false);
	GDScriptAST::ExpressionNode *parse_literal(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::LiteralNode *parse_literal();
	GDScriptAST::ExpressionNode *parse_self(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_identifier(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::IdentifierNode *parse_identifier();
	GDScriptAST::ExpressionNode *parse_builtin_constant(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_unary_operator(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_binary_operator(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_binary_not_in_operator(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_ternary_operator(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_assignment(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_array(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_dictionary(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_call(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_get_node(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_preload(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_grouping(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_cast(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_await(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_attribute(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_subscript(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_lambda(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_type_test(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_yield(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::ExpressionNode *parse_invalid_token(GDScriptAST::ExpressionNode *p_previous_operand, bool p_can_assign);
	GDScriptAST::TypeNode *parse_type(bool p_allow_void = false);
	// TODO: Remove in 5.x.
	bool parse_standalone_string();

#ifdef TOOLS_ENABLED
	int max_script_doc_line = INT_MAX;
	int min_member_doc_line = 1;
	bool has_comment(int p_line, bool p_must_be_doc = false);
	GDScriptAST::MemberDocData parse_doc_comment(int p_line, bool p_single_line = false);
	GDScriptAST::ClassDocData parse_class_doc_comment(int p_line, bool p_single_line = false);
#endif // TOOLS_ENABLED

public:
	Error parse(const String &p_source_code, const String &p_script_path, bool p_for_completion, bool p_parse_body = true);
	Error parse_binary(const Vector<uint8_t> &p_binary, const String &p_script_path);
	GDScriptAST::ClassNode *get_tree() const { return head; }
	bool is_tool() const { return _is_tool; }
	Ref<GDScriptParserRef> get_depended_parser_for(const String &p_path);
	const HashMap<String, Ref<GDScriptParserRef>> &get_depended_parsers();
	GDScriptAST::ClassNode *find_class(const String &p_qualified_name) const;
	bool has_class(const GDScriptAST::ClassNode *p_class) const;
	static Variant::Type get_builtin_type(const StringName &p_type); // Excluding `Variant::NIL` and `Variant::OBJECT`.

	CompletionContext get_completion_context() const { return completion_context; }
	void get_annotation_list(List<MethodInfo> *r_annotations) const;
	bool annotation_exists(const String &p_annotation_name) const;
	bool apply_annotation(GDScriptAST::AnnotationNode *p_annotation, GDScriptAST::Node *p_target, GDScriptAST::ClassNode *p_class);

	const List<ParserError> &get_errors() const { return errors; }
	const List<String> get_dependencies() const {
		// TODO: Keep track of deps.
		return List<String>();
	}

#ifdef DEBUG_ENABLED
	static void update_project_settings();
	const List<GDScriptWarning> &get_warnings() const { return warnings; }
#endif // DEBUG_ENABLED

#ifdef TOOLS_ENABLED
	static HashMap<String, String> theme_color_names;

	HashMap<int, GDScriptTokenizer::CommentData> comment_data;

	const HashSet<int> &get_unsafe_lines() const { return unsafe_lines; }
	int get_last_line_number() const { return current.end_line; }
#endif // TOOLS_ENABLED

	GDScriptParser();
	~GDScriptParser();

#ifdef DEBUG_ENABLED
	class TreePrinter {
		int indent_level = 0;
		String indent;
		StringBuilder printed;
		bool pending_indent = false;

		void increase_indent();
		void decrease_indent();
		void push_line(const String &p_line = String());
		void push_text(const String &p_text);

		void print_annotation(const GDScriptAST::AnnotationNode *p_annotation);
		void print_array(const GDScriptAST::ArrayNode *p_array);
		void print_assert(const GDScriptAST::AssertNode *p_assert);
		void print_assignment(const GDScriptAST::AssignmentNode *p_assignment);
		void print_await(const GDScriptAST::AwaitNode *p_await);
		void print_binary_op(const GDScriptAST::BinaryOpNode *p_binary_op);
		void print_call(const GDScriptAST::CallNode *p_call);
		void print_cast(const GDScriptAST::CastNode *p_cast);
		void print_class(const GDScriptAST::ClassNode *p_class);
		void print_constant(const GDScriptAST::ConstantNode *p_constant);
		void print_dictionary(const GDScriptAST::DictionaryNode *p_dictionary);
		void print_expression(const GDScriptAST::ExpressionNode *p_expression);
		void print_enum(const GDScriptAST::EnumNode *p_enum);
		void print_for(const GDScriptAST::ForNode *p_for);
		void print_function(const GDScriptAST::FunctionNode *p_function, const String &p_context = "Function");
		void print_get_node(const GDScriptAST::GetNodeNode *p_get_node);
		void print_if(const GDScriptAST::IfNode *p_if, bool p_is_elif = false);
		void print_identifier(const GDScriptAST::IdentifierNode *p_identifier);
		void print_lambda(const GDScriptAST::LambdaNode *p_lambda);
		void print_literal(const GDScriptAST::LiteralNode *p_literal);
		void print_match(const GDScriptAST::MatchNode *p_match);
		void print_match_branch(const GDScriptAST::MatchBranchNode *p_match_branch);
		void print_match_pattern(const GDScriptAST::PatternNode *p_match_pattern);
		void print_parameter(const GDScriptAST::ParameterNode *p_parameter);
		void print_preload(const GDScriptAST::PreloadNode *p_preload);
		void print_return(const GDScriptAST::ReturnNode *p_return);
		void print_self(const GDScriptAST::SelfNode *p_self);
		void print_signal(const GDScriptAST::SignalNode *p_signal);
		void print_statement(const GDScriptAST::Node *p_statement);
		void print_subscript(const GDScriptAST::SubscriptNode *p_subscript);
		void print_suite(const GDScriptAST::SuiteNode *p_suite);
		void print_ternary_op(const GDScriptAST::TernaryOpNode *p_ternary_op);
		void print_type(const GDScriptAST::TypeNode *p_type);
		void print_type_test(const GDScriptAST::TypeTestNode *p_type_test);
		void print_unary_op(const GDScriptAST::UnaryOpNode *p_unary_op);
		void print_variable(const GDScriptAST::VariableNode *p_variable);
		void print_while(const GDScriptAST::WhileNode *p_while);

	public:
		void print_tree(const GDScriptParser &p_parser);
	};
#endif // DEBUG_ENABLED
	static void cleanup();
};
