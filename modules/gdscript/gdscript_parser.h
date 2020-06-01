/*************************************************************************/
/*  gdscript_parser.h                                                    */
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

#ifndef GDSCRIPT_PARSER_H
#define GDSCRIPT_PARSER_H

#include "core/hash_map.h"
#include "core/io/multiplayer_api.h"
#include "core/list.h"
#include "core/map.h"
#include "core/reference.h"
#include "core/resource.h"
#include "core/script_language.h"
#include "core/string_name.h"
#include "core/ustring.h"
#include "core/variant.h"
#include "core/vector.h"
#include "gdscript_functions.h"
#include "gdscript_tokenizer.h"

#ifdef DEBUG_ENABLED
#include "core/string_builder.h"
#endif // DEBUG_ENABLED

class GDScriptParser {
	struct AnnotationInfo;

public:
	// Forward-declare all parser nodes, to avoid ordering issues.
	struct AnnotationNode;
	struct ArrayNode;
	struct AssertNode;
	struct AssignmentNode;
	struct AwaitNode;
	struct BinaryOpNode;
	struct BreakNode;
	struct BreakpointNode;
	struct CallNode;
	struct CastNode;
	struct ClassNode;
	struct ConstantNode;
	struct ContinueNode;
	struct DictionaryNode;
	struct EnumNode;
	struct ExpressionNode;
	struct ForNode;
	struct FunctionNode;
	struct GetNodeNode;
	struct IdentifierNode;
	struct IfNode;
	struct LiteralNode;
	struct MatchNode;
	struct MatchBranchNode;
	struct ParameterNode;
	struct PassNode;
	struct PatternNode;
	struct PreloadNode;
	struct ReturnNode;
	struct SelfNode;
	struct SignalNode;
	struct SubscriptNode;
	struct SuiteNode;
	struct TernaryOpNode;
	struct TypeNode;
	struct UnaryOpNode;
	struct VariableNode;
	struct WhileNode;

	struct DataType {
		enum Kind {
			BUILTIN,
			NATIVE,
			SCRIPT,
			CLASS, // GDScript.
			UNRESOLVED,
		};
		Kind kind = UNRESOLVED;

		enum TypeSource {
			UNDETECTED, // Can be any type.
			INFERRED, // Has inferred type, but still dynamic.
			ANNOTATED_EXPLICIT, // Has a specific type annotated.
			ANNOTATED_INFERRED, // Has a static type but comes from the assigned value.
		};
		TypeSource type_source = UNDETECTED;

		bool is_constant = false;
		bool is_meta_type = false;
		bool infer_type = false;

		Variant::Type builtin_type = Variant::NIL;
		StringName native_type;
		Ref<Script> script_type;
		ClassNode *gdscript_type = nullptr;

		_FORCE_INLINE_ bool is_set() const { return type_source != UNDETECTED; }
		String to_string() const;

		bool operator==(const DataType &p_other) const {
			if (type_source == UNDETECTED || p_other.type_source == UNDETECTED) {
				return true; // Can be consireded equal for parsing purposes.
			}

			if (type_source == INFERRED || p_other.type_source == INFERRED) {
				return true; // Can be consireded equal for parsing purposes.
			}

			if (kind != p_other.kind) {
				return false;
			}

			switch (kind) {
				case BUILTIN:
					return builtin_type == p_other.builtin_type;
				case NATIVE:
					return native_type == p_other.native_type;
				case SCRIPT:
					return script_type == p_other.script_type;
				case CLASS:
					return gdscript_type == p_other.gdscript_type;
				case UNRESOLVED:
					break;
			}

			return false;
		}
	};

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
		int line = 0, column = 0;
	};

	struct Node {
		enum Type {
			NONE,
			ANNOTATION,
			ARRAY,
			ASSERT,
			ASSIGNMENT,
			AWAIT,
			BINARY_OPERATOR,
			BREAK,
			BREAKPOINT,
			CALL,
			CAST,
			CLASS,
			CONSTANT,
			CONTINUE,
			DICTIONARY,
			ENUM,
			FOR,
			FUNCTION,
			GET_NODE,
			IDENTIFIER,
			IF,
			LITERAL,
			MATCH,
			MATCH_BRANCH,
			PARAMETER,
			PASS,
			PATTERN,
			PRELOAD,
			RETURN,
			SELF,
			SIGNAL,
			SUBSCRIPT,
			SUITE,
			TERNARY_OPERATOR,
			TYPE,
			UNARY_OPERATOR,
			VARIABLE,
			WHILE,
		};

		Type type = NONE;
		int start_line = 0, end_line = 0;
		int leftmost_column = 0, rightmost_column = 0;
		Node *next = nullptr;
		List<AnnotationNode *> annotations;

		virtual DataType get_datatype() const { return DataType(); }
		virtual void set_datatype(const DataType &p_datatype) {}

		virtual bool is_expression() const { return false; }

		virtual ~Node() {}
	};

	struct ExpressionNode : public Node {
		// Base type for all expression kinds.
		virtual bool is_expression() const { return true; }
		virtual ~ExpressionNode() {}

	protected:
		ExpressionNode() {}
	};

	struct AnnotationNode : public Node {
		StringName name;
		Vector<ExpressionNode *> arguments;
		Vector<Variant> resolved_arguments;

		AnnotationInfo *info = nullptr;

		bool apply(GDScriptParser *p_this, Node *p_target) const;
		bool applies_to(uint32_t p_target_kinds) const;

		AnnotationNode() {
			type = ANNOTATION;
		}
	};

	struct ArrayNode : public ExpressionNode {
		Vector<ExpressionNode *> elements;

		ArrayNode() {
			type = ARRAY;
		}
	};

	struct AssertNode : public Node {
		ExpressionNode *condition = nullptr;
		LiteralNode *message = nullptr;

		AssertNode() {
			type = ASSERT;
		}
	};

	struct AssignmentNode : public ExpressionNode {
		// Assignment is not really an expression but it's easier to parse as if it were.
		enum Operation {
			OP_NONE,
			OP_ADDITION,
			OP_SUBTRACTION,
			OP_MULTIPLICATION,
			OP_DIVISION,
			OP_MODULO,
			OP_BIT_SHIFT_LEFT,
			OP_BIT_SHIFT_RIGHT,
			OP_BIT_AND,
			OP_BIT_OR,
			OP_BIT_XOR,
		};

		Operation operation = OP_NONE;
		ExpressionNode *assignee = nullptr;
		ExpressionNode *assigned_value = nullptr;

		AssignmentNode() {
			type = ASSIGNMENT;
		}
	};

	struct AwaitNode : public ExpressionNode {
		ExpressionNode *to_await = nullptr;

		AwaitNode() {
			type = AWAIT;
		}
	};

	struct BinaryOpNode : public ExpressionNode {
		enum OpType {
			OP_ADDITION,
			OP_SUBTRACTION,
			OP_MULTIPLICATION,
			OP_DIVISION,
			OP_MODULO,
			OP_BIT_LEFT_SHIFT,
			OP_BIT_RIGHT_SHIFT,
			OP_BIT_AND,
			OP_BIT_OR,
			OP_BIT_XOR,
			OP_LOGIC_AND,
			OP_LOGIC_OR,
			OP_TYPE_TEST,
			OP_CONTENT_TEST,
			OP_COMP_EQUAL,
			OP_COMP_NOT_EQUAL,
			OP_COMP_LESS,
			OP_COMP_LESS_EQUAL,
			OP_COMP_GREATER,
			OP_COMP_GREATER_EQUAL,
		};

		OpType operation;
		ExpressionNode *left_operand = nullptr;
		ExpressionNode *right_operand = nullptr;

		BinaryOpNode() {
			type = BINARY_OPERATOR;
		}
	};

	struct BreakNode : public Node {
		BreakNode() {
			type = BREAK;
		}
	};

	struct BreakpointNode : public Node {
		BreakpointNode() {
			type = BREAKPOINT;
		}
	};

	struct CallNode : public ExpressionNode {
		ExpressionNode *callee = nullptr;
		Vector<ExpressionNode *> arguments;
		bool is_super = false;

		CallNode() {
			type = CALL;
		}
	};

	struct CastNode : public ExpressionNode {
		ExpressionNode *operand = nullptr;
		TypeNode *cast_type = nullptr;

		CastNode() {
			type = CAST;
		}
	};

	struct EnumNode : public Node {
		struct Value {
			IdentifierNode *identifier = nullptr;
			LiteralNode *custom_value = nullptr;
			int value = 0;
		};
		IdentifierNode *identifier = nullptr;
		Vector<Value> values;

		EnumNode() {
			type = ENUM;
		}
	};

	struct ClassNode : public Node {
		struct Member {
			enum Type {
				UNDEFINED,
				CLASS,
				CONSTANT,
				FUNCTION,
				SIGNAL,
				VARIABLE,
				ENUM,
				ENUM_VALUE, // For unnamed enums.
			};

			Type type = UNDEFINED;

			union {
				ClassNode *m_class = nullptr;
				ConstantNode *constant;
				FunctionNode *function;
				SignalNode *signal;
				VariableNode *variable;
				EnumNode *m_enum;
			};
			EnumNode::Value enum_value;

			String get_type_name() const {
				switch (type) {
					case UNDEFINED:
						return "???";
					case CLASS:
						return "class";
					case CONSTANT:
						return "constant";
					case FUNCTION:
						return "function";
					case SIGNAL:
						return "signal";
					case VARIABLE:
						return "variable";
					case ENUM:
						return "enum";
					case ENUM_VALUE:
						return "enum value";
				}
				return "";
			}

			Member() {}

			Member(ClassNode *p_class) {
				type = CLASS;
				m_class = p_class;
			}
			Member(ConstantNode *p_constant) {
				type = CONSTANT;
				constant = p_constant;
			}
			Member(VariableNode *p_variable) {
				type = VARIABLE;
				variable = p_variable;
			}
			Member(SignalNode *p_signal) {
				type = SIGNAL;
				signal = p_signal;
			}
			Member(FunctionNode *p_function) {
				type = FUNCTION;
				function = p_function;
			}
			Member(EnumNode *p_enum) {
				type = ENUM;
				m_enum = p_enum;
			}
			Member(const EnumNode::Value &p_enum_value) {
				type = ENUM_VALUE;
				enum_value = p_enum_value;
			}
		};

		IdentifierNode *identifier = nullptr;
		String icon_path;
		Vector<Member> members;
		HashMap<StringName, int> members_indices;
		ClassNode *outer = nullptr;
		bool extends_used = false;
		bool onready_used = false;
		String extends_path;
		Vector<StringName> extends; // List for indexing: extends A.B.C
		DataType base_type;

		Member get_member(const StringName &p_name) const {
			return members[members_indices[p_name]];
		}
		template <class T>
		void add_member(T *p_member_node) {
			members_indices[p_member_node->identifier->name] = members.size();
			members.push_back(Member(p_member_node));
		}
		void add_member(const EnumNode::Value &p_enum_value) {
			members_indices[p_enum_value.identifier->name] = members.size();
			members.push_back(Member(p_enum_value));
		}
		virtual DataType get_datatype() const {
			return base_type;
		}
		virtual void set_datatype(const DataType &p_datatype) {
			base_type = p_datatype;
		}

		ClassNode() {
			type = CLASS;
		}
	};

	struct ConstantNode : public Node {
		IdentifierNode *identifier = nullptr;
		ExpressionNode *initializer = nullptr;
		TypeNode *datatype_specifier = nullptr;
		bool infer_datatype = false;

		ConstantNode() {
			type = CONSTANT;
		}
	};

	struct ContinueNode : public Node {
		ContinueNode() {
			type = CONTINUE;
		}
	};

	struct DictionaryNode : public ExpressionNode {
		struct Pair {
			ExpressionNode *key = nullptr;
			ExpressionNode *value = nullptr;
		};
		Vector<Pair> elements;

		enum Style {
			LUA_TABLE,
			PYTHON_DICT,
		};
		Style style = PYTHON_DICT;

		DictionaryNode() {
			type = DICTIONARY;
		}
	};

	struct ForNode : public Node {
		IdentifierNode *variable = nullptr;
		ExpressionNode *list = nullptr;
		SuiteNode *loop = nullptr;

		ForNode() {
			type = FOR;
		}
	};

	struct FunctionNode : public Node {
		IdentifierNode *identifier = nullptr;
		Vector<ParameterNode *> parameters;
		HashMap<StringName, int> parameters_indices;
		TypeNode *return_type = nullptr;
		SuiteNode *body = nullptr;
		bool is_static = false;
		MultiplayerAPI::RPCMode rpc_mode = MultiplayerAPI::RPC_MODE_DISABLED;

		FunctionNode() {
			type = FUNCTION;
		}
	};

	struct GetNodeNode : public ExpressionNode {
		LiteralNode *string = nullptr;
		Vector<IdentifierNode *> chain;

		GetNodeNode() {
			type = GET_NODE;
		}
	};

	struct IdentifierNode : public ExpressionNode {
		StringName name;

		IdentifierNode() {
			type = IDENTIFIER;
		}
	};

	struct IfNode : public Node {
		ExpressionNode *condition = nullptr;
		SuiteNode *true_block = nullptr;
		SuiteNode *false_block = nullptr;

		IfNode() {
			type = IF;
		}
	};

	struct LiteralNode : public ExpressionNode {
		Variant value;

		LiteralNode() {
			type = LITERAL;
		}
	};

	struct MatchNode : public Node {
		ExpressionNode *test = nullptr;
		Vector<MatchBranchNode *> branches;

		MatchNode() {
			type = MATCH;
		}
	};

	struct MatchBranchNode : public Node {
		Vector<PatternNode *> patterns;
		SuiteNode *block;

		MatchBranchNode() {
			type = MATCH_BRANCH;
		}
	};

	struct ParameterNode : public Node {
		IdentifierNode *identifier = nullptr;
		ExpressionNode *default_value = nullptr;
		TypeNode *datatype_specifier = nullptr;
		bool infer_datatype = false;

		ParameterNode() {
			type = PARAMETER;
		}
	};

	struct PassNode : public Node {
		PassNode() {
			type = PASS;
		}
	};

	struct PatternNode : public Node {
		enum Type {
			PT_LITERAL,
			PT_EXPRESSION,
			PT_BIND,
			PT_ARRAY,
			PT_DICTIONARY,
			PT_REST,
			PT_WILDCARD,
		};
		Type pattern_type = PT_LITERAL;

		union {
			LiteralNode *literal = nullptr;
			IdentifierNode *bind;
			ExpressionNode *expression;
		};
		Vector<PatternNode *> array;
		bool rest_used = false; // For array/dict patterns.

		struct Pair {
			ExpressionNode *key = nullptr;
			PatternNode *value_pattern = nullptr;
		};
		Vector<Pair> dictionary;

		PatternNode() {
			type = PATTERN;
		}
	};
	struct PreloadNode : public ExpressionNode {
		ExpressionNode *path = nullptr;
		String resolved_path;
		Ref<Resource> resource;

		PreloadNode() {
			type = PRELOAD;
		}
	};

	struct ReturnNode : public Node {
		ExpressionNode *return_value = nullptr;

		ReturnNode() {
			type = RETURN;
		}
	};

	struct SelfNode : public ExpressionNode {
		ClassNode *current_class = nullptr;

		SelfNode() {
			type = SELF;
		}
	};

	struct SignalNode : public Node {
		IdentifierNode *identifier = nullptr;
		Vector<ParameterNode *> parameters;
		HashMap<StringName, int> parameters_indices;

		SignalNode() {
			type = SIGNAL;
		}
	};

	struct SubscriptNode : public ExpressionNode {
		ExpressionNode *base = nullptr;
		union {
			ExpressionNode *index = nullptr;
			IdentifierNode *attribute;
		};

		bool is_attribute = false;

		SubscriptNode() {
			type = SUBSCRIPT;
		}
	};

	struct SuiteNode : public Node {
		SuiteNode *parent_block = nullptr;
		Vector<Node *> statements;
		struct Local {
			enum Type {
				UNDEFINED,
				CONSTANT,
				VARIABLE,
			};
			Type type = UNDEFINED;
			union {
				ConstantNode *constant = nullptr;
				VariableNode *variable;
			};

			Local() {}
			Local(ConstantNode *p_constant) {
				type = CONSTANT;
				constant = p_constant;
			}
			Local(VariableNode *p_variable) {
				type = VARIABLE;
				variable = p_variable;
			}
		};
		Local empty;
		Vector<Local> locals;
		HashMap<StringName, int> locals_indices;

		bool has_local(const StringName &p_name) const;
		const Local &get_local(const StringName &p_name) const;
		template <class T>
		void add_local(T *p_local) {
			locals_indices[p_local->identifier->name] = locals.size();
			locals.push_back(Local(p_local));
		}

		SuiteNode() {
			type = SUITE;
		}
	};

	struct TernaryOpNode : public ExpressionNode {
		// Only one ternary operation exists, so no abstraction here.
		ExpressionNode *condition = nullptr;
		ExpressionNode *true_expr = nullptr;
		ExpressionNode *false_expr = nullptr;

		TernaryOpNode() {
			type = TERNARY_OPERATOR;
		}
	};

	struct TypeNode : public Node {
		IdentifierNode *type_base = nullptr;
		SubscriptNode *type_specifier = nullptr;

		TypeNode() {
			type = TYPE;
		}
	};

	struct UnaryOpNode : public ExpressionNode {
		enum OpType {
			OP_POSITIVE,
			OP_NEGATIVE,
			OP_COMPLEMENT,
			OP_LOGIC_NOT,
		};

		OpType operation;
		ExpressionNode *operand = nullptr;

		UnaryOpNode() {
			type = UNARY_OPERATOR;
		}
	};

	struct VariableNode : public Node {
		enum PropertyStyle {
			PROP_NONE,
			PROP_INLINE,
			PROP_SETGET,
		};

		IdentifierNode *identifier = nullptr;
		ExpressionNode *initializer = nullptr;
		TypeNode *datatype_specifier = nullptr;
		bool infer_datatype = false;

		PropertyStyle property = PROP_NONE;
		union {
			SuiteNode *setter = nullptr;
			IdentifierNode *setter_pointer;
		};
		IdentifierNode *setter_parameter = nullptr;
		union {
			SuiteNode *getter = nullptr;
			IdentifierNode *getter_pointer;
		};

		bool exported = false;
		bool onready = false;
		PropertyInfo export_info;
		MultiplayerAPI::RPCMode rpc_mode = MultiplayerAPI::RPC_MODE_DISABLED;

		VariableNode() {
			type = VARIABLE;
		}
	};

	struct WhileNode : public Node {
		ExpressionNode *condition = nullptr;
		SuiteNode *loop = nullptr;

		WhileNode() {
			type = WHILE;
		}
	};

private:
	friend class GDScriptAnalyzer;

	bool _is_tool = false;
	String script_path;
	bool for_completion = false;
	bool panic_mode = false;
	bool can_break = false;
	bool can_continue = false;
	List<bool> multiline_stack;

	ClassNode *head = nullptr;
	Node *list = nullptr;
	List<ParserError> errors;

	GDScriptTokenizer tokenizer;
	GDScriptTokenizer::Token previous;
	GDScriptTokenizer::Token current;

	ClassNode *current_class = nullptr;
	FunctionNode *current_function = nullptr;
	SuiteNode *current_suite = nullptr;

	typedef bool (GDScriptParser::*AnnotationAction)(const AnnotationNode *p_annotation, Node *p_target);
	struct AnnotationInfo {
		enum TargetKind {
			NONE = 0,
			SCRIPT = 1 << 0,
			CLASS = 1 << 1,
			VARIABLE = 1 << 2,
			CONSTANT = 1 << 3,
			SIGNAL = 1 << 4,
			FUNCTION = 1 << 5,
			STATEMENT = 1 << 6,
			CLASS_LEVEL = CLASS | VARIABLE | FUNCTION,
		};
		uint32_t target_kind = 0; // Flags.
		AnnotationAction apply = nullptr;
		MethodInfo info;
	};
	HashMap<StringName, AnnotationInfo> valid_annotations;
	List<AnnotationNode *> annotation_stack;

	typedef ExpressionNode *(GDScriptParser::*ParseFunction)(ExpressionNode *p_previous_operand, bool p_can_assign);
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
		PREC_SUBTRACTION,
		PREC_ADDITION,
		PREC_FACTOR,
		PREC_SIGN,
		PREC_BIT_NOT,
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

	template <class T>
	T *alloc_node();
	void clear();
	void push_error(const String &p_message, const Node *p_origin = nullptr);

	GDScriptTokenizer::Token advance();
	bool match(GDScriptTokenizer::Token::Type p_token_type);
	bool check(GDScriptTokenizer::Token::Type p_token_type);
	bool consume(GDScriptTokenizer::Token::Type p_token_type, const String &p_error_message);
	bool is_at_end();
	bool is_statement_end();
	void end_statement(const String &p_context);
	void synchronize();
	void push_multiline(bool p_state);
	void pop_multiline();

	// Main blocks.
	void parse_program();
	ClassNode *parse_class();
	void parse_class_name();
	void parse_extends();
	void parse_class_body();
	template <class T>
	void parse_class_member(T *(GDScriptParser::*p_parse_function)(), AnnotationInfo::TargetKind p_target, const String &p_member_kind);
	SignalNode *parse_signal();
	EnumNode *parse_enum();
	ParameterNode *parse_parameter();
	FunctionNode *parse_function();
	SuiteNode *parse_suite(const String &p_context);
	// Annotations
	AnnotationNode *parse_annotation(uint32_t p_valid_targets);
	bool register_annotation(const MethodInfo &p_info, uint32_t p_target_kinds, AnnotationAction p_apply, int p_optional_arguments = 0, bool p_is_vararg = false);
	bool validate_annotation_arguments(AnnotationNode *p_annotation);
	void clear_unused_annotations();
	bool tool_annotation(const AnnotationNode *p_annotation, Node *p_target);
	bool icon_annotation(const AnnotationNode *p_annotation, Node *p_target);
	bool onready_annotation(const AnnotationNode *p_annotation, Node *p_target);
	template <PropertyHint t_hint, Variant::Type t_type>
	bool export_annotations(const AnnotationNode *p_annotation, Node *p_target);
	bool warning_annotations(const AnnotationNode *p_annotation, Node *p_target);
	template <MultiplayerAPI::RPCMode t_mode>
	bool network_annotations(const AnnotationNode *p_annotation, Node *p_target);
	// Statements.
	Node *parse_statement();
	VariableNode *parse_variable();
	VariableNode *parse_variable(bool p_allow_property);
	VariableNode *parse_property(VariableNode *p_variable, bool p_need_indent);
	void parse_property_getter(VariableNode *p_variable);
	void parse_property_setter(VariableNode *p_variable);
	ConstantNode *parse_constant();
	AssertNode *parse_assert();
	BreakNode *parse_break();
	ContinueNode *parse_continue();
	ForNode *parse_for();
	IfNode *parse_if(const String &p_token = "if");
	MatchNode *parse_match();
	MatchBranchNode *parse_match_branch();
	PatternNode *parse_match_pattern();
	WhileNode *parse_while();
	// Expressions.
	ExpressionNode *parse_expression(bool p_can_assign, bool p_stop_on_assign = false);
	ExpressionNode *parse_precedence(Precedence p_precedence, bool p_can_assign, bool p_stop_on_assign = false);
	ExpressionNode *parse_literal(ExpressionNode *p_previous_operand, bool p_can_assign);
	LiteralNode *parse_literal();
	ExpressionNode *parse_self(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_identifier(ExpressionNode *p_previous_operand, bool p_can_assign);
	IdentifierNode *parse_identifier();
	ExpressionNode *parse_builtin_constant(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_unary_operator(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_binary_operator(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_ternary_operator(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_assignment(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_array(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_dictionary(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_call(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_get_node(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_preload(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_grouping(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_cast(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_await(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_attribute(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_subscript(ExpressionNode *p_previous_operand, bool p_can_assign);
	ExpressionNode *parse_invalid_token(ExpressionNode *p_previous_operand, bool p_can_assign);
	TypeNode *parse_type(bool p_allow_void = false);

public:
	Error parse(const String &p_source_code, const String &p_script_path, bool p_for_completion);
	ClassNode *get_tree() const { return head; }
	bool is_tool() const { return _is_tool; }
	static Variant::Type get_builtin_type(const StringName &p_type);
	static GDScriptFunctions::Function get_builtin_function(const StringName &p_name);

	const List<ParserError> &get_errors() const { return errors; }
	const List<String> get_dependencies() const {
		// TODO: Keep track of deps.
		return List<String>();
	}

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

		void print_annotation(AnnotationNode *p_annotation);
		void print_array(ArrayNode *p_array);
		void print_assert(AssertNode *p_assert);
		void print_assignment(AssignmentNode *p_assignment);
		void print_await(AwaitNode *p_await);
		void print_binary_op(BinaryOpNode *p_binary_op);
		void print_call(CallNode *p_call);
		void print_cast(CastNode *p_cast);
		void print_class(ClassNode *p_class);
		void print_constant(ConstantNode *p_constant);
		void print_dictionary(DictionaryNode *p_dictionary);
		void print_expression(ExpressionNode *p_expression);
		void print_enum(EnumNode *p_enum);
		void print_for(ForNode *p_for);
		void print_function(FunctionNode *p_function);
		void print_get_node(GetNodeNode *p_get_node);
		void print_if(IfNode *p_if, bool p_is_elif = false);
		void print_identifier(IdentifierNode *p_identifier);
		void print_literal(LiteralNode *p_literal);
		void print_match(MatchNode *p_match);
		void print_match_branch(MatchBranchNode *p_match_branch);
		void print_match_pattern(PatternNode *p_match_pattern);
		void print_parameter(ParameterNode *p_parameter);
		void print_preload(PreloadNode *p_preload);
		void print_return(ReturnNode *p_return);
		void print_self(SelfNode *p_self);
		void print_signal(SignalNode *p_signal);
		void print_statement(Node *p_statement);
		void print_subscript(SubscriptNode *p_subscript);
		void print_suite(SuiteNode *p_suite);
		void print_type(TypeNode *p_type);
		void print_ternary_op(TernaryOpNode *p_ternary_op);
		void print_unary_op(UnaryOpNode *p_unary_op);
		void print_variable(VariableNode *p_variable);
		void print_while(WhileNode *p_while);

	public:
		void print_tree(const GDScriptParser &p_parser);
	};
#endif // DEBUG_ENABLED
};

#endif // GDSCRIPT_PARSER_H
