/*************************************************************************/
/*  gdscript_parser.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "gdscript_functions.h"
#include "gdscript_tokenizer.h"
#include "map.h"
#include "object.h"
#include "script_language.h"

class GDScriptParser {
public:
	struct Node {

		enum Type {
			TYPE_CLASS,
			TYPE_FUNCTION,
			TYPE_BUILT_IN_FUNCTION,
			TYPE_BLOCK,
			TYPE_IDENTIFIER,
			TYPE_TYPE,
			TYPE_CONSTANT,
			TYPE_ARRAY,
			TYPE_DICTIONARY,
			TYPE_SELF,
			TYPE_OPERATOR,
			TYPE_CONTROL_FLOW,
			TYPE_LOCAL_VAR,
			TYPE_ASSERT,
			TYPE_BREAKPOINT,
			TYPE_NEWLINE,
		};

		Node *next;
		int line;
		int column;
		Type type;

		virtual ~Node() {}
	};

	struct FunctionNode;
	struct BlockNode;

	struct ClassNode : public Node {

		bool tool;
		StringName name;
		bool extends_used;
		StringName extends_file;
		Vector<StringName> extends_class;

		struct Member {
			PropertyInfo _export;
#ifdef TOOLS_ENABLED
			Variant default_value;
#endif
			StringName identifier;
			StringName setter;
			StringName getter;
			int line;
			Node *expression;
			ScriptInstance::RPCMode rpc_mode;
		};
		struct Constant {
			StringName identifier;
			Node *expression;
		};

		struct Signal {
			StringName name;
			Vector<StringName> arguments;
		};

		Vector<ClassNode *> subclasses;
		Vector<Member> variables;
		Vector<Constant> constant_expressions;
		Vector<FunctionNode *> functions;
		Vector<FunctionNode *> static_functions;
		Vector<Signal> _signals;
		BlockNode *initializer;
		BlockNode *ready;
		ClassNode *owner;
		//Vector<Node*> initializers;
		int end_line;

		ClassNode() {
			tool = false;
			type = TYPE_CLASS;
			extends_used = false;
			end_line = -1;
			owner = NULL;
		}
	};

	struct FunctionNode : public Node {

		bool _static;
		ScriptInstance::RPCMode rpc_mode;
		StringName name;
		Vector<StringName> arguments;
		Vector<Node *> default_values;
		BlockNode *body;

		FunctionNode() {
			type = TYPE_FUNCTION;
			_static = false;
			rpc_mode = ScriptInstance::RPC_MODE_DISABLED;
		}
	};

	struct BlockNode : public Node {

		ClassNode *parent_class;
		BlockNode *parent_block;
		Map<StringName, int> locals;
		List<Node *> statements;
		Vector<StringName> variables;
		Vector<int> variable_lines;

		Node *if_condition; //tiny hack to improve code completion on if () blocks

		//the following is useful for code completion
		List<BlockNode *> sub_blocks;
		int end_line;
		BlockNode() {
			if_condition = NULL;
			type = TYPE_BLOCK;
			end_line = -1;
			parent_block = NULL;
			parent_class = NULL;
		}
	};

	struct TypeNode : public Node {

		Variant::Type vtype;
		TypeNode() { type = TYPE_TYPE; }
	};
	struct BuiltInFunctionNode : public Node {
		GDScriptFunctions::Function function;
		BuiltInFunctionNode() { type = TYPE_BUILT_IN_FUNCTION; }
	};

	struct IdentifierNode : public Node {

		StringName name;
		IdentifierNode() { type = TYPE_IDENTIFIER; }
	};

	struct LocalVarNode : public Node {

		StringName name;
		Node *assign;
		LocalVarNode() {
			type = TYPE_LOCAL_VAR;
			assign = NULL;
		}
	};

	struct ConstantNode : public Node {
		Variant value;
		ConstantNode() { type = TYPE_CONSTANT; }
	};

	struct ArrayNode : public Node {

		Vector<Node *> elements;
		ArrayNode() { type = TYPE_ARRAY; }
	};

	struct DictionaryNode : public Node {

		struct Pair {

			Node *key;
			Node *value;
		};

		Vector<Pair> elements;
		DictionaryNode() { type = TYPE_DICTIONARY; }
	};

	struct SelfNode : public Node {
		SelfNode() { type = TYPE_SELF; }
	};

	struct OperatorNode : public Node {
		enum Operator {
			//call/constructor operator
			OP_CALL,
			OP_PARENT_CALL,
			OP_YIELD,
			OP_IS,
			//indexing operator
			OP_INDEX,
			OP_INDEX_NAMED,
			//unary operators
			OP_NEG,
			OP_POS,
			OP_NOT,
			OP_BIT_INVERT,
			OP_PREINC,
			OP_PREDEC,
			OP_INC,
			OP_DEC,
			//binary operators (in precedence order)
			OP_IN,
			OP_EQUAL,
			OP_NOT_EQUAL,
			OP_LESS,
			OP_LESS_EQUAL,
			OP_GREATER,
			OP_GREATER_EQUAL,
			OP_AND,
			OP_OR,
			OP_ADD,
			OP_SUB,
			OP_MUL,
			OP_DIV,
			OP_MOD,
			OP_SHIFT_LEFT,
			OP_SHIFT_RIGHT,
			OP_INIT_ASSIGN,
			OP_ASSIGN,
			OP_ASSIGN_ADD,
			OP_ASSIGN_SUB,
			OP_ASSIGN_MUL,
			OP_ASSIGN_DIV,
			OP_ASSIGN_MOD,
			OP_ASSIGN_SHIFT_LEFT,
			OP_ASSIGN_SHIFT_RIGHT,
			OP_ASSIGN_BIT_AND,
			OP_ASSIGN_BIT_OR,
			OP_ASSIGN_BIT_XOR,
			OP_BIT_AND,
			OP_BIT_OR,
			OP_BIT_XOR,
			//ternary operators
			OP_TERNARY_IF,
			OP_TERNARY_ELSE,
		};

		Operator op;

		Vector<Node *> arguments;
		OperatorNode() { type = TYPE_OPERATOR; }
	};

	struct PatternNode : public Node {

		enum PatternType {
			PT_CONSTANT,
			PT_BIND,
			PT_DICTIONARY,
			PT_ARRAY,
			PT_IGNORE_REST,
			PT_WILDCARD
		};

		PatternType pt_type;

		Node *constant;
		StringName bind;
		Map<ConstantNode *, PatternNode *> dictionary;
		Vector<PatternNode *> array;
	};

	struct PatternBranchNode : public Node {
		Vector<PatternNode *> patterns;
		BlockNode *body;
	};

	struct MatchNode : public Node {
		Node *val_to_match;
		Vector<PatternBranchNode *> branches;

		struct CompiledPatternBranch {
			Node *compiled_pattern;
			BlockNode *body;
		};

		Vector<CompiledPatternBranch> compiled_pattern_branches;
	};

	struct ControlFlowNode : public Node {
		enum CFType {
			CF_IF,
			CF_FOR,
			CF_WHILE,
			CF_SWITCH,
			CF_BREAK,
			CF_CONTINUE,
			CF_RETURN,
			CF_MATCH
		};

		CFType cf_type;
		Vector<Node *> arguments;
		BlockNode *body;
		BlockNode *body_else;

		MatchNode *match;

		ControlFlowNode *_else; //used for if
		ControlFlowNode() {
			type = TYPE_CONTROL_FLOW;
			cf_type = CF_IF;
			body = NULL;
			body_else = NULL;
		}
	};

	struct AssertNode : public Node {
		Node *condition;
		AssertNode() { type = TYPE_ASSERT; }
	};

	struct BreakpointNode : public Node {
		BreakpointNode() { type = TYPE_BREAKPOINT; }
	};

	struct NewLineNode : public Node {
		NewLineNode() { type = TYPE_NEWLINE; }
	};

	struct Expression {

		bool is_op;
		union {
			OperatorNode::Operator op;
			Node *node;
		};
	};

	/*
	struct OperatorNode : public Node {

		DataType return_cache;
		Operator op;
		Vector<Node*> arguments;
		virtual DataType get_datatype() const { return return_cache; }

		OperatorNode() { type=TYPE_OPERATOR; return_cache=TYPE_VOID; }
	};

	struct VariableNode : public Node {

		DataType datatype_cache;
		StringName name;
		virtual DataType get_datatype() const { return datatype_cache; }

		VariableNode() { type=TYPE_VARIABLE; datatype_cache=TYPE_VOID; }
	};

	struct ConstantNode : public Node {

		DataType datatype;
		Variant value;
		virtual DataType get_datatype() const { return datatype; }

		ConstantNode() { type=TYPE_CONSTANT; }
	};

	struct BlockNode : public Node {

		Map<StringName,DataType> variables;
		List<Node*> statements;
		BlockNode() { type=TYPE_BLOCK; }
	};

	struct ControlFlowNode : public Node {

		FlowOperation flow_op;
		Vector<Node*> statements;
		ControlFlowNode() { type=TYPE_CONTROL_FLOW; flow_op=FLOW_OP_IF;}
	};

	struct MemberNode : public Node {

		DataType datatype;
		StringName name;
		Node* owner;
		virtual DataType get_datatype() const { return datatype; }
		MemberNode() { type=TYPE_MEMBER; }
	};


	struct ProgramNode : public Node {

		struct Function {
			StringName name;
			FunctionNode*function;
		};

		Map<StringName,DataType> builtin_variables;
		Map<StringName,DataType> preexisting_variables;

		Vector<Function> functions;
		BlockNode *body;

		ProgramNode() { type=TYPE_PROGRAM; }
	};
*/

	enum CompletionType {
		COMPLETION_NONE,
		COMPLETION_BUILT_IN_TYPE_CONSTANT,
		COMPLETION_GET_NODE,
		COMPLETION_FUNCTION,
		COMPLETION_IDENTIFIER,
		COMPLETION_PARENT_FUNCTION,
		COMPLETION_METHOD,
		COMPLETION_CALL_ARGUMENTS,
		COMPLETION_RESOURCE_PATH,
		COMPLETION_INDEX,
		COMPLETION_VIRTUAL_FUNC,
		COMPLETION_YIELD,
		COMPLETION_ASSIGN,
	};

private:
	GDScriptTokenizer *tokenizer;

	Node *head;
	Node *list;
	template <class T>
	T *alloc_node();

	bool validating;
	bool for_completion;
	int parenthesis;
	bool error_set;
	String error;
	int error_line;
	int error_column;

	int pending_newline;

	List<int> tab_level;

	String base_path;
	String self_path;

	ClassNode *current_class;
	FunctionNode *current_function;
	BlockNode *current_block;

	bool _get_completable_identifier(CompletionType p_type, StringName &identifier);
	void _make_completable_call(int p_arg);

	CompletionType completion_type;
	StringName completion_cursor;
	bool completion_static;
	Variant::Type completion_built_in_constant;
	Node *completion_node;
	ClassNode *completion_class;
	FunctionNode *completion_function;
	BlockNode *completion_block;
	int completion_line;
	int completion_argument;
	bool completion_found;
	bool completion_ident_is_call;

	PropertyInfo current_export;

	ScriptInstance::RPCMode rpc_mode;

	void _set_error(const String &p_error, int p_line = -1, int p_column = -1);
	bool _recover_from_completion();

	bool _parse_arguments(Node *p_parent, Vector<Node *> &p_args, bool p_static, bool p_can_codecomplete = false);
	bool _enter_indent_block(BlockNode *p_block = NULL);
	bool _parse_newline();
	Node *_parse_expression(Node *p_parent, bool p_static, bool p_allow_assign = false, bool p_parsing_constant = false);
	Node *_reduce_expression(Node *p_node, bool p_to_const = false);
	Node *_parse_and_reduce_expression(Node *p_parent, bool p_static, bool p_reduce_const = false, bool p_allow_assign = false);

	PatternNode *_parse_pattern(bool p_static);
	void _parse_pattern_block(BlockNode *p_block, Vector<PatternBranchNode *> &p_branches, bool p_static);
	void _transform_match_statment(BlockNode *p_block, MatchNode *p_match_statement);
	void _generate_pattern(PatternNode *p_pattern, Node *p_node_to_match, Node *&p_resulting_node, Map<StringName, Node *> &p_bindings);

	void _parse_block(BlockNode *p_block, bool p_static);
	void _parse_extends(ClassNode *p_class);
	void _parse_class(ClassNode *p_class);
	bool _end_statement();

	Error _parse(const String &p_base_path);

public:
	String get_error() const;
	int get_error_line() const;
	int get_error_column() const;
	Error parse(const String &p_code, const String &p_base_path = "", bool p_just_validate = false, const String &p_self_path = "", bool p_for_completion = false);
	Error parse_bytecode(const Vector<uint8_t> &p_bytecode, const String &p_base_path = "", const String &p_self_path = "");

	bool is_tool_script() const;
	const Node *get_parse_tree() const;

	//completion info

	CompletionType get_completion_type();
	StringName get_completion_cursor();
	int get_completion_line();
	Variant::Type get_completion_built_in_constant();
	Node *get_completion_node();
	ClassNode *get_completion_class();
	BlockNode *get_completion_block();
	FunctionNode *get_completion_function();
	int get_completion_argument_index();
	int get_completion_identifier_is_function();

	void clear();
	GDScriptParser();
	~GDScriptParser();
};

#endif // GDSCRIPT_PARSER_H
