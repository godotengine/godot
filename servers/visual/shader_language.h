/*************************************************************************/
/*  shader_language.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef SHADER_LANGUAGE_H
#define SHADER_LANGUAGE_H

#include "list.h"
#include "map.h"
#include "string_db.h"
#include "typedefs.h"
#include "ustring.h"
#include "variant.h"

class ShaderLanguage {

public:
	enum TokenType {

		TK_EMPTY,
		TK_INDENTIFIER,
		TK_TRUE,
		TK_FALSE,
		TK_REAL_CONSTANT,
		TK_TYPE_VOID,
		TK_TYPE_BOOL,
		TK_TYPE_FLOAT,
		TK_TYPE_VEC2,
		TK_TYPE_VEC3,
		TK_TYPE_VEC4,
		TK_TYPE_MAT2,
		TK_TYPE_MAT3,
		TK_TYPE_MAT4,
		TK_TYPE_TEXTURE,
		TK_TYPE_CUBEMAP,
		TK_TYPE_COLOR,
		TK_OP_EQUAL,
		TK_OP_NOT_EQUAL,
		TK_OP_LESS,
		TK_OP_LESS_EQUAL,
		TK_OP_GREATER,
		TK_OP_GREATER_EQUAL,
		TK_OP_AND,
		TK_OP_OR,
		TK_OP_NOT,
		TK_OP_ADD,
		TK_OP_SUB,
		TK_OP_MUL,
		TK_OP_DIV,
		TK_OP_NEG,
		TK_OP_ASSIGN,
		TK_OP_ASSIGN_ADD,
		TK_OP_ASSIGN_SUB,
		TK_OP_ASSIGN_MUL,
		TK_OP_ASSIGN_DIV,
		TK_CF_IF,
		TK_CF_ELSE,
		TK_CF_RETURN,
		TK_BRACKET_OPEN,
		TK_BRACKET_CLOSE,
		TK_CURLY_BRACKET_OPEN,
		TK_CURLY_BRACKET_CLOSE,
		TK_PARENTHESIS_OPEN,
		TK_PARENTHESIS_CLOSE,
		TK_COMMA,
		TK_SEMICOLON,
		TK_PERIOD,
		TK_UNIFORM,
		TK_ERROR,
		TK_MAX
	};

	/* COMPILER */

	enum ShaderType {
		SHADER_MATERIAL_VERTEX,
		SHADER_MATERIAL_FRAGMENT,
		SHADER_MATERIAL_LIGHT,
		SHADER_CANVAS_ITEM_VERTEX,
		SHADER_CANVAS_ITEM_FRAGMENT,
		SHADER_CANVAS_ITEM_LIGHT,
		SHADER_POST_PROCESS,
	};

	enum DataType {
		TYPE_VOID,
		TYPE_BOOL,
		TYPE_FLOAT,
		TYPE_VEC2,
		TYPE_VEC3,
		TYPE_VEC4,
		TYPE_MAT2,
		TYPE_MAT3,
		TYPE_MAT4,
		TYPE_TEXTURE,
		TYPE_CUBEMAP,
	};

	enum Operator {
		OP_ASSIGN,
		OP_ADD,
		OP_SUB,
		OP_MUL,
		OP_DIV,
		OP_ASSIGN_ADD,
		OP_ASSIGN_SUB,
		OP_ASSIGN_MUL,
		OP_ASSIGN_DIV,
		OP_NEG,
		OP_NOT,
		OP_CMP_EQ,
		OP_CMP_NEQ,
		OP_CMP_LEQ,
		OP_CMP_GEQ,
		OP_CMP_LESS,
		OP_CMP_GREATER,
		OP_CMP_OR,
		OP_CMP_AND,
		OP_CALL,
		OP_CONSTRUCT,
		OP_MAX
	};

	enum FlowOperation {
		FLOW_OP_IF,
		FLOW_OP_RETURN,
		//FLOW_OP_FOR,
		//FLOW_OP_WHILE,
		//FLOW_OP_DO,
		//FLOW_OP_BREAK,
		//FLOW_OP_CONTINUE,

	};

	struct Node {

		enum Type {
			TYPE_PROGRAM,
			TYPE_FUNCTION,
			TYPE_BLOCK,
			TYPE_VARIABLE,
			TYPE_CONSTANT,
			TYPE_OPERATOR,
			TYPE_CONTROL_FLOW,
			TYPE_MEMBER
		};

		Node *parent;
		Type type;

		virtual DataType get_datatype() const { return TYPE_VOID; }

		virtual ~Node() {}
	};

	struct OperatorNode : public Node {

		DataType return_cache;
		Operator op;
		Vector<Node *> arguments;
		virtual DataType get_datatype() const { return return_cache; }

		OperatorNode() {
			type = TYPE_OPERATOR;
			return_cache = TYPE_VOID;
		}
	};

	struct VariableNode : public Node {
		bool uniform;
		DataType datatype_cache;
		StringName name;
		virtual DataType get_datatype() const { return datatype_cache; }

		VariableNode() {
			type = TYPE_VARIABLE;
			datatype_cache = TYPE_VOID;
			uniform = false;
		}
	};

	struct ConstantNode : public Node {

		DataType datatype;
		Variant value;
		virtual DataType get_datatype() const { return datatype; }

		ConstantNode() { type = TYPE_CONSTANT; }
	};

	struct BlockNode : public Node {

		Map<StringName, DataType> variables;
		List<Node *> statements;
		BlockNode() { type = TYPE_BLOCK; }
	};

	struct ControlFlowNode : public Node {

		FlowOperation flow_op;
		Vector<Node *> statements;
		ControlFlowNode() {
			type = TYPE_CONTROL_FLOW;
			flow_op = FLOW_OP_IF;
		}
	};

	struct MemberNode : public Node {

		DataType basetype;
		DataType datatype;
		StringName name;
		Node *owner;
		virtual DataType get_datatype() const { return datatype; }
		MemberNode() { type = TYPE_MEMBER; }
	};

	struct FunctionNode : public Node {

		struct Argument {

			StringName name;
			DataType type;
		};

		StringName name;
		DataType return_type;
		Vector<Argument> arguments;
		BlockNode *body;

		FunctionNode() { type = TYPE_FUNCTION; }
	};

	struct Uniform {

		int order;
		DataType type;
		Variant default_value;
	};

	struct ProgramNode : public Node {

		struct Function {
			StringName name;
			FunctionNode *function;
		};

		Map<StringName, DataType> builtin_variables;
		Map<StringName, Uniform> uniforms;

		Vector<Function> functions;
		BlockNode *body;

		ProgramNode() { type = TYPE_PROGRAM; }
	};

	struct Expression {

		bool is_op;
		union {
			TokenType op;
			Node *node;
		};
	};

	typedef Error (*CompileFunc)(void *, ProgramNode *);

	struct VarInfo {

		StringName name;
		DataType type;
	};

private:
	static const char *token_names[TK_MAX];

	struct Token {

		TokenType type;
		StringName text;
		uint16_t line, col;

		Token(TokenType p_type = TK_EMPTY, const String &p_text = String()) {
			type = p_type;
			text = p_text;
			line = 0;
			col = 0;
		}
	};

	static Token read_token(const CharType *p_text, int p_len, int &r_line, int &r_chars);
	static Error tokenize(const String &p_text, Vector<Token> *p_tokens, String *r_error, int *r_err_line, int *r_err_column);

	class Parser {

		Vector<Token> tokens;
		int pos;
		String error;

	public:
		void set_error(const String &p_error) { error = p_error; }
		void get_error(String *r_error, int *r_line, int *r_column) {

			*r_error = error;
			*r_line = get_next_token(0).line;
			*r_column = get_next_token(0).col;
		}

		Token get_next_token(int ofs = 0) const {
			int idx = pos + ofs;
			if (idx < 0 || idx >= tokens.size()) return Token(TK_ERROR);
			return tokens[idx];
		}
		TokenType get_next_token_type(int ofs = 0) const {
			int idx = pos + ofs;
			if (idx < 0 || idx >= tokens.size()) return TK_ERROR;
			return tokens[idx].type;
		}
		void advance(int p_amount = 1) { pos += p_amount; }
		bool is_at_end() const { return pos >= tokens.size(); }

		ProgramNode *program;
		template <class T>
		T *create_node(Node *p_parent) {
			T *n = memnew(T);
			nodegc.push_back(n);
			n->parent = p_parent;
			return n;
		}
		List<Node *> nodegc;

		Parser(const Vector<Token> &p_tokens) {
			tokens = p_tokens;
			pos = 0;
		}
	};

	struct IntrinsicFuncDef {

		enum { MAX_ARGS = 5 };
		const char *name;
		DataType rettype;
		const DataType args[MAX_ARGS];
	};

	static const IntrinsicFuncDef intrinsic_func_defs[];

	struct OperatorDef {

		enum { MAX_ARGS = 2 };
		Operator op;
		DataType rettype;
		const DataType args[MAX_ARGS];
	};

	static const OperatorDef operator_defs[];

	struct BuiltinsDef {

		const char *name;
		DataType type;
	};

	static const BuiltinsDef vertex_builtins_defs[];
	static const BuiltinsDef fragment_builtins_defs[];
	static const BuiltinsDef light_builtins_defs[];

	static const BuiltinsDef ci_vertex_builtins_defs[];
	static const BuiltinsDef ci_fragment_builtins_defs[];
	static const BuiltinsDef ci_light_builtins_defs[];

	static const BuiltinsDef postprocess_fragment_builtins_defs[];

	static DataType get_token_datatype(TokenType p_type);
	static String get_datatype_name(DataType p_type);
	static bool is_token_datatype(TokenType p_type);
	static bool is_token_nonvoid_datatype(TokenType p_type);

	static bool test_existing_identifier(Node *p_node, const StringName p_identifier, bool p_func = true, bool p_var = true, bool p_builtin = true);

	static bool parser_is_at_function(Parser &parser);
	static DataType compute_node_type(Node *p_node);

	static Node *validate_function_call(Parser &parser, OperatorNode *p_func);
	static Node *validate_operator(Parser &parser, OperatorNode *p_func);
	static bool is_token_operator(TokenType p_type);
	static Operator get_token_operator(TokenType p_type);

	static Error parse_expression(Parser &parser, Node *p_parent, Node **r_expr);

	static Error parse_variable_declaration(Parser &parser, BlockNode *p_block);
	static Error parse_function(Parser &parser, BlockNode *p_block);
	static Error parse_flow_if(Parser &parser, Node *p_parent, Node **r_statement);
	static Error parse_flow_return(Parser &parser, Node *p_parent, Node **r_statement);
	static Error parse_statement(Parser &parser, Node *p_parent, Node **r_statement);
	static Error parse_block(Parser &parser, BlockNode *p_block);

	static Error parse(const Vector<Token> &p_tokens, ShaderType p_type, CompileFunc p_compile_func, void *p_userdata, String *r_error, int *r_err_line, int *r_err_column);

	;

public:
	static void get_keyword_list(ShaderType p_type, List<String> *p_keywords);

	static Error compile(const String &p_code, ShaderType p_type, CompileFunc p_compile_func, void *p_userdata, String *r_error, int *r_err_line, int *r_err_column);
	static String lex_debug(const String &p_code);
};

#endif // SHADER_LANGUAGE_H
