/**************************************************************************/
/*  shader_language.h                                                     */
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

#ifndef SHADER_LANGUAGE_H
#define SHADER_LANGUAGE_H

#include "core/list.h"
#include "core/map.h"
#include "core/ordered_hash_map.h"
#include "core/script_language.h"
#include "core/string_name.h"
#include "core/typedefs.h"
#include "core/ustring.h"
#include "core/variant.h"

class ShaderLanguage {
public:
	struct TkPos {
		int char_idx;
		int tk_line;
	};

	enum TokenType {
		TK_EMPTY,
		TK_IDENTIFIER,
		TK_TRUE,
		TK_FALSE,
		TK_REAL_CONSTANT,
		TK_INT_CONSTANT,
		TK_UINT_CONSTANT,
		TK_TYPE_VOID,
		TK_TYPE_BOOL,
		TK_TYPE_BVEC2,
		TK_TYPE_BVEC3,
		TK_TYPE_BVEC4,
		TK_TYPE_INT,
		TK_TYPE_IVEC2,
		TK_TYPE_IVEC3,
		TK_TYPE_IVEC4,
		TK_TYPE_UINT,
		TK_TYPE_UVEC2,
		TK_TYPE_UVEC3,
		TK_TYPE_UVEC4,
		TK_TYPE_FLOAT,
		TK_TYPE_VEC2,
		TK_TYPE_VEC3,
		TK_TYPE_VEC4,
		TK_TYPE_MAT2,
		TK_TYPE_MAT3,
		TK_TYPE_MAT4,
		TK_TYPE_SAMPLER2D,
		TK_TYPE_ISAMPLER2D,
		TK_TYPE_USAMPLER2D,
		TK_TYPE_SAMPLER2DARRAY,
		TK_TYPE_ISAMPLER2DARRAY,
		TK_TYPE_USAMPLER2DARRAY,
		TK_TYPE_SAMPLER3D,
		TK_TYPE_ISAMPLER3D,
		TK_TYPE_USAMPLER3D,
		TK_TYPE_SAMPLERCUBE,
		TK_TYPE_SAMPLEREXT,
		TK_INTERPOLATION_FLAT,
		TK_INTERPOLATION_SMOOTH,
		TK_CONST,
		TK_STRUCT,
		TK_PRECISION_LOW,
		TK_PRECISION_MID,
		TK_PRECISION_HIGH,
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
		TK_OP_MOD,
		TK_OP_SHIFT_LEFT,
		TK_OP_SHIFT_RIGHT,
		TK_OP_ASSIGN,
		TK_OP_ASSIGN_ADD,
		TK_OP_ASSIGN_SUB,
		TK_OP_ASSIGN_MUL,
		TK_OP_ASSIGN_DIV,
		TK_OP_ASSIGN_MOD,
		TK_OP_ASSIGN_SHIFT_LEFT,
		TK_OP_ASSIGN_SHIFT_RIGHT,
		TK_OP_ASSIGN_BIT_AND,
		TK_OP_ASSIGN_BIT_OR,
		TK_OP_ASSIGN_BIT_XOR,
		TK_OP_BIT_AND,
		TK_OP_BIT_OR,
		TK_OP_BIT_XOR,
		TK_OP_BIT_INVERT,
		TK_OP_INCREMENT,
		TK_OP_DECREMENT,
		TK_CF_IF,
		TK_CF_ELSE,
		TK_CF_FOR,
		TK_CF_WHILE,
		TK_CF_DO,
		TK_CF_SWITCH,
		TK_CF_CASE,
		TK_CF_DEFAULT,
		TK_CF_BREAK,
		TK_CF_CONTINUE,
		TK_CF_RETURN,
		TK_CF_DISCARD,
		TK_BRACKET_OPEN,
		TK_BRACKET_CLOSE,
		TK_CURLY_BRACKET_OPEN,
		TK_CURLY_BRACKET_CLOSE,
		TK_PARENTHESIS_OPEN,
		TK_PARENTHESIS_CLOSE,
		TK_QUESTION,
		TK_COMMA,
		TK_COLON,
		TK_SEMICOLON,
		TK_PERIOD,
		TK_UNIFORM,
		TK_VARYING,
		TK_ARG_IN,
		TK_ARG_OUT,
		TK_ARG_INOUT,
		TK_RENDER_MODE,
		TK_HINT_WHITE_TEXTURE,
		TK_HINT_BLACK_TEXTURE,
		TK_HINT_TRANSPARENT_TEXTURE,
		TK_HINT_NORMAL_TEXTURE,
		TK_HINT_ANISO_TEXTURE,
		TK_HINT_ALBEDO_TEXTURE,
		TK_HINT_BLACK_ALBEDO_TEXTURE,
		TK_HINT_COLOR,
		TK_HINT_RANGE,
		TK_SHADER_TYPE,
		TK_CURSOR,
		TK_ERROR,
		TK_EOF,
		TK_MAX
	};

/* COMPILER */

// lame work around to Apple defining this as a macro in 10.12 SDK
#ifdef TYPE_BOOL
#undef TYPE_BOOL
#endif

	enum DataType {
		TYPE_VOID,
		TYPE_BOOL,
		TYPE_BVEC2,
		TYPE_BVEC3,
		TYPE_BVEC4,
		TYPE_INT,
		TYPE_IVEC2,
		TYPE_IVEC3,
		TYPE_IVEC4,
		TYPE_UINT,
		TYPE_UVEC2,
		TYPE_UVEC3,
		TYPE_UVEC4,
		TYPE_FLOAT,
		TYPE_VEC2,
		TYPE_VEC3,
		TYPE_VEC4,
		TYPE_MAT2,
		TYPE_MAT3,
		TYPE_MAT4,
		TYPE_SAMPLER2D,
		TYPE_ISAMPLER2D,
		TYPE_USAMPLER2D,
		TYPE_SAMPLER2DARRAY,
		TYPE_ISAMPLER2DARRAY,
		TYPE_USAMPLER2DARRAY,
		TYPE_SAMPLER3D,
		TYPE_ISAMPLER3D,
		TYPE_USAMPLER3D,
		TYPE_SAMPLERCUBE,
		TYPE_SAMPLEREXT,
		TYPE_STRUCT,
		TYPE_MAX,
	};

	enum DataPrecision {
		PRECISION_LOWP,
		PRECISION_MEDIUMP,
		PRECISION_HIGHP,
		PRECISION_DEFAULT,
	};

	enum DataInterpolation {
		INTERPOLATION_FLAT,
		INTERPOLATION_SMOOTH,
	};

	enum Operator {
		OP_EQUAL,
		OP_NOT_EQUAL,
		OP_LESS,
		OP_LESS_EQUAL,
		OP_GREATER,
		OP_GREATER_EQUAL,
		OP_AND,
		OP_OR,
		OP_NOT,
		OP_NEGATE,
		OP_ADD,
		OP_SUB,
		OP_MUL,
		OP_DIV,
		OP_MOD,
		OP_SHIFT_LEFT,
		OP_SHIFT_RIGHT,
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
		OP_BIT_INVERT,
		OP_INCREMENT,
		OP_DECREMENT,
		OP_SELECT_IF,
		OP_SELECT_ELSE, //used only internally, then only IF appears with 3 arguments
		OP_POST_INCREMENT,
		OP_POST_DECREMENT,
		OP_CALL,
		OP_CONSTRUCT,
		OP_STRUCT,
		OP_INDEX,
		OP_MAX
	};

	enum FlowOperation {
		FLOW_OP_IF,
		FLOW_OP_RETURN,
		FLOW_OP_FOR,
		FLOW_OP_WHILE,
		FLOW_OP_DO,
		FLOW_OP_BREAK,
		FLOW_OP_SWITCH,
		FLOW_OP_CASE,
		FLOW_OP_DEFAULT,
		FLOW_OP_CONTINUE,
		FLOW_OP_DISCARD
	};

	enum ArgumentQualifier {
		ARGUMENT_QUALIFIER_IN,
		ARGUMENT_QUALIFIER_OUT,
		ARGUMENT_QUALIFIER_INOUT,
	};

	enum SubClassTag {
		TAG_GLOBAL,
		TAG_ARRAY,
	};

	struct VaryingFunctionNames {
		StringName fragment;
		StringName vertex;
		StringName light;
		VaryingFunctionNames() {
			fragment = "fragment";
			vertex = "vertex";
			light = "light";
		}
	};

	struct Node {
		Node *next;

		enum Type {
			TYPE_SHADER,
			TYPE_FUNCTION,
			TYPE_BLOCK,
			TYPE_VARIABLE,
			TYPE_VARIABLE_DECLARATION,
			TYPE_CONSTANT,
			TYPE_OPERATOR,
			TYPE_CONTROL_FLOW,
			TYPE_MEMBER,
			TYPE_ARRAY,
			TYPE_ARRAY_DECLARATION,
			TYPE_ARRAY_CONSTRUCT,
			TYPE_STRUCT,
		};

		Type type;

		virtual DataType get_datatype() const { return TYPE_VOID; }
		virtual String get_datatype_name() const { return ""; }

		Node(Type t) :
				next(nullptr),
				type(t) {}
		virtual ~Node() {}
	};

	template <class T>
	T *alloc_node() {
		T *node = memnew(T);
		node->next = nodes;
		nodes = node;
		return node;
	}

	Node *nodes;

	struct OperatorNode : public Node {
		DataType return_cache;
		DataPrecision return_precision_cache;
		Operator op;
		StringName struct_name;
		Vector<Node *> arguments;
		virtual DataType get_datatype() const { return return_cache; }
		virtual String get_datatype_name() const { return String(struct_name); }

		OperatorNode() :
				Node(TYPE_OPERATOR),
				return_cache(TYPE_VOID),
				return_precision_cache(PRECISION_DEFAULT),
				op(OP_EQUAL),
				struct_name("") {}
	};

	struct VariableNode : public Node {
		DataType datatype_cache;
		StringName name;
		StringName struct_name;
		virtual DataType get_datatype() const { return datatype_cache; }
		virtual String get_datatype_name() const { return String(struct_name); }
		bool is_const;
		bool is_local;

		VariableNode() :
				Node(TYPE_VARIABLE),
				datatype_cache(TYPE_VOID),
				is_const(false),
				is_local(false) {}
	};

	struct VariableDeclarationNode : public Node {
		DataPrecision precision;
		DataType datatype;
		String struct_name;
		bool is_const;

		struct Declaration {
			StringName name;
			Node *initializer;
		};

		Vector<Declaration> declarations;
		virtual DataType get_datatype() const { return datatype; }

		VariableDeclarationNode() :
				Node(TYPE_VARIABLE_DECLARATION),
				precision(PRECISION_DEFAULT),
				datatype(TYPE_VOID),
				is_const(false) {}
	};

	struct ArrayNode : public Node {
		DataType datatype_cache;
		StringName struct_name;
		StringName name;
		Node *index_expression;
		Node *call_expression;
		Node *assign_expression;
		bool is_const;
		bool is_local;

		virtual DataType get_datatype() const { return datatype_cache; }
		virtual String get_datatype_name() const { return String(struct_name); }

		ArrayNode() :
				Node(TYPE_ARRAY),
				datatype_cache(TYPE_VOID),
				index_expression(nullptr),
				call_expression(nullptr),
				assign_expression(nullptr),
				is_const(false),
				is_local(false) {}
	};

	struct ArrayConstructNode : public Node {
		DataType datatype;
		String struct_name;
		Vector<Node *> initializer;

		ArrayConstructNode() :
				Node(TYPE_ARRAY_CONSTRUCT),
				datatype(TYPE_VOID) {
		}
	};

	struct ArrayDeclarationNode : public Node {
		DataPrecision precision;
		DataType datatype;
		String struct_name;
		bool is_const;

		struct Declaration {
			StringName name;
			uint32_t size;
			Vector<Node *> initializer;
		};

		Vector<Declaration> declarations;
		virtual DataType get_datatype() const { return datatype; }

		ArrayDeclarationNode() :
				Node(TYPE_ARRAY_DECLARATION),
				precision(PRECISION_DEFAULT),
				datatype(TYPE_VOID),
				is_const(false) {}
	};

	struct ConstantNode : public Node {
		DataType datatype;
		String struct_name = "";
		int array_size = 0;

		union Value {
			bool boolean;
			float real;
			int32_t sint;
			uint32_t uint;
		};

		Vector<Value> values;
		Vector<ArrayDeclarationNode::Declaration> array_declarations;
		virtual DataType get_datatype() const { return datatype; }
		virtual String get_datatype_name() const { return struct_name; }

		ConstantNode() :
				Node(TYPE_CONSTANT),
				datatype(TYPE_VOID) {}
	};

	struct FunctionNode;

	struct BlockNode : public Node {
		FunctionNode *parent_function;
		BlockNode *parent_block;

		enum BlockType {
			BLOCK_TYPE_STANDART,
			BLOCK_TYPE_SWITCH,
			BLOCK_TYPE_CASE,
			BLOCK_TYPE_DEFAULT,
		};

		int block_type;
		SubClassTag block_tag;

		struct Variable {
			DataType type;
			StringName struct_name;
			DataPrecision precision;
			int line; //for completion
			int array_size;
			bool is_const;
		};

		Map<StringName, Variable> variables;
		List<Node *> statements;
		bool single_statement;

		BlockNode() :
				Node(TYPE_BLOCK),
				parent_function(nullptr),
				parent_block(nullptr),
				block_type(BLOCK_TYPE_STANDART),
				block_tag(SubClassTag::TAG_GLOBAL),
				single_statement(false) {}
	};

	struct ControlFlowNode : public Node {
		FlowOperation flow_op;
		Vector<Node *> expressions;
		Vector<BlockNode *> blocks;

		ControlFlowNode() :
				Node(TYPE_CONTROL_FLOW),
				flow_op(FLOW_OP_IF) {}
	};

	struct MemberNode : public Node {
		DataType basetype;
		bool basetype_const;
		StringName base_struct_name;
		DataPrecision precision;
		DataType datatype;
		int array_size;
		StringName struct_name;
		StringName name;
		Node *owner;
		Node *index_expression;
		Node *assign_expression;
		bool has_swizzling_duplicates;

		virtual DataType get_datatype() const { return datatype; }
		virtual String get_datatype_name() const { return String(struct_name); }

		MemberNode() :
				Node(TYPE_MEMBER),
				basetype(TYPE_VOID),
				basetype_const(false),
				datatype(TYPE_VOID),
				array_size(0),
				owner(nullptr),
				index_expression(nullptr),
				assign_expression(nullptr),
				has_swizzling_duplicates(false) {}
	};

	struct StructNode : public Node {
		List<MemberNode *> members;
		StructNode() :
				Node(TYPE_STRUCT) {}
	};

	struct FunctionNode : public Node {
		struct Argument {
			ArgumentQualifier qualifier;
			StringName name;
			DataType type;
			StringName type_str;
			DataPrecision precision;
			bool is_const;
		};

		StringName name;
		DataType return_type;
		StringName return_struct_name;
		DataPrecision return_precision;
		Vector<Argument> arguments;
		BlockNode *body;
		bool can_discard;

		FunctionNode() :
				Node(TYPE_FUNCTION),
				return_type(TYPE_VOID),
				return_precision(PRECISION_DEFAULT),
				body(nullptr),
				can_discard(false) {}
	};

	struct ShaderNode : public Node {
		struct Constant {
			StringName name;
			DataType type;
			StringName type_str;
			DataPrecision precision;
			ConstantNode *initializer;
			int array_size;
		};

		struct Function {
			StringName name;
			FunctionNode *function;
			Vector<StringName> uses_function;
			bool callable;
		};

		struct Struct {
			StringName name;
			StructNode *shader_struct;
		};

		struct Varying {
			enum Stage {
				STAGE_UNKNOWN,
				STAGE_VERTEX, // transition stage to STAGE_VERTEX_TO_FRAGMENT_LIGHT, emits warning if it's not used
				STAGE_FRAGMENT, // transition stage to STAGE_FRAGMENT_TO_LIGHT, emits warning if it's not used
				STAGE_VERTEX_TO_FRAGMENT_LIGHT,
				STAGE_FRAGMENT_TO_LIGHT,
			};

			Stage stage;
			DataType type;
			DataInterpolation interpolation;
			DataPrecision precision;
			int array_size;
			TkPos tkpos;

			Varying() :
					stage(STAGE_UNKNOWN),
					type(TYPE_VOID),
					interpolation(INTERPOLATION_FLAT),
					precision(PRECISION_DEFAULT),
					array_size(0) {}
		};

		struct Uniform {
			enum Hint {
				HINT_NONE,
				HINT_COLOR,
				HINT_RANGE,
				HINT_ALBEDO,
				HINT_BLACK_ALBEDO,
				HINT_NORMAL,
				HINT_BLACK,
				HINT_WHITE,
				HINT_TRANSPARENT,
				HINT_ANISO,
				HINT_MAX
			};

			int order;
			int texture_order;
			DataType type;
			DataPrecision precision;
			Vector<ConstantNode::Value> default_value;
			Hint hint;
			float hint_range[3];

			Uniform() :
					order(0),
					texture_order(0),
					type(TYPE_VOID),
					precision(PRECISION_DEFAULT),
					hint(HINT_NONE) {
				hint_range[0] = 0.0f;
				hint_range[1] = 1.0f;
				hint_range[2] = 0.001f;
			}
		};

		Map<StringName, Constant> constants;
		OrderedHashMap<StringName, Varying> varyings;
		OrderedHashMap<StringName, Uniform> uniforms;
		Map<StringName, Struct> structs;
		Vector<StringName> render_modes;

		Vector<Function> functions;
		Vector<Constant> vconstants;
		Vector<Struct> vstructs;

		ShaderNode() :
				Node(TYPE_SHADER) {}
	};

	struct Expression {
		bool is_op;
		union {
			Operator op;
			Node *node;
		};
	};

	struct VarInfo {
		StringName name;
		DataType type;
	};

	enum CompletionType {
		COMPLETION_NONE,
		COMPLETION_SHADER_TYPE,
		COMPLETION_RENDER_MODE,
		COMPLETION_MAIN_FUNCTION,
		COMPLETION_IDENTIFIER,
		COMPLETION_FUNCTION_CALL,
		COMPLETION_CALL_ARGUMENTS,
		COMPLETION_INDEX,
		COMPLETION_STRUCT,
		COMPLETION_HINT,
	};

	struct Token {
		TokenType type;
		StringName text;
		double constant;
		uint16_t line;
		bool is_integer_constant() const {
			return type == TK_INT_CONSTANT || type == TK_UINT_CONSTANT;
		}
	};

	static String get_operator_text(Operator p_op);
	static String get_token_text(Token p_token);

	static bool is_token_datatype(TokenType p_type);
	static bool is_token_variable_datatype(TokenType p_type);
	static DataType get_token_datatype(TokenType p_type);
	static bool is_token_interpolation(TokenType p_type);
	static DataInterpolation get_token_interpolation(TokenType p_type);
	static bool is_token_precision(TokenType p_type);
	static DataPrecision get_token_precision(TokenType p_type);
	static String get_precision_name(DataPrecision p_type);
	static String get_datatype_name(DataType p_type);
	static bool is_token_nonvoid_datatype(TokenType p_type);
	static bool is_token_operator(TokenType p_type);
	static bool is_token_operator_assign(TokenType p_type);
	static bool is_token_hint(TokenType p_type);

	static bool convert_constant(ConstantNode *p_constant, DataType p_to_type, ConstantNode::Value *p_value = nullptr);
	static DataType get_scalar_type(DataType p_type);
	static int get_cardinality(DataType p_type);
	static bool is_scalar_type(DataType p_type);
	static bool is_sampler_type(DataType p_type);
	static Variant constant_value_to_variant(const Vector<ShaderLanguage::ConstantNode::Value> &p_value, DataType p_type, ShaderLanguage::ShaderNode::Uniform::Hint p_hint = ShaderLanguage::ShaderNode::Uniform::HINT_NONE);

	static void get_keyword_list(List<String> *r_keywords);
	static bool is_control_flow_keyword(String p_keyword);
	static void get_builtin_funcs(List<String> *r_keywords);

	struct BuiltInInfo {
		DataType type;
		bool constant;

		BuiltInInfo() :
				type(TYPE_VOID),
				constant(false) {}

		BuiltInInfo(DataType p_type, bool p_constant = false) :
				type(p_type),
				constant(p_constant) {}
	};

	struct FunctionInfo {
		Map<StringName, BuiltInInfo> built_ins;
		bool can_discard;
		bool main_function;

		FunctionInfo() :
				can_discard(false), main_function(false) {
		}
	};
	static bool has_builtin(const Map<StringName, ShaderLanguage::FunctionInfo> &p_functions, const StringName &p_name);

private:
	struct KeyWord {
		TokenType token;
		const char *text;
	};

	static const KeyWord keyword_list[];

	bool error_set;
	String error_str;
	int error_line;

	String code;
	int char_idx;
	int tk_line;

	StringName current_function;
	bool last_const = false;

	VaryingFunctionNames varying_function_names;

	TkPos _get_tkpos() {
		TkPos tkp;
		tkp.char_idx = char_idx;
		tkp.tk_line = tk_line;
		return tkp;
	}

	void _set_tkpos(TkPos p_pos) {
		char_idx = p_pos.char_idx;
		tk_line = p_pos.tk_line;
	}

	void _set_error(const String &p_str) {
		if (error_set) {
			return;
		}

		error_line = tk_line;
		error_set = true;
		error_str = p_str;
	}

	static const char *token_names[TK_MAX];

	Token _make_token(TokenType p_type, const StringName &p_text = StringName());
	Token _get_token();

	ShaderNode *shader;

	enum IdentifierType {
		IDENTIFIER_FUNCTION,
		IDENTIFIER_UNIFORM,
		IDENTIFIER_VARYING,
		IDENTIFIER_FUNCTION_ARGUMENT,
		IDENTIFIER_LOCAL_VAR,
		IDENTIFIER_BUILTIN_VAR,
		IDENTIFIER_CONSTANT,
	};

	bool _find_identifier(const BlockNode *p_block, const Map<StringName, BuiltInInfo> &p_builtin_types, const StringName &p_identifier, DataType *r_data_type = nullptr, IdentifierType *r_type = nullptr, bool *r_is_const = nullptr, int *r_array_size = nullptr, StringName *r_struct_name = nullptr, ConstantNode::Value *r_constant_value = nullptr);
	bool _is_operator_assign(Operator p_op) const;
	bool _validate_assign(Node *p_node, const Map<StringName, BuiltInInfo> &p_builtin_types, String *r_message = nullptr);
	bool _validate_operator(OperatorNode *p_op, DataType *r_ret_type = nullptr);

	struct BuiltinFuncDef {
		enum { MAX_ARGS = 5 };
		const char *name;
		DataType rettype;
		const DataType args[MAX_ARGS];
		SubClassTag tag;
		bool high_end;
	};

	struct BuiltinFuncOutArgs { //arguments used as out in built in functions
		const char *name;
		int argument;
	};

	CompletionType completion_type;
	int completion_line;
	BlockNode *completion_block;
	DataType completion_base;
	SubClassTag completion_class;
	StringName completion_function;
	StringName completion_struct;
	int completion_argument;

	bool _get_completable_identifier(BlockNode *p_block, CompletionType p_type, StringName &identifier);
	static const BuiltinFuncDef builtin_func_defs[];
	static const BuiltinFuncOutArgs builtin_func_out_args[];

	static bool is_const_suffix_lut_initialized;

	Error _validate_datatype(DataType p_type);
	bool _compare_datatypes_in_nodes(Node *a, Node *b) const;

	bool _validate_function_call(BlockNode *p_block, OperatorNode *p_func, DataType *r_ret_type, StringName *r_ret_type_str);
	bool _parse_function_arguments(BlockNode *p_block, const Map<StringName, BuiltInInfo> &p_builtin_types, OperatorNode *p_func, int *r_complete_arg = nullptr);
	bool _validate_varying_assign(ShaderNode::Varying &p_varying, String *r_message);

	Node *_parse_expression(BlockNode *p_block, const Map<StringName, BuiltInInfo> &p_builtin_types);
	Error _parse_array_size(BlockNode *p_block, int *r_array_size);
	Node *_parse_array_constructor(BlockNode *p_block, const Map<StringName, BuiltInInfo> &p_builtin_types, DataType p_type, const StringName &p_struct_name, int p_array_size);
	ShaderLanguage::Node *_reduce_expression(BlockNode *p_block, ShaderLanguage::Node *p_node);

	Node *_parse_and_reduce_expression(BlockNode *p_block, const Map<StringName, BuiltInInfo> &p_builtin_types);
	Error _parse_block(BlockNode *p_block, const Map<StringName, BuiltInInfo> &p_builtin_types, bool p_just_one = false, bool p_can_break = false, bool p_can_continue = false);
	String _get_shader_type_list(const Set<String> &p_shader_types) const;
	String _get_qualifier_str(ArgumentQualifier p_qualifier) const;

	Error _parse_shader(const Map<StringName, FunctionInfo> &p_functions, const Vector<StringName> &p_render_modes, const Set<String> &p_shader_types);

	Error _find_last_flow_op_in_block(BlockNode *p_block, FlowOperation p_op);
	Error _find_last_flow_op_in_op(ControlFlowNode *p_flow, FlowOperation p_op);

public:
	//static void get_keyword_list(ShaderType p_type,List<String> *p_keywords);

	void clear();

	static String get_shader_type(const String &p_code);
	Error compile(const String &p_code, const Map<StringName, FunctionInfo> &p_functions, const Vector<StringName> &p_render_modes, const Set<String> &p_shader_types);
	Error complete(const String &p_code, const Map<StringName, FunctionInfo> &p_functions, const Vector<StringName> &p_render_modes, const Set<String> &p_shader_types, List<ScriptCodeCompletionOption> *r_options, String &r_call_hint);

	String get_error_text();
	int get_error_line();

	ShaderNode *get_shader();

	String token_debug(const String &p_code);

	ShaderLanguage();
	~ShaderLanguage();
};

#endif // SHADER_LANGUAGE_H
