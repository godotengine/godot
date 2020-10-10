/*************************************************************************/
/*  expression.h                                                         */
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

#ifndef EXPRESSION_H
#define EXPRESSION_H

#include "core/reference.h"

class Expression : public Reference {
	GDCLASS(Expression, Reference);

public:
	enum BuiltinFunc {
		MATH_SIN,
		MATH_COS,
		MATH_TAN,
		MATH_SINH,
		MATH_COSH,
		MATH_TANH,
		MATH_ASIN,
		MATH_ACOS,
		MATH_ATAN,
		MATH_ATAN2,
		MATH_SQRT,
		MATH_FMOD,
		MATH_FPOSMOD,
		MATH_POSMOD,
		MATH_FLOOR,
		MATH_CEIL,
		MATH_ROUND,
		MATH_ABS,
		MATH_SIGN,
		MATH_POW,
		MATH_LOG,
		MATH_EXP,
		MATH_ISNAN,
		MATH_ISINF,
		MATH_EASE,
		MATH_STEP_DECIMALS,
		MATH_STEPIFY,
		MATH_LERP,
		MATH_LERP_ANGLE,
		MATH_INVERSE_LERP,
		MATH_RANGE_LERP,
		MATH_SMOOTHSTEP,
		MATH_MOVE_TOWARD,
		MATH_DECTIME,
		MATH_RANDOMIZE,
		MATH_RAND,
		MATH_RANDF,
		MATH_RANDOM,
		MATH_SEED,
		MATH_RANDSEED,
		MATH_DEG2RAD,
		MATH_RAD2DEG,
		MATH_LINEAR2DB,
		MATH_DB2LINEAR,
		MATH_POLAR2CARTESIAN,
		MATH_CARTESIAN2POLAR,
		MATH_WRAP,
		MATH_WRAPF,
		LOGIC_MAX,
		LOGIC_MIN,
		LOGIC_CLAMP,
		LOGIC_NEAREST_PO2,
		OBJ_WEAKREF,
		FUNC_FUNCREF,
		TYPE_CONVERT,
		TYPE_OF,
		TYPE_EXISTS,
		TEXT_CHAR,
		TEXT_ORD,
		TEXT_STR,
		TEXT_PRINT,
		TEXT_PRINTERR,
		TEXT_PRINTRAW,
		VAR_TO_STR,
		STR_TO_VAR,
		VAR_TO_BYTES,
		BYTES_TO_VAR,
		COLORN,
		FUNC_MAX
	};

	static int get_func_argument_count(BuiltinFunc p_func);
	static String get_func_name(BuiltinFunc p_func);
	static void exec_func(BuiltinFunc p_func, const Variant **p_inputs, Variant *r_return, Callable::CallError &r_error, String &r_error_str);
	static BuiltinFunc find_function(const String &p_string);

private:
	static const char *func_name[FUNC_MAX];

	struct Input {
		Variant::Type type = Variant::NIL;
		String name;

		Input() {}
	};

	Vector<Input> inputs;
	Variant::Type output_type = Variant::NIL;

	String expression;

	bool sequenced = false;
	int str_ofs = 0;
	bool expression_dirty = false;

	bool _compile_expression();

	enum TokenType {
		TK_CURLY_BRACKET_OPEN,
		TK_CURLY_BRACKET_CLOSE,
		TK_BRACKET_OPEN,
		TK_BRACKET_CLOSE,
		TK_PARENTHESIS_OPEN,
		TK_PARENTHESIS_CLOSE,
		TK_IDENTIFIER,
		TK_BUILTIN_FUNC,
		TK_SELF,
		TK_CONSTANT,
		TK_BASIC_TYPE,
		TK_COLON,
		TK_COMMA,
		TK_PERIOD,
		TK_OP_IN,
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
		TK_OP_BIT_AND,
		TK_OP_BIT_OR,
		TK_OP_BIT_XOR,
		TK_OP_BIT_INVERT,
		TK_INPUT,
		TK_EOF,
		TK_ERROR,
		TK_MAX
	};

	static const char *token_name[TK_MAX];
	struct Token {
		TokenType type;
		Variant value;
	};

	void _set_error(const String &p_err) {
		if (error_set) {
			return;
		}
		error_str = p_err;
		error_set = true;
	}

	Error _get_token(Token &r_token);

	String error_str;
	bool error_set = true;

	struct ENode {
		enum Type {
			TYPE_INPUT,
			TYPE_CONSTANT,
			TYPE_SELF,
			TYPE_OPERATOR,
			TYPE_INDEX,
			TYPE_NAMED_INDEX,
			TYPE_ARRAY,
			TYPE_DICTIONARY,
			TYPE_CONSTRUCTOR,
			TYPE_BUILTIN_FUNC,
			TYPE_CALL
		};

		ENode *next = nullptr;

		Type type;

		ENode() {}
		virtual ~ENode() {
			if (next) {
				memdelete(next);
			}
		}
	};

	struct ExpressionNode {
		bool is_op;
		union {
			Variant::Operator op;
			ENode *node;
		};
	};

	ENode *_parse_expression();

	struct InputNode : public ENode {
		int index;
		InputNode() {
			type = TYPE_INPUT;
		}
	};

	struct ConstantNode : public ENode {
		Variant value;
		ConstantNode() {
			type = TYPE_CONSTANT;
		}
	};

	struct OperatorNode : public ENode {
		Variant::Operator op;

		ENode *nodes[2];

		OperatorNode() {
			type = TYPE_OPERATOR;
		}
	};

	struct SelfNode : public ENode {
		SelfNode() {
			type = TYPE_SELF;
		}
	};

	struct IndexNode : public ENode {
		ENode *base;
		ENode *index;

		IndexNode() {
			type = TYPE_INDEX;
		}
	};

	struct NamedIndexNode : public ENode {
		ENode *base;
		StringName name;

		NamedIndexNode() {
			type = TYPE_NAMED_INDEX;
		}
	};

	struct ConstructorNode : public ENode {
		Variant::Type data_type;
		Vector<ENode *> arguments;

		ConstructorNode() {
			type = TYPE_CONSTRUCTOR;
		}
	};

	struct CallNode : public ENode {
		ENode *base;
		StringName method;
		Vector<ENode *> arguments;

		CallNode() {
			type = TYPE_CALL;
		}
	};

	struct ArrayNode : public ENode {
		Vector<ENode *> array;
		ArrayNode() {
			type = TYPE_ARRAY;
		}
	};

	struct DictionaryNode : public ENode {
		Vector<ENode *> dict;
		DictionaryNode() {
			type = TYPE_DICTIONARY;
		}
	};

	struct BuiltinFuncNode : public ENode {
		BuiltinFunc func;
		Vector<ENode *> arguments;
		BuiltinFuncNode() {
			type = TYPE_BUILTIN_FUNC;
		}
	};

	template <class T>
	T *alloc_node() {
		T *node = memnew(T);
		node->next = nodes;
		nodes = node;
		return node;
	}

	ENode *root = nullptr;
	ENode *nodes = nullptr;

	Vector<String> input_names;

	bool execution_error = false;
	bool _execute(const Array &p_inputs, Object *p_instance, Expression::ENode *p_node, Variant &r_ret, String &r_error_str);

protected:
	static void _bind_methods();

public:
	Error parse(const String &p_expression, const Vector<String> &p_input_names = Vector<String>());
	Variant execute(Array p_inputs = Array(), Object *p_base = nullptr, bool p_show_error = true);
	bool has_execute_failed() const;
	String get_error_text() const;

	Expression() {}
	~Expression();
};

#endif // EXPRESSION_H
