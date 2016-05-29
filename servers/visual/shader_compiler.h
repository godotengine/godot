/*************************************************************************/
/*  shader_compiler.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef SHADER_COMPILER_H
#define SHADER_COMPILER_H

#include "map.h"
#include "list.h"
#include "vector.h"
#if 0
class ShaderSyntax {
public:


	enum DataType {
		TYPE_BOOL,
		TYPE_FLOAT,
		TYPE_VEC3,
		TYPE_TRANSFORM,
		TYPE_TEXTURE
	};

	enum Operator {
		OP_ASSIGN,
		OP_ADD,
		OP_SUB,
		OP_MUL,
		OP_DIV,
		OP_NEG,
		OP_CMP_EQ,
		OP_CMP_NEQ,
		OP_CMP_LEQ,
		OP_CMP_GEQ,
		OP_CMP_OR,
		OP_CMP_AND,
		OP_CALL
	};

	struct Node {

		enum Type {
			TYPE_PROGRAM,
			TYPE_FUNCTION,
			TYPE_BLOCK,
			TYPE_VARIABLE,
			TYPE_OPERATOR,
			TYPE_IF,
		};

		Node * parent;
		Type type;

		virtual ~Node() {}
	};


	struct OperatorNode : public Node {

		Operator op;
		Vector<Node*> arguments;
		OperatorNode() { type=TYPE_OPERATOR; }
	};

	struct VariableNode : public Node {

		StringName variable;
		VariableNode() { type=TYPE_VARIABLE; }
	};

	struct BlockNode : public Node {

		Map<StringName,DataType> variables;
		List<Node*> subnodes;
		BlockNode() { type=TYPE_BLOCK; }
	};

	struct ConditionalNode : public Node {

		Node *test;
		Node *do_if;
		Node *do_else;
		ConditionalNode() { type=TYPE_CONDITIONAL; }
	};


	struct FunctionNode : public Node {

		struct Argument {

			StringName name;
			DataType type;
		};

		Vector<Argument> arguments;
		Node *body;

		FunctionNode() { type=TYPE_FUNCTION; }

	};


	struct ProgramNode : public Node {

		Vector<FunctionNode*> functions;
		Node *body;

		ProgramNode() { type=TYPE_PROGRAM; }
	};




	ShaderCompiler();
};

#endif // SHADER_COMPILER_H
#endif
