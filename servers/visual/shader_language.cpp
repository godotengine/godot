/*************************************************************************/
/*  shader_language.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "shader_language.h"
#include "os/os.h"
#include "print_string.h"
static bool _is_text_char(CharType c) {

	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

static bool _is_number(CharType c) {

	return (c >= '0' && c <= '9');
}

static bool _is_hex(CharType c) {

	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

const char *ShaderLanguage::token_names[TK_MAX] = {
	"EMPTY",
	"INDENTIFIER",
	"TRUE",
	"FALSE",
	"REAL_CONSTANT",
	"TYPE_VOID",
	"TYPE_BOOL",
	"TYPE_FLOAT",
	"TYPE_VEC2",
	"TYPE_VEC3",
	"TYPE_VEC4",
	"TYPE_MAT2",
	"TYPE_MAT3",
	"TYPE_MAT4",
	"TYPE_TEXTURE",
	"TYPE_CUBEMAP",
	"TYPE_COLOR",
	"OP_EQUAL",
	"OP_NOT_EQUAL",
	"OP_LESS",
	"OP_LESS_EQUAL",
	"OP_GREATER",
	"OP_GREATER_EQUAL",
	"OP_AND",
	"OP_OR",
	"OP_NOT",
	"OP_ADD",
	"OP_SUB",
	"OP_MUL",
	"OP_DIV",
	"OP_NEG",
	"OP_ASSIGN",
	"OP_ASSIGN_ADD",
	"OP_ASSIGN_SUB",
	"OP_ASSIGN_MUL",
	"OP_ASSIGN_DIV",
	"CF_IF",
	"CF_ELSE",
	"CF_RETURN",
	"BRACKET_OPEN",
	"BRACKET_CLOSE",
	"CURLY_BRACKET_OPEN",
	"CURLY_BRACKET_CLOSE",
	"PARENTHESIS_OPEN",
	"PARENTHESIS_CLOSE",
	"COMMA",
	"SEMICOLON",
	"PERIOD",
	"UNIFORM",
	"ERROR",
};

ShaderLanguage::Token ShaderLanguage::read_token(const CharType *p_text, int p_len, int &r_line, int &r_chars) {

#define GETCHAR(m_idx) ((m_idx < p_len) ? p_text[m_idx] : CharType(0))

	r_chars = 1; //by default everything eats one char
	switch (GETCHAR(0)) {

		case '\t':
		case '\r':
		case ' ':
			return Token();
		case '\n':
			r_line++;
			return Token();
		case '/': {

			switch (GETCHAR(1)) {
				case '*': { // block comment

					while (true) {
						if (GETCHAR(r_chars + 1) == 0) {
							r_chars += 1;
							break;
						}
						if (GETCHAR(r_chars + 1) == '*' && GETCHAR(r_chars + 2) == '/') {
							r_chars += 3;
							break;
						}
						if (GETCHAR(r_chars + 1) == '\n') {
							r_line++;
						}

						r_chars++;
					}

					return Token();

				} break;
				case '/': { // line comment skip

					while (GETCHAR(r_chars + 1) != '\n' && GETCHAR(r_chars + 1) != 0) {
						r_chars++;
					}
					r_chars++;
					//r_line++;

					return Token();

				} break;
				case '=': { // diveq

					r_chars = 2;
					return Token(TK_OP_ASSIGN_DIV);

				} break;
				default:
					return Token(TK_OP_DIV);
			}
		} break;
		case '=': {
			if (GETCHAR(1) == '=') {
				r_chars++;
				return Token(TK_OP_EQUAL);
			}

			return Token(TK_OP_ASSIGN);

		} break;
		case '<': {
			if (GETCHAR(1) == '=') {
				r_chars++;
				return Token(TK_OP_LESS_EQUAL);
			} /*else if (GETCHAR(1)=='<') {
				r_chars++;
				if (GETCHAR(2)=='=') {
					r_chars++;
					return Token(TK_OP_ASSIGN_SHIFT_LEFT);
				}

				return Token(TK_OP_SHIFT_LEFT);
			}*/

			return Token(TK_OP_LESS);

		} break;
		case '>': {
			if (GETCHAR(1) == '=') {
				r_chars++;
				return Token(TK_OP_GREATER_EQUAL);
			} /* else if (GETCHAR(1)=='<') {
				r_chars++;
				if (GETCHAR(2)=='=') {
					r_chars++;
					return Token(TK_OP_ASSIGN_SHIFT_RIGHT);
				}

				return Token(TK_OP_SHIFT_RIGHT);
			}*/

			return Token(TK_OP_GREATER);

		} break;
		case '!': {
			if (GETCHAR(1) == '=') {
				r_chars++;
				return Token(TK_OP_NOT_EQUAL);
			}

			return Token(TK_OP_NOT);

		} break;
		//case '"' //string - no strings in shader
		//case '\'' //string - no strings in shader
		case '{':
			return Token(TK_CURLY_BRACKET_OPEN);
		case '}':
			return Token(TK_CURLY_BRACKET_CLOSE);
		//case '[':
		//	return Token(TK_BRACKET_OPEN);
		//case ']':
		//		return Token(TK_BRACKET_CLOSE);
		case '(':
			return Token(TK_PARENTHESIS_OPEN);
		case ')':
			return Token(TK_PARENTHESIS_CLOSE);
		case ',':
			return Token(TK_COMMA);
		case ';':
			return Token(TK_SEMICOLON);
		//case '?':
		//	return Token(TK_QUESTION_MARK);
		//case ':':
		//	return Token(TK_COLON); //for methods maybe but now useless.
		//case '^':
		//	return Token(TK_OP_BIT_XOR);
		//case '~':
		//		return Token(TK_OP_BIT_INVERT);
		case '&': {

			if (GETCHAR(1) == '&') {

				r_chars++;
				return Token(TK_OP_AND);
			}

			return Token(TK_ERROR, "Unknown character");

			/*
			if (GETCHAR(1)=='=') {
				r_chars++;
				return Token(TK_OP_ASSIGN_BIT_AND);
			} else if (GETCHAR(1)=='&') {
				r_chars++;
				return Token(TK_OP_AND);
			}
			return TK_OP_BIT_AND;*/
		} break;
		case '|': {

			if (GETCHAR(1) == '|') {

				r_chars++;
				return Token(TK_OP_OR);
			}

			return Token(TK_ERROR, "Unknown character");

			/*
			if (GETCHAR(1)=='=') {
				r_chars++;
				return Token(TK_OP_ASSIGN_BIT_OR);
			} else if (GETCHAR(1)=='|') {
				r_chars++;
				return Token(TK_OP_OR);
			}
			return TK_OP_BIT_OR;
			*/
		} break;
		case '*': {

			if (GETCHAR(1) == '=') {
				r_chars++;
				return Token(TK_OP_ASSIGN_MUL);
			}
			return TK_OP_MUL;
		} break;
		case '+': {

			if (GETCHAR(1) == '=') {
				r_chars++;
				return Token(TK_OP_ASSIGN_ADD);
			} /*else if (GETCHAR(1)=='+') {

				r_chars++;
				return Token(TK_OP_PLUS_PLUS);
			}*/

			return TK_OP_ADD;
		} break;
		case '-': {

			if (GETCHAR(1) == '=') {
				r_chars++;
				return Token(TK_OP_ASSIGN_SUB);
			} /* else if (GETCHAR(1)=='-') {

				r_chars++;
				return Token(TK_OP_MINUS_MINUS);
			}*/

			return TK_OP_SUB;
		} break;
		/*case '%': {

			if (GETCHAR(1)=='=') {
				r_chars++;
				return Token(TK_OP_ASSIGN_MOD);
			}

			return TK_OP_MOD;
		} break;*/
		default: {

			if (_is_number(GETCHAR(0)) || (GETCHAR(0) == '.' && _is_number(GETCHAR(1)))) {
				// parse number
				bool period_found = false;
				bool exponent_found = false;
				bool hexa_found = false;
				bool sign_found = false;

				String str;
				int i = 0;

				while (true) {
					if (GETCHAR(i) == '.') {
						if (period_found || exponent_found)
							return Token(TK_ERROR, "Invalid numeric constant");
						period_found = true;
					} else if (GETCHAR(i) == 'x') {
						if (hexa_found || str.length() != 1 || str[0] != '0')
							return Token(TK_ERROR, "Invalid numeric constant");
						hexa_found = true;
					} else if (GETCHAR(i) == 'e') {
						if (hexa_found || exponent_found)
							return Token(TK_ERROR, "Invalid numeric constant");
						exponent_found = true;
					} else if (_is_number(GETCHAR(i))) {
						//all ok
					} else if (hexa_found && _is_hex(GETCHAR(i))) {

					} else if ((GETCHAR(i) == '-' || GETCHAR(i) == '+') && exponent_found) {
						if (sign_found)
							return Token(TK_ERROR, "Invalid numeric constant");
						sign_found = true;
					} else
						break;

					str += CharType(GETCHAR(i));
					i++;
				}

				if (!_is_number(str[str.length() - 1]))
					return Token(TK_ERROR, "Invalid numeric constant");

				r_chars += str.length() - 1;
				return Token(TK_REAL_CONSTANT, str);
				/*
				if (period_found)
					return Token(TK_NUMBER_REAL,str);
				else
					return Token(TK_NUMBER_INTEGER,str);*/
			}

			if (GETCHAR(0) == '.') {
				//parse period
				return Token(TK_PERIOD);
			}

			if (_is_text_char(GETCHAR(0))) {
				// parse identifier
				String str;
				str += CharType(GETCHAR(0));

				while (_is_text_char(GETCHAR(r_chars))) {

					str += CharType(GETCHAR(r_chars));
					r_chars++;
				}

				//see if keyword
				struct _kws {
					TokenType token;
					const char *text;
				};
				static const _kws keyword_list[] = {
					{ TK_TRUE, "true" },
					{ TK_FALSE, "false" },
					{ TK_TYPE_VOID, "void" },
					{ TK_TYPE_BOOL, "bool" },
					/*{TK_TYPE_INT,"int"},
					{TK_TYPE_INT2,"int2"},
					{TK_TYPE_INT3,"int3"},
					{TK_TYPE_INT4,"int4"},*/
					{ TK_TYPE_FLOAT, "float" },
					/*{TK_TYPE_FLOAT2,"float2"},
					{TK_TYPE_FLOAT3,"float3"},
					{TK_TYPE_FLOAT4,"float4"},*/
					{ TK_TYPE_VEC2, "vec2" },
					{ TK_TYPE_VEC3, "vec3" },
					{ TK_TYPE_VEC4, "vec4" },
					{ TK_TYPE_TEXTURE, "texture" },
					{ TK_TYPE_CUBEMAP, "cubemap" },
					{ TK_TYPE_COLOR, "color" },

					{ TK_TYPE_MAT2, "mat2" },
					/*{TK_TYPE_MAT3,"mat3"},
					{TK_TYPE_MAT4,"mat3"},*/
					{ TK_TYPE_MAT3, "mat3" },
					{ TK_TYPE_MAT4, "mat4" },
					{ TK_CF_IF, "if" },
					{ TK_CF_ELSE, "else" },
					/*
					{TK_CF_FOR,"for"},
					{TK_CF_WHILE,"while"},
					{TK_CF_DO,"do"},
					{TK_CF_SWITCH,"switch"},
					{TK_CF_BREAK,"break"},
					{TK_CF_CONTINUE,"continue"},*/
					{ TK_CF_RETURN, "return" },
					{ TK_UNIFORM, "uniform" },
					{ TK_ERROR, NULL }
				};

				int idx = 0;

				while (keyword_list[idx].text) {

					if (str == keyword_list[idx].text)
						return Token(keyword_list[idx].token);
					idx++;
				}

				return Token(TK_INDENTIFIER, str);
			}

			if (GETCHAR(0) > 32)
				return Token(TK_ERROR, "Tokenizer: Unknown character #" + itos(GETCHAR(0)) + ": '" + String::chr(GETCHAR(0)) + "'");
			else
				return Token(TK_ERROR, "Tokenizer: Unknown character #" + itos(GETCHAR(0)));

		} break;
	}

	ERR_PRINT("BUG");
	return Token();
}

Error ShaderLanguage::tokenize(const String &p_text, Vector<Token> *p_tokens, String *r_error, int *r_err_line, int *r_err_column) {

	int len = p_text.length();
	int pos = 0;

	int line = 0;
	int col = 0;

	while (pos < len) {

		int advance = 0;
		int prev_line = line;
		Token t = read_token(&p_text[pos], len - pos, line, advance);
		t.line = line;
		t.col = col;

		if (t.type == TK_ERROR) {

			if (r_error) {
				*r_error = t.text;
				*r_err_line = line;
				*r_err_column = col;
				return ERR_COMPILATION_FAILED;
			}
		}

		if (line == prev_line) {
			col += advance;
		} else {
			col = 0;
			//p_tokens->push_back(Token(TK_LINE,itos(line)))
		}

		if (t.type != TK_EMPTY)
			p_tokens->push_back(t);

		pos += advance;
	}

	return OK;
}

String ShaderLanguage::lex_debug(const String &p_code) {

	Vector<Token> tokens;
	String error;
	int errline, errcol;
	if (tokenize(p_code, &tokens, &error, &errline, &errcol) != OK)
		return error;
	String ret;
	for (int i = 0; i < tokens.size(); i++) {
		ret += String(token_names[tokens[i].type]) + ":" + itos(tokens[i].line) + ":" + itos(tokens[i].col) + ":" + tokens[i].text + "\n";
	}

	return ret;
}

bool ShaderLanguage::is_token_datatype(TokenType p_type) {

	return (p_type == TK_TYPE_VOID) ||
		   (p_type == TK_TYPE_BOOL) ||
		   (p_type == TK_TYPE_FLOAT) ||
		   (p_type == TK_TYPE_VEC2) ||
		   (p_type == TK_TYPE_VEC3) ||
		   (p_type == TK_TYPE_VEC4) ||
		   (p_type == TK_TYPE_COLOR) ||
		   (p_type == TK_TYPE_MAT2) ||
		   (p_type == TK_TYPE_MAT3) ||
		   (p_type == TK_TYPE_MAT4) ||
		   (p_type == TK_TYPE_CUBEMAP) ||
		   (p_type == TK_TYPE_TEXTURE);
}

ShaderLanguage::DataType ShaderLanguage::get_token_datatype(TokenType p_type) {

	switch (p_type) {

		case TK_TYPE_VOID: return TYPE_VOID;
		case TK_TYPE_BOOL: return TYPE_BOOL;
		case TK_TYPE_FLOAT: return TYPE_FLOAT;
		case TK_TYPE_VEC2: return TYPE_VEC2;
		case TK_TYPE_VEC3: return TYPE_VEC3;
		case TK_TYPE_VEC4: return TYPE_VEC4;
		case TK_TYPE_COLOR: return TYPE_VEC4;
		case TK_TYPE_MAT2: return TYPE_MAT2;
		case TK_TYPE_MAT3: return TYPE_MAT3;
		case TK_TYPE_MAT4: return TYPE_MAT4;
		case TK_TYPE_TEXTURE: return TYPE_TEXTURE;
		case TK_TYPE_CUBEMAP: return TYPE_CUBEMAP;
		default: return TYPE_VOID;
	}

	return TYPE_VOID;
}

String ShaderLanguage::get_datatype_name(DataType p_type) {

	switch (p_type) {

		case TYPE_VOID: return "void";
		case TYPE_BOOL: return "bool";
		case TYPE_FLOAT: return "float";
		case TYPE_VEC2: return "vec2";
		case TYPE_VEC3: return "vec3";
		case TYPE_VEC4: return "vec4";
		case TYPE_MAT2: return "mat2";
		case TYPE_MAT3: return "mat3";
		case TYPE_MAT4: return "mat4";
		case TYPE_TEXTURE: return "texture";
		case TYPE_CUBEMAP: return "cubemap";
		default: return "";
	}

	return "";
}

bool ShaderLanguage::is_token_nonvoid_datatype(TokenType p_type) {

	return (p_type == TK_TYPE_BOOL) ||
		   (p_type == TK_TYPE_FLOAT) ||
		   (p_type == TK_TYPE_VEC2) ||
		   (p_type == TK_TYPE_VEC3) ||
		   (p_type == TK_TYPE_VEC4) ||
		   (p_type == TK_TYPE_COLOR) ||
		   (p_type == TK_TYPE_MAT2) ||
		   (p_type == TK_TYPE_MAT3) ||
		   (p_type == TK_TYPE_MAT4) ||
		   (p_type == TK_TYPE_TEXTURE) ||
		   (p_type == TK_TYPE_CUBEMAP);
}

bool ShaderLanguage::parser_is_at_function(Parser &parser) {

	return (is_token_datatype(parser.get_next_token_type(0)) && parser.get_next_token_type(1) == TK_INDENTIFIER && parser.get_next_token_type(2) == TK_PARENTHESIS_OPEN);
}

bool ShaderLanguage::test_existing_identifier(Node *p_node, const StringName p_identifier, bool p_func, bool p_var, bool p_builtin) {

	Node *node = p_node;

	while (node) {

		if (node->type == Node::TYPE_BLOCK) {

			BlockNode *block = (BlockNode *)node;
			if (block->variables.has(p_identifier))
				return true;
		} else if (node->type == Node::TYPE_PROGRAM) {

			ProgramNode *program = (ProgramNode *)node;
			for (int i = 0; i < program->functions.size(); i++) {

				if (program->functions[i].name == p_identifier) {
					return true;
				}
			}

			if (program->builtin_variables.has(p_identifier)) {
				return true;
			}
			if (program->uniforms.has(p_identifier)) {
				return true;
			}

		} else if (node->type == Node::TYPE_FUNCTION) {

			FunctionNode *func = (FunctionNode *)node;
			for (int i = 0; i < func->arguments.size(); i++)
				if (func->arguments[i].name == p_identifier)
					return true;
		}

		node = node->parent;
	}

	// try keywords

	int idx = 0;

	//todo optimize
	while (intrinsic_func_defs[idx].name) {

		if (p_identifier.operator String() == intrinsic_func_defs[idx].name)
			return true;
		idx++;
	}

	return false;
}

Error ShaderLanguage::parse_function(Parser &parser, BlockNode *p_block) {

	if (!p_block->parent || p_block->parent->type != Node::TYPE_PROGRAM) {
		parser.set_error("Misplaced function");
		return ERR_PARSE_ERROR;
	}

	ProgramNode *program = (ProgramNode *)p_block->parent;

	StringName name = parser.get_next_token(1).text;

	if (test_existing_identifier(p_block, name)) {

		parser.set_error("Duplicate Identifier (existing variable/builtin/function): " + name);
		return ERR_PARSE_ERROR;
	}

	FunctionNode *function = parser.create_node<FunctionNode>(program);
	function->body = parser.create_node<BlockNode>(function);

	function->name = name;

	function->return_type = get_token_datatype(parser.get_next_token_type(0));

	{ //add to programnode
		ProgramNode::Function f;
		f.name = name;
		f.function = function;
		program->functions.push_back(f);
	}

	int ofs = 3;

	while (true) {

		//end of arguments
		if (parser.get_next_token_type(ofs) == TK_PARENTHESIS_CLOSE) {
			ofs++;
			break;
		}
		//next argument awaits
		if (parser.get_next_token_type(ofs) == TK_COMMA) {
			if (!is_token_nonvoid_datatype(parser.get_next_token_type(ofs + 1))) {
				parser.set_error("Expected Identifier or ')' following ','");
				return ERR_PARSE_ERROR;
			}
			ofs++;
			continue;
		}

		if (!is_token_nonvoid_datatype(parser.get_next_token_type(ofs + 0))) {
			parser.set_error("Invalid Argument Type");
			return ERR_PARSE_ERROR;
		}

		DataType identtype = get_token_datatype(parser.get_next_token_type(ofs + 0));

		if (parser.get_next_token_type(ofs + 1) != TK_INDENTIFIER) {
			parser.set_error("Expected Argument Identifier");
			return ERR_PARSE_ERROR;
		}

		StringName identname = parser.get_next_token(ofs + 1).text;

		if (test_existing_identifier(function, identname)) {
			parser.set_error("Duplicate Argument Identifier: " + identname);
			return ERR_DUPLICATE_SYMBOL;
		}

		FunctionNode::Argument arg;
		arg.name = identname;
		arg.type = identtype;
		//function->body->variables[arg.name]=arg.type;
		function->arguments.push_back(arg);

		ofs += 2;
	}

	parser.advance(ofs);
	// match {
	if (parser.get_next_token_type() != TK_CURLY_BRACKET_OPEN) {
		parser.set_error("Expected '{'");
		return ERR_PARSE_ERROR;
	}

	parser.advance();
	Error err = parse_block(parser, function->body);

	if (err)
		return err;

	// make sure that if the function has a return type, it does return something..
	if (function->return_type != TYPE_VOID) {
		bool found = false;
		for (int i = 0; i < function->body->statements.size(); i++) {
			if (function->body->statements[i]->type == Node::TYPE_CONTROL_FLOW) {

				ControlFlowNode *cf = (ControlFlowNode *)function->body->statements[i];
				if (cf->flow_op == FLOW_OP_RETURN) {
					// type of return was already checked when inserted
					// no need to check here
					found = true;
				}
			}
		}

		if (!found) {
			parser.set_error("Function must return a value (use the main block)");
			return ERR_PARSE_ERROR;
		}
	}

	return OK;
}

const ShaderLanguage::IntrinsicFuncDef ShaderLanguage::intrinsic_func_defs[] = {
	//constructors
	{ "bool", TYPE_BOOL, { TYPE_BOOL, TYPE_VOID } },
	{ "float", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "vec2", TYPE_VEC2, { TYPE_FLOAT, TYPE_VOID } },
	{ "vec2", TYPE_VEC2, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "vec3", TYPE_VEC3, { TYPE_FLOAT, TYPE_VOID } },
	{ "vec3", TYPE_VEC3, { TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "vec3", TYPE_VEC3, { TYPE_VEC2, TYPE_FLOAT, TYPE_VOID } },
	{ "vec3", TYPE_VEC3, { TYPE_FLOAT, TYPE_VEC2, TYPE_VOID } },
	{ "vec4", TYPE_VEC4, { TYPE_FLOAT, TYPE_VOID } },
	{ "vec4", TYPE_VEC4, { TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "vec4", TYPE_VEC4, { TYPE_FLOAT, TYPE_VEC2, TYPE_FLOAT, TYPE_VOID } },
	{ "vec4", TYPE_VEC4, { TYPE_VEC2, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "vec4", TYPE_VEC4, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VEC2, TYPE_VOID } },
	{ "vec4", TYPE_VEC4, { TYPE_FLOAT, TYPE_VEC3, TYPE_VOID } },
	{ "vec4", TYPE_VEC4, { TYPE_VEC3, TYPE_FLOAT, TYPE_VOID } },
	{ "vec4", TYPE_VEC4, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID } },
	{ "mat2", TYPE_MAT2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID } },
	{ "mat3", TYPE_MAT3, { TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "mat4", TYPE_MAT4, { TYPE_VEC4, TYPE_VEC4, TYPE_VEC4, TYPE_VEC4, TYPE_VOID } },
	//intrinsics - trigonometry
	{ "sin", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "cos", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "tan", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "asin", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "acos", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "atan", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "atan2", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "sinh", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "cosh", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "tanh", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	//intrinsics - exponential
	{ "pow", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "pow", TYPE_VEC2, { TYPE_VEC2, TYPE_FLOAT, TYPE_VOID } },
	{ "pow", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID } },
	{ "pow", TYPE_VEC3, { TYPE_VEC3, TYPE_FLOAT, TYPE_VOID } },
	{ "pow", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "pow", TYPE_VEC4, { TYPE_VEC4, TYPE_FLOAT, TYPE_VOID } },
	{ "pow", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID } },
	{ "exp", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "exp", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID } },
	{ "exp", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },
	{ "exp", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID } },
	{ "log", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "log", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID } },
	{ "log", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },
	{ "log", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID } },
	{ "sqrt", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "sqrt", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID } },
	{ "sqrt", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },
	{ "sqrt", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID } },
	//intrinsics - common
	{ "abs", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "abs", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID } },
	{ "abs", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },
	{ "abs", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID } },
	{ "sign", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "sign", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID } },
	{ "sign", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },
	{ "sign", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID } },
	{ "floor", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "floor", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID } },
	{ "floor", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },
	{ "floor", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID } },
	{ "trunc", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "trunc", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID } },
	{ "trunc", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },
	{ "trunc", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID } },
	{ "round", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "round", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID } },
	{ "round", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },
	{ "round", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID } },
	{ "ceil", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "ceil", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID } },
	{ "ceil", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },
	{ "ceil", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID } },
	{ "fract", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ "fract", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID } },
	{ "fract", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },
	{ "fract", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID } },
	{ "mod", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "mod", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID } },
	{ "mod", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "mod", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID } },
	{ "min", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "min", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID } },
	{ "min", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "min", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID } },
	{ "max", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "max", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID } },
	{ "max", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "max", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID } },
	{ "clamp", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "clamp", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VEC2, TYPE_VOID } },
	{ "clamp", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "clamp", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VEC4, TYPE_VOID } },
	{ "clamp", TYPE_VEC2, { TYPE_VEC2, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "clamp", TYPE_VEC3, { TYPE_VEC3, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "clamp", TYPE_VEC4, { TYPE_VEC4, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "mix", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "mix", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_FLOAT, TYPE_VOID } },
	{ "mix", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VEC2, TYPE_VOID } },
	{ "mix", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID } },
	{ "mix", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "mix", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID } },
	{ "mix", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VEC4, TYPE_VOID } },
	{ "step", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "step", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID } },
	{ "step", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "step", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID } },
	{ "step", TYPE_VEC2, { TYPE_FLOAT, TYPE_VEC2, TYPE_VOID } },
	{ "step", TYPE_VEC3, { TYPE_FLOAT, TYPE_VEC3, TYPE_VOID } },
	{ "step", TYPE_VEC4, { TYPE_FLOAT, TYPE_VEC4, TYPE_VOID } },
	{ "smoothstep", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID } },
	{ "smoothstep", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VEC2, TYPE_VOID } },
	{ "smoothstep", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "smoothstep", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VEC4, TYPE_VOID } },
	{ "smoothstep", TYPE_VEC2, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VEC2, TYPE_VOID } },
	{ "smoothstep", TYPE_VEC3, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VEC3, TYPE_VOID } },
	{ "smoothstep", TYPE_VEC4, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VEC4, TYPE_VOID } },

	//intrinsics - geometric
	{ "length", TYPE_FLOAT, { TYPE_VEC2, TYPE_VOID } },
	{ "length", TYPE_FLOAT, { TYPE_VEC3, TYPE_VOID } },
	{ "length", TYPE_FLOAT, { TYPE_VEC4, TYPE_VOID } },
	{ "distance", TYPE_FLOAT, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID } },
	{ "distance", TYPE_FLOAT, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "distance", TYPE_FLOAT, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID } },
	{ "dot", TYPE_FLOAT, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID } },
	{ "dot", TYPE_FLOAT, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "dot", TYPE_FLOAT, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID } },
	{ "cross", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "normalize", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID } },
	{ "normalize", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },
	{ "normalize", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID } },
	{ "reflect", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID } },
	{ "refract", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID } },
	//intrinsics - texture
	{ "tex", TYPE_VEC4, { TYPE_TEXTURE, TYPE_VEC2, TYPE_VOID } },
	{ "texcube", TYPE_VEC4, { TYPE_CUBEMAP, TYPE_VEC3, TYPE_VOID } },
	{ "texscreen", TYPE_VEC3, { TYPE_VEC2, TYPE_VOID } },
	{ "texpos", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },

	{ NULL, TYPE_VOID, { TYPE_VOID } }

};

const ShaderLanguage::OperatorDef ShaderLanguage::operator_defs[] = {

	{ OP_ASSIGN, TYPE_VOID, { TYPE_BOOL, TYPE_BOOL } },
	{ OP_ASSIGN, TYPE_VOID, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_ASSIGN, TYPE_VOID, { TYPE_VEC2, TYPE_VEC2 } },
	{ OP_ASSIGN, TYPE_VOID, { TYPE_VEC3, TYPE_VEC3 } },
	{ OP_ASSIGN, TYPE_VOID, { TYPE_VEC4, TYPE_VEC4 } },
	{ OP_ASSIGN, TYPE_VOID, { TYPE_MAT2, TYPE_MAT2 } },
	{ OP_ASSIGN, TYPE_VOID, { TYPE_MAT3, TYPE_MAT3 } },
	{ OP_ASSIGN, TYPE_VOID, { TYPE_MAT4, TYPE_MAT4 } },
	{ OP_ADD, TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_ADD, TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2 } },
	{ OP_ADD, TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3 } },
	{ OP_ADD, TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4 } },
	{ OP_SUB, TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_SUB, TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2 } },
	{ OP_SUB, TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3 } },
	{ OP_SUB, TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4 } },
	{ OP_MUL, TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_MUL, TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2 } },
	{ OP_MUL, TYPE_VEC2, { TYPE_VEC2, TYPE_FLOAT } },
	{ OP_MUL, TYPE_VEC2, { TYPE_FLOAT, TYPE_VEC2 } },
	{ OP_MUL, TYPE_VEC2, { TYPE_VEC2, TYPE_MAT3 } },
	{ OP_MUL, TYPE_VEC2, { TYPE_MAT2, TYPE_VEC2 } },
	{ OP_MUL, TYPE_VEC2, { TYPE_VEC2, TYPE_MAT2 } },
	{ OP_MUL, TYPE_VEC2, { TYPE_MAT3, TYPE_VEC2 } },
	{ OP_MUL, TYPE_VEC2, { TYPE_VEC2, TYPE_MAT4 } },
	{ OP_MUL, TYPE_VEC2, { TYPE_MAT4, TYPE_VEC2 } },
	{ OP_MUL, TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3 } },
	{ OP_MUL, TYPE_VEC3, { TYPE_VEC3, TYPE_FLOAT } },
	{ OP_MUL, TYPE_VEC3, { TYPE_FLOAT, TYPE_VEC3 } },
	{ OP_MUL, TYPE_VEC3, { TYPE_MAT3, TYPE_VEC3 } },
	{ OP_MUL, TYPE_VEC3, { TYPE_MAT4, TYPE_VEC3 } },
	{ OP_MUL, TYPE_VEC3, { TYPE_VEC3, TYPE_MAT3 } },
	{ OP_MUL, TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4 } },
	{ OP_MUL, TYPE_VEC4, { TYPE_VEC4, TYPE_FLOAT } },
	{ OP_MUL, TYPE_VEC4, { TYPE_FLOAT, TYPE_VEC4 } },
	{ OP_MUL, TYPE_VEC4, { TYPE_MAT4, TYPE_VEC4 } },
	{ OP_MUL, TYPE_VEC4, { TYPE_VEC4, TYPE_MAT4 } },
	{ OP_MUL, TYPE_MAT2, { TYPE_MAT2, TYPE_MAT2 } },
	{ OP_MUL, TYPE_MAT3, { TYPE_MAT3, TYPE_MAT3 } },
	{ OP_MUL, TYPE_MAT4, { TYPE_MAT4, TYPE_MAT4 } },
	{ OP_DIV, TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_DIV, TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2 } },
	{ OP_DIV, TYPE_VEC2, { TYPE_VEC2, TYPE_FLOAT } },
	{ OP_DIV, TYPE_VEC2, { TYPE_FLOAT, TYPE_VEC2 } },
	{ OP_DIV, TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3 } },
	{ OP_DIV, TYPE_VEC3, { TYPE_VEC3, TYPE_FLOAT } },
	{ OP_DIV, TYPE_VEC3, { TYPE_FLOAT, TYPE_VEC3 } },
	{ OP_DIV, TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4 } },
	{ OP_DIV, TYPE_VEC4, { TYPE_VEC4, TYPE_FLOAT } },
	{ OP_DIV, TYPE_VEC4, { TYPE_FLOAT, TYPE_VEC4 } },
	{ OP_ASSIGN_ADD, TYPE_VOID, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_ASSIGN_ADD, TYPE_VOID, { TYPE_VEC2, TYPE_VEC2 } },
	{ OP_ASSIGN_ADD, TYPE_VOID, { TYPE_VEC3, TYPE_VEC3 } },
	{ OP_ASSIGN_ADD, TYPE_VOID, { TYPE_VEC4, TYPE_VEC4 } },
	{ OP_ASSIGN_ADD, TYPE_VOID, { TYPE_VEC2, TYPE_FLOAT } },
	{ OP_ASSIGN_ADD, TYPE_VOID, { TYPE_VEC3, TYPE_FLOAT } },
	{ OP_ASSIGN_ADD, TYPE_VOID, { TYPE_VEC4, TYPE_FLOAT } },
	{ OP_ASSIGN_SUB, TYPE_VOID, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_ASSIGN_SUB, TYPE_VOID, { TYPE_VEC2, TYPE_VEC2 } },
	{ OP_ASSIGN_SUB, TYPE_VOID, { TYPE_VEC3, TYPE_VEC3 } },
	{ OP_ASSIGN_SUB, TYPE_VOID, { TYPE_VEC4, TYPE_VEC4 } },
	{ OP_ASSIGN_SUB, TYPE_VOID, { TYPE_VEC2, TYPE_FLOAT } },
	{ OP_ASSIGN_SUB, TYPE_VOID, { TYPE_VEC3, TYPE_FLOAT } },
	{ OP_ASSIGN_SUB, TYPE_VOID, { TYPE_VEC4, TYPE_FLOAT } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_VEC2, TYPE_VEC2 } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_VEC2, TYPE_FLOAT } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_VEC2, TYPE_MAT2 } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_MAT2, TYPE_MAT2 } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_VEC3, TYPE_MAT3 } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_VEC3, TYPE_VEC3 } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_VEC3, TYPE_FLOAT } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_VEC3, TYPE_MAT4 } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_VEC4, TYPE_VEC4 } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_VEC4, TYPE_FLOAT } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_VEC4, TYPE_MAT4 } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_MAT3, TYPE_MAT3 } },
	{ OP_ASSIGN_MUL, TYPE_VOID, { TYPE_MAT4, TYPE_MAT4 } },
	{ OP_ASSIGN_DIV, TYPE_VOID, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_ASSIGN_DIV, TYPE_VOID, { TYPE_VEC2, TYPE_VEC2 } },
	{ OP_ASSIGN_DIV, TYPE_VOID, { TYPE_VEC2, TYPE_FLOAT } },
	{ OP_ASSIGN_DIV, TYPE_VOID, { TYPE_VEC3, TYPE_VEC3 } },
	{ OP_ASSIGN_DIV, TYPE_VOID, { TYPE_VEC3, TYPE_FLOAT } },
	{ OP_ASSIGN_DIV, TYPE_VOID, { TYPE_VEC4, TYPE_VEC4 } },
	{ OP_ASSIGN_DIV, TYPE_VOID, { TYPE_VEC4, TYPE_FLOAT } },
	{ OP_NEG, TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID } },
	{ OP_NEG, TYPE_VEC2, { TYPE_VEC2, TYPE_VOID } },
	{ OP_NEG, TYPE_VEC3, { TYPE_VEC3, TYPE_VOID } },
	{ OP_NEG, TYPE_VEC4, { TYPE_VEC4, TYPE_VOID } },
	{ OP_NOT, TYPE_BOOL, { TYPE_BOOL, TYPE_VOID } },
	{ OP_CMP_EQ, TYPE_BOOL, { TYPE_BOOL, TYPE_BOOL } },
	{ OP_CMP_EQ, TYPE_BOOL, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_CMP_EQ, TYPE_BOOL, { TYPE_VEC3, TYPE_VEC2 } },
	{ OP_CMP_EQ, TYPE_BOOL, { TYPE_VEC3, TYPE_VEC3 } },
	{ OP_CMP_EQ, TYPE_BOOL, { TYPE_VEC3, TYPE_VEC4 } },
	//{OP_CMP_EQ,TYPE_MAT3,{TYPE_MAT4,TYPE_MAT3}}, ??
	//{OP_CMP_EQ,TYPE_MAT4,{TYPE_MAT4,TYPE_MAT4}}, ??
	{ OP_CMP_NEQ, TYPE_BOOL, { TYPE_BOOL, TYPE_BOOL } },
	{ OP_CMP_NEQ, TYPE_BOOL, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_CMP_NEQ, TYPE_BOOL, { TYPE_VEC2, TYPE_VEC2 } },
	{ OP_CMP_NEQ, TYPE_BOOL, { TYPE_VEC3, TYPE_VEC3 } },
	{ OP_CMP_NEQ, TYPE_BOOL, { TYPE_VEC4, TYPE_VEC4 } },
	//{OP_CMP_NEQ,TYPE_MAT4,{TYPE_MAT4,TYPE_MAT4}}, //?
	{ OP_CMP_LEQ, TYPE_BOOL, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_CMP_GEQ, TYPE_BOOL, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_CMP_LESS, TYPE_BOOL, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_CMP_GREATER, TYPE_BOOL, { TYPE_FLOAT, TYPE_FLOAT } },
	{ OP_CMP_OR, TYPE_BOOL, { TYPE_BOOL, TYPE_BOOL } },
	{ OP_CMP_AND, TYPE_BOOL, { TYPE_BOOL, TYPE_BOOL } },
	{ OP_MAX, TYPE_VOID, { TYPE_VOID, TYPE_VOID } }
};

const ShaderLanguage::BuiltinsDef ShaderLanguage::vertex_builtins_defs[] = {

	{ "SRC_VERTEX", TYPE_VEC3 },
	{ "SRC_NORMAL", TYPE_VEC3 },
	{ "SRC_TANGENT", TYPE_VEC3 },
	{ "SRC_BINORMALF", TYPE_FLOAT },

	{ "POSITION", TYPE_VEC4 },
	{ "VERTEX", TYPE_VEC3 },
	{ "NORMAL", TYPE_VEC3 },
	{ "TANGENT", TYPE_VEC3 },
	{ "BINORMAL", TYPE_VEC3 },
	{ "UV", TYPE_VEC2 },
	{ "UV2", TYPE_VEC2 },
	{ "COLOR", TYPE_VEC4 },
	{ "BONES", TYPE_VEC4 },
	{ "WEIGHTS", TYPE_VEC4 },
	{ "VAR1", TYPE_VEC4 },
	{ "VAR2", TYPE_VEC4 },
	{ "SPEC_EXP", TYPE_FLOAT },
	{ "POINT_SIZE", TYPE_FLOAT },

	//builtins
	{ "WORLD_MATRIX", TYPE_MAT4 },
	{ "INV_CAMERA_MATRIX", TYPE_MAT4 },
	{ "PROJECTION_MATRIX", TYPE_MAT4 },
	{ "MODELVIEW_MATRIX", TYPE_MAT4 },
	{ "INSTANCE_ID", TYPE_FLOAT },
	{ "TIME", TYPE_FLOAT },
	{ NULL, TYPE_VOID },
};
const ShaderLanguage::BuiltinsDef ShaderLanguage::fragment_builtins_defs[] = {

	{ "VERTEX", TYPE_VEC3 },
	{ "POSITION", TYPE_VEC4 },
	{ "NORMAL", TYPE_VEC3 },
	{ "TANGENT", TYPE_VEC3 },
	{ "BINORMAL", TYPE_VEC3 },
	{ "NORMALMAP", TYPE_VEC3 },
	{ "NORMALMAP_DEPTH", TYPE_FLOAT },
	{ "UV", TYPE_VEC2 },
	{ "UV2", TYPE_VEC2 },
	{ "COLOR", TYPE_VEC4 },
	{ "NORMAL", TYPE_VEC3 },
	{ "VAR1", TYPE_VEC4 },
	{ "VAR2", TYPE_VEC4 },
	{ "DIFFUSE", TYPE_VEC3 },
	{ "DIFFUSE_ALPHA", TYPE_VEC4 },
	{ "SPECULAR", TYPE_VEC3 },
	{ "EMISSION", TYPE_VEC3 },
	{ "SPEC_EXP", TYPE_FLOAT },
	{ "GLOW", TYPE_FLOAT },
	{ "SHADE_PARAM", TYPE_FLOAT },
	{ "DISCARD", TYPE_BOOL },
	{ "SCREEN_UV", TYPE_VEC2 },
	{ "POINT_COORD", TYPE_VEC2 },
	{ "INV_CAMERA_MATRIX", TYPE_MAT4 },

	//	{ "SCREEN_POS", TYPE_VEC2},
	//	{ "SCREEN_TEXEL_SIZE", TYPE_VEC2},
	{ "TIME", TYPE_FLOAT },
	{ NULL, TYPE_VOID }

};

const ShaderLanguage::BuiltinsDef ShaderLanguage::light_builtins_defs[] = {

	{ "NORMAL", TYPE_VEC3 },
	{ "LIGHT_DIR", TYPE_VEC3 },
	{ "LIGHT_DIFFUSE", TYPE_VEC3 },
	{ "LIGHT_SPECULAR", TYPE_VEC3 },
	{ "EYE_VEC", TYPE_VEC3 },
	{ "DIFFUSE", TYPE_VEC3 },
	{ "SPECULAR", TYPE_VEC3 },
	{ "SPECULAR_EXP", TYPE_FLOAT },
	{ "SHADE_PARAM", TYPE_FLOAT },
	{ "LIGHT", TYPE_VEC3 },
	{ "SHADOW", TYPE_VEC3 },
	{ "POINT_COORD", TYPE_VEC2 },
	//	{ "SCREEN_POS", TYPE_VEC2},
	//	{ "SCREEN_TEXEL_SIZE", TYPE_VEC2},
	{ "TIME", TYPE_FLOAT },
	{ NULL, TYPE_VOID }

};

const ShaderLanguage::BuiltinsDef ShaderLanguage::ci_vertex_builtins_defs[] = {

	{ "SRC_VERTEX", TYPE_VEC2 },
	{ "VERTEX", TYPE_VEC2 },
	{ "WORLD_VERTEX", TYPE_VEC2 },
	{ "UV", TYPE_VEC2 },
	{ "COLOR", TYPE_VEC4 },
	{ "VAR1", TYPE_VEC4 },
	{ "VAR2", TYPE_VEC4 },
	{ "POINT_SIZE", TYPE_FLOAT },

	//builtins
	{ "WORLD_MATRIX", TYPE_MAT4 },
	{ "PROJECTION_MATRIX", TYPE_MAT4 },
	{ "EXTRA_MATRIX", TYPE_MAT4 },
	{ "TIME", TYPE_FLOAT },
	{ "AT_LIGHT_PASS", TYPE_BOOL },
	{ NULL, TYPE_VOID },
};
const ShaderLanguage::BuiltinsDef ShaderLanguage::ci_fragment_builtins_defs[] = {

	{ "SRC_COLOR", TYPE_VEC4 },
	{ "POSITION", TYPE_VEC2 },
	{ "NORMAL", TYPE_VEC3 },
	{ "NORMALMAP", TYPE_VEC3 },
	{ "NORMALMAP_DEPTH", TYPE_FLOAT },
	{ "UV", TYPE_VEC2 },
	{ "COLOR", TYPE_VEC4 },
	{ "TEXTURE", TYPE_TEXTURE },
	{ "TEXTURE_PIXEL_SIZE", TYPE_VEC2 },
	{ "VAR1", TYPE_VEC4 },
	{ "VAR2", TYPE_VEC4 },
	{ "SCREEN_UV", TYPE_VEC2 },
	{ "POINT_COORD", TYPE_VEC2 },

	//	{ "SCREEN_POS", TYPE_VEC2},
	//	{ "SCREEN_TEXEL_SIZE", TYPE_VEC2},
	{ "TIME", TYPE_FLOAT },
	{ "AT_LIGHT_PASS", TYPE_BOOL },
	{ NULL, TYPE_VOID }

};

const ShaderLanguage::BuiltinsDef ShaderLanguage::ci_light_builtins_defs[] = {

	{ "POSITION", TYPE_VEC2 },
	{ "NORMAL", TYPE_VEC3 },
	{ "UV", TYPE_VEC2 },
	{ "COLOR", TYPE_VEC4 },
	{ "TEXTURE", TYPE_TEXTURE },
	{ "TEXTURE_PIXEL_SIZE", TYPE_VEC2 },
	{ "VAR1", TYPE_VEC4 },
	{ "VAR2", TYPE_VEC4 },
	{ "SCREEN_UV", TYPE_VEC2 },
	{ "LIGHT_VEC", TYPE_VEC2 },
	{ "LIGHT_HEIGHT", TYPE_FLOAT },
	{ "LIGHT_COLOR", TYPE_VEC4 },
	{ "LIGHT_UV", TYPE_VEC2 },
	{ "LIGHT_SHADOW", TYPE_VEC4 },
	{ "LIGHT", TYPE_VEC4 },
	{ "SHADOW", TYPE_VEC4 },
	{ "POINT_COORD", TYPE_VEC2 },
	//	{ "SCREEN_POS", TYPE_VEC2},
	//	{ "SCREEN_TEXEL_SIZE", TYPE_VEC2},
	{ "TIME", TYPE_FLOAT },
	{ NULL, TYPE_VOID }

};

const ShaderLanguage::BuiltinsDef ShaderLanguage::postprocess_fragment_builtins_defs[] = {

	{ "IN_COLOR", TYPE_VEC3 },
	{ "IN_POSITION", TYPE_VEC3 },
	{ "OUT_COLOR", TYPE_VEC3 },
	{ "SCREEN_POS", TYPE_VEC2 },
	{ "SCREEN_TEXEL_SIZE", TYPE_VEC2 },
	{ "TIME", TYPE_FLOAT },
	{ NULL, TYPE_VOID }
};

ShaderLanguage::DataType ShaderLanguage::compute_node_type(Node *p_node) {

	switch (p_node->type) {

		case Node::TYPE_PROGRAM: ERR_FAIL_V(TYPE_VOID);
		case Node::TYPE_FUNCTION: return static_cast<FunctionNode *>(p_node)->return_type;
		case Node::TYPE_BLOCK: ERR_FAIL_V(TYPE_VOID);
		case Node::TYPE_VARIABLE: return static_cast<VariableNode *>(p_node)->datatype_cache;
		case Node::TYPE_CONSTANT: return static_cast<ConstantNode *>(p_node)->datatype;
		case Node::TYPE_OPERATOR: return static_cast<OperatorNode *>(p_node)->return_cache;
		case Node::TYPE_CONTROL_FLOW: ERR_FAIL_V(TYPE_VOID);
		case Node::TYPE_MEMBER: return static_cast<MemberNode *>(p_node)->datatype;
	}

	return TYPE_VOID;
}

ShaderLanguage::Node *ShaderLanguage::validate_function_call(Parser &parser, OperatorNode *p_func) {

	ERR_FAIL_COND_V(p_func->op != OP_CALL && p_func->op != OP_CONSTRUCT, NULL);

	Vector<DataType> args;

	ERR_FAIL_COND_V(p_func->arguments[0]->type != Node::TYPE_VARIABLE, NULL);

	String name = static_cast<VariableNode *>(p_func->arguments[0])->name.operator String();

	bool all_const = true;
	for (int i = 1; i < p_func->arguments.size(); i++) {
		if (p_func->arguments[i]->type != Node::TYPE_CONSTANT)
			all_const = false;
		args.push_back(compute_node_type(p_func->arguments[i]));
	}

	int argcount = args.size();

	bool found_intrinsic = false;

	if (argcount <= 4) {
		// test intrinsics
		int idx = 0;

		while (intrinsic_func_defs[idx].name) {

			if (name == intrinsic_func_defs[idx].name) {

				bool fail = false;
				for (int i = 0; i < argcount; i++) {

					if (args[i] != intrinsic_func_defs[idx].args[i]) {
						fail = true;
						break;
					}
				}

				if (!fail && argcount < 4 && intrinsic_func_defs[idx].args[argcount] != TYPE_VOID)
					fail = true; //make sure the number of arguments matches

				if (!fail) {
					p_func->return_cache = intrinsic_func_defs[idx].rettype;
					found_intrinsic = true;
					break;
				}
			}

			idx++;
		}
	}

	if (found_intrinsic) {

		if (p_func->op == OP_CONSTRUCT && all_const) {

			Vector<float> cdata;
			for (int i = 0; i < argcount; i++) {

				Variant v = static_cast<ConstantNode *>(p_func->arguments[i + 1])->value;
				switch (v.get_type()) {

					case Variant::REAL: cdata.push_back(v); break;
					case Variant::VECTOR2: {
						Vector2 v2 = v;
						cdata.push_back(v2.x);
						cdata.push_back(v2.y);
					} break;
					case Variant::VECTOR3: {
						Vector3 v3 = v;
						cdata.push_back(v3.x);
						cdata.push_back(v3.y);
						cdata.push_back(v3.z);
					} break;
					case Variant::PLANE: {
						Plane v4 = v;
						cdata.push_back(v4.normal.x);
						cdata.push_back(v4.normal.y);
						cdata.push_back(v4.normal.z);
						cdata.push_back(v4.d);
					} break;
					default: ERR_FAIL_V(NULL);
				}
			}

			ConstantNode *cn = parser.create_node<ConstantNode>(p_func->parent);
			Variant data;
			switch (p_func->return_cache) {
				case TYPE_FLOAT: data = cdata[0]; break;
				case TYPE_VEC2:
					if (cdata.size() == 1)
						data = Vector2(cdata[0], cdata[0]);
					else
						data = Vector2(cdata[0], cdata[1]);

					break;
				case TYPE_VEC3:
					if (cdata.size() == 1)
						data = Vector3(cdata[0], cdata[0], cdata[0]);
					else
						data = Vector3(cdata[0], cdata[1], cdata[2]);
					break;
				case TYPE_VEC4:
					if (cdata.size() == 1)
						data = Plane(cdata[0], cdata[0], cdata[0], cdata[0]);
					else
						data = Plane(cdata[0], cdata[1], cdata[2], cdata[3]);
					break;
			}

			cn->datatype = p_func->return_cache;
			cn->value = data;
			return cn;
		}
		return p_func;
	}

	// try existing functions..

	FunctionNode *exclude_function = NULL; //exclude current function (in case inside one)

	Node *node = p_func;

	while (node->parent) {

		if (node->type == Node::TYPE_FUNCTION) {

			exclude_function = (FunctionNode *)node;
		}

		node = node->parent;
	}

	ERR_FAIL_COND_V(node->type != Node::TYPE_PROGRAM, NULL);
	ProgramNode *program = (ProgramNode *)node;

	for (int i = 0; i < program->functions.size(); i++) {

		if (program->functions[i].function == exclude_function)
			continue;

		FunctionNode *pfunc = program->functions[i].function;

		if (pfunc->arguments.size() != args.size())
			continue;

		bool fail = false;

		for (int i = 0; i < args.size(); i++) {
			if (args[i] != pfunc->arguments[i].type) {
				fail = true;
				break;
			}
		}

		if (!fail && name == program->functions[i].name) {
			p_func->return_cache = pfunc->return_type;
			return p_func;
		}
	}

	return NULL;
}

ShaderLanguage::Node *ShaderLanguage::validate_operator(Parser &parser, OperatorNode *p_func) {

	int argcount = p_func->arguments.size();
	ERR_FAIL_COND_V(argcount > 2, NULL);

	DataType argtype[2] = { TYPE_VOID, TYPE_VOID };
	bool all_const = true;

	for (int i = 0; i < argcount; i++) {

		argtype[i] = compute_node_type(p_func->arguments[i]);
		if (p_func->arguments[i]->type != Node::TYPE_CONSTANT)
			all_const = false;
	}
	int idx = 0;

	bool valid = false;
	while (operator_defs[idx].op != OP_MAX) {

		if (p_func->op == operator_defs[idx].op) {

			if (operator_defs[idx].args[0] == argtype[0] && operator_defs[idx].args[1] == argtype[1]) {

				p_func->return_cache = operator_defs[idx].rettype;
				valid = true;
				break;
			}
		}

		idx++;
	}

	if (!valid)
		return NULL;

#define _RCO2(m_op, m_vop)                                                                                                                                              \
	case m_op: {                                                                                                                                                        \
		ConstantNode *cn = parser.create_node<ConstantNode>(p_func->parent);                                                                                            \
		cn->datatype = p_func->return_cache;                                                                                                                            \
		Variant::evaluate(m_vop, static_cast<ConstantNode *>(p_func->arguments[0])->value, static_cast<ConstantNode *>(p_func->arguments[1])->value, cn->value, valid); \
		if (!valid)                                                                                                                                                     \
			return NULL;                                                                                                                                                \
		return cn;                                                                                                                                                      \
	} break;

#define _RCO1(m_op, m_vop)                                                                                               \
	case m_op: {                                                                                                         \
		ConstantNode *cn = parser.create_node<ConstantNode>(p_func->parent);                                             \
		cn->datatype = p_func->return_cache;                                                                             \
		Variant::evaluate(m_vop, static_cast<ConstantNode *>(p_func->arguments[0])->value, Variant(), cn->value, valid); \
		if (!valid)                                                                                                      \
			return NULL;                                                                                                 \
		return cn;                                                                                                       \
	} break;

	if (all_const) {
		//reduce constant operator
		switch (p_func->op) {
			_RCO2(OP_ADD, Variant::OP_ADD);
			_RCO2(OP_SUB, Variant::OP_SUBSTRACT);
			_RCO2(OP_MUL, Variant::OP_MULTIPLY);
			_RCO2(OP_DIV, Variant::OP_DIVIDE);
			_RCO1(OP_NEG, Variant::OP_NEGATE);
			_RCO1(OP_NOT, Variant::OP_NOT);
			_RCO2(OP_CMP_EQ, Variant::OP_EQUAL);
			_RCO2(OP_CMP_NEQ, Variant::OP_NOT_EQUAL);
			_RCO2(OP_CMP_LEQ, Variant::OP_LESS_EQUAL);
			_RCO2(OP_CMP_GEQ, Variant::OP_GREATER_EQUAL);
			_RCO2(OP_CMP_LESS, Variant::OP_LESS);
			_RCO2(OP_CMP_GREATER, Variant::OP_GREATER);
			_RCO2(OP_CMP_OR, Variant::OP_OR);
			_RCO2(OP_CMP_AND, Variant::OP_AND);
			default: {}
		}
	}

	return p_func;
}

bool ShaderLanguage::is_token_operator(TokenType p_type) {

	return (p_type == TK_OP_EQUAL) ||
		   (p_type == TK_OP_NOT_EQUAL) ||
		   (p_type == TK_OP_LESS) ||
		   (p_type == TK_OP_LESS_EQUAL) ||
		   (p_type == TK_OP_GREATER) ||
		   (p_type == TK_OP_GREATER_EQUAL) ||
		   (p_type == TK_OP_AND) ||
		   (p_type == TK_OP_OR) ||
		   (p_type == TK_OP_NOT) ||
		   (p_type == TK_OP_ADD) ||
		   (p_type == TK_OP_SUB) ||
		   (p_type == TK_OP_MUL) ||
		   (p_type == TK_OP_DIV) ||
		   (p_type == TK_OP_NEG) ||
		   (p_type == TK_OP_ASSIGN) ||
		   (p_type == TK_OP_ASSIGN_ADD) ||
		   (p_type == TK_OP_ASSIGN_SUB) ||
		   (p_type == TK_OP_ASSIGN_MUL) ||
		   (p_type == TK_OP_ASSIGN_DIV);
}
ShaderLanguage::Operator ShaderLanguage::get_token_operator(TokenType p_type) {

	switch (p_type) {
		case TK_OP_EQUAL: return OP_CMP_EQ;
		case TK_OP_NOT_EQUAL: return OP_CMP_NEQ;
		case TK_OP_LESS: return OP_CMP_LESS;
		case TK_OP_LESS_EQUAL: return OP_CMP_LEQ;
		case TK_OP_GREATER: return OP_CMP_GREATER;
		case TK_OP_GREATER_EQUAL: return OP_CMP_GEQ;
		case TK_OP_AND: return OP_CMP_AND;
		case TK_OP_OR: return OP_CMP_OR;
		case TK_OP_NOT: return OP_NOT;
		case TK_OP_ADD: return OP_ADD;
		case TK_OP_SUB: return OP_SUB;
		case TK_OP_MUL: return OP_MUL;
		case TK_OP_DIV: return OP_DIV;
		case TK_OP_NEG: return OP_NEG;
		case TK_OP_ASSIGN: return OP_ASSIGN;
		case TK_OP_ASSIGN_ADD: return OP_ASSIGN_ADD;
		case TK_OP_ASSIGN_SUB: return OP_ASSIGN_SUB;
		case TK_OP_ASSIGN_MUL: return OP_ASSIGN_MUL;
		case TK_OP_ASSIGN_DIV: return OP_ASSIGN_DIV;
		default: ERR_FAIL_V(OP_MAX);
	}

	return OP_MAX;
}

Error ShaderLanguage::parse_expression(Parser &parser, Node *p_parent, Node **r_expr) {

	Vector<Expression> expression;
	//Vector<TokenType> operators;

	while (true) {

		Node *expr = NULL;

		if (parser.get_next_token_type() == TK_PARENTHESIS_OPEN) {
			//handle subexpression
			parser.advance();
			Error err = parse_expression(parser, p_parent, &expr);
			if (err)
				return err;

			if (parser.get_next_token_type() != TK_PARENTHESIS_CLOSE) {

				parser.set_error("Expected ')' in expression");
				return ERR_PARSE_ERROR;
			}

			parser.advance();

		} else if (parser.get_next_token_type() == TK_REAL_CONSTANT) {

			ConstantNode *constant = parser.create_node<ConstantNode>(p_parent);
			constant->value = parser.get_next_token().text.operator String().to_double();
			constant->datatype = TYPE_FLOAT;
			expr = constant;
			parser.advance();
		} else if (parser.get_next_token_type() == TK_TRUE) {
			//print_line("found true");

			//handle true constant
			ConstantNode *constant = parser.create_node<ConstantNode>(p_parent);
			constant->value = true;
			constant->datatype = TYPE_BOOL;
			expr = constant;
			parser.advance();
		} else if (parser.get_next_token_type() == TK_FALSE) {

			//handle false constant
			ConstantNode *constant = parser.create_node<ConstantNode>(p_parent);
			constant->value = false;
			constant->datatype = TYPE_BOOL;
			expr = constant;
			parser.advance();
		} else if (parser.get_next_token_type() == TK_TYPE_VOID) {

			//make sure void is not used in expression
			parser.set_error("Void value not allowed in Expression");
			return ERR_PARSE_ERROR;
		} else if (parser.get_next_token_type(1) == TK_PARENTHESIS_OPEN && (is_token_nonvoid_datatype(parser.get_next_token_type()) || parser.get_next_token_type() == TK_INDENTIFIER)) {

			//function or constructor
			StringName name;
			DataType constructor = TYPE_VOID;
			if (is_token_nonvoid_datatype(parser.get_next_token_type())) {

				constructor = get_token_datatype(parser.get_next_token_type());
				switch (get_token_datatype(parser.get_next_token_type())) {
					case TYPE_BOOL: name = "bool"; break;
					case TYPE_FLOAT: name = "float"; break;
					case TYPE_VEC2: name = "vec2"; break;
					case TYPE_VEC3: name = "vec3"; break;
					case TYPE_VEC4: name = "vec4"; break;
					case TYPE_MAT2: name = "mat2"; break;
					case TYPE_MAT3: name = "mat3"; break;
					case TYPE_MAT4: name = "mat4"; break;
					default: ERR_FAIL_V(ERR_BUG);
				}
			} else {

				name = parser.get_next_token().text;
			}

			if (!test_existing_identifier(p_parent, name)) {

				parser.set_error("Unknown identifier in expression: " + name);
				return ERR_PARSE_ERROR;
			}

			parser.advance(2);

			OperatorNode *func = parser.create_node<OperatorNode>(p_parent);

			func->op = constructor != TYPE_VOID ? OP_CONSTRUCT : OP_CALL;

			VariableNode *funcname = parser.create_node<VariableNode>(func);
			funcname->name = name;
			func->arguments.push_back(funcname);

			//parse parameters

			if (parser.get_next_token_type() == TK_PARENTHESIS_CLOSE) {
				parser.advance();
			} else {

				while (true) {

					Node *arg = NULL;
					Error err = parse_expression(parser, func, &arg);
					if (err)
						return err;
					func->arguments.push_back(arg);

					if (parser.get_next_token_type() == TK_PARENTHESIS_CLOSE) {
						parser.advance();
						break;

					} else if (parser.get_next_token_type() == TK_COMMA) {

						if (parser.get_next_token_type(1) == TK_PARENTHESIS_CLOSE) {

							parser.set_error("Expression expected");
							return ERR_PARSE_ERROR;
						}

						parser.advance();
					} else {
						// something is broken
						parser.set_error("Expected ',' or ')'");
						return ERR_PARSE_ERROR;
					}
				}
			}

			expr = validate_function_call(parser, func);
			if (!expr) {

				parser.set_error("Invalid arguments to function/constructor: " + StringName(name));
				return ERR_PARSE_ERROR;
			}

		} else if (parser.get_next_token_type() == TK_INDENTIFIER) {
			//probably variable

			Node *node = p_parent;
			bool existing = false;
			DataType datatype;
			StringName identifier = parser.get_next_token().text;

			while (node) {

				if (node->type == Node::TYPE_BLOCK) {

					BlockNode *block = (BlockNode *)node;

					if (block->variables.has(identifier)) {
						existing = true;
						datatype = block->variables[identifier];
						break;
					}
				}

				if (node->type == Node::TYPE_FUNCTION) {

					FunctionNode *function = (FunctionNode *)node;
					for (int i = 0; i < function->arguments.size(); i++) {
						if (function->arguments[i].name == identifier) {
							existing = true;
							datatype = function->arguments[i].type;
							break;
						}
					}

					if (existing)
						break;
				}

				if (node->type == Node::TYPE_PROGRAM) {

					ProgramNode *program = (ProgramNode *)node;
					if (program->builtin_variables.has(identifier)) {
						datatype = program->builtin_variables[identifier];
						existing = true;
						break;
					}
					if (program->uniforms.has(identifier)) {
						datatype = program->uniforms[identifier].type;
						existing = true;
						break;
					}
				}

				node = node->parent;
			}

			if (!existing) {

				parser.set_error("Nonexistent identifier in expression: " + identifier);
				return ERR_PARSE_ERROR;
			}

			VariableNode *varname = parser.create_node<VariableNode>(p_parent);
			varname->name = identifier;
			varname->datatype_cache = datatype;
			parser.advance();
			expr = varname;

		} else if (parser.get_next_token_type() == TK_OP_SUB || parser.get_next_token_type() == TK_OP_NOT) {

			//single prefix operators
			TokenType token_type = parser.get_next_token_type();
			parser.advance();
			//Node *subexpr=NULL;
			//Error err = parse_expression(parser,p_parent,&subexpr);
			//if (err)
			//	return err;

			//OperatorNode *op = parser.create_node<OperatorNode>(p_parent);

			Expression e;
			e.is_op = true;

			switch (token_type) {
				case TK_OP_SUB: e.op = TK_OP_NEG; break;
				case TK_OP_NOT:
					e.op = TK_OP_NOT;
					break;
				//case TK_OP_PLUS_PLUS: op->op=OP_PLUS_PLUS; break;
				//case TK_OP_MINUS_MINUS: op->op=OP_MINUS_MINUS; break;
				default: ERR_FAIL_V(ERR_BUG);
			}

			expression.push_back(e);

			continue;

		} else {
			print_line("found bug?");
			print_line("misplaced token: " + String(token_names[parser.get_next_token_type()]));

			parser.set_error("Error parsing expression, misplaced: " + String(token_names[parser.get_next_token_type()]));
			return ERR_PARSE_ERROR;
			//nothing
		}

		ERR_FAIL_COND_V(!expr, ERR_BUG);

		/* OK now see what's NEXT to the operator.. */
		/* OK now see what's NEXT to the operator.. */
		/* OK now see what's NEXT to the operator.. */

		if (parser.get_next_token_type() == TK_PERIOD) {

			if (parser.get_next_token_type(1) != TK_INDENTIFIER) {
				parser.set_error("Expected identifier as member");
				return ERR_PARSE_ERROR;
			}

			DataType dt = compute_node_type(expr);
			String ident = parser.get_next_token(1).text;

			bool ok = true;
			DataType member_type;
			switch (dt) {
				case TYPE_VEC2: {

					int l = ident.length();
					if (l == 1) {
						member_type = TYPE_FLOAT;
					} else if (l == 2) {
						member_type = TYPE_VEC2;
					} else {
						ok = false;
						break;
					}

					const CharType *c = ident.ptr();
					for (int i = 0; i < l; i++) {

						switch (c[i]) {
							case 'r':
							case 'g':
							case 'x':
							case 'y':
								break;
							default:
								ok = false;
								break;
						}
					}

				} break;
				case TYPE_VEC3: {

					int l = ident.length();
					if (l == 1) {
						member_type = TYPE_FLOAT;
					} else if (l == 2) {
						member_type = TYPE_VEC2;
					} else if (l == 3) {
						member_type = TYPE_VEC3;
					} else {
						ok = false;
						break;
					}

					const CharType *c = ident.ptr();
					for (int i = 0; i < l; i++) {

						switch (c[i]) {
							case 'r':
							case 'g':
							case 'b':
							case 'x':
							case 'y':
							case 'z':
								break;
							default:
								ok = false;
								break;
						}
					}

				} break;
				case TYPE_VEC4: {

					int l = ident.length();
					if (l == 1) {
						member_type = TYPE_FLOAT;
					} else if (l == 2) {
						member_type = TYPE_VEC2;
					} else if (l == 3) {
						member_type = TYPE_VEC3;
					} else if (l == 4) {
						member_type = TYPE_VEC4;
					} else {
						ok = false;
						break;
					}

					const CharType *c = ident.ptr();
					for (int i = 0; i < l; i++) {

						switch (c[i]) {
							case 'r':
							case 'g':
							case 'b':
							case 'a':
							case 'x':
							case 'y':
							case 'z':
							case 'w':
								break;
							default:
								ok = false;
								break;
						}
					}

				} break;
				case TYPE_MAT2:
					ok = (ident == "x" || ident == "y");
					member_type = TYPE_VEC2;
					break;
				case TYPE_MAT3:
					ok = (ident == "x" || ident == "y" || ident == "z");
					member_type = TYPE_VEC3;
					break;
				case TYPE_MAT4:
					ok = (ident == "x" || ident == "y" || ident == "z" || ident == "w");
					member_type = TYPE_VEC4;
					break;
				default: {}
			}

			if (!ok) {

				parser.set_error("Invalid member for expression: ." + ident);
				return ERR_PARSE_ERROR;
			}

			MemberNode *mn = parser.create_node<MemberNode>(p_parent);
			mn->basetype = dt;
			mn->datatype = member_type;
			mn->name = ident;
			mn->owner = expr;
			expr = mn;

			parser.advance(2);
			//todo
			//member (period) has priority over any operator
			//creates a subindexing expression in place

		} else if (parser.get_next_token_type() == TK_BRACKET_OPEN) {
			//todo
			//subindexing has priority over any operator
			//creates a subindexing expression in place

		} /*else if (parser.get_next_token_type()==TK_OP_PLUS_PLUS || parser.get_next_token_type()==TK_OP_MINUS_MINUS) {
			//todo
			//inc/dec operators have priority over any operator
			//creates a subindexing expression in place
			//return OK; //wtfs

		} */

		Expression e;
		e.is_op = false;
		e.node = expr;
		expression.push_back(e);

		if (is_token_operator(parser.get_next_token_type())) {

			Expression o;
			o.is_op = true;
			o.op = parser.get_next_token_type();
			expression.push_back(o);
			parser.advance();
		} else {
			break;
		}
	}

	/* Reduce the set set of expressions and place them in an operator tree, respecting precedence */

	while (expression.size() > 1) {

		int next_op = -1;
		int min_priority = 0xFFFFF;
		bool is_unary = false;

		for (int i = 0; i < expression.size(); i++) {

			if (!expression[i].is_op) {

				continue;
			}

			bool unary = false;

			int priority;
			switch (expression[i].op) {

				case TK_OP_NOT:
					priority = 0;
					unary = true;
					break;
				case TK_OP_NEG:
					priority = 0;
					unary = true;
					break;

				case TK_OP_MUL: priority = 1; break;
				case TK_OP_DIV: priority = 1; break;

				case TK_OP_ADD: priority = 2; break;
				case TK_OP_SUB:
					priority = 2;
					break;

				// shift left/right =2

				case TK_OP_LESS: priority = 4; break;
				case TK_OP_LESS_EQUAL: priority = 4; break;
				case TK_OP_GREATER: priority = 4; break;
				case TK_OP_GREATER_EQUAL: priority = 4; break;

				case TK_OP_EQUAL: priority = 5; break;
				case TK_OP_NOT_EQUAL:
					priority = 5;
					break;

				//bit and =5
				//bit xor =6
				//bit or=7

				case TK_OP_AND: priority = 8; break;
				case TK_OP_OR:
					priority = 9;
					break;

				// ?: = 10

				case TK_OP_ASSIGN_ADD: priority = 11; break;
				case TK_OP_ASSIGN_SUB: priority = 11; break;
				case TK_OP_ASSIGN_MUL: priority = 11; break;
				case TK_OP_ASSIGN_DIV: priority = 11; break;
				case TK_OP_ASSIGN: priority = 11; break;

				default:
					ERR_FAIL_V(ERR_BUG); //unexpected operator
			}

			if (priority < min_priority) {
				// < is used for left to right (default)
				// <= is used for right to left
				next_op = i;
				min_priority = priority;
				is_unary = unary;
			}
		}

		ERR_FAIL_COND_V(next_op == -1, ERR_BUG);

		// OK! create operator..
		// OK! create operator..
		if (is_unary) {

			int expr_pos = next_op;
			while (expression[expr_pos].is_op) {

				expr_pos++;
				if (expr_pos == expression.size()) {
					//can happen..
					parser.set_error("Unexpected end of expression..");
					return ERR_BUG;
				}
			}

			//consecutively do unary opeators
			for (int i = expr_pos - 1; i >= next_op; i--) {

				OperatorNode *op = parser.create_node<OperatorNode>(p_parent);
				op->op = get_token_operator(expression[i].op);
				op->arguments.push_back(expression[i + 1].node);

				expression[i].is_op = false;
				expression[i].node = validate_operator(parser, op);
				if (!expression[i].node) {

					String at;
					for (int i = 0; i < op->arguments.size(); i++) {
						if (i > 0)
							at += " and ";
						at += get_datatype_name(compute_node_type(op->arguments[i]));
					}
					parser.set_error("Invalid argument to unary operator " + String(token_names[op->op]) + ": " + at);
					return ERR_PARSE_ERROR;
				}
				expression.remove(i + 1);
			}
		} else {

			if (next_op < 1 || next_op >= (expression.size() - 1)) {
				parser.set_error("Parser bug..");
				ERR_FAIL_V(ERR_BUG);
			}

			OperatorNode *op = parser.create_node<OperatorNode>(p_parent);
			op->op = get_token_operator(expression[next_op].op);

			if (expression[next_op - 1].is_op) {

				parser.set_error("Parser bug..");
				ERR_FAIL_V(ERR_BUG);
			}

			if (expression[next_op + 1].is_op) {
				// this is not invalid and can really appear
				// but it becomes invalid anyway because no binary op
				// can be followed by an unary op in a valid combination,
				// due to how precedence works, unaries will always dissapear first

				parser.set_error("Parser bug..");
			}

			op->arguments.push_back(expression[next_op - 1].node); //expression goes as left
			op->arguments.push_back(expression[next_op + 1].node); //next expression goes as right

			//replace all 3 nodes by this operator and make it an expression
			expression[next_op - 1].node = validate_operator(parser, op);
			if (!expression[next_op - 1].node) {

				String at;
				for (int i = 0; i < op->arguments.size(); i++) {
					if (i > 0)
						at += " and ";
					at += get_datatype_name(compute_node_type(op->arguments[i]));
				}
				static const char *op_names[OP_MAX] = { "=", "+", "-", "*", "/", "+=", "-=", "*=", "/=", "-", "!", "==", "!=", "<=", ">=", "<", ">", "||", "&&", "call", "()" };

				parser.set_error("Invalid arguments to operator " + String(op_names[op->op]) + ": " + at);
				return ERR_PARSE_ERROR;
			}
			expression.remove(next_op);
			expression.remove(next_op);
		}

#if 0
		OperatorNode *op = parser.create_node<OperatorNode>(p_parent);
		op->op=get_token_operator(operators[next_op]);

		op->arguments.push_back(expressions[next_op]); //expression goes as left
		op->arguments.push_back(expressions[next_op+1]); //next expression goes as right

		expressions[next_op]=validate_operator(parser,op);
		if (!expressions[next_op]) {

			String at;
			for(int i=0;i<op->arguments.size();i++) {
				if (i>0)
					at+=" and ";
				at+=get_datatype_name(compute_node_type(op->arguments[i]));

			}
			parser.set_error("Invalid arguments to operator "+String(token_names[operators[next_op]])+": "+at);
			return ERR_PARSE_ERROR;
		}


		expressions.remove(next_op+1);
		operators.remove(next_op);
#endif
	}

	*r_expr = expression[0].node;

	return OK;

	/*
			TokenType token_type=parser.get_next_token_type();
			OperatorNode *op = parser.create_node<OperatorNode>(p_parent);
			op->op=get_token_operator(parser.get_next_token_type());

			op->arguments.push_back(*r_expr); //expression goes as left
			parser.advance();
			Node *right_expr=NULL;
			Error err = parse_expression(parser,p_parent,&right_expr);
			if (err)
				return err;
			op->arguments.push_back(right_expr);

			if (!validate_operator(op)) {

				parser.set_error("Invalid arguments to operator "+String(token_names[token_type]));
				return ERR_PARSE_ERROR;
			}

*/
}

Error ShaderLanguage::parse_variable_declaration(Parser &parser, BlockNode *p_block) {

	bool uniform = parser.get_next_token(-1).type == TK_UNIFORM;

	DataType type = get_token_datatype(parser.get_next_token_type(0));
	bool iscolor = parser.get_next_token_type(0) == TK_TYPE_COLOR;

	if (type == TYPE_VOID) {

		parser.set_error("Cannot Declare a 'void' Variable");
		return ERR_PARSE_ERROR;
	}

	if (type == TYPE_TEXTURE && !uniform) {

		parser.set_error("Cannot Declare a Non-Uniform Texture");
		return ERR_PARSE_ERROR;
	}
	if (type == TYPE_CUBEMAP && !uniform) {

		parser.set_error("Cannot Declare a Non-Uniform Cubemap");
		return ERR_PARSE_ERROR;
	}

	parser.advance();
	int found = 0;

	while (true) {

		if (found && parser.get_next_token_type() != TK_COMMA) {
			break;
		}

		if (parser.get_next_token_type() != TK_INDENTIFIER) {

			parser.set_error("Identifier Expected");
			return ERR_PARSE_ERROR;
		}

		StringName name = parser.get_next_token().text;

		if (test_existing_identifier(p_block, name)) {
			parser.set_error("Duplicate Identifier (existing variable/function): " + name);
			return ERR_PARSE_ERROR;
		}

		found = true;

		parser.advance();
		//see if declaration has an initializer
		if (parser.get_next_token_type() == TK_OP_ASSIGN) {
			parser.advance();
			OperatorNode *op = parser.create_node<OperatorNode>(p_block);
			VariableNode *var = parser.create_node<VariableNode>(op);
			var->name = name;
			var->datatype_cache = type;
			var->uniform = uniform;
			Node *expr;
			Error err = parse_expression(parser, p_block, &expr);

			if (err)
				return err;

			if (var->uniform) {

				if (expr->type != Node::TYPE_CONSTANT) {

					parser.set_error("Uniform can only be initialized to a constant.");
					return ERR_PARSE_ERROR;
				}

				Uniform u;
				u.order = parser.program->uniforms.size();
				u.type = type;
				u.default_value = static_cast<ConstantNode *>(expr)->value;
				if (iscolor && u.default_value.get_type() == Variant::PLANE) {
					Color c;
					Plane p = u.default_value;
					c = Color(p.normal.x, p.normal.y, p.normal.z, p.d);
					u.default_value = c;
				}
				parser.program->uniforms[var->name] = u;
			} else {
				op->op = OP_ASSIGN;
				op->arguments.push_back(var);
				op->arguments.push_back(expr);
				Node *n = validate_operator(parser, op);
				if (!n) {
					parser.set_error("Invalid initializer for variable: " + name);
					return ERR_PARSE_ERROR;
				}
				p_block->statements.push_back(n);
			}

		} else {
			//initialize it EMPTY

			OperatorNode *op = parser.create_node<OperatorNode>(p_block);
			VariableNode *var = parser.create_node<VariableNode>(op);
			ConstantNode *con = parser.create_node<ConstantNode>(op);

			var->name = name;
			var->datatype_cache = type;
			var->uniform = uniform;
			con->datatype = type;

			switch (type) {
				case TYPE_BOOL: con->value = false; break;
				case TYPE_FLOAT: con->value = 0.0; break;
				case TYPE_VEC2: con->value = Vector2(); break;
				case TYPE_VEC3: con->value = Vector3(); break;
				case TYPE_VEC4: con->value = iscolor ? Variant(Color()) : Variant(Plane()); break;
				case TYPE_MAT2: con->value = Matrix32(); break;
				case TYPE_MAT3: con->value = Matrix3(); break;
				case TYPE_MAT4: con->value = Transform(); break;
				case TYPE_TEXTURE:
				case TYPE_CUBEMAP: con->value = RID(); break;
				default: {}
			}

			if (uniform) {
				Uniform u;
				u.type = type;
				u.default_value = con->value;
				u.order = parser.program->uniforms.size();
				parser.program->uniforms[var->name] = u;

			} else {
				op->op = OP_ASSIGN;
				op->arguments.push_back(var);
				op->arguments.push_back(con);
				p_block->statements.push_back(op);
			}
		}

		if (!uniform)
			p_block->variables[name] = type;
	}

	if (parser.get_next_token_type() != TK_SEMICOLON) {
		parser.set_error("Expected ';'");
		return ERR_PARSE_ERROR;
	}

	return OK;
}

Error ShaderLanguage::parse_flow_if(Parser &parser, Node *p_parent, Node **r_statement) {

	ControlFlowNode *cf = parser.create_node<ControlFlowNode>(p_parent);

	cf->flow_op = FLOW_OP_IF;

	parser.advance();

	if (parser.get_next_token_type() != TK_PARENTHESIS_OPEN) {
		parser.set_error("Expected '(' after 'if'");
		return ERR_PARSE_ERROR;
	}
	parser.advance();

	Node *expression = NULL;
	Error err = parse_expression(parser, cf, &expression);
	if (err)
		return err;

	if (compute_node_type(expression) != TYPE_BOOL) {

		parser.set_error("Expression for 'if' is not boolean");
		return ERR_PARSE_ERROR;
	}

	cf->statements.push_back(expression);

	if (parser.get_next_token_type() != TK_PARENTHESIS_CLOSE) {
		parser.set_error("Expected ')' after expression");
		return ERR_PARSE_ERROR;
	}

	parser.advance();

	if (parser.get_next_token_type() != TK_CURLY_BRACKET_OPEN) {
		parser.set_error("Expected statement block after 'if()'");
		return ERR_PARSE_ERROR;
	}

	Node *substatement = NULL;
	err = parse_statement(parser, cf, &substatement);
	if (err)
		return err;

	cf->statements.push_back(substatement);

	if (parser.get_next_token_type() == TK_CF_ELSE) {

		parser.advance();

		if (parser.get_next_token_type() != TK_CURLY_BRACKET_OPEN) {
			parser.set_error("Expected statement block after 'else'");
			return ERR_PARSE_ERROR;
		}

		substatement = NULL;
		err = parse_statement(parser, cf, &substatement);
		if (err)
			return err;

		cf->statements.push_back(substatement);
	}

	*r_statement = cf;

	return OK;
}

Error ShaderLanguage::parse_flow_return(Parser &parser, Node *p_parent, Node **r_statement) {

	FunctionNode *function = NULL;

	Node *parent = p_parent;

	while (parent) {

		if (parent->type == Node::TYPE_FUNCTION) {

			function = (FunctionNode *)parent;
			break;
		}

		parent = parent->parent;
	}

	if (!function) {

		parser.set_error("'return' must be inside a function");
		return ERR_PARSE_ERROR;
	}

	ControlFlowNode *cf = parser.create_node<ControlFlowNode>(p_parent);

	cf->flow_op = FLOW_OP_RETURN;

	parser.advance();

	if (function->return_type != TYPE_VOID) {
		// should expect a return expression.

		Node *expr = NULL;
		Error err = parse_expression(parser, cf, &expr);
		if (err)
			return err;

		if (compute_node_type(expr) != function->return_type) {
			parser.set_error("Invalid type for 'return' expression");
			return ERR_PARSE_ERROR;
		}
		cf->statements.push_back(expr);
	}

	*r_statement = cf;

	if (parser.get_next_token_type() != TK_SEMICOLON) {
		parser.set_error("Expected ';'");
		return ERR_PARSE_ERROR;
	}

	return OK;
}

Error ShaderLanguage::parse_statement(Parser &parser, Node *p_parent, Node **r_statement) {

	*r_statement = NULL;

	TokenType token_type = parser.get_next_token_type();

	if (token_type == TK_CURLY_BRACKET_OPEN) {
		//sub-block
		parser.advance();
		BlockNode *block = parser.create_node<BlockNode>(p_parent);

		*r_statement = block;
		return parse_block(parser, block);
	} else if (token_type == TK_SEMICOLON) {
		// empty ;
		parser.advance();
		return OK;
	} else if (token_type == TK_CF_IF) {
		return parse_flow_if(parser, p_parent, r_statement);

	} else if (token_type == TK_CF_RETURN) {
		return parse_flow_return(parser, p_parent, r_statement);
	} else {
		Error err = parse_expression(parser, p_parent, r_statement);

		if (err)
			return err;

		if (parser.get_next_token_type() != TK_SEMICOLON) {
			parser.set_error("Expected ';'");
			return ERR_PARSE_ERROR;
		}
	}

	return OK;
}

Error ShaderLanguage::parse_block(Parser &parser, BlockNode *p_block) {

	while (true) {

		if (parser.is_at_end()) {
			if (p_block->parent->type != Node::TYPE_PROGRAM) {
				parser.set_error("Unexpected End of File");
				return ERR_PARSE_ERROR;
			}
			return OK; //bye
		}

		TokenType token_type = parser.get_next_token_type();

		if (token_type == TK_CURLY_BRACKET_CLOSE) {
			if (p_block->parent->type == Node::TYPE_PROGRAM) {
				parser.set_error("Unexpected '}'");
				return ERR_PARSE_ERROR;
			}
			parser.advance();
			return OK; // exit block

		} else if (token_type == TK_UNIFORM) {

			if (p_block != parser.program->body) {

				parser.set_error("Uniform only allowed in main program body.");
				return ERR_PARSE_ERROR;
			}
			parser.advance();
			Error err = parse_variable_declaration(parser, p_block);
			if (err)
				return err;

		} else if (is_token_datatype(token_type)) {

			Error err = OK;
			if (parser_is_at_function(parser))
				err = parse_function(parser, p_block);
			else {
				err = parse_variable_declaration(parser, p_block);
			}

			if (err)
				return err;

		} else {
			// must be a statement
			Node *statement = NULL;

			Error err = parse_statement(parser, p_block, &statement);
			if (err)
				return err;
			if (statement) {
				p_block->statements.push_back(statement);
			}
		}
	}

	return OK;
}

Error ShaderLanguage::parse(const Vector<Token> &p_tokens, ShaderType p_type, CompileFunc p_compile_func, void *p_userdata, String *r_error, int *r_err_line, int *r_err_column) {

	Parser parser(p_tokens);
	parser.program = parser.create_node<ProgramNode>(NULL);
	parser.program->body = parser.create_node<BlockNode>(parser.program);

	//add builtins
	switch (p_type) {
		case SHADER_MATERIAL_VERTEX: {
			int idx = 0;
			while (vertex_builtins_defs[idx].name) {
				parser.program->builtin_variables[vertex_builtins_defs[idx].name] = vertex_builtins_defs[idx].type;
				idx++;
			}
		} break;
		case SHADER_MATERIAL_FRAGMENT: {
			int idx = 0;
			while (fragment_builtins_defs[idx].name) {
				parser.program->builtin_variables[fragment_builtins_defs[idx].name] = fragment_builtins_defs[idx].type;
				idx++;
			}
		} break;
		case SHADER_MATERIAL_LIGHT: {
			int idx = 0;
			while (light_builtins_defs[idx].name) {
				parser.program->builtin_variables[light_builtins_defs[idx].name] = light_builtins_defs[idx].type;
				idx++;
			}
		} break;
		case SHADER_CANVAS_ITEM_VERTEX: {
			int idx = 0;
			while (ci_vertex_builtins_defs[idx].name) {
				parser.program->builtin_variables[ci_vertex_builtins_defs[idx].name] = ci_vertex_builtins_defs[idx].type;
				idx++;
			}
		} break;
		case SHADER_CANVAS_ITEM_FRAGMENT: {
			int idx = 0;
			while (ci_fragment_builtins_defs[idx].name) {
				parser.program->builtin_variables[ci_fragment_builtins_defs[idx].name] = ci_fragment_builtins_defs[idx].type;
				idx++;
			}
		} break;
		case SHADER_CANVAS_ITEM_LIGHT: {
			int idx = 0;
			while (ci_light_builtins_defs[idx].name) {
				parser.program->builtin_variables[ci_light_builtins_defs[idx].name] = ci_light_builtins_defs[idx].type;
				idx++;
			}
		} break;
		case SHADER_POST_PROCESS: {
			int idx = 0;
			while (postprocess_fragment_builtins_defs[idx].name) {
				parser.program->builtin_variables[postprocess_fragment_builtins_defs[idx].name] = postprocess_fragment_builtins_defs[idx].type;
				idx++;
			}
		} break;
	}

	Error err = parse_block(parser, parser.program->body);
	if (err) {
		parser.get_error(r_error, r_err_line, r_err_column);
		return err;
	}

	if (p_compile_func) {
		err = p_compile_func(p_userdata, parser.program);
	}

	//clean up nodes created
	while (parser.nodegc.size()) {

		memdelete(parser.nodegc.front()->get());
		parser.nodegc.pop_front();
	}
	return err;
}

Error ShaderLanguage::compile(const String &p_code, ShaderType p_type, CompileFunc p_compile_func, void *p_userdata, String *r_error, int *r_err_line, int *r_err_column) {

	*r_error = "";
	*r_err_line = 0;
	*r_err_column = 0;
	Vector<Token> tokens;

	Error err = tokenize(p_code, &tokens, r_error, r_err_line, r_err_column);
	if (err != OK) {
		print_line("tokenizer error!");
	}

	if (err != OK) {
		return err;
	}
	err = parse(tokens, p_type, p_compile_func, p_userdata, r_error, r_err_line, r_err_column);
	if (err != OK) {
		return err;
	}
	return OK;
}

void ShaderLanguage::get_keyword_list(ShaderType p_type, List<String> *p_keywords) {

	int idx = 0;

	p_keywords->push_back("uniform");
	p_keywords->push_back("texture");
	p_keywords->push_back("cubemap");
	p_keywords->push_back("color");
	p_keywords->push_back("if");
	p_keywords->push_back("else");

	while (intrinsic_func_defs[idx].name) {

		p_keywords->push_back(intrinsic_func_defs[idx].name);
		idx++;
	}

	switch (p_type) {
		case SHADER_MATERIAL_VERTEX: {
			idx = 0;
			while (vertex_builtins_defs[idx].name) {
				p_keywords->push_back(vertex_builtins_defs[idx].name);
				idx++;
			}
		} break;
		case SHADER_MATERIAL_FRAGMENT: {
			idx = 0;
			while (fragment_builtins_defs[idx].name) {
				p_keywords->push_back(fragment_builtins_defs[idx].name);
				idx++;
			}
		} break;
		case SHADER_MATERIAL_LIGHT: {
			idx = 0;
			while (light_builtins_defs[idx].name) {
				p_keywords->push_back(light_builtins_defs[idx].name);
				idx++;
			}
		} break;
		case SHADER_CANVAS_ITEM_VERTEX: {
			idx = 0;
			while (ci_vertex_builtins_defs[idx].name) {
				p_keywords->push_back(ci_vertex_builtins_defs[idx].name);
				idx++;
			}
		} break;
		case SHADER_CANVAS_ITEM_FRAGMENT: {
			idx = 0;
			while (ci_fragment_builtins_defs[idx].name) {
				p_keywords->push_back(ci_fragment_builtins_defs[idx].name);
				idx++;
			}
		} break;
		case SHADER_CANVAS_ITEM_LIGHT: {
			idx = 0;
			while (ci_light_builtins_defs[idx].name) {
				p_keywords->push_back(ci_light_builtins_defs[idx].name);
				idx++;
			}
		} break;

		case SHADER_POST_PROCESS: {
			idx = 0;
			while (postprocess_fragment_builtins_defs[idx].name) {
				p_keywords->push_back(postprocess_fragment_builtins_defs[idx].name);
				idx++;
			}
		} break;
	}
}
