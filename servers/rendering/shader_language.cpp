/*************************************************************************/
/*  shader_language.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "servers/rendering_server.h"

#define HAS_WARNING(flag) (warning_flags & flag)

static bool _is_text_char(char32_t c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

static bool _is_number(char32_t c) {
	return (c >= '0' && c <= '9');
}

static bool _is_hex(char32_t c) {
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

String ShaderLanguage::get_operator_text(Operator p_op) {
	static const char *op_names[OP_MAX] = { "==",
		"!=",
		"<",
		"<=",
		">",
		">=",
		"&&",
		"||",
		"!",
		"-",
		"+",
		"-",
		"*",
		"/",
		"%",
		"<<",
		">>",
		"=",
		"+=",
		"-=",
		"*=",
		"/=",
		"%=",
		"<<=",
		">>=",
		"&=",
		"|=",
		"^=",
		"&",
		"|",
		"^",
		"~",
		"++",
		"--",
		"?",
		":",
		"++",
		"--",
		"()",
		"construct",
		"index" };

	return op_names[p_op];
}

const char *ShaderLanguage::token_names[TK_MAX] = {
	"EMPTY",
	"IDENTIFIER",
	"TRUE",
	"FALSE",
	"REAL_CONSTANT",
	"INT_CONSTANT",
	"TYPE_VOID",
	"TYPE_BOOL",
	"TYPE_BVEC2",
	"TYPE_BVEC3",
	"TYPE_BVEC4",
	"TYPE_INT",
	"TYPE_IVEC2",
	"TYPE_IVEC3",
	"TYPE_IVEC4",
	"TYPE_UINT",
	"TYPE_UVEC2",
	"TYPE_UVEC3",
	"TYPE_UVEC4",
	"TYPE_FLOAT",
	"TYPE_VEC2",
	"TYPE_VEC3",
	"TYPE_VEC4",
	"TYPE_MAT2",
	"TYPE_MAT3",
	"TYPE_MAT4",
	"TYPE_SAMPLER2D",
	"TYPE_ISAMPLER2D",
	"TYPE_USAMPLER2D",
	"TYPE_SAMPLER2DARRAY",
	"TYPE_ISAMPLER2DARRAY",
	"TYPE_USAMPLER2DARRAY",
	"TYPE_SAMPLER3D",
	"TYPE_ISAMPLER3D",
	"TYPE_USAMPLER3D",
	"TYPE_SAMPLERCUBE",
	"TYPE_SAMPLERCUBEARRAY",
	"INTERPOLATION_FLAT",
	"INTERPOLATION_SMOOTH",
	"CONST",
	"PRECISION_LOW",
	"PRECISION_MID",
	"PRECISION_HIGH",
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
	"OP_MOD",
	"OP_SHIFT_LEFT",
	"OP_SHIFT_RIGHT",
	"OP_ASSIGN",
	"OP_ASSIGN_ADD",
	"OP_ASSIGN_SUB",
	"OP_ASSIGN_MUL",
	"OP_ASSIGN_DIV",
	"OP_ASSIGN_MOD",
	"OP_ASSIGN_SHIFT_LEFT",
	"OP_ASSIGN_SHIFT_RIGHT",
	"OP_ASSIGN_BIT_AND",
	"OP_ASSIGN_BIT_OR",
	"OP_ASSIGN_BIT_XOR",
	"OP_BIT_AND",
	"OP_BIT_OR",
	"OP_BIT_XOR",
	"OP_BIT_INVERT",
	"OP_INCREMENT",
	"OP_DECREMENT",
	"CF_IF",
	"CF_ELSE",
	"CF_FOR",
	"CF_WHILE",
	"CF_DO",
	"CF_SWITCH",
	"CF_CASE",
	"CF_BREAK",
	"CF_CONTINUE",
	"CF_RETURN",
	"CF_DISCARD",
	"BRACKET_OPEN",
	"BRACKET_CLOSE",
	"CURLY_BRACKET_OPEN",
	"CURLY_BRACKET_CLOSE",
	"PARENTHESIS_OPEN",
	"PARENTHESIS_CLOSE",
	"QUESTION",
	"COMMA",
	"COLON",
	"SEMICOLON",
	"PERIOD",
	"UNIFORM",
	"INSTANCE",
	"GLOBAL",
	"VARYING",
	"IN",
	"OUT",
	"INOUT",
	"RENDER_MODE",
	"HINT_WHITE_TEXTURE",
	"HINT_BLACK_TEXTURE",
	"HINT_NORMAL_TEXTURE",
	"HINT_ANISO_TEXTURE",
	"HINT_ALBEDO_TEXTURE",
	"HINT_BLACK_ALBEDO_TEXTURE",
	"HINT_COLOR",
	"HINT_RANGE",
	"HINT_INSTANCE_INDEX",
	"FILTER_NEAREST",
	"FILTER_LINEAR",
	"FILTER_NEAREST_MIPMAP",
	"FILTER_LINEAR_MIPMAP",
	"FILTER_NEAREST_MIPMAP_ANISO",
	"FILTER_LINEAR_MIPMAP_ANISO",
	"REPEAT_ENABLE",
	"REPEAT_DISABLE",
	"SHADER_TYPE",
	"CURSOR",
	"ERROR",
	"EOF",
};

String ShaderLanguage::get_token_text(Token p_token) {
	String name = token_names[p_token.type];
	if (p_token.type == TK_INT_CONSTANT || p_token.type == TK_FLOAT_CONSTANT) {
		name += "(" + rtos(p_token.constant) + ")";
	} else if (p_token.type == TK_IDENTIFIER) {
		name += "(" + String(p_token.text) + ")";
	} else if (p_token.type == TK_ERROR) {
		name += "(" + String(p_token.text) + ")";
	}

	return name;
}

ShaderLanguage::Token ShaderLanguage::_make_token(TokenType p_type, const StringName &p_text) {
	Token tk;
	tk.type = p_type;
	tk.text = p_text;
	tk.line = tk_line;
	if (tk.type == TK_ERROR) {
		_set_error(p_text);
	}
	return tk;
}

const ShaderLanguage::KeyWord ShaderLanguage::keyword_list[] = {
	{ TK_TRUE, "true" },
	{ TK_FALSE, "false" },
	{ TK_TYPE_VOID, "void" },
	{ TK_TYPE_BOOL, "bool" },
	{ TK_TYPE_BVEC2, "bvec2" },
	{ TK_TYPE_BVEC3, "bvec3" },
	{ TK_TYPE_BVEC4, "bvec4" },
	{ TK_TYPE_INT, "int" },
	{ TK_TYPE_IVEC2, "ivec2" },
	{ TK_TYPE_IVEC3, "ivec3" },
	{ TK_TYPE_IVEC4, "ivec4" },
	{ TK_TYPE_UINT, "uint" },
	{ TK_TYPE_UVEC2, "uvec2" },
	{ TK_TYPE_UVEC3, "uvec3" },
	{ TK_TYPE_UVEC4, "uvec4" },
	{ TK_TYPE_FLOAT, "float" },
	{ TK_TYPE_VEC2, "vec2" },
	{ TK_TYPE_VEC3, "vec3" },
	{ TK_TYPE_VEC4, "vec4" },
	{ TK_TYPE_MAT2, "mat2" },
	{ TK_TYPE_MAT3, "mat3" },
	{ TK_TYPE_MAT4, "mat4" },
	{ TK_TYPE_SAMPLER2D, "sampler2D" },
	{ TK_TYPE_ISAMPLER2D, "isampler2D" },
	{ TK_TYPE_USAMPLER2D, "usampler2D" },
	{ TK_TYPE_SAMPLER2DARRAY, "sampler2DArray" },
	{ TK_TYPE_ISAMPLER2DARRAY, "isampler2DArray" },
	{ TK_TYPE_USAMPLER2DARRAY, "usampler2DArray" },
	{ TK_TYPE_SAMPLER3D, "sampler3D" },
	{ TK_TYPE_ISAMPLER3D, "isampler3D" },
	{ TK_TYPE_USAMPLER3D, "usampler3D" },
	{ TK_TYPE_SAMPLERCUBE, "samplerCube" },
	{ TK_TYPE_SAMPLERCUBEARRAY, "samplerCubeArray" },
	{ TK_INTERPOLATION_FLAT, "flat" },
	{ TK_INTERPOLATION_SMOOTH, "smooth" },
	{ TK_CONST, "const" },
	{ TK_STRUCT, "struct" },
	{ TK_PRECISION_LOW, "lowp" },
	{ TK_PRECISION_MID, "mediump" },
	{ TK_PRECISION_HIGH, "highp" },
	{ TK_CF_IF, "if" },
	{ TK_CF_ELSE, "else" },
	{ TK_CF_FOR, "for" },
	{ TK_CF_WHILE, "while" },
	{ TK_CF_DO, "do" },
	{ TK_CF_SWITCH, "switch" },
	{ TK_CF_CASE, "case" },
	{ TK_CF_DEFAULT, "default" },
	{ TK_CF_BREAK, "break" },
	{ TK_CF_CONTINUE, "continue" },
	{ TK_CF_RETURN, "return" },
	{ TK_CF_DISCARD, "discard" },
	{ TK_UNIFORM, "uniform" },
	{ TK_INSTANCE, "instance" },
	{ TK_GLOBAL, "global" },
	{ TK_VARYING, "varying" },
	{ TK_ARG_IN, "in" },
	{ TK_ARG_OUT, "out" },
	{ TK_ARG_INOUT, "inout" },
	{ TK_RENDER_MODE, "render_mode" },
	{ TK_HINT_WHITE_TEXTURE, "hint_white" },
	{ TK_HINT_BLACK_TEXTURE, "hint_black" },
	{ TK_HINT_NORMAL_TEXTURE, "hint_normal" },
	{ TK_HINT_ROUGHNESS_NORMAL_TEXTURE, "hint_roughness_normal" },
	{ TK_HINT_ROUGHNESS_R, "hint_roughness_r" },
	{ TK_HINT_ROUGHNESS_G, "hint_roughness_g" },
	{ TK_HINT_ROUGHNESS_B, "hint_roughness_b" },
	{ TK_HINT_ROUGHNESS_A, "hint_roughness_a" },
	{ TK_HINT_ROUGHNESS_GRAY, "hint_roughness_gray" },
	{ TK_HINT_ANISO_TEXTURE, "hint_aniso" },
	{ TK_HINT_ALBEDO_TEXTURE, "hint_albedo" },
	{ TK_HINT_BLACK_ALBEDO_TEXTURE, "hint_black_albedo" },
	{ TK_HINT_COLOR, "hint_color" },
	{ TK_HINT_RANGE, "hint_range" },
	{ TK_HINT_INSTANCE_INDEX, "instance_index" },
	{ TK_FILTER_NEAREST, "filter_nearest" },
	{ TK_FILTER_LINEAR, "filter_linear" },
	{ TK_FILTER_NEAREST_MIPMAP, "filter_nearest_mipmap" },
	{ TK_FILTER_LINEAR_MIPMAP, "filter_linear_mipmap" },
	{ TK_FILTER_NEAREST_MIPMAP_ANISO, "filter_nearest_mipmap_aniso" },
	{ TK_FILTER_LINEAR_MIPMAP_ANISO, "filter_linear_mipmap_aniso" },
	{ TK_REPEAT_ENABLE, "repeat_enable" },
	{ TK_REPEAT_DISABLE, "repeat_disable" },
	{ TK_SHADER_TYPE, "shader_type" },
	{ TK_ERROR, nullptr }
};

ShaderLanguage::Token ShaderLanguage::_get_token() {
#define GETCHAR(m_idx) (((char_idx + m_idx) < code.length()) ? code[char_idx + m_idx] : char32_t(0))

	while (true) {
		char_idx++;
		switch (GETCHAR(-1)) {
			case 0:
				return _make_token(TK_EOF);
			case 0xFFFF:
				return _make_token(TK_CURSOR); //for completion
			case '\t':
			case '\r':
			case ' ':
				continue;
			case '\n':
				tk_line++;
				continue;
			case '/': {
				switch (GETCHAR(0)) {
					case '*': { // block comment

						char_idx++;
						while (true) {
							if (GETCHAR(0) == 0) {
								return _make_token(TK_EOF);
							}
							if (GETCHAR(0) == '*' && GETCHAR(1) == '/') {
								char_idx += 2;
								break;
							} else if (GETCHAR(0) == '\n') {
								tk_line++;
							}

							char_idx++;
						}

					} break;
					case '/': { // line comment skip

						while (true) {
							if (GETCHAR(0) == '\n') {
								tk_line++;
								char_idx++;
								break;
							}
							if (GETCHAR(0) == 0) {
								return _make_token(TK_EOF);
							}
							char_idx++;
						}

					} break;
					case '=': { // diveq

						char_idx++;
						return _make_token(TK_OP_ASSIGN_DIV);

					} break;
					default:
						return _make_token(TK_OP_DIV);
				}

				continue; //a comment, continue to next token
			} break;
			case '=': {
				if (GETCHAR(0) == '=') {
					char_idx++;
					return _make_token(TK_OP_EQUAL);
				}

				return _make_token(TK_OP_ASSIGN);

			} break;
			case '<': {
				if (GETCHAR(0) == '=') {
					char_idx++;
					return _make_token(TK_OP_LESS_EQUAL);
				} else if (GETCHAR(0) == '<') {
					char_idx++;
					if (GETCHAR(0) == '=') {
						char_idx++;
						return _make_token(TK_OP_ASSIGN_SHIFT_LEFT);
					}

					return _make_token(TK_OP_SHIFT_LEFT);
				}

				return _make_token(TK_OP_LESS);

			} break;
			case '>': {
				if (GETCHAR(0) == '=') {
					char_idx++;
					return _make_token(TK_OP_GREATER_EQUAL);
				} else if (GETCHAR(0) == '>') {
					char_idx++;
					if (GETCHAR(0) == '=') {
						char_idx++;
						return _make_token(TK_OP_ASSIGN_SHIFT_RIGHT);
					}

					return _make_token(TK_OP_SHIFT_RIGHT);
				}

				return _make_token(TK_OP_GREATER);

			} break;
			case '!': {
				if (GETCHAR(0) == '=') {
					char_idx++;
					return _make_token(TK_OP_NOT_EQUAL);
				}

				return _make_token(TK_OP_NOT);

			} break;
			//case '"' //string - no strings in shader
			//case '\'' //string - no strings in shader
			case '{':
				return _make_token(TK_CURLY_BRACKET_OPEN);
			case '}':
				return _make_token(TK_CURLY_BRACKET_CLOSE);
			case '[':
				return _make_token(TK_BRACKET_OPEN);
			case ']':
				return _make_token(TK_BRACKET_CLOSE);
			case '(':
				return _make_token(TK_PARENTHESIS_OPEN);
			case ')':
				return _make_token(TK_PARENTHESIS_CLOSE);
			case ',':
				return _make_token(TK_COMMA);
			case ';':
				return _make_token(TK_SEMICOLON);
			case '?':
				return _make_token(TK_QUESTION);
			case ':':
				return _make_token(TK_COLON);
			case '^':
				return _make_token(TK_OP_BIT_XOR);
			case '~':
				return _make_token(TK_OP_BIT_INVERT);
			case '&': {
				if (GETCHAR(0) == '=') {
					char_idx++;
					return _make_token(TK_OP_ASSIGN_BIT_AND);
				} else if (GETCHAR(0) == '&') {
					char_idx++;
					return _make_token(TK_OP_AND);
				}
				return _make_token(TK_OP_BIT_AND);
			} break;
			case '|': {
				if (GETCHAR(0) == '=') {
					char_idx++;
					return _make_token(TK_OP_ASSIGN_BIT_OR);
				} else if (GETCHAR(0) == '|') {
					char_idx++;
					return _make_token(TK_OP_OR);
				}
				return _make_token(TK_OP_BIT_OR);

			} break;
			case '*': {
				if (GETCHAR(0) == '=') {
					char_idx++;
					return _make_token(TK_OP_ASSIGN_MUL);
				}
				return _make_token(TK_OP_MUL);
			} break;
			case '+': {
				if (GETCHAR(0) == '=') {
					char_idx++;
					return _make_token(TK_OP_ASSIGN_ADD);
				} else if (GETCHAR(0) == '+') {
					char_idx++;
					return _make_token(TK_OP_INCREMENT);
				}

				return _make_token(TK_OP_ADD);
			} break;
			case '-': {
				if (GETCHAR(0) == '=') {
					char_idx++;
					return _make_token(TK_OP_ASSIGN_SUB);
				} else if (GETCHAR(0) == '-') {
					char_idx++;
					return _make_token(TK_OP_DECREMENT);
				}

				return _make_token(TK_OP_SUB);
			} break;
			case '%': {
				if (GETCHAR(0) == '=') {
					char_idx++;
					return _make_token(TK_OP_ASSIGN_MOD);
				}

				return _make_token(TK_OP_MOD);
			} break;
			default: {
				char_idx--; //go back one, since we have no idea what this is

				if (_is_number(GETCHAR(0)) || (GETCHAR(0) == '.' && _is_number(GETCHAR(1)))) {
					// parse number
					bool period_found = false;
					bool exponent_found = false;
					bool hexa_found = false;
					bool sign_found = false;
					bool float_suffix_found = false;

					String str;
					int i = 0;

					while (true) {
						if (GETCHAR(i) == '.') {
							if (period_found || exponent_found || hexa_found || float_suffix_found) {
								return _make_token(TK_ERROR, "Invalid numeric constant");
							}
							period_found = true;
						} else if (GETCHAR(i) == 'x') {
							if (hexa_found || str.length() != 1 || str[0] != '0') {
								return _make_token(TK_ERROR, "Invalid numeric constant");
							}
							hexa_found = true;
						} else if (GETCHAR(i) == 'e' && !hexa_found) {
							if (exponent_found || float_suffix_found) {
								return _make_token(TK_ERROR, "Invalid numeric constant");
							}
							exponent_found = true;
						} else if (GETCHAR(i) == 'f' && !hexa_found) {
							if (exponent_found) {
								return _make_token(TK_ERROR, "Invalid numeric constant");
							}
							float_suffix_found = true;
						} else if (_is_number(GETCHAR(i))) {
							if (float_suffix_found) {
								return _make_token(TK_ERROR, "Invalid numeric constant");
							}
						} else if (hexa_found && _is_hex(GETCHAR(i))) {
						} else if ((GETCHAR(i) == '-' || GETCHAR(i) == '+') && exponent_found) {
							if (sign_found) {
								return _make_token(TK_ERROR, "Invalid numeric constant");
							}
							sign_found = true;
						} else {
							break;
						}

						str += char32_t(GETCHAR(i));
						i++;
					}

					char32_t last_char = str[str.length() - 1];

					if (hexa_found) {
						//integer(hex)
						if (str.size() > 11 || !str.is_valid_hex_number(true)) { // > 0xFFFFFFFF
							return _make_token(TK_ERROR, "Invalid (hexadecimal) numeric constant");
						}
					} else if (period_found || exponent_found || float_suffix_found) {
						//floats
						if (period_found) {
							if (float_suffix_found) {
								//checks for eg "1.f" or "1.99f" notations
								if (last_char != 'f') {
									return _make_token(TK_ERROR, "Invalid (float) numeric constant");
								}
							} else {
								//checks for eg. "1." or "1.99" notations
								if (last_char != '.' && !_is_number(last_char)) {
									return _make_token(TK_ERROR, "Invalid (float) numeric constant");
								}
							}
						} else if (float_suffix_found) {
							// if no period found the float suffix must be the last character, like in "2f" for "2.0"
							if (last_char != 'f') {
								return _make_token(TK_ERROR, "Invalid (float) numeric constant");
							}
						}

						if (float_suffix_found) {
							//strip the suffix
							str = str.left(str.length() - 1);
							//compensate reading cursor position
							char_idx += 1;
						}

						if (!str.is_valid_float()) {
							return _make_token(TK_ERROR, "Invalid (float) numeric constant");
						}
					} else {
						//integers
						if (!_is_number(last_char)) {
							return _make_token(TK_ERROR, "Invalid (integer) numeric constant");
						}
						if (!str.is_valid_int()) {
							return _make_token(TK_ERROR, "Invalid numeric constant");
						}
					}

					char_idx += str.length();
					Token tk;
					if (period_found || exponent_found || float_suffix_found) {
						tk.type = TK_FLOAT_CONSTANT;
					} else {
						tk.type = TK_INT_CONSTANT;
					}

					if (hexa_found) {
						tk.constant = (double)str.hex_to_int();
					} else {
						tk.constant = str.to_float();
					}
					tk.line = tk_line;

					return tk;
				}

				if (GETCHAR(0) == '.') {
					//parse period
					char_idx++;
					return _make_token(TK_PERIOD);
				}

				if (_is_text_char(GETCHAR(0))) {
					// parse identifier
					String str;

					while (_is_text_char(GETCHAR(0))) {
						str += char32_t(GETCHAR(0));
						char_idx++;
					}

					//see if keyword
					//should be converted to a static map
					int idx = 0;

					while (keyword_list[idx].text) {
						if (str == keyword_list[idx].text) {
							return _make_token(keyword_list[idx].token);
						}
						idx++;
					}

					str = str.replace("dus_", "_");

					return _make_token(TK_IDENTIFIER, str);
				}

				if (GETCHAR(0) > 32) {
					return _make_token(TK_ERROR, "Tokenizer: Unknown character #" + itos(GETCHAR(0)) + ": '" + String::chr(GETCHAR(0)) + "'");
				} else {
					return _make_token(TK_ERROR, "Tokenizer: Unknown character #" + itos(GETCHAR(0)));
				}

			} break;
		}
	}
	ERR_PRINT("BUG");
	return Token();

#undef GETCHAR
}

String ShaderLanguage::token_debug(const String &p_code) {
	clear();

	code = p_code;

	String output;

	Token tk = _get_token();
	while (tk.type != TK_EOF && tk.type != TK_ERROR) {
		output += itos(tk_line) + ": " + get_token_text(tk) + "\n";
		tk = _get_token();
	}

	return output;
}

bool ShaderLanguage::is_token_variable_datatype(TokenType p_type) {
	return (
			p_type == TK_TYPE_VOID ||
			p_type == TK_TYPE_BOOL ||
			p_type == TK_TYPE_BVEC2 ||
			p_type == TK_TYPE_BVEC3 ||
			p_type == TK_TYPE_BVEC4 ||
			p_type == TK_TYPE_INT ||
			p_type == TK_TYPE_IVEC2 ||
			p_type == TK_TYPE_IVEC3 ||
			p_type == TK_TYPE_IVEC4 ||
			p_type == TK_TYPE_UINT ||
			p_type == TK_TYPE_UVEC2 ||
			p_type == TK_TYPE_UVEC3 ||
			p_type == TK_TYPE_UVEC4 ||
			p_type == TK_TYPE_FLOAT ||
			p_type == TK_TYPE_VEC2 ||
			p_type == TK_TYPE_VEC3 ||
			p_type == TK_TYPE_VEC4 ||
			p_type == TK_TYPE_MAT2 ||
			p_type == TK_TYPE_MAT3 ||
			p_type == TK_TYPE_MAT4);
}

bool ShaderLanguage::is_token_datatype(TokenType p_type) {
	return (
			p_type == TK_TYPE_VOID ||
			p_type == TK_TYPE_BOOL ||
			p_type == TK_TYPE_BVEC2 ||
			p_type == TK_TYPE_BVEC3 ||
			p_type == TK_TYPE_BVEC4 ||
			p_type == TK_TYPE_INT ||
			p_type == TK_TYPE_IVEC2 ||
			p_type == TK_TYPE_IVEC3 ||
			p_type == TK_TYPE_IVEC4 ||
			p_type == TK_TYPE_UINT ||
			p_type == TK_TYPE_UVEC2 ||
			p_type == TK_TYPE_UVEC3 ||
			p_type == TK_TYPE_UVEC4 ||
			p_type == TK_TYPE_FLOAT ||
			p_type == TK_TYPE_VEC2 ||
			p_type == TK_TYPE_VEC3 ||
			p_type == TK_TYPE_VEC4 ||
			p_type == TK_TYPE_MAT2 ||
			p_type == TK_TYPE_MAT3 ||
			p_type == TK_TYPE_MAT4 ||
			p_type == TK_TYPE_SAMPLER2D ||
			p_type == TK_TYPE_ISAMPLER2D ||
			p_type == TK_TYPE_USAMPLER2D ||
			p_type == TK_TYPE_SAMPLER2DARRAY ||
			p_type == TK_TYPE_ISAMPLER2DARRAY ||
			p_type == TK_TYPE_USAMPLER2DARRAY ||
			p_type == TK_TYPE_SAMPLER3D ||
			p_type == TK_TYPE_ISAMPLER3D ||
			p_type == TK_TYPE_USAMPLER3D ||
			p_type == TK_TYPE_SAMPLERCUBE ||
			p_type == TK_TYPE_SAMPLERCUBEARRAY);
}

ShaderLanguage::DataType ShaderLanguage::get_token_datatype(TokenType p_type) {
	return DataType(p_type - TK_TYPE_VOID);
}

bool ShaderLanguage::is_token_interpolation(TokenType p_type) {
	return (
			p_type == TK_INTERPOLATION_FLAT ||
			p_type == TK_INTERPOLATION_SMOOTH);
}

ShaderLanguage::DataInterpolation ShaderLanguage::get_token_interpolation(TokenType p_type) {
	if (p_type == TK_INTERPOLATION_FLAT) {
		return INTERPOLATION_FLAT;
	} else {
		return INTERPOLATION_SMOOTH;
	}
}

bool ShaderLanguage::is_token_precision(TokenType p_type) {
	return (
			p_type == TK_PRECISION_LOW ||
			p_type == TK_PRECISION_MID ||
			p_type == TK_PRECISION_HIGH);
}

ShaderLanguage::DataPrecision ShaderLanguage::get_token_precision(TokenType p_type) {
	if (p_type == TK_PRECISION_LOW) {
		return PRECISION_LOWP;
	} else if (p_type == TK_PRECISION_HIGH) {
		return PRECISION_HIGHP;
	} else {
		return PRECISION_MEDIUMP;
	}
}

String ShaderLanguage::get_precision_name(DataPrecision p_type) {
	switch (p_type) {
		case PRECISION_LOWP:
			return "lowp";
		case PRECISION_MEDIUMP:
			return "mediump";
		case PRECISION_HIGHP:
			return "highp";
		default:
			break;
	}
	return "";
}

String ShaderLanguage::get_datatype_name(DataType p_type) {
	switch (p_type) {
		case TYPE_VOID:
			return "void";
		case TYPE_BOOL:
			return "bool";
		case TYPE_BVEC2:
			return "bvec2";
		case TYPE_BVEC3:
			return "bvec3";
		case TYPE_BVEC4:
			return "bvec4";
		case TYPE_INT:
			return "int";
		case TYPE_IVEC2:
			return "ivec2";
		case TYPE_IVEC3:
			return "ivec3";
		case TYPE_IVEC4:
			return "ivec4";
		case TYPE_UINT:
			return "uint";
		case TYPE_UVEC2:
			return "uvec2";
		case TYPE_UVEC3:
			return "uvec3";
		case TYPE_UVEC4:
			return "uvec4";
		case TYPE_FLOAT:
			return "float";
		case TYPE_VEC2:
			return "vec2";
		case TYPE_VEC3:
			return "vec3";
		case TYPE_VEC4:
			return "vec4";
		case TYPE_MAT2:
			return "mat2";
		case TYPE_MAT3:
			return "mat3";
		case TYPE_MAT4:
			return "mat4";
		case TYPE_SAMPLER2D:
			return "sampler2D";
		case TYPE_ISAMPLER2D:
			return "isampler2D";
		case TYPE_USAMPLER2D:
			return "usampler2D";
		case TYPE_SAMPLER2DARRAY:
			return "sampler2DArray";
		case TYPE_ISAMPLER2DARRAY:
			return "isampler2DArray";
		case TYPE_USAMPLER2DARRAY:
			return "usampler2DArray";
		case TYPE_SAMPLER3D:
			return "sampler3D";
		case TYPE_ISAMPLER3D:
			return "isampler3D";
		case TYPE_USAMPLER3D:
			return "usampler3D";
		case TYPE_SAMPLERCUBE:
			return "samplerCube";
		case TYPE_SAMPLERCUBEARRAY:
			return "samplerCubeArray";
		case TYPE_STRUCT:
			return "struct";
		case TYPE_MAX:
			return "invalid";
	}

	return "";
}

bool ShaderLanguage::is_token_nonvoid_datatype(TokenType p_type) {
	return is_token_datatype(p_type) && p_type != TK_TYPE_VOID;
}

void ShaderLanguage::clear() {
	current_function = StringName();
	last_name = StringName();
	last_type = IDENTIFIER_MAX;

	completion_type = COMPLETION_NONE;
	completion_block = nullptr;
	completion_function = StringName();
	completion_class = SubClassTag::TAG_GLOBAL;
	completion_struct = StringName();

	unknown_varying_usages.clear();

#ifdef DEBUG_ENABLED
	used_constants.clear();
	used_varyings.clear();
	used_uniforms.clear();
	used_functions.clear();
	used_structs.clear();
	used_local_vars.clear();
	warnings.clear();
#endif // DEBUG_ENABLED

	error_line = 0;
	tk_line = 1;
	char_idx = 0;
	error_set = false;
	error_str = "";
	last_const = false;
	while (nodes) {
		Node *n = nodes;
		nodes = nodes->next;
		memdelete(n);
	}
}

#ifdef DEBUG_ENABLED
void ShaderLanguage::_parse_used_identifier(const StringName &p_identifier, IdentifierType p_type, const StringName &p_function) {
	switch (p_type) {
		case IdentifierType::IDENTIFIER_CONSTANT:
			if (HAS_WARNING(ShaderWarning::UNUSED_CONSTANT_FLAG) && used_constants.has(p_identifier)) {
				used_constants[p_identifier].used = true;
			}
			break;
		case IdentifierType::IDENTIFIER_VARYING:
			if (HAS_WARNING(ShaderWarning::UNUSED_VARYING_FLAG) && used_varyings.has(p_identifier)) {
				if (shader->varyings[p_identifier].stage != ShaderNode::Varying::STAGE_VERTEX && shader->varyings[p_identifier].stage != ShaderNode::Varying::STAGE_FRAGMENT) {
					used_varyings[p_identifier].used = true;
				}
			}
			break;
		case IdentifierType::IDENTIFIER_UNIFORM:
			if (HAS_WARNING(ShaderWarning::UNUSED_UNIFORM_FLAG) && used_uniforms.has(p_identifier)) {
				used_uniforms[p_identifier].used = true;
			}
			break;
		case IdentifierType::IDENTIFIER_FUNCTION:
			if (HAS_WARNING(ShaderWarning::UNUSED_FUNCTION_FLAG) && used_functions.has(p_identifier)) {
				used_functions[p_identifier].used = true;
			}
			break;
		case IdentifierType::IDENTIFIER_LOCAL_VAR:
			if (HAS_WARNING(ShaderWarning::UNUSED_LOCAL_VARIABLE_FLAG) && used_local_vars.has(p_function) && used_local_vars[p_function].has(p_identifier)) {
				used_local_vars[p_function][p_identifier].used = true;
			}
			break;
		default:
			break;
	}
}
#endif // DEBUG_ENABLED

bool ShaderLanguage::_find_identifier(const BlockNode *p_block, bool p_allow_reassign, const FunctionInfo &p_function_info, const StringName &p_identifier, DataType *r_data_type, IdentifierType *r_type, bool *r_is_const, int *r_array_size, StringName *r_struct_name, ConstantNode::Value *r_constant_value) {
	if (p_function_info.built_ins.has(p_identifier)) {
		if (r_data_type) {
			*r_data_type = p_function_info.built_ins[p_identifier].type;
		}
		if (r_is_const) {
			*r_is_const = p_function_info.built_ins[p_identifier].constant;
		}
		if (r_type) {
			*r_type = IDENTIFIER_BUILTIN_VAR;
		}
		return true;
	}

	if (p_function_info.stage_functions.has(p_identifier)) {
		if (r_data_type) {
			*r_data_type = p_function_info.stage_functions[p_identifier].return_type;
		}
		if (r_is_const) {
			*r_is_const = true;
		}
		if (r_type) {
			*r_type = IDENTIFIER_FUNCTION;
		}
		return true;
	}

	FunctionNode *function = nullptr;

	while (p_block) {
		if (p_block->variables.has(p_identifier)) {
			if (r_data_type) {
				*r_data_type = p_block->variables[p_identifier].type;
			}
			if (r_is_const) {
				*r_is_const = p_block->variables[p_identifier].is_const;
			}
			if (r_array_size) {
				*r_array_size = p_block->variables[p_identifier].array_size;
			}
			if (r_struct_name) {
				*r_struct_name = p_block->variables[p_identifier].struct_name;
			}
			if (r_constant_value) {
				*r_constant_value = p_block->variables[p_identifier].value;
			}
			if (r_type) {
				*r_type = IDENTIFIER_LOCAL_VAR;
			}
			return true;
		}

		if (p_block->parent_function) {
			function = p_block->parent_function;
			break;
		} else {
			if (p_allow_reassign) {
				break;
			}
			ERR_FAIL_COND_V(!p_block->parent_block, false);
			p_block = p_block->parent_block;
		}
	}

	if (function) {
		for (int i = 0; i < function->arguments.size(); i++) {
			if (function->arguments[i].name == p_identifier) {
				if (r_data_type) {
					*r_data_type = function->arguments[i].type;
				}
				if (r_struct_name) {
					*r_struct_name = function->arguments[i].type_str;
				}
				if (r_array_size) {
					*r_array_size = function->arguments[i].array_size;
				}
				if (r_is_const) {
					*r_is_const = function->arguments[i].is_const;
				}
				if (r_type) {
					*r_type = IDENTIFIER_FUNCTION_ARGUMENT;
				}
				return true;
			}
		}
	}

	if (shader->varyings.has(p_identifier)) {
		if (r_data_type) {
			*r_data_type = shader->varyings[p_identifier].type;
		}
		if (r_array_size) {
			*r_array_size = shader->varyings[p_identifier].array_size;
		}
		if (r_type) {
			*r_type = IDENTIFIER_VARYING;
		}
		return true;
	}

	if (shader->uniforms.has(p_identifier)) {
		if (r_data_type) {
			*r_data_type = shader->uniforms[p_identifier].type;
		}
		if (r_array_size) {
			*r_array_size = shader->uniforms[p_identifier].array_size;
		}
		if (r_type) {
			*r_type = IDENTIFIER_UNIFORM;
		}
		return true;
	}

	if (shader->constants.has(p_identifier)) {
		if (r_is_const) {
			*r_is_const = true;
		}
		if (r_data_type) {
			*r_data_type = shader->constants[p_identifier].type;
		}
		if (r_array_size) {
			*r_array_size = shader->constants[p_identifier].array_size;
		}
		if (r_struct_name) {
			*r_struct_name = shader->constants[p_identifier].type_str;
		}
		if (r_constant_value) {
			if (shader->constants[p_identifier].initializer && shader->constants[p_identifier].initializer->values.size() == 1) {
				*r_constant_value = shader->constants[p_identifier].initializer->values[0];
			}
		}
		if (r_type) {
			*r_type = IDENTIFIER_CONSTANT;
		}
		return true;
	}

	for (int i = 0; i < shader->functions.size(); i++) {
		if (!shader->functions[i].callable) {
			continue;
		}

		if (shader->functions[i].name == p_identifier) {
			if (r_data_type) {
				*r_data_type = shader->functions[i].function->return_type;
			}
			if (r_array_size) {
				*r_array_size = shader->functions[i].function->return_array_size;
			}
			if (r_type) {
				*r_type = IDENTIFIER_FUNCTION;
			}
			return true;
		}
	}

	return false;
}

bool ShaderLanguage::_validate_operator(OperatorNode *p_op, DataType *r_ret_type, int *r_ret_size) {
	bool valid = false;
	DataType ret_type = TYPE_VOID;
	int ret_size = 0;

	switch (p_op->op) {
		case OP_EQUAL:
		case OP_NOT_EQUAL: {
			if ((!p_op->arguments[0]->is_indexed() && p_op->arguments[0]->get_array_size() > 0) || (!p_op->arguments[1]->is_indexed() && p_op->arguments[1]->get_array_size() > 0)) {
				break; // don't accept arrays
			}

			DataType na = p_op->arguments[0]->get_datatype();
			DataType nb = p_op->arguments[1]->get_datatype();
			valid = na == nb;
			ret_type = TYPE_BOOL;
		} break;
		case OP_LESS:
		case OP_LESS_EQUAL:
		case OP_GREATER:
		case OP_GREATER_EQUAL: {
			if ((!p_op->arguments[0]->is_indexed() && p_op->arguments[0]->get_array_size() > 0) || (!p_op->arguments[1]->is_indexed() && p_op->arguments[1]->get_array_size() > 0)) {
				break; // don't accept arrays
			}

			DataType na = p_op->arguments[0]->get_datatype();
			DataType nb = p_op->arguments[1]->get_datatype();

			valid = na == nb && (na == TYPE_UINT || na == TYPE_INT || na == TYPE_FLOAT);
			ret_type = TYPE_BOOL;

		} break;
		case OP_AND:
		case OP_OR: {
			if ((!p_op->arguments[0]->is_indexed() && p_op->arguments[0]->get_array_size() > 0) || (!p_op->arguments[1]->is_indexed() && p_op->arguments[1]->get_array_size() > 0)) {
				break; // don't accept arrays
			}

			DataType na = p_op->arguments[0]->get_datatype();
			DataType nb = p_op->arguments[1]->get_datatype();

			valid = na == nb && na == TYPE_BOOL;
			ret_type = TYPE_BOOL;

		} break;
		case OP_NOT: {
			if (!p_op->arguments[0]->is_indexed() && p_op->arguments[0]->get_array_size() > 0) {
				break; // don't accept arrays
			}

			DataType na = p_op->arguments[0]->get_datatype();
			valid = na == TYPE_BOOL;
			ret_type = TYPE_BOOL;

		} break;
		case OP_INCREMENT:
		case OP_DECREMENT:
		case OP_POST_INCREMENT:
		case OP_POST_DECREMENT:
		case OP_NEGATE: {
			if (!p_op->arguments[0]->is_indexed() && p_op->arguments[0]->get_array_size() > 0) {
				break; // don't accept arrays
			}

			DataType na = p_op->arguments[0]->get_datatype();
			valid = na > TYPE_BOOL && na < TYPE_MAT2;
			ret_type = na;
		} break;
		case OP_ADD:
		case OP_SUB:
		case OP_MUL:
		case OP_DIV: {
			if ((!p_op->arguments[0]->is_indexed() && p_op->arguments[0]->get_array_size() > 0) || (!p_op->arguments[1]->is_indexed() && p_op->arguments[1]->get_array_size() > 0)) {
				break; // don't accept arrays
			}

			DataType na = p_op->arguments[0]->get_datatype();
			DataType nb = p_op->arguments[1]->get_datatype();

			if (na > nb) {
				//make things easier;
				SWAP(na, nb);
			}

			if (na == nb) {
				valid = (na > TYPE_BOOL && na <= TYPE_MAT4);
				ret_type = na;
			} else if (na == TYPE_INT && nb == TYPE_IVEC2) {
				valid = true;
				ret_type = TYPE_IVEC2;
			} else if (na == TYPE_INT && nb == TYPE_IVEC3) {
				valid = true;
				ret_type = TYPE_IVEC3;
			} else if (na == TYPE_INT && nb == TYPE_IVEC4) {
				valid = true;
				ret_type = TYPE_IVEC4;
			} else if (na == TYPE_UINT && nb == TYPE_UVEC2) {
				valid = true;
				ret_type = TYPE_UVEC2;
			} else if (na == TYPE_UINT && nb == TYPE_UVEC3) {
				valid = true;
				ret_type = TYPE_UVEC3;
			} else if (na == TYPE_UINT && nb == TYPE_UVEC4) {
				valid = true;
				ret_type = TYPE_UVEC4;
			} else if (na == TYPE_FLOAT && nb == TYPE_VEC2) {
				valid = true;
				ret_type = TYPE_VEC2;
			} else if (na == TYPE_FLOAT && nb == TYPE_VEC3) {
				valid = true;
				ret_type = TYPE_VEC3;
			} else if (na == TYPE_FLOAT && nb == TYPE_VEC4) {
				valid = true;
				ret_type = TYPE_VEC4;
			} else if (na == TYPE_FLOAT && nb == TYPE_MAT2) {
				valid = true;
				ret_type = TYPE_MAT2;
			} else if (na == TYPE_FLOAT && nb == TYPE_MAT3) {
				valid = true;
				ret_type = TYPE_MAT3;
			} else if (na == TYPE_FLOAT && nb == TYPE_MAT4) {
				valid = true;
				ret_type = TYPE_MAT4;
			} else if (p_op->op == OP_MUL && na == TYPE_VEC2 && nb == TYPE_MAT2) {
				valid = true;
				ret_type = TYPE_VEC2;
			} else if (p_op->op == OP_MUL && na == TYPE_VEC3 && nb == TYPE_MAT3) {
				valid = true;
				ret_type = TYPE_VEC3;
			} else if (p_op->op == OP_MUL && na == TYPE_VEC4 && nb == TYPE_MAT4) {
				valid = true;
				ret_type = TYPE_VEC4;
			}
		} break;
		case OP_ASSIGN_MOD:
		case OP_MOD: {
			/*
			 * The operator modulus (%) operates on signed or unsigned integers or integer vectors. The operand
			 * types must both be signed or both be unsigned. The operands cannot be vectors of differing size. If
			 * one operand is a scalar and the other vector, then the scalar is applied component-wise to the vector,
			 * resulting in the same type as the vector. If both are vectors of the same size, the result is computed
			 * component-wise.
			 */

			if ((!p_op->arguments[0]->is_indexed() && p_op->arguments[0]->get_array_size() > 0) || (!p_op->arguments[1]->is_indexed() && p_op->arguments[1]->get_array_size() > 0)) {
				break; // don't accept arrays
			}

			DataType na = p_op->arguments[0]->get_datatype();
			DataType nb = p_op->arguments[1]->get_datatype();

			if (na == TYPE_INT && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_INT;
			} else if (na == TYPE_IVEC2 && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_IVEC2;
			} else if (na == TYPE_IVEC3 && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_IVEC3;
			} else if (na == TYPE_IVEC4 && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_IVEC4;
			} else if (na == TYPE_IVEC2 && nb == TYPE_IVEC2) {
				valid = true;
				ret_type = TYPE_IVEC2;
			} else if (na == TYPE_IVEC3 && nb == TYPE_IVEC3) {
				valid = true;
				ret_type = TYPE_IVEC3;
			} else if (na == TYPE_IVEC4 && nb == TYPE_IVEC4) {
				valid = true;
				ret_type = TYPE_IVEC4;
				/////
			} else if (na == TYPE_UINT && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UINT;
			} else if (na == TYPE_UVEC2 && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UVEC2;
			} else if (na == TYPE_UVEC3 && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UVEC3;
			} else if (na == TYPE_UVEC4 && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UVEC4;
			} else if (na == TYPE_UVEC2 && nb == TYPE_UVEC2) {
				valid = true;
				ret_type = TYPE_UVEC2;
			} else if (na == TYPE_UVEC3 && nb == TYPE_UVEC3) {
				valid = true;
				ret_type = TYPE_UVEC3;
			} else if (na == TYPE_UVEC4 && nb == TYPE_UVEC4) {
				valid = true;
				ret_type = TYPE_UVEC4;
			}
		} break;
		case OP_ASSIGN_SHIFT_LEFT:
		case OP_ASSIGN_SHIFT_RIGHT:
		case OP_SHIFT_LEFT:
		case OP_SHIFT_RIGHT: {
			if ((!p_op->arguments[0]->is_indexed() && p_op->arguments[0]->get_array_size() > 0) || (!p_op->arguments[1]->is_indexed() && p_op->arguments[1]->get_array_size() > 0)) {
				break; // don't accept arrays
			}

			DataType na = p_op->arguments[0]->get_datatype();
			DataType nb = p_op->arguments[1]->get_datatype();

			if (na == TYPE_INT && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_INT;
			} else if (na == TYPE_IVEC2 && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_IVEC2;
			} else if (na == TYPE_IVEC3 && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_IVEC3;
			} else if (na == TYPE_IVEC4 && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_IVEC4;
			} else if (na == TYPE_IVEC2 && nb == TYPE_IVEC2) {
				valid = true;
				ret_type = TYPE_IVEC2;
			} else if (na == TYPE_IVEC3 && nb == TYPE_IVEC3) {
				valid = true;
				ret_type = TYPE_IVEC3;
			} else if (na == TYPE_IVEC4 && nb == TYPE_IVEC4) {
				valid = true;
				ret_type = TYPE_IVEC4;
			} else if (na == TYPE_UINT && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UINT;
			} else if (na == TYPE_UVEC2 && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UVEC2;
			} else if (na == TYPE_UVEC3 && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UVEC3;
			} else if (na == TYPE_UVEC4 && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UVEC4;
			} else if (na == TYPE_UVEC2 && nb == TYPE_UVEC2) {
				valid = true;
				ret_type = TYPE_UVEC2;
			} else if (na == TYPE_UVEC3 && nb == TYPE_UVEC3) {
				valid = true;
				ret_type = TYPE_UVEC3;
			} else if (na == TYPE_UVEC4 && nb == TYPE_UVEC4) {
				valid = true;
				ret_type = TYPE_UVEC4;
			}
		} break;
		case OP_ASSIGN: {
			int sa = 0;
			int sb = 0;
			if (!p_op->arguments[0]->is_indexed()) {
				sa = p_op->arguments[0]->get_array_size();
			}
			if (!p_op->arguments[1]->is_indexed()) {
				sb = p_op->arguments[1]->get_array_size();
			}
			if (sa != sb) {
				break; // don't accept arrays if their sizes are not equal
			}

			DataType na = p_op->arguments[0]->get_datatype();
			DataType nb = p_op->arguments[1]->get_datatype();
			if (na == TYPE_STRUCT || nb == TYPE_STRUCT) {
				valid = p_op->arguments[0]->get_datatype_name() == p_op->arguments[1]->get_datatype_name();
			} else {
				valid = na == nb;
			}
			ret_type = na;
			ret_size = sa;
		} break;
		case OP_ASSIGN_ADD:
		case OP_ASSIGN_SUB:
		case OP_ASSIGN_MUL:
		case OP_ASSIGN_DIV: {
			int sa = 0;
			int sb = 0;
			if (!p_op->arguments[0]->is_indexed()) {
				sa = p_op->arguments[0]->get_array_size();
			}
			if (!p_op->arguments[1]->is_indexed()) {
				sb = p_op->arguments[1]->get_array_size();
			}
			if (sa > 0 || sb > 0) {
				break; // don't accept arrays
			}

			DataType na = p_op->arguments[0]->get_datatype();
			DataType nb = p_op->arguments[1]->get_datatype();

			if (na == nb) {
				valid = (na > TYPE_BOOL && na <= TYPE_MAT4);
				ret_type = na;
			} else if (na == TYPE_IVEC2 && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_IVEC2;
			} else if (na == TYPE_IVEC3 && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_IVEC3;
			} else if (na == TYPE_IVEC4 && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_IVEC4;
			} else if (na == TYPE_UVEC2 && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UVEC2;
			} else if (na == TYPE_UVEC3 && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UVEC3;
			} else if (na == TYPE_UVEC4 && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UVEC4;
			} else if (na == TYPE_VEC2 && nb == TYPE_FLOAT) {
				valid = true;
				ret_type = TYPE_VEC2;
			} else if (na == TYPE_VEC3 && nb == TYPE_FLOAT) {
				valid = true;
				ret_type = TYPE_VEC3;
			} else if (na == TYPE_VEC4 && nb == TYPE_FLOAT) {
				valid = true;
				ret_type = TYPE_VEC4;
			} else if (na == TYPE_MAT2 && nb == TYPE_FLOAT) {
				valid = true;
				ret_type = TYPE_MAT2;
			} else if (na == TYPE_MAT3 && nb == TYPE_FLOAT) {
				valid = true;
				ret_type = TYPE_MAT3;
			} else if (na == TYPE_MAT4 && nb == TYPE_FLOAT) {
				valid = true;
				ret_type = TYPE_MAT4;
			} else if (p_op->op == OP_ASSIGN_MUL && na == TYPE_VEC2 && nb == TYPE_MAT2) {
				valid = true;
				ret_type = TYPE_VEC2;
			} else if (p_op->op == OP_ASSIGN_MUL && na == TYPE_VEC3 && nb == TYPE_MAT3) {
				valid = true;
				ret_type = TYPE_VEC3;
			} else if (p_op->op == OP_ASSIGN_MUL && na == TYPE_VEC4 && nb == TYPE_MAT4) {
				valid = true;
				ret_type = TYPE_VEC4;
			}
		} break;
		case OP_ASSIGN_BIT_AND:
		case OP_ASSIGN_BIT_OR:
		case OP_ASSIGN_BIT_XOR:
		case OP_BIT_AND:
		case OP_BIT_OR:
		case OP_BIT_XOR: {
			/*
			 * The bitwise operators and (&), exclusive-or (^), and inclusive-or (|). The operands must be of type
			 * signed or unsigned integers or integer vectors. The operands cannot be vectors of differing size. If
			 * one operand is a scalar and the other a vector, the scalar is applied component-wise to the vector,
			 * resulting in the same type as the vector. The fundamental types of the operands (signed or unsigned)
			 * must match.
			 */

			int sa = 0;
			int sb = 0;
			if (!p_op->arguments[0]->is_indexed()) {
				sa = p_op->arguments[0]->get_array_size();
			}
			if (!p_op->arguments[1]->is_indexed()) {
				sb = p_op->arguments[1]->get_array_size();
			}
			if (sa > 0 || sb > 0) {
				break; // don't accept arrays
			}

			DataType na = p_op->arguments[0]->get_datatype();
			DataType nb = p_op->arguments[1]->get_datatype();

			if (na > nb && p_op->op >= OP_BIT_AND) {
				//can swap for non assign
				SWAP(na, nb);
			}

			if (na == TYPE_INT && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_INT;
			} else if (na == TYPE_IVEC2 && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_IVEC2;
			} else if (na == TYPE_IVEC3 && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_IVEC3;
			} else if (na == TYPE_IVEC4 && nb == TYPE_INT) {
				valid = true;
				ret_type = TYPE_IVEC4;
			} else if (na == TYPE_IVEC2 && nb == TYPE_IVEC2) {
				valid = true;
				ret_type = TYPE_IVEC2;
			} else if (na == TYPE_IVEC3 && nb == TYPE_IVEC3) {
				valid = true;
				ret_type = TYPE_IVEC3;
			} else if (na == TYPE_IVEC4 && nb == TYPE_IVEC4) {
				valid = true;
				ret_type = TYPE_IVEC4;
				/////
			} else if (na == TYPE_UINT && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UINT;
			} else if (na == TYPE_UVEC2 && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UVEC2;
			} else if (na == TYPE_UVEC3 && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UVEC3;
			} else if (na == TYPE_UVEC4 && nb == TYPE_UINT) {
				valid = true;
				ret_type = TYPE_UVEC4;
			} else if (na == TYPE_UVEC2 && nb == TYPE_UVEC2) {
				valid = true;
				ret_type = TYPE_UVEC2;
			} else if (na == TYPE_UVEC3 && nb == TYPE_UVEC3) {
				valid = true;
				ret_type = TYPE_UVEC3;
			} else if (na == TYPE_UVEC4 && nb == TYPE_UVEC4) {
				valid = true;
				ret_type = TYPE_UVEC4;
			}
		} break;
		case OP_BIT_INVERT: { //unaries
			if (!p_op->arguments[0]->is_indexed() && p_op->arguments[0]->get_array_size() > 0) {
				break; // don't accept arrays
			}

			DataType na = p_op->arguments[0]->get_datatype();
			valid = na >= TYPE_INT && na < TYPE_FLOAT;
			ret_type = na;
		} break;
		case OP_SELECT_IF: {
			int sa = 0;
			int sb = 0;
			if (!p_op->arguments[1]->is_indexed()) {
				sa = p_op->arguments[1]->get_array_size();
			}
			if (!p_op->arguments[2]->is_indexed()) {
				sb = p_op->arguments[2]->get_array_size();
			}
			if (sa != sb) {
				break; // don't accept arrays if their sizes are not equal
			}

			DataType na = p_op->arguments[0]->get_datatype();
			DataType nb = p_op->arguments[1]->get_datatype();
			DataType nc = p_op->arguments[2]->get_datatype();

			valid = na == TYPE_BOOL && (nb == nc);
			ret_type = nb;
			ret_size = sa;
		} break;
		default: {
			ERR_FAIL_V(false);
		}
	}

	if (r_ret_type) {
		*r_ret_type = ret_type;
	}
	if (r_ret_size) {
		*r_ret_size = ret_size;
	}
	return valid;
}

const ShaderLanguage::BuiltinFuncDef ShaderLanguage::builtin_func_defs[] = {
	// Constructors.

	{ "bool", TYPE_BOOL, { TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec2", TYPE_BVEC2, { TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec2", TYPE_BVEC2, { TYPE_BOOL, TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec3", TYPE_BVEC3, { TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec3", TYPE_BVEC3, { TYPE_BOOL, TYPE_BOOL, TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec3", TYPE_BVEC3, { TYPE_BVEC2, TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec3", TYPE_BVEC3, { TYPE_BOOL, TYPE_BVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec4", TYPE_BVEC4, { TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec4", TYPE_BVEC4, { TYPE_BOOL, TYPE_BOOL, TYPE_BOOL, TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec4", TYPE_BVEC4, { TYPE_BOOL, TYPE_BVEC2, TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec4", TYPE_BVEC4, { TYPE_BVEC2, TYPE_BOOL, TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec4", TYPE_BVEC4, { TYPE_BOOL, TYPE_BOOL, TYPE_BVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec4", TYPE_BVEC4, { TYPE_BOOL, TYPE_BVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec4", TYPE_BVEC4, { TYPE_BVEC3, TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec4", TYPE_BVEC4, { TYPE_BVEC2, TYPE_BVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "float", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec2", TYPE_VEC2, { TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec2", TYPE_VEC2, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec3", TYPE_VEC3, { TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec3", TYPE_VEC3, { TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec3", TYPE_VEC3, { TYPE_VEC2, TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec3", TYPE_VEC3, { TYPE_FLOAT, TYPE_VEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec4", TYPE_VEC4, { TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec4", TYPE_VEC4, { TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec4", TYPE_VEC4, { TYPE_FLOAT, TYPE_VEC2, TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec4", TYPE_VEC4, { TYPE_VEC2, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec4", TYPE_VEC4, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec4", TYPE_VEC4, { TYPE_FLOAT, TYPE_VEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec4", TYPE_VEC4, { TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec4", TYPE_VEC4, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "int", TYPE_INT, { TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec2", TYPE_IVEC2, { TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec2", TYPE_IVEC2, { TYPE_INT, TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec3", TYPE_IVEC3, { TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec3", TYPE_IVEC3, { TYPE_INT, TYPE_INT, TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec3", TYPE_IVEC3, { TYPE_IVEC2, TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec3", TYPE_IVEC3, { TYPE_INT, TYPE_IVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec4", TYPE_IVEC4, { TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec4", TYPE_IVEC4, { TYPE_INT, TYPE_INT, TYPE_INT, TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec4", TYPE_IVEC4, { TYPE_INT, TYPE_IVEC2, TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec4", TYPE_IVEC4, { TYPE_IVEC2, TYPE_INT, TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec4", TYPE_IVEC4, { TYPE_INT, TYPE_INT, TYPE_IVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec4", TYPE_IVEC4, { TYPE_INT, TYPE_IVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec4", TYPE_IVEC4, { TYPE_IVEC3, TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec4", TYPE_IVEC4, { TYPE_IVEC2, TYPE_IVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "uint", TYPE_UINT, { TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec2", TYPE_UVEC2, { TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec2", TYPE_UVEC2, { TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec3", TYPE_UVEC3, { TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec3", TYPE_UVEC3, { TYPE_UINT, TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec3", TYPE_UVEC3, { TYPE_UVEC2, TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec3", TYPE_UVEC3, { TYPE_UINT, TYPE_UVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec4", TYPE_UVEC4, { TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec4", TYPE_UVEC4, { TYPE_UINT, TYPE_UINT, TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec4", TYPE_UVEC4, { TYPE_UINT, TYPE_UVEC2, TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec4", TYPE_UVEC4, { TYPE_UVEC2, TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec4", TYPE_UVEC4, { TYPE_UINT, TYPE_UINT, TYPE_UVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec4", TYPE_UVEC4, { TYPE_UINT, TYPE_UVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec4", TYPE_UVEC4, { TYPE_UVEC3, TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec4", TYPE_UVEC4, { TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, true },

	{ "mat2", TYPE_MAT2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "mat3", TYPE_MAT3, { TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "mat4", TYPE_MAT4, { TYPE_VEC4, TYPE_VEC4, TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "mat2", TYPE_MAT2, { TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "mat3", TYPE_MAT3, { TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "mat4", TYPE_MAT4, { TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	// Conversion scalars.

	{ "int", TYPE_INT, { TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "int", TYPE_INT, { TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "int", TYPE_INT, { TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "int", TYPE_INT, { TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "float", TYPE_FLOAT, { TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "float", TYPE_FLOAT, { TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "float", TYPE_FLOAT, { TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "float", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "uint", TYPE_UINT, { TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uint", TYPE_UINT, { TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uint", TYPE_UINT, { TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uint", TYPE_UINT, { TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },

	{ "bool", TYPE_BOOL, { TYPE_BOOL, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bool", TYPE_BOOL, { TYPE_INT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bool", TYPE_BOOL, { TYPE_UINT, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "bool", TYPE_BOOL, { TYPE_FLOAT, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	// Conversion vectors.

	{ "ivec2", TYPE_IVEC2, { TYPE_BVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec2", TYPE_IVEC2, { TYPE_IVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec2", TYPE_IVEC2, { TYPE_UVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec2", TYPE_IVEC2, { TYPE_VEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "vec2", TYPE_VEC2, { TYPE_BVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec2", TYPE_VEC2, { TYPE_IVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec2", TYPE_VEC2, { TYPE_UVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "vec2", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "uvec2", TYPE_UVEC2, { TYPE_BVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec2", TYPE_UVEC2, { TYPE_IVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec2", TYPE_UVEC2, { TYPE_UVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec2", TYPE_UVEC2, { TYPE_VEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, true },

	{ "bvec2", TYPE_BVEC2, { TYPE_BVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec2", TYPE_BVEC2, { TYPE_IVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec2", TYPE_BVEC2, { TYPE_UVEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "bvec2", TYPE_BVEC2, { TYPE_VEC2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "ivec3", TYPE_IVEC3, { TYPE_BVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec3", TYPE_IVEC3, { TYPE_IVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec3", TYPE_IVEC3, { TYPE_UVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "ivec3", TYPE_IVEC3, { TYPE_VEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "vec3", TYPE_VEC3, { TYPE_BVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec3", TYPE_VEC3, { TYPE_IVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec3", TYPE_VEC3, { TYPE_UVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "vec3", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "uvec3", TYPE_UVEC3, { TYPE_BVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec3", TYPE_UVEC3, { TYPE_IVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec3", TYPE_UVEC3, { TYPE_UVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec3", TYPE_UVEC3, { TYPE_VEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, true },

	{ "bvec3", TYPE_BVEC3, { TYPE_BVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec3", TYPE_BVEC3, { TYPE_IVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec3", TYPE_BVEC3, { TYPE_UVEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "bvec3", TYPE_BVEC3, { TYPE_VEC3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "ivec4", TYPE_IVEC4, { TYPE_BVEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec4", TYPE_IVEC4, { TYPE_IVEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "ivec4", TYPE_IVEC4, { TYPE_UVEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "ivec4", TYPE_IVEC4, { TYPE_VEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "vec4", TYPE_VEC4, { TYPE_BVEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec4", TYPE_VEC4, { TYPE_IVEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "vec4", TYPE_VEC4, { TYPE_UVEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "vec4", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	{ "uvec4", TYPE_UVEC4, { TYPE_BVEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec4", TYPE_UVEC4, { TYPE_IVEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec4", TYPE_UVEC4, { TYPE_UVEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "uvec4", TYPE_UVEC4, { TYPE_VEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, true },

	{ "bvec4", TYPE_BVEC4, { TYPE_BVEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec4", TYPE_BVEC4, { TYPE_IVEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "bvec4", TYPE_BVEC4, { TYPE_UVEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, true },
	{ "bvec4", TYPE_BVEC4, { TYPE_VEC4, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	// Conversion between matrixes.

	{ "mat2", TYPE_MAT2, { TYPE_MAT3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "mat2", TYPE_MAT2, { TYPE_MAT4, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "mat3", TYPE_MAT3, { TYPE_MAT2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "mat3", TYPE_MAT3, { TYPE_MAT4, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "mat4", TYPE_MAT4, { TYPE_MAT2, TYPE_VOID }, { "" }, TAG_GLOBAL, false },
	{ "mat4", TYPE_MAT4, { TYPE_MAT3, TYPE_VOID }, { "" }, TAG_GLOBAL, false },

	// Built-ins - trigonometric functions.
	// radians

	{ "radians", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "degrees" }, TAG_GLOBAL, false },
	{ "radians", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "degrees" }, TAG_GLOBAL, false },
	{ "radians", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "degrees" }, TAG_GLOBAL, false },
	{ "radians", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "degrees" }, TAG_GLOBAL, false },

	// degrees

	{ "degrees", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "radians" }, TAG_GLOBAL, false },
	{ "degrees", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "radians" }, TAG_GLOBAL, false },
	{ "degrees", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "radians" }, TAG_GLOBAL, false },
	{ "degrees", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "radians" }, TAG_GLOBAL, false },

	// sin

	{ "sin", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "angle" }, TAG_GLOBAL, false },
	{ "sin", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "angle" }, TAG_GLOBAL, false },
	{ "sin", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "angle" }, TAG_GLOBAL, false },
	{ "sin", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "angle" }, TAG_GLOBAL, false },

	// cos

	{ "cos", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "angle" }, TAG_GLOBAL, false },
	{ "cos", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "angle" }, TAG_GLOBAL, false },
	{ "cos", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "angle" }, TAG_GLOBAL, false },
	{ "cos", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "angle" }, TAG_GLOBAL, false },

	// tan

	{ "tan", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "angle" }, TAG_GLOBAL, false },
	{ "tan", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "angle" }, TAG_GLOBAL, false },
	{ "tan", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "angle" }, TAG_GLOBAL, false },
	{ "tan", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "angle" }, TAG_GLOBAL, false },

	// asin

	{ "asin", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "asin", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "asin", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "asin", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// acos

	{ "acos", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "acos", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "acos", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "acos", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// atan

	{ "atan", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "y_over_x" }, TAG_GLOBAL, false },
	{ "atan", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "y_over_x" }, TAG_GLOBAL, false },
	{ "atan", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "y_over_x" }, TAG_GLOBAL, false },
	{ "atan", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "y_over_x" }, TAG_GLOBAL, false },
	{ "atan", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "y", "x" }, TAG_GLOBAL, false },
	{ "atan", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "y", "x" }, TAG_GLOBAL, false },
	{ "atan", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "y", "x" }, TAG_GLOBAL, false },
	{ "atan", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "y", "x" }, TAG_GLOBAL, false },

	// sinh

	{ "sinh", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "sinh", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "sinh", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "sinh", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// cosh

	{ "cosh", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "cosh", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "cosh", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "cosh", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// tanh

	{ "tanh", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "tanh", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "tanh", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "tanh", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// asinh

	{ "asinh", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "asinh", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "asinh", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "asinh", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// acosh

	{ "acosh", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "acosh", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "acosh", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "acosh", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// atanh

	{ "atanh", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "atanh", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "atanh", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "atanh", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// Builtins - exponential functions.
	// pow

	{ "pow", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "x", "y" }, TAG_GLOBAL, false },
	{ "pow", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "x", "y" }, TAG_GLOBAL, false },
	{ "pow", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "x", "y" }, TAG_GLOBAL, false },
	{ "pow", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "x", "y" }, TAG_GLOBAL, false },

	// exp

	{ "exp", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "exp", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "exp", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "exp", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// log

	{ "log", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "log", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "log", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "log", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// exp2

	{ "exp2", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "exp2", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "exp2", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "exp2", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// log2

	{ "log2", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "log2", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "log2", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "log2", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// sqrt

	{ "sqrt", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "sqrt", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "sqrt", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "sqrt", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// inversesqrt

	{ "inversesqrt", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "inversesqrt", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "inversesqrt", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "inversesqrt", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// Built-ins - common functions.
	// abs

	{ "abs", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "abs", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "abs", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "abs", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	{ "abs", TYPE_INT, { TYPE_INT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "abs", TYPE_IVEC2, { TYPE_IVEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "abs", TYPE_IVEC3, { TYPE_IVEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "abs", TYPE_IVEC4, { TYPE_IVEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// sign

	{ "sign", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "sign", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "sign", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "sign", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	{ "sign", TYPE_INT, { TYPE_INT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "sign", TYPE_IVEC2, { TYPE_IVEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "sign", TYPE_IVEC3, { TYPE_IVEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "sign", TYPE_IVEC4, { TYPE_IVEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// floor

	{ "floor", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "floor", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "floor", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "floor", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// trunc

	{ "trunc", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "trunc", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "trunc", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "trunc", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// round

	{ "round", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "round", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "round", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "round", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// roundEven

	{ "roundEven", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "roundEven", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "roundEven", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "roundEven", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// ceil

	{ "ceil", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "ceil", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "ceil", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "ceil", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// fract

	{ "fract", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "fract", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "fract", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "fract", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// mod

	{ "mod", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "x", "y" }, TAG_GLOBAL, false },
	{ "mod", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "x", "y" }, TAG_GLOBAL, false },
	{ "mod", TYPE_VEC2, { TYPE_VEC2, TYPE_FLOAT, TYPE_VOID }, { "x", "y" }, TAG_GLOBAL, false },
	{ "mod", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "x", "y" }, TAG_GLOBAL, false },
	{ "mod", TYPE_VEC3, { TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "x", "y" }, TAG_GLOBAL, false },
	{ "mod", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "x", "y" }, TAG_GLOBAL, false },
	{ "mod", TYPE_VEC4, { TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "x", "y" }, TAG_GLOBAL, false },

	// modf

	{ "modf", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "x", "i" }, TAG_GLOBAL, true },
	{ "modf", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "x", "i" }, TAG_GLOBAL, true },
	{ "modf", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "x", "i" }, TAG_GLOBAL, true },
	{ "modf", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "x", "i" }, TAG_GLOBAL, true },

	// min

	{ "min", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "min", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "min", TYPE_VEC2, { TYPE_VEC2, TYPE_FLOAT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "min", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "min", TYPE_VEC3, { TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "min", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "min", TYPE_VEC4, { TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "min", TYPE_INT, { TYPE_INT, TYPE_INT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "min", TYPE_IVEC2, { TYPE_IVEC2, TYPE_IVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "min", TYPE_IVEC2, { TYPE_IVEC2, TYPE_INT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "min", TYPE_IVEC3, { TYPE_IVEC3, TYPE_IVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "min", TYPE_IVEC3, { TYPE_IVEC3, TYPE_INT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "min", TYPE_IVEC4, { TYPE_IVEC4, TYPE_IVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "min", TYPE_IVEC4, { TYPE_IVEC4, TYPE_INT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "min", TYPE_UINT, { TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "min", TYPE_UVEC2, { TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "min", TYPE_UVEC2, { TYPE_UVEC2, TYPE_UINT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "min", TYPE_UVEC3, { TYPE_UVEC3, TYPE_UVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "min", TYPE_UVEC3, { TYPE_UVEC3, TYPE_UINT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "min", TYPE_UVEC4, { TYPE_UVEC4, TYPE_UVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "min", TYPE_UVEC4, { TYPE_UVEC4, TYPE_UINT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },

	// max

	{ "max", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "max", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "max", TYPE_VEC2, { TYPE_VEC2, TYPE_FLOAT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "max", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "max", TYPE_VEC3, { TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "max", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "max", TYPE_VEC4, { TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "max", TYPE_INT, { TYPE_INT, TYPE_INT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "max", TYPE_IVEC2, { TYPE_IVEC2, TYPE_IVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "max", TYPE_IVEC2, { TYPE_IVEC2, TYPE_INT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "max", TYPE_IVEC3, { TYPE_IVEC3, TYPE_IVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "max", TYPE_IVEC3, { TYPE_IVEC3, TYPE_INT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "max", TYPE_IVEC4, { TYPE_IVEC4, TYPE_IVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "max", TYPE_IVEC4, { TYPE_IVEC4, TYPE_INT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "max", TYPE_UINT, { TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "max", TYPE_UVEC2, { TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "max", TYPE_UVEC2, { TYPE_UVEC2, TYPE_UINT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "max", TYPE_UVEC3, { TYPE_UVEC3, TYPE_UVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "max", TYPE_UVEC3, { TYPE_UVEC3, TYPE_UINT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "max", TYPE_UVEC4, { TYPE_UVEC4, TYPE_UVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "max", TYPE_UVEC4, { TYPE_UVEC4, TYPE_UINT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },

	// clamp

	{ "clamp", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },
	{ "clamp", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },
	{ "clamp", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },
	{ "clamp", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },
	{ "clamp", TYPE_VEC2, { TYPE_VEC2, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },
	{ "clamp", TYPE_VEC3, { TYPE_VEC3, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },
	{ "clamp", TYPE_VEC4, { TYPE_VEC4, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },

	{ "clamp", TYPE_INT, { TYPE_INT, TYPE_INT, TYPE_INT, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },
	{ "clamp", TYPE_IVEC2, { TYPE_IVEC2, TYPE_IVEC2, TYPE_IVEC2, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },
	{ "clamp", TYPE_IVEC3, { TYPE_IVEC3, TYPE_IVEC3, TYPE_IVEC3, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },
	{ "clamp", TYPE_IVEC4, { TYPE_IVEC4, TYPE_IVEC4, TYPE_IVEC4, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },
	{ "clamp", TYPE_IVEC2, { TYPE_IVEC2, TYPE_INT, TYPE_INT, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },
	{ "clamp", TYPE_IVEC3, { TYPE_IVEC3, TYPE_INT, TYPE_INT, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },
	{ "clamp", TYPE_IVEC4, { TYPE_IVEC4, TYPE_INT, TYPE_INT, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, false },

	{ "clamp", TYPE_UINT, { TYPE_UINT, TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, true },
	{ "clamp", TYPE_UVEC2, { TYPE_UVEC2, TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, true },
	{ "clamp", TYPE_UVEC3, { TYPE_UVEC3, TYPE_UVEC3, TYPE_UVEC3, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, true },
	{ "clamp", TYPE_UVEC4, { TYPE_UVEC4, TYPE_UVEC4, TYPE_UVEC4, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, true },
	{ "clamp", TYPE_UVEC2, { TYPE_UVEC2, TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, true },
	{ "clamp", TYPE_UVEC3, { TYPE_UVEC3, TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, true },
	{ "clamp", TYPE_UVEC4, { TYPE_UVEC4, TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "x", "minVal", "maxVal" }, TAG_GLOBAL, true },

	// mix

	{ "mix", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "a", "b", "value" }, TAG_GLOBAL, false },
	{ "mix", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_FLOAT, TYPE_VOID }, { "a", "b", "value" }, TAG_GLOBAL, false },
	{ "mix", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_BVEC2, TYPE_VOID }, { "a", "b", "value" }, TAG_GLOBAL, false },
	{ "mix", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "a", "b", "value" }, TAG_GLOBAL, false },
	{ "mix", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "a", "b", "value" }, TAG_GLOBAL, false },
	{ "mix", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_BVEC3, TYPE_VOID }, { "a", "b", "value" }, TAG_GLOBAL, false },
	{ "mix", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b", "value" }, TAG_GLOBAL, false },
	{ "mix", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "a", "b", "value" }, TAG_GLOBAL, false },
	{ "mix", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_BVEC4, TYPE_VOID }, { "a", "b", "value" }, TAG_GLOBAL, false },
	{ "mix", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "a", "b", "value" }, TAG_GLOBAL, false },

	// step

	{ "step", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "edge", "x" }, TAG_GLOBAL, false },
	{ "step", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "edge", "x" }, TAG_GLOBAL, false },
	{ "step", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "edge", "x" }, TAG_GLOBAL, false },
	{ "step", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "edge", "x" }, TAG_GLOBAL, false },
	{ "step", TYPE_VEC2, { TYPE_FLOAT, TYPE_VEC2, TYPE_VOID }, { "edge", "x" }, TAG_GLOBAL, false },
	{ "step", TYPE_VEC3, { TYPE_FLOAT, TYPE_VEC3, TYPE_VOID }, { "edge", "x" }, TAG_GLOBAL, false },
	{ "step", TYPE_VEC4, { TYPE_FLOAT, TYPE_VEC4, TYPE_VOID }, { "edge", "x" }, TAG_GLOBAL, false },

	// smoothstep

	{ "smoothstep", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "edge0", "edge1", "value" }, TAG_GLOBAL, false },
	{ "smoothstep", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "edge0", "edge1", "value" }, TAG_GLOBAL, false },
	{ "smoothstep", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "edge0", "edge1", "value" }, TAG_GLOBAL, false },
	{ "smoothstep", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "edge0", "edge1", "value" }, TAG_GLOBAL, false },
	{ "smoothstep", TYPE_VEC2, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VEC2, TYPE_VOID }, { "edge0", "edge1", "value" }, TAG_GLOBAL, false },
	{ "smoothstep", TYPE_VEC3, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VEC3, TYPE_VOID }, { "edge0", "edge1", "value" }, TAG_GLOBAL, false },
	{ "smoothstep", TYPE_VEC4, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VEC4, TYPE_VOID }, { "edge0", "edge1", "value" }, TAG_GLOBAL, false },

	// isnan

	{ "isnan", TYPE_BOOL, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "isnan", TYPE_BVEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "isnan", TYPE_BVEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "isnan", TYPE_BVEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// isinf

	{ "isinf", TYPE_BOOL, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "isinf", TYPE_BVEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "isinf", TYPE_BVEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "isinf", TYPE_BVEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// floatBitsToInt

	{ "floatBitsToInt", TYPE_INT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },
	{ "floatBitsToInt", TYPE_IVEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },
	{ "floatBitsToInt", TYPE_IVEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },
	{ "floatBitsToInt", TYPE_IVEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },

	// floatBitsToUint

	{ "floatBitsToUint", TYPE_UINT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },
	{ "floatBitsToUint", TYPE_UVEC2, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },
	{ "floatBitsToUint", TYPE_UVEC3, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },
	{ "floatBitsToUint", TYPE_UVEC4, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },

	// intBitsToFloat

	{ "intBitsToFloat", TYPE_FLOAT, { TYPE_INT, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },
	{ "intBitsToFloat", TYPE_VEC2, { TYPE_IVEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },
	{ "intBitsToFloat", TYPE_VEC3, { TYPE_IVEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },
	{ "intBitsToFloat", TYPE_VEC4, { TYPE_IVEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },

	// uintBitsToFloat

	{ "uintBitsToFloat", TYPE_FLOAT, { TYPE_UINT, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },
	{ "uintBitsToFloat", TYPE_VEC2, { TYPE_UVEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },
	{ "uintBitsToFloat", TYPE_VEC3, { TYPE_UVEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },
	{ "uintBitsToFloat", TYPE_VEC4, { TYPE_UVEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, true },

	// Built-ins - geometric functions.
	// length

	{ "length", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "length", TYPE_FLOAT, { TYPE_VEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "length", TYPE_FLOAT, { TYPE_VEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "length", TYPE_FLOAT, { TYPE_VEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// distance

	{ "distance", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "distance", TYPE_FLOAT, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "distance", TYPE_FLOAT, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "distance", TYPE_FLOAT, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	// dot

	{ "dot", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "dot", TYPE_FLOAT, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "dot", TYPE_FLOAT, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "dot", TYPE_FLOAT, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	// cross

	{ "cross", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	// normalize

	{ "normalize", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "v" }, TAG_GLOBAL, false },
	{ "normalize", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "v" }, TAG_GLOBAL, false },
	{ "normalize", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "v" }, TAG_GLOBAL, false },
	{ "normalize", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "v" }, TAG_GLOBAL, false },

	// reflect

	{ "reflect", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "I", "N" }, TAG_GLOBAL, false },

	// refract

	{ "refract", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "I", "N", "eta" }, TAG_GLOBAL, false },

	// faceforward

	{ "faceforward", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "N", "I", "Nref" }, TAG_GLOBAL, false },
	{ "faceforward", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "N", "I", "Nref" }, TAG_GLOBAL, false },
	{ "faceforward", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "N", "I", "Nref" }, TAG_GLOBAL, false },

	// matrixCompMult

	{ "matrixCompMult", TYPE_MAT2, { TYPE_MAT2, TYPE_MAT2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "matrixCompMult", TYPE_MAT3, { TYPE_MAT3, TYPE_MAT3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "matrixCompMult", TYPE_MAT4, { TYPE_MAT4, TYPE_MAT4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	// outerProduct

	{ "outerProduct", TYPE_MAT2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "c", "r" }, TAG_GLOBAL, false },
	{ "outerProduct", TYPE_MAT3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "c", "r" }, TAG_GLOBAL, false },
	{ "outerProduct", TYPE_MAT4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "c", "r" }, TAG_GLOBAL, false },

	// transpose

	{ "transpose", TYPE_MAT2, { TYPE_MAT2, TYPE_VOID }, { "m" }, TAG_GLOBAL, false },
	{ "transpose", TYPE_MAT3, { TYPE_MAT3, TYPE_VOID }, { "m" }, TAG_GLOBAL, false },
	{ "transpose", TYPE_MAT4, { TYPE_MAT4, TYPE_VOID }, { "m" }, TAG_GLOBAL, false },

	// determinant

	{ "determinant", TYPE_FLOAT, { TYPE_MAT2, TYPE_VOID }, { "m" }, TAG_GLOBAL, false },
	{ "determinant", TYPE_FLOAT, { TYPE_MAT3, TYPE_VOID }, { "m" }, TAG_GLOBAL, false },
	{ "determinant", TYPE_FLOAT, { TYPE_MAT4, TYPE_VOID }, { "m" }, TAG_GLOBAL, false },

	// inverse

	{ "inverse", TYPE_MAT2, { TYPE_MAT2, TYPE_VOID }, { "m" }, TAG_GLOBAL, false },
	{ "inverse", TYPE_MAT3, { TYPE_MAT3, TYPE_VOID }, { "m" }, TAG_GLOBAL, false },
	{ "inverse", TYPE_MAT4, { TYPE_MAT4, TYPE_VOID }, { "m" }, TAG_GLOBAL, false },

	// lessThan

	{ "lessThan", TYPE_BVEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "lessThan", TYPE_BVEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "lessThan", TYPE_BVEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "lessThan", TYPE_BVEC2, { TYPE_IVEC2, TYPE_IVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "lessThan", TYPE_BVEC3, { TYPE_IVEC3, TYPE_IVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "lessThan", TYPE_BVEC4, { TYPE_IVEC4, TYPE_IVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "lessThan", TYPE_BVEC2, { TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "lessThan", TYPE_BVEC3, { TYPE_UVEC3, TYPE_UVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "lessThan", TYPE_BVEC4, { TYPE_UVEC4, TYPE_UVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },

	// greaterThan

	{ "greaterThan", TYPE_BVEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "greaterThan", TYPE_BVEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "greaterThan", TYPE_BVEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "greaterThan", TYPE_BVEC2, { TYPE_IVEC2, TYPE_IVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "greaterThan", TYPE_BVEC3, { TYPE_IVEC3, TYPE_IVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "greaterThan", TYPE_BVEC4, { TYPE_IVEC4, TYPE_IVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "greaterThan", TYPE_BVEC2, { TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "greaterThan", TYPE_BVEC3, { TYPE_UVEC3, TYPE_UVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "greaterThan", TYPE_BVEC4, { TYPE_UVEC4, TYPE_UVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },

	// lessThanEqual

	{ "lessThanEqual", TYPE_BVEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "lessThanEqual", TYPE_BVEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "lessThanEqual", TYPE_BVEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "lessThanEqual", TYPE_BVEC2, { TYPE_IVEC2, TYPE_IVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "lessThanEqual", TYPE_BVEC3, { TYPE_IVEC3, TYPE_IVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "lessThanEqual", TYPE_BVEC4, { TYPE_IVEC4, TYPE_IVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "lessThanEqual", TYPE_BVEC2, { TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "lessThanEqual", TYPE_BVEC3, { TYPE_UVEC3, TYPE_UVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "lessThanEqual", TYPE_BVEC4, { TYPE_UVEC4, TYPE_UVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },

	// greaterThanEqual

	{ "greaterThanEqual", TYPE_BVEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "greaterThanEqual", TYPE_BVEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "greaterThanEqual", TYPE_BVEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "greaterThanEqual", TYPE_BVEC2, { TYPE_IVEC2, TYPE_IVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "greaterThanEqual", TYPE_BVEC3, { TYPE_IVEC3, TYPE_IVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "greaterThanEqual", TYPE_BVEC4, { TYPE_IVEC4, TYPE_IVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "greaterThanEqual", TYPE_BVEC2, { TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "greaterThanEqual", TYPE_BVEC3, { TYPE_UVEC3, TYPE_UVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "greaterThanEqual", TYPE_BVEC4, { TYPE_UVEC4, TYPE_UVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },

	// equal

	{ "equal", TYPE_BVEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "equal", TYPE_BVEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "equal", TYPE_BVEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "equal", TYPE_BVEC2, { TYPE_IVEC2, TYPE_IVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "equal", TYPE_BVEC3, { TYPE_IVEC3, TYPE_IVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "equal", TYPE_BVEC4, { TYPE_IVEC4, TYPE_IVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "equal", TYPE_BVEC2, { TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "equal", TYPE_BVEC3, { TYPE_UVEC3, TYPE_UVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "equal", TYPE_BVEC4, { TYPE_UVEC4, TYPE_UVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },

	{ "equal", TYPE_BVEC2, { TYPE_BVEC2, TYPE_BVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "equal", TYPE_BVEC3, { TYPE_BVEC3, TYPE_BVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "equal", TYPE_BVEC4, { TYPE_BVEC4, TYPE_BVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	// notEqual

	{ "notEqual", TYPE_BVEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "notEqual", TYPE_BVEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "notEqual", TYPE_BVEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "notEqual", TYPE_BVEC2, { TYPE_IVEC2, TYPE_IVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "notEqual", TYPE_BVEC3, { TYPE_IVEC3, TYPE_IVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "notEqual", TYPE_BVEC4, { TYPE_IVEC4, TYPE_IVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	{ "notEqual", TYPE_BVEC2, { TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "notEqual", TYPE_BVEC3, { TYPE_UVEC3, TYPE_UVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },
	{ "notEqual", TYPE_BVEC4, { TYPE_UVEC4, TYPE_UVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, true },

	{ "notEqual", TYPE_BVEC2, { TYPE_BVEC2, TYPE_BVEC2, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "notEqual", TYPE_BVEC3, { TYPE_BVEC3, TYPE_BVEC3, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },
	{ "notEqual", TYPE_BVEC4, { TYPE_BVEC4, TYPE_BVEC4, TYPE_VOID }, { "a", "b" }, TAG_GLOBAL, false },

	// any

	{ "any", TYPE_BOOL, { TYPE_BVEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "any", TYPE_BOOL, { TYPE_BVEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "any", TYPE_BOOL, { TYPE_BVEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// all

	{ "all", TYPE_BOOL, { TYPE_BVEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "all", TYPE_BOOL, { TYPE_BVEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "all", TYPE_BOOL, { TYPE_BVEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// not

	{ "not", TYPE_BVEC2, { TYPE_BVEC2, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "not", TYPE_BVEC3, { TYPE_BVEC3, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },
	{ "not", TYPE_BVEC4, { TYPE_BVEC4, TYPE_VOID }, { "x" }, TAG_GLOBAL, false },

	// Built-ins: texture functions.
	// textureSize

	{ "textureSize", TYPE_IVEC2, { TYPE_SAMPLER2D, TYPE_INT, TYPE_VOID }, { "sampler", "lod" }, TAG_GLOBAL, true },
	{ "textureSize", TYPE_IVEC2, { TYPE_ISAMPLER2D, TYPE_INT, TYPE_VOID }, { "sampler", "lod" }, TAG_GLOBAL, true },
	{ "textureSize", TYPE_IVEC2, { TYPE_USAMPLER2D, TYPE_INT, TYPE_VOID }, { "sampler", "lod" }, TAG_GLOBAL, true },
	{ "textureSize", TYPE_IVEC3, { TYPE_SAMPLER2DARRAY, TYPE_INT, TYPE_VOID }, { "sampler", "lod" }, TAG_GLOBAL, true },
	{ "textureSize", TYPE_IVEC3, { TYPE_ISAMPLER2DARRAY, TYPE_INT, TYPE_VOID }, { "sampler", "lod" }, TAG_GLOBAL, true },
	{ "textureSize", TYPE_IVEC3, { TYPE_USAMPLER2DARRAY, TYPE_INT, TYPE_VOID }, { "sampler", "lod" }, TAG_GLOBAL, true },
	{ "textureSize", TYPE_IVEC3, { TYPE_SAMPLER3D, TYPE_INT, TYPE_VOID }, { "sampler", "lod" }, TAG_GLOBAL, true },
	{ "textureSize", TYPE_IVEC3, { TYPE_ISAMPLER3D, TYPE_INT, TYPE_VOID }, { "sampler", "lod" }, TAG_GLOBAL, true },
	{ "textureSize", TYPE_IVEC3, { TYPE_USAMPLER3D, TYPE_INT, TYPE_VOID }, { "sampler", "lod" }, TAG_GLOBAL, true },
	{ "textureSize", TYPE_IVEC2, { TYPE_SAMPLERCUBE, TYPE_INT, TYPE_VOID }, { "sampler", "lod" }, TAG_GLOBAL, true },
	{ "textureSize", TYPE_IVEC2, { TYPE_SAMPLERCUBEARRAY, TYPE_INT, TYPE_VOID }, { "sampler", "lod" }, TAG_GLOBAL, true },

	// texture

	{ "texture", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_VEC2, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, false },
	{ "texture", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_VEC2, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, false },
	{ "texture", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_VEC2, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "texture", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_VEC2, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "texture", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_VEC2, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "texture", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_VEC2, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "texture", TYPE_VEC4, { TYPE_SAMPLER2DARRAY, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, false },
	{ "texture", TYPE_VEC4, { TYPE_SAMPLER2DARRAY, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, false },
	{ "texture", TYPE_UVEC4, { TYPE_USAMPLER2DARRAY, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "texture", TYPE_UVEC4, { TYPE_USAMPLER2DARRAY, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "texture", TYPE_IVEC4, { TYPE_ISAMPLER2DARRAY, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "texture", TYPE_IVEC4, { TYPE_ISAMPLER2DARRAY, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "texture", TYPE_VEC4, { TYPE_SAMPLER3D, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, false },
	{ "texture", TYPE_VEC4, { TYPE_SAMPLER3D, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, false },
	{ "texture", TYPE_UVEC4, { TYPE_USAMPLER3D, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "texture", TYPE_UVEC4, { TYPE_USAMPLER3D, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "texture", TYPE_IVEC4, { TYPE_ISAMPLER3D, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "texture", TYPE_IVEC4, { TYPE_ISAMPLER3D, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "texture", TYPE_VEC4, { TYPE_SAMPLERCUBE, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, false },
	{ "texture", TYPE_VEC4, { TYPE_SAMPLERCUBE, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, false },
	{ "texture", TYPE_VEC4, { TYPE_SAMPLERCUBEARRAY, TYPE_VEC4, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, false },
	{ "texture", TYPE_VEC4, { TYPE_SAMPLERCUBEARRAY, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, false },

	// textureProj

	{ "textureProj", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_VEC4, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_VEC4, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_VEC4, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_VEC4, { TYPE_SAMPLER3D, TYPE_VEC4, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_VEC4, { TYPE_SAMPLER3D, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_IVEC4, { TYPE_ISAMPLER3D, TYPE_VEC4, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_IVEC4, { TYPE_ISAMPLER3D, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_UVEC4, { TYPE_USAMPLER3D, TYPE_VEC4, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureProj", TYPE_UVEC4, { TYPE_USAMPLER3D, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "bias" }, TAG_GLOBAL, true },

	// textureLod

	{ "textureLod", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_VEC2, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, false },
	{ "textureLod", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_VEC2, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureLod", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_VEC2, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureLod", TYPE_VEC4, { TYPE_SAMPLER2DARRAY, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, false },
	{ "textureLod", TYPE_IVEC4, { TYPE_ISAMPLER2DARRAY, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureLod", TYPE_UVEC4, { TYPE_USAMPLER2DARRAY, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureLod", TYPE_VEC4, { TYPE_SAMPLER3D, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, false },
	{ "textureLod", TYPE_IVEC4, { TYPE_ISAMPLER3D, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureLod", TYPE_UVEC4, { TYPE_USAMPLER3D, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureLod", TYPE_VEC4, { TYPE_SAMPLERCUBE, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, false },
	{ "textureLod", TYPE_VEC4, { TYPE_SAMPLERCUBEARRAY, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, false },

	// texelFetch

	{ "texelFetch", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_IVEC2, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "texelFetch", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_IVEC2, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "texelFetch", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_IVEC2, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "texelFetch", TYPE_VEC4, { TYPE_SAMPLER2DARRAY, TYPE_IVEC3, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "texelFetch", TYPE_IVEC4, { TYPE_ISAMPLER2DARRAY, TYPE_IVEC3, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "texelFetch", TYPE_UVEC4, { TYPE_USAMPLER2DARRAY, TYPE_IVEC3, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "texelFetch", TYPE_VEC4, { TYPE_SAMPLER3D, TYPE_IVEC3, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "texelFetch", TYPE_IVEC4, { TYPE_ISAMPLER3D, TYPE_IVEC3, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "texelFetch", TYPE_UVEC4, { TYPE_USAMPLER3D, TYPE_IVEC3, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },

	// textureProjLod

	{ "textureProjLod", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureProjLod", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureProjLod", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureProjLod", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureProjLod", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_VEC3, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureProjLod", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureProjLod", TYPE_VEC4, { TYPE_SAMPLER3D, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureProjLod", TYPE_IVEC4, { TYPE_ISAMPLER3D, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },
	{ "textureProjLod", TYPE_UVEC4, { TYPE_USAMPLER3D, TYPE_VEC4, TYPE_FLOAT, TYPE_VOID }, { "sampler", "coords", "lod" }, TAG_GLOBAL, true },

	// textureGrad

	{ "textureGrad", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_VEC2, TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "sampler", "coords", "dPdx", "dPdy" }, TAG_GLOBAL, true },
	{ "textureGrad", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_VEC2, TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "sampler", "coords", "dPdx", "dPdy" }, TAG_GLOBAL, true },
	{ "textureGrad", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_VEC2, TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "sampler", "coords", "dPdx", "dPdy" }, TAG_GLOBAL, true },
	{ "textureGrad", TYPE_VEC4, { TYPE_SAMPLER2DARRAY, TYPE_VEC3, TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "sampler", "coords", "dPdx", "dPdy" }, TAG_GLOBAL, true },
	{ "textureGrad", TYPE_IVEC4, { TYPE_ISAMPLER2DARRAY, TYPE_VEC3, TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "sampler", "coords", "dPdx", "dPdy" }, TAG_GLOBAL, true },
	{ "textureGrad", TYPE_UVEC4, { TYPE_USAMPLER2DARRAY, TYPE_VEC3, TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "sampler", "coords", "dPdx", "dPdy" }, TAG_GLOBAL, true },
	{ "textureGrad", TYPE_VEC4, { TYPE_SAMPLER3D, TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords", "dPdx", "dPdy" }, TAG_GLOBAL, true },
	{ "textureGrad", TYPE_IVEC4, { TYPE_ISAMPLER3D, TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords", "dPdx", "dPdy" }, TAG_GLOBAL, true },
	{ "textureGrad", TYPE_UVEC4, { TYPE_USAMPLER3D, TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords", "dPdx", "dPdy" }, TAG_GLOBAL, true },
	{ "textureGrad", TYPE_VEC4, { TYPE_SAMPLERCUBE, TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords", "dPdx", "dPdy" }, TAG_GLOBAL, true },
	{ "textureGrad", TYPE_VEC4, { TYPE_SAMPLERCUBEARRAY, TYPE_VEC4, TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords", "dPdx", "dPdy" }, TAG_GLOBAL, true },

	// textureGather

	{ "textureGather", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_VEC2, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_VEC2, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_VEC2, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_VEC4, { TYPE_SAMPLER2D, TYPE_VEC2, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "comp" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_IVEC4, { TYPE_ISAMPLER2D, TYPE_VEC2, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "comp" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_UVEC4, { TYPE_USAMPLER2D, TYPE_VEC2, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "comp" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_VEC4, { TYPE_SAMPLER2DARRAY, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_IVEC4, { TYPE_ISAMPLER2DARRAY, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_UVEC4, { TYPE_USAMPLER2DARRAY, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_VEC4, { TYPE_SAMPLER2DARRAY, TYPE_VEC3, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "comp" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_IVEC4, { TYPE_ISAMPLER2DARRAY, TYPE_VEC3, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "comp" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_UVEC4, { TYPE_USAMPLER2DARRAY, TYPE_VEC3, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "comp" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_VEC4, { TYPE_SAMPLERCUBE, TYPE_VEC3, TYPE_VOID }, { "sampler", "coords" }, TAG_GLOBAL, true },
	{ "textureGather", TYPE_VEC4, { TYPE_SAMPLERCUBE, TYPE_VEC3, TYPE_INT, TYPE_VOID }, { "sampler", "coords", "comp" }, TAG_GLOBAL, true },

	// dFdx

	{ "dFdx", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "p" }, TAG_GLOBAL, true },
	{ "dFdx", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "p" }, TAG_GLOBAL, true },
	{ "dFdx", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "p" }, TAG_GLOBAL, true },
	{ "dFdx", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "p" }, TAG_GLOBAL, true },

	// dFdy

	{ "dFdy", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "p" }, TAG_GLOBAL, true },
	{ "dFdy", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "p" }, TAG_GLOBAL, true },
	{ "dFdy", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "p" }, TAG_GLOBAL, true },
	{ "dFdy", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "p" }, TAG_GLOBAL, true },

	// fwidth

	{ "fwidth", TYPE_FLOAT, { TYPE_FLOAT, TYPE_VOID }, { "p" }, TAG_GLOBAL, true },
	{ "fwidth", TYPE_VEC2, { TYPE_VEC2, TYPE_VOID }, { "p" }, TAG_GLOBAL, true },
	{ "fwidth", TYPE_VEC3, { TYPE_VEC3, TYPE_VOID }, { "p" }, TAG_GLOBAL, true },
	{ "fwidth", TYPE_VEC4, { TYPE_VEC4, TYPE_VOID }, { "p" }, TAG_GLOBAL, true },

	// Sub-functions.
	// array

	{ "length", TYPE_INT, { TYPE_VOID }, { "" }, TAG_ARRAY, true },

	// Modern functions.
	// fma

	{ "fma", TYPE_FLOAT, { TYPE_FLOAT, TYPE_FLOAT, TYPE_FLOAT, TYPE_VOID }, { "a", "b", "c" }, TAG_GLOBAL, false },
	{ "fma", TYPE_VEC2, { TYPE_VEC2, TYPE_VEC2, TYPE_VEC2, TYPE_VOID }, { "a", "b", "c" }, TAG_GLOBAL, false },
	{ "fma", TYPE_VEC3, { TYPE_VEC3, TYPE_VEC3, TYPE_VEC3, TYPE_VOID }, { "a", "b", "c" }, TAG_GLOBAL, false },
	{ "fma", TYPE_VEC4, { TYPE_VEC4, TYPE_VEC4, TYPE_VEC4, TYPE_VOID }, { "a", "b", "c" }, TAG_GLOBAL, false },

	// Packing/Unpacking functions.

	{ "packHalf2x16", TYPE_UINT, { TYPE_VEC2, TYPE_VOID }, { "v" }, TAG_GLOBAL, true },
	{ "packUnorm2x16", TYPE_UINT, { TYPE_VEC2, TYPE_VOID }, { "v" }, TAG_GLOBAL, true },
	{ "packSnorm2x16", TYPE_UINT, { TYPE_VEC2, TYPE_VOID }, { "v" }, TAG_GLOBAL, true },
	{ "packUnorm4x8", TYPE_UINT, { TYPE_VEC4, TYPE_VOID }, { "v" }, TAG_GLOBAL, true },
	{ "packSnorm4x8", TYPE_UINT, { TYPE_VEC4, TYPE_VOID }, { "v" }, TAG_GLOBAL, true },

	{ "unpackHalf2x16", TYPE_VEC2, { TYPE_UINT, TYPE_VOID }, { "v" }, TAG_GLOBAL, true },
	{ "unpackUnorm2x16", TYPE_VEC2, { TYPE_UINT, TYPE_VOID }, { "v" }, TAG_GLOBAL, true },
	{ "unpackSnorm2x16", TYPE_VEC2, { TYPE_UINT, TYPE_VOID }, { "v" }, TAG_GLOBAL, true },
	{ "unpackUnorm4x8", TYPE_VEC4, { TYPE_UINT, TYPE_VOID }, { "v" }, TAG_GLOBAL, true },
	{ "unpackSnorm4x8", TYPE_VEC4, { TYPE_UINT, TYPE_VOID }, { "v" }, TAG_GLOBAL, true },

	// bitfieldExtract

	{ "bitfieldExtract", TYPE_INT, { TYPE_INT, TYPE_INT, TYPE_INT, TYPE_VOID }, { "value", "offset", "bits" }, TAG_GLOBAL, true },
	{ "bitfieldExtract", TYPE_IVEC2, { TYPE_IVEC2, TYPE_INT, TYPE_INT, TYPE_VOID }, { "value", "offset", "bits" }, TAG_GLOBAL, true },
	{ "bitfieldExtract", TYPE_IVEC3, { TYPE_IVEC3, TYPE_INT, TYPE_INT, TYPE_VOID }, { "value", "offset", "bits" }, TAG_GLOBAL, true },
	{ "bitfieldExtract", TYPE_IVEC4, { TYPE_IVEC4, TYPE_INT, TYPE_INT, TYPE_VOID }, { "value", "offset", "bits" }, TAG_GLOBAL, true },

	{ "bitfieldExtract", TYPE_UINT, { TYPE_UINT, TYPE_INT, TYPE_INT, TYPE_VOID }, { "value", "offset", "bits" }, TAG_GLOBAL, true },
	{ "bitfieldExtract", TYPE_UVEC2, { TYPE_UVEC2, TYPE_INT, TYPE_INT, TYPE_VOID }, { "value", "offset", "bits" }, TAG_GLOBAL, true },
	{ "bitfieldExtract", TYPE_UVEC3, { TYPE_UVEC3, TYPE_INT, TYPE_INT, TYPE_VOID }, { "value", "offset", "bits" }, TAG_GLOBAL, true },
	{ "bitfieldExtract", TYPE_UVEC4, { TYPE_UVEC4, TYPE_INT, TYPE_INT, TYPE_VOID }, { "value", "offset", "bits" }, TAG_GLOBAL, true },

	// bitfieldInsert

	{ "bitfieldInsert", TYPE_INT, { TYPE_INT, TYPE_INT, TYPE_INT, TYPE_INT, TYPE_VOID }, { "base", "insert", "offset", "bits" }, TAG_GLOBAL, true },
	{ "bitfieldInsert", TYPE_IVEC2, { TYPE_IVEC2, TYPE_IVEC2, TYPE_INT, TYPE_INT, TYPE_VOID }, { "base", "insert", "offset", "bits" }, TAG_GLOBAL, true },
	{ "bitfieldInsert", TYPE_IVEC3, { TYPE_IVEC3, TYPE_IVEC3, TYPE_INT, TYPE_INT, TYPE_VOID }, { "base", "insert", "offset", "bits" }, TAG_GLOBAL, true },
	{ "bitfieldInsert", TYPE_IVEC4, { TYPE_IVEC4, TYPE_IVEC4, TYPE_INT, TYPE_INT, TYPE_VOID }, { "base", "insert", "offset", "bits" }, TAG_GLOBAL, true },

	{ "bitfieldInsert", TYPE_UINT, { TYPE_UINT, TYPE_UINT, TYPE_INT, TYPE_INT, TYPE_VOID }, { "base", "insert", "offset", "bits" }, TAG_GLOBAL, true },
	{ "bitfieldInsert", TYPE_UVEC2, { TYPE_UVEC2, TYPE_UVEC2, TYPE_INT, TYPE_INT, TYPE_VOID }, { "base", "insert", "offset", "bits" }, TAG_GLOBAL, true },
	{ "bitfieldInsert", TYPE_UVEC3, { TYPE_UVEC3, TYPE_UVEC3, TYPE_INT, TYPE_INT, TYPE_VOID }, { "base", "insert", "offset", "bits" }, TAG_GLOBAL, true },
	{ "bitfieldInsert", TYPE_UVEC4, { TYPE_UVEC4, TYPE_UVEC4, TYPE_INT, TYPE_INT, TYPE_VOID }, { "base", "insert", "offset", "bits" }, TAG_GLOBAL, true },

	// bitfieldReverse

	{ "bitfieldReverse", TYPE_INT, { TYPE_INT, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "bitfieldReverse", TYPE_IVEC2, { TYPE_IVEC2, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "bitfieldReverse", TYPE_IVEC3, { TYPE_IVEC3, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "bitfieldReverse", TYPE_IVEC4, { TYPE_IVEC4, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },

	{ "bitfieldReverse", TYPE_UINT, { TYPE_UINT, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "bitfieldReverse", TYPE_UVEC2, { TYPE_UVEC2, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "bitfieldReverse", TYPE_UVEC3, { TYPE_UVEC3, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "bitfieldReverse", TYPE_UVEC4, { TYPE_UVEC4, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },

	// bitCount

	{ "bitCount", TYPE_INT, { TYPE_INT, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "bitCount", TYPE_IVEC2, { TYPE_IVEC2, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "bitCount", TYPE_IVEC3, { TYPE_IVEC3, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "bitCount", TYPE_IVEC4, { TYPE_IVEC4, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },

	{ "bitCount", TYPE_UINT, { TYPE_UINT, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "bitCount", TYPE_UVEC2, { TYPE_UVEC2, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "bitCount", TYPE_UVEC3, { TYPE_UVEC3, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "bitCount", TYPE_UVEC4, { TYPE_UVEC4, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },

	// findLSB

	{ "findLSB", TYPE_INT, { TYPE_INT, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "findLSB", TYPE_IVEC2, { TYPE_IVEC2, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "findLSB", TYPE_IVEC3, { TYPE_IVEC3, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "findLSB", TYPE_IVEC4, { TYPE_IVEC4, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },

	{ "findLSB", TYPE_UINT, { TYPE_UINT, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "findLSB", TYPE_UVEC2, { TYPE_UVEC2, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "findLSB", TYPE_UVEC3, { TYPE_UVEC3, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "findLSB", TYPE_UVEC4, { TYPE_UVEC4, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },

	// findMSB

	{ "findMSB", TYPE_INT, { TYPE_INT, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "findMSB", TYPE_IVEC2, { TYPE_IVEC2, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "findMSB", TYPE_IVEC3, { TYPE_IVEC3, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "findMSB", TYPE_IVEC4, { TYPE_IVEC4, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },

	{ "findMSB", TYPE_UINT, { TYPE_UINT, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "findMSB", TYPE_UVEC2, { TYPE_UVEC2, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "findMSB", TYPE_UVEC3, { TYPE_UVEC3, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },
	{ "findMSB", TYPE_UVEC4, { TYPE_UVEC4, TYPE_VOID }, { "value" }, TAG_GLOBAL, true },

	// umulExtended

	{ "umulExtended", TYPE_VOID, { TYPE_UINT, TYPE_UINT, TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "x", "y", "msb", "lsb" }, TAG_GLOBAL, true },
	{ "umulExtended", TYPE_VOID, { TYPE_UVEC2, TYPE_UVEC2, TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "x", "y", "msb", "lsb" }, TAG_GLOBAL, true },
	{ "umulExtended", TYPE_VOID, { TYPE_UVEC3, TYPE_UVEC3, TYPE_UVEC3, TYPE_UVEC3, TYPE_VOID }, { "x", "y", "msb", "lsb" }, TAG_GLOBAL, true },
	{ "umulExtended", TYPE_VOID, { TYPE_UVEC4, TYPE_UVEC4, TYPE_UVEC4, TYPE_UVEC4, TYPE_VOID }, { "x", "y", "msb", "lsb" }, TAG_GLOBAL, true },

	// imulExtended

	{ "imulExtended", TYPE_VOID, { TYPE_INT, TYPE_INT, TYPE_INT, TYPE_INT, TYPE_VOID }, { "x", "y", "msb", "lsb" }, TAG_GLOBAL, true },
	{ "imulExtended", TYPE_VOID, { TYPE_IVEC2, TYPE_IVEC2, TYPE_IVEC2, TYPE_IVEC2, TYPE_VOID }, { "x", "y", "msb", "lsb" }, TAG_GLOBAL, true },
	{ "imulExtended", TYPE_VOID, { TYPE_IVEC3, TYPE_IVEC3, TYPE_IVEC3, TYPE_IVEC3, TYPE_VOID }, { "x", "y", "msb", "lsb" }, TAG_GLOBAL, true },
	{ "imulExtended", TYPE_VOID, { TYPE_IVEC4, TYPE_IVEC4, TYPE_IVEC4, TYPE_IVEC4, TYPE_VOID }, { "x", "y", "msb", "lsb" }, TAG_GLOBAL, true },

	// uaddCarry

	{ "uaddCarry", TYPE_UINT, { TYPE_UINT, TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "x", "y", "carry" }, TAG_GLOBAL, true },
	{ "uaddCarry", TYPE_UVEC2, { TYPE_UVEC2, TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "x", "y", "carry" }, TAG_GLOBAL, true },
	{ "uaddCarry", TYPE_UVEC3, { TYPE_UVEC3, TYPE_UVEC3, TYPE_UVEC3, TYPE_VOID }, { "x", "y", "carry" }, TAG_GLOBAL, true },
	{ "uaddCarry", TYPE_UVEC4, { TYPE_UVEC4, TYPE_UVEC4, TYPE_UVEC4, TYPE_VOID }, { "x", "y", "carry" }, TAG_GLOBAL, true },

	// usubBorrow

	{ "usubBorrow", TYPE_UINT, { TYPE_UINT, TYPE_UINT, TYPE_UINT, TYPE_VOID }, { "x", "y", "borrow" }, TAG_GLOBAL, true },
	{ "usubBorrow", TYPE_UVEC2, { TYPE_UVEC2, TYPE_UVEC2, TYPE_UVEC2, TYPE_VOID }, { "x", "y", "borrow" }, TAG_GLOBAL, true },
	{ "usubBorrow", TYPE_UVEC3, { TYPE_UVEC3, TYPE_UVEC3, TYPE_UVEC3, TYPE_VOID }, { "x", "y", "borrow" }, TAG_GLOBAL, true },
	{ "usubBorrow", TYPE_UVEC4, { TYPE_UVEC4, TYPE_UVEC4, TYPE_UVEC4, TYPE_VOID }, { "x", "y", "borrow" }, TAG_GLOBAL, true },

	// ldexp

	{ "ldexp", TYPE_FLOAT, { TYPE_FLOAT, TYPE_INT, TYPE_VOID }, { "x", "exp" }, TAG_GLOBAL, true },
	{ "ldexp", TYPE_VEC2, { TYPE_VEC2, TYPE_IVEC2, TYPE_VOID }, { "x", "exp" }, TAG_GLOBAL, true },
	{ "ldexp", TYPE_VEC3, { TYPE_VEC3, TYPE_IVEC3, TYPE_VOID }, { "x", "exp" }, TAG_GLOBAL, true },
	{ "ldexp", TYPE_VEC4, { TYPE_VEC4, TYPE_IVEC4, TYPE_VOID }, { "x", "exp" }, TAG_GLOBAL, true },

	// frexp

	{ "frexp", TYPE_FLOAT, { TYPE_FLOAT, TYPE_INT, TYPE_VOID }, { "x", "exp" }, TAG_GLOBAL, true },
	{ "frexp", TYPE_VEC2, { TYPE_VEC2, TYPE_IVEC2, TYPE_VOID }, { "x", "exp" }, TAG_GLOBAL, true },
	{ "frexp", TYPE_VEC3, { TYPE_VEC3, TYPE_IVEC3, TYPE_VOID }, { "x", "exp" }, TAG_GLOBAL, true },
	{ "frexp", TYPE_VEC4, { TYPE_VEC4, TYPE_IVEC4, TYPE_VOID }, { "x", "exp" }, TAG_GLOBAL, true },

	{ nullptr, TYPE_VOID, { TYPE_VOID }, { "" }, TAG_GLOBAL, false }
};

const ShaderLanguage::BuiltinFuncOutArgs ShaderLanguage::builtin_func_out_args[] = {
	{ "modf", { 1, -1 } },
	{ "umulExtended", { 2, 3 } },
	{ "imulExtended", { 2, 3 } },
	{ "uaddCarry", { 2, -1 } },
	{ "usubBorrow", { 2, -1 } },
	{ "ldexp", { 1, -1 } },
	{ "frexp", { 1, -1 } },
	{ nullptr, { 0, -1 } }
};

const ShaderLanguage::BuiltinFuncConstArgs ShaderLanguage::builtin_func_const_args[] = {
	{ "textureGather", 2, 0, 3 },
	{ nullptr, 0, 0, 0 }
};

bool ShaderLanguage::_validate_function_call(BlockNode *p_block, const FunctionInfo &p_function_info, OperatorNode *p_func, DataType *r_ret_type, StringName *r_ret_type_str) {
	ERR_FAIL_COND_V(p_func->op != OP_CALL && p_func->op != OP_CONSTRUCT, false);

	Vector<DataType> args;
	Vector<StringName> args2;
	Vector<int> args3;

	ERR_FAIL_COND_V(p_func->arguments[0]->type != Node::TYPE_VARIABLE, false);

	StringName name = static_cast<VariableNode *>(p_func->arguments[0])->name.operator String();

	for (int i = 1; i < p_func->arguments.size(); i++) {
		args.push_back(p_func->arguments[i]->get_datatype());
		args2.push_back(p_func->arguments[i]->get_datatype_name());
		args3.push_back(p_func->arguments[i]->get_array_size());
	}

	int argcount = args.size();

	if (p_function_info.stage_functions.has(name)) {
		//stage based function
		const StageFunctionInfo &sf = p_function_info.stage_functions[name];
		if (argcount != sf.arguments.size()) {
			_set_error(vformat("Invalid number of arguments when calling stage function '%s', which expects %d arguments.", String(name), sf.arguments.size()));
			return false;
		}
		//validate arguments
		for (int i = 0; i < argcount; i++) {
			if (args[i] != sf.arguments[i].type) {
				_set_error(vformat("Invalid argument type when calling stage function '%s', type expected is '%s'.", String(name), String(get_datatype_name(sf.arguments[i].type))));
				return false;
			}
		}

		if (r_ret_type) {
			*r_ret_type = sf.return_type;
		}
		if (r_ret_type_str) {
			*r_ret_type_str = "";
		}
		return true;
	}

	bool failed_builtin = false;
	bool unsupported_builtin = false;
	int builtin_idx = 0;

	if (argcount <= 4) {
		// test builtins
		int idx = 0;

		while (builtin_func_defs[idx].name) {
			if (completion_class != builtin_func_defs[idx].tag) {
				idx++;
				continue;
			}

			if (name == builtin_func_defs[idx].name) {
				failed_builtin = true;
				bool fail = false;
				for (int i = 0; i < argcount; i++) {
					if (p_func->arguments[i + 1]->type == Node::TYPE_ARRAY) {
						const ArrayNode *anode = static_cast<const ArrayNode *>(p_func->arguments[i + 1]);
						if (anode->call_expression == nullptr && !anode->is_indexed()) {
							fail = true;
							break;
						}
					}
					if (get_scalar_type(args[i]) == args[i] && p_func->arguments[i + 1]->type == Node::TYPE_CONSTANT && convert_constant(static_cast<ConstantNode *>(p_func->arguments[i + 1]), builtin_func_defs[idx].args[i])) {
						//all good, but needs implicit conversion later
					} else if (args[i] != builtin_func_defs[idx].args[i]) {
						fail = true;
						break;
					}
				}

				if (!fail) {
					if (RenderingServer::get_singleton()->is_low_end()) {
						if (builtin_func_defs[idx].high_end) {
							fail = true;
							unsupported_builtin = true;
							builtin_idx = idx;
						}
					}
				}

				if (!fail && argcount < 4 && builtin_func_defs[idx].args[argcount] != TYPE_VOID) {
					fail = true; //make sure the number of arguments matches
				}

				if (!fail) {
					{
						int constarg_idx = 0;
						while (builtin_func_const_args[constarg_idx].name) {
							if (String(name) == builtin_func_const_args[constarg_idx].name) {
								int arg = builtin_func_const_args[constarg_idx].arg + 1;
								if (p_func->arguments.size() <= arg) {
									break;
								}

								int min = builtin_func_const_args[constarg_idx].min;
								int max = builtin_func_const_args[constarg_idx].max;

								bool error = false;
								if (p_func->arguments[arg]->type == Node::TYPE_VARIABLE) {
									const VariableNode *vn = (VariableNode *)p_func->arguments[arg];

									bool is_const = false;
									ConstantNode::Value value;
									value.sint = -1;

									_find_identifier(p_block, false, p_function_info, vn->name, nullptr, nullptr, &is_const, nullptr, nullptr, &value);
									if (!is_const || value.sint < min || value.sint > max) {
										error = true;
									}
								} else {
									if (p_func->arguments[arg]->type == Node::TYPE_CONSTANT) {
										ConstantNode *cn = (ConstantNode *)p_func->arguments[arg];

										if (cn->get_datatype() == TYPE_INT && cn->values.size() == 1) {
											int value = cn->values[0].sint;

											if (value < min || value > max) {
												error = true;
											}
										} else {
											error = true;
										}
									} else {
										error = true;
									}
								}
								if (error) {
									_set_error(vformat("Expected integer constant within %s..%s range.", min, max));
									return false;
								}
							}
							constarg_idx++;
						}
					}

					//make sure its not an out argument used in the wrong way
					int outarg_idx = 0;
					while (builtin_func_out_args[outarg_idx].name) {
						if (String(name) == builtin_func_out_args[outarg_idx].name) {
							for (int arg = 0; arg < BuiltinFuncOutArgs::MAX_ARGS; arg++) {
								int arg_idx = builtin_func_out_args[outarg_idx].arguments[arg];
								if (arg_idx == -1) {
									break;
								}
								if (arg_idx < argcount) {
									if (p_func->arguments[arg_idx + 1]->type != Node::TYPE_VARIABLE && p_func->arguments[arg_idx + 1]->type != Node::TYPE_MEMBER && p_func->arguments[arg_idx + 1]->type != Node::TYPE_ARRAY) {
										_set_error("Argument " + itos(arg_idx + 1) + " of function '" + String(name) + "' is not a variable, array or member.");
										return false;
									}

									if (p_func->arguments[arg_idx + 1]->type == Node::TYPE_ARRAY) {
										ArrayNode *mn = static_cast<ArrayNode *>(p_func->arguments[arg_idx + 1]);
										if (mn->is_const) {
											fail = true;
										}
									} else if (p_func->arguments[arg_idx + 1]->type == Node::TYPE_MEMBER) {
										MemberNode *mn = static_cast<MemberNode *>(p_func->arguments[arg_idx + 1]);
										if (mn->basetype_const) {
											fail = true;
										}
									} else { // TYPE_VARIABLE
										VariableNode *vn = static_cast<VariableNode *>(p_func->arguments[arg_idx + 1]);
										if (vn->is_const) {
											fail = true;
										} else {
											StringName varname = vn->name;
											if (shader->uniforms.has(varname)) {
												fail = true;
											} else {
												if (shader->varyings.has(varname)) {
													_set_error(vformat("Varyings cannot be passed for '%s' parameter!", "out"));
													return false;
												}
												if (p_function_info.built_ins.has(varname)) {
													BuiltInInfo info = p_function_info.built_ins[varname];
													if (info.constant) {
														fail = true;
													}
												}
											}
										}
									}
									if (fail) {
										_set_error(vformat("Constant value cannot be passed for '%s' parameter!", "out"));
										return false;
									}

									StringName var_name;
									if (p_func->arguments[arg_idx + 1]->type == Node::TYPE_ARRAY) {
										var_name = static_cast<const ArrayNode *>(p_func->arguments[arg_idx + 1])->name;
									} else if (p_func->arguments[arg_idx + 1]->type == Node::TYPE_MEMBER) {
										Node *n = static_cast<const MemberNode *>(p_func->arguments[arg_idx + 1])->owner;
										while (n->type == Node::TYPE_MEMBER) {
											n = static_cast<const MemberNode *>(n)->owner;
										}
										if (n->type != Node::TYPE_VARIABLE && n->type != Node::TYPE_ARRAY) {
											_set_error("Argument " + itos(arg_idx + 1) + " of function '" + String(name) + "' is not a variable, array or member.");
											return false;
										}
										if (n->type == Node::TYPE_VARIABLE) {
											var_name = static_cast<const VariableNode *>(n)->name;
										} else { // TYPE_ARRAY
											var_name = static_cast<const ArrayNode *>(n)->name;
										}
									} else { // TYPE_VARIABLE
										var_name = static_cast<const VariableNode *>(p_func->arguments[arg_idx + 1])->name;
									}
									const BlockNode *b = p_block;
									bool valid = false;
									while (b) {
										if (b->variables.has(var_name) || p_function_info.built_ins.has(var_name)) {
											valid = true;
											break;
										}
										if (b->parent_function) {
											for (int i = 0; i < b->parent_function->arguments.size(); i++) {
												if (b->parent_function->arguments[i].name == var_name) {
													valid = true;
													break;
												}
											}
										}
										b = b->parent_block;
									}

									if (!valid) {
										_set_error("Argument " + itos(arg_idx + 1) + " of function '" + String(name) + "' can only take a local variable, array or member.");
										return false;
									}
								}
							}
						}
						outarg_idx++;
					}
					//implicitly convert values if possible
					for (int i = 0; i < argcount; i++) {
						if (get_scalar_type(args[i]) != args[i] || args[i] == builtin_func_defs[idx].args[i] || p_func->arguments[i + 1]->type != Node::TYPE_CONSTANT) {
							//can't do implicit conversion here
							continue;
						}

						//this is an implicit conversion
						ConstantNode *constant = static_cast<ConstantNode *>(p_func->arguments[i + 1]);
						ConstantNode *conversion = alloc_node<ConstantNode>();

						conversion->datatype = builtin_func_defs[idx].args[i];
						conversion->values.resize(1);

						convert_constant(constant, builtin_func_defs[idx].args[i], conversion->values.ptrw());
						p_func->arguments.write[i + 1] = conversion;
					}

					if (r_ret_type) {
						*r_ret_type = builtin_func_defs[idx].rettype;
					}

					return true;
				}
			}

			idx++;
		}
	}

	if (unsupported_builtin) {
		String arglist = "";
		for (int i = 0; i < argcount; i++) {
			if (i > 0) {
				arglist += ", ";
			}
			arglist += get_datatype_name(builtin_func_defs[builtin_idx].args[i]);
		}

		String err = "Built-in function \"" + String(name) + "(" + arglist + ")\" is supported only on high-end platform!";
		_set_error(err);
		return false;
	}

	if (failed_builtin) {
		String err = "Invalid arguments for built-in function: " + String(name) + "(";
		for (int i = 0; i < argcount; i++) {
			if (i > 0) {
				err += ",";
			}

			String arg_name;
			if (args[i] == TYPE_STRUCT) {
				arg_name = args2[i];
			} else {
				arg_name = get_datatype_name(args[i]);
			}
			if (args3[i] > 0) {
				arg_name += "[";
				arg_name += itos(args3[i]);
				arg_name += "]";
			}
			err += arg_name;
		}
		err += ")";
		_set_error(err);
		return false;
	}

	// try existing functions..

	StringName exclude_function;
	BlockNode *block = p_block;

	while (block) {
		if (block->parent_function) {
			exclude_function = block->parent_function->name;
		}
		block = block->parent_block;
	}

	if (name == exclude_function) {
		_set_error("Recursion is not allowed");
		return false;
	}

	int last_arg_count = 0;
	String arg_list = "";

	for (int i = 0; i < shader->functions.size(); i++) {
		if (name != shader->functions[i].name) {
			continue;
		}

		if (!shader->functions[i].callable) {
			_set_error("Function '" + String(name) + " can't be called from source code.");
			return false;
		}

		FunctionNode *pfunc = shader->functions[i].function;
		if (arg_list == "") {
			for (int j = 0; j < pfunc->arguments.size(); j++) {
				if (j > 0) {
					arg_list += ", ";
				}
				String func_arg_name;
				if (pfunc->arguments[j].type == TYPE_STRUCT) {
					func_arg_name = pfunc->arguments[j].type_str;
				} else {
					func_arg_name = get_datatype_name(pfunc->arguments[j].type);
				}
				if (pfunc->arguments[j].array_size > 0) {
					func_arg_name += "[";
					func_arg_name += itos(pfunc->arguments[j].array_size);
					func_arg_name += "]";
				}
				arg_list += func_arg_name;
			}
		}

		if (pfunc->arguments.size() != args.size()) {
			last_arg_count = pfunc->arguments.size();
			continue;
		}

		bool fail = false;

		for (int j = 0; j < args.size(); j++) {
			if (get_scalar_type(args[j]) == args[j] && p_func->arguments[j + 1]->type == Node::TYPE_CONSTANT && args3[j] == 0 && convert_constant(static_cast<ConstantNode *>(p_func->arguments[j + 1]), pfunc->arguments[j].type)) {
				//all good, but it needs implicit conversion later
			} else if (args[j] != pfunc->arguments[j].type || (args[j] == TYPE_STRUCT && args2[j] != pfunc->arguments[j].type_str) || args3[j] != pfunc->arguments[j].array_size) {
				String func_arg_name;
				if (pfunc->arguments[j].type == TYPE_STRUCT) {
					func_arg_name = pfunc->arguments[j].type_str;
				} else {
					func_arg_name = get_datatype_name(pfunc->arguments[j].type);
				}
				if (pfunc->arguments[j].array_size > 0) {
					func_arg_name += "[";
					func_arg_name += itos(pfunc->arguments[j].array_size);
					func_arg_name += "]";
				}
				String arg_name;
				if (args[j] == TYPE_STRUCT) {
					arg_name = args2[j];
				} else {
					arg_name = get_datatype_name(args[j]);
				}
				if (args3[j] > 0) {
					arg_name += "[";
					arg_name += itos(args3[j]);
					arg_name += "]";
				}

				_set_error(vformat("Invalid argument for \"%s(%s)\" function: argument %s should be %s but is %s.", String(name), arg_list, j + 1, func_arg_name, arg_name));
				fail = true;
				break;
			}
		}

		if (!fail) {
			//implicitly convert values if possible
			for (int k = 0; k < args.size(); k++) {
				if (get_scalar_type(args[k]) != args[k] || args[k] == pfunc->arguments[k].type || p_func->arguments[k + 1]->type != Node::TYPE_CONSTANT) {
					//can't do implicit conversion here
					continue;
				}

				//this is an implicit conversion
				ConstantNode *constant = static_cast<ConstantNode *>(p_func->arguments[k + 1]);
				ConstantNode *conversion = alloc_node<ConstantNode>();

				conversion->datatype = pfunc->arguments[k].type;
				conversion->values.resize(1);

				convert_constant(constant, pfunc->arguments[k].type, conversion->values.ptrw());
				p_func->arguments.write[k + 1] = conversion;
			}

			if (r_ret_type) {
				*r_ret_type = pfunc->return_type;
				if (pfunc->return_type == TYPE_STRUCT) {
					*r_ret_type_str = pfunc->return_struct_name;
				}
			}

			return true;
		}
	}

	if (last_arg_count > args.size()) {
		_set_error(vformat("Too few arguments for \"%s(%s)\" call. Expected at least %s but received %s.", String(name), arg_list, last_arg_count, args.size()));
	} else if (last_arg_count < args.size()) {
		_set_error(vformat("Too many arguments for \"%s(%s)\" call. Expected at most %s but received %s.", String(name), arg_list, last_arg_count, args.size()));
	}

	return false;
}

bool ShaderLanguage::_compare_datatypes(DataType p_datatype_a, String p_datatype_name_a, int p_array_size_a, DataType p_datatype_b, String p_datatype_name_b, int p_array_size_b) {
	bool result = true;

	if (p_datatype_a == TYPE_STRUCT || p_datatype_b == TYPE_STRUCT) {
		if (p_datatype_name_a != p_datatype_name_b) {
			result = false;
		}
	} else {
		if (p_datatype_a != p_datatype_b) {
			result = false;
		}
	}

	if (p_array_size_a != p_array_size_b) {
		result = false;
	}

	if (!result) {
		String type_name = p_datatype_a == TYPE_STRUCT ? p_datatype_name_a : get_datatype_name(p_datatype_a);
		if (p_array_size_a > 0) {
			type_name += "[";
			type_name += itos(p_array_size_a);
			type_name += "]";
		}

		String type_name2 = p_datatype_b == TYPE_STRUCT ? p_datatype_name_b : get_datatype_name(p_datatype_b);
		if (p_array_size_b > 0) {
			type_name2 += "[";
			type_name2 += itos(p_array_size_b);
			type_name2 += "]";
		}

		_set_error("Invalid assignment of '" + type_name2 + "' to '" + type_name + "'");
	}
	return result;
}

bool ShaderLanguage::_compare_datatypes_in_nodes(Node *a, Node *b) {
	return _compare_datatypes(a->get_datatype(), a->get_datatype_name(), a->get_array_size(), b->get_datatype(), b->get_datatype_name(), b->get_array_size());
}

bool ShaderLanguage::_parse_function_arguments(BlockNode *p_block, const FunctionInfo &p_function_info, OperatorNode *p_func, int *r_complete_arg) {
	TkPos pos = _get_tkpos();
	Token tk = _get_token();

	if (tk.type == TK_PARENTHESIS_CLOSE) {
		return true;
	}

	_set_tkpos(pos);

	while (true) {
		if (r_complete_arg) {
			pos = _get_tkpos();
			tk = _get_token();

			if (tk.type == TK_CURSOR) {
				*r_complete_arg = p_func->arguments.size() - 1;
			} else {
				_set_tkpos(pos);
			}
		}

		Node *arg = _parse_and_reduce_expression(p_block, p_function_info);

		if (!arg) {
			return false;
		}

		p_func->arguments.push_back(arg);

		tk = _get_token();

		if (tk.type == TK_PARENTHESIS_CLOSE) {
			return true;
		} else if (tk.type != TK_COMMA) {
			// something is broken
			_set_error("Expected ',' or ')' after argument");
			return false;
		}
	}

	return true;
}

bool ShaderLanguage::is_token_operator(TokenType p_type) {
	return (p_type == TK_OP_EQUAL ||
			p_type == TK_OP_NOT_EQUAL ||
			p_type == TK_OP_LESS ||
			p_type == TK_OP_LESS_EQUAL ||
			p_type == TK_OP_GREATER ||
			p_type == TK_OP_GREATER_EQUAL ||
			p_type == TK_OP_AND ||
			p_type == TK_OP_OR ||
			p_type == TK_OP_NOT ||
			p_type == TK_OP_ADD ||
			p_type == TK_OP_SUB ||
			p_type == TK_OP_MUL ||
			p_type == TK_OP_DIV ||
			p_type == TK_OP_MOD ||
			p_type == TK_OP_SHIFT_LEFT ||
			p_type == TK_OP_SHIFT_RIGHT ||
			p_type == TK_OP_ASSIGN ||
			p_type == TK_OP_ASSIGN_ADD ||
			p_type == TK_OP_ASSIGN_SUB ||
			p_type == TK_OP_ASSIGN_MUL ||
			p_type == TK_OP_ASSIGN_DIV ||
			p_type == TK_OP_ASSIGN_MOD ||
			p_type == TK_OP_ASSIGN_SHIFT_LEFT ||
			p_type == TK_OP_ASSIGN_SHIFT_RIGHT ||
			p_type == TK_OP_ASSIGN_BIT_AND ||
			p_type == TK_OP_ASSIGN_BIT_OR ||
			p_type == TK_OP_ASSIGN_BIT_XOR ||
			p_type == TK_OP_BIT_AND ||
			p_type == TK_OP_BIT_OR ||
			p_type == TK_OP_BIT_XOR ||
			p_type == TK_OP_BIT_INVERT ||
			p_type == TK_OP_INCREMENT ||
			p_type == TK_OP_DECREMENT ||
			p_type == TK_QUESTION ||
			p_type == TK_COLON);
}

bool ShaderLanguage::is_token_operator_assign(TokenType p_type) {
	return (p_type == TK_OP_ASSIGN ||
			p_type == TK_OP_ASSIGN_ADD ||
			p_type == TK_OP_ASSIGN_SUB ||
			p_type == TK_OP_ASSIGN_MUL ||
			p_type == TK_OP_ASSIGN_DIV ||
			p_type == TK_OP_ASSIGN_MOD ||
			p_type == TK_OP_ASSIGN_SHIFT_LEFT ||
			p_type == TK_OP_ASSIGN_SHIFT_RIGHT ||
			p_type == TK_OP_ASSIGN_BIT_AND ||
			p_type == TK_OP_ASSIGN_BIT_OR ||
			p_type == TK_OP_ASSIGN_BIT_XOR);
}

bool ShaderLanguage::convert_constant(ConstantNode *p_constant, DataType p_to_type, ConstantNode::Value *p_value) {
	if (p_constant->datatype == p_to_type) {
		if (p_value) {
			for (int i = 0; i < p_constant->values.size(); i++) {
				p_value[i] = p_constant->values[i];
			}
		}
		return true;
	} else if (p_constant->datatype == TYPE_INT && p_to_type == TYPE_FLOAT) {
		if (p_value) {
			p_value->real = p_constant->values[0].sint;
		}
		return true;
	} else if (p_constant->datatype == TYPE_UINT && p_to_type == TYPE_FLOAT) {
		if (p_value) {
			p_value->real = p_constant->values[0].uint;
		}
		return true;
	} else if (p_constant->datatype == TYPE_INT && p_to_type == TYPE_UINT) {
		if (p_constant->values[0].sint < 0) {
			return false;
		}
		if (p_value) {
			p_value->uint = p_constant->values[0].sint;
		}
		return true;
	} else if (p_constant->datatype == TYPE_UINT && p_to_type == TYPE_INT) {
		if (p_constant->values[0].uint > 0x7FFFFFFF) {
			return false;
		}
		if (p_value) {
			p_value->sint = p_constant->values[0].uint;
		}
		return true;
	} else {
		return false;
	}
}

bool ShaderLanguage::is_scalar_type(DataType p_type) {
	return p_type == TYPE_BOOL || p_type == TYPE_INT || p_type == TYPE_UINT || p_type == TYPE_FLOAT;
}

bool ShaderLanguage::is_float_type(DataType p_type) {
	switch (p_type) {
		case TYPE_FLOAT:
		case TYPE_VEC2:
		case TYPE_VEC3:
		case TYPE_VEC4:
		case TYPE_MAT2:
		case TYPE_MAT3:
		case TYPE_MAT4:
		case TYPE_SAMPLER2D:
		case TYPE_SAMPLER2DARRAY:
		case TYPE_SAMPLER3D:
		case TYPE_SAMPLERCUBE:
		case TYPE_SAMPLERCUBEARRAY: {
			return true;
		}
		default: {
			return false;
		}
	}
}
bool ShaderLanguage::is_sampler_type(DataType p_type) {
	return p_type == TYPE_SAMPLER2D ||
		   p_type == TYPE_ISAMPLER2D ||
		   p_type == TYPE_USAMPLER2D ||
		   p_type == TYPE_SAMPLER2DARRAY ||
		   p_type == TYPE_ISAMPLER2DARRAY ||
		   p_type == TYPE_USAMPLER2DARRAY ||
		   p_type == TYPE_SAMPLER3D ||
		   p_type == TYPE_ISAMPLER3D ||
		   p_type == TYPE_USAMPLER3D ||
		   p_type == TYPE_SAMPLERCUBE ||
		   p_type == TYPE_SAMPLERCUBEARRAY;
}

Variant ShaderLanguage::constant_value_to_variant(const Vector<ShaderLanguage::ConstantNode::Value> &p_value, DataType p_type, int p_array_size, ShaderLanguage::ShaderNode::Uniform::Hint p_hint) {
	int array_size = p_array_size;

	if (p_value.size() > 0) {
		Variant value;
		switch (p_type) {
			case ShaderLanguage::TYPE_BOOL:
				if (array_size > 0) {
					PackedInt32Array array = PackedInt32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].boolean);
					}
					value = Variant(array);
				} else {
					value = Variant(p_value[0].boolean);
				}
				break;
			case ShaderLanguage::TYPE_BVEC2:
				array_size *= 2;

				if (array_size > 0) {
					PackedInt32Array array = PackedInt32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].boolean);
					}
					value = Variant(array);
				} else {
					value = Variant(p_value[0].boolean);
				}
				break;
			case ShaderLanguage::TYPE_BVEC3:
				array_size *= 3;

				if (array_size > 0) {
					PackedInt32Array array = PackedInt32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].boolean);
					}
					value = Variant(array);
				} else {
					value = Variant(p_value[0].boolean);
				}
				break;
			case ShaderLanguage::TYPE_BVEC4:
				array_size *= 4;

				if (array_size > 0) {
					PackedInt32Array array = PackedInt32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].boolean);
					}
					value = Variant(array);
				} else {
					value = Variant(p_value[0].boolean);
				}
				break;
			case ShaderLanguage::TYPE_INT:
				if (array_size > 0) {
					PackedInt32Array array = PackedInt32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].sint);
					}
					value = Variant(array);
				} else {
					value = Variant(p_value[0].sint);
				}
				break;
			case ShaderLanguage::TYPE_IVEC2:
				if (array_size > 0) {
					array_size *= 2;

					PackedInt32Array array = PackedInt32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].sint);
					}
					value = Variant(array);
				} else {
					value = Variant(Vector2(p_value[0].sint, p_value[1].sint));
				}
				break;
			case ShaderLanguage::TYPE_IVEC3:
				if (array_size > 0) {
					array_size *= 3;

					PackedInt32Array array = PackedInt32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].sint);
					}
					value = Variant(array);
				} else {
					value = Variant(Vector3(p_value[0].sint, p_value[1].sint, p_value[2].sint));
				}
				break;
			case ShaderLanguage::TYPE_IVEC4:
				if (array_size > 0) {
					array_size *= 4;

					PackedInt32Array array = PackedInt32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].sint);
					}
					value = Variant(array);
				} else {
					value = Variant(Plane(p_value[0].sint, p_value[1].sint, p_value[2].sint, p_value[3].sint));
				}
				break;
			case ShaderLanguage::TYPE_UINT:
				if (array_size > 0) {
					PackedInt32Array array = PackedInt32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].uint);
					}
					value = Variant(array);
				} else {
					value = Variant(p_value[0].uint);
				}
				break;
			case ShaderLanguage::TYPE_UVEC2:
				if (array_size > 0) {
					array_size *= 2;

					PackedInt32Array array = PackedInt32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].uint);
					}
					value = Variant(array);
				} else {
					value = Variant(Vector2(p_value[0].uint, p_value[1].uint));
				}
				break;
			case ShaderLanguage::TYPE_UVEC3:
				if (array_size > 0) {
					array_size *= 3;

					PackedInt32Array array = PackedInt32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].uint);
					}
					value = Variant(array);
				} else {
					value = Variant(Vector3(p_value[0].uint, p_value[1].uint, p_value[2].uint));
				}
				break;
			case ShaderLanguage::TYPE_UVEC4:
				if (array_size > 0) {
					array_size *= 4;

					PackedInt32Array array = PackedInt32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].uint);
					}
					value = Variant(array);
				} else {
					value = Variant(Plane(p_value[0].uint, p_value[1].uint, p_value[2].uint, p_value[3].uint));
				}
				break;
			case ShaderLanguage::TYPE_FLOAT:
				if (array_size > 0) {
					PackedFloat32Array array = PackedFloat32Array();
					for (int i = 0; i < array_size; i++) {
						array.push_back(p_value[i].real);
					}
					value = Variant(array);
				} else {
					value = Variant(p_value[0].real);
				}
				break;
			case ShaderLanguage::TYPE_VEC2:
				if (array_size > 0) {
					array_size *= 2;

					PackedVector2Array array = PackedVector2Array();
					for (int i = 0; i < array_size; i += 2) {
						array.push_back(Vector2(p_value[i].real, p_value[i + 1].real));
					}
					value = Variant(array);
				} else {
					value = Variant(Vector2(p_value[0].real, p_value[1].real));
				}
				break;
			case ShaderLanguage::TYPE_VEC3:
				if (array_size > 0) {
					array_size *= 3;

					PackedVector3Array array = PackedVector3Array();
					for (int i = 0; i < array_size; i += 3) {
						array.push_back(Vector3(p_value[i].real, p_value[i + 1].real, p_value[i + 2].real));
					}
					value = Variant(array);
				} else {
					value = Variant(Vector3(p_value[0].real, p_value[1].real, p_value[2].real));
				}
				break;
			case ShaderLanguage::TYPE_VEC4:
				if (array_size > 0) {
					array_size *= 4;

					if (p_hint == ShaderLanguage::ShaderNode::Uniform::HINT_COLOR) {
						PackedColorArray array = PackedColorArray();
						for (int i = 0; i < array_size; i += 4) {
							array.push_back(Color(p_value[i].real, p_value[i + 1].real, p_value[i + 2].real, p_value[i + 3].real));
						}
						value = Variant(array);
					} else {
						PackedFloat32Array array = PackedFloat32Array();
						for (int i = 0; i < array_size; i += 4) {
							array.push_back(p_value[i].real);
							array.push_back(p_value[i + 1].real);
							array.push_back(p_value[i + 2].real);
							array.push_back(p_value[i + 3].real);
						}
						value = Variant(array);
					}
				} else {
					if (p_hint == ShaderLanguage::ShaderNode::Uniform::HINT_COLOR) {
						value = Variant(Color(p_value[0].real, p_value[1].real, p_value[2].real, p_value[3].real));
					} else {
						value = Variant(Plane(p_value[0].real, p_value[1].real, p_value[2].real, p_value[3].real));
					}
				}
				break;
			case ShaderLanguage::TYPE_MAT2:
				if (array_size > 0) {
					array_size *= 4;

					PackedFloat32Array array = PackedFloat32Array();
					for (int i = 0; i < array_size; i += 4) {
						array.push_back(p_value[i].real);
						array.push_back(p_value[i + 1].real);
						array.push_back(p_value[i + 2].real);
						array.push_back(p_value[i + 3].real);
					}
					value = Variant(array);
				} else {
					value = Variant(Transform2D(p_value[0].real, p_value[2].real, p_value[1].real, p_value[3].real, 0.0, 0.0));
				}
				break;
			case ShaderLanguage::TYPE_MAT3: {
				if (array_size > 0) {
					array_size *= 9;

					PackedFloat32Array array = PackedFloat32Array();
					for (int i = 0; i < array_size; i += 9) {
						for (int j = 0; j < 9; j++) {
							array.push_back(p_value[i + j].real);
						}
					}
					value = Variant(array);
				} else {
					Basis p;
					p[0][0] = p_value[0].real;
					p[0][1] = p_value[1].real;
					p[0][2] = p_value[2].real;
					p[1][0] = p_value[3].real;
					p[1][1] = p_value[4].real;
					p[1][2] = p_value[5].real;
					p[2][0] = p_value[6].real;
					p[2][1] = p_value[7].real;
					p[2][2] = p_value[8].real;
					value = Variant(p);
				}
				break;
			}
			case ShaderLanguage::TYPE_MAT4: {
				if (array_size > 0) {
					array_size *= 16;

					PackedFloat32Array array = PackedFloat32Array();
					for (int i = 0; i < array_size; i += 16) {
						for (int j = 0; j < 16; j++) {
							array.push_back(p_value[i + j].real);
						}
					}
					value = Variant(array);
				} else {
					Basis p;
					p[0][0] = p_value[0].real;
					p[0][1] = p_value[1].real;
					p[0][2] = p_value[2].real;
					p[1][0] = p_value[4].real;
					p[1][1] = p_value[5].real;
					p[1][2] = p_value[6].real;
					p[2][0] = p_value[8].real;
					p[2][1] = p_value[9].real;
					p[2][2] = p_value[10].real;
					Transform3D t = Transform3D(p, Vector3(p_value[3].real, p_value[7].real, p_value[11].real));
					value = Variant(t);
				}
				break;
			}
			case ShaderLanguage::TYPE_ISAMPLER2DARRAY:
			case ShaderLanguage::TYPE_ISAMPLER2D:
			case ShaderLanguage::TYPE_ISAMPLER3D:
			case ShaderLanguage::TYPE_SAMPLER2DARRAY:
			case ShaderLanguage::TYPE_SAMPLER2D:
			case ShaderLanguage::TYPE_SAMPLER3D:
			case ShaderLanguage::TYPE_USAMPLER2DARRAY:
			case ShaderLanguage::TYPE_USAMPLER2D:
			case ShaderLanguage::TYPE_USAMPLER3D:
			case ShaderLanguage::TYPE_SAMPLERCUBE:
			case ShaderLanguage::TYPE_SAMPLERCUBEARRAY: {
				// Texture types, likely not relevant here.
				break;
			}
			case ShaderLanguage::TYPE_STRUCT:
				break;
			case ShaderLanguage::TYPE_VOID:
				break;
			case ShaderLanguage::TYPE_MAX:
				break;
		}
		return value;
	}
	return Variant();
}

PropertyInfo ShaderLanguage::uniform_to_property_info(const ShaderNode::Uniform &p_uniform) {
	PropertyInfo pi;
	switch (p_uniform.type) {
		case ShaderLanguage::TYPE_VOID:
			pi.type = Variant::NIL;
			break;
		case ShaderLanguage::TYPE_BOOL:
			if (p_uniform.array_size > 0) {
				pi.type = Variant::PACKED_INT32_ARRAY;
			} else {
				pi.type = Variant::BOOL;
			}
			break;
		case ShaderLanguage::TYPE_BVEC2:
			if (p_uniform.array_size > 0) {
				pi.type = Variant::PACKED_INT32_ARRAY;
			} else {
				pi.type = Variant::INT;
				pi.hint = PROPERTY_HINT_FLAGS;
				pi.hint_string = "x,y";
			}
			break;
		case ShaderLanguage::TYPE_BVEC3:
			if (p_uniform.array_size > 0) {
				pi.type = Variant::PACKED_INT32_ARRAY;
			} else {
				pi.type = Variant::INT;
				pi.hint = PROPERTY_HINT_FLAGS;
				pi.hint_string = "x,y,z";
			}
			break;
		case ShaderLanguage::TYPE_BVEC4:
			if (p_uniform.array_size > 0) {
				pi.type = Variant::PACKED_INT32_ARRAY;
			} else {
				pi.type = Variant::INT;
				pi.hint = PROPERTY_HINT_FLAGS;
				pi.hint_string = "x,y,z,w";
			}
			break;
		case ShaderLanguage::TYPE_UINT:
		case ShaderLanguage::TYPE_INT: {
			if (p_uniform.array_size > 0) {
				pi.type = Variant::PACKED_INT32_ARRAY;
			} else {
				pi.type = Variant::INT;
				if (p_uniform.hint == ShaderLanguage::ShaderNode::Uniform::HINT_RANGE) {
					pi.hint = PROPERTY_HINT_RANGE;
					pi.hint_string = rtos(p_uniform.hint_range[0]) + "," + rtos(p_uniform.hint_range[1]) + "," + rtos(p_uniform.hint_range[2]);
				}
			}
		} break;
		case ShaderLanguage::TYPE_IVEC2:
		case ShaderLanguage::TYPE_IVEC3:
		case ShaderLanguage::TYPE_IVEC4:
		case ShaderLanguage::TYPE_UVEC2:
		case ShaderLanguage::TYPE_UVEC3:
		case ShaderLanguage::TYPE_UVEC4: {
			pi.type = Variant::PACKED_INT32_ARRAY;
		} break;
		case ShaderLanguage::TYPE_FLOAT: {
			if (p_uniform.array_size > 0) {
				pi.type = Variant::PACKED_FLOAT32_ARRAY;
			} else {
				pi.type = Variant::FLOAT;
				if (p_uniform.hint == ShaderLanguage::ShaderNode::Uniform::HINT_RANGE) {
					pi.hint = PROPERTY_HINT_RANGE;
					pi.hint_string = rtos(p_uniform.hint_range[0]) + "," + rtos(p_uniform.hint_range[1]) + "," + rtos(p_uniform.hint_range[2]);
				}
			}
		} break;
		case ShaderLanguage::TYPE_VEC2:
			if (p_uniform.array_size > 0) {
				pi.type = Variant::PACKED_VECTOR2_ARRAY;
			} else {
				pi.type = Variant::VECTOR2;
			}
			break;
		case ShaderLanguage::TYPE_VEC3:
			if (p_uniform.array_size > 0) {
				pi.type = Variant::PACKED_VECTOR3_ARRAY;
			} else {
				pi.type = Variant::VECTOR3;
			}
			break;
		case ShaderLanguage::TYPE_VEC4: {
			if (p_uniform.array_size > 0) {
				if (p_uniform.hint == ShaderLanguage::ShaderNode::Uniform::HINT_COLOR) {
					pi.type = Variant::PACKED_COLOR_ARRAY;
				} else {
					pi.type = Variant::PACKED_FLOAT32_ARRAY;
				}
			} else {
				if (p_uniform.hint == ShaderLanguage::ShaderNode::Uniform::HINT_COLOR) {
					pi.type = Variant::COLOR;
				} else {
					pi.type = Variant::PLANE;
				}
			}
		} break;
		case ShaderLanguage::TYPE_MAT2:
			if (p_uniform.array_size > 0) {
				pi.type = Variant::PACKED_FLOAT32_ARRAY;
			} else {
				pi.type = Variant::TRANSFORM2D;
			}
			break;
		case ShaderLanguage::TYPE_MAT3:
			if (p_uniform.array_size > 0) {
				pi.type = Variant::PACKED_FLOAT32_ARRAY;
			} else {
				pi.type = Variant::BASIS;
			}
			break;
		case ShaderLanguage::TYPE_MAT4:
			if (p_uniform.array_size > 0) {
				pi.type = Variant::PACKED_FLOAT32_ARRAY;
			} else {
				pi.type = Variant::TRANSFORM3D;
			}
			break;
		case ShaderLanguage::TYPE_SAMPLER2D:
		case ShaderLanguage::TYPE_ISAMPLER2D:
		case ShaderLanguage::TYPE_USAMPLER2D: {
			if (p_uniform.array_size > 0) {
				pi.type = Variant::ARRAY;
			} else {
				pi.type = Variant::OBJECT;
			}
			pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
			pi.hint_string = "Texture2D";
		} break;
		case ShaderLanguage::TYPE_SAMPLER2DARRAY:
		case ShaderLanguage::TYPE_ISAMPLER2DARRAY:
		case ShaderLanguage::TYPE_USAMPLER2DARRAY: {
			if (p_uniform.array_size > 0) {
				pi.type = Variant::ARRAY;
			} else {
				pi.type = Variant::OBJECT;
			}
			pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
			pi.hint_string = "TextureLayered";
		} break;
		case ShaderLanguage::TYPE_SAMPLER3D:
		case ShaderLanguage::TYPE_ISAMPLER3D:
		case ShaderLanguage::TYPE_USAMPLER3D: {
			if (p_uniform.array_size > 0) {
				pi.type = Variant::ARRAY;
			} else {
				pi.type = Variant::OBJECT;
			}
			pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
			pi.hint_string = "Texture3D";
		} break;
		case ShaderLanguage::TYPE_SAMPLERCUBE:
		case ShaderLanguage::TYPE_SAMPLERCUBEARRAY: {
			if (p_uniform.array_size > 0) {
				pi.type = Variant::ARRAY;
			} else {
				pi.type = Variant::OBJECT;
			}
			pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
			pi.hint_string = "TextureLayered";
		} break;
		case ShaderLanguage::TYPE_STRUCT: {
			// FIXME: Implement this.
		} break;
		case ShaderLanguage::TYPE_MAX:
			break;
	}
	return pi;
}

uint32_t ShaderLanguage::get_type_size(DataType p_type) {
	switch (p_type) {
		case TYPE_VOID:
			return 0;
		case TYPE_BOOL:
		case TYPE_INT:
		case TYPE_UINT:
		case TYPE_FLOAT:
			return 4;
		case TYPE_BVEC2:
		case TYPE_IVEC2:
		case TYPE_UVEC2:
		case TYPE_VEC2:
			return 8;
		case TYPE_BVEC3:
		case TYPE_IVEC3:
		case TYPE_UVEC3:
		case TYPE_VEC3:
			return 12;
		case TYPE_BVEC4:
		case TYPE_IVEC4:
		case TYPE_UVEC4:
		case TYPE_VEC4:
			return 16;
		case TYPE_MAT2:
			return 8;
		case TYPE_MAT3:
			return 12;
		case TYPE_MAT4:
			return 16;
		case TYPE_SAMPLER2D:
		case TYPE_ISAMPLER2D:
		case TYPE_USAMPLER2D:
		case TYPE_SAMPLER2DARRAY:
		case TYPE_ISAMPLER2DARRAY:
		case TYPE_USAMPLER2DARRAY:
		case TYPE_SAMPLER3D:
		case TYPE_ISAMPLER3D:
		case TYPE_USAMPLER3D:
		case TYPE_SAMPLERCUBE:
		case TYPE_SAMPLERCUBEARRAY:
			return 4; //not really, but useful for indices
		case TYPE_STRUCT:
			// FIXME: Implement.
			return 0;
		case ShaderLanguage::TYPE_MAX:
			return 0;
	}
	return 0;
}

void ShaderLanguage::get_keyword_list(List<String> *r_keywords) {
	Set<String> kws;

	int idx = 0;

	while (keyword_list[idx].text) {
		kws.insert(keyword_list[idx].text);
		idx++;
	}

	idx = 0;

	while (builtin_func_defs[idx].name) {
		kws.insert(builtin_func_defs[idx].name);

		idx++;
	}

	for (Set<String>::Element *E = kws.front(); E; E = E->next()) {
		r_keywords->push_back(E->get());
	}
}

bool ShaderLanguage::is_control_flow_keyword(String p_keyword) {
	return p_keyword == "break" ||
		   p_keyword == "case" ||
		   p_keyword == "continue" ||
		   p_keyword == "default" ||
		   p_keyword == "do" ||
		   p_keyword == "else" ||
		   p_keyword == "for" ||
		   p_keyword == "if" ||
		   p_keyword == "return" ||
		   p_keyword == "switch" ||
		   p_keyword == "while";
}

void ShaderLanguage::get_builtin_funcs(List<String> *r_keywords) {
	Set<String> kws;

	int idx = 0;

	while (builtin_func_defs[idx].name) {
		kws.insert(builtin_func_defs[idx].name);

		idx++;
	}

	for (Set<String>::Element *E = kws.front(); E; E = E->next()) {
		r_keywords->push_back(E->get());
	}
}

ShaderLanguage::DataType ShaderLanguage::get_scalar_type(DataType p_type) {
	static const DataType scalar_types[] = {
		TYPE_VOID,
		TYPE_BOOL,
		TYPE_BOOL,
		TYPE_BOOL,
		TYPE_BOOL,
		TYPE_INT,
		TYPE_INT,
		TYPE_INT,
		TYPE_INT,
		TYPE_UINT,
		TYPE_UINT,
		TYPE_UINT,
		TYPE_UINT,
		TYPE_FLOAT,
		TYPE_FLOAT,
		TYPE_FLOAT,
		TYPE_FLOAT,
		TYPE_FLOAT,
		TYPE_FLOAT,
		TYPE_FLOAT,
		TYPE_FLOAT,
		TYPE_INT,
		TYPE_UINT,
		TYPE_FLOAT,
	};

	return scalar_types[p_type];
}

int ShaderLanguage::get_cardinality(DataType p_type) {
	static const int cardinality_table[] = {
		0,
		1,
		2,
		3,
		4,
		1,
		2,
		3,
		4,
		1,
		2,
		3,
		4,
		1,
		2,
		3,
		4,
		4,
		9,
		16,
		1,
		1,
		1,
		1,
	};

	return cardinality_table[p_type];
}

bool ShaderLanguage::_get_completable_identifier(BlockNode *p_block, CompletionType p_type, StringName &identifier) {
	identifier = StringName();

	TkPos pos = { 0, 0 };

	Token tk = _get_token();

	if (tk.type == TK_IDENTIFIER) {
		identifier = tk.text;
		pos = _get_tkpos();
		tk = _get_token();
	}

	if (tk.type == TK_CURSOR) {
		completion_type = p_type;
		completion_line = tk_line;
		completion_block = p_block;

		pos = _get_tkpos();
		tk = _get_token();

		if (tk.type == TK_IDENTIFIER) {
			identifier = identifier.operator String() + tk.text.operator String();
		} else {
			_set_tkpos(pos);
		}
		return true;
	} else if (identifier != StringName()) {
		_set_tkpos(pos);
	}

	return false;
}

bool ShaderLanguage::_is_operator_assign(Operator p_op) const {
	switch (p_op) {
		case OP_ASSIGN:
		case OP_ASSIGN_ADD:
		case OP_ASSIGN_SUB:
		case OP_ASSIGN_MUL:
		case OP_ASSIGN_DIV:
		case OP_ASSIGN_MOD:
		case OP_ASSIGN_SHIFT_LEFT:
		case OP_ASSIGN_SHIFT_RIGHT:
		case OP_ASSIGN_BIT_AND:
		case OP_ASSIGN_BIT_OR:
		case OP_ASSIGN_BIT_XOR:
			return true;
		default:
			return false;
	}

	return false;
}

bool ShaderLanguage::_validate_varying_assign(ShaderNode::Varying &p_varying, String *r_message) {
	if (current_function != String("vertex") && current_function != String("fragment")) {
		*r_message = vformat(RTR("Varying may not be assigned in the '%s' function."), current_function);
		return false;
	}
	switch (p_varying.stage) {
		case ShaderNode::Varying::STAGE_UNKNOWN: // first assign
			if (current_function == varying_function_names.vertex) {
				p_varying.stage = ShaderNode::Varying::STAGE_VERTEX;
			} else if (current_function == varying_function_names.fragment) {
				p_varying.stage = ShaderNode::Varying::STAGE_FRAGMENT;
			}
			break;
		case ShaderNode::Varying::STAGE_VERTEX_TO_FRAGMENT_LIGHT:
		case ShaderNode::Varying::STAGE_VERTEX:
			if (current_function == varying_function_names.fragment) {
				*r_message = RTR("Varyings which assigned in 'vertex' function may not be reassigned in 'fragment' or 'light'.");
				return false;
			}
			break;
		case ShaderNode::Varying::STAGE_FRAGMENT_TO_LIGHT:
		case ShaderNode::Varying::STAGE_FRAGMENT:
			if (current_function == varying_function_names.vertex) {
				*r_message = RTR("Varyings which assigned in 'fragment' function may not be reassigned in 'vertex' or 'light'.");
				return false;
			}
			break;
		default:
			break;
	}
	return true;
}

bool ShaderLanguage::_validate_varying_using(ShaderNode::Varying &p_varying, String *r_message) {
	switch (p_varying.stage) {
		case ShaderNode::Varying::STAGE_UNKNOWN:
			VaryingUsage usage;
			usage.var = &p_varying;
			usage.line = tk_line;
			unknown_varying_usages.push_back(usage);
			break;
		case ShaderNode::Varying::STAGE_VERTEX:
			if (current_function == varying_function_names.fragment || current_function == varying_function_names.light) {
				p_varying.stage = ShaderNode::Varying::STAGE_VERTEX_TO_FRAGMENT_LIGHT;
			}
			break;
		case ShaderNode::Varying::STAGE_FRAGMENT:
			if (current_function == varying_function_names.light) {
				p_varying.stage = ShaderNode::Varying::STAGE_FRAGMENT_TO_LIGHT;
			}
			break;
		default:
			break;
	}
	return true;
}

bool ShaderLanguage::_check_varying_usages(int *r_error_line, String *r_error_message) const {
	for (const List<ShaderLanguage::VaryingUsage>::Element *E = unknown_varying_usages.front(); E; E = E->next()) {
		ShaderNode::Varying::Stage stage = E->get().var->stage;
		if (stage != ShaderNode::Varying::STAGE_UNKNOWN && stage != ShaderNode::Varying::STAGE_VERTEX && stage != ShaderNode::Varying::STAGE_VERTEX_TO_FRAGMENT_LIGHT) {
			*r_error_line = E->get().line;
			*r_error_message = RTR("Fragment-stage varying could not been accessed in custom function!");
			return false;
		}
	}

	return true;
}

bool ShaderLanguage::_check_node_constness(const Node *p_node) const {
	switch (p_node->type) {
		case Node::TYPE_OPERATOR: {
			OperatorNode *op_node = (OperatorNode *)p_node;
			for (int i = int(op_node->op == OP_CALL); i < op_node->arguments.size(); i++) {
				if (!_check_node_constness(op_node->arguments[i])) {
					return false;
				}
			}
		} break;
		case Node::TYPE_CONSTANT:
			break;
		case Node::TYPE_VARIABLE: {
			VariableNode *varn = (VariableNode *)p_node;
			if (!varn->is_const) {
				return false;
			}
		} break;
		case Node::TYPE_ARRAY: {
			ArrayNode *arrn = (ArrayNode *)p_node;
			if (!arrn->is_const) {
				return false;
			}
		} break;
		default:
			return false;
	}
	return true;
}

bool ShaderLanguage::_validate_assign(Node *p_node, const FunctionInfo &p_function_info, String *r_message) {
	if (p_node->type == Node::TYPE_OPERATOR) {
		OperatorNode *op = static_cast<OperatorNode *>(p_node);

		if (op->op == OP_INDEX) {
			return _validate_assign(op->arguments[0], p_function_info, r_message);

		} else if (_is_operator_assign(op->op)) {
			//chained assignment
			return _validate_assign(op->arguments[1], p_function_info, r_message);

		} else if (op->op == OP_CALL) {
			if (r_message) {
				*r_message = RTR("Assignment to function.");
			}
			return false;
		}

	} else if (p_node->type == Node::TYPE_MEMBER) {
		MemberNode *member = static_cast<MemberNode *>(p_node);

		if (member->has_swizzling_duplicates) {
			if (r_message) {
				*r_message = RTR("Swizzling assignment contains duplicates.");
			}
			return false;
		}

		return _validate_assign(member->owner, p_function_info, r_message);

	} else if (p_node->type == Node::TYPE_VARIABLE) {
		VariableNode *var = static_cast<VariableNode *>(p_node);

		if (shader->uniforms.has(var->name)) {
			if (r_message) {
				*r_message = RTR("Assignment to uniform.");
			}
			return false;
		}

		if (shader->constants.has(var->name) || var->is_const) {
			if (r_message) {
				*r_message = RTR("Constants cannot be modified.");
			}
			return false;
		}

		if (!(p_function_info.built_ins.has(var->name) && p_function_info.built_ins[var->name].constant)) {
			return true;
		}
	} else if (p_node->type == Node::TYPE_ARRAY) {
		ArrayNode *arr = static_cast<ArrayNode *>(p_node);

		if (shader->constants.has(arr->name) || arr->is_const) {
			if (r_message) {
				*r_message = RTR("Constants cannot be modified.");
			}
			return false;
		}

		return true;
	}

	if (r_message) {
		*r_message = "Assignment to constant expression.";
	}
	return false;
}

bool ShaderLanguage::_propagate_function_call_sampler_uniform_settings(StringName p_name, int p_argument, TextureFilter p_filter, TextureRepeat p_repeat) {
	for (int i = 0; i < shader->functions.size(); i++) {
		if (shader->functions[i].name == p_name) {
			ERR_FAIL_INDEX_V(p_argument, shader->functions[i].function->arguments.size(), false);
			FunctionNode::Argument *arg = &shader->functions[i].function->arguments.write[p_argument];
			if (arg->tex_builtin_check) {
				_set_error("Sampler argument #" + itos(p_argument) + " of function '" + String(p_name) + "' called more than once using both built-ins and uniform textures, this is not supported (use either one or the other).");
				return false;
			} else if (arg->tex_argument_check) {
				//was checked, verify that filter and repeat are the same
				if (arg->tex_argument_filter == p_filter && arg->tex_argument_repeat == p_repeat) {
					return true;
				} else {
					_set_error("Sampler argument #" + itos(p_argument) + " of function '" + String(p_name) + "' called more than once using textures that differ in either filter or repeat setting.");
					return false;
				}
			} else {
				arg->tex_argument_check = true;
				arg->tex_argument_filter = p_filter;
				arg->tex_argument_repeat = p_repeat;
				for (KeyValue<StringName, Set<int>> &E : arg->tex_argument_connect) {
					for (Set<int>::Element *F = E.value.front(); F; F = F->next()) {
						if (!_propagate_function_call_sampler_uniform_settings(E.key, F->get(), p_filter, p_repeat)) {
							return false;
						}
					}
				}
				return true;
			}
		}
	}
	ERR_FAIL_V(false); //bug? function not found
}

bool ShaderLanguage::_propagate_function_call_sampler_builtin_reference(StringName p_name, int p_argument, const StringName &p_builtin) {
	for (int i = 0; i < shader->functions.size(); i++) {
		if (shader->functions[i].name == p_name) {
			ERR_FAIL_INDEX_V(p_argument, shader->functions[i].function->arguments.size(), false);
			FunctionNode::Argument *arg = &shader->functions[i].function->arguments.write[p_argument];
			if (arg->tex_argument_check) {
				_set_error("Sampler argument #" + itos(p_argument) + " of function '" + String(p_name) + "' called more than once using both built-ins and uniform textures, this is not supported (use either one or the other).");
				return false;
			} else if (arg->tex_builtin_check) {
				//was checked, verify that the built-in is the same
				if (arg->tex_builtin == p_builtin) {
					return true;
				} else {
					_set_error("Sampler argument #" + itos(p_argument) + " of function '" + String(p_name) + "' called more than once using different built-ins. Only calling with the same built-in is supported.");
					return false;
				}
			} else {
				arg->tex_builtin_check = true;
				arg->tex_builtin = p_builtin;

				for (KeyValue<StringName, Set<int>> &E : arg->tex_argument_connect) {
					for (Set<int>::Element *F = E.value.front(); F; F = F->next()) {
						if (!_propagate_function_call_sampler_builtin_reference(E.key, F->get(), p_builtin)) {
							return false;
						}
					}
				}
				return true;
			}
		}
	}
	ERR_FAIL_V(false); //bug? function not found
}

ShaderLanguage::Node *ShaderLanguage::_parse_array_size(BlockNode *p_block, const FunctionInfo &p_function_info, int &r_array_size) {
	int array_size = 0;

	Node *n = _parse_and_reduce_expression(p_block, p_function_info);
	if (n) {
		if (n->type == Node::TYPE_VARIABLE) {
			VariableNode *vn = static_cast<VariableNode *>(n);
			if (vn) {
				ConstantNode::Value v;
				DataType data_type;
				bool is_const = false;

				_find_identifier(p_block, false, p_function_info, vn->name, &data_type, nullptr, &is_const, nullptr, nullptr, &v);

				if (is_const) {
					if (data_type == TYPE_INT) {
						int32_t value = v.sint;
						if (value > 0) {
							array_size = value;
						}
					} else if (data_type == TYPE_UINT) {
						uint32_t value = v.uint;
						if (value > 0U) {
							array_size = value;
						}
					}
				}
			}
		} else if (n->type == Node::TYPE_OPERATOR) {
			_set_error("Array size expressions are not yet implemented.");
			return nullptr;
		}
	}

	r_array_size = array_size;
	return n;
}

Error ShaderLanguage::_parse_global_array_size(int &r_array_size) {
	if (r_array_size > 0) {
		_set_error("Array size is already defined!");
		return ERR_PARSE_ERROR;
	}
	TkPos pos = _get_tkpos();
	Token tk = _get_token();

	int array_size = 0;

	if (tk.type != TK_INT_CONSTANT || ((int)tk.constant) <= 0) {
		_set_tkpos(pos);
		Node *n = _parse_array_size(nullptr, FunctionInfo(), array_size);
		if (!n) {
			return ERR_PARSE_ERROR;
		}
	} else if (((int)tk.constant) > 0) {
		array_size = (uint32_t)tk.constant;
	}

	if (array_size <= 0) {
		_set_error("Expected single integer constant > 0");
		return ERR_PARSE_ERROR;
	}

	tk = _get_token();
	if (tk.type != TK_BRACKET_CLOSE) {
		_set_error("Expected ']'");
		return ERR_PARSE_ERROR;
	}

	r_array_size = array_size;
	return OK;
}

Error ShaderLanguage::_parse_local_array_size(BlockNode *p_block, const FunctionInfo &p_function_info, ArrayDeclarationNode *p_node, ArrayDeclarationNode::Declaration *p_decl, int &r_array_size, bool &r_is_unknown_size) {
	TkPos pos = _get_tkpos();
	Token tk = _get_token();

	if (tk.type == TK_BRACKET_CLOSE) {
		r_is_unknown_size = true;
	} else {
		if (tk.type != TK_INT_CONSTANT || ((int)tk.constant) <= 0) {
			_set_tkpos(pos);
			int array_size = 0;
			Node *n = _parse_array_size(p_block, p_function_info, array_size);
			if (!n) {
				return ERR_PARSE_ERROR;
			}
			p_decl->size = array_size;
			p_node->size_expression = n;
		} else if (((int)tk.constant) > 0) {
			p_decl->size = (uint32_t)tk.constant;
		}

		if (p_decl->size <= 0) {
			_set_error("Expected single integer constant > 0");
			return ERR_PARSE_ERROR;
		}

		tk = _get_token();
		if (tk.type != TK_BRACKET_CLOSE) {
			_set_error("Expected ']'");
			return ERR_PARSE_ERROR;
		}

		r_array_size = p_decl->size;
	}

	return OK;
}

ShaderLanguage::Node *ShaderLanguage::_parse_array_constructor(BlockNode *p_block, const FunctionInfo &p_function_info) {
	DataType type = TYPE_VOID;
	String struct_name = "";
	int array_size = 0;
	bool auto_size = false;
	bool undefined_size = false;
	Token tk = _get_token();

	if (tk.type == TK_CURLY_BRACKET_OPEN) {
		auto_size = true;
	} else {
		if (shader->structs.has(tk.text)) {
			type = TYPE_STRUCT;
			struct_name = tk.text;
		} else {
			if (!is_token_variable_datatype(tk.type)) {
				_set_error("Invalid data type for array");
				return nullptr;
			}
			type = get_token_datatype(tk.type);
		}
		tk = _get_token();
		if (tk.type == TK_BRACKET_OPEN) {
			TkPos pos = _get_tkpos();
			tk = _get_token();
			if (tk.type == TK_BRACKET_CLOSE) {
				undefined_size = true;
				tk = _get_token();
			} else {
				_set_tkpos(pos);

				Node *n = _parse_and_reduce_expression(p_block, p_function_info);
				if (!n || n->type != Node::TYPE_CONSTANT || n->get_datatype() != TYPE_INT) {
					_set_error("Expected single integer constant > 0");
					return nullptr;
				}

				ConstantNode *cnode = (ConstantNode *)n;
				if (cnode->values.size() == 1) {
					array_size = cnode->values[0].sint;
					if (array_size <= 0) {
						_set_error("Expected single integer constant > 0");
						return nullptr;
					}
				} else {
					_set_error("Expected single integer constant > 0");
					return nullptr;
				}

				tk = _get_token();
				if (tk.type != TK_BRACKET_CLOSE) {
					_set_error("Expected ']'");
					return nullptr;
				} else {
					tk = _get_token();
				}
			}
		} else {
			_set_error("Expected '['");
			return nullptr;
		}
	}

	ArrayConstructNode *an = alloc_node<ArrayConstructNode>();

	if (tk.type == TK_PARENTHESIS_OPEN || auto_size) { // initialization
		int idx = 0;
		while (true) {
			Node *n = _parse_and_reduce_expression(p_block, p_function_info);
			if (!n) {
				return nullptr;
			}

			// define type by using the first member
			if (auto_size && idx == 0) {
				type = n->get_datatype();
				if (type == TYPE_STRUCT) {
					struct_name = n->get_datatype_name();
				}
			} else {
				if (!_compare_datatypes(type, struct_name, 0, n->get_datatype(), n->get_datatype_name(), 0)) {
					return nullptr;
				}
			}

			tk = _get_token();
			if (tk.type == TK_COMMA) {
				an->initializer.push_back(n);
			} else if (!auto_size && tk.type == TK_PARENTHESIS_CLOSE) {
				an->initializer.push_back(n);
				break;
			} else if (auto_size && tk.type == TK_CURLY_BRACKET_CLOSE) {
				an->initializer.push_back(n);
				break;
			} else {
				if (auto_size) {
					_set_error("Expected '}' or ','");
				} else {
					_set_error("Expected ')' or ','");
				}
				return nullptr;
			}
			idx++;
		}
		if (!auto_size && !undefined_size && an->initializer.size() != array_size) {
			_set_error("Array size mismatch");
			return nullptr;
		}
	} else {
		_set_error("Expected array initialization!");
		return nullptr;
	}

	an->datatype = type;
	an->struct_name = struct_name;
	return an;
}

ShaderLanguage::Node *ShaderLanguage::_parse_array_constructor(BlockNode *p_block, const FunctionInfo &p_function_info, DataType p_type, const StringName &p_struct_name, int p_array_size) {
	DataType type = TYPE_VOID;
	String struct_name = "";
	int array_size = 0;
	bool auto_size = false;
	TkPos prev_pos = _get_tkpos();
	Token tk = _get_token();

	if (tk.type == TK_CURLY_BRACKET_OPEN) {
		auto_size = true;
	} else {
		if (shader->structs.has(tk.text)) {
			type = TYPE_STRUCT;
			struct_name = tk.text;
		} else {
			if (!is_token_variable_datatype(tk.type)) {
				_set_tkpos(prev_pos);

				Node *n = _parse_and_reduce_expression(p_block, p_function_info);

				if (!n) {
					_set_error("Invalid data type for array");
					return nullptr;
				}

				if (!_compare_datatypes(p_type, p_struct_name, p_array_size, n->get_datatype(), n->get_datatype_name(), n->get_array_size())) {
					return nullptr;
				}
				return n;
			}
			type = get_token_datatype(tk.type);
		}
		tk = _get_token();
		if (tk.type == TK_BRACKET_OPEN) {
			TkPos pos = _get_tkpos();
			tk = _get_token();
			if (tk.type == TK_BRACKET_CLOSE) {
				array_size = p_array_size;
				tk = _get_token();
			} else {
				_set_tkpos(pos);

				Node *n = _parse_and_reduce_expression(p_block, p_function_info);
				if (!n || n->type != Node::TYPE_CONSTANT || n->get_datatype() != TYPE_INT) {
					_set_error("Expected single integer constant > 0");
					return nullptr;
				}

				ConstantNode *cnode = (ConstantNode *)n;
				if (cnode->values.size() == 1) {
					array_size = cnode->values[0].sint;
					if (array_size <= 0) {
						_set_error("Expected single integer constant > 0");
						return nullptr;
					}
				} else {
					_set_error("Expected single integer constant > 0");
					return nullptr;
				}

				tk = _get_token();
				if (tk.type != TK_BRACKET_CLOSE) {
					_set_error("Expected ']'");
					return nullptr;
				} else {
					tk = _get_token();
				}
			}
		} else {
			_set_error("Expected '['");
			return nullptr;
		}

		if (type != p_type || struct_name != p_struct_name || array_size != p_array_size) {
			String error_str = "Cannot convert from '";
			if (type == TYPE_STRUCT) {
				error_str += struct_name;
			} else {
				error_str += get_datatype_name(type);
			}
			error_str += "[";
			error_str += itos(array_size);
			error_str += "]'";
			error_str += " to '";
			if (type == TYPE_STRUCT) {
				error_str += p_struct_name;
			} else {
				error_str += get_datatype_name(p_type);
			}
			error_str += "[";
			error_str += itos(p_array_size);
			error_str += "]'";
			_set_error(error_str);
			return nullptr;
		}
	}

	ArrayConstructNode *an = alloc_node<ArrayConstructNode>();
	an->datatype = p_type;
	an->struct_name = p_struct_name;

	if (tk.type == TK_PARENTHESIS_OPEN || auto_size) { // initialization
		while (true) {
			Node *n = _parse_and_reduce_expression(p_block, p_function_info);
			if (!n) {
				return nullptr;
			}

			if (!_compare_datatypes(p_type, p_struct_name, 0, n->get_datatype(), n->get_datatype_name(), n->get_array_size())) {
				return nullptr;
			}

			tk = _get_token();
			if (tk.type == TK_COMMA) {
				an->initializer.push_back(n);
			} else if (!auto_size && tk.type == TK_PARENTHESIS_CLOSE) {
				an->initializer.push_back(n);
				break;
			} else if (auto_size && tk.type == TK_CURLY_BRACKET_CLOSE) {
				an->initializer.push_back(n);
				break;
			} else {
				if (auto_size) {
					_set_error("Expected '}' or ','");
				} else {
					_set_error("Expected ')' or ','");
				}
				return nullptr;
			}
		}
		if (an->initializer.size() != p_array_size) {
			_set_error("Array size mismatch");
			return nullptr;
		}
	} else {
		_set_error("Expected array initialization!");
		return nullptr;
	}

	return an;
}

ShaderLanguage::Node *ShaderLanguage::_parse_expression(BlockNode *p_block, const FunctionInfo &p_function_info) {
	Vector<Expression> expression;

	//Vector<TokenType> operators;

	while (true) {
		Node *expr = nullptr;
		TkPos prepos = _get_tkpos();
		Token tk = _get_token();
		TkPos pos = _get_tkpos();

		bool is_const = false;

		if (tk.type == TK_PARENTHESIS_OPEN) {
			//handle subexpression

			expr = _parse_and_reduce_expression(p_block, p_function_info);
			if (!expr) {
				return nullptr;
			}

			tk = _get_token();

			if (tk.type != TK_PARENTHESIS_CLOSE) {
				_set_error("Expected ')' in expression");
				return nullptr;
			}

		} else if (tk.type == TK_FLOAT_CONSTANT) {
			ConstantNode *constant = alloc_node<ConstantNode>();
			ConstantNode::Value v;
			v.real = tk.constant;
			constant->values.push_back(v);
			constant->datatype = TYPE_FLOAT;
			expr = constant;

		} else if (tk.type == TK_INT_CONSTANT) {
			ConstantNode *constant = alloc_node<ConstantNode>();
			ConstantNode::Value v;
			v.sint = tk.constant;
			constant->values.push_back(v);
			constant->datatype = TYPE_INT;
			expr = constant;

		} else if (tk.type == TK_TRUE) {
			//handle true constant
			ConstantNode *constant = alloc_node<ConstantNode>();
			ConstantNode::Value v;
			v.boolean = true;
			constant->values.push_back(v);
			constant->datatype = TYPE_BOOL;
			expr = constant;

		} else if (tk.type == TK_FALSE) {
			//handle false constant
			ConstantNode *constant = alloc_node<ConstantNode>();
			ConstantNode::Value v;
			v.boolean = false;
			constant->values.push_back(v);
			constant->datatype = TYPE_BOOL;
			expr = constant;

		} else if (tk.type == TK_TYPE_VOID) {
			//make sure void is not used in expression
			_set_error("Void value not allowed in Expression");
			return nullptr;
		} else if (is_token_nonvoid_datatype(tk.type) || tk.type == TK_CURLY_BRACKET_OPEN) {
			if (tk.type == TK_CURLY_BRACKET_OPEN) {
				//array constructor

				_set_tkpos(prepos);
				expr = _parse_array_constructor(p_block, p_function_info);
			} else {
				DataType datatype;
				DataPrecision precision;
				bool precision_defined = false;

				if (is_token_precision(tk.type)) {
					precision = get_token_precision(tk.type);
					precision_defined = true;
					tk = _get_token();
				}

				datatype = get_token_datatype(tk.type);
				tk = _get_token();

				if (tk.type == TK_BRACKET_OPEN) {
					//array constructor

					_set_tkpos(prepos);
					expr = _parse_array_constructor(p_block, p_function_info);
				} else {
					if (tk.type != TK_PARENTHESIS_OPEN) {
						_set_error("Expected '(' after type name");
						return nullptr;
					}
					//basic type constructor

					OperatorNode *func = alloc_node<OperatorNode>();
					func->op = OP_CONSTRUCT;

					if (precision_defined) {
						func->return_precision_cache = precision;
					}

					VariableNode *funcname = alloc_node<VariableNode>();
					funcname->name = get_datatype_name(datatype);
					func->arguments.push_back(funcname);

					int carg = -1;

					bool ok = _parse_function_arguments(p_block, p_function_info, func, &carg);

					if (carg >= 0) {
						completion_type = COMPLETION_CALL_ARGUMENTS;
						completion_line = tk_line;
						completion_block = p_block;
						completion_function = funcname->name;
						completion_argument = carg;
					}

					if (!ok) {
						return nullptr;
					}

					if (!_validate_function_call(p_block, p_function_info, func, &func->return_cache, &func->struct_name)) {
						_set_error("No matching constructor found for: '" + String(funcname->name) + "'");
						return nullptr;
					}

					expr = _reduce_expression(p_block, func);
				}
			}
		} else if (tk.type == TK_IDENTIFIER) {
			_set_tkpos(prepos);

			StringName identifier;

			StructNode *pstruct = nullptr;
			bool struct_init = false;

			_get_completable_identifier(p_block, COMPLETION_IDENTIFIER, identifier);

			if (shader->structs.has(identifier)) {
				pstruct = shader->structs[identifier].shader_struct;
#ifdef DEBUG_ENABLED
				if (check_warnings && HAS_WARNING(ShaderWarning::UNUSED_STRUCT_FLAG) && used_structs.has(identifier)) {
					used_structs[identifier].used = true;
				}
#endif // DEBUG_ENABLED
				struct_init = true;
			}

			tk = _get_token();
			if (tk.type == TK_PARENTHESIS_OPEN) {
				if (struct_init) { //a struct constructor

					const StringName &name = identifier;

					OperatorNode *func = alloc_node<OperatorNode>();
					func->op = OP_STRUCT;
					func->struct_name = name;
					func->return_cache = TYPE_STRUCT;
					VariableNode *funcname = alloc_node<VariableNode>();
					funcname->name = name;
					func->arguments.push_back(funcname);

					for (int i = 0; i < pstruct->members.size(); i++) {
						Node *nexpr;

						if (pstruct->members[i]->array_size != 0) {
							nexpr = _parse_array_constructor(p_block, p_function_info, pstruct->members[i]->get_datatype(), pstruct->members[i]->struct_name, pstruct->members[i]->array_size);
							if (!nexpr) {
								return nullptr;
							}
						} else {
							nexpr = _parse_and_reduce_expression(p_block, p_function_info);
							if (!nexpr) {
								return nullptr;
							}
							if (!_compare_datatypes_in_nodes(pstruct->members[i], nexpr)) {
								return nullptr;
							}
						}

						if (i + 1 < pstruct->members.size()) {
							tk = _get_token();
							if (tk.type != TK_COMMA) {
								_set_error("Expected ','");
								return nullptr;
							}
						}
						func->arguments.push_back(nexpr);
					}
					tk = _get_token();
					if (tk.type != TK_PARENTHESIS_CLOSE) {
						_set_error("Expected ')'");
						return nullptr;
					}

					expr = func;

				} else { //a function call

					const StringName &name = identifier;

					OperatorNode *func = alloc_node<OperatorNode>();
					func->op = OP_CALL;
					VariableNode *funcname = alloc_node<VariableNode>();
					funcname->name = name;
					func->arguments.push_back(funcname);

					int carg = -1;

					bool ok = _parse_function_arguments(p_block, p_function_info, func, &carg);

					// Check if block has a variable with the same name as function to prevent shader crash.
					ShaderLanguage::BlockNode *bnode = p_block;
					while (bnode) {
						if (bnode->variables.has(name)) {
							_set_error("Expected function name");
							return nullptr;
						}
						bnode = bnode->parent_block;
					}

					//test if function was parsed first
					int function_index = -1;
					for (int i = 0; i < shader->functions.size(); i++) {
						if (shader->functions[i].name == name) {
							//add to current function as dependency
							for (int j = 0; j < shader->functions.size(); j++) {
								if (shader->functions[j].name == current_function) {
									shader->functions.write[j].uses_function.insert(name);
									break;
								}
							}

							//see if texture arguments must connect
							function_index = i;
							break;
						}
					}

					if (carg >= 0) {
						completion_type = COMPLETION_CALL_ARGUMENTS;
						completion_line = tk_line;
						completion_block = p_block;
						completion_function = funcname->name;
						completion_argument = carg;
					}

					if (!ok) {
						return nullptr;
					}

					if (!_validate_function_call(p_block, p_function_info, func, &func->return_cache, &func->struct_name)) {
						_set_error("No matching function found for: '" + String(funcname->name) + "'");
						return nullptr;
					}
					completion_class = TAG_GLOBAL; // reset sub-class
					if (function_index >= 0) {
						//connect texture arguments, so we can cache in the
						//argument what type of filter and repeat to use

						FunctionNode *call_function = shader->functions[function_index].function;
						if (call_function) {
							func->return_cache = call_function->get_datatype();
							func->struct_name = call_function->get_datatype_name();
							func->return_array_size = call_function->get_array_size();

							//get current base function
							FunctionNode *base_function = nullptr;
							{
								BlockNode *b = p_block;

								while (b) {
									if (b->parent_function) {
										base_function = b->parent_function;
										break;
									} else {
										b = b->parent_block;
									}
								}
							}

							ERR_FAIL_COND_V(!base_function, nullptr); //bug, wtf

							for (int i = 0; i < call_function->arguments.size(); i++) {
								int argidx = i + 1;
								if (argidx < func->arguments.size()) {
									if (call_function->arguments[i].is_const || call_function->arguments[i].qualifier == ArgumentQualifier::ARGUMENT_QUALIFIER_OUT || call_function->arguments[i].qualifier == ArgumentQualifier::ARGUMENT_QUALIFIER_INOUT) {
										bool error = false;
										Node *n = func->arguments[argidx];
										if (n->type == Node::TYPE_CONSTANT || n->type == Node::TYPE_OPERATOR) {
											error = true;
										} else if (n->type == Node::TYPE_ARRAY) {
											ArrayNode *an = static_cast<ArrayNode *>(n);
											if (an->call_expression != nullptr || an->is_const) {
												error = true;
											}
										} else if (n->type == Node::TYPE_VARIABLE) {
											VariableNode *vn = static_cast<VariableNode *>(n);
											if (vn->is_const) {
												error = true;
											} else {
												StringName varname = vn->name;
												if (shader->constants.has(varname)) {
													error = true;
												} else if (shader->uniforms.has(varname)) {
													error = true;
												} else {
													if (shader->varyings.has(varname)) {
														_set_error(vformat("Varyings cannot be passed for '%s' parameter!", _get_qualifier_str(call_function->arguments[i].qualifier)));
														return nullptr;
													}
													if (p_function_info.built_ins.has(varname)) {
														BuiltInInfo info = p_function_info.built_ins[varname];
														if (info.constant) {
															error = true;
														}
													}
												}
											}
										} else if (n->type == Node::TYPE_MEMBER) {
											MemberNode *mn = static_cast<MemberNode *>(n);
											if (mn->basetype_const) {
												error = true;
											}
										}
										if (error) {
											_set_error(vformat("Constant value cannot be passed for '%s' parameter!", _get_qualifier_str(call_function->arguments[i].qualifier)));
											return nullptr;
										}
									}
									if (is_sampler_type(call_function->arguments[i].type)) {
										//let's see where our argument comes from
										Node *n = func->arguments[argidx];
										ERR_CONTINUE(n->type != Node::TYPE_VARIABLE); //bug? this should always be a variable
										VariableNode *vn = static_cast<VariableNode *>(n);
										StringName varname = vn->name;
										if (shader->uniforms.has(varname)) {
											//being sampler, this either comes from a uniform
											ShaderNode::Uniform *u = &shader->uniforms[varname];
											ERR_CONTINUE(u->type != call_function->arguments[i].type); //this should have been validated previously
											//propagate
											if (!_propagate_function_call_sampler_uniform_settings(name, i, u->filter, u->repeat)) {
												return nullptr;
											}
										} else if (p_function_info.built_ins.has(varname)) {
											//a built-in
											if (!_propagate_function_call_sampler_builtin_reference(name, i, varname)) {
												return nullptr;
											}
										} else {
											//or this comes from an argument, but nothing else can be a sampler
											bool found = false;
											for (int j = 0; j < base_function->arguments.size(); j++) {
												if (base_function->arguments[j].name == varname) {
													if (!base_function->arguments[j].tex_argument_connect.has(call_function->name)) {
														base_function->arguments.write[j].tex_argument_connect[call_function->name] = Set<int>();
													}
													base_function->arguments.write[j].tex_argument_connect[call_function->name].insert(i);
													found = true;
													break;
												}
											}
											ERR_CONTINUE(!found);
										}
									}
								} else {
									break;
								}
							}
						}
					}
					expr = func;
#ifdef DEBUG_ENABLED
					if (check_warnings) {
						StringName func_name;

						if (p_block && p_block->parent_function) {
							func_name = p_block->parent_function->name;
						}

						_parse_used_identifier(name, IdentifierType::IDENTIFIER_FUNCTION, func_name);
					}
#endif // DEBUG_ENABLED
				}
			} else {
				//an identifier

				last_name = identifier;
				last_type = IDENTIFIER_MAX;
				_set_tkpos(pos);

				DataType data_type;
				IdentifierType ident_type;
				int array_size = 0;
				StringName struct_name;
				bool is_local = false;

				if (p_block && p_block->block_tag != SubClassTag::TAG_GLOBAL) {
					int idx = 0;
					bool found = false;

					while (builtin_func_defs[idx].name) {
						if (builtin_func_defs[idx].tag == p_block->block_tag && builtin_func_defs[idx].name == identifier) {
							found = true;
							break;
						}
						idx++;
					}
					if (!found) {
						_set_error("Unknown identifier in expression: " + String(identifier));
						return nullptr;
					}
				} else {
					if (!_find_identifier(p_block, false, p_function_info, identifier, &data_type, &ident_type, &is_const, &array_size, &struct_name)) {
						_set_error("Unknown identifier in expression: " + String(identifier));
						return nullptr;
					}
					if (ident_type == IDENTIFIER_VARYING) {
						TkPos prev_pos = _get_tkpos();
						Token next_token = _get_token();

						// An array of varyings.
						if (next_token.type == TK_BRACKET_OPEN) {
							_get_token(); // Pass constant.
							_get_token(); // Pass TK_BRACKET_CLOSE.
							next_token = _get_token();
						}
						_set_tkpos(prev_pos);

						String error;
						if (is_token_operator_assign(next_token.type)) {
							if (!_validate_varying_assign(shader->varyings[identifier], &error)) {
								_set_error(error);
								return nullptr;
							}
						} else {
							if (!_validate_varying_using(shader->varyings[identifier], &error)) {
								_set_error(error);
								return nullptr;
							}
						}
					}

					if (ident_type == IDENTIFIER_FUNCTION) {
						_set_error("Can't use function as identifier: " + String(identifier));
						return nullptr;
					}
					if (is_const) {
						last_type = IDENTIFIER_CONSTANT;
					} else {
						last_type = ident_type;
					}

					is_local = ident_type == IDENTIFIER_LOCAL_VAR || ident_type == IDENTIFIER_FUNCTION_ARGUMENT;
				}

				Node *index_expression = nullptr;
				Node *call_expression = nullptr;
				Node *assign_expression = nullptr;

				if (array_size > 0) {
					prepos = _get_tkpos();
					tk = _get_token();

					if (tk.type == TK_OP_ASSIGN) {
						if (is_const) {
							_set_error("Constants cannot be modified.");
							return nullptr;
						}
						assign_expression = _parse_array_constructor(p_block, p_function_info, data_type, struct_name, array_size);
						if (!assign_expression) {
							return nullptr;
						}
					} else if (tk.type == TK_PERIOD) {
						completion_class = TAG_ARRAY;
						p_block->block_tag = SubClassTag::TAG_ARRAY;
						call_expression = _parse_and_reduce_expression(p_block, p_function_info);
						p_block->block_tag = SubClassTag::TAG_GLOBAL;
						if (!call_expression) {
							return nullptr;
						}
						data_type = call_expression->get_datatype();
					} else if (tk.type == TK_BRACKET_OPEN) { // indexing
						index_expression = _parse_and_reduce_expression(p_block, p_function_info);
						if (!index_expression) {
							return nullptr;
						}

						if (index_expression->get_array_size() != 0 || (index_expression->get_datatype() != TYPE_INT && index_expression->get_datatype() != TYPE_UINT)) {
							_set_error("Only integer expressions are allowed for indexing.");
							return nullptr;
						}

						if (index_expression->type == Node::TYPE_CONSTANT) {
							ConstantNode *cnode = (ConstantNode *)index_expression;
							if (cnode) {
								if (!cnode->values.is_empty()) {
									int value = cnode->values[0].sint;
									if (value < 0 || value >= array_size) {
										_set_error(vformat("Index [%s] out of range [%s..%s]", value, 0, array_size - 1));
										return nullptr;
									}
								}
							}
						}

						tk = _get_token();
						if (tk.type != TK_BRACKET_CLOSE) {
							_set_error("Expected ']'");
							return nullptr;
						}
					} else {
						_set_tkpos(prepos);
					}

					ArrayNode *arrname = alloc_node<ArrayNode>();
					arrname->name = identifier;
					arrname->datatype_cache = data_type;
					arrname->struct_name = struct_name;
					arrname->index_expression = index_expression;
					arrname->call_expression = call_expression;
					arrname->assign_expression = assign_expression;
					arrname->is_const = is_const;
					arrname->array_size = array_size;
					arrname->is_local = is_local;
					expr = arrname;
				} else {
					VariableNode *varname = alloc_node<VariableNode>();
					varname->name = identifier;
					varname->datatype_cache = data_type;
					varname->is_const = is_const;
					varname->struct_name = struct_name;
					varname->is_local = is_local;
					expr = varname;
				}
#ifdef DEBUG_ENABLED
				if (check_warnings) {
					StringName func_name;

					if (p_block && p_block->parent_function) {
						func_name = p_block->parent_function->name;
					}

					_parse_used_identifier(identifier, ident_type, func_name);
				}
#endif // DEBUG_ENABLED
			}
		} else if (tk.type == TK_OP_ADD) {
			continue; //this one does nothing
		} else if (tk.type == TK_OP_SUB || tk.type == TK_OP_NOT || tk.type == TK_OP_BIT_INVERT || tk.type == TK_OP_INCREMENT || tk.type == TK_OP_DECREMENT) {
			Expression e;
			e.is_op = true;

			switch (tk.type) {
				case TK_OP_SUB:
					e.op = OP_NEGATE;
					break;
				case TK_OP_NOT:
					e.op = OP_NOT;
					break;
				case TK_OP_BIT_INVERT:
					e.op = OP_BIT_INVERT;
					break;
				case TK_OP_INCREMENT:
					e.op = OP_INCREMENT;
					break;
				case TK_OP_DECREMENT:
					e.op = OP_DECREMENT;
					break;
				default:
					ERR_FAIL_V(nullptr);
			}

			expression.push_back(e);
			continue;
		} else {
			_set_error("Expected expression, found: " + get_token_text(tk));
			return nullptr;
			//nothing
		}

		ERR_FAIL_COND_V(!expr, nullptr);

		/* OK now see what's NEXT to the operator.. */

		while (true) {
			TkPos pos2 = _get_tkpos();
			tk = _get_token();

			if (tk.type == TK_CURSOR) {
				//do nothing
			} else if (tk.type == TK_IDENTIFIER) {
			} else if (tk.type == TK_PERIOD) {
				DataType dt = expr->get_datatype();
				String st = expr->get_datatype_name();

				if (!expr->is_indexed() && expr->get_array_size() > 0) {
					completion_class = TAG_ARRAY;
					p_block->block_tag = SubClassTag::TAG_ARRAY;
					Node *call_expression = _parse_and_reduce_expression(p_block, p_function_info);
					p_block->block_tag = SubClassTag::TAG_GLOBAL;
					if (!call_expression) {
						return nullptr;
					}
					expr = call_expression;
					break;
				}

				StringName identifier;
				if (_get_completable_identifier(p_block, dt == TYPE_STRUCT ? COMPLETION_STRUCT : COMPLETION_INDEX, identifier)) {
					if (dt == TYPE_STRUCT) {
						completion_struct = st;
					} else {
						completion_base = dt;
					}
				}

				if (identifier == StringName()) {
					_set_error("Expected identifier as member");
					return nullptr;
				}
				String ident = identifier;

				bool ok = true;
				bool repeated = false;
				DataType member_type = TYPE_VOID;
				StringName member_struct_name = "";
				int array_size = 0;

				Set<char> position_symbols;
				Set<char> color_symbols;
				Set<char> texture_symbols;

				bool mix_error = false;

				switch (dt) {
					case TYPE_STRUCT: {
						ok = false;
						String member_name = String(ident.ptr());
						if (shader->structs.has(st)) {
							StructNode *n = shader->structs[st].shader_struct;
							for (const MemberNode *E : n->members) {
								if (String(E->name) == member_name) {
									member_type = E->datatype;
									array_size = E->array_size;
									if (member_type == TYPE_STRUCT) {
										member_struct_name = E->struct_name;
									}
									ok = true;
									break;
								}
							}
						}

					} break;
					case TYPE_BVEC2:
					case TYPE_IVEC2:
					case TYPE_UVEC2:
					case TYPE_VEC2: {
						int l = ident.length();
						if (l == 1) {
							member_type = DataType(dt - 1);
						} else if (l == 2) {
							member_type = dt;
						} else if (l == 3) {
							member_type = DataType(dt + 1);
						} else if (l == 4) {
							member_type = DataType(dt + 2);
						} else {
							ok = false;
							break;
						}

						const char32_t *c = ident.ptr();
						for (int i = 0; i < l; i++) {
							switch (c[i]) {
								case 'r':
								case 'g':
									if (position_symbols.size() > 0 || texture_symbols.size() > 0) {
										mix_error = true;
										break;
									}
									if (!color_symbols.has(c[i])) {
										color_symbols.insert(c[i]);
									} else {
										repeated = true;
									}
									break;
								case 'x':
								case 'y':
									if (color_symbols.size() > 0 || texture_symbols.size() > 0) {
										mix_error = true;
										break;
									}
									if (!position_symbols.has(c[i])) {
										position_symbols.insert(c[i]);
									} else {
										repeated = true;
									}
									break;
								case 's':
								case 't':
									if (color_symbols.size() > 0 || position_symbols.size() > 0) {
										mix_error = true;
										break;
									}
									if (!texture_symbols.has(c[i])) {
										texture_symbols.insert(c[i]);
									} else {
										repeated = true;
									}
									break;
								default:
									ok = false;
									break;
							}
						}

					} break;
					case TYPE_BVEC3:
					case TYPE_IVEC3:
					case TYPE_UVEC3:
					case TYPE_VEC3: {
						int l = ident.length();
						if (l == 1) {
							member_type = DataType(dt - 2);
						} else if (l == 2) {
							member_type = DataType(dt - 1);
						} else if (l == 3) {
							member_type = dt;
						} else if (l == 4) {
							member_type = DataType(dt + 1);
						} else {
							ok = false;
							break;
						}

						const char32_t *c = ident.ptr();
						for (int i = 0; i < l; i++) {
							switch (c[i]) {
								case 'r':
								case 'g':
								case 'b':
									if (position_symbols.size() > 0 || texture_symbols.size() > 0) {
										mix_error = true;
										break;
									}
									if (!color_symbols.has(c[i])) {
										color_symbols.insert(c[i]);
									} else {
										repeated = true;
									}
									break;
								case 'x':
								case 'y':
								case 'z':
									if (color_symbols.size() > 0 || texture_symbols.size() > 0) {
										mix_error = true;
										break;
									}
									if (!position_symbols.has(c[i])) {
										position_symbols.insert(c[i]);
									} else {
										repeated = true;
									}
									break;
								case 's':
								case 't':
								case 'p':
									if (color_symbols.size() > 0 || position_symbols.size() > 0) {
										mix_error = true;
										break;
									}
									if (!texture_symbols.has(c[i])) {
										texture_symbols.insert(c[i]);
									} else {
										repeated = true;
									}
									break;
								default:
									ok = false;
									break;
							}
						}

					} break;
					case TYPE_BVEC4:
					case TYPE_IVEC4:
					case TYPE_UVEC4:
					case TYPE_VEC4: {
						int l = ident.length();
						if (l == 1) {
							member_type = DataType(dt - 3);
						} else if (l == 2) {
							member_type = DataType(dt - 2);
						} else if (l == 3) {
							member_type = DataType(dt - 1);
						} else if (l == 4) {
							member_type = dt;
						} else {
							ok = false;
							break;
						}

						const char32_t *c = ident.ptr();
						for (int i = 0; i < l; i++) {
							switch (c[i]) {
								case 'r':
								case 'g':
								case 'b':
								case 'a':
									if (position_symbols.size() > 0 || texture_symbols.size() > 0) {
										mix_error = true;
										break;
									}
									if (!color_symbols.has(c[i])) {
										color_symbols.insert(c[i]);
									} else {
										repeated = true;
									}
									break;
								case 'x':
								case 'y':
								case 'z':
								case 'w':
									if (color_symbols.size() > 0 || texture_symbols.size() > 0) {
										mix_error = true;
										break;
									}
									if (!position_symbols.has(c[i])) {
										position_symbols.insert(c[i]);
									} else {
										repeated = true;
									}
									break;
								case 's':
								case 't':
								case 'p':
								case 'q':
									if (color_symbols.size() > 0 || position_symbols.size() > 0) {
										mix_error = true;
										break;
									}
									if (!texture_symbols.has(c[i])) {
										texture_symbols.insert(c[i]);
									} else {
										repeated = true;
									}
									break;
								default:
									ok = false;
									break;
							}
						}

					} break;

					default: {
						ok = false;
					}
				}

				if (mix_error) {
					_set_error("Cannot combine symbols from different sets in expression ." + ident);
					return nullptr;
				}

				if (!ok) {
					_set_error("Invalid member for " + (dt == TYPE_STRUCT ? st : get_datatype_name(dt)) + " expression: ." + ident);
					return nullptr;
				}

				MemberNode *mn = alloc_node<MemberNode>();
				mn->basetype = dt;
				mn->basetype_const = is_const;
				mn->datatype = member_type;
				mn->base_struct_name = st;
				mn->struct_name = member_struct_name;
				mn->array_size = array_size;
				mn->name = ident;
				mn->owner = expr;
				mn->has_swizzling_duplicates = repeated;

				if (array_size > 0) {
					TkPos prev_pos = _get_tkpos();
					tk = _get_token();
					if (tk.type == TK_OP_ASSIGN) {
						if (last_type == IDENTIFIER_CONSTANT) {
							_set_error("Constants cannot be modified.");
							return nullptr;
						}
						Node *assign_expression = _parse_array_constructor(p_block, p_function_info, member_type, member_struct_name, array_size);
						if (!assign_expression) {
							return nullptr;
						}
						mn->assign_expression = assign_expression;
					} else if (tk.type == TK_PERIOD) {
						completion_class = TAG_ARRAY;
						p_block->block_tag = SubClassTag::TAG_ARRAY;
						Node *call_expression = _parse_and_reduce_expression(p_block, p_function_info);
						p_block->block_tag = SubClassTag::TAG_GLOBAL;
						if (!call_expression) {
							return nullptr;
						}
						mn->datatype = call_expression->get_datatype();
						mn->call_expression = call_expression;
					} else if (tk.type == TK_BRACKET_OPEN) {
						Node *index_expression = _parse_and_reduce_expression(p_block, p_function_info);
						if (!index_expression) {
							return nullptr;
						}

						if (index_expression->get_array_size() != 0 || (index_expression->get_datatype() != TYPE_INT && index_expression->get_datatype() != TYPE_UINT)) {
							_set_error("Only integer expressions are allowed for indexing.");
							return nullptr;
						}

						if (index_expression->type == Node::TYPE_CONSTANT) {
							ConstantNode *cnode = (ConstantNode *)index_expression;
							if (cnode) {
								if (!cnode->values.is_empty()) {
									int value = cnode->values[0].sint;
									if (value < 0 || value >= array_size) {
										_set_error(vformat("Index [%s] out of range [%s..%s]", value, 0, array_size - 1));
										return nullptr;
									}
								}
							}
						}

						tk = _get_token();
						if (tk.type != TK_BRACKET_CLOSE) {
							_set_error("Expected ']'");
							return nullptr;
						}
						mn->index_expression = index_expression;
					} else {
						_set_tkpos(prev_pos);
					}
				}
				expr = mn;

				//todo
				//member (period) has priority over any operator
				//creates a subindexing expression in place

				/*} else if (tk.type==TK_BRACKET_OPEN) {
				//todo
				//subindexing has priority over any operator
				//creates a subindexing expression in place

	*/
			} else if (tk.type == TK_BRACKET_OPEN) {
				Node *index = _parse_and_reduce_expression(p_block, p_function_info);
				if (!index) {
					return nullptr;
				}

				if (index->get_array_size() != 0 || (index->get_datatype() != TYPE_INT && index->get_datatype() != TYPE_UINT)) {
					_set_error("Only integer expressions are allowed for indexing.");
					return nullptr;
				}

				DataType member_type = TYPE_VOID;
				String member_struct_name;

				if (expr->get_array_size() > 0) {
					if (index->type == Node::TYPE_CONSTANT) {
						uint32_t index_constant = static_cast<ConstantNode *>(index)->values[0].uint;
						if (index_constant >= (uint32_t)expr->get_array_size()) {
							_set_error(vformat("Index [%s] out of range [%s..%s]", index_constant, 0, expr->get_array_size() - 1));
							return nullptr;
						}
					}
					member_type = expr->get_datatype();
					if (member_type == TYPE_STRUCT) {
						member_struct_name = expr->get_datatype_name();
					}
				} else {
					switch (expr->get_datatype()) {
						case TYPE_BVEC2:
						case TYPE_VEC2:
						case TYPE_IVEC2:
						case TYPE_UVEC2:
						case TYPE_MAT2:
							if (index->type == Node::TYPE_CONSTANT) {
								uint32_t index_constant = static_cast<ConstantNode *>(index)->values[0].uint;
								if (index_constant >= 2) {
									_set_error("Index out of range (0-1)");
									return nullptr;
								}
							}

							switch (expr->get_datatype()) {
								case TYPE_BVEC2:
									member_type = TYPE_BOOL;
									break;
								case TYPE_VEC2:
									member_type = TYPE_FLOAT;
									break;
								case TYPE_IVEC2:
									member_type = TYPE_INT;
									break;
								case TYPE_UVEC2:
									member_type = TYPE_UINT;
									break;
								case TYPE_MAT2:
									member_type = TYPE_VEC2;
									break;
								default:
									break;
							}

							break;
						case TYPE_BVEC3:
						case TYPE_VEC3:
						case TYPE_IVEC3:
						case TYPE_UVEC3:
						case TYPE_MAT3:
							if (index->type == Node::TYPE_CONSTANT) {
								uint32_t index_constant = static_cast<ConstantNode *>(index)->values[0].uint;
								if (index_constant >= 3) {
									_set_error("Index out of range (0-2)");
									return nullptr;
								}
							}

							switch (expr->get_datatype()) {
								case TYPE_BVEC3:
									member_type = TYPE_BOOL;
									break;
								case TYPE_VEC3:
									member_type = TYPE_FLOAT;
									break;
								case TYPE_IVEC3:
									member_type = TYPE_INT;
									break;
								case TYPE_UVEC3:
									member_type = TYPE_UINT;
									break;
								case TYPE_MAT3:
									member_type = TYPE_VEC3;
									break;
								default:
									break;
							}
							break;
						case TYPE_BVEC4:
						case TYPE_VEC4:
						case TYPE_IVEC4:
						case TYPE_UVEC4:
						case TYPE_MAT4:
							if (index->type == Node::TYPE_CONSTANT) {
								uint32_t index_constant = static_cast<ConstantNode *>(index)->values[0].uint;
								if (index_constant >= 4) {
									_set_error("Index out of range (0-3)");
									return nullptr;
								}
							}

							switch (expr->get_datatype()) {
								case TYPE_BVEC4:
									member_type = TYPE_BOOL;
									break;
								case TYPE_VEC4:
									member_type = TYPE_FLOAT;
									break;
								case TYPE_IVEC4:
									member_type = TYPE_INT;
									break;
								case TYPE_UVEC4:
									member_type = TYPE_UINT;
									break;
								case TYPE_MAT4:
									member_type = TYPE_VEC4;
									break;
								default:
									break;
							}
							break;
						default: {
							_set_error("Object of type '" + (expr->get_datatype() == TYPE_STRUCT ? expr->get_datatype_name() : get_datatype_name(expr->get_datatype())) + "' can't be indexed");
							return nullptr;
						}
					}
				}
				OperatorNode *op = alloc_node<OperatorNode>();
				op->op = OP_INDEX;
				op->return_cache = member_type;
				op->struct_name = member_struct_name;
				op->arguments.push_back(expr);
				op->arguments.push_back(index);
				expr = op;

				tk = _get_token();
				if (tk.type != TK_BRACKET_CLOSE) {
					_set_error("Expected ']' after indexing expression");
					return nullptr;
				}

			} else if (tk.type == TK_OP_INCREMENT || tk.type == TK_OP_DECREMENT) {
				OperatorNode *op = alloc_node<OperatorNode>();
				op->op = tk.type == TK_OP_DECREMENT ? OP_POST_DECREMENT : OP_POST_INCREMENT;
				op->arguments.push_back(expr);

				if (!_validate_operator(op, &op->return_cache, &op->return_array_size)) {
					_set_error("Invalid base type for increment/decrement operator");
					return nullptr;
				}

				if (!_validate_assign(expr, p_function_info)) {
					_set_error("Invalid use of increment/decrement operator in constant expression.");
					return nullptr;
				}
				expr = op;
			} else {
				_set_tkpos(pos2);
				break;
			}
		}

		Expression e;
		e.is_op = false;
		e.node = expr;
		expression.push_back(e);

		pos = _get_tkpos();
		tk = _get_token();

		if (is_token_operator(tk.type)) {
			Expression o;
			o.is_op = true;

			switch (tk.type) {
				case TK_OP_EQUAL:
					o.op = OP_EQUAL;
					break;
				case TK_OP_NOT_EQUAL:
					o.op = OP_NOT_EQUAL;
					break;
				case TK_OP_LESS:
					o.op = OP_LESS;
					break;
				case TK_OP_LESS_EQUAL:
					o.op = OP_LESS_EQUAL;
					break;
				case TK_OP_GREATER:
					o.op = OP_GREATER;
					break;
				case TK_OP_GREATER_EQUAL:
					o.op = OP_GREATER_EQUAL;
					break;
				case TK_OP_AND:
					o.op = OP_AND;
					break;
				case TK_OP_OR:
					o.op = OP_OR;
					break;
				case TK_OP_ADD:
					o.op = OP_ADD;
					break;
				case TK_OP_SUB:
					o.op = OP_SUB;
					break;
				case TK_OP_MUL:
					o.op = OP_MUL;
					break;
				case TK_OP_DIV:
					o.op = OP_DIV;
					break;
				case TK_OP_MOD:
					o.op = OP_MOD;
					break;
				case TK_OP_SHIFT_LEFT:
					o.op = OP_SHIFT_LEFT;
					break;
				case TK_OP_SHIFT_RIGHT:
					o.op = OP_SHIFT_RIGHT;
					break;
				case TK_OP_ASSIGN:
					o.op = OP_ASSIGN;
					break;
				case TK_OP_ASSIGN_ADD:
					o.op = OP_ASSIGN_ADD;
					break;
				case TK_OP_ASSIGN_SUB:
					o.op = OP_ASSIGN_SUB;
					break;
				case TK_OP_ASSIGN_MUL:
					o.op = OP_ASSIGN_MUL;
					break;
				case TK_OP_ASSIGN_DIV:
					o.op = OP_ASSIGN_DIV;
					break;
				case TK_OP_ASSIGN_MOD:
					o.op = OP_ASSIGN_MOD;
					break;
				case TK_OP_ASSIGN_SHIFT_LEFT:
					o.op = OP_ASSIGN_SHIFT_LEFT;
					break;
				case TK_OP_ASSIGN_SHIFT_RIGHT:
					o.op = OP_ASSIGN_SHIFT_RIGHT;
					break;
				case TK_OP_ASSIGN_BIT_AND:
					o.op = OP_ASSIGN_BIT_AND;
					break;
				case TK_OP_ASSIGN_BIT_OR:
					o.op = OP_ASSIGN_BIT_OR;
					break;
				case TK_OP_ASSIGN_BIT_XOR:
					o.op = OP_ASSIGN_BIT_XOR;
					break;
				case TK_OP_BIT_AND:
					o.op = OP_BIT_AND;
					break;
				case TK_OP_BIT_OR:
					o.op = OP_BIT_OR;
					break;
				case TK_OP_BIT_XOR:
					o.op = OP_BIT_XOR;
					break;
				case TK_QUESTION:
					o.op = OP_SELECT_IF;
					break;
				case TK_COLON:
					o.op = OP_SELECT_ELSE;
					break;
				default: {
					_set_error("Invalid token for operator: " + get_token_text(tk));
					return nullptr;
				}
			}

			expression.push_back(o);

		} else {
			_set_tkpos(pos); //something else, so rollback and end
			break;
		}
	}

	/* Reduce the set set of expressions and place them in an operator tree, respecting precedence */

	while (expression.size() > 1) {
		int next_op = -1;
		int min_priority = 0xFFFFF;
		bool is_unary = false;
		bool is_ternary = false;

		for (int i = 0; i < expression.size(); i++) {
			if (!expression[i].is_op) {
				continue;
			}

			bool unary = false;
			bool ternary = false;
			Operator op = expression[i].op;

			int priority;
			switch (op) {
				case OP_EQUAL:
					priority = 8;
					break;
				case OP_NOT_EQUAL:
					priority = 8;
					break;
				case OP_LESS:
					priority = 7;
					break;
				case OP_LESS_EQUAL:
					priority = 7;
					break;
				case OP_GREATER:
					priority = 7;
					break;
				case OP_GREATER_EQUAL:
					priority = 7;
					break;
				case OP_AND:
					priority = 12;
					break;
				case OP_OR:
					priority = 14;
					break;
				case OP_NOT:
					priority = 3;
					unary = true;
					break;
				case OP_NEGATE:
					priority = 3;
					unary = true;
					break;
				case OP_ADD:
					priority = 5;
					break;
				case OP_SUB:
					priority = 5;
					break;
				case OP_MUL:
					priority = 4;
					break;
				case OP_DIV:
					priority = 4;
					break;
				case OP_MOD:
					priority = 4;
					break;
				case OP_SHIFT_LEFT:
					priority = 6;
					break;
				case OP_SHIFT_RIGHT:
					priority = 6;
					break;
				case OP_ASSIGN:
					priority = 16;
					break;
				case OP_ASSIGN_ADD:
					priority = 16;
					break;
				case OP_ASSIGN_SUB:
					priority = 16;
					break;
				case OP_ASSIGN_MUL:
					priority = 16;
					break;
				case OP_ASSIGN_DIV:
					priority = 16;
					break;
				case OP_ASSIGN_MOD:
					priority = 16;
					break;
				case OP_ASSIGN_SHIFT_LEFT:
					priority = 16;
					break;
				case OP_ASSIGN_SHIFT_RIGHT:
					priority = 16;
					break;
				case OP_ASSIGN_BIT_AND:
					priority = 16;
					break;
				case OP_ASSIGN_BIT_OR:
					priority = 16;
					break;
				case OP_ASSIGN_BIT_XOR:
					priority = 16;
					break;
				case OP_BIT_AND:
					priority = 9;
					break;
				case OP_BIT_OR:
					priority = 11;
					break;
				case OP_BIT_XOR:
					priority = 10;
					break;
				case OP_BIT_INVERT:
					priority = 3;
					unary = true;
					break;
				case OP_INCREMENT:
					priority = 3;
					unary = true;
					break;
				case OP_DECREMENT:
					priority = 3;
					unary = true;
					break;
				case OP_SELECT_IF:
					priority = 15;
					ternary = true;
					break;
				case OP_SELECT_ELSE:
					priority = 15;
					ternary = true;
					break;

				default:
					ERR_FAIL_V(nullptr); //unexpected operator
			}

#if DEBUG_ENABLED
			if (check_warnings && HAS_WARNING(ShaderWarning::FLOAT_COMPARISON_FLAG) && (op == OP_EQUAL || op == OP_NOT_EQUAL) &&
					(!expression[i - 1].is_op && !expression[i + 1].is_op) &&
					(expression[i - 1].node->get_datatype() == TYPE_FLOAT && expression[i + 1].node->get_datatype() == TYPE_FLOAT)) {
				_add_line_warning(ShaderWarning::FLOAT_COMPARISON);
			}
#endif // DEBUG_ENABLED

			if (priority < min_priority) {
				// < is used for left to right (default)
				// <= is used for right to left
				next_op = i;
				min_priority = priority;
				is_unary = unary;
				is_ternary = ternary;
			}
		}

		ERR_FAIL_COND_V(next_op == -1, nullptr);

		// OK! create operator..
		if (is_unary) {
			int expr_pos = next_op;
			while (expression[expr_pos].is_op) {
				expr_pos++;
				if (expr_pos == expression.size()) {
					//can happen..
					_set_error("Unexpected end of expression...");
					return nullptr;
				}
			}

			//consecutively do unary operators
			for (int i = expr_pos - 1; i >= next_op; i--) {
				OperatorNode *op = alloc_node<OperatorNode>();
				op->op = expression[i].op;
				if ((op->op == OP_INCREMENT || op->op == OP_DECREMENT) && !_validate_assign(expression[i + 1].node, p_function_info)) {
					_set_error("Can't use increment/decrement operator in constant expression.");
					return nullptr;
				}
				op->arguments.push_back(expression[i + 1].node);

				expression.write[i].is_op = false;
				expression.write[i].node = op;

				if (!_validate_operator(op, &op->return_cache, &op->return_array_size)) {
					String at;
					for (int j = 0; j < op->arguments.size(); j++) {
						if (j > 0) {
							at += " and ";
						}
						at += get_datatype_name(op->arguments[j]->get_datatype());
						if (!op->arguments[j]->is_indexed() && op->arguments[j]->get_array_size() > 0) {
							at += "[";
							at += itos(op->arguments[j]->get_array_size());
							at += "]";
						}
					}
					_set_error("Invalid arguments to unary operator '" + get_operator_text(op->op) + "' :" + at);
					return nullptr;
				}
				expression.remove(i + 1);
			}

		} else if (is_ternary) {
			if (next_op < 1 || next_op >= (expression.size() - 1)) {
				_set_error("Parser bug...");
				ERR_FAIL_V(nullptr);
			}

			if (next_op + 2 >= expression.size() || !expression[next_op + 2].is_op || expression[next_op + 2].op != OP_SELECT_ELSE) {
				_set_error("Missing matching ':' for select operator");
				return nullptr;
			}

			OperatorNode *op = alloc_node<OperatorNode>();
			op->op = expression[next_op].op;
			op->arguments.push_back(expression[next_op - 1].node);
			op->arguments.push_back(expression[next_op + 1].node);
			op->arguments.push_back(expression[next_op + 3].node);

			expression.write[next_op - 1].is_op = false;
			expression.write[next_op - 1].node = op;
			if (!_validate_operator(op, &op->return_cache, &op->return_array_size)) {
				String at;
				for (int i = 0; i < op->arguments.size(); i++) {
					if (i > 0) {
						at += " and ";
					}
					at += get_datatype_name(op->arguments[i]->get_datatype());
					if (!op->arguments[i]->is_indexed() && op->arguments[i]->get_array_size() > 0) {
						at += "[";
						at += itos(op->arguments[i]->get_array_size());
						at += "]";
					}
				}
				_set_error("Invalid argument to ternary ?: operator: " + at);
				return nullptr;
			}

			for (int i = 0; i < 4; i++) {
				expression.remove(next_op);
			}

		} else {
			if (next_op < 1 || next_op >= (expression.size() - 1)) {
				_set_error("Parser bug...");
				ERR_FAIL_V(nullptr);
			}

			OperatorNode *op = alloc_node<OperatorNode>();
			op->op = expression[next_op].op;

			if (expression[next_op - 1].is_op) {
				_set_error("Parser bug...");
				ERR_FAIL_V(nullptr);
			}

			if (_is_operator_assign(op->op)) {
				String assign_message;
				if (!_validate_assign(expression[next_op - 1].node, p_function_info, &assign_message)) {
					_set_error(assign_message);
					return nullptr;
				}
			}

			if (expression[next_op + 1].is_op) {
				// this is not invalid and can really appear
				// but it becomes invalid anyway because no binary op
				// can be followed by a unary op in a valid combination,
				// due to how precedence works, unaries will always disappear first

				_set_error("Parser bug...");
			}

			op->arguments.push_back(expression[next_op - 1].node); //expression goes as left
			op->arguments.push_back(expression[next_op + 1].node); //next expression goes as right
			expression.write[next_op - 1].node = op;

			//replace all 3 nodes by this operator and make it an expression

			if (!_validate_operator(op, &op->return_cache, &op->return_array_size)) {
				String at;
				for (int i = 0; i < op->arguments.size(); i++) {
					if (i > 0) {
						at += " and ";
					}
					if (op->arguments[i]->get_datatype() == TYPE_STRUCT) {
						at += op->arguments[i]->get_datatype_name();
					} else {
						at += get_datatype_name(op->arguments[i]->get_datatype());
					}
					if (!op->arguments[i]->is_indexed() && op->arguments[i]->get_array_size() > 0) {
						at += "[";
						at += itos(op->arguments[i]->get_array_size());
						at += "]";
					}
				}
				_set_error("Invalid arguments to operator '" + get_operator_text(op->op) + "' :" + at);
				return nullptr;
			}

			expression.remove(next_op);
			expression.remove(next_op);
		}
	}

	return expression[0].node;
}

ShaderLanguage::Node *ShaderLanguage::_reduce_expression(BlockNode *p_block, ShaderLanguage::Node *p_node) {
	if (p_node->type != Node::TYPE_OPERATOR) {
		return p_node;
	}

	//for now only reduce simple constructors
	OperatorNode *op = static_cast<OperatorNode *>(p_node);

	if (op->op == OP_CONSTRUCT) {
		ERR_FAIL_COND_V(op->arguments[0]->type != Node::TYPE_VARIABLE, p_node);

		DataType type = op->get_datatype();
		DataType base = get_scalar_type(type);
		int cardinality = get_cardinality(type);

		Vector<ConstantNode::Value> values;

		for (int i = 1; i < op->arguments.size(); i++) {
			op->arguments.write[i] = _reduce_expression(p_block, op->arguments[i]);
			if (op->arguments[i]->type == Node::TYPE_CONSTANT) {
				ConstantNode *cn = static_cast<ConstantNode *>(op->arguments[i]);

				if (get_scalar_type(cn->datatype) == base) {
					for (int j = 0; j < cn->values.size(); j++) {
						values.push_back(cn->values[j]);
					}
				} else if (get_scalar_type(cn->datatype) == cn->datatype) {
					ConstantNode::Value v;
					if (!convert_constant(cn, base, &v)) {
						return p_node;
					}
					values.push_back(v);
				} else {
					return p_node;
				}

			} else {
				return p_node;
			}
		}

		if (values.size() == 1) {
			if (type >= TYPE_MAT2 && type <= TYPE_MAT4) {
				ConstantNode::Value value = values[0];
				ConstantNode::Value zero;
				zero.real = 0.0f;
				int size = 2 + (type - TYPE_MAT2);

				values.clear();
				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {
						values.push_back(i == j ? value : zero);
					}
				}
			} else {
				ConstantNode::Value value = values[0];
				for (int i = 1; i < cardinality; i++) {
					values.push_back(value);
				}
			}
		} else if (values.size() != cardinality) {
			ERR_PRINT("Failed to reduce expression, values and cardinality mismatch.");
			return p_node;
		}

		ConstantNode *cn = alloc_node<ConstantNode>();
		cn->datatype = op->get_datatype();
		cn->values = values;
		return cn;
	} else if (op->op == OP_NEGATE) {
		op->arguments.write[0] = _reduce_expression(p_block, op->arguments[0]);
		if (op->arguments[0]->type == Node::TYPE_CONSTANT) {
			ConstantNode *cn = static_cast<ConstantNode *>(op->arguments[0]);

			DataType base = get_scalar_type(cn->datatype);

			Vector<ConstantNode::Value> values;

			for (int i = 0; i < cn->values.size(); i++) {
				ConstantNode::Value nv;
				switch (base) {
					case TYPE_BOOL: {
						nv.boolean = !cn->values[i].boolean;
					} break;
					case TYPE_INT: {
						nv.sint = -cn->values[i].sint;
					} break;
					case TYPE_UINT: {
						// Intentionally wrap the unsigned int value, because GLSL does.
						nv.uint = 0 - cn->values[i].uint;
					} break;
					case TYPE_FLOAT: {
						nv.real = -cn->values[i].real;
					} break;
					default: {
					}
				}

				values.push_back(nv);
			}

			cn->values = values;
			return cn;
		}
	}

	return p_node;
}

ShaderLanguage::Node *ShaderLanguage::_parse_and_reduce_expression(BlockNode *p_block, const FunctionInfo &p_function_info) {
	ShaderLanguage::Node *expr = _parse_expression(p_block, p_function_info);
	if (!expr) { //errored
		return nullptr;
	}

	expr = _reduce_expression(p_block, expr);

	return expr;
}

Error ShaderLanguage::_parse_block(BlockNode *p_block, const FunctionInfo &p_function_info, bool p_just_one, bool p_can_break, bool p_can_continue) {
	while (true) {
		TkPos pos = _get_tkpos();

		Token tk = _get_token();

		if (p_block && p_block->block_type == BlockNode::BLOCK_TYPE_SWITCH) {
			if (tk.type != TK_CF_CASE && tk.type != TK_CF_DEFAULT && tk.type != TK_CURLY_BRACKET_CLOSE) {
				_set_error("Switch may contains only case and default blocks");
				return ERR_PARSE_ERROR;
			}
		}

		bool is_struct = shader->structs.has(tk.text);

		if (tk.type == TK_CURLY_BRACKET_CLOSE) { //end of block
			if (p_just_one) {
				_set_error("Unexpected '}'");
				return ERR_PARSE_ERROR;
			}

			return OK;

		} else if (tk.type == TK_CONST || is_token_precision(tk.type) || is_token_nonvoid_datatype(tk.type) || is_struct) {
			String struct_name = "";
			if (is_struct) {
				struct_name = tk.text;
			}

			bool is_const = false;

			if (tk.type == TK_CONST) {
				is_const = true;
				tk = _get_token();

				if (!is_struct) {
					is_struct = shader->structs.has(tk.text); // check again.
					struct_name = tk.text;
				}
			}

			DataPrecision precision = PRECISION_DEFAULT;
			if (is_token_precision(tk.type)) {
				precision = get_token_precision(tk.type);
				tk = _get_token();

				if (!is_struct) {
					is_struct = shader->structs.has(tk.text); // check again.
				}
				if (is_struct && precision != PRECISION_DEFAULT) {
					_set_error("Precision modifier cannot be used on structs.");
					return ERR_PARSE_ERROR;
				}
				if (!is_token_nonvoid_datatype(tk.type)) {
					_set_error("Expected datatype after precision");
					return ERR_PARSE_ERROR;
				}
			}

			if (!is_struct) {
				if (!is_token_variable_datatype(tk.type)) {
					_set_error("Invalid data type for variable (samplers not allowed)");
					return ERR_PARSE_ERROR;
				}
			}

			DataType type = is_struct ? TYPE_STRUCT : get_token_datatype(tk.type);

			if (_validate_datatype(type) != OK) {
				return ERR_PARSE_ERROR;
			}

			tk = _get_token();

			Node *vardecl = nullptr;

			while (true) {
				bool unknown_size = false;
				int array_size = 0;

				ArrayDeclarationNode *anode = nullptr;
				ArrayDeclarationNode::Declaration adecl;

				if (tk.type != TK_IDENTIFIER && tk.type != TK_BRACKET_OPEN) {
					_set_error("Expected identifier or '[' after type.");
					return ERR_PARSE_ERROR;
				}

				if (tk.type == TK_BRACKET_OPEN) {
					anode = alloc_node<ArrayDeclarationNode>();

					if (is_struct) {
						anode->struct_name = struct_name;
						anode->datatype = TYPE_STRUCT;
					} else {
						anode->datatype = type;
					}

					anode->precision = precision;
					anode->is_const = is_const;
					vardecl = (Node *)anode;

					adecl.size = 0U;
					adecl.single_expression = false;

					Error error = _parse_local_array_size(p_block, p_function_info, anode, &adecl, array_size, unknown_size);
					if (error != OK) {
						return error;
					}
					tk = _get_token();

					if (tk.type != TK_IDENTIFIER) {
						_set_error("Expected identifier!");
						return ERR_PARSE_ERROR;
					}
				}

				StringName name = tk.text;
				ShaderLanguage::IdentifierType itype;
				if (_find_identifier(p_block, true, p_function_info, name, (ShaderLanguage::DataType *)nullptr, &itype)) {
					if (itype != IDENTIFIER_FUNCTION) {
						_set_error("Redefinition of '" + String(name) + "'");
						return ERR_PARSE_ERROR;
					}
				}

				adecl.name = name;

#ifdef DEBUG_ENABLED
				if (check_warnings && HAS_WARNING(ShaderWarning::UNUSED_LOCAL_VARIABLE_FLAG)) {
					if (p_block && p_block->parent_function) {
						StringName func_name = p_block->parent_function->name;

						if (!used_local_vars.has(func_name)) {
							used_local_vars.insert(func_name, Map<StringName, Usage>());
						}

						used_local_vars[func_name].insert(name, Usage(tk_line));
					}
				}
#endif // DEBUG_ENABLED

				BlockNode::Variable var;
				var.type = type;
				var.precision = precision;
				var.line = tk_line;
				var.array_size = array_size;
				var.is_const = is_const;
				var.struct_name = struct_name;

				tk = _get_token();

				if (tk.type == TK_BRACKET_OPEN) {
					if (var.array_size > 0 || unknown_size) {
						_set_error("Array size is already defined!");
						return ERR_PARSE_ERROR;
					}

					if (RenderingServer::get_singleton()->is_low_end() && is_const) {
						_set_error("Local const arrays are supported only on high-end platform!");
						return ERR_PARSE_ERROR;
					}

					anode = alloc_node<ArrayDeclarationNode>();
					if (is_struct) {
						anode->struct_name = struct_name;
						anode->datatype = TYPE_STRUCT;
					} else {
						anode->datatype = type;
					}
					anode->precision = precision;
					anode->is_const = is_const;
					vardecl = (Node *)anode;

					adecl.size = 0U;
					adecl.single_expression = false;

					Error error = _parse_local_array_size(p_block, p_function_info, anode, &adecl, var.array_size, unknown_size);
					if (error != OK) {
						return error;
					}
					tk = _get_token();
				}

				if (var.array_size > 0 || unknown_size) {
					bool full_def = false;

					if (tk.type == TK_OP_ASSIGN) {
						if (RenderingServer::get_singleton()->is_low_end()) {
							_set_error("Array initialization is supported only on high-end platform!");
							return ERR_PARSE_ERROR;
						}

						TkPos prev_pos = _get_tkpos();
						tk = _get_token();

						if (tk.type == TK_IDENTIFIER) { // a function call array initialization
							_set_tkpos(prev_pos);
							Node *n = _parse_and_reduce_expression(p_block, p_function_info);

							if (!n) {
								_set_error("Expected correct array initializer!");
								return ERR_PARSE_ERROR;
							} else {
								if (unknown_size) {
									adecl.size = n->get_array_size();
									var.array_size = n->get_array_size();
								}

								if (!_compare_datatypes(var.type, var.struct_name, var.array_size, n->get_datatype(), n->get_datatype_name(), n->get_array_size())) {
									return ERR_PARSE_ERROR;
								}

								adecl.single_expression = true;
								adecl.initializer.push_back(n);
							}

							tk = _get_token();
						} else {
							if (tk.type != TK_CURLY_BRACKET_OPEN) {
								if (unknown_size) {
									_set_error("Expected '{'");
									return ERR_PARSE_ERROR;
								}

								full_def = true;

								DataPrecision precision2 = PRECISION_DEFAULT;
								if (is_token_precision(tk.type)) {
									precision2 = get_token_precision(tk.type);
									tk = _get_token();
									if (shader->structs.has(tk.text)) {
										_set_error("Precision modifier cannot be used on structs.");
										return ERR_PARSE_ERROR;
									}
									if (!is_token_nonvoid_datatype(tk.type)) {
										_set_error("Expected datatype after precision");
										return ERR_PARSE_ERROR;
									}
								}

								DataType type2;
								StringName struct_name2 = "";

								if (shader->structs.has(tk.text)) {
									type2 = TYPE_STRUCT;
									struct_name2 = tk.text;
								} else {
									if (!is_token_variable_datatype(tk.type)) {
										_set_error("Invalid data type for array");
										return ERR_PARSE_ERROR;
									}
									type2 = get_token_datatype(tk.type);
								}

								int array_size2 = 0;

								tk = _get_token();
								if (tk.type == TK_BRACKET_OPEN) {
									TkPos pos2 = _get_tkpos();
									tk = _get_token();
									if (tk.type == TK_BRACKET_CLOSE) {
										array_size2 = var.array_size;
										tk = _get_token();
									} else {
										_set_tkpos(pos2);

										Node *n = _parse_and_reduce_expression(p_block, p_function_info);
										if (!n || n->type != Node::TYPE_CONSTANT || n->get_datatype() != TYPE_INT) {
											_set_error("Expected single integer constant > 0");
											return ERR_PARSE_ERROR;
										}

										ConstantNode *cnode = (ConstantNode *)n;
										if (cnode->values.size() == 1) {
											array_size2 = cnode->values[0].sint;
											if (array_size2 <= 0) {
												_set_error("Expected single integer constant > 0");
												return ERR_PARSE_ERROR;
											}
										} else {
											_set_error("Expected single integer constant > 0");
											return ERR_PARSE_ERROR;
										}

										tk = _get_token();
										if (tk.type != TK_BRACKET_CLOSE) {
											_set_error("Expected ']'");
											return ERR_PARSE_ERROR;
										} else {
											tk = _get_token();
										}
									}
								} else {
									_set_error("Expected '['");
									return ERR_PARSE_ERROR;
								}

								if (precision != precision2 || type != type2 || struct_name != struct_name2 || var.array_size != array_size2) {
									String error_str = "Cannot convert from '";
									if (precision2 != PRECISION_DEFAULT) {
										error_str += get_precision_name(precision2);
										error_str += " ";
									}
									if (type2 == TYPE_STRUCT) {
										error_str += struct_name2;
									} else {
										error_str += get_datatype_name(type2);
									}
									error_str += "[";
									error_str += itos(array_size2);
									error_str += "]'";
									error_str += " to '";
									if (precision != PRECISION_DEFAULT) {
										error_str += get_precision_name(precision);
										error_str += " ";
									}
									if (type == TYPE_STRUCT) {
										error_str += struct_name;
									} else {
										error_str += get_datatype_name(type);
									}
									error_str += "[";
									error_str += itos(var.array_size);
									error_str += "]'";
									_set_error(error_str);
									return ERR_PARSE_ERROR;
								}
							}

							bool curly = tk.type == TK_CURLY_BRACKET_OPEN;

							if (unknown_size) {
								if (!curly) {
									_set_error("Expected '{'");
									return ERR_PARSE_ERROR;
								}
							} else {
								if (full_def) {
									if (curly) {
										_set_error("Expected '('");
										return ERR_PARSE_ERROR;
									}
								}
							}

							if (tk.type == TK_PARENTHESIS_OPEN || curly) { // initialization
								while (true) {
									Node *n = _parse_and_reduce_expression(p_block, p_function_info);
									if (!n) {
										return ERR_PARSE_ERROR;
									}

									if (anode->is_const && n->type == Node::TYPE_OPERATOR && ((OperatorNode *)n)->op == OP_CALL) {
										_set_error("Expected constant expression");
										return ERR_PARSE_ERROR;
									}

									if (!_compare_datatypes(var.type, struct_name, 0, n->get_datatype(), n->get_datatype_name(), 0)) {
										return ERR_PARSE_ERROR;
									}

									tk = _get_token();
									if (tk.type == TK_COMMA) {
										adecl.initializer.push_back(n);
										continue;
									} else if (!curly && tk.type == TK_PARENTHESIS_CLOSE) {
										adecl.initializer.push_back(n);
										break;
									} else if (curly && tk.type == TK_CURLY_BRACKET_CLOSE) {
										adecl.initializer.push_back(n);
										break;
									} else {
										if (curly) {
											_set_error("Expected '}' or ','");
										} else {
											_set_error("Expected ')' or ','");
										}
										return ERR_PARSE_ERROR;
									}
								}
								if (unknown_size) {
									adecl.size = adecl.initializer.size();
									var.array_size = adecl.initializer.size();
								} else if (adecl.initializer.size() != var.array_size) {
									_set_error("Array size mismatch");
									return ERR_PARSE_ERROR;
								}
								tk = _get_token();
							}
						}
					} else {
						if (unknown_size) {
							_set_error("Expected array initialization");
							return ERR_PARSE_ERROR;
						}
						if (anode->is_const) {
							_set_error("Expected initialization of constant");
							return ERR_PARSE_ERROR;
						}
					}

					anode->declarations.push_back(adecl);
				} else if (tk.type == TK_OP_ASSIGN) {
					VariableDeclarationNode *node = alloc_node<VariableDeclarationNode>();
					if (is_struct) {
						node->struct_name = struct_name;
						node->datatype = TYPE_STRUCT;
					} else {
						node->datatype = type;
					}
					node->precision = precision;
					node->is_const = is_const;
					vardecl = (Node *)node;

					VariableDeclarationNode::Declaration decl;
					decl.name = name;
					decl.initializer = nullptr;

					//variable created with assignment! must parse an expression
					Node *n = _parse_and_reduce_expression(p_block, p_function_info);
					if (!n) {
						return ERR_PARSE_ERROR;
					}
					if (node->is_const && n->type == Node::TYPE_OPERATOR && ((OperatorNode *)n)->op == OP_CALL) {
						OperatorNode *op = ((OperatorNode *)n);
						for (int i = 1; i < op->arguments.size(); i++) {
							if (!_check_node_constness(op->arguments[i])) {
								_set_error("Expected constant expression for argument '" + itos(i - 1) + "' of function call after '='");
								return ERR_PARSE_ERROR;
							}
						}
					}
					decl.initializer = n;

					if (n->type == Node::TYPE_CONSTANT) {
						ConstantNode *const_node = static_cast<ConstantNode *>(n);
						if (const_node && const_node->values.size() == 1) {
							var.value = const_node->values[0];
						}
					}

					if (!_compare_datatypes(var.type, var.struct_name, var.array_size, n->get_datatype(), n->get_datatype_name(), n->get_array_size())) {
						return ERR_PARSE_ERROR;
					}
					tk = _get_token();
					node->declarations.push_back(decl);
				} else {
					if (is_const) {
						_set_error("Expected initialization of constant");
						return ERR_PARSE_ERROR;
					}

					VariableDeclarationNode *node = alloc_node<VariableDeclarationNode>();
					if (is_struct) {
						node->struct_name = struct_name;
						node->datatype = TYPE_STRUCT;
					} else {
						node->datatype = type;
					}
					node->precision = precision;
					vardecl = (Node *)node;

					VariableDeclarationNode::Declaration decl;
					decl.name = name;
					decl.initializer = nullptr;
					node->declarations.push_back(decl);
				}

				p_block->statements.push_back(vardecl);

				p_block->variables[name] = var;
				if (tk.type == TK_COMMA) {
					if (p_block->block_type == BlockNode::BLOCK_TYPE_FOR) {
						_set_error("Multiple declarations in 'for' loop are not implemented yet.");
						return ERR_PARSE_ERROR;
					}
					tk = _get_token();
					//another variable
				} else if (tk.type == TK_SEMICOLON) {
					break;
				} else {
					_set_error("Expected ',' or ';' after variable");
					return ERR_PARSE_ERROR;
				}
			}
		} else if (tk.type == TK_CURLY_BRACKET_OPEN) {
			//a sub block, just because..
			BlockNode *block = alloc_node<BlockNode>();
			block->parent_block = p_block;
			if (_parse_block(block, p_function_info, false, p_can_break, p_can_continue) != OK) {
				return ERR_PARSE_ERROR;
			}
			p_block->statements.push_back(block);
		} else if (tk.type == TK_CF_IF) {
			//if () {}
			tk = _get_token();
			if (tk.type != TK_PARENTHESIS_OPEN) {
				_set_error("Expected '(' after if");
				return ERR_PARSE_ERROR;
			}

			ControlFlowNode *cf = alloc_node<ControlFlowNode>();
			cf->flow_op = FLOW_OP_IF;
			Node *n = _parse_and_reduce_expression(p_block, p_function_info);
			if (!n) {
				return ERR_PARSE_ERROR;
			}

			if (n->get_datatype() != TYPE_BOOL) {
				_set_error("Expected boolean expression");
				return ERR_PARSE_ERROR;
			}

			tk = _get_token();
			if (tk.type != TK_PARENTHESIS_CLOSE) {
				_set_error("Expected ')' after expression");
				return ERR_PARSE_ERROR;
			}

			BlockNode *block = alloc_node<BlockNode>();
			block->parent_block = p_block;
			cf->expressions.push_back(n);
			cf->blocks.push_back(block);
			p_block->statements.push_back(cf);

			Error err = _parse_block(block, p_function_info, true, p_can_break, p_can_continue);
			if (err) {
				return err;
			}

			pos = _get_tkpos();
			tk = _get_token();
			if (tk.type == TK_CF_ELSE) {
				block = alloc_node<BlockNode>();
				block->parent_block = p_block;
				cf->blocks.push_back(block);
				err = _parse_block(block, p_function_info, true, p_can_break, p_can_continue);

			} else {
				_set_tkpos(pos); //rollback
			}
		} else if (tk.type == TK_CF_SWITCH) {
			if (RenderingServer::get_singleton()->is_low_end()) {
				_set_error("\"switch\" operator is supported only on high-end platform!");
				return ERR_PARSE_ERROR;
			}

			// switch() {}
			tk = _get_token();
			if (tk.type != TK_PARENTHESIS_OPEN) {
				_set_error("Expected '(' after switch");
				return ERR_PARSE_ERROR;
			}
			ControlFlowNode *cf = alloc_node<ControlFlowNode>();
			cf->flow_op = FLOW_OP_SWITCH;
			Node *n = _parse_and_reduce_expression(p_block, p_function_info);
			if (!n) {
				return ERR_PARSE_ERROR;
			}
			if (n->get_datatype() != TYPE_INT) {
				_set_error("Expected integer expression");
				return ERR_PARSE_ERROR;
			}
			tk = _get_token();
			if (tk.type != TK_PARENTHESIS_CLOSE) {
				_set_error("Expected ')' after expression");
				return ERR_PARSE_ERROR;
			}
			tk = _get_token();
			if (tk.type != TK_CURLY_BRACKET_OPEN) {
				_set_error("Expected '{' after switch statement");
				return ERR_PARSE_ERROR;
			}
			BlockNode *switch_block = alloc_node<BlockNode>();
			switch_block->block_type = BlockNode::BLOCK_TYPE_SWITCH;
			switch_block->parent_block = p_block;
			cf->expressions.push_back(n);
			cf->blocks.push_back(switch_block);
			p_block->statements.push_back(cf);

			int prev_type = TK_CF_CASE;
			while (true) { // Go-through multiple cases.

				if (_parse_block(switch_block, p_function_info, true, true, false) != OK) {
					return ERR_PARSE_ERROR;
				}
				pos = _get_tkpos();
				tk = _get_token();
				if (tk.type == TK_CF_CASE || tk.type == TK_CF_DEFAULT) {
					if (prev_type == TK_CF_DEFAULT) {
						if (tk.type == TK_CF_CASE) {
							_set_error("Cases must be defined before default case.");
							return ERR_PARSE_ERROR;
						} else if (prev_type == TK_CF_DEFAULT) {
							_set_error("Default case must be defined only once.");
							return ERR_PARSE_ERROR;
						}
					}
					prev_type = tk.type;
					_set_tkpos(pos);
					continue;
				} else {
					Set<int> constants;
					for (int i = 0; i < switch_block->statements.size(); i++) { // Checks for duplicates.
						ControlFlowNode *flow = (ControlFlowNode *)switch_block->statements[i];
						if (flow) {
							if (flow->flow_op == FLOW_OP_CASE) {
								if (flow->expressions[0]->type == Node::TYPE_CONSTANT) {
									ConstantNode *cn = static_cast<ConstantNode *>(flow->expressions[0]);
									if (!cn || cn->values.is_empty()) {
										return ERR_PARSE_ERROR;
									}
									if (constants.has(cn->values[0].sint)) {
										_set_error("Duplicated case label: '" + itos(cn->values[0].sint) + "'");
										return ERR_PARSE_ERROR;
									}
									constants.insert(cn->values[0].sint);
								} else if (flow->expressions[0]->type == Node::TYPE_VARIABLE) {
									VariableNode *vn = static_cast<VariableNode *>(flow->expressions[0]);
									if (!vn) {
										return ERR_PARSE_ERROR;
									}
									ConstantNode::Value v;
									_find_identifier(p_block, false, p_function_info, vn->name, nullptr, nullptr, nullptr, nullptr, nullptr, &v);
									if (constants.has(v.sint)) {
										_set_error("Duplicated case label: '" + itos(v.sint) + "'");
										return ERR_PARSE_ERROR;
									}
									constants.insert(v.sint);
								}
							} else if (flow->flow_op == FLOW_OP_DEFAULT) {
								continue;
							} else {
								return ERR_PARSE_ERROR;
							}
						} else {
							return ERR_PARSE_ERROR;
						}
					}
					break;
				}
			}

		} else if (tk.type == TK_CF_CASE) {
			// case x : break; | return;

			if (p_block && p_block->block_type == BlockNode::BLOCK_TYPE_CASE) {
				_set_tkpos(pos);
				return OK;
			}

			if (!p_block || (p_block->block_type != BlockNode::BLOCK_TYPE_SWITCH)) {
				_set_error("case must be placed within switch block");
				return ERR_PARSE_ERROR;
			}

			tk = _get_token();

			int sign = 1;

			if (tk.type == TK_OP_SUB) {
				sign = -1;
				tk = _get_token();
			}

			Node *n = nullptr;

			if (tk.type != TK_INT_CONSTANT) {
				bool correct_constant_expression = false;
				DataType data_type;

				if (tk.type == TK_IDENTIFIER) {
					bool is_const;
					_find_identifier(p_block, false, p_function_info, tk.text, &data_type, nullptr, &is_const);
					if (is_const) {
						if (data_type == TYPE_INT) {
							correct_constant_expression = true;
						}
					}
				}
				if (!correct_constant_expression) {
					_set_error("Expected integer constant");
					return ERR_PARSE_ERROR;
				}

				VariableNode *vn = alloc_node<VariableNode>();
				vn->name = tk.text;
				n = vn;
			} else {
				ConstantNode::Value v;
				v.sint = (int)tk.constant * sign;

				ConstantNode *cn = alloc_node<ConstantNode>();
				cn->values.push_back(v);
				cn->datatype = TYPE_INT;
				n = cn;
			}

			tk = _get_token();

			if (tk.type != TK_COLON) {
				_set_error("Expected ':'");
				return ERR_PARSE_ERROR;
			}

			ControlFlowNode *cf = alloc_node<ControlFlowNode>();
			cf->flow_op = FLOW_OP_CASE;

			BlockNode *case_block = alloc_node<BlockNode>();
			case_block->block_type = BlockNode::BLOCK_TYPE_CASE;
			case_block->parent_block = p_block;
			cf->expressions.push_back(n);
			cf->blocks.push_back(case_block);
			p_block->statements.push_back(cf);

			Error err = _parse_block(case_block, p_function_info, false, true, false);
			if (err) {
				return err;
			}

			return OK;

		} else if (tk.type == TK_CF_DEFAULT) {
			if (p_block && p_block->block_type == BlockNode::BLOCK_TYPE_CASE) {
				_set_tkpos(pos);
				return OK;
			}

			if (!p_block || (p_block->block_type != BlockNode::BLOCK_TYPE_SWITCH)) {
				_set_error("default must be placed within switch block");
				return ERR_PARSE_ERROR;
			}

			tk = _get_token();

			if (tk.type != TK_COLON) {
				_set_error("Expected ':'");
				return ERR_PARSE_ERROR;
			}

			ControlFlowNode *cf = alloc_node<ControlFlowNode>();
			cf->flow_op = FLOW_OP_DEFAULT;

			BlockNode *default_block = alloc_node<BlockNode>();
			default_block->block_type = BlockNode::BLOCK_TYPE_DEFAULT;
			default_block->parent_block = p_block;
			cf->blocks.push_back(default_block);
			p_block->statements.push_back(cf);

			Error err = _parse_block(default_block, p_function_info, false, true, false);
			if (err) {
				return err;
			}

			return OK;

		} else if (tk.type == TK_CF_DO || tk.type == TK_CF_WHILE) {
			// do {} while()
			// while() {}
			bool is_do = tk.type == TK_CF_DO;

			BlockNode *do_block = nullptr;
			if (is_do) {
				do_block = alloc_node<BlockNode>();
				do_block->parent_block = p_block;

				Error err = _parse_block(do_block, p_function_info, true, true, true);
				if (err) {
					return err;
				}

				tk = _get_token();
				if (tk.type != TK_CF_WHILE) {
					_set_error("Expected while after do");
					return ERR_PARSE_ERROR;
				}
			}
			tk = _get_token();

			if (tk.type != TK_PARENTHESIS_OPEN) {
				_set_error("Expected '(' after while");
				return ERR_PARSE_ERROR;
			}

			ControlFlowNode *cf = alloc_node<ControlFlowNode>();
			if (is_do) {
				cf->flow_op = FLOW_OP_DO;
			} else {
				cf->flow_op = FLOW_OP_WHILE;
			}
			Node *n = _parse_and_reduce_expression(p_block, p_function_info);
			if (!n) {
				return ERR_PARSE_ERROR;
			}

			tk = _get_token();
			if (tk.type != TK_PARENTHESIS_CLOSE) {
				_set_error("Expected ')' after expression");
				return ERR_PARSE_ERROR;
			}
			if (!is_do) {
				BlockNode *block = alloc_node<BlockNode>();
				block->parent_block = p_block;
				cf->expressions.push_back(n);
				cf->blocks.push_back(block);
				p_block->statements.push_back(cf);

				Error err = _parse_block(block, p_function_info, true, true, true);
				if (err) {
					return err;
				}
			} else {
				cf->expressions.push_back(n);
				cf->blocks.push_back(do_block);
				p_block->statements.push_back(cf);

				tk = _get_token();
				if (tk.type != TK_SEMICOLON) {
					_set_error("Expected ';'");
					return ERR_PARSE_ERROR;
				}
			}
		} else if (tk.type == TK_CF_FOR) {
			// for() {}
			tk = _get_token();
			if (tk.type != TK_PARENTHESIS_OPEN) {
				_set_error("Expected '(' after for");
				return ERR_PARSE_ERROR;
			}

			ControlFlowNode *cf = alloc_node<ControlFlowNode>();
			cf->flow_op = FLOW_OP_FOR;

			BlockNode *init_block = alloc_node<BlockNode>();
			init_block->block_type = BlockNode::BLOCK_TYPE_FOR;
			init_block->parent_block = p_block;
			init_block->single_statement = true;
			cf->blocks.push_back(init_block);
			if (_parse_block(init_block, p_function_info, true, false, false) != OK) {
				return ERR_PARSE_ERROR;
			}

			Node *n = _parse_and_reduce_expression(init_block, p_function_info);
			if (!n) {
				return ERR_PARSE_ERROR;
			}

			if (n->get_datatype() != TYPE_BOOL) {
				_set_error("Middle expression is expected to be boolean.");
				return ERR_PARSE_ERROR;
			}

			tk = _get_token();
			if (tk.type != TK_SEMICOLON) {
				_set_error("Expected ';' after middle expression");
				return ERR_PARSE_ERROR;
			}

			cf->expressions.push_back(n);

			n = _parse_and_reduce_expression(init_block, p_function_info);
			if (!n) {
				return ERR_PARSE_ERROR;
			}

			cf->expressions.push_back(n);

			tk = _get_token();
			if (tk.type != TK_PARENTHESIS_CLOSE) {
				_set_error("Expected ')' after third expression");
				return ERR_PARSE_ERROR;
			}

			BlockNode *block = alloc_node<BlockNode>();
			block->parent_block = init_block;
			cf->blocks.push_back(block);
			p_block->statements.push_back(cf);

			Error err = _parse_block(block, p_function_info, true, true, true);
			if (err) {
				return err;
			}

		} else if (tk.type == TK_CF_RETURN) {
			//check return type
			BlockNode *b = p_block;

			if (b && b->parent_function && p_function_info.main_function) {
				_set_error(vformat("Using 'return' in '%s' processor function results in undefined behavior!", b->parent_function->name));
				return ERR_PARSE_ERROR;
			}

			while (b && !b->parent_function) {
				b = b->parent_block;
			}

			if (!b) {
				_set_error("Bug");
				return ERR_BUG;
			}

			String return_struct_name = String(b->parent_function->return_struct_name);
			String array_size_string;

			if (b->parent_function->return_array_size > 0) {
				array_size_string = "[" + itos(b->parent_function->return_array_size) + "]";
			}

			ControlFlowNode *flow = alloc_node<ControlFlowNode>();
			flow->flow_op = FLOW_OP_RETURN;

			pos = _get_tkpos();
			tk = _get_token();
			if (tk.type == TK_SEMICOLON) {
				//all is good
				if (b->parent_function->return_type != TYPE_VOID) {
					_set_error("Expected return with an expression of type '" + (return_struct_name != "" ? return_struct_name : get_datatype_name(b->parent_function->return_type)) + array_size_string + "'");
					return ERR_PARSE_ERROR;
				}
			} else {
				_set_tkpos(pos); //rollback, wants expression

				Node *expr = _parse_and_reduce_expression(p_block, p_function_info);
				if (!expr) {
					return ERR_PARSE_ERROR;
				}

				if (b->parent_function->return_type != expr->get_datatype() || b->parent_function->return_array_size != expr->get_array_size() || return_struct_name != expr->get_datatype_name()) {
					_set_error("Expected return with an expression of type '" + (return_struct_name != "" ? return_struct_name : get_datatype_name(b->parent_function->return_type)) + array_size_string + "'");
					return ERR_PARSE_ERROR;
				}

				tk = _get_token();
				if (tk.type != TK_SEMICOLON) {
					_set_error("Expected ';' after return expression");
					return ERR_PARSE_ERROR;
				}

				flow->expressions.push_back(expr);
			}

			p_block->statements.push_back(flow);

			BlockNode *block = p_block;
			while (block) {
				if (block->block_type == BlockNode::BLOCK_TYPE_CASE || block->block_type == BlockNode::BLOCK_TYPE_DEFAULT) {
					return OK;
				}
				block = block->parent_block;
			}
		} else if (tk.type == TK_CF_DISCARD) {
			//check return type
			BlockNode *b = p_block;
			while (b && !b->parent_function) {
				b = b->parent_block;
			}
			if (!b) {
				_set_error("Bug");
				return ERR_BUG;
			}

			if (!b->parent_function->can_discard) {
				_set_error("Use of 'discard' is not allowed here.");
				return ERR_PARSE_ERROR;
			}

			ControlFlowNode *flow = alloc_node<ControlFlowNode>();
			flow->flow_op = FLOW_OP_DISCARD;

			pos = _get_tkpos();
			tk = _get_token();
			if (tk.type != TK_SEMICOLON) {
				_set_error("Expected ';' after discard");
				return ERR_PARSE_ERROR;
			}

			p_block->statements.push_back(flow);
		} else if (tk.type == TK_CF_BREAK) {
			if (!p_can_break) {
				_set_error("'break' is not allowed outside of a loop or 'switch' statement");
				return ERR_PARSE_ERROR;
			}

			ControlFlowNode *flow = alloc_node<ControlFlowNode>();
			flow->flow_op = FLOW_OP_BREAK;

			pos = _get_tkpos();
			tk = _get_token();
			if (tk.type != TK_SEMICOLON) {
				_set_error("Expected ';' after break");
				return ERR_PARSE_ERROR;
			}

			p_block->statements.push_back(flow);

			BlockNode *block = p_block;
			while (block) {
				if (block->block_type == BlockNode::BLOCK_TYPE_CASE || block->block_type == BlockNode::BLOCK_TYPE_DEFAULT) {
					return OK;
				}
				block = block->parent_block;
			}

		} else if (tk.type == TK_CF_CONTINUE) {
			if (!p_can_continue) {
				_set_error("'continue' is not allowed outside of a loop");
				return ERR_PARSE_ERROR;
			}

			ControlFlowNode *flow = alloc_node<ControlFlowNode>();
			flow->flow_op = FLOW_OP_CONTINUE;

			pos = _get_tkpos();
			tk = _get_token();
			if (tk.type != TK_SEMICOLON) {
				//all is good
				_set_error("Expected ';' after continue");
				return ERR_PARSE_ERROR;
			}

			p_block->statements.push_back(flow);

		} else {
			//nothing else, so expression
			_set_tkpos(pos); //rollback
			Node *expr = _parse_and_reduce_expression(p_block, p_function_info);
			if (!expr) {
				return ERR_PARSE_ERROR;
			}
			p_block->statements.push_back(expr);
			tk = _get_token();

			if (tk.type != TK_SEMICOLON) {
				_set_error("Expected ';' after statement");
				return ERR_PARSE_ERROR;
			}
		}

		if (p_just_one) {
			break;
		}
	}

	return OK;
}

String ShaderLanguage::_get_shader_type_list(const Set<String> &p_shader_types) const {
	// Return a list of shader types as an human-readable string
	String valid_types;
	for (const Set<String>::Element *E = p_shader_types.front(); E; E = E->next()) {
		if (valid_types != String()) {
			valid_types += ", ";
		}

		valid_types += "'" + E->get() + "'";
	}

	return valid_types;
}

String ShaderLanguage::_get_qualifier_str(ArgumentQualifier p_qualifier) const {
	switch (p_qualifier) {
		case ArgumentQualifier::ARGUMENT_QUALIFIER_IN:
			return "in";
		case ArgumentQualifier::ARGUMENT_QUALIFIER_OUT:
			return "out";
		case ArgumentQualifier::ARGUMENT_QUALIFIER_INOUT:
			return "inout";
	}
	return "";
}

Error ShaderLanguage::_validate_datatype(DataType p_type) {
	if (RenderingServer::get_singleton()->is_low_end()) {
		bool invalid_type = false;

		switch (p_type) {
			case TYPE_UINT:
			case TYPE_UVEC2:
			case TYPE_UVEC3:
			case TYPE_UVEC4:
			case TYPE_ISAMPLER2D:
			case TYPE_USAMPLER2D:
			case TYPE_ISAMPLER3D:
			case TYPE_USAMPLER3D:
			case TYPE_USAMPLER2DARRAY:
			case TYPE_ISAMPLER2DARRAY:
				invalid_type = true;
				break;
			default:
				break;
		}

		if (invalid_type) {
			_set_error(vformat("\"%s\" type is supported only on high-end platform!", get_datatype_name(p_type)));
			return ERR_UNAVAILABLE;
		}
	}
	return OK;
}

Error ShaderLanguage::_parse_shader(const Map<StringName, FunctionInfo> &p_functions, const Vector<StringName> &p_render_modes, const Set<String> &p_shader_types) {
	Token tk = _get_token();
	TkPos prev_pos;

	if (tk.type != TK_SHADER_TYPE) {
		_set_error("Expected 'shader_type' at the beginning of shader. Valid types are: " + _get_shader_type_list(p_shader_types));
		return ERR_PARSE_ERROR;
	}

	tk = _get_token();

	if (tk.type != TK_IDENTIFIER) {
		_set_error("Expected identifier after 'shader_type', indicating type of shader. Valid types are: " + _get_shader_type_list(p_shader_types));
		return ERR_PARSE_ERROR;
	}

	String shader_type_identifier;

	shader_type_identifier = tk.text;

	if (!p_shader_types.has(shader_type_identifier)) {
		_set_error("Invalid shader type. Valid types are: " + _get_shader_type_list(p_shader_types));
		return ERR_PARSE_ERROR;
	}
	prev_pos = _get_tkpos();
	tk = _get_token();

	if (tk.type != TK_SEMICOLON) {
		_set_tkpos(prev_pos);
		_set_error("Expected ';' after 'shader_type <type>'.");
		return ERR_PARSE_ERROR;
	}

	tk = _get_token();

	int texture_uniforms = 0;
	int texture_binding = 0;
	int uniforms = 0;
	int instance_index = 0;
	ShaderNode::Uniform::Scope uniform_scope = ShaderNode::Uniform::SCOPE_LOCAL;

	stages = &p_functions;

	while (tk.type != TK_EOF) {
		switch (tk.type) {
			case TK_RENDER_MODE: {
				while (true) {
					StringName mode;
					_get_completable_identifier(nullptr, COMPLETION_RENDER_MODE, mode);

					if (mode == StringName()) {
						_set_error("Expected identifier for render mode");
						return ERR_PARSE_ERROR;
					}

					if (p_render_modes.find(mode) == -1) {
						_set_error("Invalid render mode: '" + String(mode) + "'");
						return ERR_PARSE_ERROR;
					}

					if (shader->render_modes.find(mode) != -1) {
						_set_error("Duplicate render mode: '" + String(mode) + "'");
						return ERR_PARSE_ERROR;
					}

					shader->render_modes.push_back(mode);

					tk = _get_token();
					if (tk.type == TK_COMMA) {
						//all good, do nothing
					} else if (tk.type == TK_SEMICOLON) {
						break; //done
					} else {
						_set_error("Unexpected token: " + get_token_text(tk));
						return ERR_PARSE_ERROR;
					}
				}
			} break;
			case TK_STRUCT: {
				ShaderNode::Struct st;
				DataType type;

				tk = _get_token();
				if (tk.type == TK_IDENTIFIER) {
					st.name = tk.text;
					if (shader->structs.has(st.name)) {
						_set_error("Redefinition of '" + String(st.name) + "'");
						return ERR_PARSE_ERROR;
					}
					tk = _get_token();
					if (tk.type != TK_CURLY_BRACKET_OPEN) {
						_set_error("Expected '{'");
						return ERR_PARSE_ERROR;
					}
				} else {
					_set_error("Expected struct identifier!");
					return ERR_PARSE_ERROR;
				}

				StructNode *st_node = alloc_node<StructNode>();
				st.shader_struct = st_node;

				int member_count = 0;
				Set<String> member_names;
				while (true) { // variables list
					tk = _get_token();
					if (tk.type == TK_CURLY_BRACKET_CLOSE) {
						break;
					}
					StringName struct_name = "";
					bool struct_dt = false;
					bool use_precision = false;
					DataPrecision precision = DataPrecision::PRECISION_DEFAULT;

					if (tk.type == TK_STRUCT) {
						_set_error("nested structs are not allowed!");
						return ERR_PARSE_ERROR;
					}

					if (is_token_precision(tk.type)) {
						precision = get_token_precision(tk.type);
						use_precision = true;
						tk = _get_token();
					}

					if (shader->structs.has(tk.text)) {
						struct_name = tk.text;
						struct_dt = true;
						if (use_precision) {
							_set_error("Precision modifier cannot be used on structs.");
							return ERR_PARSE_ERROR;
						}
					}

					if (!is_token_datatype(tk.type) && !struct_dt) {
						_set_error("Expected datatype.");
						return ERR_PARSE_ERROR;
					} else {
						type = struct_dt ? TYPE_STRUCT : get_token_datatype(tk.type);

						if (is_sampler_type(type)) {
							_set_error("sampler datatype not allowed here");
							return ERR_PARSE_ERROR;
						} else if (type == TYPE_VOID) {
							_set_error("void datatype not allowed here");
							return ERR_PARSE_ERROR;
						}
						tk = _get_token();

						if (tk.type != TK_IDENTIFIER && tk.type != TK_BRACKET_OPEN) {
							_set_error("Expected identifier or '['.");
							return ERR_PARSE_ERROR;
						}

						int array_size = 0;

						if (tk.type == TK_BRACKET_OPEN) {
							Error error = _parse_global_array_size(array_size);
							if (error != OK) {
								return error;
							}
							tk = _get_token();
						}

						if (tk.type != TK_IDENTIFIER) {
							_set_error("Expected identifier!");
							return ERR_PARSE_ERROR;
						}

						MemberNode *member = alloc_node<MemberNode>();
						member->precision = precision;
						member->datatype = type;
						member->struct_name = struct_name;
						member->name = tk.text;
						member->array_size = array_size;

						if (member_names.has(member->name)) {
							_set_error("Redefinition of '" + String(member->name) + "'");
							return ERR_PARSE_ERROR;
						}
						member_names.insert(member->name);
						tk = _get_token();

						if (tk.type == TK_BRACKET_OPEN) {
							Error error = _parse_global_array_size(member->array_size);
							if (error != OK) {
								return error;
							}
							tk = _get_token();
						}

						if (tk.type != TK_SEMICOLON) {
							_set_error("Expected ';'");
							return ERR_PARSE_ERROR;
						}

						st_node->members.push_back(member);
						member_count++;
					}
				}
				if (member_count == 0) {
					_set_error("Empty structs are not allowed!");
					return ERR_PARSE_ERROR;
				}

				tk = _get_token();
				if (tk.type != TK_SEMICOLON) {
					_set_error("Expected ';'");
					return ERR_PARSE_ERROR;
				}
				shader->structs[st.name] = st;
				shader->vstructs.push_back(st); // struct's order is important!
#ifdef DEBUG_ENABLED
				if (check_warnings && HAS_WARNING(ShaderWarning::UNUSED_STRUCT_FLAG)) {
					used_structs.insert(st.name, Usage(tk_line));
				}
#endif // DEBUG_ENABLED
			} break;
			case TK_GLOBAL: {
				tk = _get_token();
				if (tk.type != TK_UNIFORM) {
					_set_error("Expected 'uniform' after 'global'");
					return ERR_PARSE_ERROR;
				}
				uniform_scope = ShaderNode::Uniform::SCOPE_GLOBAL;
			};
				[[fallthrough]];
			case TK_INSTANCE: {
				if (uniform_scope == ShaderNode::Uniform::SCOPE_LOCAL) {
					tk = _get_token();
					if (tk.type != TK_UNIFORM) {
						_set_error("Expected 'uniform' after 'instance'");
						return ERR_PARSE_ERROR;
					}
					uniform_scope = ShaderNode::Uniform::SCOPE_INSTANCE;
				}
			};
				[[fallthrough]];
			case TK_UNIFORM:
			case TK_VARYING: {
				bool uniform = tk.type == TK_UNIFORM;

				if (!uniform) {
					if (shader_type_identifier == "particles" || shader_type_identifier == "sky") {
						_set_error(vformat("Varyings cannot be used in '%s' shaders!", shader_type_identifier));
						return ERR_PARSE_ERROR;
					}
				}

				bool precision_defined = false;
				DataPrecision precision = PRECISION_DEFAULT;
				DataInterpolation interpolation = INTERPOLATION_SMOOTH;
				DataType type;
				StringName name;
				int array_size = 0;

				tk = _get_token();
				if (is_token_interpolation(tk.type)) {
					if (uniform) {
						_set_error("Interpolation qualifiers are not supported for uniforms!");
						return ERR_PARSE_ERROR;
					}
					interpolation = get_token_interpolation(tk.type);
					tk = _get_token();
				}

				if (is_token_precision(tk.type)) {
					precision = get_token_precision(tk.type);
					precision_defined = true;
					tk = _get_token();
				}

				if (shader->structs.has(tk.text)) {
					if (uniform) {
						if (precision_defined) {
							_set_error("Precision modifier cannot be used on structs.");
							return ERR_PARSE_ERROR;
						}
						_set_error("struct datatype is not yet supported for uniforms!");
						return ERR_PARSE_ERROR;
					} else {
						_set_error("struct datatype not allowed here");
						return ERR_PARSE_ERROR;
					}
				}

				if (!is_token_datatype(tk.type)) {
					_set_error("Expected datatype. ");
					return ERR_PARSE_ERROR;
				}

				type = get_token_datatype(tk.type);

				if (type == TYPE_VOID) {
					_set_error("void datatype not allowed here");
					return ERR_PARSE_ERROR;
				}

				if (!uniform && (type < TYPE_FLOAT || type > TYPE_MAT4)) {
					_set_error("Invalid type for varying, only float,vec2,vec3,vec4,mat2,mat3,mat4 or array of these types allowed.");
					return ERR_PARSE_ERROR;
				}

				tk = _get_token();

				if (tk.type != TK_IDENTIFIER && tk.type != TK_BRACKET_OPEN) {
					_set_error("Expected identifier or '['.");
					return ERR_PARSE_ERROR;
				}

				if (tk.type == TK_BRACKET_OPEN) {
					Error error = _parse_global_array_size(array_size);
					if (error != OK) {
						return error;
					}
					tk = _get_token();
				}

				if (tk.type != TK_IDENTIFIER) {
					_set_error("Expected identifier!");
					return ERR_PARSE_ERROR;
				}

				prev_pos = _get_tkpos();
				name = tk.text;

				if (_find_identifier(nullptr, false, FunctionInfo(), name)) {
					_set_error("Redefinition of '" + String(name) + "'");
					return ERR_PARSE_ERROR;
				}

				if (has_builtin(p_functions, name)) {
					_set_error("Redefinition of '" + String(name) + "'");
					return ERR_PARSE_ERROR;
				}

				if (uniform) {
					if (uniform_scope == ShaderNode::Uniform::SCOPE_GLOBAL) {
						//validate global uniform
						DataType gvtype = global_var_get_type_func(name);
						if (gvtype == TYPE_MAX) {
							_set_error("Global uniform '" + String(name) + "' does not exist. Create it in Project Settings.");
							return ERR_PARSE_ERROR;
						}

						if (type != gvtype) {
							_set_error("Global uniform '" + String(name) + "' must be of type '" + get_datatype_name(gvtype) + "'.");
							return ERR_PARSE_ERROR;
						}
					}
					ShaderNode::Uniform uniform2;

					uniform2.type = type;
					uniform2.scope = uniform_scope;
					uniform2.precision = precision;
					uniform2.array_size = array_size;

					tk = _get_token();
					if (tk.type == TK_BRACKET_OPEN) {
						Error error = _parse_global_array_size(uniform2.array_size);
						if (error != OK) {
							return error;
						}
						tk = _get_token();
					}

					if (is_sampler_type(type)) {
						if (uniform_scope == ShaderNode::Uniform::SCOPE_INSTANCE) {
							_set_error("Uniforms with 'instance' qualifiers can't be of sampler type.");
							return ERR_PARSE_ERROR;
						}
						uniform2.texture_order = texture_uniforms++;
						uniform2.texture_binding = texture_binding;
						if (uniform2.array_size > 0) {
							texture_binding += uniform2.array_size;
						} else {
							++texture_binding;
						}
						uniform2.order = -1;
						if (_validate_datatype(type) != OK) {
							return ERR_PARSE_ERROR;
						}
					} else {
						if (uniform_scope == ShaderNode::Uniform::SCOPE_INSTANCE && (type == TYPE_MAT2 || type == TYPE_MAT3 || type == TYPE_MAT4)) {
							_set_error("Uniforms with 'instance' qualifiers can't be of matrix type.");
							return ERR_PARSE_ERROR;
						}
						uniform2.texture_order = -1;
						if (uniform_scope != ShaderNode::Uniform::SCOPE_INSTANCE) {
							uniform2.order = uniforms++;
						}
					}

					if (uniform2.array_size > 0) {
						if (uniform_scope == ShaderNode::Uniform::SCOPE_GLOBAL) {
							_set_error("'SCOPE_GLOBAL' qualifier is not yet supported for uniform array!");
							return ERR_PARSE_ERROR;
						}
						if (uniform_scope == ShaderNode::Uniform::SCOPE_INSTANCE) {
							_set_error("'SCOPE_INSTANCE' qualifier is not yet supported for uniform array!");
							return ERR_PARSE_ERROR;
						}
					}

					int custom_instance_index = -1;

					if (tk.type == TK_COLON) {
						//hint
						do {
							tk = _get_token();

							if (uniform2.array_size > 0) {
								if (tk.type != TK_HINT_COLOR) {
									_set_error("This hint is not yet supported for uniform arrays!");
									return ERR_PARSE_ERROR;
								}
							}

							if (tk.type == TK_HINT_WHITE_TEXTURE) {
								uniform2.hint = ShaderNode::Uniform::HINT_WHITE;
							} else if (tk.type == TK_HINT_BLACK_TEXTURE) {
								uniform2.hint = ShaderNode::Uniform::HINT_BLACK;
							} else if (tk.type == TK_HINT_NORMAL_TEXTURE) {
								uniform2.hint = ShaderNode::Uniform::HINT_NORMAL;
							} else if (tk.type == TK_HINT_ROUGHNESS_NORMAL_TEXTURE) {
								uniform2.hint = ShaderNode::Uniform::HINT_ROUGHNESS_NORMAL;
							} else if (tk.type == TK_HINT_ROUGHNESS_R) {
								uniform2.hint = ShaderNode::Uniform::HINT_ROUGHNESS_R;
							} else if (tk.type == TK_HINT_ROUGHNESS_G) {
								uniform2.hint = ShaderNode::Uniform::HINT_ROUGHNESS_G;
							} else if (tk.type == TK_HINT_ROUGHNESS_B) {
								uniform2.hint = ShaderNode::Uniform::HINT_ROUGHNESS_B;
							} else if (tk.type == TK_HINT_ROUGHNESS_A) {
								uniform2.hint = ShaderNode::Uniform::HINT_ROUGHNESS_A;
							} else if (tk.type == TK_HINT_ROUGHNESS_GRAY) {
								uniform2.hint = ShaderNode::Uniform::HINT_ROUGHNESS_GRAY;
							} else if (tk.type == TK_HINT_ANISO_TEXTURE) {
								uniform2.hint = ShaderNode::Uniform::HINT_ANISO;
							} else if (tk.type == TK_HINT_ALBEDO_TEXTURE) {
								uniform2.hint = ShaderNode::Uniform::HINT_ALBEDO;
							} else if (tk.type == TK_HINT_BLACK_ALBEDO_TEXTURE) {
								uniform2.hint = ShaderNode::Uniform::HINT_BLACK_ALBEDO;
							} else if (tk.type == TK_HINT_COLOR) {
								if (type != TYPE_VEC4) {
									_set_error("Color hint is for vec4 only");
									return ERR_PARSE_ERROR;
								}
								uniform2.hint = ShaderNode::Uniform::HINT_COLOR;
							} else if (tk.type == TK_HINT_RANGE) {
								uniform2.hint = ShaderNode::Uniform::HINT_RANGE;
								if (type != TYPE_FLOAT && type != TYPE_INT) {
									_set_error("Range hint is for float and int only");
									return ERR_PARSE_ERROR;
								}

								tk = _get_token();
								if (tk.type != TK_PARENTHESIS_OPEN) {
									_set_error("Expected '(' after hint_range");
									return ERR_PARSE_ERROR;
								}

								tk = _get_token();

								float sign = 1.0;

								if (tk.type == TK_OP_SUB) {
									sign = -1.0;
									tk = _get_token();
								}

								if (tk.type != TK_FLOAT_CONSTANT && tk.type != TK_INT_CONSTANT) {
									_set_error("Expected integer constant");
									return ERR_PARSE_ERROR;
								}

								uniform2.hint_range[0] = tk.constant;
								uniform2.hint_range[0] *= sign;

								tk = _get_token();

								if (tk.type != TK_COMMA) {
									_set_error("Expected ',' after integer constant");
									return ERR_PARSE_ERROR;
								}

								tk = _get_token();

								sign = 1.0;

								if (tk.type == TK_OP_SUB) {
									sign = -1.0;
									tk = _get_token();
								}

								if (tk.type != TK_FLOAT_CONSTANT && tk.type != TK_INT_CONSTANT) {
									_set_error("Expected integer constant after ','");
									return ERR_PARSE_ERROR;
								}

								uniform2.hint_range[1] = tk.constant;
								uniform2.hint_range[1] *= sign;

								tk = _get_token();

								if (tk.type == TK_COMMA) {
									tk = _get_token();

									if (tk.type != TK_FLOAT_CONSTANT && tk.type != TK_INT_CONSTANT) {
										_set_error("Expected integer constant after ','");
										return ERR_PARSE_ERROR;
									}

									uniform2.hint_range[2] = tk.constant;
									tk = _get_token();
								} else {
									if (type == TYPE_INT) {
										uniform2.hint_range[2] = 1;
									} else {
										uniform2.hint_range[2] = 0.001;
									}
								}

								if (tk.type != TK_PARENTHESIS_CLOSE) {
									_set_error("Expected ')'");
									return ERR_PARSE_ERROR;
								}
							} else if (tk.type == TK_HINT_INSTANCE_INDEX) {
								if (custom_instance_index != -1) {
									_set_error("Can only specify 'instance_index' once.");
									return ERR_PARSE_ERROR;
								}

								tk = _get_token();
								if (tk.type != TK_PARENTHESIS_OPEN) {
									_set_error("Expected '(' after 'instance_index'");
									return ERR_PARSE_ERROR;
								}

								tk = _get_token();

								if (tk.type == TK_OP_SUB) {
									_set_error("The instance index can't be negative.");
									return ERR_PARSE_ERROR;
								}

								if (tk.type != TK_INT_CONSTANT) {
									_set_error("Expected integer constant");
									return ERR_PARSE_ERROR;
								}

								custom_instance_index = tk.constant;

								if (custom_instance_index >= MAX_INSTANCE_UNIFORM_INDICES) {
									_set_error("Allowed instance uniform indices are 0-" + itos(MAX_INSTANCE_UNIFORM_INDICES - 1));
									return ERR_PARSE_ERROR;
								}

								tk = _get_token();

								if (tk.type != TK_PARENTHESIS_CLOSE) {
									_set_error("Expected ')'");
									return ERR_PARSE_ERROR;
								}
							} else if (tk.type == TK_FILTER_LINEAR) {
								uniform2.filter = FILTER_LINEAR;
							} else if (tk.type == TK_FILTER_NEAREST) {
								uniform2.filter = FILTER_NEAREST;
							} else if (tk.type == TK_FILTER_NEAREST_MIPMAP) {
								uniform2.filter = FILTER_NEAREST_MIPMAP;
							} else if (tk.type == TK_FILTER_LINEAR_MIPMAP) {
								uniform2.filter = FILTER_LINEAR_MIPMAP;
							} else if (tk.type == TK_FILTER_NEAREST_MIPMAP_ANISO) {
								uniform2.filter = FILTER_NEAREST_MIPMAP_ANISO;
							} else if (tk.type == TK_FILTER_LINEAR_MIPMAP_ANISO) {
								uniform2.filter = FILTER_LINEAR_MIPMAP_ANISO;
							} else if (tk.type == TK_REPEAT_DISABLE) {
								uniform2.repeat = REPEAT_DISABLE;
							} else if (tk.type == TK_REPEAT_ENABLE) {
								uniform2.repeat = REPEAT_ENABLE;
							} else {
								_set_error("Expected valid type hint after ':'.");
							}

							if (uniform2.hint != ShaderNode::Uniform::HINT_RANGE && uniform2.hint != ShaderNode::Uniform::HINT_NONE && uniform2.hint != ShaderNode::Uniform::HINT_COLOR && type <= TYPE_MAT4) {
								_set_error("This hint is only for sampler types");
								return ERR_PARSE_ERROR;
							}

							tk = _get_token();

						} while (tk.type == TK_COMMA);
					}

					if (uniform_scope == ShaderNode::Uniform::SCOPE_INSTANCE) {
						if (custom_instance_index >= 0) {
							uniform2.instance_index = custom_instance_index;
						} else {
							uniform2.instance_index = instance_index++;
							if (instance_index > MAX_INSTANCE_UNIFORM_INDICES) {
								_set_error("Too many 'instance' uniforms in shader, maximum supported is " + itos(MAX_INSTANCE_UNIFORM_INDICES));
								return ERR_PARSE_ERROR;
							}
						}
					}

					//reset scope for next uniform

					if (tk.type == TK_OP_ASSIGN) {
						if (uniform2.array_size > 0) {
							_set_error("Setting default value to a uniform array is not yet supported!");
							return ERR_PARSE_ERROR;
						}

						Node *expr = _parse_and_reduce_expression(nullptr, FunctionInfo());
						if (!expr) {
							return ERR_PARSE_ERROR;
						}
						if (expr->type != Node::TYPE_CONSTANT) {
							_set_error("Expected constant expression after '='");
							return ERR_PARSE_ERROR;
						}

						ConstantNode *cn = static_cast<ConstantNode *>(expr);

						uniform2.default_value.resize(cn->values.size());

						if (!convert_constant(cn, uniform2.type, uniform2.default_value.ptrw())) {
							_set_error("Can't convert constant to " + get_datatype_name(uniform2.type));
							return ERR_PARSE_ERROR;
						}
						tk = _get_token();
					}

					shader->uniforms[name] = uniform2;
#ifdef DEBUG_ENABLED
					if (check_warnings && HAS_WARNING(ShaderWarning::UNUSED_UNIFORM_FLAG)) {
						used_uniforms.insert(name, Usage(tk_line));
					}
#endif // DEBUG_ENABLED

					//reset scope for next uniform
					uniform_scope = ShaderNode::Uniform::SCOPE_LOCAL;

					if (tk.type != TK_SEMICOLON) {
						_set_error("Expected ';'");
						return ERR_PARSE_ERROR;
					}
				} else { // varying
					ShaderNode::Varying varying;
					varying.type = type;
					varying.precision = precision;
					varying.interpolation = interpolation;
					varying.tkpos = prev_pos;
					varying.array_size = array_size;

					tk = _get_token();
					if (tk.type != TK_SEMICOLON && tk.type != TK_BRACKET_OPEN) {
						if (array_size == 0) {
							_set_error("Expected ';' or '['");
						} else {
							_set_error("Expected ';'");
						}
						return ERR_PARSE_ERROR;
					}

					if (tk.type == TK_BRACKET_OPEN) {
						if (array_size > 0) {
							_set_error("Array size is already defined!");
							return ERR_PARSE_ERROR;
						}
						tk = _get_token();
						if (tk.type == TK_INT_CONSTANT && tk.constant > 0) {
							varying.array_size = (int)tk.constant;

							tk = _get_token();
							if (tk.type == TK_BRACKET_CLOSE) {
								tk = _get_token();
								if (tk.type != TK_SEMICOLON) {
									_set_error("Expected ';'");
									return ERR_PARSE_ERROR;
								}
							} else {
								_set_error("Expected ']'");
								return ERR_PARSE_ERROR;
							}
						} else {
							_set_error("Expected integer constant > 0");
							return ERR_PARSE_ERROR;
						}
					}

					shader->varyings[name] = varying;
#ifdef DEBUG_ENABLED
					if (check_warnings && HAS_WARNING(ShaderWarning::UNUSED_VARYING_FLAG)) {
						used_varyings.insert(name, Usage(tk_line));
					}
#endif // DEBUG_ENABLED
				}

			} break;
			default: {
				//function or constant variable

				bool is_constant = false;
				bool is_struct = false;
				StringName struct_name;
				DataPrecision precision = PRECISION_DEFAULT;
				DataType type;
				StringName name;
				int return_array_size = 0;

				if (tk.type == TK_CONST) {
					is_constant = true;
					tk = _get_token();
				}

				if (is_token_precision(tk.type)) {
					precision = get_token_precision(tk.type);
					tk = _get_token();
				}

				if (shader->structs.has(tk.text)) {
					if (precision != PRECISION_DEFAULT) {
						_set_error("Precision modifier cannot be used on structs.");
						return ERR_PARSE_ERROR;
					}
					is_struct = true;
					struct_name = tk.text;
				} else {
					if (!is_token_datatype(tk.type)) {
						_set_error("Expected constant, function, uniform or varying");
						return ERR_PARSE_ERROR;
					}

					if (!is_token_variable_datatype(tk.type)) {
						_set_error("Invalid data type for constants or function return (samplers not allowed)");
						return ERR_PARSE_ERROR;
					}
				}

				if (is_struct) {
					type = TYPE_STRUCT;
				} else {
					type = get_token_datatype(tk.type);
				}
				prev_pos = _get_tkpos();
				tk = _get_token();

				if (tk.type == TK_BRACKET_OPEN) {
					bool error = false;
					tk = _get_token();

					if (tk.type == TK_INT_CONSTANT) {
						return_array_size = (int)tk.constant;
						if (return_array_size > 0) {
							tk = _get_token();
							if (tk.type != TK_BRACKET_CLOSE) {
								_set_error("Expected ']'");
								return ERR_PARSE_ERROR;
							}
						} else {
							error = true;
						}
					} else {
						error = true;
					}
					if (error) {
						_set_error("Expected integer constant > 0");
						return ERR_PARSE_ERROR;
					}

					prev_pos = _get_tkpos();
				}

				_set_tkpos(prev_pos);

				_get_completable_identifier(nullptr, COMPLETION_MAIN_FUNCTION, name);

				if (name == StringName()) {
					_set_error("Expected function name after datatype");
					return ERR_PARSE_ERROR;
				}

				if (_find_identifier(nullptr, false, FunctionInfo(), name)) {
					_set_error("Redefinition of '" + String(name) + "'");
					return ERR_PARSE_ERROR;
				}

				if (has_builtin(p_functions, name)) {
					_set_error("Redefinition of '" + String(name) + "'");
					return ERR_PARSE_ERROR;
				}

				tk = _get_token();
				if (tk.type != TK_PARENTHESIS_OPEN) {
					if (type == TYPE_VOID) {
						_set_error("Expected '(' after function identifier");
						return ERR_PARSE_ERROR;
					}

					//variable

					while (true) {
						ShaderNode::Constant constant;
						constant.name = name;
						constant.type = is_struct ? TYPE_STRUCT : type;
						constant.type_str = struct_name;
						constant.precision = precision;
						constant.initializer = nullptr;
						constant.array_size = 0;

						bool unknown_size = false;

						if (tk.type == TK_BRACKET_OPEN) {
							if (RenderingServer::get_singleton()->is_low_end()) {
								_set_error("Global const arrays are supported only on high-end platform!");
								return ERR_PARSE_ERROR;
							}

							tk = _get_token();
							if (tk.type == TK_BRACKET_CLOSE) {
								unknown_size = true;
								tk = _get_token();
							} else if (tk.type == TK_INT_CONSTANT && ((int)tk.constant) > 0) {
								constant.array_size = (int)tk.constant;
								tk = _get_token();
								if (tk.type != TK_BRACKET_CLOSE) {
									_set_error("Expected ']'");
									return ERR_PARSE_ERROR;
								}
								tk = _get_token();
							} else {
								_set_error("Expected integer constant > 0 or ']'");
								return ERR_PARSE_ERROR;
							}
						}

						if (tk.type == TK_OP_ASSIGN) {
							if (!is_constant) {
								_set_error("Expected 'const' keyword before constant definition");
								return ERR_PARSE_ERROR;
							}

							if (constant.array_size > 0 || unknown_size) {
								bool full_def = false;

								ArrayDeclarationNode::Declaration decl;
								decl.name = name;
								decl.size = constant.array_size;

								tk = _get_token();

								if (tk.type != TK_CURLY_BRACKET_OPEN) {
									if (unknown_size) {
										_set_error("Expected '{'");
										return ERR_PARSE_ERROR;
									}

									full_def = true;

									DataPrecision precision2 = PRECISION_DEFAULT;
									if (is_token_precision(tk.type)) {
										precision2 = get_token_precision(tk.type);
										tk = _get_token();
										if (!is_token_nonvoid_datatype(tk.type)) {
											_set_error("Expected datatype after precision");
											return ERR_PARSE_ERROR;
										}
									}

									StringName struct_name2;
									DataType type2;

									if (shader->structs.has(tk.text)) {
										type2 = TYPE_STRUCT;
										struct_name2 = tk.text;
									} else {
										if (!is_token_variable_datatype(tk.type)) {
											_set_error("Invalid data type for array");
											return ERR_PARSE_ERROR;
										}
										type2 = get_token_datatype(tk.type);
									}

									int array_size2 = 0;

									tk = _get_token();
									if (tk.type == TK_BRACKET_OPEN) {
										prev_pos = _get_tkpos();
										tk = _get_token();
										if (tk.type == TK_BRACKET_CLOSE) {
											array_size2 = constant.array_size;
											tk = _get_token();
										} else {
											_set_tkpos(prev_pos);

											Node *n = _parse_and_reduce_expression(nullptr, FunctionInfo());
											if (!n || n->type != Node::TYPE_CONSTANT || n->get_datatype() != TYPE_INT) {
												_set_error("Expected single integer constant > 0");
												return ERR_PARSE_ERROR;
											}

											ConstantNode *cnode = (ConstantNode *)n;
											if (cnode->values.size() == 1) {
												array_size2 = cnode->values[0].sint;
												if (array_size2 <= 0) {
													_set_error("Expected single integer constant > 0");
													return ERR_PARSE_ERROR;
												}
											} else {
												_set_error("Expected single integer constant > 0");
												return ERR_PARSE_ERROR;
											}

											tk = _get_token();
											if (tk.type != TK_BRACKET_CLOSE) {
												_set_error("Expected ']");
												return ERR_PARSE_ERROR;
											} else {
												tk = _get_token();
											}
										}
									} else {
										_set_error("Expected '[");
										return ERR_PARSE_ERROR;
									}

									if (constant.precision != precision2 || constant.type != type2 || struct_name != struct_name2 || constant.array_size != array_size2) {
										String error_str = "Cannot convert from '";
										if (type2 == TYPE_STRUCT) {
											error_str += struct_name2;
										} else {
											if (precision2 != PRECISION_DEFAULT) {
												error_str += get_precision_name(precision2);
												error_str += " ";
											}
											error_str += get_datatype_name(type2);
										}
										error_str += "[";
										error_str += itos(array_size2);
										error_str += "]'";
										error_str += " to '";
										if (type == TYPE_STRUCT) {
											error_str += struct_name;
										} else {
											if (precision != PRECISION_DEFAULT) {
												error_str += get_precision_name(precision);
												error_str += " ";
											}
											error_str += get_datatype_name(type);
										}
										error_str += "[";
										error_str += itos(constant.array_size);
										error_str += "]'";
										_set_error(error_str);
										return ERR_PARSE_ERROR;
									}
								}

								bool curly = tk.type == TK_CURLY_BRACKET_OPEN;

								if (unknown_size) {
									if (!curly) {
										_set_error("Expected '{'");
										return ERR_PARSE_ERROR;
									}
								} else {
									if (full_def) {
										if (curly) {
											_set_error("Expected '('");
											return ERR_PARSE_ERROR;
										}
									}
								}

								if (tk.type == TK_PARENTHESIS_OPEN || curly) { // initialization
									while (true) {
										Node *n = _parse_and_reduce_expression(nullptr, FunctionInfo());
										if (!n) {
											return ERR_PARSE_ERROR;
										}

										if (n->type == Node::TYPE_OPERATOR && ((OperatorNode *)n)->op == OP_CALL) {
											_set_error("Expected constant expression");
											return ERR_PARSE_ERROR;
										}

										if (!_compare_datatypes(constant.type, struct_name, 0, n->get_datatype(), n->get_datatype_name(), 0)) {
											return ERR_PARSE_ERROR;
										}

										tk = _get_token();
										if (tk.type == TK_COMMA) {
											decl.initializer.push_back(n);
											continue;
										} else if (!curly && tk.type == TK_PARENTHESIS_CLOSE) {
											decl.initializer.push_back(n);
											break;
										} else if (curly && tk.type == TK_CURLY_BRACKET_CLOSE) {
											decl.initializer.push_back(n);
											break;
										} else {
											if (curly) {
												_set_error("Expected '}' or ','");
											} else {
												_set_error("Expected ')' or ','");
											}
											return ERR_PARSE_ERROR;
										}
									}
									if (unknown_size) {
										decl.size = decl.initializer.size();
										constant.array_size = decl.initializer.size();
									} else if (decl.initializer.size() != constant.array_size) {
										_set_error("Array size mismatch");
										return ERR_PARSE_ERROR;
									}
								}

								ConstantNode *expr = memnew(ConstantNode);

								expr->datatype = constant.type;

								expr->struct_name = constant.type_str;

								expr->array_size = constant.array_size;

								expr->array_declarations.push_back(decl);

								constant.initializer = static_cast<ConstantNode *>(expr);
							} else {
								//variable created with assignment! must parse an expression
								Node *expr = _parse_and_reduce_expression(nullptr, FunctionInfo());
								if (!expr) {
									return ERR_PARSE_ERROR;
								}
								if (expr->type == Node::TYPE_OPERATOR && ((OperatorNode *)expr)->op == OP_CALL) {
									OperatorNode *op = ((OperatorNode *)expr);
									for (int i = 1; i < op->arguments.size(); i++) {
										if (!_check_node_constness(op->arguments[i])) {
											_set_error("Expected constant expression for argument '" + itos(i - 1) + "' of function call after '='");
											return ERR_PARSE_ERROR;
										}
									}
								}

								constant.initializer = static_cast<ConstantNode *>(expr);

								if (!_compare_datatypes(type, struct_name, 0, expr->get_datatype(), expr->get_datatype_name(), 0)) {
									return ERR_PARSE_ERROR;
								}
							}
							tk = _get_token();
						} else {
							if (constant.array_size > 0 || unknown_size) {
								_set_error("Expected array initialization");
								return ERR_PARSE_ERROR;
							} else {
								_set_error("Expected initialization of constant");
								return ERR_PARSE_ERROR;
							}
						}

						shader->constants[name] = constant;
						shader->vconstants.push_back(constant);
#ifdef DEBUG_ENABLED
						if (check_warnings && HAS_WARNING(ShaderWarning::UNUSED_CONSTANT_FLAG)) {
							used_constants.insert(name, Usage(tk_line));
						}
#endif // DEBUG_ENABLED

						if (tk.type == TK_COMMA) {
							tk = _get_token();
							if (tk.type != TK_IDENTIFIER) {
								_set_error("Expected identifier after type");
								return ERR_PARSE_ERROR;
							}

							name = tk.text;
							if (_find_identifier(nullptr, false, FunctionInfo(), name)) {
								_set_error("Redefinition of '" + String(name) + "'");
								return ERR_PARSE_ERROR;
							}

							if (has_builtin(p_functions, name)) {
								_set_error("Redefinition of '" + String(name) + "'");
								return ERR_PARSE_ERROR;
							}

							tk = _get_token();

						} else if (tk.type == TK_SEMICOLON) {
							break;
						} else {
							_set_error("Expected ',' or ';' after constant");
							return ERR_PARSE_ERROR;
						}
					}

					break;
				}

				FunctionInfo builtins;
				if (p_functions.has(name)) {
					builtins = p_functions[name];
				}

				if (p_functions.has("global")) { // Adds global variables: 'TIME'
					for (const KeyValue<StringName, BuiltInInfo> &E : p_functions["global"].built_ins) {
						builtins.built_ins.insert(E.key, E.value);
					}
				}

				ShaderNode::Function function;

				function.callable = !p_functions.has(name);
				function.name = name;

				FunctionNode *func_node = alloc_node<FunctionNode>();

				function.function = func_node;

				shader->functions.push_back(function);

				func_node->name = name;
				func_node->return_type = type;
				func_node->return_struct_name = struct_name;
				func_node->return_precision = precision;
				func_node->return_array_size = return_array_size;

				if (p_functions.has(name)) {
					func_node->can_discard = p_functions[name].can_discard;
				} else {
#ifdef DEBUG_ENABLED
					if (check_warnings && HAS_WARNING(ShaderWarning::UNUSED_FUNCTION_FLAG)) {
						used_functions.insert(name, Usage(tk_line));
					}
#endif // DEBUG_ENABLED
				}

				func_node->body = alloc_node<BlockNode>();
				func_node->body->parent_function = func_node;

				tk = _get_token();

				while (true) {
					if (tk.type == TK_PARENTHESIS_CLOSE) {
						break;
					}

					bool is_const = false;
					if (tk.type == TK_CONST) {
						is_const = true;
						tk = _get_token();
					}

					ArgumentQualifier qualifier = ARGUMENT_QUALIFIER_IN;

					if (tk.type == TK_ARG_IN) {
						qualifier = ARGUMENT_QUALIFIER_IN;
						tk = _get_token();
					} else if (tk.type == TK_ARG_OUT) {
						if (is_const) {
							_set_error("'out' qualifier cannot be used within a function parameter declared with 'const'.");
							return ERR_PARSE_ERROR;
						}
						qualifier = ARGUMENT_QUALIFIER_OUT;
						tk = _get_token();
					} else if (tk.type == TK_ARG_INOUT) {
						if (is_const) {
							_set_error("'inout' qualifier cannot be used within a function parameter declared with 'const'.");
							return ERR_PARSE_ERROR;
						}
						qualifier = ARGUMENT_QUALIFIER_INOUT;
						tk = _get_token();
					}

					DataType ptype;
					StringName pname;
					StringName param_struct_name;
					DataPrecision pprecision = PRECISION_DEFAULT;
					bool use_precision = false;
					int array_size = 0;

					if (is_token_precision(tk.type)) {
						pprecision = get_token_precision(tk.type);
						tk = _get_token();
						use_precision = true;
					}

					is_struct = false;

					if (shader->structs.has(tk.text)) {
						is_struct = true;
						param_struct_name = tk.text;
						if (use_precision) {
							_set_error("Precision modifier cannot be used on structs.");
							return ERR_PARSE_ERROR;
						}
					}

					if (!is_struct && !is_token_datatype(tk.type)) {
						_set_error("Expected a valid datatype for argument");
						return ERR_PARSE_ERROR;
					}

					if (qualifier == ARGUMENT_QUALIFIER_OUT || qualifier == ARGUMENT_QUALIFIER_INOUT) {
						if (is_sampler_type(get_token_datatype(tk.type))) {
							_set_error("Opaque types cannot be output parameters.");
							return ERR_PARSE_ERROR;
						}
					}

					if (is_struct) {
						ptype = TYPE_STRUCT;
					} else {
						ptype = get_token_datatype(tk.type);
						if (_validate_datatype(ptype) != OK) {
							return ERR_PARSE_ERROR;
						}
						if (ptype == TYPE_VOID) {
							_set_error("void not allowed in argument");
							return ERR_PARSE_ERROR;
						}
					}

					tk = _get_token();

					if (tk.type == TK_BRACKET_OPEN) {
						bool error = false;
						tk = _get_token();

						if (tk.type == TK_INT_CONSTANT) {
							array_size = (int)tk.constant;

							if (array_size > 0) {
								tk = _get_token();
								if (tk.type != TK_BRACKET_CLOSE) {
									_set_error("Expected ']'");
									return ERR_PARSE_ERROR;
								}
							} else {
								error = true;
							}
						} else {
							error = true;
						}
						if (error) {
							_set_error("Expected integer constant > 0");
							return ERR_PARSE_ERROR;
						}
						tk = _get_token();
					}
					if (tk.type != TK_IDENTIFIER) {
						_set_error("Expected identifier for argument name");
						return ERR_PARSE_ERROR;
					}

					pname = tk.text;

					ShaderLanguage::IdentifierType itype;
					if (_find_identifier(func_node->body, false, builtins, pname, (ShaderLanguage::DataType *)nullptr, &itype)) {
						if (itype != IDENTIFIER_FUNCTION) {
							_set_error("Redefinition of '" + String(pname) + "'");
							return ERR_PARSE_ERROR;
						}
					}

					if (has_builtin(p_functions, pname)) {
						_set_error("Redefinition of '" + String(pname) + "'");
						return ERR_PARSE_ERROR;
					}

					FunctionNode::Argument arg;
					arg.type = ptype;
					arg.name = pname;
					arg.type_str = param_struct_name;
					arg.precision = pprecision;
					arg.qualifier = qualifier;
					arg.tex_argument_check = false;
					arg.tex_builtin_check = false;
					arg.tex_argument_filter = FILTER_DEFAULT;
					arg.tex_argument_repeat = REPEAT_DEFAULT;
					arg.is_const = is_const;

					tk = _get_token();
					if (tk.type == TK_BRACKET_OPEN) {
						if (array_size > 0) {
							_set_error("Array size is already defined!");
							return ERR_PARSE_ERROR;
						}
						bool error = false;
						tk = _get_token();

						if (tk.type == TK_INT_CONSTANT) {
							array_size = (int)tk.constant;

							if (array_size > 0) {
								tk = _get_token();
								if (tk.type != TK_BRACKET_CLOSE) {
									_set_error("Expected ']'");
									return ERR_PARSE_ERROR;
								}
							} else {
								error = true;
							}
						} else {
							error = true;
						}

						if (error) {
							_set_error("Expected integer constant > 0");
							return ERR_PARSE_ERROR;
						}
						tk = _get_token();
					}

					arg.array_size = array_size;
					func_node->arguments.push_back(arg);

					if (tk.type == TK_COMMA) {
						tk = _get_token();
						//do none and go on
					} else if (tk.type != TK_PARENTHESIS_CLOSE) {
						_set_error("Expected ',' or ')' after identifier");
						return ERR_PARSE_ERROR;
					}
				}

				if (p_functions.has(name)) {
					//if one of the core functions, make sure they are of the correct form
					if (func_node->arguments.size() > 0) {
						_set_error("Function '" + String(name) + "' expects no arguments.");
						return ERR_PARSE_ERROR;
					}
					if (func_node->return_type != TYPE_VOID) {
						_set_error("Function '" + String(name) + "' must be of void return type.");
						return ERR_PARSE_ERROR;
					}
				}

				//all good let's parse inside the function!
				tk = _get_token();
				if (tk.type != TK_CURLY_BRACKET_OPEN) {
					_set_error("Expected '{' to begin function");
					return ERR_PARSE_ERROR;
				}

				current_function = name;

				Error err = _parse_block(func_node->body, builtins);
				if (err) {
					return err;
				}

				if (func_node->return_type != DataType::TYPE_VOID) {
					BlockNode *block = func_node->body;
					if (_find_last_flow_op_in_block(block, FlowOperation::FLOW_OP_RETURN) != OK) {
						_set_error("Expected at least one return statement in a non-void function.");
						return ERR_PARSE_ERROR;
					}
				}
				current_function = StringName();
			}
		}

		tk = _get_token();
	}

	int error_line;
	String error_message;
	if (!_check_varying_usages(&error_line, &error_message)) {
		_set_tkpos({ 0, error_line });
		_set_error(error_message);
		return ERR_PARSE_ERROR;
	}

	return OK;
}

bool ShaderLanguage::has_builtin(const Map<StringName, ShaderLanguage::FunctionInfo> &p_functions, const StringName &p_name) {
	for (const KeyValue<StringName, ShaderLanguage::FunctionInfo> &E : p_functions) {
		if (E.value.built_ins.has(p_name)) {
			return true;
		}
	}

	return false;
}

Error ShaderLanguage::_find_last_flow_op_in_op(ControlFlowNode *p_flow, FlowOperation p_op) {
	bool found = false;

	for (int i = p_flow->blocks.size() - 1; i >= 0; i--) {
		if (p_flow->blocks[i]->type == Node::TYPE_BLOCK) {
			BlockNode *last_block = (BlockNode *)p_flow->blocks[i];
			if (_find_last_flow_op_in_block(last_block, p_op) == OK) {
				found = true;
				break;
			}
		}
	}
	if (found) {
		return OK;
	}
	return FAILED;
}

Error ShaderLanguage::_find_last_flow_op_in_block(BlockNode *p_block, FlowOperation p_op) {
	bool found = false;

	for (int i = p_block->statements.size() - 1; i >= 0; i--) {
		if (p_block->statements[i]->type == Node::TYPE_CONTROL_FLOW) {
			ControlFlowNode *flow = (ControlFlowNode *)p_block->statements[i];
			if (flow->flow_op == p_op) {
				found = true;
				break;
			} else {
				if (_find_last_flow_op_in_op(flow, p_op) == OK) {
					found = true;
					break;
				}
			}
		} else if (p_block->statements[i]->type == Node::TYPE_BLOCK) {
			BlockNode *block = (BlockNode *)p_block->statements[i];
			if (_find_last_flow_op_in_block(block, p_op) == OK) {
				found = true;
				break;
			}
		}
	}

	if (found) {
		return OK;
	}
	return FAILED;
}

// skips over whitespace and /* */ and // comments
static int _get_first_ident_pos(const String &p_code) {
	int idx = 0;

#define GETCHAR(m_idx) (((idx + m_idx) < p_code.length()) ? p_code[idx + m_idx] : char32_t(0))

	while (true) {
		if (GETCHAR(0) == '/' && GETCHAR(1) == '/') {
			idx += 2;
			while (true) {
				if (GETCHAR(0) == 0) {
					return 0;
				}
				if (GETCHAR(0) == '\n') {
					idx++;
					break; // loop
				}
				idx++;
			}
		} else if (GETCHAR(0) == '/' && GETCHAR(1) == '*') {
			idx += 2;
			while (true) {
				if (GETCHAR(0) == 0) {
					return 0;
				}
				if (GETCHAR(0) == '*' && GETCHAR(1) == '/') {
					idx += 2;
					break; // loop
				}
				idx++;
			}
		} else {
			switch (GETCHAR(0)) {
				case ' ':
				case '\t':
				case '\r':
				case '\n': {
					idx++;
				} break; // switch
				default:
					return idx;
			}
		}
	}

#undef GETCHAR
}

String ShaderLanguage::get_shader_type(const String &p_code) {
	bool reading_type = false;

	String cur_identifier;

	for (int i = _get_first_ident_pos(p_code); i < p_code.length(); i++) {
		if (p_code[i] == ';') {
			break;

		} else if (p_code[i] <= 32) {
			if (cur_identifier != String()) {
				if (!reading_type) {
					if (cur_identifier != "shader_type") {
						return String();
					}

					reading_type = true;
					cur_identifier = String();
				} else {
					return cur_identifier;
				}
			}
		} else {
			cur_identifier += String::chr(p_code[i]);
		}
	}

	if (reading_type) {
		return cur_identifier;
	}

	return String();
}

#ifdef DEBUG_ENABLED
void ShaderLanguage::_check_warning_accums() {
	for (const KeyValue<ShaderWarning::Code, Map<StringName, Map<StringName, Usage>> *> &E : warnings_check_map2) {
		for (Map<StringName, Map<StringName, Usage>>::Element *T = (*E.value).front(); T; T = T->next()) {
			for (const KeyValue<StringName, Usage> &U : T->get()) {
				if (!U.value.used) {
					_add_warning(E.key, U.value.decl_line, U.key);
				}
			}
		}
	}
	for (const KeyValue<ShaderWarning::Code, Map<StringName, Usage> *> &E : warnings_check_map) {
		for (const Map<StringName, Usage>::Element *U = (*E.value).front(); U; U = U->next()) {
			if (!U->get().used) {
				_add_warning(E.key, U->get().decl_line, U->key());
			}
		}
	}
}
List<ShaderWarning>::Element *ShaderLanguage::get_warnings_ptr() {
	return warnings.front();
}
void ShaderLanguage::enable_warning_checking(bool p_enabled) {
	check_warnings = p_enabled;
}
bool ShaderLanguage::is_warning_checking_enabled() const {
	return check_warnings;
}
void ShaderLanguage::set_warning_flags(uint32_t p_flags) {
	warning_flags = p_flags;
}
uint32_t ShaderLanguage::get_warning_flags() const {
	return warning_flags;
}
#endif // DEBUG_ENABLED

Error ShaderLanguage::compile(const String &p_code, const Map<StringName, FunctionInfo> &p_functions, const Vector<StringName> &p_render_modes, const VaryingFunctionNames &p_varying_function_names, const Set<String> &p_shader_types, GlobalVariableGetTypeFunc p_global_variable_type_func) {
	clear();

	code = p_code;
	global_var_get_type_func = p_global_variable_type_func;
	varying_function_names = p_varying_function_names;

	nodes = nullptr;

	shader = alloc_node<ShaderNode>();
	Error err = _parse_shader(p_functions, p_render_modes, p_shader_types);

#ifdef DEBUG_ENABLED
	if (check_warnings) {
		_check_warning_accums();
	}
#endif // DEBUG_ENABLED

	if (err != OK) {
		return err;
	}
	return OK;
}

Error ShaderLanguage::complete(const String &p_code, const Map<StringName, FunctionInfo> &p_functions, const Vector<StringName> &p_render_modes, const VaryingFunctionNames &p_varying_function_names, const Set<String> &p_shader_types, GlobalVariableGetTypeFunc p_global_variable_type_func, List<ScriptCodeCompletionOption> *r_options, String &r_call_hint) {
	clear();

	code = p_code;
	varying_function_names = p_varying_function_names;

	nodes = nullptr;
	global_var_get_type_func = p_global_variable_type_func;

	shader = alloc_node<ShaderNode>();
	_parse_shader(p_functions, p_render_modes, p_shader_types);

	switch (completion_type) {
		case COMPLETION_NONE: {
			//do nothing
			return OK;
		} break;
		case COMPLETION_RENDER_MODE: {
			for (int i = 0; i < p_render_modes.size(); i++) {
				ScriptCodeCompletionOption option(p_render_modes[i], ScriptCodeCompletionOption::KIND_ENUM);
				r_options->push_back(option);
			}

			return OK;
		} break;
		case COMPLETION_STRUCT: {
			if (shader->structs.has(completion_struct)) {
				StructNode *node = shader->structs[completion_struct].shader_struct;
				for (int i = 0; i < node->members.size(); i++) {
					ScriptCodeCompletionOption option(node->members[i]->name, ScriptCodeCompletionOption::KIND_MEMBER);
					r_options->push_back(option);
				}
			}

			return OK;
		} break;
		case COMPLETION_MAIN_FUNCTION: {
			for (const KeyValue<StringName, FunctionInfo> &E : p_functions) {
				ScriptCodeCompletionOption option(E.key, ScriptCodeCompletionOption::KIND_FUNCTION);
				r_options->push_back(option);
			}

			return OK;
		} break;
		case COMPLETION_IDENTIFIER:
		case COMPLETION_FUNCTION_CALL: {
			bool comp_ident = completion_type == COMPLETION_IDENTIFIER;
			Map<String, ScriptCodeCompletionOption::Kind> matches;
			StringName skip_function;
			BlockNode *block = completion_block;

			if (completion_class == TAG_GLOBAL) {
				while (block) {
					if (comp_ident) {
						for (const KeyValue<StringName, BlockNode::Variable> &E : block->variables) {
							if (E.value.line < completion_line) {
								matches.insert(E.key, ScriptCodeCompletionOption::KIND_VARIABLE);
							}
						}
					}

					if (block->parent_function) {
						if (comp_ident) {
							for (int i = 0; i < block->parent_function->arguments.size(); i++) {
								matches.insert(block->parent_function->arguments[i].name, ScriptCodeCompletionOption::KIND_VARIABLE);
							}
						}
						skip_function = block->parent_function->name;
					}
					block = block->parent_block;
				}

				if (comp_ident) {
					if (p_functions.has("global")) {
						for (const KeyValue<StringName, BuiltInInfo> &E : p_functions["global"].built_ins) {
							ScriptCodeCompletionOption::Kind kind = ScriptCodeCompletionOption::KIND_MEMBER;
							if (E.value.constant) {
								kind = ScriptCodeCompletionOption::KIND_CONSTANT;
							}
							matches.insert(E.key, kind);
						}
					}

					if (skip_function != StringName() && p_functions.has(skip_function)) {
						for (const KeyValue<StringName, BuiltInInfo> &E : p_functions[skip_function].built_ins) {
							ScriptCodeCompletionOption::Kind kind = ScriptCodeCompletionOption::KIND_MEMBER;
							if (E.value.constant) {
								kind = ScriptCodeCompletionOption::KIND_CONSTANT;
							}
							matches.insert(E.key, kind);
						}
					}

					for (const KeyValue<StringName, ShaderNode::Varying> &E : shader->varyings) {
						matches.insert(E.key, ScriptCodeCompletionOption::KIND_VARIABLE);
					}
					for (const KeyValue<StringName, ShaderNode::Uniform> &E : shader->uniforms) {
						matches.insert(E.key, ScriptCodeCompletionOption::KIND_MEMBER);
					}
				}

				for (int i = 0; i < shader->functions.size(); i++) {
					if (!shader->functions[i].callable || shader->functions[i].name == skip_function) {
						continue;
					}
					matches.insert(String(shader->functions[i].name), ScriptCodeCompletionOption::KIND_FUNCTION);
				}

				int idx = 0;
				bool low_end = RenderingServer::get_singleton()->is_low_end();

				if (stages && stages->has(skip_function)) {
					for (const KeyValue<StringName, StageFunctionInfo> &E : (*stages)[skip_function].stage_functions) {
						matches.insert(String(E.key), ScriptCodeCompletionOption::KIND_FUNCTION);
					}
				}

				while (builtin_func_defs[idx].name) {
					if (low_end && builtin_func_defs[idx].high_end) {
						idx++;
						continue;
					}
					matches.insert(String(builtin_func_defs[idx].name), ScriptCodeCompletionOption::KIND_FUNCTION);
					idx++;
				}

			} else { // sub-class
				int idx = 0;
				bool low_end = RenderingServer::get_singleton()->is_low_end();

				while (builtin_func_defs[idx].name) {
					if (low_end && builtin_func_defs[idx].high_end) {
						idx++;
						continue;
					}
					if (builtin_func_defs[idx].tag == completion_class) {
						matches.insert(String(builtin_func_defs[idx].name), ScriptCodeCompletionOption::KIND_FUNCTION);
					}
					idx++;
				}
			}

			for (const KeyValue<String, ScriptCodeCompletionOption::Kind> &E : matches) {
				ScriptCodeCompletionOption option(E.key, E.value);
				if (E.value == ScriptCodeCompletionOption::KIND_FUNCTION) {
					option.insert_text += "(";
				}
				r_options->push_back(option);
			}

			return OK;
		} break;
		case COMPLETION_CALL_ARGUMENTS: {
			StringName block_function;
			BlockNode *block = completion_block;

			while (block) {
				if (block->parent_function) {
					block_function = block->parent_function->name;
				}
				block = block->parent_block;
			}

			for (int i = 0; i < shader->functions.size(); i++) {
				if (!shader->functions[i].callable) {
					continue;
				}
				if (shader->functions[i].name == completion_function) {
					String calltip;

					calltip += get_datatype_name(shader->functions[i].function->return_type);

					if (shader->functions[i].function->return_array_size > 0) {
						calltip += "[";
						calltip += itos(shader->functions[i].function->return_array_size);
						calltip += "]";
					}

					calltip += " ";
					calltip += shader->functions[i].name;
					calltip += "(";

					for (int j = 0; j < shader->functions[i].function->arguments.size(); j++) {
						if (j > 0) {
							calltip += ", ";
						} else {
							calltip += " ";
						}

						if (j == completion_argument) {
							calltip += char32_t(0xFFFF);
						}

						if (shader->functions[i].function->arguments[j].is_const) {
							calltip += "const ";
						}

						if (shader->functions[i].function->arguments[j].qualifier != ArgumentQualifier::ARGUMENT_QUALIFIER_IN) {
							if (shader->functions[i].function->arguments[j].qualifier == ArgumentQualifier::ARGUMENT_QUALIFIER_OUT) {
								calltip += "out ";
							} else { // ArgumentQualifier::ARGUMENT_QUALIFIER_INOUT
								calltip += "inout ";
							}
						}

						calltip += get_datatype_name(shader->functions[i].function->arguments[j].type);
						calltip += " ";
						calltip += shader->functions[i].function->arguments[j].name;

						if (shader->functions[i].function->arguments[j].array_size > 0) {
							calltip += "[";
							calltip += itos(shader->functions[i].function->arguments[j].array_size);
							calltip += "]";
						}

						if (j == completion_argument) {
							calltip += char32_t(0xFFFF);
						}
					}

					if (shader->functions[i].function->arguments.size()) {
						calltip += " ";
					}
					calltip += ")";

					r_call_hint = calltip;
					return OK;
				}
			}

			int idx = 0;

			String calltip;
			bool low_end = RenderingServer::get_singleton()->is_low_end();

			if (stages && stages->has(block_function)) {
				for (const KeyValue<StringName, StageFunctionInfo> &E : (*stages)[block_function].stage_functions) {
					if (completion_function == E.key) {
						calltip += get_datatype_name(E.value.return_type);
						calltip += " ";
						calltip += E.key;
						calltip += "(";

						for (int i = 0; i < E.value.arguments.size(); i++) {
							if (i > 0) {
								calltip += ", ";
							} else {
								calltip += " ";
							}

							if (i == completion_argument) {
								calltip += char32_t(0xFFFF);
							}

							calltip += get_datatype_name(E.value.arguments[i].type);
							calltip += " ";
							calltip += E.value.arguments[i].name;

							if (i == completion_argument) {
								calltip += char32_t(0xFFFF);
							}
						}

						if (E.value.arguments.size()) {
							calltip += " ";
						}
						calltip += ")";

						r_call_hint = calltip;
						return OK;
					}
				}
			}

			while (builtin_func_defs[idx].name) {
				if (low_end && builtin_func_defs[idx].high_end) {
					idx++;
					continue;
				}

				int idx2 = 0;
				Set<int> out_args;
				while (builtin_func_out_args[idx2].name != nullptr) {
					if (builtin_func_out_args[idx2].name == builtin_func_defs[idx].name) {
						for (int i = 0; i < BuiltinFuncOutArgs::MAX_ARGS; i++) {
							int arg = builtin_func_out_args[idx2].arguments[i];
							if (arg == -1) {
								break;
							}
							out_args.insert(arg);
						}
						break;
					}
					idx2++;
				}

				if (completion_function == builtin_func_defs[idx].name) {
					if (builtin_func_defs[idx].tag != completion_class) {
						idx++;
						continue;
					}

					if (calltip.length()) {
						calltip += "\n";
					}

					calltip += get_datatype_name(builtin_func_defs[idx].rettype);
					calltip += " ";
					calltip += builtin_func_defs[idx].name;
					calltip += "(";

					bool found_arg = false;
					for (int i = 0; i < BuiltinFuncDef::MAX_ARGS - 1; i++) {
						if (builtin_func_defs[idx].args[i] == TYPE_VOID) {
							break;
						}

						if (i > 0) {
							calltip += ", ";
						} else {
							calltip += " ";
						}

						if (i == completion_argument) {
							calltip += char32_t(0xFFFF);
						}

						if (out_args.has(i)) {
							calltip += "out ";
						}

						calltip += get_datatype_name(builtin_func_defs[idx].args[i]);

						String arg_name = (String)builtin_func_defs[idx].args_names[i];
						if (!arg_name.is_empty()) {
							calltip += " ";
							calltip += arg_name;
						}

						if (i == completion_argument) {
							calltip += char32_t(0xFFFF);
						}

						found_arg = true;
					}

					if (found_arg) {
						calltip += " ";
					}
					calltip += ")";
				}
				idx++;
			}

			r_call_hint = calltip;

			return OK;

		} break;
		case COMPLETION_INDEX: {
			const char colv[4] = { 'r', 'g', 'b', 'a' };
			const char coordv[4] = { 'x', 'y', 'z', 'w' };
			const char coordt[4] = { 's', 't', 'p', 'q' };

			int limit = 0;

			switch (completion_base) {
				case TYPE_BVEC2:
				case TYPE_IVEC2:
				case TYPE_UVEC2:
				case TYPE_VEC2: {
					limit = 2;

				} break;
				case TYPE_BVEC3:
				case TYPE_IVEC3:
				case TYPE_UVEC3:
				case TYPE_VEC3: {
					limit = 3;

				} break;
				case TYPE_BVEC4:
				case TYPE_IVEC4:
				case TYPE_UVEC4:
				case TYPE_VEC4: {
					limit = 4;

				} break;
				case TYPE_MAT2:
					limit = 2;
					break;
				case TYPE_MAT3:
					limit = 3;
					break;
				case TYPE_MAT4:
					limit = 4;
					break;
				default: {
				}
			}

			for (int i = 0; i < limit; i++) {
				r_options->push_back(ScriptCodeCompletionOption(String::chr(colv[i]), ScriptCodeCompletionOption::KIND_PLAIN_TEXT));
				r_options->push_back(ScriptCodeCompletionOption(String::chr(coordv[i]), ScriptCodeCompletionOption::KIND_PLAIN_TEXT));
				r_options->push_back(ScriptCodeCompletionOption(String::chr(coordt[i]), ScriptCodeCompletionOption::KIND_PLAIN_TEXT));
			}

		} break;
	}

	return ERR_PARSE_ERROR;
}

String ShaderLanguage::get_error_text() {
	return error_str;
}

int ShaderLanguage::get_error_line() {
	return error_line;
}

ShaderLanguage::ShaderNode *ShaderLanguage::get_shader() {
	return shader;
}

ShaderLanguage::ShaderLanguage() {
	nodes = nullptr;
	completion_class = TAG_GLOBAL;

#if DEBUG_ENABLED
	warnings_check_map.insert(ShaderWarning::UNUSED_CONSTANT, &used_constants);
	warnings_check_map.insert(ShaderWarning::UNUSED_FUNCTION, &used_functions);
	warnings_check_map.insert(ShaderWarning::UNUSED_STRUCT, &used_structs);
	warnings_check_map.insert(ShaderWarning::UNUSED_UNIFORM, &used_uniforms);
	warnings_check_map.insert(ShaderWarning::UNUSED_VARYING, &used_varyings);

	warnings_check_map2.insert(ShaderWarning::UNUSED_LOCAL_VARIABLE, &used_local_vars);
#endif // DEBUG_ENABLED
}

ShaderLanguage::~ShaderLanguage() {
	clear();
}
