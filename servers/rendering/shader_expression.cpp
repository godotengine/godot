/**************************************************************************/
/*  shader_expression.cpp                                                 */
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

#include "shader_expression.h"

Error ShaderExpression::_get_token(Token &r_token) {
	while (true) {
#define GET_CHAR() (str_ofs >= expression.length() ? 0 : expression[str_ofs++])

		char32_t cchar = GET_CHAR();
		bool invalid_character = false;

		switch (cchar) {
			case 0: {
				r_token.type = TK_EOF;
				return OK;
			}
			case '{': {
				invalid_character = true;
				break;
			}
			case '}': {
				invalid_character = true;
				break;
			}
			case '[': {
				invalid_character = true;
				break;
			}
			case ']': {
				invalid_character = true;
				break;
			}
			case '(': {
				r_token.type = TK_PARENTHESIS_OPEN;
				return OK;
			}
			case ')': {
				r_token.type = TK_PARENTHESIS_CLOSE;
				return OK;
			}
			case ',': {
				invalid_character = true;
				break;
			}
			case ':': {
				invalid_character = true;
				break;
			}
			case '$': {
				invalid_character = true;
				break;
			}
			case '=': {
				cchar = GET_CHAR();
				if (cchar == '=') {
					r_token.type = TK_OP_EQUAL;
				} else {
					_set_error("Expected '='.");
					r_token.type = TK_ERROR;
					return ERR_PARSE_ERROR;
				}
				return OK;
			}
			case '!': {
				if (expression[str_ofs] == '=') {
					r_token.type = TK_OP_NOT_EQUAL;
					str_ofs++;
				} else {
					r_token.type = TK_OP_NOT;
				}
				return OK;
			}
			case '>': {
				if (expression[str_ofs] == '=') {
					r_token.type = TK_OP_GREATER_EQUAL;
					str_ofs++;
				} else if (expression[str_ofs] == '>') {
					r_token.type = TK_OP_SHIFT_RIGHT;
					str_ofs++;
				} else {
					r_token.type = TK_OP_GREATER;
				}
				return OK;
			}
			case '<': {
				if (expression[str_ofs] == '=') {
					r_token.type = TK_OP_LESS_EQUAL;
					str_ofs++;
				} else if (expression[str_ofs] == '<') {
					r_token.type = TK_OP_SHIFT_LEFT;
					str_ofs++;
				} else {
					r_token.type = TK_OP_LESS;
				}
				return OK;
			}
			case '+': {
				r_token.type = TK_OP_ADD;
				return OK;
			}
			case '-': {
				r_token.type = TK_OP_SUB;
				return OK;
			}
			case '/': {
				r_token.type = TK_OP_DIV;
				return OK;
			}
			case '*': {
				r_token.type = TK_OP_MUL;
				return OK;
			}
			case '%': {
				r_token.type = TK_OP_MOD;
				return OK;
			}
			case '&': {
				if (expression[str_ofs] == '&') {
					r_token.type = TK_OP_AND;
					str_ofs++;
				} else {
					r_token.type = TK_OP_BIT_AND;
				}
				return OK;
			}
			case '|': {
				if (expression[str_ofs] == '|') {
					r_token.type = TK_OP_OR;
					str_ofs++;
				} else {
					r_token.type = TK_OP_BIT_OR;
				}
				return OK;
			}
			case '^': {
				invalid_character = true;
				break;
			}
			case '~': {
				invalid_character = true;
				break;
			}
			case '\'': {
				invalid_character = true;
				break;
			}
			case '"': {
				invalid_character = true;
				break;
			}
			default: {
				if (cchar <= 32) {
					break;
				}

				char32_t next_char = (str_ofs >= expression.length()) ? 0 : expression[str_ofs];
				if (is_digit(cchar)) {
					//a number

					String num;
#define READING_SIGN 0
#define READING_INT 1
#define READING_HEX 2
#define READING_BIN 3
#define READING_DEC 4
#define READING_EXP 5
#define READING_DONE 6
					int reading = READING_INT;

					char32_t c = cchar;
					bool exp_sign = false;
					bool exp_beg = false;
					bool bin_beg = false;
					bool hex_beg = false;
					bool is_float = false;
					bool is_first_char = true;

					while (true) {
						switch (reading) {
							case READING_INT: {
								if (is_digit(c)) {
									if (is_first_char && c == '0') {
										if (next_char == 'b' || next_char == 'B') {
											reading = READING_BIN;
										} else if (next_char == 'x' || next_char == 'X') {
											reading = READING_HEX;
										}
									}
								} else if (c == '.') {
									reading = READING_DEC;
									is_float = true;
								} else if (c == 'e' || c == 'E') {
									reading = READING_EXP;
									is_float = true;
								} else {
									reading = READING_DONE;
								}
								if (is_float) {
									_set_error("Floating-point number operations are not supported by the shader preprocessor.");
									r_token.type = TK_ERROR;
									return ERR_PARSE_ERROR;
								}
							} break;
							case READING_BIN: {
								if (bin_beg && !is_binary_digit(c)) {
									reading = READING_DONE;
								} else if (c == 'b' || c == 'B') {
									bin_beg = true;
								}

							} break;
							case READING_HEX: {
								if (hex_beg && !is_hex_digit(c)) {
									reading = READING_DONE;
								} else if (c == 'x' || c == 'X') {
									hex_beg = true;
								}

							} break;
							case READING_DEC: {
								if (is_digit(c)) {
								} else if (c == 'e' || c == 'E') {
									reading = READING_EXP;
								} else {
									reading = READING_DONE;
								}

							} break;
							case READING_EXP: {
								if (is_digit(c)) {
									exp_beg = true;

								} else if ((c == '-' || c == '+') && !exp_sign && !exp_beg) {
									exp_sign = true;

								} else {
									reading = READING_DONE;
								}
							} break;
						}

						if (reading == READING_DONE) {
							break;
						}
						num += c;
						c = GET_CHAR();
						is_first_char = false;
					}

					if (c != 0) {
						str_ofs--;
					}

					r_token.type = TK_CONSTANT;

					if (is_float) {
						r_token.value = num.to_float();
					} else if (bin_beg) {
						r_token.value = num.bin_to_int();
					} else if (hex_beg) {
						r_token.value = num.hex_to_int();
					} else {
						r_token.value = num.to_int();
					}
					return OK;

				} else if (is_unicode_identifier_start(cchar)) {
					String id = String::chr(cchar);
					cchar = GET_CHAR();

					while (is_unicode_identifier_continue(cchar)) {
						id += cchar;
						cchar = GET_CHAR();
					}

					str_ofs--; //go back one

					if (id == "true") {
						r_token.type = TK_CONSTANT;
						r_token.value = true;
					} else if (id == "false") {
						r_token.type = TK_CONSTANT;
						r_token.value = false;
					} else {
						r_token.type = TK_IDENTIFIER;
						r_token.value = id;
					}

					return OK;
				} else {
					invalid_character = true;
				}
			}
		}
		if (invalid_character) {
			_set_error("Unexpected character.");
			r_token.type = TK_ERROR;
			return ERR_PARSE_ERROR;
		}
#undef GET_CHAR
	}

	r_token.type = TK_ERROR;
	return ERR_PARSE_ERROR;
}
