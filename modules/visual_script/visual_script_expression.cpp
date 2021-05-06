/*************************************************************************/
/*  visual_script_expression.cpp                                         */
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

#include "visual_script_expression.h"

bool VisualScriptExpression::_set(const StringName &p_name, const Variant &p_value) {
	if (String(p_name) == "expression") {
		expression = p_value;
		expression_dirty = true;
		ports_changed_notify();
		return true;
	}

	if (String(p_name) == "out_type") {
		output_type = Variant::Type(int(p_value));
		expression_dirty = true;
		ports_changed_notify();
		return true;
	}
	if (String(p_name) == "sequenced") {
		sequenced = p_value;
		ports_changed_notify();
		return true;
	}

	if (String(p_name) == "input_count") {
		int from = inputs.size();
		inputs.resize(int(p_value));
		for (int i = from; i < inputs.size(); i++) {
			inputs.write[i].name = String::chr('a' + i);
			if (from == 0) {
				inputs.write[i].type = output_type;
			} else {
				inputs.write[i].type = inputs[from - 1].type;
			}
		}
		expression_dirty = true;
		ports_changed_notify();
		notify_property_list_changed();
		return true;
	}

	if (String(p_name).begins_with("input_")) {
		int idx = String(p_name).get_slicec('_', 1).get_slicec('/', 0).to_int();
		ERR_FAIL_INDEX_V(idx, inputs.size(), false);

		String what = String(p_name).get_slice("/", 1);

		if (what == "type") {
			inputs.write[idx].type = Variant::Type(int(p_value));
		} else if (what == "name") {
			inputs.write[idx].name = p_value;
		} else {
			return false;
		}

		expression_dirty = true;
		ports_changed_notify();
		return true;
	}

	return false;
}

bool VisualScriptExpression::_get(const StringName &p_name, Variant &r_ret) const {
	if (String(p_name) == "expression") {
		r_ret = expression;
		return true;
	}

	if (String(p_name) == "out_type") {
		r_ret = output_type;
		return true;
	}

	if (String(p_name) == "sequenced") {
		r_ret = sequenced;
		return true;
	}

	if (String(p_name) == "input_count") {
		r_ret = inputs.size();
		return true;
	}

	if (String(p_name).begins_with("input_")) {
		int idx = String(p_name).get_slicec('_', 1).get_slicec('/', 0).to_int();
		ERR_FAIL_INDEX_V(idx, inputs.size(), false);

		String what = String(p_name).get_slice("/", 1);

		if (what == "type") {
			r_ret = inputs[idx].type;
		} else if (what == "name") {
			r_ret = inputs[idx].name;
		} else {
			return false;
		}

		return true;
	}

	return false;
}

void VisualScriptExpression::_get_property_list(List<PropertyInfo> *p_list) const {
	String argt = "Any";
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		argt += "," + Variant::get_type_name(Variant::Type(i));
	}

	p_list->push_back(PropertyInfo(Variant::STRING, "expression", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	p_list->push_back(PropertyInfo(Variant::INT, "out_type", PROPERTY_HINT_ENUM, argt));
	p_list->push_back(PropertyInfo(Variant::INT, "input_count", PROPERTY_HINT_RANGE, "0,64,1"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "sequenced"));

	for (int i = 0; i < inputs.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::INT, "input_" + itos(i) + "/type", PROPERTY_HINT_ENUM, argt));
		p_list->push_back(PropertyInfo(Variant::STRING, "input_" + itos(i) + "/name"));
	}
}

int VisualScriptExpression::get_output_sequence_port_count() const {
	return sequenced ? 1 : 0;
}

bool VisualScriptExpression::has_input_sequence_port() const {
	return sequenced;
}

String VisualScriptExpression::get_output_sequence_port_text(int p_port) const {
	return String();
}

int VisualScriptExpression::get_input_value_port_count() const {
	return inputs.size();
}

int VisualScriptExpression::get_output_value_port_count() const {
	return 1;
}

PropertyInfo VisualScriptExpression::get_input_value_port_info(int p_idx) const {
	return PropertyInfo(inputs[p_idx].type, inputs[p_idx].name);
}

PropertyInfo VisualScriptExpression::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(output_type, "result");
}

String VisualScriptExpression::get_caption() const {
	return "Expression";
}

String VisualScriptExpression::get_text() const {
	return expression;
}

Error VisualScriptExpression::_get_token(Token &r_token) {
	while (true) {
#define GET_CHAR() (str_ofs >= expression.length() ? 0 : expression[str_ofs++])

		char32_t cchar = GET_CHAR();
		if (cchar == 0) {
			r_token.type = TK_EOF;
			return OK;
		}

		switch (cchar) {
			case 0: {
				r_token.type = TK_EOF;
				return OK;
			} break;
			case '{': {
				r_token.type = TK_CURLY_BRACKET_OPEN;
				return OK;
			};
			case '}': {
				r_token.type = TK_CURLY_BRACKET_CLOSE;
				return OK;
			};
			case '[': {
				r_token.type = TK_BRACKET_OPEN;
				return OK;
			};
			case ']': {
				r_token.type = TK_BRACKET_CLOSE;
				return OK;
			};
			case '(': {
				r_token.type = TK_PARENTHESIS_OPEN;
				return OK;
			};
			case ')': {
				r_token.type = TK_PARENTHESIS_CLOSE;
				return OK;
			};
			case ',': {
				r_token.type = TK_COMMA;
				return OK;
			};
			case ':': {
				r_token.type = TK_COLON;
				return OK;
			};
			case '.': {
				r_token.type = TK_PERIOD;
				return OK;
			};
			case '=': {
				cchar = GET_CHAR();
				if (cchar == '=') {
					r_token.type = TK_OP_EQUAL;
				} else {
					_set_error("Expected '='");
					r_token.type = TK_ERROR;
					return ERR_PARSE_ERROR;
				}
				return OK;
			};
			case '!': {
				if (expression[str_ofs] == '=') {
					r_token.type = TK_OP_NOT_EQUAL;
					str_ofs++;
				} else {
					r_token.type = TK_OP_NOT;
				}
				return OK;
			};
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
			};
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
			};
			case '+': {
				r_token.type = TK_OP_ADD;
				return OK;
			};
			case '-': {
				r_token.type = TK_OP_SUB;
				return OK;
			};
			case '/': {
				r_token.type = TK_OP_DIV;
				return OK;
			};
			case '*': {
				r_token.type = TK_OP_MUL;
				return OK;
			};
			case '%': {
				r_token.type = TK_OP_MOD;
				return OK;
			};
			case '&': {
				if (expression[str_ofs] == '&') {
					r_token.type = TK_OP_AND;
					str_ofs++;
				} else {
					r_token.type = TK_OP_BIT_AND;
				}
				return OK;
			};
			case '|': {
				if (expression[str_ofs] == '|') {
					r_token.type = TK_OP_OR;
					str_ofs++;
				} else {
					r_token.type = TK_OP_BIT_OR;
				}
				return OK;
			};
			case '^': {
				r_token.type = TK_OP_BIT_XOR;

				return OK;
			};
			case '~': {
				r_token.type = TK_OP_BIT_INVERT;

				return OK;
			};
			case '"': {
				String str;
				while (true) {
					char32_t ch = GET_CHAR();

					if (ch == 0) {
						_set_error("Unterminated String");
						r_token.type = TK_ERROR;
						return ERR_PARSE_ERROR;
					} else if (ch == '"') {
						break;
					} else if (ch == '\\') {
						//escaped characters...

						char32_t next = GET_CHAR();
						if (next == 0) {
							_set_error("Unterminated String");
							r_token.type = TK_ERROR;
							return ERR_PARSE_ERROR;
						}
						char32_t res = 0;

						switch (next) {
							case 'b':
								res = 8;
								break;
							case 't':
								res = 9;
								break;
							case 'n':
								res = 10;
								break;
							case 'f':
								res = 12;
								break;
							case 'r':
								res = 13;
								break;
							case 'u': {
								// hex number
								for (int j = 0; j < 4; j++) {
									char32_t c = GET_CHAR();

									if (c == 0) {
										_set_error("Unterminated String");
										r_token.type = TK_ERROR;
										return ERR_PARSE_ERROR;
									}
									if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
										_set_error("Malformed hex constant in string");
										r_token.type = TK_ERROR;
										return ERR_PARSE_ERROR;
									}
									char32_t v;
									if (c >= '0' && c <= '9') {
										v = c - '0';
									} else if (c >= 'a' && c <= 'f') {
										v = c - 'a';
										v += 10;
									} else if (c >= 'A' && c <= 'F') {
										v = c - 'A';
										v += 10;
									} else {
										ERR_PRINT("Bug parsing hex constant.");
										v = 0;
									}

									res <<= 4;
									res |= v;
								}

							} break;
							default: {
								res = next;
							} break;
						}

						str += res;

					} else {
						str += ch;
					}
				}

				r_token.type = TK_CONSTANT;
				r_token.value = str;
				return OK;

			} break;
			default: {
				if (cchar <= 32) {
					break;
				}

				if (cchar >= '0' && cchar <= '9') {
					//a number

					String num;
#define READING_SIGN 0
#define READING_INT 1
#define READING_DEC 2
#define READING_EXP 3
#define READING_DONE 4
					int reading = READING_INT;

					char32_t c = cchar;
					bool exp_sign = false;
					bool exp_beg = false;
					bool is_float = false;

					while (true) {
						switch (reading) {
							case READING_INT: {
								if (c >= '0' && c <= '9') {
									//pass
								} else if (c == '.') {
									reading = READING_DEC;
									is_float = true;
								} else if (c == 'e') {
									reading = READING_EXP;
								} else {
									reading = READING_DONE;
								}

							} break;
							case READING_DEC: {
								if (c >= '0' && c <= '9') {
								} else if (c == 'e') {
									reading = READING_EXP;

								} else {
									reading = READING_DONE;
								}

							} break;
							case READING_EXP: {
								if (c >= '0' && c <= '9') {
									exp_beg = true;

								} else if ((c == '-' || c == '+') && !exp_sign && !exp_beg) {
									if (c == '-') {
										is_float = true;
									}
									exp_sign = true;

								} else {
									reading = READING_DONE;
								}
							} break;
						}

						if (reading == READING_DONE) {
							break;
						}
						num += String::chr(c);
						c = GET_CHAR();
					}

					str_ofs--;

					r_token.type = TK_CONSTANT;

					if (is_float) {
						r_token.value = num.to_float();
					} else {
						r_token.value = num.to_int();
					}
					return OK;

				} else if ((cchar >= 'A' && cchar <= 'Z') || (cchar >= 'a' && cchar <= 'z') || cchar == '_') {
					String id;
					bool first = true;

					while ((cchar >= 'A' && cchar <= 'Z') || (cchar >= 'a' && cchar <= 'z') || cchar == '_' || (!first && cchar >= '0' && cchar <= '9')) {
						id += String::chr(cchar);
						cchar = GET_CHAR();
						first = false;
					}

					str_ofs--; //go back one

					if (id == "in") {
						r_token.type = TK_OP_IN;
					} else if (id == "null") {
						r_token.type = TK_CONSTANT;
						r_token.value = Variant();
					} else if (id == "true") {
						r_token.type = TK_CONSTANT;
						r_token.value = true;
					} else if (id == "false") {
						r_token.type = TK_CONSTANT;
						r_token.value = false;
					} else if (id == "PI") {
						r_token.type = TK_CONSTANT;
						r_token.value = Math_PI;
					} else if (id == "TAU") {
						r_token.type = TK_CONSTANT;
						r_token.value = Math_TAU;
					} else if (id == "INF") {
						r_token.type = TK_CONSTANT;
						r_token.value = Math_INF;
					} else if (id == "NAN") {
						r_token.type = TK_CONSTANT;
						r_token.value = Math_NAN;
					} else if (id == "not") {
						r_token.type = TK_OP_NOT;
					} else if (id == "or") {
						r_token.type = TK_OP_OR;
					} else if (id == "and") {
						r_token.type = TK_OP_AND;
					} else if (id == "self") {
						r_token.type = TK_SELF;
					} else {
						for (int i = 0; i < Variant::VARIANT_MAX; i++) {
							if (id == Variant::get_type_name(Variant::Type(i))) {
								r_token.type = TK_BASIC_TYPE;
								r_token.value = i;
								return OK;
							}
						}

						VisualScriptBuiltinFunc::BuiltinFunc bifunc = VisualScriptBuiltinFunc::find_function(id);
						if (bifunc != VisualScriptBuiltinFunc::FUNC_MAX) {
							r_token.type = TK_BUILTIN_FUNC;
							r_token.value = bifunc;
							return OK;
						}

						r_token.type = TK_IDENTIFIER;
						r_token.value = id;
					}

					return OK;
				} else {
					_set_error("Unexpected character.");
					r_token.type = TK_ERROR;
					return ERR_PARSE_ERROR;
				}
			}
		}
	}

	r_token.type = TK_ERROR;
	return ERR_PARSE_ERROR;
}

const char *VisualScriptExpression::token_name[TK_MAX] = {
	"CURLY BRACKET OPEN",
	"CURLY BRACKET CLOSE",
	"BRACKET OPEN",
	"BRACKET CLOSE",
	"PARENTHESIS OPEN",
	"PARENTHESIS CLOSE",
	"IDENTIFIER",
	"BUILTIN FUNC",
	"SELF",
	"CONSTANT",
	"BASIC TYPE",
	"COLON",
	"COMMA",
	"PERIOD",
	"OP IN",
	"OP EQUAL",
	"OP NOT EQUAL",
	"OP LESS",
	"OP LESS EQUAL",
	"OP GREATER",
	"OP GREATER EQUAL",
	"OP AND",
	"OP OR",
	"OP NOT",
	"OP ADD",
	"OP SUB",
	"OP MUL",
	"OP DIV",
	"OP MOD",
	"OP SHIFT LEFT",
	"OP SHIFT RIGHT",
	"OP BIT AND",
	"OP BIT OR",
	"OP BIT XOR",
	"OP BIT INVERT",
	"EOF",
	"ERROR"
};

VisualScriptExpression::ENode *VisualScriptExpression::_parse_expression() {
	Vector<Expression> expression;

	while (true) {
		//keep appending stuff to expression
		ENode *expr = nullptr;

		Token tk;
		_get_token(tk);
		if (error_set) {
			return nullptr;
		}

		switch (tk.type) {
			case TK_CURLY_BRACKET_OPEN: {
				//a dictionary
				DictionaryNode *dn = alloc_node<DictionaryNode>();

				while (true) {
					int cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_CURLY_BRACKET_CLOSE) {
						break;
					}
					str_ofs = cofs; //revert
					//parse an expression
					ENode *expr2 = _parse_expression();
					if (!expr2) {
						return nullptr;
					}
					dn->dict.push_back(expr2);

					_get_token(tk);
					if (tk.type != TK_COLON) {
						_set_error("Expected ':'");
						return nullptr;
					}

					expr2 = _parse_expression();
					if (!expr2) {
						return nullptr;
					}

					dn->dict.push_back(expr2);

					cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_COMMA) {
						//all good
					} else if (tk.type == TK_CURLY_BRACKET_CLOSE) {
						str_ofs = cofs;
					} else {
						_set_error("Expected ',' or '}'");
					}
				}

				expr = dn;
			} break;
			case TK_BRACKET_OPEN: {
				//an array

				ArrayNode *an = alloc_node<ArrayNode>();

				while (true) {
					int cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_BRACKET_CLOSE) {
						break;
					}
					str_ofs = cofs; //revert
					//parse an expression
					ENode *expr2 = _parse_expression();
					if (!expr2) {
						return nullptr;
					}
					an->array.push_back(expr2);

					cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_COMMA) {
						//all good
					} else if (tk.type == TK_BRACKET_CLOSE) {
						str_ofs = cofs;
					} else {
						_set_error("Expected ',' or ']'");
					}
				}

				expr = an;
			} break;
			case TK_PARENTHESIS_OPEN: {
				//a suexpression
				ENode *e = _parse_expression();
				if (error_set) {
					return nullptr;
				}
				_get_token(tk);
				if (tk.type != TK_PARENTHESIS_CLOSE) {
					_set_error("Expected ')'");
					return nullptr;
				}

				expr = e;

			} break;
			case TK_IDENTIFIER: {
				String what = tk.value;
				int index = -1;
				for (int i = 0; i < inputs.size(); i++) {
					if (what == inputs[i].name) {
						index = i;
						break;
					}
				}

				if (index != -1) {
					InputNode *input = alloc_node<InputNode>();
					input->index = index;
					expr = input;
				} else {
					_set_error("Invalid input identifier '" + what + "'. For script variables, use self (locals are for inputs)." + what);
					return nullptr;
				}
			} break;
			case TK_SELF: {
				SelfNode *self = alloc_node<SelfNode>();
				expr = self;
			} break;
			case TK_CONSTANT: {
				ConstantNode *constant = alloc_node<ConstantNode>();
				constant->value = tk.value;
				expr = constant;
			} break;
			case TK_BASIC_TYPE: {
				//constructor..

				Variant::Type bt = Variant::Type(int(tk.value));
				_get_token(tk);
				if (tk.type != TK_PARENTHESIS_OPEN) {
					_set_error("Expected '('");
					return nullptr;
				}

				ConstructorNode *constructor = alloc_node<ConstructorNode>();
				constructor->data_type = bt;

				while (true) {
					int cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_PARENTHESIS_CLOSE) {
						break;
					}
					str_ofs = cofs; //revert
					//parse an expression
					ENode *expr2 = _parse_expression();
					if (!expr2) {
						return nullptr;
					}

					constructor->arguments.push_back(expr2);

					cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_COMMA) {
						//all good
					} else if (tk.type == TK_PARENTHESIS_CLOSE) {
						str_ofs = cofs;
					} else {
						_set_error("Expected ',' or ')'");
					}
				}

				expr = constructor;

			} break;
			case TK_BUILTIN_FUNC: {
				//builtin function

				_get_token(tk);
				if (tk.type != TK_PARENTHESIS_OPEN) {
					_set_error("Expected '('");
					return nullptr;
				}

				BuiltinFuncNode *bifunc = alloc_node<BuiltinFuncNode>();
				bifunc->func = VisualScriptBuiltinFunc::BuiltinFunc(int(tk.value));

				while (true) {
					int cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_PARENTHESIS_CLOSE) {
						break;
					}
					str_ofs = cofs; //revert
					//parse an expression
					ENode *expr2 = _parse_expression();
					if (!expr2) {
						return nullptr;
					}

					bifunc->arguments.push_back(expr2);

					cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_COMMA) {
						//all good
					} else if (tk.type == TK_PARENTHESIS_CLOSE) {
						str_ofs = cofs;
					} else {
						_set_error("Expected ',' or ')'");
					}
				}

				int expected_args = VisualScriptBuiltinFunc::get_func_argument_count(bifunc->func);
				if (bifunc->arguments.size() != expected_args) {
					_set_error("Builtin func '" + VisualScriptBuiltinFunc::get_func_name(bifunc->func) + "' expects " + itos(expected_args) + " arguments.");
				}

				expr = bifunc;

			} break;
			case TK_OP_SUB: {
				Expression e;
				e.is_op = true;
				e.op = Variant::OP_NEGATE;
				expression.push_back(e);
				continue;
			} break;
			case TK_OP_NOT: {
				Expression e;
				e.is_op = true;
				e.op = Variant::OP_NOT;
				expression.push_back(e);
				continue;
			} break;

			default: {
				_set_error("Expected expression.");
				return nullptr;
			} break;
		}

		//before going to operators, must check indexing!

		while (true) {
			int cofs2 = str_ofs;
			_get_token(tk);
			if (error_set) {
				return nullptr;
			}

			bool done = false;

			switch (tk.type) {
				case TK_BRACKET_OPEN: {
					//value indexing

					IndexNode *index = alloc_node<IndexNode>();
					index->base = expr;

					ENode *what = _parse_expression();
					if (!what) {
						return nullptr;
					}

					index->index = what;

					_get_token(tk);
					if (tk.type != TK_BRACKET_CLOSE) {
						_set_error("Expected ']' at end of index.");
						return nullptr;
					}
					expr = index;

				} break;
				case TK_PERIOD: {
					//named indexing or function call
					_get_token(tk);
					if (tk.type != TK_IDENTIFIER) {
						_set_error("Expected identifier after '.'");
						return nullptr;
					}

					StringName identifier = tk.value;

					int cofs = str_ofs;
					_get_token(tk);
					if (tk.type == TK_PARENTHESIS_OPEN) {
						//function call
						CallNode *func_call = alloc_node<CallNode>();
						func_call->method = identifier;
						func_call->base = expr;

						while (true) {
							int cofs3 = str_ofs;
							_get_token(tk);
							if (tk.type == TK_PARENTHESIS_CLOSE) {
								break;
							}
							str_ofs = cofs3; //revert
							//parse an expression
							ENode *expr2 = _parse_expression();
							if (!expr2) {
								return nullptr;
							}

							func_call->arguments.push_back(expr2);

							cofs3 = str_ofs;
							_get_token(tk);
							if (tk.type == TK_COMMA) {
								//all good
							} else if (tk.type == TK_PARENTHESIS_CLOSE) {
								str_ofs = cofs3;
							} else {
								_set_error("Expected ',' or ')'");
							}
						}

						expr = func_call;
					} else {
						//named indexing
						str_ofs = cofs;

						NamedIndexNode *index = alloc_node<NamedIndexNode>();
						index->base = expr;
						index->name = identifier;
						expr = index;
					}

				} break;
				default: {
					str_ofs = cofs2;
					done = true;
				} break;
			}

			if (done) {
				break;
			}
		}

		//push expression
		{
			Expression e;
			e.is_op = false;
			e.node = expr;
			expression.push_back(e);
		}

		//ok finally look for an operator

		int cofs = str_ofs;
		_get_token(tk);
		if (error_set) {
			return nullptr;
		}

		Variant::Operator op = Variant::OP_MAX;

		switch (tk.type) {
			case TK_OP_IN:
				op = Variant::OP_IN;
				break;
			case TK_OP_EQUAL:
				op = Variant::OP_EQUAL;
				break;
			case TK_OP_NOT_EQUAL:
				op = Variant::OP_NOT_EQUAL;
				break;
			case TK_OP_LESS:
				op = Variant::OP_LESS;
				break;
			case TK_OP_LESS_EQUAL:
				op = Variant::OP_LESS_EQUAL;
				break;
			case TK_OP_GREATER:
				op = Variant::OP_GREATER;
				break;
			case TK_OP_GREATER_EQUAL:
				op = Variant::OP_GREATER_EQUAL;
				break;
			case TK_OP_AND:
				op = Variant::OP_AND;
				break;
			case TK_OP_OR:
				op = Variant::OP_OR;
				break;
			case TK_OP_NOT:
				op = Variant::OP_NOT;
				break;
			case TK_OP_ADD:
				op = Variant::OP_ADD;
				break;
			case TK_OP_SUB:
				op = Variant::OP_SUBTRACT;
				break;
			case TK_OP_MUL:
				op = Variant::OP_MULTIPLY;
				break;
			case TK_OP_DIV:
				op = Variant::OP_DIVIDE;
				break;
			case TK_OP_MOD:
				op = Variant::OP_MODULE;
				break;
			case TK_OP_SHIFT_LEFT:
				op = Variant::OP_SHIFT_LEFT;
				break;
			case TK_OP_SHIFT_RIGHT:
				op = Variant::OP_SHIFT_RIGHT;
				break;
			case TK_OP_BIT_AND:
				op = Variant::OP_BIT_AND;
				break;
			case TK_OP_BIT_OR:
				op = Variant::OP_BIT_OR;
				break;
			case TK_OP_BIT_XOR:
				op = Variant::OP_BIT_XOR;
				break;
			case TK_OP_BIT_INVERT:
				op = Variant::OP_BIT_NEGATE;
				break;
			default: {
			};
		}

		if (op == Variant::OP_MAX) { //stop appending stuff
			str_ofs = cofs;
			break;
		}

		//push operator and go on
		{
			Expression e;
			e.is_op = true;
			e.op = op;
			expression.push_back(e);
		}
	}

	/* Reduce the set of expressions and place them in an operator tree, respecting precedence */

	while (expression.size() > 1) {
		int next_op = -1;
		int min_priority = 0xFFFFF;
		bool is_unary = false;

		for (int i = 0; i < expression.size(); i++) {
			if (!expression[i].is_op) {
				continue;
			}

			int priority;

			bool unary = false;

			switch (expression[i].op) {
				case Variant::OP_BIT_NEGATE:
					priority = 0;
					unary = true;
					break;
				case Variant::OP_NEGATE:
					priority = 1;
					unary = true;
					break;

				case Variant::OP_MULTIPLY:
					priority = 2;
					break;
				case Variant::OP_DIVIDE:
					priority = 2;
					break;
				case Variant::OP_MODULE:
					priority = 2;
					break;

				case Variant::OP_ADD:
					priority = 3;
					break;
				case Variant::OP_SUBTRACT:
					priority = 3;
					break;

				case Variant::OP_SHIFT_LEFT:
					priority = 4;
					break;
				case Variant::OP_SHIFT_RIGHT:
					priority = 4;
					break;

				case Variant::OP_BIT_AND:
					priority = 5;
					break;
				case Variant::OP_BIT_XOR:
					priority = 6;
					break;
				case Variant::OP_BIT_OR:
					priority = 7;
					break;

				case Variant::OP_LESS:
					priority = 8;
					break;
				case Variant::OP_LESS_EQUAL:
					priority = 8;
					break;
				case Variant::OP_GREATER:
					priority = 8;
					break;
				case Variant::OP_GREATER_EQUAL:
					priority = 8;
					break;

				case Variant::OP_EQUAL:
					priority = 8;
					break;
				case Variant::OP_NOT_EQUAL:
					priority = 8;
					break;

				case Variant::OP_IN:
					priority = 10;
					break;

				case Variant::OP_NOT:
					priority = 11;
					unary = true;
					break;
				case Variant::OP_AND:
					priority = 12;
					break;
				case Variant::OP_OR:
					priority = 13;
					break;

				default: {
					_set_error("Parser bug, invalid operator in expression: " + itos(expression[i].op));
					return nullptr;
				}
			}

			if (priority < min_priority) {
				// < is used for left to right (default)
				// <= is used for right to left

				next_op = i;
				min_priority = priority;
				is_unary = unary;
			}
		}

		if (next_op == -1) {
			_set_error("Yet another parser bug....");
			ERR_FAIL_V(nullptr);
		}

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
				op->nodes[0] = expression[i + 1].node;
				op->nodes[1] = nullptr;
				expression.write[i].is_op = false;
				expression.write[i].node = op;
				expression.remove(i + 1);
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

			if (expression[next_op + 1].is_op) {
				// this is not invalid and can really appear
				// but it becomes invalid anyway because no binary op
				// can be followed by a unary op in a valid combination,
				// due to how precedence works, unaries will always disappear first

				_set_error("Unexpected two consecutive operators.");
				return nullptr;
			}

			op->nodes[0] = expression[next_op - 1].node; //expression goes as left
			op->nodes[1] = expression[next_op + 1].node; //next expression goes as right

			//replace all 3 nodes by this operator and make it an expression
			expression.write[next_op - 1].node = op;
			expression.remove(next_op);
			expression.remove(next_op);
		}
	}

	return expression[0].node;
}

bool VisualScriptExpression::_compile_expression() {
	if (!expression_dirty) {
		return error_set;
	}

	if (nodes) {
		memdelete(nodes);
		nodes = nullptr;
		root = nullptr;
	}

	error_str = String();
	error_set = false;
	str_ofs = 0;

	root = _parse_expression();

	if (error_set) {
		root = nullptr;
		if (nodes) {
			memdelete(nodes);
		}
		nodes = nullptr;
		return true;
	}

	expression_dirty = false;
	return false;
}

class VisualScriptNodeInstanceExpression : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;
	VisualScriptExpression *expression;

	//virtual int get_working_memory_size() const { return 0; }
	//execute by parsing the tree directly
	virtual bool _execute(const Variant **p_inputs, VisualScriptExpression::ENode *p_node, Variant &r_ret, String &r_error_str, Callable::CallError &ce) {
		switch (p_node->type) {
			case VisualScriptExpression::ENode::TYPE_INPUT: {
				const VisualScriptExpression::InputNode *in = static_cast<const VisualScriptExpression::InputNode *>(p_node);
				r_ret = *p_inputs[in->index];
			} break;
			case VisualScriptExpression::ENode::TYPE_CONSTANT: {
				const VisualScriptExpression::ConstantNode *c = static_cast<const VisualScriptExpression::ConstantNode *>(p_node);
				r_ret = c->value;

			} break;
			case VisualScriptExpression::ENode::TYPE_SELF: {
				r_ret = instance->get_owner_ptr();
			} break;
			case VisualScriptExpression::ENode::TYPE_OPERATOR: {
				const VisualScriptExpression::OperatorNode *op = static_cast<const VisualScriptExpression::OperatorNode *>(p_node);

				Variant a;
				bool ret = _execute(p_inputs, op->nodes[0], a, r_error_str, ce);
				if (ret) {
					return true;
				}

				Variant b;

				if (op->nodes[1]) {
					ret = _execute(p_inputs, op->nodes[1], b, r_error_str, ce);
					if (ret) {
						return true;
					}
				}

				bool valid = true;
				Variant::evaluate(op->op, a, b, r_ret, valid);
				if (!valid) {
					r_error_str = "Invalid operands to operator " + Variant::get_operator_name(op->op) + ": " + Variant::get_type_name(a.get_type()) + " and " + Variant::get_type_name(b.get_type()) + ".";
					return true;
				}

			} break;
			case VisualScriptExpression::ENode::TYPE_INDEX: {
				const VisualScriptExpression::IndexNode *index = static_cast<const VisualScriptExpression::IndexNode *>(p_node);

				Variant base;
				bool ret = _execute(p_inputs, index->base, base, r_error_str, ce);
				if (ret) {
					return true;
				}

				Variant idx;

				ret = _execute(p_inputs, index->index, idx, r_error_str, ce);
				if (ret) {
					return true;
				}

				bool valid;
				r_ret = base.get(idx, &valid);
				if (!valid) {
					r_error_str = "Invalid index of type " + Variant::get_type_name(idx.get_type()) + " for base of type " + Variant::get_type_name(base.get_type()) + ".";
					return true;
				}

			} break;
			case VisualScriptExpression::ENode::TYPE_NAMED_INDEX: {
				const VisualScriptExpression::NamedIndexNode *index = static_cast<const VisualScriptExpression::NamedIndexNode *>(p_node);

				Variant base;
				bool ret = _execute(p_inputs, index->base, base, r_error_str, ce);
				if (ret) {
					return true;
				}

				bool valid;
				r_ret = base.get_named(index->name, valid);
				if (!valid) {
					r_error_str = "Invalid index '" + String(index->name) + "' for base of type " + Variant::get_type_name(base.get_type()) + ".";
					return true;
				}

			} break;
			case VisualScriptExpression::ENode::TYPE_ARRAY: {
				const VisualScriptExpression::ArrayNode *array = static_cast<const VisualScriptExpression::ArrayNode *>(p_node);

				Array arr;
				arr.resize(array->array.size());
				for (int i = 0; i < array->array.size(); i++) {
					Variant value;
					bool ret = _execute(p_inputs, array->array[i], value, r_error_str, ce);
					if (ret) {
						return true;
					}
					arr[i] = value;
				}

				r_ret = arr;

			} break;
			case VisualScriptExpression::ENode::TYPE_DICTIONARY: {
				const VisualScriptExpression::DictionaryNode *dictionary = static_cast<const VisualScriptExpression::DictionaryNode *>(p_node);

				Dictionary d;
				for (int i = 0; i < dictionary->dict.size(); i += 2) {
					Variant key;
					bool ret = _execute(p_inputs, dictionary->dict[i + 0], key, r_error_str, ce);
					if (ret) {
						return true;
					}

					Variant value;
					ret = _execute(p_inputs, dictionary->dict[i + 1], value, r_error_str, ce);
					if (ret) {
						return true;
					}

					d[key] = value;
				}

				r_ret = d;
			} break;
			case VisualScriptExpression::ENode::TYPE_CONSTRUCTOR: {
				const VisualScriptExpression::ConstructorNode *constructor = static_cast<const VisualScriptExpression::ConstructorNode *>(p_node);

				Vector<Variant> arr;
				Vector<const Variant *> argp;
				arr.resize(constructor->arguments.size());
				argp.resize(constructor->arguments.size());

				for (int i = 0; i < constructor->arguments.size(); i++) {
					Variant value;
					bool ret = _execute(p_inputs, constructor->arguments[i], value, r_error_str, ce);
					if (ret) {
						return true;
					}
					arr.write[i] = value;
					argp.write[i] = &arr[i];
				}

				Variant::construct(constructor->data_type, r_ret, (const Variant **)argp.ptr(), argp.size(), ce);

				if (ce.error != Callable::CallError::CALL_OK) {
					r_error_str = "Invalid arguments to construct '" + Variant::get_type_name(constructor->data_type) + "'.";
					return true;
				}

			} break;
			case VisualScriptExpression::ENode::TYPE_BUILTIN_FUNC: {
				const VisualScriptExpression::BuiltinFuncNode *bifunc = static_cast<const VisualScriptExpression::BuiltinFuncNode *>(p_node);

				Vector<Variant> arr;
				Vector<const Variant *> argp;
				arr.resize(bifunc->arguments.size());
				argp.resize(bifunc->arguments.size());

				for (int i = 0; i < bifunc->arguments.size(); i++) {
					Variant value;
					bool ret = _execute(p_inputs, bifunc->arguments[i], value, r_error_str, ce);
					if (ret) {
						return true;
					}
					arr.write[i] = value;
					argp.write[i] = &arr[i];
				}

				VisualScriptBuiltinFunc::exec_func(bifunc->func, (const Variant **)argp.ptr(), &r_ret, ce, r_error_str);

				if (ce.error != Callable::CallError::CALL_OK) {
					r_error_str = "Builtin Call Failed. " + r_error_str;
					return true;
				}

			} break;
			case VisualScriptExpression::ENode::TYPE_CALL: {
				const VisualScriptExpression::CallNode *call = static_cast<const VisualScriptExpression::CallNode *>(p_node);

				Variant base;
				bool ret = _execute(p_inputs, call->base, base, r_error_str, ce);
				if (ret) {
					return true;
				}

				Vector<Variant> arr;
				Vector<const Variant *> argp;
				arr.resize(call->arguments.size());
				argp.resize(call->arguments.size());

				for (int i = 0; i < call->arguments.size(); i++) {
					Variant value;
					bool ret2 = _execute(p_inputs, call->arguments[i], value, r_error_str, ce);
					if (ret2) {
						return true;
					}
					arr.write[i] = value;
					argp.write[i] = &arr[i];
				}

				base.call(call->method, (const Variant **)argp.ptr(), argp.size(), r_ret, ce);

				if (ce.error != Callable::CallError::CALL_OK) {
					r_error_str = "On call to '" + String(call->method) + "':";
					return true;
				}

			} break;
		}
		return false;
	}

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		if (!expression->root || expression->error_set) {
			r_error_str = expression->error_str;
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
			return 0;
		}

		bool error = _execute(p_inputs, expression->root, *p_outputs[0], r_error_str, r_error);
		if (error && r_error.error == Callable::CallError::CALL_OK) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		}

#ifdef DEBUG_ENABLED
		if (!error && expression->output_type != Variant::NIL && !Variant::can_convert_strict(p_outputs[0]->get_type(), expression->output_type)) {
			r_error_str += "Can't convert expression result from " + Variant::get_type_name(p_outputs[0]->get_type()) + " to " + Variant::get_type_name(expression->output_type) + ".";
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		}
#endif

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptExpression::instance(VisualScriptInstance *p_instance) {
	_compile_expression();
	VisualScriptNodeInstanceExpression *instance = memnew(VisualScriptNodeInstanceExpression);
	instance->instance = p_instance;
	instance->expression = this;
	return instance;
}

void VisualScriptExpression::reset_state() {
	if (nodes) {
		memdelete(nodes);
		nodes = nullptr;
		root = nullptr;
	}

	error_str = String();
	error_set = false;
	str_ofs = 0;
	inputs.clear();
}

VisualScriptExpression::VisualScriptExpression() {
}

VisualScriptExpression::~VisualScriptExpression() {
	if (nodes) {
		memdelete(nodes);
	}
}

void register_visual_script_expression_node() {
	VisualScriptLanguage::singleton->add_register_func("operators/expression", create_node_generic<VisualScriptExpression>);
}
