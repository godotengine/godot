/*************************************************************************/
/*  json.cpp                                                             */
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

#include "json.h"

#include "core/string/print_string.h"

const char *JSON::tk_name[TK_MAX] = {
	"'{'",
	"'}'",
	"'['",
	"']'",
	"identifier",
	"string",
	"number",
	"':'",
	"','",
	"EOF",
};

static String _make_indent(const String &p_indent, int p_size) {
	String indent_text = "";
	if (!p_indent.is_empty()) {
		for (int i = 0; i < p_size; i++) {
			indent_text += p_indent;
		}
	}
	return indent_text;
}

String JSON::_print_var(const Variant &p_var, const String &p_indent, int p_cur_indent, bool p_sort_keys) {
	String colon = ":";
	String end_statement = "";

	if (!p_indent.is_empty()) {
		colon += " ";
		end_statement += "\n";
	}

	switch (p_var.get_type()) {
		case Variant::NIL:
			return "null";
		case Variant::BOOL:
			return p_var.operator bool() ? "true" : "false";
		case Variant::INT:
			return itos(p_var);
		case Variant::FLOAT:
			return rtos(p_var);
		case Variant::PACKED_INT32_ARRAY:
		case Variant::PACKED_INT64_ARRAY:
		case Variant::PACKED_FLOAT32_ARRAY:
		case Variant::PACKED_FLOAT64_ARRAY:
		case Variant::PACKED_STRING_ARRAY:
		case Variant::ARRAY: {
			String s = "[";
			s += end_statement;
			Array a = p_var;
			for (int i = 0; i < a.size(); i++) {
				if (i > 0) {
					s += ",";
					s += end_statement;
				}
				s += _make_indent(p_indent, p_cur_indent + 1) + _print_var(a[i], p_indent, p_cur_indent + 1, p_sort_keys);
			}
			s += end_statement + _make_indent(p_indent, p_cur_indent) + "]";
			return s;
		}
		case Variant::DICTIONARY: {
			String s = "{";
			s += end_statement;
			Dictionary d = p_var;
			List<Variant> keys;
			d.get_key_list(&keys);

			if (p_sort_keys) {
				keys.sort();
			}

			for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
				if (E != keys.front()) {
					s += ",";
					s += end_statement;
				}
				s += _make_indent(p_indent, p_cur_indent + 1) + _print_var(String(E->get()), p_indent, p_cur_indent + 1, p_sort_keys);
				s += colon;
				s += _print_var(d[E->get()], p_indent, p_cur_indent + 1, p_sort_keys);
			}

			s += end_statement + _make_indent(p_indent, p_cur_indent) + "}";
			return s;
		}
		default:
			return "\"" + String(p_var).json_escape() + "\"";
	}
}

String JSON::print(const Variant &p_var, const String &p_indent, bool p_sort_keys) {
	return _print_var(p_var, p_indent, 0, p_sort_keys);
}

Error JSON::_get_token(const char32_t *p_str, int &index, int p_len, Token &r_token, int &line, String &r_err_str) {
	while (p_len > 0) {
		switch (p_str[index]) {
			case '\n': {
				line++;
				index++;
				break;
			}
			case 0: {
				r_token.type = TK_EOF;
				return OK;
			} break;
			case '{': {
				r_token.type = TK_CURLY_BRACKET_OPEN;
				index++;
				return OK;
			}
			case '}': {
				r_token.type = TK_CURLY_BRACKET_CLOSE;
				index++;
				return OK;
			}
			case '[': {
				r_token.type = TK_BRACKET_OPEN;
				index++;
				return OK;
			}
			case ']': {
				r_token.type = TK_BRACKET_CLOSE;
				index++;
				return OK;
			}
			case ':': {
				r_token.type = TK_COLON;
				index++;
				return OK;
			}
			case ',': {
				r_token.type = TK_COMMA;
				index++;
				return OK;
			}
			case '"': {
				index++;
				String str;
				while (true) {
					if (p_str[index] == 0) {
						r_err_str = "Unterminated String";
						return ERR_PARSE_ERROR;
					} else if (p_str[index] == '"') {
						index++;
						break;
					} else if (p_str[index] == '\\') {
						//escaped characters...
						index++;
						char32_t next = p_str[index];
						if (next == 0) {
							r_err_str = "Unterminated String";
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
									char32_t c = p_str[index + j + 1];
									if (c == 0) {
										r_err_str = "Unterminated String";
										return ERR_PARSE_ERROR;
									}
									if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
										r_err_str = "Malformed hex constant in string";
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
								index += 4; //will add at the end anyway

								if ((res & 0xfffffc00) == 0xd800) {
									if (p_str[index + 1] != '\\' || p_str[index + 2] != 'u') {
										r_err_str = "Invalid UTF-16 sequence in string, unpaired lead surrogate";
										return ERR_PARSE_ERROR;
									}
									index += 2;
									char32_t trail = 0;
									for (int j = 0; j < 4; j++) {
										char32_t c = p_str[index + j + 1];
										if (c == 0) {
											r_err_str = "Unterminated String";
											return ERR_PARSE_ERROR;
										}
										if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
											r_err_str = "Malformed hex constant in string";
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

										trail <<= 4;
										trail |= v;
									}
									if ((trail & 0xfffffc00) == 0xdc00) {
										res = (res << 10UL) + trail - ((0xd800 << 10UL) + 0xdc00 - 0x10000);
										index += 4; //will add at the end anyway
									} else {
										r_err_str = "Invalid UTF-16 sequence in string, unpaired lead surrogate";
										return ERR_PARSE_ERROR;
									}
								} else if ((res & 0xfffffc00) == 0xdc00) {
									r_err_str = "Invalid UTF-16 sequence in string, unpaired trail surrogate";
									return ERR_PARSE_ERROR;
								}

							} break;
							default: {
								res = next;
							} break;
						}

						str += res;

					} else {
						if (p_str[index] == '\n') {
							line++;
						}
						str += p_str[index];
					}
					index++;
				}

				r_token.type = TK_STRING;
				r_token.value = str;
				return OK;

			} break;
			default: {
				if (p_str[index] <= 32) {
					index++;
					break;
				}

				if (p_str[index] == '-' || (p_str[index] >= '0' && p_str[index] <= '9')) {
					//a number
					const char32_t *rptr;
					double number = String::to_float(&p_str[index], &rptr);
					index += (rptr - &p_str[index]);
					r_token.type = TK_NUMBER;
					r_token.value = number;
					return OK;

				} else if ((p_str[index] >= 'A' && p_str[index] <= 'Z') || (p_str[index] >= 'a' && p_str[index] <= 'z')) {
					String id;

					while ((p_str[index] >= 'A' && p_str[index] <= 'Z') || (p_str[index] >= 'a' && p_str[index] <= 'z')) {
						id += p_str[index];
						index++;
					}

					r_token.type = TK_IDENTIFIER;
					r_token.value = id;
					return OK;
				} else {
					r_err_str = "Unexpected character.";
					return ERR_PARSE_ERROR;
				}
			}
		}
	}

	return ERR_PARSE_ERROR;
}

Error JSON::_parse_value(Variant &value, Token &token, const char32_t *p_str, int &index, int p_len, int &line, String &r_err_str) {
	if (token.type == TK_CURLY_BRACKET_OPEN) {
		Dictionary d;
		Error err = _parse_object(d, p_str, index, p_len, line, r_err_str);
		if (err) {
			return err;
		}
		value = d;
	} else if (token.type == TK_BRACKET_OPEN) {
		Array a;
		Error err = _parse_array(a, p_str, index, p_len, line, r_err_str);
		if (err) {
			return err;
		}
		value = a;
	} else if (token.type == TK_IDENTIFIER) {
		String id = token.value;
		if (id == "true") {
			value = true;
		} else if (id == "false") {
			value = false;
		} else if (id == "null") {
			value = Variant();
		} else {
			r_err_str = "Expected 'true','false' or 'null', got '" + id + "'.";
			return ERR_PARSE_ERROR;
		}
	} else if (token.type == TK_NUMBER) {
		value = token.value;
	} else if (token.type == TK_STRING) {
		value = token.value;
	} else {
		r_err_str = "Expected value, got " + String(tk_name[token.type]) + ".";
		return ERR_PARSE_ERROR;
	}

	return OK;
}

Error JSON::_parse_array(Array &array, const char32_t *p_str, int &index, int p_len, int &line, String &r_err_str) {
	Token token;
	bool need_comma = false;

	while (index < p_len) {
		Error err = _get_token(p_str, index, p_len, token, line, r_err_str);
		if (err != OK) {
			return err;
		}

		if (token.type == TK_BRACKET_CLOSE) {
			return OK;
		}

		if (need_comma) {
			if (token.type != TK_COMMA) {
				r_err_str = "Expected ','";
				return ERR_PARSE_ERROR;
			} else {
				need_comma = false;
				continue;
			}
		}

		Variant v;
		err = _parse_value(v, token, p_str, index, p_len, line, r_err_str);
		if (err) {
			return err;
		}

		array.push_back(v);
		need_comma = true;
	}

	r_err_str = "Expected ']'";
	return ERR_PARSE_ERROR;
}

Error JSON::_parse_object(Dictionary &object, const char32_t *p_str, int &index, int p_len, int &line, String &r_err_str) {
	bool at_key = true;
	String key;
	Token token;
	bool need_comma = false;

	while (index < p_len) {
		if (at_key) {
			Error err = _get_token(p_str, index, p_len, token, line, r_err_str);
			if (err != OK) {
				return err;
			}

			if (token.type == TK_CURLY_BRACKET_CLOSE) {
				return OK;
			}

			if (need_comma) {
				if (token.type != TK_COMMA) {
					r_err_str = "Expected '}' or ','";
					return ERR_PARSE_ERROR;
				} else {
					need_comma = false;
					continue;
				}
			}

			if (token.type != TK_STRING) {
				r_err_str = "Expected key";
				return ERR_PARSE_ERROR;
			}

			key = token.value;
			err = _get_token(p_str, index, p_len, token, line, r_err_str);
			if (err != OK) {
				return err;
			}
			if (token.type != TK_COLON) {
				r_err_str = "Expected ':'";
				return ERR_PARSE_ERROR;
			}
			at_key = false;
		} else {
			Error err = _get_token(p_str, index, p_len, token, line, r_err_str);
			if (err != OK) {
				return err;
			}

			Variant v;
			err = _parse_value(v, token, p_str, index, p_len, line, r_err_str);
			if (err) {
				return err;
			}
			object[key] = v;
			need_comma = true;
			at_key = true;
		}
	}

	r_err_str = "Expected '}'";
	return ERR_PARSE_ERROR;
}

Error JSON::parse(const String &p_json, Variant &r_ret, String &r_err_str, int &r_err_line) {
	const char32_t *str = p_json.ptr();
	int idx = 0;
	int len = p_json.length();
	Token token;
	r_err_line = 0;
	String aux_key;

	Error err = _get_token(str, idx, len, token, r_err_line, r_err_str);
	if (err) {
		return err;
	}

	err = _parse_value(r_ret, token, str, idx, len, r_err_line, r_err_str);

	// Check if EOF is reached
	// or it's a type of the next token.
	if (err == OK && idx < len) {
		err = _get_token(str, idx, len, token, r_err_line, r_err_str);

		if (err || token.type != TK_EOF) {
			r_err_str = "Expected 'EOF'";
			// Reset return value to empty `Variant`
			r_ret = Variant();
			return ERR_PARSE_ERROR;
		}
	}

	return err;
}

Error JSONParser::parse_string(const String &p_json_string) {
	return JSON::parse(p_json_string, data, err_text, err_line);
}
String JSONParser::get_error_text() const {
	return err_text;
}
int JSONParser::get_error_line() const {
	return err_line;
}
Variant JSONParser::get_data() const {
	return data;
}

Error JSONParser::decode_data(const Variant &p_data, const String &p_indent, bool p_sort_keys) {
	string = JSON::print(p_data, p_indent, p_sort_keys);
	data = p_data;
	return OK;
}

String JSONParser::get_string() const {
	return string;
}

void JSONParser::_bind_methods() {
	ClassDB::bind_method(D_METHOD("parse_string", "json_string"), &JSONParser::parse_string);
	ClassDB::bind_method(D_METHOD("get_error_text"), &JSONParser::get_error_text);
	ClassDB::bind_method(D_METHOD("get_error_line"), &JSONParser::get_error_line);
	ClassDB::bind_method(D_METHOD("get_data"), &JSONParser::get_data);
	ClassDB::bind_method(D_METHOD("decode_data", "data", "indent", "sort_keys"), &JSONParser::decode_data, DEFVAL(""), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_string"), &JSONParser::get_string);
}
