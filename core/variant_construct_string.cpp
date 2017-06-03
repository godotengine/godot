/*************************************************************************/
/*  variant_construct_string.cpp                                         */
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
#include "variant.h"

class VariantConstruct {

	enum TokenType {
		TK_CURLY_BRACKET_OPEN,
		TK_CURLY_BRACKET_CLOSE,
		TK_BRACKET_OPEN,
		TK_BRACKET_CLOSE,
		TK_IDENTIFIER,
		TK_STRING,
		TK_NUMBER,
		TK_COLON,
		TK_COMMA,
		TK_EOF,
		TK_MAX
	};

	enum Expecting {

		EXPECT_OBJECT,
		EXPECT_OBJECT_KEY,
		EXPECT_COLON,
		EXPECT_OBJECT_VALUE,
	};

	struct Token {

		TokenType type;
		Variant value;
	};

	static const char *tk_name[TK_MAX];

	static String _print_var(const Variant &p_var);

	static Error _get_token(const CharType *p_str, int &index, int p_len, Token &r_token, int &line, String &r_err_str);
	static Error _parse_value(Variant &value, Token &token, const CharType *p_str, int &index, int p_len, int &line, String &r_err_str, Variant::ObjectConstruct *p_construct, void *p_ud);
	static Error _parse_array(Array &array, const CharType *p_str, int &index, int p_len, int &line, String &r_err_str, Variant::ObjectConstruct *p_construct, void *p_ud);
	static Error _parse_dict(Dictionary &object, const CharType *p_str, int &index, int p_len, int &line, String &r_err_str, Variant::ObjectConstruct *p_construct, void *p_ud);

public:
	static Error parse(const String &p_string, Variant &r_ret, String &r_err_str, int &r_err_line, Variant::ObjectConstruct *p_construct, void *p_ud);
};

const char *VariantConstruct::tk_name[TK_MAX] = {
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

Error VariantConstruct::_get_token(const CharType *p_str, int &idx, int p_len, Token &r_token, int &line, String &r_err_str) {

	while (true) {
		switch (p_str[idx]) {

			case '\n': {

				line++;
				idx++;
				break;
			};
			case 0: {
				r_token.type = TK_EOF;
				return OK;
			} break;
			case '{': {

				r_token.type = TK_CURLY_BRACKET_OPEN;
				idx++;
				return OK;
			};
			case '}': {

				r_token.type = TK_CURLY_BRACKET_CLOSE;
				idx++;
				return OK;
			};
			case '[': {

				r_token.type = TK_BRACKET_OPEN;
				idx++;
				return OK;
			};
			case ']': {

				r_token.type = TK_BRACKET_CLOSE;
				idx++;
				return OK;
			};
			case ':': {

				r_token.type = TK_COLON;
				idx++;
				return OK;
			};
			case ',': {

				r_token.type = TK_COMMA;
				idx++;
				return OK;
			};
			case '"': {

				idx++;
				String str;
				while (true) {
					if (p_str[idx] == 0) {
						r_err_str = "Unterminated String";
						return ERR_PARSE_ERROR;
					} else if (p_str[idx] == '"') {
						idx++;
						break;
					} else if (p_str[idx] == '\\') {
						//escaped characters...
						idx++;
						CharType next = p_str[idx];
						if (next == 0) {
							r_err_str = "Unterminated String";
							return ERR_PARSE_ERROR;
						}
						CharType res = 0;

						switch (next) {

							case 'b': res = 8; break;
							case 't': res = 9; break;
							case 'n': res = 10; break;
							case 'f': res = 12; break;
							case 'r': res = 13; break;
							case '\"': res = '\"'; break;
							case '\\': res = '\\'; break;
							case '/': res = '/'; break;
							case 'u': {
								//hexnumbarh - oct is deprecated

								for (int j = 0; j < 4; j++) {
									CharType c = p_str[idx + j + 1];
									if (c == 0) {
										r_err_str = "Unterminated String";
										return ERR_PARSE_ERROR;
									}
									if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {

										r_err_str = "Malformed hex constant in string";
										return ERR_PARSE_ERROR;
									}
									CharType v;
									if (c >= '0' && c <= '9') {
										v = c - '0';
									} else if (c >= 'a' && c <= 'f') {
										v = c - 'a';
										v += 10;
									} else if (c >= 'A' && c <= 'F') {
										v = c - 'A';
										v += 10;
									} else {
										ERR_PRINT("BUG");
										v = 0;
									}

									res <<= 4;
									res |= v;
								}
								idx += 4; //will add at the end anyway

							} break;
							default: {

								r_err_str = "Invalid escape sequence";
								return ERR_PARSE_ERROR;
							} break;
						}

						str += res;

					} else {
						if (p_str[idx] == '\n')
							line++;
						str += p_str[idx];
					}
					idx++;
				}

				r_token.type = TK_STRING;
				r_token.value = str;
				return OK;

			} break;
			default: {

				if (p_str[idx] <= 32) {
					idx++;
					break;
				}

				if (p_str[idx] == '-' || (p_str[idx] >= '0' && p_str[idx] <= '9')) {
					//a number
					const CharType *rptr;
					double number = String::to_double(&p_str[idx], &rptr);
					idx += (rptr - &p_str[idx]);
					r_token.type = TK_NUMBER;
					r_token.value = number;
					return OK;

				} else if ((p_str[idx] >= 'A' && p_str[idx] <= 'Z') || (p_str[idx] >= 'a' && p_str[idx] <= 'z')) {

					String id;

					while ((p_str[idx] >= 'A' && p_str[idx] <= 'Z') || (p_str[idx] >= 'a' && p_str[idx] <= 'z')) {

						id += p_str[idx];
						idx++;
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

Error VariantConstruct::_parse_value(Variant &value, Token &token, const CharType *p_str, int &index, int p_len, int &line, String &r_err_str, Variant::ObjectConstruct *p_construct, void *p_ud) {

	if (token.type == TK_CURLY_BRACKET_OPEN) {

		Dictionary d;
		Error err = _parse_dict(d, p_str, index, p_len, line, r_err_str, p_construct, p_ud);
		if (err)
			return err;
		value = d;
		return OK;
	} else if (token.type == TK_BRACKET_OPEN) {

		Array a;
		Error err = _parse_array(a, p_str, index, p_len, line, r_err_str, p_construct, p_ud);
		if (err)
			return err;
		value = a;
		return OK;

	} else if (token.type == TK_IDENTIFIER) {

		String id = token.value;
		if (id == "true")
			value = true;
		else if (id == "false")
			value = false;
		else if (id == "null")
			value = Variant();
		else {
			r_err_str = "Expected 'true','false' or 'null', got '" + id + "'.";
			return ERR_PARSE_ERROR;
		}
		return OK;

	} else if (token.type == TK_NUMBER) {

		value = token.value;
		return OK;
	} else if (token.type == TK_STRING) {

		value = token.value;
		return OK;
	} else {
		r_err_str = "Expected value, got " + String(tk_name[token.type]) + ".";
		return ERR_PARSE_ERROR;
	}

	return ERR_PARSE_ERROR;
}

Error VariantConstruct::_parse_array(Array &array, const CharType *p_str, int &index, int p_len, int &line, String &r_err_str, Variant::ObjectConstruct *p_construct, void *p_ud) {

	Token token;
	bool need_comma = false;

	while (index < p_len) {

		Error err = _get_token(p_str, index, p_len, token, line, r_err_str);
		if (err != OK)
			return err;

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
		err = _parse_value(v, token, p_str, index, p_len, line, r_err_str, p_construct, p_ud);
		if (err)
			return err;

		array.push_back(v);
		need_comma = true;
	}

	return OK;
}

Error VariantConstruct::_parse_dict(Dictionary &dict, const CharType *p_str, int &index, int p_len, int &line, String &r_err_str, Variant::ObjectConstruct *p_construct, void *p_ud) {

	bool at_key = true;
	Variant key;
	Token token;
	bool need_comma = false;

	while (index < p_len) {

		if (at_key) {

			Error err = _get_token(p_str, index, p_len, token, line, r_err_str);
			if (err != OK)
				return err;

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

			err = _parse_value(key, token, p_str, index, p_len, line, r_err_str, p_construct, p_ud);

			if (err != OK)
				return err;

			err = _get_token(p_str, index, p_len, token, line, r_err_str);

			if (err != OK)
				return err;

			if (token.type != TK_COLON) {

				r_err_str = "Expected ':'";
				return ERR_PARSE_ERROR;
			}
			at_key = false;
		} else {

			Error err = _get_token(p_str, index, p_len, token, line, r_err_str);
			if (err != OK)
				return err;

			Variant v;
			err = _parse_value(v, token, p_str, index, p_len, line, r_err_str, p_construct, p_ud);
			if (err)
				return err;
			dict[key] = v;
			need_comma = true;
			at_key = true;
		}
	}

	return OK;
}

Error VariantConstruct::parse(const String &p_string, Variant &r_ret, String &r_err_str, int &r_err_line, Variant::ObjectConstruct *p_construct, void *p_ud) {

	const CharType *str = p_string.ptr();
	int idx = 0;
	int len = p_string.length();
	Token token;
	r_err_line = 0;
	String aux_key;

	Error err = _get_token(str, idx, len, token, r_err_line, r_err_str);
	if (err)
		return err;

	return _parse_value(r_ret, token, str, idx, len, r_err_line, r_err_str, p_construct, p_ud);
}
