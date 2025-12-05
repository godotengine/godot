/**************************************************************************/
/*  variant_parser.cpp                                                    */
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

#include "variant_parser.h"

#include "core/crypto/crypto_core.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_uid.h"
#include "core/object/script_language.h"
#include "core/string/string_buffer.h"

char32_t VariantParser::Stream::get_char() {
	// is within buffer?
	if (readahead_pointer < readahead_filled) {
		return readahead_buffer[readahead_pointer++];
	}

	// attempt to readahead
	readahead_filled = _read_buffer(readahead_buffer, readahead_enabled ? READAHEAD_SIZE : 1);
	if (readahead_filled) {
		readahead_pointer = 0;
	} else {
		// EOF
		readahead_pointer = 1;
		eof = true;
		return 0;
	}
	return get_char();
}

bool VariantParser::Stream::is_eof() const {
	if (readahead_enabled) {
		return eof;
	}
	return _is_eof();
}

bool VariantParser::StreamFile::is_utf8() const {
	return true;
}

bool VariantParser::StreamFile::_is_eof() const {
	return f->eof_reached();
}

uint32_t VariantParser::StreamFile::_read_buffer(char32_t *p_buffer, uint32_t p_num_chars) {
	// The buffer is assumed to include at least one character (for null terminator)
	ERR_FAIL_COND_V(!p_num_chars, 0);

	uint8_t *temp = (uint8_t *)alloca(p_num_chars);
	uint64_t num_read = f->get_buffer(temp, p_num_chars);
	ERR_FAIL_COND_V(num_read == UINT64_MAX, 0);

	// translate to wchar
	for (uint32_t n = 0; n < num_read; n++) {
		p_buffer[n] = temp[n];
	}

	// could be less than p_num_chars, or zero
	return num_read;
}

bool VariantParser::StreamString::is_utf8() const {
	return false;
}

bool VariantParser::StreamString::_is_eof() const {
	return pos > s.length();
}

uint32_t VariantParser::StreamString::_read_buffer(char32_t *p_buffer, uint32_t p_num_chars) {
	// The buffer is assumed to include at least one character (for null terminator)
	ERR_FAIL_COND_V(!p_num_chars, 0);
	ERR_FAIL_NULL_V(p_buffer, 0);

	int available = MAX(s.length() - pos, 0);
	if (available >= (int)p_num_chars) {
		const char32_t *src = s.ptr();
		src += pos;
		memcpy(p_buffer, src, p_num_chars * sizeof(char32_t));
		pos += p_num_chars;

		return p_num_chars;
	}

	// going to reach EOF
	if (available) {
		const char32_t *src = s.ptr();
		src += pos;
		memcpy(p_buffer, src, available * sizeof(char32_t));
		pos += available;
	}

	// add a zero
	p_buffer[available] = 0;

	return available;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

const char *VariantParser::tk_name[TK_MAX] = {
	"'{'",
	"'}'",
	"'['",
	"']'",
	"'('",
	"')'",
	"identifier",
	"string",
	"string_name",
	"number",
	"color",
	"':'",
	"','",
	"'.'",
	"'='",
	"EOF",
	"ERROR"
};

static double stor_fix(const String &p_str) {
	if (p_str == "inf") {
		return Math::INF;
	} else if (p_str == "-inf" || p_str == "inf_neg") {
		// inf_neg kept for compatibility.
		return -Math::INF;
	} else if (p_str == "nan") {
		return Math::NaN;
	}
	return -1;
}

Error VariantParser::get_token(Stream *p_stream, Token &r_token, int &line, String &r_err_str) {
	bool string_name = false;

	while (true) {
		char32_t cchar;
		if (p_stream->saved) {
			cchar = p_stream->saved;
			p_stream->saved = 0;
		} else {
			cchar = p_stream->get_char();
			if (p_stream->is_eof()) {
				r_token.type = TK_EOF;
				return OK;
			}
		}

		switch (cchar) {
			case '\n': {
				line++;
				break;
			}
			case 0: {
				r_token.type = TK_EOF;
				return OK;
			} break;
			case '{': {
				r_token.type = TK_CURLY_BRACKET_OPEN;
				return OK;
			}
			case '}': {
				r_token.type = TK_CURLY_BRACKET_CLOSE;
				return OK;
			}
			case '[': {
				r_token.type = TK_BRACKET_OPEN;
				return OK;
			}
			case ']': {
				r_token.type = TK_BRACKET_CLOSE;
				return OK;
			}
			case '(': {
				r_token.type = TK_PARENTHESIS_OPEN;
				return OK;
			}
			case ')': {
				r_token.type = TK_PARENTHESIS_CLOSE;
				return OK;
			}
			case ':': {
				r_token.type = TK_COLON;
				return OK;
			}
			case ';': {
				while (true) {
					char32_t ch = p_stream->get_char();
					if (p_stream->is_eof()) {
						r_token.type = TK_EOF;
						return OK;
					}
					if (ch == '\n') {
						line++;
						break;
					}
				}

				break;
			}
			case ',': {
				r_token.type = TK_COMMA;
				return OK;
			}
			case '.': {
				r_token.type = TK_PERIOD;
				return OK;
			}
			case '=': {
				r_token.type = TK_EQUAL;
				return OK;
			}
			case '#': {
				StringBuffer<> color_str;
				color_str += '#';
				while (true) {
					char32_t ch = p_stream->get_char();
					if (p_stream->is_eof()) {
						r_token.type = TK_EOF;
						return OK;
					} else if (is_hex_digit(ch)) {
						color_str += ch;

					} else {
						p_stream->saved = ch;
						break;
					}
				}

				r_token.value = Color::html(color_str.as_string());
				r_token.type = TK_COLOR;
				return OK;
			}
#ifndef DISABLE_DEPRECATED
			case '@': // Compatibility with 3.x StringNames.
#endif
			case '&': { // StringName.
				cchar = p_stream->get_char();
				if (cchar != '"') {
					r_err_str = "Expected '\"' after '&'";
					r_token.type = TK_ERROR;
					return ERR_PARSE_ERROR;
				}

				string_name = true;
				[[fallthrough]];
			}
			case '"': {
				String str;
				char32_t prev = 0;
				while (true) {
					char32_t ch = p_stream->get_char();

					if (ch == 0) {
						r_err_str = "Unterminated string";
						r_token.type = TK_ERROR;
						return ERR_PARSE_ERROR;
					} else if (ch == '"') {
						break;
					} else if (ch == '\\') {
						//escaped characters...
						char32_t next = p_stream->get_char();
						if (next == 0) {
							r_err_str = "Unterminated string";
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
							case 'U':
							case 'u': {
								// Hexadecimal sequence.
								int hex_len = (next == 'U') ? 6 : 4;
								for (int j = 0; j < hex_len; j++) {
									char32_t c = p_stream->get_char();

									if (c == 0) {
										r_err_str = "Unterminated string";
										r_token.type = TK_ERROR;
										return ERR_PARSE_ERROR;
									}
									if (!is_hex_digit(c)) {
										r_err_str = "Malformed hex constant in string";
										r_token.type = TK_ERROR;
										return ERR_PARSE_ERROR;
									}
									char32_t v;
									if (is_digit(c)) {
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

						// Parse UTF-16 pair.
						if ((res & 0xfffffc00) == 0xd800) {
							if (prev == 0) {
								prev = res;
								continue;
							} else {
								r_err_str = "Invalid UTF-16 sequence in string, unpaired lead surrogate";
								r_token.type = TK_ERROR;
								return ERR_PARSE_ERROR;
							}
						} else if ((res & 0xfffffc00) == 0xdc00) {
							if (prev == 0) {
								r_err_str = "Invalid UTF-16 sequence in string, unpaired trail surrogate";
								r_token.type = TK_ERROR;
								return ERR_PARSE_ERROR;
							} else {
								res = (prev << 10UL) + res - ((0xd800 << 10UL) + 0xdc00 - 0x10000);
								prev = 0;
							}
						}
						if (prev != 0) {
							r_err_str = "Invalid UTF-16 sequence in string, unpaired lead surrogate";
							r_token.type = TK_ERROR;
							return ERR_PARSE_ERROR;
						}
						str += res;
					} else {
						if (prev != 0) {
							r_err_str = "Invalid UTF-16 sequence in string, unpaired lead surrogate";
							r_token.type = TK_ERROR;
							return ERR_PARSE_ERROR;
						}
						if (ch == '\n') {
							line++;
						}
						str += ch;
					}
				}
				if (prev != 0) {
					r_err_str = "Invalid UTF-16 sequence in string, unpaired lead surrogate";
					r_token.type = TK_ERROR;
					return ERR_PARSE_ERROR;
				}

				if (p_stream->is_utf8()) {
					// Re-interpret the string we built as ascii.
					CharString string_as_ascii = str.ascii(true);
					str.clear();
					str.append_utf8(string_as_ascii);
				}
				if (string_name) {
					r_token.type = TK_STRING_NAME;
					r_token.value = StringName(str);
				} else {
					r_token.type = TK_STRING;
					r_token.value = str;
				}
				return OK;

			} break;
			default: {
				if (cchar <= 32) {
					break;
				}
				StringBuffer<> token_text;
				if (cchar == '-') {
					token_text += '-';
					cchar = p_stream->get_char();
				}
				if (cchar >= '0' && cchar <= '9') {
					//a number
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
								if (is_digit(c)) {
									//pass
								} else if (c == '.') {
									reading = READING_DEC;
									is_float = true;
								} else if (c == 'e' || c == 'E') {
									reading = READING_EXP;
									is_float = true;
								} else {
									reading = READING_DONE;
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
						token_text += c;
						c = p_stream->get_char();
					}

					p_stream->saved = c;

					r_token.type = TK_NUMBER;

					if (is_float) {
						r_token.value = token_text.as_double();
					} else {
						r_token.value = token_text.as_int();
					}
					return OK;
				} else if (is_ascii_alphabet_char(cchar) || is_underscore(cchar)) {
					bool first = true;

					while (is_ascii_alphabet_char(cchar) || is_underscore(cchar) || (!first && is_digit(cchar))) {
						token_text += cchar;
						cchar = p_stream->get_char();
						first = false;
					}

					p_stream->saved = cchar;

					r_token.type = TK_IDENTIFIER;
					r_token.value = token_text.as_string();
					return OK;
				} else {
					r_err_str = "Unexpected character";
					r_token.type = TK_ERROR;
					return ERR_PARSE_ERROR;
				}
			}
		}
	}

	r_err_str = "Unknown error getting token";
	r_token.type = TK_ERROR;
	return ERR_PARSE_ERROR;
}

Error VariantParser::_parse_enginecfg(Stream *p_stream, Vector<String> &strings, int &line, String &r_err_str) {
	Token token;
	get_token(p_stream, token, line, r_err_str);
	if (token.type != TK_PARENTHESIS_OPEN) {
		r_err_str = "Expected '(' in old-style project.godot construct";
		return ERR_PARSE_ERROR;
	}

	String accum;

	while (true) {
		char32_t c = p_stream->get_char();

		if (p_stream->is_eof()) {
			r_err_str = "Unexpected EOF while parsing old-style project.godot construct";
			return ERR_PARSE_ERROR;
		}

		if (c == ',') {
			strings.push_back(accum.strip_edges());
			accum = String();
		} else if (c == ')') {
			strings.push_back(accum.strip_edges());
			return OK;
		} else if (c == '\n') {
			line++;
		}
	}
}

template <typename T>
Error VariantParser::_parse_construct(Stream *p_stream, Vector<T> &r_construct, int &line, String &r_err_str) {
	Token token;
	get_token(p_stream, token, line, r_err_str);
	if (token.type != TK_PARENTHESIS_OPEN) {
		r_err_str = "Expected '(' in constructor";
		return ERR_PARSE_ERROR;
	}

	bool first = true;
	while (true) {
		if (!first) {
			get_token(p_stream, token, line, r_err_str);
			if (token.type == TK_COMMA) {
				//do none
			} else if (token.type == TK_PARENTHESIS_CLOSE) {
				break;
			} else {
				r_err_str = "Expected ',' or ')' in constructor";
				return ERR_PARSE_ERROR;
			}
		}
		get_token(p_stream, token, line, r_err_str);

		if (first && token.type == TK_PARENTHESIS_CLOSE) {
			break;
		} else if (token.type != TK_NUMBER) {
			bool valid = false;
			if (token.type == TK_IDENTIFIER) {
				double real = stor_fix(token.value);
				if (real != -1) {
					token.type = TK_NUMBER;
					token.value = real;
					valid = true;
				}
			}
			if (!valid) {
				r_err_str = "Expected float in constructor";
				return ERR_PARSE_ERROR;
			}
		}

		r_construct.push_back(token.value);
		first = false;
	}

	return OK;
}

Error VariantParser::_parse_byte_array(Stream *p_stream, Vector<uint8_t> &r_construct, int &line, String &r_err_str) {
	Token token;
	get_token(p_stream, token, line, r_err_str);
	if (token.type != TK_PARENTHESIS_OPEN) {
		r_err_str = "Expected '(' in constructor";
		return ERR_PARSE_ERROR;
	}

	get_token(p_stream, token, line, r_err_str);
	if (token.type == TK_STRING) {
		// Base64 encoded array.
		String base64_encoded_string = token.value;
		int strlen = base64_encoded_string.length();
		CharString cstr = base64_encoded_string.ascii();

		size_t arr_len = 0;
		r_construct.resize(strlen / 4 * 3 + 1);
		uint8_t *w = r_construct.ptrw();
		Error err = CryptoCore::b64_decode(&w[0], r_construct.size(), &arr_len, (unsigned char *)cstr.get_data(), strlen);
		if (err) {
			r_err_str = "Invalid base64-encoded string";
			return ERR_PARSE_ERROR;
		}
		r_construct.resize(arr_len);

		get_token(p_stream, token, line, r_err_str);
		if (token.type != TK_PARENTHESIS_CLOSE) {
			r_err_str = "Expected ')' in constructor";
			return ERR_PARSE_ERROR;
		}

	} else if (token.type == TK_NUMBER || token.type == TK_IDENTIFIER) {
		// Individual elements.
		while (true) {
			if (token.type != TK_NUMBER) {
				bool valid = false;
				if (token.type == TK_IDENTIFIER) {
					double real = stor_fix(token.value);
					if (real != -1) {
						token.type = TK_NUMBER;
						token.value = real;
						valid = true;
					}
				}
				if (!valid) {
					r_err_str = "Expected number in constructor";
					return ERR_PARSE_ERROR;
				}
			}

			r_construct.push_back(token.value);

			get_token(p_stream, token, line, r_err_str);

			if (token.type == TK_COMMA) {
				//do none
			} else if (token.type == TK_PARENTHESIS_CLOSE) {
				break;
			} else {
				r_err_str = "Expected ',' or ')' in constructor";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream, token, line, r_err_str);
		}
	} else if (token.type == TK_PARENTHESIS_CLOSE) {
		// Empty array.
		return OK;
	} else {
		r_err_str = "Expected base64 string, or list of numbers in constructor";
		return ERR_PARSE_ERROR;
	}

	return OK;
}

VariantParser::ParsedTypeResult VariantParser::_parse_type(Token &token, Stream *p_stream, int &line, String &r_err_str, ResourceParser *p_res_parser, TokenType expected_token_type) {
	static HashMap<StringName, Variant::Type> builtin_types;
	if (builtin_types.is_empty()) {
		for (int i = 1; i < Variant::VARIANT_MAX; i++) {
			builtin_types[Variant::get_type_name((Variant::Type)i)] = (Variant::Type)i;
		}
	}

	VariantParser::ParsedTypeResult parsed_type;
	parsed_type.error = OK;
	parsed_type.got_expected_token = false;

	if (builtin_types.has(token.value)) {
		parsed_type.type.builtin_type = builtin_types.get(token.value);
		parsed_type.type.class_name = StringName();
		parsed_type.type.script = Variant();

	} else if (token.value == "Resource" || token.value == "SubResource" || token.value == "ExtResource") {
		Variant resource;
		parsed_type.error = parse_value(token, resource, p_stream, line, r_err_str, p_res_parser);
		if (parsed_type.error) {
			if (token.value == "Resource" && parsed_type.error == ERR_PARSE_ERROR && r_err_str == "Expected '('" && token.type == expected_token_type) {
				parsed_type.error = OK;
				r_err_str = String();
				parsed_type.type.builtin_type = Variant::OBJECT;
				parsed_type.type.class_name = token.value;
				parsed_type.type.script = Variant();
				parsed_type.got_expected_token = true;

			} else {
				return parsed_type;
			}
		} else {
			Ref<Script> script = resource;
			if (script.is_valid() && script->is_valid()) {
				parsed_type.type.builtin_type = Variant::OBJECT;
				parsed_type.type.class_name = script->get_instance_base_type();
				parsed_type.type.script = script;
			}
		}
	} else if (ClassDB::class_exists(token.value)) {
		parsed_type.type.builtin_type = Variant::OBJECT;
		parsed_type.type.class_name = token.value;
		parsed_type.type.script = Variant();
	}
	return parsed_type;
}

Error VariantParser::parse_value(Token &token, Variant &value, Stream *p_stream, int &line, String &r_err_str, ResourceParser *p_res_parser) {
	if (token.type == TK_CURLY_BRACKET_OPEN) {
		Dictionary d;
		Error err = _parse_dictionary(d, p_stream, line, r_err_str, p_res_parser);
		if (err) {
			return err;
		}
		value = d;
		return OK;
	} else if (token.type == TK_BRACKET_OPEN) {
		Array a;
		Error err = _parse_array(a, p_stream, line, r_err_str, p_res_parser);
		if (err) {
			return err;
		}
		value = a;
		return OK;
	} else if (token.type == TK_IDENTIFIER) {
		String id = token.value;
		if (id == "true") {
			value = true;
		} else if (id == "false") {
			value = false;
		} else if (id == "null" || id == "nil") {
			value = Variant();
		} else if (id == "inf") {
			value = Math::INF;
		} else if (id == "-inf" || id == "inf_neg") {
			// inf_neg kept for compatibility.
			value = -Math::INF;
		} else if (id == "nan") {
			value = Math::NaN;
		} else if (id == "Vector2") {
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 2) {
				r_err_str = "Expected 2 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Vector2(args[0], args[1]);
		} else if (id == "Vector2i") {
			Vector<int32_t> args;
			Error err = _parse_construct<int32_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 2) {
				r_err_str = "Expected 2 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Vector2i(args[0], args[1]);
		} else if (id == "Rect2") {
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 4) {
				r_err_str = "Expected 4 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Rect2(args[0], args[1], args[2], args[3]);
		} else if (id == "Rect2i") {
			Vector<int32_t> args;
			Error err = _parse_construct<int32_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 4) {
				r_err_str = "Expected 4 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Rect2i(args[0], args[1], args[2], args[3]);
		} else if (id == "Vector3") {
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 3) {
				r_err_str = "Expected 3 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Vector3(args[0], args[1], args[2]);
		} else if (id == "Vector3i") {
			Vector<int32_t> args;
			Error err = _parse_construct<int32_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 3) {
				r_err_str = "Expected 3 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Vector3i(args[0], args[1], args[2]);
		} else if (id == "Vector4") {
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 4) {
				r_err_str = "Expected 4 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Vector4(args[0], args[1], args[2], args[3]);
		} else if (id == "Vector4i") {
			Vector<int32_t> args;
			Error err = _parse_construct<int32_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 4) {
				r_err_str = "Expected 4 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Vector4i(args[0], args[1], args[2], args[3]);
		} else if (id == "Transform2D" || id == "Matrix32") { //compatibility
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 6) {
				r_err_str = "Expected 6 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			Transform2D m;
			m[0] = Vector2(args[0], args[1]);
			m[1] = Vector2(args[2], args[3]);
			m[2] = Vector2(args[4], args[5]);
			value = m;
		} else if (id == "Plane") {
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 4) {
				r_err_str = "Expected 4 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Plane(args[0], args[1], args[2], args[3]);
		} else if (id == "Quaternion" || id == "Quat") { // "Quat" kept for compatibility
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 4) {
				r_err_str = "Expected 4 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Quaternion(args[0], args[1], args[2], args[3]);
		} else if (id == "AABB" || id == "Rect3") {
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 6) {
				r_err_str = "Expected 6 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = AABB(Vector3(args[0], args[1], args[2]), Vector3(args[3], args[4], args[5]));
		} else if (id == "Basis" || id == "Matrix3") { //compatibility
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 9) {
				r_err_str = "Expected 9 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Basis(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]);
		} else if (id == "Transform3D" || id == "Transform") { // "Transform" kept for compatibility with Godot <4.
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 12) {
				r_err_str = "Expected 12 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Transform3D(Basis(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]), Vector3(args[9], args[10], args[11]));
		} else if (id == "Projection") { // "Transform" kept for compatibility with Godot <4.
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 16) {
				r_err_str = "Expected 16 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Projection(Vector4(args[0], args[1], args[2], args[3]), Vector4(args[4], args[5], args[6], args[7]), Vector4(args[8], args[9], args[10], args[11]), Vector4(args[12], args[13], args[14], args[15]));
		} else if (id == "Color") {
			Vector<float> args;
			Error err = _parse_construct<float>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			if (args.size() != 4) {
				r_err_str = "Expected 4 arguments for constructor";
				return ERR_PARSE_ERROR;
			}

			value = Color(args[0], args[1], args[2], args[3]);
		} else if (id == "NodePath") {
			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_OPEN) {
				r_err_str = "Expected '('";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_STRING) {
				r_err_str = "Expected string as argument for NodePath()";
				return ERR_PARSE_ERROR;
			}

			value = NodePath(String(token.value));

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_CLOSE) {
				r_err_str = "Expected ')'";
				return ERR_PARSE_ERROR;
			}
		} else if (id == "RID") {
			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_OPEN) {
				r_err_str = "Expected '('";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream, token, line, r_err_str);
			// Permit empty RID.
			if (token.type == TK_PARENTHESIS_CLOSE) {
				value = RID();
				return OK;
			} else if (token.type != TK_NUMBER) {
				r_err_str = "Expected number as argument or ')'";
				return ERR_PARSE_ERROR;
			}

			value = RID::from_uint64(token.value);

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_CLOSE) {
				r_err_str = "Expected ')'";
				return ERR_PARSE_ERROR;
			}
		} else if (id == "Signal") {
			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_OPEN) {
				r_err_str = "Expected '('";
				return ERR_PARSE_ERROR;
			}

			// Load as empty.
			value = Signal();

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_CLOSE) {
				r_err_str = "Expected ')'";
				return ERR_PARSE_ERROR;
			}
		} else if (id == "Callable") {
			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_OPEN) {
				r_err_str = "Expected '('";
				return ERR_PARSE_ERROR;
			}

			// Load as empty.
			value = Callable();

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_CLOSE) {
				r_err_str = "Expected ')'";
				return ERR_PARSE_ERROR;
			}
		} else if (id == "Object") {
			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_OPEN) {
				r_err_str = "Expected '('";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream, token, line, r_err_str);

			if (token.type != TK_IDENTIFIER) {
				r_err_str = "Expected identifier with type of object";
				return ERR_PARSE_ERROR;
			}

			String type = token.value;

			Object *obj = ClassDB::instantiate(type);

			if (!obj) {
				r_err_str = vformat("Can't instantiate Object() of type '%s'", type);
				return ERR_PARSE_ERROR;
			}

			Ref<RefCounted> ref = Ref<RefCounted>(Object::cast_to<RefCounted>(obj));

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_COMMA) {
				r_err_str = "Expected ',' after object type";
				return ERR_PARSE_ERROR;
			}

			bool at_key = true;
			String key;
			bool need_comma = false;

			while (true) {
				if (p_stream->is_eof()) {
					r_err_str = "Unexpected EOF while parsing Object()";
					return ERR_FILE_CORRUPT;
				}

				if (at_key) {
					Error err = get_token(p_stream, token, line, r_err_str);
					if (err != OK) {
						return err;
					}

					if (token.type == TK_PARENTHESIS_CLOSE) {
						value = ref.is_valid() ? Variant(ref) : Variant(obj);
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
						r_err_str = "Expected property name as string";
						return ERR_PARSE_ERROR;
					}

					key = token.value;

					err = get_token(p_stream, token, line, r_err_str);

					if (err != OK) {
						return err;
					}
					if (token.type != TK_COLON) {
						r_err_str = "Expected ':'";
						return ERR_PARSE_ERROR;
					}
					at_key = false;
				} else {
					Error err = get_token(p_stream, token, line, r_err_str);
					if (err != OK) {
						return err;
					}

					Variant v;
					err = parse_value(token, v, p_stream, line, r_err_str, p_res_parser);
					if (err) {
						return err;
					}
					obj->set(key, v);
					need_comma = true;
					at_key = true;
				}
			}
		} else if (id == "Resource" || id == "SubResource" || id == "ExtResource") {
			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_OPEN) {
				r_err_str = "Expected '('";
				return ERR_PARSE_ERROR;
			}

			if (p_res_parser && id == "Resource" && p_res_parser->func) {
				Ref<Resource> res;
				Error err = p_res_parser->func(p_res_parser->userdata, p_stream, res, line, r_err_str);
				if (err) {
					return err;
				}

				value = res;
			} else if (p_res_parser && id == "ExtResource" && p_res_parser->ext_func) {
				Ref<Resource> res;
				Error err = p_res_parser->ext_func(p_res_parser->userdata, p_stream, res, line, r_err_str);
				if (err) {
					// If the file is missing, the error can be ignored.
					if (err != ERR_FILE_NOT_FOUND && err != ERR_CANT_OPEN && err != ERR_FILE_CANT_OPEN) {
						return err;
					}
				}

				value = res;
			} else if (p_res_parser && id == "SubResource" && p_res_parser->sub_func) {
				Ref<Resource> res;
				Error err = p_res_parser->sub_func(p_res_parser->userdata, p_stream, res, line, r_err_str);
				if (err) {
					return err;
				}

				value = res;
			} else {
				get_token(p_stream, token, line, r_err_str);
				if (token.type == TK_STRING) {
					String path = token.value;
					String uid_string;

					get_token(p_stream, token, line, r_err_str);

					if (path.begins_with("uid://")) {
						uid_string = path;
						path = "";
					}
					if (token.type == TK_COMMA) {
						get_token(p_stream, token, line, r_err_str);
						if (token.type != TK_STRING) {
							r_err_str = "Expected string in Resource reference";
							return ERR_PARSE_ERROR;
						}
						String extra_path = token.value;
						if (extra_path.begins_with("uid://")) {
							if (!uid_string.is_empty()) {
								r_err_str = "Two uid:// paths in one Resource reference";
								return ERR_PARSE_ERROR;
							}
							uid_string = extra_path;
						} else {
							if (!path.is_empty()) {
								r_err_str = "Two non-uid paths in one Resource reference";
								return ERR_PARSE_ERROR;
							}
							path = extra_path;
						}
						get_token(p_stream, token, line, r_err_str);
					}

					Ref<Resource> res;
					if (!uid_string.is_empty()) {
						ResourceUID::ID uid = ResourceUID::get_singleton()->text_to_id(uid_string);
						if (uid != ResourceUID::INVALID_ID && ResourceUID::get_singleton()->has_id(uid)) {
							const String id_path = ResourceUID::get_singleton()->get_id_path(uid);
							if (!id_path.is_empty()) {
								res = ResourceLoader::load(id_path);
							}
						}
					}
					if (res.is_null() && !path.is_empty()) {
						res = ResourceLoader::load(path);
					}
					if (res.is_null()) {
						r_err_str = "Can't load resource at path: " + path + " with uid: " + uid_string;
						return ERR_PARSE_ERROR;
					}

					if (token.type != TK_PARENTHESIS_CLOSE) {
						r_err_str = "Expected ')'";
						return ERR_PARSE_ERROR;
					}

					value = res;
				} else {
					r_err_str = "Expected string as argument for Resource()";
					return ERR_PARSE_ERROR;
				}
			}
		} else if (id == "Dictionary") {
			Error err = OK;

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_BRACKET_OPEN) {
				r_err_str = "Expected '['";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_IDENTIFIER) {
				r_err_str = "Expected type identifier for key";
				return ERR_PARSE_ERROR;
			}

			Dictionary dict;
			VariantParser::ParsedTypeResult key_result = _parse_type(token, p_stream, line, r_err_str, p_res_parser, VariantParser::TokenType::TK_COMMA);

			if (key_result.error) {
				return key_result.error;
			}

			if (!key_result.got_expected_token) {
				get_token(p_stream, token, line, r_err_str);
				if (token.type != TK_COMMA) {
					r_err_str = "Expected ',' after key type";
					return ERR_PARSE_ERROR;
				}
			}

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_IDENTIFIER) {
				r_err_str = "Expected type identifier for value";
				return ERR_PARSE_ERROR;
			}

			VariantParser::ParsedTypeResult value_result = _parse_type(token, p_stream, line, r_err_str, p_res_parser, VariantParser::TokenType::TK_BRACKET_CLOSE);

			if (value_result.error) {
				return value_result.error;
			}

			if (key_result.type.builtin_type != Variant::NIL || value_result.type.builtin_type != Variant::NIL) {
				dict.set_typed(key_result.type.builtin_type, key_result.type.class_name, key_result.type.script, value_result.type.builtin_type, value_result.type.class_name, value_result.type.script);
			}

			if (!value_result.got_expected_token) {
				get_token(p_stream, token, line, r_err_str);
				if (token.type != TK_BRACKET_CLOSE) {
					r_err_str = "Expected ']'";
					return ERR_PARSE_ERROR;
				}
			}

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_OPEN) {
				r_err_str = "Expected '('";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_CURLY_BRACKET_OPEN) {
				r_err_str = "Expected '{'";
				return ERR_PARSE_ERROR;
			}

			Dictionary values;
			err = _parse_dictionary(values, p_stream, line, r_err_str, p_res_parser);
			if (err) {
				return err;
			}

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_CLOSE) {
				r_err_str = "Expected ')'";
				return ERR_PARSE_ERROR;
			}

			dict.assign(values);

			value = dict;
		} else if (id == "Array") {
			Error err = OK;

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_BRACKET_OPEN) {
				r_err_str = "Expected '['";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_IDENTIFIER) {
				r_err_str = "Expected type identifier";
				return ERR_PARSE_ERROR;
			}

			static HashMap<String, Variant::Type> builtin_types;
			if (builtin_types.is_empty()) {
				for (int i = 1; i < Variant::VARIANT_MAX; i++) {
					builtin_types[Variant::get_type_name((Variant::Type)i)] = (Variant::Type)i;
				}
			}

			Array array = Array();
			VariantParser::ParsedTypeResult array_type = _parse_type(token, p_stream, line, r_err_str, p_res_parser, TK_BRACKET_CLOSE);

			if (array_type.error) {
				return array_type.error;
			}

			array.set_typed(array_type.type.builtin_type, array_type.type.class_name, array_type.type.script);

			if (!array_type.got_expected_token) {
				get_token(p_stream, token, line, r_err_str);
				if (token.type != TK_BRACKET_CLOSE) {
					r_err_str = "Expected ']'";
					return ERR_PARSE_ERROR;
				}
			}

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_OPEN) {
				r_err_str = "Expected '('";
				return ERR_PARSE_ERROR;
			}

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_BRACKET_OPEN) {
				r_err_str = "Expected '['";
				return ERR_PARSE_ERROR;
			}

			Array values;
			err = _parse_array(values, p_stream, line, r_err_str, p_res_parser);
			if (err) {
				return err;
			}

			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_CLOSE) {
				r_err_str = "Expected ')'";
				return ERR_PARSE_ERROR;
			}

			array.assign(values);

			value = array;
		} else if (id == "PackedByteArray" || id == "PoolByteArray" || id == "ByteArray") {
			Vector<uint8_t> args;
			Error err = _parse_byte_array(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			Vector<uint8_t> arr;
			{
				int len = args.size();
				arr.resize(len);
				uint8_t *w = arr.ptrw();
				for (int i = 0; i < len; i++) {
					w[i] = args[i];
				}
			}

			value = arr;
		} else if (id == "PackedInt32Array" || id == "PackedIntArray" || id == "PoolIntArray" || id == "IntArray") {
			Vector<int32_t> args;
			Error err = _parse_construct<int32_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			Vector<int32_t> arr;
			{
				int32_t len = args.size();
				arr.resize(len);
				int32_t *w = arr.ptrw();
				for (int32_t i = 0; i < len; i++) {
					w[i] = int32_t(args[i]);
				}
			}

			value = arr;
		} else if (id == "PackedInt64Array") {
			Vector<int64_t> args;
			Error err = _parse_construct<int64_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			Vector<int64_t> arr;
			{
				int64_t len = args.size();
				arr.resize(len);
				int64_t *w = arr.ptrw();
				for (int64_t i = 0; i < len; i++) {
					w[i] = int64_t(args[i]);
				}
			}

			value = arr;
		} else if (id == "PackedFloat32Array" || id == "PackedRealArray" || id == "PoolRealArray" || id == "FloatArray") {
			Vector<float> args;
			Error err = _parse_construct<float>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			Vector<float> arr;
			{
				int len = args.size();
				arr.resize(len);
				float *w = arr.ptrw();
				for (int i = 0; i < len; i++) {
					w[i] = args[i];
				}
			}

			value = arr;
		} else if (id == "PackedFloat64Array") {
			Vector<double> args;
			Error err = _parse_construct<double>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			Vector<double> arr;
			{
				int len = args.size();
				arr.resize(len);
				double *w = arr.ptrw();
				for (int i = 0; i < len; i++) {
					w[i] = args[i];
				}
			}

			value = arr;
		} else if (id == "PackedStringArray" || id == "PoolStringArray" || id == "StringArray") {
			get_token(p_stream, token, line, r_err_str);
			if (token.type != TK_PARENTHESIS_OPEN) {
				r_err_str = "Expected '('";
				return ERR_PARSE_ERROR;
			}

			Vector<String> cs;

			bool first = true;
			while (true) {
				if (!first) {
					get_token(p_stream, token, line, r_err_str);
					if (token.type == TK_COMMA) {
						//do none
					} else if (token.type == TK_PARENTHESIS_CLOSE) {
						break;
					} else {
						r_err_str = "Expected ',' or ')'";
						return ERR_PARSE_ERROR;
					}
				}
				get_token(p_stream, token, line, r_err_str);

				if (token.type == TK_PARENTHESIS_CLOSE) {
					break;
				} else if (token.type != TK_STRING) {
					r_err_str = "Expected string";
					return ERR_PARSE_ERROR;
				}

				first = false;
				cs.push_back(token.value);
			}

			Vector<String> arr;
			{
				int len = cs.size();
				arr.resize(len);
				String *w = arr.ptrw();
				for (int i = 0; i < len; i++) {
					w[i] = cs[i];
				}
			}

			value = arr;
		} else if (id == "PackedVector2Array" || id == "PoolVector2Array" || id == "Vector2Array") {
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			Vector<Vector2> arr;
			{
				int len = args.size() / 2;
				arr.resize(len);
				Vector2 *w = arr.ptrw();
				for (int i = 0; i < len; i++) {
					w[i] = Vector2(args[i * 2 + 0], args[i * 2 + 1]);
				}
			}

			value = arr;
		} else if (id == "PackedVector3Array" || id == "PoolVector3Array" || id == "Vector3Array") {
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			Vector<Vector3> arr;
			{
				int len = args.size() / 3;
				arr.resize(len);
				Vector3 *w = arr.ptrw();
				for (int i = 0; i < len; i++) {
					w[i] = Vector3(args[i * 3 + 0], args[i * 3 + 1], args[i * 3 + 2]);
				}
			}

			value = arr;
		} else if (id == "PackedVector4Array" || id == "PoolVector4Array" || id == "Vector4Array") {
			Vector<real_t> args;
			Error err = _parse_construct<real_t>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			Vector<Vector4> arr;
			{
				int len = args.size() / 4;
				arr.resize(len);
				Vector4 *w = arr.ptrw();
				for (int i = 0; i < len; i++) {
					w[i] = Vector4(args[i * 4 + 0], args[i * 4 + 1], args[i * 4 + 2], args[i * 4 + 3]);
				}
			}

			value = arr;
		} else if (id == "PackedColorArray" || id == "PoolColorArray" || id == "ColorArray") {
			Vector<float> args;
			Error err = _parse_construct<float>(p_stream, args, line, r_err_str);
			if (err) {
				return err;
			}

			Vector<Color> arr;
			{
				int len = args.size() / 4;
				arr.resize(len);
				Color *w = arr.ptrw();
				for (int i = 0; i < len; i++) {
					w[i] = Color(args[i * 4 + 0], args[i * 4 + 1], args[i * 4 + 2], args[i * 4 + 3]);
				}
			}

			value = arr;
		} else {
			r_err_str = vformat("Unexpected identifier '%s'", id);
			return ERR_PARSE_ERROR;
		}

		// All above branches end up here unless they had an early return.
		return OK;
	} else if (token.type == TK_NUMBER) {
		value = token.value;
		return OK;
	} else if (token.type == TK_STRING) {
		value = token.value;
		return OK;
	} else if (token.type == TK_STRING_NAME) {
		value = token.value;
		return OK;
	} else if (token.type == TK_COLOR) {
		value = token.value;
		return OK;
	} else {
		r_err_str = vformat("Expected value, got '%s'", String(tk_name[token.type]));
		return ERR_PARSE_ERROR;
	}
}

Error VariantParser::_parse_array(Array &array, Stream *p_stream, int &line, String &r_err_str, ResourceParser *p_res_parser) {
	Token token;
	bool need_comma = false;

	while (true) {
		if (p_stream->is_eof()) {
			r_err_str = "Unexpected EOF while parsing array";
			return ERR_FILE_CORRUPT;
		}

		Error err = get_token(p_stream, token, line, r_err_str);
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
		err = parse_value(token, v, p_stream, line, r_err_str, p_res_parser);
		if (err) {
			return err;
		}

		array.push_back(v);
		need_comma = true;
	}
}

Error VariantParser::_parse_dictionary(Dictionary &object, Stream *p_stream, int &line, String &r_err_str, ResourceParser *p_res_parser) {
	bool at_key = true;
	Variant key;
	Token token;
	bool need_comma = false;

	while (true) {
		if (p_stream->is_eof()) {
			r_err_str = "Unexpected EOF while parsing dictionary";
			return ERR_FILE_CORRUPT;
		}

		if (at_key) {
			Error err = get_token(p_stream, token, line, r_err_str);
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

			err = parse_value(token, key, p_stream, line, r_err_str, p_res_parser);

			if (err) {
				return err;
			}

			err = get_token(p_stream, token, line, r_err_str);

			if (err != OK) {
				return err;
			}
			if (token.type != TK_COLON) {
				r_err_str = "Expected ':'";
				return ERR_PARSE_ERROR;
			}
			at_key = false;
		} else {
			Error err = get_token(p_stream, token, line, r_err_str);
			if (err != OK) {
				return err;
			}

			Variant v;
			err = parse_value(token, v, p_stream, line, r_err_str, p_res_parser);
			if (err && err != ERR_FILE_MISSING_DEPENDENCIES) {
				return err;
			}
			object[key] = v;
			need_comma = true;
			at_key = true;
		}
	}
}

Error VariantParser::_parse_tag(Token &token, Stream *p_stream, int &line, String &r_err_str, Tag &r_tag, ResourceParser *p_res_parser, bool p_simple_tag) {
	r_tag.fields.clear();

	if (token.type != TK_BRACKET_OPEN) {
		r_err_str = "Expected '['";
		return ERR_PARSE_ERROR;
	}

	if (p_simple_tag) {
		r_tag.name = "";
		r_tag.fields.clear();
		bool escaping = false;

		if (p_stream->is_utf8()) {
			CharString cs;
			while (true) {
				char c = p_stream->get_char();
				if (p_stream->is_eof()) {
					r_err_str = "Unexpected EOF while parsing simple tag";
					return ERR_PARSE_ERROR;
				}
				if (c == ']') {
					if (escaping) {
						escaping = false;
					} else {
						break;
					}
				} else if (c == '\\') {
					escaping = true;
				} else {
					escaping = false;
				}
				cs += c;
			}
			r_tag.name.clear();
			r_tag.name.append_utf8(cs.get_data(), cs.length());
		} else {
			while (true) {
				char32_t c = p_stream->get_char();
				if (p_stream->is_eof()) {
					r_err_str = "Unexpected EOF while parsing simple tag";
					return ERR_PARSE_ERROR;
				}
				if (c == ']') {
					if (escaping) {
						escaping = false;
					} else {
						break;
					}
				} else if (c == '\\') {
					escaping = true;
				} else {
					escaping = false;
				}
				r_tag.name += c;
			}
		}

		r_tag.name = r_tag.name.strip_edges();

		return OK;
	}

	get_token(p_stream, token, line, r_err_str);

	if (token.type != TK_IDENTIFIER) {
		r_err_str = "Expected identifier (tag name)";
		return ERR_PARSE_ERROR;
	}

	r_tag.name = token.value;
	bool parsing_tag = true;

	while (true) {
		if (p_stream->is_eof()) {
			r_err_str = vformat("Unexpected EOF while parsing tag '%s'", r_tag.name);
			return ERR_FILE_CORRUPT;
		}

		get_token(p_stream, token, line, r_err_str);
		if (token.type == TK_BRACKET_CLOSE) {
			break;
		}

		if (parsing_tag && token.type == TK_PERIOD) {
			r_tag.name += "."; //support tags such as [someprop.Android] for specific platforms
			get_token(p_stream, token, line, r_err_str);
		} else if (parsing_tag && token.type == TK_COLON) {
			r_tag.name += ":"; //support tags such as [someprop.Android] for specific platforms
			get_token(p_stream, token, line, r_err_str);
		} else {
			parsing_tag = false;
		}

		if (token.type != TK_IDENTIFIER) {
			r_err_str = "Expected identifier";
			return ERR_PARSE_ERROR;
		}

		String id = token.value;

		if (parsing_tag) {
			r_tag.name += id;
			continue;
		}

		get_token(p_stream, token, line, r_err_str);
		if (token.type != TK_EQUAL) {
			r_err_str = "Expected '=' after identifier";
			return ERR_PARSE_ERROR;
		}

		get_token(p_stream, token, line, r_err_str);
		Variant value;
		Error err = parse_value(token, value, p_stream, line, r_err_str, p_res_parser);
		if (err) {
			return err;
		}

		r_tag.fields[id] = value;
	}

	return OK;
}

Error VariantParser::parse_tag(Stream *p_stream, int &line, String &r_err_str, Tag &r_tag, ResourceParser *p_res_parser, bool p_simple_tag) {
	Token token;
	get_token(p_stream, token, line, r_err_str);

	if (token.type == TK_EOF) {
		return ERR_FILE_EOF;
	}

	if (token.type != TK_BRACKET_OPEN) {
		r_err_str = "Expected '['";
		return ERR_PARSE_ERROR;
	}

	return _parse_tag(token, p_stream, line, r_err_str, r_tag, p_res_parser, p_simple_tag);
}

Error VariantParser::parse_tag_assign_eof(Stream *p_stream, int &line, String &r_err_str, Tag &r_tag, String &r_assign, Variant &r_value, ResourceParser *p_res_parser, bool p_simple_tag) {
	//assign..
	r_assign = "";
	String what;

	while (true) {
		char32_t c;
		if (p_stream->saved) {
			c = p_stream->saved;
			p_stream->saved = 0;

		} else {
			c = p_stream->get_char();
		}

		if (p_stream->is_eof()) {
			return ERR_FILE_EOF;
		}

		if (c == ';') { //comment
			while (true) {
				char32_t ch = p_stream->get_char();
				if (p_stream->is_eof()) {
					return ERR_FILE_EOF;
				}
				if (ch == '\n') {
					line++;
					break;
				}
			}
			continue;
		}

		if (c == '[' && what.length() == 0) {
			//it's a tag!
			p_stream->saved = '['; //go back one

			Error err = parse_tag(p_stream, line, r_err_str, r_tag, p_res_parser, p_simple_tag);

			return err;
		}

		if (c > 32) {
			if (c == '"') { //quoted
				p_stream->saved = '"';
				Token tk;
				Error err = get_token(p_stream, tk, line, r_err_str);
				if (err) {
					return err;
				}
				if (tk.type != TK_STRING) {
					r_err_str = "Error reading quoted string";
					return ERR_INVALID_DATA;
				}

				what = tk.value;

			} else if (c != '=') {
				what += c;
			} else {
				r_assign = what;
				Token token;
				get_token(p_stream, token, line, r_err_str);
				Error err = parse_value(token, r_value, p_stream, line, r_err_str, p_res_parser);
				return err;
			}
		} else if (c == '\n') {
			line++;
		}
	}
}

Error VariantParser::parse(Stream *p_stream, Variant &r_ret, String &r_err_str, int &r_err_line, ResourceParser *p_res_parser) {
	Token token;
	Error err = get_token(p_stream, token, r_err_line, r_err_str);
	if (err) {
		return err;
	}

	if (token.type == TK_EOF) {
		return ERR_FILE_EOF;
	}

	return parse_value(token, r_ret, p_stream, r_err_line, r_err_str, p_res_parser);
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

// These two functions serialize floats or doubles using num_scientific to ensure
// it can be read back in the same way, except collapsing -0 to 0, collapsing
// NaN values, handling old inf_neg for compatibility, and collapsing doubles
// that match their 32-bit float representation to avoid serializing garbage
// digits when the underlying float is 32-bit.
static String rtos_fix(double p_value, bool p_compat) {
	if (p_value == 0.0) {
		return "0"; // Avoid negative zero (-0) being written, which may annoy git, svn, etc. for changes when they don't exist.
	} else if (p_compat) {
		// Write old inf_neg for compatibility.
		if (std::isinf(p_value) && p_value < 0.0) {
			return "inf_neg";
		}
	}
	// Hack to avoid garbage digits when the underlying float is 32-bit.
	if ((double)(float)p_value == p_value) {
		return String::num_scientific((float)p_value);
	}
	return String::num_scientific(p_value);
}

static String encode_resource_reference(const String &path) {
	ResourceUID::ID uid = ResourceLoader::get_resource_uid(path);
	if (uid != ResourceUID::INVALID_ID) {
		return "Resource(\"" + ResourceUID::get_singleton()->id_to_text(uid) +
				"\", \"" + path.c_escape_multiline() + "\")";
	} else {
		return "Resource(\"" + path.c_escape_multiline() + "\")";
	}
}

Error VariantWriter::write(const Variant &p_variant, StoreStringFunc p_store_string_func, void *p_store_string_ud, EncodeResourceFunc p_encode_res_func, void *p_encode_res_ud, int p_recursion_count, bool p_compat) {
	switch (p_variant.get_type()) {
		case Variant::NIL: {
			p_store_string_func(p_store_string_ud, "null");
		} break;
		case Variant::BOOL: {
			p_store_string_func(p_store_string_ud, p_variant.operator bool() ? "true" : "false");
		} break;
		case Variant::INT: {
			p_store_string_func(p_store_string_ud, itos(p_variant.operator int64_t()));
		} break;
		case Variant::FLOAT: {
			const double value = p_variant.operator double();
			String s = rtos_fix(value, p_compat);
			// Append ".0" to floats to ensure they are float literals.
			if (s != "inf" && s != "-inf" && s != "nan" && !s.contains_char('.') && !s.contains_char('e') && !s.contains_char('E')) {
				s += ".0";
			}
			p_store_string_func(p_store_string_ud, s);
		} break;
		case Variant::STRING: {
			String str = p_variant;
			str = "\"" + str.c_escape_multiline() + "\"";
			p_store_string_func(p_store_string_ud, str);
		} break;

		// Math types.
		case Variant::VECTOR2: {
			Vector2 v = p_variant;
			p_store_string_func(p_store_string_ud, "Vector2(" + rtos_fix(v.x, p_compat) + ", " + rtos_fix(v.y, p_compat) + ")");
		} break;
		case Variant::VECTOR2I: {
			Vector2i v = p_variant;
			p_store_string_func(p_store_string_ud, "Vector2i(" + itos(v.x) + ", " + itos(v.y) + ")");
		} break;
		case Variant::RECT2: {
			Rect2 aabb = p_variant;
			p_store_string_func(p_store_string_ud, "Rect2(" + rtos_fix(aabb.position.x, p_compat) + ", " + rtos_fix(aabb.position.y, p_compat) + ", " + rtos_fix(aabb.size.x, p_compat) + ", " + rtos_fix(aabb.size.y, p_compat) + ")");
		} break;
		case Variant::RECT2I: {
			Rect2i aabb = p_variant;
			p_store_string_func(p_store_string_ud, "Rect2i(" + itos(aabb.position.x) + ", " + itos(aabb.position.y) + ", " + itos(aabb.size.x) + ", " + itos(aabb.size.y) + ")");
		} break;
		case Variant::VECTOR3: {
			Vector3 v = p_variant;
			p_store_string_func(p_store_string_ud, "Vector3(" + rtos_fix(v.x, p_compat) + ", " + rtos_fix(v.y, p_compat) + ", " + rtos_fix(v.z, p_compat) + ")");
		} break;
		case Variant::VECTOR3I: {
			Vector3i v = p_variant;
			p_store_string_func(p_store_string_ud, "Vector3i(" + itos(v.x) + ", " + itos(v.y) + ", " + itos(v.z) + ")");
		} break;
		case Variant::VECTOR4: {
			Vector4 v = p_variant;
			p_store_string_func(p_store_string_ud, "Vector4(" + rtos_fix(v.x, p_compat) + ", " + rtos_fix(v.y, p_compat) + ", " + rtos_fix(v.z, p_compat) + ", " + rtos_fix(v.w, p_compat) + ")");
		} break;
		case Variant::VECTOR4I: {
			Vector4i v = p_variant;
			p_store_string_func(p_store_string_ud, "Vector4i(" + itos(v.x) + ", " + itos(v.y) + ", " + itos(v.z) + ", " + itos(v.w) + ")");
		} break;
		case Variant::PLANE: {
			Plane p = p_variant;
			p_store_string_func(p_store_string_ud, "Plane(" + rtos_fix(p.normal.x, p_compat) + ", " + rtos_fix(p.normal.y, p_compat) + ", " + rtos_fix(p.normal.z, p_compat) + ", " + rtos_fix(p.d, p_compat) + ")");
		} break;
		case Variant::AABB: {
			AABB aabb = p_variant;
			p_store_string_func(p_store_string_ud, "AABB(" + rtos_fix(aabb.position.x, p_compat) + ", " + rtos_fix(aabb.position.y, p_compat) + ", " + rtos_fix(aabb.position.z, p_compat) + ", " + rtos_fix(aabb.size.x, p_compat) + ", " + rtos_fix(aabb.size.y, p_compat) + ", " + rtos_fix(aabb.size.z, p_compat) + ")");
		} break;
		case Variant::QUATERNION: {
			Quaternion quaternion = p_variant;
			p_store_string_func(p_store_string_ud, "Quaternion(" + rtos_fix(quaternion.x, p_compat) + ", " + rtos_fix(quaternion.y, p_compat) + ", " + rtos_fix(quaternion.z, p_compat) + ", " + rtos_fix(quaternion.w, p_compat) + ")");
		} break;
		case Variant::TRANSFORM2D: {
			String s = "Transform2D(";
			Transform2D m3 = p_variant;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 2; j++) {
					if (i != 0 || j != 0) {
						s += ", ";
					}
					s += rtos_fix(m3.columns[i][j], p_compat);
				}
			}

			p_store_string_func(p_store_string_ud, s + ")");
		} break;
		case Variant::BASIS: {
			String s = "Basis(";
			Basis m3 = p_variant;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					if (i != 0 || j != 0) {
						s += ", ";
					}
					s += rtos_fix(m3.rows[i][j], p_compat);
				}
			}

			p_store_string_func(p_store_string_ud, s + ")");
		} break;
		case Variant::TRANSFORM3D: {
			String s = "Transform3D(";
			Transform3D t = p_variant;
			Basis &m3 = t.basis;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					if (i != 0 || j != 0) {
						s += ", ";
					}
					s += rtos_fix(m3.rows[i][j], p_compat);
				}
			}

			s = s + ", " + rtos_fix(t.origin.x, p_compat) + ", " + rtos_fix(t.origin.y, p_compat) + ", " + rtos_fix(t.origin.z, p_compat);

			p_store_string_func(p_store_string_ud, s + ")");
		} break;
		case Variant::PROJECTION: {
			String s = "Projection(";
			Projection t = p_variant;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					if (i != 0 || j != 0) {
						s += ", ";
					}
					s += rtos_fix(t.columns[i][j], p_compat);
				}
			}

			p_store_string_func(p_store_string_ud, s + ")");
		} break;

		// Misc types.
		case Variant::COLOR: {
			Color c = p_variant;
			p_store_string_func(p_store_string_ud, "Color(" + rtos_fix(c.r, p_compat) + ", " + rtos_fix(c.g, p_compat) + ", " + rtos_fix(c.b, p_compat) + ", " + rtos_fix(c.a, p_compat) + ")");
		} break;
		case Variant::STRING_NAME: {
			String str = p_variant;
			str = "&\"" + str.c_escape() + "\"";
			p_store_string_func(p_store_string_ud, str);
		} break;
		case Variant::NODE_PATH: {
			String str = p_variant;
			str = "NodePath(\"" + str.c_escape() + "\")";
			p_store_string_func(p_store_string_ud, str);
		} break;
		case Variant::RID: {
			RID rid = p_variant;
			if (rid == RID()) {
				p_store_string_func(p_store_string_ud, "RID()");
			} else {
				p_store_string_func(p_store_string_ud, "RID(" + itos(rid.get_id()) + ")");
			}
		} break;

		// Do not really store these, but ensure that assignments are not empty.
		case Variant::SIGNAL: {
			p_store_string_func(p_store_string_ud, "Signal()");
		} break;
		case Variant::CALLABLE: {
			p_store_string_func(p_store_string_ud, "Callable()");
		} break;

		case Variant::OBJECT: {
			if (unlikely(p_recursion_count > MAX_RECURSION)) {
				ERR_PRINT("Max recursion reached");
				p_store_string_func(p_store_string_ud, "null");
				return OK;
			}
			p_recursion_count++;

			Object *obj = p_variant.get_validated_object();

			if (!obj) {
				p_store_string_func(p_store_string_ud, "null");
				break; // don't save it
			}

			Ref<Resource> res = p_variant;
			if (res.is_valid()) {
				String res_text;

				// Try external function.
				if (p_encode_res_func) {
					res_text = p_encode_res_func(p_encode_res_ud, res);
				}

				// Try path, because it's a file.
				if (res_text.is_empty() && res->get_path().is_resource_file()) {
					// External resource.
					String path = res->get_path();
					res_text = encode_resource_reference(path);
				}

				// Could come up with some sort of text.
				if (!res_text.is_empty()) {
					p_store_string_func(p_store_string_ud, res_text);
					break;
				}
			}

			//store as generic object

			p_store_string_func(p_store_string_ud, "Object(" + obj->get_class() + ",");

			List<PropertyInfo> props;
			obj->get_property_list(&props);
			bool first = true;
			for (const PropertyInfo &E : props) {
				if (E.usage & PROPERTY_USAGE_STORAGE || E.usage & PROPERTY_USAGE_SCRIPT_VARIABLE) {
					//must be serialized

					if (first) {
						first = false;
					} else {
						p_store_string_func(p_store_string_ud, ",");
					}

					p_store_string_func(p_store_string_ud, "\"" + E.name + "\":");
					write(obj->get(E.name), p_store_string_func, p_store_string_ud, p_encode_res_func, p_encode_res_ud, p_recursion_count, p_compat);
				}
			}

			p_store_string_func(p_store_string_ud, ")\n");
		} break;

		case Variant::DICTIONARY: {
			Dictionary dict = p_variant;

			if (dict.is_typed()) {
				p_store_string_func(p_store_string_ud, "Dictionary[");

				Variant::Type key_builtin_type = (Variant::Type)dict.get_typed_key_builtin();
				StringName key_class_name = dict.get_typed_key_class_name();
				Ref<Script> key_script = dict.get_typed_key_script();

				if (key_script.is_valid()) {
					String resource_text;
					if (p_encode_res_func) {
						resource_text = p_encode_res_func(p_encode_res_ud, key_script);
					}
					if (resource_text.is_empty() && key_script->get_path().is_resource_file()) {
						resource_text = encode_resource_reference(key_script->get_path());
					}

					if (!resource_text.is_empty()) {
						p_store_string_func(p_store_string_ud, resource_text);
					} else {
						ERR_PRINT("Failed to encode a path to a custom script for a dictionary key type.");
						p_store_string_func(p_store_string_ud, key_class_name);
					}
				} else if (key_class_name != StringName()) {
					p_store_string_func(p_store_string_ud, key_class_name);
				} else if (key_builtin_type == Variant::NIL) {
					p_store_string_func(p_store_string_ud, "Variant");
				} else {
					p_store_string_func(p_store_string_ud, Variant::get_type_name(key_builtin_type));
				}

				p_store_string_func(p_store_string_ud, ", ");

				Variant::Type value_builtin_type = (Variant::Type)dict.get_typed_value_builtin();
				StringName value_class_name = dict.get_typed_value_class_name();
				Ref<Script> value_script = dict.get_typed_value_script();

				if (value_script.is_valid()) {
					String resource_text;
					if (p_encode_res_func) {
						resource_text = p_encode_res_func(p_encode_res_ud, value_script);
					}
					if (resource_text.is_empty() && value_script->get_path().is_resource_file()) {
						resource_text = encode_resource_reference(value_script->get_path());
					}

					if (!resource_text.is_empty()) {
						p_store_string_func(p_store_string_ud, resource_text);
					} else {
						ERR_PRINT("Failed to encode a path to a custom script for a dictionary value type.");
						p_store_string_func(p_store_string_ud, value_class_name);
					}
				} else if (value_class_name != StringName()) {
					p_store_string_func(p_store_string_ud, value_class_name);
				} else if (value_builtin_type == Variant::NIL) {
					p_store_string_func(p_store_string_ud, "Variant");
				} else {
					p_store_string_func(p_store_string_ud, Variant::get_type_name(value_builtin_type));
				}

				p_store_string_func(p_store_string_ud, "](");
			}

			if (unlikely(p_recursion_count > MAX_RECURSION)) {
				ERR_PRINT("Max recursion reached");
				p_store_string_func(p_store_string_ud, "{}");
			} else {
				LocalVector<Variant> keys = dict.get_key_list();
				keys.sort_custom<StringLikeVariantOrder>();

				if (keys.is_empty()) {
					// Avoid unnecessary line break.
					p_store_string_func(p_store_string_ud, "{}");
				} else {
					p_recursion_count++;

					p_store_string_func(p_store_string_ud, "{\n");

					for (uint32_t i = 0; i < keys.size(); i++) {
						const Variant &key = keys[i];
						write(key, p_store_string_func, p_store_string_ud, p_encode_res_func, p_encode_res_ud, p_recursion_count, p_compat);
						p_store_string_func(p_store_string_ud, ": ");
						write(dict[key], p_store_string_func, p_store_string_ud, p_encode_res_func, p_encode_res_ud, p_recursion_count, p_compat);
						if (i + 1 < keys.size()) {
							p_store_string_func(p_store_string_ud, ",\n");
						} else {
							p_store_string_func(p_store_string_ud, "\n");
						}
					}

					p_store_string_func(p_store_string_ud, "}");
				}
			}

			if (dict.is_typed()) {
				p_store_string_func(p_store_string_ud, ")");
			}
		} break;

		case Variant::ARRAY: {
			Array array = p_variant;

			if (array.is_typed()) {
				p_store_string_func(p_store_string_ud, "Array[");

				Variant::Type builtin_type = (Variant::Type)array.get_typed_builtin();
				StringName class_name = array.get_typed_class_name();
				Ref<Script> script = array.get_typed_script();

				if (script.is_valid()) {
					String resource_text = String();
					if (p_encode_res_func) {
						resource_text = p_encode_res_func(p_encode_res_ud, script);
					}
					if (resource_text.is_empty() && script->get_path().is_resource_file()) {
						resource_text = encode_resource_reference(script->get_path());
					}

					if (!resource_text.is_empty()) {
						p_store_string_func(p_store_string_ud, resource_text);
					} else {
						ERR_PRINT("Failed to encode a path to a custom script for an array type.");
						p_store_string_func(p_store_string_ud, class_name);
					}
				} else if (class_name != StringName()) {
					p_store_string_func(p_store_string_ud, class_name);
				} else {
					p_store_string_func(p_store_string_ud, Variant::get_type_name(builtin_type));
				}

				p_store_string_func(p_store_string_ud, "](");
			}

			if (unlikely(p_recursion_count > MAX_RECURSION)) {
				ERR_PRINT("Max recursion reached");
				p_store_string_func(p_store_string_ud, "[]");
			} else {
				p_recursion_count++;

				p_store_string_func(p_store_string_ud, "[");

				bool first = true;
				for (const Variant &var : array) {
					if (first) {
						first = false;
					} else {
						p_store_string_func(p_store_string_ud, ", ");
					}
					write(var, p_store_string_func, p_store_string_ud, p_encode_res_func, p_encode_res_ud, p_recursion_count, p_compat);
				}

				p_store_string_func(p_store_string_ud, "]");
			}

			if (array.is_typed()) {
				p_store_string_func(p_store_string_ud, ")");
			}
		} break;

		case Variant::PACKED_BYTE_ARRAY: {
			p_store_string_func(p_store_string_ud, "PackedByteArray(");
			Vector<uint8_t> data = p_variant;
			if (p_compat) {
				int len = data.size();
				const uint8_t *ptr = data.ptr();
				for (int i = 0; i < len; i++) {
					if (i > 0) {
						p_store_string_func(p_store_string_ud, ", ");
					}
					p_store_string_func(p_store_string_ud, itos(ptr[i]));
				}
			} else if (data.size() > 0) {
				p_store_string_func(p_store_string_ud, "\"");
				p_store_string_func(p_store_string_ud, CryptoCore::b64_encode_str(data.ptr(), data.size()));
				p_store_string_func(p_store_string_ud, "\"");
			}
			p_store_string_func(p_store_string_ud, ")");
		} break;
		case Variant::PACKED_INT32_ARRAY: {
			p_store_string_func(p_store_string_ud, "PackedInt32Array(");
			Vector<int32_t> data = p_variant;
			int32_t len = data.size();
			const int32_t *ptr = data.ptr();

			for (int32_t i = 0; i < len; i++) {
				if (i > 0) {
					p_store_string_func(p_store_string_ud, ", ");
				}

				p_store_string_func(p_store_string_ud, itos(ptr[i]));
			}

			p_store_string_func(p_store_string_ud, ")");
		} break;
		case Variant::PACKED_INT64_ARRAY: {
			p_store_string_func(p_store_string_ud, "PackedInt64Array(");
			Vector<int64_t> data = p_variant;
			int64_t len = data.size();
			const int64_t *ptr = data.ptr();

			for (int64_t i = 0; i < len; i++) {
				if (i > 0) {
					p_store_string_func(p_store_string_ud, ", ");
				}

				p_store_string_func(p_store_string_ud, itos(ptr[i]));
			}

			p_store_string_func(p_store_string_ud, ")");
		} break;
		case Variant::PACKED_FLOAT32_ARRAY: {
			p_store_string_func(p_store_string_ud, "PackedFloat32Array(");
			Vector<float> data = p_variant;
			int len = data.size();
			const float *ptr = data.ptr();

			for (int i = 0; i < len; i++) {
				if (i > 0) {
					p_store_string_func(p_store_string_ud, ", ");
				}
				p_store_string_func(p_store_string_ud, rtos_fix(ptr[i], p_compat));
			}

			p_store_string_func(p_store_string_ud, ")");
		} break;
		case Variant::PACKED_FLOAT64_ARRAY: {
			p_store_string_func(p_store_string_ud, "PackedFloat64Array(");
			Vector<double> data = p_variant;
			int len = data.size();
			const double *ptr = data.ptr();

			for (int i = 0; i < len; i++) {
				if (i > 0) {
					p_store_string_func(p_store_string_ud, ", ");
				}
				p_store_string_func(p_store_string_ud, rtos_fix(ptr[i], p_compat));
			}

			p_store_string_func(p_store_string_ud, ")");
		} break;
		case Variant::PACKED_STRING_ARRAY: {
			p_store_string_func(p_store_string_ud, "PackedStringArray(");
			Vector<String> data = p_variant;
			int len = data.size();
			const String *ptr = data.ptr();

			for (int i = 0; i < len; i++) {
				if (i > 0) {
					p_store_string_func(p_store_string_ud, ", ");
				}
				p_store_string_func(p_store_string_ud, "\"" + ptr[i].c_escape() + "\"");
			}

			p_store_string_func(p_store_string_ud, ")");
		} break;
		case Variant::PACKED_VECTOR2_ARRAY: {
			p_store_string_func(p_store_string_ud, "PackedVector2Array(");
			Vector<Vector2> data = p_variant;
			int len = data.size();
			const Vector2 *ptr = data.ptr();

			for (int i = 0; i < len; i++) {
				if (i > 0) {
					p_store_string_func(p_store_string_ud, ", ");
				}
				p_store_string_func(p_store_string_ud, rtos_fix(ptr[i].x, p_compat) + ", " + rtos_fix(ptr[i].y, p_compat));
			}

			p_store_string_func(p_store_string_ud, ")");
		} break;
		case Variant::PACKED_VECTOR3_ARRAY: {
			p_store_string_func(p_store_string_ud, "PackedVector3Array(");
			Vector<Vector3> data = p_variant;
			int len = data.size();
			const Vector3 *ptr = data.ptr();

			for (int i = 0; i < len; i++) {
				if (i > 0) {
					p_store_string_func(p_store_string_ud, ", ");
				}
				p_store_string_func(p_store_string_ud, rtos_fix(ptr[i].x, p_compat) + ", " + rtos_fix(ptr[i].y, p_compat) + ", " + rtos_fix(ptr[i].z, p_compat));
			}

			p_store_string_func(p_store_string_ud, ")");
		} break;
		case Variant::PACKED_COLOR_ARRAY: {
			p_store_string_func(p_store_string_ud, "PackedColorArray(");
			Vector<Color> data = p_variant;
			int len = data.size();
			const Color *ptr = data.ptr();

			for (int i = 0; i < len; i++) {
				if (i > 0) {
					p_store_string_func(p_store_string_ud, ", ");
				}
				p_store_string_func(p_store_string_ud, rtos_fix(ptr[i].r, p_compat) + ", " + rtos_fix(ptr[i].g, p_compat) + ", " + rtos_fix(ptr[i].b, p_compat) + ", " + rtos_fix(ptr[i].a, p_compat));
			}

			p_store_string_func(p_store_string_ud, ")");
		} break;
		case Variant::PACKED_VECTOR4_ARRAY: {
			p_store_string_func(p_store_string_ud, "PackedVector4Array(");
			Vector<Vector4> data = p_variant;
			int len = data.size();
			const Vector4 *ptr = data.ptr();

			for (int i = 0; i < len; i++) {
				if (i > 0) {
					p_store_string_func(p_store_string_ud, ", ");
				}
				p_store_string_func(p_store_string_ud, rtos_fix(ptr[i].x, p_compat) + ", " + rtos_fix(ptr[i].y, p_compat) + ", " + rtos_fix(ptr[i].z, p_compat) + ", " + rtos_fix(ptr[i].w, p_compat));
			}

			p_store_string_func(p_store_string_ud, ")");
		} break;

		default: {
			ERR_PRINT("Unknown variant type");
			return ERR_BUG;
		}
	}

	return OK;
}

static Error _write_to_str(void *ud, const String &p_string) {
	String *str = (String *)ud;
	(*str) += p_string;
	return OK;
}

Error VariantWriter::write_to_string(const Variant &p_variant, String &r_string, EncodeResourceFunc p_encode_res_func, void *p_encode_res_ud, bool p_compat) {
	r_string = String();

	return write(p_variant, _write_to_str, &r_string, p_encode_res_func, p_encode_res_ud, 0, p_compat);
}
