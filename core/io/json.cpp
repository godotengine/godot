/**************************************************************************/
/*  json.cpp                                                              */
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

#include "json.h"

#include "core/config/engine.h"
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

String JSON::_make_indent(const String &p_indent, int p_size) {
	return p_indent.repeat(p_size);
}

String JSON::_stringify(const Variant &p_var, const String &p_indent, int p_cur_indent, bool p_sort_keys, HashSet<const void *> &p_markers, bool p_full_precision) {
	ERR_FAIL_COND_V_MSG(p_cur_indent > Variant::MAX_RECURSION_DEPTH, "...", "JSON structure is too deep. Bailing.");

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
		case Variant::FLOAT: {
			double num = p_var;
			if (p_full_precision) {
				// Store unreliable digits (17) instead of just reliable
				// digits (14) so that the value can be decoded exactly.
				return String::num(num, 17 - (int)floor(log10(num)));
			} else {
				// Store only reliable digits (14) by default.
				return String::num(num, 14 - (int)floor(log10(num)));
			}
		}
		case Variant::PACKED_INT32_ARRAY:
		case Variant::PACKED_INT64_ARRAY:
		case Variant::PACKED_FLOAT32_ARRAY:
		case Variant::PACKED_FLOAT64_ARRAY:
		case Variant::PACKED_STRING_ARRAY:
		case Variant::ARRAY: {
			Array a = p_var;
			if (a.is_empty()) {
				return "[]";
			}
			String s = "[";
			s += end_statement;

			ERR_FAIL_COND_V_MSG(p_markers.has(a.id()), "\"[...]\"", "Converting circular structure to JSON.");
			p_markers.insert(a.id());

			bool first = true;
			for (const Variant &var : a) {
				if (first) {
					first = false;
				} else {
					s += ",";
					s += end_statement;
				}
				s += _make_indent(p_indent, p_cur_indent + 1) + _stringify(var, p_indent, p_cur_indent + 1, p_sort_keys, p_markers);
			}
			s += end_statement + _make_indent(p_indent, p_cur_indent) + "]";
			p_markers.erase(a.id());
			return s;
		}
		case Variant::DICTIONARY: {
			String s = "{";
			s += end_statement;
			Dictionary d = p_var;

			ERR_FAIL_COND_V_MSG(p_markers.has(d.id()), "\"{...}\"", "Converting circular structure to JSON.");
			p_markers.insert(d.id());

			List<Variant> keys;
			d.get_key_list(&keys);

			if (p_sort_keys) {
				keys.sort();
			}

			bool first_key = true;
			for (const Variant &E : keys) {
				if (first_key) {
					first_key = false;
				} else {
					s += ",";
					s += end_statement;
				}
				s += _make_indent(p_indent, p_cur_indent + 1) + _stringify(String(E), p_indent, p_cur_indent + 1, p_sort_keys, p_markers);
				s += colon;
				s += _stringify(d[E], p_indent, p_cur_indent + 1, p_sort_keys, p_markers);
			}

			s += end_statement + _make_indent(p_indent, p_cur_indent) + "}";
			p_markers.erase(d.id());
			return s;
		}
		default:
			return "\"" + String(p_var).json_escape() + "\"";
	}
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
									if (!is_hex_digit(c)) {
										r_err_str = "Malformed hex constant in string";
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
										if (!is_hex_digit(c)) {
											r_err_str = "Malformed hex constant in string";
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
							case '"':
							case '\\':
							case '/': {
								res = next;
							} break;
							default: {
								r_err_str = "Invalid escape sequence.";
								return ERR_PARSE_ERROR;
							}
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

				if (p_str[index] == '-' || is_digit(p_str[index])) {
					//a number
					const char32_t *rptr;
					double number = String::to_float(&p_str[index], &rptr);
					index += (rptr - &p_str[index]);
					r_token.type = TK_NUMBER;
					r_token.value = number;
					return OK;

				} else if (is_ascii_alphabet_char(p_str[index])) {
					String id;

					while (is_ascii_alphabet_char(p_str[index])) {
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

Error JSON::_parse_value(Variant &value, Token &token, const char32_t *p_str, int &index, int p_len, int &line, int p_depth, String &r_err_str) {
	if (p_depth > Variant::MAX_RECURSION_DEPTH) {
		r_err_str = "JSON structure is too deep. Bailing.";
		return ERR_OUT_OF_MEMORY;
	}

	if (token.type == TK_CURLY_BRACKET_OPEN) {
		Dictionary d;
		Error err = _parse_object(d, p_str, index, p_len, line, p_depth + 1, r_err_str);
		if (err) {
			return err;
		}
		value = d;
	} else if (token.type == TK_BRACKET_OPEN) {
		Array a;
		Error err = _parse_array(a, p_str, index, p_len, line, p_depth + 1, r_err_str);
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

Error JSON::_parse_array(Array &array, const char32_t *p_str, int &index, int p_len, int &line, int p_depth, String &r_err_str) {
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
		err = _parse_value(v, token, p_str, index, p_len, line, p_depth, r_err_str);
		if (err) {
			return err;
		}

		array.push_back(v);
		need_comma = true;
	}

	r_err_str = "Expected ']'";
	return ERR_PARSE_ERROR;
}

Error JSON::_parse_object(Dictionary &object, const char32_t *p_str, int &index, int p_len, int &line, int p_depth, String &r_err_str) {
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
			err = _parse_value(v, token, p_str, index, p_len, line, p_depth, r_err_str);
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

void JSON::set_data(const Variant &p_data) {
	data = p_data;
	text.clear();
}

Error JSON::_parse_string(const String &p_json, Variant &r_ret, String &r_err_str, int &r_err_line) {
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

	err = _parse_value(r_ret, token, str, idx, len, r_err_line, 0, r_err_str);

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

Error JSON::parse(const String &p_json_string, bool p_keep_text) {
	Error err = _parse_string(p_json_string, data, err_str, err_line);
	if (err == Error::OK) {
		err_line = 0;
	}
	if (p_keep_text) {
		text = p_json_string;
	}
	return err;
}

String JSON::get_parsed_text() const {
	return text;
}

String JSON::stringify(const Variant &p_var, const String &p_indent, bool p_sort_keys, bool p_full_precision) {
	Ref<JSON> jason;
	jason.instantiate();
	HashSet<const void *> markers;
	return jason->_stringify(p_var, p_indent, 0, p_sort_keys, markers, p_full_precision);
}

Variant JSON::parse_string(const String &p_json_string) {
	Ref<JSON> jason;
	jason.instantiate();
	Error error = jason->parse(p_json_string);
	ERR_FAIL_COND_V_MSG(error != Error::OK, Variant(), vformat("Parse JSON failed. Error at line %d: %s", jason->get_error_line(), jason->get_error_message()));
	return jason->get_data();
}

void JSON::_bind_methods() {
	ClassDB::bind_static_method("JSON", D_METHOD("stringify", "data", "indent", "sort_keys", "full_precision"), &JSON::stringify, DEFVAL(""), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_static_method("JSON", D_METHOD("parse_string", "json_string"), &JSON::parse_string);
	ClassDB::bind_method(D_METHOD("parse", "json_text", "keep_text"), &JSON::parse, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("get_data"), &JSON::get_data);
	ClassDB::bind_method(D_METHOD("set_data", "data"), &JSON::set_data);
	ClassDB::bind_method(D_METHOD("get_parsed_text"), &JSON::get_parsed_text);
	ClassDB::bind_method(D_METHOD("get_error_line"), &JSON::get_error_line);
	ClassDB::bind_method(D_METHOD("get_error_message"), &JSON::get_error_message);

	ADD_PROPERTY(PropertyInfo(Variant::NIL, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT), "set_data", "get_data"); // Ensures that it can be serialized as binary.
}

////

////////////

Ref<Resource> ResourceFormatLoaderJSON::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	if (!FileAccess::exists(p_path)) {
		*r_error = ERR_FILE_NOT_FOUND;
		return Ref<Resource>();
	}

	Ref<JSON> json;
	json.instantiate();

	Error err = json->parse(FileAccess::get_file_as_string(p_path), Engine::get_singleton()->is_editor_hint());
	if (err != OK) {
		String err_text = "Error parsing JSON file at '" + p_path + "', on line " + itos(json->get_error_line()) + ": " + json->get_error_message();

		if (Engine::get_singleton()->is_editor_hint()) {
			// If running on editor, still allow opening the JSON so the code editor can edit it.
			WARN_PRINT(err_text);
		} else {
			if (r_error) {
				*r_error = err;
			}
			ERR_PRINT(err_text);
			return Ref<Resource>();
		}
	}

	if (r_error) {
		*r_error = OK;
	}

	return json;
}

void ResourceFormatLoaderJSON::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("json");
}

bool ResourceFormatLoaderJSON::handles_type(const String &p_type) const {
	return (p_type == "JSON");
}

String ResourceFormatLoaderJSON::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "json") {
		return "JSON";
	}
	return "";
}

Error ResourceFormatSaverJSON::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	Ref<JSON> json = p_resource;
	ERR_FAIL_COND_V(json.is_null(), ERR_INVALID_PARAMETER);

	String source = json->get_parsed_text().is_empty() ? JSON::stringify(json->get_data(), "\t", false, true) : json->get_parsed_text();

	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);

	ERR_FAIL_COND_V_MSG(err, err, "Cannot save json '" + p_path + "'.");

	file->store_string(source);
	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		return ERR_CANT_CREATE;
	}

	return OK;
}

void ResourceFormatSaverJSON::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	Ref<JSON> json = p_resource;
	if (json.is_valid()) {
		p_extensions->push_back("json");
	}
}

bool ResourceFormatSaverJSON::recognize(const Ref<Resource> &p_resource) const {
	return p_resource->get_class_name() == "JSON"; //only json, not inherited
}
