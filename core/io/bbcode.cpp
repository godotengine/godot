/**************************************************************************/
/*  bbcode.cpp                                                            */
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

#include "bbcode.h"

String BBCodeToken::get_normalized_bbcode() const {
	switch (type) {
		case TOKEN_TYPE_TEXT:
			return value;
		case TOKEN_TYPE_OPEN_TAG: {
			PackedStringArray contents;
			contents.append(value);
			for (const KeyValue<Variant, Variant> &E : parameters) {
				const String &param_name = static_cast<String>(E.key);
				const Variant &param_value = E.value;

				String param;
				if (param_value.is_null()) {
					param = param_name;
				} else {
					param = vformat("%s=\"%s\"", param_name, param_value.operator String().replace("\"", "\\\""));
				}
				contents.append(param);
			}
			return vformat("[%s]", String(" ").join(contents));
		}
		case TOKEN_TYPE_CLOSE_TAG:
			return vformat("[/%s]", value);
		default:
			ERR_FAIL_V(String());
	}
}

void BBCodeToken::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_parsed_bbcode"), &BBCodeToken::get_parsed_bbcode);
	ClassDB::bind_method(D_METHOD("get_normalized_bbcode"), &BBCodeToken::get_normalized_bbcode);
	ClassDB::bind_method(D_METHOD("get_position_start"), &BBCodeToken::get_position_start);
	ClassDB::bind_method(D_METHOD("get_position_end"), &BBCodeToken::get_position_end);
	ClassDB::bind_method(D_METHOD("get_token_type"), &BBCodeToken::get_type);
	ClassDB::bind_method(D_METHOD("get_value"), &BBCodeToken::get_value);
	ClassDB::bind_method(D_METHOD("get_parameters"), &BBCodeToken::get_parameters);

	BIND_ENUM_CONSTANT(TOKEN_TYPE_TEXT);
	BIND_ENUM_CONSTANT(TOKEN_TYPE_OPEN_TAG);
	BIND_ENUM_CONSTANT(TOKEN_TYPE_CLOSE_TAG);
}

Dictionary BBCodeParser::validate_tag(const String &p_tag, const Dictionary &p_parameters) {
	Dictionary result;
	if (!GDVIRTUAL_CALL(_validate_tag, p_tag, p_parameters, result)) {
		result = _validate_tag(p_tag, p_parameters);
	}
	return result;
}

Error BBCodeParser::validate_text(const String &p_text) {
	Error result;
	if (!GDVIRTUAL_CALL(_validate_text, p_text, result)) {
		result = _validate_text(p_text);
	}
	return result;
}

void BBCodeParser::_parse_tag(const String &p_bbcode, const int p_start, String &r_tag, Dictionary &r_parameters, int &r_end) {
	enum State {
		TAG_NAME,
		PARAM_NAME,
		PARAM_VALUE,
	};
	enum Quotes {
		NONE,
		SINGLE_QUOTE,
		DOUBLE_QUOTE,
	};

	String buffer;
	String parameter;
	State state = State::TAG_NAME;
	Quotes quotes = Quotes::NONE;
	bool is_handling_backslashes = backslash_escape_quotes || (escape_brackets & ESCAPE_BRACKETS_BACKSLASH);

	int pos;
	int len = p_bbcode.length();
	for (pos = p_start; pos < len; pos++) {
		const char32_t cchar = p_bbcode[pos];

		if (is_handling_backslashes) {
			if (cchar == '\\') {
				if (pos + 1 >= len) {
					// Can't end a string with an unmatched escape.
					error = ERR_PARSE_ERROR;
					continue;
				}
				const char32_t peek = p_bbcode[pos + 1];
				if (peek == '\\' ||
						((peek == '[' || peek == ']') && (escape_brackets & ESCAPE_BRACKETS_BACKSLASH)) ||
						((peek == '"' || peek == '\'') && backslash_escape_quotes)) {
					// Insert character literally.
					buffer += peek;
					pos++;
					continue;
				}
			}
		}

		switch (state) {
			case State::TAG_NAME:
				switch (cchar) {
					case ' ':
						if (buffer.is_empty()) {
							continue;
						}
						if (buffer[0] == '/') {
							// Closing tag can't have parameters.
							_update_error(ERR_PARSE_ERROR);
							goto END;
						}
						r_tag = buffer.strip_edges();
						buffer.clear();
						state = State::PARAM_NAME;
						break;
					case '=':
						if (buffer.is_empty()) {
							continue;
						}
						if (buffer[0] == '/') {
							// Closing tag can't have parameters.
							_update_error(ERR_PARSE_ERROR);
							goto END;
						}
						r_tag = buffer.strip_edges();
						buffer.clear();
						state = State::PARAM_NAME;
						break;
					case ']':
						r_tag = buffer.strip_edges();
						goto END;
					default:
						buffer += cchar;
						break;
				}
				break;
			case State::PARAM_NAME:
				switch (cchar) {
					case ' ':
						if (buffer.is_empty()) {
							continue;
						}
						r_parameters.set(buffer.strip_edges(), Variant());
						buffer.clear();
						break;
					case '=':
						if (buffer.is_empty()) {
							continue;
						}
						parameter = buffer.strip_edges();
						buffer.clear();
						state = State::PARAM_VALUE;
						break;
					case ']':
						if (!buffer.is_empty()) {
							r_parameters.set(buffer.strip_edges(), Variant());
						}
						goto END;
					default:
						buffer += cchar;
						break;
				}
				break;
			case State::PARAM_VALUE:
				switch (quotes) {
					case Quotes::NONE:
						switch (cchar) {
							case ' ':
								if (buffer.is_empty()) {
									continue;
								}
								r_parameters.set(parameter, buffer.strip_edges());
								parameter.clear();
								buffer.clear();
								state = State::PARAM_NAME;
								break;
							case '\'':
								if (!buffer.is_empty()) {
									buffer += cchar;
									break;
								}
								quotes = Quotes::SINGLE_QUOTE;
								break;
							case '"':
								if (!buffer.is_empty()) {
									buffer += cchar;
									break;
								}
								quotes = Quotes::DOUBLE_QUOTE;
								break;
							case ']':
								if (!buffer.is_empty()) {
									r_parameters.set(parameter, buffer.strip_edges());
								}
								goto END;
							default:
								buffer += cchar;
								break;
						}
						break;
					case Quotes::SINGLE_QUOTE:
						switch (cchar) {
							case '\'':
								// When using quotes, do not strip spaces and allow an empty string value.
								quotes = Quotes::NONE;
								r_parameters.set(parameter, buffer);
								parameter.clear();
								buffer.clear();
								state = State::PARAM_NAME;
								break;
							default:
								buffer += cchar;
								break;
						}
						break;
					case Quotes::DOUBLE_QUOTE:
						switch (cchar) {
							case '"':
								// When using quotes, do not strip spaces and allow an empty string value.
								quotes = Quotes::NONE;
								r_parameters.set(parameter, buffer);
								parameter.clear();
								buffer.clear();
								state = State::PARAM_NAME;
								break;
							default:
								buffer += cchar;
								break;
						}
						break;
				}
				break;
		}
	}

	// Missing closing bracket.
	_update_error(ERR_PARSE_ERROR);

END:
	r_end = pos;
}

void BBCodeParser::_push_token(const String &p_parsed_bbcode, BBCodeToken *p_token) {
	p_token->set_position_start(position);
	if (p_parsed_bbcode.is_empty()) {
		p_token->set_parsed_bbcode(p_token->get_normalized_bbcode());
	} else {
		p_token->set_parsed_bbcode(p_parsed_bbcode);
	}
	position += p_token->get_parsed_bbcode().length();
	tokens.append(p_token);
}

TypedArray<BBCodeToken> BBCodeParser::get_items() const {
	TypedArray<BBCodeToken> arr;
	int size = tokens.size();
	arr.resize(size);
	for (int i = 0; i < size; i++) {
		arr[i] = tokens[i];
	}
	return arr;
}

void BBCodeParser::clear() {
	for (BBCodeToken *token : tokens) {
		memdelete(token);
	}
	tokens.clear();
	position = 0;
}

void BBCodeParser::push_bbcode(const String &p_bbcode) {
	String bbcode = p_bbcode;
	int pos = 0;
	int escape_pos = -1;
	while (pos <= bbcode.length()) {
		// TODO: we are manipulating the bbcode string when escaping brackets, which desyncs token position start/end info...
		// We must keep track of our modifications, such that we can refer back to the unmodified p_bbcode for indexing
		// and passing it along to _parsed_* methods 1st parameter.

		if (escape_brackets & ESCAPE_BRACKETS_BACKSLASH) {
			if (pos < bbcode.length()) {
				int escape_char_pos = bbcode.find_char('\\', escape_pos + 1);
				if (escape_char_pos >= 0) {
					const char32_t peek = bbcode[escape_char_pos + 1];
					switch (peek) {
						case '[':
							// brk_pos should skip over the escaped opening bracket.
							escape_pos = escape_char_pos;
							[[fallthrough]];
						case ']':
						case '\\':
							// Remove the leading backslash.
							bbcode.remove_at(escape_char_pos);
							break;
					}
					if (escape_pos >= 0) {
						continue;
					}
				}
			}
		}

		int brk_pos = bbcode.find_char('[', escape_pos < 0 ? pos : (escape_pos + 1));
		escape_pos = -1;

		if (brk_pos < 0) {
			brk_pos = bbcode.length();
		}

		String txt = brk_pos > pos ? bbcode.substr(pos, brk_pos - pos) : "";

		if (!txt.is_empty()) {
			push_text(txt);
		}

		if (brk_pos == bbcode.length()) {
			break; // Nothing else to add.
		}

		if (escape_brackets & ESCAPE_BRACKETS_WRAPPED) {
			String escape = bbcode.substr(brk_pos, 3);
			if (escape == "[[]") {
				push_text("[");
				pos = brk_pos + 3;
				continue;
			} else if (escape == "[]]") {
				push_text("]");
				pos = brk_pos + 3;
				continue;
			}
		}

		String tag;
		Dictionary parameters;
		_parse_tag(bbcode, brk_pos + 1, tag, parameters, pos);
		// Move after the closing brace.
		pos++;

		bool is_closing_tag = !tag.is_empty() && tag[0] == '/';
		tag = is_closing_tag ? tag.substr(1) : tag;

		if (escape_brackets & ESCAPE_BRACKETS_ABBREVIATION) {
			if (tag == "lb") {
				push_text("[");
				continue;
			} else if (tag == "rb") {
				push_text("]");
				continue;
			}
		}

		if (escape_contents) {
			DEV_ASSERT(!tag_stack.is_empty());

			// Wait for closing tag to stop escaping.
			if (tag_stack[tag_stack.size() - 1] == tag) {
				escape_contents = false;
			} else {
				push_text(bbcode.substr(brk_pos, pos - brk_pos));
				continue;
			}
		}

		if (is_closing_tag) {
			_parsed_pop_tag(bbcode.substr(brk_pos, pos - brk_pos), tag);
		} else {
			_parsed_push_tag(bbcode.substr(brk_pos, pos - brk_pos), tag, parameters);
		}
	}
}

void BBCodeParser::_parsed_push_text(const String &p_parsed_bbcode, const String &p_text) {
	_update_error(validate_text(p_text));
	if (_is_ok() || error_handling == ERROR_HANDLING_PARSE_AS_TEXT) {
		BBCodeToken *token = memnew(BBCodeToken);
		token->set_type(BBCodeToken::TOKEN_TYPE_TEXT);
		token->set_value(p_text);
		_push_token(p_parsed_bbcode, token);
	}
}

void BBCodeParser::_parsed_push_tag(const String &p_parsed_bbcode, const String &p_tag, const Dictionary &p_parameters) {
	Dictionary result = validate_tag(p_tag, p_parameters);

	Variant error_var = result.get("error", OK);
	ERR_FAIL_COND(error_var.get_type() != Variant::Type::INT);
	_update_error(static_cast<Error>(error_var));

	if (_is_ok()) {
		BBCodeToken *token = memnew(BBCodeToken);
		token->set_type(BBCodeToken::TOKEN_TYPE_OPEN_TAG);
		token->set_value(p_tag);
		token->set_parameters(p_parameters);
		_push_token(p_parsed_bbcode, token);

		Variant self_closing_var = result.get("self_closing", false);
		ERR_FAIL_COND(self_closing_var.get_type() != Variant::Type::BOOL);
		if (!static_cast<bool>(self_closing_var)) {
			tag_stack.append(p_tag);
		}

		Variant escape_var = result.get("escape_contents", false);
		ERR_FAIL_COND(escape_var.get_type() != Variant::Type::BOOL);
		escape_contents = static_cast<bool>(escape_var);
	} else if (error_handling == ERROR_HANDLING_PARSE_AS_TEXT) {
		_parsed_push_text(p_parsed_bbcode, p_parsed_bbcode);
	}
}

void BBCodeParser::_parsed_pop_tag(const String &p_parsed_bbcode, const String &p_tag) {
	if (!tag_stack.is_empty()) {
		const String &top_tag = tag_stack[tag_stack.size() - 1];
		if (top_tag != p_tag) {
			// Mismatching top tag.
			_update_error(Error::ERR_INVALID_DATA);
			return;
		}
		tag_stack.remove_at(tag_stack.size() - 1);
	}

	BBCodeToken *token = memnew(BBCodeToken);
	token->set_type(BBCodeToken::TOKEN_TYPE_CLOSE_TAG);
	token->set_value(p_tag);
	_push_token(p_parsed_bbcode, token);
}

void BBCodeParser::push_text(const String &p_text) {
	_parsed_push_text("", p_text);
}

void BBCodeParser::push_tag(const String &p_tag, const Dictionary &p_parameters) {
	_parsed_push_tag("", p_tag, p_parameters);
}

void BBCodeParser::pop_tag(const String &p_tag) {
	_parsed_pop_tag("", p_tag);
}

void BBCodeParser::_bind_methods() {
	ClassDB::bind_method(D_METHOD("validate_tag", "tag", "parameters"), &BBCodeParser::validate_tag);
	ClassDB::bind_method(D_METHOD("validate_text", "text"), &BBCodeParser::validate_text);

	ClassDB::bind_method(D_METHOD("get_error"), &BBCodeParser::get_error);
	ClassDB::bind_method(D_METHOD("get_items"), &BBCodeParser::get_items);

	ClassDB::bind_method(D_METHOD("get_escape_brackets"), &BBCodeParser::get_escape_brackets);
	ClassDB::bind_method(D_METHOD("set_escape_brackets", "value"), &BBCodeParser::set_escape_brackets);
	ClassDB::bind_method(D_METHOD("get_backslash_escape_quotes"), &BBCodeParser::get_backslash_escape_quotes);
	ClassDB::bind_method(D_METHOD("set_backslash_escape_quotes", "value"), &BBCodeParser::set_backslash_escape_quotes);
	ClassDB::bind_method(D_METHOD("get_error_handling"), &BBCodeParser::get_error_handling);
	ClassDB::bind_method(D_METHOD("set_error_handling", "error_handling"), &BBCodeParser::set_error_handling);

	ClassDB::bind_method(D_METHOD("clear"), &BBCodeParser::clear);
	ClassDB::bind_method(D_METHOD("push_bbcode", "bbcode"), &BBCodeParser::push_bbcode);
	ClassDB::bind_method(D_METHOD("push_text", "text"), &BBCodeParser::push_text);
	ClassDB::bind_method(D_METHOD("push_tag", "tag", "parameters"), &BBCodeParser::push_tag);
	ClassDB::bind_method(D_METHOD("pop_tag", "tag"), &BBCodeParser::pop_tag);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "escape_brackets", PROPERTY_HINT_FLAGS, "Double Brackets,Wrapped Brackets,Backslash,Abbreviation"), "set_escape_brackets", "get_escape_brackets");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "backslash_escape_quotes"), "set_backslash_escape_quotes", "get_backslash_escape_quotes");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "error_handling", PROPERTY_HINT_ENUM, "Remove,Parse as Text,Keep"), "set_error_handling", "get_error_handling");

	GDVIRTUAL_BIND(_validate_tag, "tag", "parameters")
	GDVIRTUAL_BIND(_validate_text, "text")

	BIND_BITFIELD_FLAG(ESCAPE_BRACKETS_NONE);
	BIND_BITFIELD_FLAG(ESCAPE_BRACKETS_WRAPPED);
	BIND_BITFIELD_FLAG(ESCAPE_BRACKETS_BACKSLASH);
	BIND_BITFIELD_FLAG(ESCAPE_BRACKETS_ABBREVIATION);

	BIND_ENUM_CONSTANT(ERROR_HANDLING_REMOVE);
	BIND_ENUM_CONSTANT(ERROR_HANDLING_PARSE_AS_TEXT);
	BIND_ENUM_CONSTANT(ERROR_HANDLING_KEEP);
}

BBCodeParser::~BBCodeParser() {
	clear();
}
