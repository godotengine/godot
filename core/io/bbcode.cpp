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

void BBCodeToken::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_type"), &BBCodeToken::get_type);
	ClassDB::bind_method(D_METHOD("set_type", "type"), &BBCodeToken::set_type);
	ClassDB::bind_method(D_METHOD("get_value"), &BBCodeToken::get_value);
	ClassDB::bind_method(D_METHOD("set_value", "value"), &BBCodeToken::set_value);
	ClassDB::bind_method(D_METHOD("get_parameters"), &BBCodeToken::get_parameters);
	ClassDB::bind_method(D_METHOD("set_parameters", "parameters"), &BBCodeToken::set_parameters);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "type"), "set_type", "get_type"); // TODO: bind enum
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "value"), "set_value", "get_value");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "parameters"), "set_parameters", "get_parameters");
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
	// TODO: allow escaping quotes with backslash?
	// TODO: how should a string like abc"def"ghi be handled?
	enum Escape {
		NONE,
		SINGLE_QUOTE,
		DOUBLE_QUOTE,
	};

	String buffer;
	String parameter;
	State state = State::TAG_NAME;
	Escape escaping = Escape::NONE;

	int pos;
	for (pos = p_start; pos < p_bbcode.length(); pos++) {
		const char32_t cchar = p_bbcode[pos];

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
				switch (escaping) {
					case Escape::NONE:
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
								escaping = Escape::SINGLE_QUOTE;
								break;
							case '"':
								escaping = Escape::DOUBLE_QUOTE;
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
					case Escape::SINGLE_QUOTE:
						switch (cchar) {
							case '\'':
								// When using quotes, do not strip spaces and allow an empty string value.
								escaping = Escape::NONE;
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
					case Escape::DOUBLE_QUOTE:
						switch (cchar) {
							case '"':
								// When using quotes, do not strip spaces and allow an empty string value.
								escaping = Escape::NONE;
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
}

void BBCodeParser::push_bbcode(const String &p_bbcode) {
	int pos = 0;
	while (pos <= p_bbcode.length()) {
		int brk_pos = p_bbcode.find_char('[', pos);

		if (brk_pos < 0) {
			brk_pos = p_bbcode.length();
		}

		String txt = brk_pos > pos ? p_bbcode.substr(pos, brk_pos - pos) : "";

		if (!txt.is_empty()) {
			push_text(txt);
		}

		if (brk_pos == p_bbcode.length()) {
			break; // Nothing else to add.
		}

		String tag;
		Dictionary parameters;
		_parse_tag(p_bbcode, brk_pos + 1, tag, parameters, pos);
		// Move after the closing brace.
		pos++;

		bool is_closing_tag = !tag.is_empty() && tag[0] == '/';
		tag = is_closing_tag ? tag.substr(1) : tag;

		// TODO: this unnecessarily splits escaped tags into their own text tokens
		if (escape_contents) {
			DEV_ASSERT(!tag_stack.is_empty());

			// Wait for closing tag to stop escaping.
			if (tag_stack[tag_stack.size() - 1] == tag) {
				escape_contents = false;
			} else {
				push_text(p_bbcode.substr(brk_pos, pos - brk_pos));
				continue;
			}
		}

		if (is_closing_tag) {
			push_close_tag(tag);
		} else {
			push_open_tag(tag, parameters);
		}
	}
}

void BBCodeParser::push_text(const String &p_text) {
	_update_error(validate_text(p_text));
	if (error == OK) {
		BBCodeToken *token = memnew(BBCodeToken);
		token->set_type(BBCodeToken::Type::TEXT);
		token->set_value(p_text);
		tokens.append(token);
	}
}

void BBCodeParser::push_open_tag(const String &p_tag, const Dictionary &p_parameters) {
	Dictionary result = validate_tag(p_tag, p_parameters);

	Variant error_var = result.get("error", OK);
	ERR_FAIL_COND(error_var.get_type() != Variant::Type::INT);
	_update_error(static_cast<Error>(error_var));

	if (error == OK) {
		BBCodeToken *token = memnew(BBCodeToken);
		token->set_type(BBCodeToken::Type::OPEN_TAG);
		token->set_value(p_tag);
		token->set_parameters(p_parameters);
		tokens.append(token);

		Variant self_closing_var = result.get("self_closing", false);
		ERR_FAIL_COND(self_closing_var.get_type() != Variant::Type::BOOL);
		if (!static_cast<bool>(self_closing_var)) {
			tag_stack.append(p_tag);
		}

		Variant escape_var = result.get("escape_contents", false);
		ERR_FAIL_COND(escape_var.get_type() != Variant::Type::BOOL);
		escape_contents = static_cast<bool>(escape_var);
	}
}

void BBCodeParser::push_close_tag(const String &p_tag) {
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
	token->set_type(BBCodeToken::Type::CLOSE_TAG);
	token->set_value(p_tag);
	tokens.append(token);
}

void BBCodeParser::_bind_methods() {
	ClassDB::bind_method(D_METHOD("validate_tag", "tag", "parameters"), &BBCodeParser::validate_tag);
	ClassDB::bind_method(D_METHOD("validate_text", "text"), &BBCodeParser::validate_text);

	ClassDB::bind_method(D_METHOD("get_error"), &BBCodeParser::get_error);
	ClassDB::bind_method(D_METHOD("get_items"), &BBCodeParser::get_items);

	ClassDB::bind_method(D_METHOD("clear"), &BBCodeParser::clear);
	ClassDB::bind_method(D_METHOD("push_bbcode", "bbcode"), &BBCodeParser::push_bbcode);
	ClassDB::bind_method(D_METHOD("push_text", "text"), &BBCodeParser::push_text);
	ClassDB::bind_method(D_METHOD("push_open_tag", "tag", "parameters"), &BBCodeParser::push_open_tag);
	ClassDB::bind_method(D_METHOD("push_close_tag", "tag"), &BBCodeParser::push_close_tag);

	GDVIRTUAL_BIND(_validate_tag, "tag", "parameters")
	GDVIRTUAL_BIND(_validate_text, "text")
}

BBCodeParser::BBCodeParser() {
	set_local_to_scene(true);
}

BBCodeParser::~BBCodeParser() {
	clear();
}
