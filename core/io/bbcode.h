/**************************************************************************/
/*  bbcode.h                                                              */
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

#pragma once

#include "core/io/resource.h"
#include "core/variant/typed_array.h"

class BBCodeToken : public Object {
	GDCLASS(BBCodeToken, Object);

public:
	enum TokenType {
		TOKEN_TYPE_TEXT,
		TOKEN_TYPE_OPEN_TAG,
		TOKEN_TYPE_CLOSE_TAG,
	};

private:
	int position_start = 0;
	String parsed_bbcode;
	TokenType type = TokenType::TOKEN_TYPE_TEXT;
	String value;
	Dictionary parameters;

protected:
	static void _bind_methods();

public:
	String get_normalized_bbcode() const;
	String get_parsed_bbcode() const { return parsed_bbcode; }
	void set_parsed_bbcode(const String &p_parsed_bbcode) { parsed_bbcode = p_parsed_bbcode; }
	TokenType get_type() const { return type; }
	void set_type(TokenType p_type) { type = p_type; }
	String get_value() const { return value; }
	void set_value(const String &p_value) { value = p_value; }
	Dictionary get_parameters() const { return parameters; }
	void set_parameters(const Dictionary &p_parameters) { parameters = p_parameters; }
	int get_position_start() const { return position_start; }
	void set_position_start(int p_position_start) { position_start = p_position_start; }
	int get_position_end() const { return position_start + parsed_bbcode.length(); }
};

class BBCodeParser : public Resource {
	GDCLASS(BBCodeParser, Resource);

public:
	enum EscapeBrackets {
		ESCAPE_BRACKETS_NONE = 0,
		ESCAPE_BRACKETS_WRAPPED = (1 << 0),
		ESCAPE_BRACKETS_BACKSLASH = (1 << 1),
		ESCAPE_BRACKETS_ABBREVIATION = (1 << 2),
	};

	enum ErrorHandling {
		ERROR_HANDLING_REMOVE = 0,
		ERROR_HANDLING_PARSE_AS_TEXT = 1,
		ERROR_HANDLING_KEEP = 2,
	};

protected:
	static void _bind_methods();

	GDVIRTUAL2RC(Dictionary, _validate_tag, String, Dictionary)
	GDVIRTUAL1RC(Error, _validate_text, String)

	int position = 0;
	bool backslash_escape_quotes = true;
	bool escape_contents = false;
	BitField<EscapeBrackets> escape_brackets = ESCAPE_BRACKETS_BACKSLASH;
	Error error = OK;
	ErrorHandling error_handling = ERROR_HANDLING_REMOVE;
	Vector<BBCodeToken *> tokens;
	Vector<String> tag_stack;

	void _parse_tag(const String &p_bbcode, const int p_start, String &r_tag, Dictionary &r_parameters, int &r_end);
	void _update_error(Error p_error) {
		if (error == OK) {
			error = p_error;
		}
	}
	void _parsed_push_text(const String &p_parsed_bbcode, const String &p_text);
	void _parsed_push_tag(const String &p_parsed_bbcode, const String &p_tag, const Dictionary &p_parameters);
	void _parsed_pop_tag(const String &p_parsed_bbcode, const String &p_tag);
	void _push_token(const String &p_parsed_bbcode, BBCodeToken *p_token);

	bool _is_ok() const { return error == OK || error_handling == ERROR_HANDLING_KEEP; }

public:
	// Returns { error: Error = OK, escape_contents: boolean = false, self_closing: boolean = false }
	virtual Dictionary _validate_tag(const String &p_tag, const Dictionary &p_parameters) const { return Dictionary(); }
	Dictionary validate_tag(const String &p_tag, const Dictionary &p_parameters);
	virtual Error _validate_text(const String &p_text) const { return OK; }
	Error validate_text(const String &p_text);

	Error get_error() const { return error; }
	TypedArray<BBCodeToken> get_items() const;

	EscapeBrackets get_escape_brackets() const { return escape_brackets; }
	void set_escape_brackets(const EscapeBrackets p_value) { escape_brackets = p_value; }

	bool get_backslash_escape_quotes() const { return backslash_escape_quotes; }
	void set_backslash_escape_quotes(const bool p_value) { backslash_escape_quotes = p_value; }

	ErrorHandling get_error_handling() const { return error_handling; }
	void set_error_handling(ErrorHandling p_error_handling) { error_handling = p_error_handling; }

	void clear();
	void push_bbcode(const String &p_bbcode);
	void push_text(const String &p_text);
	void push_tag(const String &p_tag, const Dictionary &p_parameters);
	void pop_tag(const String &p_tag);

	~BBCodeParser();
};

VARIANT_ENUM_CAST(BBCodeToken::TokenType);
VARIANT_ENUM_CAST(BBCodeParser::ErrorHandling);
VARIANT_BITFIELD_CAST(BBCodeParser::EscapeBrackets);
