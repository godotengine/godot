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
	enum Type {
		TEXT,
		OPEN_TAG,
		CLOSE_TAG,
	};

private:
	int type;
	String value;
	Dictionary parameters;

protected:
	static void _bind_methods();

public:
	int get_type() const { return type; }
	void set_type(int p_type) { type = p_type; }
	String get_value() const { return value; }
	void set_value(String p_value) { value = p_value; }
	Dictionary get_parameters() const { return parameters; }
	void set_parameters(Dictionary p_parameters) { parameters = p_parameters; }
};

class BBCodeParser : public Resource {
	GDCLASS(BBCodeParser, Resource);

protected:
	static void _bind_methods();

	GDVIRTUAL2RC(Dictionary, _validate_tag, String, Dictionary)
	GDVIRTUAL1RC(Error, _validate_text, String)

	bool escape_contents = false;
	Error error = OK;
	Vector<BBCodeToken *> tokens;
	Vector<String> tag_stack;

	void _parse_tag(const String &p_bbcode, const int p_start, String &r_tag, Dictionary &r_parameters, int &r_end);
	void _update_error(Error p_error) {
		if (error == OK) {
			error = p_error;
		}
	}

public:
	// Returns { error: Error = OK, escape_contents: boolean = false, self_closing: boolean = false }
	virtual Dictionary _validate_tag(const String &p_tag, const Dictionary &p_parameters) const { return Dictionary(); }
	Dictionary validate_tag(const String &p_tag, const Dictionary &p_parameters);
	virtual Error _validate_text(const String &p_text) const { return OK; }
	Error validate_text(const String &p_text);

	Error get_error() const { return error; }
	TypedArray<BBCodeToken> get_items() const;

	void clear();
	void push_bbcode(const String &p_bbcode);
	void push_text(const String &p_text);
	void push_open_tag(const String &p_tag, const Dictionary &p_parameters);
	void push_close_tag(const String &p_tag);

	BBCodeParser();
	~BBCodeParser();
};
