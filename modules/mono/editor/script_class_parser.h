/*************************************************************************/
/*  script_class_parser.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SCRIPT_CLASS_PARSER_H
#define SCRIPT_CLASS_PARSER_H

#include "core/ustring.h"
#include "core/variant.h"
#include "core/vector.h"

class ScriptClassParser {
public:
	struct NameDecl {
		enum Type {
			NAMESPACE_DECL,
			CLASS_DECL,
			STRUCT_DECL
		};

		String name;
		Type type;
	};

	struct ClassDecl {
		String name;
		String namespace_;
		Vector<String> base;
		bool nested;
	};

private:
	String code;
	int idx;
	int line;
	String error_str;
	bool error;
	Variant value;

	Vector<ClassDecl> classes;

	enum Token {
		TK_BRACKET_OPEN,
		TK_BRACKET_CLOSE,
		TK_CURLY_BRACKET_OPEN,
		TK_CURLY_BRACKET_CLOSE,
		TK_PERIOD,
		TK_COLON,
		TK_COMMA,
		TK_SYMBOL,
		TK_IDENTIFIER,
		TK_STRING,
		TK_NUMBER,
		TK_OP_LESS,
		TK_OP_GREATER,
		TK_EOF,
		TK_ERROR,
		TK_MAX
	};

	static const char *token_names[TK_MAX];
	static String get_token_name(Token p_token);

	Token get_token();

	Error _skip_generic_type_params();

	Error _parse_type_full_name(String &r_full_name);
	Error _parse_class_base(Vector<String> &r_base);
	Error _parse_type_constraints();
	Error _parse_namespace_name(String &r_name, int &r_curly_stack);

public:
	Error parse(const String &p_code);
	Error parse_file(const String &p_filepath);

	String get_error();

	Vector<ClassDecl> get_classes();
};

#endif // SCRIPT_CLASS_PARSER_H
