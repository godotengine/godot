/**************************************************************************/
/*  csharp_editor_language.cpp                                            */
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

#include "csharp_editor_language.h"

CSharpEditorLanguage *CSharpEditorLanguage::singleton = nullptr;

const Vector<String> CSharpEditorLanguage::get_reserved_words() const {
	static const Vector<String> ret = {
		// Reserved keywords
		"abstract",
		"as",
		"base",
		"bool",
		"break",
		"byte",
		"case",
		"catch",
		"char",
		"checked",
		"class",
		"const",
		"continue",
		"decimal",
		"default",
		"delegate",
		"do",
		"double",
		"else",
		"enum",
		"event",
		"explicit",
		"extern",
		"false",
		"finally",
		"fixed",
		"float",
		"for",
		"foreach",
		"goto",
		"if",
		"implicit",
		"in",
		"int",
		"interface",
		"internal",
		"is",
		"lock",
		"long",
		"namespace",
		"new",
		"null",
		"object",
		"operator",
		"out",
		"override",
		"params",
		"private",
		"protected",
		"public",
		"readonly",
		"ref",
		"return",
		"sbyte",
		"sealed",
		"short",
		"sizeof",
		"stackalloc",
		"static",
		"string",
		"struct",
		"switch",
		"this",
		"throw",
		"true",
		"try",
		"typeof",
		"uint",
		"ulong",
		"unchecked",
		"unsafe",
		"ushort",
		"using",
		"virtual",
		"void",
		"volatile",
		"while",

		// Contextual keywords. Not reserved words, but I guess we should include
		// them because this seems to be used only for syntax highlighting.
		"add",
		"alias",
		"ascending",
		"async",
		"await",
		"by",
		"descending",
		"dynamic",
		"equals",
		"from",
		"get",
		"global",
		"group",
		"into",
		"join",
		"let",
		"nameof",
		"on",
		"orderby",
		"partial",
		"remove",
		"select",
		"set",
		"value",
		"var",
		"when",
		"where",
		"yield",
	};

	return ret;
}

bool CSharpEditorLanguage::is_control_flow_keyword(const String &p_keyword) const {
	return p_keyword == "break" ||
			p_keyword == "case" ||
			p_keyword == "catch" ||
			p_keyword == "continue" ||
			p_keyword == "default" ||
			p_keyword == "do" ||
			p_keyword == "else" ||
			p_keyword == "finally" ||
			p_keyword == "for" ||
			p_keyword == "foreach" ||
			p_keyword == "goto" ||
			p_keyword == "if" ||
			p_keyword == "return" ||
			p_keyword == "switch" ||
			p_keyword == "throw" ||
			p_keyword == "try" ||
			p_keyword == "while";
}

const Vector<String> CSharpEditorLanguage::get_comment_delimiters() const {
	static const Vector<String> delimiters = {
		"//", // single-line comment
		"/* */" // delimited comment
	};
	return delimiters;
}

const Vector<String> CSharpEditorLanguage::get_doc_comment_delimiters() const {
	static const Vector<String> delimiters = {
		"///", // single-line doc comment
		"/** */" // delimited doc comment
	};
	return delimiters;
}

const Vector<String> CSharpEditorLanguage::get_string_delimiters() const {
	static const Vector<String> delimiters = {
		"' '", // character literal
		"\" \"", // regular string literal
		"@\" \"" // verbatim string literal
	};
	// Generic string highlighting suffices as a workaround for now.
	return delimiters;
}

EditorLanguage::NameCasing CSharpEditorLanguage::get_preferred_file_name_casing() const {
	return NameCasing::PASCAL_CASE;
}
