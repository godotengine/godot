/*************************************************************************/
/*  syntax_highlighter.cpp                                               */
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

#include "syntax_highlighter.h"

#include "core/script_language.h"
#include "scene/gui/text_edit.h"

Dictionary SyntaxHighlighter::get_line_syntax_highlighting(int p_line) {
	return call("_get_line_syntax_highlighting", p_line);
}

void SyntaxHighlighter::update_cache() {
	call("_update_cache");
}

String SyntaxHighlighter::_get_name() const {
	ScriptInstance *si = get_script_instance();
	if (si && si->has_method("_get_name")) {
		return si->call("_get_name");
	}
	return "Unamed";
}

Array SyntaxHighlighter::_get_supported_languages() const {
	ScriptInstance *si = get_script_instance();
	if (si && si->has_method("_get_supported_languages")) {
		return si->call("_get_supported_languages");
	}
	return Array();
}

void SyntaxHighlighter::set_text_edit(TextEdit *p_text_edit) {
	text_edit = p_text_edit;
}

TextEdit *SyntaxHighlighter::get_text_edit() {
	return text_edit;
}

Ref<SyntaxHighlighter> SyntaxHighlighter::_create() const {
	Ref<SyntaxHighlighter> syntax_highlighter;
	syntax_highlighter.instance();
	if (get_script_instance()) {
		syntax_highlighter->set_script(get_script_instance()->get_script());
	}
	return syntax_highlighter;
}

void SyntaxHighlighter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_get_line_syntax_highlighting", "p_line"), &SyntaxHighlighter::_get_line_syntax_highlighting);
	ClassDB::bind_method(D_METHOD("_update_cache"), &SyntaxHighlighter::_update_cache);
	ClassDB::bind_method(D_METHOD("get_text_edit"), &SyntaxHighlighter::get_text_edit);

	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_name"));
	BIND_VMETHOD(MethodInfo(Variant::ARRAY, "_get_supported_languages"));
	BIND_VMETHOD(MethodInfo(Variant::DICTIONARY, "_get_line_syntax_highlighting", PropertyInfo(Variant::INT, "p_line")));
	BIND_VMETHOD(MethodInfo("_update_cache"));
}
