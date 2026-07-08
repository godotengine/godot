/**************************************************************************/
/*  text_editor.cpp                                                       */
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

#include "text_editor.h"

#include "core/io/json.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/settings/editor_settings.h"

void TextEditor::_validate_script() {
	TextEditorBase::_validate_script();

	Ref<JSON> json_file = edited_res;
	if (json_file.is_valid()) {
		CodeEdit *te = code_editor->get_text_editor();

		te->set_line_background_color(code_editor->get_error_pos().x, Color(0, 0, 0, 0));
		code_editor->set_error("");

		if (json_file->parse(te->get_text(), true) != OK) {
			code_editor->set_error(json_file->get_error_message().replace("[", "[lb]"));
			code_editor->set_error_pos(json_file->get_error_line(), 0);
			te->set_line_background_color(code_editor->get_error_pos().x, EDITOR_GET("text_editor/theme/highlighting/mark_color"));
		}
	}
}

void TextEditor::apply_code() {
	Ref<TextFile> text_file = edited_res;
	if (text_file.is_valid()) {
		text_file->set_text(code_editor->get_text_editor()->get_text());
	}

	Ref<JSON> json_file = edited_res;
	if (json_file.is_valid()) {
		json_file->parse(code_editor->get_text_editor()->get_text(), true);
	}
	code_editor->get_text_editor()->get_syntax_highlighter()->update_cache();
}

ScriptEditorBase *TextEditor::create_editor(const Ref<Resource> &p_resource) {
	if (Object::cast_to<TextFile>(*p_resource) || Object::cast_to<JSON>(*p_resource)) {
		return memnew(TextEditor);
	}
	return nullptr;
}

Control *TextEditor::get_edit_menu() {
	if (!edit_menus) {
		edit_menus = memnew(EditMenus);
	}
	return edit_menus;
}

void TextEditor::register_editor() {
	ScriptEditor::register_create_script_editor_function(create_editor);
}

TextEditor::TextEditor() {
	add_child(code_editor);
}
