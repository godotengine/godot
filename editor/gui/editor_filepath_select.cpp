/**************************************************************************/
/*  editor_filepath_select.cpp                                            */
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

#include "editor_filepath_select.h"

void EditorFilepathSelect::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED:
			browse_button->set_button_icon(get_editor_theme_icon(SNAME("FileBrowse")));
			break;
	}
}
void EditorFilepathSelect::_path_pressed() {
	String full_path = edit->get_text();

	dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	dialog->popup_file_dialog();
}

void EditorFilepathSelect::_dialog_path_selected(const String &p_path) {
	edit->set_text(p_path);

	// TODO: Make a different signal, instead of manually emitting the LineEdit's
	// signal?
	edit->emit_signal("text_submitted", p_path);
}

EditorFilepathSelect::EditorFilepathSelect() {
	edit = memnew(LineEdit);
	edit->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_FILE);
	add_child(edit);
	edit->set_h_size_flags(SIZE_EXPAND_FILL);

	browse_button = memnew(Button);
	add_child(browse_button);
	browse_button->set_clip_text(true);
	browse_button->connect(SceneStringName(pressed), callable_mp(this, &EditorFilepathSelect::_path_pressed));

	dialog = memnew(EditorFileDialog);
	dialog->connect("file_selected", callable_mp(this, &EditorFilepathSelect::_dialog_path_selected));
	add_child(dialog);
}
