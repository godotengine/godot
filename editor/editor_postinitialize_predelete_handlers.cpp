/**************************************************************************/
/*  editor_postinitialize_predelete_handlers.cpp                          */
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

#include "editor/settings/editor_settings_helper.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/text_edit.h"

void postinitialize_handler_in_editor(TextEdit *p_text_edit) {
	_post_initialize_in_editor(static_cast<TextEdit::super_type *>(p_text_edit));
	if (EditorSettingsHelper *helper = EditorSettingsHelper::get_singleton()) {
		helper->postinitialize_text_edit(p_text_edit);
	}
}

void predelete_handler_in_editor(TextEdit *p_text_edit) {
	if (EditorSettingsHelper *helper = EditorSettingsHelper::get_singleton()) {
		helper->predelete_text_edit(p_text_edit);
	}
	_predelete_in_editor(static_cast<TextEdit::super_type *>(p_text_edit));
}

void postinitialize_handler_in_editor(LineEdit *p_line_edit) {
	_post_initialize_in_editor(static_cast<LineEdit::super_type *>(p_line_edit));
	if (EditorSettingsHelper *helper = EditorSettingsHelper::get_singleton()) {
		helper->postinitialize_line_edit(p_line_edit);
	}
}

void predelete_handler_in_editor(LineEdit *p_line_edit) {
	if (EditorSettingsHelper *helper = EditorSettingsHelper::get_singleton()) {
		helper->predelete_line_edit(p_line_edit);
	}
	_predelete_in_editor(static_cast<LineEdit::super_type *>(p_line_edit));
}
