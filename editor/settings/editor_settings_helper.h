/**************************************************************************/
/*  editor_settings_helper.h                                              */
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

#include "core/object/ref_counted.h"
#include "core/templates/list.h"

class TextEdit;
class LineEdit;

class EditorSettingsHelper : public RefCounted {
	GDCLASS(EditorSettingsHelper, RefCounted)

public:
	static void create();
	static EditorSettingsHelper *get_singleton();
	static void destroy();

	void postinitialize_text_edit(TextEdit *p_text_edit);
	void predelete_text_edit(TextEdit *p_text_edit);
	void postinitialize_line_edit(LineEdit *p_line_edit);
	void predelete_line_edit(LineEdit *p_line_edit);

private:
	void _settings_changed();
	void _text_edit_tree_entered(TextEdit *p_text_edit);
	void _text_edit_tree_exited(TextEdit *p_text_edit);
	void _line_edit_tree_entered(LineEdit *p_line_edit);
	void _line_edit_tree_exited(LineEdit *p_line_edit);

	List<TextEdit *> text_edits;
	List<LineEdit *> line_edits;
};
