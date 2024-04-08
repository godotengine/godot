/**************************************************************************/
/*  editor_verify_dialog.h                                                */
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

#ifndef EDITOR_VERIFY_DIALOG_H
#define EDITOR_VERIFY_DIALOG_H

#include "scene/gui/dialogs.h"

class Label;
class Tree;
class LineEdit;

class EditorVerifyDialog : public ConfirmationDialog {
	GDCLASS(EditorVerifyDialog, ConfirmationDialog);

	int items_count = 0;
	unsigned evaluation_bit = 0;

	String phrase = "";

	Label *checklists_label = nullptr;
	Tree *checklists_tree = nullptr;

	Label *phrase_label = nullptr;
	LineEdit *phrase_line_edit = nullptr;

	void _show_phrase();
	void _show_checklists(const Dictionary &p_checklists);

	void _item_edited();
	void _validate_phrase(const String &p_text);

	void _evaluate();

public:
	bool reload(int p_export_type, const Callable &p_export_all_callable, const Callable &p_export_callable, const Callable &p_export_pck_zip_callable);

	EditorVerifyDialog();
};

#endif // EDITOR_VERIFY_DIALOG_H
