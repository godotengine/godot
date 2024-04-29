/**************************************************************************/
/*  editor_dir_dialog.h                                                   */
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

#ifndef EDITOR_DIR_DIALOG_H
#define EDITOR_DIR_DIALOG_H

#include "scene/gui/dialogs.h"

class CheckBox;
class EditorFileSystemDirectory;
class Tree;
class TreeItem;

class EditorDirDialog : public ConfirmationDialog {
	GDCLASS(EditorDirDialog, ConfirmationDialog);

	ConfirmationDialog *makedialog = nullptr;
	LineEdit *makedirname = nullptr;
	AcceptDialog *mkdirerr = nullptr;

	Button *makedir = nullptr;
	HashSet<String> opened_paths;

	Tree *tree = nullptr;
	bool updating = false;
	CheckBox *copy = nullptr;

	void _copy_toggled(bool p_pressed);
	void _item_collapsed(Object *p_item);
	void _item_activated();
	void _update_dir(TreeItem *p_item, EditorFileSystemDirectory *p_dir, const String &p_select_path = String());

	void _make_dir();
	void _make_dir_confirm();

	void ok_pressed() override;

	bool must_reload = false;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void reload(const String &p_path = "");
	bool is_copy_pressed() const;

	EditorDirDialog();
};

#endif // EDITOR_DIR_DIALOG_H
