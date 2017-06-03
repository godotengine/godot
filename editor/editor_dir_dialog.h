/*************************************************************************/
/*  editor_dir_dialog.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef EDITOR_DIR_DIALOG_H
#define EDITOR_DIR_DIALOG_H

#include "os/dir_access.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/tree.h"

class EditorDirDialog : public ConfirmationDialog {
	GDCLASS(EditorDirDialog, ConfirmationDialog);

	ConfirmationDialog *makedialog;
	LineEdit *makedirname;
	AcceptDialog *mkdirerr;

	Button *makedir;

	Tree *tree;
	bool updating;

	void _item_collapsed(Object *p_item);
	void _update_dir(TreeItem *p_item);

	void _make_dir();
	void _make_dir_confirm();

	void ok_pressed();

	bool must_reload;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_current_path(const String &p_path);
	void reload();
	EditorDirDialog();
};

#endif // EDITOR_DIR_DIALOG_H
