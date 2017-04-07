/*************************************************************************/
/*  script_create_dialog.h                                               */
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
#ifndef SCRIPT_CREATE_DIALOG_H
#define SCRIPT_CREATE_DIALOG_H

#include "editor/editor_file_dialog.h"
#include "editor/editor_settings.h"
#include "scene/gui/check_button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"

class ScriptCreateDialog : public ConfirmationDialog {
	GDCLASS(ScriptCreateDialog, ConfirmationDialog);

	LineEdit *class_name;
	Label *error_label;
	Label *path_error_label;
	LineEdit *parent_name;
	OptionButton *language_menu;
	LineEdit *file_path;
	EditorFileDialog *file_browse;
	CheckButton *internal;
	VBoxContainer *path_vb;
	AcceptDialog *alert;
	bool path_valid;
	bool create_new;
	String initial_bp;
	EditorSettings *editor_settings;

	void _path_changed(const String &p_path = String());
	void _lang_changed(int l = 0);
	void _built_in_pressed();
	bool _validate(const String &p_strin);
	void _class_name_changed(const String &p_name);
	void _browse_path();
	void _file_selected(const String &p_file);
	virtual void ok_pressed();
	void _create_new();
	void _load_exist();
	void _update_controls();

protected:
	static void _bind_methods();

public:
	void config(const String &p_base_name, const String &p_base_path);

	ScriptCreateDialog();
};

#endif // SCRIPT_CREATE_DIALOG_H
