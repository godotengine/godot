/**************************************************************************/
/*  project_dialog.h                                                      */
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

#ifndef PROJECT_DIALOG_H
#define PROJECT_DIALOG_H

#include "scene/gui/dialogs.h"

class Button;
class EditorFileDialog;
class LineEdit;
class OptionButton;
class TextureRect;

class ProjectDialog : public ConfirmationDialog {
	GDCLASS(ProjectDialog, ConfirmationDialog);

public:
	enum Mode {
		MODE_NEW,
		MODE_IMPORT,
		MODE_INSTALL,
		MODE_RENAME,
	};

private:
	enum MessageType {
		MESSAGE_ERROR,
		MESSAGE_WARNING,
		MESSAGE_SUCCESS,
	};

	enum InputType {
		PROJECT_PATH,
		INSTALL_PATH,
	};

	Mode mode = MODE_NEW;
	bool is_folder_empty = true;

	Button *browse = nullptr;
	Button *install_browse = nullptr;
	Button *create_dir = nullptr;
	VBoxContainer *name_container = nullptr;
	VBoxContainer *path_container = nullptr;
	VBoxContainer *install_path_container = nullptr;

	VBoxContainer *renderer_container = nullptr;
	Label *renderer_info = nullptr;
	HBoxContainer *default_files_container = nullptr;
	Ref<ButtonGroup> renderer_button_group;

	Label *msg = nullptr;
	LineEdit *project_path = nullptr;
	LineEdit *project_name = nullptr;
	LineEdit *install_path = nullptr;
	TextureRect *status_rect = nullptr;
	TextureRect *install_status_rect = nullptr;

	OptionButton *vcs_metadata_selection = nullptr;

	EditorFileDialog *fdialog = nullptr;
	EditorFileDialog *fdialog_install = nullptr;
	AcceptDialog *dialog_error = nullptr;

	String zip_path;
	String zip_title;
	String fav_dir;

	String created_folder_path;

	void _set_message(const String &p_msg, MessageType p_type = MESSAGE_SUCCESS, InputType input_type = PROJECT_PATH);

	String _test_path();
	void _update_path(const String &p_path);
	void _path_text_changed(const String &p_path);
	void _path_selected(const String &p_path);
	void _file_selected(const String &p_path);
	void _install_path_selected(const String &p_path);

	void _browse_path();
	void _browse_install_path();
	void _create_folder();

	void _text_changed(const String &p_text);
	void _nonempty_confirmation_ok_pressed();
	void _renderer_selected();
	void _remove_created_folder();

	void ok_pressed() override;
	void cancel_pressed() override;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_zip_path(const String &p_path);
	void set_zip_title(const String &p_title);
	void set_mode(Mode p_mode);
	void set_project_path(const String &p_path);

	void ask_for_path_and_show();
	void show_dialog();

	ProjectDialog();
};

#endif // PROJECT_DIALOG_H
