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

#pragma once

#include "scene/gui/dialogs.h"

class Button;
class CheckBox;
class CheckButton;
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
		MODE_DUPLICATE,
	};

private:
	enum FocusTarget {
		FOCUS_TARGET_NAME,
		FOCUS_TARGET_ORIGIN,
		FOCUS_TARGET_DESTINATION,
	};

	enum MessageType {
		MESSAGE_ERROR,
		MESSAGE_WARNING,
		MESSAGE_SUCCESS,
	};

	enum InputType {
		ORIGIN_PATH,
		DESTINATION_PATH,
	};

	enum InvalidStateFlag {
		INVALID_STATE_FLAG_NONE = 0,
		INVALID_STATE_FLAG_PATH_INPUT = 1 << 0,
		INVALID_STATE_FLAG_RENDERER_SELECT = 1 << 1,
	};

	Mode mode = MODE_NEW;
	bool is_folder_empty = true;
	ConfirmationDialog *nonempty_confirmation = nullptr;

	CheckButton *create_dir = nullptr;
	Button *origin_browse = nullptr;
	Button *destination_browse = nullptr;
	VBoxContainer *name_container = nullptr;
	VBoxContainer *origin_path_container = nullptr;
	VBoxContainer *destination_path_container = nullptr;

	VBoxContainer *renderer_container = nullptr;
	Label *renderer_info = nullptr;
	HBoxContainer *default_files_container = nullptr;
	Ref<ButtonGroup> renderer_button_group;
	bool rendering_device_supported = false;
	bool rendering_device_checked = false;
	Label *rd_not_supported = nullptr;

	Label *msg = nullptr;
	LineEdit *project_name = nullptr;
	Label *origin_path_label = nullptr;
	LineEdit *origin_path = nullptr;
	Label *destination_path_label = nullptr;
	LineEdit *destination_path = nullptr;
	TextureRect *origin_status_rect = nullptr;
	TextureRect *destination_status_rect = nullptr;

	OptionButton *vcs_metadata_selection = nullptr;

	CheckBox *edit_check_box = nullptr;

	EditorFileDialog *fdialog_origin = nullptr;
	EditorFileDialog *fdialog_destination = nullptr;
	AcceptDialog *dialog_error = nullptr;

	bool is_origin_zip = false;
	String zip_title;

	String default_destination_dir;
	String project_root;

	bool duplicate_can_edit = false;

	BitField<InvalidStateFlag> invalid_state_flags = INVALID_STATE_FLAG_NONE;

	void _set_message(const String &p_msg, MessageType p_type, InputType input_type);
	void _update_ok_button();
	void _validate_path();
	bool _pre_validate_path(const String &p_path, const InputType &p_type);

	// Calculated from project name / ZIP name.
	String auto_dir;

	// Updates `auto_dir`. If the destination path dir name is equal to `auto_dir` (the default state), the destination path is also updated.
	void _update_destination_auto_dir();
	void _reset_auto_dir();

	// While `create_dir` is disabled, stores the last destination path dir name, or an empty string if equal to `auto_dir`.
	String last_custom_destination_dir;
	void _create_dir_toggled(bool p_pressed);

	void _project_name_changed();
	void _origin_path_changed();
	void _destination_path_changed();

	void _browse_origin_path();
	void _browse_destination_path();

	void _config_name_container(const bool p_reset_name = true);
	void _config_origin_container(const String &p_path_title, const String &p_default_path = String(), const bool p_editable = true);
	void _config_destination_container(const String &p_path_title, const String &p_default_path = String(), bool p_editable = true);
	void _set_dialog_focus(FocusTarget p_focus_target);

	void _origin_path_selected(const String &p_path);
	void _destination_path_selected(const String &p_path);

	void _reset_name();
	void _renderer_selected();
	void _nonempty_confirmation_ok_pressed();

	void ok_pressed() override;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void set_mode(Mode p_mode);
	void set_project_name(const String &p_name);
	void set_origin_path(const String &p_path, const bool &p_working_on_origin = false);
	void set_zip_title(const String &p_title);
	void set_zip_project_root(const String &p_path);
	void set_duplicate_can_edit(bool p_duplicate_can_edit);

	void ask_for_path_and_show();
	void show_dialog(bool p_reset_name = true, bool p_is_confirmed = true);

	ProjectDialog();
};
