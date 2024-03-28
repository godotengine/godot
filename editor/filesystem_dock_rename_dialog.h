/**************************************************************************/
/*  filesystem_dock_rename_dialog.h										  */
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

#ifndef FILESYSTEM_DOCK_RENAME_DIALOG_H
#define FILESYSTEM_DOCK_RENAME_DIALOG_H

#include "modules/modules_enabled.gen.h" // For regex.
#ifdef MODULE_REGEX_ENABLED

#include "editor/filesystem_dock.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"

class Button;
class CheckBox;
class CheckButton;
class Label;
class OptionButton;
class SpinBox;
class TabContainer;

class FileSystemRenameDialog : public ConfirmationDialog {
	GDCLASS(FileSystemRenameDialog, ConfirmationDialog);

	virtual void ok_pressed() override { rename(); };
	void _cancel_pressed() {}
	void _features_toggled(bool p_pressed);
	void _insert_text(const String &p_text);
	void _update_substitute();
	bool _is_main_field(LineEdit *p_line_edit);

	void _iterate_files(const Vector<String> &p_selection, int *p_counter);
	String _apply_rename(const String &p_filepath, int p_count);
	String _substitute(const String &p_subject, const String &p_filepath, int p_count);
	String _regex(const String &p_pattern, const String &p_subject, const String &p_replacement);
	String _postprocess(const String &p_subject);
	void _update_preview(const String &p_new_text = "");
	void _update_preview_int(int p_new_value = 0);
	static void _error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, bool p_editor_notify, ErrorHandlerType p_type);

	FileSystemDock *file_system_dock = nullptr;
	int global_count = 0;

	LineEdit *lne_search = nullptr;
	LineEdit *lne_replace = nullptr;
	LineEdit *lne_prefix = nullptr;
	LineEdit *lne_suffix = nullptr;

	TabContainer *tabc_features = nullptr;

	CheckBox *cbut_substitute = nullptr;
	CheckButton *cbut_regex = nullptr;
	CheckBox *cbut_process = nullptr;
	CheckBox *chk_per_level_counter = nullptr;

	Button *but_insert_name = nullptr;
	Button *but_insert_parent = nullptr;
	Button *but_insert_count = nullptr;

	SpinBox *spn_count_start = nullptr;
	SpinBox *spn_count_step = nullptr;
	SpinBox *spn_count_padding = nullptr;

	OptionButton *opt_style = nullptr;
	OptionButton *opt_case = nullptr;

	Label *lbl_preview_title = nullptr;
	Label *lbl_preview = nullptr;

	List<Pair<String, String>> to_rename;
	String preview_file;
	bool lock_preview_update = false;
	ErrorHandlerList eh;
	bool has_errors = false;
	Vector<String> selected_files;
	bool is_tree = false;

protected:
	static void _bind_methods();
	virtual void _post_popup() override;

public:
	void reset();
	void rename();
	void set_selected_files(const Vector<String> &p_paths, const bool &p_is_tree);

	FileSystemRenameDialog(FileSystemDock *p_file_system_dock);
};

#endif // MODULE_REGEX_ENABLED

#endif // FILESYSTEM_DOCK_RENAME_DIALOG_H
