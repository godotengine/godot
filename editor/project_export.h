/**************************************************************************/
/*  project_export.h                                                      */
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

#ifndef PROJECT_EXPORT_H
#define PROJECT_EXPORT_H

#include "core/os/dir_access.h"
#include "core/os/thread.h"
#include "editor/editor_export.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_file_system.h"
#include "editor/editor_inspector.h"
#include "editor/editor_properties.h"
#include "scene/gui/button.h"
#include "scene/gui/check_button.h"
#include "scene/gui/control.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/label.h"
#include "scene/gui/link_button.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/slider.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/tree.h"
#include "scene/main/timer.h"

class EditorNode;

class ProjectExportDialog : public ConfirmationDialog {
	GDCLASS(ProjectExportDialog, ConfirmationDialog);

private:
	TabContainer *sections;

	MenuButton *add_preset;
	Button *duplicate_preset;
	Button *delete_preset;
	ItemList *presets;

	LineEdit *name;
	EditorPropertyPath *export_path;
	EditorInspector *parameters;
	CheckButton *runnable;

	Button *button_export;
	bool updating;

	RichTextLabel *result_dialog_log;
	AcceptDialog *result_dialog;
	ConfirmationDialog *delete_confirm;

	OptionButton *export_filter;
	LineEdit *include_filters;
	LineEdit *exclude_filters;
	Tree *include_files;

	Label *include_label;
	MarginContainer *include_margin;

	StringName editor_icons;

	Button *export_button;
	Button *export_all_button;
	AcceptDialog *export_all_dialog;

	LineEdit *custom_features;
	RichTextLabel *custom_feature_display;

	OptionButton *script_mode;
	LineEdit *script_key;
	Label *script_key_error;

	Label *export_error;
	Label *export_warning;
	HBoxContainer *export_templates_error;

	String default_filename;

	void _runnable_pressed();
	void _update_parameters(const String &p_edited_property);
	void _name_changed(const String &p_string);
	void _export_path_changed(const StringName &p_property, const Variant &p_value, const String &p_field, bool p_changing);
	void _add_preset(int p_platform);
	void _edit_preset(int p_index);
	void _duplicate_preset();
	void _delete_preset();
	void _delete_preset_confirm();
	void _update_export_all();

	void _force_update_current_preset_parameters();
	void _update_current_preset();
	void _update_presets();

	void _export_type_changed(int p_which);
	void _filter_changed(const String &p_filter);
	void _fill_resource_tree();
	bool _fill_tree(EditorFileSystemDirectory *p_dir, TreeItem *p_item, Ref<EditorExportPreset> &current, bool p_only_scenes);
	void _tree_changed();
	void _check_dir_recursive(TreeItem *p_dir, bool p_checked);
	void _refresh_parent_checks(TreeItem *p_item);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	EditorFileDialog *export_pck_zip;
	EditorFileDialog *export_project;
	CheckBox *export_debug;
	CheckBox *export_pck_zip_debug;

	void _open_export_template_manager();

	void _export_pck_zip();
	void _export_pck_zip_selected(const String &p_path);

	void _validate_export_path(const String &p_path);
	void _export_project();
	void _export_project_to_path(const String &p_path);
	void _export_all_dialog();
	void _export_all_dialog_action(const String &p_str);
	void _export_all(bool p_debug);

	void _update_feature_list();
	void _custom_features_changed(const String &p_text);

	bool updating_script_key;
	void _script_export_mode_changed(int p_mode);
	void _script_encryption_key_changed(const String &p_key);
	bool _validate_script_encryption_key(const String &p_key);

	void _open_key_help_link();

	void _tab_changed(int);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void popup_export();

	void set_export_path(const String &p_value);
	String get_export_path();

	Ref<EditorExportPreset> get_current_preset() const;

	ProjectExportDialog();
	~ProjectExportDialog();
};

#endif // PROJECT_EXPORT_H
