/*************************************************************************/
/*  project_export.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef PROJECT_EXPORT_SETTINGS_H
#define PROJECT_EXPORT_SETTINGS_H

#include "core/io/dir_access.h"
#include "core/os/thread.h"
#include "editor/editor_export.h"
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

class EditorFileDialog;

class ProjectExportDialog : public ConfirmationDialog {
	GDCLASS(ProjectExportDialog, ConfirmationDialog);

private:
	TabContainer *sections = nullptr;

	MenuButton *add_preset = nullptr;
	Button *duplicate_preset = nullptr;
	Button *delete_preset = nullptr;
	ItemList *presets = nullptr;

	LineEdit *name = nullptr;
	EditorPropertyPath *export_path = nullptr;
	EditorInspector *parameters = nullptr;
	CheckButton *runnable = nullptr;

	Button *button_export = nullptr;
	bool updating = false;

	RichTextLabel *result_dialog_log = nullptr;
	AcceptDialog *result_dialog = nullptr;
	ConfirmationDialog *delete_confirm = nullptr;

	OptionButton *export_filter = nullptr;
	LineEdit *include_filters = nullptr;
	LineEdit *exclude_filters = nullptr;
	Tree *include_files = nullptr;

	Label *include_label = nullptr;
	MarginContainer *include_margin = nullptr;

	Button *export_button = nullptr;
	Button *export_all_button = nullptr;
	AcceptDialog *export_all_dialog = nullptr;

	LineEdit *custom_features = nullptr;
	RichTextLabel *custom_feature_display = nullptr;

	OptionButton *script_mode = nullptr;
	LineEdit *script_key = nullptr;
	Label *script_key_error = nullptr;

	Label *export_error = nullptr;
	Label *export_warning = nullptr;
	HBoxContainer *export_templates_error = nullptr;

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
	void _check_propagated_to_item(Object *p_obj, int column);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	EditorFileDialog *export_pck_zip = nullptr;
	EditorFileDialog *export_project = nullptr;
	CheckBox *export_debug = nullptr;
	CheckBox *export_pck_zip_debug = nullptr;

	CheckButton *enc_pck = nullptr;
	CheckButton *enc_directory = nullptr;
	LineEdit *enc_in_filters = nullptr;
	LineEdit *enc_ex_filters = nullptr;

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

	bool updating_script_key = false;
	bool updating_enc_filters = false;
	void _enc_pck_changed(bool p_pressed);
	void _enc_directory_changed(bool p_pressed);
	void _enc_filters_changed(const String &p_text);
	void _script_export_mode_changed(int p_mode);
	void _script_encryption_key_changed(const String &p_key);
	bool _validate_script_encryption_key(const String &p_key);

	void _open_key_help_link();

	void _tab_changed(int);

protected:
	void _theme_changed();
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

#endif // PROJECT_EXPORT_SETTINGS_H
