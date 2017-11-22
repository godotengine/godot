/*************************************************************************/
/*  project_export.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef PROJECT_EXPORT_SETTINGS_H
#define PROJECT_EXPORT_SETTINGS_H

#include "editor/editor_file_dialog.h"
#include "os/dir_access.h"
#include "os/thread.h"
#include "scene/gui/button.h"
#include "scene/gui/control.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/label.h"
#include "scene/gui/link_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/tree.h"
#include "scene/main/timer.h"

#include "editor/editor_file_system.h"
#include "editor_export.h"
#include "property_editor.h"
#include "scene/gui/slider.h"

class EditorNode;

class ProjectExportDialog : public ConfirmationDialog {
	GDCLASS(ProjectExportDialog, ConfirmationDialog);

private:
	TabContainer *sections;

	MenuButton *add_preset;
	Button *delete_preset;
	ItemList *presets;

	LineEdit *name;
	PropertyEditor *parameters;
	CheckButton *runnable;

	//EditorFileDialog *pck_export;
	//EditorFileDialog *file_export;

	Button *button_export;
	bool updating;

	AcceptDialog *error_dialog;
	ConfirmationDialog *delete_confirm;

	OptionButton *export_filter;
	LineEdit *include_filters;
	LineEdit *exclude_filters;
	Tree *include_files;

	Label *include_label;
	MarginContainer *include_margin;

	StringName editor_icons;

	Tree *patches;
	Button *patch_export;
	int patch_index;
	FileDialog *patch_dialog;
	ConfirmationDialog *patch_erase;

	Button *export_button;

	LineEdit *custom_features;
	RichTextLabel *custom_feature_display;

	Label *export_error;
	HBoxContainer *export_templates_error;

	String default_filename;

	void _patch_selected(const String &p_path);
	void _patch_deleted();

	void _runnable_pressed();
	void _update_parameters(const String &p_edited_property);
	void _name_changed(const String &p_string);
	void _add_preset(int p_platform);
	void _edit_preset(int p_index);
	void _delete_preset();
	void _delete_preset_confirm();

	void _update_presets();

	void _export_type_changed(int p_which);
	void _filter_changed(const String &p_filter);
	void _fill_resource_tree();
	bool _fill_tree(EditorFileSystemDirectory *p_dir, TreeItem *p_item, Ref<EditorExportPreset> &current, bool p_only_scenes);
	void _tree_changed();

	void _patch_button_pressed(Object *p_item, int p_column, int p_id);
	void _patch_edited();

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	FileDialog *export_pck_zip;
	FileDialog *export_project;
	CheckButton *export_debug;

	void _open_export_template_manager();

	void _export_pck_zip();
	void _export_pck_zip_selected(const String &p_path);

	void _export_project();
	void _export_project_to_path(const String &p_path);

	void _update_feature_list();
	void _custom_features_changed(const String &p_text);

	void _tab_changed(int);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void popup_export();

	ProjectExportDialog();
	~ProjectExportDialog();
};

#endif // PROJECT_EXPORT_SETTINGS_H
