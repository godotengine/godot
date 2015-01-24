/*************************************************************************/
/*  project_export.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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

#include "scene/main/timer.h"
#include "scene/gui/control.h"
#include "scene/gui/tree.h"
#include "scene/gui/label.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/tab_container.h"
#include "os/dir_access.h"
#include "os/thread.h"
#include "scene/gui/option_button.h"

#include "scene/gui/slider.h"
#include "tools/editor/editor_file_system.h"
#include "property_editor.h"
#include "editor_import_export.h"

class EditorNode;

class ProjectExportDialog : public ConfirmationDialog {
	OBJ_TYPE( ProjectExportDialog, ConfirmationDialog );

public:
	enum ExportAction {
		ACTION_NONE,
		ACTION_COPY,
		ACTION_BUNDLE,
		ACTION_MAX

	};

	static const char *da_string[ACTION_MAX];

private:


	EditorNode *editor;
	String expopt;

	TabContainer *sections;
	bool updating_tree;
	AcceptDialog *error;
	ConfirmationDialog *confirm;

	Button *button_reload;
	LineEdit *filters;
	HBoxContainer *plat_errors;
	Label *platform_error_string;

	Tree * tree;

	FileDialog *pck_export;
	FileDialog *file_export;
	CheckButton *file_export_check;
	LineEdit *file_export_password;

	Button *button_export;
	String _delete_attempt;

	bool updating;

	void _tree_changed();
	void _update_tree();

	bool _create_tree(TreeItem *p_parent,EditorFileSystemDirectory *p_dir);
	void _rescan();
//	void _confirmed();
	void _scan_finished();

	void _validate_platform();
	///////////////////

	Tree * platforms;
	PropertyEditor *platform_options;

	OptionButton *export_mode;
	VBoxContainer *tree_vb;

	VBoxContainer *image_vb;
	OptionButton *image_action;
	HSlider *image_quality;
	SpinBox *image_shrink;
	Tree *image_formats;
	Vector<TreeItem*> formats;

	LineEdit *group_new_name;
	HSlider *group_lossy_quality;
	Label *group_new_name_error;
	VBoxContainer *group_options;
	Tree *groups;
	SpinBox *group_shrink;
	CheckButton *group_atlas;
	OptionButton *group_image_action;
	Button *group_add;
	Tree *group_images;
	LineEdit *group_images_filter;
	Button *atlas_preview;


	AcceptDialog *atlas_preview_dialog;
	TextureFrame *atlas_preview_frame;


	VBoxContainer *script_vbox;
	OptionButton *script_mode;
	LineEdit *script_key;



	void _export_mode_changed(int p_idx);
	void _prop_edited(String what);

	void _update_platform();
	void _update_exporter();
	void _platform_selected();

	void _filters_edited(String what);
	void _update_group_tree();

	void _image_filter_changed(String);
	bool _update_group_treef(TreeItem *p_parent,EditorFileSystemDirectory *p_dir,const Set<String>& p_extensions,const String& p_groups,const Map<StringName,int>& p_group_index);
	void _group_item_edited();
	void _group_atlas_preview();



	void _quality_edited(float what);
	void _image_export_edited(int what);
	void _shrink_edited(float what);

	void _update_group_list();
	void _select_group(const String& p_by_name);


	String _get_selected_group();
	void _update_group();
	void _group_changed(Variant v);
	void _group_selected();
	void _group_add();
	void _group_select_all();
	void _group_select_none();
	void _group_del(Object *item,int p_column, int p_button);

	bool updating_script;
	void _update_script();
	void _script_edited(Variant v);
	void _export_action(const String& p_file);
	void _export_action_pck(const String& p_file);
	void ok_pressed();
	void custom_action(const String&);

	void _save_export_cfg();
	void _format_toggled();


protected:
	void _notification(int p_what);
	static void _bind_methods();
public:

	String get_selected_path() const;

	Error export_platform(const String& p_platform, const String& p_path, bool p_debug,const String& p_password,bool p_quit_after=false);

	ProjectExportDialog(EditorNode *p_editor);
	~ProjectExportDialog();
};

class EditorData;

class ProjectExport : public ConfirmationDialog {
	OBJ_TYPE( ProjectExport, ConfirmationDialog );

	EditorData *editor_data;

	AcceptDialog *error;
	Label *label;
	OptionButton *export_preset;
public:

	Error export_project(const String& p_preset);
	void popup_export();


	ProjectExport(EditorData* p_data);

};


#endif // PROJECT_EXPORT_SETTINGS_H
