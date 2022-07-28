/*************************************************************************/
/*  version_control_editor_plugin.h                                      */
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

#ifndef VERSION_CONTROL_EDITOR_PLUGIN_H
#define VERSION_CONTROL_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"
#include "editor/editor_vcs_interface.h"
#include "scene/gui/box_container.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/split_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/tree.h"

class VersionControlEditorPlugin : public EditorPlugin {
	GDCLASS(VersionControlEditorPlugin, EditorPlugin)

public:
	enum ChangeType {
		CHANGE_TYPE_NEW = 0,
		CHANGE_TYPE_MODIFIED = 1,
		CHANGE_TYPE_RENAMED = 2,
		CHANGE_TYPE_DELETED = 3,
		CHANGE_TYPE_TYPECHANGE = 4
	};

private:
	static VersionControlEditorPlugin *singleton;

	int staged_files_count;
	List<StringName> available_addons;

	PopupMenu *version_control_actions = nullptr;
	ConfirmationDialog *metadata_dialog = nullptr;
	OptionButton *metadata_selection = nullptr;
	AcceptDialog *set_up_dialog = nullptr;
	VBoxContainer *set_up_vbc = nullptr;
	HBoxContainer *set_up_hbc = nullptr;
	Label *set_up_vcs_label = nullptr;
	OptionButton *set_up_choice = nullptr;
	PanelContainer *set_up_init_settings = nullptr;
	Button *set_up_init_button = nullptr;
	RichTextLabel *set_up_vcs_status = nullptr;
	Button *set_up_ok_button = nullptr;

	HashMap<ChangeType, String> change_type_to_strings;
	HashMap<ChangeType, Color> change_type_to_color;

	VBoxContainer *version_commit_dock = nullptr;
	VBoxContainer *commit_box_vbc = nullptr;
	HSplitContainer *stage_tools = nullptr;
	Tree *stage_files = nullptr;
	TreeItem *new_files = nullptr;
	TreeItem *modified_files = nullptr;
	TreeItem *renamed_files = nullptr;
	TreeItem *deleted_files = nullptr;
	TreeItem *typechange_files = nullptr;
	Label *staging_area_label = nullptr;
	HSplitContainer *stage_buttons = nullptr;
	Button *stage_all_button = nullptr;
	Button *stage_selected_button = nullptr;
	Button *refresh_button = nullptr;
	TextEdit *commit_message = nullptr;
	Button *commit_button = nullptr;
	Label *commit_status = nullptr;

	PanelContainer *version_control_dock = nullptr;
	Button *version_control_dock_button = nullptr;
	VBoxContainer *diff_vbc = nullptr;
	HBoxContainer *diff_hbc = nullptr;
	Button *diff_refresh_button = nullptr;
	Label *diff_file_name = nullptr;
	Label *diff_heading = nullptr;
	RichTextLabel *diff = nullptr;

	void _populate_available_vcs_names();
	void _create_vcs_metadata_files();
	void _selected_a_vcs(int p_id);
	void _initialize_vcs();
	void _send_commit_msg();
	void _refresh_stage_area();
	void _stage_selected();
	void _stage_all();
	void _view_file_diff();
	void _display_file_diff(String p_file_path);
	void _refresh_file_diff();
	void _clear_file_diff();
	void _update_stage_status();
	void _update_commit_status();
	void _update_commit_button();
	void _commit_message_gui_input(const Ref<InputEvent> &p_event);

	friend class EditorVCSInterface;

protected:
	static void _bind_methods();

public:
	static VersionControlEditorPlugin *get_singleton();

	void popup_vcs_metadata_dialog();
	void popup_vcs_set_up_dialog(const Control *p_gui_base);
	void set_version_control_tool_button(Button *p_button) { version_control_dock_button = p_button; }

	PopupMenu *get_version_control_actions_panel() const { return version_control_actions; }
	VBoxContainer *get_version_commit_dock() const { return version_commit_dock; }
	PanelContainer *get_version_control_dock() const { return version_control_dock; }

	List<StringName> get_available_vcs_names() const { return available_addons; }
	bool is_vcs_initialized() const;
	const String get_vcs_name() const;

	void register_editor();
	void fetch_available_vcs_addon_names();
	void clear_stage_area();
	void shut_down();

	VersionControlEditorPlugin();
	~VersionControlEditorPlugin();
};

VARIANT_ENUM_CAST(VersionControlEditorPlugin::ChangeType);

#endif // VERSION_CONTROL_EDITOR_PLUGIN_H
