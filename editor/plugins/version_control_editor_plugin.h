#ifndef VERSION_CONTROL_EDITOR_PLUGIN_H
#define VERSION_CONTROL_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"
#include "editor/editor_vcs_interface.h"
#include "scene/gui/container.h"
#include "scene/gui/rich_text_label.h"
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

	PopupMenu *version_control_actions;
	AcceptDialog *set_up_dialog;
	VBoxContainer *set_up_vbc;
	HBoxContainer *set_up_hbc;
	Label *set_up_vcs_label;
	OptionButton *set_up_choice;
	PanelContainer *set_up_init_settings;
	Button *set_up_init_button;
	RichTextLabel *set_up_vcs_status;
	Button *set_up_ok_button;

	HashMap<ChangeType, String> change_type_to_strings;
	HashMap<ChangeType, Color> change_type_to_color;

	VBoxContainer *version_commit_dock;
	VBoxContainer *commit_box_vbc;
	HSplitContainer *stage_tools;
	Tree *stage_files;
	TreeItem *new_files;
	TreeItem *modified_files;
	TreeItem *renamed_files;
	TreeItem *deleted_files;
	TreeItem *typechange_files;
	Label *staging_area_label;
	HSplitContainer *stage_buttons;
	Button *stage_all_button;
	Button *stage_selected_button;
	Button *refresh_button;
	TextEdit *commit_message;
	Button *commit_button;
	Label *commit_status;

	PanelContainer *version_control_dock;
	ToolButton *version_control_dock_button;
	VBoxContainer *diff_vbc;
	HBoxContainer *diff_hbc;
	Button *diff_refresh_button;
	Label *diff_file_name;
	Label *diff_heading;
	RichTextLabel *diff;

	void _populate_available_vcs_names();
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

	friend class EditorVCSInterface;

protected:
	static void _bind_methods();

public:
	static VersionControlEditorPlugin *get_singleton();

	void popup_vcs_set_up_dialog(const Control *p_gui_base);
	void set_version_control_tool_button(ToolButton *p_button) { version_control_dock_button = p_button; }

	PopupMenu *get_version_control_actions_panel() const { return version_control_actions; }
	VBoxContainer *get_version_commit_dock() const { return version_commit_dock; }
	PanelContainer *get_version_control_dock() const { return version_control_dock; }

	List<StringName> get_available_vcs_names() const { return available_addons; }
	bool get_is_vcs_intialized() const;
	const String get_vcs_name() const;

	void register_editor();
	void fetch_available_vcs_addon_names();
	void clear_stage_area();
	void shut_down();

	VersionControlEditorPlugin();
	~VersionControlEditorPlugin();
};

VARIANT_ENUM_CAST(VersionControlEditorPlugin::ChangeType);

#endif // !VERSION_CONTROL_EDITOR_PLUGIN_H
