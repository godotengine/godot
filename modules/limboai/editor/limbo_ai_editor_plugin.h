/**
 * limbo_ai_editor_plugin.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#ifndef LIMBO_AI_EDITOR_PLUGIN_H
#define LIMBO_AI_EDITOR_PLUGIN_H

#include "../bt/behavior_tree.h"
#include "../bt/tasks/bt_task.h"
#include "editor_property_variable_name.h"
#include "owner_picker.h"
#include "task_palette.h"
#include "task_tree.h"

#ifdef LIMBOAI_MODULE
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/templates/hash_set.h"
#include "editor/editor_node.h"
#include "editor/gui/editor_spin_slider.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/gui/control.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/link_button.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/popup.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tree.h"
#include "scene/resources/texture.h"

#define GET_UNDO_REDO() EditorUndoRedoManager::get_singleton()

#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include "godot_cpp/classes/accept_dialog.hpp"
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/editor_plugin.hpp>
#include <godot_cpp/classes/editor_spin_slider.hpp>
#include <godot_cpp/classes/file_dialog.hpp>
#include <godot_cpp/classes/h_box_container.hpp>
#include <godot_cpp/classes/h_split_container.hpp>
#include <godot_cpp/classes/input_event.hpp>
#include <godot_cpp/classes/link_button.hpp>
#include <godot_cpp/classes/menu_button.hpp>
#include <godot_cpp/classes/panel.hpp>
#include <godot_cpp/classes/popup_menu.hpp>
#include <godot_cpp/classes/tab_bar.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/variant.hpp>

using namespace godot;

#define GET_UNDO_REDO() plugin->get_undo_redo()

#endif // LIMBOAI_GDEXTENSION

class LimboAIEditor : public Control {
	GDCLASS(LimboAIEditor, Control);

private:
	enum Action {
		ACTION_EDIT_PROBABILITY,
		ACTION_RENAME,
		ACTION_CHANGE_TYPE,
		ACTION_EDIT_SCRIPT,
		ACTION_OPEN_DOC,
		ACTION_CUT,
		ACTION_COPY,
		ACTION_PASTE,
		ACTION_PASTE_AFTER,
		ACTION_MOVE_UP,
		ACTION_MOVE_DOWN,
		ACTION_DUPLICATE,
		ACTION_MAKE_ROOT,
		ACTION_EXTRACT_SUBTREE,
		ACTION_REMOVE,
	};

	enum MiscMenu {
		MISC_ONLINE_DOCUMENTATION,
		MISC_DOC_INTRODUCTION,
		MISC_DOC_CUSTOM_TASKS,
		MISC_OPEN_DEBUGGER,
		MISC_LAYOUT_CLASSIC,
		MISC_LAYOUT_WIDESCREEN_OPTIMIZED,
		MISC_PROJECT_SETTINGS,
		MISC_CREATE_SCRIPT_TEMPLATE,
	};

	enum TabMenu {
		TAB_SHOW_IN_FILESYSTEM,
		TAB_JUMP_TO_OWNER,
		TAB_CLOSE,
		TAB_CLOSE_OTHER,
		TAB_CLOSE_RIGHT,
		TAB_CLOSE_ALL,
	};

	enum EditorLayout {
		LAYOUT_CLASSIC,
		LAYOUT_WIDESCREEN_OPTIMIZED,
	};

	struct ThemeCache {
		Ref<Texture2D> duplicate_task_icon;
		Ref<Texture2D> edit_script_icon;
		Ref<Texture2D> make_root_icon;
		Ref<Texture2D> move_task_down_icon;
		Ref<Texture2D> move_task_up_icon;
		Ref<Texture2D> open_debugger_icon;
		Ref<Texture2D> doc_icon;
		Ref<Texture2D> introduction_icon;
		Ref<Texture2D> percent_icon;
		Ref<Texture2D> remove_task_icon;
		Ref<Texture2D> rename_task_icon;
		Ref<Texture2D> change_type_icon;
		Ref<Texture2D> extract_subtree_icon;
		Ref<Texture2D> behavior_tree_icon;
		Ref<Texture2D> cut_icon;
		Ref<Texture2D> copy_icon;
		Ref<Texture2D> paste_icon;
	} theme_cache;

	EditorPlugin *plugin;
	EditorLayout editor_layout;
	Vector<Ref<BehaviorTree>> history;
	int idx_history;
	bool updating_tabs = false;
	HashSet<Ref<BehaviorTree>> dirty;
	Ref<BTTask> clipboard_task;

	VBoxContainer *vbox;
	PanelContainer *tab_bar_panel;
	HBoxContainer *tab_bar_container;
	LinkButton *version_btn;
	TabBar *tab_bar;
	PopupMenu *tab_menu;
	OwnerPicker *owner_picker;
	HSplitContainer *hsc;
	TaskTree *task_tree;
	VBoxContainer *banners;
	Panel *usage_hint;
	PopupMenu *menu;
	HBoxContainer *fav_tasks_hbox;
	TaskPalette *task_palette;

	PopupPanel *probability_popup;
	EditorSpinSlider *probability_edit;
	Button *weight_mode;
	Button *percent_mode;

	PopupPanel *change_type_popup;
	TaskPalette *change_type_palette;

	FileDialog *save_dialog;
	FileDialog *load_dialog;
	FileDialog *extract_dialog;

	Button *new_btn;
	Button *load_btn;
	Button *save_btn;
	Button *new_script_btn;
	MenuButton *misc_btn;

	ConfirmationDialog *rename_dialog;
	LineEdit *rename_edit;

	ConfirmationDialog *disk_changed;
	Tree *disk_changed_list;
	HashSet<String> disk_changed_files;

	AcceptDialog *info_dialog;

	void _add_task(const Ref<BTTask> &p_task, bool p_as_sibling);
	void _add_task_with_prototype(const Ref<BTTask> &p_prototype);
	Ref<BTTask> _create_task_by_class_or_path(const String &p_class_or_path) const;
	void _add_task_by_class_or_path(const String &p_class_or_path);
	void _remove_task(const Ref<BTTask> &p_task);
	void _update_favorite_tasks();
	void _update_misc_menu();
	void _update_banners();
	void _new_bt();
	void _save_bt(String p_path);
	void _load_bt(String p_path);
	void _disable_editing();
	void _mark_as_dirty(bool p_dirty);
	void _create_user_task_dir();
	void _remove_task_from_favorite(const String &p_task);
	void _save_and_restart();
	void _extract_subtree(const String &p_path);
	void _replace_task(const Ref<BTTask> &p_task, const Ref<BTTask> &p_by_task);

	void _tab_clicked(int p_tab);
	void _tab_closed(int p_tab);
	void _update_tabs();
	void _move_active_tab(int p_to_index);
	void _tab_input(const Ref<InputEvent> &p_input);
	void _show_tab_context_menu();
	void _tab_menu_option_selected(int p_id);
	void _tab_plan_edited(int p_tab);

	void _reload_modified();
	void _resave_modified(String _str = "");
	void _popup_file_dialog(FileDialog *p_dialog) { p_dialog->popup_centered_clamped(Size2i(700, 500), 0.8f); }
	void _popup_info_dialog(const String &p_text);

	void _rename_task_confirmed();
	void _on_tree_rmb(const Vector2 &p_menu_pos);
	void _action_selected(int p_id);
	void _misc_option_selected(int p_id);
	void _on_probability_edited(double p_value);
	void _update_probability_edit();
	void _probability_popup_closed();
	void _on_tree_task_selected(const Ref<BTTask> &p_task);
	void _on_visibility_changed();
	void _on_header_pressed();
	void _on_save_pressed();
	void _on_history_back();
	void _on_history_forward();
	void _on_task_dragged(Ref<BTTask> p_task, Ref<BTTask> p_to_task, int p_type);
	void _on_resources_reload(const PackedStringArray &p_resources);
	void _on_new_script_pressed();
	void _task_type_selected(const String &p_class_or_path);
	void _copy_version_info();

	void _edit_project_settings();
	void _process_shortcut_input(const Ref<InputEvent> &p_event);

#ifdef LIMBOAI_MODULE
	virtual void shortcut_input(const Ref<InputEvent> &p_event) override { _process_shortcut_input(p_event); }
#endif // LIMBOAI_MODULE

protected:
	virtual void _do_update_theme_item_cache();

	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_plugin(EditorPlugin *p_plugin) { plugin = p_plugin; };
	void edit_bt(Ref<BehaviorTree> p_behavior_tree, bool p_force_refresh = false);
	Ref<BlackboardPlan> get_edited_blackboard_plan();

	void apply_changes();

#ifdef LIMBOAI_GDEXTENSION
	virtual void _shortcut_input(const Ref<InputEvent> &p_event) override { _process_shortcut_input(p_event); }
#endif

	LimboAIEditor();
	~LimboAIEditor();
};

class LimboAIEditorPlugin : public EditorPlugin {
	GDCLASS(LimboAIEditorPlugin, EditorPlugin);

private:
	LimboAIEditor *limbo_ai_editor;

protected:
	static void _bind_methods();
	void _notification(int p_notification);

public:
#ifdef LIMBOAI_MODULE
	bool has_main_screen() const override { return true; }

	virtual String get_name() const override { return "LimboAI"; }
	virtual void make_visible(bool p_visible) override;
	virtual void apply_changes() override;
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;

#elif LIMBOAI_GDEXTENSION
	bool _has_main_screen() const override { return true; }

	virtual String _get_plugin_name() const override { return "LimboAI"; }
	virtual void _make_visible(bool p_visible) override;
	virtual void _apply_changes() override;
	virtual void _edit(Object *p_object) override;
	virtual bool _handles(Object *p_object) const override;
	virtual Ref<Texture2D> _get_plugin_icon() const override;
#endif // LIMBOAI_MODULE & LIMBOAI_GDEXTENSION

	LimboAIEditorPlugin();
	~LimboAIEditorPlugin();
};

#endif // LIMBO_AI_EDITOR_PLUGIN_H

#endif // ! TOOLS_ENABLED
