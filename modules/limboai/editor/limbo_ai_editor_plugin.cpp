/**
 * limbo_ai_editor_plugin.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#include "limbo_ai_editor_plugin.h"

#include "../bt/behavior_tree.h"
#include "../bt/tasks/bt_comment.h"
#include "../bt/tasks/composites/bt_probability_selector.h"
#include "../bt/tasks/composites/bt_selector.h"
#include "../bt/tasks/decorators/bt_subtree.h"
#include "../util/limbo_compat.h"
#include "../util/limbo_utility.h"
#include "../util/limboai_version.h"
#include "action_banner.h"
#include "blackboard_plan_editor.h"
#include "debugger/limbo_debugger_plugin.h"
#include "editor_property_bb_param.h"

#ifdef LIMBOAI_MODULE
#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/input/input.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/editor_file_system.h"
#include "editor/editor_help.h"
#include "editor/editor_interface.h"
#include "editor/editor_paths.h"
#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/filesystem_dock.h"
#include "editor/gui/editor_bottom_panel.h"
#include "editor/inspector_dock.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/project_settings_editor.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/button_group.hpp>
#include <godot_cpp/classes/config_file.hpp>
#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/display_server.hpp>
#include <godot_cpp/classes/editor_file_system.hpp>
#include <godot_cpp/classes/editor_inspector.hpp>
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/editor_paths.hpp>
#include <godot_cpp/classes/editor_settings.hpp>
#include <godot_cpp/classes/editor_undo_redo_manager.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/file_system_dock.hpp>
#include <godot_cpp/classes/input.hpp>
#include <godot_cpp/classes/input_event.hpp>
#include <godot_cpp/classes/input_event_mouse_button.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/resource_saver.hpp>
#include <godot_cpp/classes/script.hpp>
#include <godot_cpp/classes/script_editor.hpp>
#include <godot_cpp/classes/script_editor_base.hpp>
#include <godot_cpp/classes/v_separator.hpp>
#include <godot_cpp/core/error_macros.hpp>
#endif // LIMBOAI_GDEXTENSION

//**** LimboAIEditor

_FORCE_INLINE_ String _get_script_template_path() {
	String templates_search_path = GLOBAL_GET("editor/script/templates_search_path");
	return templates_search_path.path_join("BTTask").path_join("custom_task.gd");
}

void LimboAIEditor::_add_task(const Ref<BTTask> &p_task, bool p_as_sibling) {
	if (task_tree->get_bt().is_null()) {
		return;
	}
	ERR_FAIL_COND(p_task.is_null());

	EditorUndoRedoManager *undo_redo = GET_UNDO_REDO();
	undo_redo->create_action(TTR("Add BT Task"));

	int insert_idx = -1;
	Ref<BTTask> selected = task_tree->get_selected();
	Ref<BTTask> parent = selected;
	if (parent.is_null()) {
		// When no task is selected, use the root task.
		parent = task_tree->get_bt()->get_root_task();
		selected = parent;
	}
	if (parent.is_null()) {
		// When tree is empty.
		undo_redo->add_do_method(task_tree->get_bt().ptr(), LW_NAME(set_root_task), p_task);
		undo_redo->add_undo_method(task_tree->get_bt().ptr(), LW_NAME(set_root_task), task_tree->get_bt()->get_root_task());
	} else {
		if (p_as_sibling && selected.is_valid() && selected->get_parent().is_valid()) {
			// Insert task after the currently selected and on the same level (usually when shift is pressed).
			parent = selected->get_parent();
			insert_idx = selected->get_index() + 1;
		}
		undo_redo->add_do_method(parent.ptr(), LW_NAME(add_child_at_index), p_task, insert_idx);
		undo_redo->add_undo_method(parent.ptr(), LW_NAME(remove_child), p_task);
	}
	undo_redo->add_do_method(task_tree, LW_NAME(update_tree));
	undo_redo->add_undo_method(task_tree, LW_NAME(update_tree));

	undo_redo->commit_action();
	_mark_as_dirty(true);
}

void LimboAIEditor::_add_task_with_prototype(const Ref<BTTask> &p_prototype) {
	Ref<BTTask> selected = task_tree->get_selected();
	bool as_sibling = Input::get_singleton()->is_key_pressed(LW_KEY(SHIFT));
	_add_task(p_prototype->clone(), as_sibling);
}

Ref<BTTask> LimboAIEditor::_create_task_by_class_or_path(const String &p_class_or_path) const {
	ERR_FAIL_COND_V(p_class_or_path.is_empty(), nullptr);

	Ref<BTTask> ret;

	if (p_class_or_path.begins_with("res:")) {
		Ref<Script> s = RESOURCE_LOAD(p_class_or_path, "Script");
		ERR_FAIL_COND_V_MSG(s.is_null(), nullptr, vformat("LimboAI: Can't add task. Bad script: %s", p_class_or_path));
		StringName base_type = s->get_instance_base_type();
		if (base_type == StringName()) {
			// Try reloading script.
			s->reload(true);
			base_type = s->get_instance_base_type();
		}
		ERR_FAIL_COND_V_MSG(base_type == StringName(), nullptr, vformat("LimboAI: Can't add task. Bad script: %s", p_class_or_path));

		Variant inst = ClassDB::instantiate(base_type);
		Object *obj = inst;
		ERR_FAIL_NULL_V_MSG(obj, nullptr, vformat("LimboAI: Can't add task. Failed to create base type \"%s\".", base_type));

		if (unlikely(!IS_CLASS(obj, BTTask))) {
			ERR_PRINT_ED(vformat("LimboAI: Can't add task. Script is not a BTTask: %s", p_class_or_path));
			VARIANT_DELETE_IF_OBJECT(inst);
			return nullptr;
		}

		ret.reference_ptr(Object::cast_to<BTTask>(obj));
		ret->set_script(s);
	} else {
		ERR_FAIL_COND_V(!ClassDB::is_parent_class(p_class_or_path, "BTTask"), nullptr);
		ret = ClassDB::instantiate(p_class_or_path);
	}
	return ret;
}

void LimboAIEditor::_add_task_by_class_or_path(const String &p_class_or_path) {
	Ref<BTTask> selected = task_tree->get_selected();
	bool as_sibling = Input::get_singleton()->is_key_pressed(LW_KEY(SHIFT));
	_add_task(_create_task_by_class_or_path(p_class_or_path), as_sibling);
}

void LimboAIEditor::_remove_task(const Ref<BTTask> &p_task) {
	ERR_FAIL_COND(p_task.is_null());
	ERR_FAIL_COND(task_tree->get_bt().is_null());
	EditorUndoRedoManager *undo_redo = GET_UNDO_REDO();
	undo_redo->create_action(TTR("Remove BT Task"));
	if (p_task->get_parent() == nullptr) {
		ERR_FAIL_COND(task_tree->get_bt()->get_root_task() != p_task);
		undo_redo->add_do_method(task_tree->get_bt().ptr(), LW_NAME(set_root_task), Variant());
		undo_redo->add_undo_method(task_tree->get_bt().ptr(), LW_NAME(set_root_task), task_tree->get_bt()->get_root_task());
	} else {
		undo_redo->add_do_method(p_task->get_parent().ptr(), LW_NAME(remove_child), p_task);
		undo_redo->add_undo_method(p_task->get_parent().ptr(), LW_NAME(add_child_at_index), p_task, p_task->get_index());
	}
	undo_redo->add_do_method(task_tree, LW_NAME(update_tree));
	undo_redo->add_undo_method(task_tree, LW_NAME(update_tree));
	undo_redo->commit_action();
}

void LimboAIEditor::_new_bt() {
	Ref<BehaviorTree> bt = memnew(BehaviorTree);
	bt->set_root_task(memnew(BTSelector));
	bt->set_blackboard_plan(memnew(BlackboardPlan));
	EDIT_RESOURCE(bt);
}

void LimboAIEditor::_save_bt(String p_path) {
	ERR_FAIL_COND_MSG(p_path.is_empty(), "Empty p_path");
	ERR_FAIL_COND_MSG(task_tree->get_bt().is_null(), "Behavior Tree is null.");
#ifdef LIMBOAI_MODULE
	task_tree->get_bt()->set_path(p_path, true);
#elif LIMBOAI_GDEXTENSION
	task_tree->get_bt()->take_over_path(p_path);
#endif
	RESOURCE_SAVE(task_tree->get_bt(), p_path, ResourceSaver::FLAG_CHANGE_PATH);
	_update_tabs();
	_mark_as_dirty(false);
}

void LimboAIEditor::_load_bt(String p_path) {
	ERR_FAIL_COND_MSG(p_path.is_empty(), "Empty p_path");
	Ref<BehaviorTree> bt = RESOURCE_LOAD(p_path, "BehaviorTree");
	ERR_FAIL_COND(!bt.is_valid());
	if (bt->get_blackboard_plan().is_null()) {
		bt->set_blackboard_plan(memnew(BlackboardPlan));
	}
	// if (history.find(bt) != -1) {
	// 	history.erase(bt);
	// 	history.push_back(bt);
	// }

	EDIT_RESOURCE(bt);
}

void LimboAIEditor::_disable_editing() {
	task_tree->unload();
	task_palette->hide();
	task_tree->hide();
	usage_hint->show();
}

void LimboAIEditor::edit_bt(Ref<BehaviorTree> p_behavior_tree, bool p_force_refresh) {
	ERR_FAIL_COND_MSG(p_behavior_tree.is_null(), "p_behavior_tree is null");

	if (!p_force_refresh && task_tree->get_bt() == p_behavior_tree) {
		return;
	}

#ifdef LIMBOAI_MODULE
	p_behavior_tree->editor_set_section_unfold("blackboard_plan", true);
	p_behavior_tree->notify_property_list_changed();
#endif // LIMBOAI_MODULE

	task_tree->load_bt(p_behavior_tree);

	if (task_tree->get_bt().is_valid() && !task_tree->get_bt()->is_connected(LW_NAME(changed), callable_mp(this, &LimboAIEditor::_mark_as_dirty))) {
		task_tree->get_bt()->connect(LW_NAME(changed), callable_mp(this, &LimboAIEditor::_mark_as_dirty).bind(true));
	}

	int idx = history.find(p_behavior_tree);
	if (idx != -1) {
		idx_history = idx;
	} else {
		history.push_back(p_behavior_tree);
		idx_history = history.size() - 1;
	}

	usage_hint->hide();
	task_tree->show();
	task_palette->show();

	_update_tabs();
}

Ref<BlackboardPlan> LimboAIEditor::get_edited_blackboard_plan() {
	if (task_tree->get_bt().is_null()) {
		return nullptr;
	}
	if (task_tree->get_bt()->get_blackboard_plan().is_null()) {
		task_tree->get_bt()->set_blackboard_plan(memnew(BlackboardPlan));
	}
	return task_tree->get_bt()->get_blackboard_plan();
}

void LimboAIEditor::_mark_as_dirty(bool p_dirty) {
	Ref<BehaviorTree> bt = task_tree->get_bt();
	if (p_dirty && !dirty.has(bt)) {
		dirty.insert(bt);
	} else if (p_dirty == false && dirty.has(bt)) {
		dirty.erase(bt);
	}
}

void LimboAIEditor::_create_user_task_dir() {
	String task_dir = GLOBAL_GET("limbo_ai/behavior_tree/user_task_dir_1");
	ERR_FAIL_COND_MSG(DirAccess::dir_exists_absolute(task_dir), "LimboAIEditor: Directory already exists: " + task_dir);

	Error err = DirAccess::make_dir_recursive_absolute(task_dir);
	ERR_FAIL_COND_MSG(err != OK, "LimboAIEditor: Failed to create directory: " + task_dir);

#ifdef LIMBOAI_MODULE
	EditorFileSystem::get_singleton()->scan_changes();
#elif LIMBOAI_GDEXTENSION
	EditorInterface::get_singleton()->get_resource_filesystem()->scan_sources();
#endif
	_update_banners();
}

void LimboAIEditor::_edit_project_settings() {
#ifdef LIMBOAI_MODULE
	ProjectSettingsEditor::get_singleton()->set_general_page("limbo_ai/behavior_tree");
	ProjectSettingsEditor::get_singleton()->popup_project_settings();
	ProjectSettingsEditor::get_singleton()->connect(LW_NAME(visibility_changed), callable_mp(this, &LimboAIEditor::_update_banners), CONNECT_ONE_SHOT);
#elif LIMBOAI_GDEXTENSION
	// TODO: Find a way to show project setting in GDExtension.
	String text = "Can't open settings in GDExtension, sorry :(\n"
				  "To edit project settings, navigate to \"Project->Project Settings\",\n"
				  "enable \"Advanced settings\", and scroll down to the \"LimboAI\" section.";
	_popup_info_dialog(text);
#endif
}

void LimboAIEditor::_remove_task_from_favorite(const String &p_task) {
	PackedStringArray favorite_tasks = GLOBAL_GET("limbo_ai/behavior_tree/favorite_tasks");
	int idx = favorite_tasks.find(p_task);
	if (idx >= 0) {
		favorite_tasks.remove_at(idx);
	}
	ProjectSettings::get_singleton()->set_setting("limbo_ai/behavior_tree/favorite_tasks", favorite_tasks);
	ProjectSettings::get_singleton()->save();
}

void LimboAIEditor::_save_and_restart() {
	ProjectSettings::get_singleton()->save();
	EditorInterface::get_singleton()->save_all_scenes();
	EditorInterface::get_singleton()->restart_editor(true);
}

void LimboAIEditor::_extract_subtree(const String &p_path) {
	Ref<BTTask> selected = task_tree->get_selected();
	ERR_FAIL_COND(selected.is_null());

	EditorUndoRedoManager *undo_redo = GET_UNDO_REDO();
	undo_redo->create_action(TTR("Extract Subtree"));

	Ref<BehaviorTree> bt = memnew(BehaviorTree);
	bt->set_root_task(selected->clone());
	bt->set_path(p_path);
	RESOURCE_SAVE(bt, p_path, ResourceSaver::FLAG_CHANGE_PATH);

	Ref<BTSubtree> subtree = memnew(BTSubtree);
	subtree->set_subtree(bt);

	if (selected->is_root()) {
		undo_redo->add_do_method(task_tree->get_bt().ptr(), LW_NAME(set_root_task), subtree);
		undo_redo->add_undo_method(task_tree->get_bt().ptr(), LW_NAME(set_root_task), selected);
	} else {
		int idx = selected->get_index();
		undo_redo->add_do_method(selected->get_parent().ptr(), LW_NAME(remove_child), selected);
		undo_redo->add_do_method(selected->get_parent().ptr(), LW_NAME(add_child_at_index), subtree, idx);
		undo_redo->add_undo_method(selected->get_parent().ptr(), LW_NAME(remove_child), subtree);
		undo_redo->add_undo_method(selected->get_parent().ptr(), LW_NAME(add_child_at_index), selected, idx);
	}
	undo_redo->add_do_method(task_tree, LW_NAME(update_tree));
	undo_redo->add_undo_method(task_tree, LW_NAME(update_tree));

	undo_redo->commit_action();
	EDIT_RESOURCE(task_tree->get_selected());
	_mark_as_dirty(true);
}

void LimboAIEditor::_process_shortcut_input(const Ref<InputEvent> &p_event) {
	if (!p_event->is_pressed() || p_event->is_echo()) {
		return;
	}

	bool handled = false;

	// * Global shortcuts.

	if (LW_IS_SHORTCUT("limbo_ai/open_debugger", p_event)) {
		_misc_option_selected(MISC_OPEN_DEBUGGER);
		handled = true;
	}

	// * When editor is on screen.

	if (!handled && is_visible_in_tree()) {
		if (LW_IS_SHORTCUT("limbo_ai/jump_to_owner", p_event)) {
			_tab_menu_option_selected(TAB_JUMP_TO_OWNER);
			handled = true;
		} else if (LW_IS_SHORTCUT("limbo_ai/close_tab", p_event)) {
			_tab_menu_option_selected(TAB_CLOSE);
			handled = true;
		}
	}

	// * When editor is focused.

	if (!handled && (has_focus() || (get_viewport()->gui_get_focus_owner() && is_ancestor_of(get_viewport()->gui_get_focus_owner())))) {
		handled = true;
		if (LW_IS_SHORTCUT("limbo_ai/rename_task", p_event)) {
			_action_selected(ACTION_RENAME);
		} else if (LW_IS_SHORTCUT("limbo_ai/cut_task", p_event)) {
			_action_selected(ACTION_CUT);
		} else if (LW_IS_SHORTCUT("limbo_ai/copy_task", p_event)) {
			_action_selected(ACTION_COPY);
		} else if (LW_IS_SHORTCUT("limbo_ai/paste_task", p_event)) {
			_action_selected(ACTION_PASTE);
		} else if (LW_IS_SHORTCUT("limbo_ai/paste_task_after", p_event)) {
			_action_selected(ACTION_PASTE_AFTER);
		} else if (LW_IS_SHORTCUT("limbo_ai/move_task_up", p_event)) {
			_action_selected(ACTION_MOVE_UP);
		} else if (LW_IS_SHORTCUT("limbo_ai/move_task_down", p_event)) {
			_action_selected(ACTION_MOVE_DOWN);
		} else if (LW_IS_SHORTCUT("limbo_ai/duplicate_task", p_event)) {
			_action_selected(ACTION_DUPLICATE);
		} else if (LW_IS_SHORTCUT("limbo_ai/remove_task", p_event)) {
			_action_selected(ACTION_REMOVE);
		} else if (LW_IS_SHORTCUT("limbo_ai/new_behavior_tree", p_event)) {
			_new_bt();
		} else if (LW_IS_SHORTCUT("limbo_ai/save_behavior_tree", p_event)) {
			_on_save_pressed();
		} else if (LW_IS_SHORTCUT("limbo_ai/load_behavior_tree", p_event)) {
			_popup_file_dialog(load_dialog);
		} else {
			handled = false;
		}
	}

	if (handled) {
		get_viewport()->set_input_as_handled();
	}
}

void LimboAIEditor::_on_tree_rmb(const Vector2 &p_menu_pos) {
	menu->clear();

	Ref<BTTask> task = task_tree->get_selected();
	ERR_FAIL_COND_MSG(task.is_null(), "LimboAIEditor: get_selected() returned null");

	if (task_tree->selected_has_probability()) {
		menu->add_icon_item(theme_cache.percent_icon, TTR("Edit Probability"), ACTION_EDIT_PROBABILITY);
	}
	menu->add_icon_shortcut(theme_cache.rename_task_icon, LW_GET_SHORTCUT("limbo_ai/rename_task"), ACTION_RENAME);
	menu->add_icon_item(theme_cache.change_type_icon, TTR("Change Type"), ACTION_CHANGE_TYPE);
	menu->add_icon_item(theme_cache.edit_script_icon, TTR("Edit Script"), ACTION_EDIT_SCRIPT);
	menu->add_icon_item(theme_cache.doc_icon, TTR("Open Documentation"), ACTION_OPEN_DOC);
	menu->set_item_disabled(menu->get_item_index(ACTION_EDIT_SCRIPT), task->get_script() == Variant());

	menu->add_separator();
	menu->add_icon_shortcut(theme_cache.cut_icon, LW_GET_SHORTCUT("limbo_ai/cut_task"), ACTION_CUT);
	menu->add_icon_shortcut(theme_cache.copy_icon, LW_GET_SHORTCUT("limbo_ai/copy_task"), ACTION_COPY);
	menu->add_icon_shortcut(theme_cache.paste_icon, LW_GET_SHORTCUT("limbo_ai/paste_task"), ACTION_PASTE);
	menu->add_icon_shortcut(theme_cache.paste_icon, LW_GET_SHORTCUT("limbo_ai/paste_task_after"), ACTION_PASTE_AFTER);
	menu->set_item_disabled(ACTION_PASTE, clipboard_task.is_null());
	menu->set_item_disabled(ACTION_PASTE_AFTER, clipboard_task.is_null());

	menu->add_separator();
	menu->add_icon_shortcut(theme_cache.move_task_up_icon, LW_GET_SHORTCUT("limbo_ai/move_task_up"), ACTION_MOVE_UP);
	menu->add_icon_shortcut(theme_cache.move_task_down_icon, LW_GET_SHORTCUT("limbo_ai/move_task_down"), ACTION_MOVE_DOWN);
	menu->add_icon_shortcut(theme_cache.duplicate_task_icon, LW_GET_SHORTCUT("limbo_ai/duplicate_task"), ACTION_DUPLICATE);
	menu->add_icon_item(theme_cache.make_root_icon, TTR("Make Root"), ACTION_MAKE_ROOT);
	menu->add_icon_item(theme_cache.extract_subtree_icon, TTR("Extract Subtree"), ACTION_EXTRACT_SUBTREE);

	menu->add_separator();
	menu->add_icon_shortcut(theme_cache.remove_task_icon, LW_GET_SHORTCUT("limbo_ai/remove_task"), ACTION_REMOVE);

	menu->reset_size();
	menu->set_position(p_menu_pos);
	menu->popup();
}

void LimboAIEditor::_action_selected(int p_id) {
	EditorUndoRedoManager *undo_redo = GET_UNDO_REDO();
	switch (p_id) {
		case ACTION_RENAME: {
			if (!task_tree->get_selected().is_valid()) {
				return;
			}
			Ref<BTTask> task = task_tree->get_selected();
			if (IS_CLASS(task, BTComment)) {
				rename_dialog->set_title(TTR("Edit Comment"));
				rename_dialog->get_ok_button()->set_text(TTR("OK"));
				rename_edit->set_placeholder(TTR("Comment"));
			} else {
				rename_dialog->set_title(TTR("Rename Task"));
				rename_dialog->get_ok_button()->set_text(TTR("Rename"));
				rename_edit->set_placeholder(TTR("Custom Name"));
			}
			rename_edit->set_text(task->get_custom_name());
			rename_dialog->popup_centered();
			rename_edit->select_all();
			rename_edit->grab_focus();
		} break;
		case ACTION_CHANGE_TYPE: {
			change_type_palette->clear_filter();
			change_type_palette->refresh();
			Rect2 rect = Rect2(get_global_mouse_position(), Size2(400.0, 600.0) * EDSCALE);
			change_type_popup->popup(rect);
		} break;
		case ACTION_EDIT_PROBABILITY: {
			Rect2 rect = task_tree->get_selected_probability_rect();
			ERR_FAIL_COND(rect == Rect2());
			rect.position.y += rect.size.y;
			rect.position += task_tree->get_rect().position;
			rect = task_tree->get_screen_transform().xform(rect);
			_update_probability_edit();
			probability_popup->popup(rect);
		} break;
		case ACTION_EDIT_SCRIPT: {
			ERR_FAIL_COND(task_tree->get_selected().is_null());
			EDIT_RESOURCE(task_tree->get_selected()->get_script());
		} break;
		case ACTION_OPEN_DOC: {
			Ref<BTTask> task = task_tree->get_selected();
			ERR_FAIL_COND(task.is_null());
			String help_class;

			Ref<Script> sc = GET_SCRIPT(task);
			if (sc.is_valid() && sc->get_path().is_absolute_path()) {
				help_class = sc->get_path();
			}
			if (help_class.is_empty()) {
				// Assuming context task is core class.
				help_class = task->get_class();
			}

			LimboUtility::get_singleton()->open_doc_class(help_class);
		} break;
		case ACTION_COPY: {
			Ref<BTTask> sel = task_tree->get_selected();
			if (sel.is_valid()) {
				clipboard_task = sel->clone();
			}
		} break;
		case ACTION_PASTE: {
			if (clipboard_task.is_valid()) {
				_add_task(clipboard_task->clone(), false);
			}
		} break;
		case ACTION_PASTE_AFTER: {
			if (clipboard_task.is_valid()) {
				_add_task(clipboard_task->clone(), true);
			}
		} break;
		case ACTION_MOVE_UP: {
			Ref<BTTask> sel = task_tree->get_selected();
			if (sel.is_valid() && sel->get_parent().is_valid()) {
				Ref<BTTask> parent = sel->get_parent();
				int idx = sel->get_index();
				if (idx > 0 && idx < parent->get_child_count()) {
					undo_redo->create_action(TTR("Move BT Task"));
					undo_redo->add_do_method(parent.ptr(), LW_NAME(remove_child), sel);
					undo_redo->add_do_method(parent.ptr(), LW_NAME(add_child_at_index), sel, idx - 1);
					undo_redo->add_undo_method(parent.ptr(), LW_NAME(remove_child), sel);
					undo_redo->add_undo_method(parent.ptr(), LW_NAME(add_child_at_index), sel, idx);
					undo_redo->add_do_method(task_tree, LW_NAME(update_tree));
					undo_redo->add_undo_method(task_tree, LW_NAME(update_tree));
					undo_redo->commit_action();
					_mark_as_dirty(true);
				}
			}
		} break;
		case ACTION_MOVE_DOWN: {
			Ref<BTTask> sel = task_tree->get_selected();
			if (sel.is_valid() && sel->get_parent().is_valid()) {
				Ref<BTTask> parent = sel->get_parent();
				int idx = sel->get_index();
				if (idx >= 0 && idx < (parent->get_child_count() - 1)) {
					undo_redo->create_action(TTR("Move BT Task"));
					undo_redo->add_do_method(parent.ptr(), LW_NAME(remove_child), sel);
					undo_redo->add_do_method(parent.ptr(), LW_NAME(add_child_at_index), sel, idx + 1);
					undo_redo->add_undo_method(parent.ptr(), LW_NAME(remove_child), sel);
					undo_redo->add_undo_method(parent.ptr(), LW_NAME(add_child_at_index), sel, idx);
					undo_redo->add_do_method(task_tree, LW_NAME(update_tree));
					undo_redo->add_undo_method(task_tree, LW_NAME(update_tree));
					undo_redo->commit_action();
					_mark_as_dirty(true);
				}
			}
		} break;
		case ACTION_DUPLICATE: {
			Ref<BTTask> sel = task_tree->get_selected();
			if (sel.is_valid()) {
				undo_redo->create_action(TTR("Duplicate BT Task"));
				Ref<BTTask> parent = sel->get_parent();
				if (parent.is_null()) {
					parent = sel;
				}
				const Ref<BTTask> &sel_dup = sel->clone();
				undo_redo->add_do_method(parent.ptr(), LW_NAME(add_child_at_index), sel_dup, sel->get_index() + 1);
				undo_redo->add_undo_method(parent.ptr(), LW_NAME(remove_child), sel_dup);
				undo_redo->add_do_method(task_tree, LW_NAME(update_tree));
				undo_redo->add_undo_method(task_tree, LW_NAME(update_tree));
				undo_redo->commit_action();
				_mark_as_dirty(true);
			}
		} break;
		case ACTION_MAKE_ROOT: {
			Ref<BTTask> sel = task_tree->get_selected();
			if (sel.is_valid() && task_tree->get_bt()->get_root_task() != sel) {
				Ref<BTTask> parent = sel->get_parent();
				ERR_FAIL_COND(parent.is_null());
				undo_redo->create_action(TTR("Make Root"));
				undo_redo->add_do_method(parent.ptr(), LW_NAME(remove_child), sel);
				Ref<BTTask> old_root = task_tree->get_bt()->get_root_task();
				undo_redo->add_do_method(task_tree->get_bt().ptr(), LW_NAME(set_root_task), sel);
				undo_redo->add_do_method(sel.ptr(), LW_NAME(add_child), old_root);
				undo_redo->add_undo_method(sel.ptr(), LW_NAME(remove_child), old_root);
				undo_redo->add_undo_method(task_tree->get_bt().ptr(), LW_NAME(set_root_task), old_root);
				undo_redo->add_undo_method(parent.ptr(), LW_NAME(add_child_at_index), sel, sel->get_index());
				undo_redo->add_do_method(task_tree, LW_NAME(update_tree));
				undo_redo->add_undo_method(task_tree, LW_NAME(update_tree));
				undo_redo->commit_action();
				_mark_as_dirty(true);
			}
		} break;
		case ACTION_EXTRACT_SUBTREE: {
			Ref<BTTask> sel = task_tree->get_selected();
			if (sel.is_valid() && !IS_CLASS(sel, BTSubtree)) {
				extract_dialog->popup_centered_ratio();
			}
		} break;
		case ACTION_CUT:
		case ACTION_REMOVE: {
			Ref<BTTask> sel = task_tree->get_selected();
			if (sel.is_valid()) {
				if (p_id == ACTION_CUT) {
					clipboard_task = sel->clone();
				}

				undo_redo->create_action(TTR("Remove BT Task"));
				if (sel->is_root()) {
					undo_redo->add_do_method(task_tree->get_bt().ptr(), LW_NAME(set_root_task), Variant());
					undo_redo->add_undo_method(task_tree->get_bt().ptr(), LW_NAME(set_root_task), task_tree->get_bt()->get_root_task());
				} else {
					undo_redo->add_do_method(sel->get_parent().ptr(), LW_NAME(remove_child), sel);
					undo_redo->add_undo_method(sel->get_parent().ptr(), LW_NAME(add_child_at_index), sel, sel->get_index());
				}
				undo_redo->add_do_method(task_tree, LW_NAME(update_tree));
				undo_redo->add_undo_method(task_tree, LW_NAME(update_tree));
				undo_redo->commit_action();
				EDIT_RESOURCE(task_tree->get_selected());
				_mark_as_dirty(true);
			}
		} break;
	}
}

void LimboAIEditor::_on_probability_edited(double p_value) {
	Ref<BTTask> selected = task_tree->get_selected();
	ERR_FAIL_COND(selected == nullptr);
	Ref<BTProbabilitySelector> probability_selector = selected->get_parent();
	ERR_FAIL_COND(probability_selector.is_null());
	if (percent_mode->is_pressed()) {
		probability_selector->set_probability(selected->get_index(), p_value * 0.01);
	} else {
		probability_selector->set_weight(selected->get_index(), p_value);
	}
}

void LimboAIEditor::_update_probability_edit() {
	Ref<BTTask> selected = task_tree->get_selected();
	ERR_FAIL_COND(selected.is_null());
	Ref<BTProbabilitySelector> prob = selected->get_parent();
	ERR_FAIL_COND(prob.is_null());
	double others_weight = prob->get_total_weight() - prob->get_weight(selected->get_index());
	bool cannot_edit_percent = others_weight == 0.0;
	percent_mode->set_disabled(cannot_edit_percent);
	if (cannot_edit_percent && percent_mode->is_pressed()) {
		weight_mode->set_pressed(true);
	}

	if (percent_mode->is_pressed()) {
		probability_edit->set_suffix("%");
		probability_edit->set_max(99.0);
		probability_edit->set_allow_greater(false);
		probability_edit->set_step(0.01);
		probability_edit->set_value_no_signal(task_tree->get_selected_probability_percent());
	} else {
		probability_edit->set_suffix("");
		probability_edit->set_allow_greater(true);
		probability_edit->set_max(10.0);
		probability_edit->set_step(0.01);
		probability_edit->set_value_no_signal(task_tree->get_selected_probability_weight());
	}
}

void LimboAIEditor::_probability_popup_closed() {
	probability_edit->grab_focus(); // Hack: Workaround for an EditorSpinSlider bug keeping LineEdit visible and "stuck" with ghost value.
}

void LimboAIEditor::_misc_option_selected(int p_id) {
	switch (p_id) {
		case MISC_ONLINE_DOCUMENTATION: {
			LimboUtility::get_singleton()->open_doc_online();
		} break;
		case MISC_DOC_INTRODUCTION: {
			LimboUtility::get_singleton()->open_doc_introduction();
		} break;
		case MISC_DOC_CUSTOM_TASKS: {
			LimboUtility::get_singleton()->open_doc_custom_tasks();
		} break;
		case MISC_OPEN_DEBUGGER: {
			ERR_FAIL_COND(LimboDebuggerPlugin::get_singleton() == nullptr);
			if (LimboDebuggerPlugin::get_singleton()->get_first_session_window()->get_window_enabled()) {
				LimboDebuggerPlugin::get_singleton()->get_first_session_window()->set_window_enabled(true);
			} else {
#ifdef LIMBOAI_MODULE
				EditorNode::get_bottom_panel()->make_item_visible(EditorDebuggerNode::get_singleton());
				EditorDebuggerNode::get_singleton()->get_default_debugger()->switch_to_debugger(
						LimboDebuggerPlugin::get_singleton()->get_first_session_tab_index());
#elif LIMBOAI_GDEXTENSION
				// TODO: Unsure how to switch to debugger pane with GDExtension.
#endif
			}
		} break;
		case MISC_PROJECT_SETTINGS: {
			_edit_project_settings();
		} break;
		case MISC_LAYOUT_CLASSIC: {
			EDITOR_SETTINGS()->set_setting("limbo_ai/editor/layout", LAYOUT_CLASSIC);
			EDITOR_SETTINGS()->mark_setting_changed("limbo_ai/editor/layout");
			_update_banners();
		} break;
		case MISC_LAYOUT_WIDESCREEN_OPTIMIZED: {
			EDITOR_SETTINGS()->set_setting("limbo_ai/editor/layout", LAYOUT_WIDESCREEN_OPTIMIZED);
			EDITOR_SETTINGS()->mark_setting_changed("limbo_ai/editor/layout");
			_update_banners();
		} break;
		case MISC_CREATE_SCRIPT_TEMPLATE: {
			String template_path = _get_script_template_path();
			String template_dir = template_path.get_base_dir();

			if (!FILE_EXISTS(template_path)) {
				if (!DirAccess::dir_exists_absolute(template_dir)) {
					Error err = DirAccess::make_dir_recursive_absolute(template_dir);
					ERR_FAIL_COND(err != OK);
				}

				Ref<FileAccess> f = FileAccess::open(template_path, FileAccess::WRITE);
				ERR_FAIL_COND(f.is_null());

				String script_template =
						"# meta-name: Custom Task\n"
						"# meta-description: Custom task to be used in a BehaviorTree\n"
						"# meta-default: true\n"
						"@tool\n"
						"extends _BASE_\n"
						"## _CLASS_\n"
						"\n\n"
						"# Display a customized name (requires @tool).\n"
						"func _generate_name() -> String:\n"
						"_TS_return \"_CLASS_\"\n"
						"\n\n"
						"# Called once during initialization.\n"
						"func _setup() -> void:\n"
						"_TS_pass\n"
						"\n\n"
						"# Called each time this task is entered.\n"
						"func _enter() -> void:\n"
						"_TS_pass\n"
						"\n\n"
						"# Called each time this task is exited.\n"
						"func _exit() -> void:\n"
						"_TS_pass\n"
						"\n\n"
						"# Called each time this task is ticked (aka executed).\n"
						"func _tick(delta: float) -> Status:\n"
						"_TS_return SUCCESS\n"
						"\n\n"
						"# Strings returned from this method are displayed as warnings in the behavior tree editor (requires @tool).\n"
						"func _get_configuration_warnings() -> PackedStringArray:\n"
						"_TS_var warnings := PackedStringArray()\n"
						"_TS_return warnings\n";

				f->store_string(script_template);
				f->close();
			}

			EDITOR_FILE_SYSTEM()->scan();
			EDIT_SCRIPT(template_path);
		} break;
	}
}

void LimboAIEditor::_on_tree_task_selected(const Ref<BTTask> &p_task) {
	EDIT_RESOURCE(p_task);
}

void LimboAIEditor::_on_visibility_changed() {
	if (task_tree->is_visible_in_tree()) {
		Ref<BTTask> sel = task_tree->get_selected();
		if (sel.is_valid()) {
			EDIT_RESOURCE(sel);
		} else if (task_tree->get_bt().is_valid() && INSPECTOR_GET_EDITED_OBJECT() != task_tree->get_bt().ptr()) {
			EDIT_RESOURCE(task_tree->get_bt());
		}

		task_palette->refresh();
	}
	_update_favorite_tasks();
}

void LimboAIEditor::_on_header_pressed() {
	task_tree->deselect();
#ifdef LIMBOAI_MODULE
	if (task_tree->get_bt().is_valid()) {
		task_tree->get_bt()->editor_set_section_unfold("blackboard_plan", true);
	}
#endif // LIMBOAI_MODULE
	EDIT_RESOURCE(task_tree->get_bt());
}

void LimboAIEditor::_on_save_pressed() {
	if (task_tree->get_bt().is_null()) {
		return;
	}
	String path = task_tree->get_bt()->get_path();
	if (path.is_empty()) {
		save_dialog->popup_centered_ratio();
	} else {
		_save_bt(path);
	}
}

void LimboAIEditor::_on_history_back() {
	ERR_FAIL_COND(history.size() == 0);
	idx_history = MAX(idx_history - 1, 0);
	EDIT_RESOURCE(history[idx_history]);
}

void LimboAIEditor::_on_history_forward() {
	ERR_FAIL_COND(history.size() == 0);
	idx_history = MIN(idx_history + 1, history.size() - 1);
	EDIT_RESOURCE(history[idx_history]);
}

void LimboAIEditor::_on_task_dragged(Ref<BTTask> p_task, Ref<BTTask> p_to_task, int p_type) {
	ERR_FAIL_COND(p_type < -1 || p_type > 1);
	ERR_FAIL_COND(p_type != 0 && p_to_task->get_parent().is_null());

	if (p_task == p_to_task) {
		return;
	}

	EditorUndoRedoManager *undo_redo = GET_UNDO_REDO();
	undo_redo->create_action(TTR("Drag BT Task"));
	undo_redo->add_do_method(p_task->get_parent().ptr(), LW_NAME(remove_child), p_task);

	if (p_type == 0) {
		undo_redo->add_do_method(p_to_task.ptr(), LW_NAME(add_child), p_task);
		undo_redo->add_undo_method(p_to_task.ptr(), LW_NAME(remove_child), p_task);
	} else {
		int drop_idx = p_to_task->get_index();
		if (p_to_task->get_parent() == p_task->get_parent() && drop_idx > p_task->get_index()) {
			drop_idx -= 1;
		}
		if (p_type == -1) {
			undo_redo->add_do_method(p_to_task->get_parent().ptr(), LW_NAME(add_child_at_index), p_task, drop_idx);
			undo_redo->add_undo_method(p_to_task->get_parent().ptr(), LW_NAME(remove_child), p_task);
		} else if (p_type == 1) {
			undo_redo->add_do_method(p_to_task->get_parent().ptr(), LW_NAME(add_child_at_index), p_task, drop_idx + 1);
			undo_redo->add_undo_method(p_to_task->get_parent().ptr(), LW_NAME(remove_child), p_task);
		}
	}

	undo_redo->add_undo_method(p_task->get_parent().ptr(), "add_child_at_index", p_task, p_task->get_index());

	undo_redo->add_do_method(task_tree, LW_NAME(update_tree));
	undo_redo->add_undo_method(task_tree, LW_NAME(update_tree));

	undo_redo->commit_action();
	_mark_as_dirty(true);
}

void LimboAIEditor::_on_resources_reload(const PackedStringArray &p_resources) {
	for (const String &res_path : p_resources) {
		if (!RESOURCE_IS_CACHED(res_path)) {
			continue;
		}

		if (RESOURCE_EXISTS(res_path, "BehaviorTree")) {
			Ref<BehaviorTree> res = RESOURCE_LOAD(res_path, "BehaviorTree");
			if (res.is_valid()) {
				if (history.has(res)) {
					disk_changed_files.insert(res_path);
				} else {
					Ref<BehaviorTree> reloaded = RESOURCE_LOAD_NO_CACHE(res_path, "BehaviorTree");
					res->copy_other(reloaded);
				}
			}
		}
	}

	// TODO: Find a way to allow resaving trees when they change outside of Godot.
	// * Currently, editor reloads them without asking in GDExtension. There is no Resource::editor_can_reload_from_file().
#ifdef LIMBOAI_MODULE
	if (disk_changed_files.size() > 0) {
		disk_changed_list->clear();
		disk_changed_list->set_hide_root(true);
		disk_changed_list->create_item();
		for (const String &fn : disk_changed_files) {
			TreeItem *ti = disk_changed_list->create_item();
			ti->set_text(0, fn);
		}

		if (!is_visible()) {
			SET_MAIN_SCREEN_EDITOR("LimboAI");
		}
		disk_changed->call_deferred("popup_centered_ratio", 0.5);
	}
#elif LIMBOAI_GDEXTENSION
	task_tree->update_tree();
#endif
}

void LimboAIEditor::_on_new_script_pressed() {
	SCRIPT_EDITOR()->open_script_create_dialog("BTAction", String(GLOBAL_GET("limbo_ai/behavior_tree/user_task_dir_1")).path_join("new_task"));
}

void LimboAIEditor::_task_type_selected(const String &p_class_or_path) {
	change_type_popup->hide();

	Ref<BTTask> selected_task = task_tree->get_selected();
	ERR_FAIL_COND(selected_task.is_null());
	Ref<BTTask> new_task = _create_task_by_class_or_path(p_class_or_path);
	ERR_FAIL_COND_MSG(new_task.is_null(), "LimboAI: Unable to construct task.");

	EditorUndoRedoManager *undo_redo = GET_UNDO_REDO();
	undo_redo->create_action(TTR("Change BT task type"));
	undo_redo->add_do_method(this, LW_NAME(_replace_task), selected_task, new_task);
	undo_redo->add_undo_method(this, LW_NAME(_replace_task), new_task, selected_task);
	undo_redo->add_do_method(task_tree, LW_NAME(update_tree));
	undo_redo->add_undo_method(task_tree, LW_NAME(update_tree));
	undo_redo->commit_action();
	_mark_as_dirty(true);
}

void LimboAIEditor::_copy_version_info() {
	DisplayServer::get_singleton()->clipboard_set(version_btn->get_text());
}

void LimboAIEditor::_replace_task(const Ref<BTTask> &p_task, const Ref<BTTask> &p_by_task) {
	ERR_FAIL_COND(p_task.is_null());
	ERR_FAIL_COND(p_by_task.is_null());
	ERR_FAIL_COND(p_by_task->get_child_count() > 0);
	ERR_FAIL_COND(p_by_task->get_parent().is_valid());

	while (p_task->get_child_count() > 0) {
		Ref<BTTask> child = p_task->get_child(0);
		p_task->remove_child_at_index(0);
		p_by_task->add_child(child);
	}
	p_by_task->set_custom_name(p_task->get_custom_name());

	Ref<BTTask> parent = p_task->get_parent();
	if (parent.is_null()) {
		// Assuming root task is replaced.
		ERR_FAIL_COND(task_tree->get_bt().is_null());
		ERR_FAIL_COND(task_tree->get_bt()->get_root_task() != p_task);
		task_tree->get_bt()->set_root_task(p_by_task);
	} else {
		// Non-root task is replaced.
		int idx = p_task->get_index();

		double weight = 0.0;
		Ref<BTProbabilitySelector> probability_selector = parent;
		if (probability_selector.is_valid()) {
			weight = probability_selector->get_weight(idx);
		}

		parent->remove_child(p_task);
		parent->add_child_at_index(p_by_task, idx);

		if (probability_selector.is_valid()) {
			probability_selector->set_weight(idx, weight);
		}
	}
}

void LimboAIEditor::_tab_clicked(int p_tab) {
	if (updating_tabs) {
		return;
	}
	ERR_FAIL_INDEX(p_tab, history.size());
	EDIT_RESOURCE(history[p_tab]);
}

void LimboAIEditor::_tab_closed(int p_tab) {
	ERR_FAIL_INDEX(p_tab, history.size());
	Ref<BehaviorTree> history_bt = history[p_tab];
	if (history_bt.is_valid() && history_bt->is_connected(LW_NAME(changed), callable_mp(this, &LimboAIEditor::_mark_as_dirty))) {
		history_bt->disconnect(LW_NAME(changed), callable_mp(this, &LimboAIEditor::_mark_as_dirty));
	}
	history.remove_at(p_tab);
	idx_history = MIN(idx_history, history.size() - 1);
	if (idx_history < 0) {
		_disable_editing();
	} else {
		EDIT_RESOURCE(history[idx_history]);
	}
	_update_tabs();
}

void LimboAIEditor::_update_tabs() {
	updating_tabs = true;
	tab_bar->clear_tabs();

	Vector<String> short_names;
	// Keep track of how many times each short name is used.
	HashMap<String, int> usage_counts;

	for (int i = 0; i < history.size(); i++) {
		String tab_name;
		if (history[i]->get_path().contains("::")) {
			tab_name = history[i]->get_path().get_file();
		} else {
			tab_name = history[i]->get_path().get_file().get_basename();
		}
		short_names.append(tab_name);
		if (usage_counts.has(tab_name)) {
			usage_counts[tab_name] += 1;
		} else {
			usage_counts[tab_name] = 1;
		}
	}

	for (int i = 0; i < short_names.size(); i++) {
		String tab_name = short_names[i];
		if (tab_name.is_empty()) {
			tab_name = "[new]";
		} else if (usage_counts[tab_name] > 1) {
			// Use the full name if the short name is not unique.
			tab_name = history[i]->get_path().trim_prefix("res://");
		}
		tab_bar->add_tab(tab_name, LimboUtility::get_singleton()->get_task_icon("BehaviorTree"));
		if (i == idx_history) {
			tab_bar->set_tab_button_icon(tab_bar->get_tab_count() - 1, LimboUtility::get_singleton()->get_task_icon("LimboEditBlackboard"));
		}
	}

	if (idx_history >= 0) {
		ERR_FAIL_INDEX(idx_history, history.size());
		tab_bar->set_current_tab(idx_history);
	}

	updating_tabs = false;
}

void LimboAIEditor::_move_active_tab(int p_to_index) {
	ERR_FAIL_INDEX(p_to_index, history.size());
	if (idx_history == p_to_index) {
		return;
	}
	Ref<BehaviorTree> bt = history[idx_history];
	history.remove_at(idx_history);
	history.insert(p_to_index, bt);
	idx_history = p_to_index;
	_update_tabs();
}

void LimboAIEditor::_tab_input(const Ref<InputEvent> &p_input) {
	Ref<InputEventMouseButton> mb = p_input;
	if (mb.is_null()) {
		return;
	}
	int tab_idx = tab_bar->get_tab_idx_at_point(tab_bar->get_local_mouse_position());
	if (tab_idx < 0) {
		return;
	}
	if (mb->is_pressed() && mb->get_button_index() == LW_MBTN(MIDDLE)) {
		_tab_closed(tab_idx);
	} else if (mb->is_pressed() && mb->get_button_index() == LW_MBTN(RIGHT)) {
		_show_tab_context_menu();
	}
}

void LimboAIEditor::_show_tab_context_menu() {
	tab_menu->clear();
	tab_menu->add_shortcut(LW_GET_SHORTCUT("limbo_ai/jump_to_owner"), TabMenu::TAB_JUMP_TO_OWNER);
	tab_menu->add_item(TTR("Show in FileSystem"), TabMenu::TAB_SHOW_IN_FILESYSTEM);
	tab_menu->add_separator();
	tab_menu->add_shortcut(LW_GET_SHORTCUT("limbo_ai/close_tab"), TabMenu::TAB_CLOSE);
	tab_menu->add_item(TTR("Close Other Tabs"), TabMenu::TAB_CLOSE_OTHER);
	tab_menu->add_item(TTR("Close Tabs to the Right"), TabMenu::TAB_CLOSE_RIGHT);
	tab_menu->add_item(TTR("Close All Tabs"), TabMenu::TAB_CLOSE_ALL);
	tab_menu->set_position(get_screen_position() + get_local_mouse_position());
	tab_menu->reset_size();
	tab_menu->popup();
}

void LimboAIEditor::_tab_menu_option_selected(int p_id) {
	if (history.size() == 0) {
		// No tabs open, returning.
		return;
	}
	ERR_FAIL_INDEX(idx_history, history.size());

	switch (p_id) {
		case TAB_SHOW_IN_FILESYSTEM: {
			Ref<BehaviorTree> bt = history[idx_history];
			String path = bt->get_path();
			if (!path.is_empty()) {
				FS_DOCK_SELECT_FILE(path.get_slice("::", 0));
			}
		} break;
		case TAB_JUMP_TO_OWNER: {
			Ref<BehaviorTree> bt = history[idx_history];
			ERR_FAIL_NULL(bt);
			String bt_path = bt->get_path();
			if (!bt_path.is_empty()) {
				owner_picker->pick_and_open_owner_of_resource(bt_path);
			}
		} break;
		case TAB_CLOSE: {
			_tab_closed(idx_history);
		} break;
		case TAB_CLOSE_OTHER: {
			Ref<BehaviorTree> bt = history[idx_history];
			history.clear();
			history.append(bt);
			idx_history = 0;
			_update_tabs();
		} break;
		case TAB_CLOSE_RIGHT: {
			for (int i = history.size() - 1; i > idx_history; i--) {
				history.remove_at(i);
			}
			_update_tabs();
		} break;
		case TAB_CLOSE_ALL: {
			history.clear();
			idx_history = -1;
			_disable_editing();
			_update_tabs();
		} break;
	}
}

void LimboAIEditor::_tab_plan_edited(int p_tab) {
	ERR_FAIL_INDEX(p_tab, history.size());
	if (history[p_tab]->get_blackboard_plan().is_valid()) {
		EDIT_RESOURCE(history[p_tab]->get_blackboard_plan());
	}
}

void LimboAIEditor::_reload_modified() {
	for (const String &res_path : disk_changed_files) {
		Ref<BehaviorTree> res = RESOURCE_LOAD(res_path, "BehaviorTree");
		if (res.is_valid()) {
			Ref<BehaviorTree> reloaded = RESOURCE_LOAD_NO_CACHE(res_path, "BehaviorTree");
			res->copy_other(reloaded);
			if (idx_history >= 0 && history.get(idx_history) == res) {
				edit_bt(res, true);
			}
		}
	}
	disk_changed_files.clear();
	task_tree->update_tree();
}

void LimboAIEditor::_resave_modified(String _str) {
	for (const String &res_path : disk_changed_files) {
		Ref<BehaviorTree> res = RESOURCE_LOAD(res_path, "BehaviorTree");
		if (res.is_valid()) {
			ERR_FAIL_COND(!res->is_class("BehaviorTree"));
			RESOURCE_SAVE(res, res->get_path(), 0);
		}
	}
	task_tree->update_tree();
	disk_changed->hide();
	disk_changed_files.clear();
}

void LimboAIEditor::_popup_info_dialog(const String &p_text) {
	info_dialog->set_text(p_text);
	info_dialog->popup_centered();
}

void LimboAIEditor::_rename_task_confirmed() {
	ERR_FAIL_COND(!task_tree->get_selected().is_valid());
	rename_dialog->hide();

	EditorUndoRedoManager *undo_redo = GET_UNDO_REDO();
	undo_redo->create_action(TTR("Set Custom Name"));
	undo_redo->add_do_method(task_tree->get_selected().ptr(), LW_NAME(set_custom_name), rename_edit->get_text());
	undo_redo->add_undo_method(task_tree->get_selected().ptr(), LW_NAME(set_custom_name), task_tree->get_selected()->get_custom_name());
	undo_redo->add_do_method(task_tree, LW_NAME(update_task), task_tree->get_selected());
	undo_redo->add_undo_method(task_tree, LW_NAME(update_task), task_tree->get_selected());
	undo_redo->commit_action();
}

void LimboAIEditor::apply_changes() {
	for (int i = 0; i < history.size(); i++) {
		Ref<BehaviorTree> bt = history.get(i);
		String path = bt->get_path();
		if (RESOURCE_EXISTS(path, "BehaviorTree")) {
			RESOURCE_SAVE(bt, path, 0);
		}
		dirty.clear();
	}
}

void LimboAIEditor::_update_favorite_tasks() {
	for (int i = 0; i < fav_tasks_hbox->get_child_count(); i++) {
		fav_tasks_hbox->get_child(i)->queue_free();
	}
	Array favorite_tasks = GLOBAL_GET("limbo_ai/behavior_tree/favorite_tasks");
	for (int i = 0; i < favorite_tasks.size(); i++) {
		String task_meta = favorite_tasks[i];

		if (task_meta.is_empty() || (!FILE_EXISTS(task_meta) && !ClassDB::class_exists(task_meta))) {
			call_deferred(LW_NAME(_update_banners));
			continue;
		}

		Button *btn = memnew(Button);
		String task_name;
		if (task_meta.begins_with("res:")) {
			task_name = task_meta.get_file().get_basename().trim_prefix("BT").to_pascal_case();
		} else {
			task_name = task_meta.trim_prefix("BT");
		}
		btn->set_text(task_name);
		btn->set_meta(LW_NAME(task_meta), task_meta);
		BUTTON_SET_ICON(btn, LimboUtility::get_singleton()->get_task_icon(task_meta));
		btn->set_tooltip_text(vformat(TTR("Add %s task."), task_name));
		btn->set_flat(true);
		btn->add_theme_constant_override(LW_NAME(icon_max_width), 16 * EDSCALE); // Force user icons to be of the proper size.
		btn->set_focus_mode(Control::FOCUS_NONE);
		btn->connect(LW_NAME(pressed), callable_mp(this, &LimboAIEditor::_add_task_by_class_or_path).bind(task_meta));
		fav_tasks_hbox->add_child(btn);
	}
}

void LimboAIEditor::_update_misc_menu() {
	PopupMenu *misc_menu = misc_btn->get_popup();

	misc_menu->clear();

	misc_menu->add_icon_item(theme_cache.doc_icon, TTR("Online Documentation"), MISC_ONLINE_DOCUMENTATION);
	misc_menu->add_icon_item(theme_cache.introduction_icon, TTR("Introduction"), MISC_DOC_INTRODUCTION);
	misc_menu->add_icon_item(theme_cache.introduction_icon, TTR("Creating custom tasks in GDScript"), MISC_DOC_CUSTOM_TASKS);

	misc_menu->add_separator();
#ifdef LIMBOAI_MODULE
	// * Disabled in GDExtension: Not sure how to switch to debugger pane.
	misc_menu->add_icon_shortcut(theme_cache.open_debugger_icon, LW_GET_SHORTCUT("limbo_ai/open_debugger"), MISC_OPEN_DEBUGGER);
#endif // LIMBOAI_MODULE
	misc_menu->add_item(TTR("Project Settings..."), MISC_PROJECT_SETTINGS);

	PopupMenu *layout_menu = Object::cast_to<PopupMenu>(misc_menu->get_node_or_null(NodePath("LayoutMenu")));
	if (layout_menu == nullptr) {
		layout_menu = memnew(PopupMenu);
		layout_menu->set_name("LayoutMenu");
		layout_menu->connect(LW_NAME(id_pressed), callable_mp(this, &LimboAIEditor::_misc_option_selected));
		misc_menu->add_child(layout_menu);
	}
	layout_menu->add_radio_check_item(TTR("Classic"), MISC_LAYOUT_CLASSIC);
	layout_menu->add_radio_check_item(TTR("Widescreen Optimized"), MISC_LAYOUT_WIDESCREEN_OPTIMIZED);
	misc_menu->add_submenu_item(TTR("Layout"), "LayoutMenu");
	EditorLayout saved_layout = (EditorLayout)(int)EDITOR_GET("limbo_ai/editor/layout");
	layout_menu->set_item_checked(0, saved_layout == LAYOUT_CLASSIC);
	layout_menu->set_item_checked(1, saved_layout == LAYOUT_WIDESCREEN_OPTIMIZED);

	misc_menu->add_separator();
	misc_menu->add_item(
			FILE_EXISTS(_get_script_template_path()) ? TTR("Edit Script Template") : TTR("Create Script Template"),
			MISC_CREATE_SCRIPT_TEMPLATE);
}

void LimboAIEditor::_update_banners() {
	for (int i = 0; i < banners->get_child_count(); i++) {
		if (banners->get_child(i)->has_meta(LW_NAME(managed))) {
			banners->get_child(i)->queue_free();
		}
	}

	for (String dir_setting : { "limbo_ai/behavior_tree/user_task_dir_1", "limbo_ai/behavior_tree/user_task_dir_2", "limbo_ai/behavior_tree/user_task_dir_3" }) {
		String task_dir = GLOBAL_GET(dir_setting);
		if (!task_dir.is_empty() && !DirAccess::dir_exists_absolute(task_dir)) {
			ActionBanner *banner = memnew(ActionBanner);
			banner->set_text(vformat(TTR("Task folder not found: %s"), task_dir));
			banner->add_action(TTR("Create"), callable_mp(this, &LimboAIEditor::_create_user_task_dir), true);
			banner->add_action(TTR("Edit Path..."), callable_mp(this, &LimboAIEditor::_edit_project_settings));
			banner->add_spacer();
			banner->add_action(TTR("Help..."), callable_mp(LimboUtility::get_singleton(), &LimboUtility::open_doc_custom_tasks));
			banner->set_meta(LW_NAME(managed), Variant(true));
			banners->call_deferred(LW_NAME(add_child), banner);
		}
	}

	Array favorite_tasks = GLOBAL_GET("limbo_ai/behavior_tree/favorite_tasks");
	for (int i = 0; i < favorite_tasks.size(); i++) {
		String task_meta = favorite_tasks[i];

		if (task_meta.is_empty() || (!FILE_EXISTS(task_meta) && !ClassDB::class_exists(task_meta))) {
			ActionBanner *banner = memnew(ActionBanner);
			banner->set_text(vformat(TTR("Favorite task not found: %s"), task_meta));
			banner->add_action(TTR("Remove"), callable_mp(this, &LimboAIEditor::_remove_task_from_favorite).bind(task_meta), true);
			banner->add_action(TTR("Edit Favorite Tasks..."), callable_mp(this, &LimboAIEditor::_edit_project_settings));
			banner->set_meta(LW_NAME(managed), Variant(true));
			banners->call_deferred(LW_NAME(add_child), banner);
		}
	}

	EditorLayout saved_layout = (EditorLayout)(int)EDITOR_GET("limbo_ai/editor/layout");
	if (saved_layout != editor_layout) {
		ActionBanner *banner = memnew(ActionBanner);
		banner->set_text(TTR("Restart required to apply changes to editor layout"));
		banner->add_action(TTR("Save & Restart"), callable_mp(this, &LimboAIEditor::_save_and_restart), true);
		banner->set_meta(LW_NAME(managed), Variant(true));
		banners->call_deferred(LW_NAME(add_child), banner);
	}
}

void LimboAIEditor::_do_update_theme_item_cache() {
	theme_cache.duplicate_task_icon = get_theme_icon(LW_NAME(Duplicate), LW_NAME(EditorIcons));
	theme_cache.edit_script_icon = get_theme_icon(LW_NAME(Script), LW_NAME(EditorIcons));
	theme_cache.make_root_icon = get_theme_icon(LW_NAME(NewRoot), LW_NAME(EditorIcons));
	theme_cache.move_task_down_icon = get_theme_icon(LW_NAME(MoveDown), LW_NAME(EditorIcons));
	theme_cache.move_task_up_icon = get_theme_icon(LW_NAME(MoveUp), LW_NAME(EditorIcons));
	theme_cache.open_debugger_icon = get_theme_icon(LW_NAME(Debug), LW_NAME(EditorIcons));
	theme_cache.doc_icon = get_theme_icon(LW_NAME(Help), LW_NAME(EditorIcons));
	theme_cache.introduction_icon = get_theme_icon(LW_NAME(Info), LW_NAME(EditorIcons));
	theme_cache.remove_task_icon = get_theme_icon(LW_NAME(Remove), LW_NAME(EditorIcons));
	theme_cache.rename_task_icon = get_theme_icon(LW_NAME(Rename), LW_NAME(EditorIcons));
	theme_cache.change_type_icon = get_theme_icon(LW_NAME(Reload), LW_NAME(EditorIcons));
	theme_cache.cut_icon = get_theme_icon(LW_NAME(ActionCut), LW_NAME(EditorIcons));
	theme_cache.copy_icon = get_theme_icon(LW_NAME(ActionCopy), LW_NAME(EditorIcons));
	theme_cache.paste_icon = get_theme_icon(LW_NAME(ActionPaste), LW_NAME(EditorIcons));

	theme_cache.behavior_tree_icon = LimboUtility::get_singleton()->get_task_icon("BehaviorTree");
	theme_cache.percent_icon = LimboUtility::get_singleton()->get_task_icon("LimboPercent");
	theme_cache.extract_subtree_icon = LimboUtility::get_singleton()->get_task_icon("LimboExtractSubtree");
}

void LimboAIEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			Ref<ConfigFile> cf;
			cf.instantiate();
			String conf_path = PROJECT_CONFIG_FILE();
			if (cf->load(conf_path) == OK) {
				hsc->set_split_offset(cf->get_value("bt_editor", "bteditor_hsplit", hsc->get_split_offset()));
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			Ref<ConfigFile> cf;
			cf.instantiate();
			String conf_path = PROJECT_CONFIG_FILE();
			cf->load(conf_path);
			int split_offset = hsc->get_split_offset();
			if (editor_layout != (int)EDITOR_GET("limbo_ai/editor/layout")) {
				// Editor layout settings changed - flip split offset.
				split_offset *= -1;
			}
			cf->set_value("bt_editor", "bteditor_hsplit", split_offset);
			cf->save(conf_path);

			task_tree->unload();
			for (int i = 0; i < history.size(); i++) {
				if (history[i]->is_connected(LW_NAME(changed), callable_mp(this, &LimboAIEditor::_mark_as_dirty))) {
					history[i]->disconnect(LW_NAME(changed), callable_mp(this, &LimboAIEditor::_mark_as_dirty));
				}
			}
		} break;
		case NOTIFICATION_READY: {
			// **** Signals
			save_dialog->connect("file_selected", callable_mp(this, &LimboAIEditor::_save_bt));
			load_dialog->connect("file_selected", callable_mp(this, &LimboAIEditor::_load_bt));
			extract_dialog->connect("file_selected", callable_mp(this, &LimboAIEditor::_extract_subtree));
			new_btn->connect(LW_NAME(pressed), callable_mp(this, &LimboAIEditor::_new_bt));
			load_btn->connect(LW_NAME(pressed), callable_mp(this, &LimboAIEditor::_popup_file_dialog).bind(load_dialog));
			task_tree->connect("rmb_pressed", callable_mp(this, &LimboAIEditor::_on_tree_rmb));
			task_tree->connect("task_selected", callable_mp(this, &LimboAIEditor::_on_tree_task_selected));
			task_tree->connect("task_dragged", callable_mp(this, &LimboAIEditor::_on_task_dragged));
			task_tree->connect("task_activated", callable_mp(this, &LimboAIEditor::_action_selected).bind(ACTION_RENAME));
			task_tree->connect("probability_clicked", callable_mp(this, &LimboAIEditor::_action_selected).bind(ACTION_EDIT_PROBABILITY));
			task_tree->connect("visibility_changed", callable_mp(this, &LimboAIEditor::_on_visibility_changed));
			task_tree->connect("visibility_changed", callable_mp(this, &LimboAIEditor::_update_banners));
			save_btn->connect(LW_NAME(pressed), callable_mp(this, &LimboAIEditor::_on_save_pressed));
			misc_btn->connect(LW_NAME(pressed), callable_mp(this, &LimboAIEditor::_update_misc_menu));
			misc_btn->get_popup()->connect("id_pressed", callable_mp(this, &LimboAIEditor::_misc_option_selected));
			task_palette->connect("task_selected", callable_mp(this, &LimboAIEditor::_add_task_by_class_or_path));
			task_palette->connect("favorite_tasks_changed", callable_mp(this, &LimboAIEditor::_update_favorite_tasks));
			change_type_palette->connect("task_selected", callable_mp(this, &LimboAIEditor::_task_type_selected));
			menu->connect("id_pressed", callable_mp(this, &LimboAIEditor::_action_selected));
			weight_mode->connect(LW_NAME(pressed), callable_mp(this, &LimboAIEditor::_update_probability_edit));
			percent_mode->connect(LW_NAME(pressed), callable_mp(this, &LimboAIEditor::_update_probability_edit));
			probability_edit->connect("value_changed", callable_mp(this, &LimboAIEditor::_on_probability_edited));
			probability_popup->connect("popup_hide", callable_mp(this, &LimboAIEditor::_probability_popup_closed));
			disk_changed->connect("confirmed", callable_mp(this, &LimboAIEditor::_reload_modified));
			disk_changed->connect("custom_action", callable_mp(this, &LimboAIEditor::_resave_modified));
			rename_dialog->connect("confirmed", callable_mp(this, &LimboAIEditor::_rename_task_confirmed));
			new_script_btn->connect(LW_NAME(pressed), callable_mp(this, &LimboAIEditor::_on_new_script_pressed));
			tab_bar->connect("tab_clicked", callable_mp(this, &LimboAIEditor::_tab_clicked));
			tab_bar->connect("active_tab_rearranged", callable_mp(this, &LimboAIEditor::_move_active_tab));
			tab_bar->connect("tab_close_pressed", callable_mp(this, &LimboAIEditor::_tab_closed));
			tab_bar->connect(LW_NAME(gui_input), callable_mp(this, &LimboAIEditor::_tab_input));
			tab_menu->connect(LW_NAME(id_pressed), callable_mp(this, &LimboAIEditor::_tab_menu_option_selected));
			tab_bar->connect("tab_button_pressed", callable_mp(this, &LimboAIEditor::_tab_plan_edited));
			version_btn->connect(LW_NAME(pressed), callable_mp(this, &LimboAIEditor::_copy_version_info));

			EDITOR_FILE_SYSTEM()->connect("resources_reload", callable_mp(this, &LimboAIEditor::_on_resources_reload));

		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_do_update_theme_item_cache();

			ADD_STYLEBOX_OVERRIDE(tab_bar_panel, "panel", get_theme_stylebox("tabbar_background", "TabContainer"));

			BUTTON_SET_ICON(new_btn, get_theme_icon(LW_NAME(New), LW_NAME(EditorIcons)));
			BUTTON_SET_ICON(load_btn, get_theme_icon(LW_NAME(Load), LW_NAME(EditorIcons)));
			BUTTON_SET_ICON(save_btn, get_theme_icon(LW_NAME(Save), LW_NAME(EditorIcons)));
			BUTTON_SET_ICON(new_script_btn, get_theme_icon(LW_NAME(ScriptCreate), LW_NAME(EditorIcons)));
			BUTTON_SET_ICON(misc_btn, get_theme_icon(LW_NAME(Tools), LW_NAME(EditorIcons)));

			_update_favorite_tasks();
		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (is_visible_in_tree()) {
				_update_banners();
			}
		} break;
	}
}

void LimboAIEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_add_task", "task"), &LimboAIEditor::_add_task);
	ClassDB::bind_method(D_METHOD("_remove_task", "task"), &LimboAIEditor::_remove_task);
	ClassDB::bind_method(D_METHOD("_add_task_with_prototype", "prototype_task"), &LimboAIEditor::_add_task_with_prototype);
	ClassDB::bind_method(D_METHOD("_new_bt"), &LimboAIEditor::_new_bt);
	ClassDB::bind_method(D_METHOD("_save_bt", "path"), &LimboAIEditor::_save_bt);
	ClassDB::bind_method(D_METHOD("_load_bt", "path"), &LimboAIEditor::_load_bt);
	ClassDB::bind_method(D_METHOD("edit_bt", "behavior_tree", "force_refresh"), &LimboAIEditor::edit_bt, Variant(false));
	ClassDB::bind_method(D_METHOD("_reload_modified"), &LimboAIEditor::_reload_modified);
	ClassDB::bind_method(D_METHOD("_resave_modified"), &LimboAIEditor::_resave_modified);
	ClassDB::bind_method(D_METHOD("_replace_task", "task", "by_task"), &LimboAIEditor::_replace_task);
	ClassDB::bind_method(D_METHOD("_popup_file_dialog"), &LimboAIEditor::_popup_file_dialog);
	ClassDB::bind_method(D_METHOD("get_edited_blackboard_plan"), &LimboAIEditor::get_edited_blackboard_plan);
}

LimboAIEditor::LimboAIEditor() {
	plugin = nullptr;
	idx_history = 0;

#ifdef LIMBOAI_MODULE
	EDITOR_DEF("limbo_ai/editor/layout", 0);
	EDITOR_SETTINGS()->add_property_hint(PropertyInfo(Variant::INT, "limbo_ai/editor/layout", PROPERTY_HINT_ENUM, "Classic:0,Widescreen Optimized:1"));
	EDITOR_SETTINGS()->set_restart_if_changed("limbo_ai/editor/layout", true);
#elif LIMBOAI_GDEXTENSION
	EDITOR_SETTINGS()->set_initial_value("limbo_ai/editor/layout", 0, false);
	Dictionary pinfo;
	pinfo["name"] = "limbo_ai/editor/layout";
	pinfo["type"] = Variant::INT;
	pinfo["hint"] = PROPERTY_HINT_ENUM;
	pinfo["hint_string"] = "Classic:0,Widescreen Optimized:1";
	EDITOR_SETTINGS()->add_property_info(pinfo);
#endif

	LW_SHORTCUT("limbo_ai/rename_task", TTR("Rename"), LW_KEY(F2));
	// Todo: Add override support for shortcuts.
	// LW_SHORTCUT_OVERRIDE("limbo_ai/rename_task", "macos", Key::ENTER);
	LW_SHORTCUT("limbo_ai/move_task_up", TTR("Move Up"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY(UP)));
	LW_SHORTCUT("limbo_ai/move_task_down", TTR("Move Down"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY(DOWN)));
	LW_SHORTCUT("limbo_ai/duplicate_task", TTR("Duplicate"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY(D)));
	LW_SHORTCUT("limbo_ai/remove_task", TTR("Remove"), Key::KEY_DELETE);
	LW_SHORTCUT("limbo_ai/cut_task", TTR("Cut"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY(X)));
	LW_SHORTCUT("limbo_ai/copy_task", TTR("Copy"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY(C)));
	LW_SHORTCUT("limbo_ai/paste_task", TTR("Paste"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY(V)));
	LW_SHORTCUT("limbo_ai/paste_task_after", TTR("Paste After Selected"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY_MASK(SHIFT) | LW_KEY(V)));

	LW_SHORTCUT("limbo_ai/new_behavior_tree", TTR("New Behavior Tree"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY_MASK(ALT) | LW_KEY(N)));
	LW_SHORTCUT("limbo_ai/save_behavior_tree", TTR("Save Behavior Tree"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY_MASK(ALT) | LW_KEY(S)));
	LW_SHORTCUT("limbo_ai/load_behavior_tree", TTR("Load Behavior Tree"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY_MASK(ALT) | LW_KEY(L)));
	LW_SHORTCUT("limbo_ai/open_debugger", TTR("Open Debugger"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY_MASK(ALT) | LW_KEY(D)));
	LW_SHORTCUT("limbo_ai/jump_to_owner", TTR("Jump to Owner"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY(J)));
	LW_SHORTCUT("limbo_ai/close_tab", TTR("Close Tab"), (Key)(LW_KEY_MASK(CMD_OR_CTRL) | LW_KEY(W)));

	set_process_shortcut_input(true);

	save_dialog = memnew(FileDialog);
	save_dialog->set_file_mode(FileDialog::FILE_MODE_SAVE_FILE);
	save_dialog->set_title(TTR("Save Behavior Tree"));
	save_dialog->add_filter("*.tres");
	save_dialog->hide();
	add_child(save_dialog);

	load_dialog = memnew(FileDialog);
	load_dialog->set_file_mode(FileDialog::FILE_MODE_OPEN_FILE);
	load_dialog->set_title(TTR("Load Behavior Tree"));
	load_dialog->add_filter("*.tres");
	load_dialog->hide();
	add_child(load_dialog);

	extract_dialog = memnew(FileDialog);
	extract_dialog->set_file_mode(FileDialog::FILE_MODE_SAVE_FILE);
	extract_dialog->set_title(TTR("Save Extracted Tree"));
	extract_dialog->add_filter("*.tres");
	extract_dialog->hide();
	add_child(extract_dialog);

	vbox = memnew(VBoxContainer);
	vbox->set_anchor(SIDE_RIGHT, ANCHOR_END);
	vbox->set_anchor(SIDE_BOTTOM, ANCHOR_END);
	add_child(vbox);

	HBoxContainer *toolbar = memnew(HBoxContainer);
	vbox->add_child(toolbar);

	PackedStringArray favorite_tasks_default;
	favorite_tasks_default.append("BTSelector");
	favorite_tasks_default.append("BTSequence");
	favorite_tasks_default.append("BTComment");
	GLOBAL_DEF(PropertyInfo(Variant::PACKED_STRING_ARRAY, "limbo_ai/behavior_tree/favorite_tasks", PROPERTY_HINT_ARRAY_TYPE, "String"), favorite_tasks_default);

	fav_tasks_hbox = memnew(HBoxContainer);
	toolbar->add_child(fav_tasks_hbox);

	toolbar->add_child(memnew(VSeparator));

	new_btn = memnew(Button);
	new_btn->set_text(TTR("New"));
	new_btn->set_tooltip_text(TTR("Create a new behavior tree."));
	new_btn->set_shortcut(LW_GET_SHORTCUT("limbo_ai/new_behavior_tree"));
	new_btn->set_flat(true);
	new_btn->set_focus_mode(Control::FOCUS_NONE);
	toolbar->add_child(new_btn);

	load_btn = memnew(Button);
	load_btn->set_text(TTR("Load"));
	load_btn->set_tooltip_text(TTR("Load behavior tree from a resource file."));
	load_btn->set_shortcut(LW_GET_SHORTCUT("limbo_ai/load_behavior_tree"));
	load_btn->set_flat(true);
	load_btn->set_focus_mode(Control::FOCUS_NONE);
	toolbar->add_child(load_btn);

	save_btn = memnew(Button);
	save_btn->set_text(TTR("Save"));
	save_btn->set_tooltip_text(TTR("Save edited behavior tree to a resource file."));
	save_btn->set_shortcut(LW_GET_SHORTCUT("limbo_ai/save_behavior_tree"));
	save_btn->set_flat(true);
	save_btn->set_focus_mode(Control::FOCUS_NONE);
	toolbar->add_child(save_btn);

	toolbar->add_child(memnew(VSeparator));

	new_script_btn = memnew(Button);
	new_script_btn->set_text(TTR("New Task"));
	new_script_btn->set_tooltip_text(TTR("Create new task script and edit it."));
	new_script_btn->set_flat(true);
	new_script_btn->set_focus_mode(Control::FOCUS_NONE);
	toolbar->add_child(new_script_btn);

	misc_btn = memnew(MenuButton);
	misc_btn->set_text(TTR("Misc"));
	misc_btn->set_flat(true);
	toolbar->add_child(misc_btn);

	HBoxContainer *version_hbox = memnew(HBoxContainer);
	version_hbox->set_h_size_flags(SIZE_EXPAND | SIZE_SHRINK_END);
	toolbar->add_child(version_hbox);

	TextureRect *logo = memnew(TextureRect);
	logo->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	logo->set_texture(LimboUtility::get_singleton()->get_task_icon("LimboAI"));
	version_hbox->add_child(logo);

	version_btn = memnew(LinkButton);
	version_btn->set_text(TTR("v") + String(GET_LIMBOAI_FULL_VERSION()));
	version_btn->set_tooltip_text(TTR("Click to copy."));
	version_btn->set_self_modulate(Color(1, 1, 1, 0.65));
	version_btn->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	version_btn->set_v_size_flags(SIZE_SHRINK_CENTER);
	version_hbox->add_child(version_btn);

	Control *version_spacer = memnew(Control);
	version_spacer->set_custom_minimum_size(Size2(2, 0) * EDSCALE);
	version_hbox->add_child(version_spacer);

	tab_bar_panel = memnew(PanelContainer);
	vbox->add_child(tab_bar_panel);
	tab_bar_container = memnew(HBoxContainer);
	tab_bar_panel->add_child(tab_bar_container);

	tab_bar = memnew(TabBar);
	tab_bar->set_select_with_rmb(true);
	tab_bar->set_drag_to_rearrange_enabled(true);
	tab_bar->set_max_tab_width(int(EDITOR_GET("interface/scene_tabs/maximum_width")) * EDSCALE);
	tab_bar->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	tab_bar->set_tab_close_display_policy(TabBar::CLOSE_BUTTON_SHOW_ACTIVE_ONLY);
	tab_bar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tab_bar->set_focus_mode(FocusMode::FOCUS_NONE);
	tab_bar_container->add_child(tab_bar);

	tab_menu = memnew(PopupMenu);
	add_child(tab_menu);

	owner_picker = memnew(OwnerPicker);
	add_child(owner_picker);

	hsc = memnew(HSplitContainer);
	hsc->set_h_size_flags(SIZE_EXPAND_FILL);
	hsc->set_v_size_flags(SIZE_EXPAND_FILL);
	hsc->set_focus_mode(FOCUS_NONE);
	vbox->add_child(hsc);

	task_tree = memnew(TaskTree);
	task_tree->set_v_size_flags(SIZE_EXPAND_FILL);
	task_tree->set_h_size_flags(SIZE_EXPAND_FILL);
	task_tree->hide();
	hsc->add_child(task_tree);

	usage_hint = memnew(Panel);
	usage_hint->set_v_size_flags(SIZE_EXPAND_FILL);
	usage_hint->set_h_size_flags(SIZE_EXPAND_FILL);
	hsc->add_child(usage_hint);

	Label *usage_label = memnew(Label);
	usage_label->set_anchor(SIDE_RIGHT, 1);
	usage_label->set_anchor(SIDE_BOTTOM, 1);
	usage_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	usage_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	usage_label->set_text(TTR("Create a new or load an existing behavior tree."));
	usage_hint->add_child(usage_label);

	task_palette = memnew(TaskPalette());
	task_palette->hide();
	hsc->add_child(task_palette);

	banners = memnew(VBoxContainer);
	vbox->add_child(banners);

	editor_layout = (EditorLayout)(int)EDITOR_GET("limbo_ai/editor/layout");
	if (editor_layout == LAYOUT_WIDESCREEN_OPTIMIZED) {
		// * Alternative layout optimized for wide screen.
		VBoxContainer *sidebar_vbox = memnew(VBoxContainer);
		hsc->add_child(sidebar_vbox);
		sidebar_vbox->set_v_size_flags(SIZE_EXPAND_FILL);

		HBoxContainer *header_bar = memnew(HBoxContainer);
		sidebar_vbox->add_child(header_bar);
		Control *header_spacer = memnew(Control);
		header_bar->add_child(header_spacer);
		header_spacer->set_custom_minimum_size(Size2(6, 0) * EDSCALE);
		TextureRect *header_logo = Object::cast_to<TextureRect>(logo->duplicate());
		header_bar->add_child(header_logo);
		Label *header_title = memnew(Label);
		header_bar->add_child(header_title);
		header_title->set_text(TTR("Behavior Tree Editor"));
		header_title->set_v_size_flags(SIZE_SHRINK_CENTER);
		header_title->set_theme_type_variation("HeaderMedium");

		task_palette->reparent(sidebar_vbox);
		task_palette->set_v_size_flags(SIZE_EXPAND_FILL);

		VBoxContainer *editor_vbox = memnew(VBoxContainer);
		hsc->add_child(editor_vbox);
		toolbar->reparent(editor_vbox);
		tab_bar_panel->reparent(editor_vbox);
		task_tree->reparent(editor_vbox);
		usage_hint->reparent(editor_vbox);
		banners->reparent(editor_vbox);
	}

	hsc->set_split_offset((editor_layout == LAYOUT_CLASSIC ? -320 : 320) * EDSCALE);

	change_type_popup = memnew(PopupPanel);
	add_child(change_type_popup);
	{
		VBoxContainer *change_type_vbox = memnew(VBoxContainer);
		change_type_popup->add_child(change_type_vbox);

		Label *change_type_title = memnew(Label);
		change_type_vbox->add_child(change_type_title);
		change_type_title->set_theme_type_variation("HeaderSmall");
		change_type_title->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
		change_type_title->set_text(TTR("Choose New Type"));

		change_type_palette = memnew(TaskPalette);
		change_type_vbox->add_child(change_type_palette);
		change_type_palette->use_dialog_mode();
		change_type_palette->set_v_size_flags(SIZE_EXPAND_FILL);
	}

	menu = memnew(PopupMenu);
	add_child(menu);

	probability_popup = memnew(PopupPanel);
	{
		VBoxContainer *vbc = memnew(VBoxContainer);
		probability_popup->add_child(vbc);

		PanelContainer *mode_panel = memnew(PanelContainer);
		vbc->add_child(mode_panel);

		HBoxContainer *mode_hbox = memnew(HBoxContainer);
		mode_panel->add_child(mode_hbox);

		Ref<ButtonGroup> button_group;
		button_group.instantiate();

		weight_mode = memnew(Button);
		mode_hbox->add_child(weight_mode);
		weight_mode->set_toggle_mode(true);
		weight_mode->set_button_group(button_group);
		weight_mode->set_focus_mode(Control::FOCUS_NONE);
		weight_mode->set_text(TTR("Weight"));
		weight_mode->set_tooltip_text(TTR("Edit weight"));
		weight_mode->set_pressed_no_signal(true);

		percent_mode = memnew(Button);
		mode_hbox->add_child(percent_mode);
		percent_mode->set_toggle_mode(true);
		percent_mode->set_button_group(button_group);
		percent_mode->set_focus_mode(Control::FOCUS_NONE);
		percent_mode->set_text(TTR("Percent"));
		percent_mode->set_tooltip_text(TTR("Edit percent"));

		probability_edit = memnew(EditorSpinSlider);
		vbc->add_child(probability_edit);
		probability_edit->set_min(0.0);
		probability_edit->set_max(10.0);
		probability_edit->set_step(0.01);
		probability_edit->set_allow_greater(true);
		probability_edit->set_custom_minimum_size(Size2(200.0 * EDSCALE, 0.0));
	}
	add_child(probability_popup);

	rename_dialog = memnew(ConfirmationDialog);
	{
		VBoxContainer *vbc = memnew(VBoxContainer);
		rename_dialog->add_child(vbc);

		rename_edit = memnew(LineEdit);
		vbc->add_child(rename_edit);
		rename_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		rename_edit->set_custom_minimum_size(Size2(350.0, 0.0));

		rename_dialog->register_text_enter(rename_edit);
	}
	add_child(rename_dialog);

	disk_changed = memnew(ConfirmationDialog);
	{
		VBoxContainer *vbc = memnew(VBoxContainer);
		disk_changed->add_child(vbc);

		Label *dl = memnew(Label);
		dl->set_text(TTR("The following BehaviorTree resources are newer on disk.\nWhat action should be taken?"));
		vbc->add_child(dl);

		disk_changed_list = memnew(Tree);
		vbc->add_child(disk_changed_list);
		disk_changed_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);

		disk_changed->get_ok_button()->set_text(TTR("Reload"));
		disk_changed->add_button(TTR("Resave"), !DisplayServer::get_singleton()->get_swap_cancel_ok(), "resave");
	}

	info_dialog = memnew(AcceptDialog);
	add_child(info_dialog);

	BASE_CONTROL()->add_child(disk_changed);

	GLOBAL_DEF(PropertyInfo(Variant::STRING, "limbo_ai/behavior_tree/behavior_tree_default_dir", PROPERTY_HINT_DIR), "res://ai/trees");
	GLOBAL_DEF(PropertyInfo(Variant::STRING, "limbo_ai/behavior_tree/user_task_dir_1", PROPERTY_HINT_DIR), "res://ai/tasks");
	GLOBAL_DEF(PropertyInfo(Variant::STRING, "limbo_ai/behavior_tree/user_task_dir_2", PROPERTY_HINT_DIR), "");
	GLOBAL_DEF(PropertyInfo(Variant::STRING, "limbo_ai/behavior_tree/user_task_dir_3", PROPERTY_HINT_DIR), "");

	String bt_default_dir = GLOBAL_GET("limbo_ai/behavior_tree/behavior_tree_default_dir");
	save_dialog->set_current_dir(bt_default_dir);
	load_dialog->set_current_dir(bt_default_dir);
	extract_dialog->set_current_dir(bt_default_dir);
}

LimboAIEditor::~LimboAIEditor() {
}

//**** LimboAIEditor ^

//**** LimboAIEditorPlugin

#ifdef LIMBOAI_MODULE
void LimboAIEditorPlugin::apply_changes() {
#elif LIMBOAI_GDEXTENSION
void LimboAIEditorPlugin::_apply_changes() {
#endif
	limbo_ai_editor->apply_changes();
}

void LimboAIEditorPlugin::_bind_methods() {
}

void LimboAIEditorPlugin::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_READY: {
			add_debugger_plugin(memnew(LimboDebuggerPlugin));
			add_inspector_plugin(memnew(EditorInspectorPluginBBPlan));
			EditorInspectorPluginVariableName *var_plugin = memnew(EditorInspectorPluginVariableName);
			var_plugin->set_editor_plan_provider(Callable(limbo_ai_editor, "get_edited_blackboard_plan"));
			add_inspector_plugin(var_plugin);
#ifdef LIMBOAI_MODULE
			// ! Only used in the module version.
			EditorInspectorPluginBBParam *param_plugin = memnew(EditorInspectorPluginBBParam);
			param_plugin->set_plan_getter(Callable(limbo_ai_editor, "get_edited_blackboard_plan"));
			add_inspector_plugin(param_plugin);
#endif // LIMBOAI_MODULE
		} break;
		case NOTIFICATION_ENTER_TREE: {
			// Add BehaviorTree to the list of resources that should open in a new inspector.
			PackedStringArray open_in_new_inspector = EDITOR_GET("interface/inspector/resources_to_open_in_new_inspector");
			if (!open_in_new_inspector.has("BehaviorTree")) {
				open_in_new_inspector.push_back("BehaviorTree");
				EDITOR_SETTINGS()->set_setting("interface/inspector/resources_to_open_in_new_inspector", open_in_new_inspector);
			}
		} break;
	}
}

#ifdef LIMBOAI_MODULE
void LimboAIEditorPlugin::make_visible(bool p_visible) {
#elif LIMBOAI_GDEXTENSION
void LimboAIEditorPlugin::_make_visible(bool p_visible) {
#endif
	limbo_ai_editor->set_visible(p_visible);
}

#ifdef LIMBOAI_MODULE
void LimboAIEditorPlugin::edit(Object *p_object) {
#elif LIMBOAI_GDEXTENSION
void LimboAIEditorPlugin::_edit(Object *p_object) {
#endif
	if (Object::cast_to<BehaviorTree>(p_object)) {
		limbo_ai_editor->edit_bt(Object::cast_to<BehaviorTree>(p_object));
	}
}

#ifdef LIMBOAI_MODULE
bool LimboAIEditorPlugin::handles(Object *p_object) const {
#elif LIMBOAI_GDEXTENSION
bool LimboAIEditorPlugin::_handles(Object *p_object) const {
#endif
	if (Object::cast_to<BehaviorTree>(p_object)) {
		return true;
	}
	return false;
}

#ifdef LIMBOAI_GDEXTENSION
Ref<Texture2D> LimboAIEditorPlugin::_get_plugin_icon() const {
	return LimboUtility::get_singleton()->get_task_icon("LimboAI");
}
#endif // LIMBOAI_GDEXTENSION

LimboAIEditorPlugin::LimboAIEditorPlugin() {
	limbo_ai_editor = memnew(LimboAIEditor());
	limbo_ai_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	MAIN_SCREEN_CONTROL()->add_child(limbo_ai_editor);
	limbo_ai_editor->hide();
	limbo_ai_editor->set_plugin(this);
}

LimboAIEditorPlugin::~LimboAIEditorPlugin() {
}

#endif // ! TOOLS_ENABLED
