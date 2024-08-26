/**************************************************************************/
/*  editor_scene_tabs.cpp                                                 */
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

#include "editor_scene_tabs.h"

#include "editor/editor_node.h"
#include "editor/editor_resource_preview.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/inspector_dock.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/panel.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/tab_bar.h"
#include "scene/gui/texture_rect.h"

static const String EDITOR_TABS_CONFIG_SECTION = "EditorTabs";

EditorSceneTabs *EditorSceneTabs::singleton = nullptr;
MainEditorTabContent EditorSceneTabs::tab_content_setting = MainEditorTabContent::SCENES_ONLY;

void EditorSceneTabs::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			tabbar_panel->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("tabbar_background"), SNAME("TabContainer")));
			scene_tabs->add_theme_constant_override("icon_max_width", get_theme_constant(SNAME("class_icon_size"), EditorStringName(Editor)));

			scene_tab_add->set_icon(get_editor_theme_icon(SNAME("Add")));
			scene_tab_add->add_theme_color_override("icon_normal_color", Color(0.6f, 0.6f, 0.6f, 0.8f));

			scene_tab_add_ph->set_custom_minimum_size(scene_tab_add->get_minimum_size());
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (EditorSettings::get_singleton()->check_changed_settings_in_group("interface/scene_tabs")) {
				scene_tabs->set_tab_close_display_policy((TabBar::CloseButtonDisplayPolicy)EDITOR_GET("interface/scene_tabs/display_close_button").operator int());
				scene_tabs->set_max_tab_width(int(EDITOR_GET("interface/scene_tabs/maximum_width")) * EDSCALE);
				_scene_tabs_resized();
			}
		} break;

		case NOTIFICATION_PROCESS: {
			if (_selected_index >= 0) {
				int selected_index = _selected_index;
				_selected_index = -1;
				if (selected_index >= 0 && selected_index < _tabs.size()) {
					_tabs[selected_index]->emit_signal("selected", _tabs[selected_index]);
				}
			}
		} break;
	}
}

void EditorSceneTabs::_scene_tab_changed(int p_tab) {
	tab_preview_panel->hide();

	_current_tab_index = scene_tabs->get_current_tab();
	_tabs[p_tab]->emit_signal("selected", _tabs[p_tab]);
}

void EditorSceneTabs::_scene_tab_button_pressed(int p_tab) {
	_tabs[p_tab]->emit_signal("tab_button_pressed", _tabs[p_tab]);
}

void EditorSceneTabs::_scene_tab_closed_pressed(int p_tab) {
	remove_tab(_tabs[p_tab], true);
}

void EditorSceneTabs::_scene_tab_hovered(int p_tab) {
	if (!bool(EDITOR_GET("interface/scene_tabs/show_thumbnail_on_hover"))) {
		return;
	}
	int current_tab = scene_tabs->get_current_tab();

	if (p_tab == current_tab || p_tab < 0) {
		tab_preview_panel->hide();
	} else {
		String path = _tabs[p_tab]->get_resource_path();
		if (!path.is_empty()) {
			EditorResourcePreview::get_singleton()->queue_resource_preview(path, this, "_tab_preview_done", p_tab);
		}
	}
}

void EditorSceneTabs::_scene_tab_exit() {
	tab_preview_panel->hide();
}

void EditorSceneTabs::_scene_tab_input(const Ref<InputEvent> &p_input) {
	int tab_index = scene_tabs->get_hovered_tab();
	Ref<InputEventMouseButton> mb = p_input;

	if (mb.is_valid()) {
		if (tab_index >= 0) {
			if (mb->get_button_index() == MouseButton::MIDDLE && mb->is_pressed()) {
				remove_tab(_tabs[tab_index], true);
			}
		} else if (mb->get_button_index() == MouseButton::LEFT && mb->is_double_click()) {
			int tab_buttons = 0;
			if (scene_tabs->get_offset_buttons_visible()) {
				tab_buttons = get_theme_icon(SNAME("increment"), SNAME("TabBar"))->get_width() + get_theme_icon(SNAME("decrement"), SNAME("TabBar"))->get_width();
			}

			if ((is_layout_rtl() && mb->get_position().x > tab_buttons) || (!is_layout_rtl() && mb->get_position().x < scene_tabs->get_size().width - tab_buttons)) {
				EditorNode::get_singleton()->trigger_menu_option(EditorNode::FILE_NEW_SCENE, true);
			}
		}
		if (mb->get_button_index() == MouseButton::RIGHT && mb->is_pressed()) {
			// The tab must be selected to have the right context menu.
			if (tab_index >= 0) {
				select_tab(_tabs[tab_index]);
			}

			// Context menu.
			_update_context_menu();

			scene_tabs_context_menu->set_position(scene_tabs->get_screen_position() + mb->get_position());
			scene_tabs_context_menu->reset_size();
			scene_tabs_context_menu->popup();
		}
	}
}

void EditorSceneTabs::unhandled_key_input(const Ref<InputEvent> &p_event) {
	if (!tab_preview_panel->is_visible()) {
		return;
	}

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->is_action_pressed(SNAME("ui_cancel"), false, true)) {
		tab_preview_panel->hide();
	}
}

void EditorSceneTabs::_reposition_active_tab(int p_to_index) {
	if (_current_tab_index < 0 || _current_tab_index >= _tabs.size()) {
		return;
	}

	EditorTab *tab = _tabs[_current_tab_index];
	_tabs.remove_at(_current_tab_index);
	_tabs.insert(p_to_index, tab);

	_current_tab_index = p_to_index;
}

void EditorSceneTabs::_update_context_menu() {
	scene_tabs_context_menu->clear();
	scene_tabs_context_menu->reset_size();

	if (_current_tab_index < 0 || _current_tab_index >= _tabs.size()) {
		return;
	}

	_tabs[_current_tab_index]->emit_signal(SNAME("context_menu_needed"), _tabs[_current_tab_index], scene_tabs_context_menu);

	scene_tabs_context_menu->add_item(TTR("Close Other Tabs"), MenuOptions::CLOSE_OTHERS);
	_disable_menu_option_if(MenuOptions::CLOSE_OTHERS, _tabs.size() <= 1);
	scene_tabs_context_menu->add_item(TTR("Close Tabs to the Right"), MenuOptions::CLOSE_RIGHT);
	_disable_menu_option_if(MenuOptions::CLOSE_RIGHT, _tabs.size() == _current_tab_index + 1);
	scene_tabs_context_menu->add_item(TTR("Close All Tabs"), MenuOptions::CLOSE_ALL);
}

void EditorSceneTabs::_context_menu_id_pressed(int p_option) {
	if (p_option < CLOSE_OTHERS) {
		// Custom context menu from the context_menu_needed.
		EditorTab *tab = get_current_tab();
		if (tab) {
			tab->emit_signal("context_menu_pressed", _tabs[_current_tab_index], p_option);
		}
		return;
	}

	switch (p_option) {
		case CLOSE_OTHERS: {
			_tabs_to_close.clear();
			for (int i = 0; i < _tabs.size(); i++) {
				if (i == _current_tab_index) {
					continue;
				}
				_tabs_to_close.push_back(_tabs[i]);
			}
			if (_tabs_to_close.size() > 0) {
				remove_tab(_tabs_to_close[0], true);
			}
		} break;
		case CLOSE_RIGHT: {
			_tabs_to_close.clear();
			for (int i = _current_tab_index + 1; i < _tabs.size(); i++) {
				_tabs_to_close.push_back(_tabs[i]);
			}
			if (_tabs_to_close.size() > 0) {
				remove_tab(_tabs_to_close[0], true);
			}
		} break;
		case CLOSE_ALL: {
			_tabs_to_close.clear();
			for (int i = 0; i < _tabs.size(); i++) {
				_tabs_to_close.push_back(_tabs[i]);
			}
			if (_tabs_to_close.size() > 0) {
				remove_tab(_tabs_to_close[0], true);
			}
		} break;
	}
}

void EditorSceneTabs::_disable_menu_option_if(int p_option, bool p_condition) {
	if (p_condition) {
		scene_tabs_context_menu->set_item_disabled(scene_tabs_context_menu->get_item_index(p_option), true);
	}
}

void EditorSceneTabs::update_scene_tabs() {
	static bool menu_initialized = false;
	tab_preview_panel->hide();

	if (menu_initialized && scene_tabs->get_tab_count() == _tabs.size()) {
		_update_tab_titles();
		return;
	}
	menu_initialized = true;

	if (NativeMenu::get_singleton()->has_feature(NativeMenu::FEATURE_GLOBAL_MENU)) {
		RID dock_rid = NativeMenu::get_singleton()->get_system_menu(NativeMenu::DOCK_MENU_ID);
		NativeMenu::get_singleton()->clear(dock_rid);
	}

	scene_tabs->set_block_signals(true);
	scene_tabs->set_tab_count(_tabs.size());
	scene_tabs->set_block_signals(false);

	if (NativeMenu::get_singleton()->has_feature(NativeMenu::FEATURE_GLOBAL_MENU)) {
		RID dock_rid = NativeMenu::get_singleton()->get_system_menu(NativeMenu::DOCK_MENU_ID);
		for (int i = 0; i < _tabs.size(); i++) {
			int global_menu_index = NativeMenu::get_singleton()->add_item(dock_rid, _tabs[i]->get_name(), callable_mp(this, &EditorSceneTabs::_global_menu_scene), Callable(), i);
			scene_tabs->set_tab_metadata(i, global_menu_index);
		}
		NativeMenu::get_singleton()->add_separator(dock_rid);
		NativeMenu::get_singleton()->add_item(dock_rid, TTR("New Window"), callable_mp(this, &EditorSceneTabs::_global_menu_new_window));
	}

	_update_tab_titles();
}

void EditorSceneTabs::_update_tab_titles() {
	// Get all scene names, which may be ambiguous.
	Vector<String> disambiguated_scene_names;
	Vector<String> full_path_names;
	for (EditorTab *&tab : _tabs) {
		// When closing, the state could be delete, skipping the
		// update prevents null ref or error lookups in for closed tabs.
		if (!tab->get_closing()) {
			tab->emit_signal(SNAME("update_needed"), tab);
		}
		disambiguated_scene_names.append(tab->get_name());
		full_path_names.append(tab->get_resource_path());
	}
	EditorNode::disambiguate_filenames(full_path_names, disambiguated_scene_names);

	for (int i = 0; i < _tabs.size(); i++) {
		scene_tabs->set_tab_title(i, disambiguated_scene_names[i]);
		scene_tabs->set_tab_icon(i, _tabs[i]->get_icon());
		scene_tabs->set_tab_button_icon(i, _tabs[i]->get_tab_button_icon());

		if (NativeMenu::get_singleton()->has_feature(NativeMenu::FEATURE_GLOBAL_MENU)) {
			RID dock_rid = NativeMenu::get_singleton()->get_system_menu(NativeMenu::DOCK_MENU_ID);
			int global_menu_index = scene_tabs->get_tab_metadata(i);
			NativeMenu::get_singleton()->set_item_text(dock_rid, global_menu_index, _tabs[i]->get_name());
			NativeMenu::get_singleton()->set_item_tag(dock_rid, global_menu_index, i);
		}
	}

	if (_current_tab_index >= 0) {
		if (scene_tabs->get_tab_count() > 0 && scene_tabs->get_current_tab() != _current_tab_index && _current_tab_index < scene_tabs->get_tab_count()) {
			scene_tabs->set_block_signals(true);
			scene_tabs->set_current_tab(_current_tab_index);
			scene_tabs->set_block_signals(false);
		}
	}

	_scene_tabs_resized();
}

void EditorSceneTabs::_scene_tabs_resized() {
	const Size2 add_button_size = Size2(scene_tab_add->get_size().x, scene_tabs->get_size().y);
	if (scene_tabs->get_offset_buttons_visible()) {
		// Move the add button to a fixed position.
		if (scene_tab_add->get_parent() == scene_tabs) {
			scene_tabs->remove_child(scene_tab_add);
			scene_tab_add_ph->add_child(scene_tab_add);
			scene_tab_add->set_rect(Rect2(Point2(), add_button_size));
		}
	} else {
		// Move the add button to be after the last tab.
		if (scene_tab_add->get_parent() == scene_tab_add_ph) {
			scene_tab_add_ph->remove_child(scene_tab_add);
			scene_tabs->add_child(scene_tab_add);
		}

		if (scene_tabs->get_tab_count() == 0) {
			scene_tab_add->set_rect(Rect2(Point2(), add_button_size));
			return;
		}

		Rect2 last_tab = scene_tabs->get_tab_rect(scene_tabs->get_tab_count() - 1);
		int hsep = scene_tabs->get_theme_constant(SNAME("h_separation"));
		if (scene_tabs->is_layout_rtl()) {
			scene_tab_add->set_rect(Rect2(Point2(last_tab.position.x - add_button_size.x - hsep, last_tab.position.y), add_button_size));
		} else {
			scene_tab_add->set_rect(Rect2(Point2(last_tab.position.x + last_tab.size.width + hsep, last_tab.position.y), add_button_size));
		}
	}
}

void EditorSceneTabs::_tab_preview_done(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, const Variant &p_udata) {
	int p_tab = p_udata;
	if (p_preview.is_valid()) {
		tab_preview->set_texture(p_preview);

		Rect2 rect = scene_tabs->get_tab_rect(p_tab);
		rect.position += scene_tabs->get_global_position();
		tab_preview_panel->set_global_position(rect.position + Vector2(0, rect.size.height));
		tab_preview_panel->show();
	}
}

void EditorSceneTabs::_global_menu_scene(const Variant &p_tag) {
	int idx = (int)p_tag;
	_set_current_tab_index(idx);
}

void EditorSceneTabs::_global_menu_new_window(const Variant &p_tag) {
	if (OS::get_singleton()->get_main_loop()) {
		List<String> args;
		args.push_back("-p");
		OS::get_singleton()->create_instance(args);
	}
}

void EditorSceneTabs::shortcut_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventKey> k = p_event;
	if ((k.is_valid() && k->is_pressed() && !k->is_echo()) || Object::cast_to<InputEventShortcut>(*p_event)) {
		if (ED_IS_SHORTCUT("editor/next_tab", p_event)) {
			nav_tabs->popup_dialog(_tabs, true);
		}
		if (ED_IS_SHORTCUT("editor/prev_tab", p_event)) {
			nav_tabs->popup_dialog(_tabs, false);
		}
	}
}

EditorTab *EditorSceneTabs::add_tab() {
	EditorTab *p_tab = memnew(EditorTab);
	p_tab->update_last_used();
	_tabs.append(p_tab);
	return p_tab;
}

EditorTab *EditorSceneTabs::get_current_tab() const {
	if (_current_tab_index < 0) {
		return nullptr;
	}
	return _tabs[_current_tab_index];
}

int EditorSceneTabs::get_current_tab_index() const {
	return _current_tab_index;
}

void EditorSceneTabs::remove_tab(EditorTab *p_tab, bool p_user_removal) {
	int idx = _get_tab_index(p_tab);
	if (idx < 0) {
		return;
	}

	if (p_tab->get_closing()) {
		// Already in the process of closing.
		// Prevent problems when the source can recall remove_tab
		// when already in remove_tab.
		p_tab->set_cancel(false);
		return;
	}

	if (p_user_removal) {
		p_tab->set_closing(true);
		p_tab->set_cancel(false);

		p_tab->emit_signal("closing", p_tab);

		if (p_tab->get_cancel()) {
			// User has cancelled the tab closing.
			p_tab->set_cancel(false);
			p_tab->set_closing(false);
			return;
		}

		// Maybe in the signal, remove_tab was called and the tab
		// would not be in _tabs anymore.
		idx = _get_tab_index(p_tab);
		if (idx < 0) {
			return;
		}
	}

	_tabs.remove_at(idx);

	if (_tabs_to_close.size() > 0) {
		if (_tabs_to_close[0] == p_tab) {
			_tabs_to_close.remove_at(0);
		} else {
			// Something is wrong. It should be that tab.
			_tabs_to_close.clear();
		}
	}

	// Sometimes, the tab is removed while in the emit signal of the context menu
	// and freeing objects while emitting is not a good idea.
	callable_mp(this, &EditorSceneTabs::_memdel_edit_tab_deferred).call_deferred(p_tab);

	// Force a reselect of the next tab.
	_current_tab_index = -1;

	// Select the previous tab.
	int new_current_tab = _get_most_recent_tab_index();
	if (new_current_tab >= 0) {
		select_tab_index(new_current_tab);
	} else {
		update_scene_tabs();
	}

	if (_tabs_to_close.size() > 0) {
		// Deferred call caused problem with EditorData::check_and_update_scene and the EditorProgress.
		EditorSceneTabs::remove_tab(_tabs_to_close[0], true);
	}
}

void EditorSceneTabs::_memdel_edit_tab_deferred(EditorTab *p_tab) {
	memdelete(p_tab);
}

void EditorSceneTabs::select_tab(const EditorTab *p_tab) {
	select_tab_index(_get_tab_index(p_tab));
}

void EditorSceneTabs::select_tab_index(int p_index) {
	if (p_index >= 0 && p_index < _tabs.size()) {
		if (_current_tab_index != p_index) {
			_current_tab_index = p_index;
			update_scene_tabs();
			_tabs[_current_tab_index]->update_last_used();
			// We cannot directly emit selected when flushing message que because it could trigger
			// a reloading of a scene in EditorNode if the scene was updated which cause an error
			// in ProgressDialog::add_task because it's not authorized in a deferred call.
			if (MessageQueue::get_singleton()->is_flushing()) {
				_selected_index = _current_tab_index;
			} else {
				_tabs[_current_tab_index]->emit_signal("selected", _tabs[_current_tab_index]);
			}
		}
	}
}

EditorTab *EditorSceneTabs::get_tab_by_state(Variant p_state) {
	for (EditorTab *tab : _tabs) {
		if (tab->get_state() == p_state) {
			return tab;
		}
	}
	return nullptr;
}

int EditorSceneTabs::_get_tab_index(const EditorTab *p_tab) const {
	int idx = 0;
	for (const EditorTab *tab : _tabs) {
		if (tab == p_tab) {
			return idx;
		}
		idx++;
	}
	return -1;
}

int EditorSceneTabs::_get_tab_index_by_resource_path(const String &p_resource_path) const {
	int idx = 0;
	for (const EditorTab *tab : _tabs) {
		if (tab->get_resource_path() == p_resource_path) {
			return idx;
		}
		idx++;
	}
	return -1;
}

int EditorSceneTabs::_get_tab_index_by_name(const String &p_name) const {
	int idx = 0;
	for (const EditorTab *tab : _tabs) {
		if (tab->get_name().replace("(*)", "") == p_name) {
			return idx;
		}
		idx++;
	}
	return -1;
}

int EditorSceneTabs::_get_most_recent_tab_index() const {
	int idx = 0;
	int most_recent = -1;
	uint64_t higher_last_used = 0;
	for (const EditorTab *tab : _tabs) {
		if (tab->get_last_used() > higher_last_used) {
			most_recent = idx;
			higher_last_used = tab->get_last_used();
		}
		idx++;
	}
	return most_recent;
}

void EditorSceneTabs::add_extra_button(Button *p_button) {
	tabbar_container->add_child(p_button);
}

void EditorSceneTabs::cancel_close_process() {
	_tabs_to_close.clear();
}

void EditorSceneTabs::_set_current_tab_index(int p_tab) {
	if (p_tab >= 0 && p_tab < scene_tabs->get_tab_count()) {
		_current_tab_index = p_tab;
		scene_tabs->set_current_tab(p_tab);
	}
}

void EditorSceneTabs::save_tabs_layout(Ref<ConfigFile> p_layout) {
	Array tabs_order;
	Array tabs_last_used_index;

	for (EditorTab *tab : _tabs) {
		if (!tab->get_resource_path().is_empty()) {
			tabs_order.append(tab->get_resource_path());
		} else {
			tabs_order.append(tab->get_name().replace("(*)", ""));
		}
	}

	p_layout->set_value(EDITOR_TABS_CONFIG_SECTION, "tabs_order", tabs_order);
	p_layout->set_value(EDITOR_TABS_CONFIG_SECTION, "tab_selected_idx", _current_tab_index);
}

void EditorSceneTabs::restore_tabs_layout(Ref<ConfigFile> p_layout) {
	if (tab_content_setting != MainEditorTabContent::ALL) {
		return;
	}

	if (p_layout->has_section_key(EDITOR_TABS_CONFIG_SECTION, "tabs_order")) {
		int index = 0;
		Array tabs_config = p_layout->get_value(EDITOR_TABS_CONFIG_SECTION, "tabs_order");
		Array tabs_last_used = p_layout->get_value(EDITOR_TABS_CONFIG_SECTION, "tabs_last_used_index", Array());
		for (const String tab_info : tabs_config) {
			int tab_index_to_move;
			if (tab_info.begins_with("res://")) {
				tab_index_to_move = _get_tab_index_by_resource_path(tab_info);
			} else {
				tab_index_to_move = _get_tab_index_by_name(tab_info);
			}
			if (tab_index_to_move >= 0 && tab_index_to_move > index) {
				EditorTab *old_tab = _tabs[index];
				_tabs.write[index] = _tabs[tab_index_to_move];
				_tabs.write[tab_index_to_move] = old_tab;
			}

			index++;
		}
		update_scene_tabs();
	}

	// Editor tabs.
	if (p_layout->has_section_key(EDITOR_TABS_CONFIG_SECTION, "tab_selected_idx")) {
		int tab_selected_idx = p_layout->get_value(EDITOR_TABS_CONFIG_SECTION, "tab_selected_idx");
		if (tab_selected_idx >= 0 && tab_selected_idx < _tabs.size()) {
			callable_mp(this, &EditorSceneTabs::select_tab_index).call_deferred(tab_selected_idx);
		}
	}
}

void EditorSceneTabs::_bind_methods() {
	ClassDB::bind_method("_tab_preview_done", &EditorSceneTabs::_tab_preview_done);
}

EditorSceneTabs::EditorSceneTabs() {
	singleton = this;
	tab_content_setting = (MainEditorTabContent)(int)EDITOR_GET("interface/editor/main_editor_tab_content");

	set_process(true);
	set_process_shortcut_input(true);
	set_process_unhandled_key_input(true);

	tabbar_panel = memnew(PanelContainer);
	add_child(tabbar_panel);
	tabbar_container = memnew(HBoxContainer);
	tabbar_panel->add_child(tabbar_container);

	scene_tabs = memnew(TabBar);
	scene_tabs->set_select_with_rmb(true);
	scene_tabs->add_tab("unsaved");
	scene_tabs->set_tab_close_display_policy((TabBar::CloseButtonDisplayPolicy)EDITOR_GET("interface/scene_tabs/display_close_button").operator int());
	scene_tabs->set_max_tab_width(int(EDITOR_GET("interface/scene_tabs/maximum_width")) * EDSCALE);
	scene_tabs->set_drag_to_rearrange_enabled(true);
	scene_tabs->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	scene_tabs->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tabbar_container->add_child(scene_tabs);

	scene_tabs->connect("tab_changed", callable_mp(this, &EditorSceneTabs::_scene_tab_changed));
	scene_tabs->connect("tab_button_pressed", callable_mp(this, &EditorSceneTabs::_scene_tab_button_pressed));
	scene_tabs->connect("tab_close_pressed", callable_mp(this, &EditorSceneTabs::_scene_tab_closed_pressed));
	scene_tabs->connect("tab_hovered", callable_mp(this, &EditorSceneTabs::_scene_tab_hovered));
	scene_tabs->connect(SceneStringName(mouse_exited), callable_mp(this, &EditorSceneTabs::_scene_tab_exit));
	scene_tabs->connect(SceneStringName(gui_input), callable_mp(this, &EditorSceneTabs::_scene_tab_input));
	scene_tabs->connect("active_tab_rearranged", callable_mp(this, &EditorSceneTabs::_reposition_active_tab));
	scene_tabs->connect(SceneStringName(resized), callable_mp(this, &EditorSceneTabs::_scene_tabs_resized), CONNECT_DEFERRED);

	scene_tabs_context_menu = memnew(PopupMenu);
	tabbar_container->add_child(scene_tabs_context_menu);
	scene_tabs_context_menu->connect(SceneStringName(id_pressed), callable_mp(this, &EditorSceneTabs::_context_menu_id_pressed));

	scene_tab_add = memnew(Button);
	scene_tab_add->set_flat(true);
	scene_tab_add->set_tooltip_text(TTR("Add a new scene."));
	scene_tabs->add_child(scene_tab_add);
	scene_tab_add->connect(SceneStringName(pressed), callable_mp(EditorNode::get_singleton(), &EditorNode::trigger_menu_option).bind(EditorNode::FILE_NEW_SCENE, false));

	scene_tab_add_ph = memnew(Control);
	scene_tab_add_ph->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	scene_tab_add_ph->set_custom_minimum_size(scene_tab_add->get_minimum_size());
	tabbar_container->add_child(scene_tab_add_ph);

	// On-hover tab preview.

	Control *tab_preview_anchor = memnew(Control);
	tab_preview_anchor->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	add_child(tab_preview_anchor);

	tab_preview_panel = memnew(Panel);
	tab_preview_panel->set_size(Size2(100, 100) * EDSCALE);
	tab_preview_panel->hide();
	tab_preview_panel->set_self_modulate(Color(1, 1, 1, 0.7));
	tab_preview_anchor->add_child(tab_preview_panel);

	tab_preview = memnew(TextureRect);
	tab_preview->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	tab_preview->set_size(Size2(96, 96) * EDSCALE);
	tab_preview->set_position(Point2(2, 2) * EDSCALE);
	tab_preview_panel->add_child(tab_preview);

	nav_tabs = memnew(EditorNavTabs);
	add_child(nav_tabs);
	nav_tabs->connect("selected", callable_mp(this, &EditorSceneTabs::select_tab));
}

EditorSceneTabs::~EditorSceneTabs() {
	for (EditorTab *&tab : _tabs) {
		memdelete(tab);
	}
	_tabs.clear();
}
