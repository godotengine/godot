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
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/inspector_dock.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/panel.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/tab_bar.h"
#include "scene/gui/texture_rect.h"

EditorSceneTabs *EditorSceneTabs::singleton = nullptr;

void EditorSceneTabs::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			tabbar_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("tabbar_background"), SNAME("TabContainer")));
			scene_tabs->add_theme_constant_override("icon_max_width", get_theme_constant(SNAME("class_icon_size"), EditorStringName(Editor)));

			scene_tab_add->set_icon(get_editor_theme_icon(SNAME("Add")));
			scene_tab_add->add_theme_color_override("icon_normal_color", Color(0.6f, 0.6f, 0.6f, 0.8f));

			scene_tab_add_ph->set_custom_minimum_size(scene_tab_add->get_minimum_size());
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			scene_tabs->set_tab_close_display_policy((TabBar::CloseButtonDisplayPolicy)EDITOR_GET("interface/scene_tabs/display_close_button").operator int());
			scene_tabs->set_max_tab_width(int(EDITOR_GET("interface/scene_tabs/maximum_width")) * EDSCALE);
		} break;
	}
}

void EditorSceneTabs::_scene_tab_changed(int p_tab) {
	tab_preview_panel->hide();

	emit_signal("tab_changed", p_tab);
}

void EditorSceneTabs::_scene_tab_script_edited(int p_tab) {
	Ref<Script> scr = EditorNode::get_editor_data().get_scene_root_script(p_tab);
	if (scr.is_valid()) {
		InspectorDock::get_singleton()->edit_resource(scr);
	}
}

void EditorSceneTabs::_scene_tab_closed(int p_tab) {
	emit_signal("tab_closed", p_tab);
}

void EditorSceneTabs::_scene_tab_hovered(int p_tab) {
	if (!bool(EDITOR_GET("interface/scene_tabs/show_thumbnail_on_hover"))) {
		return;
	}
	int current_tab = scene_tabs->get_current_tab();

	if (p_tab == current_tab || p_tab < 0) {
		tab_preview_panel->hide();
	} else {
		String path = EditorNode::get_editor_data().get_scene_path(p_tab);
		if (!path.is_empty()) {
			EditorResourcePreview::get_singleton()->queue_resource_preview(path, this, "_tab_preview_done", p_tab);
		}
	}
}

void EditorSceneTabs::_scene_tab_exit() {
	tab_preview_panel->hide();
}

void EditorSceneTabs::_scene_tab_input(const Ref<InputEvent> &p_input) {
	int tab_id = scene_tabs->get_hovered_tab();
	Ref<InputEventMouseButton> mb = p_input;

	if (mb.is_valid()) {
		if (tab_id >= 0) {
			if (mb->get_button_index() == MouseButton::MIDDLE && mb->is_pressed()) {
				_scene_tab_closed(tab_id);
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
			// Context menu.
			_update_context_menu();

			scene_tabs_context_menu->set_position(scene_tabs->get_screen_position() + mb->get_position());
			scene_tabs_context_menu->reset_size();
			scene_tabs_context_menu->popup();
		}
	}
}

void EditorSceneTabs::_reposition_active_tab(int p_to_index) {
	EditorNode::get_editor_data().move_edited_scene_to_index(p_to_index);
	update_scene_tabs();
}

void EditorSceneTabs::_update_context_menu() {
	scene_tabs_context_menu->clear();
	scene_tabs_context_menu->reset_size();

	int tab_id = scene_tabs->get_hovered_tab();
	bool no_root_node = !EditorNode::get_editor_data().get_edited_scene_root(tab_id);

	scene_tabs_context_menu->add_shortcut(ED_GET_SHORTCUT("editor/new_scene"), EditorNode::FILE_NEW_SCENE);
	if (tab_id >= 0) {
		scene_tabs_context_menu->add_shortcut(ED_GET_SHORTCUT("editor/save_scene"), EditorNode::FILE_SAVE_SCENE);
		_disable_menu_option_if(EditorNode::FILE_SAVE_SCENE, no_root_node);
		scene_tabs_context_menu->add_shortcut(ED_GET_SHORTCUT("editor/save_scene_as"), EditorNode::FILE_SAVE_AS_SCENE);
		_disable_menu_option_if(EditorNode::FILE_SAVE_AS_SCENE, no_root_node);
	}

	scene_tabs_context_menu->add_shortcut(ED_GET_SHORTCUT("editor/save_all_scenes"), EditorNode::FILE_SAVE_ALL_SCENES);
	bool can_save_all_scenes = false;
	for (int i = 0; i < EditorNode::get_editor_data().get_edited_scene_count(); i++) {
		if (!EditorNode::get_editor_data().get_scene_path(i).is_empty() && EditorNode::get_editor_data().get_edited_scene_root(i)) {
			can_save_all_scenes = true;
			break;
		}
	}
	_disable_menu_option_if(EditorNode::FILE_SAVE_ALL_SCENES, !can_save_all_scenes);

	if (tab_id >= 0) {
		scene_tabs_context_menu->add_separator();
		scene_tabs_context_menu->add_item(TTR("Show in FileSystem"), EditorNode::FILE_SHOW_IN_FILESYSTEM);
		_disable_menu_option_if(EditorNode::FILE_SHOW_IN_FILESYSTEM, !ResourceLoader::exists(EditorNode::get_editor_data().get_scene_path(tab_id)));
		scene_tabs_context_menu->add_item(TTR("Play This Scene"), EditorNode::FILE_RUN_SCENE);
		_disable_menu_option_if(EditorNode::FILE_RUN_SCENE, no_root_node);

		scene_tabs_context_menu->add_separator();
		scene_tabs_context_menu->add_shortcut(ED_GET_SHORTCUT("editor/close_scene"), EditorNode::FILE_CLOSE);
		scene_tabs_context_menu->set_item_text(scene_tabs_context_menu->get_item_index(EditorNode::FILE_CLOSE), TTR("Close Tab"));
		scene_tabs_context_menu->add_shortcut(ED_GET_SHORTCUT("editor/reopen_closed_scene"), EditorNode::FILE_OPEN_PREV);
		scene_tabs_context_menu->set_item_text(scene_tabs_context_menu->get_item_index(EditorNode::FILE_OPEN_PREV), TTR("Undo Close Tab"));
		_disable_menu_option_if(EditorNode::FILE_OPEN_PREV, !EditorNode::get_singleton()->has_previous_scenes());
		scene_tabs_context_menu->add_item(TTR("Close Other Tabs"), EditorNode::FILE_CLOSE_OTHERS);
		_disable_menu_option_if(EditorNode::FILE_CLOSE_OTHERS, EditorNode::get_editor_data().get_edited_scene_count() <= 1);
		scene_tabs_context_menu->add_item(TTR("Close Tabs to the Right"), EditorNode::FILE_CLOSE_RIGHT);
		_disable_menu_option_if(EditorNode::FILE_CLOSE_RIGHT, EditorNode::get_editor_data().get_edited_scene_count() == tab_id + 1);
		scene_tabs_context_menu->add_item(TTR("Close All Tabs"), EditorNode::FILE_CLOSE_ALL);
	}
}

void EditorSceneTabs::_disable_menu_option_if(int p_option, bool p_condition) {
	if (p_condition) {
		scene_tabs_context_menu->set_item_disabled(scene_tabs_context_menu->get_item_index(p_option), true);
	}
}

// TODO: This REALLY should be done in a better way than replacing all tabs after almost EVERY action.
void EditorSceneTabs::update_scene_tabs() {
	tab_preview_panel->hide();

	bool show_rb = EDITOR_GET("interface/scene_tabs/show_script_button");

	if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_GLOBAL_MENU)) {
		DisplayServer::get_singleton()->global_menu_clear("_dock");
	}

	// Get all scene names, which may be ambiguous.
	Vector<String> disambiguated_scene_names;
	Vector<String> full_path_names;
	for (int i = 0; i < EditorNode::get_editor_data().get_edited_scene_count(); i++) {
		disambiguated_scene_names.append(EditorNode::get_editor_data().get_scene_title(i));
		full_path_names.append(EditorNode::get_editor_data().get_scene_path(i));
	}

	EditorNode::disambiguate_filenames(full_path_names, disambiguated_scene_names);

	// Workaround to ignore the tab_changed signal from the first added tab.
	scene_tabs->disconnect("tab_changed", callable_mp(this, &EditorSceneTabs::_scene_tab_changed));

	scene_tabs->clear_tabs();
	Ref<Texture2D> script_icon = get_editor_theme_icon(SNAME("Script"));
	for (int i = 0; i < EditorNode::get_editor_data().get_edited_scene_count(); i++) {
		Node *type_node = EditorNode::get_editor_data().get_edited_scene_root(i);
		Ref<Texture2D> icon;
		if (type_node) {
			icon = EditorNode::get_singleton()->get_object_icon(type_node, "Node");
		}

		bool unsaved = EditorUndoRedoManager::get_singleton()->is_history_unsaved(EditorNode::get_editor_data().get_scene_history_id(i));
		scene_tabs->add_tab(disambiguated_scene_names[i] + (unsaved ? "(*)" : ""), icon);

		if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_GLOBAL_MENU)) {
			DisplayServer::get_singleton()->global_menu_add_item("_dock", EditorNode::get_editor_data().get_scene_title(i) + (unsaved ? "(*)" : ""), callable_mp(this, &EditorSceneTabs::_global_menu_scene), Callable(), i);
		}

		if (show_rb && EditorNode::get_editor_data().get_scene_root_script(i).is_valid()) {
			scene_tabs->set_tab_button_icon(i, script_icon);
		}
	}

	if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_GLOBAL_MENU)) {
		DisplayServer::get_singleton()->global_menu_add_separator("_dock");
		DisplayServer::get_singleton()->global_menu_add_item("_dock", TTR("New Window"), callable_mp(this, &EditorSceneTabs::_global_menu_new_window));
	}

	if (scene_tabs->get_tab_count() > 0) {
		scene_tabs->set_current_tab(EditorNode::get_editor_data().get_edited_scene());
	}

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

	// Reconnect after everything is done.
	scene_tabs->connect("tab_changed", callable_mp(this, &EditorSceneTabs::_scene_tab_changed));
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
	scene_tabs->set_current_tab(idx);
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
			int next_tab = EditorNode::get_editor_data().get_edited_scene() + 1;
			next_tab %= EditorNode::get_editor_data().get_edited_scene_count();
			_scene_tab_changed(next_tab);
		}
		if (ED_IS_SHORTCUT("editor/prev_tab", p_event)) {
			int next_tab = EditorNode::get_editor_data().get_edited_scene() - 1;
			next_tab = next_tab >= 0 ? next_tab : EditorNode::get_editor_data().get_edited_scene_count() - 1;
			_scene_tab_changed(next_tab);
		}
	}
}

void EditorSceneTabs::add_extra_button(Button *p_button) {
	tabbar_container->add_child(p_button);
}

void EditorSceneTabs::set_current_tab(int p_tab) {
	scene_tabs->set_current_tab(p_tab);
}

int EditorSceneTabs::get_current_tab() const {
	return scene_tabs->get_current_tab();
}

void EditorSceneTabs::_bind_methods() {
	ADD_SIGNAL(MethodInfo("tab_changed", PropertyInfo(Variant::INT, "tab_index")));
	ADD_SIGNAL(MethodInfo("tab_closed", PropertyInfo(Variant::INT, "tab_index")));

	ClassDB::bind_method("_tab_preview_done", &EditorSceneTabs::_tab_preview_done);
}

EditorSceneTabs::EditorSceneTabs() {
	singleton = this;

	set_process_shortcut_input(true);

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
	scene_tabs->set_auto_translate(false);
	scene_tabs->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tabbar_container->add_child(scene_tabs);

	scene_tabs->connect("tab_changed", callable_mp(this, &EditorSceneTabs::_scene_tab_changed));
	scene_tabs->connect("tab_button_pressed", callable_mp(this, &EditorSceneTabs::_scene_tab_script_edited));
	scene_tabs->connect("tab_close_pressed", callable_mp(this, &EditorSceneTabs::_scene_tab_closed));
	scene_tabs->connect("tab_hovered", callable_mp(this, &EditorSceneTabs::_scene_tab_hovered));
	scene_tabs->connect("mouse_exited", callable_mp(this, &EditorSceneTabs::_scene_tab_exit));
	scene_tabs->connect("gui_input", callable_mp(this, &EditorSceneTabs::_scene_tab_input));
	scene_tabs->connect("active_tab_rearranged", callable_mp(this, &EditorSceneTabs::_reposition_active_tab));
	scene_tabs->connect("resized", callable_mp(this, &EditorSceneTabs::update_scene_tabs));

	scene_tabs_context_menu = memnew(PopupMenu);
	tabbar_container->add_child(scene_tabs_context_menu);
	scene_tabs_context_menu->connect("id_pressed", callable_mp(EditorNode::get_singleton(), &EditorNode::trigger_menu_option).bind(false));

	scene_tab_add = memnew(Button);
	scene_tab_add->set_flat(true);
	scene_tab_add->set_tooltip_text(TTR("Add a new scene."));
	scene_tabs->add_child(scene_tab_add);
	scene_tab_add->connect("pressed", callable_mp(EditorNode::get_singleton(), &EditorNode::trigger_menu_option).bind(EditorNode::FILE_NEW_SCENE, false));

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
}
