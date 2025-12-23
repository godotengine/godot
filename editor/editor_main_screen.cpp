/**************************************************************************/
/*  editor_main_screen.cpp                                                */
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

#include "editor_main_screen.h"

#include "core/io/config_file.h"
#include "editor/docks/editor_dock.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/plugins/editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "scene/gui/button.h"

void LegacyMainScreenContainer::_force_dock_visible(EditorDock *p_dock, CanvasItem *p_child) {
	if (p_dock->is_visible_in_tree()) {
		p_child->show();
	}
}

void LegacyMainScreenContainer::add_child_notify(Node *p_child) {
	EditorMainScreen *ms = EditorNode::get_editor_main_screen();
	if (!ms->adding_plugin || !ms->adding_plugin->has_main_screen()) {
		return;
	}

	EditorDock *dock = memnew(EditorDock);
	dock->set_default_slot(DockConstants::DOCK_SLOT_MAIN_SCREEN);
	dock->set_available_layouts(EditorDock::DOCK_LAYOUT_MAIN_SCREEN);
	dock->set_title(ms->adding_plugin->get_plugin_name());
	dock->set_dock_icon(ms->adding_plugin->get_plugin_icon());
	dock->set_icon_name(ms->adding_plugin->get_plugin_name());
	EditorDockManager::get_singleton()->add_dock(dock);

	ms->adding_plugin->set_meta("_dock", dock);

	CanvasItem *ci_child = Object::cast_to<CanvasItem>(p_child);
	if (ci_child) {
		ci_child->show();
		dock->connect(SceneStringName(visibility_changed), callable_mp(this, &LegacyMainScreenContainer::_force_dock_visible).bind(ci_child));
	}

	p_child->reparent(dock);
}

void EditorMainScreen::_on_tab_changed(int p_tab) {
	EditorNode::get_singleton()->update_distraction_free_mode();

	EditorDock *dock = Object::cast_to<EditorDock>(get_current_tab_control());
	if (!dock) {
		return;
	}
	const String new_main_screen = dock->get_display_title();

	EditorData &editor_data = EditorNode::get_editor_data();
	int plugin_count = editor_data.get_editor_plugin_count();
	for (int i = 0; i < plugin_count; i++) {
		editor_data.get_editor_plugin(i)->notify_main_screen_changed(new_main_screen);
	}
	// selected_plugin->selected_notify();
}

void EditorMainScreen::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			set_tab_alignment(TabBar::ALIGNMENT_CENTER);
			connect("tab_changed", callable_mp(this, &EditorMainScreen::_on_tab_changed));
		} break;

		case NOTIFICATION_READY: {
			// if (EDITOR_3D < buttons.size() && buttons[EDITOR_3D]->is_visible()) {
			// 	// If the 3D editor is enabled, use this as the default.
			// 	select(EDITOR_3D);
			// 	return;
			// }

			// Switch to the first main screen plugin that is enabled. Usually this is
			// 2D, but may be subsequent ones if 2D is disabled in the feature profile.
			// for (int i = 0; i < buttons.size(); i++) {
			// 	Button *editor_button = buttons[i];
			// 	if (editor_button->is_visible()) {
			// 		select(i);
			// 		return;
			// 	}
			// }

			// select(-1);
		} break;
	}
}

void EditorMainScreen::set_button_container(HBoxContainer *p_button_hb) {
	get_internal_container()->reparent(p_button_hb);
	get_internal_container()->set_h_size_flags(SIZE_EXPAND_FILL);
	get_internal_container()->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	button_hb = p_button_hb;
	button_hb->set_h_size_flags(SIZE_EXPAND_FILL);
}

void EditorMainScreen::edit(Object *p_object) {
	EditorPlugin *handling_plugin = EditorNode::get_editor_data().get_handling_main_editor(p_object);
	if (selected_plugin) {
		if (handling_plugin == selected_plugin) {
			selected_plugin->edit(p_object);
		} else {
			selected_plugin->edit(nullptr);
			if (handling_plugin) {
				selected_plugin->make_visible(false);
			}
		}
	}
	selected_plugin = handling_plugin;
	if (selected_plugin) {
		selected_plugin->edit(p_object);
		selected_plugin->make_visible(true);
	}
}

void EditorMainScreen::select_next() {
	if (get_current_tab() < get_tab_count() - 1) {
		set_current_tab(get_current_tab() + 1);
	}
}

void EditorMainScreen::select_prev() {
	if (get_current_tab() > 0) {
		set_current_tab(get_current_tab() - 1);
	}
}

void EditorMainScreen::select_by_name(const String &p_name) {
	ERR_FAIL_COND(p_name.is_empty());

	for (int i = 0; i < get_tab_count(); i++) {
		EditorDock *dock = Object::cast_to<EditorDock>(get_tab_control(i));
		ERR_FAIL_NULL(dock);
		if (dock->get_display_title() == p_name) {
			dock->make_visible();
			return;
		}
	}
	ERR_FAIL_MSG("The editor name '" + p_name + "' was not found.");
}

void EditorMainScreen::select(int p_index) {
	if (EditorNode::get_singleton()->is_changing_scene()) {
		return;
	}

	ERR_FAIL_INDEX(p_index, editor_table.size());

	// if (!buttons[p_index]->is_visible()) { // Button hidden, no editor.
	// 	return;
	// }

	// for (int i = 0; i < buttons.size(); i++) {
	// 	buttons[i]->set_pressed_no_signal(i == p_index);
	// }

	EditorPlugin *new_editor = editor_table[p_index];
	ERR_FAIL_NULL(new_editor);

	if (selected_plugin == new_editor) {
		return;
	}

	if (selected_plugin) {
		selected_plugin->make_visible(false);
	}

	selected_plugin = new_editor;
	selected_plugin->make_visible(true);
	selected_plugin->selected_notify();

	EditorData &editor_data = EditorNode::get_editor_data();
	int plugin_count = editor_data.get_editor_plugin_count();
	for (int i = 0; i < plugin_count; i++) {
		editor_data.get_editor_plugin(i)->notify_main_screen_changed(selected_plugin->get_plugin_name());
	}

	EditorNode::get_singleton()->update_distraction_free_mode();
}

EditorPlugin *EditorMainScreen::get_selected_plugin() const {
	return selected_plugin;
}

bool EditorMainScreen::can_auto_switch_screens() const {
	if (selected_plugin == nullptr) {
		return true;
	}
	// Only allow auto-switching if the selected button is to the left of the Script button.
	for (int i = 0; i < get_tab_count(); i++) {
		EditorDock *dock = Object::cast_to<EditorDock>(get_tab_control(i));
		if (dock->get_display_title() == "Script") {
			// Selected button is at or after the Script button.
			return false;
		}
		if (dock->get_display_title() == selected_plugin->get_plugin_name()) {
			// Selected button is before the Script button.
			return true;
		}
	}
	return false;
}

VBoxContainer *EditorMainScreen::get_control() const {
	return main_screen_vbox;
}

void EditorMainScreen::add_main_plugin(EditorPlugin *p_editor) {
	editor_table.push_back(p_editor);
}

void EditorMainScreen::remove_main_plugin(EditorPlugin *p_editor) {
	if (selected_plugin == p_editor) {
		selected_plugin = nullptr;
	}
	if (p_editor->has_meta("_dock")) {
		EditorDock *dock = Object::cast_to<EditorDock>(p_editor->get_meta("_dock").get_validated_object());
		if (dock) {
			EditorDockManager::get_singleton()->remove_dock(dock);
		}
	}

	editor_table.erase(p_editor);
}

EditorMainScreen::EditorMainScreen() {
	set_theme_type_variation("MainScreenContainer");

	main_screen_vbox = memnew(LegacyMainScreenContainer);
	main_screen_vbox->hide();
	EditorNode::get_singleton()->get_gui_base()->add_child(main_screen_vbox);

	Ref<StyleBoxEmpty> sb;
	sb.instantiate();
	add_theme_style_override(SceneStringName(panel), sb);
}
