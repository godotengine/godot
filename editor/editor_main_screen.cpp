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
#include "core/object/callable_mp.h"
#include "editor/docks/editor_dock.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/plugins/editor_plugin.h"
#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "scene/gui/button.h"

#ifndef DISABLE_DEPRECATED
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
	dock->set_default_slot(EditorDock::DOCK_SLOT_MAIN_SCREEN);
	dock->set_available_layouts(EditorDock::DOCK_LAYOUT_MAIN_SCREEN);
	dock->set_title(ms->adding_plugin->get_plugin_name());
	dock->set_dock_icon(ms->adding_plugin->get_plugin_icon());
	dock->set_icon_name(ms->adding_plugin->get_plugin_name());
	EditorDockManager::get_singleton()->add_dock(dock);

	ms->adding_plugin->set_meta("_dock", dock);

	CanvasItem *ci_child = Object::cast_to<CanvasItem>(p_child);
	if (ci_child) {
		ci_child->show();
		dock->connect(SceneStringName(visibility_changed), callable_mp(this, &LegacyMainScreenContainer::_force_dock_visible).bind(dock, ci_child));
	}

	callable_mp((Node *)p_child, &Node::reparent).call_deferred(dock, false);
}
#endif

void EditorMainScreen::_on_tab_changed(int p_tab) {
	EditorNode::get_singleton()->update_distraction_free_mode();

	EditorDock *dock = get_dock(p_tab);
	if (!dock) {
		return;
	}
	const String new_main_screen = dock->get_display_title();

	EditorData &editor_data = EditorNode::get_editor_data();
	int plugin_count = editor_data.get_editor_plugin_count();
	for (int i = 0; i < plugin_count; i++) {
		editor_data.get_editor_plugin(i)->notify_main_screen_changed(new_main_screen);
	}
}

void EditorMainScreen::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			set_tab_alignment(TabBar::ALIGNMENT_CENTER);
			get_internal_container()->set_alignment(BoxContainer::ALIGNMENT_CENTER);
			connect("tab_changed", callable_mp(this, &EditorMainScreen::_on_tab_changed));
		} break;

		case NOTIFICATION_READY: {
			popup_id = dock_context_popup->get_instance_id();
			for (int i = 0; i < get_tab_count(); i++) {
				EditorDock *dock = get_dock(i);
				if (dock == Node3DEditor::get_singleton()) {
					dock->make_visible();
					break;
				}
			}
		} break;
	}
}

void EditorMainScreen::update_visibility() {
	if (popup_id.is_valid() && !ObjectDB::get_instance(popup_id)) {
		//The popup was freed, likely due to editor exiting.
		return;
	}
	show();
	if (get_tab_count() == 0) {
		// Hide the popup button when there are no tabs.
		set_popup(nullptr);
	} else {
		set_popup(dock_context_popup);
	}
}

DockTabContainer::TabStyle EditorMainScreen::get_tab_style() const {
	return (TabStyle)EDITOR_GET("interface/editor/docks/main_screen_dock_tab_style").operator int();
}

Rect2 EditorMainScreen::get_drag_hint_rect() const {
	const Rect2 tab_rect = get_internal_container()->get_global_rect();
	const Rect2 content_rect = get_global_rect();
	Rect2 final_rect = tab_rect.merge(content_rect);
	float tab_x = get_tab_bar()->get_global_transform().xform(get_tab_bar()->get_tab_rect(0).position).x;
	final_rect.position.x = MIN(tab_x, content_rect.position.x);
	tab_x = get_tab_bar()->get_global_transform().xform(get_tab_bar()->get_tab_rect(-1).get_end()).x;
	final_rect.set_end(Vector2(MAX(tab_x, content_rect.get_end().x), final_rect.get_end().y));
	return final_rect;
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
	if (get_tab_count() == 0) {
		return;
	}
	if (get_current_tab() < get_tab_count() - 1) {
		set_current_tab(get_current_tab() + 1);
	} else {
		set_current_tab(0);
	}
}

void EditorMainScreen::select_prev() {
	if (get_tab_count() == 0) {
		return;
	}
	if (get_current_tab() > 0) {
		set_current_tab(get_current_tab() - 1);
	} else {
		set_current_tab(get_tab_count() - 1);
	}
}

EditorPlugin *EditorMainScreen::get_selected_plugin() const {
	return selected_plugin;
}

bool EditorMainScreen::can_auto_switch_screens() const {
	if (selected_plugin == nullptr) {
		return true;
	}
	// Only allow auto-switching if the selected tab is to the left of the Script tab, or is not in the TabBar.
	bool was_script_tab = false;
	for (int i = 0; i < get_tab_count(); i++) {
		EditorDock *dock = get_dock(i);
		if (dock == ScriptEditor::get_singleton()) {
			was_script_tab = true;
		}
		if (dock->get_display_title() == selected_plugin->get_plugin_name()) {
			return was_script_tab;
		}
	}
	return true;
}

#ifndef DISABLE_DEPRECATED
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
#endif

EditorMainScreen::EditorMainScreen() :
		DockTabContainer(EditorDock::DOCK_SLOT_MAIN_SCREEN) {
	layout = EditorDock::DOCK_LAYOUT_MAIN_SCREEN;
	grid_rect = Rect2i(2, 0, 4, 4);

	set_theme_type_variation("MainScreenContainer");
	set_custom_minimum_size(Size2(0, 80) * EDSCALE);
	set_v_size_flags(Control::SIZE_EXPAND_FILL);
	set_draw_behind_parent(true);

#ifndef DISABLE_DEPRECATED
	main_screen_vbox = memnew(LegacyMainScreenContainer);
	main_screen_vbox->hide();
	EditorNode::get_singleton()->get_gui_base()->add_child(main_screen_vbox);
#endif

	Ref<StyleBoxEmpty> sb;
	sb.instantiate();
	add_theme_style_override(SceneStringName(panel), sb);
}
