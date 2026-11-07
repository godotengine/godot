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

#include "core/object/callable_mp.h"
#include "editor/docks/editor_dock.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/editor_node.h"
#include "editor/plugins/editor_plugin.h"
#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"
#include "scene/gui/tab_bar.h"

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

	callable_mp(p_child, &Node::reparent).call_deferred(dock, false);
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
			set_popup(nullptr);

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

DockTabContainer::TabStyle EditorMainScreen::get_tab_style() const {
	return (TabStyle)EDITOR_GET("interface/editor/docks/main_screen_dock_tab_style").operator int();
}

Rect2 EditorMainScreen::get_drag_hint_rect() const {
	const Rect2 tab_rect = get_internal_container()->get_global_rect();
	const Rect2 content_rect = get_global_rect();
	Rect2 final_rect = tab_rect.merge(content_rect);
	if (get_tab_count() > 0) {
		float tab_x = get_tab_bar()->get_global_transform().xform(get_tab_bar()->get_tab_rect(0).position).x;
		final_rect.position.x = MIN(tab_x, content_rect.position.x);
		tab_x = get_tab_bar()->get_global_transform().xform(get_tab_bar()->get_tab_rect(-1).get_end()).x;
		final_rect.set_end(Vector2(MAX(tab_x, content_rect.get_end().x), final_rect.get_end().y));
	}
	return final_rect;
}

bool EditorMainScreen::can_switch_dock() const {
	if (get_current_tab() == -1) {
		return true;
	}
	EditorDock *dock = get_dock(get_current_tab());
	return dock->is_allow_switch_screen();
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

#ifndef DISABLE_DEPRECATED
VBoxContainer *EditorMainScreen::get_control() const {
	return main_screen_vbox;
}

void EditorMainScreen::add_main_plugin(EditorPlugin *p_editor) {
	editor_table.push_back(p_editor);
}

void EditorMainScreen::remove_main_plugin(EditorPlugin *p_editor) {
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
	// The right half of the center cells is the split panel (EditorMainScreenSplit).
	grid_rect = Rect2i(2, 0, 2, 4);

	set_theme_type_variation("MainScreenContainer");
	set_custom_minimum_size(Size2(80, 80) * EDSCALE);
	set_h_size_flags(Control::SIZE_EXPAND_FILL);
	set_v_size_flags(Control::SIZE_EXPAND_FILL);
	set_draw_behind_parent(true);
	get_tab_bar()->set_mouse_filter(Control::MOUSE_FILTER_PASS);

#ifndef DISABLE_DEPRECATED
	main_screen_vbox = memnew(LegacyMainScreenContainer);
	main_screen_vbox->hide();
	EditorNode::get_singleton()->get_gui_base()->add_child(main_screen_vbox);
#endif

	Ref<StyleBoxEmpty> sb;
	sb.instantiate();
	add_theme_style_override(SceneStringName(panel), sb);
}

DockTabContainer::TabStyle EditorMainScreenSplit::get_tab_style() const {
	return (TabStyle)EDITOR_GET("interface/editor/docks/main_screen_dock_tab_style").operator int();
}

bool EditorMainScreenSplit::can_switch_dock() const {
	if (get_current_tab() == -1) {
		return true;
	}
	EditorDock *dock = get_dock(get_current_tab());
	return dock && dock->is_allow_switch_screen();
}

void EditorMainScreenSplit::_update_header_height() {
	int height = 0;
	if (header_reference && header_reference->is_visible()) {
		Ref<StyleBox> tabbar_style = get_theme_stylebox(SNAME("tabbar_background"));
		height = header_reference->get_size().height;
		if (tabbar_style.is_valid()) {
			height -= tabbar_style->get_margin(SIDE_TOP) + tabbar_style->get_margin(SIDE_BOTTOM);
		}
	}
	get_internal_container()->set_custom_minimum_size(Size2(0, MAX(0, height)));
}

void EditorMainScreenSplit::_queue_update_title_tabs() {
	if (title_tabs_queued) {
		return;
	}
	title_tabs_queued = true;
	callable_mp(this, &EditorMainScreenSplit::_update_title_tabs).call_deferred();
}

void EditorMainScreenSplit::_update_title_tabs() {
	title_tabs_queued = false;
	if (!title_tabs) {
		return;
	}
	title_tabs->set_tab_count(get_tab_count());
	for (int i = 0; i < get_tab_count(); i++) {
		title_tabs->set_tab_title(i, get_tab_title(i));
		title_tabs->set_tab_icon(i, get_tab_icon(i));
		title_tabs->set_tab_tooltip(i, get_tab_tooltip(i));
	}
	title_tabs->set_visible(get_tab_count() > 0);
	// Selected while the dock is shown, like the main screen tabs.
	title_tabs->set_current_tab(is_visible() ? get_current_tab() : -1);
}

void EditorMainScreenSplit::_update_title_tabs_theme() {
	// The same look as the main screen tabs next to them (see TabContainer::_on_theme_changed()).
	const StringName type = SNAME("MainScreenContainer");
	title_tabs->begin_bulk_theme_override();
	for (const char *name : { "tab_unselected", "tab_hovered", "tab_selected", "tab_disabled", "tab_focus" }) {
		title_tabs->add_theme_style_override(name, get_theme_stylebox(name, type));
	}
	for (const char *name : { "font_selected_color", "font_hovered_color", "font_unselected_color", "font_disabled_color", "font_outline_color", "icon_selected_color", "icon_hovered_color", "icon_unselected_color", "icon_disabled_color" }) {
		title_tabs->add_theme_color_override(name, get_theme_color(name, type));
	}
	title_tabs->add_theme_font_override(SceneStringName(font), get_theme_font(SceneStringName(font), type));
	title_tabs->add_theme_font_size_override(SceneStringName(font_size), get_theme_font_size(SceneStringName(font_size), type));
	title_tabs->add_theme_constant_override(SNAME("h_separation"), get_theme_constant(SNAME("icon_separation"), type));
	title_tabs->add_theme_constant_override(SNAME("tab_separation"), get_theme_constant(SNAME("tab_separation"), type));
	title_tabs->add_theme_constant_override(SNAME("icon_max_width"), get_theme_constant(SNAME("icon_max_width"), type));
	title_tabs->add_theme_constant_override(SNAME("outline_size"), get_theme_constant(SNAME("outline_size"), type));
	title_tabs->end_bulk_theme_override();
}

void EditorMainScreenSplit::_title_tab_clicked(int p_tab) {
	if (maximized == MAXIMIZED_MAIN) {
		EditorNode::get_singleton()->set_distraction_free_mode(false);
	}
	EditorDock *dock = get_dock(p_tab);
	if (dock) {
		EditorDockManager::get_singleton()->focus_dock(dock);
	}
	_queue_update_title_tabs();
}

void EditorMainScreenSplit::_maximize_toggled(bool p_pressed) {
	EditorNode::get_singleton()->set_workspace_split_maximized(p_pressed);
}

void EditorMainScreenSplit::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			connect("tab_changed", callable_mp(this, &EditorMainScreenSplit::_queue_update_title_tabs).unbind(1));
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_update_header_height();
			maximize_button->set_button_icon(get_editor_theme_icon(SNAME("DistractionFree")));
			// Pressed is shown by the panel filling the workspace, as for the scene tabs' button.
			maximize_button->add_theme_style_override(SceneStringName(pressed), get_theme_stylebox(CoreStringName(normal), SNAME("FlatMenuButton")));
			if (title_tabs) {
				_update_title_tabs_theme();
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			_queue_update_title_tabs();
		} break;
	}
}

void EditorMainScreenSplit::set_header_reference(Control *p_control) {
	header_reference = p_control;
	header_reference->connect(SceneStringName(resized), callable_mp(this, &EditorMainScreenSplit::_update_header_height));
	header_reference->connect(SceneStringName(visibility_changed), callable_mp(this, &EditorMainScreenSplit::_update_header_height));
	_update_header_height();
}

void EditorMainScreenSplit::set_main_side(Control *p_main_side) {
	main_side = p_main_side;
}

void EditorMainScreenSplit::set_title_container(Control *p_container) {
	title_tabs = memnew(TabBar);
	title_tabs->set_focus_mode(FOCUS_NONE);
	title_tabs->set_deselect_enabled(true);
	title_tabs->set_clip_tabs(false);
	title_tabs->set_v_size_flags(SIZE_SHRINK_CENTER);
	title_tabs->hide();
	title_tabs->connect("tab_clicked", callable_mp(this, &EditorMainScreenSplit::_title_tab_clicked));
	p_container->add_child(title_tabs);
	_update_title_tabs_theme();
	_queue_update_title_tabs();
}

void EditorMainScreenSplit::set_maximized(Maximized p_maximized) {
	maximized = p_maximized;
	update_visibility();
}

void EditorMainScreenSplit::update_visibility() {
	// Unlike side docks, the split stays visible in distraction-free mode: it is part of the workspace.
	// Only maximizing one side of the workspace hides the other.
	const bool active = get_tab_count() > 0;
	if (!active && maximized == MAXIMIZED_SPLIT) {
		// The maximized panel was closed or emptied: leave the maximized mode with it.
		callable_mp(EditorNode::get_singleton(), &EditorNode::set_distraction_free_mode).call_deferred(false);
	}
	set_visible(active && maximized != MAXIMIZED_MAIN);
	if (main_side) {
		main_side->set_visible(!active || maximized != MAXIMIZED_SPLIT);
	}
	if (maximize_button) {
		maximize_button->set_pressed_no_signal(maximized == MAXIMIZED_SPLIT);
	}
	_queue_update_title_tabs();
}

EditorMainScreenSplit::EditorMainScreenSplit() :
		DockTabContainer(EditorDock::DOCK_SLOT_MAIN_SCREEN_SPLIT) {
	layout = EditorDock::DOCK_LAYOUT_MAIN_SCREEN;
	grid_rect = Rect2i(4, 0, 2, 4);

	set_custom_minimum_size(Size2(80, 80) * EDSCALE);
	set_h_size_flags(Control::SIZE_EXPAND_FILL);
	set_v_size_flags(Control::SIZE_EXPAND_FILL);

	maximize_button = memnew(Button);
	maximize_button->set_theme_type_variation("FlatMenuButton");
	maximize_button->set_toggle_mode(true);
	maximize_button->set_tooltip_text(TTRC("Maximize this panel: hide the docks and the main screen."));
	maximize_button->connect(SceneStringName(toggled), callable_mp(this, &EditorMainScreenSplit::_maximize_toggled));
	get_internal_container()->add_child(maximize_button);
}
