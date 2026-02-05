/**************************************************************************/
/*  dock_tab_container.cpp                                                */
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

#include "dock_tab_container.h"

#include "editor/docks/editor_dock.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/resources/style_box_flat.h"

bool EditorDockDragHint::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
	return can_drop_dock;
}

void EditorDockDragHint::drop_data(const Point2 &p_point, const Variant &p_data) {
	// Drop dock into last spot if not over tabbar.
	if (drop_tabbar->get_rect().has_point(p_point)) {
		drop_tabbar->_handle_drop_data("tab_container_tab", p_point, p_data, callable_mp(this, &EditorDockDragHint::_drag_move_tab), callable_mp(this, &EditorDockDragHint::_drag_move_tab_from));
	} else {
		EditorDockManager *dock_manager = EditorDockManager::get_singleton();
		dock_manager->_move_dock(dock_manager->_get_dock_tab_dragged(), dock_container, drop_tabbar->get_tab_count());
	}
}

void EditorDockDragHint::_drag_move_tab(int p_from_index, int p_to_index) {
	dock_container->get_dock(p_from_index)->set_tab_index(p_to_index, true);
}

void EditorDockDragHint::_drag_move_tab_from(TabBar *p_from_tabbar, int p_from_index, int p_to_index) {
	EditorDockManager *dock_manager = EditorDockManager::get_singleton();
	dock_manager->_move_dock(dock_manager->_get_dock_tab_dragged(), dock_container, p_to_index);
}

void EditorDockDragHint::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		Point2 pos = mm->get_position();

		// Redraw when inside the tabbar and just exited.
		if (mouse_inside_tabbar) {
			queue_redraw();
		}
		mouse_inside_tabbar = drop_tabbar->get_rect().has_point(pos);
	}
}

void EditorDockDragHint::set_slot(DockTabContainer *p_slot) {
	dock_container = p_slot;
	drop_tabbar = p_slot->get_tab_bar();
}

void EditorDockDragHint::_notification(int p_what) {
	switch (p_what) {
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (EditorSettings::get_singleton()->check_changed_settings_in_group("interface/theme")) {
				dock_drop_highlight->set_corner_radius_all(EDSCALE * EDITOR_GET("interface/theme/corner_radius").operator int());
				if (mouse_inside) {
					queue_redraw();
				}
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			valid_drop_color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
		} break;

		case NOTIFICATION_MOUSE_ENTER:
		case NOTIFICATION_MOUSE_EXIT: {
			mouse_inside = p_what == NOTIFICATION_MOUSE_ENTER;
			queue_redraw();
		} break;

		case NOTIFICATION_DRAG_BEGIN: {
			EditorDock *dragged_dock = EditorDockManager::get_singleton()->_get_dock_tab_dragged();
			if (!dragged_dock) {
				return;
			}

			can_drop_dock = dragged_dock->get_available_layouts() & dock_container->layout;

			dock_drop_highlight->set_border_color(valid_drop_color);
			dock_drop_highlight->set_bg_color(valid_drop_color * Color(1, 1, 1, 0.1));
		} break;
		case NOTIFICATION_DRAG_END: {
			EditorDockManager::get_singleton()->_dock_drag_stopped();
			can_drop_dock = false;
			mouse_inside = false;
			hide();
		} break;

		case NOTIFICATION_DRAW: {
			if (!mouse_inside || !can_drop_dock) {
				return;
			}

			// Draw highlights around docks that can be dropped.
			Rect2 dock_rect = Rect2(Point2(), get_size()).grow(2 * EDSCALE);
			draw_style_box(dock_drop_highlight, dock_rect);

			// Only display tabbar hint if the mouse is over the tabbar.
			if (drop_tabbar->get_global_rect().has_point(get_global_mouse_position())) {
				draw_set_transform(drop_tabbar->get_position()); // The TabBar isn't always on top.
				drop_tabbar->_draw_tab_drop(get_canvas_item());
			}
		} break;
	}
}

EditorDockDragHint::EditorDockDragHint() {
	set_as_top_level(true);
	hide();

	dock_drop_highlight.instantiate();
	dock_drop_highlight->set_corner_radius_all(EDSCALE * EDITOR_GET("interface/theme/corner_radius").operator int());
	dock_drop_highlight->set_border_width_all(Math::round(2 * EDSCALE));
}

void DockTabContainer::_pre_popup() {
	dock_context_popup->set_dock(get_dock(get_current_tab()));
}

void DockTabContainer::_tab_rmb_clicked(int p_tab_idx) {
	EditorDock *hovered_dock = get_dock(p_tab_idx);
	if (!hovered_dock) {
		return;
	}

	// Right click context menu.
	dock_context_popup->set_dock(hovered_dock);
	dock_context_popup->set_position(get_tab_bar()->get_screen_position() + get_tab_bar()->get_local_mouse_position());
	dock_context_popup->popup();
}

void DockTabContainer::_notification(int p_what) {
	if (p_what == NOTIFICATION_POSTINITIALIZE) {
		connect("pre_popup_pressed", callable_mp(this, &DockTabContainer::_pre_popup));
		connect("child_order_changed", callable_mp(this, &DockTabContainer::update_visibility));
	}
}

void DockTabContainer::update_visibility() {
	// Hide the dock container if there are no tabs.
	set_visible(EditorDockManager::get_singleton()->are_docks_visible() && get_tab_count() > 0);
}

DockTabContainer::TabStyle DockTabContainer::get_tab_style() const {
	return (TabStyle)EDITOR_GET("interface/editor/dock_tab_style").operator int();
}

bool DockTabContainer::can_switch_dock() const {
	return EditorDockManager::get_singleton()->are_docks_visible();
}

void DockTabContainer::save_docks_to_config(Ref<ConfigFile> p_layout, const String &p_section) {
	PackedStringArray names;
	names.reserve_exact(get_tab_count());
	for (int i = 0; i < get_tab_count(); i++) {
		const String name = get_dock(i)->get_effective_layout_key();
		names.append(name);
	}

	const String config_key = DockTabContainer::get_config_key(dock_slot);
	if (!names.is_empty()) {
		p_layout->set_value(p_section, config_key, String(",").join(names));
	} else if (p_layout->has_section_key(p_section, config_key)) {
		p_layout->erase_section_key(p_section, config_key);
	}

	const String tab_key = config_key + "_selected_tab_idx";
	int selected_tab_idx = get_current_tab();
	if (selected_tab_idx >= 0) {
		p_layout->set_value(p_section, tab_key, selected_tab_idx);
	} else if (p_layout->has_section_key(p_section, tab_key)) {
		p_layout->erase_section_key(p_section, tab_key);
	}
}

void DockTabContainer::load_selected_tab(int p_idx) {
	EditorDock *selected_dock = get_dock(p_idx);
	if (!selected_dock) {
		return;
	}
	set_block_signals(true);
	set_current_tab(p_idx);
	set_block_signals(false);
}

void DockTabContainer::set_dock_context_popup(DockContextPopup *p_popup) {
	dock_context_popup = p_popup;
	set_popup(dock_context_popup);
}

void DockTabContainer::move_dock_index(EditorDock *p_dock, int p_to_index, bool p_set_current) {
	set_block_signals(true);
	int target_index = CLAMP(p_to_index, 0, get_tab_count() - 1);
	move_child(p_dock, get_dock(target_index)->get_index(false));

	if (p_set_current) {
		set_current_tab(target_index);
	}
	set_block_signals(false);
}

EditorDock *DockTabContainer::get_dock(int p_idx) const {
	return Object::cast_to<EditorDock>(get_tab_control(p_idx));
}

void DockTabContainer::show_drag_hint() {
	if (!is_visible_in_tree()) {
		return;
	}
	drag_hint->set_rect(get_global_rect());
	drag_hint->show();
}

DockTabContainer::DockTabContainer(EditorDock::DockSlot p_slot) {
	ERR_FAIL_INDEX(p_slot, EditorDock::DOCK_SLOT_MAX);
	dock_slot = p_slot;

	set_drag_to_rearrange_enabled(true);
	set_tabs_rearrange_group(1);
	hide();

	drag_hint = memnew(EditorDockDragHint);
	drag_hint->set_slot(this);
	drag_hint->hide();
	EditorNode::get_singleton()->get_gui_base()->add_child(drag_hint);

	get_tab_bar()->set_switch_on_release(true);
	get_tab_bar()->connect("tab_rmb_clicked", callable_mp(this, &DockTabContainer::_tab_rmb_clicked));
}

SideDockTabContainer::SideDockTabContainer(EditorDock::DockSlot p_slot) :
		DockTabContainer(p_slot) {
	set_custom_minimum_size(Size2(170 * EDSCALE, 0));
	set_v_size_flags(Control::SIZE_EXPAND_FILL);
	set_use_hidden_tabs_for_min_size(true);
}
