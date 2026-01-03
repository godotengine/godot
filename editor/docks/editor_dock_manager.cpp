/**************************************************************************/
/*  editor_dock_manager.cpp                                               */
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

#include "editor_dock_manager.h"

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"
#include "scene/main/window.h"

#include "editor/docks/editor_dock.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_bottom_panel.h"
#include "editor/gui/window_wrapper.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/resources/style_box_flat.h"

enum class TabStyle {
	TEXT_ONLY,
	ICON_ONLY,
	TEXT_AND_ICON,
};

static inline Ref<Texture2D> _get_dock_icon(const EditorDock *p_dock, const Callable &p_icon_fetch) {
	Ref<Texture2D> icon = p_dock->get_dock_icon();
	if (icon.is_null() && !p_dock->get_icon_name().is_empty()) {
		icon = p_icon_fetch.call(p_dock->get_icon_name());
	}
	return icon;
}

bool EditorDockDragHint::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
	return can_drop_dock;
}

void EditorDockDragHint::drop_data(const Point2 &p_point, const Variant &p_data) {
	// Drop dock into last spot if not over tabbar.
	if (drop_tabbar->get_rect().has_point(p_point)) {
		drop_tabbar->_handle_drop_data("tab_container_tab", p_point, p_data, callable_mp(this, &EditorDockDragHint::_drag_move_tab), callable_mp(this, &EditorDockDragHint::_drag_move_tab_from));
	} else {
		dock_manager->_move_dock(dock_manager->_get_dock_tab_dragged(), dock_manager->dock_slots[occupied_slot].container, drop_tabbar->get_tab_count());
	}
}

void EditorDockDragHint::_drag_move_tab(int p_from_index, int p_to_index) {
	dock_manager->_move_dock_tab_index(dock_manager->_get_dock_tab_dragged(), p_to_index, true);
}

void EditorDockDragHint::_drag_move_tab_from(TabBar *p_from_tabbar, int p_from_index, int p_to_index) {
	dock_manager->_move_dock(dock_manager->_get_dock_tab_dragged(), dock_manager->dock_slots[occupied_slot].container, p_to_index);
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

void EditorDockDragHint::set_slot(DockConstants::DockSlot p_slot) {
	occupied_slot = p_slot;
	drop_tabbar = dock_manager->dock_slots[occupied_slot].container->get_tab_bar();
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
			EditorDock *dragged_dock = dock_manager->_get_dock_tab_dragged();
			if (!dragged_dock) {
				return;
			}

			can_drop_dock = dragged_dock->get_available_layouts() & (EditorDock::DockLayout)EditorDockManager::get_singleton()->dock_slots[occupied_slot].layout;

			dock_drop_highlight->set_border_color(valid_drop_color);
			dock_drop_highlight->set_bg_color(valid_drop_color * Color(1, 1, 1, 0.1));
		} break;
		case NOTIFICATION_DRAG_END: {
			dock_manager->_dock_drag_stopped();
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
	dock_manager = EditorDockManager::get_singleton();

	set_as_top_level(true);
	dock_drop_highlight.instantiate();
	dock_drop_highlight->set_corner_radius_all(EDSCALE * EDITOR_GET("interface/theme/corner_radius").operator int());
	dock_drop_highlight->set_border_width_all(Math::round(2 * EDSCALE));
}

////////////////////////////////////////////////
////////////////////////////////////////////////

void DockSplitContainer::_update_visibility() {
	if (is_updating) {
		return;
	}
	is_updating = true;
	bool any_visible = false;
	for (int i = 0; i < get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(get_child(i, false));
		if (!c || !c->is_visible() || c->is_set_as_top_level()) {
			continue;
		}
		any_visible = c;
		break;
	}
	set_visible(any_visible);
	is_updating = false;
}

void DockSplitContainer::add_child_notify(Node *p_child) {
	SplitContainer::add_child_notify(p_child);

	Control *child_control = nullptr;
	for (int i = 0; i < get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(get_child(i, false));
		if (!c || c->is_set_as_top_level()) {
			continue;
		}
		if (p_child == c) {
			child_control = c;
			break;
		}
	}
	if (!child_control) {
		return;
	}

	child_control->connect(SceneStringName(visibility_changed), callable_mp(this, &DockSplitContainer::_update_visibility));
	_update_visibility();
}

void DockSplitContainer::remove_child_notify(Node *p_child) {
	SplitContainer::remove_child_notify(p_child);

	Control *child_control = nullptr;
	for (int i = 0; i < get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(get_child(i, false));
		if (!c || c->is_set_as_top_level()) {
			continue;
		}
		if (p_child == c) {
			child_control = c;
			break;
		}
	}
	if (!child_control) {
		return;
	}

	child_control->disconnect(SceneStringName(visibility_changed), callable_mp(this, &DockSplitContainer::_update_visibility));
	_update_visibility();
}

DockSplitContainer::DockSplitContainer() {
	if (EDITOR_GET("interface/touchscreen/enable_touch_optimizations")) {
		callable_mp((SplitContainer *)this, &SplitContainer::set_touch_dragger_enabled).call_deferred(true);
	}
}

////////////////////////////////////////////////
////////////////////////////////////////////////

EditorDock *EditorDockManager::_get_dock_tab_dragged() {
	if (dock_tab_dragged) {
		return dock_tab_dragged;
	}

	Dictionary dock_drop_data = dock_slots[DockConstants::DOCK_SLOT_LEFT_BL].container->get_viewport()->gui_get_drag_data();

	// Check if we are dragging a dock.
	if (dock_drop_data.get("type", "").operator String() != "tab") {
		return nullptr;
	}

	const String tab_type = dock_drop_data.get("tab_type", "");
	if (tab_type == "tab_container_tab") {
		Node *source_tab_bar = EditorNode::get_singleton()->get_node(dock_drop_data["from_path"]);
		if (!source_tab_bar) {
			return nullptr;
		}

		TabContainer *source_tab_container = Object::cast_to<TabContainer>(source_tab_bar->get_parent());
		if (!source_tab_container) {
			return nullptr;
		}

		dock_tab_dragged = Object::cast_to<EditorDock>(source_tab_container->get_tab_control(dock_drop_data["tab_index"]));
		if (!dock_tab_dragged) {
			return nullptr;
		}

		for (int i = 0; i < DockConstants::DOCK_SLOT_MAX; i++) {
			if (dock_slots[i].container->is_visible_in_tree()) {
				dock_slots[i].drag_hint->set_rect(dock_slots[i].container->get_global_rect());
				dock_slots[i].drag_hint->show();
			}
		}

		return dock_tab_dragged;
	}
	return nullptr;
}

void EditorDockManager::_dock_drag_stopped() {
	dock_tab_dragged = nullptr;
}

void EditorDockManager::_dock_split_dragged(int p_offset) {
	EditorNode::get_singleton()->save_editor_layout_delayed();
}

void EditorDockManager::_dock_container_popup(int p_tab_idx, TabContainer *p_dock_container) {
	EditorDock *hovered_dock = Object::cast_to<EditorDock>(p_dock_container->get_tab_control(p_tab_idx));
	if (hovered_dock == nullptr) {
		return;
	}

	// Right click context menu.
	dock_context_popup->set_dock(hovered_dock);
	dock_context_popup->set_position(p_dock_container->get_tab_bar()->get_screen_position() + p_dock_container->get_tab_bar()->get_local_mouse_position());
	dock_context_popup->popup();
}

void EditorDockManager::_dock_container_update_visibility(TabContainer *p_dock_container) {
	if (!docks_visible) {
		return;
	}
	// Hide the dock container if there are no tabs.
	p_dock_container->set_visible(p_dock_container->get_tab_count() > 0);
}

void EditorDockManager::_update_layout() {
	if (!dock_context_popup->is_inside_tree() || EditorNode::get_singleton()->is_exiting()) {
		return;
	}
	dock_context_popup->docks_updated();
	update_docks_menu();
	EditorNode::get_singleton()->save_editor_layout_delayed();
}

void EditorDockManager::update_docks_menu() {
	docks_menu->clear();
	docks_menu->reset_size();

	const Ref<Texture2D> default_icon = get_editor_theme_icon(SNAME("Window"));
	const Color closed_icon_color_mod = Color(1, 1, 1, 0.5);

	bool global_menu = !bool(EDITOR_GET("interface/editor/use_embedded_menu")) && NativeMenu::get_singleton()->has_feature(NativeMenu::FEATURE_GLOBAL_MENU);
	bool dark_mode = DisplayServer::get_singleton()->is_dark_mode_supported() && DisplayServer::get_singleton()->is_dark_mode();
	int icon_max_width = get_theme_constant(SNAME("class_icon_size"), EditorStringName(Editor));

	// Add docks.
	docks_menu_docks.clear();
	int id = 0;
	const Callable icon_fetch = callable_mp(EditorNode::get_singleton(), &EditorNode::get_editor_theme_native_menu_icon).bind(global_menu, dark_mode);
	for (EditorDock *dock : all_docks) {
		if (!dock->enabled || !dock->global) {
			continue;
		}
		if (dock->shortcut.is_valid()) {
			docks_menu->add_shortcut(dock->shortcut, id);
			docks_menu->set_item_text(id, dock->get_display_title());
		} else {
			docks_menu->add_item(dock->get_display_title(), id);
		}
		docks_menu->set_item_icon_max_width(id, icon_max_width);

		const Ref<Texture2D> icon = _get_dock_icon(dock, icon_fetch);
		docks_menu->set_item_icon(id, icon.is_valid() ? icon : default_icon);
		if (!dock->is_open) {
			docks_menu->set_item_icon_modulate(id, closed_icon_color_mod);
			docks_menu->set_item_tooltip(id, vformat(TTR("Open the %s dock."), dock->get_display_title()));
		} else {
			docks_menu->set_item_tooltip(id, vformat(TTR("Focus on the %s dock."), dock->get_display_title()));
		}
		docks_menu_docks.push_back(dock);
		id++;
	}
}

void EditorDockManager::_docks_menu_option(int p_id) {
	EditorDock *dock = docks_menu_docks[p_id];
	ERR_FAIL_NULL(dock);
	ERR_FAIL_COND_MSG(!all_docks.has(dock), vformat("Menu option for unknown dock '%s'.", dock->get_display_title()));
	if (dock->enabled && dock->is_open) {
		PopupMenu *parent_menu = Object::cast_to<PopupMenu>(docks_menu->get_parent());
		ERR_FAIL_NULL(parent_menu);
		parent_menu->hide();
	}
	focus_dock(dock);
}

void EditorDockManager::_window_close_request(WindowWrapper *p_wrapper) {
	// Give the dock back to the original owner.
	EditorDock *dock = _close_window(p_wrapper);
	ERR_FAIL_COND(!all_docks.has(dock));

	if (dock->dock_slot_index != DockConstants::DOCK_SLOT_NONE) {
		dock->is_open = false;
		open_dock(dock);
		focus_dock(dock);
	} else {
		close_dock(dock);
	}
}

EditorDock *EditorDockManager::_close_window(WindowWrapper *p_wrapper) {
	p_wrapper->set_block_signals(true);
	EditorDock *dock = Object::cast_to<EditorDock>(p_wrapper->release_wrapped_control());
	p_wrapper->set_block_signals(false);
	ERR_FAIL_COND_V(!all_docks.has(dock), nullptr);

	dock->dock_window = nullptr;
	dock_windows.erase(p_wrapper);
	p_wrapper->queue_free();
	return dock;
}

void EditorDockManager::_open_dock_in_window(EditorDock *p_dock, bool p_show_window, bool p_reset_size) {
	ERR_FAIL_NULL(p_dock);

	Size2 borders = Size2(4, 4) * EDSCALE;
	// Remember size and position before removing it from the main window.
	Size2 dock_size = p_dock->get_size() + borders * 2;
	Point2 dock_screen_pos = p_dock->get_screen_position();

	WindowWrapper *wrapper = memnew(WindowWrapper);
	wrapper->set_window_title(vformat(TTR("%s - Godot Engine"), TTR(p_dock->get_display_title())));
	wrapper->set_margins_enabled(true);

	add_child(wrapper);

	_move_dock(p_dock, nullptr);
	p_dock->update_layout(EditorDock::DOCK_LAYOUT_FLOATING);
	wrapper->set_wrapped_control(p_dock);

	p_dock->dock_window = wrapper;
	p_dock->is_open = true;
	p_dock->show();

	wrapper->connect("window_close_requested", callable_mp(this, &EditorDockManager::_window_close_request).bind(wrapper));
	dock_windows.push_back(wrapper);

	if (p_show_window) {
		wrapper->restore_window(Rect2i(dock_screen_pos, dock_size), get_window()->get_current_screen());
		_update_layout();
		if (p_reset_size) {
			// Use a default size of one third the current window size.
			Size2i popup_size = get_window()->get_size() / 3.0;
			p_dock->get_window()->set_size(popup_size);
			p_dock->get_window()->move_to_center();
		}
		p_dock->get_window()->grab_focus();
	}
}

void EditorDockManager::_restore_dock_to_saved_window(EditorDock *p_dock, const Dictionary &p_window_dump) {
	if (!p_dock->dock_window) {
		_open_dock_in_window(p_dock, false);
	}

	p_dock->dock_window->restore_window_from_saved_position(
			p_window_dump.get("window_rect", Rect2i()),
			p_window_dump.get("window_screen", -1),
			p_window_dump.get("window_screen_rect", Rect2i()));
}

void EditorDockManager::_move_dock_tab_index(EditorDock *p_dock, int p_tab_index, bool p_set_current) {
	TabContainer *dock_tab_container = Object::cast_to<TabContainer>(p_dock->get_parent());
	if (!dock_tab_container) {
		return;
	}

	dock_tab_container->set_block_signals(true);
	int target_index = CLAMP(p_tab_index, 0, dock_tab_container->get_tab_count() - 1);
	dock_tab_container->move_child(p_dock, dock_tab_container->get_tab_control(target_index)->get_index(false));
	p_dock->previous_tab_index = target_index;

	if (p_set_current) {
		dock_tab_container->set_current_tab(target_index);
	}
	dock_tab_container->set_block_signals(false);
}

void EditorDockManager::_move_dock(EditorDock *p_dock, Control *p_target, int p_tab_index, bool p_set_current) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(!all_docks.has(p_dock), vformat("Cannot move unknown dock '%s'.", p_dock->get_display_title()));

	Node *parent = p_dock->get_parent();
	if (parent == p_target) {
		if (parent && p_tab_index >= 0) {
			// Only change the tab index.
			_move_dock_tab_index(p_dock, p_tab_index, p_set_current);
		}
		return;
	}

	// Remove dock from its existing parent.
	if (parent) {
		if (p_dock->dock_window) {
			_close_window(p_dock->dock_window);
		} else {
			TabContainer *parent_tabs = Object::cast_to<TabContainer>(parent);
			if (parent_tabs) {
				p_dock->previous_tab_index = parent_tabs->get_tab_idx_from_control(p_dock);
			}
			parent->set_block_signals(true);
			parent->remove_child(p_dock);
			parent->set_block_signals(false);
			if (parent_tabs) {
				_dock_container_update_visibility(parent_tabs);
			}
		}
	}

	if (!p_target) {
		return;
	}

	if (p_target != closed_dock_parent) {
		p_dock->update_layout(p_target->get_meta("dock_layout"));
		p_dock->dock_slot_index = p_target->get_meta("dock_slot");
	}

	// Add dock to its new parent, at the given tab index.
	p_target->set_block_signals(true);
	p_target->add_child(p_dock);
	p_target->set_block_signals(false);

	TabContainer *dock_tab_container = Object::cast_to<TabContainer>(p_target);
	if (dock_tab_container) {
		if (dock_tab_container->is_inside_tree()) {
			_update_tab_style(p_dock);
		}
		if (p_tab_index >= 0) {
			_move_dock_tab_index(p_dock, p_tab_index, p_set_current);
		}
		_dock_container_update_visibility(dock_tab_container);
	}
}

void EditorDockManager::_queue_update_tab_style(EditorDock *p_dock) {
	if (dirty_docks.is_empty()) {
		callable_mp(this, &EditorDockManager::_update_dirty_dock_tabs).call_deferred();
	}
	dirty_docks.insert(p_dock);
}

void EditorDockManager::_update_dirty_dock_tabs() {
	bool update_menu = false;
	for (EditorDock *dock : dirty_docks) {
		update_menu = update_menu || dock->global;
		_update_tab_style(dock);
	}
	dirty_docks.clear();

	if (update_menu) {
		update_docks_menu();
	}
}

void EditorDockManager::_update_tab_style(EditorDock *p_dock) {
	if (!p_dock->enabled || !p_dock->is_open) {
		return; // Disabled by feature profile or manually closed by user.
	}
	if (p_dock->dock_window) {
		return; // Floating.
	}

	TabContainer *tab_container = get_dock_tab_container(p_dock);
	ERR_FAIL_NULL(tab_container);

	int index = tab_container->get_tab_idx_from_control(p_dock);
	ERR_FAIL_COND(index == -1);

	tab_container->get_tab_bar()->set_font_color_override_all(index, p_dock->title_color);

	const TabStyle style = (tab_container == EditorNode::get_bottom_panel())
			? (TabStyle)EDITOR_GET("interface/editor/bottom_dock_tab_style").operator int()
			: (TabStyle)EDITOR_GET("interface/editor/dock_tab_style").operator int();

	const Ref<Texture2D> icon = _get_dock_icon(p_dock, callable_mp((Control *)this, &Control::get_editor_theme_icon));
	bool assign_icon = p_dock->force_show_icon;
	String tooltip;
	switch (style) {
		case TabStyle::TEXT_ONLY: {
			tab_container->set_tab_title(index, p_dock->get_display_title());
		} break;
		case TabStyle::ICON_ONLY: {
			tab_container->set_tab_title(index, icon.is_valid() ? String() : p_dock->get_display_title());
			tooltip = TTR(p_dock->get_display_title());
			assign_icon = true;
		} break;
		case TabStyle::TEXT_AND_ICON: {
			tab_container->set_tab_title(index, p_dock->get_display_title());
			assign_icon = true;
		} break;
	}

	if (p_dock->shortcut.is_valid() && p_dock->shortcut->has_valid_event()) {
		tooltip += (tooltip.is_empty() ? "" : "\n") + TTR(p_dock->shortcut->get_name()) + " (" + p_dock->shortcut->get_as_text() + ")";
	}
	tab_container->set_tab_tooltip(index, tooltip);

	if (assign_icon) {
		tab_container->set_tab_icon(index, icon);
	} else {
		tab_container->set_tab_icon(index, Ref<Texture2D>());
	}
}

void EditorDockManager::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			update_docks_menu();
			update_tab_styles();
		} break;
	}
}

void EditorDockManager::shortcut_input(const Ref<InputEvent> &p_event) {
	if (p_event.is_null() || !p_event->is_pressed() || p_event->is_echo()) {
		return;
	}

	for (EditorDock *dock : all_docks) {
		const Ref<Shortcut> &dock_shortcut = dock->get_dock_shortcut();
		if (dock_shortcut.is_valid() && dock_shortcut->matches_event(p_event)) {
			if (dock->is_visible() && dock->get_parent() == EditorNode::get_bottom_panel()) {
				EditorNode::get_bottom_panel()->hide_bottom_panel();
			} else if (!dock->transient || dock->is_open) {
				focus_dock(dock);
			}
			accept_event();
			break;
		}
	}
}

void EditorDockManager::save_docks_to_config(Ref<ConfigFile> p_layout, const String &p_section) const {
	// Save docks by dock slot.
	for (int i = 0; i < DockConstants::DOCK_SLOT_MAX; i++) {
		const DockSlot &dock_slot = dock_slots[i];

		PackedStringArray names;
		names.reserve_exact(dock_slot.container->get_tab_count());
		for (int j = 0; j < dock_slot.container->get_tab_count(); j++) {
			const String name = Object::cast_to<EditorDock>(dock_slot.container->get_tab_control(j))->get_effective_layout_key();
			names.append(name);
		}

		const String config_key = "dock_" + itos(i + 1);
		if (p_layout->has_section_key(p_section, config_key)) {
			p_layout->erase_section_key(p_section, config_key);
		}

		if (!names.is_empty()) {
			p_layout->set_value(p_section, config_key, String(",").join(names));
		}

		const String tab_key = config_key + "_selected_tab_idx";

		int selected_tab_idx = dock_slots[i].container->get_current_tab();
		if (selected_tab_idx >= 0) {
			p_layout->set_value(p_section, tab_key, selected_tab_idx);
		} else if (p_layout->has_section_key(p_section, tab_key)) {
			p_layout->erase_section_key(p_section, tab_key);
		}
	}

	// Clear the special dock slot for docks without default slots (index -1 = dock_0).
	// This prevents closed docks from being infinitely appended to the config on each save.
	const String no_slot_config_key = "dock_0";
	if (p_layout->has_section_key(p_section, no_slot_config_key)) {
		p_layout->erase_section_key(p_section, no_slot_config_key);
	}

	// Save docks in windows.
	Dictionary floating_docks_dump;
	for (WindowWrapper *wrapper : dock_windows) {
		EditorDock *dock = Object::cast_to<EditorDock>(wrapper->get_wrapped_control());

		Dictionary window_dump;
		window_dump["window_rect"] = wrapper->get_window_rect();

		int screen = wrapper->get_window_screen();
		window_dump["window_screen"] = wrapper->get_window_screen();
		window_dump["window_screen_rect"] = DisplayServer::get_singleton()->screen_get_usable_rect(screen);

		String name = dock->get_effective_layout_key();
		if (!dock->transient) {
			floating_docks_dump[name] = window_dump;
		}

		// Append to regular dock section so we know where to restore it to.
		int dock_slot_id = dock->dock_slot_index;
		String config_key = "dock_" + itos(dock_slot_id + 1);

		String names = p_layout->get_value(p_section, config_key, "");
		if (names.is_empty()) {
			names = name;
		} else {
			names += "," + name;
		}
		p_layout->set_value(p_section, config_key, names);
	}
	p_layout->set_value(p_section, "dock_floating", floating_docks_dump);

	// Save closed docks.
	Array closed_docks_dump;
	for (const EditorDock *dock : all_docks) {
		const String section_name = p_section + "/" + dock->get_effective_layout_key();
		dock->save_layout_to_config(p_layout, section_name);

		if (dock->is_open) {
			continue;
		}

		const String name = dock->get_effective_layout_key();
		if (!dock->is_open && !dock->transient) {
			closed_docks_dump.push_back(name);
		}

		int dock_slot_id = dock->dock_slot_index;
		String config_key = "dock_" + itos(dock_slot_id + 1);

		String names = p_layout->get_value(p_section, config_key, "");
		if (names.is_empty()) {
			names = name;
		} else {
			names += "," + name;
		}
		p_layout->set_value(p_section, config_key, names);
	}
	p_layout->set_value(p_section, "dock_closed", closed_docks_dump);

	// Save SplitContainer offsets.
	for (int i = 0; i < vsplits.size(); i++) {
		if (vsplits[i]->is_visible_in_tree()) {
			p_layout->set_value(p_section, "dock_split_" + itos(i + 1), vsplits[i]->get_split_offset());
		}
	}

	PackedInt32Array split_offsets = main_hsplit->get_split_offsets();
	int index = 0;
	for (int i = 0; i < vsplits.size(); i++) {
		int value = 0;
		if (vsplits[i]->is_visible() && index < split_offsets.size()) {
			value = split_offsets[index] / EDSCALE;
			index++;
		}
		p_layout->set_value(p_section, "dock_hsplit_" + itos(i + 1), value);
	}
}

void EditorDockManager::load_docks_from_config(Ref<ConfigFile> p_layout, const String &p_section, bool p_first_load) {
	Dictionary floating_docks_dump = p_layout->get_value(p_section, "dock_floating", Dictionary());
	Array closed_docks = p_layout->get_value(p_section, "dock_closed", Array());

	bool allow_floating_docks = EditorNode::get_singleton()->is_multi_window_enabled() && (!p_first_load || EDITOR_GET("interface/multi_window/restore_windows_on_load"));

	// Store the docks by name for easy lookup.
	HashMap<String, EditorDock *> dock_map;
	for (EditorDock *dock : all_docks) {
		dock_map[dock->get_effective_layout_key()] = dock;
	}

	// Load docks by slot. Index -1 is for docks that have no slot.
	for (int i = -1; i < DockConstants::DOCK_SLOT_MAX; i++) {
		if (!p_layout->has_section_key(p_section, "dock_" + itos(i + 1))) {
			continue;
		}

		Vector<String> names = String(p_layout->get_value(p_section, "dock_" + itos(i + 1))).split(",");

		for (int j = names.size() - 1; j >= 0; j--) {
			const String &name = names[j];
			const String section_name = p_section + "/" + name;

			if (!dock_map.has(name)) {
				continue;
			}
			EditorDock *dock = dock_map[name];

			if (!dock->enabled) {
				// Don't is_open disabled docks.
				dock->load_layout_from_config(p_layout, section_name);
				continue;
			}

			if (allow_floating_docks && floating_docks_dump.has(name)) {
				_restore_dock_to_saved_window(dock, floating_docks_dump[name]);
			} else if (i >= 0 && !(dock->transient && !dock->is_open)) {
				// Safe to include transient open docks here because they won't be in the closed dock dump.
				if (closed_docks.has(name)) {
					dock->is_open = false;
					dock->hide();
					_move_dock(dock, closed_dock_parent);
				} else {
					dock->is_open = true;
					_move_dock(dock, dock_slots[i].container, 0);
				}
			}
			dock->load_layout_from_config(p_layout, section_name);

			dock->dock_slot_index = i;
			dock->previous_tab_index = i >= 0 ? j : 0;
		}
	}

	// Set the selected tabs.
	for (int i = 0; i < DockConstants::DOCK_SLOT_MAX; i++) {
		const DockSlot &dock_slot = dock_slots[i];

		int selected_tab_idx = p_layout->get_value(p_section, "dock_" + itos(i + 1) + "_selected_tab_idx", -1);
		if (selected_tab_idx <= 0 || selected_tab_idx >= dock_slot.container->get_tab_count()) {
			if (i == DockConstants::DOCK_SLOT_BOTTOM) {
				dock_slot.container->set_current_tab(-1);
			}
			continue;
		}

		EditorDock *selected_dock = Object::cast_to<EditorDock>(dock_slot.container->get_tab_control(selected_tab_idx));
		if (!selected_dock) {
			continue;
		}
		dock_slot.container->set_block_signals(true);
		dock_slot.container->set_current_tab(selected_tab_idx);
		dock_slot.container->set_block_signals(false);
	}

	// Load SplitContainer offsets.
	PackedInt32Array offsets;
	for (int i = 0; i < vsplits.size(); i++) {
		if (!p_layout->has_section_key(p_section, "dock_split_" + itos(i + 1))) {
			continue;
		}
		int ofs = p_layout->get_value(p_section, "dock_split_" + itos(i + 1));
		vsplits[i]->set_split_offset(ofs);

		// Only visible ones need a split offset for the main hsplit, even though they all have a value saved.
		if (vsplits[i]->is_visible() && p_layout->has_section_key(p_section, "dock_hsplit_" + itos(i + 1))) {
			int offset = p_layout->get_value(p_section, "dock_hsplit_" + itos(i + 1));
			offsets.push_back(offset * EDSCALE);
		}
	}
	main_hsplit->set_split_offsets(offsets);

	update_docks_menu();
}

void EditorDockManager::set_dock_enabled(EditorDock *p_dock, bool p_enabled) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(!all_docks.has(p_dock), vformat("Cannot set enabled unknown dock '%s'.", p_dock->get_display_title()));

	if (p_dock->enabled == p_enabled) {
		return;
	}

	p_dock->enabled = p_enabled;
	if (p_enabled) {
		open_dock(p_dock, false);
	} else {
		close_dock(p_dock);
	}
}

void EditorDockManager::close_dock(EditorDock *p_dock) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(!all_docks.has(p_dock), vformat("Cannot close unknown dock '%s'.", p_dock->get_display_title()));

	if (!p_dock->is_open) {
		return;
	}

	p_dock->is_open = false;

	EditorBottomPanel *bottom_panel = EditorNode::get_bottom_panel();
	if (get_dock_tab_container(p_dock) == bottom_panel && bottom_panel->get_current_tab_control() == p_dock) {
		bottom_panel->hide_bottom_panel();
	}
	// Hide before moving to remove inconsistent signals.
	p_dock->hide();
	_move_dock(p_dock, closed_dock_parent);

	_update_layout();
}

void EditorDockManager::open_dock(EditorDock *p_dock, bool p_set_current) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(!all_docks.has(p_dock), vformat("Cannot open unknown dock '%s'.", p_dock->get_display_title()));

	if (p_dock->is_open) {
		// Show the dock if it is already open.
		if (p_set_current) {
			if (p_dock->dock_window) {
				p_dock->get_window()->grab_focus();
				return;
			}

			TabContainer *dock_tab_container = get_dock_tab_container(p_dock);
			if (dock_tab_container) {
				int tab_index = dock_tab_container->get_tab_idx_from_control(p_dock);
				dock_tab_container->set_current_tab(tab_index);
			}
		}
		return;
	}

	p_dock->is_open = true;

	// Open dock to its previous location.
	if (p_dock->dock_slot_index != DockConstants::DOCK_SLOT_NONE) {
		TabContainer *slot = dock_slots[p_dock->dock_slot_index].container;
		int tab_index = p_dock->previous_tab_index;
		if (tab_index < 0) {
			tab_index = slot->get_tab_count();
		}
		_move_dock(p_dock, slot, tab_index, p_set_current);
	} else {
		_open_dock_in_window(p_dock, true, true);
		return;
	}

	_update_layout();
}

void EditorDockManager::make_dock_floating(EditorDock *p_dock) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(!all_docks.has(p_dock), vformat("Cannot make unknown dock '%s' floating.", p_dock->get_display_title()));

	if (!p_dock->dock_window) {
		_open_dock_in_window(p_dock);
	}
}

TabContainer *EditorDockManager::get_dock_tab_container(Control *p_dock) const {
	return Object::cast_to<TabContainer>(p_dock->get_parent());
}

void EditorDockManager::focus_dock(EditorDock *p_dock) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(!all_docks.has(p_dock), vformat("Cannot focus unknown dock '%s'.", p_dock->get_display_title()));

	if (!p_dock->enabled) {
		return;
	}

	if (!p_dock->is_open) {
		open_dock(p_dock, false);
	}

	if (p_dock->dock_window) {
		p_dock->get_window()->grab_focus();
		return;
	}

	if (p_dock->get_parent() == EditorNode::get_bottom_panel()) {
		if (EditorNode::get_bottom_panel()->is_locked()) {
			return;
		}
	} else if (!docks_visible) {
		return;
	}

	TabContainer *tab_container = get_dock_tab_container(p_dock);
	if (!tab_container) {
		return;
	}
	int tab_index = tab_container->get_tab_idx_from_control(p_dock);
	tab_container->get_tab_bar()->grab_focus();
	tab_container->set_current_tab(tab_index);
}

void EditorDockManager::add_dock(EditorDock *p_dock) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(all_docks.has(p_dock), vformat("Cannot add dock '%s', already added.", p_dock->get_display_title()));

	p_dock->dock_slot_index = p_dock->default_slot;
	all_docks.push_back(p_dock);
	p_dock->connect("_tab_style_changed", callable_mp(this, &EditorDockManager::_queue_update_tab_style).bind(p_dock));
	p_dock->connect("renamed", callable_mp(this, &EditorDockManager::_queue_update_tab_style).bind(p_dock));

	if (p_dock->default_slot != DockConstants::DOCK_SLOT_NONE) {
		open_dock(p_dock, false);
	} else {
		closed_dock_parent->add_child(p_dock);
		p_dock->hide();
		_update_layout();
	}
}

void EditorDockManager::remove_dock(EditorDock *p_dock) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(!all_docks.has(p_dock), vformat("Cannot remove unknown dock '%s'.", p_dock->get_display_title()));

	_move_dock(p_dock, nullptr);

	all_docks.erase(p_dock);
	p_dock->disconnect("_tab_style_changed", callable_mp(this, &EditorDockManager::_queue_update_tab_style));
	p_dock->disconnect("renamed", callable_mp(this, &EditorDockManager::_queue_update_tab_style));
	_update_layout();
}

void EditorDockManager::set_docks_visible(bool p_show) {
	if (docks_visible == p_show) {
		return;
	}
	docks_visible = p_show;
	for (int i = 0; i < DockConstants::DOCK_SLOT_BOTTOM; i++) {
		// Show and hide in reverse order due to the SplitContainer prioritizing the last split offset.
		TabContainer *container = dock_slots[docks_visible ? i : DockConstants::DOCK_SLOT_BOTTOM - i - 1].container;
		container->set_visible(docks_visible && container->get_tab_count() > 0);
	}
	_update_layout();
}

bool EditorDockManager::are_docks_visible() const {
	return docks_visible;
}

void EditorDockManager::update_tab_styles() {
	for (EditorDock *dock : all_docks) {
		_update_tab_style(dock);
	}
}

void EditorDockManager::set_tab_icon_max_width(int p_max_width) {
	for (int i = 0; i < DockConstants::DOCK_SLOT_MAX; i++) {
		TabContainer *tab_container = dock_slots[i].container;
		tab_container->add_theme_constant_override(SNAME("icon_max_width"), p_max_width);
	}
}

void EditorDockManager::add_vsplit(DockSplitContainer *p_split) {
	vsplits.push_back(p_split);
	p_split->connect("dragged", callable_mp(this, &EditorDockManager::_dock_split_dragged));
}

void EditorDockManager::set_hsplit(DockSplitContainer *p_split) {
	main_hsplit = p_split;
	p_split->connect("dragged", callable_mp(this, &EditorDockManager::_dock_split_dragged));
}

void EditorDockManager::register_dock_slot(DockConstants::DockSlot p_dock_slot, TabContainer *p_tab_container, DockConstants::DockLayout p_layout) {
	ERR_FAIL_NULL(p_tab_container);
	ERR_FAIL_INDEX(p_dock_slot, DockConstants::DOCK_SLOT_MAX);

	DockSlot slot;
	slot.layout = p_layout;

	slot.container = p_tab_container;
	p_tab_container->set_popup(dock_context_popup);
	p_tab_container->connect("pre_popup_pressed", callable_mp(dock_context_popup, &DockContextPopup::select_current_dock_in_dock_slot).bind(p_dock_slot));
	p_tab_container->get_tab_bar()->set_switch_on_release(true);
	p_tab_container->get_tab_bar()->connect("tab_rmb_clicked", callable_mp(this, &EditorDockManager::_dock_container_popup).bind(p_tab_container));
	p_tab_container->set_drag_to_rearrange_enabled(true);
	p_tab_container->set_tabs_rearrange_group(1);
	p_tab_container->connect("tab_changed", callable_mp(this, &EditorDockManager::_update_layout).unbind(1));
	p_tab_container->connect("active_tab_rearranged", callable_mp(this, &EditorDockManager::_update_layout).unbind(1));
	p_tab_container->connect("child_order_changed", callable_mp(this, &EditorDockManager::_dock_container_update_visibility).bind(p_tab_container));
	p_tab_container->hide();
	p_tab_container->set_meta("dock_slot", p_dock_slot);
	p_tab_container->set_meta("dock_layout", p_layout);

	if (p_layout == DockConstants::DOCK_LAYOUT_VERTICAL) {
		p_tab_container->set_custom_minimum_size(Size2(170, 0) * EDSCALE);
		p_tab_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		p_tab_container->set_use_hidden_tabs_for_min_size(true);
	}

	// Create dock dragging hint.
	slot.drag_hint = memnew(EditorDockDragHint);
	slot.drag_hint->hide();
	EditorNode::get_singleton()->get_gui_base()->add_child(slot.drag_hint);

	dock_slots[p_dock_slot] = slot;
	slot.drag_hint->set_slot(p_dock_slot);
}

int EditorDockManager::get_vsplit_count() const {
	return vsplits.size();
}

PopupMenu *EditorDockManager::get_docks_menu() {
	return docks_menu;
}

EditorDockManager::EditorDockManager() {
	singleton = this;
	closed_dock_parent = this;
	hide();
	set_process_shortcut_input(true);

	dock_context_popup = memnew(DockContextPopup);
	add_child(dock_context_popup);

	docks_menu = memnew(PopupMenu);
	docks_menu->set_hide_on_item_selection(false);
	docks_menu->connect(SceneStringName(id_pressed), callable_mp(this, &EditorDockManager::_docks_menu_option));
}

////////////////////////////////////////////////
////////////////////////////////////////////////

void DockContextPopup::_notification(int p_what) {
	switch (p_what) {
		case Control::NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			if (make_float_button) {
				make_float_button->set_button_icon(get_editor_theme_icon(SNAME("MakeFloating")));
			}
			if (is_layout_rtl()) {
				tab_move_left_button->set_button_icon(get_editor_theme_icon(SNAME("Forward")));
				tab_move_right_button->set_button_icon(get_editor_theme_icon(SNAME("Back")));
				tab_move_left_button->set_tooltip_text(TTR("Move this dock right one tab."));
				tab_move_right_button->set_tooltip_text(TTR("Move this dock left one tab."));
			} else {
				tab_move_left_button->set_button_icon(get_editor_theme_icon(SNAME("Back")));
				tab_move_right_button->set_button_icon(get_editor_theme_icon(SNAME("Forward")));
				tab_move_left_button->set_tooltip_text(TTR("Move this dock left one tab."));
				tab_move_right_button->set_tooltip_text(TTR("Move this dock right one tab."));
			}
			close_button->set_button_icon(get_editor_theme_icon(SNAME("Close")));
		} break;
	}
}

void DockContextPopup::_tab_move_left() {
	TabContainer *tab_container = dock_manager->get_dock_tab_container(context_dock);
	if (!tab_container) {
		return;
	}
	int new_index = tab_container->get_tab_idx_from_control(context_dock) - 1;
	dock_manager->_move_dock(context_dock, tab_container, new_index);
	dock_manager->_update_layout();
	dock_select->queue_redraw();
}

void DockContextPopup::_tab_move_right() {
	TabContainer *tab_container = dock_manager->get_dock_tab_container(context_dock);
	if (!tab_container) {
		return;
	}
	int new_index = tab_container->get_tab_idx_from_control(context_dock) + 1;
	dock_manager->_move_dock(context_dock, tab_container, new_index);
	dock_manager->_update_layout();
	dock_select->queue_redraw();
}

void DockContextPopup::_close_dock() {
	hide();
	context_dock->emit_signal("closed");
	dock_manager->close_dock(context_dock);
}

void DockContextPopup::_float_dock() {
	hide();
	dock_manager->_open_dock_in_window(context_dock);
}

bool DockContextPopup::_is_slot_available(int p_slot) const {
	return context_dock->available_layouts & (EditorDock::DockLayout)EditorDockManager::get_singleton()->dock_slots[p_slot].layout;
}

void DockContextPopup::_dock_select_input(const Ref<InputEvent> &p_input) {
	Ref<InputEventMouse> me = p_input;

	if (me.is_valid()) {
		Vector2 point = me->get_position();

		int over_dock_slot = -1;
		for (int i = 0; i < DockConstants::DOCK_SLOT_MAX; i++) {
			if (dock_select_rects[i].has_point(point)) {
				over_dock_slot = i;
				break;
			}
		}

		if (over_dock_slot != dock_select_rect_over_idx) {
			dock_select->queue_redraw();
			dock_select_rect_over_idx = over_dock_slot;
		}

		if (over_dock_slot == -1) {
			return;
		}

		Ref<InputEventMouseButton> mb = me;
		TabContainer *target_tab_container = dock_manager->dock_slots[over_dock_slot].container;

		if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && mb->is_pressed()) {
			if (dock_manager->get_dock_tab_container(context_dock) != target_tab_container && _is_slot_available(over_dock_slot)) {
				dock_manager->_move_dock(context_dock, target_tab_container, target_tab_container->get_tab_count());
				dock_manager->_update_layout();
				hide();
			}
		}
	}
}

void DockContextPopup::_dock_select_mouse_exited() {
	dock_select_rect_over_idx = -1;
	dock_select->queue_redraw();
}

void DockContextPopup::_dock_select_draw() {
	Color used_dock_color = Color(0.6, 0.6, 0.6, 0.8);
	Color hovered_dock_color = Color(0.8, 0.8, 0.8, 0.8);
	Color tab_selected_color = dock_select->get_theme_color(SNAME("mono_color"), EditorStringName(Editor));
	Color tab_unselected_color = used_dock_color;
	Color unused_dock_color = used_dock_color;
	unused_dock_color.a = 0.4;
	Color unusable_dock_color = unused_dock_color;
	unusable_dock_color.a = 0.1;

	// Update sizes.
	Size2 dock_size = dock_select->get_size();
	dock_size.x /= 6.0;
	dock_size.y /= 2.0;

	real_t center_panel_width = dock_size.x * 2.0;
	Rect2 center_panel_rect(center_panel_width, 0, center_panel_width, dock_size.y);

	if (dock_select->is_layout_rtl()) {
		dock_select_rects[DockConstants::DOCK_SLOT_RIGHT_UR] = Rect2(Point2(), dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_RIGHT_BR] = Rect2(Point2(0, dock_size.y), dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_RIGHT_UL] = Rect2(Point2(dock_size.x, 0), dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_RIGHT_BL] = Rect2(dock_size, dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_LEFT_UR] = Rect2(Point2(dock_size.x * 4, 0), dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_LEFT_BR] = Rect2(Point2(dock_size.x * 4, dock_size.y), dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_LEFT_UL] = Rect2(Point2(dock_size.x * 5, 0), dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_LEFT_BL] = Rect2(Point2(dock_size.x * 5, dock_size.y), dock_size);
	} else {
		dock_select_rects[DockConstants::DOCK_SLOT_LEFT_UL] = Rect2(Point2(), dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_LEFT_BL] = Rect2(Point2(0, dock_size.y), dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_LEFT_UR] = Rect2(Point2(dock_size.x, 0), dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_LEFT_BR] = Rect2(dock_size, dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_RIGHT_UL] = Rect2(Point2(dock_size.x * 4, 0), dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_RIGHT_BL] = Rect2(Point2(dock_size.x * 4, dock_size.y), dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_RIGHT_UR] = Rect2(Point2(dock_size.x * 5, 0), dock_size);
		dock_select_rects[DockConstants::DOCK_SLOT_RIGHT_BR] = Rect2(Point2(dock_size.x * 5, dock_size.y), dock_size);
	}
	dock_select_rects[DockConstants::DOCK_SLOT_BOTTOM] = Rect2(center_panel_width, dock_size.y, center_panel_width, dock_size.y);

	int rtl_dir = dock_select->is_layout_rtl() ? -1 : 1;
	real_t tab_height = 3.0 * EDSCALE;
	real_t tab_spacing = 1.0 * EDSCALE;
	real_t dock_spacing = 2.0 * EDSCALE;
	real_t dock_top_spacing = tab_height + dock_spacing;

	TabContainer *context_tab_container = dock_manager->get_dock_tab_container(context_dock);
	int context_tab_index = -1;
	if (context_tab_container && context_tab_container->get_tab_count() > 0) {
		context_tab_index = context_tab_container->get_tab_idx_from_control(context_dock);
	}

	// Draw center panel.
	Rect2 center_panel_draw_rect = center_panel_rect.grow_individual(-dock_spacing, -dock_top_spacing, -dock_spacing, -dock_spacing);
	dock_select->draw_rect(center_panel_draw_rect, unusable_dock_color);

	// Draw all dock slots.
	for (int i = 0; i < DockConstants::DOCK_SLOT_MAX; i++) {
		int max_tabs = (i == DockConstants::DOCK_SLOT_BOTTOM) ? 6 : 3;
		const EditorDockManager::DockSlot &dock_slot = dock_manager->dock_slots[i];

		Rect2 dock_slot_draw_rect = dock_select_rects[i].grow_individual(-dock_spacing, -dock_top_spacing, -dock_spacing, -dock_spacing);
		real_t tab_width = Math::round(dock_slot_draw_rect.size.width / max_tabs);
		Rect2 tab_draw_rect = Rect2(dock_slot_draw_rect.position.x, dock_select_rects[i].position.y, tab_width - tab_spacing, tab_height);

		real_t max_width = tab_width * max_tabs;
		// Tabs may not fit perfectly, so they need to be re-centered.
		if (max_width > dock_slot_draw_rect.size.x) {
			tab_draw_rect.position.x -= int(max_width - dock_slot_draw_rect.size.x) / 2 * rtl_dir;
		}
		if (dock_select->is_layout_rtl()) {
			tab_draw_rect.position.x += dock_slot_draw_rect.size.x - tab_draw_rect.size.x;
		}

		int tabs_to_draw = MIN(max_tabs, dock_slot.container->get_tab_count());
		bool is_context_dock = context_tab_container == dock_slot.container;
		if (i == context_dock->dock_slot_index) {
			dock_select->draw_rect(dock_slot_draw_rect, tab_selected_color);
		} else if (!_is_slot_available(i)) {
			dock_select->draw_rect(dock_slot_draw_rect, unusable_dock_color);
		} else if (i == dock_select_rect_over_idx) {
			dock_select->draw_rect(dock_slot_draw_rect, hovered_dock_color);
		} else if (tabs_to_draw == 0) {
			dock_select->draw_rect(dock_slot_draw_rect, unused_dock_color);
		} else {
			dock_select->draw_rect(dock_slot_draw_rect, used_dock_color);
		}

		// Draw tabs above each used dock slot.
		for (int j = 0; j < tabs_to_draw; j++) {
			Color tab_color = tab_unselected_color;
			if (is_context_dock && context_tab_index == j) {
				tab_color = tab_selected_color;
			}
			Rect2 tabj_draw_rect = tab_draw_rect;
			tabj_draw_rect.position.x += tab_width * j * rtl_dir;
			dock_select->draw_rect(tabj_draw_rect, tab_color);
		}
	}
}

void DockContextPopup::_update_buttons() {
	TabContainer *context_tab_container = dock_manager->get_dock_tab_container(context_dock);
	if (context_dock->global || context_dock->closable) {
		close_button->set_tooltip_text(TTRC("Close this dock."));
		close_button->set_disabled(false);
	} else {
		close_button->set_tooltip_text(TTRC("This dock can't be closed."));
		close_button->set_disabled(true);
	}
	if (EditorNode::get_singleton()->is_multi_window_enabled()) {
		if (!(context_dock->available_layouts & EditorDock::DOCK_LAYOUT_FLOATING)) {
			make_float_button->set_tooltip_text(TTRC("This dock does not support floating."));
			make_float_button->set_disabled(true);
		} else {
			make_float_button->set_tooltip_text(TTRC("Make this dock floating."));
			make_float_button->set_disabled(false);
		}
	}

	// Update tab move buttons.
	tab_move_left_button->set_disabled(true);
	tab_move_right_button->set_disabled(true);
	if (context_tab_container && context_tab_container->get_tab_count() > 0) {
		int context_tab_index = context_tab_container->get_tab_idx_from_control(context_dock);
		tab_move_left_button->set_disabled(context_tab_index == 0);
		tab_move_right_button->set_disabled(context_tab_index >= context_tab_container->get_tab_count() - 1);
	}
	reset_size();
}

void DockContextPopup::select_current_dock_in_dock_slot(int p_dock_slot) {
	context_dock = Object::cast_to<EditorDock>(dock_manager->dock_slots[p_dock_slot].container->get_current_tab_control());
	_update_buttons();
}

void DockContextPopup::set_dock(EditorDock *p_dock) {
	context_dock = p_dock;
	_update_buttons();
}

EditorDock *DockContextPopup::get_dock() const {
	return context_dock;
}

void DockContextPopup::docks_updated() {
	if (!is_visible()) {
		return;
	}
	_update_buttons();
}

DockContextPopup::DockContextPopup() {
	dock_manager = EditorDockManager::get_singleton();

	dock_select_popup_vb = memnew(VBoxContainer);
	add_child(dock_select_popup_vb);

	HBoxContainer *header_hb = memnew(HBoxContainer);
	tab_move_left_button = memnew(Button);
	tab_move_left_button->set_accessibility_name(TTRC("Move Tab Left"));
	tab_move_left_button->set_flat(true);
	tab_move_left_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	tab_move_left_button->connect(SceneStringName(pressed), callable_mp(this, &DockContextPopup::_tab_move_left));
	header_hb->add_child(tab_move_left_button);

	Label *position_label = memnew(Label);
	position_label->set_text(TTRC("Dock Position"));
	position_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	position_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	header_hb->add_child(position_label);

	tab_move_right_button = memnew(Button);
	tab_move_right_button->set_accessibility_name(TTRC("Move Tab Right"));
	tab_move_right_button->set_flat(true);
	tab_move_right_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	tab_move_right_button->connect(SceneStringName(pressed), callable_mp(this, &DockContextPopup::_tab_move_right));

	header_hb->add_child(tab_move_right_button);
	dock_select_popup_vb->add_child(header_hb);

	dock_select = memnew(Control);
	dock_select->set_custom_minimum_size(Size2(128, 64) * EDSCALE);
	dock_select->connect(SceneStringName(gui_input), callable_mp(this, &DockContextPopup::_dock_select_input));
	dock_select->connect(SceneStringName(draw), callable_mp(this, &DockContextPopup::_dock_select_draw));
	dock_select->connect(SceneStringName(mouse_exited), callable_mp(this, &DockContextPopup::_dock_select_mouse_exited));
	dock_select->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	dock_select_popup_vb->add_child(dock_select);

	make_float_button = memnew(Button);
	make_float_button->set_text(TTRC("Make Floating"));
	if (!EditorNode::get_singleton()->is_multi_window_enabled()) {
		make_float_button->set_disabled(true);
		make_float_button->set_tooltip_text(EditorNode::get_singleton()->get_multiwindow_support_tooltip_text());
	}
	make_float_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	make_float_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	make_float_button->connect(SceneStringName(pressed), callable_mp(this, &DockContextPopup::_float_dock));
	dock_select_popup_vb->add_child(make_float_button);

	close_button = memnew(Button);
	close_button->set_text(TTRC("Close"));
	close_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	close_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	close_button->connect(SceneStringName(pressed), callable_mp(this, &DockContextPopup::_close_dock));
	dock_select_popup_vb->add_child(close_button);
}
