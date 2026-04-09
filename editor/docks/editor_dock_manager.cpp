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

#include "core/object/callable_mp.h"
#include "core/object/class_db.h" // IWYU pragma: keep. `ADD_SIGNAL` macro.
#include "editor/docks/dock_tab_container.h"
#include "editor/docks/editor_dock.h"
#include "editor/docks/floating_dock_container.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"
#include "scene/main/window.h"
#include "servers/display/display_server.h"

////////////////////////////////////////////////
////////////////////////////////////////////////

void DockSplitContainer::_notification(int p_what) {
	switch (p_what) {
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (!EditorSettings::get_singleton()->check_changed_settings_in_group("interface/touchscreen")) {
				return;
			}
			set_touch_dragger_enabled(EDITOR_GET("interface/touchscreen/enable_touch_optimizations"));
		} break;
	}
}

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
	set_drag_nested_intersections(true);
}

////////////////////////////////////////////////
////////////////////////////////////////////////

EditorDock *EditorDockManager::_get_dock_tab_dragged() {
	if (dock_tab_dragged) {
		return dock_tab_dragged;
	}

	Dictionary dock_drop_data = EditorNode::get_singleton()->get_viewport()->gui_get_drag_data();

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
		HBoxContainer *parent = Object::cast_to<HBoxContainer>(source_tab_bar->get_parent()); // The internal container.
		if (!parent) {
			return nullptr;
		}
		DockTabContainer *source_tab_container = Object::cast_to<DockTabContainer>(parent->get_parent());
		if (!source_tab_container) {
			return nullptr;
		}

		dock_tab_dragged = source_tab_container->get_dock(dock_drop_data["tab_index"]);
		if (!dock_tab_dragged) {
			return nullptr;
		}

		for (int i = 0; i < EditorDock::DOCK_SLOT_MAX; i++) {
			dock_slots[i]->show_drag_hint();
		}
		for (KeyValue<int, FloatingDockContainer *> &slot : floating_slots) {
			slot.value->show_drag_hint();
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

	const Ref<Texture2D> default_icon = docks_menu->get_editor_theme_icon(SNAME("Window"));
	const Color closed_icon_color_mod = Color(1, 1, 1, 0.5);

	bool global_menu = !bool(EDITOR_GET("interface/editor/appearance/use_embedded_menu")) && NativeMenu::get_singleton()->has_feature(NativeMenu::FEATURE_GLOBAL_MENU);
	bool dark_mode = DisplayServer::get_singleton()->is_dark_mode_supported() && DisplayServer::get_singleton()->is_dark_mode();
	int icon_max_width = EditorNode::get_singleton()->get_editor_theme()->get_constant(SNAME("class_icon_size"), EditorStringName(Editor));

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

		const Ref<Texture2D> icon = dock->get_effective_icon(icon_fetch);
		docks_menu->set_item_icon(id, icon.is_valid() ? icon : default_icon);
		if (!dock->is_open) {
			docks_menu->set_item_icon_modulate(id, closed_icon_color_mod);
			docks_menu->set_item_tooltip(id, vformat(TTR("Open the %s dock."), TTR(dock->get_display_title())));
		} else {
			docks_menu->set_item_tooltip(id, vformat(TTR("Focus on the %s dock."), TTR(dock->get_display_title())));
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

void EditorDockManager::_load_docks_in_slot(int p_slot, Ref<ConfigFile> p_layout, const String &p_section, bool p_first_load, const HashMap<String, EditorDock *> &p_dock_map, const Array &p_closed_docks) {
	const String key = DockTabContainer::get_config_key(p_slot);
	if (!p_layout->has_section_key(p_section, key)) {
		return;
	}

	DockTabContainer *dock_container = _get_dock_slot(p_slot);
	ERR_FAIL_NULL(dock_container);

	const Vector<String> names = String(p_layout->get_value(p_section, key)).split(",");
	for (int i = names.size() - 1; i >= 0; i--) {
		const String &name = names[i];
		const String section_name = p_section + "/" + name;

		if (!p_dock_map.has(name)) {
			continue;
		}
		EditorDock *dock = p_dock_map[name];

		if (!dock->enabled) {
			// Don't open disabled docks.
			dock->load_layout_from_config(p_layout, section_name);
			continue;
		}

		if (p_slot >= 0 && !(dock->transient && !dock->is_open)) {
			// Safe to include transient open docks here because they won't be in the closed dock dump.
			if (p_closed_docks.has(name)) {
				dock->is_open = false;
				dock->hide();
				_move_dock(dock, closed_dock_parent);
			} else {
				dock->is_open = true;
				_move_dock(dock, dock_container, 0, false);
			}
		}
		dock->load_layout_from_config(p_layout, section_name);

		dock->dock_slot_index = p_slot;
		dock->previous_tab_index = p_slot >= 0 ? i : 0;
	}
	int selected_tab_idx = p_layout->get_value(p_section, key + "_selected_tab_idx", -1);
	dock_container->load_selected_tab(selected_tab_idx);
}

FloatingDockContainer *EditorDockManager::_create_floating_dock_slot(const Vector2i &p_position, const Vector2i &p_size, int p_idx) {
	if (p_idx == -1) {
		bool added = false;
		for (int i = EditorDock::DOCK_SLOT_BASE_FLOATING; i < EditorDock::DOCK_SLOT_BASE_FLOATING + 1000; i++) {
			if (!floating_slots.has(i)) {
				p_idx = i;
				added = true;
				break;
			}
		}
		ERR_FAIL_COND_V_MSG(!added, nullptr, "No space found for floating dock slot.");
	}
	FloatingDockContainer *floating_container = memnew(FloatingDockContainer(p_idx));
	floating_container->window->set_position(p_position);
	floating_container->window->set_size(p_size);
	if (EDITOR_GET("interface/multi_window/maximize_window").operator bool()) {
		floating_container->window->set_mode(Window::MODE_MAXIMIZED);
	}
	_register_floating_dock_slot(floating_container);
	floating_container->show();
	return floating_container;
}

void EditorDockManager::_open_dock_in_window(EditorDock *p_dock) {
	ERR_FAIL_NULL(p_dock);

	Size2 borders = Size2(4, 4) * EDSCALE;
	// Remember size and position before removing it from the main window.
	Size2 dock_size = p_dock->get_size() + borders * 2;
	Point2 dock_screen_pos = p_dock->get_screen_position();

	Window *current_window = p_dock->get_window();
	FloatingDockContainer *floating_container;
	if (current_window != EditorNode::get_singleton()->get_window()) {
		// The dock is already floating, so copy the current window's rect.
		floating_container = _create_floating_dock_slot(current_window->get_position(), current_window->get_size());
	} else {
		floating_container = _create_floating_dock_slot(dock_screen_pos, dock_size);
	}
	_move_dock(p_dock, floating_container);

	p_dock->is_open = true;
	p_dock->show();

	_update_layout();
	p_dock->get_window()->grab_focus();
}

void EditorDockManager::_move_dock(EditorDock *p_dock, Control *p_target, int p_tab_index, bool p_set_current) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(!all_docks.has(p_dock), vformat("Cannot move unknown dock '%s'.", p_dock->get_display_title()));

	Node *parent = p_dock->get_parent();
	if (parent == p_target) {
		if (parent && p_tab_index >= 0) {
			// Only change the tab index.
			p_dock->set_tab_index(p_tab_index, p_set_current);
		}
		return;
	}

	DockTabContainer *dock_tab_container = Object::cast_to<DockTabContainer>(p_target);

	// Remove dock from its existing parent.
	if (parent) {
		DockTabContainer *parent_tabs = Object::cast_to<DockTabContainer>(parent);
		if (parent_tabs) {
			p_dock->previous_tab_index = parent_tabs->get_tab_idx_from_control(p_dock);

			// Swap to previous tab when closing current tab.
			if (parent_tabs->get_current_tab() == p_dock->previous_tab_index && parent_tabs->get_previous_tab() != -1) {
				parent_tabs->set_current_tab(parent_tabs->get_previous_tab());
			}
		}
		parent->set_block_signals(true);
		parent->remove_child(p_dock);
		parent->set_block_signals(false);
		if (parent_tabs) {
			parent_tabs->update_visibility();
			if (p_target != closed_dock_parent) {
				parent_tabs->dock_removed(p_dock);
			}
		} else if (p_target != closed_dock_parent && p_dock->dock_slot_index > -1) {
			parent_tabs = _get_dock_slot(p_dock->dock_slot_index);
			if (parent_tabs && (!dock_tab_container || parent_tabs != dock_tab_container)) {
				// Closed dock moved to another dock (likely result of layout change). Notify previous owner.
				parent_tabs->dock_removed(p_dock);
			}
		}
	}

	if (!p_target) {
		p_dock->is_open = false;
		return;
	}

	// Prevent extra visibility signals from firing.
	p_dock->hide();

	if (p_target != closed_dock_parent) {
		if (dock_tab_container->layout != p_dock->current_layout) {
			p_dock->update_layout(dock_tab_container->layout);
			p_dock->current_layout = dock_tab_container->layout;
		}
		p_dock->dock_slot_index = dock_tab_container->dock_slot;
	}

	// Add dock to its new parent, at the given tab index.
	p_target->set_block_signals(true);
	p_target->add_child(p_dock);
	p_target->set_block_signals(false);

	if (dock_tab_container) {
		dock_tab_container->dock_added(p_dock);
		if (dock_tab_container->is_inside_tree()) {
			p_dock->update_tab_style();
		}
		if (p_tab_index >= 0) {
			p_dock->set_tab_index(p_tab_index, p_set_current);
		}
		dock_tab_container->update_visibility();
	}
}

DockTabContainer *EditorDockManager::_get_dock_slot(int p_idx) {
	if (p_idx >= EditorDock::DOCK_SLOT_BASE_FLOATING) {
		FloatingDockContainer **floatainer = floating_slots.getptr(p_idx);
		ERR_FAIL_NULL_V_MSG(floatainer, nullptr, vformat("No floating dock slot with index %d.", p_idx));
		return *floatainer;
	}
	ERR_FAIL_INDEX_V(p_idx, EditorDock::DOCK_SLOT_MAX, nullptr);
	return dock_slots[p_idx];
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
		dock->update_tab_style();
	}
	dirty_docks.clear();

	if (update_menu) {
		update_docks_menu();
	}
}

void EditorDockManager::_register_floating_dock_slot(FloatingDockContainer *p_tab_container) {
	ERR_FAIL_NULL(p_tab_container);
	floating_slots[p_tab_container->dock_slot] = p_tab_container;
	EditorNode::get_singleton()->get_gui_base()->add_child(p_tab_container->window);

	p_tab_container->set_dock_context_popup(dock_context_popup);
	p_tab_container->connect("tab_changed", callable_mp(this, &EditorDockManager::_update_layout).unbind(1));
	p_tab_container->connect("active_tab_rearranged", callable_mp(this, &EditorDockManager::_update_layout).unbind(1));
	p_tab_container->window->connect("close_requested", callable_mp(this, &EditorDockManager::_close_floating_dock_slot).bind(p_tab_container));
}

void EditorDockManager::_close_floating_dock_slot(FloatingDockContainer *p_tab_container) {
	int dock_count = p_tab_container->get_tab_count();
	for (int i = dock_count - 1; i >= 0; i--) {
		EditorDock *dock = p_tab_container->get_dock(i);
		if (dock->global || dock->closable || dock->default_slot == EditorDock::DOCK_SLOT_NONE) {
			close_dock(dock);
		} else {
			_move_dock(dock, dock_slots[dock->default_slot]);
		}
	}
}

void EditorDockManager::save_docks_to_config(Ref<ConfigFile> p_layout, const String &p_section) const {
	// Save docks by dock slot.
	for (int i = 0; i < EditorDock::DOCK_SLOT_MAX; i++) {
		dock_slots[i]->save_docks_to_config(p_layout, p_section);
	}
	for (const KeyValue<int, FloatingDockContainer *> &slot : floating_slots) {
		slot.value->save_docks_to_config(p_layout, p_section);
	}

	// Clear the special dock slot for docks without default slots (index -1 = dock_0).
	// This prevents closed docks from being infinitely appended to the config on each save.
	p_layout->erase_section_key_if_exists(p_section, "dock_0");

	// Cleanup removed floating dock containers.
	for (int idx : removed_floating_docks) {
		const String config_key = DockTabContainer::get_config_key(idx);
		p_layout->erase_section_key_if_exists(p_section, config_key);
		p_layout->erase_section_key_if_exists(p_section, config_key + "_selected_tab_idx");
		p_layout->erase_section_key_if_exists(p_section, "floating_" + config_key);
	}
	removed_floating_docks.clear();
	// Save floating dock containers.
	for (const KeyValue<int, FloatingDockContainer *> slot : floating_slots) {
		const String config_key = "floating_" + DockTabContainer::get_config_key(slot.key);
		p_layout->set_value(p_section, config_key, slot.value->get_window_layout());
	}

	Array closed_docks_dump;
	for (const EditorDock *dock : all_docks) {
		const String section_name = p_section + "/" + dock->get_effective_layout_key();
		dock->save_layout_to_config(p_layout, section_name);

		if (dock->is_open) {
			continue;
		}

		// Save closed docks.
		const String name = dock->get_effective_layout_key();
		if (!dock->transient) {
			closed_docks_dump.push_back(name);
		}

		int dock_slot_id = dock->dock_slot_index;
		String config_key = DockTabContainer::get_config_key(dock_slot_id);

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
	Array closed_docks = p_layout->get_value(p_section, "dock_closed", Array());

	bool allow_floating_docks = EditorNode::get_singleton()->is_multi_window_enabled() && (!p_first_load || EDITOR_GET("interface/multi_window/restore_windows_on_load"));
	if (allow_floating_docks) {
		const PackedStringArray keys = p_layout->get_section_keys(p_section);
		for (const String &key : keys) {
			if (!key.begins_with("floating_dock")) {
				continue;
			}
			int idx = key.trim_prefix("floating_dock_").to_int() - 1; // Slots stored in layout have index + 1.
			const Rect2i window_rect = FloatingDockContainer::get_window_rect_from_layout(p_layout->get_value(p_section, key));
			_create_floating_dock_slot(window_rect.position, window_rect.size, idx);
		}
	}

	// Store the docks by name for easy lookup.
	HashMap<String, EditorDock *> dock_map;
	for (EditorDock *dock : all_docks) {
		dock_map[dock->get_effective_layout_key()] = dock;
	}

	// Load docks by slot. Index -1 is for docks that have no slot.
	for (int i = -1; i < EditorDock::DOCK_SLOT_MAX; i++) {
		_load_docks_in_slot(i, p_layout, p_section, p_first_load, dock_map, closed_docks);
	}
	for (const KeyValue<int, FloatingDockContainer *> slot : floating_slots) {
		_load_docks_in_slot(slot.key, p_layout, p_section, p_first_load, dock_map, closed_docks);
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
	DockTabContainer *parent_container = p_dock->get_parent_container();
	if (parent_container) {
		parent_container->dock_closed(p_dock);
	}

	_move_dock(p_dock, closed_dock_parent);

	_update_layout();
}

void EditorDockManager::open_dock(EditorDock *p_dock, bool p_set_current) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(!all_docks.has(p_dock), vformat("Cannot open unknown dock '%s'.", p_dock->get_display_title()));

	if (p_dock->is_open) {
		// Show the dock if it is already open.
		if (p_set_current) {
			_make_dock_visible(p_dock, false);
		}
		return;
	}

	p_dock->is_open = true;

	// Open dock to its previous location.
	if (p_dock->dock_slot_index != EditorDock::DOCK_SLOT_NONE) {
		DockTabContainer *slot = _get_dock_slot(p_dock->dock_slot_index);
		int tab_index = p_dock->previous_tab_index;
		if (tab_index < 0) {
			tab_index = slot->get_tab_count();
		}

		_move_dock(p_dock, slot, tab_index, p_set_current && slot->can_switch_dock());
	} else {
		_open_dock_in_window(p_dock);
		return;
	}

	_update_layout();
}

void EditorDockManager::make_dock_floating(EditorDock *p_dock) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(!all_docks.has(p_dock), vformat("Cannot make unknown dock '%s' floating.", p_dock->get_display_title()));

	_open_dock_in_window(p_dock);
}

void EditorDockManager::_make_dock_visible(EditorDock *p_dock, bool p_grab_focus) {
	DockTabContainer *tab_container = p_dock->get_parent_container();
	if (!tab_container || !tab_container->can_switch_dock()) {
		return;
	}

	if (!p_dock->is_visible_in_tree()) {
		int tab_index = tab_container->get_tab_idx_from_control(p_dock);
		tab_container->set_current_tab(tab_index);
	}
}

void EditorDockManager::focus_dock(EditorDock *p_dock) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(!all_docks.has(p_dock), vformat("Cannot focus unknown dock '%s'.", p_dock->get_display_title()));

	if (!p_dock->enabled) {
		return;
	}

	bool was_visible = p_dock->is_visible();
	if (!p_dock->is_open) {
		p_dock->emit_signal("opened");
		open_dock(p_dock, false);
	}

	_make_dock_visible(p_dock, true);

	DockTabContainer *tab_container = p_dock->get_parent_container();
	ERR_FAIL_NULL(tab_container);
	tab_container->dock_focused(p_dock, was_visible);
}

void EditorDockManager::add_dock(EditorDock *p_dock) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND_MSG(all_docks.has(p_dock), vformat("Cannot add dock '%s', already added.", p_dock->get_display_title()));

	p_dock->dock_slot_index = p_dock->default_slot;
	all_docks.push_back(p_dock);
	p_dock->connect("_tab_style_changed", callable_mp(this, &EditorDockManager::_queue_update_tab_style).bind(p_dock));
	p_dock->connect("renamed", callable_mp(this, &EditorDockManager::_queue_update_tab_style).bind(p_dock));

	if (p_dock->default_slot != EditorDock::DOCK_SLOT_NONE) {
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
	for (int i = 0; i < EditorDock::DOCK_SLOT_MAX; i++) {
		// Show and hide in reverse order due to the SplitContainer prioritizing the last split offset.
		dock_slots[docks_visible ? i : EditorDock::DOCK_SLOT_MAX - i - 1]->update_visibility();
	}
	_update_layout();
}

bool EditorDockManager::are_docks_visible() const {
	return docks_visible;
}

void EditorDockManager::update_tab_styles() {
	for (EditorDock *dock : all_docks) {
		dock->update_tab_style();
	}
}

void EditorDockManager::set_tab_icon_max_width(int p_max_width) {
	for (int i = 0; i < EditorDock::DOCK_SLOT_MAX; i++) {
		dock_slots[i]->add_theme_constant_override(SNAME("icon_max_width"), p_max_width);
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

void EditorDockManager::register_dock_slot(DockTabContainer *p_tab_container) {
	ERR_FAIL_NULL(p_tab_container);
	dock_slots[p_tab_container->dock_slot] = p_tab_container;

	p_tab_container->set_dock_context_popup(dock_context_popup);
	p_tab_container->connect("tab_changed", callable_mp(this, &EditorDockManager::_update_layout).unbind(1));
	p_tab_container->connect("active_tab_rearranged", callable_mp(this, &EditorDockManager::_update_layout).unbind(1));
}

void EditorDockManager::destroy_floating_slot(FloatingDockContainer *p_tab_container) {
	if (p_tab_container->window->is_ancestor_of(dock_context_popup)) {
		dock_context_popup->reparent(EditorNode::get_singleton()->get_window());
	}
	_close_floating_dock_slot(p_tab_container);
	floating_slots.erase(p_tab_container->dock_slot);
	removed_floating_docks.push_back(p_tab_container->dock_slot);
	p_tab_container->window->queue_free();
	EditorNode::get_singleton()->save_editor_layout_delayed();
}

int EditorDockManager::get_vsplit_count() const {
	return vsplits.size();
}

PopupMenu *EditorDockManager::get_docks_menu() {
	return docks_menu;
}

EditorDockManager::EditorDockManager() {
	singleton = this;

	closed_dock_parent = memnew(Control);
	closed_dock_parent->hide();
	EditorNode::get_singleton()->get_gui_base()->add_child(closed_dock_parent);

	dock_context_popup = memnew(DockContextPopup);
	EditorNode::get_singleton()->get_gui_base()->add_child(dock_context_popup);
	EditorNode::get_singleton()->add_child(memnew(DockShortcutHandler));

	docks_menu = memnew(PopupMenu);
	docks_menu->set_hide_on_item_selection(false);
	docks_menu->connect(SceneStringName(id_pressed), callable_mp(this, &EditorDockManager::_docks_menu_option));
	EditorNode::get_singleton()->get_gui_base()->connect(SceneStringName(theme_changed), callable_mp(this, &EditorDockManager::update_docks_menu));
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

void DockContextPopup::_slot_clicked(int p_slot) {
	DockTabContainer *target_tab_container = dock_manager->dock_slots[p_slot];
	if (context_dock->get_parent_container() != target_tab_container) {
		dock_manager->_move_dock(context_dock, target_tab_container, target_tab_container->get_tab_count());
		dock_manager->_update_layout();
		hide();
	}
}

void DockContextPopup::_tab_move_left() {
	TabContainer *tab_container = context_dock->get_parent_container();
	if (!tab_container) {
		return;
	}
	int new_index = tab_container->get_tab_idx_from_control(context_dock) - 1;
	context_dock->set_tab_index(new_index, true);
	dock_manager->_update_layout();
	dock_select->queue_redraw();
}

void DockContextPopup::_tab_move_right() {
	TabContainer *tab_container = context_dock->get_parent_container();
	if (!tab_container) {
		return;
	}
	int new_index = tab_container->get_tab_idx_from_control(context_dock) + 1;
	context_dock->set_tab_index(new_index, true);
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

void DockContextPopup::_update_buttons() {
	if (context_dock->global || context_dock->closable) {
		close_button->set_tooltip_text(TTRC("Close this dock."));
		close_button->set_disabled(false);
	} else {
		close_button->set_tooltip_text(TTRC("This dock can't be closed."));
		close_button->set_disabled(true);
	}
	if (EditorNode::get_singleton()->is_multi_window_enabled()) {
		String tooltip;
		make_float_button->set_disabled(!context_dock->get_parent_container()->can_dock_float(context_dock, tooltip));
		make_float_button->set_tooltip_text(tooltip);
	}

	// Update tab move buttons.
	tab_move_left_button->set_disabled(true);
	tab_move_right_button->set_disabled(true);
	TabContainer *context_tab_container = context_dock->get_parent_container();
	if (context_tab_container && context_tab_container->get_tab_count() > 0) {
		int context_tab_index = context_tab_container->get_tab_idx_from_control(context_dock);
		tab_move_left_button->set_disabled(context_tab_index == 0);
		tab_move_right_button->set_disabled(context_tab_index >= context_tab_container->get_tab_count() - 1);
	}
	reset_size();
}

void DockContextPopup::set_dock(EditorDock *p_dock) {
	context_dock = p_dock;
	dock_select->context_dock = p_dock;
	_update_buttons();
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

	dock_select = memnew(DockSlotGrid);
	dock_select_popup_vb->add_child(dock_select);
	dock_select->connect("slot_clicked", callable_mp(this, &DockContextPopup::_slot_clicked));

	Control *separator = memnew(Control);
	separator->set_custom_minimum_size(Vector2(0, 8 * EDSCALE));
	dock_select_popup_vb->add_child(separator);

	make_float_button = memnew(Button);
	make_float_button->set_text(TTRC("Detach Window"));
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

void DockShortcutHandler::shortcut_input(const Ref<InputEvent> &p_event) {
	if (p_event.is_null() || !p_event->is_pressed() || p_event->is_echo()) {
		return;
	}

	for (EditorDock *dock : EditorDockManager::get_singleton()->all_docks) {
		const Ref<Shortcut> &dock_shortcut = dock->get_dock_shortcut();
		if (dock_shortcut.is_valid() && dock_shortcut->matches_event(p_event)) {
			if (!dock->transient || dock->is_open) {
				EditorDockManager::get_singleton()->focus_dock(dock);
			}
			get_viewport()->set_input_as_handled();
			break;
		}
	}
}

void DockSlotGrid::_update_rect_cache() {
	for (int i = 0; i < EditorDock::DOCK_SLOT_MAX; i++) {
		Rect2 rect = EditorDockManager::get_singleton()->dock_slots[i]->grid_rect;
		if (is_layout_rtl()) {
			rect.position.x = GRID_SIZE.x - rect.position.x - rect.size.x;
		}
		rect.position = rect.position * CELL_SIZE * EDSCALE + (rect.position + Vector2i(0, 1)) * MARGINS * EDSCALE;
		rect.size = rect.size * CELL_SIZE * EDSCALE + (rect.size - Vector2i(1, 1)) * MARGINS * EDSCALE;
		rect_cache[i] = rect;
	}

	// Temporarily hard-coded, until main screen is registered as a slot.
	{
		Rect2 rect = Rect2i(2, 0, 4, 4);
		if (is_layout_rtl()) {
			rect.position.x = GRID_SIZE.x - rect.position.x - rect.size.x;
		}
		rect.position = rect.position * CELL_SIZE * EDSCALE + (rect.position + Vector2i(0, 1)) * MARGINS * EDSCALE;
		rect.size = rect.size * CELL_SIZE * EDSCALE + (rect.size - Vector2i(1, 1)) * MARGINS * EDSCALE;
		main_screen_rect = rect;
	}
}

void DockSlotGrid::_bind_methods() {
	ADD_SIGNAL(MethodInfo("slot_clicked", PropertyInfo(Variant::INT, "slot")));
}

void DockSlotGrid::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED: {
			rect_cache_dirty = true;
		} break;

		case NOTIFICATION_DRAW: {
			if (rect_cache_dirty) {
				_update_rect_cache();
				rect_cache_dirty = false;
			}
			Color used_dock_color = Color(0.6, 0.6, 0.6, 0.8);
			Color hovered_dock_color = Color(0.8, 0.8, 0.8, 0.8);
			Color tab_selected_color = get_theme_color(SNAME("mono_color"), EditorStringName(Editor));
			Color tab_unselected_color = used_dock_color;
			Color unused_dock_color = used_dock_color;
			unused_dock_color.a = 0.4;
			Color unusable_dock_color = unused_dock_color;
			unusable_dock_color.a = 0.1;
			Color tab_unusable_color = unusable_dock_color;

			TabContainer *context_tab_container = context_dock->get_parent_container();
			int context_tab_index = -1;
			if (context_tab_container && context_tab_container->get_tab_count() > 0) {
				context_tab_index = context_tab_container->get_tab_idx_from_control(context_dock);
			}

			for (int i = 0; i < EditorDock::DOCK_SLOT_MAX; i++) {
				const Rect2i slot_rect = rect_cache[i];
				int max_tabs = EditorDockManager::get_singleton()->dock_slots[i]->grid_rect.size.x * TABS_PER_CELL;

				DockTabContainer *dock_slot = EditorDockManager::get_singleton()->dock_slots[i];
				bool is_context_slot = context_tab_container == dock_slot;
				bool is_slot_available = context_dock->available_layouts & dock_slot->layout;
				int tabs_to_draw = MIN(max_tabs, dock_slot->get_tab_count());

				if (i == context_dock->dock_slot_index) {
					draw_rect(slot_rect, tab_selected_color);
				} else if (!is_slot_available) {
					draw_rect(slot_rect, unusable_dock_color);
				} else if (i == hovered_slot) {
					draw_rect(slot_rect, hovered_dock_color);
				} else if (tabs_to_draw == 0) {
					draw_rect(slot_rect, unused_dock_color);
				} else {
					draw_rect(slot_rect, used_dock_color);
				}

				real_t tab_width = ((slot_rect.size.x - (max_tabs - 1) * TAB_MARGIN * EDSCALE) / max_tabs);
				real_t initial_offset = (slot_rect.size.x - (max_tabs * tab_width + (max_tabs - 1) * TAB_MARGIN * EDSCALE)) * 0.5;

				for (int j = 0; j < tabs_to_draw; j++) {
					real_t pos_x = is_layout_rtl()
							? slot_rect.size.x - (initial_offset + (j + 1) * tab_width + j * TAB_MARGIN * EDSCALE)
							: initial_offset + j * (tab_width + TAB_MARGIN * EDSCALE);
					const Rect2 tab_rect = Rect2(slot_rect.position + Vector2(pos_x, -MARGINS.y * EDSCALE + MARGINS.y * EDSCALE / 4), Vector2(tab_width, MARGINS.y * EDSCALE / 2));
					if (is_context_slot && context_tab_index == j) {
						draw_rect(tab_rect, tab_selected_color);
					} else if (is_slot_available) {
						draw_rect(tab_rect, tab_unselected_color);
					} else {
						draw_rect(tab_rect, tab_unusable_color);
					}
				}
			}
			draw_rect(main_screen_rect, unusable_dock_color);
		} break;

		case NOTIFICATION_MOUSE_EXIT: {
			if (hovered_slot > -1) {
				hovered_slot = -1;
				queue_redraw();
			}
		} break;
	}
}

void DockSlotGrid::gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouse> me = p_event;
	if (me.is_valid()) {
		Vector2 point = me->get_position();

		int over_dock_slot = -1;
		for (int i = 0; i < EditorDock::DOCK_SLOT_MAX; i++) {
			if (rect_cache[i].has_point(point)) {
				over_dock_slot = i;
				break;
			}
		}

		if (over_dock_slot != hovered_slot) {
			queue_redraw();
			hovered_slot = over_dock_slot;
		}

		if (over_dock_slot == -1) {
			return;
		}

		Ref<InputEventMouseButton> mb = me;
		DockTabContainer *target_tab_container = EditorDockManager::get_singleton()->dock_slots[over_dock_slot];
		if (context_dock->get_parent_container() == target_tab_container) {
			return;
		}

		if (!(context_dock->available_layouts & target_tab_container->layout)) {
			return;
		}

		if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && mb->is_pressed()) {
			emit_signal("slot_clicked", over_dock_slot);
		}
	}
}

Size2 DockSlotGrid::get_minimum_size() const {
	return GRID_SIZE * CELL_SIZE * EDSCALE + (GRID_SIZE - Vector2i(1, 0)) * MARGINS * EDSCALE;
}
