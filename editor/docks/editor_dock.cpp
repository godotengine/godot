/**************************************************************************/
/*  editor_dock.cpp                                                       */
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

#include "editor_dock.h"

#include "core/input/shortcut.h"
#include "core/io/config_file.h"
#include "editor/docks/editor_dock_manager.h"

void EditorDock::_set_default_slot_bind(EditorPlugin::DockSlot p_slot) {
	ERR_FAIL_COND(p_slot < EditorPlugin::DOCK_SLOT_NONE || p_slot >= EditorPlugin::DOCK_SLOT_MAX);
	default_slot = (DockConstants::DockSlot)p_slot;
}

void EditorDock::_emit_changed() {
	emit_signal(SNAME("_tab_style_changed"));
}

void EditorDock::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			set_accessibility_region(true);
			set_accessibility_name(get_display_title());
		} break;
	}
}

void EditorDock::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open"), &EditorDock::open);
	ClassDB::bind_method(D_METHOD("make_visible"), &EditorDock::make_visible);
	ClassDB::bind_method(D_METHOD("close"), &EditorDock::close);

	ClassDB::bind_method(D_METHOD("set_title", "title"), &EditorDock::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &EditorDock::get_title);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");

	ClassDB::bind_method(D_METHOD("set_layout_key", "layout_key"), &EditorDock::set_layout_key);
	ClassDB::bind_method(D_METHOD("get_layout_key"), &EditorDock::get_layout_key);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "layout_key"), "set_layout_key", "get_layout_key");

	ClassDB::bind_method(D_METHOD("set_global", "global"), &EditorDock::set_global);
	ClassDB::bind_method(D_METHOD("is_global"), &EditorDock::is_global);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "global"), "set_global", "is_global");

	ClassDB::bind_method(D_METHOD("set_transient", "transient"), &EditorDock::set_transient);
	ClassDB::bind_method(D_METHOD("is_transient"), &EditorDock::is_transient);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "transient"), "set_transient", "is_transient");

	ClassDB::bind_method(D_METHOD("set_closable", "closable"), &EditorDock::set_closable);
	ClassDB::bind_method(D_METHOD("is_closable"), &EditorDock::is_closable);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "closable"), "set_closable", "is_closable");

	ClassDB::bind_method(D_METHOD("set_icon_name", "icon_name"), &EditorDock::set_icon_name);
	ClassDB::bind_method(D_METHOD("get_icon_name"), &EditorDock::get_icon_name);
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "icon_name"), "set_icon_name", "get_icon_name");

	ClassDB::bind_method(D_METHOD("set_dock_icon", "icon"), &EditorDock::set_dock_icon);
	ClassDB::bind_method(D_METHOD("get_dock_icon"), &EditorDock::get_dock_icon);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "dock_icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_dock_icon", "get_dock_icon");

	ClassDB::bind_method(D_METHOD("set_force_show_icon", "force"), &EditorDock::set_force_show_icon);
	ClassDB::bind_method(D_METHOD("get_force_show_icon"), &EditorDock::get_force_show_icon);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "force_show_icon"), "set_force_show_icon", "get_force_show_icon");

	ClassDB::bind_method(D_METHOD("set_title_color", "color"), &EditorDock::set_title_color);
	ClassDB::bind_method(D_METHOD("get_title_color"), &EditorDock::get_title_color);
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "title_color"), "set_title_color", "get_title_color");

	ClassDB::bind_method(D_METHOD("set_dock_shortcut", "shortcut"), &EditorDock::set_dock_shortcut);
	ClassDB::bind_method(D_METHOD("get_dock_shortcut"), &EditorDock::get_dock_shortcut);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "dock_shortcut", PROPERTY_HINT_RESOURCE_TYPE, "Shortcut"), "set_dock_shortcut", "get_dock_shortcut");

	ClassDB::bind_method(D_METHOD("set_default_slot", "slot"), &EditorDock::_set_default_slot_bind);
	ClassDB::bind_method(D_METHOD("get_default_slot"), &EditorDock::_get_default_slot_bind);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "default_slot", PROPERTY_HINT_ENUM, "None:-1,Left Side Upper-Left,Left Side Bottom-Left,Left Side Upper-Right,Left Side Bottom-Right,Right Side Upper-Left,Right Side Bottom-Left,Right Side Upper-Right,Right Side Bottom-Right,Bottom"), "set_default_slot", "get_default_slot");

	ClassDB::bind_method(D_METHOD("set_available_layouts", "layouts"), &EditorDock::set_available_layouts);
	ClassDB::bind_method(D_METHOD("get_available_layouts"), &EditorDock::get_available_layouts);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "available_layouts", PROPERTY_HINT_FLAGS, "Vertical:1,Horizontal:2,Floating:4"), "set_available_layouts", "get_available_layouts");

	ADD_SIGNAL(MethodInfo("closed"));
	ADD_SIGNAL(MethodInfo("_tab_style_changed"));

	BIND_BITFIELD_FLAG(DOCK_LAYOUT_VERTICAL);
	BIND_BITFIELD_FLAG(DOCK_LAYOUT_HORIZONTAL);
	BIND_BITFIELD_FLAG(DOCK_LAYOUT_FLOATING);
	BIND_BITFIELD_FLAG(DOCK_LAYOUT_ALL);

	GDVIRTUAL_BIND(_update_layout, "layout");
	GDVIRTUAL_BIND(_save_layout_to_config, "config", "section");
	GDVIRTUAL_BIND(_load_layout_from_config, "config", "section");
}

void EditorDock::open() {
	if (!is_open) {
		EditorDockManager::get_singleton()->open_dock(this, false);
	}
}

void EditorDock::make_visible() {
	EditorDockManager::get_singleton()->open_dock(this, true);
}

void EditorDock::close() {
	if (is_open) {
		EditorDockManager::get_singleton()->close_dock(this);
	}
}

void EditorDock::set_title(const String &p_title) {
	if (title == p_title) {
		return;
	}
	title = p_title;
	set_accessibility_name(get_display_title());
	_emit_changed();
}

void EditorDock::set_global(bool p_global) {
	if (global == p_global) {
		return;
	}
	global = p_global;
	if (is_inside_tree()) {
		EditorDockManager::get_singleton()->update_docks_menu();
	}
}

void EditorDock::set_icon_name(const StringName &p_name) {
	if (icon_name == p_name) {
		return;
	}
	icon_name = p_name;
	_emit_changed();
}

void EditorDock::set_dock_icon(const Ref<Texture2D> &p_icon) {
	if (dock_icon == p_icon) {
		return;
	}
	dock_icon = p_icon;
	_emit_changed();
}

void EditorDock::set_force_show_icon(bool p_force) {
	if (force_show_icon == p_force) {
		return;
	}
	force_show_icon = p_force;
	_emit_changed();
}

void EditorDock::set_title_color(const Color &p_color) {
	if (title_color == p_color) {
		return;
	}
	title_color = p_color;
	_emit_changed();
}

void EditorDock::set_dock_shortcut(const Ref<Shortcut> &p_shortcut) {
	if (shortcut == p_shortcut) {
		return;
	}

	const Callable changed_callback = callable_mp(this, &EditorDock::_emit_changed);
	if (shortcut.is_valid()) {
		shortcut->disconnect_changed(changed_callback);
	}
	shortcut = p_shortcut;
	if (shortcut.is_valid()) {
		shortcut->connect_changed(changed_callback);
	}
	_emit_changed();
}

void EditorDock::set_default_slot(DockConstants::DockSlot p_slot) {
	ERR_FAIL_INDEX(p_slot, DockConstants::DOCK_SLOT_MAX);
	default_slot = p_slot;
}

String EditorDock::get_display_title() const {
	if (!title.is_empty()) {
		return title;
	}

	const String sname = get_name();
	if (sname.contains_char('@')) {
		// Auto-generated name, try to use something better.
		const Node *child = get_child_count() > 0 ? get_child(0) : nullptr;
		if (child) {
			// In user plugins, the child will usually be dock's content and have a proper name.
			return child->get_name();
		}
	}
	return sname;
}

String EditorDock::get_effective_layout_key() const {
	return layout_key.is_empty() ? get_display_title() : layout_key;
}
