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

void EditorDock::_set_default_slot_bind(EditorPlugin::DockSlot p_slot) {
	ERR_FAIL_COND(p_slot < EditorPlugin::DOCK_SLOT_NONE || p_slot >= EditorPlugin::DOCK_SLOT_MAX);
	default_slot = (EditorDockManager::DockSlot)p_slot;
}

void EditorDock::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_title", "title"), &EditorDock::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &EditorDock::get_title);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");

	ClassDB::bind_method(D_METHOD("set_layout_key", "layout_key"), &EditorDock::set_layout_key);
	ClassDB::bind_method(D_METHOD("get_layout_key"), &EditorDock::get_layout_key);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "layout_key"), "set_layout_key", "get_layout_key");

	ClassDB::bind_method(D_METHOD("set_icon_name", "icon_name"), &EditorDock::set_icon_name);
	ClassDB::bind_method(D_METHOD("get_icon_name"), &EditorDock::get_icon_name);
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "icon_name"), "set_icon_name", "get_icon_name");

	ClassDB::bind_method(D_METHOD("set_dock_icon", "icon"), &EditorDock::set_dock_icon);
	ClassDB::bind_method(D_METHOD("get_dock_icon"), &EditorDock::get_dock_icon);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "dock_icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_dock_icon", "get_dock_icon");

	ClassDB::bind_method(D_METHOD("set_dock_shortcut", "shortcut"), &EditorDock::set_dock_shortcut);
	ClassDB::bind_method(D_METHOD("get_dock_shortcut"), &EditorDock::get_dock_shortcut);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "dock_shortcut", PROPERTY_HINT_RESOURCE_TYPE, "ShortCut"), "set_dock_shortcut", "get_dock_shortcut");

	ClassDB::bind_method(D_METHOD("set_default_slot", "slot"), &EditorDock::_set_default_slot_bind);
	ClassDB::bind_method(D_METHOD("get_default_slot"), &EditorDock::_get_default_slot_bind);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "default_slot"), "set_default_slot", "get_default_slot");

	ClassDB::bind_method(D_METHOD("set_available_layouts", "layouts"), &EditorDock::set_available_layouts);
	ClassDB::bind_method(D_METHOD("get_available_layouts"), &EditorDock::get_available_layouts);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "available_layouts", PROPERTY_HINT_FLAGS, "Vertical:1,Horizontal:2"), "set_available_layouts", "get_available_layouts");

	BIND_BITFIELD_FLAG(DOCK_LAYOUT_VERTICAL);
	BIND_BITFIELD_FLAG(DOCK_LAYOUT_HORIZONTAL);

	GDVIRTUAL_BIND(_update_layout, "layout");
	GDVIRTUAL_BIND(_save_layout_to_config, "config", "section");
	GDVIRTUAL_BIND(_load_layout_from_config, "config", "section");
}

EditorDock::EditorDock() {
	set_clip_contents(true);
	add_user_signal(MethodInfo("tab_style_changed"));
}

void EditorDock::set_title(const String &p_title) {
	if (title == p_title) {
		return;
	}
	title = p_title;
	emit_signal("tab_style_changed");
}

void EditorDock::set_icon_name(const StringName &p_name) {
	if (icon_name == p_name) {
		return;
	}
	icon_name = p_name;
	emit_signal("tab_style_changed");
}

void EditorDock::set_dock_icon(const Ref<Texture2D> &p_icon) {
	if (dock_icon == p_icon) {
		return;
	}
	dock_icon = p_icon;
	emit_signal("tab_style_changed");
}

void EditorDock::set_default_slot(EditorDockManager::DockSlot p_slot) {
	ERR_FAIL_INDEX(p_slot, EditorDockManager::DOCK_SLOT_MAX);
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
