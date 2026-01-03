/**************************************************************************/
/*  editor_dock.h                                                         */
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

#pragma once

#include "editor/docks/editor_dock_manager.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/gui/margin_container.h"

class ConfigFile;
class Shortcut;
class WindowWrapper;

class EditorDock : public MarginContainer {
	GDCLASS(EditorDock, MarginContainer);

public:
	enum DockLayout {
		DOCK_LAYOUT_VERTICAL = DockConstants::DOCK_LAYOUT_VERTICAL,
		DOCK_LAYOUT_HORIZONTAL = DockConstants::DOCK_LAYOUT_HORIZONTAL,
		DOCK_LAYOUT_FLOATING = DockConstants::DOCK_LAYOUT_FLOATING,
		DOCK_LAYOUT_ALL = DOCK_LAYOUT_VERTICAL | DOCK_LAYOUT_HORIZONTAL | DOCK_LAYOUT_FLOATING,
	};

private:
	friend class EditorDockManager;
	friend class DockContextPopup;
	friend class DockShortcutHandler;

	String title;
	String layout_key;
	StringName icon_name;
	Ref<Texture2D> dock_icon;
	bool force_show_icon = false;
	Color title_color = Color(0, 0, 0, 0);
	Ref<Shortcut> shortcut;
	DockConstants::DockSlot default_slot = DockConstants::DOCK_SLOT_NONE;
	bool global = true;
	bool transient = false;
	bool closable = false;

	BitField<DockLayout> available_layouts = DOCK_LAYOUT_VERTICAL | DOCK_LAYOUT_FLOATING;

	bool is_open = false;
	bool enabled = true;
	int previous_tab_index = -1;
	WindowWrapper *dock_window = nullptr;
	int dock_slot_index = DockConstants::DOCK_SLOT_NONE;

	void _set_default_slot_bind(EditorPlugin::DockSlot p_slot);
	EditorPlugin::DockSlot _get_default_slot_bind() const { return (EditorPlugin::DockSlot)default_slot; }

	void _emit_changed();

protected:
	static void _bind_methods();

	GDVIRTUAL1(_update_layout, int)
	GDVIRTUAL2C(_save_layout_to_config, Ref<ConfigFile>, const String &)
	GDVIRTUAL2(_load_layout_from_config, Ref<ConfigFile>, const String &)

public:
	void open();
	void make_visible();
	void close();
	void try_hide();

	void set_title(const String &p_title);
	String get_title() const { return title; }

	void set_layout_key(const String &p_key) { layout_key = p_key; }
	String get_layout_key() const { return layout_key; }

	void set_global(bool p_global);
	bool is_global() const { return global; }

	void set_transient(bool p_transient) { transient = p_transient; }
	bool is_transient() const { return transient; }

	void set_closable(bool p_closable) { closable = p_closable; }
	bool is_closable() const { return closable; }

	void set_icon_name(const StringName &p_name);
	StringName get_icon_name() const { return icon_name; }

	void set_dock_icon(const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_dock_icon() const { return dock_icon; }

	void set_force_show_icon(bool p_force);
	bool get_force_show_icon() const { return force_show_icon; }

	void set_title_color(const Color &p_color);
	Color get_title_color() const { return title_color; }

	void set_dock_shortcut(const Ref<Shortcut> &p_shortcut);
	Ref<Shortcut> get_dock_shortcut() const { return shortcut; }

	void set_default_slot(DockConstants::DockSlot p_slot);
	DockConstants::DockSlot get_default_slot() const { return default_slot; }

	void set_available_layouts(BitField<DockLayout> p_layouts) { available_layouts = p_layouts; }
	BitField<DockLayout> get_available_layouts() const { return available_layouts; }

	String get_display_title() const;
	String get_effective_layout_key() const;

	virtual void update_layout(DockLayout p_layout) { GDVIRTUAL_CALL(_update_layout, p_layout); }
	virtual void save_layout_to_config(Ref<ConfigFile> &p_layout, const String &p_section) const { GDVIRTUAL_CALL(_save_layout_to_config, p_layout, p_section); }
	virtual void load_layout_from_config(const Ref<ConfigFile> &p_layout, const String &p_section) { GDVIRTUAL_CALL(_load_layout_from_config, p_layout, p_section); }
};

VARIANT_BITFIELD_CAST(EditorDock::DockLayout);
