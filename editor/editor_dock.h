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

#include "editor/editor_dock_manager.h"
#include "scene/gui/margin_container.h"

class ConfigFile;
class Shortcut;
class WindowWrapper;

class EditorDock : public MarginContainer {
	GDCLASS(EditorDock, MarginContainer);

public:
	enum DockLayout {
		DOCK_LAYOUT_VERTICAL = 1,
		DOCK_LAYOUT_HORIZONTAL = 2,
	};

private:
	friend class EditorDockManager;
	friend class DockContextPopup;

	String title;
	StringName icon_name;
	Ref<Texture2D> dock_icon;
	Ref<Shortcut> shortcut;

	BitField<DockLayout> available_layouts = DOCK_LAYOUT_VERTICAL;

	bool open = false;
	bool enabled = true;
	bool at_bottom = false;
	int previous_tab_index = -1;
	bool previous_at_bottom = false;
	WindowWrapper *dock_window = nullptr;
	int dock_slot_index = EditorDockManager::DOCK_SLOT_NONE;

protected:
	static void _bind_methods();

	GDVIRTUAL1(_set_horizontal_layout, bool)
	GDVIRTUAL2C(_save_layout_to_config, Ref<ConfigFile>, const String &)
	GDVIRTUAL2(_load_layout_from_config, Ref<ConfigFile>, const String &)

public:
	void set_title(const String &p_title);
	String get_title() const;

	void set_icon_name(const StringName &p_name);
	StringName get_icon_name() const;

	void set_dock_icon(const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_dock_icon() const;

	void set_dock_shortcut(const Ref<Shortcut> &p_shortcut);
	Ref<Shortcut> get_dock_shortcut() const;

	void set_available_layouts(BitField<DockLayout> p_layouts);
	BitField<DockLayout> get_available_layouts() const;

	String get_display_title() const;

	virtual void set_horizontal_layout(bool p_enable);
	virtual void save_layout_to_config(Ref<ConfigFile> &p_layout, const String &p_section) const;
	virtual void load_layout_from_config(const Ref<ConfigFile> &p_layout, const String &p_section);
};

VARIANT_BITFIELD_CAST(EditorDock::DockLayout);
