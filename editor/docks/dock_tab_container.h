/**************************************************************************/
/*  dock_tab_container.h                                                  */
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

#include "editor/docks/editor_dock.h"
#include "scene/gui/tab_container.h"

class ConfigFile;
class DockContextPopup;
class DockTabContainer;
class EditorDockManager;
class StyleBoxFlat;

class EditorDockDragHint : public Control {
	GDCLASS(EditorDockDragHint, Control);

	DockTabContainer *dock_container = nullptr;
	TabBar *drop_tabbar = nullptr;

	Color valid_drop_color;
	Ref<StyleBoxFlat> dock_drop_highlight;
	bool can_drop_dock = false;
	bool mouse_inside = false;
	bool mouse_inside_tabbar = false;

	void _drag_move_tab(int p_from_index, int p_to_index);
	void _drag_move_tab_from(TabBar *p_from_tabbar, int p_from_index, int p_to_index);

protected:
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

	void _notification(int p_what);
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const override;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data) override;

public:
	void set_slot(DockTabContainer *p_slot);

	EditorDockDragHint();
};

class DockTabContainer : public TabContainer {
	GDCLASS(DockTabContainer, TabContainer);

	EditorDockDragHint *drag_hint = nullptr;

	void _pre_popup();
	void _tab_rmb_clicked(int p_tab_idx);

protected:
	DockContextPopup *dock_context_popup = nullptr;

	void _notification(int p_what);

public:
	enum class TabStyle {
		TEXT_ONLY,
		ICON_ONLY,
		TEXT_AND_ICON,
	};

	EditorDock::DockSlot dock_slot = EditorDock::DOCK_SLOT_NONE;
	EditorDock::DockLayout layout = EditorDock::DOCK_LAYOUT_VERTICAL;

	static String get_config_key(int p_idx) { return "dock_" + itos(p_idx + 1); }

	virtual void dock_closed(EditorDock *p_dock) {}
	virtual void dock_focused(EditorDock *p_dock, bool p_was_visible) {}
	virtual void update_visibility();
	virtual TabStyle get_tab_style() const;
	virtual bool can_switch_dock() const;

	// There is no equivalent load method, because loading needs to handle floating and closing.
	void save_docks_to_config(Ref<ConfigFile> p_layout, const String &p_section);
	virtual void load_selected_tab(int p_idx);

	// This method should only be called by EditorDock.
	void move_dock_index(EditorDock *p_dock, int p_to_index, bool p_set_current);

	void set_dock_context_popup(DockContextPopup *p_popup);
	EditorDock *get_dock(int p_idx) const;
	void show_drag_hint();

	DockTabContainer(EditorDock::DockSlot p_slot);
};

class SideDockTabContainer : public DockTabContainer {
	GDCLASS(SideDockTabContainer, DockTabContainer);

public:
	SideDockTabContainer(EditorDock::DockSlot p_slot);
};
