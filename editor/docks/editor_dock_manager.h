/**************************************************************************/
/*  editor_dock_manager.h                                                 */
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

#include "editor/docks/dock_constants.h"
#include "scene/gui/popup.h"
#include "scene/gui/split_container.h"

class Button;
class ConfigFile;
class Control;
class EditorDock;
class PopupMenu;
class TabBar;
class TabContainer;
class VBoxContainer;
class WindowWrapper;
class StyleBoxFlat;

class DockSplitContainer : public SplitContainer {
	GDCLASS(DockSplitContainer, SplitContainer);

private:
	bool is_updating = false;

protected:
	void _update_visibility();

	virtual void add_child_notify(Node *p_child) override;
	virtual void remove_child_notify(Node *p_child) override;

public:
	DockSplitContainer();
};

class DockShortcutHandler : public Node {
	GDCLASS(DockShortcutHandler, Node);

protected:
	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;

public:
	DockShortcutHandler() { set_process_shortcut_input(true); }
};

class DockContextPopup;
class EditorDockDragHint;

class EditorDockManager : public Object {
	GDCLASS(EditorDockManager, Object);

private:
	friend class DockContextPopup;
	friend class EditorDockDragHint;
	friend class DockShortcutHandler;

	static inline EditorDockManager *singleton = nullptr;

	// To access splits easily by index.
	Vector<DockSplitContainer *> vsplits;
	DockSplitContainer *main_hsplit = nullptr;

	struct DockSlot {
		TabContainer *container = nullptr;
		EditorDockDragHint *drag_hint = nullptr;
		DockConstants::DockLayout layout = DockConstants::DOCK_LAYOUT_VERTICAL;
	};

	DockSlot dock_slots[DockConstants::DOCK_SLOT_MAX];
	Vector<WindowWrapper *> dock_windows;
	LocalVector<EditorDock *> all_docks;
	HashSet<EditorDock *> dirty_docks;

	EditorDock *dock_tab_dragged = nullptr;
	bool docks_visible = true;

	DockContextPopup *dock_context_popup = nullptr;
	PopupMenu *docks_menu = nullptr;
	LocalVector<EditorDock *> docks_menu_docks;
	Control *closed_dock_parent = nullptr;

	EditorDock *_get_dock_tab_dragged();
	void _dock_drag_stopped();
	void _dock_split_dragged(int p_offset);
	void _dock_container_popup(int p_tab_idx, TabContainer *p_dock_container);
	void _dock_container_update_visibility(TabContainer *p_dock_container);
	void _update_layout();

	void _docks_menu_option(int p_id);

	void _window_close_request(WindowWrapper *p_wrapper);
	EditorDock *_close_window(WindowWrapper *p_wrapper);
	void _open_dock_in_window(EditorDock *p_dock, bool p_show_window = true, bool p_reset_size = false);
	void _restore_dock_to_saved_window(EditorDock *p_dock, const Dictionary &p_window_dump);

	void _move_dock_tab_index(EditorDock *p_dock, int p_tab_index, bool p_set_current);
	void _move_dock(EditorDock *p_dock, Control *p_target, int p_tab_index = -1, bool p_set_current = true);

	void _queue_update_tab_style(EditorDock *p_dock);
	void _update_dirty_dock_tabs();
	void _update_tab_style(EditorDock *p_dock);

public:
	static EditorDockManager *get_singleton() { return singleton; }

	void update_docks_menu();
	void update_tab_styles();
	void set_tab_icon_max_width(int p_max_width);

	void add_vsplit(DockSplitContainer *p_split);
	void set_hsplit(DockSplitContainer *p_split);
	void register_dock_slot(DockConstants::DockSlot p_dock_slot, TabContainer *p_tab_container, DockConstants::DockLayout p_layout);
	int get_vsplit_count() const;
	PopupMenu *get_docks_menu();

	void save_docks_to_config(Ref<ConfigFile> p_layout, const String &p_section) const;
	void load_docks_from_config(Ref<ConfigFile> p_layout, const String &p_section, bool p_first_load = false);

	void set_dock_enabled(EditorDock *p_dock, bool p_enabled);
	void close_dock(EditorDock *p_dock);
	void open_dock(EditorDock *p_dock, bool p_set_current = true);
	void focus_dock(EditorDock *p_dock);
	void make_dock_floating(EditorDock *p_dock);

	TabContainer *get_dock_tab_container(Control *p_dock) const;

	void set_docks_visible(bool p_show);
	bool are_docks_visible() const;

	void add_dock(EditorDock *p_dock);
	void remove_dock(EditorDock *p_dock);

	EditorDockManager();
};

class EditorDockDragHint : public Control {
	GDCLASS(EditorDockDragHint, Control);

private:
	EditorDockManager *dock_manager = nullptr;
	DockConstants::DockSlot occupied_slot = DockConstants::DOCK_SLOT_MAX;
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
	bool can_drop_data(const Point2 &p_point, const Variant &p_data) const override;
	void drop_data(const Point2 &p_point, const Variant &p_data) override;

public:
	void set_slot(DockConstants::DockSlot p_slot);

	EditorDockDragHint();
};

class DockContextPopup : public PopupPanel {
	GDCLASS(DockContextPopup, PopupPanel);

private:
	VBoxContainer *dock_select_popup_vb = nullptr;

	Button *make_float_button = nullptr;
	Button *tab_move_left_button = nullptr;
	Button *tab_move_right_button = nullptr;
	Button *close_button = nullptr;

	Control *dock_select = nullptr;
	Rect2 dock_select_rects[DockConstants::DOCK_SLOT_MAX];
	int dock_select_rect_over_idx = -1;

	EditorDock *context_dock = nullptr;

	EditorDockManager *dock_manager = nullptr;

	void _tab_move_left();
	void _tab_move_right();
	void _close_dock();
	void _float_dock();
	bool _is_slot_available(int p_slot) const;

	void _dock_select_input(const Ref<InputEvent> &p_input);
	void _dock_select_mouse_exited();
	void _dock_select_draw();

	void _update_buttons();

protected:
	void _notification(int p_what);

public:
	void select_current_dock_in_dock_slot(int p_dock_slot);
	void set_dock(EditorDock *p_dock);
	EditorDock *get_dock() const;
	void docks_updated();

	DockContextPopup();
};
