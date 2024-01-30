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

#ifndef EDITOR_DOCK_MANAGER_H
#define EDITOR_DOCK_MANAGER_H

#include "scene/gui/split_container.h"

class Button;
class ConfigFile;
class Control;
class PopupPanel;
class TabContainer;
class WindowWrapper;

class DockSplitContainer : public SplitContainer {
	GDCLASS(DockSplitContainer, SplitContainer);

private:
	bool is_updating = false;

protected:
	void _update_visibility();

	virtual void add_child_notify(Node *p_child) override;
	virtual void remove_child_notify(Node *p_child) override;
};

class EditorDockManager : public Object {
	GDCLASS(EditorDockManager, Object);

public:
	enum DockSlot {
		DOCK_SLOT_LEFT_UL,
		DOCK_SLOT_LEFT_BL,
		DOCK_SLOT_LEFT_UR,
		DOCK_SLOT_LEFT_BR,
		DOCK_SLOT_RIGHT_UL,
		DOCK_SLOT_RIGHT_BL,
		DOCK_SLOT_RIGHT_UR,
		DOCK_SLOT_RIGHT_BR,
		DOCK_SLOT_MAX
	};

private:
	static EditorDockManager *singleton;

	// To access splits easily by index.
	Vector<DockSplitContainer *> vsplits;
	Vector<DockSplitContainer *> hsplits;

	Vector<WindowWrapper *> floating_docks;
	Vector<Control *> bottom_docks;
	TabContainer *dock_slot[DOCK_SLOT_MAX];
	bool docks_visible = true;

	PopupPanel *dock_select_popup = nullptr;
	Button *dock_float = nullptr;
	Button *dock_to_bottom = nullptr;
	Button *dock_tab_move_left = nullptr;
	Button *dock_tab_move_right = nullptr;
	Control *dock_select = nullptr;
	Rect2 dock_select_rect[DOCK_SLOT_MAX];
	int dock_select_rect_over_idx = -1;
	int dock_popup_selected_idx = -1;
	int dock_bottom_selected_idx = -1;

	void _dock_select_popup_theme_changed();
	void _dock_popup_exit();
	void _dock_pre_popup(int p_dock_slot);
	void _dock_move_left();
	void _dock_move_right();
	void _dock_select_input(const Ref<InputEvent> &p_input);
	void _dock_select_draw();
	void _dock_split_dragged(int p_offset);

	void _dock_tab_changed(int p_tab);
	void _edit_current();

	void _dock_floating_close_request(WindowWrapper *p_wrapper);
	void _dock_make_selected_float();
	void _dock_make_float(Control *p_control, int p_slot_index, bool p_show_window = true);
	void _restore_floating_dock(const Dictionary &p_dock_dump, Control *p_wrapper, int p_slot_index);

	void _dock_move_selected_to_bottom();

protected:
	static void _bind_methods();

public:
	static EditorDockManager *get_singleton() { return singleton; }

	void add_vsplit(DockSplitContainer *p_split);
	void add_hsplit(DockSplitContainer *p_split);
	void register_dock_slot(DockSlot p_dock_slot, TabContainer *p_tab_container);
	int get_vsplit_count() const;

	void save_docks_to_config(Ref<ConfigFile> p_layout, const String &p_section) const;
	void load_docks_from_config(Ref<ConfigFile> p_layout, const String &p_section);
	void update_dock_slots_visibility(bool p_keep_selected_tabs = false);

	void bottom_dock_show_placement_popup(const Rect2i &p_position, Control *p_dock);

	void close_all_floating_docks();

	void set_docks_visible(bool p_show);
	bool are_docks_visible() const;

	void add_control_to_dock(DockSlot p_slot, Control *p_control, const String &p_name = "");
	void remove_control_from_dock(Control *p_control);

	EditorDockManager();
};

#endif // EDITOR_DOCK_MANAGER_H
