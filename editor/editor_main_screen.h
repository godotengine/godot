/**************************************************************************/
/*  editor_main_screen.h                                                  */
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

#include "editor/docks/dock_tab_container.h"
#include "scene/gui/box_container.h"

class Button;
class ConfigFile;
class EditorDock;
class EditorPlugin;
class TabBar;

#ifndef DISABLE_DEPRECATED
class LegacyMainScreenContainer : public VBoxContainer {
	GDCLASS(LegacyMainScreenContainer, VBoxContainer);

	void _force_dock_visible(EditorDock *p_dock, CanvasItem *p_child);

protected:
	virtual void add_child_notify(Node *p_child) override;
	virtual void remove_child_notify(Node *p_child) override {} // Disable parent method.
};
#endif

class EditorMainScreen : public DockTabContainer {
	GDCLASS(EditorMainScreen, DockTabContainer);

private:
#ifndef DISABLE_DEPRECATED
	VBoxContainer *main_screen_vbox = nullptr;
	Vector<EditorPlugin *> editor_table;
#endif
	void _on_tab_changed(int p_tab);

protected:
	void _notification(int p_what);

	virtual void update_visibility() override { show(); } // Never hide main screen.
	virtual TabStyle get_tab_style() const override;
	virtual Rect2 get_drag_hint_rect() const override;
	virtual bool can_switch_dock() const override;

public:
#ifndef DISABLE_DEPRECATED
	EditorPlugin *adding_plugin = nullptr;
	VBoxContainer *get_control() const;
	void add_main_plugin(EditorPlugin *p_editor);
	void remove_main_plugin(EditorPlugin *p_editor);
#endif

	void select_next();
	void select_prev();

	EditorMainScreen();
};

// Second panel of the main screen workspace. Main screen docks (Game, 2D, 3D, Script...)
// moved here are shown next to the main screen, so two of them are visible at once.
// It is hidden while it holds no dock.
class EditorMainScreenSplit : public DockTabContainer {
	GDCLASS(EditorMainScreenSplit, DockTabContainer);

public:
	enum Maximized {
		MAXIMIZED_NONE,
		MAXIMIZED_MAIN, // The main screen fills the workspace; this panel is hidden.
		MAXIMIZED_SPLIT, // This panel fills the workspace; the main screen side is hidden.
	};

private:
	Control *header_reference = nullptr;
	Control *main_side = nullptr;
	TabBar *title_tabs = nullptr; // This panel's docks, in the title bar next to the main screen tabs.
	Button *maximize_button = nullptr;
	Maximized maximized = MAXIMIZED_NONE;
	bool title_tabs_queued = false;

	void _update_header_height();
	void _queue_update_title_tabs();
	void _update_title_tabs();
	void _update_title_tabs_theme();
	void _title_tab_clicked(int p_tab);
	void _maximize_toggled(bool p_pressed);

protected:
	void _notification(int p_what);

	virtual TabStyle get_tab_style() const override;
	virtual bool can_switch_dock() const override;

public:
	virtual void update_visibility() override;

	// Keeps the tab bar as tall as p_control (the scene tabs above the main screen), so the
	// panel's toolbar and viewport line up with the main screen's.
	void set_header_reference(Control *p_control);
	// p_main_side holds the scene tabs and the main screen; it is hidden while this panel is maximized.
	void set_main_side(Control *p_main_side);
	// Shows this panel's docks as tabs in p_container (the title bar's main screen tabs), so they
	// stay listed there when moved out of the main screen.
	void set_title_container(Control *p_container);

	void set_maximized(Maximized p_maximized);
	Maximized get_maximized() const { return maximized; }

	EditorMainScreenSplit();
};
