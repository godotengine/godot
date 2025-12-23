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

#include "scene/gui/box_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/tab_container.h"

class ConfigFile;
class EditorDock;
class EditorPlugin;

class LegacyMainScreenContainer : public VBoxContainer {
	GDCLASS(LegacyMainScreenContainer, VBoxContainer);

	void _force_dock_visible(EditorDock *p_dock, CanvasItem *p_child);

protected:
	virtual void add_child_notify(Node *p_child) override;
};

class EditorMainScreen : public TabContainer {
	GDCLASS(EditorMainScreen, TabContainer);

private:
	VBoxContainer *main_screen_vbox = nullptr;
	EditorPlugin *selected_plugin = nullptr;

	HBoxContainer *button_hb = nullptr;
	Vector<EditorPlugin *> editor_table;

	void _on_tab_changed(int p_tab);

protected:
	void _notification(int p_what);

public:
	EditorPlugin *adding_plugin = nullptr;

	void set_button_container(HBoxContainer *p_button_hb);
	void edit(Object *p_object);

	void select_next();
	void select_prev();
	void select_by_name(const String &p_name);
	void select(int p_index);
	EditorPlugin *get_selected_plugin() const;
	bool can_auto_switch_screens() const;

	VBoxContainer *get_control() const;

	void add_main_plugin(EditorPlugin *p_editor);
	void remove_main_plugin(EditorPlugin *p_editor);

	EditorMainScreen();
};
