/**************************************************************************/
/*  editor_bottom_panel.h                                                 */
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

#include "scene/gui/tab_container.h"

class Button;
class ConfigFile;
class EditorToaster;
class HBoxContainer;

class EditorBottomPanel : public TabContainer {
	GDCLASS(EditorBottomPanel, TabContainer);

	HBoxContainer *bottom_hbox = nullptr;
	EditorToaster *editor_toaster = nullptr;
	Button *pin_button = nullptr;
	Button *expand_button = nullptr;

	bool lock_panel_switching = false;
	LocalVector<Control *> bottom_docks;
	LocalVector<Ref<Shortcut>> dock_shortcuts;

	LocalVector<Button *> legacy_buttons;
	void _on_button_visibility_changed(Button *p_button, Control *p_control);

	void _repaint();
	void _on_tab_changed(int p_idx);
	void _pin_button_toggled(bool p_pressed);
	void _expand_button_toggled(bool p_pressed);

protected:
	void _notification(int p_what);

	virtual void _update_margins() override;
	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;

public:
	void save_layout_to_config(Ref<ConfigFile> p_config_file, const String &p_section) const;
	void load_layout_from_config(Ref<ConfigFile> p_config_file, const String &p_section);

	Button *add_item(String p_text, Control *p_item, const Ref<Shortcut> &p_shortcut = nullptr, bool p_at_front = false);
	void remove_item(Control *p_item);
	void make_item_visible(Control *p_item, bool p_visible = true, bool p_ignore_lock = false);
	void move_item_to_end(Control *p_item);
	void hide_bottom_panel();
	void toggle_last_opened_bottom_panel();
	void set_expanded(bool p_expanded);

	EditorBottomPanel();
	~EditorBottomPanel();
};
