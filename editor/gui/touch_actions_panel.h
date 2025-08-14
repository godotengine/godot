/**************************************************************************/
/*  touch_actions_panel.h                                                 */
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

#include "scene/gui/panel_container.h"

class BoxContainer;
class Button;
class TextureRect;

class TouchActionsPanel : public PanelContainer {
	GDCLASS(TouchActionsPanel, PanelContainer);

private:
	BoxContainer *box = nullptr;
	Button *save_button = nullptr;
	Button *delete_button = nullptr;
	Button *undo_button = nullptr;
	Button *redo_button = nullptr;
	Button *cut_button = nullptr;
	Button *copy_button = nullptr;
	Button *paste_button = nullptr;

	TextureRect *drag_handle = nullptr;
	Button *layout_toggle_button = nullptr;
	Button *lock_panel_button = nullptr;
	Button *panel_pos_button = nullptr;

	bool locked_panel = false;
	bool dragging = false;
	Vector2 drag_offset;

	enum Modifier {
		MODIFIER_CTRL,
		MODIFIER_SHIFT,
		MODIFIER_ALT
	};

	bool ctrl_btn_pressed = false;
	bool shift_btn_pressed = false;
	bool alt_btn_pressed = false;

	bool is_floating = false; // Embedded panel mode is default.
	int embedded_panel_index = 0;

	void _notification(int p_what);
	virtual void input(const Ref<InputEvent> &event) override;

	void _simulate_editor_shortcut(const String &p_shortcut_name);
	void _simulate_key_press(Key p_keycode);
	void _on_drag_handle_gui_input(const Ref<InputEvent> &p_event);
	void _switch_layout();
	void _lock_panel_toggled(bool p_pressed);
	void _switch_embedded_panel_side();

	Button *_add_new_action_button(const String &p_shortcut, const String &p_name, Key p_keycode = Key::NONE);
	void _add_new_modifier_button(Modifier p_modifier);
	void _on_modifier_button_toggled(bool p_pressed, int p_modifier);

	void _hardware_keyboard_connected(bool p_connected);

public:
	TouchActionsPanel();
};
