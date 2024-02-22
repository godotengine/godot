/**************************************************************************/
/*  text_edit_menu.h                                                      */
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

#ifndef TEXT_EDIT_MENU_H
#define TEXT_EDIT_MENU_H

#include "scene/gui/control.h"

enum class Key;
class PopupMenu;

class TextEditMenu {
private:
	Key _get_menu_action_accelerator(const String &p_action);

public:
	enum MenuItems {
		MENU_CUT,
		MENU_COPY,
		MENU_PASTE,
		MENU_CLEAR,
		MENU_SELECT_ALL,
		MENU_UNDO,
		MENU_REDO,
		MENU_SUBMENU_TEXT_DIR,
		MENU_DIR_INHERITED,
		MENU_DIR_AUTO,
		MENU_DIR_LTR,
		MENU_DIR_RTL,
		MENU_DISPLAY_UCC,
		MENU_SUBMENU_INSERT_UCC,
		MENU_INSERT_LRM,
		MENU_INSERT_RLM,
		MENU_INSERT_LRE,
		MENU_INSERT_RLE,
		MENU_INSERT_LRO,
		MENU_INSERT_RLO,
		MENU_INSERT_PDF,
		MENU_INSERT_ALM,
		MENU_INSERT_LRI,
		MENU_INSERT_RLI,
		MENU_INSERT_FSI,
		MENU_INSERT_PDI,
		MENU_INSERT_ZWJ,
		MENU_INSERT_ZWNJ,
		MENU_INSERT_WJ,
		MENU_INSERT_SHY,
		MENU_MAX
	};

	PopupMenu *menu = nullptr;
	PopupMenu *menu_dir = nullptr;
	PopupMenu *menu_ctl = nullptr;

	void generate_menu(Control *p_control);
	void add_option_selected_callback(const Callable &p_option_selected_callback);
	void add_focus_changed_callback(const Callable &p_focus_changed_callback);
	void update(bool p_editable, bool p_selecting_enabled, bool p_shortcut_keys_enabled, Control::TextDirection p_text_direction, bool p_draw_control_chars, bool p_has_undo, bool p_has_redo);

	void popup_at(const Point2 &p_point, bool p_grab_focus = false);
	bool has_focus();
	void set_text_direction(Control::TextDirection p_text_direction);
	void set_draw_control_chars(bool p_draw_control_chars);
	bool is_visible() const;
	bool is_initialized() const;
	PopupMenu *get_main_menu() const;
};

#endif // TEXT_EDIT_MENU_H
