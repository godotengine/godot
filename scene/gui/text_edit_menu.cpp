/**************************************************************************/
/*  text_edit_menu.cpp                                                    */
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

#include "text_edit_menu.h"

#include "core/input/input_map.h"
#include "scene/gui/popup_menu.h"

Key TextEditMenu::_get_menu_action_accelerator(const String &p_action) {
	const List<Ref<InputEvent>> *events = InputMap::get_singleton()->action_get_events(p_action);
	if (!events) {
		return Key::NONE;
	}

	// Use first event in the list for the accelerator.
	const List<Ref<InputEvent>>::Element *first_event = events->front();
	if (!first_event) {
		return Key::NONE;
	}

	const Ref<InputEventKey> event = first_event->get();
	if (event.is_null()) {
		return Key::NONE;
	}

	// Use physical keycode if non-zero.
	if (event->get_physical_keycode() != Key::NONE) {
		return event->get_physical_keycode_with_modifiers();
	} else {
		return event->get_keycode_with_modifiers();
	}
}

void TextEditMenu::generate_menu(Control *p_control) {
	menu = memnew(PopupMenu);
	p_control->add_child(menu, false, Node::INTERNAL_MODE_FRONT);

	menu_dir = memnew(PopupMenu);
	menu_dir->set_name("DirMenu");
	menu_dir->add_radio_check_item(RTR("Same as Layout Direction"), MENU_DIR_INHERITED);
	menu_dir->add_radio_check_item(RTR("Auto-Detect Direction"), MENU_DIR_AUTO);
	menu_dir->add_radio_check_item(RTR("Left-to-Right"), MENU_DIR_LTR);
	menu_dir->add_radio_check_item(RTR("Right-to-Left"), MENU_DIR_RTL);
	menu->add_child(menu_dir, false, Node::INTERNAL_MODE_FRONT);

	menu_ctl = memnew(PopupMenu);
	menu_ctl->set_name("CTLMenu");
	menu_ctl->add_item(RTR("Left-to-Right Mark (LRM)"), MENU_INSERT_LRM);
	menu_ctl->add_item(RTR("Right-to-Left Mark (RLM)"), MENU_INSERT_RLM);
	menu_ctl->add_item(RTR("Start of Left-to-Right Embedding (LRE)"), MENU_INSERT_LRE);
	menu_ctl->add_item(RTR("Start of Right-to-Left Embedding (RLE)"), MENU_INSERT_RLE);
	menu_ctl->add_item(RTR("Start of Left-to-Right Override (LRO)"), MENU_INSERT_LRO);
	menu_ctl->add_item(RTR("Start of Right-to-Left Override (RLO)"), MENU_INSERT_RLO);
	menu_ctl->add_item(RTR("Pop Direction Formatting (PDF)"), MENU_INSERT_PDF);
	menu_ctl->add_separator();
	menu_ctl->add_item(RTR("Arabic Letter Mark (ALM)"), MENU_INSERT_ALM);
	menu_ctl->add_item(RTR("Left-to-Right Isolate (LRI)"), MENU_INSERT_LRI);
	menu_ctl->add_item(RTR("Right-to-Left Isolate (RLI)"), MENU_INSERT_RLI);
	menu_ctl->add_item(RTR("First Strong Isolate (FSI)"), MENU_INSERT_FSI);
	menu_ctl->add_item(RTR("Pop Direction Isolate (PDI)"), MENU_INSERT_PDI);
	menu_ctl->add_separator();
	menu_ctl->add_item(RTR("Zero-Width Joiner (ZWJ)"), MENU_INSERT_ZWJ);
	menu_ctl->add_item(RTR("Zero-Width Non-Joiner (ZWNJ)"), MENU_INSERT_ZWNJ);
	menu_ctl->add_item(RTR("Word Joiner (WJ)"), MENU_INSERT_WJ);
	menu_ctl->add_item(RTR("Soft Hyphen (SHY)"), MENU_INSERT_SHY);
	menu->add_child(menu_ctl, false, Node::INTERNAL_MODE_FRONT);

	menu->add_item(RTR("Cut"), MENU_CUT);
	menu->add_item(RTR("Copy"), MENU_COPY);
	menu->add_item(RTR("Paste"), MENU_PASTE);
	menu->add_separator();
	menu->add_item(RTR("Select All"), MENU_SELECT_ALL);
	menu->add_item(RTR("Clear"), MENU_CLEAR);
	menu->add_separator();
	menu->add_item(RTR("Undo"), MENU_UNDO);
	menu->add_item(RTR("Redo"), MENU_REDO);
	menu->add_separator();
	menu->add_submenu_item(RTR("Text Writing Direction"), "DirMenu", MENU_SUBMENU_TEXT_DIR);
	menu->add_separator();
	menu->add_check_item(RTR("Display Control Characters"), MENU_DISPLAY_UCC);
	menu->add_submenu_item(RTR("Insert Control Character"), "CTLMenu", MENU_SUBMENU_INSERT_UCC);
}

void TextEditMenu::add_option_selected_callback(const Callable &p_option_selected_callback) {
	ERR_FAIL_NULL(menu);
	ERR_FAIL_NULL(menu_dir);
	ERR_FAIL_NULL(menu_ctl);

	menu->connect(SNAME("id_pressed"), p_option_selected_callback);
	menu_dir->connect(SNAME("id_pressed"), p_option_selected_callback);
	menu_ctl->connect(SNAME("id_pressed"), p_option_selected_callback);
}
void TextEditMenu::add_focus_changed_callback(const Callable &p_focus_changed_callback) {
	ERR_FAIL_NULL(menu);
	menu->connect(SNAME("focus_entered"), p_focus_changed_callback);
	menu->connect(SNAME("focus_exited"), p_focus_changed_callback);
}

void TextEditMenu::update(bool p_editable, bool p_selecting_enabled, bool p_shortcut_keys_enabled, Control::TextDirection p_text_direction, bool p_draw_control_chars, bool p_has_undo, bool p_has_redo) {
	ERR_FAIL_NULL(menu);

	int idx = -1;

#define MENU_ITEM_ACTION_DISABLED(m_menu, m_id, m_action, m_disabled)                                                    \
	idx = m_menu->get_item_index(m_id);                                                                                  \
	if (idx >= 0) {                                                                                                      \
		m_menu->set_item_accelerator(idx, p_shortcut_keys_enabled ? _get_menu_action_accelerator(m_action) : Key::NONE); \
		m_menu->set_item_disabled(idx, m_disabled);                                                                      \
	}

#define MENU_ITEM_ACTION(m_menu, m_id, m_action)                                                                         \
	idx = m_menu->get_item_index(m_id);                                                                                  \
	if (idx >= 0) {                                                                                                      \
		m_menu->set_item_accelerator(idx, p_shortcut_keys_enabled ? _get_menu_action_accelerator(m_action) : Key::NONE); \
	}

#define MENU_ITEM_DISABLED(m_menu, m_id, m_disabled) \
	idx = m_menu->get_item_index(m_id);              \
	if (idx >= 0) {                                  \
		m_menu->set_item_disabled(idx, m_disabled);  \
	}

#define MENU_ITEM_CHECKED(m_menu, m_id, m_checked) \
	idx = m_menu->get_item_index(m_id);            \
	if (idx >= 0) {                                \
		m_menu->set_item_checked(idx, m_checked);  \
	}

	MENU_ITEM_ACTION_DISABLED(menu, MENU_CUT, "ui_cut", !p_editable)
	MENU_ITEM_ACTION(menu, MENU_COPY, "ui_copy")
	MENU_ITEM_ACTION_DISABLED(menu, MENU_PASTE, "ui_paste", !p_editable)
	MENU_ITEM_ACTION_DISABLED(menu, MENU_SELECT_ALL, "ui_text_select_all", !p_selecting_enabled)
	MENU_ITEM_DISABLED(menu, MENU_CLEAR, !p_editable)
	MENU_ITEM_ACTION_DISABLED(menu, MENU_UNDO, "ui_undo", !p_editable || !p_has_undo)
	MENU_ITEM_ACTION_DISABLED(menu, MENU_REDO, "ui_redo", !p_editable || !p_has_redo)
	MENU_ITEM_CHECKED(menu_dir, MENU_DIR_INHERITED, p_text_direction == Control::TEXT_DIRECTION_INHERITED)
	MENU_ITEM_CHECKED(menu_dir, MENU_DIR_AUTO, p_text_direction == Control::TEXT_DIRECTION_AUTO)
	MENU_ITEM_CHECKED(menu_dir, MENU_DIR_LTR, p_text_direction == Control::TEXT_DIRECTION_LTR)
	MENU_ITEM_CHECKED(menu_dir, MENU_DIR_RTL, p_text_direction == Control::TEXT_DIRECTION_RTL)
	MENU_ITEM_CHECKED(menu, MENU_DISPLAY_UCC, p_draw_control_chars)
	MENU_ITEM_DISABLED(menu, MENU_SUBMENU_INSERT_UCC, !p_editable)

#undef MENU_ITEM_ACTION_DISABLED
#undef MENU_ITEM_ACTION
#undef MENU_ITEM_DISABLED
#undef MENU_ITEM_CHECKED
}

void TextEditMenu::popup_at(const Point2 &p_point, bool p_grab_focus) {
	ERR_FAIL_NULL(menu);
	menu->set_position(p_point);
	menu->reset_size();
	menu->popup();
	if (p_grab_focus) {
		menu->grab_focus();
	}
}

bool TextEditMenu::has_focus() {
	return menu && menu->has_focus();
}

void TextEditMenu::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_NULL(menu_dir);
	menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_INHERITED), p_text_direction == Control::TEXT_DIRECTION_INHERITED);
	menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_AUTO), p_text_direction == Control::TEXT_DIRECTION_AUTO);
	menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_LTR), p_text_direction == Control::TEXT_DIRECTION_LTR);
	menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_RTL), p_text_direction == Control::TEXT_DIRECTION_RTL);
}

void TextEditMenu::set_draw_control_chars(bool p_draw_control_chars) {
	ERR_FAIL_NULL(menu);
	if (menu->get_item_index(MENU_DISPLAY_UCC) >= 0) {
		menu->set_item_checked(menu->get_item_index(MENU_DISPLAY_UCC), p_draw_control_chars);
	}
}

bool TextEditMenu::is_visible() const {
	return menu && menu->is_visible();
}

bool TextEditMenu::is_initialized() const {
	return menu;
}

PopupMenu *TextEditMenu::get_main_menu() const {
	return menu;
}
