/**************************************************************************/
/*  text_control.cpp                                                      */
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

#include "text_control.h"

void TextControl::_generate_context_menu() {
	menu = memnew(PopupMenu);
	add_child(menu, false, INTERNAL_MODE_FRONT);

	menu_dir = memnew(PopupMenu);
	menu_dir->set_name("DirMenu");
	menu_dir->add_radio_check_item(RTR("Same as Layout Direction"), MENU_DIR_INHERITED);
	menu_dir->add_radio_check_item(RTR("Auto-Detect Direction"), MENU_DIR_AUTO);
	menu_dir->add_radio_check_item(RTR("Left-to-Right"), MENU_DIR_LTR);
	menu_dir->add_radio_check_item(RTR("Right-to-Left"), MENU_DIR_RTL);
	menu->add_child(menu_dir, false, INTERNAL_MODE_FRONT);

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
	menu->add_child(menu_ctl, false, INTERNAL_MODE_FRONT);

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

	menu->connect("id_pressed", callable_mp(this, &TextControl::menu_option));
	menu_dir->connect("id_pressed", callable_mp(this, &TextControl::menu_option));
	menu_ctl->connect("id_pressed", callable_mp(this, &TextControl::menu_option));

	menu->connect(SNAME("focus_entered"), callable_mp(this, &TextControl::_context_menu_focus_changed));
	menu->connect(SNAME("focus_exited"), callable_mp(this, &TextControl::_context_menu_focus_changed));
}

void TextControl::_context_menu_focus_changed() {
}
void TextControl::menu_option(int p_option) {
}