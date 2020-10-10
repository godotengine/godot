/*************************************************************************/
/*  editor_layouts_dialog.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "editor_layouts_dialog.h"

#include "core/class_db.h"
#include "core/io/config_file.h"
#include "core/os/keyboard.h"
#include "editor/editor_settings.h"
#include "scene/gui/item_list.h"
#include "scene/gui/line_edit.h"

void EditorLayoutsDialog::_line_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (!k->is_pressed()) {
			return;
		}

		switch (k->get_keycode()) {
			case KEY_KP_ENTER:
			case KEY_ENTER: {
				if (get_hide_on_ok()) {
					hide();
				}
				ok_pressed();
				set_input_as_handled();
			} break;
			case KEY_ESCAPE: {
				hide();
				set_input_as_handled();
			} break;
		}
	}
}

void EditorLayoutsDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("name_confirmed", PropertyInfo(Variant::STRING, "name")));
}

void EditorLayoutsDialog::ok_pressed() {
	if (layout_names->is_anything_selected()) {
		Vector<int> const selected_items = layout_names->get_selected_items();
		for (int i = 0; i < selected_items.size(); ++i) {
			emit_signal("name_confirmed", layout_names->get_item_text(selected_items[i]));
		}
	} else if (name->is_visible() && name->get_text() != "") {
		emit_signal("name_confirmed", name->get_text());
	}
}

void EditorLayoutsDialog::_post_popup() {
	ConfirmationDialog::_post_popup();
	name->clear();
	layout_names->clear();

	Ref<ConfigFile> config;
	config.instance();
	Error err = config->load(EditorSettings::get_singleton()->get_editor_layouts_config());
	if (err != OK) {
		return;
	}

	List<String> layouts;
	config.ptr()->get_sections(&layouts);

	for (List<String>::Element *E = layouts.front(); E; E = E->next()) {
		layout_names->add_item(**E);
	}
}

EditorLayoutsDialog::EditorLayoutsDialog() {
	makevb = memnew(VBoxContainer);
	add_child(makevb);
	makevb->set_anchor_and_margin(MARGIN_LEFT, Control::ANCHOR_BEGIN, 5);
	makevb->set_anchor_and_margin(MARGIN_RIGHT, Control::ANCHOR_END, -5);

	layout_names = memnew(ItemList);
	makevb->add_child(layout_names);
	layout_names->set_visible(true);
	layout_names->set_margin(MARGIN_TOP, 5);
	layout_names->set_anchor_and_margin(MARGIN_LEFT, Control::ANCHOR_BEGIN, 5);
	layout_names->set_anchor_and_margin(MARGIN_RIGHT, Control::ANCHOR_END, -5);
	layout_names->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	layout_names->set_select_mode(ItemList::SELECT_MULTI);
	layout_names->set_allow_rmb_select(true);

	name = memnew(LineEdit);
	makevb->add_child(name);
	name->set_margin(MARGIN_TOP, 5);
	name->set_anchor_and_margin(MARGIN_LEFT, Control::ANCHOR_BEGIN, 5);
	name->set_anchor_and_margin(MARGIN_RIGHT, Control::ANCHOR_END, -5);
	name->connect("gui_input", callable_mp(this, &EditorLayoutsDialog::_line_gui_input));
	name->connect("focus_entered", callable_mp(layout_names, &ItemList::unselect_all));
}

void EditorLayoutsDialog::set_name_line_enabled(bool p_enabled) {
	name->set_visible(p_enabled);
}
