/*************************************************************************/
/*  editor_name_dialog.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor_name_dialog.h"

#include "class_db.h"
#include "os/keyboard.h"

void EditorNameDialog::_line_gui_input(const InputEvent &p_event) {

	if (p_event.type == InputEvent::KEY) {

		if (!p_event.key.pressed)
			return;

		switch (p_event.key.scancode) {
			case KEY_ENTER:
			case KEY_RETURN: {

				if (get_hide_on_ok())
					hide();
				ok_pressed();
				accept_event();
			} break;
			case KEY_ESCAPE: {

				hide();
				accept_event();
			} break;
		}
	}
}

void EditorNameDialog::_post_popup() {

	ConfirmationDialog::_post_popup();
	name->clear();
	name->grab_focus();
}

void EditorNameDialog::ok_pressed() {

	if (name->get_text() != "") {
		emit_signal("name_confirmed", name->get_text());
	}
}

void EditorNameDialog::_bind_methods() {

	ClassDB::bind_method("_line_gui_input", &EditorNameDialog::_line_gui_input);

	ADD_SIGNAL(MethodInfo("name_confirmed", PropertyInfo(Variant::STRING, "name")));
}

EditorNameDialog::EditorNameDialog() {
	name = memnew(LineEdit);
	add_child(name);
	move_child(name, get_label()->get_index() + 1);
	name->set_margin(MARGIN_TOP, 5);
	name->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_BEGIN, 5);
	name->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 5);
	name->connect("gui_input", this, "_line_gui_input");
}
