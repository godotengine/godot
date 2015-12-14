/*************************************************************************/
/*  editor_layout_dialog.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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

#include "editor_layout_dialog.h"
#include "object_type_db.h"

void EditorLayoutDialog::clear_layout_name() {

	layout_name->clear();
}

void EditorLayoutDialog::_post_popup() {

	ConfirmationDialog::_post_popup();
	layout_name->grab_focus();
}

void EditorLayoutDialog::ok_pressed() {

	if (layout_name->get_text()!="") {
		emit_signal("layout_selected", layout_name->get_text());
	}
}

void EditorLayoutDialog::_bind_methods() {

	ADD_SIGNAL(MethodInfo("layout_selected",PropertyInfo( Variant::STRING,"layout_name")));
}

EditorLayoutDialog::EditorLayoutDialog()
{

	layout_name = memnew( LineEdit );
	layout_name->set_margin(MARGIN_TOP,5);
	layout_name->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,5);
	layout_name->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,5);
	add_child(layout_name);
	move_child(layout_name, get_label()->get_index()+1);
}
