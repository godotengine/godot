/**************************************************************************/
/*  pane_drag.cpp                                                         */
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

#include "pane_drag.h"

void PaneDrag::_gui_input(const Ref<InputEvent> &p_input) {
	Ref<InputEventMouseMotion> mm = p_input;
	if (mm.is_valid() && mm->get_button_mask() & BUTTON_MASK_LEFT) {
		emit_signal("dragged", Point2(mm->get_relative().x, mm->get_relative().y));
	}
}

void PaneDrag::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			Ref<Texture> icon = mouse_over ? get_icon("PaneDragHover", "EditorIcons") : get_icon("PaneDrag", "EditorIcons");
			if (!icon.is_null()) {
				icon->draw(get_canvas_item(), Point2(0, 0));
			}

		} break;
		case NOTIFICATION_MOUSE_ENTER:
			mouse_over = true;
			update();
			break;
		case NOTIFICATION_MOUSE_EXIT:
			mouse_over = false;
			update();
			break;
	}
}
Size2 PaneDrag::get_minimum_size() const {
	Ref<Texture> icon = get_icon("PaneDrag", "EditorIcons");
	if (!icon.is_null()) {
		return icon->get_size();
	}
	return Size2();
}

void PaneDrag::_bind_methods() {
	ClassDB::bind_method("_gui_input", &PaneDrag::_gui_input);
	ADD_SIGNAL(MethodInfo("dragged", PropertyInfo(Variant::VECTOR2, "amount")));
}

PaneDrag::PaneDrag() {
	mouse_over = false;
}
