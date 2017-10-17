/*************************************************************************/
/*  check_button.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "check_box.h"

#include "servers/visual_server.h"

void CheckBox::_notification(int p_what) {

	if (p_what == NOTIFICATION_DRAW) {

		RID ci = get_canvas_item();

		Ref<Texture> on = Control::get_icon(is_radio() ? "radio_checked" : "checked");
		Ref<Texture> off = Control::get_icon(is_radio() ? "radio_unchecked" : "unchecked");

		Vector2 ofs;
		ofs.x = 0;
		ofs.y = int((get_size().height - on->get_height()) / 2);

		if (is_pressed())
			on->draw(ci, ofs);
		else
			off->draw(ci, ofs);
	}
}

bool CheckBox::is_radio() {

	return get_button_group().is_valid();
}

CheckBox::CheckBox(const String &p_text)
	: Button(p_text) {
	set_toggle_mode(true);
	set_text_align(ALIGN_LEFT);
}

CheckBox::~CheckBox() {
}
