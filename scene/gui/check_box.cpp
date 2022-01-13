/*************************************************************************/
/*  check_box.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

Size2 CheckBox::get_icon_size() const {
	Ref<Texture> checked = Control::get_icon("checked");
	Ref<Texture> checked_disabled = Control::get_icon("checked_disabled");
	Ref<Texture> unchecked = Control::get_icon("unchecked");
	Ref<Texture> unchecked_disabled = Control::get_icon("unchecked_disabled");
	Ref<Texture> radio_checked = Control::get_icon("radio_checked");
	Ref<Texture> radio_unchecked = Control::get_icon("radio_unchecked");

	Size2 tex_size = Size2(0, 0);
	if (!checked.is_null()) {
		tex_size = Size2(checked->get_width(), checked->get_height());
	}
	if (!unchecked.is_null()) {
		tex_size = Size2(MAX(tex_size.width, unchecked->get_width()), MAX(tex_size.height, unchecked->get_height()));
	}
	if (!radio_checked.is_null()) {
		tex_size = Size2(MAX(tex_size.width, radio_checked->get_width()), MAX(tex_size.height, radio_checked->get_height()));
	}
	if (!radio_unchecked.is_null()) {
		tex_size = Size2(MAX(tex_size.width, radio_unchecked->get_width()), MAX(tex_size.height, radio_unchecked->get_height()));
	}
	return tex_size;
}

Size2 CheckBox::get_minimum_size() const {
	Size2 minsize = Button::get_minimum_size();
	Size2 tex_size = get_icon_size();
	minsize.width += tex_size.width;
	if (get_text().length() > 0) {
		minsize.width += get_constant("hseparation");
	}
	Ref<StyleBox> sb = get_stylebox("normal");
	minsize.height = MAX(minsize.height, tex_size.height + sb->get_margin(MARGIN_TOP) + sb->get_margin(MARGIN_BOTTOM));

	return minsize;
}

void CheckBox::_notification(int p_what) {
	if (p_what == NOTIFICATION_THEME_CHANGED) {
		_set_internal_margin(MARGIN_LEFT, get_icon_size().width);
	} else if (p_what == NOTIFICATION_DRAW) {
		RID ci = get_canvas_item();

		Ref<Texture> on = Control::get_icon(vformat("%s%s", is_radio() ? "radio_checked" : "checked", is_disabled() ? "_disabled" : ""));
		Ref<Texture> off = Control::get_icon(vformat("%s%s", is_radio() ? "radio_unchecked" : "unchecked", is_disabled() ? "_disabled" : ""));
		Ref<StyleBox> sb = get_stylebox("normal");

		Vector2 ofs;
		ofs.x = sb->get_margin(MARGIN_LEFT);
		ofs.y = int((get_size().height - get_icon_size().height) / 2) + get_constant("check_vadjust");

		if (is_pressed()) {
			on->draw(ci, ofs);
		} else {
			off->draw(ci, ofs);
		}
	}
}

bool CheckBox::is_radio() {
	return get_button_group().is_valid();
}

CheckBox::CheckBox(const String &p_text) :
		Button(p_text) {
	set_toggle_mode(true);
	set_text_align(ALIGN_LEFT);
	_set_internal_margin(MARGIN_LEFT, get_icon_size().width);
}

CheckBox::~CheckBox() {
}
