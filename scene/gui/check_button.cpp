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
#include "check_button.h"

#include "print_string.h"
#include "servers/visual_server.h"

Size2 CheckButton::get_minimum_size() const {

	Size2 minsize = Button::get_minimum_size();

	Ref<Texture> on = Control::get_icon("on");
	Ref<Texture> off = Control::get_icon("off");
	Size2 tex_size = Size2(0, 0);
	if (!on.is_null())
		tex_size = Size2(on->get_width(), on->get_height());
	if (!off.is_null())
		tex_size = Size2(MAX(tex_size.width, off->get_width()), MAX(tex_size.height, off->get_height()));
	minsize += Size2(tex_size.width + get_constant("hseparation"), 0);
	minsize.height = MAX(minsize.height, tex_size.height);

	return get_stylebox("normal")->get_minimum_size() + minsize;
}

void CheckButton::_notification(int p_what) {

	if (p_what == NOTIFICATION_DRAW) {

		RID ci = get_canvas_item();

		Ref<Texture> on = Control::get_icon("on");
		Ref<Texture> off = Control::get_icon("off");

		Ref<StyleBox> sb = get_stylebox("normal");
		Size2 sb_ofs = Size2(sb->get_margin(MARGIN_RIGHT), sb->get_margin(MARGIN_TOP));
		Vector2 ofs;
		ofs.x = get_minimum_size().width - (on->get_width() + sb_ofs.width);
		ofs.y = sb_ofs.height;

		if (is_pressed())
			on->draw(ci, ofs);
		else
			off->draw(ci, ofs);
	}
}

CheckButton::CheckButton() {

	set_toggle_mode(true);
	set_text_align(ALIGN_LEFT);
}

CheckButton::~CheckButton() {
}
