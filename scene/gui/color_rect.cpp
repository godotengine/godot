/*************************************************************************/
/*  color_rect.cpp                                                       */
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

#include "color_rect.h"

void ColorFrame::set_frame_color(const Color &p_color) {

	color = p_color;
	update();
}

Color ColorFrame::get_frame_color() const {

	return color;
}

void ColorFrame::_notification(int p_what) {

	if (p_what == NOTIFICATION_DRAW) {
		draw_rect(Rect2(Point2(), get_size()), color);
	}
}

void ColorFrame::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_frame_color", "color"), &ColorFrame::set_frame_color);
	ObjectTypeDB::bind_method(_MD("get_frame_color"), &ColorFrame::get_frame_color);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), _SCS("set_frame_color"), _SCS("get_frame_color"));
}

ColorFrame::ColorFrame() {

	color = Color(1, 1, 1);
}
