/*************************************************************************/
/*  reference_frame.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "reference_frame.h"

void ReferenceFrame::_notification(int p_what) {

	if (p_what==NOTIFICATION_DRAW) {

		if (!is_inside_tree())
			return;
		if (get_tree()->is_editor_hint() || always_show) {

			//draw_style_box(get_stylebox("border"),Rect2(Point2(),get_size()),color);
			Size2 size = get_size();
			draw_line(Point2(0, 0), Point2(size.width, 0), color, 1);
			draw_line(Point2(size.width, 0), Point2(size.width, size.height), color, 1);
			draw_line(Point2(size.width, size.height), Point2(0, size.height), color, 1);
			draw_line(Point2(0, size.height), Point2(0, 0), color, 1);
		}
	}
}

void ReferenceFrame::set_color(const Color& p_color) {
	
	if (color == p_color)
		return;
	color = p_color;
	update();
}

Color ReferenceFrame::get_color() const {

	return color;
}

void ReferenceFrame::set_always_show(bool p_show) {

	if (always_show == p_show)
		return;
	always_show = p_show;
	update();
}

bool ReferenceFrame::is_always_show() const {

	return always_show;
}


void ReferenceFrame::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_color","color"), & ReferenceFrame::set_color );
	ObjectTypeDB::bind_method(_MD("get_color"), & ReferenceFrame::get_color );
	ObjectTypeDB::bind_method(_MD("set_always_show","show"), & ReferenceFrame::set_always_show );
	ObjectTypeDB::bind_method(_MD("is_always_show"), & ReferenceFrame::is_always_show );

	ADD_PROPERTY( PropertyInfo( Variant::COLOR, "color"), _SCS("set_color"),_SCS("get_color") );
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "always_show" ), _SCS("set_always_show"),_SCS("is_always_show") );
}

ReferenceFrame::ReferenceFrame()
{
	always_show = false;
	color = Color(1, 0, 0, 1);
}
