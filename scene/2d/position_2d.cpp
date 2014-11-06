/*************************************************************************/
/*  position_2d.cpp                                                      */
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
#include "position_2d.h"
#include "scene/resources/texture.h"

void Position2D::_draw_cross() {

	draw_line(Point2(-10,0),Point2(+10,0),Color(1,0.5,0.5));
	draw_line(Point2(0,-10),Point2(0,+10),Color(0.5,1,0.5));

}

Rect2 Position2D::get_item_rect() const {

	return Rect2(Point2(-8,-8),Size2(16,16));
}

void Position2D::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {

			update();
		} break;
		case NOTIFICATION_DRAW: {
			if (!is_inside_tree())
				break;
			if (get_tree()->is_editor_hint() || always_show)
				_draw_cross();

		} break;
	}

}

bool Position2D::is_always_show() const {

	return alway_show;
}

void Position2D::set_always_show(bool p_show) {

	alway_show = p_show;
}

void Position2D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_always_show","disable"),&Position2D::set_always_show);
	ObjectTypeDB::bind_method(_MD("is_always_show"),&Position2D::is_always_show);

	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "always_show"),_SCS("set_always_show"),_SCS("is_always_show") );
}

Position2D::Position2D()
{
	alway_show = false;
}
