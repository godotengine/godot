/**************************************************************************/
/*  helper_rect_2d.cpp                                                    */
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

#include "helper_rect_2d.h"

Dictionary HelperRect2D::_edit_get_state() const {
	Dictionary ret = Node2D::_edit_get_state();
	ret["rect"] = rect;
	return ret;
}

void HelperRect2D::_edit_set_state(const Dictionary &p_state) {
	set_rect(p_state["rect"]);
	Node2D::_edit_set_state(p_state);
}

void HelperRect2D::_edit_set_rect(const Rect2 &p_edit_rect) {
	dragged = true;
	set_rect(p_edit_rect);
	dragged = false;
}

void HelperRect2D::_edit_set_pivot(const Point2 &p_pivot) {
	dragged = true;

	set_position(get_transform().xform(p_pivot));

	Rect2 tmp = rect;
	tmp.position -= p_pivot;
	set_rect(tmp);

	dragged = false;
}

bool HelperRect2D::_edit_use_pivot() const {
	return true;
}

Rect2 HelperRect2D::_edit_get_rect() const {
	return rect;
}

void HelperRect2D::set_rect(const Rect2 &p_rect) {
	rect = p_rect;
	GDVIRTUAL_CALL(_rect_updated);
	item_rect_changed();
	queue_redraw();
}

Rect2 HelperRect2D::get_rect() const {
	return rect;
}

bool HelperRect2D::_edit_use_rect() const {
	return true;
}

void HelperRect2D::set_border_color(const Color &p_border_color) {
	border_color = p_border_color;
	queue_redraw();
}

Color HelperRect2D::get_border_color() const {
	return border_color;
}

bool HelperRect2D::is_being_dragged() const {
	return dragged;
}

void HelperRect2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			Rect2 drawn_rect = rect;
			drawn_rect.size *= get_global_scale();

			draw_set_transform(-get_global_position(), 0.0, Vector2(1.0, 1.0) / get_global_scale());
			draw_rect(drawn_rect, border_color, false, 2.0);

			// Restore the drawing transform to ensure `_draw()` work as expected
			draw_set_transform(Vector2(), 0.0, Vector2(1.0, 1.0));
		} break;
	}
}

void HelperRect2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_rect", "rect"), &HelperRect2D::set_rect);
	ClassDB::bind_method(D_METHOD("set_border_color", "border_color"), &HelperRect2D::set_border_color);

	ClassDB::bind_method(D_METHOD("get_rect"), &HelperRect2D::get_rect);
	ClassDB::bind_method(D_METHOD("get_border_color"), &HelperRect2D::get_border_color);

	ClassDB::bind_method(D_METHOD("is_being_dragged"), &HelperRect2D::is_being_dragged);

	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "rect", PROPERTY_HINT_NONE, "suffix:px"), "set_rect", "get_rect");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_color", PROPERTY_HINT_NONE, ""), "set_border_color", "get_border_color");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dragged", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "is_being_dragged");

	GDVIRTUAL_BIND(_rect_updated);
}
