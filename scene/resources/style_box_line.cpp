/**************************************************************************/
/*  style_box_line.cpp                                                    */
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

#include "style_box_line.h"

#include "servers/rendering_server.h"

float StyleBoxLine::get_style_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0);

	if (vertical) {
		if (p_side == SIDE_LEFT || p_side == SIDE_RIGHT) {
			return thickness / 2.0;
		}
	} else if (p_side == SIDE_TOP || p_side == SIDE_BOTTOM) {
		return thickness / 2.0;
	}

	return 0;
}

void StyleBoxLine::set_color(const Color &p_color) {
	color = p_color;
	emit_changed();
}

Color StyleBoxLine::get_color() const {
	return color;
}

void StyleBoxLine::set_thickness(int p_thickness) {
	thickness = p_thickness;
	emit_changed();
}

int StyleBoxLine::get_thickness() const {
	return thickness;
}

void StyleBoxLine::set_vertical(bool p_vertical) {
	vertical = p_vertical;
	emit_changed();
}

bool StyleBoxLine::is_vertical() const {
	return vertical;
}

void StyleBoxLine::set_grow_end(float p_grow_end) {
	grow_end = p_grow_end;
	emit_changed();
}

float StyleBoxLine::get_grow_end() const {
	return grow_end;
}

void StyleBoxLine::set_grow_begin(float p_grow_begin) {
	grow_begin = p_grow_begin;
	emit_changed();
}

float StyleBoxLine::get_grow_begin() const {
	return grow_begin;
}

void StyleBoxLine::draw(RID p_canvas_item, const Rect2 &p_rect) const {
	RenderingServer *vs = RenderingServer::get_singleton();
	Rect2i r = p_rect;

	if (vertical) {
		r.position.y -= grow_begin;
		r.size.y += (grow_begin + grow_end);
		r.size.x = thickness;
	} else {
		r.position.x -= grow_begin;
		r.size.x += (grow_begin + grow_end);
		r.size.y = thickness;
	}

	vs->canvas_item_add_rect(p_canvas_item, r, color);
}

void StyleBoxLine::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_color", "color"), &StyleBoxLine::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &StyleBoxLine::get_color);
	ClassDB::bind_method(D_METHOD("set_thickness", "thickness"), &StyleBoxLine::set_thickness);
	ClassDB::bind_method(D_METHOD("get_thickness"), &StyleBoxLine::get_thickness);
	ClassDB::bind_method(D_METHOD("set_grow_begin", "offset"), &StyleBoxLine::set_grow_begin);
	ClassDB::bind_method(D_METHOD("get_grow_begin"), &StyleBoxLine::get_grow_begin);
	ClassDB::bind_method(D_METHOD("set_grow_end", "offset"), &StyleBoxLine::set_grow_end);
	ClassDB::bind_method(D_METHOD("get_grow_end"), &StyleBoxLine::get_grow_end);
	ClassDB::bind_method(D_METHOD("set_vertical", "vertical"), &StyleBoxLine::set_vertical);
	ClassDB::bind_method(D_METHOD("is_vertical"), &StyleBoxLine::is_vertical);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "grow_begin", PROPERTY_HINT_RANGE, "-300,300,1,suffix:px"), "set_grow_begin", "get_grow_begin");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "grow_end", PROPERTY_HINT_RANGE, "-300,300,1,suffix:px"), "set_grow_end", "get_grow_end");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "thickness", PROPERTY_HINT_RANGE, "0,100,suffix:px"), "set_thickness", "get_thickness");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vertical"), "set_vertical", "is_vertical");
}

StyleBoxLine::StyleBoxLine() {}

StyleBoxLine::~StyleBoxLine() {}
