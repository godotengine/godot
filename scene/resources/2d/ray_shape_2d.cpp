/*************************************************************************/
/*  ray_shape_2d.cpp                                                     */
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

#include "ray_shape_2d.h"

#include "servers/physics_server_2d.h"
#include "servers/rendering_server.h"

void RayShape2D::_update_shape() {
	Dictionary d;
	d["length"] = length;
	d["slips_on_slope"] = slips_on_slope;
	PhysicsServer2D::get_singleton()->shape_set_data(get_rid(), d);
	emit_changed();
}

void RayShape2D::draw(const RID &p_to_rid, const Color &p_color) {
	Vector2 tip = Vector2(0, get_length());
	RS::get_singleton()->canvas_item_add_line(p_to_rid, Vector2(), tip, p_color, 3);
	Vector<Vector2> pts;
	float tsize = 4;
	pts.push_back(tip + Vector2(0, tsize));
	pts.push_back(tip + Vector2(Math_SQRT12 * tsize, 0));
	pts.push_back(tip + Vector2(-Math_SQRT12 * tsize, 0));
	Vector<Color> cols;
	for (int i = 0; i < 3; i++) {
		cols.push_back(p_color);
	}
	RS::get_singleton()->canvas_item_add_primitive(p_to_rid, pts, cols, Vector<Point2>(), RID());
}

Rect2 RayShape2D::get_rect() const {
	Rect2 rect;
	rect.position = Vector2();
	rect.expand_to(Vector2(0, length));
	rect = rect.grow(Math_SQRT12 * 4);
	return rect;
}

real_t RayShape2D::get_enclosing_radius() const {
	return length;
}

void RayShape2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_length", "length"), &RayShape2D::set_length);
	ClassDB::bind_method(D_METHOD("get_length"), &RayShape2D::get_length);

	ClassDB::bind_method(D_METHOD("set_slips_on_slope", "active"), &RayShape2D::set_slips_on_slope);
	ClassDB::bind_method(D_METHOD("get_slips_on_slope"), &RayShape2D::get_slips_on_slope);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length"), "set_length", "get_length");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "slips_on_slope"), "set_slips_on_slope", "get_slips_on_slope");
}

void RayShape2D::set_length(real_t p_length) {
	length = p_length;
	_update_shape();
}

real_t RayShape2D::get_length() const {
	return length;
}

void RayShape2D::set_slips_on_slope(bool p_active) {
	slips_on_slope = p_active;
	_update_shape();
}

bool RayShape2D::get_slips_on_slope() const {
	return slips_on_slope;
}

RayShape2D::RayShape2D() :
		Shape2D(PhysicsServer2D::get_singleton()->ray_shape_create()) {
	length = 20;
	slips_on_slope = false;
	_update_shape();
}
