/*************************************************************************/
/*  segment_shape_2d.cpp                                                 */
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

#include "segment_shape_2d.h"

#include "servers/physics_2d_server.h"
#include "servers/visual_server.h"

bool SegmentShape2D::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	Vector2 l[2] = { a, b };
	Vector2 closest = Geometry::get_closest_point_to_segment_2d(p_point, l);
	return p_point.distance_to(closest) < p_tolerance;
}

void SegmentShape2D::_update_shape() {
	Rect2 r;
	r.position = a;
	r.size = b;
	Physics2DServer::get_singleton()->shape_set_data(get_rid(), r);
	emit_changed();
}

void SegmentShape2D::set_a(const Vector2 &p_a) {
	a = p_a;
	_update_shape();
}
Vector2 SegmentShape2D::get_a() const {
	return a;
}

void SegmentShape2D::set_b(const Vector2 &p_b) {
	b = p_b;
	_update_shape();
}
Vector2 SegmentShape2D::get_b() const {
	return b;
}

void SegmentShape2D::draw(const RID &p_to_rid, const Color &p_color) {
	VisualServer::get_singleton()->canvas_item_add_line(p_to_rid, a, b, p_color, 3);
}

Rect2 SegmentShape2D::get_rect() const {
	Rect2 rect;
	rect.position = a;
	rect.expand_to(b);
	return rect;
}

real_t SegmentShape2D::get_enclosing_radius() const {
	return (a + b).length();
}

void SegmentShape2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_a", "a"), &SegmentShape2D::set_a);
	ClassDB::bind_method(D_METHOD("get_a"), &SegmentShape2D::get_a);

	ClassDB::bind_method(D_METHOD("set_b", "b"), &SegmentShape2D::set_b);
	ClassDB::bind_method(D_METHOD("get_b"), &SegmentShape2D::get_b);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "a"), "set_a", "get_a");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "b"), "set_b", "get_b");
}

SegmentShape2D::SegmentShape2D() :
		Shape2D(Physics2DServer::get_singleton()->segment_shape_create()) {
	a = Vector2();
	b = Vector2(0, 10);
	_update_shape();
}

////////////////////////////////////////////////////////////

void RayShape2D::_update_shape() {
	Dictionary d;
	d["length"] = length;
	d["slips_on_slope"] = slips_on_slope;
	Physics2DServer::get_singleton()->shape_set_data(get_rid(), d);
	emit_changed();
}

void RayShape2D::draw(const RID &p_to_rid, const Color &p_color) {
	Vector2 tip = Vector2(0, get_length());
	VS::get_singleton()->canvas_item_add_line(p_to_rid, Vector2(), tip, p_color, 3);
	Vector<Vector2> pts;
	float tsize = 4;
	pts.push_back(tip + Vector2(0, tsize));
	pts.push_back(tip + Vector2(Math_SQRT12 * tsize, 0));
	pts.push_back(tip + Vector2(-Math_SQRT12 * tsize, 0));
	Vector<Color> cols;
	for (int i = 0; i < 3; i++) {
		cols.push_back(p_color);
	}

	VS::get_singleton()->canvas_item_add_primitive(p_to_rid, pts, cols, Vector<Point2>(), RID());
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

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "length"), "set_length", "get_length");
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
		Shape2D(Physics2DServer::get_singleton()->ray_shape_create()) {
	length = 20;
	slips_on_slope = false;
	_update_shape();
}
