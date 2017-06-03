/*************************************************************************/
/*  segment_shape_2d.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "segment_shape_2d.h"

#include "servers/physics_2d_server.h"
#include "servers/visual_server.h"

void SegmentShape2D::_update_shape() {

	Rect2 r;
	r.pos = a;
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
	rect.pos = a;
	rect.expand_to(b);
	return rect;
}

void SegmentShape2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_a", "a"), &SegmentShape2D::set_a);
	ClassDB::bind_method(D_METHOD("get_a"), &SegmentShape2D::get_a);

	ClassDB::bind_method(D_METHOD("set_b", "b"), &SegmentShape2D::set_b);
	ClassDB::bind_method(D_METHOD("get_b"), &SegmentShape2D::get_b);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "a"), "set_a", "get_a");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "b"), "set_b", "get_b");
}

SegmentShape2D::SegmentShape2D()
	: Shape2D(Physics2DServer::get_singleton()->shape_create(Physics2DServer::SHAPE_SEGMENT)) {

	a = Vector2();
	b = Vector2(0, 10);
	_update_shape();
}

////////////////////////////////////////////////////////////

void RayShape2D::_update_shape() {

	Physics2DServer::get_singleton()->shape_set_data(get_rid(), length);
	emit_changed();
}

void RayShape2D::draw(const RID &p_to_rid, const Color &p_color) {

	Vector2 tip = Vector2(0, get_length());
	VS::get_singleton()->canvas_item_add_line(p_to_rid, Vector2(), tip, p_color, 3);
	Vector<Vector2> pts;
	float tsize = 4;
	pts.push_back(tip + Vector2(0, tsize));
	pts.push_back(tip + Vector2(0.707 * tsize, 0));
	pts.push_back(tip + Vector2(-0.707 * tsize, 0));
	Vector<Color> cols;
	for (int i = 0; i < 3; i++)
		cols.push_back(p_color);

	VS::get_singleton()->canvas_item_add_primitive(p_to_rid, pts, cols, Vector<Point2>(), RID());
}

Rect2 RayShape2D::get_rect() const {

	Rect2 rect;
	rect.pos = Vector2();
	rect.expand_to(Vector2(0, length));
	rect = rect.grow(0.707 * 4);
	return rect;
}

void RayShape2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_length", "length"), &RayShape2D::set_length);
	ClassDB::bind_method(D_METHOD("get_length"), &RayShape2D::get_length);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "length"), "set_length", "get_length");
}

void RayShape2D::set_length(real_t p_length) {

	length = p_length;
	_update_shape();
}
real_t RayShape2D::get_length() const {

	return length;
}

RayShape2D::RayShape2D()
	: Shape2D(Physics2DServer::get_singleton()->shape_create(Physics2DServer::SHAPE_RAY)) {

	length = 20;
	_update_shape();
}
