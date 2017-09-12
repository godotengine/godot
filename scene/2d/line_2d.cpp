/*************************************************************************/
/*  line_2d.cpp                                                          */
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
#include "line_2d.h"
#include "line_builder.h"

#include "core_string_names.h"

// Needed so we can bind functions
VARIANT_ENUM_CAST(Line2D::LineJointMode)
VARIANT_ENUM_CAST(Line2D::LineCapMode)
VARIANT_ENUM_CAST(Line2D::LineTextureMode)

Line2D::Line2D()
	: Node2D() {
	_joint_mode = LINE_JOINT_SHARP;
	_begin_cap_mode = LINE_CAP_NONE;
	_end_cap_mode = LINE_CAP_NONE;
	_width = 10;
	_default_color = Color(0.4, 0.5, 1);
	_sharp_limit = 2.f;
	_round_precision = 8;
}

void Line2D::set_points(const PoolVector<Vector2> &p_points) {
	_points = p_points;
	update();
}

void Line2D::set_width(float width) {
	if (width < 0.0)
		width = 0.0;
	_width = width;
	update();
}

float Line2D::get_width() const {
	return _width;
}

PoolVector<Vector2> Line2D::get_points() const {
	return _points;
}

void Line2D::set_point_pos(int i, Vector2 pos) {
	_points.set(i, pos);
	update();
}

Vector2 Line2D::get_point_pos(int i) const {
	return _points.get(i);
}

int Line2D::get_point_count() const {
	return _points.size();
}

void Line2D::add_point(Vector2 pos) {
	_points.append(pos);
	update();
}

void Line2D::remove_point(int i) {
	_points.remove(i);
	update();
}

void Line2D::set_default_color(Color color) {
	_default_color = color;
	update();
}

Color Line2D::get_default_color() const {
	return _default_color;
}

void Line2D::set_gradient(const Ref<Gradient> &gradient) {

	// Cleanup previous connection if any
	if (_gradient.is_valid()) {
		(**_gradient).disconnect(CoreStringNames::get_singleton()->changed, this, "_gradient_changed");
	}

	_gradient = gradient;

	// Connect to the gradient so the line will update when the ColorRamp is changed
	if (_gradient.is_valid()) {
		(**_gradient).connect(CoreStringNames::get_singleton()->changed, this, "_gradient_changed");
	}

	update();
}

Ref<Gradient> Line2D::get_gradient() const {
	return _gradient;
}

void Line2D::set_texture(const Ref<Texture> &texture) {
	_texture = texture;
	update();
}

Ref<Texture> Line2D::get_texture() const {
	return _texture;
}

void Line2D::set_texture_mode(const LineTextureMode mode) {
	_texture_mode = mode;
	update();
}

Line2D::LineTextureMode Line2D::get_texture_mode() const {
	return _texture_mode;
}

void Line2D::set_joint_mode(LineJointMode mode) {
	_joint_mode = mode;
	update();
}

Line2D::LineJointMode Line2D::get_joint_mode() const {
	return _joint_mode;
}

void Line2D::set_begin_cap_mode(LineCapMode mode) {
	_begin_cap_mode = mode;
	update();
}

Line2D::LineCapMode Line2D::get_begin_cap_mode() const {
	return _begin_cap_mode;
}

void Line2D::set_end_cap_mode(LineCapMode mode) {
	_end_cap_mode = mode;
	update();
}

Line2D::LineCapMode Line2D::get_end_cap_mode() const {
	return _end_cap_mode;
}

void Line2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW:
			_draw();
			break;
	}
}

void Line2D::set_sharp_limit(float limit) {
	if (limit < 0.f)
		limit = 0.f;
	_sharp_limit = limit;
	update();
}

float Line2D::get_sharp_limit() const {
	return _sharp_limit;
}

void Line2D::set_round_precision(int precision) {
	if (precision < 1)
		precision = 1;
	_round_precision = precision;
	update();
}

int Line2D::get_round_precision() const {
	return _round_precision;
}

void Line2D::_draw() {
	if (_points.size() <= 1 || _width == 0.f)
		return;

	// TODO Is this really needed?
	// Copy points for faster access
	Vector<Vector2> points;
	points.resize(_points.size());
	int len = points.size();
	{
		PoolVector<Vector2>::Read points_read = _points.read();
		for (int i = 0; i < len; ++i) {
			points[i] = points_read[i];
		}
	}

	// TODO Maybe have it as member rather than copying parameters and allocating memory?
	LineBuilder lb;
	lb.points = points;
	lb.default_color = _default_color;
	lb.gradient = *_gradient;
	lb.texture_mode = _texture_mode;
	lb.joint_mode = _joint_mode;
	lb.begin_cap_mode = _begin_cap_mode;
	lb.end_cap_mode = _end_cap_mode;
	lb.round_precision = _round_precision;
	lb.sharp_limit = _sharp_limit;
	lb.width = _width;

	lb.build();

	RID texture_rid;
	if (_texture.is_valid())
		texture_rid = (**_texture).get_rid();

	VS::get_singleton()->canvas_item_add_triangle_array(
			get_canvas_item(),
			lb.indices,
			lb.vertices,
			lb.colors,
			lb.uvs,
			texture_rid);

	// DEBUG
	// Draw wireframe
	//	if(lb.indices.size() % 3 == 0) {
	//		Color col(0,0,0);
	//		for(int i = 0; i < lb.indices.size(); i += 3) {
	//			int vi = lb.indices[i];
	//			int lbvsize = lb.vertices.size();
	//			Vector2 a = lb.vertices[lb.indices[i]];
	//			Vector2 b = lb.vertices[lb.indices[i+1]];
	//			Vector2 c = lb.vertices[lb.indices[i+2]];
	//			draw_line(a, b, col);
	//			draw_line(b, c, col);
	//			draw_line(c, a, col);
	//		}
	//		for(int i = 0; i < lb.vertices.size(); ++i) {
	//			Vector2 p = lb.vertices[i];
	//			draw_rect(Rect2(p.x-1, p.y-1, 2, 2), Color(0,0,0,0.5));
	//		}
	//	}
}

void Line2D::_gradient_changed() {
	update();
}

// static
void Line2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_points", "points"), &Line2D::set_points);
	ClassDB::bind_method(D_METHOD("get_points"), &Line2D::get_points);

	ClassDB::bind_method(D_METHOD("set_point_pos", "i", "pos"), &Line2D::set_point_pos);
	ClassDB::bind_method(D_METHOD("get_point_pos", "i"), &Line2D::get_point_pos);

	ClassDB::bind_method(D_METHOD("get_point_count"), &Line2D::get_point_count);

	ClassDB::bind_method(D_METHOD("add_point", "pos"), &Line2D::add_point);
	ClassDB::bind_method(D_METHOD("remove_point", "i"), &Line2D::remove_point);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &Line2D::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &Line2D::get_width);

	ClassDB::bind_method(D_METHOD("set_default_color", "color"), &Line2D::set_default_color);
	ClassDB::bind_method(D_METHOD("get_default_color"), &Line2D::get_default_color);

	ClassDB::bind_method(D_METHOD("set_gradient", "color"), &Line2D::set_gradient);
	ClassDB::bind_method(D_METHOD("get_gradient"), &Line2D::get_gradient);

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &Line2D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &Line2D::get_texture);

	ClassDB::bind_method(D_METHOD("set_texture_mode", "mode"), &Line2D::set_texture_mode);
	ClassDB::bind_method(D_METHOD("get_texture_mode"), &Line2D::get_texture_mode);

	ClassDB::bind_method(D_METHOD("set_joint_mode", "mode"), &Line2D::set_joint_mode);
	ClassDB::bind_method(D_METHOD("get_joint_mode"), &Line2D::get_joint_mode);

	ClassDB::bind_method(D_METHOD("set_begin_cap_mode", "mode"), &Line2D::set_begin_cap_mode);
	ClassDB::bind_method(D_METHOD("get_begin_cap_mode"), &Line2D::get_begin_cap_mode);

	ClassDB::bind_method(D_METHOD("set_end_cap_mode", "mode"), &Line2D::set_end_cap_mode);
	ClassDB::bind_method(D_METHOD("get_end_cap_mode"), &Line2D::get_end_cap_mode);

	ClassDB::bind_method(D_METHOD("set_sharp_limit", "limit"), &Line2D::set_sharp_limit);
	ClassDB::bind_method(D_METHOD("get_sharp_limit"), &Line2D::get_sharp_limit);

	ClassDB::bind_method(D_METHOD("set_round_precision", "precision"), &Line2D::set_round_precision);
	ClassDB::bind_method(D_METHOD("get_round_precision"), &Line2D::get_round_precision);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR2_ARRAY, "points"), "set_points", "get_points");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "width"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "default_color"), "set_default_color", "get_default_color");
	ADD_GROUP("Fill", "");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "gradient", PROPERTY_HINT_RESOURCE_TYPE, "Gradient"), "set_gradient", "get_gradient");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_PROPERTYNZ(PropertyInfo(Variant::INT, "texture_mode", PROPERTY_HINT_ENUM, "None,Tile"), "set_texture_mode", "get_texture_mode");
	ADD_GROUP("Capping", "");
	ADD_PROPERTYNZ(PropertyInfo(Variant::INT, "joint_mode", PROPERTY_HINT_ENUM, "Sharp,Bevel,Round"), "set_joint_mode", "get_joint_mode");
	ADD_PROPERTYNZ(PropertyInfo(Variant::INT, "begin_cap_mode", PROPERTY_HINT_ENUM, "None,Box,Round"), "set_begin_cap_mode", "get_begin_cap_mode");
	ADD_PROPERTYNZ(PropertyInfo(Variant::INT, "end_cap_mode", PROPERTY_HINT_ENUM, "None,Box,Round"), "set_end_cap_mode", "get_end_cap_mode");
	ADD_GROUP("Border", "");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "sharp_limit"), "set_sharp_limit", "get_sharp_limit");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "round_precision"), "set_round_precision", "get_round_precision");

	BIND_ENUM_CONSTANT(LINE_JOINT_SHARP);
	BIND_ENUM_CONSTANT(LINE_JOINT_BEVEL);
	BIND_ENUM_CONSTANT(LINE_JOINT_ROUND);

	BIND_ENUM_CONSTANT(LINE_CAP_NONE);
	BIND_ENUM_CONSTANT(LINE_CAP_BOX);
	BIND_ENUM_CONSTANT(LINE_CAP_ROUND);

	BIND_ENUM_CONSTANT(LINE_TEXTURE_NONE);
	BIND_ENUM_CONSTANT(LINE_TEXTURE_TILE);

	ClassDB::bind_method(D_METHOD("_gradient_changed"), &Line2D::_gradient_changed);
}
