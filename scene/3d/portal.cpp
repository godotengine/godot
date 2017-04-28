/*************************************************************************/
/*  portal.cpp                                                           */
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
#include "portal.h"
#include "global_config.h"
#include "scene/resources/surface_tool.h"
#include "servers/visual_server.h"

bool Portal::_set(const StringName &p_name, const Variant &p_value) {

	if (p_name == "shape") {
		PoolVector<float> src_coords = p_value;
		Vector<Point2> points;
		int src_coords_size = src_coords.size();
		ERR_FAIL_COND_V(src_coords_size % 2, false);
		points.resize(src_coords_size / 2);
		for (int i = 0; i < points.size(); i++) {

			points[i].x = src_coords[i * 2 + 0];
			points[i].y = src_coords[i * 2 + 1];
			set_shape(points);
		}
	} else if (p_name == "enabled") {
		set_enabled(p_value);
	} else if (p_name == "disable_distance") {
		set_disable_distance(p_value);
	} else if (p_name == "disabled_color") {
		set_disabled_color(p_value);
	} else if (p_name == "connect_range") {
		set_connect_range(p_value);
	} else
		return false;

	return true;
}

bool Portal::_get(const StringName &p_name, Variant &r_ret) const {

	if (p_name == "shape") {
		Vector<Point2> points = get_shape();
		PoolVector<float> dst_coords;
		dst_coords.resize(points.size() * 2);

		for (int i = 0; i < points.size(); i++) {

			dst_coords.set(i * 2 + 0, points[i].x);
			dst_coords.set(i * 2 + 1, points[i].y);
		}

		r_ret = dst_coords;
	} else if (p_name == "enabled") {
		r_ret = is_enabled();
	} else if (p_name == "disable_distance") {
		r_ret = get_disable_distance();
	} else if (p_name == "disabled_color") {
		r_ret = get_disabled_color();
	} else if (p_name == "connect_range") {
		r_ret = get_connect_range();
	} else
		return false;
	return true;
}

void Portal::_get_property_list(List<PropertyInfo> *p_list) const {

	p_list->push_back(PropertyInfo(Variant::POOL_REAL_ARRAY, "shape"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "enabled"));
	p_list->push_back(PropertyInfo(Variant::REAL, "disable_distance", PROPERTY_HINT_RANGE, "0,4096,0.01"));
	p_list->push_back(PropertyInfo(Variant::COLOR, "disabled_color"));
	p_list->push_back(PropertyInfo(Variant::REAL, "connect_range", PROPERTY_HINT_RANGE, "0.1,4096,0.01"));
}

Rect3 Portal::get_aabb() const {

	return aabb;
}
PoolVector<Face3> Portal::get_faces(uint32_t p_usage_flags) const {

	if (!(p_usage_flags & FACES_ENCLOSING))
		return PoolVector<Face3>();

	Vector<Point2> shape = get_shape();
	if (shape.size() == 0)
		return PoolVector<Face3>();

	Vector2 center;
	for (int i = 0; i < shape.size(); i++) {

		center += shape[i];
	}

	PoolVector<Face3> ret;
	center /= shape.size();

	for (int i = 0; i < shape.size(); i++) {

		int n = (i + 1) % shape.size();

		Face3 f;
		f.vertex[0] = Vector3(center.x, center.y, 0);
		f.vertex[1] = Vector3(shape[i].x, shape[i].y, 0);
		f.vertex[2] = Vector3(shape[n].x, shape[n].y, 0);
		ret.push_back(f);
	}

	return ret;
}

void Portal::set_shape(const Vector<Point2> &p_shape) {

	VisualServer::get_singleton()->portal_set_shape(portal, p_shape);
	shape = p_shape;
	update_gizmo();
}

Vector<Point2> Portal::get_shape() const {

	return shape;
}

void Portal::set_connect_range(float p_range) {

	connect_range = p_range;
	//VisualServer::get_singleton()->portal_set_connect_range(portal,p_range);
}

float Portal::get_connect_range() const {

	return connect_range;
}

void Portal::set_enabled(bool p_enabled) {

	enabled = p_enabled;
	VisualServer::get_singleton()->portal_set_enabled(portal, enabled);
}

bool Portal::is_enabled() const {

	return enabled;
}

void Portal::set_disable_distance(float p_distance) {

	disable_distance = p_distance;
	VisualServer::get_singleton()->portal_set_disable_distance(portal, disable_distance);
}
float Portal::get_disable_distance() const {

	return disable_distance;
}

void Portal::set_disabled_color(const Color &p_disabled_color) {

	disabled_color = p_disabled_color;
	VisualServer::get_singleton()->portal_set_disabled_color(portal, disabled_color);
}

Color Portal::get_disabled_color() const {

	return disabled_color;
}

void Portal::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_shape", "points"), &Portal::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &Portal::get_shape);

	ClassDB::bind_method(D_METHOD("set_enabled", "enable"), &Portal::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &Portal::is_enabled);

	ClassDB::bind_method(D_METHOD("set_disable_distance", "distance"), &Portal::set_disable_distance);
	ClassDB::bind_method(D_METHOD("get_disable_distance"), &Portal::get_disable_distance);

	ClassDB::bind_method(D_METHOD("set_disabled_color", "color"), &Portal::set_disabled_color);
	ClassDB::bind_method(D_METHOD("get_disabled_color"), &Portal::get_disabled_color);

	ClassDB::bind_method(D_METHOD("set_connect_range", "range"), &Portal::set_connect_range);
	ClassDB::bind_method(D_METHOD("get_connect_range"), &Portal::get_connect_range);
}

Portal::Portal() {

	portal = VisualServer::get_singleton()->portal_create();
	Vector<Point2> points;
	points.push_back(Point2(-1, 1));
	points.push_back(Point2(1, 1));
	points.push_back(Point2(1, -1));
	points.push_back(Point2(-1, -1));
	set_shape(points); // default shape

	set_connect_range(0.8);
	set_disable_distance(50);
	set_enabled(true);

	set_base(portal);
}

Portal::~Portal() {

	VisualServer::get_singleton()->free(portal);
}
