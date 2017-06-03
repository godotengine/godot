/*************************************************************************/
/*  path_2d.cpp                                                          */
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
#include "path_2d.h"
#include "scene/scene_string_names.h"

void Path2D::_notification(int p_what) {

	if (p_what == NOTIFICATION_DRAW && curve.is_valid()) {
		//draw the curve!!

		if (!get_tree()->is_editor_hint() && !get_tree()->is_debugging_navigation_hint()) {
			return;
		}

		for (int i = 0; i < curve->get_point_count(); i++) {

			Vector2 prev_p = curve->get_point_pos(i);

			for (int j = 1; j <= 8; j++) {

				real_t frac = j / 8.0;
				Vector2 p = curve->interpolate(i, frac);
				draw_line(prev_p, p, Color(0.5, 0.6, 1.0, 0.7), 2);
				prev_p = p;
			}
		}
	}
}

void Path2D::_curve_changed() {

	if (is_inside_tree() && get_tree()->is_editor_hint())
		update();
}

void Path2D::set_curve(const Ref<Curve2D> &p_curve) {

	if (curve.is_valid()) {
		curve->disconnect("changed", this, "_curve_changed");
	}

	curve = p_curve;

	if (curve.is_valid()) {
		curve->connect("changed", this, "_curve_changed");
	}

	_curve_changed();
}

Ref<Curve2D> Path2D::get_curve() const {

	return curve;
}

void Path2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_curve", "curve:Curve2D"), &Path2D::set_curve);
	ClassDB::bind_method(D_METHOD("get_curve:Curve2D", "curve"), &Path2D::get_curve);
	ClassDB::bind_method(D_METHOD("_curve_changed"), &Path2D::_curve_changed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve2D"), "set_curve", "get_curve");
}

Path2D::Path2D() {

	set_curve(Ref<Curve2D>(memnew(Curve2D))); //create one by default
}

/////////////////////////////////////////////////////////////////////////////////

void PathFollow2D::_update_transform() {

	if (!path)
		return;

	Ref<Curve2D> c = path->get_curve();
	if (!c.is_valid())
		return;

	float o = offset;
	if (loop)
		o = Math::fposmod(o, c->get_baked_length());

	Vector2 pos = c->interpolate_baked(o, cubic);

	if (rotate) {

		Vector2 n = (c->interpolate_baked(o + lookahead, cubic) - pos).normalized();
		Vector2 t = -n.tangent();
		pos += n * h_offset;
		pos += t * v_offset;

		set_rotation(t.angle());

	} else {

		pos.x += h_offset;
		pos.y += v_offset;
	}

	set_position(pos);
}

void PathFollow2D::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {

			Node *parent = get_parent();
			if (parent) {

				path = parent->cast_to<Path2D>();
				if (path) {
					_update_transform();
				}
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {

			path = NULL;
		} break;
	}
}

void PathFollow2D::set_cubic_interpolation(bool p_enable) {

	cubic = p_enable;
}

bool PathFollow2D::get_cubic_interpolation() const {

	return cubic;
}

bool PathFollow2D::_set(const StringName &p_name, const Variant &p_value) {

	if (p_name == SceneStringNames::get_singleton()->offset) {
		set_offset(p_value);
	} else if (p_name == SceneStringNames::get_singleton()->unit_offset) {
		set_unit_offset(p_value);
	} else if (p_name == SceneStringNames::get_singleton()->rotate) {
		set_rotate(p_value);
	} else if (p_name == SceneStringNames::get_singleton()->v_offset) {
		set_v_offset(p_value);
	} else if (p_name == SceneStringNames::get_singleton()->h_offset) {
		set_h_offset(p_value);
	} else if (String(p_name) == "cubic_interp") {
		set_cubic_interpolation(p_value);
	} else if (String(p_name) == "loop") {
		set_loop(p_value);
	} else if (String(p_name) == "lookahead") {
		set_lookahead(p_value);
	} else
		return false;

	return true;
}

bool PathFollow2D::_get(const StringName &p_name, Variant &r_ret) const {

	if (p_name == SceneStringNames::get_singleton()->offset) {
		r_ret = get_offset();
	} else if (p_name == SceneStringNames::get_singleton()->unit_offset) {
		r_ret = get_unit_offset();
	} else if (p_name == SceneStringNames::get_singleton()->rotate) {
		r_ret = is_rotating();
	} else if (p_name == SceneStringNames::get_singleton()->v_offset) {
		r_ret = get_v_offset();
	} else if (p_name == SceneStringNames::get_singleton()->h_offset) {
		r_ret = get_h_offset();
	} else if (String(p_name) == "cubic_interp") {
		r_ret = cubic;
	} else if (String(p_name) == "loop") {
		r_ret = loop;
	} else if (String(p_name) == "lookahead") {
		r_ret = lookahead;
	} else
		return false;

	return true;
}
void PathFollow2D::_get_property_list(List<PropertyInfo> *p_list) const {

	float max = 10000;
	if (path && path->get_curve().is_valid())
		max = path->get_curve()->get_baked_length();
	p_list->push_back(PropertyInfo(Variant::REAL, "offset", PROPERTY_HINT_RANGE, "0," + rtos(max) + ",0.01"));
	p_list->push_back(PropertyInfo(Variant::REAL, "unit_offset", PROPERTY_HINT_RANGE, "0,1,0.0001", PROPERTY_USAGE_EDITOR));
	p_list->push_back(PropertyInfo(Variant::REAL, "h_offset"));
	p_list->push_back(PropertyInfo(Variant::REAL, "v_offset"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "rotate"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "cubic_interp"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "loop"));
	p_list->push_back(PropertyInfo(Variant::REAL, "lookahead", PROPERTY_HINT_RANGE, "0.001,1024.0,0.001"));
}

String PathFollow2D::get_configuration_warning() const {

	if (!is_visible_in_tree() || !is_inside_tree())
		return String();

	if (!get_parent() || !get_parent()->cast_to<Path2D>()) {
		return TTR("PathFollow2D only works when set as a child of a Path2D node.");
	}

	return String();
}

void PathFollow2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &PathFollow2D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &PathFollow2D::get_offset);

	ClassDB::bind_method(D_METHOD("set_h_offset", "h_offset"), &PathFollow2D::set_h_offset);
	ClassDB::bind_method(D_METHOD("get_h_offset"), &PathFollow2D::get_h_offset);

	ClassDB::bind_method(D_METHOD("set_v_offset", "v_offset"), &PathFollow2D::set_v_offset);
	ClassDB::bind_method(D_METHOD("get_v_offset"), &PathFollow2D::get_v_offset);

	ClassDB::bind_method(D_METHOD("set_unit_offset", "unit_offset"), &PathFollow2D::set_unit_offset);
	ClassDB::bind_method(D_METHOD("get_unit_offset"), &PathFollow2D::get_unit_offset);

	ClassDB::bind_method(D_METHOD("set_rotate", "enable"), &PathFollow2D::set_rotate);
	ClassDB::bind_method(D_METHOD("is_rotating"), &PathFollow2D::is_rotating);

	ClassDB::bind_method(D_METHOD("set_cubic_interpolation", "enable"), &PathFollow2D::set_cubic_interpolation);
	ClassDB::bind_method(D_METHOD("get_cubic_interpolation"), &PathFollow2D::get_cubic_interpolation);

	ClassDB::bind_method(D_METHOD("set_loop", "loop"), &PathFollow2D::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &PathFollow2D::has_loop);
}

void PathFollow2D::set_offset(float p_offset) {

	offset = p_offset;
	if (path)
		_update_transform();
	_change_notify("offset");
	_change_notify("unit_offset");
}

void PathFollow2D::set_h_offset(float p_h_offset) {

	h_offset = p_h_offset;
	if (path)
		_update_transform();
}

float PathFollow2D::get_h_offset() const {

	return h_offset;
}

void PathFollow2D::set_v_offset(float p_v_offset) {

	v_offset = p_v_offset;
	if (path)
		_update_transform();
}

float PathFollow2D::get_v_offset() const {

	return v_offset;
}

float PathFollow2D::get_offset() const {

	return offset;
}

void PathFollow2D::set_unit_offset(float p_unit_offset) {

	if (path && path->get_curve().is_valid() && path->get_curve()->get_baked_length())
		set_offset(p_unit_offset * path->get_curve()->get_baked_length());
}

float PathFollow2D::get_unit_offset() const {

	if (path && path->get_curve().is_valid() && path->get_curve()->get_baked_length())
		return get_offset() / path->get_curve()->get_baked_length();
	else
		return 0;
}

void PathFollow2D::set_lookahead(float p_lookahead) {

	lookahead = p_lookahead;
}

float PathFollow2D::get_lookahead() const {

	return lookahead;
}

void PathFollow2D::set_rotate(bool p_rotate) {

	rotate = p_rotate;
	_update_transform();
}

bool PathFollow2D::is_rotating() const {

	return rotate;
}

void PathFollow2D::set_loop(bool p_loop) {

	loop = p_loop;
}

bool PathFollow2D::has_loop() const {

	return loop;
}

PathFollow2D::PathFollow2D() {

	offset = 0;
	h_offset = 0;
	v_offset = 0;
	path = NULL;
	rotate = true;
	cubic = true;
	loop = true;
	lookahead = 4;
}
