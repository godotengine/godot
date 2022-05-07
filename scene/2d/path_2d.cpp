/*************************************************************************/
/*  path_2d.cpp                                                          */
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

#include "path_2d.h"

#include "core/engine.h"
#include "scene/scene_string_names.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_scale.h"
#endif

#ifdef TOOLS_ENABLED
Rect2 Path2D::_edit_get_rect() const {
	if (!curve.is_valid() || curve->get_point_count() == 0) {
		return Rect2(0, 0, 0, 0);
	}

	Rect2 aabb = Rect2(curve->get_point_position(0), Vector2(0, 0));

	for (int i = 0; i < curve->get_point_count(); i++) {
		for (int j = 0; j <= 8; j++) {
			real_t frac = j / 8.0;
			Vector2 p = curve->interpolate(i, frac);
			aabb.expand_to(p);
		}
	}

	return aabb;
}

bool Path2D::_edit_use_rect() const {
	return curve.is_valid() && curve->get_point_count() != 0;
}

bool Path2D::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	if (curve.is_null()) {
		return false;
	}

	for (int i = 0; i < curve->get_point_count(); i++) {
		Vector2 s[2];
		s[0] = curve->get_point_position(i);

		for (int j = 1; j <= 8; j++) {
			real_t frac = j / 8.0;
			s[1] = curve->interpolate(i, frac);

			Vector2 p = Geometry::get_closest_point_to_segment_2d(p_point, s);
			if (p.distance_to(p_point) <= p_tolerance) {
				return true;
			}

			s[0] = s[1];
		}
	}

	return false;
}
#endif

void Path2D::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW && curve.is_valid()) {
		//draw the curve!!

		if (!Engine::get_singleton()->is_editor_hint() && !get_tree()->is_debugging_navigation_hint()) {
			return;
		}

		if (curve->get_point_count() < 2) {
			return;
		}

#ifdef TOOLS_ENABLED
		const float line_width = 2 * EDSCALE;
#else
		const float line_width = 2;
#endif
		const Color color = Color(1.0, 1.0, 1.0, 1.0);

		_cached_draw_pts.resize(curve->get_point_count() * 8);
		int count = 0;

		for (int i = 0; i < curve->get_point_count(); i++) {
			for (int j = 0; j < 8; j++) {
				real_t frac = j * (1.0 / 8.0);
				Vector2 p = curve->interpolate(i, frac);
				_cached_draw_pts.set(count++, p);
			}
		}

		draw_polyline(_cached_draw_pts, color, line_width, true);
	}
}

void Path2D::_curve_changed() {
	if (!is_inside_tree()) {
		return;
	}

	if (!Engine::get_singleton()->is_editor_hint() && !get_tree()->is_debugging_navigation_hint()) {
		return;
	}

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
	ClassDB::bind_method(D_METHOD("set_curve", "curve"), &Path2D::set_curve);
	ClassDB::bind_method(D_METHOD("get_curve"), &Path2D::get_curve);
	ClassDB::bind_method(D_METHOD("_curve_changed"), &Path2D::_curve_changed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve2D"), "set_curve", "get_curve");
}

Path2D::Path2D() {
	set_curve(Ref<Curve2D>(memnew(Curve2D))); //create one by default
	set_self_modulate(Color(0.5, 0.6, 1.0, 0.7));
}

/////////////////////////////////////////////////////////////////////////////////

void PathFollow2D::_update_transform() {
	if (!path) {
		return;
	}

	Ref<Curve2D> c = path->get_curve();
	if (!c.is_valid()) {
		return;
	}

	float path_length = c->get_baked_length();
	if (path_length == 0) {
		return;
	}
	Vector2 pos = c->interpolate_baked(offset, cubic);

	if (rotate) {
		float ahead = offset + lookahead;

		if (loop && ahead >= path_length) {
			// If our lookahead will loop, we need to check if the path is closed.
			int point_count = c->get_point_count();
			if (point_count > 0) {
				Vector2 start_point = c->get_point_position(0);
				Vector2 end_point = c->get_point_position(point_count - 1);
				if (start_point == end_point) {
					// Since the path is closed we want to 'smooth off'
					// the corner at the start/end.
					// So we wrap the lookahead back round.
					ahead = Math::fmod(ahead, path_length);
				}
			}
		}

		Vector2 ahead_pos = c->interpolate_baked(ahead, cubic);

		Vector2 tangent_to_curve;
		if (ahead_pos == pos) {
			// This will happen at the end of non-looping or non-closed paths.
			// We'll try a look behind instead, in order to get a meaningful angle.
			tangent_to_curve =
					(pos - c->interpolate_baked(offset - lookahead, cubic)).normalized();
		} else {
			tangent_to_curve = (ahead_pos - pos).normalized();
		}

		Vector2 normal_of_curve = -tangent_to_curve.tangent();

		pos += tangent_to_curve * h_offset;
		pos += normal_of_curve * v_offset;

		set_rotation(tangent_to_curve.angle());

	} else {
		pos.x += h_offset;
		pos.y += v_offset;
	}

	set_position(pos);
}

void PathFollow2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			path = Object::cast_to<Path2D>(get_parent());
			if (path) {
				_update_transform();
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			path = nullptr;
		} break;
	}
}

void PathFollow2D::set_cubic_interpolation(bool p_enable) {
	cubic = p_enable;
}

bool PathFollow2D::get_cubic_interpolation() const {
	return cubic;
}

void PathFollow2D::_validate_property(PropertyInfo &property) const {
	if (property.name == "offset") {
		float max = 10000;
		if (path && path->get_curve().is_valid()) {
			max = path->get_curve()->get_baked_length();
		}

		property.hint_string = "0," + rtos(max) + ",0.01,or_lesser,or_greater";
	}
}

String PathFollow2D::get_configuration_warning() const {
	if (!is_visible_in_tree() || !is_inside_tree()) {
		return String();
	}

	String warning = Node2D::get_configuration_warning();
	if (!Object::cast_to<Path2D>(get_parent())) {
		if (warning != String()) {
			warning += "\n\n";
		}
		warning += TTR("PathFollow2D only works when set as a child of a Path2D node.");
	}

	return warning;
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

	ClassDB::bind_method(D_METHOD("set_lookahead", "lookahead"), &PathFollow2D::set_lookahead);
	ClassDB::bind_method(D_METHOD("get_lookahead"), &PathFollow2D::get_lookahead);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "offset", PROPERTY_HINT_RANGE, "0,10000,0.01,or_lesser,or_greater"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "unit_offset", PROPERTY_HINT_RANGE, "0,1,0.0001,or_lesser,or_greater", PROPERTY_USAGE_EDITOR), "set_unit_offset", "get_unit_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "h_offset"), "set_h_offset", "get_h_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "v_offset"), "set_v_offset", "get_v_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rotate"), "set_rotate", "is_rotating");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cubic_interp"), "set_cubic_interpolation", "get_cubic_interpolation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lookahead", PROPERTY_HINT_RANGE, "0.001,1024.0,0.001"), "set_lookahead", "get_lookahead");
}

void PathFollow2D::set_offset(float p_offset) {
	offset = p_offset;
	if (path) {
		if (path->get_curve().is_valid()) {
			float path_length = path->get_curve()->get_baked_length();

			if (loop) {
				offset = Math::fposmod(offset, path_length);
				if (!Math::is_zero_approx(p_offset) && Math::is_zero_approx(offset)) {
					offset = path_length;
				}
			} else {
				offset = CLAMP(offset, 0, path_length);
			}
		}

		_update_transform();
	}
	_change_notify("offset");
	_change_notify("unit_offset");
}

void PathFollow2D::set_h_offset(float p_h_offset) {
	h_offset = p_h_offset;
	if (path) {
		_update_transform();
	}
}

float PathFollow2D::get_h_offset() const {
	return h_offset;
}

void PathFollow2D::set_v_offset(float p_v_offset) {
	v_offset = p_v_offset;
	if (path) {
		_update_transform();
	}
}

float PathFollow2D::get_v_offset() const {
	return v_offset;
}

float PathFollow2D::get_offset() const {
	return offset;
}

void PathFollow2D::set_unit_offset(float p_unit_offset) {
	if (path && path->get_curve().is_valid() && path->get_curve()->get_baked_length()) {
		set_offset(p_unit_offset * path->get_curve()->get_baked_length());
	}
}

float PathFollow2D::get_unit_offset() const {
	if (path && path->get_curve().is_valid() && path->get_curve()->get_baked_length()) {
		return get_offset() / path->get_curve()->get_baked_length();
	} else {
		return 0;
	}
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
	path = nullptr;
	rotate = true;
	cubic = true;
	loop = true;
	lookahead = 4;
}
