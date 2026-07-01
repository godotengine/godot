/**************************************************************************/
/*  trail_2d.cpp                                                          */
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

#include "trail_2d.h"

#include "core/math/geometry_2d.h"
#include "line_builder.h"

void Trail2D::set_emitting(bool p_emitting) {
	if (emitting == p_emitting) {
		return;
	}

	emitting = p_emitting;

	if (emitting) {
		_points.clear();
		active = true;
		set_process_internal(true);
		_last_position = get_global_position();
		_end_age_diff = 1.0;
	} else {
		// If emitting gets toggled off and we are more than 1 pixel from that leading point, we add the "fake" rendered leading point as a permanent leading point.
		if (_points.size() > 0 && get_global_position().distance_to(_points[_points.size() - 1].position) > 1.0) {
			TrailPoint point;
			point.position = get_global_position();
			point.time = 0;
			_points.push_back(point);
		}
	}
}

bool Trail2D::is_emitting() const {
	return emitting;
}

void Trail2D::set_lifetime(double p_lifetime) {
	ERR_FAIL_COND_MSG(p_lifetime <= 0, "Point lifetime must be greater than 0.");
	lifetime = p_lifetime;
}

double Trail2D::get_lifetime() const {
	return lifetime;
}

void Trail2D::set_segment_length(float p_segment_length) {
	ERR_FAIL_COND_MSG(p_segment_length <= 0, "Segment length must be greater than 0.");
	segment_length = p_segment_length;
}

float Trail2D::get_segment_length() const {
	return segment_length;
}

void Trail2D::set_width(float p_width) {
	if (p_width < 0.0) {
		p_width = 0.0;
	}
	_width = p_width;
	queue_redraw();
}

float Trail2D::get_width() const {
	return _width;
}

void Trail2D::set_curve(const Ref<Curve> &p_curve) {
	if (_curve.is_valid()) {
		_curve->disconnect_changed(callable_mp(this, &Trail2D::_curve_changed));
	}

	_curve = p_curve;

	if (_curve.is_valid()) {
		_curve->connect_changed(callable_mp(this, &Trail2D::_curve_changed));
	}

	queue_redraw();
}

Ref<Curve> Trail2D::get_curve() const {
	return _curve;
}

void Trail2D::set_default_color(Color p_color) {
	_default_color = p_color;
	queue_redraw();
}

Color Trail2D::get_default_color() const {
	return _default_color;
}

void Trail2D::set_gradient(const Ref<Gradient> &p_gradient) {
	if (_gradient.is_valid()) {
		_gradient->disconnect_changed(callable_mp(this, &Trail2D::_gradient_changed));
	}

	_gradient = p_gradient;

	if (_gradient.is_valid()) {
		_gradient->connect_changed(callable_mp(this, &Trail2D::_gradient_changed));
	}

	queue_redraw();
}

Ref<Gradient> Trail2D::get_gradient() const {
	return _gradient;
}

void Trail2D::set_texture(const Ref<Texture2D> &p_texture) {
	_texture = p_texture;
	queue_redraw();
}

Ref<Texture2D> Trail2D::get_texture() const {
	return _texture;
}

void Trail2D::set_texture_mode(const Line2D::LineTextureMode p_mode) {
	_texture_mode = p_mode;
	queue_redraw();
}

Line2D::LineTextureMode Trail2D::get_texture_mode() const {
	return _texture_mode;
}

void Trail2D::set_joint_mode(Line2D::LineJointMode p_mode) {
	_joint_mode = p_mode;
	queue_redraw();
}

Line2D::LineJointMode Trail2D::get_joint_mode() const {
	return _joint_mode;
}

void Trail2D::set_begin_cap_mode(Line2D::LineCapMode p_mode) {
	_begin_cap_mode = p_mode;
	queue_redraw();
}

Line2D::LineCapMode Trail2D::get_begin_cap_mode() const {
	return _begin_cap_mode;
}

void Trail2D::set_end_cap_mode(Line2D::LineCapMode p_mode) {
	_end_cap_mode = p_mode;
	queue_redraw();
}

Line2D::LineCapMode Trail2D::get_end_cap_mode() const {
	return _end_cap_mode;
}

void Trail2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			set_process_internal(emitting);
		} break;

		case NOTIFICATION_DRAW: {
			_draw();
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			_update_internal();
		} break;
	}
}

void Trail2D::set_round_precision(int p_precision) {
	_round_precision = MAX(1, p_precision);
	queue_redraw();
}

int Trail2D::get_round_precision() const {
	return _round_precision;
}

void Trail2D::set_sharp_limit(float p_limit) {
	if (p_limit < 0.f) {
		p_limit = 0.f;
	}
	_sharp_limit = p_limit;
	queue_redraw();
}

float Trail2D::get_sharp_limit() const {
	return _sharp_limit;
}

void Trail2D::_update_internal() {
	if (emitting) {
		if (_last_position.distance_to(get_global_position()) > segment_length || _points.size() == 0) {
			TrailPoint point;
			point.position = get_global_position();
			point.time = 0;
			_last_position = get_global_position();
			_points.push_back(point);
		}
	}
	if (!emitting && !active) {
		set_process_internal(false);
	}

	double delta = get_process_delta_time();

	for (int i = 0; i < _points.size(); i++) {
		_points.write[i].time += delta;
	}

	int expired_points = 0;
	for (int i = 0; i < _points.size(); i++) {
		if (_points[i].time > lifetime) {
			expired_points++;
		} else {
			break;
		}
	}

	if (expired_points > 0) {
		_points = _points.slice(expired_points);
		if (_points.size() >= 2) {
			_end_age_diff = _points[0].time - _points[1].time;
		}
	}

	if (_points.size() == 0) {
		_end_age_diff = 1.0f;
		emit_signal(SceneStringName(finished));
		if (!emitting) {
			active = false;
		}
	}

	queue_redraw();
}

void Trail2D::_draw() {
	if (_points.size() < 1 || _width == 0.f || _points.size() < 2) {
		return;
	}

	Vector<Vector2> local_space_points;

	// We only want to spawn the "fake" leading point if we are at least 1 pixel from the last actual point.
	bool add_leading_point = emitting && get_global_position().distance_to(_points[_points.size() - 1].position) > 1.0;
	if (add_leading_point) {
		local_space_points.push_back(Vector2(0, 0));
	}

	// Invert the order of the points when copying for desired texture mapping.
	// Don't include the last point of the trail.
	for (int i = _points.size() - 1; i >= 1; i--) {
		local_space_points.push_back(to_local(_points[i].position));
	}

	// We add the last point separately where it lerps towards towards the
	// previous point to make the tail shrink smoothly.
	Vector2 last_point;
	float t = CLAMP((lifetime - _points[0].time) / _end_age_diff, 0.0, 1.0);
	Vector2 next_to_last_point = _points.size() >= 2 ? _points[1].position : get_global_position();
	last_point = next_to_last_point.lerp(_points[0].position, t);

	// We only want to spawn the trailing point if we are at least 1 pixel away from the point we are lerping towards.
	if (last_point.distance_to(next_to_last_point) > 1.0) {
		local_space_points.push_back(to_local(last_point));
	}

	LineBuilder lb;
	lb.points = local_space_points;
	lb.default_color = _default_color;
	lb.gradient = *_gradient;
	lb.texture_mode = _texture_mode;
	lb.joint_mode = _joint_mode;
	lb.begin_cap_mode = _begin_cap_mode;
	lb.end_cap_mode = _end_cap_mode;
	lb.round_precision = _round_precision;
	lb.sharp_limit = _sharp_limit;
	lb.width = _width;
	lb.curve = *_curve;

	RID texture_rid;
	if (_texture.is_valid()) {
		texture_rid = _texture->get_rid();

		lb.tile_aspect = _texture->get_size().aspect();
	}

	lb.build();
	if (lb.indices.is_empty()) {
		return;
	}

	RS::get_singleton()->canvas_item_add_triangle_array(
			get_canvas_item(),
			lb.indices,
			lb.vertices,
			lb.colors,
			lb.uvs, Vector<int>(), Vector<float>(),
			texture_rid);
}

void Trail2D::_gradient_changed() {
	queue_redraw();
}

void Trail2D::_curve_changed() {
	queue_redraw();
}

// static
void Trail2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_emitting", "emitting"), &Trail2D::set_emitting);
	ClassDB::bind_method(D_METHOD("is_emitting"), &Trail2D::is_emitting);

	ClassDB::bind_method(D_METHOD("set_lifetime", "lifetime"), &Trail2D::set_lifetime);
	ClassDB::bind_method(D_METHOD("get_lifetime"), &Trail2D::get_lifetime);

	ClassDB::bind_method(D_METHOD("set_segment_length", "segment_length"), &Trail2D::set_segment_length);
	ClassDB::bind_method(D_METHOD("get_segment_length"), &Trail2D::get_segment_length);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &Trail2D::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &Trail2D::get_width);

	ClassDB::bind_method(D_METHOD("set_curve", "curve"), &Trail2D::set_curve);
	ClassDB::bind_method(D_METHOD("get_curve"), &Trail2D::get_curve);

	ClassDB::bind_method(D_METHOD("set_default_color", "color"), &Trail2D::set_default_color);
	ClassDB::bind_method(D_METHOD("get_default_color"), &Trail2D::get_default_color);

	ClassDB::bind_method(D_METHOD("set_gradient", "color"), &Trail2D::set_gradient);
	ClassDB::bind_method(D_METHOD("get_gradient"), &Trail2D::get_gradient);

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &Trail2D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &Trail2D::get_texture);

	ClassDB::bind_method(D_METHOD("set_joint_mode", "mode"), &Trail2D::set_joint_mode);
	ClassDB::bind_method(D_METHOD("get_joint_mode"), &Trail2D::get_joint_mode);

	ClassDB::bind_method(D_METHOD("set_begin_cap_mode", "mode"), &Trail2D::set_begin_cap_mode);
	ClassDB::bind_method(D_METHOD("get_begin_cap_mode"), &Trail2D::get_begin_cap_mode);

	ClassDB::bind_method(D_METHOD("set_end_cap_mode", "mode"), &Trail2D::set_end_cap_mode);
	ClassDB::bind_method(D_METHOD("get_end_cap_mode"), &Trail2D::get_end_cap_mode);

	ClassDB::bind_method(D_METHOD("set_texture_mode", "mode"), &Trail2D::set_texture_mode);
	ClassDB::bind_method(D_METHOD("get_texture_mode"), &Trail2D::get_texture_mode);

	ClassDB::bind_method(D_METHOD("set_sharp_limit", "limit"), &Trail2D::set_sharp_limit);
	ClassDB::bind_method(D_METHOD("get_sharp_limit"), &Trail2D::get_sharp_limit);

	ClassDB::bind_method(D_METHOD("set_round_precision", "precision"), &Trail2D::set_round_precision);
	ClassDB::bind_method(D_METHOD("get_round_precision"), &Trail2D::get_round_precision);

	ADD_SIGNAL(MethodInfo("finished"));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "emitting"), "set_emitting", "is_emitting");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lifetime", PROPERTY_HINT_NONE, "suffix:s"), "set_lifetime", "get_lifetime");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "segment_length", PROPERTY_HINT_NONE, "suffix:px"), "set_segment_length", "get_segment_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width", PROPERTY_HINT_NONE, "suffix:px"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "width_curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"), "set_curve", "get_curve");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "default_color"), "set_default_color", "get_default_color");
	ADD_GROUP("Fill", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gradient", PROPERTY_HINT_RESOURCE_TYPE, "Gradient"), "set_gradient", "get_gradient");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_mode", PROPERTY_HINT_ENUM, "None,Tile,Stretch"), "set_texture_mode", "get_texture_mode");
	ADD_GROUP("Capping", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "joint_mode", PROPERTY_HINT_ENUM, "Sharp,Bevel,Round"), "set_joint_mode", "get_joint_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "begin_cap_mode", PROPERTY_HINT_ENUM, "None,Box,Round"), "set_begin_cap_mode", "get_begin_cap_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "end_cap_mode", PROPERTY_HINT_ENUM, "None,Box,Round"), "set_end_cap_mode", "get_end_cap_mode");
	ADD_GROUP("Border", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sharp_limit"), "set_sharp_limit", "get_sharp_limit");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "round_precision", PROPERTY_HINT_RANGE, "1,32,1"), "set_round_precision", "get_round_precision");
}

Trail2D::Trail2D() {
	set_emitting(true);
}
