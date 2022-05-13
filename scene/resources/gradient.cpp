/*************************************************************************/
/*  gradient.cpp                                                         */
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

#include "gradient.h"

#include "core/core_string_names.h"

//setter and getter names for property serialization
#define COLOR_RAMP_GET_OFFSETS "get_offsets"
#define COLOR_RAMP_GET_COLORS "get_colors"
#define COLOR_RAMP_SET_OFFSETS "set_offsets"
#define COLOR_RAMP_SET_COLORS "set_colors"

Gradient::Gradient() {
	//Set initial color ramp transition from black to white
	points.resize(2);
	points.write[0].color = Color(0, 0, 0, 1);
	points.write[0].offset = 0;
	points.write[1].color = Color(1, 1, 1, 1);
	points.write[1].offset = 1;
	is_sorted = true;
}

Gradient::~Gradient() {
}

void Gradient::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_point", "offset", "color"), &Gradient::add_point);
	ClassDB::bind_method(D_METHOD("remove_point", "point"), &Gradient::remove_point);

	ClassDB::bind_method(D_METHOD("set_offset", "point", "offset"), &Gradient::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset", "point"), &Gradient::get_offset);

	ClassDB::bind_method(D_METHOD("set_color", "point", "color"), &Gradient::set_color);
	ClassDB::bind_method(D_METHOD("get_color", "point"), &Gradient::get_color);

	ClassDB::bind_method(D_METHOD("interpolate", "offset"), &Gradient::get_color_at_offset);

	ClassDB::bind_method(D_METHOD("get_point_count"), &Gradient::get_points_count);

	ClassDB::bind_method(D_METHOD(COLOR_RAMP_SET_OFFSETS, "offsets"), &Gradient::set_offsets);
	ClassDB::bind_method(D_METHOD(COLOR_RAMP_GET_OFFSETS), &Gradient::get_offsets);

	ClassDB::bind_method(D_METHOD(COLOR_RAMP_SET_COLORS, "colors"), &Gradient::set_colors);
	ClassDB::bind_method(D_METHOD(COLOR_RAMP_GET_COLORS), &Gradient::get_colors);

	ClassDB::bind_method(D_METHOD("set_interpolation_mode", "interpolation_mode"), &Gradient::set_interpolation_mode);
	ClassDB::bind_method(D_METHOD("get_interpolation_mode"), &Gradient::get_interpolation_mode);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "interpolation_mode", PROPERTY_HINT_ENUM, "Linear,Constant,Cubic"), "set_interpolation_mode", "get_interpolation_mode");

	ADD_GROUP("Raw Data", "");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_REAL_ARRAY, "offsets"), COLOR_RAMP_SET_OFFSETS, COLOR_RAMP_GET_OFFSETS);
	ADD_PROPERTY(PropertyInfo(Variant::POOL_COLOR_ARRAY, "colors"), COLOR_RAMP_SET_COLORS, COLOR_RAMP_GET_COLORS);

	BIND_ENUM_CONSTANT(GRADIENT_INTERPOLATE_LINEAR);
	BIND_ENUM_CONSTANT(GRADIENT_INTERPOLATE_CONSTANT);
	BIND_ENUM_CONSTANT(GRADIENT_INTERPOLATE_CUBIC);
}

Vector<float> Gradient::get_offsets() const {
	Vector<float> offsets;
	offsets.resize(points.size());
	for (int i = 0; i < points.size(); i++) {
		offsets.write[i] = points[i].offset;
	}
	return offsets;
}

Vector<Color> Gradient::get_colors() const {
	Vector<Color> colors;
	colors.resize(points.size());
	for (int i = 0; i < points.size(); i++) {
		colors.write[i] = points[i].color;
	}
	return colors;
}

void Gradient::set_interpolation_mode(Gradient::InterpolationMode p_interp_mode) {
	interpolation_mode = p_interp_mode;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Gradient::InterpolationMode Gradient::get_interpolation_mode() {
	return interpolation_mode;
}

void Gradient::set_offsets(const Vector<float> &p_offsets) {
	points.resize(p_offsets.size());
	for (int i = 0; i < points.size(); i++) {
		points.write[i].offset = p_offsets[i];
	}
	is_sorted = false;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

void Gradient::set_colors(const Vector<Color> &p_colors) {
	if (points.size() < p_colors.size()) {
		is_sorted = false;
	}
	points.resize(p_colors.size());
	for (int i = 0; i < points.size(); i++) {
		points.write[i].color = p_colors[i];
	}
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Vector<Gradient::Point> &Gradient::get_points() {
	return points;
}

void Gradient::add_point(float p_offset, const Color &p_color) {
	Point p;
	p.offset = p_offset;
	p.color = p_color;
	is_sorted = false;
	points.push_back(p);

	emit_signal(CoreStringNames::get_singleton()->changed);
}

void Gradient::remove_point(int p_index) {
	ERR_FAIL_INDEX(p_index, points.size());
	ERR_FAIL_COND(points.size() <= 1);
	points.remove(p_index);
	emit_signal(CoreStringNames::get_singleton()->changed);
}

void Gradient::set_points(Vector<Gradient::Point> &p_points) {
	points = p_points;
	is_sorted = false;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

void Gradient::set_offset(int pos, const float offset) {
	ERR_FAIL_INDEX(pos, points.size());
	_update_sorting();
	points.write[pos].offset = offset;
	is_sorted = false;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

float Gradient::get_offset(int pos) {
	ERR_FAIL_INDEX_V(pos, points.size(), 0.0);
	_update_sorting();
	return points[pos].offset;
}

void Gradient::set_color(int pos, const Color &color) {
	ERR_FAIL_INDEX(pos, points.size());
	_update_sorting();
	points.write[pos].color = color;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Color Gradient::get_color(int pos) {
	ERR_FAIL_INDEX_V(pos, points.size(), Color());
	_update_sorting();
	return points[pos].color;
}

int Gradient::get_points_count() const {
	return points.size();
}
