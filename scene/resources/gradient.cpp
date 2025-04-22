/**************************************************************************/
/*  gradient.cpp                                                          */
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

#include "gradient.h"

Gradient::Gradient() {
	//Set initial gradient transition from black to white
	points.resize(2);
	points[0].color = Color(0, 0, 0, 1);
	points[0].offset = 0;
	points[1].color = Color(1, 1, 1, 1);
	points[1].offset = 1;
}

Gradient::~Gradient() {
}

void Gradient::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_point", "offset", "color"), &Gradient::add_point);
	ClassDB::bind_method(D_METHOD("remove_point", "point"), &Gradient::remove_point);

	ClassDB::bind_method(D_METHOD("set_offset", "point", "offset"), &Gradient::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset", "point"), &Gradient::get_offset);

	ClassDB::bind_method(D_METHOD("reverse"), &Gradient::reverse);

	ClassDB::bind_method(D_METHOD("set_color", "point", "color"), &Gradient::set_color);
	ClassDB::bind_method(D_METHOD("get_color", "point"), &Gradient::get_color);

	ClassDB::bind_method(D_METHOD("sample", "offset"), &Gradient::get_color_at_offset);

	ClassDB::bind_method(D_METHOD("get_point_count"), &Gradient::get_point_count);

	ClassDB::bind_method(D_METHOD("set_offsets", "offsets"), &Gradient::set_offsets);
	ClassDB::bind_method(D_METHOD("get_offsets"), &Gradient::get_offsets);

	ClassDB::bind_method(D_METHOD("set_colors", "colors"), &Gradient::set_colors);
	ClassDB::bind_method(D_METHOD("get_colors"), &Gradient::get_colors);

	ClassDB::bind_method(D_METHOD("set_interpolation_mode", "interpolation_mode"), &Gradient::set_interpolation_mode);
	ClassDB::bind_method(D_METHOD("get_interpolation_mode"), &Gradient::get_interpolation_mode);

	ClassDB::bind_method(D_METHOD("set_interpolation_color_space", "interpolation_color_space"), &Gradient::set_interpolation_color_space);
	ClassDB::bind_method(D_METHOD("get_interpolation_color_space"), &Gradient::get_interpolation_color_space);

	ClassDB::bind_method(D_METHOD("set_dithering", "dithering"), &Gradient::set_dithering);
	ClassDB::bind_method(D_METHOD("get_dithering"), &Gradient::get_dithering);

	ClassDB::bind_method(D_METHOD("set_dithering_noise_granularity", "dithering_noise_granularity"), &Gradient::set_dithering_noise_granularity);
	ClassDB::bind_method(D_METHOD("get_dithering_noise_granularity"), &Gradient::get_dithering_noise_granularity);

	ADD_GROUP("Interpolation", "interpolation_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "interpolation_mode", PROPERTY_HINT_ENUM, "Linear,Constant,Cubic"), "set_interpolation_mode", "get_interpolation_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "interpolation_color_space", PROPERTY_HINT_ENUM, "sRGB,Linear sRGB,Oklab"), "set_interpolation_color_space", "get_interpolation_color_space");

	ADD_GROUP("Dithering", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dithering"), "set_dithering", "get_dithering");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dithering_noise_granularity", PROPERTY_HINT_RANGE, "0,10,0.01,or_greater"), "set_dithering_noise_granularity", "get_dithering_noise_granularity");

	ADD_GROUP("Raw Data", "");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "offsets"), "set_offsets", "get_offsets");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_COLOR_ARRAY, "colors"), "set_colors", "get_colors");

	BIND_ENUM_CONSTANT(GRADIENT_INTERPOLATE_LINEAR);
	BIND_ENUM_CONSTANT(GRADIENT_INTERPOLATE_CONSTANT);
	BIND_ENUM_CONSTANT(GRADIENT_INTERPOLATE_CUBIC);

	BIND_ENUM_CONSTANT(GRADIENT_COLOR_SPACE_SRGB);
	BIND_ENUM_CONSTANT(GRADIENT_COLOR_SPACE_LINEAR_SRGB);
	BIND_ENUM_CONSTANT(GRADIENT_COLOR_SPACE_OKLAB);
}

void Gradient::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "interpolation_color_space" && interpolation_mode == GRADIENT_INTERPOLATE_CONSTANT) {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

Vector<float> Gradient::get_offsets() const {
	Vector<float> offsets;
	offsets.resize(points.size());
	for (uint32_t i = 0; i < points.size(); i++) {
		offsets.write[i] = points[i].offset;
	}
	return offsets;
}

Vector<Color> Gradient::get_colors() const {
	Vector<Color> colors;
	colors.resize(points.size());
	for (uint32_t i = 0; i < points.size(); i++) {
		colors.write[i] = points[i].color;
	}
	return colors;
}

void Gradient::set_dithering(bool p_dithering) {
	if (p_dithering == dithering) {
		return;
	}

	dithering = p_dithering;
	emit_changed();
}

bool Gradient::get_dithering() {
	return dithering;
}

void Gradient::set_dithering_noise_granularity(float p_dithering_noise_granularity) {
	if (dithering_noise_granularity == p_dithering_noise_granularity) {
		return;
	}

	dithering_noise_granularity = p_dithering_noise_granularity;
	emit_changed();
}

float Gradient::get_dithering_noise_granularity() {
	return dithering_noise_granularity;
}

void Gradient::set_interpolation_mode(Gradient::InterpolationMode p_interp_mode) {
	if (p_interp_mode == interpolation_mode) {
		return;
	}

	interpolation_mode = p_interp_mode;
	emit_changed();
	notify_property_list_changed();
}

Gradient::InterpolationMode Gradient::get_interpolation_mode() {
	return interpolation_mode;
}

void Gradient::set_interpolation_color_space(Gradient::ColorSpace p_color_space) {
	if (p_color_space == interpolation_color_space) {
		return;
	}

	interpolation_color_space = p_color_space;
	emit_changed();
}

Gradient::ColorSpace Gradient::get_interpolation_color_space() {
	return interpolation_color_space;
}

void Gradient::set_offsets(const Vector<float> &p_offsets) {
	points.resize(p_offsets.size());
	for (uint32_t i = 0; i < points.size(); i++) {
		points[i].offset = p_offsets[i];
	}
	is_sorted = false;
	emit_changed();
}

void Gradient::set_colors(const Vector<Color> &p_colors) {
	if (points.size() < p_colors.size()) {
		is_sorted = false;
	}
	points.resize(p_colors.size());
	for (uint32_t i = 0; i < points.size(); i++) {
		points[i].color = p_colors[i];
	}
	emit_changed();
}

void Gradient::add_point(float p_offset, const Color &p_color) {
	Point p;
	p.offset = p_offset;
	p.color = p_color;
	is_sorted = false;
	points.push_back(p);

	emit_changed();
}

void Gradient::remove_point(int p_index) {
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_index, points.size());
	ERR_FAIL_COND(points.size() <= 1);
	points.remove_at(p_index);
	emit_changed();
}

void Gradient::reverse() {
	for (uint32_t i = 0; i < points.size(); i++) {
		points[i].offset = 1.0 - points[i].offset;
	}

	is_sorted = false;
	_update_sorting();
	emit_changed();
}

void Gradient::set_offset(int pos, const float offset) {
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)pos, points.size());
	_update_sorting();
	points[pos].offset = offset;
	is_sorted = false;
	emit_changed();
}

float Gradient::get_offset(int pos) {
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)pos, points.size(), 0.0);
	_update_sorting();
	return points[pos].offset;
}

void Gradient::set_color(int pos, const Color &color) {
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)pos, points.size());
	_update_sorting();
	points[pos].color = color;
	emit_changed();
}

Color Gradient::get_color(int pos) {
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)pos, points.size(), Color());
	_update_sorting();
	return points[pos].color;
}

int Gradient::get_point_count() const {
	return points.size();
}

Color Gradient::get_color_at_offset(float p_offset) {
	if (points.is_empty()) {
		return Color(0, 0, 0, 1);
	}

	_update_sorting();

	// Binary search.
	int low = 0;
	int high = points.size() - 1;
	int middle = 0;

#ifdef DEBUG_ENABLED
	if (low > high) {
		ERR_PRINT("low > high, this may be a bug");
	}
#endif

	while (low <= high) {
		middle = (low + high) / 2;
		const Point &point = points[middle];
		if (point.offset > p_offset) {
			high = middle - 1; //search low end of array
		} else if (point.offset < p_offset) {
			low = middle + 1; //search high end of array
		} else {
			return point.color;
		}
	}

	// Return sampled value.
	if (points[middle].offset > p_offset) {
		middle--;
	}
	int first = middle;
	int second = middle + 1;
	if (second >= (int)points.size()) {
		return points[points.size() - 1].color;
	}
	if (first < 0) {
		return points[0].color;
	}
	const Point &point1 = points[first];
	const Point &point2 = points[second];
	float weight = (p_offset - point1.offset) / (point2.offset - point1.offset);

	if (dithering) {
		const float noise_granularity = dithering_noise_granularity / 255.f;
		weight += Math::lerp(-noise_granularity, noise_granularity, Math::randf());
	}

	switch (interpolation_mode) {
		case GRADIENT_INTERPOLATE_CONSTANT: {
			return point1.color;
		}
		case GRADIENT_INTERPOLATE_LINEAR:
		default: { // Fallback to linear interpolation.
			Color color1 = transform_color_space(point1.color);
			Color color2 = transform_color_space(point2.color);
			Color interpolated = color1.lerp(color2, weight);
			return inv_transform_color_space(interpolated);
		}
		case GRADIENT_INTERPOLATE_CUBIC: {
			int p0 = first - 1;
			int p3 = second + 1;
			if (p3 >= (int)points.size()) {
				p3 = second;
			}
			if (p0 < 0) {
				p0 = first;
			}
			const Point &point0 = points[p0];
			const Point &point3 = points[p3];

			Color color0 = transform_color_space(point0.color);
			Color color1 = transform_color_space(point1.color);
			Color color2 = transform_color_space(point2.color);
			Color color3 = transform_color_space(point3.color);

			Color interpolated;
			interpolated[0] = Math::cubic_interpolate(color1[0], color2[0], color0[0], color3[0], weight);
			interpolated[1] = Math::cubic_interpolate(color1[1], color2[1], color0[1], color3[1], weight);
			interpolated[2] = Math::cubic_interpolate(color1[2], color2[2], color0[2], color3[2], weight);
			interpolated[3] = Math::cubic_interpolate(color1[3], color2[3], color0[3], color3[3], weight);

			return inv_transform_color_space(interpolated);
		}
	}
}
