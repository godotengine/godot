/**************************************************************************/
/*  shape_texture_2d.cpp                                                  */
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

#include "shape_texture_2d.h"

#include "core/core_string_names.h"
#include "core/math/color.h"
#include "core/math/geometry_2d.h"

ShapeTexture2D::ShapeTexture2D() {
	_queue_update();
}

ShapeTexture2D::~ShapeTexture2D() {
	if (texture.is_valid()) {
		ERR_FAIL_NULL(RenderingServer::get_singleton());
		RS::get_singleton()->free(texture);
	}
}

void ShapeTexture2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_width", "width"), &ShapeTexture2D::set_width);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &ShapeTexture2D::set_height);
	// `get_width()` and `get_height()` methods are already exposed by the parent class Texture2D.
	ClassDB::bind_method(D_METHOD("set_points", "points"), &ShapeTexture2D::set_points);
	ClassDB::bind_method(D_METHOD("get_points"), &ShapeTexture2D::get_points);
	ClassDB::bind_method(D_METHOD("set_star_enabled", "star_enabled"), &ShapeTexture2D::set_star_enabled);
	ClassDB::bind_method(D_METHOD("is_star_enabled"), &ShapeTexture2D::is_star_enabled);
	ClassDB::bind_method(D_METHOD("set_star_inset", "star_inset"), &ShapeTexture2D::set_star_inset);
	ClassDB::bind_method(D_METHOD("get_star_inset"), &ShapeTexture2D::get_star_inset);
	ClassDB::bind_method(D_METHOD("set_rotation_degrees", "rotation_degrees"), &ShapeTexture2D::set_rotation_degrees);
	ClassDB::bind_method(D_METHOD("get_rotation_degrees"), &ShapeTexture2D::get_rotation_degrees);
	ClassDB::bind_method(D_METHOD("set_fill_color", "fill_color"), &ShapeTexture2D::set_fill_color);
	ClassDB::bind_method(D_METHOD("get_fill_color"), &ShapeTexture2D::get_fill_color);
	ClassDB::bind_method(D_METHOD("set_border_color", "border_color"), &ShapeTexture2D::set_border_color);
	ClassDB::bind_method(D_METHOD("get_border_color"), &ShapeTexture2D::get_border_color);
	ClassDB::bind_method(D_METHOD("set_border_width", "border_width"), &ShapeTexture2D::set_border_width);
	ClassDB::bind_method(D_METHOD("get_border_width"), &ShapeTexture2D::get_border_width);
	ClassDB::bind_method(D_METHOD("set_antialiased", "antialiased"), &ShapeTexture2D::set_antialiased);
	ClassDB::bind_method(D_METHOD("is_antialiased"), &ShapeTexture2D::is_antialiased);

	ClassDB::bind_method(D_METHOD("is_using_hdr"), &ShapeTexture2D::is_using_hdr);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,16384,suffix:px"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "height", PROPERTY_HINT_RANGE, "1,16384,suffix:px"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "points", PROPERTY_HINT_RANGE, "3,100,or_greater"), "set_points", "get_points");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "star_enabled"), "set_star_enabled", "is_star_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "star_inset", PROPERTY_HINT_RANGE, "0,1"), "set_star_inset", "get_star_inset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rotation_degrees", PROPERTY_HINT_RANGE, "-180,180"), "set_rotation_degrees", "get_rotation_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "fill_color"), "set_fill_color", "get_fill_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "border_color"), "set_border_color", "get_border_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "border_width", PROPERTY_HINT_RANGE, "0,50,or_greater"), "set_border_width", "get_border_width");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "antialiased"), "set_antialiased", "is_antialiased");
}

void ShapeTexture2D::_queue_update() {
	if (update_pending) {
		return;
	}
	update_pending = true;
	callable_mp(this, &ShapeTexture2D::update_now).call_deferred();
}

void ShapeTexture2D::_update() {
	update_pending = false;

	Ref<Image> image;
	image.instantiate();

	Vector<Vector2> fill_polygon;
	Vector<Vector2> border_polygon;
	int total_points = star_enabled ? points * 2 : points;
	border_polygon.resize(total_points);
	fill_polygon.resize(total_points);
	Vector2 offset = Vector2(.5, .5);
	Vector2 border_point = Vector2(0, -.5);
	real_t fill_inset = border_width / height;
	Vector2 fill_point = Vector2(0, -.5 + fill_inset);
	real_t angle_delta = Math_TAU / total_points;
	real_t initial_angle = Math::deg_to_rad(rotation_degrees);
	Vector2 scale = Vector2(width, height);
	for (int i = 0; i < total_points; i++) {
		real_t inset = star_enabled && i % 2 ? 1 - star_inset : 1;
		border_polygon.write[i] = (offset + border_point.rotated(initial_angle + i * angle_delta) * inset) * scale;
		fill_polygon.write[i] = (offset + fill_point.rotated(initial_angle + i * angle_delta) * inset) * scale;
	}

	if (is_using_hdr()) {
		image->initialize_data(width, height, false, Image::FORMAT_RGBAF);
		// `create()` isn't available for non-uint8_t data, so fill in the data manually.
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				image->set_pixel(x, y, _get_color(Vector2(x, y), border_polygon, fill_polygon, antialiased));
			}
		}
	} else {
		Vector<uint8_t> data;
		data.resize(width * height * 4);
		{
			uint8_t *wd8 = data.ptrw();
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					const Color &c = _get_color(Vector2(x, y), border_polygon, fill_polygon, antialiased);

					wd8[(x + (y * width)) * 4 + 0] = uint8_t(CLAMP(c.r * 255.0, 0, 255));
					wd8[(x + (y * width)) * 4 + 1] = uint8_t(CLAMP(c.g * 255.0, 0, 255));
					wd8[(x + (y * width)) * 4 + 2] = uint8_t(CLAMP(c.b * 255.0, 0, 255));
					wd8[(x + (y * width)) * 4 + 3] = uint8_t(CLAMP(c.a * 255.0, 0, 255));
				}
			}
		}
		image->set_data(width, height, false, Image::FORMAT_RGBA8, data);
	}

	if (texture.is_valid()) {
		RID new_texture = RS::get_singleton()->texture_2d_create(image);
		RS::get_singleton()->texture_replace(texture, new_texture);
	} else {
		texture = RS::get_singleton()->texture_2d_create(image);
	}
}

Color ShapeTexture2D::_get_color(Vector2 p_point, PackedVector2Array &p_border, PackedVector2Array &p_fill, bool p_supersample) {
	if (p_supersample) {
		return (_get_color(p_point, p_border, p_fill, false) +
					   _get_color(p_point + Vector2(0.5, 0), p_border, p_fill, false) +
					   _get_color(p_point + Vector2(0, 0.5), p_border, p_fill, false) +
					   _get_color(p_point + Vector2(0.5, 0.5), p_border, p_fill, false)) /
				4;
	} else {
		if (Geometry2D::is_point_in_polygon(p_point, p_border)) {
			if (Geometry2D::is_point_in_polygon(p_point, p_fill)) {
				return fill_color;
			} else {
				return border_color;
			}
		} else {
			return background_color;
		}
	}
}

void ShapeTexture2D::set_width(int p_width) {
	ERR_FAIL_COND_MSG(p_width <= 0 || p_width > 16384, "Texture dimensions have to be within 1 to 16384 range.");
	width = p_width;
	_queue_update();
	emit_changed();
}

int ShapeTexture2D::get_width() const {
	return width;
}

void ShapeTexture2D::set_height(int p_height) {
	ERR_FAIL_COND_MSG(p_height <= 0 || p_height > 16384, "Texture dimensions have to be within 1 to 16384 range.");
	height = p_height;
	_queue_update();
	emit_changed();
}

int ShapeTexture2D::get_height() const {
	return height;
}

void ShapeTexture2D::set_points(int p_points) {
	points = p_points;
	_queue_update();
	emit_changed();
}

int ShapeTexture2D::get_points() const {
	return points;
}

void ShapeTexture2D::set_star_enabled(bool p_is_star) {
	star_enabled = p_is_star;
	_queue_update();
	emit_changed();
}

bool ShapeTexture2D::is_star_enabled() const {
	return star_enabled;
}

void ShapeTexture2D::set_star_inset(real_t p_star_inset) {
	star_inset = p_star_inset;
	_queue_update();
	emit_changed();
}

real_t ShapeTexture2D::get_star_inset() const {
	return star_inset;
}

void ShapeTexture2D::set_rotation_degrees(real_t p_rotation_degrees) {
	rotation_degrees = p_rotation_degrees;
	_queue_update();
	emit_changed();
}

real_t ShapeTexture2D::get_rotation_degrees() const {
	return rotation_degrees;
}

void ShapeTexture2D::set_fill_color(Color p_fill_color) {
	fill_color = p_fill_color;
	_queue_update();
	emit_changed();
}

Color ShapeTexture2D::get_fill_color() const {
	return fill_color;
}

void ShapeTexture2D::set_border_color(Color p_border_color) {
	border_color = p_border_color;
	background_color = p_border_color;
	background_color.a = 0;
	_queue_update();
	emit_changed();
}

Color ShapeTexture2D::get_border_color() const {
	return border_color;
}

void ShapeTexture2D::set_border_width(real_t p_border_width) {
	border_width = p_border_width;
	_queue_update();
	emit_changed();
}

real_t ShapeTexture2D::get_border_width() const {
	return border_width;
}

void ShapeTexture2D::set_antialiased(bool p_antialiased) {
	antialiased = p_antialiased;
	_queue_update();
	emit_changed();
}

bool ShapeTexture2D::is_antialiased() const {
	return antialiased;
}

bool ShapeTexture2D::is_using_hdr() const {
	return fill_color.r > 1 || fill_color.g > 1 || fill_color.b > 1 || border_color.r > 1 || border_color.g > 1 || border_color.b > 1;
}

RID ShapeTexture2D::get_rid() const {
	if (!texture.is_valid()) {
		texture = RS::get_singleton()->texture_2d_placeholder_create();
	}
	return texture;
}

Ref<Image> ShapeTexture2D::get_image() const {
	const_cast<ShapeTexture2D *>(this)->update_now();
	if (!texture.is_valid()) {
		return Ref<Image>();
	}
	return RenderingServer::get_singleton()->texture_2d_get(texture);
}

void ShapeTexture2D::update_now() {
	if (update_pending) {
		_update();
	}
}
