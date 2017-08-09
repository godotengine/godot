/*************************************************************************/
/*  texture_progress.cpp                                                 */
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
#include "texture_progress.h"

void TextureProgress::set_under_texture(const Ref<Texture> &p_texture) {

	under = p_texture;
	update();
	minimum_size_changed();
}

Ref<Texture> TextureProgress::get_under_texture() const {

	return under;
}

void TextureProgress::set_over_texture(const Ref<Texture> &p_texture) {

	over = p_texture;
	update();
	if (under.is_null()) {
		minimum_size_changed();
	}
}

Ref<Texture> TextureProgress::get_over_texture() const {

	return over;
}

Size2 TextureProgress::get_minimum_size() const {

	if (under.is_valid())
		return under->get_size();
	else if (over.is_valid())
		return over->get_size();
	else if (progress.is_valid())
		return progress->get_size();

	return Size2(1, 1);
}

void TextureProgress::set_progress_texture(const Ref<Texture> &p_texture) {

	progress = p_texture;
	update();
	minimum_size_changed();
}

Ref<Texture> TextureProgress::get_progress_texture() const {

	return progress;
}

Point2 TextureProgress::unit_val_to_uv(float val) {
	if (progress.is_null())
		return Point2();

	if (val < 0)
		val += 1;
	if (val > 1)
		val -= 1;

	Point2 p = get_relative_center();

	if (val < 0.125)
		return Point2(p.x + (1 - p.x) * val * 8, 0);
	if (val < 0.25)
		return Point2(1, p.y * (val - 0.125) * 8);
	if (val < 0.375)
		return Point2(1, p.y + (1 - p.y) * (val - 0.25) * 8);
	if (val < 0.5)
		return Point2(1 - (1 - p.x) * (val - 0.375) * 8, 1);
	if (val < 0.625)
		return Point2(p.x * (1 - (val - 0.5) * 8), 1);
	if (val < 0.75)
		return Point2(0, 1 - ((1 - p.y) * (val - 0.625) * 8));
	if (val < 0.875)
		return Point2(0, p.y - p.y * (val - 0.75) * 8);
	else
		return Point2(p.x * (val - 0.875) * 8, 0);
}

Point2 TextureProgress::get_relative_center() {
	if (progress.is_null())
		return Point2();
	Point2 p = progress->get_size() / 2;
	p += rad_center_off;
	p.x /= progress->get_width();
	p.y /= progress->get_height();
	p.x = CLAMP(p.x, 0, 1);
	p.y = CLAMP(p.y, 0, 1);
	return p;
}

void TextureProgress::_notification(int p_what) {
	const float corners[12] = { -0.125, -0.375, -0.625, -0.875, 0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875 };
	switch (p_what) {

		case NOTIFICATION_DRAW: {

			if (under.is_valid())
				draw_texture(under, Point2());
			if (progress.is_valid()) {
				Size2 s = progress->get_size();
				switch (mode) {
					case FILL_LEFT_TO_RIGHT: {
						Rect2 region = Rect2(Point2(), Size2(s.x * get_as_ratio(), s.y));
						draw_texture_rect_region(progress, region, region);
					} break;
					case FILL_RIGHT_TO_LEFT: {
						Rect2 region = Rect2(Point2(s.x - s.x * get_as_ratio(), 0), Size2(s.x * get_as_ratio(), s.y));
						draw_texture_rect_region(progress, region, region);
					} break;
					case FILL_TOP_TO_BOTTOM: {
						Rect2 region = Rect2(Point2(), Size2(s.x, s.y * get_as_ratio()));
						draw_texture_rect_region(progress, region, region);
					} break;
					case FILL_BOTTOM_TO_TOP: {
						Rect2 region = Rect2(Point2(0, s.y - s.y * get_as_ratio()), Size2(s.x, s.y * get_as_ratio()));
						draw_texture_rect_region(progress, region, region);
					} break;
					case FILL_CLOCKWISE:
					case FILL_COUNTER_CLOCKWISE: {
						float val = get_as_ratio() * rad_max_degrees / 360;
						if (val == 1) {
							Rect2 region = Rect2(Point2(), s);
							draw_texture_rect_region(progress, region, region);
						} else if (val != 0) {
							Array pts;
							float direction = mode == FILL_CLOCKWISE ? 1 : -1;
							float start = rad_init_angle / 360;
							float end = start + direction * val;
							pts.append(start);
							pts.append(end);
							float from = MIN(start, end);
							float to = MAX(start, end);
							for (int i = 0; i < 12; i++)
								if (corners[i] > from && corners[i] < to)
									pts.append(corners[i]);
							pts.sort();
							Vector<Point2> uvs;
							Vector<Point2> points;
							uvs.push_back(get_relative_center());
							points.push_back(Point2(s.x * get_relative_center().x, s.y * get_relative_center().y));
							for (int i = 0; i < pts.size(); i++) {
								Point2 uv = unit_val_to_uv(pts[i]);
								if (uvs.find(uv) >= 0)
									continue;
								uvs.push_back(uv);
								points.push_back(Point2(uv.x * s.x, uv.y * s.y));
							}
							draw_polygon(points, Vector<Color>(), uvs, progress);
						}
						if (get_tree()->is_editor_hint()) {
							Point2 p = progress->get_size();
							p.x *= get_relative_center().x;
							p.y *= get_relative_center().y;
							p = p.floor();
							draw_line(p - Point2(8, 0), p + Point2(8, 0), Color(0.9, 0.5, 0.5), 2);
							draw_line(p - Point2(0, 8), p + Point2(0, 8), Color(0.9, 0.5, 0.5), 2);
						}
					} break;
					default:
						draw_texture_rect_region(progress, Rect2(Point2(), Size2(s.x * get_as_ratio(), s.y)), Rect2(Point2(), Size2(s.x * get_as_ratio(), s.y)));
				}
			}
			if (over.is_valid())
				draw_texture(over, Point2());

		} break;
	}
}

void TextureProgress::set_fill_mode(int p_fill) {
	ERR_FAIL_INDEX(p_fill, 6);
	mode = (FillMode)p_fill;
	update();
}

int TextureProgress::get_fill_mode() {
	return mode;
}

void TextureProgress::set_radial_initial_angle(float p_angle) {
	while (p_angle > 360)
		p_angle -= 360;
	while (p_angle < 0)
		p_angle += 360;
	rad_init_angle = p_angle;
	update();
}

float TextureProgress::get_radial_initial_angle() {
	return rad_init_angle;
}

void TextureProgress::set_fill_degrees(float p_angle) {
	rad_max_degrees = CLAMP(p_angle, 0, 360);
	update();
}

float TextureProgress::get_fill_degrees() {
	return rad_max_degrees;
}

void TextureProgress::set_radial_center_offset(const Point2 &p_off) {
	rad_center_off = p_off;
	update();
}

Point2 TextureProgress::get_radial_center_offset() {
	return rad_center_off;
}

void TextureProgress::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_under_texture", "tex"), &TextureProgress::set_under_texture);
	ClassDB::bind_method(D_METHOD("get_under_texture"), &TextureProgress::get_under_texture);

	ClassDB::bind_method(D_METHOD("set_progress_texture", "tex"), &TextureProgress::set_progress_texture);
	ClassDB::bind_method(D_METHOD("get_progress_texture"), &TextureProgress::get_progress_texture);

	ClassDB::bind_method(D_METHOD("set_over_texture", "tex"), &TextureProgress::set_over_texture);
	ClassDB::bind_method(D_METHOD("get_over_texture"), &TextureProgress::get_over_texture);

	ClassDB::bind_method(D_METHOD("set_fill_mode", "mode"), &TextureProgress::set_fill_mode);
	ClassDB::bind_method(D_METHOD("get_fill_mode"), &TextureProgress::get_fill_mode);

	ClassDB::bind_method(D_METHOD("set_radial_initial_angle", "mode"), &TextureProgress::set_radial_initial_angle);
	ClassDB::bind_method(D_METHOD("get_radial_initial_angle"), &TextureProgress::get_radial_initial_angle);

	ClassDB::bind_method(D_METHOD("set_radial_center_offset", "mode"), &TextureProgress::set_radial_center_offset);
	ClassDB::bind_method(D_METHOD("get_radial_center_offset"), &TextureProgress::get_radial_center_offset);

	ClassDB::bind_method(D_METHOD("set_fill_degrees", "mode"), &TextureProgress::set_fill_degrees);
	ClassDB::bind_method(D_METHOD("get_fill_degrees"), &TextureProgress::get_fill_degrees);

	ADD_GROUP("Textures", "texture_");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture_under", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_under_texture", "get_under_texture");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture_over", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_over_texture", "get_over_texture");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture_progress", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_progress_texture", "get_progress_texture");
	ADD_PROPERTYNZ(PropertyInfo(Variant::INT, "fill_mode", PROPERTY_HINT_ENUM, "Left to Right,Right to Left,Top to Bottom,Bottom to Top,Clockwise,Counter Clockwise"), "set_fill_mode", "get_fill_mode");
	ADD_GROUP("Radial Fill", "radial_");
	ADD_PROPERTYNZ(PropertyInfo(Variant::REAL, "radial_initial_angle", PROPERTY_HINT_RANGE, "0.0,360.0,0.1,slider"), "set_radial_initial_angle", "get_radial_initial_angle");
	ADD_PROPERTYNZ(PropertyInfo(Variant::REAL, "radial_fill_degrees", PROPERTY_HINT_RANGE, "0.0,360.0,0.1,slider"), "set_fill_degrees", "get_fill_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "radial_center_offset"), "set_radial_center_offset", "get_radial_center_offset");

	BIND_CONSTANT(FILL_LEFT_TO_RIGHT);
	BIND_CONSTANT(FILL_RIGHT_TO_LEFT);
	BIND_CONSTANT(FILL_TOP_TO_BOTTOM);
	BIND_CONSTANT(FILL_BOTTOM_TO_TOP);
	BIND_CONSTANT(FILL_CLOCKWISE);
	BIND_CONSTANT(FILL_COUNTER_CLOCKWISE);
}

TextureProgress::TextureProgress() {
	mode = FILL_LEFT_TO_RIGHT;
	rad_init_angle = 0;
	rad_center_off = Point2();
	rad_max_degrees = 360;
	set_mouse_filter(MOUSE_FILTER_PASS);
}
