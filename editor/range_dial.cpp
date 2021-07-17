/*************************************************************************/
/*  range_dial.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "range_dial.h"

#include "editor/editor_scale.h"
#define SCALE EDSCALE

void RangeDial::_draw_rulers(double p_level) {
	double size = p_level * zoom;
	if (size < pow(10, 1) || size > pow(10, 2))
		return;

	double alpha = pow(size / 100, 3) / (pow(size / 100, 3) + pow((1 - size / 100), 3));
	int precision = is_using_rounded_values() ? 0 : MAX(0, -floor(log10(get_step())));
	double span = ceil(get_rect().size.x / size);
	double real_middle = get_rect().size.x / 2 - (cached_value * zoom);
	int font_size = get_theme_font_size("font_size", "Label");
	Ref<Font> font = get_theme_font("rulers", "EditorFonts");

	PackedVector2Array notches;
	PackedVector2Array notches_highlight;
	PackedVector2Array notches_outside;
	PackedVector2Array notches_highlight_outside;
	Color font_color = get_theme_color("font_color", "Label");

	for (int n = -span; n < span; n++) {
		double number_x = get_rect().size.x / 2 - fmod(cached_value * zoom, size) + n * size;
		double notch_value = Math::snapped((number_x - real_middle) / zoom, 1 / (pow(10, (double)precision + 1)));
		if (fabs(notch_value) < CMP_EPSILON2)
			notch_value = 0;
		String current_value(String::num(notch_value, precision));
		Size2 string_size = font->get_string_size(current_value, font_size);
		Color number_font_color = font_color;
		if (number_x - string_size.x / 2 > get_rect().size.x || number_x + string_size.x / 2 < 0)
			continue;

		bool outside = false;
		if ((!is_lesser_allowed() && (notch_value <= get_min() + CMP_EPSILON2)) || (!is_greater_allowed() && (notch_value >= get_max() - CMP_EPSILON2)))
			continue;
		else {
			if (fabs(notch_value - get_min()) < CMP_EPSILON2 || fabs(notch_value - get_max()) < CMP_EPSILON2)
				continue;
			if ((notch_value <= get_min() + CMP_EPSILON2) || (notch_value >= get_max() - CMP_EPSILON2)) {
				number_font_color.a = 0.25;
				outside = true;
			}
		}

		Vector2 line_start(number_x, 0);
		Vector2 line_end(number_x, get_rect().size.y - string_size.y);

		if (fmod(fabs(notch_value), p_level * 10) < CMP_EPSILON2 || fmod(fabs(notch_value), p_level * 10) >= (p_level * 10) - CMP_EPSILON2) {
			if (outside) {
				notches_highlight_outside.push_back(line_start);
				notches_highlight_outside.push_back(line_end);
			} else {
				notches_highlight.push_back(line_start);
				notches_highlight.push_back(line_end);
			}

			draw_string(font, Point2(number_x - (string_size.x / 2), get_rect().size.y).floor(), current_value, HALIGN_CENTER, -1, -1, number_font_color);
		} else {
			if (outside) {
				notches_outside.push_back(line_start);
				notches_outside.push_back(line_end);
			} else {
				notches.push_back(line_start);
				notches.push_back(line_end);
			}
			number_font_color.a *= alpha;
			draw_string(font, Point2(number_x - (string_size.x / 2), get_rect().size.y).floor(), current_value, HALIGN_CENTER, -1, -1, number_font_color);
		}
	}

	String max_value_str(String::num(get_max()));
	String min_value_str(String::num(get_min()));
	Size2 max_value_string_size = font->get_string_size(max_value_str, font_size);
	Size2 min_value_string_size = font->get_string_size(min_value_str, font_size);
	Color mono_color, mono_color_outside, mono_color_highlight, mono_color_highlight_outside;

	Color accent_color = get_theme_color("accent_color", "Editor");
	mono_color = mono_color_outside = mono_color_highlight = mono_color_highlight_outside = get_theme_color("mono_color", "Editor");
	Color highlight_color = get_theme_color("highlight_color", "Editor");

	mono_color.a = 0.5 * alpha;
	mono_color_outside.a = 0.25 * alpha;
	mono_color_highlight.a = 0.75;
	mono_color_highlight_outside.a = 0.25;
	highlight_color.a = 0.05;
	double number_x_min = real_middle + get_min() * zoom;
	double number_x_max = real_middle + get_max() * zoom;
	double min_x = MIN(MAX(0, number_x_min), get_rect().size.x);
	double max_x = MAX(MIN(get_rect().size.x, number_x_max), 0);
	draw_rect(Rect2(max_x, 0, min_x - max_x, get_rect().size.y), highlight_color);

	if (number_x_max - max_value_string_size.x / 2 < get_rect().size.x && number_x_max + max_value_string_size.x / 2 > 0) {
		draw_line(Point2(number_x_max, 0), Point2(number_x_max, get_rect().size.y - max_value_string_size.y), accent_color);
		draw_string(font, Point2(number_x_max - (max_value_string_size.x / 2), get_rect().size.y).floor(), max_value_str, HALIGN_CENTER, -1, font_size, accent_color);
	}
	if (number_x_min + min_value_string_size.x > 0 && number_x_min - min_value_string_size.x < get_rect().size.x) {
		draw_line(Point2(number_x_min, 0), Point2(number_x_min, get_rect().size.y - max_value_string_size.y), accent_color);
		draw_string(font, Point2(number_x_min - (min_value_string_size.x / 2), get_rect().size.y).floor(), min_value_str, HALIGN_CENTER, -1, font_size, accent_color);
	}

	if (!notches.is_empty())
		draw_multiline(notches, mono_color);
	if (!notches_outside.is_empty())
		draw_multiline(notches_outside, mono_color_outside);
	if (!notches_highlight.is_empty())
		draw_multiline(notches_highlight, mono_color_highlight);
	if (!notches_highlight_outside.is_empty())
		draw_multiline(notches_highlight_outside, mono_color_highlight_outside);
	if (!notches_highlight_outside.is_empty())
		draw_multiline(notches_highlight_outside, mono_color_highlight_outside);
}

void RangeDial::update_by_relative(double p_relative) {
	set_zoom(zoom + ((p_relative * -0.5) / (100 / zoom)));
}

double RangeDial::get_zoom_relative(double p_relative) {
	return (p_relative * (100 / zoom) * 3 / 1000);
}

void RangeDial::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		for (int i = 1; i < log10((fabs(get_min()) + get_max())) + 1; i++)
			_draw_rulers(pow(10, i));
		for (int i = 0; i < (is_using_rounded_values() ? 0 : MAX(0, -floor(log10(get_step())))) + 1; i++)
			_draw_rulers(1 / pow(10, i));
		draw_line(Vector2(get_rect().size.x / 2, 0), Vector2(get_rect().size.x / 2, get_rect().size.y), Color(0.96, 0.2, 0.32));
	}
}

void RangeDial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_zoom"), &RangeDial::get_zoom);
	ClassDB::bind_method(D_METHOD("set_zoom", "amount"), &RangeDial::set_zoom);
	ClassDB::bind_method(D_METHOD("update_by_relative", "relative"), &RangeDial::update_by_relative);
	ClassDB::bind_method(D_METHOD("get_zoom_relative", "relative"), &RangeDial::get_zoom_relative);

	ADD_SIGNAL(MethodInfo("zoom_changed"));
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "zoom"), "set_zoom", "get_zoom");
}

void RangeDial::set_zoom(double p_zoom) {
	int precision = is_using_rounded_values() ? 0 : MAX(0, -floor(log10(get_step())));
	double min_zoom = get_rect().size.x / (fabs(get_min()) + get_max()) / 2;
	double max_zoom = pow(10, (double)precision + 2);
	if (min_zoom > max_zoom)
		zoom = max_zoom;
	else
		zoom = CLAMP(p_zoom, min_zoom, max_zoom);
	emit_signal("zoom_changed");
	update();
}

double RangeDial::get_zoom() const {
	return zoom;
}

void RangeDial::set_value(double p_val) {
	if (!is_greater_allowed() && p_val > get_max() - get_page()) {
		p_val = get_max() - get_page();
	}

	if (!is_lesser_allowed() && p_val < get_min()) {
		p_val = get_min();
	}

	cached_value = p_val;
	Range::set_value(p_val);
}

RangeDial::RangeDial() {
	zoom = 1;
	cached_value = 0;
}

void RangeDialPopup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_dial"), &RangeDialPopup::get_dial);
}

RangeDial *RangeDialPopup::get_dial() const {
	return dial;
}

RangeDialPopup::RangeDialPopup() {
	set_min_size(Vector2(350, 48) * SCALE);
	set_transient(false);
	set_flag(Window::FLAG_NO_FOCUS, true);
	set_wrap_controls(true);
	dial = memnew(RangeDial);
	dial->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	add_child(dial);
}
